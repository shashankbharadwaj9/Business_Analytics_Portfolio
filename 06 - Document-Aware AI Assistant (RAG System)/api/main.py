# =========================
# Doc AI API (FastAPI)
# What this file does:
# - Upload documents (.pdf/.txt) to /uploads
# - Extract text from documents
# - Chunk text into overlapping pieces
# - Create embeddings (OpenAI) and index chunks into ChromaDB (batched to avoid timeouts)
# - Clear and reindex workflows
# - Search indexed chunks with optional filename filters
# - Ask questions (RAG): retrieve + rerank + return ONLY clean bullet points (no sources returned)
# =========================

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Path
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import re

from pathlib import Path as SysPath
import shutil
import fitz  # type: ignore # PyMuPDF
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb


# =========================
# 1) Request Models
# What this section does:
# - Defines JSON bodies for /ask and /reindex endpoints
# =========================
class AskRequest(BaseModel):
    question: str
    top_k: int = 10
    filenames: Optional[List[str]] = None  # Example: ["Annual.pdf"]


class ReindexRequest(BaseModel):
    filenames: List[str]                 # Example: ["Annual.pdf", "Shashank_Resume_BA.pdf"]
    chunk_size: int = 1600               # better default for reports
    overlap: int = 200                   # better default for reports
    batch_size: int = 32                 # embeddings/upsert batch size (reduces timeouts)


# =========================
# 2) App + Storage + Clients
# What this section does:
# - Creates FastAPI app
# - Ensures uploads folder exists
# - Loads environment variables
# - Initializes OpenAI client and ChromaDB collection
# =========================
app = FastAPI(title="Doc AI API")

UPLOAD_DIR = SysPath("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found. Add it to your .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "doc_ai_chunks"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

EMBED_MODEL = "text-embedding-3-small"


# =========================
# 3) Health Endpoint
# What this section does:
# - Quick check that API is running
# =========================
@app.get("/")
def root():
    return {"message": "Doc AI is running"}


# =========================
# 4) Index Management
# What this section does:
# - Clears the entire vector index by deleting & recreating the Chroma collection
# =========================
@app.delete(
    "/index",
    tags=["Indexing"],
    summary="Clear the vector index",
    description=(
        "Deletes ALL previously indexed chunks from the Chroma collection and recreates an empty collection.\n\n"
        "Use this before re-indexing selected files."
    ),
)
def clear_index():
    global collection
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    return {"status": "cleared", "collection": COLLECTION_NAME}


# =========================
# 5) Documents: Upload + List
# What this section does:
# - Uploads files to uploads/ folder
# - Lists all uploaded files
# =========================
@app.post(
    "/upload",
    tags=["Documents"],
    summary="Upload a document",
    description="Upload a document file (.pdf or .txt). The file will be saved into the local uploads folder.",
)
def upload_document(
    file: UploadFile = File(..., description="Upload a document file (.pdf or .txt).")
):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "status": "uploaded successfully"}


@app.get(
    "/documents",
    tags=["Documents"],
    summary="List uploaded documents",
    description="Lists all files currently present in the uploads folder.",
)
def list_documents():
    files = [p.name for p in UPLOAD_DIR.iterdir() if p.is_file()]
    return {"documents": files}


# =========================
# 6) Text Extraction Utilities
# What this section does:
# - Extracts text from .txt and .pdf files
# =========================
def extract_text_from_txt(path: SysPath) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_text_from_pdf(path: SysPath) -> str:
    text_parts: List[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts)


def _read_full_text(file_path: SysPath) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return extract_text_from_txt(file_path)
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    raise ValueError("Unsupported file type. Upload .txt or .pdf for now.")


# =========================
# 7) Chunking Utility
# What this section does:
# - Splits long text into overlapping chunks for better retrieval
# =========================
def chunk_text(text: str, chunk_size: int = 1600, overlap: int = 200) -> list[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if end == n:
            break

    return chunks


# =========================
# 8) Preview Endpoints
# What this section does:
# - Preview extracted text and chunk previews (useful for debugging)
# =========================
@app.get(
    "/documents/{filename}/text",
    tags=["Documents"],
    summary="Preview extracted text",
    description="Extracts full text from the document and returns a preview (first ~3000 characters).",
)
def get_document_text(
    filename: str = Path(..., description="Enter the file name with extension (example: Annual.pdf).")
):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    full_text = _read_full_text(file_path)
    return {"filename": filename, "characters_extracted": len(full_text), "preview": full_text[:3000]}


@app.get(
    "/documents/{filename}/chunks",
    tags=["Documents"],
    summary="Preview text chunks",
    description="Splits extracted text into chunks and returns a small preview of the first few chunks.",
)
def get_document_chunks(
    filename: str = Path(..., description="Enter the file name with extension (example: Annual.pdf)."),
    chunk_size: int = Query(default=1600, description="Chunk size in characters (example: 1600).", ge=200, le=10000),
    overlap: int = Query(default=200, description="Overlap between chunks (example: 200). Must be < chunk_size.", ge=0, le=5000),
):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    full_text = _read_full_text(file_path)
    chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)

    preview_count = min(3, len(chunks))
    preview = [{"chunk_index": i, "text_preview": chunks[i][:300]} for i in range(preview_count)]
    return {
        "filename": filename,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "total_chunks": len(chunks),
        "preview": preview,
    }


# =========================
# 9) Embedding Utilities
# What this section does:
# - Embeds one query string (embed_text)
# - Embeds many chunk strings in one API call (embed_texts) to prevent timeouts
# =========================
def embed_text(text: str) -> list[float]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def embed_texts(texts: List[str]) -> List[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


# =========================
# 10) Indexing Utilities + Endpoint
# What this section does:
# - Reads text, chunks it, embeds chunks in batches, and upserts into ChromaDB
# =========================
def _index_document_internal(filename: str, chunk_size: int, overlap: int, batch_size: int) -> Dict[str, Any]:
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    full_text = _read_full_text(file_path)
    chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)

    total_indexed = 0

    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        batch_chunks = chunks[start:end]

        ids = [f"{filename}::chunk::{i}" for i in range(start, end)]
        metadatas = [{"filename": filename, "chunk_index": i} for i in range(start, end)]
        documents = batch_chunks

        embeddings = embed_texts(batch_chunks)

        collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        total_indexed += len(batch_chunks)

    return {"filename": filename, "total_chunks_indexed": total_indexed, "status": "indexed"}


@app.post(
    "/documents/{filename}/index",
    tags=["Indexing"],
    summary="Index a document",
    description="Chunks + embeds + indexes a single uploaded document into ChromaDB (batched).",
)
def index_document(
    filename: str = Path(..., description="Enter the file name with extension (example: Annual.pdf)."),
    chunk_size: int = Query(default=1600, description="Chunk size in characters (example: 1600).", ge=200, le=10000),
    overlap: int = Query(default=200, description="Overlap between chunks (example: 200). Must be < chunk_size.", ge=0, le=5000),
    batch_size: int = Query(default=32, description="Embedding batch size (example: 32).", ge=1, le=256),
):
    return _index_document_internal(filename, chunk_size, overlap, batch_size)


# =========================
# 11) Reindex Workflow
# What this section does:
# - Clears index then indexes a list of uploaded files in one request
# =========================
@app.post(
    "/reindex",
    tags=["Indexing"],
    summary="Clear index and re-index selected files",
    description=(
        "One-call workflow:\n"
        "1) Clears the Chroma collection\n"
        "2) Indexes each filename you provide (must exist in /uploads)\n\n"
        "Recommended defaults: chunk_size=1600, overlap=200, batch_size=32"
    ),
)
def reindex(req: ReindexRequest):
    clear_index()

    results: List[Dict[str, Any]] = []
    for fn in req.filenames:
        try:
            out = _index_document_internal(fn, req.chunk_size, req.overlap, req.batch_size)
            results.append(out)
        except Exception as e:
            results.append({"filename": fn, "status": "failed", "error": str(e)})

    return {
        "status": "reindex_complete",
        "collection": COLLECTION_NAME,
        "chunk_size": req.chunk_size,
        "overlap": req.overlap,
        "batch_size": req.batch_size,
        "results": results,
    }


# =========================
# 12) Search Endpoint
# What this section does:
# - Vector search over indexed chunks with optional filename filtering
# =========================
@app.get(
    "/search",
    tags=["Search"],
    summary="Search indexed documents (optional filename filter)",
    description="Vector similarity search. Optionally pass filenames multiple times to filter search.",
)
def search(
    query: str = Query(..., description="Enter your search question (example: revenue, approvals, projects)."),
    top_k: int = Query(default=5, description="Number of chunks to return (example: 5).", ge=1, le=50),
    filenames: Optional[List[str]] = Query(default=None, description="Optional: filter by filenames (use Add item to add more)."),
):
    query_embedding = embed_text(query)

    where = None
    if filenames:
        where = {"filename": filenames[0]} if len(filenames) == 1 else {"filename": {"$in": filenames}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
        where=where,
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append(
            {
                "id": results["ids"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
                "text_preview": results["documents"][0][i][:300],
            }
        )

    return {"query": query, "top_k": top_k, "filenames_filter": filenames, "hits": hits}


# =========================
# 13) RAG Helpers
# What this section does:
# - Rerank chunks using keyword matches
# - Apply money boost ONLY for numeric-heavy questions
# - Detect approvals/sign-offs questions and validate evidence
# - Clean up bullet formatting
# =========================
def _build_where_from_filenames(filenames: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    if not filenames:
        return None
    if len(filenames) == 1:
        return {"filename": filenames[0]}
    return {"filename": {"$in": filenames}}


def _keyword_boost_score(text: str, keywords: List[str]) -> int:
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)


def _has_money_amount(text: str) -> bool:
    t = text.lower()
    return bool(re.search(r"(usd\s*\d)|(\$\s*\d)|(\d+(\.\d+)?\s*million)|(\d+(\.\d+)?m\b)", t))


def _money_boost_score(text: str) -> int:
    return 3 if _has_money_amount(text) else 0


def _is_numeric_heavy_question(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ["expense", "expenses", "fees", "budget", "revenue", "cost", "amount", "usd", "$"])


def _is_approvals_question(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ["approval", "approvals", "sign-off", "signoff", "signoffs", "sign-offs", "sign off"])


def _looks_like_approval_chunk(text: str) -> bool:
    tl = text.lower()
    return any(x in tl for x in ["reviewed and approved", "approvals", "approval", "sign-off", "signoffs", "executive leadership"])


def _clean_bullet_line(line: str) -> str:
    line = line.strip()
    # remove trailing citation patterns if model outputs them
    line = re.sub(r"\s*(\(?[A-Za-z0-9_.-]+\.pdf::chunk::\d+\)?)\s*\.?\s*$", "", line).strip()
    # remove trailing punctuation (optional)
    line = line.rstrip()
    if line.endswith("."):
        line = line[:-1]
    return line


# =========================
# 14) Ask Endpoint (RAG) - returns ONLY bullets
# What this section does:
# - Retrieves relevant chunks
# - Reranks them (money boost only if numeric-heavy)
# - Forces the model to output bullet points WITHOUT citations
# - Returns only: question + bullets
# =========================
@app.post(
    "/ask",
    tags=["Q&A"],
    summary="Ask a question (returns only bullet points)",
    description=(
        "RAG workflow:\n"
        "1) Retrieve relevant chunks from Chroma\n"
        "2) Rerank chunks\n"
        "3) LLM answers in bullet points\n\n"
        "Response contains ONLY bullets (no sources, no chunk ids)."
    ),
)
def ask(req: AskRequest):
    # 1) Embed question
    q_embedding = embed_text(req.question)

    # 2) Optional filename filter
    where = _build_where_from_filenames(req.filenames)

    # 3) Retrieve top_k chunks
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=req.top_k,
        include=["documents", "metadatas", "distances"],
        where=where,
    )

    retrieved: List[Dict[str, Any]] = []
    for i in range(len(results["ids"][0])):
        retrieved.append(
            {
                "id": results["ids"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
                "text": results["documents"][0][i],
            }
        )

    if not retrieved:
        return {
            "question": req.question,
            "bullets": ["I couldn't find relevant content in the indexed documents. Try a different query or re-index the correct file."],
        }

    # 4) Rerank
    apply_money = _is_numeric_heavy_question(req.question)

    base_keywords = [
        # approvals/sign-offs
        "approvals", "approval", "sign-off", "sign-offs", "signoff", "signoffs",
        "reviewed", "approved", "executive leadership", "executive leadership team",
        "ceo", "cfo", "cco",
        # finance
        "operating", "expense", "expenses", "breakdown", "budget", "revenue",
        "audit", "compliance", "fees", "cost", "amount", "usd",
    ]
    q_words = [w.lower() for w in re.findall(r"[a-zA-Z]{4,}", req.question)]
    keywords = list(dict.fromkeys(base_keywords + q_words))

    retrieved.sort(
        key=lambda x: (
            -(_keyword_boost_score(x["text"], keywords) + (_money_boost_score(x["text"]) if apply_money else 0)),
            float(x["distance"]),
        )
    )

    # Approvals-specific evidence validation: ensure top context includes approval-like text
    if _is_approvals_question(req.question):
        approval_chunks = [x for x in retrieved if _looks_like_approval_chunk(x["text"])]
        if approval_chunks:
            # put approval chunks first
            rest = [x for x in retrieved if x not in approval_chunks]
            retrieved = approval_chunks + rest

    # 5) Build context
    context_blocks = []
    for item in retrieved:
        context_blocks.append(f"{item['text']}".strip())
    context = "\n\n---\n\n".join(context_blocks)

    # 6) LLM prompt: bullets only, no citations, strict format
    system_msg = (
        "You are a helpful assistant. Answer the user's question using ONLY the provided context.\n"
        "If the answer is not in the context, say you don't have enough information.\n\n"
        "OUTPUT FORMAT:\n"
        "- Return ONLY bullet points.\n"
        "- Use '-' at the start of each bullet.\n"
        "- Keep bullets short and factual.\n"
        "- Do NOT include any citations, chunk ids, or the word SOURCE.\n"
        "- Do NOT include headings or paragraphs.\n\n"
        "Special formatting for approvals/sign-offs questions:\n"
        "1) One bullet stating who conducted approvals/sign-offs.\n"
        "2) One bullet stating the approval date and 'by:'\n"
        "3) Then list EACH person on its own bullet (name, title).\n"
        "Do NOT merge names into one bullet.\n"
    )

    user_msg = f"QUESTION:\n{req.question}\n\nCONTEXT:\n{context}"

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )

    raw_answer = completion.choices[0].message.content or ""

    # 7) Parse bullets
    bullets: List[str] = []
    for line in raw_answer.splitlines():
        line = line.strip()
        if line.startswith("- "):
            bullets.append(_clean_bullet_line(line[2:]))

    if not bullets:
        cleaned = raw_answer.strip()
        if cleaned:
            bullets = [_clean_bullet_line(cleaned)]
        else:
            bullets = ["I don't have enough information in the provided context."]

    return {"question": req.question, "bullets": bullets}
