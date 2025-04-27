
-- Create Tables (Simulating Database Tables)

-- 1. Refund Transactions Table
CREATE TABLE refund_transactions_sample_v2 (
    Transaction_ID VARCHAR(10) PRIMARY KEY,
    Customer_ID VARCHAR(10),
    Region VARCHAR(50),
    Refund_Amount INT,
    Refund_Reason VARCHAR(100),
    Transaction_Date DATE,
    Fraud_Flag INT
);

-- 2. Region Info Table
CREATE TABLE region_info (
    Region_ID INT PRIMARY KEY,
    Region_Name VARCHAR(50),
    Risk_Score VARCHAR(20)
);

-- 3. Customer Info Table
CREATE TABLE customer_info (
    Customer_ID VARCHAR(10) PRIMARY KEY,
    Customer_Segment VARCHAR(20)
);

-- ----------------------------
-- JOIN Query: Full Enriched Refund Data
-- ----------------------------

SELECT 
    rts.Transaction_ID,
    rts.Customer_ID,
    ci.Customer_Segment,
    rts.Region,
    ri.Risk_Score,
    rts.Refund_Amount,
    rts.Refund_Reason,
    rts.Transaction_Date,
    rts.Fraud_Flag
FROM refund_transactions_sample_v2 rts
INNER JOIN region_info ri
    ON rts.Region = ri.Region_Name
INNER JOIN customer_info ci
    ON rts.Customer_ID = ci.Customer_ID
ORDER BY rts.Transaction_Date;

-- ----------------------------
-- Some Additional Queries
-- ----------------------------

-- Total Refunds
SELECT COUNT(Transaction_ID) AS Total_Refunds
FROM refund_transactions_sample_v2;

-- Total Fraudulent Refunds
SELECT COUNT(Transaction_ID) AS Fraudulent_Refunds
FROM refund_transactions_sample_v2
WHERE Fraud_Flag = 1;

-- Fraud Rate by Region
SELECT 
    rts.Region,
    COUNT(rts.Transaction_ID) AS Total_Refunds,
    SUM(CASE WHEN rts.Fraud_Flag = 1 THEN 1 ELSE 0 END) AS Fraudulent_Refunds,
    ROUND(SUM(CASE WHEN rts.Fraud_Flag = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(rts.Transaction_ID), 2) AS Fraud_Percentage,
    ri.Risk_Score
FROM refund_transactions_sample_v2 rts
INNER JOIN region_info ri
    ON rts.Region = ri.Region_Name
GROUP BY rts.Region, ri.Risk_Score
ORDER BY Fraud_Percentage DESC;

-- Top 5 Refund Reasons
SELECT 
    Refund_Reason,
    COUNT(*) AS Reason_Count
FROM refund_transactions_sample_v2
GROUP BY Refund_Reason
ORDER BY Reason_Count DESC
LIMIT 5;
