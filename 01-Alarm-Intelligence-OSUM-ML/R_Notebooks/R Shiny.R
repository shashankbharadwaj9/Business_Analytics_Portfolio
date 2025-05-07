library(shiny)
library(shinydashboard)
library(tidyverse)
library(readxl)
library(ggplot2)
library(caret)
library(rpart)
library(xgboost)
library(pROC)
library(e1071)
library(Metrics)
library(shinyBS)

# Load dataset
data <- read_excel("IM009B-XLS-ENG.xlsx")

# UI
ui <- dashboardPage(
  dashboardHeader(title = "Alarm Analytics Dashboard"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Dashboard", tabName = "dashboard", icon = icon("tachometer-alt")),
      menuItem("CHB Classification", tabName = "chb_model", icon = icon("brain")),
      menuItem("ATD Prediction", tabName = "atd_model", icon = icon("clock")),
      menuItem("Download", tabName = "download", icon = icon("download")),
      checkboxGroupInput("tag_filter", "Alarm Tag Type:", 
                         choices = unique(data$`Alarm Tag Type`), 
                         selected = unique(data$`Alarm Tag Type`)),
      checkboxGroupInput("week_filter", "Week:", 
                         choices = unique(data$Week), 
                         selected = unique(data$Week)),
      checkboxInput("chb_only", "Chattering Only", value = FALSE)
    )
  ),
  dashboardBody(
    bsTooltip("barPlot", "Shows how many alarms triggered by each tag type. High counts may indicate frequent issues.", "top", options = list(container = "body")),
    bsTooltip("atdHist", "Displays the distribution of Active Time Duration (ATD). High values may indicate unresolved alarms.", "top", options = list(container = "body")),
    bsTooltip("pieWeek", "Pie chart of M count by Week. Useful to detect peak alarm periods.", "top", options = list(container = "body")),
    bsTooltip("atdByTag", "Compares average ATD across tag types. High values suggest delays in resolution.", "top", options = list(container = "body")),
    bsTooltip("modelPerf", "Performance comparison of Logistic, Tree, and XGBoost models for CHB classification.", "top", options = list(container = "body")),
    bsTooltip("rocCurve", "ROC Curve for XGBoost classifier. AUC closer to 1 indicates better model performance.", "top", options = list(container = "body")),
    bsTooltip("actualLinear", "Plots actual vs predicted ATD using Linear Regression.", "top", options = list(container = "body")),
    bsTooltip("actualXGB", "Plots actual vs predicted ATD using XGBoost.", "top", options = list(container = "body")),
    bsTooltip("residualPlot", "Residuals of XGBoost prediction vs actual ATD. Spread indicates prediction accuracy.", "top", options = list(container = "body")),
    
    tabItems(
      tabItem(tabName = "dashboard",
              fluidRow(
                valueBoxOutput("totalAlarms"),
                valueBoxOutput("avgFlow"),
                valueBoxOutput("avgTemp")
              ),
              fluidRow(
                box(title = "Alarm Count by Tag Type", width = 6, plotOutput("barPlot")),
                box(title = "ATD Distribution", width = 6, plotOutput("atdHist"))
              ),
              fluidRow(
                box(title = "M by Week (Pie Chart)", width = 6, plotOutput("pieWeek")),
                box(title = "ATD by Tag Type", width = 6, plotOutput("atdByTag"))
              )
      ),
      tabItem(tabName = "chb_model",
              fluidRow(
                box(title = "CHB Model Performance", width = 12, plotOutput("modelPerf"))
              ),
              fluidRow(
                box(title = "ROC Curve", width = 12, plotOutput("rocCurve"))
              )
      ),
      tabItem(tabName = "atd_model",
              fluidRow(
                box(title = "ATD: Linear vs XGBoost", width = 6, plotOutput("actualLinear")),
                box(title = "XGBoost Predictions", width = 6, plotOutput("actualXGB"))
              ),
              fluidRow(
                box(title = "XGBoost Residuals", width = 12, plotOutput("residualPlot"))
              )
      ),
      tabItem(tabName = "download",
              downloadButton("downloadFiltered", "Download Filtered Data")
      )
    )
  )
)
server <- function(input, output) {
  
  filtered_data <- reactive({
    df <- data %>% filter(`Alarm Tag Type` %in% input$tag_filter, Week %in% input$week_filter)
    if (input$chb_only) df <- df %>% filter(CHB == 1)
    df
  })
  
  output$totalAlarms <- renderValueBox({
    valueBox(nrow(filtered_data()), "Total Alarms", icon = icon("bell"), color = "aqua")
  })
  
  output$avgFlow <- renderValueBox({
    valueBox(round(mean(filtered_data()$Flow, na.rm = TRUE), 2), "Average Flow", icon = icon("tint"), color = "green")
  })
  
  output$avgTemp <- renderValueBox({
    valueBox(round(mean(filtered_data()$Temperature, na.rm = TRUE), 2), "Average Temperature", icon = icon("temperature-high"), color = "orange")
  })
  
  output$barPlot <- renderPlot({
    filtered_data() %>% count(`Alarm Tag Type`) %>%
      ggplot(aes(x = reorder(`Alarm Tag Type`, n), y = n, fill = `Alarm Tag Type`)) +
      geom_bar(stat = "identity") + coord_flip() + theme_minimal() + labs(x = "Alarm Tag Type", y = "Count")
  })
  
  output$atdHist <- renderPlot({
    ggplot(filtered_data(), aes(x = ATD)) +
      geom_histogram(bins = 40, fill = "steelblue", color = "black") +
      labs(x = "Active Time Duration", y = "Frequency") + theme_minimal()
  })
  
  output$pieWeek <- renderPlot({
    filtered_data() %>% count(Week) %>%
      mutate(pct = round(n / sum(n) * 100, 1), label = paste0(pct, "%")) %>%
      ggplot(aes(x = "", y = n, fill = Week)) +
      geom_bar(stat = "identity", width = 1) + coord_polar(theta = "y") +
      geom_text(aes(label = label), position = position_stack(vjust = 0.5)) + theme_void()
  })
  
  output$atdByTag <- renderPlot({
    filtered_data() %>%
      group_by(`Alarm Tag Type`) %>%
      summarise(avgATD = mean(ATD, na.rm = TRUE)) %>%
      ggplot(aes(x = `Alarm Tag Type`, y = avgATD, fill = `Alarm Tag Type`)) +
      geom_bar(stat = "identity") + theme_minimal() + labs(y = "Average ATD")
  })
  
  output$modelPerf <- renderPlot({
    withProgress(message = 'Training CHB Models', value = 0, {
      df <- filtered_data() %>% select(ATD, CHB, M, `Alarm Tag Type`, H) %>% drop_na()
      df$CHB <- as.factor(df$CHB)
      set.seed(123)
      idx <- createDataPartition(df$CHB, p = 0.7, list = FALSE)
      train <- df[idx, ]; test <- df[-idx, ]
      
      incProgress(0.1, detail = "Preparing data (10%)")
      
      # Logistic Regression
      log_model <- glm(CHB ~ ., data = train, family = "binomial")
      log_probs <- predict(log_model, test, type = "response")
      log_preds <- ifelse(log_probs > 0.5, 1, 0)
      incProgress(0.3, detail = "Training Logistic Regression (40%)")
      
      # Decision Tree
      tree_model <- rpart(CHB ~ ., data = train, method = "class")
      tree_probs <- predict(tree_model, test, type = "prob")[,2]
      tree_preds <- ifelse(tree_probs > 0.5, 1, 0)
      incProgress(0.2, detail = "Training Decision Tree (60%)")
      
      # XGBoost
      mat_train <- model.matrix(CHB ~ . -1, train)
      mat_test <- model.matrix(CHB ~ . -1, test)
      xgb_mod <- xgboost(data = xgb.DMatrix(mat_train, label = as.numeric(train$CHB) - 1),
                         nrounds = 50, objective = "binary:logistic", verbose = 0)
      xgb_probs <- predict(xgb_mod, xgb.DMatrix(mat_test))
      xgb_preds <- ifelse(xgb_probs > 0.5, 1, 0)
      incProgress(0.3, detail = "Training XGBoost (90%)")
      
      # Metrics calculation
      get_metrics <- function(true, predicted, probs) {
        cm <- confusionMatrix(factor(predicted, levels = c(0, 1)), 
                              factor(true, levels = c(0, 1)), 
                              positive = "1")
        roc_obj <- roc(response = true, predictor = probs)
        tibble(
          Accuracy = cm$overall["Accuracy"],
          Precision = cm$byClass["Precision"],
          Recall = cm$byClass["Recall"],
          F1 = cm$byClass["F1"],
          AUC = as.numeric(roc_obj$auc)
        )
      }
      
      results <- bind_rows(
        get_metrics(test$CHB, log_preds, log_probs) %>% mutate(Model = "Logistic"),
        get_metrics(test$CHB, tree_preds, tree_probs) %>% mutate(Model = "Decision Tree"),
        get_metrics(test$CHB, xgb_preds, xgb_probs) %>% mutate(Model = "XGBoost")
      )
      
      incProgress(1, detail = "Done! (100%)")
      
      # Plot
      results %>%
        pivot_longer(-Model) %>%
        ggplot(aes(x = name, y = value, fill = Model)) +
        geom_col(position = position_dodge()) +
        geom_text(aes(label = round(value, 2)), vjust = -0.5, position = position_dodge(0.9)) +
        labs(title = "CHB Model Comparison", x = "Metric", y = "Score") + 
        theme_minimal() +
        ylim(0, 1)
    })
  })
  
  
  output$rocCurve <- renderPlot({
    df <- filtered_data() %>% select(ATD, CHB, M, `Alarm Tag Type`, H) %>% drop_na()
    df$CHB <- as.factor(df$CHB)
    set.seed(123)
    idx <- createDataPartition(df$CHB, p = 0.7, list = FALSE)
    train <- df[idx, ]; test <- df[-idx, ]
    
    # XGBoost model for ROC curve
    mat_train <- model.matrix(CHB ~ . -1, train)
    mat_test <- model.matrix(CHB ~ . -1, test)
    xgb_mod <- xgboost(data = xgb.DMatrix(data = mat_train, label = as.numeric(train$CHB) - 1),
                       nrounds = 50, objective = "binary:logistic", verbose = 0)
    xgb_probs <- predict(xgb_mod, xgb.DMatrix(mat_test))
    xgb_roc <- roc(response = as.numeric(as.character(test$CHB)), predictor = xgb_probs)
    
    # Plot ROC curve
    plot(xgb_roc, main = "ROC Curve (XGBoost)", col = "blue", lwd = 2)
    legend("bottomright", legend = paste("AUC =", round(as.numeric(xgb_roc$auc), 3)), col = "blue", lwd = 2)
  })
  
  output$actualLinear <- renderPlot({
    df <- filtered_data() %>% select(ATD, CHB, M, `Alarm Tag Type`, H) %>% drop_na()
    idx <- createDataPartition(df$ATD, p = 0.7, list = FALSE)
    train <- df[idx, ]; test <- df[-idx, ]
    lm_model <- lm(ATD ~ ., data = train)
    preds <- predict(lm_model, test)
    ggplot(data.frame(Actual=test$ATD, Predicted=preds), aes(x=Actual, y=Predicted)) +
      geom_point(alpha=0.6) + geom_abline(slope=1, intercept=0, linetype="dashed") + theme_minimal()
  })
  
  output$actualXGB <- renderPlot({
    df <- filtered_data() %>% select(ATD, CHB, M, `Alarm Tag Type`, H) %>% drop_na()
    idx <- createDataPartition(df$ATD, p = 0.7, list = FALSE)
    train <- df[idx, ]; test <- df[-idx, ]
    mat_train <- model.matrix(ATD ~ . -1, train)
    mat_test <- model.matrix(ATD ~ . -1, test)
    xgb_mod <- xgboost(data = xgb.DMatrix(mat_train, label = train$ATD),
                       nrounds = 50, objective = "reg:squarederror", verbose = 0)
    preds <- predict(xgb_mod, xgb.DMatrix(mat_test))
    ggplot(data.frame(Actual=test$ATD, Predicted=preds), aes(x=Actual, y=Predicted)) +
      geom_point(alpha=0.6) + geom_abline(slope=1, intercept=0, linetype="dashed") + theme_minimal()
  })
  
  output$residualPlot <- renderPlot({
    df <- filtered_data() %>% select(ATD, CHB, M, `Alarm Tag Type`, H) %>% drop_na()
    idx <- createDataPartition(df$ATD, p = 0.7, list = FALSE)
    train <- df[idx, ]; test <- df[-idx, ]
    mat_train <- model.matrix(ATD ~ . -1, train)
    mat_test <- model.matrix(ATD ~ . -1, test)
    xgb_mod <- xgboost(data = xgb.DMatrix(mat_train, label = train$ATD),
                       nrounds = 50, objective = "reg:squarederror", verbose = 0)
    preds <- predict(xgb_mod, xgb.DMatrix(mat_test))
    residuals <- test$ATD - preds
    ggplot(data.frame(Actual = test$ATD, Residual = residuals), aes(x = Actual, y = Residual)) +
      geom_point(alpha = 0.6, color = "blue") + geom_hline(yintercept = 0, linetype = "dashed") + theme_minimal()
  })
  
  output$downloadFiltered <- downloadHandler(
    filename = function() "filtered_alarm_data.csv",
    content = function(file) {
      write.csv(filtered_data(), file, row.names = FALSE)
    }
  )
}

shinyApp(ui, server)

