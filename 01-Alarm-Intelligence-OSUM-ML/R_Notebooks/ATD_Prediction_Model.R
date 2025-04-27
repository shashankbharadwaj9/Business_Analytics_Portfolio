# Load libraries
library(tidyverse)
library(readxl)
library(caret)
library(xgboost)

# Load dataset
alarm_data <- read_excel("../dataset/IM009B-XLS-ENG.xlsx")

# Data Preparation
alarm_data <- alarm_data %>%
  drop_na()

# Train-Test Split
set.seed(123)
trainIndex <- createDataPartition(alarm_data$ATD, p = .7, list = FALSE)
train_data <- alarm_data[trainIndex, ]
test_data <- alarm_data[-trainIndex, ]

# Linear Regression
model_lm <- lm(ATD ~ ., data = train_data)
pred_lm <- predict(model_lm, test_data)

# XGBoost Regressor
train_matrix <- model.matrix(ATD ~ .-1, data = train_data)
test_matrix <- model.matrix(ATD ~ .-1, data = test_data)
train_label <- train_data$ATD
test_label <- test_data$ATD

model_xgb <- xgboost(data = train_matrix, label = train_label, nrounds = 100, objective = "reg:squarederror", verbose = 0)
pred_xgb <- predict(model_xgb, test_matrix)

# Compare RMSE
rmse_lm <- sqrt(mean((test_label - pred_lm)^2))
rmse_xgb <- sqrt(mean((test_label - pred_xgb)^2))

results <- data.frame(
  Model = c("Linear Regression", "XGBoost Regressor"),
  RMSE = c(rmse_lm, rmse_xgb)
)
print(results)

# Save results
write.csv(results, "../Results/atd_model_comparison.csv", row.names = FALSE)

# Residual Plot
residuals <- test_label - pred_xgb
plot(residuals, main = "Residual Plot - XGBoost ATD Prediction", ylab = "Residuals", xlab = "Index", pch = 19, col = "blue")
abline(h = 0, col = "red", lwd = 2)
