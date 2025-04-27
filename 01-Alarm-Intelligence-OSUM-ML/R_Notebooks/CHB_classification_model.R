# Load necessary libraries
library(tidyverse)
library(readxl)
library(caret)
library(xgboost)
library(pROC)
library(rpart)

# Load dataset
alarm_data <- read_excel("E:/dataset/IM009B-XLS-ENG.xlsx")

# Data Preparation
alarm_data <- alarm_data %>%
  mutate(CHB = as.factor(CHB)) %>%
  drop_na()

# Train-Test Split
set.seed(123)
trainIndex <- createDataPartition(alarm_data$CHB, p = .7, list = FALSE)
train_data <- alarm_data[trainIndex, ]
test_data <- alarm_data[-trainIndex, ]

# Logistic Regression
model_logit <- glm(CHB ~ ., data = train_data, family = binomial)
pred_logit <- predict(model_logit, test_data, type = "response")
roc_logit <- roc(test_data$CHB, pred_logit)

# Decision Tree
model_tree <- rpart(CHB ~ ., data = train_data, method = "class")
pred_tree <- predict(model_tree, test_data, type = "prob")[,2]
roc_tree <- roc(test_data$CHB, pred_tree)

# XGBoost
train_matrix <- model.matrix(CHB ~ .-1, data = train_data)
test_matrix <- model.matrix(CHB ~ .-1, data = test_data)
train_label <- as.numeric(train_data$CHB) - 1
test_label <- as.numeric(test_data$CHB) - 1

model_xgb <- xgboost(data = train_matrix, label = train_label, nrounds = 100, objective = "binary:logistic", verbose = 0)
pred_xgb <- predict(model_xgb, test_matrix)
roc_xgb <- roc(test_data$CHB, pred_xgb)

# Plot ROC Curves
plot(roc_logit, col = "blue", main = "ROC Curves for CHB Classification")
plot(roc_tree, col = "red", add = TRUE)
plot(roc_xgb, col = "green", add = TRUE)
legend("bottomright", legend = c("Logistic Regression", "Decision Tree", "XGBoost"), col = c("blue", "red", "green"), lwd = 2)

# AUC Values
auc_values <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "XGBoost"),
  AUC = c(auc(roc_logit), auc(roc_tree), auc(roc_xgb))
)
print(auc_values)

# Save AUC table
write.csv(auc_values, "../Results/chb_model_auc_results.csv", row.names = FALSE)
