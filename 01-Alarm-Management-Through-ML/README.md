# Enhancing Alarm Intelligence through Machine Learning at OSUM Oil Sands (2025- Still Ongoing)

## Overview
This project focuses on enhancing alarm intelligence by analyzing alarm data, identifying patterns in chattering behavior (CHB), and predicting Active Time Duration (ATD) using Machine Learning models in R.

## Dataset
- **File**: IM009B-XLS-ENG.xlsx
- **Rows**: ~20,000
- **Features**: Alarm Tags, Chattering Behaviour, Active Time Duration, Timestamps

## Technologies Used
- R Programming
- Logistic Regression
- Decision Tree
- XGBoost (Best performer)
- Tidyverse, caret, readxl, ggplot2

## Key Results
- XGBoost Classifier achieved AUC = 0.91 for CHB prediction.
- XGBoost Regressor achieved lowest RMSE and MAE for ATD prediction.
- Identified ~25% alarms as chattering → crucial insight for reducing alarm fatigue.

## Business Impact
- Prioritized alarm handling based on predicted ATD and chattering probability.
- Improved operator safety, reduced alarm overload.
- Recommendations for real-time system deployment.

## Project Files
- **dataset/**: Alarm dataset (Excel file).
- **R_Notebooks/**: R scripts for classification and regression models.
- **Results/**: ROC curves, residual plots, model comparison table.
- **Presentation/**: Capstone project slides.

---
