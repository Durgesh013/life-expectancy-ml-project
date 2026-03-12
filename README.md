![Python Version](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![GitHub Repo Size](https://img.shields.io/github/repo-size/Durgesh013/life-expectancy-ml-project)
[Dataset CSV](data/Life%20Expectancy%20Data.csv)
[Analysis Notebook](notebooks/analysis.ipynb)
# Life Expectancy Prediction using Machine Learning

## Project Overview

This project predicts **life expectancy of countries** using health, economic, and demographic indicators.  
Uses a **Linear Regression pipeline** to ensure consistent preprocessing and predictions.  
The trained model is saved locally using **joblib**.

---

## Dataset

**Life Expectancy Data** with features such as:

- Country, Year, Adult Mortality, Infant deaths, Alcohol consumption  
- Hepatitis B immunization, GDP, Population, BMI, Schooling  
- Income composition, Status (Developed / Developing)

**Target Variable:** Life expectancy

---

## Workflow

1. **Data Loading:** Using `pandas`.  
2. **Data Cleaning:** Remove missing target values, impute missing data (country-wise/global median).  
3. **Feature Preprocessing:**  
   - Categorical → One-hot encoding (`Status`)  
   - Numerical → Standard scaling  
4. **Pipeline:** Combines preprocessing + Linear Regression model.  


import joblib
# Save trained pipeline
joblib.dump(pipe, "models/life_expectancy_model.joblib")

Model Evaluation: MAE, MSE, RMSE, R² Score, 5-fold cross-validation.

Feature Importance

Feature importance shows which variables most influence life expectancy:

<img src="images/feature_importance_plot.png" alt="Feature Importance" width="600">

Key factors include:

Schooling

Adult Mortality

GDP

Income composition

BMI

Predictions

Visual comparison of actual vs predicted life expectancy:

<img src="images/prediction_plot.png" alt="Prediction Plot" width="600">

Points near the diagonal line indicate accurate predictions.

Usage

Install dependencies:

pip install pandas scikit-learn joblib

Load model:

import joblib
model = joblib.load("models/life_expectancy_model.joblib")

Predict:

import pandas as pd

input_data = pd.DataFrame({
    "Country": ["India"],
    "Year": [2025],
    "Adult Mortality": [150],
    "Infant deaths": [30],
    "Alcohol consumption": [4.5],
    "Hepatitis B immunization": [95],
    "GDP": [2000],
    "Population": [1400000000],
    "BMI": [22.5],
    "Schooling": [12],
    "Income composition of resources": [0.6],
    "Status": ["Developing"]
})

predicted_life_expectancy = model.predict(input_data)
print(predicted_life_expectancy)
Technologies Used

Python, NumPy, Pandas, Matplotlib, Scikit-learn, Joblib

Project Structure
life-expectancy-ml-project
│
├── data/Life Expectancy Data.csv
├── notebooks/analysis.ipynb
├── src/model.py
├── models/life_expectancy_model.joblib
├── images/feature_importance_plot.png
├── images/prediction_plot.png
└── README.md
Future Improvements

Try advanced models (Random Forest, Gradient Boosting)

Hyperparameter tuning

Feature selection techniques

Build a web application for prediction

Deploy using Flask or FastAPI
