![Python Version](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![GitHub Repo Size](https://img.shields.io/github/repo-size/Durgesh013/life-expectancy-ml-project)

---

[Dataset CSV](data/Life%20Expectancy%20Data.csv)  
[Analysis Notebook](notebooks/analysis.ipynb)

---

# Life Expectancy Prediction using Machine Learning

## Project Overview

This project predicts **life expectancy of countries** using various health, economic, and demographic indicators.

Several machine learning models were tested:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest

The results showed that **linear models (Linear, Ridge, and Lasso)** performed significantly better than the Random Forest model for this dataset.

The final trained model is saved using **joblib** and can be reused for predictions.

---

# Dataset

The dataset contains information about countries and factors affecting life expectancy.

Features include:

- Country
- Year
- Adult Mortality
- Infant deaths
- Alcohol consumption
- Hepatitis B immunization
- Measles cases
- BMI
- GDP
- Population
- Schooling
- Income composition of resources
- Health expenditures
- Immunization rates
- Nutritional indicators

**Target Variable**

Life expectancy

Dataset source:
data/Life Expectancy Data.csv


---

# Data Preprocessing

The following preprocessing steps were performed:

### 1. Missing Target Values Removed

Rows where **Life expectancy** was missing were removed.

### 2. Country-wise Median Imputation

For important features:

- GDP
- Population
- Hepatitis B

Missing values were filled using **median values grouped by country**.

If a country had no values, the **global median** was used.

### 3. Median Imputation for Remaining Features

The following features were filled using **training-set median values**:

- Total expenditure
- Alcohol
- Income composition of resources
- Schooling
- Thinness indicators
- BMI
- Immunization features
- Adult mortality

### 4. Feature Encoding and Scaling

- **Categorical variable**

Status (Developed / Developing)

encoded using **OneHotEncoder**

- **Numerical features**

scaled using **StandardScaler**

These transformations were combined using a **ColumnTransformer pipeline**.

---

# Machine Learning Pipeline

The project uses a **Scikit-learn Pipeline** that combines preprocessing and model training.

Example:

python
pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", LinearRegression())
])
This ensures the same preprocessing is applied during both training and prediction.

Models Tested
Four models were evaluated:
Model	                 Method Used
Linear Regression	     Baseline model
Ridge Regression	     L2 Regularization
Lasso Regression	     L1 Regularization
Random Forest	        Ensemble Tree Model

Model Evaluation

Models were evaluated using:

R² Score

Mean Squared Error (MSE)

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

Cross Validation

Example:

cv_scores = cross_val_score(
    pipe,
    X_train,
    y_train,
    cv=5,
    scoring="r2"
)

Results
Linear Regression

Linear regression achieved strong performance using the pipeline with standardized numerical features.

Ridge Regression

Hyperparameter tuning using GridSearchCV:

params = {
    "Ridge__alpha": [0.001,0.01,0.1,1,10,100]
}

Best result:

Best alpha: 0.1
Best CV Score: 0.813
Test R²: 0.8227
Lasso Regression

Hyperparameter tuning using GridSearchCV:

params = {
    "Lasso__alpha":[0.0001,0.001,0.01,0.1,1,10]
}

Best result:

Best alpha: 0.001
Best CV Score: 0.813
Test R²: 0.8228
Random Forest

Random Forest was tested with RandomizedSearchCV for hyperparameter tuning.

Example parameters:

params_random = {
    "rf__n_estimators":[100,200,300],
    "rf__max_depth":[None,10,20],
    "rf__min_samples_split":[2,5,10],
    "rf__min_samples_leaf":[1,2,4]
}

Best model performance:

OOB Score: 0.231
Test R²: 0.232
MSE: 71.44
Observation

Random Forest performed significantly worse than linear models on this dataset.

Possible reasons:

Dataset structure may favor linear relationships

Feature scaling benefits linear models

Dataset size may be insufficient for complex tree ensembles

Feature Importance

Feature importance was derived from linear regression coefficients.

<img src="images/feature_importance_plot.png" width="600">

Important predictors include:

Schooling

Adult Mortality

GDP

Income composition of resources

BMI

Prediction Visualization

Actual vs predicted life expectancy:

<img src="images/prediction_plot.png" width="600">

Points close to the diagonal line represent accurate predictions.

Saving the Model

The trained pipeline is saved using joblib.

import joblib

joblib.dump(pipe,"models/life_expectancy_model.joblib")

This allows the model to be loaded later without retraining.

How to Use the Model
Install Dependencies
pip install pandas scikit-learn joblib matplotlib
Load the Model
import joblib

model = joblib.load("models/life_expectancy_model.joblib")
Make a Prediction
import pandas as pd

input_data = pd.DataFrame({
    "Country": ["India"],
    "Year": [2025],
    "Status": ["Developing"],
    "Adult Mortality": [150],
    "infant deaths": [30],
    "Alcohol": [4.5],
    "percentage expenditure": [8.0],
    "Hepatitis B": [95],
    "Measles": [50],
    "BMI": [22.5],
    "under-five deaths": [35],
    "Polio": [98],
    "Total expenditure": [5.0],
    "Diphtheria": [97],
    "HIV/AIDS": [0.2],
    "GDP": [2000],
    "Population": [1400000000],
    "thinness  1-19 years": [3.5],
    "thinness 5-9 years": [2.0],
    "Income composition of resources": [0.6],
    "Schooling": [12]
})

prediction = model.predict(input_data)

print(prediction)
Technologies Used

Python

NumPy

Pandas

Matplotlib

Scikit-learn

Joblib

Project Structure
life-expectancy-ml-project
│
├── data
│   └── Life Expectancy Data.csv
│
├── notebooks
│   └── analysis.ipynb
│
├── src
│   └── model.py
│
├── models
│   └── life_expectancy_model.joblib
│
├── images
│   ├── feature_importance_plot.png
│   └── prediction_plot.png
│
└── README.md
Future Improvements

Possible improvements for the project:

Test Gradient Boosting models

Try XGBoost or LightGBM

Perform advanced feature engineering

Build an interactive prediction web app

Deploy the model using Flask or FastAPI
