# Life Expectancy Prediction using Machine Learning

## Project Overview
This project predicts **life expectancy of countries** using machine learning techniques.  
The model analyzes health, economic, and demographic indicators to estimate life expectancy.

A **Linear Regression model** is trained on the dataset after performing preprocessing steps such as handling missing values, encoding categorical variables, and scaling numerical features.

---

## Dataset
The dataset used in this project is **Life Expectancy Data** collected from multiple countries and years.

### Main Features
- Country
- Year
- Adult Mortality
- Infant deaths
- Alcohol consumption
- Hepatitis B immunization
- GDP
- Population
- BMI
- Schooling
- Income composition of resources
- Status (Developed / Developing)

**Target Variable:**  
Life expectancy

---

## Project Workflow

### 1. Data Loading
The dataset is loaded using **pandas**.

### 2. Data Cleaning
- Removed rows where life expectancy was missing.
- Missing values were handled using:
  - Country-wise median imputation
  - Global median imputation when country values were unavailable.

### 3. Feature Preprocessing

Two types of preprocessing were applied.

**Categorical Features**
- One-hot encoding applied to the `Status` column.

**Numerical Features**
- Standard scaling applied using a scaler.

This was implemented using **ColumnTransformer**.

### 4. Model Pipeline
A machine learning pipeline was created containing:

- Preprocessing  
- Linear Regression model  

Using a pipeline ensures preprocessing and modeling are applied consistently.

### 5. Model Evaluation

The model was evaluated using:

- Cross-validation (5-fold)
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Model Performance

Evaluation metrics used:

- **MAE** – Average absolute prediction error
- **MSE** – Average squared error
- **RMSE** – Square root of MSE
- **R² Score** – Proportion of variance explained by the model

The model was also validated using **5-fold cross-validation**.

---

## Visualization
The project includes a visualization comparing **actual vs predicted life expectancy values**.

Points closer to the **red diagonal line** indicate more accurate predictions.

![Prediction Plot](images/prediction_plot.png)

---

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Project Structure

life-expectancy-ml-project
│
├── data
│ └── Life Expectancy Data.csv
│
├── notebooks
│ └── analysis.ipynb
│
├── src
│ └── model.py
│
├── images
│ └── prediction_plot.png
│
└── README.md


---

## Future Improvements

Possible improvements for the project:

- Try advanced models like Random Forest or Gradient Boosting
- Perform feature importance analysis
- Hyperparameter tuning
- Deploy the model as a web application
