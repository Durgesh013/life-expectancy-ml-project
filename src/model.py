import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv("../data/Life Expectancy Data.csv")

df.columns = df.columns.str.strip()

df = df.dropna(subset = ["Life expectancy"])

X, y = df.drop(columns = ["Life expectancy"]), df["Life expectancy"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 13)

for col in ["GDP", "Population", "Hepatitis B"]:
    X_train[col] = X_train.groupby("Country")[col].transform(
        lambda x: x.fillna(x.median()))
    global_median = X_train[col].median()
    X_train[col] = X_train[col].fillna(global_median)
    medians = X_train.groupby("Country")[col].median()
    X_test[col] = X_test[col].fillna(X_test["Country"].map(medians))
    X_test[col] = X_test[col].fillna(global_median)

remaining_cols = [
    "Total expenditure",
    "Alcohol",
    "Income composition of resources",
    "Schooling",
    "thinness 5-9 years",
    "thinness  1-19 years",
    "BMI",
    "Diphtheria",
    "Polio",
    "Adult Mortality"
]

for col in remaining_cols:
    median = X_train[col].median()
    X_train[col] = X_train[col].fillna(median)
    X_test[col] = X_test[col].fillna(median)

categorical_cols = ["Status"]
numeric_cols = [
   'Year','Adult Mortality',
   'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
   'Measles', 'BMI', 'under-five deaths', 'Polio', 'Total expenditure',
   'Diphtheria', 'HIV/AIDS', 'GDP', 'Population',
   'thinness  1-19 years', 'thinness 5-9 years',
   'Income composition of resources', 'Schooling'
]

preprocessor = ColumnTransformer(
    transformers = [
        ("cat", OneHotEncoder(
            drop = "if_binary",
            handle_unknown = "ignore",
            # sprase_output = False
        ), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ]
)

pipe = Pipeline([
    ("preprocessing", preprocessor), 
    ("regressor", LinearRegression()),
])

cv_scores = cross_val_score(
    pipe, X_train, y_train, cv = 5, scoring = "r2")
print(f"CV Scores: {cv_scores}")
print(f"Mean CV Scores: {np.mean(cv_scores)}")
print(f"CV std: {np.std(cv_scores)}")

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
mse  = mean_squared_error(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred, alpha=0.5)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linewidth=2
)

plt.xlabel("Actual Life Expectancy")
plt.ylabel("Predicted Life Expectancy")
plt.title("Actual vs Predicted Life Expectancy")
plt.grid()

plt.tight_layout()
plt.savefig("../images/prediction_plot.png", dpi=300)
plt.show()

linear = pipe.named_steps["regressor"]
importance = linear.coef_[0]

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

print(feature_importance)

import matplotlib.pyplot as plt

feature_importance = feature_importance.sort_values(by="Importance")

plt.figure(figsize=(8,6))
plt.barh(feature_importance["Feature"], feature_importance["Importance"])
plt.title("Logistic Regression Feature Importance")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig("../images/feature_importance_plot.png", dpi=300)
plt.show()

import joblib
joblib.dump(pipe, "../models/life_expectancy_model.joblib")