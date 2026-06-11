# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 10:58:41 2026

@author: SHAFIQUE RB
"""
# ==============================
# SCENARIO 1: MULTILINEAR REGRESSION
# Student Performance Prediction
# ==============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load dataset
df = pd.read_csv("StudentsPerformance.csv")  # download from Kaggle

# 3. Create target variable (Final Score)
df["FinalScore"] = df[["math score", "reading score", "writing score"]].mean(axis=1)

# 4. Create required features
np.random.seed(42)
df["StudyHours"] = np.random.uniform(1, 5, len(df))
df["Attendance"] = np.random.uniform(70, 100, len(df))
df["SleepHours"] = np.random.uniform(5, 9, len(df))

# 5. Encode categorical features
le = LabelEncoder()
df["ParentalEdu"] = le.fit_transform(df["parental level of education"])
df["TestPrep"] = le.fit_transform(df["test preparation course"])

# 6. Select features & target
X = df[["StudyHours", "Attendance", "ParentalEdu", "TestPrep", "SleepHours"]]
y = df["FinalScore"]

# 7. Handle missing values (mean imputation)
X = X.fillna(X.mean())

# 8. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 9. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 10. Train Multilinear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# 11. Prediction
y_pred = model.predict(X_test)

# 12. Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("=== Linear Regression Performance ===")
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)

# 13. Coefficient analysis
coeff_df = pd.DataFrame({
    "Feature": ["StudyHours", "Attendance", "ParentalEdu", "TestPrep", "SleepHours"],
    "Coefficient": model.coef_
})
print("\nCoefficients:\n", coeff_df)

# 14. Feature elimination (remove low coefficient)
selected_features = coeff_df[abs(coeff_df["Coefficient"]) > 0.05]["Feature"].values
print("\nSelected Features:", selected_features)

# 15. Ridge & Lasso
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)

ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

ridge_pred = ridge.predict(X_test)
lasso_pred = lasso.predict(X_test)

print("\n=== Ridge R2:", r2_score(y_test, ridge_pred))
print("=== Lasso R2:", r2_score(y_test, lasso_pred))

# ==============================
# VISUALIZATION
# ==============================

# Predicted vs Actual
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Predicted vs Actual")
plt.show()

# Coefficient comparison
sns.barplot(x="Feature", y="Coefficient", data=coeff_df)
plt.title("Coefficient Magnitude")
plt.show()

# Residual plot
residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()

