# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 11:43:28 2026

@author: SHAFIQUE RB
"""

# ==============================
# SCENARIO 2: POLYNOMIAL REGRESSION
# MPG vs Horsepower
# ==============================

# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load dataset
df = pd.read_csv("auto-mpg.csv")  # download from Kaggle

# 3. Clean dataset
df = df.replace("?", np.nan)
df["horsepower"] = df["horsepower"].astype(float)
df = df.dropna()

# 4. Select feature & target
X = df[["horsepower"]]
y = df["mpg"]

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Feature scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

degrees = [2, 3, 4]
results = []

plt.figure()

for d in degrees:
    # Polynomial features
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train_s)
    X_test_poly = poly.transform(X_test_s)

    # Train model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Predict
    y_pred = model.predict(X_test_poly)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append((d, mse, rmse, r2))

    # Plot curve
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_plot_s = scaler.transform(X_plot)
    X_plot_poly = poly.transform(X_plot_s)
    y_plot = model.predict(X_plot_poly)

    plt.scatter(X, y)
    plt.plot(X_plot, y_plot)
    plt.title(f"Polynomial Degree {d}")
    plt.xlabel("Horsepower")
    plt.ylabel("MPG")
    plt.show()

# Results comparison
for r in results:
    print(f"Degree {r[0]} -> MSE:{r[1]:.2f}, RMSE:{r[2]:.2f}, R2:{r[3]:.3f}")

# 7. Ridge Regularization (degree 4)
poly = PolynomialFeatures(degree=4)
X_train_poly = poly.fit_transform(X_train_s)
X_test_poly = poly.transform(X_test_s)

ridge = Ridge(alpha=10)
ridge.fit(X_train_poly, y_train)
ridge_pred = ridge.predict(X_test_poly)

print("\nRidge Degree 4 R2:", r2_score(y_test, ridge_pred))
