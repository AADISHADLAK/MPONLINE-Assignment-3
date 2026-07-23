"""
Assignment 3: Salary Prediction using Polynomial Regression
Name: AADISH ADLAK
Registration Number: 23BCE10681
Application Number: IN26010985
Batch Number: 9A
Email: adlakaadish@gmail.com

Dataset: Position Salaries Dataset (Kaggle)
https://www.kaggle.com/datasets/akram24/position-salaries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

# ---------------------------------------------------------------------------
# Task 1: Data Understanding
# ---------------------------------------------------------------------------
df = pd.read_csv("data/Position_Salaries.csv")

print("=" * 70)
print("TASK 1: DATA UNDERSTANDING")
print("=" * 70)

print("\nFirst 5 records:")
print(df.head())

print("\nInput Feature : Level")
print("Target Variable : Salary")

print("\nDataset Info:")
df.info()

print("\nSummary Statistics:")
print(df.describe())

# ---------------------------------------------------------------------------
# Task 2: Data Preprocessing
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 2: DATA PREPROCESSING")
print("=" * 70)

print("\nMissing values in each column:")
print(df.isnull().sum())

X = df[['Level']].values   # Input feature
y = df['Salary'].values    # Target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")

# ---------------------------------------------------------------------------
# Task 3: Model Development
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 3: MODEL DEVELOPMENT")
print("=" * 70)

poly_reg = PolynomialFeatures(degree=3)
X_train_poly = poly_reg.fit_transform(X_train)
X_test_poly = poly_reg.transform(X_test)

print(f"\nOriginal feature shape         : {X_train.shape}")
print(f"Polynomial feature shape (deg=3): {X_train_poly.shape}")

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

print(f"\nModel coefficients: {poly_model.coef_}")
print(f"Model intercept   : {poly_model.intercept_}")

y_pred = poly_model.predict(X_test_poly)

comparison = pd.DataFrame({
    "Actual Salary": y_test,
    "Predicted Salary": np.round(y_pred, 2)
})
print("\nActual vs Predicted Salary (Test Set):")
print(comparison)

# ---------------------------------------------------------------------------
# Task 4: Model Evaluation
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 4: MODEL EVALUATION")
print("=" * 70)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Absolute Error (MAE): {mae:,.2f}")
print(f"Mean Squared Error (MSE) : {mse:,.2f}")
print(f"R2 Score                 : {r2:.4f}")

# Scatter plot of original data
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='red')
plt.title('Original Data: Position Level vs Salary')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('images/original_data_scatter.png', dpi=150)
plt.close()

# Scatter plot + Polynomial Regression curve
X_grid = np.arange(min(X.min(), 1), max(X.max(), 10) + 0.1, 0.1).reshape(-1, 1)
X_grid_poly = poly_reg.transform(X_grid)
y_grid_pred = poly_model.predict(X_grid_poly)

plt.figure(figsize=(9, 6))
plt.scatter(X, y, color='red', label='Actual Data (all points)')
plt.scatter(X_test, y_test, color='green', marker='s', s=90, label='Test Data', zorder=5)
plt.plot(X_grid, y_grid_pred, color='blue', label='Polynomial Regression Curve (degree=3)')
plt.title('Position Level vs Salary - Polynomial Regression (Degree 3)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('images/polynomial_regression_curve.png', dpi=150)
plt.close()

print("\nPlots saved to images/original_data_scatter.png and "
      "images/polynomial_regression_curve.png")

print("""
Observations:
1. The Polynomial Regression curve (degree 3) closely follows the sharp,
   non-linear rise in salary at higher position levels, which a straight
   line (simple Linear Regression) would fail to capture.
2. Because the dataset has only 10 rows, the 80/20 split produces a very
   small test set, so the MAE, MSE, and R2 values should be interpreted
   as illustrative rather than statistically robust.
3. The largest prediction errors tend to occur near the highest position
   levels (e.g., CEO), where salary grows extremely fast (non-linearly).
""")

# ---------------------------------------------------------------------------
# Task 5: Conclusion
# ---------------------------------------------------------------------------
print("=" * 70)
print("TASK 5: CONCLUSION")
print("=" * 70)
print("""
The Polynomial Regression model (degree 3) captured the non-linear
relationship between an employee's position level and their salary far
better than a straight-line fit would. Salary increases slowly at lower
levels but accelerates sharply at senior levels (e.g., Partner, C-level,
CEO), a pattern that Linear Regression -- which assumes a constant rate
of change -- cannot represent well.

Linear Regression vs Polynomial Regression: Linear Regression fits a
single straight line (y = b0 + b1*x) and assumes a constant relationship
between input and output. Polynomial Regression extends this by adding
higher-degree terms (x^2, x^3, ...), allowing the model to fit curves and
capture non-linear trends while still being solved using linear
regression techniques on the transformed features.

Advantage for this dataset: Since salary rises exponentially with
seniority rather than linearly, Polynomial Regression provides a much
closer, more accurate fit to the actual salary curve, significantly
improving prediction accuracy for higher position levels compared to a
simple linear model.
""")
