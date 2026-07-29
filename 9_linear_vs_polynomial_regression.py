"""
Program 9: Compare Linear and Polynomial Regression
Fits both a simple linear regression model and a polynomial regression
model on the same non-linear dataset and compares their performance.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score


if __name__ == "__main__":
    # Synthetic non-linear dataset
    rng = np.random.RandomState(0)
    X = np.sort(rng.uniform(-3, 3, 60)).reshape(-1, 1)
    y = 0.5 * X.flatten() ** 3 - X.flatten() ** 2 + 2 + rng.normal(0, 2, X.shape[0])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Linear Regression ---
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    lin_pred = lin_model.predict(X_test)
    lin_mse = mean_squared_error(y_test, lin_pred)
    lin_r2 = r2_score(y_test, lin_pred)

    # --- Polynomial Regression (degree 3) ---
    degree = 3
    poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    poly_model.fit(X_train, y_train)
    poly_pred = poly_model.predict(X_test)
    poly_mse = mean_squared_error(y_test, poly_pred)
    poly_r2 = r2_score(y_test, poly_pred)

    print("=== Linear Regression ===")
    print(f"MSE: {lin_mse:.4f}, R^2: {lin_r2:.4f}")

    print("\n=== Polynomial Regression (degree=3) ===")
    print(f"MSE: {poly_mse:.4f}, R^2: {poly_r2:.4f}")

    print("\nConclusion:")
    if poly_r2 > lin_r2:
        print("Polynomial regression fits this non-linear data better than "
              "linear regression.")
    else:
        print("Linear regression performs comparably or better on this data.")

    # Plot comparison
    X_plot = np.linspace(-3, 3, 200).reshape(-1, 1)
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color="gray", alpha=0.6, label="Data")
    plt.plot(X_plot, lin_model.predict(X_plot), color="blue",
             label="Linear Regression", linewidth=2)
    plt.plot(X_plot, poly_model.predict(X_plot), color="red",
             label=f"Polynomial Regression (deg={degree})", linewidth=2)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear vs Polynomial Regression")
    plt.legend()
    plt.tight_layout()
    plt.savefig("linear_vs_polynomial_regression.png")
    print("\nPlot saved as 'linear_vs_polynomial_regression.png'")
