"""
Program 8: Linear Regression Algorithm
Implements simple linear regression using scikit-learn on a synthetic
dataset (e.g., years of experience vs salary) and visualizes the fit.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


if __name__ == "__main__":
    # Synthetic dataset: Years of Experience vs Salary (in thousands)
    rng = np.random.RandomState(42)
    X = np.sort(rng.uniform(0, 10, 40)).reshape(-1, 1)  # years of experience
    y = 25 + 9 * X.flatten() + rng.normal(0, 5, size=X.shape[0])  # salary (k)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("=== Linear Regression ===")
    print(f"Coefficient (slope): {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    # Predict for a new value
    new_experience = np.array([[7.5]])
    predicted_salary = model.predict(new_experience)[0]
    print(f"\nPredicted salary for {new_experience[0][0]} years experience: "
          f"{predicted_salary:.2f}k")

    # Plot
    plt.figure(figsize=(7, 5))
    plt.scatter(X, y, color="blue", label="Data")
    plt.plot(X, model.predict(X), color="red", linewidth=2, label="Regression line")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary (in thousands)")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig("linear_regression_plot.png")
    print("\nPlot saved as 'linear_regression_plot.png'")
