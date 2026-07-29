"""
Program 7: Logistic Regression Algorithm
Implements logistic regression for binary classification using the
Breast Cancer Wisconsin dataset (built into scikit-learn).
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


if __name__ == "__main__":
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    print("=== Logistic Regression ===")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=data.target_names))

    # Classify a new (sample) patient record - using first test sample as demo
    sample = X_test[0].reshape(1, -1)
    pred_class = model.predict(sample)[0]
    print(f"\nSample prediction: {data.target_names[pred_class]} "
          f"(actual: {data.target_names[y_test[0]]})")
