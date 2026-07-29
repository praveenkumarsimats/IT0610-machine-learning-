"""
Program 5: K-Nearest Neighbours (K-NN)
Implements and demonstrates the K-NN classification algorithm on the
Iris dataset using scikit-learn, and also shows a from-scratch version.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter


def knn_from_scratch(X_train, y_train, X_test, k=5):
    predictions = []
    for test_point in X_test:
        distances = np.linalg.norm(X_train - test_point, axis=1)
        k_indices = np.argsort(distances)[:k]
        k_labels = y_train[k_indices]
        most_common = Counter(k_labels).most_common(1)[0][0]
        predictions.append(most_common)
    return np.array(predictions)


if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target
    target_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    k = 5

    # --- scikit-learn implementation ---
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    sk_predictions = model.predict(X_test)
    sk_accuracy = accuracy_score(y_test, sk_predictions)

    print("=== scikit-learn K-NN ===")
    print(f"k = {k}")
    print(f"Accuracy: {sk_accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, sk_predictions, target_names=target_names))

    # --- from-scratch implementation ---
    scratch_predictions = knn_from_scratch(X_train, y_train, X_test, k=k)
    scratch_accuracy = accuracy_score(y_test, scratch_predictions)

    print("=== From-scratch K-NN ===")
    print(f"Accuracy: {scratch_accuracy * 100:.2f}%")

    # Classify a new sample
    new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(new_sample)
    print(f"\nNew sample {new_sample.tolist()} classified as: "
          f"{target_names[prediction[0]]}")
