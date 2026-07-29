"""
Program 6: Naive Bayes Algorithm
Implements Gaussian Naive Bayes classification, and displays results
using a confusion matrix and accuracy score.

Dataset: Iris dataset (sklearn built-in)
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target
    target_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = GaussianNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)
    acc = accuracy_score(y_test, predictions)

    print("=== Naive Bayes Classifier ===")
    print("\nConfusion Matrix:")
    print(cm)

    print(f"\nAccuracy: {acc * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=target_names))

    # Try plotting the confusion matrix (saved to file, no display needed)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(cmap="Blues")
        plt.title("Naive Bayes - Confusion Matrix")
        plt.savefig("naive_bayes_confusion_matrix.png")
        print("\nConfusion matrix plot saved as 'naive_bayes_confusion_matrix.png'")
    except Exception as e:
        print(f"(Plotting skipped: {e})")
