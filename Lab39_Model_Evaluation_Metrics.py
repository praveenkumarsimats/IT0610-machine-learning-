"""
Lab 39 — Model Evaluation Metrics (Confusion Matrix, Precision, Recall, F1)
Aim: To compute and interpret standard classification evaluation metrics for a trained
model.
Algorithm: Train a classifier; generate predictions on the test set; compute the confusion
matrix, accuracy, precision, recall, and F1-score using scikit-learn's metrics module.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score,
                              precision_score, recall_score, f1_score)

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier(random_state=1).fit(X_train, y_train)
pred = clf.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print("Accuracy:", accuracy_score(y_test, pred))
print("Precision (macro):", precision_score(y_test, pred, average='macro'))
print("Recall (macro):", recall_score(y_test, pred, average='macro'))
print("F1-score (macro):", f1_score(y_test, pred, average='macro'))

# Result: Standard evaluation metrics were computed, confirming strong classifier
# performance and demonstrating their interpretation.
