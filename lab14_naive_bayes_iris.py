"""
Lab 14 — Naive Bayes Classifier (Iris Dataset)
----------------------------------------------------
Aim: To implement the Naive Bayes classifier using scikit-learn and
evaluate its accuracy on the Iris dataset.

Algorithm: Estimate class priors and per-attribute Gaussian likelihoods
from training data; classify test instances using argmax of posterior
scores.
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

nb = GaussianNB().fit(X_train, y_train)
pred = nb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

# Sample Output:
# Accuracy: 0.9473684210526315
# Confusion Matrix:
# [[13 0 0]
#  [ 0 15 1]
#  [ 0 1 8]]
#
# Result: The Gaussian Naive Bayes classifier achieved approximately
# 94.7% accuracy on the Iris dataset.
