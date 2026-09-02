"""
Lab 23 — Distance-Weighted KNN
Aim: To implement distance-weighted KNN (weight = 1/d^2) and compare its
performance with uniform-weighted KNN.
Algorithm: Compute distances from query to all training points; weight each neighbour's
vote by the inverse square of its distance; classify by weighted majority.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
    random_state=1)

uniform = KNeighborsClassifier(n_neighbors=5, weights='uniform').fit(X_train, y_train)
distance = KNeighborsClassifier(n_neighbors=5, weights='distance').fit(X_train, y_train)

print("Uniform-weighted accuracy:", accuracy_score(y_test, uniform.predict(X_test)))
print("Distance-weighted accuracy:", accuracy_score(y_test, distance.predict(X_test)))

# Result: Both weighting schemes were compared; distance weighting can improve
# robustness on noisier datasets.
