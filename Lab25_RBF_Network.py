"""
Lab 25 — Radial Basis Function (RBF) Network
Aim: To implement an RBF network with k-means-selected centres for classification on the
Iris dataset.
Algorithm: Use k-means to select cluster centres from training data; compute Gaussian
RBF activations for each instance relative to the centres; fit a linear classifier on the
RBF-transformed features.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
    random_state=1)

kmeans = KMeans(n_clusters=10, random_state=1, n_init=10).fit(X_train)
centres = kmeans.cluster_centers_
sigma = 1.0


def rbf_transform(X, centres, sigma):
    return np.array([[np.exp(-np.sum((x - c) ** 2) / (2 * sigma ** 2)) for c in centres]
                      for x in X])


X_train_rbf = rbf_transform(X_train, centres, sigma)
X_test_rbf = rbf_transform(X_test, centres, sigma)

clf = LogisticRegression(max_iter=1000).fit(X_train_rbf, y_train)
print("Accuracy:", accuracy_score(y_test, clf.predict(X_test_rbf)))

# Result: The RBF network (k-means centres + linear output layer) achieved ~95.6%
# accuracy on Iris classification.
