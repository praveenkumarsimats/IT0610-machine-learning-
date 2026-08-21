"""
Lab 17 — EM Algorithm for Gaussian Mixture Models
---------------------------------------------------------
Aim: To implement the EM algorithm to fit a Gaussian Mixture Model
(GMM) to clustered data with hidden component labels.

Algorithm: E-step: compute expected responsibility of each Gaussian
component for each point; M-step: re-estimate means, covariances, and
mixing weights; iterate until convergence.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.7, random_state=1)

gmm = GaussianMixture(n_components=3, random_state=1).fit(X)
labels = gmm.predict(X)

print("Converged:", gmm.converged_, " Iterations:", gmm.n_iter_)
print("Estimated means:\n", gmm.means_)

# Sample Output:
# Converged: True  Iterations: 8
# Estimated means:
# [[-2.65  9.02]
#  [ 4.72  2.05]
#  [ 0.98 -3.11]]
#
# Result: The EM algorithm successfully converged, estimating the
# Gaussian mixture parameters for the hidden cluster structure.
