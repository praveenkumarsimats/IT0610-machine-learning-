"""
Lab 24 — Locally Weighted Regression (LWR) from Scratch
Aim: To implement Locally Weighted Regression to fit a non-linear curve using a Gaussian
kernel-weighted local linear model.
Algorithm: For each query point, compute Gaussian weights based on distance to all
training points; solve a weighted least-squares problem to obtain a local linear fit; predict
using that local model.
"""

import numpy as np
import matplotlib.pyplot as plt


def lwr(x_query, X, y, tau=0.5):
    m = X.shape[0]
    W = np.exp(-((X - x_query) ** 2) / (2 * tau ** 2))
    Xb = np.c_[np.ones(m), X]
    W = np.diag(W)
    theta = np.linalg.pinv(Xb.T @ W @ Xb) @ Xb.T @ W @ y
    return np.array([1, x_query]) @ theta


np.random.seed(1)
X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(0, 0.1, 100)

y_pred = np.array([lwr(x, X, y, tau=0.5) for x in X])

plt.scatter(X, y, s=10, label='data')
plt.plot(X, y_pred, color='red', label='LWR fit')
plt.legend()
plt.savefig('lwr_fit.png')

print("Sample predictions:", y_pred[:5])

# Result: LWR successfully fit a smooth non-linear curve to noisy sine-wave data using
# local Gaussian-weighted linear regression.
