"""
Lab 18 — Maximum Likelihood Estimation Demonstration
--------------------------------------------------------
Aim: To estimate the parameter (mean) of a Gaussian distribution from
sample data using Maximum Likelihood Estimation and compare with
least-squares.

Algorithm: Given i.i.d. samples, the ML estimate of the mean is the
sample mean; verify this minimises the sum of squared deviations.
"""

import numpy as np

np.random.seed(1)
data = np.random.normal(loc=50, scale=5, size=200)

mle_mean = np.mean(data)
mle_var = np.var(data)
print(f"MLE Mean = {mle_mean:.3f}, MLE Variance = {mle_var:.3f}")

# Verify sample mean minimises sum of squared errors
candidates = np.linspace(45, 55, 100)
sse = [np.sum((data - c) ** 2) for c in candidates]
print("Candidate minimising SSE:", candidates[np.argmin(sse)])

# Sample Output:
# MLE Mean = 49.764, MLE Variance = 22.981
# Candidate minimising SSE: 49.747
#
# Result: The ML estimate of the mean matched the value minimising the
# sum of squared errors, confirming the Gaussian-noise least-squares
# connection.
