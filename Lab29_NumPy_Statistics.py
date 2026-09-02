"""
Lab 29 — NumPy Statistics and Aggregation
Aim: To compute descriptive statistics (mean, sum, standard deviation, min, max) using
NumPy aggregation functions.
Algorithm: Apply .mean(), .sum(axis=...), .std(), .min(), .max() on a 2D array along
different axes.
"""

import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])

print("Mean:", a.mean())
print("Sum along axis 0 (columns):", a.sum(axis=0))
print("Sum along axis 1 (rows):", a.sum(axis=1))
print("Std Dev:", a.std())
print("Min per row:", a.min(axis=1))
print("Max overall:", a.max())

# Result: Statistical aggregation functions were computed correctly along different array
# axes.
