"""
Lab 27 — NumPy Array Creation and Basic Operations
Aim: To create NumPy arrays using various methods and perform basic mathematical
operations.
Algorithm: Use np.array, np.zeros, np.ones, np.arange, np.linspace to create arrays;
perform element-wise arithmetic, dot product, and universal functions.
"""

import numpy as np

a = np.array([1, 2, 3, 4])
b = np.zeros((3, 3))
c = np.ones((2, 4))
d = np.arange(0, 10, 2)
e = np.linspace(0, 1, 5)

print("a:", a, "shape:", a.shape, "dtype:", a.dtype, "ndim:", a.ndim)
print("zeros:\n", b)
print("arange:", d)
print("linspace:", e)

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
print("x+y:", x + y, " x*y:", x * y, " dot:", np.dot(x, y))
print("sqrt(x):", np.sqrt(x), " exp(x):", np.exp(x))

# Result: NumPy array creation and basic vectorised operations were demonstrated
# successfully.
