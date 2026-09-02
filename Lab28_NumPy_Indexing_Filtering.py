"""
Lab 28 — NumPy Indexing and Filtering
Aim: To demonstrate slicing, indexing, and boolean filtering on multi-dimensional NumPy
arrays.
Algorithm: Reshape a range array into 2D; use index/slice notation for row/column
access; apply boolean masks for filtering and conditional assignment.
"""

import numpy as np

a = np.arange(12).reshape(3, 4)
print("Array:\n", a)
print("a[1,2]:", a[1, 2])
print("Column 1:", a[:, 1])
print("Elements > 5:", a[a > 5])

a[a % 2 == 0] = 0
print("After zeroing evens:\n", a)

# Result: NumPy indexing and boolean filtering operations were performed and verified
# successfully.
