"""
Lab 31 — Saving and Loading Data with NumPy
Aim: To save NumPy arrays to disk in binary and text formats, and reload them for later
use.
Algorithm: Use np.save/np.load for binary .npy storage and np.savetxt/np.loadtxt for
plain-text CSV storage.
"""

import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])

np.save('array.npy', a)
np.savetxt('array.csv', a, delimiter=',')

loaded_npy = np.load('array.npy')
loaded_csv = np.loadtxt('array.csv', delimiter=',')

print("Loaded from .npy:\n", loaded_npy)
print("Loaded from .csv:\n", loaded_csv)

# Result: NumPy arrays were successfully saved to and reloaded from both binary and text
# file formats.
