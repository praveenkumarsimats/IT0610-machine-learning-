"""
Lab 30 — NumPy Random Module for Simulation
Aim: To generate random numbers using NumPy's random module for simulation and
sampling tasks.
Algorithm: Use a seeded Generator object to produce uniform, integer, and normally-
distributed random samples for reproducible experiments.
"""

import numpy as np

rng = np.random.default_rng(seed=42)

print("Uniform random 2x3:\n", rng.random((2, 3)))
print("Random integers 0-10:", rng.integers(0, 10, size=5))
print("Gaussian samples:", rng.normal(0, 1, size=5))

# Simple Monte Carlo estimate of pi
n = 100000
points = rng.random((n, 2))
inside = np.sum(points[:, 0] ** 2 + points[:, 1] ** 2 <= 1)
print("Estimated pi:", 4 * inside / n)

# Result: NumPy's random module was used for reproducible random sampling and a Monte
# Carlo simulation to estimate pi.
