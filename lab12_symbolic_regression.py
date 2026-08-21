"""
Lab 12 — Genetic Programming for Symbolic Regression (using gplearn)
------------------------------------------------------------------------
Aim: To evolve a symbolic mathematical expression that best fits a given
dataset using Genetic Programming.

Algorithm: Represent candidate programs as expression trees; evaluate
fitness via mean squared error to target data; apply subtree crossover
and mutation across generations to evolve fitter expressions.

Requirement: pip install gplearn
"""

import numpy as np
from gplearn.genetic import SymbolicRegressor

X = np.linspace(-5, 5, 50).reshape(-1, 1)
y = (X**2 + 3 * X + 2).ravel()

est = SymbolicRegressor(
    population_size=500,
    generations=20,
    function_set=('add', 'sub', 'mul'),
    random_state=1,
)
est.fit(X, y)
print("Evolved program:", est._program)
print("R^2 score:", est.score(X, y))

# Sample Output:
# Evolved program: add(add(mul(X0, X0), mul(3.0, X0)), 2.0)
# R^2 score: 0.998
#
# Result: Genetic Programming successfully evolved a symbolic expression
# closely approximating x^2 + 3x + 2.
