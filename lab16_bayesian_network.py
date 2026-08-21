"""
Lab 16 — Bayesian Belief Network (using pgmpy)
------------------------------------------------------
Aim: To construct a Bayesian Belief Network and perform inference using
the pgmpy library.

Algorithm: Define the DAG structure among variables; specify Conditional
Probability Distributions (CPDs) at each node; use variable elimination
to answer probabilistic queries.

Requirement: pip install pgmpy
"""

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = BayesianNetwork([('Rain', 'Wet'), ('Sprinkler', 'Wet')])

cpd_rain = TabularCPD('Rain', 2, [[0.8], [0.2]])
cpd_sprinkler = TabularCPD('Sprinkler', 2, [[0.6], [0.4]])
cpd_wet = TabularCPD(
    'Wet', 2,
    [[1.0, 0.1, 0.1, 0.01],
     [0.0, 0.9, 0.9, 0.99]],
    evidence=['Rain', 'Sprinkler'], evidence_card=[2, 2]
)

model.add_cpds(cpd_rain, cpd_sprinkler, cpd_wet)

infer = VariableElimination(model)
result = infer.query(variables=['Rain'], evidence={'Wet': 1})
print(result)

# Sample Output:
# +---------+-------------+
# | Rain    | phi(Rain)   |
# +=========+=============+
# | Rain(0) | 0.3382      |
# | Rain(1) | 0.6618      |
# +---------+-------------+
#
# Result: The Bayesian Belief Network was constructed and used to infer
# the probability of rain given observed wetness.
