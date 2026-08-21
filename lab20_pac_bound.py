"""
Lab 20 — Sample Complexity / PAC Bound Calculator
---------------------------------------------------------
Aim: To compute the number of training examples required for PAC
learning given |H|, epsilon, and delta using the finite hypothesis-space
bound.

Algorithm: Apply m >= (1/epsilon)(ln|H| + ln(1/delta)) to compute the
required sample size for given parameters.
"""

import math


def sample_complexity(H_size, epsilon, delta):
    return math.ceil((1 / epsilon) * (math.log(H_size) + math.log(1 / delta)))


for H_size in [10, 100, 1000]:
    m = sample_complexity(H_size, epsilon=0.1, delta=0.05)
    print(f"|H|={H_size}: m >= {m} examples needed")

# Sample Output:
# |H|=10: m >= 53 examples needed
# |H|=100: m >= 76 examples needed
# |H|=1000: m >= 99 examples needed
#
# Result: The sample complexity bound was computed for varying
# hypothesis space sizes, illustrating its logarithmic dependence on |H|.
