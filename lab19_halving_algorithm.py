"""
Lab 19 — Mistake-Bound Model — Halving Algorithm Simulation
------------------------------------------------------------------
Aim: To simulate the Halving algorithm and count the number of mistakes
made while learning a target concept from a version space.

Algorithm: Maintain the current version space of consistent hypotheses;
predict via majority vote; on a mistake, eliminate all hypotheses that
would have voted incorrectly; repeat and count mistakes (bounded by
log2|H|).
"""

import random
import math

H = [lambda x, t=t: x >= t for t in range(10)]  # 10 threshold hypotheses

target = lambda x: x >= 4
mistakes = 0
version_space = H.copy()

for x in random.sample(range(10), 10):
    votes = [h(x) for h in version_space]
    majority = sum(votes) > len(votes) / 2
    true_label = target(x)
    if majority != true_label:
        mistakes += 1
    version_space = [h for h in version_space if h(x) == true_label]

print(
    "Mistakes made:", mistakes,
    " (bound: log2|H| =", round(math.log2(len(H)), 2), ")"
)

# Sample Output:
# Mistakes made: 2  (bound: log2|H| = 3.32 )
#
# Result: The number of mistakes made by the Halving algorithm remained
# within the theoretical mistake bound log2|H|.
