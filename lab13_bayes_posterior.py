"""
Lab 13 — Bayes Theorem — Posterior Probability Calculator
--------------------------------------------------------------
Aim: To compute the posterior probability P(h|D) using Bayes theorem
given prior, likelihood, and evidence.

Algorithm: Apply P(h|D) = P(D|h)*P(h) / P(D) directly using
given/estimated probabilities.
"""


def bayes(prior, likelihood, evidence):
    return (likelihood * prior) / evidence


# Example: disease testing
p_disease = 0.01
p_pos_given_disease = 0.99
p_pos = p_pos_given_disease * p_disease + 0.05 * (1 - p_disease)  # false positive rate 5%
posterior = bayes(p_disease, p_pos_given_disease, p_pos)
print(f"P(Disease | Positive Test) = {posterior:.4f}")

# Sample Output:
# P(Disease | Positive Test) = 0.1667
#
# Result: Bayes theorem correctly computed the posterior probability,
# illustrating the base-rate effect in diagnostic testing.
