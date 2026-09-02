"""
fusion.py
---------
Decision-fusion mechanism combining the MLP(+GA) posterior with the
Gaussian Naive Bayes posterior via weighted (log-linear) Bayesian Model
Averaging.

Mathematical justification
---------------------------
Let p_MLP = P_MLP(y=1|x) and p_NB = P_NB(y=1|x) be two independent
estimates of the same posterior P(y=1|x), produced by models with
different, complementary bias/variance profiles (MLP: high-capacity,
discriminative, correlation-aware; NB: low-variance, well-calibrated
under class-conditional-independence, robust on small folds).

Treating the two models as an ensemble/mixture of experts, Bayesian
Model Averaging combines them as:

    P(y|x) = sum_m P(y|x, M_m) * P(M_m | data)

which, with two models and mixture weight w in [0,1] standing in for
the (validation-performance-derived) model posterior P(M_MLP | data),
reduces to the weighted arithmetic mixture:

    P_fused(y=1|x) = w * p_MLP + (1-w) * p_NB

We additionally implement a log-linear (geometric) pooling variant,
which is the Bayesian-optimal fusion rule when the two classifiers'
errors are conditionally independent given the true label (a product-
of-experts / logarithmic opinion pool):

    log P_fused(y=1|x) proportional_to  w*log p_MLP + (1-w)*log p_NB
    P_fused(y=1|x) = normalise( p_MLP^w * p_NB^(1-w) )

The mixture weight w is not fixed a priori: it is selected on a held-out
validation split by a small grid search maximising F1-score, i.e. w is
itself estimated from data, which is the empirical-Bayes analogue of
estimating P(M_MLP|data) in the BMA formulation.
"""
import numpy as np
from src.evaluation import f1_score


def weighted_arithmetic_fusion(p_mlp, p_nb, w):
    return w * p_mlp + (1 - w) * p_nb


def weighted_geometric_fusion(p_mlp, p_nb, w, eps=1e-9):
    p_mlp = np.clip(p_mlp, eps, 1 - eps)
    p_nb = np.clip(p_nb, eps, 1 - eps)
    log_p1 = w * np.log(p_mlp) + (1 - w) * np.log(p_nb)
    log_p0 = w * np.log(1 - p_mlp) + (1 - w) * np.log(1 - p_nb)
    m = np.maximum(log_p1, log_p0)
    denom = np.exp(log_p1 - m) + np.exp(log_p0 - m)
    p1 = np.exp(log_p1 - m) / denom
    return p1


def fit_fusion_weight(p_mlp_val, p_nb_val, y_val, mode="arithmetic",
                       w_grid=None):
    """Grid-search the mixture weight w on a validation split, maximising
    F1-score. This is the empirical-Bayes estimate of P(M_MLP | data)."""
    if w_grid is None:
        w_grid = np.linspace(0, 1, 21)
    fuse_fn = weighted_arithmetic_fusion if mode == "arithmetic" else weighted_geometric_fusion
    best_w, best_f1 = 0.5, -1
    for w in w_grid:
        p_fused = fuse_fn(p_mlp_val, p_nb_val, w)
        pred = (p_fused >= 0.5).astype(int)
        f1 = f1_score(y_val, pred)
        if f1 > best_f1:
            best_f1, best_w = f1, w
    return best_w, best_f1


def fuse(p_mlp, p_nb, w, mode="arithmetic"):
    if mode == "arithmetic":
        return weighted_arithmetic_fusion(p_mlp, p_nb, w)
    return weighted_geometric_fusion(p_mlp, p_nb, w)
