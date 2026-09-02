"""
evaluation.py
-------------
Evaluation metrics (accuracy, precision, recall, F1, ROC-AUC) and a
paired statistical significance test (paired t-test across CV folds),
all implemented from first principles using only NumPy.
"""
import numpy as np


def confusion_counts(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn


def accuracy(y_true, y_pred):
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    return (tp + tn) / max(tp + tn + fp + fn, 1)


def precision(y_true, y_pred):
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def roc_auc(y_true, y_scores):
    """Computed from first principles as the probability that a randomly
    chosen positive example is scored higher than a randomly chosen
    negative example (Mann-Whitney U statistic formulation), which is
    equivalent to the area under the ROC curve."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    pos_scores = y_scores[y_true == 1]
    neg_scores = y_scores[y_true == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    # rank-based Mann-Whitney U (handles ties)
    all_scores = np.concatenate([pos_scores, neg_scores])
    ranks = np.empty(len(all_scores))
    order = np.argsort(all_scores)
    sorted_scores = all_scores[order]
    ranks_sorted = np.empty(len(all_scores))
    i = 0
    r = 1
    while i < len(sorted_scores):
        j = i
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (r + (r + (j - i) - 1)) / 2.0
        ranks_sorted[i:j] = avg_rank
        r += (j - i)
        i = j
    ranks[order] = ranks_sorted
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    sum_ranks_pos = ranks[:n_pos].sum()
    u_stat = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    auc = u_stat / (n_pos * n_neg)
    return auc


def evaluate_all(y_true, y_pred, y_scores):
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc(y_true, y_scores),
    }


def paired_t_test(sample_a, sample_b):
    """Two-sided paired t-test computed from first principles (no scipy
    dependency for the core statistic; scipy is only used separately as
    an independent cross-check per the assignment constraints)."""
    a, b = np.asarray(sample_a, dtype=float), np.asarray(sample_b, dtype=float)
    diffs = a - b
    n = len(diffs)
    mean_diff = diffs.mean()
    std_diff = diffs.std(ddof=1)
    if std_diff == 0:
        t_stat = 0.0 if mean_diff == 0 else np.inf * np.sign(mean_diff)
    else:
        t_stat = mean_diff / (std_diff / np.sqrt(n))
    # two-sided p-value via the incomplete beta function is avoided here;
    # we report t-statistic, dof, and use scipy ONLY as a cross-check in
    # the report notebook (per assignment constraint: library used only
    # to verify, not to implement, the core statistic).
    dof = n - 1
    return {"t_stat": t_stat, "dof": dof, "mean_diff": mean_diff, "std_diff": std_diff}
