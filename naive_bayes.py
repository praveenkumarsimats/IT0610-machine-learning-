"""
naive_bayes.py
--------------
Gaussian Naive Bayes classifier implemented from first principles.

Bayes' theorem:
    P(y | x_1..x_n) = P(y) * P(x_1..x_n | y) / P(x_1..x_n)

Naive (conditional independence) assumption:
    P(x_1..x_n | y) = prod_i P(x_i | y)

Each continuous feature is modelled as Gaussian within each class:
    P(x_i | y) = (1 / sqrt(2*pi*sigma_iy^2)) * exp(-(x_i - mu_iy)^2 / (2*sigma_iy^2))

Since P(x_1..x_n) is the same normalising constant for every class, the
MAP decision rule only needs the (unnormalised) numerator, computed here
in log-space for numerical stability:

    log P(y|x) (unnormalised) = log P(y) + sum_i log P(x_i | y)
"""
import numpy as np


class GaussianNaiveBayes:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.priors_ = {}
        self.mean_ = {}
        self.var_ = {}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n = X.shape[0]
        max_var = X.var(axis=0).max()
        for c in self.classes_:
            Xc = X[y == c]
            self.priors_[c] = Xc.shape[0] / n
            self.mean_[c] = Xc.mean(axis=0)
            self.var_[c] = Xc.var(axis=0) + self.var_smoothing * max_var
        return self

    def _log_gaussian_likelihood(self, x, mean, var):
        return -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)

    def log_joint(self, X):
        """Returns unnormalised log posterior (log prior + sum log
        likelihood) for each class -> shape (n_samples, n_classes)."""
        log_probs = []
        for c in self.classes_:
            log_prior = np.log(self.priors_[c])
            log_lik = self._log_gaussian_likelihood(X, self.mean_[c], self.var_[c]).sum(axis=1)
            log_probs.append(log_prior + log_lik)
        return np.column_stack(log_probs)

    def predict_proba(self, X):
        """Normalise the log-joint via the log-sum-exp trick to obtain
        genuine posterior probabilities P(y|x) that sum to 1 across
        classes."""
        log_joint = self.log_joint(X)
        max_log = log_joint.max(axis=1, keepdims=True)
        log_sum_exp = max_log + np.log(np.exp(log_joint - max_log).sum(axis=1, keepdims=True))
        log_posterior = log_joint - log_sum_exp
        return np.exp(log_posterior)  # columns correspond to self.classes_

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def manual_posterior_derivation(self, x, feature_names=None):
        """Produces a fully worked, human-readable derivation of the
        posterior probability for a single record x, showing every term
        of Bayes' theorem explicitly. Returns a formatted string plus the
        numeric result, for inclusion in the technical report."""
        lines = []
        n_features = len(x)
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(n_features)]

        log_joint = {}
        for c in self.classes_:
            lines.append(f"\n--- Class y = {c} ---")
            lines.append(f"Prior P(y={c}) = {self.priors_[c]:.6f}")
            log_lik_terms = []
            total_log_lik = 0.0
            for i, fname in enumerate(feature_names):
                mu, var = self.mean_[c][i], self.var_[c][i]
                xi = x[i]
                likelihood = (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-((xi - mu) ** 2) / (2 * var))
                log_likelihood = self._log_gaussian_likelihood(np.array([xi]), mu, var)[0]
                total_log_lik += log_likelihood
                lines.append(
                    f"  P({fname}={xi:.3f} | y={c}) = N(mu={mu:.3f}, var={var:.3f}) "
                    f"= {likelihood:.6e}  (log={log_likelihood:.4f})")
                log_lik_terms.append(log_likelihood)
            log_joint[c] = np.log(self.priors_[c]) + total_log_lik
            lines.append(f"  log P(y={c}) + sum_i log P(x_i|y={c}) = {log_joint[c]:.4f}")

        # normalise
        max_log = max(log_joint.values())
        denom = sum(np.exp(v - max_log) for v in log_joint.values())
        posteriors = {c: np.exp(log_joint[c] - max_log) / denom for c in self.classes_}

        lines.append("\n--- Normalisation (Bayes' theorem denominator) ---")
        lines.append("P(x) = sum_y [P(y) * P(x|y)]  (computed via log-sum-exp for stability)")
        for c in self.classes_:
            lines.append(f"P(y={c} | x) = {posteriors[c]:.6f}")

        predicted = max(posteriors, key=posteriors.get)
        lines.append(f"\nPredicted class (argmax posterior) = {predicted}")

        return "\n".join(lines), posteriors
