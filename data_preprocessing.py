"""
data_preprocessing.py
----------------------
Loads the Pima Indians Diabetes dataset, handles missing/noisy values
(zeros that are biologically implausible are treated as missing),
performs z-score standardisation, and provides a stratified k-fold
splitter and a simple class-imbalance handler (random oversampling of
the minority class) -- all implemented from first principles with
NumPy/Pandas only.
"""
import numpy as np
import pandas as pd

# Columns where a value of 0 is not physiologically possible and is
# therefore treated as a missing value (a well-known data-quality issue
# in the Pima Indians Diabetes dataset).
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_raw(path):
    df = pd.read_csv(path)
    return df


def clean_missing(df):
    """Replace biologically-impossible zeros with NaN, then impute with the
    median of the column, conditioned on the outcome class (median
    imputation is robust to the outliers/noise present in this dataset)."""
    df = df.copy()
    for col in ZERO_AS_MISSING_COLS:
        df[col] = df[col].replace(0, np.nan)

    for outcome in df["Outcome"].unique():
        mask = df["Outcome"] == outcome
        for col in ZERO_AS_MISSING_COLS:
            median_val = df.loc[mask, col].median()
            df.loc[mask & df[col].isna(), col] = median_val
    return df


def remove_outliers_iqr(df, cols, k=3.0):
    """Winsorise (clip) extreme outliers using the IQR rule instead of
    dropping rows, to avoid losing scarce positive-class examples."""
    df = df.copy()
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - k * iqr, q3 + k * iqr
        df[col] = df[col].clip(lower=low, upper=high)
    return df


def standardise(X, mean=None, std=None):
    """Z-score standardisation. If mean/std are not supplied they are
    computed from X (training fold); otherwise the supplied statistics
    (from training fold) are applied to X (validation/test fold) to avoid
    data leakage."""
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
        std[std == 0] = 1e-8
    return (X - mean) / std, mean, std


def random_oversample(X, y, random_state=42):
    """Balance classes by randomly duplicating minority-class samples
    (with replacement) until both classes are equal in size."""
    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(y, return_counts=True)
    majority_count = counts.max()

    X_bal, y_bal = [X], [y]
    for cls, cnt in zip(classes, counts):
        if cnt < majority_count:
            deficit = majority_count - cnt
            idx = np.where(y == cls)[0]
            sampled_idx = rng.choice(idx, size=deficit, replace=True)
            X_bal.append(X[sampled_idx])
            y_bal.append(y[sampled_idx])

    X_out = np.vstack(X_bal)
    y_out = np.concatenate(y_bal)
    perm = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm]


def stratified_kfold_indices(y, k=5, random_state=42):
    """Returns a list of (train_idx, val_idx) tuples implementing
    stratified k-fold cross-validation from scratch."""
    rng = np.random.RandomState(random_state)
    y = np.asarray(y)
    folds = [[] for _ in range(k)]

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        splits = np.array_split(cls_idx, k)
        for i in range(k):
            folds[i].extend(splits[i].tolist())

    fold_indices = []
    all_idx = np.arange(len(y))
    for i in range(k):
        val_idx = np.array(sorted(folds[i]))
        train_idx = np.setdiff1d(all_idx, val_idx)
        fold_indices.append((train_idx, val_idx))
    return fold_indices


def load_and_prepare(path):
    """Full pipeline: load -> clean -> outlier-clip -> return X, y (raw,
    unscaled -- scaling is done per-fold to avoid leakage)."""
    df = load_raw(path)
    df = clean_missing(df)
    df = remove_outliers_iqr(df, ZERO_AS_MISSING_COLS + ["Pregnancies", "DiabetesPedigreeFunction"])
    feature_cols = [c for c in df.columns if c != "Outcome"]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["Outcome"].to_numpy(dtype=int)
    return X, y, feature_cols
