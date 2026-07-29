# Machine Learning Lab Programs

A collection of 10 classic Machine Learning lab programs implemented in Python.

## Setup

```bash
pip install -r requirements.txt
```

## Programs

| # | File | Description |
|---|------|-------------|
| 1 | `1_find_s.py` | FIND-S algorithm — finds the most specific hypothesis from training data (`datasets/enjoysport.csv`) |
| 2 | `2_candidate_elimination.py` | Candidate-Elimination algorithm — outputs the version space (S and G boundaries) consistent with training data |
| 3 | `3_id3_decision_tree.py` | ID3 decision tree algorithm — builds a tree from `datasets/playtennis.csv` and classifies a new sample |
| 4 | `4_backpropagation_ann.py` | Artificial Neural Network trained with the Backpropagation algorithm (from scratch, numpy) |
| 5 | `5_knn.py` | K-Nearest Neighbours classifier (scikit-learn + from-scratch) on the Iris dataset |
| 6 | `6_naive_bayes.py` | Gaussian Naive Bayes classifier with confusion matrix and accuracy on the Iris dataset |
| 7 | `7_logistic_regression.py` | Logistic Regression on the Breast Cancer Wisconsin dataset |
| 8 | `8_linear_regression.py` | Linear Regression on a synthetic experience-vs-salary dataset, with plot |
| 9 | `9_linear_vs_polynomial_regression.py` | Comparison of Linear vs Polynomial Regression on non-linear data |
| 10 | `10_expectation_maximization.py` | EM algorithm (Gaussian Mixture Model) vs K-Means clustering on the Iris dataset |

## Datasets

- `datasets/enjoysport.csv` — used by programs 1 and 2
- `datasets/playtennis.csv` — used by program 3
- Programs 4–10 use synthetic data or scikit-learn's built-in datasets (Iris, Breast Cancer), so no extra downloads are needed.

## Running

Each program is self-contained and can be run directly:

```bash
python3 1_find_s.py
python3 2_candidate_elimination.py
python3 3_id3_decision_tree.py
python3 4_backpropagation_ann.py
python3 5_knn.py
python3 6_naive_bayes.py
python3 7_logistic_regression.py
python3 8_linear_regression.py
python3 9_linear_vs_polynomial_regression.py
python3 10_expectation_maximization.py
```

Programs 6, 8, 9, and 10 also save plot images (`.png`) to the working directory.

All programs were tested and verified to run without errors on Python 3, pandas, numpy, scikit-learn, and matplotlib.
