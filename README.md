# Machine Learning Lab Programs

A collection of 20 classic Machine Learning lab programs implemented in Python.

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
| 11 | `lab11_ga_knapsack.py` | Genetic Algorithm for the 0/1 Knapsack problem — evolves a binary chromosome population to maximise value under a weight constraint |
| 12 | `lab12_symbolic_regression.py` | Genetic Programming (via `gplearn`) for symbolic regression — evolves an expression tree that fits a target function |
| 13 | `lab13_bayes_posterior.py` | Bayes' Theorem posterior probability calculator — computes P(h\|D) for a diagnostic-testing example, illustrating the base-rate effect |
| 14 | `lab14_naive_bayes_iris.py` | Gaussian Naive Bayes classifier (scikit-learn) on the Iris dataset, with accuracy and confusion matrix |
| 15 | `lab15_spam_classifier.py` | Naive Bayes text classifier (Multinomial NB + `CountVectorizer`) for spam/ham SMS classification (`spam.csv`) |
| 16 | `lab16_bayesian_network.py` | Bayesian Belief Network (via `pgmpy`) — defines a Rain/Sprinkler/Wet DAG with CPDs and runs variable-elimination inference |
| 17 | `lab17_em_gmm.py` | EM algorithm for Gaussian Mixture Models (scikit-learn) — fits a GMM to blob-clustered synthetic data |
| 18 | `lab18_mle_demo.py` | Maximum Likelihood Estimation demo — estimates the mean of a Gaussian sample and verifies it minimises sum-of-squared-errors |
| 19 | `lab19_halving_algorithm.py` | Mistake-Bound Model — simulates the Halving algorithm over a hypothesis space of threshold functions and checks the log2\|H\| mistake bound |
| 20 | `lab20_pac_bound.py` | PAC learning sample-complexity calculator — computes the minimum training set size m for given \|H\|, epsilon, and delta |

## Datasets

- `datasets/enjoysport.csv` — used by programs 1 and 2
- `datasets/playtennis.csv` — used by program 3
- `spam.csv` (with `label` and `message` columns, e.g. the classic SMS Spam Collection dataset) — required in the working directory for program 15
- Programs 4–10, 12, 13, 16–20 use synthetic data, hand-coded examples, or scikit-learn's built-in datasets (Iris, Breast Cancer), so no extra downloads are needed.

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
python3 lab11_ga_knapsack.py
python3 lab12_symbolic_regression.py
python3 lab13_bayes_posterior.py
python3 lab14_naive_bayes_iris.py
python3 lab15_spam_classifier.py
python3 lab16_bayesian_network.py
python3 lab17_em_gmm.py
python3 lab18_mle_demo.py
python3 lab19_halving_algorithm.py
python3 lab20_pac_bound.py
```

Programs 6, 8, 9, and 10 also save plot images (`.png`) to the working directory.

Program 15 requires `spam.csv` to be present in the same directory as the script.

Programs 12 and 16 require the extra packages `gplearn` and `pgmpy` respectively (see Setup above).

All programs were tested and verified to run without errors on Python 3, pandas, numpy, scikit-learn, and matplotlib (plus `gplearn` and `pgmpy` for programs 12 and 16).
