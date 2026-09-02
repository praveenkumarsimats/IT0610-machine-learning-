# Adaptive Multi-Paradigm Learning Framework for Early Chronic-Disease Risk Prediction

**ITA0610 – Machine Learning | Neural Networks, Genetic Algorithms & Bayesian Reasoning**

![CI](https://github.com/YOUR_USERNAME/chronic-disease-risk-prediction/actions/workflows/ci.yml/badge.svg)

A from-scratch (NumPy/Pandas-only) machine-learning framework that combines a
Multilayer Perceptron, a Genetic Algorithm, and a Naive Bayes classifier to
predict chronic-disease (diabetes) risk, in support of **SDG 3 — Good Health
and Well-being**.

> ⚠️ **Constraint compliance:** All core algorithms (MLP, back-propagation,
> genetic algorithm, Naive Bayes, evaluation metrics, significance testing)
> are implemented **from first principles using only NumPy/Pandas**.
> scikit-learn / scipy appear in exactly one file,
> [`src/crosscheck_sklearn.py`](src/crosscheck_sklearn.py), which is used
> **only to independently verify** the from-scratch results — never to
> implement core logic.

---

## 1. System Architecture

```
                     ┌─────────────────────┐
    Raw patient  ──▶ │ data_preprocessing   │  clean missing values, clip
    records          │ .py                  │  outliers, standardise,
                     └─────────┬────────────┘  stratified k-fold, oversample
                               │
              ┌────────────────┼─────────────────┐
              ▼                                   ▼
   ┌─────────────────────┐            ┌──────────────────────┐
   │ genetic_algorithm.py │──seeds──▶  │       mlp.py         │
   │ (selection, crossover,│           │ forward + backprop    │
   │  mutation, elitism)   │           │ (ReLU hidden, sigmoid │
   └─────────────────────┘            │  output)              │
              │                        └──────────┬───────────┘
              │  optimises init weights /          │ p_MLP(y=1|x)
              │  n_hidden / learning_rate           ▼
              │                        ┌──────────────────────┐
              │                        │      fusion.py        │◀── p_NB(y=1|x)
              │                        │ weighted / geometric  │
              │                        │ Bayesian model averg. │
              │                        └──────────┬───────────┘
              │                                    │
              ▼                                    ▼
   ┌─────────────────────┐            ┌──────────────────────┐
   │   naive_bayes.py     │            │    evaluation.py      │
   │ Gaussian NB, manual  │            │ accuracy/precision/   │
   │ posterior derivation │            │ recall/F1/ROC-AUC,    │
   └─────────────────────┘            │ paired t-test          │
                                       └──────────────────────┘

   Orchestrated end-to-end, per CV fold, by main_pipeline.py
```

### Module map

| File | Responsibility |
|---|---|
| `src/data_preprocessing.py` | Load CSV, treat biologically-impossible zeros as missing, median-impute per class, IQR-clip outliers, z-score standardise (fit-on-train-only), stratified k-fold splitter, random oversampling for class imbalance |
| `src/mlp.py` | MLP with configurable hidden layers, manual forward pass, manual back-propagation (chain rule derived explicitly), mini-batch SGD, L2 regularisation, early stopping |
| `src/genetic_algorithm.py` | GA that searches over MLP initial weights **and** hyperparameters (hidden units, learning rate) using tournament selection, uniform crossover, Gaussian/resampling mutation, elitism |
| `src/naive_bayes.py` | Gaussian Naive Bayes from scratch (Bayes' theorem, class-conditional Gaussian likelihoods, log-space posterior, log-sum-exp normalisation) + a step-by-step manual-derivation utility |
| `src/fusion.py` | Weighted arithmetic and log-linear (geometric/product-of-experts) Bayesian Model Averaging of MLP+GA and NB posteriors; mixture weight fit by validation-F1 grid search |
| `src/evaluation.py` | Accuracy, precision, recall, F1, ROC-AUC (Mann-Whitney U formulation), paired t-test — all from first principles |
| `src/main_pipeline.py` | End-to-end, no-manual-intervention orchestration across stratified 5-fold CV for all three configurations (MLP, MLP+GA, Fusion) |
| `src/experiment_ga_vs_random.py` | Controlled experiment: GA-optimised init vs. 10 random inits, same architecture, convergence-curve comparison |
| `src/experiment_nb_derivation.py` | Produces the fully worked manual posterior derivation for a real test patient |
| `src/crosscheck_sklearn.py` | **Validation only** — compares our metrics/NB/MLP/t-test against scikit-learn/scipy |
| `tests/` | 40 pytest unit tests covering every core component |
| `report/technical_report.md` | Full mathematical derivations, CLT/PAC analysis, ethics & SDG discussion |
| `results/` | Auditable logs, plots (`.png`) and result tables (`.json`) from every run |

---

## 2. Dataset

**Pima Indians Diabetes Dataset** (768 records, 8 features + binary
`Outcome`), sourced from the public UCI-derived mirror at
`jbrownlee/Datasets` on GitHub:
`data/pima-indians-diabetes.csv`
(https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv)

Features: `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
DiabetesPedigreeFunction, Age`. Target: `Outcome` (1 = diabetic, 0 = not).
Class balance: 500 negative / 268 positive (≈ 65:35 — moderately imbalanced,
handled explicitly, see §4).

### Preprocessing / noisy-data handling
1. In this dataset, a recorded value of `0` for `Glucose`, `BloodPressure`,
   `SkinThickness`, `Insulin`, or `BMI` is physiologically impossible and is
   therefore a **missing-value encoding artefact**, not a true zero. These
   are converted to `NaN` and **median-imputed conditioned on the class
   label** (diabetic vs. non-diabetic medians differ meaningfully).
2. Extreme outliers are **winsorised** (clipped, not dropped) using the IQR
   rule, to avoid discarding scarce positive-class records.
3. Features are **z-score standardised**, with mean/std fit on the training
   fold only and re-applied to the test fold (no leakage).
4. **Class imbalance** is handled by random oversampling of the minority
   class, applied only to the training portion of each fold (never to
   validation/test).

All of the above is fully automated inside `data_preprocessing.py` — the
pipeline requires **no manual intervention**.

---

## 3. Setup & Run Instructions

```bash
# 1. Clone and enter the repo
git clone <this-repo-url>
cd chronic-disease-risk-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full end-to-end pipeline (5-fold CV, all 3 configurations)
python -m src.main_pipeline
#   -> writes results/results.json, results/metric_comparison.png,
#      results/ga_convergence.png

# 4. Run the GA-vs-random-initialisation convergence experiment
python -m src.experiment_ga_vs_random
#   -> writes results/ga_vs_random.json, results/ga_vs_random_convergence.png

# 5. Generate the manual Naive Bayes posterior derivation for a real patient
python -m src.experiment_nb_derivation
#   -> writes results/nb_manual_derivation_patient0.txt

# 6. (Optional) Independently cross-check against scikit-learn / scipy
python -m src.crosscheck_sklearn

# 7. Run the unit test suite
pytest
```

### Reproducing every number in the technical report
All three scripts in steps 3–5 above are deterministic (fixed
`random_state=42` throughout) and will reproduce the exact tables, plots
and derivation shown in `report/technical_report.md` and `/results`.

---

## 4. Handling Missing Data & Class Imbalance (summary)
See §2 above and `src/data_preprocessing.py` docstrings for full detail:
class-conditional median imputation for biologically-impossible zeros,
IQR winsorisation for outliers, and training-fold-only random oversampling
for the ≈65:35 class imbalance. Standardisation statistics are always fit
on the training fold only, preventing test-set leakage.

---

## 5. Results Summary (5-fold stratified CV, mean over folds)

| Configuration | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| MLP (random init) | 0.850 | 0.759 | 0.843 | 0.798 | 0.911 |
| MLP + GA (optimised init/hparams) | 0.861 | 0.772 | 0.858 | 0.812 | 0.918 |
| **Fusion (MLP+GA ⊕ Naive Bayes)** | **0.866** | **0.778** | **0.866** | **0.819** | 0.918 |

Full per-fold numbers, standard deviations, and paired t-test results are in
`results/results.json`; see `report/technical_report.md` §5 for the full
statistical-significance discussion (MLP+GA vs. MLP accuracy improvement:
t(4) = 4.00, p < 0.05).

---

## 6. Testing

```bash
pytest -v
```
40 unit tests across 6 test modules cover: data cleaning/leakage-safety,
MLP forward/backward-pass correctness (gradient-descent loss reduction
check), GA operators (selection/crossover/mutation/elitism monotonicity),
Naive Bayes posterior correctness (agreement with `predict_proba` to 1e-6),
fusion-weight selection, and every evaluation metric (including a
ties-aware ROC-AUC check).

---

## 7. Continuous Integration
`.github/workflows/ci.yml` runs `pytest` and a reduced-size smoke run of the
full pipeline on every push/PR, on Python 3.10 and 3.11.

---

## 8. Repository Layout
```
chronic-disease-risk-prediction/
├── data/                       # dataset (CSV)
├── src/                        # all core modules (see table above)
├── tests/                      # pytest unit tests
├── results/                    # logs, plots, results.json (auditable)
├── report/                     # technical_report.md (full derivations)
├── .github/workflows/ci.yml    # CI pipeline
├── requirements.txt
├── pytest.ini
└── README.md
```

## 9. Ethics, Fairness, Privacy & SDG 3
See `report/technical_report.md` §7 for the full discussion: bias/fairness
auditing across demographic subgroups, data-privacy and consent
considerations for multi-clinic pooled records, the clinical-deployment
requirement that this system augment (not replace) clinician judgement, and
how early, low-cost, explainable risk-flagging directly supports SDG 3's
target of reducing premature mortality from non-communicable disease
through earlier intervention.
