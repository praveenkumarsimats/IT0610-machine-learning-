"""
main_pipeline.py
-----------------
End-to-end, no-manual-intervention pipeline:
  1. Load & clean data (missing values, outliers, class imbalance).
  2. Stratified 5-fold cross-validation. For each fold:
       a. Standardise (fit on train fold only).
       b. Oversample minority class on the TRAIN split only.
       c. Train MLP-A (random init) as baseline.
       d. Run GA to find optimised init weights/hyperparameters -> MLP-B (MLP+GA).
       e. Train Gaussian Naive Bayes.
       f. Fit fusion weight on a small internal validation carve-out; fuse.
       g. Record accuracy/precision/recall/F1/ROC-AUC for all 3 configs.
  3. Aggregate results, run paired t-tests, save all logs/plots to /results.
"""
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import data_preprocessing as dp
from src.mlp import MLP
from src.genetic_algorithm import GeneticAlgorithm
from src.naive_bayes import GaussianNaiveBayes
from src import fusion as fz
from src.evaluation import evaluate_all, paired_t_test


def run_pipeline(data_path="data/pima-indians-diabetes.csv", k=5,
                  results_dir="results", random_state=42, ga_generations=15,
                  ga_pop_size=16, mlp_epochs=250, verbose=True):
    t0 = time.time()
    X, y, feature_names = dp.load_and_prepare(data_path)
    folds = dp.stratified_kfold_indices(y, k=k, random_state=random_state)

    records = {"mlp": [], "mlp_ga": [], "fusion": []}
    ga_convergence_logs = []
    fusion_weights = []
    per_fold_details = []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        if verbose:
            print(f"\n===== FOLD {fold_i+1}/{k} =====")
        X_train_raw, y_train = X[train_idx], y[train_idx]
        X_test_raw, y_test = X[test_idx], y[test_idx]

        # carve an internal validation split from the training fold
        # (for GA fitness + fusion-weight fitting), stratified.
        inner_folds = dp.stratified_kfold_indices(y_train, k=5, random_state=random_state)
        inner_train_idx, inner_val_idx = inner_folds[0]
        X_inner_train_raw, y_inner_train = X_train_raw[inner_train_idx], y_train[inner_train_idx]
        X_inner_val_raw, y_inner_val = X_train_raw[inner_val_idx], y_train[inner_val_idx]

        # standardise using full-train-fold statistics (no leakage from test)
        X_train, mean_, std_ = dp.standardise(X_train_raw)
        X_test, _, _ = dp.standardise(X_test_raw, mean_, std_)
        X_inner_train, _, _ = dp.standardise(X_inner_train_raw, mean_, std_)
        X_inner_val, _, _ = dp.standardise(X_inner_val_raw, mean_, std_)

        # handle class imbalance: oversample minority class on TRAIN only
        X_train_bal, y_train_bal = dp.random_oversample(X_train, y_train, random_state=random_state + fold_i)
        X_inner_train_bal, y_inner_train_bal = dp.random_oversample(
            X_inner_train, y_inner_train, random_state=random_state + fold_i)

        n_features = X_train.shape[1]

        # ---------- (a) MLP with RANDOM initial weights (baseline) ----------
        mlp_random = MLP([n_features, 12, 1], learning_rate=0.05, random_state=random_state + fold_i)
        mlp_random.train(X_train_bal, y_train_bal, X_test, y_test,
                          epochs=mlp_epochs, batch_size=32, patience=30, verbose=False)
        p_mlp_test = mlp_random.predict_proba(X_test)
        pred_mlp_test = (p_mlp_test >= 0.5).astype(int)
        m_mlp = evaluate_all(y_test, pred_mlp_test, p_mlp_test)
        records["mlp"].append(m_mlp)

        # ---------- (b) Genetic Algorithm optimises init weights/hparams ----------
        ga = GeneticAlgorithm(n_features, X_inner_train_bal, y_inner_train_bal,
                               X_inner_val, y_inner_val, pop_size=ga_pop_size,
                               generations=ga_generations, random_state=random_state + fold_i)
        best_mlp_seed, ga_info = ga.run(verbose=False)
        ga_convergence_logs.append(ga.history)

        mlp_ga = MLP([n_features, ga_info["n_hidden"], 1],
                      learning_rate=ga_info["learning_rate"],
                      init_weights=(best_mlp_seed.W, best_mlp_seed.b))
        mlp_ga.train(X_train_bal, y_train_bal, X_test, y_test,
                     epochs=mlp_epochs, batch_size=32, patience=30, verbose=False)
        p_mlp_ga_test = mlp_ga.predict_proba(X_test)
        pred_mlp_ga_test = (p_mlp_ga_test >= 0.5).astype(int)
        m_mlp_ga = evaluate_all(y_test, pred_mlp_ga_test, p_mlp_ga_test)
        records["mlp_ga"].append(m_mlp_ga)

        # ---------- (c) Naive Bayes ----------
        nb = GaussianNaiveBayes().fit(X_train_bal, y_train_bal)
        p_nb_test = nb.predict_proba(X_test)[:, list(nb.classes_).index(1)]
        pred_nb_test = nb.predict(X_test)

        # ---------- (d) Fusion ----------
        p_mlp_ga_val = mlp_ga.predict_proba(X_inner_val)
        p_nb_val = nb.predict_proba(X_inner_val)[:, list(nb.classes_).index(1)]
        best_w, _ = fz.fit_fusion_weight(p_mlp_ga_val, p_nb_val, y_inner_val, mode="arithmetic")
        p_fused_test = fz.fuse(p_mlp_ga_test, p_nb_test, best_w, mode="arithmetic")
        pred_fused_test = (p_fused_test >= 0.5).astype(int)
        m_fused = evaluate_all(y_test, pred_fused_test, p_fused_test)
        records["fusion"].append(m_fused)
        fusion_weights.append(best_w)

        if verbose:
            print(f"  MLP(random):  {m_mlp}")
            print(f"  MLP+GA:       {m_mlp_ga}  (GA chose n_hidden={ga_info['n_hidden']}, lr={ga_info['learning_rate']:.4f})")
            print(f"  Fused (w={best_w:.2f}): {m_fused}")

        per_fold_details.append({
            "fold": fold_i, "ga_info": ga_info, "fusion_weight": best_w,
            "mlp": m_mlp, "mlp_ga": m_mlp_ga, "fusion": m_fused,
        })

    # ---------- aggregate ----------
    summary = {}
    for cfg in ["mlp", "mlp_ga", "fusion"]:
        metrics_keys = records[cfg][0].keys()
        summary[cfg] = {mk: float(np.mean([r[mk] for r in records[cfg]])) for mk in metrics_keys}
        summary[cfg + "_std"] = {mk: float(np.std([r[mk] for r in records[cfg]])) for mk in metrics_keys}

    # ---------- statistical significance (paired t-test across folds) ----------
    stats = {}
    for metric in ["accuracy", "f1", "roc_auc"]:
        a = [r[metric] for r in records["mlp_ga"]]
        b = [r[metric] for r in records["mlp"]]
        stats[f"mlp_ga_vs_mlp_{metric}"] = paired_t_test(a, b)

        a = [r[metric] for r in records["fusion"]]
        b = [r[metric] for r in records["mlp_ga"]]
        stats[f"fusion_vs_mlp_ga_{metric}"] = paired_t_test(a, b)

    elapsed = time.time() - t0
    output = {
        "summary": summary,
        "per_fold": per_fold_details,
        "significance_tests": stats,
        "mean_fusion_weight": float(np.mean(fusion_weights)),
        "elapsed_seconds": elapsed,
        "n_samples": int(len(y)),
        "n_features": int(n_features),
        "k_folds": k,
    }

    with open(f"{results_dir}/results.json", "w") as f:
        json.dump(output, f, indent=2, default=float)

    _plot_results(records, ga_convergence_logs, results_dir)

    if verbose:
        print("\n===== SUMMARY (mean over folds) =====")
        for cfg in ["mlp", "mlp_ga", "fusion"]:
            print(cfg, summary[cfg])
        print("elapsed:", elapsed, "s")

    return output


def _plot_results(records, ga_logs, results_dir):
    # 1. Metric comparison bar chart
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    configs = ["mlp", "mlp_ga", "fusion"]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.25
    xpos = np.arange(len(metrics))
    for i, cfg in enumerate(configs):
        means = [np.mean([r[m] for r in records[cfg]]) for m in metrics]
        ax.bar(xpos + i * width, means, width, label=cfg)
    ax.set_xticks(xpos + width)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Mean k-fold CV performance by configuration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{results_dir}/metric_comparison.png", dpi=150)
    plt.close(fig)

    # 2. GA convergence plot (averaged across folds)
    max_gens = max(len(h["best_fitness"]) for h in ga_logs)
    best_matrix = np.array([h["best_fitness"] + [h["best_fitness"][-1]] * (max_gens - len(h["best_fitness"])) for h in ga_logs])
    mean_matrix = np.array([h["mean_fitness"] + [h["mean_fitness"][-1]] * (max_gens - len(h["mean_fitness"])) for h in ga_logs])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(best_matrix.mean(axis=0), label="Best fitness (F1) - mean across folds")
    ax.plot(mean_matrix.mean(axis=0), label="Population mean fitness (F1)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Validation F1")
    ax.set_title("GA convergence (averaged over CV folds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{results_dir}/ga_convergence.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_pipeline()
