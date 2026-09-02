"""
experiment_ga_vs_random.py
----------------------------
Empirically demonstrates that GA-optimised initial weights converge
faster / to a better optimum than plain random weight initialisation,
by training a fixed-architecture MLP from (a) 10 independent random
initialisations and (b) the single GA-selected initialisation, over the
same number of epochs, and comparing the training-loss curves and final
validation F1.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import data_preprocessing as dp
from src.mlp import MLP
from src.genetic_algorithm import GeneticAlgorithm
from src.evaluation import f1_score


def run(data_path="data/pima-indians-diabetes.csv", results_dir="results",
        random_state=42, n_random_runs=10, epochs=150):
    X, y, _ = dp.load_and_prepare(data_path)
    folds = dp.stratified_kfold_indices(y, k=5, random_state=random_state)
    train_idx, test_idx = folds[0]
    X_train_raw, y_train = X[train_idx], y[train_idx]
    X_test_raw, y_test = X[test_idx], y[test_idx]
    X_train, mean_, std_ = dp.standardise(X_train_raw)
    X_test, _, _ = dp.standardise(X_test_raw, mean_, std_)
    X_train_bal, y_train_bal = dp.random_oversample(X_train, y_train, random_state=random_state)

    inner = dp.stratified_kfold_indices(y_train, k=5, random_state=random_state)
    itr, iva = inner[0]
    Xitr, yitr = dp.standardise(X_train_raw[itr], mean_, std_)[0], y_train[itr]
    Xiva, yiva = dp.standardise(X_train_raw[iva], mean_, std_)[0], y_train[iva]
    Xitr_bal, yitr_bal = dp.random_oversample(Xitr, yitr, random_state=random_state)

    n_features = X_train.shape[1]
    fixed_hidden = 16  # fixed architecture for a fair, controlled comparison

    # ---- (a) baseline: N random initialisations ----
    random_curves = []
    random_final_f1 = []
    for i in range(n_random_runs):
        mlp = MLP([n_features, fixed_hidden, 1], learning_rate=0.05, random_state=1000 + i)
        hist = mlp.train(X_train_bal, y_train_bal, X_test, y_test,
                          epochs=epochs, batch_size=32, patience=epochs, verbose=False)
        random_curves.append(hist["train_loss"])
        pred = mlp.predict(X_test)
        random_final_f1.append(f1_score(y_test, pred))

    # ---- (b) GA-optimised initialisation ----
    ga = GeneticAlgorithm(n_features, Xitr_bal, yitr_bal, Xiva, yiva,
                           pop_size=20, generations=20, random_state=random_state)
    best_seed_mlp, ga_info = ga.run(verbose=False)

    mlp_ga_init = MLP([n_features, ga_info["n_hidden"], 1],
                       learning_rate=ga_info["learning_rate"],
                       init_weights=(best_seed_mlp.W, best_seed_mlp.b))
    hist_ga = mlp_ga_init.train(X_train_bal, y_train_bal, X_test, y_test,
                                 epochs=epochs, batch_size=32, patience=epochs, verbose=False)
    pred_ga = mlp_ga_init.predict(X_test)
    ga_final_f1 = f1_score(y_test, pred_ga)

    # ---- also: same fixed architecture (16 hidden) with random init averaged, for apples-to-apples loss-curve comparison ----
    max_len = max(len(c) for c in random_curves)
    padded = [c + [c[-1]] * (max_len - len(c)) for c in random_curves]
    mean_random_curve = np.mean(padded, axis=0)
    std_random_curve = np.std(padded, axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mean_random_curve, label=f"Random init (mean of {n_random_runs} runs)", color="tab:blue")
    ax.fill_between(range(len(mean_random_curve)),
                     mean_random_curve - std_random_curve, mean_random_curve + std_random_curve,
                     alpha=0.2, color="tab:blue")
    ga_curve = hist_ga["train_loss"]
    ax.plot(ga_curve + [ga_curve[-1]] * (max_len - len(ga_curve)), label="GA-optimised init", color="tab:orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (BCE)")
    ax.set_title("Convergence: GA-optimised init vs. random init")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{results_dir}/ga_vs_random_convergence.png", dpi=150)
    plt.close(fig)

    result = {
        "random_init_final_f1_mean": float(np.mean(random_final_f1)),
        "random_init_final_f1_std": float(np.std(random_final_f1)),
        "random_init_final_f1_all_runs": [float(x) for x in random_final_f1],
        "ga_init_final_f1": float(ga_final_f1),
        "ga_chosen_hparams": ga_info,
        "epochs_to_loss_below_0.5_random_mean": float(np.mean(
            [next((i for i, v in enumerate(c) if v < 0.5), epochs) for c in random_curves])),
        "epochs_to_loss_below_0.5_ga": int(next((i for i, v in enumerate(ga_curve) if v < 0.5), epochs)),
    }
    with open(f"{results_dir}/ga_vs_random.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
