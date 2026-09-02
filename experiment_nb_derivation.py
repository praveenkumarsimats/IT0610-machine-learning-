"""
experiment_nb_derivation.py
-----------------------------
Trains the from-scratch Gaussian Naive Bayes on fold-1 training data and
produces a fully worked manual posterior derivation for one specific,
real test-set patient record, saved to /results for inclusion in the
technical report.
"""
from src import data_preprocessing as dp
from src.naive_bayes import GaussianNaiveBayes

FEATURE_NAMES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                  "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]


def run(data_path="data/pima-indians-diabetes.csv", results_dir="results",
        random_state=42, patient_index=0):
    X, y, feature_names = dp.load_and_prepare(data_path)
    folds = dp.stratified_kfold_indices(y, k=5, random_state=random_state)
    train_idx, test_idx = folds[0]
    X_train_raw, y_train = X[train_idx], y[train_idx]
    X_test_raw, y_test = X[test_idx], y[test_idx]
    X_train, mean_, std_ = dp.standardise(X_train_raw)
    X_test, _, _ = dp.standardise(X_test_raw, mean_, std_)
    X_train_bal, y_train_bal = dp.random_oversample(X_train, y_train, random_state=random_state)

    nb = GaussianNaiveBayes().fit(X_train_bal, y_train_bal)

    x_patient = X_test[patient_index]
    x_patient_raw = X_test_raw[patient_index]
    true_label = int(y_test[patient_index])

    derivation_text, posteriors = nb.manual_posterior_derivation(x_patient, feature_names)

    header = (
        f"Manual Naive Bayes Posterior Derivation for Test Patient #{patient_index}\n"
        f"{'='*70}\n"
        f"Raw feature values: " +
        ", ".join(f"{n}={v:.2f}" for n, v in zip(feature_names, x_patient_raw)) + "\n"
        f"(Standardised feature values used internally by the model: " +
        ", ".join(f"{v:.3f}" for v in x_patient) + ")\n"
        f"True label (Outcome): {true_label}\n"
    )
    full_text = header + derivation_text

    with open(f"{results_dir}/nb_manual_derivation_patient0.txt", "w") as f:
        f.write(full_text)

    print(full_text)
    return full_text, posteriors, true_label


if __name__ == "__main__":
    run()
