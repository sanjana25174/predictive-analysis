"""
evaluate_models.py - create model_metrics.json for the dashboard's Model Performance page.

Why this file exists
--------------------
model_training.py only PRINTED the metrics (and saved only the Random Forest), so the
Logistic Regression and Decision Tree numbers were never stored anywhere. This script:

  * Random Forest : loads your SAVED random_forest_model.pkl (it is NOT retrained) and
                    evaluates it on the held-out test split.
  * Logistic Regression and Decision Tree : retrains them with exactly the same settings,
                    split and random_state as model_training.py (they are not saved anywhere,
                    and training them takes only a moment), then evaluates them.

It also checks the Random Forest result against the figures you reported earlier.
Run it again whenever you re-run model_training.py or change your scikit-learn version:

    python evaluate_models.py
"""

import json
import platform
from datetime import datetime

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import core


def metrics_pair(model, X_train, y_train, X_test, y_test):
    """Test and train metrics for one fitted model."""
    return {
        "test": core.classification_metrics(y_test, model.predict(X_test)),
        "train": core.classification_metrics(y_train, model.predict(X_train)),
    }


def main():
    rf_model, feature_names = core.load_model_assets()
    df = core.load_dataset()
    X_train, X_test, y_train, y_test = core.get_test_split(df, feature_names)

    results = {}

    # --- Logistic Regression (scaled inputs, as in model_training.py) -----------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logistic = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    logistic.fit(X_train_scaled, y_train)
    results["Logistic Regression"] = metrics_pair(logistic, X_train_scaled, y_train, X_test_scaled, y_test)
    results["Logistic Regression"]["source"] = "Retrained with the settings in model_training.py"

    # --- Decision Tree ------------------------------------------------------------------
    tree = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
    tree.fit(X_train, y_train)
    results["Decision Tree"] = metrics_pair(tree, X_train, y_train, X_test, y_test)
    results["Decision Tree"]["source"] = "Retrained with the settings in model_training.py"

    # --- Random Forest: the SAVED model, not retrained ------------------------------------
    results["Random Forest"] = metrics_pair(rf_model, X_train, y_train, X_test, y_test)
    results["Random Forest"]["source"] = "Saved random_forest_model.pkl evaluated (not retrained)"

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "split": {
            "test_size": core.TEST_SIZE,
            "random_state": core.RANDOM_STATE,
            "stratified": True,
            "train_records": int(len(y_train)),
            "test_records": int(len(y_test)),
            "test_failures": int(y_test.sum()),
        },
        "models": results,
    }
    core.METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # --- Console report -----------------------------------------------------------------
    print(f"Test set: {len(y_test)} records, {int(y_test.sum())} failures\n")
    print(f"{'Model':<22}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}")
    for name, res in results.items():
        t = res["test"]
        print(f"{name:<22}{t['accuracy']:>10.4f}{t['precision']:>11.4f}{t['recall']:>9.4f}{t['f1']:>9.4f}")

    check = core.compare_with_reported(results["Random Forest"]["test"])
    print("\nRandom Forest vs previously reported figures:")
    print(check.to_string(index=False))
    if check["Matches"].all():
        print("\nVERIFIED: the saved Random Forest reproduces every reported figure.")
    else:
        print("\nMISMATCH: some reported figures do not match the saved model. "
              "Use the recomputed values and update your report.")
    print(f"\nSaved: {core.METRICS_PATH.name}")


if __name__ == "__main__":
    try:
        main()
    except core.AssetError as error:
        raise SystemExit(f"Error: {error}")
