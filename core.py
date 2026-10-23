"""
core.py - machine-learning and data logic for the Smart Manufacturing dashboard.

This module has NO Streamlit or Plotly code, so everything here can be tested
from the command line (see test_core.py). app.py only handles the user interface.

Training-time files (created by data_cleaning.py / model_training.py):
    ai4i2020_cleaned.csv, random_forest_model.pkl, feature_names.pkl
Prediction-time work (this module): load those files, encode user input exactly
like training, call the REAL saved model, and map the probability to a risk level.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random_forest_model.pkl"
FEATURES_PATH = BASE_DIR / "feature_names.pkl"
DATA_PATH = BASE_DIR / "ai4i2020_cleaned.csv"
METRICS_PATH = BASE_DIR / "model_metrics.json"

SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET_COLUMN = "Machine failure"
MACHINE_TYPES = ["L", "M", "H"]
TYPE_COLUMNS = {"L": "Type_L", "M": "Type_M", "H": "Type_H"}

# Added to the dataframe by load_dataset() for charts and filters.
TYPE_LABEL_COLUMN = "Machine type"
OUTCOME_COLUMN = "Outcome"
OUTCOME_LABELS = {0: "No failure", 1: "Failure"}

# These two values MUST match the split used in model_training.py.
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Probability >= 0.50 is the model's own decision rule (what RandomForest.predict uses).
CLASS_DECISION_THRESHOLD = 0.50

# Prototype risk bands. They come from this project's original prediction_system.py
# and are NOT industrially validated safety limits.
DEFAULT_RISK_THRESHOLDS = (0.30, 0.70)

# Previously reported Random Forest results (percent). Used only to VERIFY the saved model.
REPORTED_RF_METRICS = {"Accuracy": 96.90, "Precision": 53.12, "Recall": 75.00, "F1-score": 62.20}
REPORTED_RF_CONFUSION = {"tn": 1887, "fp": 45, "fn": 17, "tp": 51}

RISK_INFO = {
    "Lower": {
        "label": "Lower estimated risk",
        "recommendation": "Continue normal operation and monitor machine parameters.",
    },
    "Medium": {
        "label": "Medium estimated risk",
        "recommendation": "Review sensor readings and schedule an inspection as appropriate.",
    },
    "Higher": {
        "label": "Higher estimated risk",
        "recommendation": (
            "Arrange prompt maintenance inspection and follow the site's "
            "established safety procedures."
        ),
    },
    "Not available": {
        "label": "Risk level not available",
        "recommendation": (
            "This model cannot output a probability. Review the sensor readings and "
            "follow the site's established procedures."
        ),
    },
}

# Kelvin temperatures and rpm must be > 0; torque and tool wear must be >= 0.
_STRICTLY_POSITIVE = {"Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]"}


class AssetError(RuntimeError):
    """Raised with a beginner-friendly message when a required file is missing or unusable."""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def sensor_name(column: str) -> str:
    """'Air temperature [K]' -> 'Air temperature'."""
    return column.split(" [")[0]


def sensor_unit(column: str) -> str:
    """'Air temperature [K]' -> 'K'."""
    if "[" in column and "]" in column:
        return column[column.index("[") + 1 : column.index("]")]
    return ""


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise AssetError(
            f"The {description} was not found: '{path.name}'. "
            f"It must be in the project folder: {BASE_DIR}"
        )


# --------------------------------------------------------------------------
# Loading the saved model, features and dataset
# --------------------------------------------------------------------------
def load_model_assets():
    """Load the saved Random Forest and its exact feature order. Never retrains."""
    _require_file(MODEL_PATH, "trained model file")
    _require_file(FEATURES_PATH, "feature-names file")
    try:
        with warnings.catch_warnings():
            # A version difference is reported on the About page instead of flooding the console.
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            model = joblib.load(MODEL_PATH)
            feature_names = joblib.load(FEATURES_PATH)
    except Exception as exc:  # noqa: BLE001 - we want to show any load problem clearly
        raise AssetError(
            f"The model files exist but could not be loaded ({type(exc).__name__}: {exc}). "
            "Check that scikit-learn is installed (pip install -r requirements.txt)."
        ) from exc

    if not hasattr(model, "predict"):
        raise AssetError("random_forest_model.pkl does not contain a scikit-learn model.")
    feature_names = list(feature_names)
    if getattr(model, "n_features_in_", len(feature_names)) != len(feature_names):
        raise AssetError(
            "The model and feature_names.pkl disagree on the number of features. "
            "Re-run model_training.py so both files are saved together."
        )
    return model, feature_names


def load_dataset() -> pd.DataFrame:
    """Load the cleaned dataset and add readable 'Machine type' and 'Outcome' columns."""
    _require_file(DATA_PATH, "cleaned dataset")
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as exc:  # noqa: BLE001
        raise AssetError(f"Could not read '{DATA_PATH.name}' ({type(exc).__name__}: {exc}).") from exc

    required = SENSOR_COLUMNS + [TARGET_COLUMN] + list(TYPE_COLUMNS.values())
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise AssetError(f"'{DATA_PATH.name}' is missing expected columns: {missing}")
    if df[required].isna().any().any():
        raise AssetError(f"'{DATA_PATH.name}' contains missing values. Re-run data_cleaning.py.")

    df = df.copy()
    type_columns = [TYPE_COLUMNS[t] for t in MACHINE_TYPES]
    df[TYPE_LABEL_COLUMN] = df[type_columns].idxmax(axis=1).str.replace("Type_", "", regex=False)
    df[OUTCOME_COLUMN] = df[TARGET_COLUMN].map(OUTCOME_LABELS)
    return df


def model_sklearn_version(model) -> str | None:
    """scikit-learn version the model was saved with (stored inside the pickle)."""
    try:
        return model.__getstate__().get("_sklearn_version")
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# Dataset statistics (historical data - NOT live machine readings)
# --------------------------------------------------------------------------
def dataset_overview(df: pd.DataFrame) -> dict:
    total = len(df)
    failures = int(df[TARGET_COLUMN].sum())
    non_failures = total - failures
    return {
        "total": total,
        "failures": failures,
        "non_failures": non_failures,
        "failure_pct": 100 * failures / total,
        "non_failure_pct": 100 * non_failures / total,
    }


def sensor_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, std, min, quartiles and max for each sensor."""
    stats = df[SENSOR_COLUMNS].describe().T
    stats = stats.rename(columns={"25%": "Q1", "50%": "Median", "75%": "Q3"})
    stats = stats[["mean", "std", "min", "Q1", "Median", "Q3", "max"]]
    stats.columns = ["Mean", "Std dev", "Min", "Q1", "Median", "Q3", "Max"]
    stats.index.name = "Sensor"
    return stats.reset_index()


def observed_ranges(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {c: (float(df[c].min()), float(df[c].max())) for c in SENSOR_COLUMNS}


def default_inputs(df: pd.DataFrame) -> dict:
    """Sensible form defaults: dataset medians (rounded to the sensor's resolution)."""
    med = df[SENSOR_COLUMNS].median()
    return {
        "Air temperature [K]": round(float(med["Air temperature [K]"]), 1),
        "Process temperature [K]": round(float(med["Process temperature [K]"]), 1),
        "Rotational speed [rpm]": int(round(med["Rotational speed [rpm]"])),
        "Torque [Nm]": round(float(med["Torque [Nm]"]), 1),
        "Tool wear [min]": int(round(med["Tool wear [min]"])),
        "Machine type": df[TYPE_LABEL_COLUMN].mode().iloc[0],
    }


def failure_rate_by_bin(df: pd.DataFrame, column: str, bins: int = 10) -> pd.DataFrame:
    """Observed failure rate within equal-sized (quantile) bins of one sensor."""
    binned = pd.qcut(df[column], q=bins, duplicates="drop")
    grouped = df.groupby(binned, observed=True)[TARGET_COLUMN].agg(["size", "sum"])
    decimals = 0 if column in ("Rotational speed [rpm]", "Tool wear [min]") else 1
    out = pd.DataFrame(
        {
            "Range": [f"{iv.left:.{decimals}f} to {iv.right:.{decimals}f}" for iv in grouped.index],
            "Records": grouped["size"].to_numpy(),
            "Failures": grouped["sum"].to_numpy().astype(int),
        }
    )
    out["Failure rate (%)"] = 100 * out["Failures"] / out["Records"]
    return out


def standardized_differences(df: pd.DataFrame) -> pd.DataFrame:
    """(mean of failure rows - mean of non-failure rows) / overall std, per sensor. Descriptive only."""
    failed = df[df[TARGET_COLUMN] == 1]
    ok = df[df[TARGET_COLUMN] == 0]
    rows = []
    for col in SENSOR_COLUMNS:
        rows.append(
            {
                "Sensor": sensor_name(col),
                "Mean (no failure)": ok[col].mean(),
                "Mean (failure)": failed[col].mean(),
                "Difference": failed[col].mean() - ok[col].mean(),
                "Standardized difference": (failed[col].mean() - ok[col].mean()) / df[col].std(),
                "Unit": sensor_unit(col),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Input validation, encoding and prediction (prediction-time)
# --------------------------------------------------------------------------
def validate_inputs(values: dict, machine_type: str, ranges: dict | None = None):
    """Return (errors, warnings). Errors block prediction; warnings do not."""
    errors: list[str] = []
    warnings_: list[str] = []
    numbers: dict[str, float] = {}

    for col in SENSOR_COLUMNS:
        raw = values.get(col)
        if raw is None or isinstance(raw, bool):
            errors.append(f"{col}: a value is required.")
            continue
        try:
            x = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{col}: '{raw}' is not a number.")
            continue
        if not math.isfinite(x):
            errors.append(f"{col}: the value must be a finite number.")
            continue
        if col in _STRICTLY_POSITIVE and x <= 0:
            errors.append(f"{col}: must be greater than 0.")
            continue
        if col not in _STRICTLY_POSITIVE and x < 0:
            errors.append(f"{col}: cannot be negative.")
            continue
        numbers[col] = x
        if ranges and col in ranges:
            low, high = ranges[col]
            if x < low or x > high:
                warnings_.append(
                    f"{col}: {x:g} is outside the observed dataset range ({low:g} to {high:g}). "
                    "The model has not seen readings like this, so treat the estimate with extra caution."
                )

    if machine_type not in MACHINE_TYPES:
        errors.append(f"Machine type must be one of {', '.join(MACHINE_TYPES)}.")

    air = numbers.get("Air temperature [K]")
    proc = numbers.get("Process temperature [K]")
    if air is not None and proc is not None and proc <= air:
        warnings_.append(
            "Process temperature is not higher than air temperature. In the training data it "
            "always was, so this combination is unusual for the model."
        )
    return errors, warnings_


def build_model_input(values: dict, machine_type: str, feature_names: list[str]) -> pd.DataFrame:
    """One-row DataFrame, one-hot encoded and ordered exactly like the training data."""
    row = {col: float(values[col]) for col in SENSOR_COLUMNS}
    for t in MACHINE_TYPES:
        row[TYPE_COLUMNS[t]] = 1 if t == machine_type else 0
    if set(row) != set(feature_names):
        raise AssetError(
            f"The saved feature list {feature_names} does not match the dashboard inputs "
            f"{list(row)}. Re-run model_training.py or update core.py."
        )
    return pd.DataFrame([row])[feature_names]


def validate_thresholds(thresholds) -> tuple[float, float]:
    low, high = float(thresholds[0]), float(thresholds[1])
    if not (0.0 <= low < high <= 1.0):
        raise ValueError("Risk thresholds must satisfy 0 <= medium < higher <= 1.")
    return low, high


def risk_level(probability: float | None, thresholds=DEFAULT_RISK_THRESHOLDS) -> str:
    """Map a failure probability to 'Lower', 'Medium' or 'Higher' using the prototype bands."""
    if probability is None:
        return "Not available"
    low, high = validate_thresholds(thresholds)
    if probability < low:
        return "Lower"
    if probability < high:
        return "Medium"
    return "Higher"


@dataclass
class PredictionResult:
    timestamp: str
    machine_type: str
    inputs: dict
    probability: float | None  # 0..1 from predict_proba
    predicted_class: int
    class_label: str
    risk_level: str
    recommendation: str
    thresholds: tuple
    model_input: pd.DataFrame

    def to_history_row(self) -> dict:
        return {
            "Timestamp": self.timestamp,
            "Machine type": self.machine_type,
            "Air temperature [K]": self.inputs["Air temperature [K]"],
            "Process temperature [K]": self.inputs["Process temperature [K]"],
            "Rotational speed [rpm]": self.inputs["Rotational speed [rpm]"],
            "Torque [Nm]": self.inputs["Torque [Nm]"],
            "Tool wear [min]": self.inputs["Tool wear [min]"],
            "Predicted class": self.class_label,
            "Failure probability (%)": None if self.probability is None else round(100 * self.probability, 2),
            "Risk level": self.risk_level,
            "Risk bands used": f"{self.thresholds[0]:.0%} / {self.thresholds[1]:.0%}",
            "Recommendation": self.recommendation,
        }


def predict_failure(model, feature_names, values: dict, machine_type: str,
                    thresholds=DEFAULT_RISK_THRESHOLDS) -> PredictionResult:
    """
    Run one prediction through the REAL saved model.
    Steps: validate -> encode -> predict class -> predict_proba -> risk level -> recommendation.
    """
    errors, _ = validate_inputs(values, machine_type)
    if errors:
        raise ValueError("Invalid input: " + " ".join(errors))
    thresholds = validate_thresholds(thresholds)

    X = build_model_input(values, machine_type, feature_names)
    predicted_class = int(model.predict(X)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        positive_index = list(model.classes_).index(1)
        probability = float(model.predict_proba(X)[0][positive_index])

    level = risk_level(probability, thresholds)
    return PredictionResult(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        machine_type=machine_type,
        inputs={col: values[col] for col in SENSOR_COLUMNS},
        probability=probability,
        predicted_class=predicted_class,
        class_label="Failure predicted" if predicted_class == 1 else "No failure predicted",
        risk_level=level,
        recommendation=RISK_INFO[level]["recommendation"],
        thresholds=thresholds,
        model_input=X,
    )


# --------------------------------------------------------------------------
# Model evaluation on the held-out test set
# --------------------------------------------------------------------------
def get_test_split(df: pd.DataFrame, feature_names: list[str]):
    """Re-create the exact train/test split used in model_training.py (same size, seed, stratify)."""
    X = df[feature_names]
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def classification_metrics(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def evaluate_saved_model(model, feature_names, df: pd.DataFrame) -> dict:
    """Evaluate the saved (not retrained) model on the held-out test split."""
    _, X_test, _, y_test = get_test_split(df, feature_names)
    result = classification_metrics(y_test, model.predict(X_test))
    result.update(
        y_true=y_test.to_numpy(),
        proba=model.predict_proba(X_test)[:, list(model.classes_).index(1)],
        n_test=int(len(y_test)),
        n_test_failures=int(y_test.sum()),
        baseline_accuracy=float(1 - y_test.mean()),
    )
    return result


def compare_with_reported(metrics: dict) -> pd.DataFrame:
    """Reported (previously stated) vs recomputed Random Forest metrics, in percent."""
    keys = {"Accuracy": "accuracy", "Precision": "precision", "Recall": "recall", "F1-score": "f1"}
    rows = []
    for label, key in keys.items():
        recomputed = 100 * metrics[key]
        reported = REPORTED_RF_METRICS[label]
        rows.append(
            {
                "Metric": label,
                "Reported (%)": reported,
                "Recomputed from saved model (%)": round(recomputed, 2),
                "Matches": abs(recomputed - reported) < 0.01,
            }
        )
    for label, key in [("True negatives", "tn"), ("False positives", "fp"),
                       ("False negatives", "fn"), ("True positives", "tp")]:
        rows.append(
            {
                "Metric": label,
                "Reported (%)": REPORTED_RF_CONFUSION[key],
                "Recomputed from saved model (%)": metrics[key],
                "Matches": metrics[key] == REPORTED_RF_CONFUSION[key],
            }
        )
    out = pd.DataFrame(rows)
    out = out.rename(columns={"Reported (%)": "Reported", "Recomputed from saved model (%)": "Recomputed"})
    return out


def threshold_metrics(y_true, proba, threshold: float) -> dict:
    return classification_metrics(y_true, (np.asarray(proba) >= threshold).astype(int))


def threshold_sweep(y_true, proba, thresholds=None) -> pd.DataFrame:
    thresholds = np.round(np.arange(0.05, 0.951, 0.05), 2) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        m = threshold_metrics(y_true, proba, float(t))
        rows.append({"Threshold": float(t), "Precision": m["precision"], "Recall": m["recall"], "F1-score": m["f1"]})
    return pd.DataFrame(rows)


def risk_band_table(y_true, proba, thresholds=DEFAULT_RISK_THRESHOLDS) -> pd.DataFrame:
    """How the prototype risk bands behave on held-out data: how many records, how many really failed."""
    low, high = validate_thresholds(thresholds)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    bands = [
        ("Lower", proba < low, f"below {low:.0%}"),
        ("Medium", (proba >= low) & (proba < high), f"{low:.0%} to below {high:.0%}"),
        ("Higher", proba >= high, f"{high:.0%} and above"),
    ]
    rows = []
    for name, mask, label in bands:
        n = int(mask.sum())
        failures = int(y_true[mask].sum())
        rows.append(
            {
                "Risk band": name,
                "Estimated probability": label,
                "Test records": n,
                "Actual failures": failures,
                "Observed failure rate (%)": (100 * failures / n) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def feature_importance_table(model, feature_names) -> pd.DataFrame:
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["Feature", "Importance"])
    out = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
    return out.sort_values("Importance", ascending=False).reset_index(drop=True)


def load_saved_metrics() -> dict | None:
    """Read model_metrics.json (created by evaluate_models.py). None if missing or unreadable."""
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
