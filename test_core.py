"""
test_core.py - quick automated checks for the dashboard's logic layer.

Run from the project folder:   python test_core.py
(No extra packages needed. These tests cover core.py only; the Streamlit screens are
checked manually with the checklist in README.md.)
"""

import json
import math
import warnings

import numpy as np

import core

warnings.simplefilter("ignore")

MODEL, FEATURES = core.load_model_assets()
DF = core.load_dataset()
RANGES = core.observed_ranges(DF)
DEFAULTS = core.default_inputs(DF)
SENSOR_VALUES = {c: DEFAULTS[c] for c in core.SENSOR_COLUMNS}

TESTS = []


def test(func):
    TESTS.append(func)
    return func


@test
def feature_order_is_the_training_order():
    expected = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
                "Torque [Nm]", "Tool wear [min]", "Type_H", "Type_L", "Type_M"]
    assert FEATURES == expected, FEATURES
    assert list(MODEL.feature_names_in_) == FEATURES


@test
def dataset_matches_expected_counts():
    ov = core.dataset_overview(DF)
    assert (ov["total"], ov["failures"], ov["non_failures"]) == (10000, 339, 9661), ov
    assert round(ov["failure_pct"], 2) == 3.39 and round(ov["non_failure_pct"], 2) == 96.61
    assert set(DF[core.TYPE_LABEL_COLUMN].unique()) == {"L", "M", "H"}


@test
def encoding_is_one_hot_and_correctly_ordered():
    for machine_type in core.MACHINE_TYPES:
        X = core.build_model_input(SENSOR_VALUES, machine_type, FEATURES)
        assert list(X.columns) == FEATURES
        onehot = X[["Type_H", "Type_L", "Type_M"]].iloc[0].to_dict()
        assert sum(onehot.values()) == 1 and onehot[f"Type_{machine_type}"] == 1, onehot
        assert X["Air temperature [K]"].iloc[0] == SENSOR_VALUES["Air temperature [K]"]


@test
def prediction_uses_the_real_model_not_a_hardcoded_value():
    result = core.predict_failure(MODEL, FEATURES, SENSOR_VALUES, "M")
    X = core.build_model_input(SENSOR_VALUES, "M", FEATURES)
    direct = float(MODEL.predict_proba(X)[0][1])          # computed independently here
    assert math.isclose(result.probability, direct, abs_tol=1e-12)
    assert result.predicted_class == int(MODEL.predict(X)[0])
    assert 0.0 <= result.probability <= 1.0


@test
def different_inputs_give_different_probabilities():
    calm = core.predict_failure(MODEL, FEATURES, SENSOR_VALUES, "L").probability
    stressed = dict(SENSOR_VALUES, **{"Torque [Nm]": 70.0, "Rotational speed [rpm]": 1300, "Tool wear [min]": 240})
    stressed_p = core.predict_failure(MODEL, FEATURES, stressed, "L").probability
    assert stressed_p != calm and stressed_p > calm, (calm, stressed_p)


@test
def predicted_class_follows_the_50_percent_rule():
    rng = np.random.default_rng(0)
    for _ in range(40):
        values = {c: float(rng.uniform(*RANGES[c])) for c in core.SENSOR_COLUMNS}
        res = core.predict_failure(MODEL, FEATURES, values, str(rng.choice(core.MACHINE_TYPES)))
        assert res.predicted_class == int(res.probability >= core.CLASS_DECISION_THRESHOLD)


@test
def risk_level_boundaries():
    assert core.risk_level(0.0) == "Lower"
    assert core.risk_level(0.2999) == "Lower"
    assert core.risk_level(0.30) == "Medium"
    assert core.risk_level(0.6999) == "Medium"
    assert core.risk_level(0.70) == "Higher"
    assert core.risk_level(1.0) == "Higher"
    assert core.risk_level(0.5, (0.4, 0.6)) == "Medium"
    assert core.risk_level(None) == "Not available"
    for bad in [(0.7, 0.3), (0.5, 0.5), (-0.1, 0.5), (0.2, 1.5)]:
        try:
            core.risk_level(0.5, bad)
        except ValueError:
            continue
        raise AssertionError(f"thresholds {bad} should be rejected")


@test
def recommendations_never_tell_operators_to_stop_machines():
    for info in core.RISK_INFO.values():
        assert "stop" not in info["recommendation"].lower()


@test
def validation_rejects_bad_inputs():
    for column, bad in [("Torque [Nm]", None), ("Torque [Nm]", float("nan")), ("Torque [Nm]", float("inf")),
                        ("Torque [Nm]", -1), ("Air temperature [K]", 0), ("Rotational speed [rpm]", 0),
                        ("Tool wear [min]", -5), ("Torque [Nm]", "abc")]:
        errors, _ = core.validate_inputs(dict(SENSOR_VALUES, **{column: bad}), "L", RANGES)
        assert errors, f"{column}={bad!r} should be an error"
    errors, _ = core.validate_inputs(SENSOR_VALUES, "X", RANGES)
    assert errors
    errors, _ = core.validate_inputs({}, "L", RANGES)
    assert len(errors) == 5
    try:
        core.predict_failure(MODEL, FEATURES, dict(SENSOR_VALUES, **{"Torque [Nm]": None}), "L")
    except ValueError:
        pass
    else:
        raise AssertionError("predict_failure must refuse invalid input")


@test
def out_of_range_values_warn_but_are_allowed():
    far = dict(SENSOR_VALUES, **{"Torque [Nm]": 120.0})
    errors, warns = core.validate_inputs(far, "L", RANGES)
    assert not errors and any("outside the observed dataset range" in w for w in warns)
    assert 0 <= core.predict_failure(MODEL, FEATURES, far, "L").probability <= 1
    errors, warns = core.validate_inputs(SENSOR_VALUES, "L", RANGES)
    assert not errors and not warns, (errors, warns)   # defaults are inside the range


@test
def saved_model_reproduces_reported_metrics():
    ev = core.evaluate_saved_model(MODEL, FEATURES, DF)
    check = core.compare_with_reported(ev)
    assert check["Matches"].all(), check
    assert (ev["tn"], ev["fp"], ev["fn"], ev["tp"]) == (1887, 45, 17, 51)
    assert ev["n_test"] == 2000 and ev["n_test_failures"] == 68


@test
def risk_bands_account_for_every_test_record():
    ev = core.evaluate_saved_model(MODEL, FEATURES, DF)
    table = core.risk_band_table(ev["y_true"], ev["proba"], (0.30, 0.70))
    assert table["Test records"].sum() == ev["n_test"]
    assert table["Actual failures"].sum() == ev["n_test_failures"]
    sweep = core.threshold_sweep(ev["y_true"], ev["proba"])
    assert len(sweep) == 19 and sweep["Recall"].is_monotonic_decreasing


@test
def saved_metrics_file_is_consistent_with_the_saved_model():
    saved = core.load_saved_metrics()
    assert saved is not None, "model_metrics.json missing - run: python evaluate_models.py"
    assert set(saved["models"]) == {"Logistic Regression", "Decision Tree", "Random Forest"}
    ev = core.evaluate_saved_model(MODEL, FEATURES, DF)
    rf = saved["models"]["Random Forest"]["test"]
    for key in ("accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"):
        assert math.isclose(rf[key], ev[key], abs_tol=1e-9), key
    for res in saved["models"].values():
        t = res["test"]
        assert t["tn"] + t["fp"] + t["fn"] + t["tp"] == 2000


@test
def history_row_has_every_required_field():
    row = core.predict_failure(MODEL, FEATURES, SENSOR_VALUES, "H").to_history_row()
    required = ["Timestamp", "Machine type", *core.SENSOR_COLUMNS, "Predicted class",
                "Failure probability (%)", "Risk level", "Recommendation"]
    for key in required:
        assert key in row and row[key] is not None, key


@test
def missing_files_give_friendly_errors():
    original = core.MODEL_PATH
    try:
        core.MODEL_PATH = core.BASE_DIR / "does_not_exist.pkl"
        try:
            core.load_model_assets()
        except core.AssetError as exc:
            assert "does_not_exist.pkl" in str(exc)
        else:
            raise AssertionError("expected AssetError")
    finally:
        core.MODEL_PATH = original


@test
def analytics_helpers_return_sane_tables():
    for col in core.SENSOR_COLUMNS:
        table = core.failure_rate_by_bin(DF, col)
        assert table["Records"].sum() == len(DF) and table["Failures"].sum() == 339
    assert len(core.sensor_summary(DF)) == 5
    assert len(core.standardized_differences(DF)) == 5


if __name__ == "__main__":
    failed = 0
    for func in TESTS:
        try:
            func()
            print(f"PASS  {func.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {func.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(TESTS) - failed} of {len(TESTS)} tests passed.")
    raise SystemExit(1 if failed else 0)
