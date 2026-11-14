"""Baseline quality and leakage tests on synthetic data; no PostgreSQL required."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from credit_risk.features.build import FeatureDataset
from credit_risk.modeling.evaluate import binary_metrics, select_f1_threshold
from credit_risk.modeling.split import make_splits, training_subset
from credit_risk.modeling.train import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    BaselineConfig,
    fit_baselines,
    load_baseline_config,
    positive_probability,
    run_baseline,
)


def synthetic_dataset() -> FeatureDataset:
    rng = np.random.default_rng(42)
    n = 200
    ids = pd.Series(np.arange(1, n + 1, dtype="int64"), name="client_id")
    y = pd.Series((np.arange(n) % 4 == 0).astype("int64"), name="default_next_month")
    X = pd.DataFrame({name: rng.normal(100, 10, n) for name in NUMERIC_FEATURES})
    for name in CATEGORICAL_FEATURES:
        X[name] = rng.integers(0, 3, n)
    X["limit_bal"] = rng.normal(100, 5, n) - 20 * y
    X.loc[::9, "payment_to_abs_bill_ratio"] = np.nan
    return FeatureDataset(client_ids=ids, X=X, y=y)


def test_metrics_on_perfect_predictions() -> None:
    result = binary_metrics([0, 0, 1, 1], [0, 0, 1, 1], threshold=0.5)
    for name in ("roc_auc", "average_precision", "precision", "recall", "f1"):
        assert result[name] == 1.0
    assert result["brier_score"] == 0.0
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]


def test_threshold_maximizes_f1_and_accepts_constant_scores() -> None:
    assert select_f1_threshold([0, 0, 1, 1], [0.1, 0.3, 0.8, 0.9]) == pytest.approx(0.8)
    assert select_f1_threshold([0, 1, 0, 0], [0.25] * 4) == pytest.approx(0.25)


@pytest.mark.parametrize("bad", [[0, 1.2], [0, np.nan], [0, np.inf]])
def test_invalid_probabilities_are_rejected(bad) -> None:
    with pytest.raises(ValueError, match="Probabilities"):
        binary_metrics([0, 1], bad, threshold=0.5)


def test_train_only_imputer_scaler_and_encoder() -> None:
    data = synthetic_dataset()
    splits = make_splits(data)
    train_mask = data.client_ids.isin(splits.loc[splits["split"] == "train", "client_id"])
    data.X.loc[~train_mask, "limit_bal"] = 1e9
    data.X.loc[~train_mask, "sex"] = 99
    X_train, _ = training_subset(data, splits)
    models, summary, _, probabilities = fit_baselines(data, splits, BaselineConfig())
    fitted = models["logistic"].named_steps["preprocessor"]
    numeric = fitted.named_transformers_["numeric"]
    categorical = fitted.named_transformers_["categorical"]
    limit_idx = NUMERIC_FEATURES.index("limit_bal")
    ratio_idx = NUMERIC_FEATURES.index("payment_to_abs_bill_ratio")
    sex_idx = CATEGORICAL_FEATURES.index("sex")
    assert numeric.named_steps["scaler"].mean_[limit_idx] == pytest.approx(X_train["limit_bal"].mean())
    assert numeric.named_steps["imputer"].statistics_[ratio_idx] == pytest.approx(X_train["payment_to_abs_bill_ratio"].median())
    assert 99 not in categorical.named_steps["onehot"].categories_[sex_idx]
    assert summary["train_clients"] == 120
    assert summary["validation_clients"] == 40
    assert summary["test_evaluated"] is False
    assert len(probabilities["logistic"]) == 40
    assert np.isfinite(probabilities["logistic"]).all()
    assert np.all((probabilities["logistic"] >= 0) & (probabilities["logistic"] <= 1))


def test_test_features_do_not_change_baseline_metrics_or_thresholds() -> None:
    data = synthetic_dataset()
    splits = make_splits(data)
    _, first, _, _ = fit_baselines(data, splits, BaselineConfig())
    test_mask = data.client_ids.isin(splits.loc[splits["split"] == "test", "client_id"])
    data.X.loc[test_mask, "limit_bal"] = 1e12
    data.X.loc[test_mask, "sex"] = 99
    _, second, _, _ = fit_baselines(data, splits, BaselineConfig())
    assert first["models"] == second["models"]


def test_dummy_uses_training_prevalence_and_models_reproduce() -> None:
    data = synthetic_dataset()
    splits = make_splits(data)
    _, y_train = training_subset(data, splits)
    _, first, _, probabilities = fit_baselines(data, splits, BaselineConfig())
    _, second, _, repeated = fit_baselines(data, splits, BaselineConfig())
    assert np.allclose(probabilities["dummy"], y_train.mean())
    assert first["models"]["dummy"]["validation_at_0_5"]["roc_auc"] == pytest.approx(0.5)
    np.testing.assert_allclose(probabilities["logistic"], repeated["logistic"], atol=1e-10)
    assert first["models"] == second["models"]


def test_target_column_is_rejected() -> None:
    data = synthetic_dataset()
    splits = make_splits(data)
    data.X["default_next_month"] = data.y
    with pytest.raises(ValueError, match="30 SQL model features"):
        fit_baselines(data, splits, BaselineConfig())


def test_config_and_full_report_pipeline(tmp_path) -> None:
    data = synthetic_dataset()
    assignments = make_splits(data)
    config = load_baseline_config(Path(__file__).resolve().parents[1] / "configs" / "training.yaml")
    reports, artifacts = tmp_path / "reports", tmp_path / "artifacts"
    summary = run_baseline(data, assignments, config, reports_dir=reports, artifacts_dir=artifacts)
    saved = json.loads((reports / "baseline.json").read_text(encoding="utf-8"))
    assert saved == summary
    assert saved["test_evaluated"] is False
    assert (reports / "baseline.md").is_file()
    assert len(list((reports / "figures").glob("baseline_validation_*.png"))) == 3
    assert (artifacts / "metadata.json").is_file()
    model = joblib.load(artifacts / "logistic.joblib")
    X_train, _ = training_subset(data, assignments)
    assert len(positive_probability(model, X_train)) == len(X_train)
