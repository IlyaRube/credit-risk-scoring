"""Tests for development-only model selection and frozen test evaluation."""

import json

import joblib
import numpy as np
import pytest
from test_stage4_baseline import synthetic_dataset

from credit_risk.modeling.compare import (
    CANDIDATES,
    choose_model,
    evaluate_final_test,
    fit_comparison,
    run_comparison,
)
from credit_risk.modeling.split import make_splits
from credit_risk.modeling.train import BaselineConfig, positive_probability


@pytest.fixture(scope="module")
def experiment():
    data = synthetic_dataset()
    splits = make_splits(data)
    model, summary, y_val, probabilities = fit_comparison(data, splits, BaselineConfig())
    return data, splits, model, summary, y_val, probabilities


def test_four_models_compared_and_cv_trials_recorded(experiment):
    _, _, _, summary, y_val, probabilities = experiment
    assert set(probabilities) == {"dummy", *CANDIDATES}
    assert summary["test_evaluated"] is False
    assert len(y_val) == 40
    for name in CANDIDATES[1:]:
        assert len(summary["cv_results"][name]["trials"]) == 4
        assert np.isfinite(summary["cv_results"][name]["best_mean_cv_roc_auc"])
    assert len(summary["validation_permutation_importance"]) == 30


def test_selection_is_deterministic_and_excludes_dummy():
    scores = {name: {"roc_auc": 0.7, "average_precision": 0.5, "brier_score": 0.2}
              for name in CANDIDATES}
    scores["dummy"] = {"roc_auc": 1, "average_precision": 1, "brier_score": 0}
    assert choose_model(scores) == "logistic"
    scores["random_forest"]["average_precision"] = 0.6
    assert choose_model(scores) == "random_forest"
    scores["hist_gradient_boosting"]["roc_auc"] = 0.8
    assert choose_model(scores) == "hist_gradient_boosting"


def test_test_changes_do_not_affect_comparison(experiment):
    data, splits, _, first, _, _ = experiment
    modified = synthetic_dataset()
    mask = data.client_ids.isin(splits.loc[splits["split"] == "test", "client_id"])
    modified.X.loc[mask, "limit_bal"] = 1e10
    modified.X.loc[mask, "sex"] = 999
    _, second, _, _ = fit_comparison(modified, splits, BaselineConfig())
    assert first == second


def test_winner_preprocessing_is_fitted_only_on_train(experiment):
    data, splits, model, _, _, _ = experiment
    mask = data.client_ids.isin(splits.loc[splits["split"] == "train", "client_id"])
    scaler = model.named_steps["preprocessor"].named_transformers_["numeric"].named_steps["scaler"]
    assert scaler.mean_[0] == pytest.approx(data.X.loc[mask, "limit_bal"].mean())
    assert len(positive_probability(model, data.X.loc[mask])) == 120


@pytest.fixture
def saved_experiment(tmp_path, experiment, monkeypatch):
    data, splits, model, summary, y_val, probabilities = experiment
    # Reuse the fitted synthetic experiment; still exercise actual saving and reporting.
    monkeypatch.setattr("credit_risk.modeling.compare.fit_comparison",
                        lambda *args: (model, summary.copy(), y_val, probabilities))
    reports = tmp_path / "reports"
    artifacts = tmp_path / "artifacts"
    saved = run_comparison(data, splits, BaselineConfig(), reports_dir=reports, artifacts_dir=artifacts)
    return data, splits, reports, artifacts, saved


def test_save_and_evaluate_final_once(saved_experiment):
    data, splits, reports, artifacts, saved = saved_experiment
    model = joblib.load(artifacts / "selected.joblib")
    assert len(positive_probability(model, data.X.head(5))) == 5
    assert (reports / "comparison.md").is_file()
    result = evaluate_final_test(data, splits, reports_dir=reports, artifacts_dir=artifacts)
    assert result["selected_model"] == saved["selected_model"]
    assert result["metrics"]["threshold"] == saved["selected_threshold"]
    assert result["test_clients"] == 40
    assert sum(map(sum, result["metrics"]["confusion_matrix"])) == 40
    assert json.loads((reports / "final_test.json").read_text()) == result
    assert (reports / "final_test.md").is_file()
    assert len(list((reports / "figures").glob("*.png"))) == 6
    with pytest.raises(ValueError, match="already evaluated"):
        evaluate_final_test(data, splits, reports_dir=reports, artifacts_dir=artifacts)
    with pytest.raises(ValueError, match="already evaluated"):
        run_comparison(data, splits, BaselineConfig(), reports_dir=reports, artifacts_dir=artifacts)


def test_modified_pipeline_is_rejected(saved_experiment):
    data, splits, reports, artifacts, _ = saved_experiment
    with (artifacts / "selected.joblib").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evaluate_final_test(data, splits, reports_dir=reports, artifacts_dir=artifacts)
    assert not (reports / "final_test.json").exists()


def test_changed_development_data_is_rejected(saved_experiment):
    data, splits, reports, artifacts, _ = saved_experiment
    changed = synthetic_dataset()
    mask = data.client_ids.isin(splits.loc[splits["split"] == "train", "client_id"])
    changed.X.loc[mask, "limit_bal"] += 100
    with pytest.raises(ValueError, match="development data"):
        evaluate_final_test(changed, splits, reports_dir=reports, artifacts_dir=artifacts)
    assert not (reports / "final_test.json").exists()
