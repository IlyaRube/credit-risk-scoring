"""Stage 3: exact/approximate CI checks, hypothesis testing and train-only EDA reports."""

import json

import numpy as np
import pandas as pd
import pytest

from credit_risk.reporting.eda import describe_training_data, run_eda
from credit_risk.reporting.statistics import (
    StatisticalAnalysisError,
    chi_square_delinquency,
    holm_adjust,
    mann_whitney_credit_limit,
    run_inference,
    wilson_interval,
)


def synthetic_training_data(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    index = np.arange(n)
    y = pd.Series((index % 5 == 0).astype(int), name="default_next_month")
    months_delayed = (index % 3).astype(int)
    limit = 30_000 + index * 900 + rng.integers(0, 100, n)
    X = pd.DataFrame(
        {
            "age": 23 + index % 50,
            "limit_bal": limit,
            "max_repayment_status": months_delayed,
            "delinquent_month_count": months_delayed,
            "avg_bill_amount": 5_000 + index * 110,
            "avg_payment_amount": 900 + index * 10,
            "total_payment_amount": 6 * (900 + index * 10),
            "bill_amount_change_6m": (index % 3 - 1) * 600,
            "payment_to_abs_bill_ratio": np.full(n, 0.15),
        }
    )
    for month in range(1, 7):
        X[f"bill_amount_m{month}"] = X["avg_bill_amount"] + month * 90
        X[f"payment_amount_m{month}"] = X["avg_payment_amount"] + month * 10
        X[f"repayment_status_m{month}"] = months_delayed
    return X, y


def test_wilson_95_percent_interval_known_values() -> None:
    lower, upper = wilson_interval(5, 10)
    assert lower == pytest.approx(0.2366, abs=0.0001)
    assert upper == pytest.approx(0.7634, abs=0.0001)
    assert wilson_interval(0, 10)[0] == 0
    assert wilson_interval(10, 10)[1] == 1
    with pytest.raises(StatisticalAnalysisError):
        wilson_interval(12, 10)


def test_holm_adjusted_p_values() -> None:
    assert holm_adjust([0.01, 0.04]) == pytest.approx([0.02, 0.04])
    assert holm_adjust([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    with pytest.raises(StatisticalAnalysisError):
        holm_adjust([float("nan")])


def test_inference_returns_two_tests_and_interval() -> None:
    X, y = synthetic_training_data()
    result = run_inference(X, y)
    assert result["n_clients"] == 200
    assert result["default_count"] == 40
    assert result["default_rate"] == pytest.approx(0.2)
    assert len(result["tests"]) == 2
    assert result["wilson_ci"]["lower"] < 0.2 < result["wilson_ci"]["upper"]
    for test in result["tests"]:
        assert 0 <= test["p_value"] <= 1
        assert test["holm_adjusted_p"] >= test["p_value"]
    assert result["tests"][0]["min_expected_frequency"] >= 5


def test_chi_square_rejects_small_expected_cells() -> None:
    X = pd.DataFrame({"delinquent_month_count": [0, 0, 0, 0, 0, 0, 1, 1]})
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    with pytest.raises(StatisticalAnalysisError, match="expected"):
        chi_square_delinquency(X, y)


def test_mann_whitney_direction_for_higher_limits() -> None:
    X = pd.DataFrame({"limit_bal": [100, 200, 300, 1, 2, 3]})
    y = pd.Series([0, 0, 0, 1, 1, 1])
    test = mann_whitney_credit_limit(X, y)
    assert test["rank_biserial_effect_size"] == pytest.approx(1.0)
    assert test["median_limit_no_default"] == 200
    assert test["median_limit_default"] == 2


def test_descriptive_statistics_count_missing_without_imputation() -> None:
    X, y = synthetic_training_data()
    X.loc[0, "payment_to_abs_bill_ratio"] = float("nan")
    desc = describe_training_data(X, y)
    assert desc["n_clients"] == 200
    assert desc["target_counts"] == {"0": 160, "1": 40}
    assert desc["descriptive_statistics"]["payment_to_abs_bill_ratio"]["count"] == 199
    assert desc["descriptive_statistics"]["payment_to_abs_bill_ratio"]["missing_or_nonfinite"] == 1
    assert desc["descriptive_statistics"]["age"]["variance_sample"] > 0


def test_run_eda_creates_documented_reports_and_images(tmp_path) -> None:
    X, y = synthetic_training_data()
    eda_path, stats_path, figures = run_eda(X, y, tmp_path)
    assert len(figures) == 5
    assert eda_path.exists() and stats_path.exists()
    assert all((tmp_path / "figures" / name).stat().st_size > 500 for name in figures)
    assert "train only" in eda_path.read_text(encoding="utf-8").lower()
    assert "Mann-Whitney" in stats_path.read_text(encoding="utf-8")
    assert "Holm" in stats_path.read_text(encoding="utf-8")
    eda = json.loads((tmp_path / "eda_summary.json").read_text(encoding="utf-8"))
    stats = json.loads((tmp_path / "statistics.json").read_text(encoding="utf-8"))
    assert eda["n_clients"] == 200
    assert stats["default_count"] == 40


def test_describe_rejects_target_in_x() -> None:
    X, y = synthetic_training_data()
    X["client_id"] = np.arange(len(X))
    with pytest.raises(ValueError, match="leakage"):
        describe_training_data(X, y)
