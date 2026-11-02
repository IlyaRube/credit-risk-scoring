"""Tests that enforce the feature/target boundary before model training."""

from pathlib import Path

import pandas as pd
import pytest

from credit_risk.features.build import FeatureDatasetError, separate_feature_target


def test_features_target_and_ids_are_separated_and_aligned_by_key() -> None:
    feature_frame = pd.DataFrame(
        {
            "client_id": [2, 1],
            "limit_bal": [20_000, 10_000],
            "age": [40, 30],
        }
    )
    target_frame = pd.DataFrame(
        {"client_id": [1, 2], "default_next_month": [0, 1]}
    )

    result = separate_feature_target(feature_frame, target_frame)

    assert result.client_ids.tolist() == [2, 1]
    assert result.X.columns.tolist() == ["limit_bal", "age"]
    assert result.X["limit_bal"].tolist() == [20_000, 10_000]
    assert result.y.tolist() == [1, 0]
    assert "client_id" not in result.X.columns
    assert "default_next_month" not in result.X.columns


def test_target_inside_feature_view_is_rejected() -> None:
    features = pd.DataFrame(
        {"client_id": [1], "limit_bal": [1000], "default_next_month": [0]}
    )
    targets = pd.DataFrame({"client_id": [1], "default_next_month": [0]})

    with pytest.raises(FeatureDatasetError, match="Target leakage"):
        separate_feature_target(features, targets)


def test_duplicate_feature_ids_are_rejected() -> None:
    features = pd.DataFrame({"client_id": [1, 1], "limit_bal": [1000, 2000]})
    targets = pd.DataFrame({"client_id": [1], "default_next_month": [0]})

    with pytest.raises(FeatureDatasetError, match="duplicate client_id"):
        separate_feature_target(features, targets)


def test_feature_and_target_ids_must_match_exactly() -> None:
    features = pd.DataFrame({"client_id": [1, 2], "limit_bal": [1000, 2000]})
    targets = pd.DataFrame({"client_id": [1, 3], "default_next_month": [0, 1]})

    with pytest.raises(FeatureDatasetError, match="do not match exactly"):
        separate_feature_target(features, targets)


def test_sql_feature_view_does_not_select_target() -> None:
    root = Path(__file__).resolve().parents[1]
    sql = (root / "sql" / "003_feature_dataset.sql").read_text(encoding="utf-8")

    assert "CREATE OR REPLACE VIEW ml_feature_dataset" in sql
    assert "default_next_month" not in sql
    assert "client_id" in sql
    assert "delinquent_month_count" in sql
    assert "bill_amount_change_6m" in sql
