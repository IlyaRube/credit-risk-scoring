"""Unit tests for the relational ETL; these do not require a PostgreSQL service."""

from pathlib import Path

import pandas as pd
import pytest

from credit_risk.data.db_loader import (
    MONTH_COLUMNS,
    DatabaseLoadError,
    prepare_relational_frames,
)
from credit_risk.data.schema import FEATURE_COLUMNS


def make_normalized_data(rows: int = 2) -> pd.DataFrame:
    records = []
    for index in range(rows):
        record: dict[str, int] = {"client_id": index + 1, "default_next_month": index % 2}
        record.update({column: 1 for column in FEATURE_COLUMNS})
        record.update(
            {
                "LIMIT_BAL": 10_000 + index,
                "SEX": 1,
                "EDUCATION": 2,
                "MARRIAGE": 1,
                "AGE": 30 + index,
                "PAY_0": 3,
                "PAY_2": 2,
                "PAY_3": 1,
                "PAY_4": 0,
                "PAY_5": -1,
                "PAY_6": -2,
                "BILL_AMT1": 100 + index,
                "BILL_AMT2": 200 + index,
                "BILL_AMT3": 300 + index,
                "BILL_AMT4": 400 + index,
                "BILL_AMT5": 500 + index,
                "BILL_AMT6": 600 + index,
                "PAY_AMT1": 10 + index,
                "PAY_AMT2": 20 + index,
                "PAY_AMT3": 30 + index,
                "PAY_AMT4": 40 + index,
                "PAY_AMT5": 50 + index,
                "PAY_AMT6": 60 + index,
            }
        )
        records.append(record)
    return pd.DataFrame(records, columns=["client_id", *FEATURE_COLUMNS, "default_next_month"])


def test_prepare_relational_frames_produces_expected_counts() -> None:
    clients, history, targets = prepare_relational_frames(make_normalized_data(), expected_rows=2)

    assert len(clients) == 2
    assert len(history) == 12
    assert len(targets) == 2
    assert list(clients.columns) == [
        "client_id",
        "limit_bal",
        "sex",
        "education",
        "marriage",
        "age",
    ]
    assert targets["default_next_month"].tolist() == [0, 1]


def test_history_mapping_is_chronological_from_april_to_september() -> None:
    _, history, _ = prepare_relational_frames(make_normalized_data(rows=1), expected_rows=1)

    assert [(item[0], item[1], item[2], item[3]) for item in MONTH_COLUMNS] == [
        (1, "PAY_6", "BILL_AMT6", "PAY_AMT6"),
        (2, "PAY_5", "BILL_AMT5", "PAY_AMT5"),
        (3, "PAY_4", "BILL_AMT4", "PAY_AMT4"),
        (4, "PAY_3", "BILL_AMT3", "PAY_AMT3"),
        (5, "PAY_2", "BILL_AMT2", "PAY_AMT2"),
        (6, "PAY_0", "BILL_AMT1", "PAY_AMT1"),
    ]
    assert history["month_index"].tolist() == [1, 2, 3, 4, 5, 6]
    assert history["repayment_status"].tolist() == [-2, -1, 0, 1, 2, 3]
    assert history["bill_amount"].tolist() == [600, 500, 400, 300, 200, 100]
    assert history["payment_amount"].tolist() == [60, 50, 40, 30, 20, 10]


def test_duplicate_client_ids_are_rejected() -> None:
    data = make_normalized_data()
    data.loc[1, "client_id"] = data.loc[0, "client_id"]

    with pytest.raises(DatabaseLoadError, match="duplicates"):
        prepare_relational_frames(data, expected_rows=2)


def test_non_binary_target_is_rejected() -> None:
    data = make_normalized_data()
    data.loc[0, "default_next_month"] = 2

    with pytest.raises(DatabaseLoadError, match="only 0 and 1"):
        prepare_relational_frames(data, expected_rows=2)


def test_missing_values_are_not_imputed_during_loading() -> None:
    data = make_normalized_data()
    data.loc[0, "BILL_AMT1"] = float("nan")

    with pytest.raises(DatabaseLoadError, match="missing or non-numeric"):
        prepare_relational_frames(data, expected_rows=2)


def test_unexpected_row_count_is_rejected() -> None:
    with pytest.raises(DatabaseLoadError, match="Expected 3 clients, got 2"):
        prepare_relational_frames(make_normalized_data(), expected_rows=3)


def test_analytical_sql_contains_cte_and_two_window_functions() -> None:
    root = Path(__file__).resolve().parents[1]
    sql = (root / "sql" / "002_analytical_queries.sql").read_text(encoding="utf-8").upper()

    assert "WITH CLIENT_DELAY_COUNTS AS" in sql
    assert "LAG(" in sql
    assert "ROW_NUMBER() OVER" in sql
