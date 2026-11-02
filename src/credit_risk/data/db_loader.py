"""Transactional, repeatable loading of the UCI snapshot into normalized PostgreSQL tables."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from credit_risk.data.schema import EXPECTED_ROWS, FEATURE_COLUMNS

# Chronological order: month 1 is April 2005; month 6 is September 2005.
# UCI names the repayment-status columns PAY_6, PAY_5, PAY_4, PAY_3, PAY_2, PAY_0
# from oldest to newest, while bill/payment columns use suffixes 6, 5, 4, 3, 2, 1.
MONTH_COLUMNS = (
    (1, "PAY_6", "BILL_AMT6", "PAY_AMT6"),
    (2, "PAY_5", "BILL_AMT5", "PAY_AMT5"),
    (3, "PAY_4", "BILL_AMT4", "PAY_AMT4"),
    (4, "PAY_3", "BILL_AMT3", "PAY_AMT3"),
    (5, "PAY_2", "BILL_AMT2", "PAY_AMT2"),
    (6, "PAY_0", "BILL_AMT1", "PAY_AMT1"),
)

NORMALIZED_COLUMNS = ("client_id", *FEATURE_COLUMNS, "default_next_month")
CLIENT_SOURCE_COLUMNS = (
    "client_id",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
)
CLIENT_DB_COLUMNS = (
    "client_id",
    "limit_bal",
    "sex",
    "education",
    "marriage",
    "age",
)


class DatabaseLoadError(ValueError):
    """The input snapshot or loaded relational tables violate project invariants."""


@dataclass(frozen=True)
class DatabaseLoadSummary:
    client_count: int
    history_count: int
    target_count: int
    feature_count: int


def _validate_normalized_data(data: pd.DataFrame, *, expected_rows: int) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise DatabaseLoadError("Expected a pandas DataFrame with normalized UCI columns.")
    if data.columns.has_duplicates:
        raise DatabaseLoadError("Input has duplicate column names.")

    actual = set(data.columns)
    expected = set(NORMALIZED_COLUMNS)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise DatabaseLoadError(f"Unexpected normalized schema: missing={missing}, extra={extra}.")
    if len(data) != expected_rows:
        raise DatabaseLoadError(f"Expected {expected_rows} clients, got {len(data)}.")

    frame = data.loc[:, list(NORMALIZED_COLUMNS)].copy()
    for column in NORMALIZED_COLUMNS:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.isna().any():
            raise DatabaseLoadError(f"Column {column!r} contains missing or non-numeric values.")
        values = converted.to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            raise DatabaseLoadError(f"Column {column!r} contains non-finite values.")
        if not np.equal(values, np.floor(values)).all():
            raise DatabaseLoadError(f"Column {column!r} must contain integer values.")
        frame[column] = converted.astype("int64")

    if frame["client_id"].le(0).any():
        raise DatabaseLoadError("client_id must be positive.")
    if frame["client_id"].duplicated().any():
        raise DatabaseLoadError("client_id contains duplicates.")
    if not frame["default_next_month"].isin([0, 1]).all():
        raise DatabaseLoadError("default_next_month must contain only 0 and 1.")
    return frame


def prepare_relational_frames(
    data: pd.DataFrame, *, expected_rows: int = EXPECTED_ROWS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate a normalized snapshot and split it into clients, history and targets.

    The function is pure and can be tested without a running PostgreSQL server.
    It preserves all source values; no recoding, imputation or row filtering occurs.
    """
    frame = _validate_normalized_data(data, expected_rows=expected_rows)

    clients = frame.loc[:, list(CLIENT_SOURCE_COLUMNS)].copy()
    clients.columns = CLIENT_DB_COLUMNS

    targets = frame.loc[:, ["client_id", "default_next_month"]].copy()

    month_index = np.tile(np.array([item[0] for item in MONTH_COLUMNS], dtype="int16"), len(frame))
    client_ids = np.repeat(frame["client_id"].to_numpy(dtype="int64"), len(MONTH_COLUMNS))
    repayment_status = np.column_stack(
        [frame[status].to_numpy(dtype="int64") for _, status, _, _ in MONTH_COLUMNS]
    ).reshape(-1)
    bill_amount = np.column_stack(
        [frame[bill].to_numpy(dtype="int64") for _, _, bill, _ in MONTH_COLUMNS]
    ).reshape(-1)
    payment_amount = np.column_stack(
        [frame[payment].to_numpy(dtype="int64") for _, _, _, payment in MONTH_COLUMNS]
    ).reshape(-1)

    history = pd.DataFrame(
        {
            "client_id": client_ids,
            "month_index": month_index,
            "repayment_status": repayment_status,
            "bill_amount": bill_amount,
            "payment_amount": payment_amount,
        }
    )
    if len(history) != expected_rows * 6:
        raise DatabaseLoadError(
            f"Expected {expected_rows * 6} history rows, generated {len(history)}."
        )
    return clients, history, targets


def _apply_sql_file(connection: Connection, path: Path) -> None:
    if not path.is_file():
        raise DatabaseLoadError(f"Required SQL file not found: {path}")
    contents = path.read_text(encoding="utf-8")
    for statement in contents.split(";"):
        # Ignore comments-only fragments created by splitting at semicolons.
        statement = re.sub(r"--[^\n]*", "", statement).strip()
        if statement:
            connection.exec_driver_sql(statement)


def _records(frame: pd.DataFrame) -> list[dict[str, int]]:
    return [
        {column: int(value) for column, value in zip(frame.columns, row, strict=True)}
        for row in frame.itertuples(index=False, name=None)
    ]


def _insert_frame(
    connection: Connection,
    frame: pd.DataFrame,
    statement: str,
    *,
    batch_size: int,
) -> None:
    command = text(statement)
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start : start + batch_size]
        connection.execute(command, _records(batch))


def _verify_database(connection: Connection, *, expected_rows: int) -> DatabaseLoadSummary:
    row = connection.execute(
        text(
            """
            SELECT
                (SELECT COUNT(*) FROM clients) AS client_count,
                (SELECT COUNT(*) FROM monthly_history) AS history_count,
                (SELECT COUNT(*) FROM targets) AS target_count,
                (SELECT COUNT(*) FROM ml_feature_dataset) AS feature_count,
                (SELECT COUNT(DISTINCT client_id) FROM monthly_history) AS history_clients,
                (SELECT COUNT(*) FROM (
                    SELECT client_id
                    FROM monthly_history
                    GROUP BY client_id
                    HAVING COUNT(*) <> 6
                ) AS incomplete_history) AS incomplete_history_clients,
                (SELECT COUNT(*) FROM clients c
                    JOIN targets t USING (client_id)) AS matched_targets
            """
        )
    ).mappings().one()

    expected_history = expected_rows * 6
    checks = {
        "client_count": expected_rows,
        "history_count": expected_history,
        "target_count": expected_rows,
        "feature_count": expected_rows,
        "history_clients": expected_rows,
        "incomplete_history_clients": 0,
        "matched_targets": expected_rows,
    }
    mismatches = {
        name: {"expected": expected, "actual": int(row[name])}
        for name, expected in checks.items()
        if int(row[name]) != expected
    }
    if mismatches:
        raise DatabaseLoadError(f"PostgreSQL integrity checks failed: {mismatches}")

    return DatabaseLoadSummary(
        client_count=int(row["client_count"]),
        history_count=int(row["history_count"]),
        target_count=int(row["target_count"]),
        feature_count=int(row["feature_count"]),
    )


def load_relational_data(
    engine: Engine,
    data: pd.DataFrame,
    *,
    sql_dir: Path | None = None,
    expected_rows: int = EXPECTED_ROWS,
    batch_size: int = 5_000,
) -> DatabaseLoadSummary:
    """Replace the project dataset atomically and verify all expected row counts.

    A single transaction makes a failed import roll back to the previous data.
    Re-running with the same snapshot results in the same table contents and no duplicates.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    clients, history, targets = prepare_relational_frames(data, expected_rows=expected_rows)
    sql_root = sql_dir or (Path.cwd() / "sql")

    client_insert = """
        INSERT INTO clients
            (client_id, limit_bal, sex, education, marriage, age)
        VALUES
            (:client_id, :limit_bal, :sex, :education, :marriage, :age)
    """
    history_insert = """
        INSERT INTO monthly_history
            (client_id, month_index, repayment_status, bill_amount, payment_amount)
        VALUES
            (:client_id, :month_index, :repayment_status, :bill_amount, :payment_amount)
    """
    target_insert = """
        INSERT INTO targets (client_id, default_next_month)
        VALUES (:client_id, :default_next_month)
    """

    with engine.begin() as connection:
        _apply_sql_file(connection, sql_root / "001_schema.sql")
        connection.execute(text("DELETE FROM targets"))
        connection.execute(text("DELETE FROM monthly_history"))
        connection.execute(text("DELETE FROM clients"))

        _insert_frame(connection, clients, client_insert, batch_size=batch_size)
        _insert_frame(connection, history, history_insert, batch_size=batch_size)
        _insert_frame(connection, targets, target_insert, batch_size=batch_size)
        _apply_sql_file(connection, sql_root / "003_feature_dataset.sql")
        summary = _verify_database(connection, expected_rows=expected_rows)

    return summary


def run_analytical_queries(engine: Engine, sql_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Run named SELECT queries from 002_analytical_queries.sql and return their rows."""
    if not sql_path.is_file():
        raise DatabaseLoadError(f"Required SQL file not found: {sql_path}")
    contents = sql_path.read_text(encoding="utf-8")
    pattern = re.compile(r"(?ims)^--\s*name:\s*([a-z0-9_]+)\s*\n(.*?);")
    matches = pattern.findall(contents)
    if not matches:
        raise DatabaseLoadError(f"No named SQL queries found in {sql_path}.")

    results: dict[str, list[dict[str, Any]]] = {}
    with engine.connect() as connection:
        for name, query in matches:
            result = connection.execute(text(query.strip()))
            results[name] = [dict(row) for row in result.mappings().fetchmany(10)]
    return results
