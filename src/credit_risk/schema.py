"""Strict structure checks for UCI Default of Credit Card Clients (ID 350)."""

import numpy as np
import pandas as pd

EXPECTED_ROWS = 30_000
FEATURE_COLUMNS = (
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
)
SOURCE_X_COLUMNS = tuple(f"X{i}" for i in range(1, 24))
X_TO_FEATURE = dict(zip(SOURCE_X_COLUMNS, FEATURE_COLUMNS, strict=True))
TARGET_ALIASES = (
    "default payment next month",
    "default.payment.next.month",
    "default_next_month",
    "Y",
)


class DatasetSchemaError(ValueError):
    """The dataset has an unexpected structure or invalid critical values."""


def validate_and_normalize(
    original: pd.DataFrame, *, expected_rows: int = EXPECTED_ROWS
) -> pd.DataFrame:
    """Check all columns and values, then rename ID/target without recoding features.

    NaN in feature columns is reported at the data-quality stage rather than imputed.
    ID and target must be complete, unique/valid and numeric.
    """
    if not isinstance(original, pd.DataFrame):
        raise DatasetSchemaError("Expected UCI data.original to be a pandas DataFrame.")
    if original.columns.has_duplicates:
        raise DatasetSchemaError("Dataset has duplicate column names.")
    if len(original) != expected_rows:
        raise DatasetSchemaError(
            f"Expected {expected_rows} rows, got {len(original)}. "
            "Source content may have changed."
        )

    target_names = [name for name in TARGET_ALIASES if name in original.columns]
    if len(target_names) != 1:
        raise DatasetSchemaError(
            "Expected exactly one target column from "
            f"{TARGET_ALIASES}, found {target_names}."
        )
    target_name = target_names[0]
    column_set = set(original.columns)
    semantic_columns = set(FEATURE_COLUMNS)
    coded_columns = set(SOURCE_X_COLUMNS)
    if semantic_columns.issubset(column_set):
        source_features = FEATURE_COLUMNS
    elif coded_columns.issubset(column_set):
        source_features = SOURCE_X_COLUMNS
    else:
        missing_named = sorted(semantic_columns - column_set)
        missing_x = sorted(coded_columns - column_set)
        raise DatasetSchemaError(
            "Expected the 23 UCI features as LIMIT_BAL...PAY_AMT6 or X1...X23. "
            f"Missing semantic={missing_named}; missing X-coded={missing_x}."
        )

    expected_columns = {"ID", *source_features, target_name}
    actual_columns = set(original.columns)
    missing = sorted(expected_columns - actual_columns)
    extra = sorted(actual_columns - expected_columns)
    if missing or extra:
        raise DatasetSchemaError(f"Unexpected columns: missing={missing}, extra={extra}.")

    for name in ("ID", *source_features, target_name):
        column = original[name]
        if not pd.api.types.is_numeric_dtype(column):
            raise DatasetSchemaError(f"Column {name!r} is not numeric (dtype={column.dtype}).")
        present = column.dropna().to_numpy(dtype="float64")
        if not np.isfinite(present).all():
            raise DatasetSchemaError(f"Column {name!r} contains infinite/nonfinite values.")
        if not np.equal(present, np.floor(present)).all():
            raise DatasetSchemaError(f"Column {name!r} must contain integer-coded values.")

    ids = original["ID"]
    if ids.isna().any():
        raise DatasetSchemaError("ID contains missing values.")
    if ids.duplicated().any():
        raise DatasetSchemaError("ID contains duplicate client identifiers.")
    if (ids <= 0).any():
        raise DatasetSchemaError("ID must be a positive integer.")

    target = original[target_name]
    if target.isna().any() or not target.isin([0, 1]).all():
        raise DatasetSchemaError("Target must contain only 0 and 1, without missing values.")

    # Preserve upstream values, map X1..X23 to descriptive names, fix downstream order.
    # No categorical recoding or imputation takes place here.
    normalized = original.rename(
        columns={"ID": "client_id", target_name: "default_next_month", **X_TO_FEATURE}
    )
    normalized = normalized.loc[:, ["client_id", *FEATURE_COLUMNS, "default_next_month"]]
    return normalized.copy()
