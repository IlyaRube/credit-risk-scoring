"""Read SQL-built features and target separately, keeping identifiers out of X."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


class FeatureDatasetError(ValueError):
    """The SQL feature view and target table cannot be aligned safely."""


@dataclass(frozen=True)
class FeatureDataset:
    """Aligned model inputs, labels, and IDs kept in separate objects."""

    client_ids: pd.Series
    X: pd.DataFrame
    y: pd.Series


def separate_feature_target(
    feature_frame: pd.DataFrame, target_frame: pd.DataFrame
) -> FeatureDataset:
    """Validate keys and return X, y, and IDs separately, aligned by client_id."""
    if "client_id" not in feature_frame.columns:
        raise FeatureDatasetError("Feature view must contain client_id as a join key.")
    if "default_next_month" in feature_frame.columns:
        raise FeatureDatasetError("Target leakage: default_next_month is present in the feature view.")
    if feature_frame.columns.has_duplicates:
        raise FeatureDatasetError("Feature view contains duplicate column names.")
    if not {"client_id", "default_next_month"}.issubset(target_frame.columns):
        raise FeatureDatasetError("Target query must return client_id and default_next_month.")
    if feature_frame["client_id"].isna().any() or target_frame["client_id"].isna().any():
        raise FeatureDatasetError("client_id cannot be missing.")
    if feature_frame["client_id"].duplicated().any():
        raise FeatureDatasetError("Feature view contains duplicate client_id values.")
    if target_frame["client_id"].duplicated().any():
        raise FeatureDatasetError("Target table contains duplicate client_id values.")

    feature_ids_numeric = pd.to_numeric(feature_frame["client_id"], errors="coerce")
    target_ids_numeric = pd.to_numeric(target_frame["client_id"], errors="coerce")
    for name, series in (("feature client_id", feature_ids_numeric), ("target client_id", target_ids_numeric)):
        values = series.to_numpy(dtype="float64")
        is_integer = np.equal(values, np.floor(values)).all()
        if series.isna().any() or not np.isfinite(values).all() or not is_integer:
            raise FeatureDatasetError(f"{name} must contain finite integer values.")

    feature_ids = feature_ids_numeric.astype("int64").reset_index(drop=True)
    target_frame = target_frame.copy()
    target_frame["client_id"] = target_ids_numeric.astype("int64")
    target_values = pd.to_numeric(target_frame["default_next_month"], errors="coerce")
    if target_values.isna().any() or not target_values.isin([0, 1]).all():
        raise FeatureDatasetError("Target must contain only 0 and 1 without missing values.")
    target_frame["default_next_month"] = target_values.astype("int64")
    target_index = target_frame.set_index("client_id", verify_integrity=True)
    feature_id_values = feature_ids.tolist()
    if set(feature_id_values) != set(target_index.index.astype("int64").tolist()):
        raise FeatureDatasetError("Feature and target client IDs do not match exactly.")

    y = target_index.loc[feature_id_values, "default_next_month"].reset_index(drop=True)

    X = feature_frame.drop(columns=["client_id"]).reset_index(drop=True).copy()
    forbidden = {"client_id", "default_next_month"}.intersection(X.columns)
    if forbidden:
        raise FeatureDatasetError(f"Forbidden columns in X: {sorted(forbidden)}")
    if X.empty or len(X) != len(y):
        raise FeatureDatasetError("Feature matrix is empty or its row count differs from target.")

    return FeatureDataset(
        client_ids=feature_ids,
        X=X,
        y=y.astype("int64").rename("default_next_month"),
    )


def load_feature_dataset(engine: Engine) -> FeatureDataset:
    """Load the feature view and labels with separate SQL queries."""
    with engine.connect() as connection:
        feature_frame = pd.read_sql_query(
            text("SELECT * FROM ml_feature_dataset ORDER BY client_id"), connection
        )
        target_frame = pd.read_sql_query(
            text("SELECT client_id, default_next_month FROM targets ORDER BY client_id"),
            connection,
        )
    return separate_feature_target(feature_frame, target_frame)
