"""Persist the one-time stratified 60/20/20 split before exploratory analysis.

No preprocessing, feature selection or model fitting happens in this module.
The validation and test partitions are deliberately kept out of stage 3 analysis.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk.features.build import FeatureDataset

RANDOM_STATE = 42
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_FILENAME = "client_splits.csv"
META_FILENAME = "client_splits_meta.json"


class SplitError(ValueError):
    """Client split is missing, modified, inconsistent or cannot be stratified."""


def _client_labels(dataset: FeatureDataset) -> pd.DataFrame:
    """Validate the ID/target contract and canonicalize order for stable splitting."""
    if len(dataset.client_ids) != len(dataset.y) or len(dataset.X) != len(dataset.y):
        raise SplitError("X, client_ids and y must have the same number of rows.")
    if dataset.client_ids.isna().any() or dataset.client_ids.duplicated().any():
        raise SplitError("client_ids must be complete and unique.")
    if dataset.y.isna().any() or not dataset.y.isin([0, 1]).all():
        raise SplitError("Target must contain only classes 0 and 1 without nulls.")
    if not pd.api.types.is_integer_dtype(dataset.client_ids):
        raise SplitError("client_ids must be integer typed.")
    if not pd.api.types.is_integer_dtype(dataset.y):
        raise SplitError("y must be integer typed.")

    labels = pd.DataFrame(
        {
            "client_id": dataset.client_ids.to_numpy(dtype="int64"),
            "target": dataset.y.to_numpy(dtype="int64"),
        }
    ).sort_values("client_id", kind="stable", ignore_index=True)
    if (labels["client_id"] <= 0).any():
        raise SplitError("client_id values must be positive.")
    if labels["target"].nunique() != 2:
        raise SplitError("Stratified splitting requires both target classes.")
    return labels


def _dataset_fingerprint(labels: pd.DataFrame) -> str:
    """Stable SHA-256 of sorted (client ID, target) pairs, independent of row order."""
    contents = labels.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(contents).hexdigest()


def _validate_assignment(assignment: pd.DataFrame, labels: pd.DataFrame) -> None:
    if assignment.columns.tolist() != ["client_id", "split"]:
        raise SplitError("Split file must contain exactly client_id and split columns.")
    if assignment["client_id"].isna().any() or assignment["split"].isna().any():
        raise SplitError("Split file contains null identifiers or assignments.")
    if assignment["client_id"].duplicated().any():
        raise SplitError("Split file contains duplicate client_id values.")
    if not set(assignment["split"]).issubset(SPLIT_NAMES):
        raise SplitError("Split file contains an unknown partition name.")
    if set(assignment["client_id"]) != set(labels["client_id"]):
        raise SplitError("Split IDs do not match IDs in current PostgreSQL dataset.")
    if set(assignment["split"]) != set(SPLIT_NAMES):
        raise SplitError("Expected nonempty train, validation and test partitions.")
    n = len(labels)
    counts = assignment["split"].value_counts().to_dict()
    if counts != {"train": int(0.6 * n), "validation": int(0.2 * n), "test": int(0.2 * n)}:
        raise SplitError(f"Unexpected split sizes: {counts}. Expected 60/20/20.")


def make_splits(dataset: FeatureDataset, *, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Pure deterministic stratified assignment; input row order does not matter.

    Intended for the 30,000-client dataset, whose sizes are divisible by five.
    """
    labels = _client_labels(dataset)
    n = len(labels)
    if n % 5:
        raise SplitError("Exactly 60/20/20 partitions require a row count divisible by five.")

    ids = labels["client_id"].to_numpy(dtype="int64")
    y = labels["target"].to_numpy(dtype="int64")
    try:
        train_ids, holdout_ids, _, holdout_y = train_test_split(
            ids, y, test_size=0.4, random_state=seed, stratify=y
        )
        validation_ids, test_ids = train_test_split(
            holdout_ids, test_size=0.5, random_state=seed, stratify=holdout_y
        )
    except ValueError as exc:
        raise SplitError(f"Cannot create stratified split: {exc}") from exc

    mapping = {
        **dict.fromkeys(train_ids.tolist(), "train"),
        **dict.fromkeys(validation_ids.tolist(), "validation"),
        **dict.fromkeys(test_ids.tolist(), "test"),
    }
    assignment = pd.DataFrame(
        {
            "client_id": ids,
            "split": pd.Series(
                [mapping[client_id] for client_id in ids],
                dtype="object",
            ),
        }
    )
    _validate_assignment(assignment, labels)
    return assignment


def load_splits(dataset: FeatureDataset, output_dir: Path) -> pd.DataFrame:
    """Read frozen splits and detect modifications to either input or saved CSV."""
    labels = _client_labels(dataset)
    path = output_dir / SPLIT_FILENAME
    metadata_path = output_dir / META_FILENAME
    if not path.is_file() or not metadata_path.is_file():
        raise SplitError("Splits are not found. Run 'credit-risk make-splits' first.")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        assignment = pd.read_csv(path, dtype={"client_id": "int64", "split": str})
        assignment["split"] = assignment["split"].astype("object")
    except (OSError, ValueError, KeyError) as exc:
        raise SplitError(f"Could not read persisted splits: {exc}") from exc
    if metadata.get("source_sha256") != _dataset_fingerprint(labels):
        raise SplitError("Client IDs or targets changed since split creation.")
    if metadata.get("splits_sha256") != actual_hash:
        raise SplitError("Saved split CSV has changed (SHA-256 mismatch).")
    if metadata.get("random_state") != RANDOM_STATE:
        raise SplitError("Stored split random_state differs from the project seed 42.")
    _validate_assignment(assignment, labels)
    return assignment


def create_or_load_splits(dataset: FeatureDataset, output_dir: Path) -> pd.DataFrame:
    """Create once, then reuse exactly the same partitions on subsequent runs."""
    split_path = output_dir / SPLIT_FILENAME
    metadata_path = output_dir / META_FILENAME
    if split_path.exists() or metadata_path.exists():
        return load_splits(dataset, output_dir)

    labels = _client_labels(dataset)
    assignment = make_splits(dataset)
    csv_data = assignment.to_csv(index=False, lineterminator="\n").encode("utf-8")
    metadata = {
        "source_sha256": _dataset_fingerprint(labels),
        "splits_sha256": hashlib.sha256(csv_data).hexdigest(),
        "random_state": RANDOM_STATE,
        "fractions": {"train": 0.6, "validation": 0.2, "test": 0.2},
        "n_clients": len(labels),
        "counts": {
            name: int((assignment["split"] == name).sum()) for name in SPLIT_NAMES
        },
        "stratified_by": "default_next_month",
        "note": "Retain these partitions unchanged through final model evaluation.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_csv = split_path.with_suffix(".csv.tmp")
    temporary_meta = metadata_path.with_suffix(".json.tmp")
    temporary_csv.write_bytes(csv_data)
    temporary_meta.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_csv.replace(split_path)
    temporary_meta.replace(metadata_path)
    return assignment


def training_subset(
    dataset: FeatureDataset, assignment: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Return only train rows, never validation/test observations for stage 3 EDA."""
    labels = _client_labels(dataset)
    _validate_assignment(assignment, labels)
    selection = dataset.client_ids.isin(
        assignment.loc[assignment["split"] == "train", "client_id"]
    ).to_numpy(dtype=bool)
    X_train = dataset.X.iloc[np.flatnonzero(selection)].reset_index(drop=True).copy()
    y_train = dataset.y.iloc[np.flatnonzero(selection)].reset_index(drop=True).copy()
    if len(X_train) != int(0.6 * len(dataset.y)):
        raise SplitError("Train row count does not match the saved split.")
    return X_train, y_train
