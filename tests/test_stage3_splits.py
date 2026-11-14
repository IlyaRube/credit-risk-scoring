"""Stage 3: deterministic client-level stratification and immutable saved partitions."""

import json

import numpy as np
import pandas as pd
import pytest

from credit_risk.features.build import FeatureDataset
from credit_risk.modeling.split import (
    SplitError,
    create_or_load_splits,
    load_splits,
    make_splits,
    training_subset,
)


def synthetic_dataset(n: int = 200) -> FeatureDataset:
    ids = pd.Series(np.arange(1, n + 1, dtype="int64"), name="client_id")
    y = pd.Series((np.arange(n) % 5 == 0).astype("int64"), name="default_next_month")
    X = pd.DataFrame({"limit_bal": ids.to_numpy() * 1000, "age": 25})
    return FeatureDataset(client_ids=ids, X=X, y=y)


def test_stratified_60_20_20_is_deterministic_and_order_independent() -> None:
    data = synthetic_dataset()
    first = make_splits(data)
    assert first["split"].value_counts().to_dict() == {
        "train": 120,
        "validation": 40,
        "test": 40,
    }
    assert first["client_id"].is_unique
    assert first.equals(make_splits(data))

    shuffle = np.random.default_rng(191).permutation(200)
    shuffled = FeatureDataset(
        client_ids=data.client_ids.iloc[shuffle].reset_index(drop=True),
        X=data.X.iloc[shuffle].reset_index(drop=True),
        y=data.y.iloc[shuffle].reset_index(drop=True),
    )
    assert make_splits(shuffled).equals(first)

    labels = pd.DataFrame({"client_id": data.client_ids, "y": data.y})
    joined = first.merge(labels, on="client_id", validate="one_to_one")
    for name in ("train", "validation", "test"):
        assert (joined.loc[joined["split"] == name, "y"] == 1).mean() == pytest.approx(0.2)


def test_frozen_split_is_reused_and_only_train_enters_eda(tmp_path) -> None:
    data = synthetic_dataset()
    assignments = create_or_load_splits(data, tmp_path)
    assert assignments.equals(load_splits(data, tmp_path))
    assert assignments.equals(create_or_load_splits(data, tmp_path))
    metadata = json.loads((tmp_path / "client_splits_meta.json").read_text(encoding="utf-8"))
    assert metadata["random_state"] == 42
    assert metadata["counts"] == {"train": 120, "validation": 40, "test": 40}

    X_train, y_train = training_subset(data, assignments)
    train_ids = set(assignments.loc[assignments["split"] == "train", "client_id"])
    nontrain_ids = set(assignments.loc[assignments["split"] != "train", "client_id"])
    ids_from_X = set((X_train["limit_bal"] / 1000).astype(int))
    assert ids_from_X == train_ids
    assert not ids_from_X & nontrain_ids
    assert len(y_train) == 120


def test_frozen_split_detects_tampering_and_changed_target(tmp_path) -> None:
    data = synthetic_dataset()
    create_or_load_splits(data, tmp_path)
    altered = synthetic_dataset()
    altered.y.iloc[0] = 0
    with pytest.raises(SplitError, match="IDs or targets changed"):
        load_splits(altered, tmp_path)

    csv = tmp_path / "client_splits.csv"
    csv.write_text(csv.read_text(encoding="utf-8").replace("train", "test", 1), encoding="utf-8")
    with pytest.raises(SplitError, match="SHA-256 mismatch"):
        load_splits(data, tmp_path)


def test_partial_saved_split_is_not_regenerated(tmp_path) -> None:
    data = synthetic_dataset()
    (tmp_path / "client_splits.csv").write_text("client_id,split\n1,train\n", encoding="utf-8")
    with pytest.raises(SplitError, match="not found"):
        create_or_load_splits(data, tmp_path)


def test_duplicate_client_ids_rejected() -> None:
    data = synthetic_dataset()
    data.client_ids.iloc[1] = data.client_ids.iloc[0]
    with pytest.raises(SplitError, match="complete and unique"):
        make_splits(data)


def test_too_small_or_nondivisible_dataset_rejected() -> None:
    with pytest.raises(SplitError, match="divisible by five"):
        make_splits(synthetic_dataset(201))
