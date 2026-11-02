"""Stage 1 checks: schema, categories, dataset provenance and snapshot integrity."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from credit_risk.data import fetch
from credit_risk.data.quality import build_quality_report, save_quality_report
from credit_risk.data.schema import FEATURE_COLUMNS, DatasetSchemaError, validate_and_normalize


def make_raw(rows: int = 3) -> pd.DataFrame:
    """A tiny, synthetic UCI-shaped table; no real dataset is bundled into tests."""
    data = {"ID": list(range(1, rows + 1))}
    for name in FEATURE_COLUMNS:
        data[name] = [10] * rows
    data.update(
        {
            "SEX": [1] * rows,
            "EDUCATION": [2] * rows,
            "MARRIAGE": [1] * rows,
            "AGE": [30] * rows,
            "PAY_0": [-2] * rows,
            "default payment next month": [index % 2 for index in range(rows)],
        }
    )
    return pd.DataFrame(data)


def test_schema_normalizes_id_and_target_without_changing_codes() -> None:
    clean = validate_and_normalize(make_raw(), expected_rows=3)
    assert list(clean.columns) == ["client_id", *FEATURE_COLUMNS, "default_next_month"]
    assert clean["client_id"].tolist() == [1, 2, 3]
    assert clean["PAY_0"].tolist() == [-2, -2, -2]
    assert clean["default_next_month"].tolist() == [0, 1, 0]


def test_schema_supports_official_x1_through_x23_and_y() -> None:
    raw = make_raw()
    raw = raw.rename(
        columns={
            **{name: f"X{index}" for index, name in enumerate(FEATURE_COLUMNS, 1)},
            "default payment next month": "Y",
        }
    )
    result = validate_and_normalize(raw, expected_rows=3)
    assert result.columns.tolist() == ["client_id", *FEATURE_COLUMNS, "default_next_month"]
    assert result["LIMIT_BAL"].tolist() == [10, 10, 10]
    assert result["PAY_0"].tolist() == [-2, -2, -2]
    assert result["default_next_month"].tolist() == [0, 1, 0]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda frame: frame.drop(columns=["PAY_0"]), "Expected the 23 UCI features"),
        (lambda frame: frame.assign(ID=[1, 1, 3]), "duplicate"),
        (lambda frame: frame.assign(**{"default payment next month": [0, 2, 1]}), "0 and 1"),
        (lambda frame: frame.assign(AGE=[30, 31.5, 50]), "integer-coded"),
        (lambda frame: frame.assign(AGE=[30, "INVALID", 50]), "not numeric"),
    ],
)
def test_schema_rejects_corrupt_data(change, message: str) -> None:
    with pytest.raises(DatasetSchemaError, match=message):
        validate_and_normalize(change(make_raw()), expected_rows=3)


def test_schema_rejects_wrong_number_of_rows() -> None:
    with pytest.raises(DatasetSchemaError, match="Expected 30000 rows, got 3"):
        validate_and_normalize(make_raw())


def test_quality_reports_undocumented_codes_and_negative_bill(tmp_path: Path) -> None:
    raw = make_raw()
    raw.loc[0, "EDUCATION"] = 6
    raw.loc[1, "BILL_AMT1"] = -5
    clean = validate_and_normalize(raw, expected_rows=3)
    report = build_quality_report(clean, expected_rows=3)
    assert report["undocumented_category_codes"]["PAY_0"] == {"-2": 3}
    assert report["undocumented_category_codes"]["EDUCATION"] == {"6": 1}
    assert report["suspicious_values"]["negative_bill_amounts"]["BILL_AMT1"] == 1
    assert report["row_count"] == 3
    assert report["duplicate_client_ids"] == 0
    assert clean.loc[1, "BILL_AMT1"] == -5  # Quality checks must not mutate data.
    json_path, md_path = save_quality_report(report, tmp_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["row_count"] == 3
    assert "Undocumented categorical codes" in md_path.read_text(encoding="utf-8")


def test_fetch_reuses_cache_and_rejects_modified_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = make_raw(3)
    fake_dataset = SimpleNamespace(
        metadata=SimpleNamespace(
            uci_id=350,
            data_url="https://archive.ics.uci.edu/static/public/350/data.csv",
            num_instances=30_000,
            num_features=23,
        ),
        data=SimpleNamespace(original=raw),
    )
    monkeypatch.setattr(fetch, "fetch_ucirepo", lambda *, id: fake_dataset)
    monkeypatch.setattr(
        fetch,
        "validate_and_normalize",
        lambda frame: validate_and_normalize(frame, expected_rows=3),
    )
    original_load_snapshot = fetch.load_snapshot

    def small_snapshot(raw_dir: Path) -> pd.DataFrame:
        # Verify hash exactly as production does; allow tiny synthetic fixtures for schema.
        snapshot_path, manifest_path = fetch._snapshot_paths(raw_dir)
        manifest = fetch._read_metadata(manifest_path)
        if fetch._hash_file(snapshot_path) != manifest["sha256"]:
            raise fetch.SnapshotError("SHA-256 mismatch")
        return validate_and_normalize(pd.read_csv(snapshot_path), expected_rows=3)

    monkeypatch.setattr(fetch, "load_snapshot", small_snapshot)
    first = fetch.fetch_and_store(tmp_path)
    assert len(first) == 3
    manifest = json.loads((tmp_path / fetch.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["feature_count"] == 23
    assert len(manifest["sha256"]) == 64
    assert len(fetch.fetch_and_store(tmp_path)) == 3
    (tmp_path / fetch.SNAPSHOT_FILENAME).write_text("changed", encoding="utf-8")
    with pytest.raises(fetch.SnapshotError, match="SHA-256 mismatch"):
        fetch.fetch_and_store(tmp_path)
    monkeypatch.setattr(fetch, "load_snapshot", original_load_snapshot)


def test_rejects_untrusted_origin() -> None:
    metadata = SimpleNamespace(uci_id=350, data_url="https://example.com/copy.csv")
    with pytest.raises(DatasetSchemaError, match="Unexpected UCI data URL"):
        fetch._check_official_metadata(metadata)


def test_missing_feature_is_reported_not_imputed() -> None:
    raw = make_raw()
    raw.loc[1, "BILL_AMT1"] = None
    normalized = validate_and_normalize(raw, expected_rows=3)
    report = build_quality_report(normalized, expected_rows=3)
    assert report["missing_values"]["BILL_AMT1"] == {
        "count": 1,
        "share": pytest.approx(1 / 3),
    }
    assert pd.isna(normalized.loc[1, "BILL_AMT1"])


def test_cli_full_stage1_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Integration without network: a 30k-row synthetic table through the real CLI."""
    from credit_risk.cli import main

    source = make_raw(30_000).rename(
        columns={
            **{name: f"X{index}" for index, name in enumerate(FEATURE_COLUMNS, 1)},
            "default payment next month": "Y",
        }
    )
    fake_dataset = SimpleNamespace(
        metadata=SimpleNamespace(
            uci_id=350,
            data_url="https://archive.ics.uci.edu/static/public/350/data.csv",
            num_instances=30_000,
            num_features=23,
        ),
        data=SimpleNamespace(original=source),
    )
    monkeypatch.setattr(fetch, "fetch_ucirepo", lambda *, id: fake_dataset)
    monkeypatch.chdir(tmp_path)
    assert main(["fetch-data"]) == 0
    assert main(["fetch-data"]) == 0  # offline cache reuse
    assert main(["data-quality"]) == 0
    report = json.loads((tmp_path / "reports/data_quality.json").read_text(encoding="utf-8"))
    assert report["row_count"] == 30_000
    assert report["unique_client_ids"] == 30_000
    assert report["feature_count"] == 23
