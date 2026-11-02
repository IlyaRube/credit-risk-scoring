"""Download the official UCI dataset and keep an integrity-checked local snapshot."""

import hashlib
import json
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from ucimlrepo import fetch_ucirepo

from credit_risk.data.schema import FEATURE_COLUMNS, DatasetSchemaError, validate_and_normalize

UCI_DATASET_ID = 350
SNAPSHOT_FILENAME = "uci_350_original.csv"
MANIFEST_FILENAME = "uci_350_manifest.json"


class SnapshotError(ValueError):
    """Local dataset snapshot or its manifest failed the integrity check."""


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_paths(raw_dir: Path) -> tuple[Path, Path]:
    return raw_dir / SNAPSHOT_FILENAME, raw_dir / MANIFEST_FILENAME


def _read_metadata(manifest_path: Path) -> dict:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SnapshotError(f"Cannot read snapshot manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("uci_dataset_id") != UCI_DATASET_ID:
        raise SnapshotError("Snapshot manifest has an unexpected UCI dataset ID.")
    if payload.get("snapshot_filename") != SNAPSHOT_FILENAME:
        raise SnapshotError("Snapshot manifest has an unexpected filename.")
    return payload


def load_snapshot(raw_dir: Path) -> pd.DataFrame:
    """Read a cached snapshot only after comparing its SHA-256 to the manifest."""
    snapshot_path, manifest_path = _snapshot_paths(raw_dir)
    if not snapshot_path.exists() or not manifest_path.exists():
        raise SnapshotError("Snapshot not found. Run 'credit-risk fetch-data' first.")
    manifest = _read_metadata(manifest_path)
    actual_hash = _hash_file(snapshot_path)
    if manifest.get("sha256") != actual_hash:
        raise SnapshotError(
            f"SHA-256 mismatch for {snapshot_path}. "
            "The dataset was changed or corrupted; use fetch-data --refresh to download again."
        )
    try:
        frame = pd.read_csv(snapshot_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise SnapshotError(f"Cannot parse raw snapshot {snapshot_path}: {exc}") from exc
    if frame.shape != (manifest.get("row_count"), manifest.get("column_count")):
        raise SnapshotError("Snapshot dimensions disagree with manifest.")
    return validate_and_normalize(frame)


def _check_official_metadata(metadata: object) -> str:
    """Reject records that are not explicitly labeled as UCI dataset 350."""
    uci_id = getattr(metadata, "uci_id", None)
    if uci_id != UCI_DATASET_ID:
        raise DatasetSchemaError(f"Expected UCI dataset ID 350, got {uci_id!r}.")

    source_url = getattr(metadata, "data_url", None)
    parsed = urlparse(source_url if isinstance(source_url, str) else "")
    if parsed.scheme != "https" or parsed.hostname not in {
        "archive.ics.uci.edu",
        "www.archive.ics.uci.edu",
    }:
        raise DatasetSchemaError(f"Unexpected UCI data URL: {source_url!r}.")

    for field, expected in (("num_instances", 30_000), ("num_features", 23)):
        received = getattr(metadata, field, None)
        if received is not None and int(received) != expected:
            raise DatasetSchemaError(f"UCI metadata {field}: expected {expected}, got {received}.")
    return source_url


def fetch_and_store(raw_dir: Path, *, refresh: bool = False) -> pd.DataFrame:
    """Use verified cache unless refresh is requested; never silently repair the data."""
    snapshot_path, manifest_path = _snapshot_paths(raw_dir)
    if snapshot_path.exists() or manifest_path.exists():
        if not refresh:
            return load_snapshot(raw_dir)

    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    source_url = _check_official_metadata(dataset.metadata)
    original = dataset.data.original
    validated = validate_and_normalize(original)

    raw_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = snapshot_path.with_suffix(".csv.tmp")
    original.to_csv(temporary_path, index=False, encoding="utf-8", lineterminator="\n")
    snapshot_hash = _hash_file(temporary_path)
    try:
        library_version = version("ucimlrepo")
    except PackageNotFoundError:
        library_version = "unavailable"
    manifest = {
        "uci_dataset_id": UCI_DATASET_ID,
        "dataset": "Default of Credit Card Clients",
        "author": "I-Cheng Yeh",
        "doi": "https://doi.org/10.24432/C55S3H",
        "license": "CC BY 4.0",
        "source_data_url": source_url,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "ucimlrepo_version": library_version,
        "snapshot_kind": "CSV export of ucimlrepo.data.original (not original XLS bytes)",
        "snapshot_filename": SNAPSHOT_FILENAME,
        "sha256": snapshot_hash,
        "row_count": len(original),
        "column_count": len(original.columns),
        "feature_count": len(FEATURE_COLUMNS),
        "source_column_names": list(original.columns),
        "target_original_name": original.columns[-1],
        "changes": [
            "CSV serialization of data.original without row index",
            "no filtering, imputation or categorical recoding",
            "normalization of ID/target and optional X1..X23 names in application memory",
        ],
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(snapshot_path)
    temporary_manifest.replace(manifest_path)
    return validated
