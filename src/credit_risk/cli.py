"""Command-line interface for data validation, PostgreSQL ETL and feature checks."""

import argparse
import importlib
import json
import sys
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from credit_risk.data.database import DatabaseConfigurationError, create_database_engine
from credit_risk.data.db_loader import (
    DatabaseLoadError,
    load_relational_data,
    run_analytical_queries,
)
from credit_risk.data.fetch import SnapshotError, fetch_and_store, load_snapshot
from credit_risk.data.quality import build_quality_report, save_quality_report
from credit_risk.data.schema import DatasetSchemaError
from credit_risk.features.build import FeatureDatasetError, load_feature_dataset

REQUIRED_MODULES = (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "matplotlib",
    "seaborn",
    "sqlalchemy",
    "psycopg",
    "ucimlrepo",
    "joblib",
    "yaml",
)


def check_environment() -> int:
    """Check the interpreter and import each core runtime dependency."""
    if sys.version_info[:2] != (3, 12):
        print("ERROR: Python 3.12 is required.", file=sys.stderr)
        return 1

    try:
        for module_name in REQUIRED_MODULES:
            importlib.import_module(module_name)
    except ImportError as exc:
        print(f"ERROR: dependency import failed: {exc}", file=sys.stderr)
        return 1

    print("OK: Python 3.12 and all core dependencies are importable.")
    return 0


def check_database() -> int:
    """Connect to PostgreSQL and validate its major version."""
    try:
        engine = create_database_engine()
    except DatabaseConfigurationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1")).scalar_one()
            version_number = int(
                connection.execute(text("SHOW server_version_num")).scalar_one()
            )
    except SQLAlchemyError as exc:
        print(f"ERROR: PostgreSQL connection failed: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()

    major_version = version_number // 10000
    if major_version != 16:
        print(f"ERROR: expected PostgreSQL 16, got PostgreSQL {major_version}.", file=sys.stderr)
        return 1

    print(f"OK: PostgreSQL {major_version} is reachable and SELECT 1 succeeded.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="credit-risk")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check-env", help="Verify Python and installed dependencies")
    subparsers.add_parser("check-db", help="Verify PostgreSQL 16 connection")

    fetch_parser = subparsers.add_parser(
        "fetch-data", help="Download and verify UCI dataset 350; reuse local snapshot"
    )
    fetch_parser.add_argument(
        "--refresh", action="store_true", help="Download again instead of using local snapshot"
    )
    subparsers.add_parser("data-quality", help="Write data-quality reports from verified snapshot")
    subparsers.add_parser("load-db", help="Load the verified snapshot into normalized PostgreSQL tables")
    subparsers.add_parser("db-analysis", help="Run named analytical SQL queries against PostgreSQL")
    subparsers.add_parser("check-features", help="Check the SQL feature view and target separation")
    args = parser.parse_args(argv)

    if args.command == "check-env":
        return check_environment()
    if args.command == "check-db":
        return check_database()

    raw_dir = Path.cwd() / "data" / "raw"
    sql_dir = Path.cwd() / "sql"

    if args.command == "fetch-data":
        try:
            data = fetch_and_store(raw_dir, refresh=args.refresh)
        except (DatasetSchemaError, SnapshotError, ConnectionError, OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"OK: UCI 350 snapshot verified: {len(data)} clients, 23 features.")
        print(f"Raw CSV and manifest: {raw_dir}")
        return 0

    if args.command == "data-quality":
        try:
            data = load_snapshot(raw_dir)
            report = build_quality_report(data)
            json_path, md_path = save_quality_report(report, Path.cwd() / "reports")
        except (DatasetSchemaError, SnapshotError, ConnectionError, OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"OK: Data quality report generated for {len(data)} clients.")
        print(f"JSON: {json_path}")
        print(f"Markdown: {md_path}")
        return 0

    try:
        engine = create_database_engine()
        try:
            if args.command == "load-db":
                data = load_snapshot(raw_dir)
                summary = load_relational_data(engine, data, sql_dir=sql_dir)
                print("OK: PostgreSQL load completed and integrity checks passed.")
                print(f"Clients: {summary.client_count}")
                print(f"Monthly history rows: {summary.history_count}")
                print(f"Targets: {summary.target_count}")
                print(f"Feature view rows: {summary.feature_count}")
                return 0

            if args.command == "db-analysis":
                results = run_analytical_queries(engine, sql_dir / "002_analytical_queries.sql")
                for name, rows in results.items():
                    print(f"\n[{name}]")
                    print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
                return 0

            if args.command == "check-features":
                dataset = load_feature_dataset(engine)
                print("OK: feature view and target are aligned by client_id.")
                print(f"Clients: {len(dataset.client_ids)}")
                print(f"Model features: {dataset.X.shape[1]}")
                print("client_id and default_next_month are excluded from X.")
                return 0
        finally:
            engine.dispose()
    except (
        DatabaseConfigurationError,
        DatabaseLoadError,
        DatasetSchemaError,
        FeatureDatasetError,
        SnapshotError,
        SQLAlchemyError,
        ConnectionError,
        OSError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
