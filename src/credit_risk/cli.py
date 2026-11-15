"""Command-line interface for data preparation, PostgreSQL ETL train-only EDA and baseline training."""

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
from credit_risk.modeling.split import (
    SPLIT_NAMES,
    SplitError,
    create_or_load_splits,
    load_splits,
    training_subset,
)

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
    subparsers.add_parser("load-db", help="Load verified snapshot into PostgreSQL tables")
    subparsers.add_parser("db-analysis", help="Run named analytical SQL queries against PostgreSQL")
    subparsers.add_parser("check-features", help="Check the SQL feature view and target separation")
    subparsers.add_parser("make-splits", help="Create or verify stratified client splits")
    subparsers.add_parser("run-eda", help="Generate EDA and statistics on train partition only")
    subparsers.add_parser("train-baseline", help="Fit baseline models on train; evaluate validation")
    subparsers.add_parser("compare-models", help="Train-only CV and validation model selection")
    subparsers.add_parser("evaluate-test", help="Evaluate the frozen selected model on test once")
    args = parser.parse_args(argv)

    if args.command == "check-env":
        return check_environment()
    if args.command == "check-db":
        return check_database()

    raw_dir = Path.cwd() / "data" / "raw"
    processed_dir = Path.cwd() / "data" / "processed"
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

            dataset = load_feature_dataset(engine)
            if args.command == "check-features":
                print("OK: feature view and target are aligned by client_id.")
                print(f"Clients: {len(dataset.client_ids)}")
                print(f"Model features: {dataset.X.shape[1]}")
                print("client_id and default_next_month are excluded from X.")
                return 0

            if args.command == "make-splits":
                assignments = create_or_load_splits(dataset, processed_dir)
                print("OK: train/validation/test client splits are verified (seed=42).")
                for name in SPLIT_NAMES:
                    print(f"{name}: {(assignments['split'] == name).sum()}")
                print(f"Saved in: {processed_dir}")
                return 0

            if args.command == "train-baseline":
                from credit_risk.modeling.train import load_baseline_config, run_baseline

                assignments = load_splits(dataset, processed_dir)
                config = load_baseline_config(Path.cwd() / "configs" / "training.yaml")
                summary = run_baseline(
                    dataset, assignments, config,
                    reports_dir=Path.cwd() / "reports",
                    artifacts_dir=Path.cwd() / "artifacts" / "baseline",
                )
                print("OK: baseline models fitted on train; evaluated on validation only.")
                print(f"Train: {summary['train_clients']}; validation: {summary['validation_clients']}")
                print(f"Test reserved: {summary['test_clients_reserved']} (not evaluated).")
                for name, result in summary["models"].items():
                    metrics = result["validation_at_selected_threshold"]
                    print(f"{name}: ROC-AUC={metrics['roc_auc']:.6f}, AP={metrics['average_precision']:.6f}, F1={metrics['f1']:.6f}, threshold={metrics['threshold']:.6f}")
                print("Report: reports/baseline.md")
                return 0

            if args.command in {"compare-models", "evaluate-test"}:
                from credit_risk.modeling.compare import evaluate_final_test, run_comparison
                from credit_risk.modeling.train import load_baseline_config

                assignments = load_splits(dataset, processed_dir)
                reports_dir = Path.cwd() / "reports"
                artifacts_dir = Path.cwd() / "artifacts" / "selected"
                if args.command == "compare-models":
                    config = load_baseline_config(Path.cwd() / "configs" / "training.yaml")
                    summary = run_comparison(
                        dataset, assignments, config,
                        reports_dir=reports_dir, artifacts_dir=artifacts_dir,
                    )
                    for name, metrics in summary["validation_metrics"].items():
                        print(f"{name}: ROC-AUC={metrics['roc_auc']:.6f}, AP={metrics['average_precision']:.6f}, F1={metrics['f1']:.6f}")
                    print(f"Selected: {summary['selected_model']}; threshold={summary['selected_threshold']:.8f}")
                    print("Test not evaluated. Report: reports/comparison.md")
                else:
                    result = evaluate_final_test(
                        dataset, assignments, reports_dir=reports_dir, artifacts_dir=artifacts_dir,
                    )
                    metrics = result["metrics"]
                    print(f"OK: final test evaluated for {result['selected_model']}; clients={result['test_clients']}")
                    print(f"ROC-AUC={metrics['roc_auc']:.6f}, AP={metrics['average_precision']:.6f}, F1={metrics['f1']:.6f}")
                    print("Report: reports/final_test.md")
                return 0

            if args.command == "run-eda":
                from credit_risk.reporting.eda import run_eda

                assignments = load_splits(dataset, processed_dir)
                X_train, y_train = training_subset(dataset, assignments)
                report_path, stats_path, figures = run_eda(
                    X_train, y_train, Path.cwd() / "reports"
                )
                print(f"OK: EDA completed on train only ({len(X_train)} clients).")
                print(f"EDA: {report_path}")
                print(f"Statistics: {stats_path}")
                print(f"Figures generated: {len(figures)}")
                return 0
        finally:
            engine.dispose()
    except (
        DatabaseConfigurationError,
        DatabaseLoadError,
        DatasetSchemaError,
        FeatureDatasetError,
        SnapshotError,
        SplitError,
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
