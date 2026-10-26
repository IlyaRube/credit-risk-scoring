"""Stage 0 CLI for verifying the Python environment and PostgreSQL connection."""

import argparse
import importlib
import os
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError

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
    """Connect to PostgreSQL from Windows and validate the server major version."""
    load_dotenv()
    required = ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        print(f"ERROR: missing environment variables: {', '.join(missing)}", file=sys.stderr)
        return 1

    try:
        port = int(os.getenv("POSTGRES_PORT", "55432"))
        if not 1 <= port <= 65535:
            raise ValueError("port out of range")
    except ValueError:
        print("ERROR: POSTGRES_PORT must be an integer from 1 to 65535.", file=sys.stderr)
        return 1

    database_url = URL.create(
        drivername="postgresql+psycopg",
        username=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host="127.0.0.1",
        port=port,
        database=os.environ["POSTGRES_DB"],
    )
    engine = create_engine(database_url, connect_args={"connect_timeout": 5})
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
    args = parser.parse_args(argv)

    if args.command == "check-env":
        return check_environment()
    return check_database()


if __name__ == "__main__":
    raise SystemExit(main())
