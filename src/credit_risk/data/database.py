"""Shared PostgreSQL connection configuration for local CLI commands."""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine


class DatabaseConfigurationError(ValueError):
    """Database connection settings are missing or invalid."""


def create_database_engine() -> Engine:
    """Create an engine for the local Compose PostgreSQL service."""
    load_dotenv()
    required = ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise DatabaseConfigurationError(
            f"Missing environment variables: {', '.join(missing)}. Copy .env.example to .env."
        )

    try:
        port = int(os.getenv("POSTGRES_PORT", "55432"))
    except ValueError as exc:
        raise DatabaseConfigurationError("POSTGRES_PORT must be an integer.") from exc
    if not 1 <= port <= 65535:
        raise DatabaseConfigurationError("POSTGRES_PORT must be between 1 and 65535.")

    url = URL.create(
        drivername="postgresql+psycopg",
        username=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host="127.0.0.1",
        port=port,
        database=os.environ["POSTGRES_DB"],
    )
    return create_engine(url, connect_args={"connect_timeout": 5}, pool_pre_ping=True)
