"""Checks that can run without PostgreSQL in local development and CI."""

import pytest

from credit_risk import __version__
from credit_risk.cli import main


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_cli_environment(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check-env"]) == 0
    output = capsys.readouterr().out
    assert "OK: Python 3.12" in output
