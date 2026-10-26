# Credit Risk Scoring System

An educational, reproducible Python application for predicting the probability of credit card default in the next month. **Not for real lending decisions.**

**Current stage:** 0 (repository, environment, PostgreSQL, and smoke checks). Data loading, SQL transformations, EDA, and ML are **not implemented yet**.

## Target architecture

```text
UCI dataset -> raw snapshot -> data validation -> PostgreSQL tables
                                                -> SQL feature dataset
                                                -> train/validation/test
                                                -> ML pipelines and reports
                                                -> offline prediction
```

Python code lives under `src/credit_risk/`; exploratory notebooks must call Python modules rather than duplicate production logic. PostgreSQL runs in Docker, while Python tooling runs on the host. Data files and large model artifacts stay outside Git.

## Data provenance (for upcoming stages)

- Dataset: [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), UCI Machine Learning Repository, dataset ID 350.
- Original author: I-Cheng Yeh.
- DOI: https://doi.org/10.24432/C55S3H
- License: CC BY 4.0.
- The dataset has not yet been downloaded or modified by this repository.

## Windows / PowerShell setup

Prerequisites: Git, Docker Desktop with Linux containers enabled, and `uv`.

Install uv (in PowerShell):

```powershell
winget install --id=astral-sh.uv -e
```

Open a **new** PowerShell session so `uv` is available on PATH, then, from the project root:

```powershell
uv python install 3.12
uv sync
Copy-Item .env.example .env
uv run --locked credit-risk check-env
docker compose config
docker compose up -d postgres
docker compose ps
uv run --locked credit-risk check-db
uv run --locked ruff check .
uv run --locked pytest
```

If `uv.lock` is missing on a newly created project, first run `uv lock` and commit the generated `uv.lock` before expecting `--locked` checks to succeed.

Generated data directories for later stages can be created locally with:

```powershell
New-Item -ItemType Directory -Force data/raw, data/processed, artifacts, reports/figures | Out-Null
```

`docker compose down` stops/removes the container but keeps the named volume. Do **not** use `docker compose down -v` unless deleting the development database is intentional.

`POSTGRES_PASSWORD` in `.env.example` is a **public example for an isolated local environment**, not a production secret. The working `.env` is ignored by Git. The PostgreSQL port is bound to `127.0.0.1`.

## Checks and acceptance criteria for stage 0

- `uv run --locked credit-risk check-env` prints `OK: Python 3.12...`.
- `docker compose ps` shows PostgreSQL as healthy.
- `uv run --locked credit-risk check-db` prints `OK: PostgreSQL 16...`.
- `uv run --locked ruff check .` and `uv run --locked pytest` exit with code 0.
- CI performs lint and smoke tests; it does not test the database yet.

No prediction metrics exist at this stage.
