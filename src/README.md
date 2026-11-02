# Credit Risk Scoring System

An educational, reproducible Python application for predicting the probability of credit card default in the next month. **Not for real lending decisions.**

**Current stage:** 1 (official UCI download, schema validation, raw snapshot, data quality report). PostgreSQL SQL ingestion, EDA, and ML are **not implemented yet**.

## Target architecture

```text
UCI dataset -> raw snapshot -> data validation -> PostgreSQL tables
                                                -> SQL feature dataset
                                                -> train/validation/test
                                                -> ML pipelines and reports
                                                -> offline prediction
```

Python code lives under `src/credit_risk/`; exploratory notebooks must call Python modules rather than duplicate production logic. PostgreSQL runs in Docker, while Python tooling runs on the host. Data files and large model artifacts stay outside Git.

## Data provenance

- Dataset: [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), UCI Machine Learning Repository, dataset ID 350.
- Original author: I-Cheng Yeh.
- DOI: https://doi.org/10.24432/C55S3H
- License: CC BY 4.0.
- Stage 1 downloads the standardized table directly through `ucimlrepo.fetch_ucirepo(id=350)`. The local file `data/raw/uci_350_original.csv` is a CSV export of `data.original` (not a byte-identical copy of the original UCI XLS). No rows are dropped, no values are recoded, and no missing data are imputed.
- `data/raw/uci_350_manifest.json` captures source URL, retrieval timestamp, version of `ucimlrepo`, data dimensions and SHA-256 of the local CSV snapshot. An unchanged snapshot is reused on subsequent runs; a corrupted copy fails validation.
- UCI does not provide a dataset revision in the return value; timestamp and SHA-256 identify the downloaded snapshot. Check `manifest.json` for the local version used.
- Internally, `ID` is mapped to `client_id`, the source target name to `default_next_month`, and any `X1`...`X23` headers to their documented descriptive equivalents (`LIMIT_BAL`...`PAY_AMT6`). All category values stay unchanged.

## Windows CMD setup

Prerequisites: Git, Docker Desktop with Linux containers enabled, and `uv`.

Install uv (from CMD):

```cmd
winget install --id=astral-sh.uv -e
```

Open a **new** CMD session so `uv` is available on PATH, then, from the project root:

```cmd
uv python install 3.12
uv sync --locked
copy .env.example .env
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

```cmd
mkdir data\raw data\processed artifacts reports\figures
```

`docker compose down` stops/removes the container but keeps the named volume. Do **not** use `docker compose down -v` unless deleting the development database is intentional.

`POSTGRES_PASSWORD` in `.env.example` is a **public example for an isolated local environment**, not a production secret. The working `.env` is ignored by Git. The PostgreSQL port is bound to `127.0.0.1`.

## Stage 1: official download and data quality

Run these commands **from the repository root** with network access to the official UCI site:

```cmd
uv run --locked credit-risk fetch-data
uv run --locked credit-risk data-quality
uv run --locked ruff check .
uv run --locked pytest
```

Outputs:

- `data/raw/uci_350_original.csv`: locally serialized copy of the exact `ucimlrepo.data.original` table (25 columns, 30,000 records expected).
- `data/raw/uci_350_manifest.json`: SHA-256 integrity manifest and data provenance.
- `reports/data_quality.json` and `reports/data_quality.md`: machine-readable and human-readable quality reports.

`fetch-data` uses a verified local snapshot when available. To fetch a new copy from UCI deliberately:

```cmd
uv run --locked credit-risk fetch-data --refresh
```

To inspect the outputs in CMD:

```cmd
type data\raw\uci_350_manifest.json
type reports\data_quality.md
```

**What validation means:** a missing column, unexpected numeric dtype, duplicate client ID,
non-binary target or unexpected row count aborts the command. Unusual categorical codes
(e.g. -2 and 0 in repayment history) and negative bill balances are reported, **not silently
corrected**. Any missing feature values are reported rather than imputed. Stage 2 adds
monthly history completeness and duplicate `(client_id, month_index)` checks after
normalizing the source table.

**Data policy:** don't add data files and downloaded snapshots to Git. `data/raw/` is ignored.
Do not use the entire dataset for EDA-driven model design in later stages: once client
splits exist, statistical exploration and transformations will use training data only.

## Checks and acceptance criteria for stage 0

- `uv run --locked credit-risk check-env` prints `OK: Python 3.12...`.
- `docker compose ps` shows PostgreSQL as healthy.
- `uv run --locked credit-risk check-db` prints `OK: PostgreSQL 16...`.
- `uv run --locked ruff check .` and `uv run --locked pytest` exit with code 0.
- CI performs lint and unit/integration tests with synthetic data; it does not access the live database or UCI server.

No prediction metrics exist at this stage.
