# Credit Risk Scoring System

A reproducible educational Python project for estimating the probability of credit card default in the next month. **It is not intended for real credit decisions.**

**Current implementation stage:** 2 — official data snapshot, Data Quality report, normalized PostgreSQL loading, analytical SQL and a leakage-safe feature/target interface. EDA and model training are not implemented yet.

## Architecture

```text
UCI dataset 350
    -> validated raw CSV + SHA-256 manifest
    -> Data Quality report
    -> PostgreSQL: clients / monthly_history / targets
    -> SQL analytical queries (CTE, JOIN, window functions)
    -> ml_feature_dataset view (predictors + client_id key only)
    -> Python feature/target separation (X, y and IDs are separate)
    -> later: train/validation/test, model pipelines and offline prediction
```

Python code lives under `src/credit_risk/`. SQL definitions and analytical queries live under `sql/`. Notebooks must call Python modules instead of duplicating project logic. PostgreSQL runs in Docker; Python runs on the host. Downloaded data and model artifacts are not committed to Git.

## Data provenance

- Dataset: [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), UCI Machine Learning Repository, dataset ID 350.
- Original author: I-Cheng Yeh.
- DOI: https://doi.org/10.24432/C55S3H
- License: CC BY 4.0.
- The project downloads the standardized table through `ucimlrepo.fetch_ucirepo(id=350)`. `data/raw/uci_350_original.csv` is a CSV export of `data.original`, not a byte-identical copy of the original XLS.
- `data/raw/uci_350_manifest.json` records the source URL, retrieval timestamp, `ucimlrepo` version, dimensions and SHA-256. A changed or corrupted local snapshot fails validation.
- Internally, `ID` is normalized to `client_id`, the target to `default_next_month`, and `X1`…`X23` headers, if returned by the source, are mapped to descriptive feature names. Source values are not recoded or imputed.

## Windows CMD setup

Prerequisites: Git, Docker Desktop with Linux containers enabled, and `uv`. Run commands from the repository root in **CMD**:

```cmd
uv python install 3.12
uv sync --locked
copy .env.example .env
uv run --locked credit-risk check-env
docker compose config
docker compose up -d postgres
docker compose ps
uv run --locked credit-risk check-db
```

If `uv.lock` does not exist in a fresh checkout, generate it with `uv lock` and commit the resulting lock file before using `--locked` commands.

`POSTGRES_PASSWORD` in `.env.example` is a public example for an isolated local environment, not a production secret. The real `.env` is ignored by Git. PostgreSQL is bound to `127.0.0.1` on port `55432` by default.

## Stage 1: official download and data quality

```cmd
uv run --locked credit-risk fetch-data
uv run --locked credit-risk data-quality
uv run --locked ruff check .
uv run --locked pytest
```

Outputs:

- `data/raw/uci_350_original.csv` — local CSV snapshot.
- `data/raw/uci_350_manifest.json` — source metadata and SHA-256.
- `reports/data_quality.json` and `reports/data_quality.md` — machine-readable and human-readable quality reports.

The current checked snapshot contains 30,000 clients, 23 predictors and one target. The report found no missing values or duplicate client IDs. It also reports undocumented category codes and negative bill amounts without silently correcting them. Those values are retained for documented treatment decisions later.

## Stage 2: PostgreSQL and analytical SQL

Ensure the local PostgreSQL container is running and the verified snapshot exists. Then run:

```cmd
docker compose up -d postgres
uv run --locked credit-risk check-db
uv run --locked credit-risk load-db
uv run --locked credit-risk db-analysis
uv run --locked credit-risk check-features
uv run --locked ruff check .
uv run --locked pytest
```

`load-db` creates the normalized tables, replaces their project data inside one transaction, creates the SQL feature view and verifies counts. If the import fails, PostgreSQL rolls the transaction back. Running the command again does not create duplicate clients or client-month rows.

Expected post-load counts:

| Table/view | Expected rows | Key invariant |
|---|---:|---|
| `clients` | 30,000 | `client_id` primary key |
| `monthly_history` | 180,000 | unique `(client_id, month_index)`; six periods per client |
| `targets` | 30,000 | one binary target per client |
| `ml_feature_dataset` | 30,000 | one feature row per client; target is not selected |

### Time mapping

`month_index` is chronological: `1` = April 2005, `6` = September 2005. The status columns map as `PAY_6`, `PAY_5`, `PAY_4`, `PAY_3`, `PAY_2`, `PAY_0`; bill and payment columns map as suffixes `6`, `5`, `4`, `3`, `2`, `1` respectively. This is tested explicitly because the source's `PAY_0` naming differs from the other status columns.

### Normalized schema

- `clients`: demographic and credit-limit fields.
- `monthly_history`: one row per client and month, with raw repayment status, bill amount and payment amount.
- `targets`: `default_next_month` in `{0, 1}`.

The schema keeps unusual categorical codes and negative bill balances as source values. Constraints enforce primary/foreign keys, valid month indices and a binary target; they do not attempt to reinterpret source categories.

### Analytical SQL and feature view

`sql/002_analytical_queries.sql` contains named queries for target prevalence, credit-limit/age summaries, monthly bill/payment summaries, repayment-status distributions, repeated delays, monthly bill changes with `LAG`, top payment totals with `ROW_NUMBER`, and join integrity. `credit-risk db-analysis` prints their real query results; results are not hard-coded.

`sql/003_feature_dataset.sql` creates `ml_feature_dataset`, including the six monthly values and summary features such as maximum repayment status, count of delayed months, average bill/payment, six-month bill change and payment-to-absolute-bill ratio. It contains `client_id` as a join key but does **not** include `default_next_month`.

`credit_risk.features.build.load_feature_dataset()` runs separate SQL queries for features and target, aligns them by `client_id`, and returns three separate objects: `X`, `y` and `client_ids`. Neither `client_id` nor `default_next_month` is included in `X`. The ratio feature can be `NULL` if the denominator is zero; any missing-value treatment belongs in a later train-fitted ML pipeline, not in ETL.

## Checks and scope

- `uv run --locked credit-risk check-env` validates Python and core imports.
- `uv run --locked credit-risk check-db` checks PostgreSQL connectivity and major version 16.
- `uv run --locked credit-risk load-db` verifies the 30,000 / 180,000 / 30,000 / 30,000 row counts, complete six-month history and client/target joins.
- `uv run --locked credit-risk check-features` checks feature/target key alignment and leakage boundaries.
- `uv run --locked ruff check .` and `uv run --locked pytest` run static checks and tests. Unit tests use synthetic data and do not require the live database; the CLI commands above perform local database integration checks.

Later stages will add EDA and statistical tests, reproducible train/validation/test splits, baseline and additional models, evaluation, a saved Pipeline and offline prediction. No model metrics are claimed before those experiments are actually run.
