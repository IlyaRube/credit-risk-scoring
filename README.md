# Credit Risk Scoring System

A reproducible educational system for analyzing credit-card payment defaults (UCI dataset 350). **Not for real credit decisions.**

**Current stage: 4, baseline machine learning.** Train-only preprocessing, Dummy and logistic regression; results are generated locally, and test evaluation is reserved for stage 5.

## Current data pipeline

```text
UCI 350 -- fetch-data --> verified CSV + SHA-256 manifest
                          |
                          +--> data-quality --> reports/data_quality.md
                          |
                          +--> load-db --> PostgreSQL clients / monthly_history / targets
                                              |
                                              +--> SQL feature view (30 predictors)
                                              |             + separate target by client_id
                                              |
                                              +--> make-splits (frozen 60/20/20)
                                                        |
                                                        +--> run-eda (TRAIN only)
                                                                   |
                                                                   +--> reports/eda.md
                                                                   +--> reports/statistics.md
                                                                   +--> reports/figures/*.png
```

Source: [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), ID 350; I-Cheng Yeh; [DOI 10.24432/C55S3H](https://doi.org/10.24432/C55S3H); CC BY 4.0. Source observations are for April–September 2005, with the next-payment default as the label. The UCI source returns 30,000 clients, 23 input variables and a binary target. `data/raw/uci_350_original.csv` is a local UTF-8 CSV export of the official `ucimlrepo.data.original`, **not** the original XLS file bytes. The manifest records UCI source metadata, SHA-256 and acquisition time. Values are not automatically imputed, recoded or dropped.

## Requirements and reproducibility

Python 3.12, `uv`, PostgreSQL 16 in Docker Desktop (Linux containers / WSL2), Windows CMD. Python libraries include Pandas, NumPy, SciPy, scikit-learn, matplotlib, seaborn, SQLAlchemy and psycopg. Project dependencies are controlled by `pyproject.toml` and the committed `uv.lock`; the stage 3 update introduces **no new dependencies**.

After cloning, from the repository directory in CMD:

```cmd
uv python install 3.12
uv sync --locked
copy .env.example .env
docker compose up -d postgres
uv run --locked credit-risk check-env
uv run --locked credit-risk check-db
```

The `.env` file is ignored by Git. Example credentials in `.env.example` are **local-only**. PostgreSQL is bound to loopback on port 55432 by default. Run all CLI commands from the project root.

## Stage 1: raw data and Data Quality

```cmd
uv run --locked credit-risk fetch-data
uv run --locked credit-risk data-quality
```

- `data/raw/uci_350_original.csv` and `data/raw/uci_350_manifest.json` contain the cached source snapshot and provenance/hash metadata. Raw data is excluded from Git.
- `reports/data_quality.md` and `.json` contain missing counts, schema/ID checks, unusual categorical codes and suspicious numeric values.
- A malformed schema or modified snapshot is rejected. Unusual status codes and negative bill balances are **not** silently corrected.

## Stage 2: PostgreSQL loading and SQL feature view

```cmd
uv run --locked credit-risk load-db
uv run --locked credit-risk db-analysis
uv run --locked credit-risk check-features
```

Relational tables: `clients` (one row per client), `monthly_history` (one per client-month, month 1 = April 2005, month 6 = September 2005), `targets` (binary label, separate from predictors). An atomic repeatable load verifies 30,000 clients, 180,000 history records, 30,000 targets and 30,000 SQL feature rows. 

- `sql/001_schema.sql` creates tables and constraints.
- `sql/002_analytical_queries.sql` contains aggregates, JOIN, CTE and window functions `LAG` and `ROW_NUMBER`.
- `sql/003_feature_dataset.sql` creates `ml_feature_dataset` with the 23 original predictor values, in six-month chronological order, and 7 derived features (30 total). The target is not present in the view.
- `src/credit_risk/features/build.py` separately queries `X`, `y`, and `client_ids`, preserving alignment and preventing identifier/target leakage into `X`.

## Stage 3: EDA and statistical inference

**Critical order:** create a permanent stratified split **before** looking at target-dependent EDA. This stage uses the **train partition only**. Validation and test are reserved for model selection and final evaluation in later stages.

```cmd
uv run --locked credit-risk check-features
uv run --locked credit-risk make-splits
uv run --locked credit-risk run-eda
uv run --locked ruff check .
uv run --locked pytest
```

`make-splits` creates exactly 60%/20%/20% stratified client partitions using seed `42` (30,000 clients -> 18,000 / 6,000 / 6,000). It creates:

- `data/processed/client_splits.csv`: client ID and assigned partition.
- `data/processed/client_splits_meta.json`: seed, sizes, fingerprint of sorted client IDs/targets, and SHA-256 of saved split CSV.

Both files are local and ignored by Git. Subsequent runs **reuse the saved assignments**, do not resample, and fail if the labels or saved split file change. Keep them stable for stages 4–6. Back up these two files outside Git if needed for exact reproduction across computers. A fresh machine will reconstruct the same assignments with the same locked dependencies and source data.

`run-eda` requires the saved split. It analyzes only the 18,000 train clients and generates:

| Output | Contents |
|---|---|
| `reports/eda.md` | Train-only distributions, descriptive statistics, prevalence, group comparison, links to PNG charts, limitations |
| `reports/eda_summary.json` | Structured descriptive statistics with count, mean, median, Q25/Q75, sample variance/std, IQR outlier counts |
| `reports/statistics.md` | Two prespecified statistical tests with results and assumptions, 95% Wilson interval |
| `reports/statistics.json` | Machine-readable test statistics, raw and Holm-adjusted p-values, effect sizes |
| `reports/figures/age_histogram.png` | Train age distribution |
| `reports/figures/credit_limit_histogram.png` | Train credit-limit distribution |
| `reports/figures/credit_limit_by_default_boxplot.png` | Train credit limits grouped by outcome |
| `reports/figures/spearman_correlation.png` | Selected train Spearman associations including outcome |
| `reports/figures/default_rate_by_delinquency.png` | Train default rate by number of delinquent months |

**Hypothesis 1:** Chi-square independence of having at least one delinquent month vs default. Reject asymptotic test when any expected cell count is below 5. Phi coefficient describes magnitude. **Hypothesis 2:** two-sided Mann–Whitney U for credit-limit distributions across the two default groups. Report rank-biserial effect size. Correct both p-values by **Holm** (family-wise alpha 0.05). A 95% Wilson interval estimates the train default proportion.

**Statistical cautions:** p-value is not the probability of a true null hypothesis; Mann–Whitney does not automatically test median equality; confidence interval coverage is a repeated-sampling property; correlations are not causation; significance does not imply usefulness for prediction; independent observations and representative sampling are assumptions, not guarantees. Historical Taiwanese credit-card data should not be generalized to modern banking without evidence. Unusual category values and bill balances are preserved, not automatically changed. Numeric fields coded as categories are kept distinct from continuous descriptive variables.

Reports are **generated from the real locally loaded PostgreSQL data**. No metric, conclusion or p-value is hard-coded. Raw data, saved client partitions and fitted models are git-ignored. Reviewed Markdown/JSON reports and the five generated PNG figures can be committed, so GitHub visitors can inspect actual EDA results and charts.

For a notebook view, open `notebooks/01_exploratory_analysis.ipynb` after generating reports; it reuses project functions and outputs instead of implementing a second analysis pipeline.

## Stage 4: fixed baseline models

Run from the repository root after PostgreSQL has been loaded and the stage 3 splits exist. **Do not regenerate the frozen splits.**

```cmd
uv run --locked credit-risk train-baseline
uv run --locked ruff check .
uv run --locked pytest
```

`configs/training.yaml` fixes seed 42, logistic regression C=1.0 and max_iter=2000. `modeling/train.py` reads the existing feature dataset and saved assignments, and fits two independent sklearn pipelines on train (18,000 clients):

- DummyClassifier (`strategy=prior`): returns the training default prevalence, regardless of input. A ranking baseline with no learned feature relationships.
- LogisticRegression (`lbfgs`, default L2 penalty): numeric median imputation and StandardScaler; nominal codes (sex, education, marriage, six repayment statuses and maximum repayment status) use train-fitted most-frequent imputation and one-hot encoding with unknown categories ignored. Remaining features are numeric. No rebalancing or hyperparameter search is performed.

Both pipelines use the same 30 SQL inputs; categorical encoding expands the matrix internally. Null values such as the ratio with a zero denominator are handled by train-fitted imputers. Neither client ID nor the target is a feature. Validation is transformed and predicted without fitting, and the test partition is not scored.

`modeling/evaluate.py` calculates ROC-AUC, average precision, Brier score, precision, recall, F1 and a 2x2 confusion matrix (rows actual, columns predicted, label order 0/1). It selects the threshold maximizing validation F1 separately for each model (highest threshold on exact ties; prediction is positive when probability >= threshold). The report includes train and validation metrics at 0.5 plus validation metrics at the selected threshold. The latter are selection-dependent, not unbiased final estimates. No final model is selected in stage 4. Stage 5 will use train-only cross-validation and a common documented selection policy, then evaluate the final choice once on the untouched test partition.

Generated files:

| File | Contents |
|---|---|
| `reports/baseline.md` | Actual metrics, model comparison, coefficients, limitations and figures |
| `reports/baseline.json` | Machine-readable metrics, configuration, library versions, split hash and feature lists |
| `reports/figures/baseline_validation_roc.png` | Validation ROC curves |
| `reports/figures/baseline_validation_pr.png` | Validation precision-recall curves |
| `reports/figures/baseline_validation_calibration.png` | Validation reliability curves (no calibration model fitted) |
| `artifacts/baseline/dummy.joblib`, `logistic.joblib` | Local fitted pipelines, including preprocessors |
| `artifacts/baseline/metadata.json` | Input schema, configuration, selected thresholds, metrics, split hash and versions |

Reports and figures may be published after review; `artifacts/` remains ignored. Only load locally generated trusted joblib files. Model coefficients are descriptive associations, not causal effects; correlated features and demographic fields require caution. This is an educational baseline, not a lending decision system.

## Tests and future stages

```cmd
uv run --locked ruff check .
uv run --locked pytest
```

Unit tests check data integrity, chronology, feature/target separation, deterministic splits and statistical calculations. Stage 3 tests use synthetic data; integration with local PostgreSQL is verified by `make-splits` and `run-eda` CLI runs. GitHub Actions runs unit tests and Ruff, without requiring an external DB.

**Stage 4:** baseline code, train-only preprocessing, validation metrics and leakage tests implemented; run `train-baseline` to generate real results. **Stage 5:** Random Forest, boosting, CV and one-time test evaluation. **Stage 6:** full saved Pipeline, offline prediction, model card and reproducibility checks. No model quality claims exist before these experiments are run.
