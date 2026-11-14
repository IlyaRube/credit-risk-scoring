"""Fit fixed baseline pipelines on train and compare on validation only."""

import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.features.build import FeatureDataset
from credit_risk.modeling.evaluate import binary_metrics, select_f1_threshold
from credit_risk.modeling.split import RANDOM_STATE, training_subset

matplotlib.use("Agg")

CATEGORICAL_FEATURES = (
    "sex", "education", "marriage",
    *(f"repayment_status_m{i}" for i in range(1, 7)),
    "max_repayment_status",
)
NUMERIC_FEATURES = (
    "limit_bal", "age",
    *(f"bill_amount_m{i}" for i in range(1, 7)),
    *(f"payment_amount_m{i}" for i in range(1, 7)),
    "delinquent_month_count", "avg_bill_amount", "avg_payment_amount",
    "total_payment_amount", "bill_amount_change_6m", "payment_to_abs_bill_ratio",
)
FEATURE_NAMES = frozenset((*CATEGORICAL_FEATURES, *NUMERIC_FEATURES))


@dataclass(frozen=True)
class BaselineConfig:
    random_state: int = RANDOM_STATE
    C: float = 1.0
    max_iter: int = 2000


def load_baseline_config(path: Path) -> BaselineConfig:
    try:
        with path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid training YAML: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != {"random_state", "logistic", "threshold_selection"}:
        raise ValueError("Unexpected baseline configuration fields.")
    logistic = raw["logistic"]
    if not isinstance(logistic, dict) or set(logistic) != {"C", "max_iter"}:
        raise ValueError("Expected logistic C and max_iter.")
    if raw["random_state"] != RANDOM_STATE or raw["threshold_selection"] != "max_validation_f1":
        raise ValueError("Baseline requires seed 42 and max_validation_f1 threshold selection.")
    c, max_iter = logistic["C"], logistic["max_iter"]
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not np.isfinite(c) or c <= 0:
        raise ValueError("Logistic C must be finite and positive.")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("Logistic max_iter must be a positive integer.")
    return BaselineConfig(C=float(c), max_iter=max_iter)


def make_preprocessor() -> ColumnTransformer:
    """Keep source category codes nominal; learn imputation/scaling from train."""
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("numeric", numeric, list(NUMERIC_FEATURES)),
        ("categorical", categorical, list(CATEGORICAL_FEATURES)),
    ], remainder="drop")


def make_baseline_models(config: BaselineConfig) -> dict[str, Pipeline]:
    """Each estimator owns an independent preprocessor; nothing is fitted yet."""
    return {
        "dummy": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", DummyClassifier(strategy="prior")),
        ]),
        "logistic": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", LogisticRegression(
                C=config.C, max_iter=config.max_iter, random_state=config.random_state,
                solver="lbfgs",
            )),
        ]),
    }


def positive_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Find class 1 explicitly instead of assuming its probability column."""
    classes = np.asarray(model.classes_)
    indices = np.flatnonzero(classes == 1)
    if len(indices) != 1:
        raise ValueError("Fitted model does not contain class 1.")
    probability = model.predict_proba(X)[:, int(indices[0])]
    if len(probability) != len(X) or not np.isfinite(probability).all():
        raise ValueError("Invalid predicted probabilities.")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("Predicted probability outside [0, 1].")
    return probability


def _check_features(X: pd.DataFrame) -> None:
    if X.columns.has_duplicates or set(X.columns) != FEATURE_NAMES:
        raise ValueError("Expected the 30 SQL model features, without client_id or target.")
    for name in X.columns:
        if not pd.api.types.is_numeric_dtype(X[name]):
            raise ValueError(f"Feature {name} must be numeric or integer-coded categorical.")
        values = X[name].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any():
            raise ValueError(f"Feature {name} contains infinity.")


def fit_baselines(
    dataset: FeatureDataset, assignment: pd.DataFrame, config: BaselineConfig
) -> tuple[dict[str, Pipeline], dict, pd.Series, dict[str, np.ndarray]]:
    """Never fit, predict or select thresholds using test rows."""
    # Existing stage 3 function validates IDs, sizes and disjoint assignments.
    X_train, y_train = training_subset(dataset, assignment)
    mask = dataset.client_ids.isin(
        assignment.loc[assignment["split"] == "validation", "client_id"]
    ).to_numpy(dtype=bool)
    X_val = dataset.X.iloc[np.flatnonzero(mask)].reset_index(drop=True).copy()
    y_val = dataset.y.iloc[np.flatnonzero(mask)].reset_index(drop=True).copy()
    _check_features(X_train)
    _check_features(X_val)
    if set(y_train.unique()) != {0, 1} or set(y_val.unique()) != {0, 1}:
        raise ValueError("Train and validation must each contain both classes.")
    models = make_baseline_models(config)
    results = {}
    probabilities = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_prob = positive_probability(model, X_train)
        val_prob = positive_probability(model, X_val)
        threshold = select_f1_threshold(y_val, val_prob)
        probabilities[name] = val_prob
        results[name] = {
            "train_at_0_5": binary_metrics(y_train, train_prob, threshold=0.5),
            "validation_at_0_5": binary_metrics(y_val, val_prob, threshold=0.5),
            "validation_at_selected_threshold": binary_metrics(y_val, val_prob, threshold=threshold),
        }
    logistic = models["logistic"]
    names = logistic.named_steps["preprocessor"].get_feature_names_out()
    coefficients = logistic.named_steps["model"].coef_[0]
    ordered = sorted(zip(names, coefficients, strict=True), key=lambda item: -abs(item[1]))
    summary = {
        "stage": 4,
        "seed": config.random_state,
        "configuration": {"C": config.C, "max_iter": config.max_iter, "solver": "lbfgs"},
        "threshold_rule": "max validation F1; highest threshold for exact ties; predict >= threshold",
        "split_sha256": hashlib.sha256(
            assignment.sort_values("client_id").to_csv(index=False, lineterminator="\n").encode()
        ).hexdigest(),
        "train_clients": len(X_train),
        "validation_clients": len(X_val),
        "test_clients_reserved": int((assignment["split"] == "test").sum()),
        "test_evaluated": False,
        "train_default_rate": float(y_train.mean()),
        "validation_default_rate": float(y_val.mean()),
        "input_features": X_train.columns.tolist(),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")},
        "models": results,
        "logistic_top_coefficients": [
            {"feature": str(name), "coefficient": float(value)} for name, value in ordered[:10]
        ],
    }
    return models, summary, y_val, probabilities


def _plot_validation(y: pd.Series, probabilities: dict[str, np.ndarray], directory: Path) -> None:
    from matplotlib import pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    for kind, filename, xlabel, ylabel in (
        ("roc", "baseline_validation_roc.png", "False positive rate", "True positive rate"),
        ("pr", "baseline_validation_pr.png", "Recall", "Precision"),
        ("calibration", "baseline_validation_calibration.png", "Mean predicted probability", "Observed fraction of defaults"),
    ):
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, probability in probabilities.items():
            if kind == "roc":
                x, values, _ = roc_curve(y, probability)
            elif kind == "pr":
                values, x, _ = precision_recall_curve(y, probability)
            else:
                values, x = calibration_curve(y, probability, n_bins=10, strategy="uniform")
            ax.plot(x, values, marker="o" if kind == "calibration" else None, label=name)
        if kind == "pr":
            ax.axhline(float(y.mean()), linestyle="--", color="gray", label="class prevalence")
        else:
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="reference")
        ax.set(xlabel=xlabel, ylabel=ylabel, title=f"Validation {kind.upper()}", xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / filename, dpi=150)
        plt.close(fig)


def _write_report(summary: dict, path: Path) -> None:
    lines = [
        "# Stage 4: baseline models (train / validation)", "",
        f"Train: {summary['train_clients']:,} clients. Validation: {summary['validation_clients']:,}. ",
        f"Reserved test: {summary['test_clients_reserved']:,}; not evaluated.", "",
        f"Dummy uses training prevalence; logistic regression uses fixed L2 regularization (C={summary['configuration']['C']}).",
        "Categorical codes are retained as categories; numeric medians, most-frequent category imputation,",
        "one-hot vocabularies and standardization are fitted only on train. Unknown categories are ignored.",
        "No class rebalancing, tuning or test scoring is performed.", "",
        "## Metrics", "",
        "| Model | Partition / threshold | Threshold | ROC-AUC | AP | Precision | Recall | F1 | Brier | TN | FP | FN | TP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, model in summary["models"].items():
        for label, metrics in model.items():
            (tn, fp), (fn, tp) = metrics["confusion_matrix"]
            scores = " | ".join(f"{metrics[key]:.6f}" for key in (
                "threshold", "roc_auc", "average_precision", "precision", "recall", "f1", "brier_score"
            ))
            lines.append(f"| {name} | {label} | {scores} | {tn} | {fp} | {fn} | {tp} |")
    lines += ["", "## Validation observations", ""]
    dummy = summary["models"]["dummy"]["validation_at_0_5"]
    logistic = summary["models"]["logistic"]["validation_at_0_5"]
    for key, label, higher in (("roc_auc", "ROC-AUC", True), ("average_precision", "AP", True), ("brier_score", "Brier", False)):
        delta = logistic[key] - dummy[key]
        better = delta > 0 if higher else delta < 0
        relation = "better" if better else ("equal" if delta == 0 else "worse")
        lines.append(f"- Logistic {label}: {relation} than Dummy on validation (difference {delta:+.6f}).")
    lines += [
        "- Thresholds maximize F1 on validation independently per model, with highest-threshold tie breaking.",
        "- Selected-threshold validation scores are selection-dependent, not independent test estimates.",
        "- ROC-AUC and AP assess ranking, Brier assesses probability error (lower is better).",
        "- A constant Dummy score may predict every client positive at its F1-selected threshold.", "",
        "## Largest absolute logistic coefficients", "",
        "| Transformed feature | Coefficient |", "|---|---:|",
    ]
    for item in summary["logistic_top_coefficients"]:
        lines.append(f"| {item['feature']} | {item['coefficient']:.6f} |")
    lines += [
        "", "Coefficients describe model associations on the transformed scale, not causal effects.",
        "Correlated engineered features and redundant one-hot levels limit isolated interpretations.",
        "Demographic variables may raise fairness concerns; this historical dataset is not suitable for real lending decisions.",
        "", "## Validation figures", "",
        "![ROC](figures/baseline_validation_roc.png)", "",
        "![Precision-recall](figures/baseline_validation_pr.png)", "",
        "![Calibration](figures/baseline_validation_calibration.png)", "",
        "The calibration curve is descriptive; no calibration model is fitted. Test remains untouched for stage 5.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_baseline(dataset: FeatureDataset, assignment: pd.DataFrame, config: BaselineConfig, *, reports_dir: Path, artifacts_dir: Path) -> dict:
    models, summary, y_val, probabilities = fit_baselines(dataset, assignment, config)
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _plot_validation(y_val, probabilities, reports_dir / "figures")
    _write_report(summary, reports_dir / "baseline.md")
    payload = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    (reports_dir / "baseline.json").write_text(payload, encoding="utf-8")
    for name, model in models.items():
        joblib.dump(model, artifacts_dir / f"{name}.joblib")
    (artifacts_dir / "metadata.json").write_text(payload, encoding="utf-8")
    return summary
