"""Train-only CV, validation model selection, and a separate final test evaluation."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from credit_risk.features.build import FeatureDataset
from credit_risk.modeling.evaluate import binary_metrics, select_f1_threshold
from credit_risk.modeling.split import RANDOM_STATE, training_subset
from credit_risk.modeling.train import (
    BaselineConfig,
    _check_features,
    fit_baselines,
    make_preprocessor,
    positive_probability,
)

matplotlib.use("Agg")

# Fixed before observing any stage 5 results. Keep the search small for local CPUs.
SEARCH_GRIDS = {
    "random_forest": {
        "model__max_depth": [8, None],
        "model__min_samples_leaf": [5, 20],
    },
    "hist_gradient_boosting": {
        "model__learning_rate": [0.05, 0.1],
        "model__max_leaf_nodes": [15, 31],
    },
}
CANDIDATES = ("logistic", "random_forest", "hist_gradient_boosting")
SELECTION_RULE = "Highest validation ROC-AUC, then AP, then lower Brier, then fixed candidate order."


def _partition(dataset: FeatureDataset, assignment: pd.DataFrame, name: str):
    mask = dataset.client_ids.isin(assignment.loc[assignment["split"] == name, "client_id"])
    indices = np.flatnonzero(mask.to_numpy(dtype=bool))
    return (dataset.X.iloc[indices].reset_index(drop=True).copy(),
            dataset.y.iloc[indices].reset_index(drop=True).copy())


def _split_hash(assignment: pd.DataFrame) -> str:
    return hashlib.sha256(assignment.sort_values("client_id").to_csv(
        index=False, lineterminator="\n"
    ).encode()).hexdigest()


def _development_hash(dataset: FeatureDataset, assignment: pd.DataFrame) -> str:
    """Bind the frozen model to train/validation inputs without inspecting test features."""
    mask = dataset.client_ids.isin(assignment.loc[assignment["split"] != "test", "client_id"])
    indices = np.flatnonzero(mask.to_numpy(dtype=bool))
    frame = dataset.X.iloc[indices].reset_index(drop=True).copy()
    frame.insert(0, "client_id", dataset.client_ids.iloc[indices].to_numpy())
    frame["default_next_month"] = dataset.y.iloc[indices].to_numpy()
    frame = frame.sort_values("client_id")
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def _versions() -> dict:
    return {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")}


def _search_models() -> dict[str, Pipeline]:
    return {
        "random_forest": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", RandomForestClassifier(n_estimators=120, random_state=RANDOM_STATE,
                                             n_jobs=1, max_features="sqrt")),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", HistGradientBoostingClassifier(max_iter=120, min_samples_leaf=20,
                                                    early_stopping=False, random_state=RANDOM_STATE)),
        ]),
    }


def choose_model(results: dict) -> str:
    """Dummy is a reference, not a candidate for deployment."""
    return max(CANDIDATES, key=lambda name: (
        results[name]["roc_auc"], results[name]["average_precision"],
        -results[name]["brier_score"], -CANDIDATES.index(name),
    ))


def fit_comparison(dataset: FeatureDataset, assignment: pd.DataFrame,
                   config: BaselineConfig) -> tuple[Pipeline, dict, pd.Series, dict]:
    """Fit preprocessing separately in every fold; never score test here."""
    X_train, y_train = training_subset(dataset, assignment)
    X_val, y_val = _partition(dataset, assignment, "validation")
    _check_features(X_train)
    _check_features(X_val)
    if set(y_train.unique()) != {0, 1} or set(y_val.unique()) != {0, 1}:
        raise ValueError("Train and validation must each contain both classes.")
    if y_train.value_counts().min() < 3:
        raise ValueError("Three-fold CV requires at least three train observations per class.")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search_results = {}
    with threadpool_limits(limits=1):
        models, baseline, _, probabilities = fit_baselines(dataset, assignment, config)
        results = {name: item["validation_at_selected_threshold"]
                   for name, item in baseline["models"].items()}
        for name, pipeline in _search_models().items():
            search = GridSearchCV(pipeline, SEARCH_GRIDS[name], scoring="roc_auc", cv=cv,
                                  n_jobs=1, refit=True, error_score="raise", return_train_score=False)
            search.fit(X_train, y_train)
            models[name] = search.best_estimator_
            scores = positive_probability(models[name], X_val)
            probabilities[name] = scores
            results[name] = binary_metrics(y_val, scores, threshold=select_f1_threshold(y_val, scores))
            search_results[name] = {
                "best_params": search.best_params_, "best_mean_cv_roc_auc": float(search.best_score_),
                "trials": [
                    {"params": params, "mean_cv_roc_auc": float(mean), "std_cv_roc_auc": float(std)}
                    for params, mean, std in zip(search.cv_results_["params"],
                        search.cv_results_["mean_test_score"], search.cv_results_["std_test_score"], strict=True)
                ],
            }
        selected = choose_model(results)
        winner = models[selected]
        importance = permutation_importance(winner, X_val, y_val, scoring="roc_auc",
                                            n_repeats=3, random_state=RANDOM_STATE, n_jobs=1)
    ranking = sorted(zip(X_val.columns, importance.importances_mean,
                         importance.importances_std, strict=True), key=lambda row: -row[1])
    summary = {
        "stage": 5, "seed": RANDOM_STATE, "train_clients": len(X_train),
        "validation_clients": len(X_val), "test_clients_reserved": int((assignment["split"] == "test").sum()),
        "test_evaluated": False, "split_sha256": _split_hash(assignment),
        "development_sha256": _development_hash(dataset, assignment),
        "input_features": X_train.columns.tolist(), "versions": _versions(),
        "baseline_configuration": baseline["configuration"],
        "search_configuration": {"folds": 3, "scoring": "roc_auc", "grids": SEARCH_GRIDS,
                                 "random_forest_n_estimators": 120, "boosting_max_iter": 120,
                                 "random_forest_max_features": "sqrt", "boosting_min_samples_leaf": 20,
                                 "boosting_early_stopping": False},
        "selection_rule": SELECTION_RULE, "threshold_rule": baseline["threshold_rule"],
        "validation_metrics": results, "cv_results": search_results,
        "selected_model": selected, "selected_threshold": results[selected]["threshold"],
        "validation_permutation_importance": [
            {"feature": str(name), "mean_auc_decrease": float(mean), "std": float(std)}
            for name, mean, std in ranking
        ],
    }
    return winner, summary, y_val, probabilities


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _plot_curves(y, probabilities: dict, directory: Path, prefix: str) -> None:
    from matplotlib import pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    for kind, xlabel, ylabel in (("roc", "False positive rate", "True positive rate"),
                                 ("pr", "Recall", "Precision"),
                                 ("calibration", "Mean predicted probability", "Observed default fraction")):
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, p in probabilities.items():
            if kind == "roc":
                x, z, _ = roc_curve(y, p)
            elif kind == "pr":
                z, x, _ = precision_recall_curve(y, p)
            else:
                z, x = calibration_curve(y, p, n_bins=10, strategy="uniform")
            ax.plot(x, z, marker="o" if kind == "calibration" else None, label=name)
        if kind == "pr":
            ax.axhline(float(np.mean(y)), linestyle="--", color="gray", label="prevalence")
        else:
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="reference")
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.02), title=f"{prefix}: {kind.upper()}")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / f"{prefix}_{kind}.png", dpi=150)
        plt.close(fig)


def _metric_table(results: dict) -> list[str]:
    lines = ["| Model | Threshold | ROC-AUC | AP | Precision | Recall | F1 | Brier | TN | FP | FN | TP |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        scores = " | ".join(f"{result[key]:.6f}" for key in ("threshold", "roc_auc", "average_precision",
                                                   "precision", "recall", "f1", "brier_score"))
        (tn, fp), (fn, tp) = result["confusion_matrix"]
        lines.append(f"| {name} | {scores} | {tn} | {fp} | {fn} | {tp} |")
    return lines


def _comparison_report(summary: dict, path: Path) -> None:
    lines = ["# Stage 5: model comparison (validation)", "",
             f"Train: {summary['train_clients']:,}; validation: {summary['validation_clients']:,}; "
             f"reserved test: {summary['test_clients_reserved']:,} (not scored).", "",
             "Preprocessing is fitted inside each training fold. Three stratified folds, seed 42; "
             "search maximizes mean CV ROC-AUC. Winning hyperparameters are refitted on train only.", "",
             "## Validation comparison", "", *_metric_table(summary["validation_metrics"]), "",
             f"Selection rule: {summary['selection_rule']}",
             f"Selected model: **{summary['selected_model']}**; threshold: **{summary['selected_threshold']:.8f}**.",
             "Thresholds maximize validation F1; test is not used for selection. Dummy is a reference only.", "",
             "## Train-only CV", ""]
    for name, result in summary["cv_results"].items():
        lines += [f"### {name}", "", f"Best parameters: `{json.dumps(result['best_params'])}`; "
                  f"mean CV ROC-AUC: {result['best_mean_cv_roc_auc']:.6f}.", "",
                  "| Parameters | Mean ROC-AUC | Fold SD |", "|---|---:|---:|"]
        for trial in result["trials"]:
            lines.append(f"| `{json.dumps(trial['params'])}` | {trial['mean_cv_roc_auc']:.6f} | {trial['std_cv_roc_auc']:.6f} |")
        lines.append("")
    selected = summary["selected_model"]
    delta = summary["validation_metrics"][selected]["roc_auc"] - summary["validation_metrics"]["logistic"]["roc_auc"]
    lines += [f"Selected minus logistic validation ROC-AUC: {delta:+.6f}. No improvement is assumed.", "",
              "## Validation permutation importance (top 10)", "", "| Feature | Mean AUC decrease | SD |", "|---|---:|---:|"]
    for item in summary["validation_permutation_importance"][:10]:
        lines.append(f"| {item['feature']} | {item['mean_auc_decrease']:.6f} | {item['std']:.6f} |")
    lines += ["", "Three permutations per raw input feature. Correlated features can share importance; "
              "this is not a causal analysis. Validation scores are selection-dependent, not final estimates.", "",
              "Calibration curves are descriptive; no probability calibration is fitted.",
              "The historical dataset and demographic variables limit real-world use. Not for lending decisions.", ""]
    for kind in ("roc", "pr", "calibration"):
        lines += [f"![Validation {kind}](figures/comparison_validation_{kind}.png)", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_comparison(dataset: FeatureDataset, assignment: pd.DataFrame, config: BaselineConfig,
                   *, reports_dir: Path, artifacts_dir: Path) -> dict:
    if (reports_dir / "final_test.json").exists():
        raise ValueError("Final test already evaluated. Do not reselect models using the same holdout.")
    winner, summary, y_val, probabilities = fit_comparison(dataset, assignment, config)
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _plot_curves(y_val, probabilities, reports_dir / "figures", "comparison_validation")
    _comparison_report(summary, reports_dir / "comparison.md")
    model_path = artifacts_dir / "selected.joblib"
    temporary = model_path.with_suffix(".joblib.tmp")
    joblib.dump(winner, temporary, compress=3)
    temporary.replace(model_path)
    summary["pipeline_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    _write_json(artifacts_dir / "metadata.json", summary)
    _write_json(reports_dir / "comparison.json", summary)
    return summary


def evaluate_final_test(dataset: FeatureDataset, assignment: pd.DataFrame, *,
                        reports_dir: Path, artifacts_dir: Path) -> dict:
    """Score only the frozen winner; no fitting or threshold selection on test."""
    final_path = reports_dir / "final_test.json"
    if final_path.exists():
        raise ValueError("Final test already evaluated. Read reports/final_test.json; do not rescore.")
    # Validate the frozen split and development data before accessing test features.
    training_subset(dataset, assignment)
    metadata = json.loads((artifacts_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata != json.loads((reports_dir / "comparison.json").read_text(encoding="utf-8")):
        raise ValueError("Selection metadata disagrees with the comparison report.")
    if (metadata["split_sha256"] != _split_hash(assignment)
            or metadata["development_sha256"] != _development_hash(dataset, assignment)
            or metadata["versions"] != _versions()):
        raise ValueError("Frozen split, development data or library versions changed since model selection.")
    model_path = artifacts_dir / "selected.joblib"
    if metadata["pipeline_sha256"] != hashlib.sha256(model_path.read_bytes()).hexdigest():
        raise ValueError("Selected pipeline SHA-256 mismatch.")
    name = metadata["selected_model"]
    if name != choose_model(metadata["validation_metrics"]):
        raise ValueError("Frozen selected model disagrees with the selection rule.")
    threshold = metadata["selected_threshold"]
    if threshold != metadata["validation_metrics"][name]["threshold"]:
        raise ValueError("Frozen threshold disagrees with validation results.")
    model = joblib.load(model_path)  # trusted, locally generated artifact only
    X_test, y_test = _partition(dataset, assignment, "test")
    _check_features(X_test)
    if X_test.columns.tolist() != metadata["input_features"]:
        raise ValueError("Test feature order differs from the frozen input schema.")
    with threadpool_limits(limits=1):
        probability = positive_probability(model, X_test)
    result = {"stage": 5, "selected_model": name, "test_evaluated": True,
              "test_clients": len(y_test), "test_default_rate": float(y_test.mean()),
              "split_sha256": metadata["split_sha256"], "pipeline_sha256": metadata["pipeline_sha256"],
              "metrics": binary_metrics(y_test, probability, threshold=threshold)}
    # Persist the evaluation marker before generating optional visualizations.
    # Exclusive creation also prevents silently replacing a completed result.
    with final_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    lines = ["# Final held-out test evaluation", "", f"Frozen model: **{name}**; test clients: {len(y_test):,}.",
             "No fitting, hyperparameter selection or threshold adjustment on test.", "",
             *_metric_table({name: result["metrics"]}), "",
             "Threshold was fixed on validation. This is a single independent holdout evaluation of the chosen pipeline.",
             "The pipeline remains trained on train only. Do not use these results to reselect or retune on this test set.",
             "Calibration is descriptive; historical data limit generalization. Not for real lending decisions.", ""]
    for kind in ("roc", "pr", "calibration"):
        lines += [f"![Test {kind}](figures/final_test_{kind}.png)", ""]
    (reports_dir / "final_test.md").write_text("\n".join(lines), encoding="utf-8")
    _plot_curves(y_test, {name: probability}, reports_dir / "figures", "final_test")
    return result
