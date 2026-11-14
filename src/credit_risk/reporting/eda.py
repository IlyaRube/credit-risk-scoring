"""Train-only descriptive EDA, visualizations and Markdown/JSON reports.

This module never accesses validation/test rows and does not fit ML preprocessors.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from credit_risk.reporting.statistics import StatisticalAnalysisError, run_inference

NUMERIC_FEATURES = (
    "age",
    "limit_bal",
    "bill_amount_m1",
    "bill_amount_m2",
    "bill_amount_m3",
    "bill_amount_m4",
    "bill_amount_m5",
    "bill_amount_m6",
    "payment_amount_m1",
    "payment_amount_m2",
    "payment_amount_m3",
    "payment_amount_m4",
    "payment_amount_m5",
    "payment_amount_m6",
    "max_repayment_status",
    "delinquent_month_count",
    "avg_bill_amount",
    "avg_payment_amount",
    "total_payment_amount",
    "bill_amount_change_6m",
    "payment_to_abs_bill_ratio",
)
CORRELATION_COLUMNS = (
    "age",
    "limit_bal",
    "repayment_status_m6",
    "delinquent_month_count",
    "avg_bill_amount",
    "avg_payment_amount",
    "bill_amount_change_6m",
    "payment_to_abs_bill_ratio",
)


class EDAError(ValueError):
    """Insufficient or incompatible data for the documented EDA."""


def describe_training_data(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Calculate train-only class counts and numerical summaries without imputation."""
    if len(X_train) != len(y_train) or len(X_train) < 5:
        raise EDAError("X_train and y_train must be aligned with at least 5 observations.")
    if "client_id" in X_train.columns or "default_next_month" in X_train.columns:
        raise EDAError("X_train contains an identifier or target leakage column.")
    missing = sorted(set(NUMERIC_FEATURES) - set(X_train.columns))
    if missing:
        raise EDAError(f"EDA required numerical features not found: {missing}")
    if y_train.isna().any() or not y_train.isin([0, 1]).all():
        raise EDAError("Train target must be binary and complete.")

    summaries: dict[str, dict] = {}
    for feature in NUMERIC_FEATURES:
        numeric = pd.to_numeric(X_train[feature], errors="coerce")
        finite = numeric.replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty:
            raise EDAError(f"No finite observations for feature {feature}.")
        first_quartile = float(finite.quantile(0.25))
        third_quartile = float(finite.quantile(0.75))
        iqr = third_quartile - first_quartile
        summaries[feature] = {
            "count": int(len(finite)),
            "missing_or_nonfinite": int(len(numeric) - len(finite)),
            "mean": float(finite.mean()),
            "median": float(finite.median()),
            "q25": first_quartile,
            "q75": third_quartile,
            "variance_sample": float(finite.var(ddof=1)) if len(finite) > 1 else None,
            "std_sample": float(finite.std(ddof=1)) if len(finite) > 1 else None,
            "min": float(finite.min()),
            "max": float(finite.max()),
            "iqr_outlier_count": int(
                ((finite < first_quartile - 1.5 * iqr) | (finite > third_quartile + 1.5 * iqr))
                .sum()
            ),
        }

    counts = y_train.value_counts().to_dict()
    labels = y_train.to_numpy()
    limits = X_train["limit_bal"].to_numpy(dtype=float)
    grouped = {}
    for target in (0, 1):
        group_limits = limits[labels == target]
        if not len(group_limits):
            raise EDAError("Both target classes are needed for group summaries.")
        grouped[str(target)] = {
            "n": int(len(group_limits)),
            "median_credit_limit": float(np.median(group_limits)),
            "mean_credit_limit": float(np.mean(group_limits)),
        }

    by_delays = (
        pd.DataFrame(
            {
                "delinquent_month_count": X_train["delinquent_month_count"].to_numpy(),
                "default": labels,
            }
        )
        .groupby("delinquent_month_count", observed=True)["default"]
        .agg(["count", "mean"])
        .sort_index()
    )
    return {
        "scope": "train_only",
        "n_clients": int(len(y_train)),
        "n_features": int(X_train.shape[1]),
        "target_counts": {"0": int(counts.get(0, 0)), "1": int(counts.get(1, 0))},
        "target_rate": float((y_train == 1).mean()),
        "descriptive_statistics": summaries,
        "credit_limit_by_target": grouped,
        "default_rate_by_delinquent_month_count": [
            {
                "delinquent_month_count": int(month),
                "n_clients": int(row["count"]),
                "default_rate": float(row["mean"]),
            }
            for month, row in by_delays.iterrows()
        ],
        "outlier_rule": "IQR fences (Q1 - 1.5*IQR, Q3 + 1.5*IQR); no observations removed.",
    }


def _new_figure(size: tuple[float, float]) -> tuple[Figure, object]:
    fig = Figure(figsize=size)
    FigureCanvasAgg(fig)
    return fig, fig.subplots()


def _save_chart(fig: Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=155, bbox_inches="tight")


def render_eda_figures(
    X_train: pd.DataFrame, y_train: pd.Series, output_dir: Path
) -> list[str]:
    """Create train-only PNGs: histograms, boxplot, correlation and default rate."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    fig, ax = _new_figure((8, 4.5))
    ax.hist(X_train["age"].dropna(), bins=30)
    ax.set(title="Age distribution (train)", xlabel="Age", ylabel="Clients")
    _save_chart(fig, output_dir / "age_histogram.png")
    generated.append("age_histogram.png")

    fig, ax = _new_figure((8, 4.5))
    ax.hist(X_train["limit_bal"].dropna(), bins=40)
    ax.set(title="Credit limit distribution (train)", xlabel="Credit limit", ylabel="Clients")
    _save_chart(fig, output_dir / "credit_limit_histogram.png")
    generated.append("credit_limit_histogram.png")

    fig, ax = _new_figure((8, 4.5))
    labels = y_train.to_numpy()
    limits = X_train["limit_bal"].to_numpy(dtype=float)
    ax.boxplot([limits[labels == 0], limits[labels == 1]], tick_labels=["0", "1"])
    ax.set(title="Credit limit by default status (train)", xlabel="Default (0/1)")
    _save_chart(fig, output_dir / "credit_limit_by_default_boxplot.png")
    generated.append("credit_limit_by_default_boxplot.png")

    correlation = pd.DataFrame(
        {name: pd.to_numeric(X_train[name], errors="coerce") for name in CORRELATION_COLUMNS}
    )
    correlation["default_next_month"] = y_train.to_numpy()
    fig, ax = _new_figure((10.5, 8.5))
    sns.heatmap(
        correlation.corr(method="spearman"),
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        annot_kws={"size": 8},
    )
    ax.set_title("Spearman associations on train (not causal effects)")
    _save_chart(fig, output_dir / "spearman_correlation.png")
    generated.append("spearman_correlation.png")

    delay_rates = (
        pd.DataFrame(
            {
                "delinquent_month_count": X_train["delinquent_month_count"].to_numpy(),
                "default": y_train.to_numpy(),
            }
        )
        .groupby("delinquent_month_count", observed=True)["default"]
        .mean()
    )
    fig, ax = _new_figure((8, 4.5))
    ax.bar(delay_rates.index.astype(str), delay_rates.to_numpy())
    ax.set(
        title="Default rate by months with delinquency (train)",
        xlabel="Months with delinquency",
        ylabel="Share of defaults",
        ylim=(0, 1),
    )
    _save_chart(fig, output_dir / "default_rate_by_delinquency.png")
    generated.append("default_rate_by_delinquency.png")
    return generated


def _number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:,.3f}"


def _pvalue(value: float) -> str:
    return "< 1e-300 (numerical underflow)" if value == 0 else f"{value:.6g}"


def _eda_markdown(summary: dict, figures: list[str]) -> str:
    lines = [
        "# Exploratory Data Analysis: train only",
        "",
        "**Dataset:** UCI Default of Credit Card Clients (ID 350).",
        "**Scope:** train partition only. Validation and test are not inspected by this report.",
        "",
        f"- Train clients: **{summary['n_clients']:,}**",
        f"- Feature columns: **{summary['n_features']}**",
        f"- Non-default (0): **{summary['target_counts']['0']:,}**",
        f"- Default (1): **{summary['target_counts']['1']:,}**",
        f"- Default share: **{summary['target_rate']:.2%}**",
        "",
        "## Numeric descriptive statistics",
        "",
        "Sample variance and sample standard deviation use ddof=1. "
        "IQR outliers are flagged, not removed.",
        "",
        "| Feature | N finite | Missing/nonfinite | Mean | Median | Q25 | Q75 | Variance | Std. dev. | IQR outliers |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for feature, entry in summary["descriptive_statistics"].items():
        lines.append(
            f"| `{feature}` | {entry['count']} | {entry['missing_or_nonfinite']} | "
            f"{_number(entry['mean'])} | {_number(entry['median'])} | "
            f"{_number(entry['q25'])} | {_number(entry['q75'])} | "
            f"{_number(entry['variance_sample'])} | {_number(entry['std_sample'])} | "
            f"{entry['iqr_outlier_count']} |"
        )
    lines.extend(
        [
            "",
            "## Credit limit by target",
            "",
            "| Default | Clients | Mean credit limit | Median credit limit |",
            "|---:|---:|---:|---:|",
        ]
    )
    for target, group in summary["credit_limit_by_target"].items():
        lines.append(
            f"| {target} | {group['n']} | {_number(group['mean_credit_limit'])} | "
            f"{_number(group['median_credit_limit'])} |"
        )
    lines.extend(
        [
            "",
            "## Default share by delinquent months",
            "",
            "| Months | Clients | Default share |",
            "|---:|---:|---:|",
        ]
    )
    for item in summary["default_rate_by_delinquent_month_count"]:
        lines.append(
            f"| {item['delinquent_month_count']} | {item['n_clients']} | "
            f"{item['default_rate']:.2%} |"
        )
    lines.extend(["", "## Figures", ""])
    for figure in figures:
        lines.append(f"![{figure.replace('_', ' ')}](figures/{figure})")
        lines.append("")
    lines.extend(
        [
            "## Interpretation and limitations",
            "",
            "- These are descriptive associations, not causal estimates or model performance metrics.",
            "- Large differences in means versus medians suggest potential asymmetry; inspect plots.",
            "- Boxplot outliers and unusual payment/bill codes are not automatically discarded.",
            "- Coded fields (education, sex, marriage, raw repayment statuses) are not treated as continuous descriptive measurements.",
            "- Spearman correlations are monotonic associations and do not imply causation.",
            "- These historical Taiwanese credit-card data may not represent modern banking populations.",
            "- No validation or test metrics, hypotheses or charts are produced here.",
            "",
        ]
    )
    return "\n".join(lines)


def _statistics_markdown(inference: dict) -> str:
    chi2, mw = inference["tests"]
    ci = inference["wilson_ci"]
    alpha = inference["alpha"]
    chi2_significant = chi2["holm_adjusted_p"] < alpha
    mw_significant = mw["holm_adjusted_p"] < alpha
    lines = [
        "# Statistical inference: train only",
        "",
        f"**Sample:** {inference['n_clients']:,} independent client rows from the saved train split.",
        "**Pre-specified family:** two two-sided/independence tests; Holm correction at alpha = 0.05.",
        "",
        "## Binomial default proportion: 95% Wilson confidence interval",
        "",
        f"- Observed defaults: {inference['default_count']:,}/{inference['n_clients']:,}.",
        f"- Sample rate: **{inference['default_rate']:.2%}**.",
        f"- 95% Wilson interval: **[{ci['lower']:.2%}, {ci['upper']:.2%}]**.",
        "- Interpretation: in hypothetical repeated independent samples, the 95% Wilson method covers the population proportion about 95% of the time under its assumptions.",
        "- This interval does not mean there is a 95% probability that the fixed population proportion lies in this particular computed interval.",
        "- Generalizability requires representative sampling; historical convenience datasets may violate that assumption.",
        "",
        "## H1: Chi-square independence (any delinquency vs default)",
        "",
        "- H0: presence of at least one historically delinquent month is independent of the default label.",
        "- H1: there is an association between the two binary variables.",
        f"- Chi-square = **{chi2['chi2']:.4f}**, df = {chi2['degrees_of_freedom']}, "
        f"p = **{_pvalue(chi2['p_value'])}**, Holm-adjusted p = **{_pvalue(chi2['holm_adjusted_p'])}**.",
        f"- Phi coefficient (association magnitude) = **{chi2['phi_effect_size']:.4f}**.",
        f"- Minimum expected cell count = {chi2['min_expected_frequency']:.1f} (>= 5 required here).",
        "- Conclusion: "
        + ("reject independence at the corrected threshold." if chi2_significant else "do not reject independence at the corrected threshold."),
        "",
        "## H2: Mann-Whitney U (credit limit by default status)",
        "",
        "- H0: the credit-limit distributions for default=0 and default=1 are equal.",
        "- H1: the two distributions differ.",
        f"- U = **{mw['u_statistic']:.1f}**, "
        f"p = **{_pvalue(mw['p_value'])}**, Holm-adjusted p = **{_pvalue(mw['holm_adjusted_p'])}**.",
        f"- Median non-default = {_number(mw['median_limit_no_default'])}; "
        f"median default = {_number(mw['median_limit_default'])}.",
        f"- Rank-biserial effect size = **{mw['rank_biserial_effect_size']:.4f}** "
        "(positive means higher ranks for the non-default group).",
        "- Conclusion: "
        + ("reject equality of distributions at the corrected threshold." if mw_significant else "do not reject equality of distributions at the corrected threshold."),
        "",
        "## Assumptions and interpretation",
        "",
        "- Each client contributes one independent observation. Correlated clients would weaken these inferences.",
        "- Chi-square relies on sufficiently large expected frequencies; checked above.",
        "- Mann-Whitney is a rank-based comparison. It is NOT automatically a test of equal medians unless additional shape assumptions hold; ties are handled by SciPy's asymptotic method.",
        "- The two tests were fixed before looking at target associations, and Holm controls family-wise error for these two comparisons.",
        "- p-value is not the probability the null hypothesis is true, nor a measure of effect size.",
        "- Statistical significance does not establish predictive utility or causality.",
        "- No validation or test observations were included in statistical estimation.",
        "",
    ]
    return "\n".join(lines)


def run_eda(
    X_train: pd.DataFrame, y_train: pd.Series, reports_dir: Path
) -> tuple[Path, Path, list[str]]:
    """Build numeric/JSON reports and figures; return generated locations."""
    summary = describe_training_data(X_train, y_train)
    try:
        inference = run_inference(X_train, y_train)
    except StatisticalAnalysisError as exc:
        raise EDAError(f"Statistical inference failed: {exc}") from exc
    figures = render_eda_figures(X_train, y_train, reports_dir / "figures")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "eda_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "statistics.json").write_text(
        json.dumps(inference, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    eda_path = reports_dir / "eda.md"
    stats_path = reports_dir / "statistics.md"
    eda_path.write_text(_eda_markdown(summary, figures), encoding="utf-8")
    stats_path.write_text(_statistics_markdown(inference), encoding="utf-8")
    return eda_path, stats_path, figures
