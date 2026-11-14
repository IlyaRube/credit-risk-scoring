"""Statistical procedures for stage 3, tested separately from the CLI and plotting."""

import math

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, norm


class StatisticalAnalysisError(ValueError):
    """Data cannot support the specified statistical procedure."""


def wilson_interval(successes: int, trials: int, *, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson binomial confidence interval (independent Bernoulli trials assumed)."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise StatisticalAnalysisError("Expected 0 <= successes <= trials and trials > 0.")
    if not 0 < confidence < 1:
        raise StatisticalAnalysisError("confidence must be between 0 and 1.")
    z = float(norm.ppf((1.0 + confidence) / 2.0))
    phat = successes / trials
    denominator = 1 + z * z / trials
    center = (phat + z * z / (2 * trials)) / denominator
    half = (
        z
        * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials**2))
        / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, center - half)
    upper = 1.0 if successes == trials else min(1.0, center + half)
    return lower, upper


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm family-wise-error-rate correction for a prespecified family of tests."""
    if not p_values or any(not np.isfinite(p) or p < 0 or p > 1 for p in p_values):
        raise StatisticalAnalysisError("p-values must be nonempty and between 0 and 1.")
    ordered_indices = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    previous = 0.0
    for rank, index in enumerate(ordered_indices):
        previous = max(previous, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = previous
    return adjusted


def chi_square_delinquency(X: pd.DataFrame, y: pd.Series) -> dict:
    """Chi-square independence test: any historical delay (yes/no) vs default (yes/no).

    The binary grouping is fixed in advance: delinquent_month_count > 0.
    No grouping is selected based on the target distribution.
    """
    if "delinquent_month_count" not in X.columns:
        raise StatisticalAnalysisError("Feature delinquent_month_count is required.")
    if len(X) != len(y):
        raise StatisticalAnalysisError("X and y lengths do not match.")
    delays = pd.to_numeric(X["delinquent_month_count"], errors="coerce")
    if delays.isna().any() or (delays < 0).any():
        raise StatisticalAnalysisError("Delinquency count must be nonmissing and nonnegative.")
    contingency = pd.crosstab(
        (delays.to_numpy() > 0).astype(int),
        y.to_numpy(),
    ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    if (contingency.to_numpy().sum(axis=0) == 0).any():
        raise StatisticalAnalysisError("Chi-square requires both target classes.")
    if (contingency.to_numpy().sum(axis=1) == 0).any():
        raise StatisticalAnalysisError("Chi-square requires both delinquency groups.")
    result = chi2_contingency(contingency, correction=False)
    min_expected = float(result.expected_freq.min())
    if min_expected < 5:
        raise StatisticalAnalysisError(
            f"Chi-square asymptotic approximation is unreliable: min expected={min_expected:.2f}."
        )
    n = int(contingency.to_numpy().sum())
    return {
        "name": "chi_square_any_delinquency_vs_default",
        "null": "Any historical delinquency and default are statistically independent.",
        "chi2": float(result.statistic),
        "degrees_of_freedom": int(result.dof),
        "p_value": float(result.pvalue),
        "phi_effect_size": math.sqrt(float(result.statistic) / n),
        "min_expected_frequency": min_expected,
        "contingency": {
            "no_delay": {
                "no_default": int(contingency.loc[0, 0]),
                "default": int(contingency.loc[0, 1]),
            },
            "has_delay": {
                "no_default": int(contingency.loc[1, 0]),
                "default": int(contingency.loc[1, 1]),
            },
        },
    }


def mann_whitney_credit_limit(X: pd.DataFrame, y: pd.Series) -> dict:
    """Two-sided Mann-Whitney U: credit limits differ by default label.

    The null is equality of the two distributions, not automatically equality
    of medians. Positive rank-biserial implies larger limits for nondefaulters.
    """
    if "limit_bal" not in X.columns:
        raise StatisticalAnalysisError("Feature limit_bal is required.")
    if len(X) != len(y):
        raise StatisticalAnalysisError("X and y lengths do not match.")
    values = pd.to_numeric(X["limit_bal"], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise StatisticalAnalysisError("Credit limits must be nonmissing and finite.")
    labels = y.to_numpy()
    group0 = values.loc[labels == 0].to_numpy(dtype=float)
    group1 = values.loc[labels == 1].to_numpy(dtype=float)
    if not len(group0) or not len(group1):
        raise StatisticalAnalysisError("Mann-Whitney requires both target classes.")
    result = mannwhitneyu(group0, group1, alternative="two-sided", method="asymptotic")
    rank_biserial = 2 * float(result.statistic) / (len(group0) * len(group1)) - 1
    return {
        "name": "mann_whitney_credit_limit_by_default",
        "null": "The credit-limit distributions are the same in both default groups.",
        "u_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "rank_biserial_effect_size": rank_biserial,
        "n_no_default": int(len(group0)),
        "n_default": int(len(group1)),
        "median_limit_no_default": float(np.median(group0)),
        "median_limit_default": float(np.median(group1)),
    }


def run_inference(X: pd.DataFrame, y: pd.Series) -> dict:
    """Two prespecified tests and Wilson CI, all using train only."""
    if not len(y) or y.isna().any() or not y.isin([0, 1]).all():
        raise StatisticalAnalysisError("Expected nonempty binary target without missing values.")
    successes = int((y == 1).sum())
    confidence = 0.95
    low, high = wilson_interval(successes, len(y), confidence=confidence)
    chi2 = chi_square_delinquency(X, y)
    mann_whitney = mann_whitney_credit_limit(X, y)
    adjusted = holm_adjust([chi2["p_value"], mann_whitney["p_value"]])
    chi2["holm_adjusted_p"] = adjusted[0]
    mann_whitney["holm_adjusted_p"] = adjusted[1]
    return {
        "scope": "train_only",
        "n_clients": int(len(y)),
        "default_count": successes,
        "default_rate": successes / len(y),
        "wilson_ci": {"confidence": confidence, "lower": low, "upper": high},
        "tests": [chi2, mann_whitney],
        "alpha": 0.05,
        "multiplicity": "Holm-Bonferroni correction across the two prespecified tests.",
    }
