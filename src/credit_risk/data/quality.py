"""Non-destructive data quality profiling. Findings never mutate the dataset."""

import json
from pathlib import Path

import pandas as pd

from credit_risk.data.schema import EXPECTED_ROWS, FEATURE_COLUMNS, validate_and_normalize

CATEGORICAL_EXPECTED = {
    "SEX": {1, 2},
    "EDUCATION": {1, 2, 3, 4},
    "MARRIAGE": {1, 2, 3},
}
PAY_STATUS_COLUMNS = ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")
CATEGORICAL_EXPECTED.update({name: {-1, *range(1, 10)} for name in PAY_STATUS_COLUMNS})


def build_quality_report(data: pd.DataFrame, *, expected_rows: int = EXPECTED_ROWS) -> dict:
    """Produce an auditable report without dropping or correcting suspicious rows."""
    # Accept only the canonical schema, including the exact original number of rows.
    if tuple(data.columns) != ("client_id", *FEATURE_COLUMNS, "default_next_month"):
        raise ValueError("Quality report expects canonical 25-column dataset.")
    original_shape = data.rename(
        columns={"client_id": "ID", "default_next_month": "default payment next month"}
    )
    validate_and_normalize(original_shape, expected_rows=expected_rows)

    total = len(data)
    missing = {
        name: {"count": int(data[name].isna().sum()), "share": float(data[name].isna().mean())}
        for name in data.columns
    }
    unusual_codes: dict[str, dict[str, int]] = {}
    for name, permitted in CATEGORICAL_EXPECTED.items():
        unexpected = data.loc[data[name].notna() & ~data[name].isin(permitted), name]
        if not unexpected.empty:
            counts = unexpected.value_counts().sort_index()
            unusual_codes[name] = {str(int(value)): int(count) for value, count in counts.items()}

    warnings = []
    if any(field["count"] for field in missing.values()):
        warnings.append("Missing values present; no imputation was performed.")
    if unusual_codes:
        warnings.append("Undocumented categorical codes present; original values preserved.")

    ages = data["AGE"]
    age_suspicious = int(((ages < 18) | (ages > 100)).sum())
    limit_nonpositive = int((data["LIMIT_BAL"] <= 0).sum())
    negative_payments = {
        name: int((data[name] < 0).sum()) for name in (f"PAY_AMT{i}" for i in range(1, 7))
    }
    negative_bills = {
        name: int((data[name] < 0).sum()) for name in (f"BILL_AMT{i}" for i in range(1, 7))
    }
    if age_suspicious or limit_nonpositive or any(negative_payments.values()):
        warnings.append("Suspicious age, limit or payment amounts present; no records removed.")
    if any(negative_bills.values()):
        warnings.append("Negative bill amounts present; they may represent credit balances.")

    return {
        "row_count": total,
        "column_count": len(data.columns),
        "feature_count": len(FEATURE_COLUMNS),
        "unique_client_ids": int(data["client_id"].nunique()),
        "duplicate_client_ids": int(data["client_id"].duplicated().sum()),
        "target_counts": {
            str(int(label)): int(count)
            for label, count in data["default_next_month"].value_counts().sort_index().items()
        },
        "missing_values": missing,
        "undocumented_category_codes": unusual_codes,
        "suspicious_values": {
            "age_outside_18_to_100": age_suspicious,
            "credit_limit_nonpositive": limit_nonpositive,
            "negative_payment_amounts": negative_payments,
            "negative_bill_amounts": negative_bills,
        },
        "warnings": warnings,
        "policy": (
            "Observed values are not recoded, imputed, dropped or winsorized at this stage. "
            "Interpretation and treatment are deferred until training-only analysis."
        ),
    }


def save_quality_report(report: dict, output_dir: Path) -> tuple[Path, Path]:
    """Save structured JSON and human-readable Markdown versions of the same report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "data_quality.json"
    markdown_path = output_dir / "data_quality.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Data Quality: UCI Default of Credit Card Clients",
        "",
        "Generated from the validated local UCI snapshot. No values were changed.",
        "",
        f"- Rows: **{report['row_count']}**",
        f"- Columns: **{report['column_count']}** (23 features + ID + target)",
        f"- Unique clients: **{report['unique_client_ids']}**",
        f"- Duplicate client IDs: **{report['duplicate_client_ids']}**",
        f"- Target counts: `{json.dumps(report['target_counts'], sort_keys=True)}`",
        "",
        "## Missing values",
        "",
        "| Column | Missing count | Missing share |",
        "|---|---:|---:|",
    ]
    for name, item in report["missing_values"].items():
        lines.append(f"| {name} | {item['count']} | {item['share']:.2%} |")

    lines.extend(["", "## Undocumented categorical codes", ""])
    if report["undocumented_category_codes"]:
        for name, codes in report["undocumented_category_codes"].items():
            lines.append(f"- **{name}**: {json.dumps(codes, sort_keys=True)}")
    else:
        lines.append("No undocumented category codes detected.")

    lines.extend(["", "## Suspicious numeric values", ""])
    for name, value in report["suspicious_values"].items():
        lines.append(f"- **{name}**: `{json.dumps(value, sort_keys=True)}`")
    lines.extend(["", "## Warnings and treatment policy", ""])
    lines.extend(f"- {warning}" for warning in report["warnings"])
    lines.extend(["", report["policy"], ""])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path
