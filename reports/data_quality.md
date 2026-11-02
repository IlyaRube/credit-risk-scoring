# Data Quality: UCI Default of Credit Card Clients

Generated from the validated local UCI snapshot. No values were changed.

- Rows: **30000**
- Columns: **25** (23 features + ID + target)
- Unique clients: **30000**
- Duplicate client IDs: **0**
- Target counts: `{"0": 23364, "1": 6636}`

## Missing values

| Column | Missing count | Missing share |
|---|---:|---:|
| client_id | 0 | 0.00% |
| LIMIT_BAL | 0 | 0.00% |
| SEX | 0 | 0.00% |
| EDUCATION | 0 | 0.00% |
| MARRIAGE | 0 | 0.00% |
| AGE | 0 | 0.00% |
| PAY_0 | 0 | 0.00% |
| PAY_2 | 0 | 0.00% |
| PAY_3 | 0 | 0.00% |
| PAY_4 | 0 | 0.00% |
| PAY_5 | 0 | 0.00% |
| PAY_6 | 0 | 0.00% |
| BILL_AMT1 | 0 | 0.00% |
| BILL_AMT2 | 0 | 0.00% |
| BILL_AMT3 | 0 | 0.00% |
| BILL_AMT4 | 0 | 0.00% |
| BILL_AMT5 | 0 | 0.00% |
| BILL_AMT6 | 0 | 0.00% |
| PAY_AMT1 | 0 | 0.00% |
| PAY_AMT2 | 0 | 0.00% |
| PAY_AMT3 | 0 | 0.00% |
| PAY_AMT4 | 0 | 0.00% |
| PAY_AMT5 | 0 | 0.00% |
| PAY_AMT6 | 0 | 0.00% |
| default_next_month | 0 | 0.00% |

## Undocumented categorical codes

- **EDUCATION**: {"0": 14, "5": 280, "6": 51}
- **MARRIAGE**: {"0": 54}
- **PAY_0**: {"-2": 2759, "0": 14737}
- **PAY_2**: {"-2": 3782, "0": 15730}
- **PAY_3**: {"-2": 4085, "0": 15764}
- **PAY_4**: {"-2": 4348, "0": 16455}
- **PAY_5**: {"-2": 4546, "0": 16947}
- **PAY_6**: {"-2": 4895, "0": 16286}

## Suspicious numeric values

- **age_outside_18_to_100**: `0`
- **credit_limit_nonpositive**: `0`
- **negative_payment_amounts**: `{"PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0}`
- **negative_bill_amounts**: `{"BILL_AMT1": 590, "BILL_AMT2": 669, "BILL_AMT3": 655, "BILL_AMT4": 675, "BILL_AMT5": 655, "BILL_AMT6": 688}`

## Warnings and treatment policy

- Undocumented categorical codes present; original values preserved.
- Negative bill amounts present; they may represent credit balances.

Observed values are not recoded, imputed, dropped or winsorized at this stage. Interpretation and treatment are deferred until training-only analysis.
