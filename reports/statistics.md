# Statistical inference: train only

**Sample:** 18,000 independent client rows from the saved train split.
**Pre-specified family:** two two-sided/independence tests; Holm correction at alpha = 0.05.

## Binomial default proportion: 95% Wilson confidence interval

- Observed defaults: 3,982/18,000.
- Sample rate: **22.12%**.
- 95% Wilson interval: **[21.52%, 22.73%]**.
- Interpretation: in hypothetical repeated independent samples, the 95% Wilson method covers the population proportion about 95% of the time under its assumptions.
- This interval does not mean there is a 95% probability that the fixed population proportion lies in this particular computed interval.
- Generalizability requires representative sampling; historical convenience datasets may violate that assumption.

## H1: Chi-square independence (any delinquency vs default)

- H0: presence of at least one historically delinquent month is independent of the default label.
- H1: there is an association between the two binary variables.
- Chi-square = **2329.0999**, df = 1, p = **< 1e-300 (numerical underflow)**, Holm-adjusted p = **< 1e-300 (numerical underflow)**.
- Phi coefficient (association magnitude) = **0.3597**.
- Minimum expected cell count = 1349.0 (>= 5 required here).
- Conclusion: reject independence at the corrected threshold.

## H2: Mann-Whitney U (credit limit by default status)

- H0: the credit-limit distributions for default=0 and default=1 are equal.
- H1: the two distributions differ.
- U = **34665722.0**, p = **7.5477e-121**, Holm-adjusted p = **7.5477e-121**.
- Median non-default = 150,000.000; median default = 90,000.000.
- Rank-biserial effect size = **0.2421** (positive means higher ranks for the non-default group).
- Conclusion: reject equality of distributions at the corrected threshold.

## Assumptions and interpretation

- Each client contributes one independent observation. Correlated clients would weaken these inferences.
- Chi-square relies on sufficiently large expected frequencies; checked above.
- Mann-Whitney is a rank-based comparison. It is NOT automatically a test of equal medians unless additional shape assumptions hold; ties are handled by SciPy's asymptotic method.
- The two tests were fixed before looking at target associations, and Holm controls family-wise error for these two comparisons.
- p-value is not the probability the null hypothesis is true, nor a measure of effect size.
- Statistical significance does not establish predictive utility or causality.
- No validation or test observations were included in statistical estimation.
