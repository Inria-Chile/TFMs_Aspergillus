"""Population-level comparisons used by the manuscript figures.

The historical V3 RF-reference figures compare the population of seed-level
metrics for each model against RF within one experimental group. They report
one p-value per comparison, not one p-value per seed. The primary
implementation preserves that convention with a two-sided Mann-Whitney U test.
A paired Wilcoxon helper is also provided for analyses where identical seed IDs
are intentionally paired.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def compare_populations(reference: Iterable[float], target: Iterable[float]) -> dict[str, float | int | str]:
    """Compare two seed-level metric populations with one two-sided p-value."""
    ref = _finite(reference)
    other = _finite(target)
    if not len(ref) or not len(other):
        return {"statistic": float("nan"), "p_value": float("nan"), "n_reference": int(len(ref)), "n_target": int(len(other)), "test": "mannwhitneyu_two_sided"}
    try:
        result = mannwhitneyu(ref, other, alternative="two-sided", method="auto")
    except TypeError:  # SciPy < 1.7
        result = mannwhitneyu(ref, other, alternative="two-sided")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue), "n_reference": int(len(ref)), "n_target": int(len(other)), "test": "mannwhitneyu_two_sided"}


def compare_paired(reference: Iterable[float], target: Iterable[float]) -> dict[str, float | int | str]:
    """Compare equal-length seed vectors with a two-sided Wilcoxon test."""
    ref = _finite(reference)
    other = _finite(target)
    n = min(len(ref), len(other))
    if not n:
        return {"statistic": float("nan"), "p_value": float("nan"), "n_pairs": 0, "test": "wilcoxon_signed_rank_two_sided"}
    try:
        result = wilcoxon(ref[:n], other[:n], alternative="two-sided", method="auto")
    except TypeError:  # SciPy < 1.7
        result = wilcoxon(ref[:n], other[:n], alternative="two-sided")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue), "n_pairs": int(n), "test": "wilcoxon_signed_rank_two_sided"}
