import numpy as np
from tfms_aspergillus.statistics import compare_paired, compare_populations

def test_population_comparison_returns_one_pvalue():
    result = compare_populations([0.8, 0.9, 0.7], [0.6, 0.5, 0.7])
    assert result["test"] == "mannwhitneyu_two_sided"
    assert np.isfinite(result["p_value"])

def test_paired_comparison_uses_matching_positions():
    result = compare_paired([0.8, 0.9, 0.7], [0.6, 0.5, 0.7])
    assert result["test"] == "wilcoxon_signed_rank_two_sided"
    assert result["n_pairs"] == 3
