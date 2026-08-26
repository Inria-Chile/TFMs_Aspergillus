from pathlib import Path

import pandas as pd

from tfms_aspergillus.results import load_importance_table, load_metrics_table


ROOT = Path(__file__).parents[1] / "data" / "final" / "v3_100_replicas"


def test_final_metrics_table_contract():
    frame = load_metrics_table(ROOT / "metrics" / "seed_metrics_long_for_boxplots.csv")
    assert {"family", "task", "scenario", "model", "seed", "metric", "value"}.issubset(frame.columns)
    assert frame["seed"].nunique() >= 100


def test_final_shap_table_contract():
    frame = load_importance_table(ROOT / "shap" / "shap_mean_abs_by_feature.csv")
    assert {"family", "task", "scenario", "model", "variable", "importance"}.issubset(frame.columns)
    assert frame["importance"].notna().all()
