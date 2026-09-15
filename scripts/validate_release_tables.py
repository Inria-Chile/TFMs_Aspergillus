#!/usr/bin/env python3
"""Validate the coverage contracts of the public consolidated tables."""

from pathlib import Path

import pandas as pd

from tfms_aspergillus.results import load_importance_table, load_metrics_table


def main() -> None:
    root = Path("data/final/v3_100_replicas")
    metrics = load_metrics_table(root / "metrics/seed_metrics_C_environment_microbiome.csv")
    shap = load_importance_table(root / "shap/shap_mean_abs_C_environment_microbiome_100seeds.csv")
    pysr = load_importance_table(
        root / "pysr/pysr_mean_relative_frequency_C_environment_microbiome_100seeds.csv"
    )
    coverage = pd.read_csv(root / "manifests/coverage_manifest.csv")

    assert shap.groupby(["family", "task", "model"]).ngroups == 36
    assert pysr.groupby(["family", "task", "model"]).ngroups == 9
    assert set(shap["scenario"]) == {"C_environment_microbiome"}
    assert set(pysr["scenario"]) == {"C_environment_microbiome"}
    assert (shap["importance"] >= 0).all() and (pysr["importance"] >= 0).all()
    assert coverage["status"].value_counts().to_dict() == {
        "complete_100_seeds": 147,
        "partial_metrics_60_seeds": 3,
    }
    print(f"Validated rows: metrics={len(metrics)}, SHAP={len(shap)}, PySR={len(pysr)}")


if __name__ == "__main__":
    main()
