#!/usr/bin/env python3
"""Build clean v3 seed-level outputs for family 1.

The v2 project contains valid historical family-1 outputs, but they live across
several plotting/audit directories. This script copies only seed-level
artifacts into the v3 layout and avoids fold-level CSVs in the destination.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


V2 = Path("/home/lvalenzuela/group_storage_nancy/lvalenzuela_SR_results/0_0000000000_a_Marta_Inria_v2/marta_omar_repro")
V3 = Path("/home/lvalenzuela/group_storage_nancy/lvalenzuela_SR_results/0_0000000000_a_Marta_Inria_v3")
FAMILY = "family1_no_clr_microbiome_raw"
FAMILY_LABEL = "Familia 1: microbioma sin CLR"

TASK_DIR = {
    "classification": "classification",
    "positive_abundance": "regression_18_positive_samples",
    "all_sample_abundance": "regression_50_all_samples",
}
TASK_LABEL = {
    "classification": "Clasificacion",
    "positive_abundance": "Regresion 18 positivas",
    "all_sample_abundance": "Regresion 50 muestras",
}
MODEL_SLUG = {
    "PySR": "pysr",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "TFN/TabPFN": "tabpfn",
    "TabICLv2": "tabiclv2",
}
SCENARIOS = ["A_environment", "B_microbiome", "C_environment_microbiome"]
MODEL_ORDER = ["PySR", "Random Forest", "XGBoost", "TFN/TabPFN", "TabICLv2"]


def norm_model(value: str) -> str:
    low = str(value).lower()
    if "random" in low and "forest" in low:
        return "Random Forest"
    if "xgboost" in low or re.search(r"\bxgb\b", low):
        return "XGBoost"
    if "tabicl" in low:
        return "TabICLv2"
    if "tabpfn" in low or "tfn" in low:
        return "TFN/TabPFN"
    if "pysr" in low:
        return "PySR"
    return str(value)


def norm_task(value: str) -> str:
    low = str(value).lower()
    if "occurrence" in low or "classification" in low:
        return "classification"
    if "positive" in low or low == "abundance":
        return "positive_abundance"
    if "all_sample" in low or "all-sample" in low:
        return "all_sample_abundance"
    return str(value)


def scenario_from_predictor_set(value: str) -> str:
    mapping = {
        "Environment": "A_environment",
        "Microbiome": "B_microbiome",
        "Environment+Microbiome": "C_environment_microbiome",
    }
    return mapping[str(value)]


def scenario_label(scenario: str) -> str:
    return {
        "A_environment": "A",
        "B_microbiome": "B",
        "C_environment_microbiome": "C",
    }.get(scenario, scenario)


def seed_metrics_sources() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    # Non-PySR family-1 seed metrics used by the final three-family replots.
    p51 = V2 / "outputs/51_three_family_complete_replots/three_family_seed_average_metrics_long.csv"
    df51 = pd.read_csv(p51)
    df51 = df51[df51["family"].eq(FAMILY)].copy()
    df51 = df51[~df51["model"].eq("PySR")].copy()
    df51["source_table"] = str(p51)
    rows.append(df51[["family", "model", "task", "scenario", "seed", "metric", "value", "n_folds_tested", "source_table"]])

    # PySR classification and positive-abundance metrics from the v2 replica
    # average table. Variant anova is the original family-1 no-CLR setting.
    p35 = V2 / "outputs/35_v2_replica_average_performance/v2_replica_average_metrics_long.csv"
    df35 = pd.read_csv(p35)
    df35 = df35[
        df35["variant"].astype(str).str.lower().eq("anova")
        & df35["model_family"].astype(str).str.lower().str.contains("pysr")
    ].copy()
    df35["family"] = FAMILY
    df35["model"] = "PySR"
    df35["task"] = df35["task"].map(norm_task)
    df35["source_table"] = str(p35)
    rows.append(df35[["family", "model", "task", "scenario", "seed", "metric", "value", "n_folds_tested", "source_table"]])

    # PySR all-sample seed-average metrics available in the all-sample plotting
    # table. Completion is tracked separately from audit40.
    p39 = V2 / "outputs/39_all_samples_regression_available_boxplots/all_samples_regression_available_seed_average_metrics.csv"
    df39 = pd.read_csv(p39)
    df39 = df39[df39["model"].astype(str).str.lower().eq("pysr")].copy()
    metric_cols = [c for c in ["RMSE", "MAE", "R2", "prediction_mean"] if c in df39.columns]
    long39 = df39.melt(
        id_vars=["model", "predictor_set", "seed", "n_folds_completed"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )
    long39["family"] = FAMILY
    long39["model"] = "PySR"
    long39["task"] = "all_sample_abundance"
    long39["scenario"] = long39["predictor_set"].map(scenario_from_predictor_set)
    long39["n_folds_tested"] = long39["n_folds_completed"]
    long39["source_table"] = str(p39)
    rows.append(long39[["family", "model", "task", "scenario", "seed", "metric", "value", "n_folds_tested", "source_table"]])

    out = pd.concat(rows, ignore_index=True)
    out["seed"] = out["seed"].astype(int)
    return out


def completion_table() -> pd.DataFrame:
    p40 = V2 / "outputs/40_v2_replicate_audit/v2_abc_seed_level_replicas.csv"
    df = pd.read_csv(p40)
    df["family"] = FAMILY
    df["model"] = df["model"].map(norm_model)
    df["task_label_clean"] = df["task"].map(TASK_LABEL)
    df["source_table"] = str(p40)
    return df


def shap_summary() -> pd.DataFrame:
    p51 = V2 / "outputs/51_three_family_complete_replots/three_family_shap_feature_importance_summary.csv"
    df = pd.read_csv(p51)
    df = df[df["family"].eq(FAMILY)].copy()
    df["source_table"] = str(p51)
    return df


def pysr_variable_usage() -> pd.DataFrame:
    p51 = V2 / "outputs/51_three_family_complete_replots/three_family_pysr_variable_usage_summary.csv"
    if not p51.exists():
        return pd.DataFrame()
    df = pd.read_csv(p51)
    if "family" in df.columns:
        df = df[df["family"].eq(FAMILY)].copy()
    # Do not rename features to X0/X1/etc. The source table already contains
    # standardized original predictor names when available.
    df["source_table"] = str(p51)
    return df


def write_seed_metric_files(seed_long: pd.DataFrame, completion: pd.DataFrame) -> pd.DataFrame:
    root = V3 / "outputs" / FAMILY
    manifest_rows: list[dict[str, object]] = []
    group_cols = ["task", "scenario", "model", "seed"]
    wide = (
        seed_long.pivot_table(
            index=["family", "task", "scenario", "model", "seed", "n_folds_tested", "source_table"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None
    wide["task_label"] = wide["task"].map(TASK_LABEL)
    wide["scenario_label"] = wide["scenario"].map(scenario_label)
    comp_keys = completion[["task", "scenario", "model", "seed", "completed_folds", "expected_folds", "seed_complete_all_folds"]].copy()
    comp_keys["seed"] = comp_keys["seed"].astype(int)
    wide = wide.merge(comp_keys, on=["task", "scenario", "model", "seed"], how="left")
    wide["seed_complete_all_folds"] = wide["seed_complete_all_folds"].fillna(False).astype(bool)
    wide["family_label"] = FAMILY_LABEL
    metric_first = ["family", "family_label", "task", "task_label", "scenario", "scenario_label", "model", "seed"]
    other_cols = [c for c in wide.columns if c not in metric_first]
    wide = wide[metric_first + other_cols]
    for (task, scenario, model, seed), g in wide.groupby(group_cols, sort=True):
        task_dir = TASK_DIR[task]
        model_dir = MODEL_SLUG.get(model, re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").lower())
        out_dir = root / task_dir / scenario / model_dir / "seed_metrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"seed_{int(seed):03d}_metrics.csv"
        g.to_csv(out_csv, index=False)
        manifest_rows.append(
            {
                "family": FAMILY,
                "task": task,
                "scenario": scenario,
                "model": model,
                "seed": int(seed),
                "seed_metrics_csv": str(out_csv),
                "rows": len(g),
                "complete_seed": bool(g["seed_complete_all_folds"].iloc[0]) if "seed_complete_all_folds" in g else False,
            }
        )
    return pd.DataFrame(manifest_rows), wide


def write_shap_files(shap: pd.DataFrame, pysr_usage: pd.DataFrame) -> None:
    root = V3 / "outputs" / FAMILY
    if not shap.empty:
        for (task, scenario, model), g in shap.groupby(["task", "scenario", "model"], sort=True):
            task_dir = TASK_DIR.get(task, task)
            model_dir = MODEL_SLUG.get(model, re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").lower())
            out_dir = root / task_dir / scenario / model_dir / "shap_importance"
            out_dir.mkdir(parents=True, exist_ok=True)
            g.sort_values("mean_abs_shap", ascending=False).to_csv(out_dir / "mean_abs_shap_by_feature.csv", index=False)
    if not pysr_usage.empty:
        out_dir = root / "pysr_variable_usage"
        out_dir.mkdir(parents=True, exist_ok=True)
        pysr_usage.to_csv(out_dir / "pysr_variable_usage_original_feature_names.csv", index=False)


def main() -> None:
    V3.mkdir(parents=True, exist_ok=True)
    (V3 / "outputs" / FAMILY / "tables").mkdir(parents=True, exist_ok=True)
    (V3 / "manifests").mkdir(parents=True, exist_ok=True)
    seed_long = seed_metrics_sources()
    completion = completion_table()
    shap = shap_summary()
    pysr_usage = pysr_variable_usage()
    seed_manifest, seed_wide = write_seed_metric_files(seed_long, completion)
    write_shap_files(shap, pysr_usage)

    tables = V3 / "outputs" / FAMILY / "tables"
    seed_long.to_csv(tables / "family1_seed_metrics_long.csv", index=False)
    seed_wide.to_csv(tables / "family1_seed_metrics_wide.csv", index=False)
    seed_manifest.to_csv(tables / "family1_seed_metrics_file_manifest.csv", index=False)
    completion.to_csv(tables / "family1_completion_seed_level.csv", index=False)
    shap.to_csv(tables / "family1_shap_mean_abs_by_feature.csv", index=False)
    if not pysr_usage.empty:
        pysr_usage.to_csv(tables / "family1_pysr_variable_usage_original_feature_names.csv", index=False)

    summary = {
        "family": FAMILY,
        "seed_metric_files": int(len(seed_manifest)),
        "seed_metric_rows_long": int(len(seed_long)),
        "seed_metric_rows_wide": int(len(seed_wide)),
        "completion_rows": int(len(completion)),
        "shap_rows": int(len(shap)),
        "pysr_variable_usage_rows": int(len(pysr_usage)),
        "source_policy": "seed-level metrics and mean SHAP only; no fold-level CSVs copied into v3 outputs",
    }
    (V3 / "manifests" / "family1_v3_seed_level_copy_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
