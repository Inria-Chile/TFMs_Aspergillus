#!/usr/bin/env python3
"""Build clean v3 seed-level outputs for family 2.

Family 2 is the full microbiome CLR + raw environment strategy
(`strategy48_clr_full_microbiome_raw_env`). Outputs are copied only at seed
level into v3; fold-level CSVs remain in v2 provenance tables.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


V2 = Path("/home/lvalenzuela/group_storage_nancy/lvalenzuela_SR_results/0_0000000000_a_Marta_Inria_v2/marta_omar_repro")
V3 = Path("/home/lvalenzuela/group_storage_nancy/lvalenzuela_SR_results/0_0000000000_a_Marta_Inria_v3")
FAMILY = "family2_full_microbiome_clr_raw_env"
FAMILY_LABEL = "Familia 2: microbioma completo CLR + ambiente crudo"

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


def scenario_label(scenario: str) -> str:
    return {
        "A_environment": "A",
        "B_microbiome": "B",
        "C_environment_microbiome": "C",
    }.get(scenario, scenario)


def safe_model_dir(model: str) -> str:
    return MODEL_SLUG.get(model, re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").lower())


def seed_metrics_sources() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    p51 = V2 / "outputs/51_three_family_complete_replots/three_family_seed_average_metrics_long.csv"
    df51 = pd.read_csv(p51)
    df51 = df51[df51["family"].eq(FAMILY)].copy()
    df51["source_table"] = str(p51)
    df51["source_priority"] = 0
    frames.append(df51[["family", "model", "task", "scenario", "seed", "metric", "value", "n_folds_tested", "source_table", "source_priority"]])

    p56 = V2 / "outputs/56_incremental_seed_fold_aggregation_20260628_0856/tables/updated_seed_average_metrics_long.csv"
    df56 = pd.read_csv(p56)
    df56 = df56[df56["family"].eq(FAMILY)].copy()
    df56["source_table"] = str(p56)
    df56["source_priority"] = 1
    frames.append(df56[["family", "model", "task", "scenario", "seed", "metric", "value", "n_folds_tested", "source_table", "source_priority"]])

    out = pd.concat(frames, ignore_index=True, sort=False)
    out["seed"] = out["seed"].astype(int)
    out = out.sort_values(["family", "task", "scenario", "model", "seed", "metric", "source_priority"])
    out = out.drop_duplicates(["family", "task", "scenario", "model", "seed", "metric"], keep="first")
    return out.drop(columns=["source_priority"])


def completion_table() -> pd.DataFrame:
    """Load family-2 best-available aggregate completion plus seed-level manifests."""
    p58 = V2 / "outputs/58_family2_deep_search_20260628_2000/tables/family2_combined_best_available_completion_strategy48_plus_56.csv"
    aggregate = pd.read_csv(p58)
    aggregate["source_table"] = str(p58)

    seed_sources: list[pd.DataFrame] = []
    for p in [
        V2 / "outputs/56_incremental_seed_fold_aggregation_20260628_0856/tables/updated_seed_file_manifest.csv",
        V2 / "outputs/55_seed_fold_aggregation_20260626_2056/tables/seed_file_manifest.csv",
    ]:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df = df[df["family"].eq(FAMILY)].copy()
        if df.empty:
            continue
        df["source_table"] = str(p)
        seed_sources.append(df)
    seed_level = pd.concat(seed_sources, ignore_index=True, sort=False) if seed_sources else pd.DataFrame()
    if not seed_level.empty:
        seed_level = seed_level.sort_values(["complete_seed", "n_unique_folds"], ascending=False)
        seed_level = seed_level.drop_duplicates(["family", "task", "scenario", "model", "seed"], keep="first")

    return aggregate, seed_level


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
    if df.empty:
        return df
    df["source_table"] = str(p51)
    return df


def write_seed_metric_files(seed_long: pd.DataFrame, seed_level_completion: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = V3 / "outputs" / FAMILY
    manifest_rows: list[dict[str, object]] = []
    seed_meta = (
        seed_long.groupby(["family", "task", "scenario", "model", "seed"], as_index=False)
        .agg(
            n_folds_tested=("n_folds_tested", "max"),
            source_tables=("source_table", lambda s: " | ".join(sorted(set(map(str, s.dropna()))))),
        )
    )
    wide = (
        seed_long.pivot_table(
            index=["family", "task", "scenario", "model", "seed"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None
    wide = wide.merge(seed_meta, on=["family", "task", "scenario", "model", "seed"], how="left")
    wide["task_label"] = wide["task"].map(TASK_LABEL)
    wide["scenario_label"] = wide["scenario"].map(scenario_label)
    wide["family_label"] = FAMILY_LABEL
    if not seed_level_completion.empty:
        comp = seed_level_completion[
            [
                "task",
                "scenario",
                "model",
                "seed",
                "expected_folds",
                "n_unique_folds",
                "n_valid_completion_folds",
                "complete_seed",
                "missing_folds",
            ]
        ].copy()
        comp["seed"] = comp["seed"].astype(int)
        wide = wide.merge(comp, on=["task", "scenario", "model", "seed"], how="left")
    wide["complete_seed"] = wide.get("complete_seed", False).fillna(False).astype(bool)
    first_cols = ["family", "family_label", "task", "task_label", "scenario", "scenario_label", "model", "seed"]
    wide = wide[first_cols + [c for c in wide.columns if c not in first_cols]]
    for (task, scenario, model, seed), g in wide.groupby(["task", "scenario", "model", "seed"], sort=True):
        out_dir = root / TASK_DIR[task] / scenario / safe_model_dir(model) / "seed_metrics"
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
                "complete_seed": bool(g["complete_seed"].iloc[0]),
            }
        )
    return pd.DataFrame(manifest_rows), wide


def write_shap_files(shap: pd.DataFrame, pysr_usage: pd.DataFrame) -> None:
    root = V3 / "outputs" / FAMILY
    if not shap.empty:
        for (task, scenario, model), g in shap.groupby(["task", "scenario", "model"], sort=True):
            out_dir = root / TASK_DIR[task] / scenario / safe_model_dir(model) / "shap_importance"
            out_dir.mkdir(parents=True, exist_ok=True)
            g.sort_values("mean_abs_shap", ascending=False).to_csv(out_dir / "mean_abs_shap_by_feature.csv", index=False)
    if not pysr_usage.empty:
        out_dir = root / "pysr_variable_usage"
        out_dir.mkdir(parents=True, exist_ok=True)
        pysr_usage.to_csv(out_dir / "pysr_variable_usage_original_feature_names.csv", index=False)


def main() -> None:
    root = V3 / "outputs" / FAMILY
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (V3 / "manifests").mkdir(parents=True, exist_ok=True)
    seed_long = seed_metrics_sources()
    completion_aggregate, completion_seed_level = completion_table()
    shap = shap_summary()
    pysr_usage = pysr_variable_usage()
    manifest, wide = write_seed_metric_files(seed_long, completion_seed_level)
    write_shap_files(shap, pysr_usage)

    seed_long.to_csv(tables / "family2_seed_metrics_long.csv", index=False)
    wide.to_csv(tables / "family2_seed_metrics_wide.csv", index=False)
    manifest.to_csv(tables / "family2_seed_metrics_file_manifest.csv", index=False)
    completion_aggregate.to_csv(tables / "family2_completion_aggregate_best_available.csv", index=False)
    completion_seed_level.to_csv(tables / "family2_completion_seed_level_available.csv", index=False)
    shap.to_csv(tables / "family2_shap_mean_abs_by_feature.csv", index=False)
    if not pysr_usage.empty:
        pysr_usage.to_csv(tables / "family2_pysr_variable_usage_original_feature_names.csv", index=False)

    summary = {
        "family": FAMILY,
        "seed_metric_files": int(len(manifest)),
        "seed_metric_rows_long": int(len(seed_long)),
        "seed_metric_rows_wide": int(len(wide)),
        "completion_aggregate_rows": int(len(completion_aggregate)),
        "completion_seed_level_rows": int(len(completion_seed_level)),
        "shap_rows": int(len(shap)),
        "shap_files": int(len(list(root.rglob("mean_abs_shap_by_feature.csv")))),
        "pysr_variable_usage_rows": int(len(pysr_usage)),
        "source_policy": "seed-level metrics and mean SHAP only; no fold-level CSVs copied into v3 outputs",
    }
    (V3 / "manifests" / "family2_v3_seed_level_copy_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
