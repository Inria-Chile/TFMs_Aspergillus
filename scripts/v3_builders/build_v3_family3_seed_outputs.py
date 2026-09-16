#!/usr/bin/env python3
"""Build clean v3 seed-level outputs for family 3.

Family 3 is the Marta retained subset strategy with raw environment variables
and selected microbiome variables transformed with CLR
(`strategy49_marta_subset_raw_env_selected_microbiome_clr`).

The v3 contract is intentionally stricter than v2:
- write one metrics CSV per seed;
- do not copy fold-level CSVs into v3 outputs;
- keep fold/provenance paths only in tables/manifests;
- write SHAP as mean absolute contribution by original feature name.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
V2 = Path(os.environ.get("ASPERGILLUS_LEGACY_ROOT", REPO / "data/legacy_v2"))
V3 = Path(os.environ.get("ASPERGILLUS_REPO_ROOT", REPO))
FAMILY = "family3_marta_subset_clr_raw_env"
FAMILY_LABEL = "Family 3: Marta subset + microbiome CLR + raw environment"
STRATEGY = "strategy49_marta_subset_raw_env_selected_microbiome_clr"
AUDIT_DIR = V2 / "outputs/59_family3_deep_search_20260628_2035"

TASK_DIR = {
    "classification": "classification",
    "positive_abundance": "regression_18_positive_samples",
    "all_sample_abundance": "regression_50_all_samples",
}
TASK_LABEL = {
    "classification": "Classification",
    "positive_abundance": "Positive-abundance regression",
    "all_sample_abundance": "Complete-abundance regression",
}
EXPECTED_FOLDS = {
    "classification": 15,
    "positive_abundance": 18,
    "all_sample_abundance": 15,
}
MODEL_SLUG = {
    "PySR": "pysr",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "TFN/TabPFN": "tabpfn",
    "TabICLv2": "tabiclv2",
}
MODELS = ["PySR", "Random Forest", "XGBoost", "TFN/TabPFN", "TabICLv2"]
SCENARIOS = ["A_environment", "B_microbiome", "C_environment_microbiome"]
TASKS = ["classification", "positive_abundance", "all_sample_abundance"]


def scenario_label(scenario: str) -> str:
    return {
        "A_environment": "A",
        "B_microbiome": "B",
        "C_environment_microbiome": "C",
    }.get(scenario, scenario)


def safe_model_dir(model: str) -> str:
    return MODEL_SLUG.get(model, re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").lower())


def normalize_seed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["seed"] = df["seed"].astype(int)
    return df


def seed_metrics_from_seed_tables() -> list[pd.DataFrame]:
    specs = [
        ("outputs/51_three_family_complete_replots/three_family_seed_average_metrics_long.csv", 0),
        ("outputs/56_incremental_seed_fold_aggregation_20260628_0856/tables/updated_seed_average_metrics_long.csv", 2),
        ("outputs/55_seed_fold_aggregation_20260626_2056/tables/seed_average_metrics_long.csv", 3),
    ]
    frames: list[pd.DataFrame] = []
    for rel, priority in specs:
        p = V2 / rel
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df = df[df["family"].eq(FAMILY)].copy()
        if df.empty:
            continue
        df = normalize_seed(df)
        df["source_table"] = str(p)
        df["source_priority"] = priority
        for col in ["complete_seed", "fold_metric_sd"]:
            if col not in df.columns:
                df[col] = pd.NA
        frames.append(
            df[
                [
                    "family",
                    "model",
                    "task",
                    "scenario",
                    "seed",
                    "metric",
                    "value",
                    "n_folds_tested",
                    "complete_seed",
                    "fold_metric_sd",
                    "source_table",
                    "source_priority",
                ]
            ]
        )
    return frames


def strategy49_records() -> pd.DataFrame:
    p = V2 / "outputs/50_strategy48_49_replots/tables/model_fold_records_from_jsonl.csv"
    df = pd.read_csv(p)
    df = df[df["strategy"].eq(STRATEGY)].copy()
    df["family"] = FAMILY
    return normalize_seed(df)


def pysr_equation_records() -> pd.DataFrame:
    p = V2 / "outputs/50_strategy48_49_replots/tables/pysr_equations_scanned_parallel_quick_merc7.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df = df[df["strategy"].eq(STRATEGY)].copy()
    if df.empty:
        return df
    df["family"] = FAMILY
    df["model"] = "PySR"
    return normalize_seed(df)


def seed_metrics_from_strategy49_folds(records: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["AUC", "balanced_accuracy", "F1", "prediction_mean", "RMSE", "MAE", "R2"]
    have = [c for c in metric_cols if c in records.columns]
    if records.empty or not have:
        return pd.DataFrame()
    key = ["family", "model", "task", "scenario", "seed"]
    n_folds = records.groupby(key)["fold_id"].nunique().reset_index(name="n_folds_tested")
    long = records.melt(
        id_vars=key,
        value_vars=have,
        var_name="metric",
        value_name="value",
    )
    long = long.dropna(subset=["value"])
    long = long.groupby(key + ["metric"], as_index=False)["value"].mean()
    long = long.merge(n_folds, on=key, how="left")
    long["complete_seed"] = long.apply(lambda r: int(r["n_folds_tested"]) >= EXPECTED_FOLDS.get(r["task"], 999), axis=1)
    long["fold_metric_sd"] = pd.NA
    long["source_table"] = str(V2 / "outputs/50_strategy48_49_replots/tables/model_fold_records_from_jsonl.csv")
    long["source_priority"] = 1
    return long


def seed_metrics_sources() -> pd.DataFrame:
    frames = seed_metrics_from_seed_tables()
    records = strategy49_records()
    fold_metrics = seed_metrics_from_strategy49_folds(records)
    if not fold_metrics.empty:
        frames.append(fold_metrics)
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = normalize_seed(out)
    out = out.sort_values(["family", "task", "scenario", "model", "seed", "metric", "source_priority"])
    out = out.drop_duplicates(["family", "task", "scenario", "model", "seed", "metric"], keep="first")
    return out.drop(columns=["source_priority"])


def completion_from_strategy49(records: pd.DataFrame, pysr_records: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if not records.empty:
        g = records.groupby(["task", "scenario", "model", "seed"])["fold_id"].nunique().reset_index(name="n_unique_folds")
        g["expected_folds"] = g["task"].map(EXPECTED_FOLDS)
        g["complete_seed"] = g["n_unique_folds"] >= g["expected_folds"]
        rows.append(g)
    if not pysr_records.empty:
        g = pysr_records.groupby(["task", "scenario", "model", "seed"])["fold_id"].nunique().reset_index(name="n_unique_folds")
        g["expected_folds"] = g["task"].map(EXPECTED_FOLDS)
        g["complete_seed"] = g["n_unique_folds"] >= g["expected_folds"]
        rows.append(g)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    out["family"] = FAMILY
    out["source_table"] = str(V2 / "outputs/50_strategy48_49_replots/tables")
    return normalize_seed(out)


def completion_from_seed_manifests() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for rel in [
        "outputs/56_incremental_seed_fold_aggregation_20260628_0856/tables/updated_seed_file_manifest.csv",
        "outputs/55_seed_fold_aggregation_20260626_2056/tables/seed_file_manifest.csv",
    ]:
        p = V2 / rel
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df = df[df["family"].eq(FAMILY)].copy()
        if df.empty:
            continue
        df = normalize_seed(df)
        df["source_table"] = str(p)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out


def aggregate_completion(records: pd.DataFrame, pysr_records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategy_seed = completion_from_strategy49(records, pysr_records)
    seed_manifest = completion_from_seed_manifests()

    agg56_p = V2 / "outputs/56_incremental_seed_fold_aggregation_20260628_0856/tables/updated_completion_by_family_task_model_scenario.csv"
    agg56 = pd.read_csv(agg56_p)
    agg56 = agg56[agg56["family"].eq(FAMILY)].copy()

    rows: list[dict[str, object]] = []
    for task in TASKS:
        for model in MODELS:
            row: dict[str, object] = {
                "Family": FAMILY_LABEL,
                "Task": TASK_LABEL[task],
                "task": task,
                "Model": model,
                "Recommended resource": "CPU" if model in {"PySR", "Random Forest"} else ("GPU preferred/CPU possible" if model == "XGBoost" else "GPU"),
            }
            total_ready = 0
            total_pending = 0
            for scenario in SCENARIOS:
                s49_ready = 0
                s49_any = 0
                if not strategy_seed.empty:
                    sub = strategy_seed[(strategy_seed["task"].eq(task)) & (strategy_seed["scenario"].eq(scenario)) & (strategy_seed["model"].eq(model))]
                    s49_ready = int(sub["complete_seed"].sum()) if not sub.empty else 0
                    s49_any = int(sub["seed"].nunique()) if not sub.empty else 0
                a56 = agg56[(agg56["task"].eq(task)) & (agg56["scenario"].eq(scenario)) & (agg56["model"].eq(model))]
                a56_ready = int(a56["complete_seeds"].iloc[0]) if not a56.empty else 0
                ready = max(s49_ready, a56_ready)
                pending = max(0, 100 - ready)
                source_bits = []
                if s49_ready or s49_any:
                    source_bits.append("strategy49_outputs50")
                if a56_ready:
                    source_bits.append("aggregation56")
                row[f"{scenario}_listas"] = ready
                row[f"{scenario}_pendientes"] = pending
                row[f"{scenario}_ready_strategy49"] = s49_ready
                row[f"{scenario}_any_strategy49"] = s49_any
                row[f"{scenario}_ready_aggregation56"] = a56_ready
                row[f"{scenario}_source"] = ";".join(source_bits) if source_bits else "missing"
                total_ready += ready
                total_pending += pending
            row["Listas_total_ABC"] = total_ready
            row["Pendientes_total_ABC"] = total_pending
            rows.append(row)

    combined = pd.DataFrame(rows)
    tables = AUDIT_DIR / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    combined.to_csv(tables / "family3_combined_best_available_completion_strategy49_plus_56.csv", index=False)
    strategy_seed.to_csv(tables / "family3_strategy49_seed_completion.csv", index=False)
    seed_manifest.to_csv(tables / "family3_seed_manifest_sources_55_56.csv", index=False)
    write_completion_markdown(combined, tables / "family3_combined_best_available_completion_strategy49_plus_56.md")

    seed_level_frames = []
    if not seed_manifest.empty:
        seed_level_frames.append(seed_manifest)
    if not strategy_seed.empty:
        s = strategy_seed.copy()
        s["n_valid_completion_folds"] = s["n_unique_folds"]
        s["missing_folds"] = s.apply(lambda r: ",".join(map(str, range(int(r["n_unique_folds"]) + 1, int(r["expected_folds"]) + 1))) if int(r["n_unique_folds"]) < int(r["expected_folds"]) else "", axis=1)
        seed_level_frames.append(s)
    seed_level = pd.concat(seed_level_frames, ignore_index=True, sort=False) if seed_level_frames else pd.DataFrame()
    if not seed_level.empty:
        seed_level = normalize_seed(seed_level)
        for col in ["expected_folds", "n_unique_folds", "n_valid_completion_folds", "complete_seed", "missing_folds", "source_table"]:
            if col not in seed_level.columns:
                seed_level[col] = pd.NA
        seed_level = seed_level.sort_values(["complete_seed", "n_unique_folds"], ascending=False)
        seed_level = seed_level.drop_duplicates(["family", "task", "scenario", "model", "seed"], keep="first")
    return combined, seed_level


def write_completion_markdown(df: pd.DataFrame, out_md: Path) -> None:
    parts = [f"# {FAMILY_LABEL}", ""]
    for task in TASKS:
        parts += [f"## {TASK_LABEL[task]}", ""]
        sub = df[df["task"].eq(task)].copy()
        table = sub[
            [
                "Model",
                "Recommended resource",
                "A_environment_listas",
                "A_environment_pendientes",
                "B_microbiome_listas",
                "B_microbiome_pendientes",
                "C_environment_microbiome_listas",
                "C_environment_microbiome_pendientes",
                "Listas_total_ABC",
                "Pendientes_total_ABC",
            ]
        ]
        parts.append("| " + " | ".join(table.columns) + " |")
        parts.append("|" + "|".join(["---"] * len(table.columns)) + "|")
        for row in table.itertuples(index=False):
            parts.append("| " + " | ".join(str(x) for x in row) + " |")
        parts.append("")
    out_md.write_text("\n".join(parts), encoding="utf-8")


def read_source51_shap() -> pd.DataFrame:
    p = V2 / "outputs/51_three_family_complete_replots/three_family_shap_feature_importance_summary.csv"
    df = pd.read_csv(p)
    df = df[df["family"].eq(FAMILY)].copy()
    if df.empty:
        return df
    df["source_table"] = str(p)
    return df


def aggregate_strategy49_shap(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if records.empty or "shap_csv" not in records.columns:
        return pd.DataFrame(), pd.DataFrame()
    cols = ["task", "scenario", "model", "seed", "shap_csv"]
    todo = records[cols].dropna(subset=["shap_csv"]).drop_duplicates().sort_values(cols).reset_index(drop=True)
    chunk_dir = AUDIT_DIR / "tables/shap_seed_feature_chunks"
    failure_dir = AUDIT_DIR / "tables/shap_read_failure_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)
    total = len(todo)
    chunk_size = 500
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk_idx = start // chunk_size
        chunk_out = chunk_dir / f"shap_seed_feature_chunk_{chunk_idx:04d}_{start:06d}_{end:06d}.csv"
        fail_out = failure_dir / f"shap_failures_chunk_{chunk_idx:04d}_{start:06d}_{end:06d}.csv"
        if chunk_out.exists():
            print(f"[family3 shap] checkpoint exists {end}/{total} fold files", flush=True)
            continue
        seed_feature_sum: dict[tuple[str, str, str, int, str], float] = defaultdict(float)
        seed_feature_n: dict[tuple[str, str, str, int, str], int] = defaultdict(int)
        failures: list[dict[str, object]] = []
        subset = todo.iloc[start:end]
        for row in subset.itertuples(index=False):
            path = Path(str(row.shap_csv))
            if not path.exists():
                failures.append({"shap_csv": str(path), "reason": "missing"})
                continue
            try:
                df = pd.read_csv(path, usecols=["variable", "abs_shap_value"])
                df = df.dropna(subset=["variable", "abs_shap_value"])
                if df.empty:
                    failures.append({"shap_csv": str(path), "reason": "empty"})
                    continue
                g = df.groupby("variable", as_index=False)["abs_shap_value"].mean()
            except Exception as exc:  # noqa: BLE001 - keep audit provenance
                failures.append({"shap_csv": str(path), "reason": type(exc).__name__, "message": str(exc)[:200]})
                continue
            for feat, val in zip(g["variable"], g["abs_shap_value"]):
                key = (row.task, row.scenario, row.model, int(row.seed), str(feat))
                seed_feature_sum[key] += float(val)
                seed_feature_n[key] += 1
        seed_rows = []
        for key, val_sum in seed_feature_sum.items():
            n = seed_feature_n[key]
            task, scenario, model, seed, feature = key
            seed_rows.append(
                {
                    "family": FAMILY,
                    "task": task,
                    "scenario": scenario,
                    "model": model,
                    "seed": seed,
                    "feature": feature,
                    "seed_mean_abs_shap": val_sum / n if n else math.nan,
                    "n_folds_with_shap": n,
                }
            )
        pd.DataFrame(seed_rows).to_csv(chunk_out, index=False)
        pd.DataFrame(failures).to_csv(fail_out, index=False)
        print(f"[family3 shap] checkpoint wrote {end}/{total} fold files", flush=True)

    chunk_files = sorted(chunk_dir.glob("shap_seed_feature_chunk_*.csv"))
    if not chunk_files:
        return pd.DataFrame(), pd.DataFrame()
    seed_df = pd.concat((pd.read_csv(p) for p in chunk_files), ignore_index=True, sort=False)
    failure_files = sorted(failure_dir.glob("shap_failures_chunk_*.csv"))
    failure_frames = []
    for p in failure_files:
        if p.stat().st_size:
            try:
                failure_frames.append(pd.read_csv(p))
            except pd.errors.EmptyDataError:
                pass
    failures_df = pd.concat(failure_frames, ignore_index=True, sort=False) if failure_frames else pd.DataFrame()
    if seed_df.empty:
        return pd.DataFrame(), failures_df
    seed_df = (
        seed_df.groupby(["family", "task", "scenario", "model", "seed", "feature"], as_index=False)
        .agg(
            seed_mean_abs_shap=("seed_mean_abs_shap", "mean"),
            n_folds_with_shap=("n_folds_with_shap", "sum"),
        )
    )
    summary = (
        seed_df.groupby(["family", "task", "scenario", "model", "feature"])
        .agg(
            mean_abs_shap=("seed_mean_abs_shap", "mean"),
            sd_abs_shap=("seed_mean_abs_shap", "std"),
            n_seeds=("seed", "nunique"),
            n_seed_feature_rows=("seed", "size"),
        )
        .reset_index()
    )
    summary["source_table"] = str(V2 / "outputs/50_strategy48_49_replots/tables/model_fold_records_from_jsonl.csv")
    return summary, failures_df


def shap_summary(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategy_shap, failures = aggregate_strategy49_shap(records)
    source51 = read_source51_shap()
    frames = []
    if not strategy_shap.empty:
        strategy_shap["source_priority"] = 0
        frames.append(strategy_shap)
    if not source51.empty:
        source51["source_priority"] = 1
        if "n_seed_feature_rows" not in source51.columns:
            source51["n_seed_feature_rows"] = pd.NA
        frames.append(source51)
    if not frames:
        return pd.DataFrame(), failures
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.sort_values(["task", "scenario", "model", "feature", "source_priority"])
    out = out.drop_duplicates(["family", "task", "scenario", "model", "feature"], keep="first")
    return out.drop(columns=["source_priority"]), failures


def pysr_variable_usage() -> pd.DataFrame:
    p = V2 / "outputs/51_three_family_complete_replots/three_family_pysr_variable_usage_summary.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df = df[df["family"].eq(FAMILY)].copy()
    if df.empty:
        return df
    df["source_table"] = str(p)
    return df


def write_seed_metric_files(seed_long: pd.DataFrame, seed_level_completion: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = V3 / "outputs" / FAMILY
    manifest_rows: list[dict[str, object]] = []
    seed_meta = (
        seed_long.groupby(["family", "task", "scenario", "model", "seed"], as_index=False)
        .agg(
            n_folds_tested=("n_folds_tested", "max"),
            complete_seed_from_metric_source=("complete_seed", lambda s: bool(pd.Series(s).fillna(False).astype(bool).max())),
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
        comp_cols = [
            "family",
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
        comp = seed_level_completion[[c for c in comp_cols if c in seed_level_completion.columns]].copy()
        comp = normalize_seed(comp)
        wide = wide.merge(comp, on=["family", "task", "scenario", "model", "seed"], how="left")
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
    (AUDIT_DIR / "tables").mkdir(parents=True, exist_ok=True)

    records = strategy49_records()
    pysr_records = pysr_equation_records()
    seed_long = seed_metrics_sources()
    completion_aggregate, completion_seed_level = aggregate_completion(records, pysr_records)
    shap, shap_failures = shap_summary(records)
    pysr_usage = pysr_variable_usage()
    manifest, wide = write_seed_metric_files(seed_long, completion_seed_level)
    write_shap_files(shap, pysr_usage)

    seed_long.to_csv(tables / "family3_seed_metrics_long.csv", index=False)
    wide.to_csv(tables / "family3_seed_metrics_wide.csv", index=False)
    manifest.to_csv(tables / "family3_seed_metrics_file_manifest.csv", index=False)
    completion_aggregate.to_csv(tables / "family3_completion_aggregate_best_available.csv", index=False)
    completion_seed_level.to_csv(tables / "family3_completion_seed_level_available.csv", index=False)
    shap.to_csv(tables / "family3_shap_mean_abs_by_feature.csv", index=False)
    shap_failures.to_csv(tables / "family3_shap_read_failures.csv", index=False)
    if not pysr_usage.empty:
        pysr_usage.to_csv(tables / "family3_pysr_variable_usage_original_feature_names.csv", index=False)

    summary = {
        "family": FAMILY,
        "seed_metric_files": int(len(manifest)),
        "seed_metric_rows_long": int(len(seed_long)),
        "seed_metric_rows_wide": int(len(wide)),
        "completion_aggregate_rows": int(len(completion_aggregate)),
        "completion_seed_level_rows": int(len(completion_seed_level)),
        "strategy49_fold_records": int(len(records)),
        "strategy49_pysr_equation_records": int(len(pysr_records)),
        "shap_rows": int(len(shap)),
        "shap_files": int(len(list(root.rglob("mean_abs_shap_by_feature.csv")))),
        "shap_read_failures": int(len(shap_failures)),
        "pysr_variable_usage_rows": int(len(pysr_usage)),
        "source_policy": "seed-level metrics and mean SHAP only; no fold-level CSVs copied into v3 outputs",
    }
    (V3 / "manifests" / "family3_v3_seed_level_copy_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
