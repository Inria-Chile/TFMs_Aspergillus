#!/usr/bin/env python3
"""Merge historical and current seed-level SHAP, then aggregate by feature."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


KEY = ["family", "task", "scenario", "model", "seed", "feature"]
GROUP = [
    "family",
    "family_label",
    "task",
    "task_label",
    "scenario",
    "scenario_label",
    "model",
    "model_label",
    "feature",
]

TASK_ALIAS = {
    "regression_18_positive_samples": "regression_18_positive",
    "positive_abundance": "regression_18_positive",
    "all_sample_abundance": "regression_50_all_samples",
}
MODEL_ALIAS = {
    "rf": "random_forest",
    "random forest": "random_forest",
    "tfn": "tabpfn",
    "tfn_tabpfn": "tabpfn",
    "tfn/tabpfn": "tabpfn",
    "tabicl": "tabiclv2",
    "tabicl_v2": "tabiclv2",
}


def normalize(df: pd.DataFrame, current: bool) -> pd.DataFrame:
    out = df.copy()
    out["task"] = out["task"].astype(str).replace(TASK_ALIAS)
    out["model"] = out["model"].astype(str).str.strip().str.lower().replace(MODEL_ALIAS)
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce")
    out["mean_abs_shap"] = pd.to_numeric(out["mean_abs_shap"], errors="coerce")
    out = out.dropna(subset=KEY + ["mean_abs_shap"])
    out["seed"] = out["seed"].astype(int)
    if current:
        out["source_priority"] = 3
        out["source_type"] = "current_native_seed_shap"
    else:
        out["source_priority"] = pd.to_numeric(out.get("source_priority", 1), errors="coerce").fillna(1)
        if "source_type" not in out:
            out["source_type"] = "historical_combined_shap"
    keep = GROUP + ["seed", "mean_abs_shap", "source_priority", "source_type"]
    return out[keep]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-long", type=Path, required=True)
    parser.add_argument("--current-long", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    tables = args.output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    started = datetime.now()

    historical = normalize(pd.read_csv(args.historical_long), current=False)
    current = normalize(pd.read_csv(args.current_long), current=True)
    combined = pd.concat([historical, current], ignore_index=True, sort=False)
    combined = (
        combined.sort_values(["source_priority"], ascending=False)
        .drop_duplicates(KEY, keep="first")
        .sort_values(KEY)
    )
    combined.to_csv(tables / "combined_shap_seed_feature_long.csv", index=False)

    aggregate = combined.groupby(GROUP, as_index=False).agg(
        mean_abs_shap=("mean_abs_shap", "mean"),
        sd_abs_shap=("mean_abs_shap", "std"),
        median_abs_shap=("mean_abs_shap", "median"),
        q25_abs_shap=("mean_abs_shap", lambda s: s.quantile(0.25)),
        q75_abs_shap=("mean_abs_shap", lambda s: s.quantile(0.75)),
        n_seeds=("seed", "nunique"),
        source_type=("source_type", lambda s: ";".join(sorted(set(map(str, s))))),
    )
    aggregate["sem_abs_shap"] = aggregate["sd_abs_shap"] / aggregate["n_seeds"].pow(0.5)
    aggregate = aggregate.sort_values(
        ["family", "task", "scenario", "model", "mean_abs_shap"],
        ascending=[True, True, True, True, False],
    )
    aggregate.to_csv(tables / "combined_shap_mean_abs_by_feature.csv", index=False)
    aggregate.groupby(["family", "task", "scenario", "model"], group_keys=False).head(30).to_csv(
        tables / "combined_shap_top30_by_family_task_scenario_model.csv", index=False
    )

    coverage = (
        combined.drop_duplicates(["family", "task", "scenario", "model", "seed"])
        .groupby(["family", "task", "scenario", "model", "source_type"], as_index=False)
        .agg(n_seeds=("seed", "nunique"))
    )
    coverage.to_csv(tables / "combined_shap_source_coverage.csv", index=False)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "elapsed_seconds": round((datetime.now() - started).total_seconds(), 3),
        "historical_long": str(args.historical_long),
        "current_long": str(args.current_long),
        "historical_seed_feature_rows": int(len(historical)),
        "current_seed_feature_rows": int(len(current)),
        "combined_seed_feature_rows": int(len(combined)),
        "aggregate_feature_rows": int(len(aggregate)),
        "n_groups": int(combined.groupby(["family", "task", "scenario", "model"]).ngroups),
        "deduplication_key": KEY,
        "priority_rule": "current native seed SHAP replaces historical SHAP for the same key",
    }
    (tables / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
