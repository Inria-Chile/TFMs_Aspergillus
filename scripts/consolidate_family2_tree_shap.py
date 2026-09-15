#!/usr/bin/env python3
"""Consolidate Family 2 RF/XGBoost fold-level SHAP values over 100 seeds."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd


TASKS = {
    "15A_occurrence": "classification",
    "15A_abundance": "regression_18",
    "17A_all_sample_abundance": "regression_50",
}
MODELS = {
    "random_forest": "random_forest",
    "xgboost": "xgboost",
}
SEED_RE = re.compile(r"__seed(\d+)__fold(\d+)$")


def consolidate(partials: Path, model: str) -> pd.DataFrame:
    by_seed: dict[tuple[int, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    observed_folds: dict[int, set[int]] = defaultdict(set)
    matched = 0

    for directory in partials.iterdir():
        if not directory.is_dir() or "__C_environment_microbiome__" not in directory.name:
            continue
        task = next((label for token, label in TASKS.items() if f"__{token}__" in directory.name), None)
        match = SEED_RE.search(directory.name)
        path = directory / "shap_values.csv"
        if task is None or match is None or not path.is_file():
            continue
        seed, fold = map(int, match.groups())
        frame = pd.read_csv(path, usecols=["variable", "abs_shap_value"])
        grouped = frame.groupby("variable", sort=False)["abs_shap_value"].agg(["sum", "count"])
        for feature, row in grouped.iterrows():
            key = (seed, f"{task}\0{feature}")
            by_seed[key][0] += float(row["sum"])
            by_seed[key][1] += float(row["count"])
        observed_folds[seed].add(fold)
        matched += 1

    seeds = sorted(observed_folds)
    if seeds != list(range(123, 223)):
        raise RuntimeError(f"{model}: expected seeds 123..222, found {seeds}")

    records = []
    for (seed, task_feature), (total, count) in by_seed.items():
        task, feature = task_feature.split("\0", 1)
        records.append((task, feature, seed, total / count))
    seed_means = pd.DataFrame(records, columns=["task", "feature", "seed", "seed_mean_abs_shap"])
    result = seed_means.groupby(["task", "feature"], as_index=False).agg(
        mean_abs_shap=("seed_mean_abs_shap", "mean"),
        sd_abs_shap=("seed_mean_abs_shap", "std"),
        n_seeds_with_feature=("seed", "nunique"),
    )
    result.insert(0, "family", "family2_full_microbiome_clr_raw_env")
    result.insert(1, "preprocessing", "With CLR")
    result.insert(3, "model", MODELS[model])
    result.insert(4, "scenario", "C_environment_microbiome")
    result["n_seeds_total"] = 100
    result["status"] = "complete_100_seeds"
    result["source"] = "family2_strategy48_fold_level_shap"
    print(f"{model}: {matched} fold files, {len(result)} aggregate rows")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-forest-partials", type=Path, required=True)
    parser.add_argument("--xgboost-partials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(consolidate, args.random_forest_partials, "random_forest"),
            executor.submit(consolidate, args.xgboost_partials, "xgboost"),
        ]
        frames = [future.result() for future in futures]
    output = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
