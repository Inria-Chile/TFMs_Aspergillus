#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from marta_omar_repro.data import detect_blocks, load_table
from marta_omar_repro.models import select_features_inside_fold
from run_omar_repro import load_config, parse_seeds, scenario_map


def safe_id(*parts: object) -> str:
    raw = "__".join(str(p) for p in parts)
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", raw).strip("_")


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def select_features(df: pd.DataFrame, cfg: dict, scenario: str, feature_cols: list[str], train_idx, task: str) -> list[str]:
    forced = [c for c in cfg["dummy_cols"] if c in feature_cols]
    core = [c for c in feature_cols if c not in forced]
    max_features = cfg["max_selected_features_by_scenario"].get(scenario)
    if max_features is not None:
        max_features = min(int(max_features), len(core))
    y_col = "Aspergillus_presence" if task == "classification" else cfg["target_col"]
    if task == "classification":
        y_train = df.iloc[train_idx][y_col].astype(int).to_numpy()
        X_train = df.iloc[train_idx][feature_cols]
        X_test = df.iloc[train_idx[:1]][feature_cols]
    else:
        y_train = np.log1p(df.iloc[train_idx][y_col].astype(float).to_numpy())
        X_train = df.iloc[train_idx][feature_cols]
        X_test = df.iloc[train_idx[:1]][feature_cols]
    _, _, selected = select_features_inside_fold(
        X_train,
        y_train,
        X_test,
        task=task,
        max_features=max_features,
        forced_cols=forced,
    )
    return selected


def build_occurrence_15a(df: pd.DataFrame, cfg: dict, seeds: list[int], include_full: bool) -> list[dict]:
    blocks = detect_blocks(df, cfg["target_col"], cfg["dummy_cols"])
    scenarios = scenario_map(df, blocks, cfg["target_col"])
    rows = []
    y = df["Aspergillus_presence"].astype(int).to_numpy()
    for seed in seeds:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
        for scenario, features in scenarios.items():
            for fold_id, (train_idx, test_idx) in enumerate(cv.split(df[features], y), start=1):
                variants = ["anova"]
                if include_full:
                    variants.append("full")
                for variant in variants:
                    if variant == "full":
                        selected = list(features)
                    else:
                        selected = select_features(df, cfg, scenario, list(features), train_idx, "classification")
                    task_id = safe_id("15A_occurrence", scenario, variant, f"seed{seed}", f"fold{fold_id}")
                    rows.append(
                        {
                            "task_id": task_id,
                            "analysis": "15A",
                            "task_kind": "occurrence",
                            "scenario": scenario,
                            "variant": variant,
                            "model": f"TabICLv2_{variant}",
                            "seed": seed,
                            "fold_id": fold_id,
                            "target_transform": "presence_0_1",
                            "train_rows_json": dumps([int(i) for i in train_idx]),
                            "test_rows_json": dumps([int(i) for i in test_idx]),
                            "features_json": dumps(selected),
                            "n_features": len(selected),
                            "n_train": len(train_idx),
                            "n_test": len(test_idx),
                        }
                    )
    return rows


def build_abundance_15a(df: pd.DataFrame, cfg: dict, seeds: list[int], include_full: bool) -> list[dict]:
    blocks = detect_blocks(df, cfg["target_col"], cfg["dummy_cols"])
    scenarios = scenario_map(df, blocks, cfg["target_col"])
    pos_rows = np.where(df[cfg["target_col"]].astype(float).to_numpy() > 0)[0]
    data_pos = df.iloc[pos_rows].reset_index(drop=False).rename(columns={"index": "source_row"})
    rows = []
    for seed in seeds:
        for scenario, features in scenarios.items():
            loo = LeaveOneOut()
            for fold_id, (train_local, test_local) in enumerate(loo.split(data_pos), start=1):
                train_idx = data_pos.iloc[train_local]["source_row"].astype(int).to_numpy()
                test_idx = data_pos.iloc[test_local]["source_row"].astype(int).to_numpy()
                variants = ["anova"]
                if include_full:
                    variants.append("full")
                for variant in variants:
                    if variant == "full":
                        selected = list(features)
                    else:
                        selected = select_features(df, cfg, scenario, list(features), train_idx, "regression")
                    task_id = safe_id("15A_abundance", scenario, variant, f"seed{seed}", f"fold{fold_id}")
                    rows.append(
                        {
                            "task_id": task_id,
                            "analysis": "15A",
                            "task_kind": "conditional_abundance",
                            "scenario": scenario,
                            "variant": variant,
                            "model": f"TabICLv2_{variant}",
                            "seed": seed,
                            "fold_id": fold_id,
                            "target_transform": "log1p_positive_abundance",
                            "train_rows_json": dumps([int(i) for i in train_idx]),
                            "test_rows_json": dumps([int(i) for i in test_idx]),
                            "features_json": dumps(selected),
                            "n_features": len(selected),
                            "n_train": len(train_idx),
                            "n_test": len(test_idx),
                        }
                    )
    return rows


def build_occurrence_16a(df: pd.DataFrame, cfg: dict, seeds: list[int]) -> list[dict]:
    predictors = cfg["dummy_cols"] + cfg["rf_best_occurrence_covars"]
    y = df["Aspergillus_presence"].astype(int).to_numpy()
    rows = []
    for seed in seeds:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
        for fold_id, (train_idx, test_idx) in enumerate(cv.split(df[predictors], y), start=1):
            task_id = safe_id("16A_occurrence_rfbest", f"seed{seed}", f"fold{fold_id}")
            rows.append(
                {
                    "task_id": task_id,
                    "analysis": "16A",
                    "task_kind": "occurrence",
                    "scenario": "RFbest_occurrence",
                    "variant": "rfbest",
                    "model": "TabICLv2_rfbest",
                    "seed": seed,
                    "fold_id": fold_id,
                    "target_transform": "presence_0_1",
                    "train_rows_json": dumps([int(i) for i in train_idx]),
                    "test_rows_json": dumps([int(i) for i in test_idx]),
                    "features_json": dumps(predictors),
                    "n_features": len(predictors),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                }
            )
    return rows


def build_abundance_16a(df: pd.DataFrame, cfg: dict, seeds: list[int]) -> list[dict]:
    predictors = cfg["dummy_cols"] + cfg["rf_best_occurrence_covars"]
    pos_rows = np.where(df[cfg["target_col"]].astype(float).to_numpy() > 0)[0]
    data_pos = df.iloc[pos_rows].reset_index(drop=False).rename(columns={"index": "source_row"})
    rows = []
    for seed in seeds:
        for fold_id, (train_local, test_local) in enumerate(LeaveOneOut().split(data_pos), start=1):
            train_idx = data_pos.iloc[train_local]["source_row"].astype(int).to_numpy()
            test_idx = data_pos.iloc[test_local]["source_row"].astype(int).to_numpy()
            task_id = safe_id("16A_abundance_rfbest", f"seed{seed}", f"fold{fold_id}")
            rows.append(
                {
                    "task_id": task_id,
                    "analysis": "16A",
                    "task_kind": "conditional_abundance",
                    "scenario": "RFbest_occurrence",
                    "variant": "rfbest",
                    "model": "TabICLv2_rfbest",
                    "seed": seed,
                    "fold_id": fold_id,
                    "target_transform": "log1p_positive_abundance",
                    "train_rows_json": dumps([int(i) for i in train_idx]),
                    "test_rows_json": dumps([int(i) for i in test_idx]),
                    "features_json": dumps(predictors),
                    "n_features": len(predictors),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                }
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "omar_repro_config.json"))
    ap.add_argument("--output-csv", default=str(ROOT / "monitoring" / "tabiclv2_marta" / "tasks.csv"))
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--include-full", action="store_true", help="Also run full high-dimensional variants.")
    ap.add_argument("--analyses", nargs="+", default=["15A_occurrence", "15A_abundance", "16A_occurrence"])
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    base = Path(args.config).resolve().parents[1]
    df = load_table((base / cfg["input_csv"]).resolve(), cfg["target_col"], cfg["id_col"])
    seeds = parse_seeds(args.seeds, args.n_seeds, int(cfg["random_state"]))

    rows: list[dict] = []
    if "15A_occurrence" in args.analyses:
        rows.extend(build_occurrence_15a(df, copy.deepcopy(cfg), seeds, args.include_full))
    if "15A_abundance" in args.analyses:
        rows.extend(build_abundance_15a(df, copy.deepcopy(cfg), seeds, args.include_full))
    if "16A_occurrence" in args.analyses:
        rows.extend(build_occurrence_16a(df, copy.deepcopy(cfg), seeds))
    if "16A_abundance" in args.analyses:
        rows.extend(build_abundance_16a(df, copy.deepcopy(cfg), seeds))

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    tasks = pd.DataFrame(rows)
    tasks.to_csv(out, index=False)
    by_analysis = {
        f"{analysis}:{task_kind}": int(count)
        for (analysis, task_kind), count in tasks.groupby(["analysis", "task_kind"]).size().items()
    } if not tasks.empty else {}
    print(
        json.dumps(
            {
                "output_csv": str(out),
                "n_tasks": int(len(tasks)),
                "n_seeds": len(seeds),
                "seeds": seeds,
                "include_full": bool(args.include_full),
                "by_analysis": by_analysis,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
