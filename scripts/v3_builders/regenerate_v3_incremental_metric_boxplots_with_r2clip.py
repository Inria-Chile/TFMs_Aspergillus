#!/usr/bin/env python3
"""Fast incremental refresh of V3 metric boxplots.

Uses a previous full consolidation as base and re-reads only seed metric CSVs
modified after a given timestamp. The plotting functions are imported from
regenerate_v3_latest_metric_boxplots_with_r2clip.py.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from regenerate_v3_latest_metric_boxplots_with_r2clip import (  # noqa: E402
    FAMILIES,
    MODELS,
    SCENARIOS,
    TASKS,
    make_archive,
    make_set100,
    make_set99,
    progress,
    truthy,
)


METRIC_COLUMNS = ["AUC", "F1", "balanced_accuracy", "MAE", "RMSE", "R2", "prediction_mean"]


def parse_seed(path: Path) -> int | None:
    match = re.match(r"seed_(\d+)_metrics\.csv$", path.name)
    return int(match.group(1)) if match else None


TASK_ALIAS = {}
TASK_EXPECTED_FOLDS = {}
TASK_LABEL_ES = {}
TASK_LABEL_EN = {}
for task, label_es, label_en, _short, aliases, expected_folds in TASKS:
    TASK_EXPECTED_FOLDS[task] = expected_folds
    TASK_LABEL_ES[task] = label_es
    TASK_LABEL_EN[task] = label_en
    for alias in aliases:
        TASK_ALIAS[alias] = task

MODEL_ALIAS = {}
MODEL_LABEL = {}
for model, label, aliases in MODELS:
    MODEL_LABEL[model] = label
    for alias in aliases:
        MODEL_ALIAS[alias] = model

FAMILY_LABEL = {x[0]: x[1] for x in FAMILIES}
SCENARIO_LABEL = {x[0]: x[1] for x in SCENARIOS}
SCENARIO_TITLE = {x[0]: x[2] for x in SCENARIOS}
SCENARIO_FILE = {x[0]: x[3] for x in SCENARIOS}


def valid_seed_metric(row: dict[str, object], task: str) -> bool:
    flag = truthy(row.get("seed_complete_all_folds"))
    if flag is not None:
        return flag
    if "n_folds_tested" in row:
        folds = pd.to_numeric(pd.Series([row["n_folds_tested"]]), errors="coerce").iloc[0]
        if pd.notna(folds):
            return int(folds) >= TASK_EXPECTED_FOLDS[task]
    return True


def find_recent_metric_files(v3_root: Path, since: str) -> list[Path]:
    dirs = [str(v3_root / "outputs" / family) for family, *_ in FAMILIES]
    cmd = ["find", *dirs, "-path", "*/seed_metrics/seed_*_metrics.csv", "-type", "f", "-newermt", since]
    out = subprocess.check_output(cmd, text=True)
    return [Path(line) for line in out.splitlines() if line.strip()]


def parse_metric_path(v3_root: Path, path: Path) -> dict[str, str] | None:
    try:
        rel = path.relative_to(v3_root / "outputs")
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 6:
        return None
    family, task_dir, scenario, model_dir, leaf = parts[:5]
    if leaf != "seed_metrics":
        return None
    task = TASK_ALIAS.get(task_dir)
    model = MODEL_ALIAS.get(model_dir)
    if family not in FAMILY_LABEL or task is None or scenario not in SCENARIO_LABEL or model is None:
        return None
    return {"family": family, "task": task, "scenario": scenario, "model": model, "task_dir": task_dir, "model_dir": model_dir}


def read_recent_metrics(v3_root: Path, since: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    paths = find_recent_metric_files(v3_root, since)
    progress(f"incremental: recent seed metric files since {since}: {len(paths)}")
    for path in paths:
        meta = parse_metric_path(v3_root, path)
        seed = parse_seed(path)
        if meta is None or seed is None:
            continue
        try:
            stat = path.stat()
            if stat.st_size == 0:
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
        except Exception:
            continue
        row = df.iloc[0].to_dict()
        if not valid_seed_metric(row, meta["task"]):
            continue
        row.update(
            {
                "family": meta["family"],
                "family_label": FAMILY_LABEL[meta["family"]],
                "task": meta["task"],
                "task_label": TASK_LABEL_ES[meta["task"]],
                "task_label_en": TASK_LABEL_EN[meta["task"]],
                "scenario": meta["scenario"],
                "scenario_label": SCENARIO_LABEL[meta["scenario"]],
                "scenario_title": SCENARIO_TITLE[meta["scenario"]],
                "scenario_file": SCENARIO_FILE[meta["scenario"]],
                "model": meta["model"],
                "model_label": MODEL_LABEL[meta["model"]],
                "seed": seed,
                "source_csv": str(path),
                "source_mtime": stat.st_mtime,
                "source_size": stat.st_size,
                "task_dir_used": meta["task_dir"],
                "model_dir_used": meta["model_dir"],
            }
        )
        flag = truthy(row.get("seed_complete_all_folds"))
        row["_complete_rank"] = int(bool(flag)) if flag is not None else 0
        rows.append(row)
    return pd.DataFrame(rows)


def normalize_base_wide(base_wide: pd.DataFrame) -> pd.DataFrame:
    base = base_wide.copy()
    if "task_label_en" not in base.columns:
        base["task_label_en"] = base["task"].map(TASK_LABEL_EN)
    if "scenario_title" not in base.columns:
        base["scenario_title"] = base["scenario"].map(SCENARIO_TITLE)
    if "scenario_file" not in base.columns:
        base["scenario_file"] = base["scenario"].map(SCENARIO_FILE)
    if "source_mtime" not in base.columns and "_mtime" in base.columns:
        base["source_mtime"] = base["_mtime"]
    if "source_size" not in base.columns and "_size_bytes" in base.columns:
        base["source_size"] = base["_size_bytes"]
    base["_complete_rank"] = 0
    if "source_mtime" not in base.columns:
        base["source_mtime"] = 0
    if "source_size" not in base.columns:
        base["source_size"] = 0
    base["source_mtime"] = pd.to_numeric(base["source_mtime"], errors="coerce").fillna(0)
    base["source_size"] = pd.to_numeric(base["source_size"], errors="coerce").fillna(0)
    return base


def wide_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in METRIC_COLUMNS if c in wide.columns]
    long = wide.melt(
        id_vars=[
            "family",
            "family_label",
            "task",
            "task_label",
            "task_label_en",
            "scenario",
            "scenario_label",
            "scenario_title",
            "scenario_file",
            "model",
            "model_label",
            "seed",
            "source_csv",
        ],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long.dropna(subset=["value"])


def write_completion(wide: pd.DataFrame, tables: Path) -> None:
    completed = wide.groupby(["family", "task", "scenario", "model"])["seed"].nunique().to_dict()
    rows = []
    for family, family_label, *_ in FAMILIES:
        for task, task_label, *_rest in TASKS:
            for scenario, scenario_label, *_srest in SCENARIOS:
                for model, model_label, *_mrest in MODELS:
                    n = int(completed.get((family, task, scenario, model), 0))
                    rows.append(
                        {
                            "family": family,
                            "family_label": family_label,
                            "task": task,
                            "task_label": task_label,
                            "scenario": scenario,
                            "scenario_label": scenario_label,
                            "model": model,
                            "model_label": model_label,
                            "expected": 100,
                            "completed": min(n, 100),
                            "pending": max(0, 100 - min(n, 100)),
                            "unique_valid_seed_files": n,
                        }
                    )
    completion = pd.DataFrame(rows)
    completion.to_csv(tables / "completion_by_family_task_scenario_model.csv", index=False)
    completion.groupby(["family", "family_label"], as_index=False).agg(
        expected=("expected", "sum"), completed=("completed", "sum"), pending=("pending", "sum")
    ).to_csv(tables / "completion_by_family.csv", index=False)
    completion.groupby(["family", "family_label", "model", "model_label"], as_index=False).agg(
        expected=("expected", "sum"), completed=("completed", "sum"), pending=("pending", "sum")
    ).to_csv(tables / "completion_by_family_model.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--base-tables-dir", type=Path, required=True)
    parser.add_argument("--since", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--prefix-number", default="199")
    args = parser.parse_args()

    outputs = args.v3_root / "outputs"
    audit_dir = outputs / f"{args.prefix_number}_incremental_metric_consolidation_for_boxplots_{args.timestamp}"
    set99_dir = outputs / f"{int(args.prefix_number) + 1}_incremental_99_boxplots_by_scheme_with_r2clip_{args.timestamp}"
    set100_dir = outputs / f"{int(args.prefix_number) + 2}_incremental_100_boxplots_families_by_scheme_with_r2clip_{args.timestamp}"
    tables = audit_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    base_wide_path = args.base_tables_dir / "seed_metrics_wide.csv"
    if not base_wide_path.exists():
        raise FileNotFoundError(base_wide_path)
    base = normalize_base_wide(pd.read_csv(base_wide_path))
    recent = read_recent_metrics(args.v3_root, args.since)
    all_wide = pd.concat([base, recent], ignore_index=True, sort=False)
    all_wide["source_mtime"] = pd.to_numeric(all_wide["source_mtime"], errors="coerce").fillna(0)
    all_wide["source_size"] = pd.to_numeric(all_wide["source_size"], errors="coerce").fillna(0)
    all_wide["_complete_rank"] = pd.to_numeric(all_wide.get("_complete_rank", 0), errors="coerce").fillna(0)
    all_wide = (
        all_wide.sort_values(["_complete_rank", "source_mtime", "source_size"], ascending=[False, False, False])
        .drop_duplicates(["family", "task", "scenario", "model", "seed"], keep="first")
        .drop(columns=["_complete_rank"], errors="ignore")
        .sort_values(["family", "task", "scenario", "model", "seed"])
    )
    all_wide.to_csv(tables / "seed_metrics_wide.csv", index=False)
    recent.to_csv(tables / "recent_seed_metrics_wide.csv", index=False)
    long = wide_to_long(all_wide)
    long.to_csv(tables / "seed_metrics_long_for_boxplots.csv", index=False)
    write_completion(all_wide, tables)

    progress(f"incremental: base rows={len(base)} recent rows={len(recent)} merged rows={len(all_wide)} metric rows={len(long)}")
    n99 = make_set99(long, set99_dir)
    n100 = make_set100(long, set100_dir)
    archive = outputs / f"{set99_dir.name}__{set100_dir.name}.tar.gz"
    make_archive([audit_dir, set99_dir, set100_dir], archive)
    metadata = {
        "created_at": datetime.now().isoformat(),
        "v3_root": str(args.v3_root),
        "base_tables_dir": str(args.base_tables_dir),
        "since": args.since,
        "audit_dir": str(audit_dir),
        "set99_dir": str(set99_dir),
        "set100_dir": str(set100_dir),
        "archive": str(archive),
        "base_seed_rows": int(len(base)),
        "recent_seed_rows": int(len(recent)),
        "merged_seed_rows": int(len(all_wide)),
        "metric_rows": int(len(long)),
        "set99_plots": int(n99),
        "set100_plots": int(n100),
    }
    (audit_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
