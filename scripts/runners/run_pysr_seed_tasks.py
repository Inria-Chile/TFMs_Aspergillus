#!/usr/bin/env python3
"""Run PySR folds and export v3 seed-level outputs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
V2 = REPO
V3 = REPO

TASK_LABEL = {
    "occurrence": "Clasificacion",
    "conditional_abundance": "Regresion 18 positivas",
    "all_sample_abundance": "Regresion 50 muestras",
}
TASK_DIR = {
    "occurrence": "classification",
    "conditional_abundance": "regression_18_positive",
    "all_sample_abundance": "regression_50_all_samples",
}
EXPECTED_FOLDS = {"occurrence": 15, "conditional_abundance": 18, "all_sample_abundance": 15}
SCENARIO_LABEL = {"A_environment": "A", "B_microbiome": "B", "C_environment_microbiome": "C"}
METRIC_COLS = ["AUC", "F1", "MAE", "R2", "RMSE", "balanced_accuracy", "prediction_mean", "complexity"]

sys.path.insert(0, str(REPO / "scripts/legacy_v2"))
sys.path.insert(0, str(REPO / "src"))

from marta_omar_repro.data import load_table  # noqa: E402
from run_omar_repro import load_config  # noqa: E402
from run_pysr_marta_chunk import load_done, run_one  # noqa: E402


def csv_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [x.strip() for x in value.split(",") if x.strip()]


def pysr_args(args: argparse.Namespace, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=str(output_dir),
        tmp_root=args.tmp_root,
        pysr_populations=args.pysr_populations,
        pysr_population_size=args.pysr_population_size,
        pysr_iterations=args.pysr_iterations,
        pysr_timeout_seconds=args.pysr_timeout_seconds,
        pysr_maxsize=args.pysr_maxsize,
        pysr_parallelism=args.pysr_parallelism,
        pysr_procs=args.pysr_procs,
        pysr_elementwise_loss=args.pysr_elementwise_loss,
        pysr_occurrence_elementwise_loss=args.pysr_occurrence_elementwise_loss,
        pysr_regression_elementwise_loss=args.pysr_regression_elementwise_loss,
        pysr_occurrence_target=args.pysr_occurrence_target,
        pysr_occurrence_score_transform=args.pysr_occurrence_score_transform,
    )


def seed_paths(args: argparse.Namespace, scenario: str, seed: int) -> tuple[Path, Path, Path, Path]:
    root = Path(args.output_root) if args.output_root else V3 / "results" / args.family
    base = root / TASK_DIR[args.task_kind] / scenario / "pysr"
    return (
        base / "seed_metrics" / f"seed_{seed:03d}_metrics.csv",
        base / "seed_predictions" / f"seed_{seed:03d}_predictions.csv",
        base / "seed_equations" / f"seed_{seed:03d}_equations.csv",
        base / "seed_variable_usage" / f"seed_{seed:03d}_variable_usage.csv",
    )


def existing_seed_done(args: argparse.Namespace, scenario: str, seed: int) -> bool:
    metric_csv, _, _, _ = seed_paths(args, scenario, seed)
    if not metric_csv.exists() or metric_csv.stat().st_size == 0:
        return False
    try:
        df = pd.read_csv(metric_csv)
    except Exception:
        return False
    if len(df) != 1:
        return False
    expected = EXPECTED_FOLDS[args.task_kind]
    if {"completed_folds", "expected_folds", "seed_complete_all_folds"}.issubset(df.columns):
        try:
            return (
                int(df["completed_folds"].iloc[0]) == expected
                and int(df["expected_folds"].iloc[0]) == expected
                and str(df["seed_complete_all_folds"].iloc[0]).lower() in {"true", "1"}
            )
        except Exception:
            return False
    if "n_folds_tested" in df.columns:
        try:
            return int(df["n_folds_tested"].iloc[0]) == expected
        except Exception:
            return False
    return False


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def feature_used(feature: str, equation: str) -> bool:
    if not equation:
        return False
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(str(feature))}(?![A-Za-z0-9_])", str(equation)) is not None


def aggregate_seed(records: list[dict[str, Any]], args: argparse.Namespace, scenario: str, seed: int, source_jsonl: Path) -> dict[str, str]:
    expected = EXPECTED_FOLDS[args.task_kind]
    ok = [r for r in records if r.get("status") == "ok"]
    if len(ok) != expected:
        raise RuntimeError(f"{args.family} {args.task_kind} pysr {scenario} seed={seed}: {len(ok)}/{expected} ok folds")

    metric_csv, pred_csv, eq_csv, usage_csv = seed_paths(args, scenario, seed)
    metric_csv.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "family": args.family,
        "family_label": args.family_label,
        "task": args.task_kind,
        "task_label": TASK_LABEL[args.task_kind],
        "scenario": scenario,
        "scenario_label": SCENARIO_LABEL.get(scenario, scenario),
        "model": "PySR",
        "seed": int(seed),
        "n_folds_tested": expected,
        "completed_folds": expected,
        "expected_folds": expected,
        "seed_complete_all_folds": True,
        "source_table": str(source_jsonl),
    }
    for col in METRIC_COLS:
        vals = pd.to_numeric(pd.Series([r.get(col) for r in ok]), errors="coerce")
        row[col] = vals.mean(skipna=True)
    pd.DataFrame([row]).to_csv(metric_csv, index=False)

    pred_frames = []
    eq_rows = []
    usage_rows = []
    for rec in ok:
        pred_path = Path(str(rec.get("predictions_csv", "")))
        if pred_path.exists():
            pred_frames.append(pd.read_csv(pred_path))
        features = []
        try:
            features = json.loads(str(rec.get("selected_features_json", "[]")))
        except Exception:
            features = []
        equation = str(rec.get("equation", "") or "")
        eq_rows.append(
            {
                "family": args.family,
                "task": args.task_kind,
                "scenario": scenario,
                "model": "PySR",
                "seed": int(seed),
                "fold_id": int(rec.get("fold_id")),
                "equation": equation,
                "equation_aliased": rec.get("equation_aliased"),
                "complexity": rec.get("complexity"),
                "n_features": rec.get("n_features"),
                "elapsed_sec": rec.get("elapsed_sec"),
            }
        )
        for feature in features:
            usage_rows.append(
                {
                    "family": args.family,
                    "task": args.task_kind,
                    "scenario": scenario,
                    "model": "PySR",
                    "seed": int(seed),
                    "fold_id": int(rec.get("fold_id")),
                    "feature": feature,
                    "used_in_equation": feature_used(str(feature), equation),
                }
            )

    if pred_frames:
        pred_csv.parent.mkdir(parents=True, exist_ok=True)
        preds = pd.concat(pred_frames, ignore_index=True, sort=False)
        preds.insert(0, "family", args.family)
        preds.to_csv(pred_csv, index=False)
    eq_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(eq_rows).to_csv(eq_csv, index=False)
    if usage_rows:
        usage_csv.parent.mkdir(parents=True, exist_ok=True)
        usage = pd.DataFrame(usage_rows)
        out = (
            usage.groupby(["family", "task", "scenario", "model", "seed", "feature"], as_index=False)
            .agg(folds_available=("used_in_equation", "size"), folds_used=("used_in_equation", "sum"))
            .assign(mean_usage_frequency=lambda d: d["folds_used"] / d["folds_available"])
            .sort_values(["mean_usage_frequency", "folds_used", "feature"], ascending=[False, False, True])
        )
        out.to_csv(usage_csv, index=False)
    return {"metric_csv": str(metric_csv), "prediction_csv": str(pred_csv), "equations_csv": str(eq_csv), "variable_usage_csv": str(usage_csv)}


def run_seed(df: pd.DataFrame, cfg: dict, tasks: pd.DataFrame, args: argparse.Namespace, scenario: str, seed: int) -> dict[str, Any]:
    if existing_seed_done(args, scenario, seed):
        return {"status": "skipped_existing", "scenario": scenario, "seed": seed}
    expected = EXPECTED_FOLDS[args.task_kind]
    rows = tasks[
        tasks["task_kind"].eq(args.task_kind)
        & tasks["scenario"].eq(scenario)
        & tasks["seed"].astype(int).eq(seed)
    ].sort_values("fold_id")
    if len(rows) != expected:
        return {"status": "missing_task_rows", "scenario": scenario, "seed": seed, "rows": len(rows), "expected": expected}

    t0 = time.time()
    run_dir = Path(args.work_dir) / args.run_tag / args.family / args.task_kind / "pysr" / scenario / f"seed_{seed:03d}"
    partial_dir = run_dir / "partials"
    source_jsonl = run_dir / "monitoring" / f"pysr_{scenario}_seed_{seed:03d}.jsonl"
    done = load_done(source_jsonl) if args.resume else set()
    run_args = pysr_args(args, partial_dir)
    for _, task_row in rows.iterrows():
        task_id = str(task_row["task_id"])
        if task_id in done:
            continue
        rec = run_one(df, task_row, cfg, run_args)
        append_jsonl(source_jsonl, rec)
        print(json.dumps({"event": "fold_done", "task": args.task_kind, "scenario": scenario, "seed": seed, "fold_id": int(task_row["fold_id"]), "status": rec.get("status"), "complexity": rec.get("complexity"), "error": str(rec.get("error_message", ""))[:180]}, ensure_ascii=False), flush=True)

    records: list[dict[str, Any]] = []
    if source_jsonl.exists():
        with source_jsonl.open(errors="replace") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    ok_count = sum(1 for r in records if r.get("status") == "ok")
    if ok_count < expected:
        return {"status": "incomplete", "scenario": scenario, "seed": seed, "ok_folds": ok_count, "expected_folds": expected, "source_jsonl": str(source_jsonl)}
    outputs = aggregate_seed(records[-expected:], args, scenario, seed, source_jsonl)
    if args.cleanup_partials:
        shutil.rmtree(partial_dir, ignore_errors=True)
    return {"status": "ok", "scenario": scenario, "seed": seed, "ok_folds": expected, "elapsed_sec": round(time.time() - t0, 3), **outputs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--family-label", required=True)
    ap.add_argument("--task-kind", choices=sorted(TASK_DIR), required=True)
    ap.add_argument("--tasks-csv", required=True)
    ap.add_argument("--config", default=str(REPO / "configs/v3_legacy/omar_repro_config.json"))
    ap.add_argument("--scenarios", default="A_environment,B_microbiome,C_environment_microbiome")
    ap.add_argument("--seeds", default=",".join(str(x) for x in range(123, 223)))
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit-seeds", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cleanup-partials", action="store_true", default=True)
    ap.add_argument("--work-dir", default=str(REPO / "results/monitoring/v3_pysr_seed_tasks"))
    ap.add_argument("--output-root", default="")
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--tmp-root", default="/dev/shm")
    ap.add_argument("--pysr-populations", type=int, default=8)
    ap.add_argument("--pysr-population-size", type=int, default=5000)
    ap.add_argument("--pysr-iterations", type=int, default=20)
    ap.add_argument("--pysr-timeout-seconds", type=int, default=120)
    ap.add_argument("--pysr-maxsize", type=int, default=10)
    ap.add_argument("--pysr-parallelism", choices=["serial", "multithreading", "multiprocessing"], default="multiprocessing")
    ap.add_argument("--pysr-procs", type=int, default=4)
    ap.add_argument("--pysr-elementwise-loss", default="")
    ap.add_argument("--pysr-occurrence-elementwise-loss", default="SigmoidLoss()")
    ap.add_argument("--pysr-regression-elementwise-loss", default="L1DistLoss()")
    ap.add_argument("--pysr-occurrence-target", choices=["binary", "margin"], default="margin")
    ap.add_argument("--pysr-occurrence-score-transform", choices=["clip01", "sigmoid"], default="sigmoid")
    args = ap.parse_args()

    args.run_tag = args.run_tag or f"{args.family}_{args.task_kind}_pysr_v3_seed_{time.strftime('%Y%m%d_%H%M%S')}"
    scenarios = csv_list(args.scenarios, list(SCENARIO_LABEL))
    seeds = [int(x) for x in csv_list(args.seeds, [str(x) for x in range(123, 223)])]
    cfg = load_config(Path(args.config))
    df = load_table((Path(args.config).resolve().parents[1] / cfg["input_csv"]).resolve(), cfg["target_col"], cfg["id_col"])
    tasks = pd.read_csv(args.tasks_csv)
    combos_df = (
        tasks[tasks["task_kind"].eq(args.task_kind) & tasks["scenario"].isin(scenarios) & tasks["seed"].astype(int).isin(seeds)][["scenario", "seed"]]
        .drop_duplicates()
        .assign(seed=lambda d: d["seed"].astype(int))
        .sort_values(["scenario", "seed"])
    )
    combos = [(r.scenario, int(r.seed)) for r in combos_df.itertuples(index=False) if not existing_seed_done(args, r.scenario, int(r.seed))]
    combos = combos[args.shard_index :: args.num_shards]
    if args.limit_seeds:
        combos = combos[: args.limit_seeds]
    print(json.dumps({"event": "start", "family": args.family, "task_kind": args.task_kind, "model": "pysr", "run_tag": args.run_tag, "n_combos": len(combos), "combos_head": combos[:20], "pysr_procs": args.pysr_procs, "tasks_csv": args.tasks_csv}, ensure_ascii=False, indent=2), flush=True)

    status = 0
    summary = Path(args.work_dir) / args.run_tag / args.family / args.task_kind / "pysr" / f"summary_shard_{args.shard_index}_of_{args.num_shards}.jsonl"
    for scenario, seed in combos:
        result = run_seed(df, cfg, tasks, args, scenario, seed)
        append_jsonl(summary, result)
        print(json.dumps({"event": "seed_done", **result}, ensure_ascii=False), flush=True)
        if result.get("status") not in {"ok", "skipped_existing"}:
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
