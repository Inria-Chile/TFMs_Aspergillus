#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from marta_omar_repro.data import load_table, write_json
from marta_omar_repro.environmental_preprocessing import preprocess_train_test_environmental
from marta_omar_repro.metrics import safe_auc
from run_omar_repro import load_config


def finite(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def safe_name(value: Any, max_len: int = 180) -> str:
    name = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value)).strip("_")
    return (name[:max_len] or "task")


def json_list(value: str) -> list[Any]:
    out = json.loads(value)
    if not isinstance(out, list):
        raise ValueError("Expected JSON list")
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def load_done(out_jsonl: Path) -> set[str]:
    if not out_jsonl.exists():
        return set()
    done: set[str] = set()
    with out_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("status") == "ok":
                done.add(str(rec.get("task_id")))
    return done


def standardize_pysr_variable_name(name: str) -> str:
    standardized = re.sub(r"\W+", "_", str(name)).strip("_")
    if not standardized:
        standardized = "var"
    if not re.match(r"^[A-Za-z_]", standardized):
        standardized = f"var_{standardized}"
    return standardized


def build_pysr_alias_map(predictors: list[str]) -> dict[str, str]:
    reserved = {
        "NaN",
        "Inf",
        "pi",
        "nothing",
        "true",
        "false",
        "exp",
        "log",
    }
    alias_map: dict[str, str] = {}
    used: set[str] = set()
    for predictor in predictors:
        base = standardize_pysr_variable_name(predictor)
        if base in reserved:
            base = f"var_{base}"
        alias = base
        suffix = 2
        while alias in used or alias in reserved:
            alias = f"{base}_{suffix}"
            suffix += 1
        alias_map[predictor] = alias
        used.add(alias)
    return alias_map


def restore_equation_names(equation: str, alias_map: dict[str, str]) -> str:
    restored = equation
    for predictor, alias in sorted(alias_map.items(), key=lambda item: len(item[1]), reverse=True):
        restored = re.sub(rf"\b{re.escape(alias)}\b", predictor, restored)
    return restored


def prepare_xy(df: pd.DataFrame, row: pd.Series, cfg: dict, args: argparse.Namespace):
    features = [str(x) for x in json_list(row["features_json"])]
    train_rows = [int(x) for x in json_list(row["train_rows_json"])]
    test_rows = [int(x) for x in json_list(row["test_rows_json"])]
    X_train_df = df.iloc[train_rows][features].copy()
    X_test_df = df.iloc[test_rows][features].copy()
    X_train_df, X_test_df, env_preprocessing_info = preprocess_train_test_environmental(X_train_df, X_test_df, df, cfg)
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_df).astype(np.float32)
    X_test = imputer.transform(X_test_df).astype(np.float32)
    if row["target_transform"] == "presence_0_1":
        y_binary = df.iloc[train_rows]["Aspergillus_presence"].astype(int).to_numpy(dtype=np.float32)
        if args.pysr_occurrence_target == "margin":
            y_train = (2.0 * y_binary - 1.0).astype(np.float32)
        else:
            y_train = y_binary.astype(np.float32)
        y_test = df.iloc[test_rows]["Aspergillus_presence"].astype(int).to_numpy()
        y_test_metric = y_test.astype(float)
    elif row["target_transform"] in {"log1p_positive_abundance", "log1p_all_abundance"}:
        y_train = np.log1p(df.iloc[train_rows][cfg["target_col"]].astype(float).to_numpy()).astype(np.float32)
        y_test_metric = df.iloc[test_rows][cfg["target_col"]].astype(float).to_numpy()
        y_test = np.log1p(y_test_metric).astype(np.float32)
    else:
        raise ValueError(f"Unknown transform: {row['target_transform']}")
    return X_train, X_test, y_train, y_test, y_test_metric, features, train_rows, test_rows, env_preprocessing_info


def effective_pysr_loss(task_kind: str, args: argparse.Namespace) -> str:
    if args.pysr_elementwise_loss:
        return args.pysr_elementwise_loss
    if task_kind == "occurrence":
        return args.pysr_occurrence_elementwise_loss
    return args.pysr_regression_elementwise_loss


def run_pysr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    features: list[str],
    seed: int,
    elementwise_loss: str,
    args,
):
    from pysr import PySRRegressor

    tmp_root = Path(args.tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)
    worker_dir = tmp_root / f"marta_pysr_{uuid.uuid4().hex[:10]}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    alias_map = build_pysr_alias_map(features)
    variable_names = [alias_map[predictor] for predictor in features]
    try:
        model = PySRRegressor(
            populations=args.pysr_populations,
            population_size=args.pysr_population_size,
            niterations=args.pysr_iterations,
            maxsize=args.pysr_maxsize,
            timeout_in_seconds=args.pysr_timeout_seconds,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp", "log"],
            nested_constraints={
                "exp": {"exp": 0, "log": 0},
                "log": {"exp": 0, "log": 0},
            },
            model_selection="accuracy",
            elementwise_loss=elementwise_loss,
            parallelism=args.pysr_parallelism,
            procs=args.pysr_procs,
            progress=False,
            verbosity=0,
            random_state=seed,
            temp_equation_file=str(worker_dir / "hof.csv"),
            tempdir=str(worker_dir),
        )
        model.fit(X_train, y_train, variable_names=variable_names)
        pred = np.asarray(model.predict(X_test), dtype=float).ravel()
        best = model.get_best()
        equation_raw = best.get("equation", "")
        if isinstance(equation_raw, str):
            equation = equation_raw
        elif pd.isna(equation_raw):
            equation = ""
        else:
            equation = str(equation_raw)
        return {
            "predictions": pred,
            "equation": restore_equation_names(equation, alias_map),
            "equation_aliased": equation,
            "variable_name_map": alias_map,
            "complexity": finite(best.get("complexity")),
        }
    finally:
        shutil.rmtree(worker_dir, ignore_errors=True)


def run_one(df: pd.DataFrame, row: pd.Series, cfg: dict, args: argparse.Namespace):
    t0 = time.time()
    task_id = str(row["task_id"])
    rec = row.to_dict()
    rec.update(
        {
            "status": "failed",
            "error_message": "",
            "pysr_populations": args.pysr_populations,
            "pysr_population_size": args.pysr_population_size,
            "pysr_iterations": args.pysr_iterations,
            "pysr_timeout_seconds": args.pysr_timeout_seconds,
            "pysr_maxsize": args.pysr_maxsize,
            "pysr_parallelism": args.pysr_parallelism,
            "pysr_procs": args.pysr_procs,
            "pysr_elementwise_loss": args.pysr_elementwise_loss,
            "pysr_occurrence_elementwise_loss": args.pysr_occurrence_elementwise_loss,
            "pysr_regression_elementwise_loss": args.pysr_regression_elementwise_loss,
            "pysr_occurrence_target": args.pysr_occurrence_target,
            "pysr_occurrence_score_transform": args.pysr_occurrence_score_transform,
        }
    )
    try:
        seed = int(row["seed"])
        X_train, X_test, y_train, y_test, y_test_metric, features, train_rows, test_rows, env_preprocessing_info = prepare_xy(df, row, cfg, args)
        loss = effective_pysr_loss(str(row["task_kind"]), args)
        rec["pysr_effective_elementwise_loss"] = loss
        rec["environmental_preprocessing_json"] = json.dumps(env_preprocessing_info, ensure_ascii=False)
        result = run_pysr(X_train, y_train, X_test, features, seed, loss, args)

        out_base = Path(args.output_dir)
        partial_dir = out_base / "partials" / safe_name(task_id)
        partial_dir.mkdir(parents=True, exist_ok=True)
        pred_path = partial_dir / "predictions.csv"
        summary_path = partial_dir / "summary.json"

        if row["task_kind"] == "occurrence":
            y_score_raw = np.asarray(result["predictions"], dtype=float)
            y_score_raw_clean = np.nan_to_num(y_score_raw, nan=0.0, posinf=50.0, neginf=-50.0)
            if args.pysr_occurrence_score_transform == "sigmoid":
                y_score = sigmoid(y_score_raw_clean)
            else:
                y_score = np.clip(np.nan_to_num(y_score_raw, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
            y_pred = (y_score >= 0.5).astype(int)
            metrics = {
                "AUC": finite(safe_auc(y_test, y_score)),
                "balanced_accuracy": finite(balanced_accuracy_score(y_test, y_pred)),
                "F1": finite(f1_score(y_test, y_pred, zero_division=0)),
                "prediction_mean": finite(np.nanmean(y_score)),
            }
            pred_df = pd.DataFrame(
                {
                    "row_index": test_rows,
                    "Sample": df.iloc[test_rows][cfg["id_col"]].to_numpy(),
                    "y_true": y_test,
                    "y_score_raw": y_score_raw,
                    "y_score": y_score,
                    "y_pred": y_pred,
                }
            )
        else:
            pred_log = np.nan_to_num(np.asarray(result["predictions"], dtype=float), nan=0.0, posinf=20.0, neginf=-20.0)
            pred_raw = np.clip(np.expm1(pred_log), 0.0, 1.0)
            metrics = {
                "RMSE": finite(mean_squared_error(y_test_metric, pred_raw) ** 0.5),
                "MAE": finite(mean_absolute_error(y_test_metric, pred_raw)),
                "R2": finite(r2_score(y_test_metric, pred_raw)) if len(y_test_metric) > 1 else None,
                "prediction_mean": finite(np.nanmean(pred_raw)),
            }
            pred_df = pd.DataFrame(
                {
                    "row_index": test_rows,
                    "Sample": df.iloc[test_rows][cfg["id_col"]].to_numpy(),
                    "y_true_raw": y_test_metric,
                    "y_pred_raw": pred_raw,
                    "y_true_log1p": y_test,
                    "y_pred_log1p": pred_log,
                }
            )

        pred_df["task_id"] = task_id
        pred_df["analysis"] = row["analysis"]
        pred_df["task_kind"] = row["task_kind"]
        pred_df["scenario"] = row["scenario"]
        pred_df["variant"] = row["variant"]
        pred_df["model"] = row["model"]
        pred_df["seed"] = int(row["seed"])
        pred_df["fold_id"] = int(row["fold_id"])
        pred_df.to_csv(pred_path, index=False)

        summary = {
            "task_id": task_id,
            "status": "ok",
            "metrics": metrics,
            "features": features,
            "variable_name_map": result["variable_name_map"],
            "pysr_effective_elementwise_loss": loss,
            "n_features": len(features),
            "n_train": len(train_rows),
            "n_test": len(test_rows),
            "equation": result["equation"],
            "equation_aliased": result["equation_aliased"],
            "complexity": result["complexity"],
            "predictions_csv": str(pred_path),
            "elapsed_sec": round(time.time() - t0, 3),
        }
        write_json(summary_path, summary)
        rec.update(metrics)
        rec.update(
            {
                "status": "ok",
                "predictions_csv": str(pred_path),
                "summary_json": str(summary_path),
                "selected_features_json": json.dumps(features, ensure_ascii=False),
                "equation": result["equation"],
                "equation_aliased": result["equation_aliased"],
                "complexity": result["complexity"],
            }
        )
    except Exception as exc:
        rec["error_message"] = f"{type(exc).__name__}: {exc}"
    rec["elapsed_sec"] = round(time.time() - t0, 3)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "omar_repro_config.json"))
    ap.add_argument("--tasks-csv", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--output-dir", default=str(ROOT / "outputs" / "21_pysr_marta"))
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--flush-every", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tmp-root", default="/dev/shm")
    ap.add_argument("--pysr-populations", type=int, default=8)
    ap.add_argument("--pysr-population-size", type=int, default=5000)
    ap.add_argument("--pysr-iterations", type=int, default=20)
    ap.add_argument("--pysr-timeout-seconds", type=int, default=120)
    ap.add_argument("--pysr-maxsize", type=int, default=10)
    ap.add_argument("--pysr-parallelism", choices=["serial", "multithreading", "multiprocessing"], default="multiprocessing")
    ap.add_argument("--pysr-procs", type=int, default=8)
    ap.add_argument("--pysr-elementwise-loss", default="", help="Legacy override for all tasks. Empty uses task-specific losses.")
    ap.add_argument("--pysr-occurrence-elementwise-loss", default="SigmoidLoss()")
    ap.add_argument("--pysr-regression-elementwise-loss", default="L1DistLoss()")
    ap.add_argument("--pysr-occurrence-target", choices=["binary", "margin"], default="margin")
    ap.add_argument("--pysr-occurrence-score-transform", choices=["clip01", "sigmoid"], default="sigmoid")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    base = Path(args.config).resolve().parents[1]
    df = load_table((base / cfg["input_csv"]).resolve(), cfg["target_col"], cfg["id_col"])
    tasks = pd.read_csv(args.tasks_csv)
    tasks = tasks.iloc[[i for i in range(len(tasks)) if i % args.num_shards == args.shard_index]].copy()
    if args.limit > 0:
        tasks = tasks.head(args.limit)

    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out) if args.resume else set()
    written = 0
    with out.open("a", encoding="utf-8") as fh:
        for _, row in tasks.iterrows():
            task_id = str(row["task_id"])
            if task_id in done:
                print(json.dumps({"task_id": task_id, "status": "skipped_resume"}, ensure_ascii=False), flush=True)
                continue
            rec = run_one(df, row, cfg, args)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if written % args.flush_every == 0:
                fh.flush()
                os.fsync(fh.fileno())
            print(
                json.dumps(
                    {
                        "task_id": task_id,
                        "status": rec.get("status"),
                        "analysis": rec.get("analysis"),
                        "task_kind": rec.get("task_kind"),
                        "scenario": rec.get("scenario"),
                        "seed": rec.get("seed"),
                        "AUC": rec.get("AUC"),
                        "RMSE": rec.get("RMSE"),
                        "complexity": rec.get("complexity"),
                        "elapsed_sec": rec.get("elapsed_sec"),
                        "error": str(rec.get("error_message", ""))[:180],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
