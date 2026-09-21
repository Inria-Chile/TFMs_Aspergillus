#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from tfms_aspergillus.v3_workflow.data import load_table, write_json
from tfms_aspergillus.v3_workflow.environmental_preprocessing import preprocess_train_test_environmental
from tfms_aspergillus.v3_workflow.metrics import safe_auc
from tfms_aspergillus.v3_workflow.models import impute_train_test, rf_classifier, rf_regressor, tabpfn_classifier, tabpfn_regressor
from tfms_aspergillus.v3_workflow.shap_utils import (
    ShapConfig,
    finite,
    kernel_shap,
    plot_shap_figures,
    shap_rows,
    summarize_shap,
    tree_shap,
    xgboost_pred_contribs,
)
from build_tabiclv2_tasks import (
    build_abundance_15a,
    build_abundance_16a,
    build_occurrence_15a,
    build_occurrence_16a,
)
from run_validated_v3_workflow import load_config, parse_seeds
from run_tabiclv2_seed_chunk import make_model as make_tabicl_model
from run_tabiclv2_seed_chunk import resolve_device as resolve_tabicl_device
from run_xgboost_seed_tasks import make_classifier as make_xgb_classifier
from run_xgboost_seed_tasks import make_regressor as make_xgb_regressor


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def loads_list(value: str) -> list[Any]:
    out = json.loads(value)
    if not isinstance(out, list):
        raise ValueError("Expected JSON list")
    return out


def safe_name(value: Any, max_len: int = 180) -> str:
    out = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value)).strip("_")
    return (out[:max_len] or "task")


def build_tasks(df: pd.DataFrame, cfg: dict, seeds: list[int], model_name: str, analyses: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if "15A_occurrence" in analyses:
        rows.extend(build_occurrence_15a(df, cfg, seeds, include_full=False))
    if "15A_abundance" in analyses:
        rows.extend(build_abundance_15a(df, cfg, seeds, include_full=False))
    if "16A_occurrence" in analyses:
        rows.extend(build_occurrence_16a(df, cfg, seeds))
    if "16A_abundance" in analyses:
        rows.extend(build_abundance_16a(df, cfg, seeds))
    tasks = pd.DataFrame(rows)
    if tasks.empty:
        return tasks
    label = {
        "random_forest": "Random forest",
        "xgboost": "XGBoost_RF_like",
        "tabpfn": "TabPFN",
        "tabiclv2": "TabICLv2",
    }[model_name]
    tasks["model"] = tasks["variant"].map(lambda v: f"{label}_{v}" if v in {"anova", "rfbest"} else label)
    tasks["task_id"] = tasks["task_id"].map(lambda x: safe_name(f"{model_name}__{x}"))
    return tasks


def keep_one_fold_per_task_scheme(tasks: pd.DataFrame) -> pd.DataFrame:
    if tasks.empty:
        return tasks
    group_cols = ["analysis", "task_kind", "scenario", "variant"]
    return (
        tasks.sort_values(["seed", "fold_id", "task_id"])
        .groupby(group_cols, as_index=False, dropna=False)
        .head(1)
        .reset_index(drop=True)
    )


def make_xgb_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        n_estimators=args.xgb_n_estimators,
        learning_rate=args.xgb_learning_rate,
        max_depth=args.xgb_max_depth,
        min_child_weight=args.xgb_min_child_weight,
        subsample=args.xgb_subsample,
        reg_alpha=args.xgb_reg_alpha,
        reg_lambda=args.xgb_reg_lambda,
        max_bin=args.xgb_max_bin,
        n_jobs=args.xgb_n_jobs,
        verbosity=0,
    )


def parse_shap_nsamples(value: str) -> int | str:
    if value == "auto":
        return value
    return int(value)


def make_model(args: argparse.Namespace, task_kind: str, seed: int, n_features: int, y_train: np.ndarray):
    if args.model == "random_forest":
        if task_kind == "occurrence":
            return rf_classifier(seed, args.rf_n_estimators, args.model_n_jobs)
        return rf_regressor(seed, args.rf_n_estimators, args.model_n_jobs)
    if args.model == "tabpfn":
        return tabpfn_classifier(args.device, seed) if task_kind == "occurrence" else tabpfn_regressor(args.device, seed)
    if args.model == "tabiclv2":
        device, use_amp, _ = resolve_tabicl_device(args.device)
        tabicl_args = argparse.Namespace(
            n_estimators=args.tabicl_n_estimators,
            batch_size=args.tabicl_batch_size,
            classifier_checkpoint_version=args.tabicl_classifier_checkpoint_version,
            regressor_checkpoint_version=args.tabicl_regressor_checkpoint_version,
        )
        return make_tabicl_model(tabicl_args, seed, device, use_amp, task_kind)
    if args.model == "xgboost":
        xgb_args = make_xgb_args(args)
        device = "cuda" if args.device.startswith("cuda") else "cpu"
        if task_kind == "occurrence":
            return make_xgb_classifier(xgb_args, seed, n_features, y_train, device)
        return make_xgb_regressor(xgb_args, seed, n_features, device)
    raise ValueError(f"Unknown model: {args.model}")


def fit_model(model: Any, args: argparse.Namespace, X_train: np.ndarray, y_train: np.ndarray) -> None:
    if args.model == "tabiclv2":
        kv_cache = False if args.tabicl_kv_cache == "false" else args.tabicl_kv_cache
        model.fit(X_train, y_train, kv_cache=kv_cache)
    else:
        model.fit(X_train, y_train)


def explain_model(
    model: Any,
    args: argparse.Namespace,
    X_train: np.ndarray,
    X_test: np.ndarray,
    features: list[str],
    task_kind: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if args.model == "random_forest":
        return tree_shap(model, X_test, task_kind)
    if args.model == "xgboost":
        return xgboost_pred_contribs(model, X_test, features)
    shap_cfg = ShapConfig(
        background_size=args.shap_background_size,
        nsamples=parse_shap_nsamples(args.shap_nsamples),
        random_state=seed,
    )
    if task_kind == "occurrence":
        predict_fn = lambda X: np.clip(np.asarray(model.predict_proba(X))[:, 1].astype(float), 0.0, 1.0)
    else:
        predict_fn = lambda X: np.clip(np.expm1(np.asarray(model.predict(X)).ravel().astype(float)), 0.0, 1.0)
    return kernel_shap(predict_fn, X_train, X_test, features, task_kind, shap_cfg)


def read_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("status") == "ok" and row.get("task_id"):
                done.add(str(row["task_id"]))
    return done


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def run_one(df: pd.DataFrame, cfg: dict, row: pd.Series, args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    t0 = time.time()
    task_id = str(row["task_id"])
    rec = row.to_dict()
    rec.update({"status": "failed", "error_message": "", "shap_enabled": True})
    try:
        features = [str(x) for x in loads_list(row["features_json"])]
        train_rows = [int(x) for x in loads_list(row["train_rows_json"])]
        test_rows = [int(x) for x in loads_list(row["test_rows_json"])]
        X_train_df = df.iloc[train_rows][features].copy()
        X_test_df = df.iloc[test_rows][features].copy()
        X_train_df, X_test_df, env_preprocessing_info = preprocess_train_test_environmental(X_train_df, X_test_df, df, cfg)
        rec["environmental_preprocessing_json"] = dumps(env_preprocessing_info)
        X_train, X_test = impute_train_test(X_train_df, X_test_df)
        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)
        seed = int(row["seed"])
        if args.model == "tabpfn":
            rec["tabpfn_random_state"] = seed
            rec["tabpfn_seeded_replicate"] = True

        if row["task_kind"] == "occurrence":
            y_train = df.iloc[train_rows]["Aspergillus_presence"].astype(int).to_numpy()
            y_test = df.iloc[test_rows]["Aspergillus_presence"].astype(int).to_numpy()
            model = make_model(args, "occurrence", seed, len(features), y_train)
            fit_model(model, args, X_train, y_train)
            pred = np.clip(np.asarray(model.predict_proba(X_test))[:, 1].astype(float), 0.0, 1.0)
            y_pred = (pred >= 0.5).astype(int)
            metrics = {
                "AUC": finite(safe_auc(y_test, pred)),
                "balanced_accuracy": finite(balanced_accuracy_score(y_test, y_pred)),
                "F1": finite(f1_score(y_test, y_pred, zero_division=0)),
                "prediction_mean": finite(np.nanmean(pred)),
            }
            pred_df = pd.DataFrame(
                {
                    "row_index": test_rows,
                    "Sample": df.iloc[test_rows][cfg["id_col"]].to_numpy(),
                    "y_true": y_test,
                    "y_score": pred,
                    "y_pred": y_pred,
                }
            )
            shap_y = y_test.astype(float)
            pred_col = "y_score"
        else:
            y_train = np.log1p(df.iloc[train_rows][cfg["target_col"]].astype(float).to_numpy()).astype(np.float32)
            y_test_raw = df.iloc[test_rows][cfg["target_col"]].astype(float).to_numpy()
            model = make_model(args, str(row["task_kind"]), seed, len(features), y_train)
            fit_model(model, args, X_train, y_train)
            pred_log = np.asarray(model.predict(X_test)).ravel().astype(float)
            pred = np.clip(np.expm1(pred_log), 0.0, 1.0)
            metrics = {
                "RMSE": finite(mean_squared_error(y_test_raw, pred) ** 0.5),
                "MAE": finite(mean_absolute_error(y_test_raw, pred)),
                "R2": finite(r2_score(y_test_raw, pred)) if len(y_test_raw) > 1 else None,
                "prediction_mean": finite(np.nanmean(pred)),
            }
            pred_df = pd.DataFrame(
                {
                    "row_index": test_rows,
                    "Sample": df.iloc[test_rows][cfg["id_col"]].to_numpy(),
                    "y_true_raw": y_test_raw,
                    "y_pred_raw": pred,
                    "y_true_log1p": np.log1p(y_test_raw),
                    "y_pred_log1p": pred_log,
                }
            )
            shap_y = y_test_raw.astype(float)
            pred_col = "y_pred_raw"

        shap_values, base_values = explain_model(model, args, X_train, X_test, features, str(row["task_kind"]), seed)
        partial_dir = run_dir / "partials" / safe_name(task_id)
        partial_dir.mkdir(parents=True, exist_ok=True)
        pred_df["task_id"] = task_id
        pred_df["analysis"] = row["analysis"]
        pred_df["task_kind"] = row["task_kind"]
        pred_df["scenario"] = row["scenario"]
        pred_df["variant"] = row["variant"]
        pred_df["model"] = row["model"]
        pred_df["seed"] = seed
        pred_df["fold_id"] = int(row["fold_id"])
        pred_path = partial_dir / "predictions.csv"
        shap_path = partial_dir / "shap_values.csv"
        pred_df.to_csv(pred_path, index=False)
        pd.DataFrame(
            shap_rows(
                task_id=task_id,
                analysis=str(row["analysis"]),
                task_kind=str(row["task_kind"]),
                scenario=str(row["scenario"]),
                variant=str(row["variant"]),
                model_name=str(row["model"]),
                seed=seed,
                fold_id=int(row["fold_id"]),
                row_indices=test_rows,
                sample_ids=list(df.iloc[test_rows][cfg["id_col"]].to_numpy()),
                feature_names=features,
                X_test_df=X_test_df,
                shap_values=shap_values,
                base_values=base_values,
                y_true=shap_y,
                prediction=pred,
                prediction_col=pred_col,
            )
        ).to_csv(shap_path, index=False)
        summary = {
            "task_id": task_id,
            "status": "ok",
            "metrics": metrics,
            "features": features,
            "n_features": len(features),
            "n_train": len(train_rows),
            "n_test": len(test_rows),
            "predictions_csv": str(pred_path),
            "shap_csv": str(shap_path),
            "elapsed_sec": round(time.time() - t0, 3),
            "environmental_preprocessing": env_preprocessing_info,
        }
        summary_path = partial_dir / "summary.json"
        write_json(summary_path, summary)
        rec.update(metrics)
        rec.update(
            {
                "status": "ok",
                "predictions_csv": str(pred_path),
                "shap_csv": str(shap_path),
                "summary_json": str(summary_path),
                "selected_features_json": dumps(features),
                "n_features": len(features),
            }
        )
    except Exception as exc:
        rec["error_message"] = f"{type(exc).__name__}: {exc}"
    rec["elapsed_sec"] = round(time.time() - t0, 3)
    return rec


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def consolidate(run_dir: Path) -> None:
    out = run_dir / "consolidated"
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for p in sorted((run_dir / "monitoring").glob("*.jsonl")):
        records.extend(read_jsonl(p))
    rec = pd.DataFrame(records)
    rec.to_csv(out / "shap_task_records.csv", index=False)
    if rec.empty or "status" not in rec:
        return
    ok = rec[rec["status"] == "ok"].drop_duplicates("task_id", keep="last").copy()
    ok.to_csv(out / "shap_task_records_unique_ok.csv", index=False)
    pred_frames = [pd.read_csv(p) for p in ok.get("predictions_csv", pd.Series(dtype=str)).dropna() if Path(p).exists()]
    shap_frames = [pd.read_csv(p) for p in ok.get("shap_csv", pd.Series(dtype=str)).dropna() if Path(p).exists()]
    preds = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    shap_values = pd.concat(shap_frames, ignore_index=True) if shap_frames else pd.DataFrame()
    preds.to_csv(out / "predictions_all_completed.csv", index=False)
    shap_values.to_csv(out / "shap_values_all.csv", index=False)
    summarize_shap(shap_values).to_csv(out / "shap_importance_summary.csv", index=False)
    plot_manifest = plot_shap_figures(shap_values, out / "plots", top_n=30)
    plot_manifest.to_csv(out / "shap_plot_manifest.csv", index=False)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "task_records": int(len(rec)),
                "unique_ok_tasks": int(len(ok)),
                "predictions": int(len(preds)),
                "shap_rows": int(len(shap_values)),
                "consolidated_dir": str(out),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "v3_repro_config.json"))
    ap.add_argument("--model", choices=["random_forest", "xgboost", "tabpfn", "tabiclv2"], required=True)
    ap.add_argument("--output-dir", default=str(ROOT / "outputs" / "33_model_shap"))
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--tasks-csv", default="")
    ap.add_argument("--seeds", default=",".join(map(str, range(123, 223))))
    ap.add_argument("--analyses", nargs="+", default=["15A_occurrence", "15A_abundance", "16A_occurrence", "16A_abundance"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--build-tasks-only", action="store_true")
    ap.add_argument("--consolidate-only", action="store_true")
    ap.add_argument("--onefold-smoke", action="store_true")
    ap.add_argument("--shap-background-size", type=int, default=20)
    ap.add_argument("--shap-nsamples", default="auto")
    ap.add_argument("--rf-n-estimators", type=int, default=500)
    ap.add_argument("--model-n-jobs", type=int, default=1)
    ap.add_argument("--tabicl-n-estimators", type=int, default=6)
    ap.add_argument("--tabicl-batch-size", type=int, default=2)
    ap.add_argument("--tabicl-kv-cache", default="repr", choices=["repr", "kv", "false"])
    ap.add_argument("--tabicl-regressor-checkpoint-version", default="tabicl-regressor-v2-20260212.ckpt")
    ap.add_argument("--tabicl-classifier-checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    ap.add_argument("--xgb-n-estimators", type=int, default=500)
    ap.add_argument("--xgb-learning-rate", type=float, default=0.03)
    ap.add_argument("--xgb-max-depth", type=int, default=3)
    ap.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    ap.add_argument("--xgb-subsample", type=float, default=0.632)
    ap.add_argument("--xgb-reg-alpha", type=float, default=0.0)
    ap.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    ap.add_argument("--xgb-max-bin", type=int, default=256)
    ap.add_argument("--xgb-n-jobs", type=int, default=1)
    args = ap.parse_args()

    run_tag = args.run_tag or f"{args.model}_shap_100seeds_v1"
    run_dir = Path(args.output_dir) / run_tag
    if args.consolidate_only:
        consolidate(run_dir)
        return 0

    cfg = load_config(Path(args.config))
    base = Path(args.config).resolve().parents[1]
    df = load_table((base / cfg["input_csv"]).resolve(), cfg["target_col"], cfg["id_col"])
    seeds = parse_seeds(args.seeds, None, int(cfg["random_state"]))
    if args.tasks_csv:
        tasks = pd.read_csv(args.tasks_csv)
    else:
        tasks = build_tasks(df, cfg, seeds, args.model, args.analyses)
    if args.onefold_smoke:
        tasks = keep_one_fold_per_task_scheme(tasks)
    task_path = run_dir / "monitoring" / "tasks.csv"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(task_path, index=False)
    if args.build_tasks_only:
        print(json.dumps({"tasks_csv": str(task_path), "n_tasks": int(len(tasks))}, ensure_ascii=False, indent=2))
        return 0

    shard = tasks.iloc[args.shard_index :: args.num_shards].copy()
    if args.limit > 0:
        shard = shard.head(args.limit)
    out_jsonl = run_dir / "monitoring" / f"{args.model}_shap_shard_{args.shard_index}_of_{args.num_shards}.jsonl"
    done = read_done(out_jsonl) if args.resume else set()
    completed = 0
    for _, row in shard.iterrows():
        task_id = str(row["task_id"])
        if task_id in done:
            continue
        rec = run_one(df, cfg, row, args, run_dir)
        append_jsonl(out_jsonl, rec)
        completed += 1
        print(
            json.dumps(
                {
                    "completed": completed,
                    "task_id": task_id,
                    "status": rec.get("status"),
                    "model": args.model,
                    "task_kind": rec.get("task_kind"),
                    "scenario": rec.get("scenario"),
                    "seed": rec.get("seed"),
                    "AUC": rec.get("AUC"),
                    "RMSE": rec.get("RMSE"),
                    "elapsed_sec": rec.get("elapsed_sec"),
                    "error": str(rec.get("error_message", ""))[:200],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
