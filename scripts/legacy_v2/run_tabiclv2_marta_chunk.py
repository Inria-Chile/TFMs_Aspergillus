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
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from marta_omar_repro.data import load_table, write_json
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


def resolve_device(requested: str) -> tuple[str, bool, dict[str, Any]]:
    import torch

    info = {
        "requested_device": requested,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
    }
    if requested == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = requested
    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    info["resolved_device"] = device
    info["use_amp"] = use_amp
    if torch.cuda.is_available():
        info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return device, use_amp, info


def make_model(args: argparse.Namespace, seed: int, device: str, use_amp: bool, task_kind: str):
    if task_kind == "occurrence":
        from tabicl import TabICLClassifier

        return TabICLClassifier(
            n_estimators=args.n_estimators,
            batch_size=args.batch_size,
            checkpoint_version=args.classifier_checkpoint_version,
            allow_auto_download=True,
            device=device,
            use_amp=use_amp,
            use_fa3=False,
            offload_mode="auto",
            random_state=seed,
            verbose=False,
        )

    from tabicl import TabICLRegressor

    return TabICLRegressor(
        n_estimators=args.n_estimators,
        batch_size=args.batch_size,
        checkpoint_version=args.regressor_checkpoint_version,
        allow_auto_download=True,
        device=device,
        use_amp=use_amp,
        use_fa3=False,
        offload_mode="auto",
        random_state=seed,
        verbose=False,
    )


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


def prepare_xy(df: pd.DataFrame, row: pd.Series, cfg: dict):
    features = [str(x) for x in json_list(row["features_json"])]
    train_rows = [int(x) for x in json_list(row["train_rows_json"])]
    test_rows = [int(x) for x in json_list(row["test_rows_json"])]
    X_train_df = df.iloc[train_rows][features].copy()
    X_test_df = df.iloc[test_rows][features].copy()
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_df).astype(np.float32)
    X_test = imputer.transform(X_test_df).astype(np.float32)
    if row["target_transform"] == "presence_0_1":
        y_train = df.iloc[train_rows]["Aspergillus_presence"].astype(int).to_numpy()
        y_test = df.iloc[test_rows]["Aspergillus_presence"].astype(int).to_numpy()
        y_test_metric = y_test.astype(float)
    elif row["target_transform"] == "log1p_positive_abundance":
        y_train = np.log1p(df.iloc[train_rows][cfg["target_col"]].astype(float).to_numpy()).astype(np.float32)
        y_test_metric = df.iloc[test_rows][cfg["target_col"]].astype(float).to_numpy()
        y_test = np.log1p(y_test_metric).astype(np.float32)
    else:
        raise ValueError(f"Unknown transform: {row['target_transform']}")
    return X_train, X_test, y_train, y_test, y_test_metric, features, train_rows, test_rows


def run_one(df: pd.DataFrame, row: pd.Series, cfg: dict, args: argparse.Namespace, device: str, use_amp: bool, device_info: dict):
    t0 = time.time()
    task_id = str(row["task_id"])
    rec = row.to_dict()
    rec.update(
        {
            "status": "failed",
            "error_message": "",
            "device": device,
            "device_info_json": json.dumps(device_info, ensure_ascii=False),
            "n_estimators": args.n_estimators,
            "batch_size": args.batch_size,
            "regressor_checkpoint_version": args.regressor_checkpoint_version,
            "classifier_checkpoint_version": args.classifier_checkpoint_version,
        }
    )
    try:
        seed = int(row["seed"])
        X_train, X_test, y_train, y_test, y_test_metric, features, train_rows, test_rows = prepare_xy(df, row, cfg)
        model = make_model(args, seed, device, use_amp, str(row["task_kind"]))
        kv_cache = False if args.kv_cache == "false" else args.kv_cache
        model.fit(X_train, y_train, kv_cache=kv_cache)

        out_base = Path(args.output_dir)
        partial_dir = out_base / "partials" / safe_name(task_id)
        partial_dir.mkdir(parents=True, exist_ok=True)
        pred_path = partial_dir / "predictions.csv"
        summary_path = partial_dir / "summary.json"

        if row["task_kind"] == "occurrence":
            prob = np.asarray(model.predict_proba(X_test))[:, 1].astype(float)
            pred = prob.copy()
            prob = np.clip(pred.astype(float), 0.0, 1.0)
            y_pred = (prob >= 0.5).astype(int)
            metrics = {
                "AUC": finite(safe_auc(y_test, prob)),
                "balanced_accuracy": finite(balanced_accuracy_score(y_test, y_pred)),
                "F1": finite(f1_score(y_test, y_pred, zero_division=0)),
                "prediction_mean": finite(np.nanmean(prob)),
            }
            pred_df = pd.DataFrame(
                {
                    "row_index": test_rows,
                    "Sample": df.iloc[test_rows][cfg["id_col"]].to_numpy(),
                    "y_true": y_test,
                    "y_score": prob,
                    "y_pred": y_pred,
                }
            )
        else:
            pred = np.asarray(model.predict(X_test)).ravel()
            pred_raw = np.expm1(pred.astype(float))
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
                    "y_pred_log1p": pred,
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
            "n_features": len(features),
            "n_train": len(train_rows),
            "n_test": len(test_rows),
            "predictions_csv": str(pred_path),
            "elapsed_sec": round(time.time() - t0, 3),
            "device_info": device_info,
        }
        write_json(summary_path, summary)
        rec.update(metrics)
        rec.update(
            {
                "status": "ok",
                "predictions_csv": str(pred_path),
                "summary_json": str(summary_path),
                "selected_features_json": json.dumps(features, ensure_ascii=False),
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
    ap.add_argument("--output-dir", default=str(ROOT / "outputs" / "17_tabiclv2_marta"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--n-estimators", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--kv-cache", default="repr", choices=["repr", "kv", "false"])
    ap.add_argument("--regressor-checkpoint-version", default="tabicl-regressor-v2-20260212.ckpt")
    ap.add_argument("--classifier-checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--flush-every", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    base = Path(args.config).resolve().parents[1]
    df = load_table((base / cfg["input_csv"]).resolve(), cfg["target_col"], cfg["id_col"])
    tasks = pd.read_csv(args.tasks_csv)
    tasks = tasks.iloc[[i for i in range(len(tasks)) if i % args.num_shards == args.shard_index]].copy()
    if args.limit > 0:
        tasks = tasks.head(args.limit)

    device, use_amp, device_info = resolve_device(args.device)
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
            rec = run_one(df, row, cfg, args, device, use_amp, device_info)
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
