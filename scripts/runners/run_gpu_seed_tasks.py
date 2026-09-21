#!/usr/bin/env python3
"""Run v2 GPU model tasks and export v3 seed-level CSVs.

This wrapper keeps the v2 model implementation/hyperparameters but changes the
public output contract: one metrics CSV, one predictions CSV, and one mean SHAP
CSV per seed. It is designed for resumable multi-node OAR runs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score


REPO = Path(__file__).resolve().parents[2]
V2 = REPO
V3 = REPO

MODEL_LABEL = {"random_forest": "Random Forest", "tabpfn": "TFN/TabPFN", "tabiclv2": "TabICLv2", "xgboost": "XGBoost", "tabfm": "TabFM"}
MODEL_DIR = {"random_forest": "random_forest", "tabpfn": "tabpfn", "tabiclv2": "tabiclv2", "xgboost": "xgboost", "tabfm": "tabfm"}
SCENARIO_LABEL = {"A_environment": "A", "B_microbiome": "B", "C_environment_microbiome": "C"}
TASK_LABEL = {
    "occurrence": "Classification",
    "positive_abundance": "Positive-abundance regression",
    "conditional_abundance": "Positive-abundance regression",
    "all_sample_abundance": "Complete-abundance regression",
}
TASK_DIR = {
    "occurrence": "classification",
    "positive_abundance": "regression_18_positive",
    "conditional_abundance": "regression_18_positive_samples",
    "all_sample_abundance": "regression_50_all_samples",
}
EXPECTED_FOLDS = {"occurrence": 15, "positive_abundance": 18, "conditional_abundance": 18, "all_sample_abundance": 15}
METRIC_COLS = ["AUC", "F1", "MAE", "R2", "RMSE", "balanced_accuracy", "prediction_mean"]

sys.path.insert(0, str(REPO / "scripts/validated_v3"))
sys.path.insert(0, str(REPO / "src"))
TABFM_REPO = Path(os.environ.get("TABFM_REPO", REPO / "vendor/tabfm"))
if TABFM_REPO.is_dir():
    sys.path.insert(0, str(TABFM_REPO))

from tfms_aspergillus.v3_workflow.data import load_table  # noqa: E402
from tfms_aspergillus.v3_workflow.environmental_preprocessing import preprocess_train_test_environmental  # noqa: E402
from tfms_aspergillus.v3_workflow.metrics import safe_auc  # noqa: E402
from tfms_aspergillus.v3_workflow.models import impute_train_test  # noqa: E402
from tfms_aspergillus.v3_workflow.shap_utils import ShapConfig, finite, kernel_shap, shap_rows  # noqa: E402
from run_model_shap_seed_tasks import read_done, run_one  # noqa: E402
from run_model_shap_seed_tasks import dumps, loads_list, parse_shap_nsamples, safe_name  # noqa: E402
from run_validated_v3_workflow import load_config  # noqa: E402
from tfms_aspergillus.resources import ResourceMonitor  # noqa: E402
from tfms_aspergillus.shap_products import export_shap_products  # noqa: E402


_TABFM_BASE_MODELS: dict[tuple[str, str], Any] = {}


def csv_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [x.strip() for x in value.split(",") if x.strip()]


def model_args(model: str, device: str, parsed_args: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        device=device,
        shap_background_size=20,
        shap_nsamples="auto",
        rf_n_estimators=500,
        model_n_jobs=1,
        tabicl_n_estimators=6,
        tabicl_batch_size=2,
        tabicl_kv_cache="repr",
        tabicl_regressor_checkpoint_version="tabicl-regressor-v2-20260212.ckpt",
        tabicl_classifier_checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
        xgb_n_estimators=500,
        xgb_learning_rate=0.03,
        xgb_max_depth=3,
        xgb_min_child_weight=1.0,
        xgb_subsample=0.632,
        xgb_reg_alpha=0.0,
        xgb_reg_lambda=1.0,
        xgb_max_bin=256,
        xgb_n_jobs=1,
        tabfm_backend=getattr(parsed_args, "tabfm_backend", os.environ.get("TABFM_BACKEND", "jax")),
        tabfm_n_estimators=getattr(parsed_args, "tabfm_n_estimators", 32),
        tabfm_batch_size=getattr(parsed_args, "tabfm_batch_size", 1),
        tabfm_max_num_features=getattr(parsed_args, "tabfm_max_num_features", 500),
        tabfm_max_num_rows=getattr(parsed_args, "tabfm_max_num_rows", None),
    )


def seed_dirs(args: argparse.Namespace, model: str, scenario: str, seed: int) -> tuple[Path, Path, Path]:
    root = Path(args.output_root) if args.output_root else V3 / "results" / args.family
    base = root / TASK_DIR[args.task_kind] / scenario / MODEL_DIR[model]
    return (
        base / "seed_metrics" / f"seed_{seed:03d}_metrics.csv",
        base / "seed_predictions" / f"seed_{seed:03d}_predictions.csv",
        base / "seed_shap" / f"seed_{seed:03d}_mean_abs_shap.csv",
    )


def existing_seed_done(args: argparse.Namespace, model: str, scenario: str, seed: int) -> bool:
    metric_csv, _, _ = seed_dirs(args, model, scenario, seed)
    if not metric_csv.exists() or metric_csv.stat().st_size == 0:
        return False
    try:
        df = pd.read_csv(metric_csv)
    except Exception:
        return False
    if len(df) != 1:
        return False
    if model == "tabpfn":
        if "tabpfn_random_state" not in df.columns or "tabpfn_seeded_replicate" not in df.columns:
            return False
        try:
            if int(df["tabpfn_random_state"].iloc[0]) != int(seed):
                return False
            if str(df["tabpfn_seeded_replicate"].iloc[0]).lower() not in {"true", "1"}:
                return False
        except Exception:
            return False
    if model == "tabfm":
        if "tabfm_random_state" not in df.columns or "tabfm_seeded_replicate" not in df.columns:
            return False
        try:
            if int(df["tabfm_random_state"].iloc[0]) != int(seed):
                return False
            if str(df["tabfm_seeded_replicate"].iloc[0]).lower() not in {"true", "1"}:
                return False
        except Exception:
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
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def aggregate_seed(records: list[dict[str, Any]], args: argparse.Namespace, model: str, scenario: str, seed: int, source_jsonl: Path) -> dict[str, str]:
    expected = int(args.max_folds) if getattr(args, "max_folds", 0) else EXPECTED_FOLDS[args.task_kind]
    ok = [r for r in records if r.get("status") == "ok"]
    if len(ok) != expected:
        raise RuntimeError(f"{args.family} {args.task_kind} {model} {scenario} seed={seed}: {len(ok)}/{expected} ok folds")

    metric_csv, pred_csv, shap_csv = seed_dirs(args, model, scenario, seed)
    metric_csv.parent.mkdir(parents=True, exist_ok=True)

    metric_row: dict[str, Any] = {
        "family": args.family,
        "family_label": args.family_label,
        "task": args.task_kind,
        "task_label": TASK_LABEL[args.task_kind],
        "scenario": scenario,
        "scenario_label": SCENARIO_LABEL.get(scenario, scenario),
        "model": MODEL_LABEL[model],
        "seed": int(seed),
        "n_folds_tested": expected,
        "completed_folds": expected,
        "expected_folds": expected,
        "seed_complete_all_folds": True,
        "source_table": str(source_jsonl),
    }
    if model == "tabpfn":
        metric_row["tabpfn_random_state"] = int(seed)
        metric_row["tabpfn_seeded_replicate"] = True
    if model == "tabfm":
        metric_row["tabfm_random_state"] = int(seed)
        metric_row["tabfm_seeded_replicate"] = True
        metric_row["tabfm_backend"] = getattr(args, "tabfm_backend", os.environ.get("TABFM_BACKEND", "jax"))
        metric_row["tabfm_n_estimators"] = getattr(args, "tabfm_n_estimators", 32)
    for col in METRIC_COLS:
        vals = pd.to_numeric(pd.Series([r.get(col) for r in ok]), errors="coerce")
        metric_row[col] = vals.mean(skipna=True)
    pd.DataFrame([metric_row]).to_csv(metric_csv, index=False)

    pred_frames = []
    shap_frames = []
    for rec in ok:
        pred_path = Path(str(rec.get("predictions_csv", "")))
        shap_path = Path(str(rec.get("shap_csv", "")))
        if pred_path.exists():
            pred_frames.append(pd.read_csv(pred_path))
        if shap_path.exists():
            shap_frames.append(pd.read_csv(shap_path))

    if pred_frames:
        pred_csv.parent.mkdir(parents=True, exist_ok=True)
        preds = pd.concat(pred_frames, ignore_index=True, sort=False)
        preds.insert(0, "family", args.family)
        preds.to_csv(pred_csv, index=False)

    if shap_frames:
        shap_csv.parent.mkdir(parents=True, exist_ok=True)
        shap = pd.concat(shap_frames, ignore_index=True, sort=False)
        feature_col = "variable" if "variable" in shap.columns else "feature"
        value_col = "abs_shap_value" if "abs_shap_value" in shap.columns else "shap_value"
        if value_col == "shap_value":
            shap["_abs_shap_value"] = pd.to_numeric(shap[value_col], errors="coerce").abs()
            value_col = "_abs_shap_value"
        out = (
            shap.groupby(feature_col, as_index=False)
            .agg(mean_abs_shap=(value_col, "mean"), sd_abs_shap=(value_col, "std"), n_rows=(value_col, "size"))
            .rename(columns={feature_col: "feature"})
            .sort_values("mean_abs_shap", ascending=False)
        )
        out.insert(0, "seed", int(seed))
        out.insert(0, "model", MODEL_LABEL[model])
        out.insert(0, "scenario", scenario)
        out.insert(0, "task", args.task_kind)
        out.insert(0, "family", args.family)
        out.to_csv(shap_csv, index=False)
        product_root=Path(args.output_root) if args.output_root else REPO/"results"
        shap_products=export_shap_products(shap,product_root,family=args.family,task=args.task_kind,scenario=scenario,model=MODEL_LABEL[model],seed=seed)
    else:
        shap_products={}
    return {"metric_csv":str(metric_csv),"prediction_csv":str(pred_csv),"seed_shap_csv":str(shap_csv),**shap_products}


def load_tabfm_base_model(backend: str, task_kind: str) -> Any:
    try:
        import tabfm  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "TabFM is not importable. Set TABFM_REPO to the cloned google-research/tabfm repo "
            "and activate a Python >=3.11 conda environment with tabfm dependencies."
        ) from exc

    model_type = "classification" if task_kind == "occurrence" else "regression"
    key = (backend, model_type)
    if key in _TABFM_BASE_MODELS:
        return _TABFM_BASE_MODELS[key]
    if backend == "jax":
        loader = tabfm.tabfm_v1_0_0_jax
    elif backend == "pytorch":
        loader = tabfm.tabfm_v1_0_0_pytorch
    else:
        raise ValueError(f"Unknown TabFM backend: {backend!r}. Expected 'jax' or 'pytorch'.")
    _TABFM_BASE_MODELS[key] = loader.load(model_type=model_type)
    return _TABFM_BASE_MODELS[key]


def make_tabfm_model(args: argparse.Namespace, run_args: SimpleNamespace, task_kind: str, seed: int) -> Any:
    import tabfm  # type: ignore

    base_model = load_tabfm_base_model(run_args.tabfm_backend, task_kind)
    common = dict(
        model=base_model,
        n_estimators=run_args.tabfm_n_estimators,
        batch_size=run_args.tabfm_batch_size,
        random_state=seed,
        max_num_features=run_args.tabfm_max_num_features,
        max_num_rows=run_args.tabfm_max_num_rows,
        verbose=False,
    )
    if task_kind == "occurrence":
        return tabfm.TabFMClassifier(**common)
    return tabfm.TabFMRegressor(**common)


def run_one_tabfm(
    df: pd.DataFrame,
    cfg: dict,
    row: pd.Series,
    args: argparse.Namespace,
    run_args: SimpleNamespace,
    run_dir: Path,
) -> dict[str, Any]:
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
        X_train_arr, X_test_arr = impute_train_test(X_train_df, X_test_df)
        X_train_arr = np.asarray(X_train_arr, dtype=np.float32)
        X_test_arr = np.asarray(X_test_arr, dtype=np.float32)
        X_train = pd.DataFrame(X_train_arr, columns=features)
        X_test = pd.DataFrame(X_test_arr, columns=features)

        seed = int(row["seed"])
        rec["tabfm_random_state"] = seed
        rec["tabfm_seeded_replicate"] = True
        rec["tabfm_backend"] = run_args.tabfm_backend
        rec["tabfm_n_estimators"] = run_args.tabfm_n_estimators

        if row["task_kind"] == "occurrence":
            y_train = df.iloc[train_rows]["Aspergillus_presence"].astype(int).to_numpy()
            y_test = df.iloc[test_rows]["Aspergillus_presence"].astype(int).to_numpy()
            model = make_tabfm_model(args, run_args, "occurrence", seed)
            model.fit(X_train, y_train)
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
            predict_fn = lambda X: np.clip(
                np.asarray(model.predict_proba(pd.DataFrame(np.asarray(X, dtype=np.float32), columns=features)))[:, 1].astype(float),
                0.0,
                1.0,
            )
        else:
            y_train = np.log1p(df.iloc[train_rows][cfg["target_col"]].astype(float).to_numpy()).astype(np.float32)
            y_test_raw = df.iloc[test_rows][cfg["target_col"]].astype(float).to_numpy()
            model = make_tabfm_model(args, run_args, str(row["task_kind"]), seed)
            model.fit(X_train, y_train)
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
            predict_fn = lambda X: np.clip(
                np.expm1(np.asarray(model.predict(pd.DataFrame(np.asarray(X, dtype=np.float32), columns=features))).ravel().astype(float)),
                0.0,
                1.0,
            )

        shap_cfg = ShapConfig(
            background_size=args.shap_background_size,
            nsamples=parse_shap_nsamples(args.shap_nsamples),
            random_state=seed,
        )
        shap_values, base_values = kernel_shap(predict_fn, X_train_arr, X_test_arr, features, str(row["task_kind"]), shap_cfg)

        partial_dir = run_dir / "partials" / safe_name(task_id)
        partial_dir.mkdir(parents=True, exist_ok=True)
        pred_df["task_id"] = task_id
        pred_df["analysis"] = row["analysis"]
        pred_df["task_kind"] = row["task_kind"]
        pred_df["scenario"] = row["scenario"]
        pred_df["variant"] = row["variant"]
        pred_df["model"] = "TabFM"
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
                model_name="TabFM",
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
            "fit_seconds": fit_seconds,
            "shap_seconds": shap_seconds,
            "model_n_estimators": getattr(model, "n_estimators", None),
            "environmental_preprocessing": env_preprocessing_info,
            "tabfm_backend": run_args.tabfm_backend,
            "tabfm_n_estimators": run_args.tabfm_n_estimators,
        }
        summary_path = partial_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
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


def run_seed(df: pd.DataFrame, cfg: dict, tasks: pd.DataFrame, args: argparse.Namespace, run_args: SimpleNamespace, scenario: str, seed: int) -> dict[str, Any]:
    if existing_seed_done(args, args.model, scenario, seed):
        return {"status": "skipped_existing", "scenario": scenario, "seed": seed}

    expected = EXPECTED_FOLDS[args.task_kind]
    rows = tasks[
        tasks["task_kind"].eq(args.task_kind)
        & tasks["scenario"].eq(scenario)
        & tasks["seed"].astype(int).eq(seed)
    ].sort_values("fold_id")
    if len(rows) != expected:
        return {"status": "missing_task_rows", "scenario": scenario, "seed": seed, "rows": len(rows), "expected": expected}
    if args.max_folds:
        rows = rows.head(args.max_folds)
        expected = int(args.max_folds)

    run_dir = Path(args.work_dir) / args.run_tag / args.family / args.task_kind / args.model / scenario / f"seed_{seed:03d}"
    source_jsonl = run_dir / "monitoring" / f"{args.model}_{scenario}_seed_{seed:03d}.jsonl"
    done = read_done(source_jsonl) if args.resume else set()
    t0 = time.time()
    monitor=ResourceMonitor(run_dir/"resource_usage.json")

    for _, row in rows.iterrows():
        if str(row["task_id"]) in done:
            continue
        if args.model == "tabfm":
            rec = run_one_tabfm(df, cfg, row, args, run_args, run_dir)
        else:
            rec = run_one(df, cfg, row, run_args, run_dir)
        monitor.observe()
        append_jsonl(source_jsonl, rec)
        print(json.dumps({"event": "fold_done", "model": args.model, "task": args.task_kind, "scenario": scenario, "seed": seed, "fold_id": int(row["fold_id"]), "status": rec.get("status"), "error": str(rec.get("error_message", ""))[:180]}, ensure_ascii=False), flush=True)

    all_records: list[dict[str, Any]] = []
    if source_jsonl.exists():
        with source_jsonl.open(errors="replace") as fh:
            for line in fh:
                if line.strip():
                    all_records.append(json.loads(line))
    ok_records = [r for r in all_records if r.get("status") == "ok"]
    monitor.stop(model=args.model,task=args.task_kind,scenario=scenario,seed=seed,fold_count=len(ok_records),workers=1,device=args.device)
    if len(ok_records) < expected:
        return {"status":"incomplete","scenario":scenario,"seed":seed,"ok_folds":len(ok_records),"expected_folds":expected,"source_jsonl":str(source_jsonl),"resource_usage":str(run_dir/"resource_usage.json")}

    outputs = aggregate_seed(ok_records[-expected:], args, args.model, scenario, seed, source_jsonl)
    if args.cleanup_partials:
        shutil.rmtree(run_dir / "partials", ignore_errors=True)
    return {"status":"ok","scenario":scenario,"seed":seed,"ok_folds":expected,"elapsed_sec":round(time.time()-t0,3),"resource_usage":str(run_dir/"resource_usage.json"),**outputs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--family-label", required=True)
    ap.add_argument("--task-kind", choices=sorted(TASK_DIR), required=True)
    ap.add_argument("--model", choices=sorted(MODEL_DIR), required=True)
    ap.add_argument("--tasks-csv", required=True)
    ap.add_argument("--config", default=str(REPO / "configs/validated_v3/v3_repro_config.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scenarios", default="A_environment,B_microbiome,C_environment_microbiome")
    ap.add_argument("--seeds", default=",".join(str(x) for x in range(123, 223)))
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit-seeds", type=int, default=0)
    ap.add_argument("--max-folds", type=int, default=0)
    ap.add_argument("--shap-background-size", type=int, default=20)
    ap.add_argument("--shap-nsamples", default="auto")
    ap.add_argument("--tabfm-backend", default=os.environ.get("TABFM_BACKEND", "jax"), choices=["jax", "pytorch"])
    ap.add_argument("--tabfm-n-estimators", type=int, default=int(os.environ.get("TABFM_N_ESTIMATORS", "32")))
    ap.add_argument("--tabfm-batch-size", type=int, default=int(os.environ.get("TABFM_BATCH_SIZE", "1")))
    ap.add_argument("--tabfm-max-num-features", type=int, default=int(os.environ.get("TABFM_MAX_NUM_FEATURES", "500")))
    ap.add_argument("--tabfm-max-num-rows", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cleanup-partials", action="store_true", default=True)
    ap.add_argument("--work-dir", default=str(REPO / "results/monitoring/v3_gpu_seed_tasks"))
    ap.add_argument("--output-root", default="")
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()

    args.run_tag = args.run_tag or f"{args.family}_{args.task_kind}_{args.model}_v3_seed_{time.strftime('%Y%m%d_%H%M%S')}"
    scenarios = csv_list(args.scenarios, list(SCENARIO_LABEL))
    seeds = [int(x) for x in csv_list(args.seeds, [str(x) for x in range(123, 223)])]

    cfg = load_config(Path(args.config))
    df = load_table((Path(args.config).resolve().parents[1] / cfg["input_csv"]).resolve(), cfg["target_col"], cfg["id_col"])
    tasks = pd.read_csv(args.tasks_csv)
    tasks = tasks[tasks["task_kind"].eq(args.task_kind)].copy()
    combos = (
        tasks[tasks["scenario"].isin(scenarios) & tasks["seed"].astype(int).isin(seeds)][["scenario", "seed"]]
        .drop_duplicates()
        .assign(seed=lambda d: d["seed"].astype(int))
        .sort_values(["scenario", "seed"])
    )
    combos = [(r.scenario, int(r.seed)) for r in combos.itertuples(index=False) if not existing_seed_done(args, args.model, r.scenario, int(r.seed))]
    combos = combos[args.shard_index :: args.num_shards]
    if args.limit_seeds:
        combos = combos[: args.limit_seeds]

    args.tabfm_max_num_rows = args.tabfm_max_num_rows or None
    run_args = model_args(args.model, args.device, args)
    print(json.dumps({"event": "start", "family": args.family, "task_kind": args.task_kind, "model": args.model, "device": args.device, "run_tag": args.run_tag, "n_combos": len(combos), "combos_head": combos[:20], "tasks_csv": args.tasks_csv, "output_root": args.output_root or str(V3 / "results" / args.family)}, ensure_ascii=False, indent=2), flush=True)

    status = 0
    summary_path = Path(args.work_dir) / args.run_tag / args.family / args.task_kind / args.model / f"summary_shard_{args.shard_index}_of_{args.num_shards}.jsonl"
    for scenario, seed in combos:
        result = run_seed(df, cfg, tasks, args, run_args, scenario, seed)
        append_jsonl(summary_path, result)
        print(json.dumps({"event": "seed_done", **result}, ensure_ascii=False), flush=True)
        if result.get("status") not in {"ok", "skipped_existing"}:
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
