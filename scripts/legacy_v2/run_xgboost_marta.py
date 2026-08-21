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
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from marta_omar_repro.data import detect_blocks, load_table, write_json
from marta_omar_repro.metrics import safe_auc
from marta_omar_repro.models import impute_train_test, select_features_inside_fold
from run_omar_repro import load_config, parse_seeds, scenario_map


def safe_name(*parts: Any, max_len: int = 180) -> str:
    value = "__".join(str(part) for part in parts)
    out = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("_")
    return (out[:max_len] or "task")


def finite(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def loads_list(value: str) -> list[Any]:
    out = json.loads(value)
    if not isinstance(out, list):
        raise ValueError("expected JSON list")
    return out


def sqrt_colsample(n_features: int) -> float:
    if n_features <= 1:
        return 1.0
    return min(1.0, max(1.0 / float(np.sqrt(n_features)), 1.0 / float(n_features)))


def build_tasks(df: pd.DataFrame, cfg: dict, seeds: list[int]) -> pd.DataFrame:
    blocks = detect_blocks(df, cfg["target_col"], cfg["dummy_cols"])
    scenarios = scenario_map(df, blocks, cfg["target_col"])
    rows: list[dict] = []

    y_occ = df["Aspergillus_presence"].astype(int).to_numpy()
    for seed in seeds:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
        for scenario, features in scenarios.items():
            for fold_id, (train_idx, test_idx) in enumerate(cv.split(df[features], y_occ), start=1):
                rows.append(
                    {
                        "task_id": safe_name("xgb", "15A_occurrence", scenario, f"seed{seed}", f"fold{fold_id}"),
                        "analysis": "15A",
                        "task_kind": "occurrence",
                        "scenario": scenario,
                        "variant": "anova",
                        "model": "XGBoost_RF_like",
                        "seed": seed,
                        "fold_id": fold_id,
                        "target_transform": "presence_0_1",
                        "train_rows_json": dumps([int(i) for i in train_idx]),
                        "test_rows_json": dumps([int(i) for i in test_idx]),
                        "features_json": dumps(list(features)),
                    }
                )

        predictors = cfg["dummy_cols"] + cfg["rf_best_occurrence_covars"]
        cv_rfbest = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
        for fold_id, (train_idx, test_idx) in enumerate(cv_rfbest.split(df[predictors], y_occ), start=1):
            rows.append(
                {
                    "task_id": safe_name("xgb", "16A_occurrence_rfbest", f"seed{seed}", f"fold{fold_id}"),
                    "analysis": "16A",
                    "task_kind": "occurrence",
                    "scenario": "RFbest_occurrence",
                    "variant": "rfbest",
                    "model": "XGBoost_RF_like",
                    "seed": seed,
                    "fold_id": fold_id,
                    "target_transform": "presence_0_1",
                    "train_rows_json": dumps([int(i) for i in train_idx]),
                    "test_rows_json": dumps([int(i) for i in test_idx]),
                    "features_json": dumps(list(predictors)),
                }
            )

    pos_rows = np.where(df[cfg["target_col"]].astype(float).to_numpy() > 0)[0]
    data_pos = df.iloc[pos_rows].reset_index(drop=False).rename(columns={"index": "source_row"})
    for seed in seeds:
        for scenario, features in scenarios.items():
            for fold_id, (train_local, test_local) in enumerate(LeaveOneOut().split(data_pos), start=1):
                train_idx = data_pos.iloc[train_local]["source_row"].astype(int).to_numpy()
                test_idx = data_pos.iloc[test_local]["source_row"].astype(int).to_numpy()
                rows.append(
                    {
                        "task_id": safe_name("xgb", "15A_abundance", scenario, f"seed{seed}", f"fold{fold_id}"),
                        "analysis": "15A",
                        "task_kind": "conditional_abundance",
                        "scenario": scenario,
                        "variant": "anova",
                        "model": "XGBoost_RF_like",
                        "seed": seed,
                        "fold_id": fold_id,
                        "target_transform": "log1p_positive_abundance",
                        "train_rows_json": dumps([int(i) for i in train_idx]),
                        "test_rows_json": dumps([int(i) for i in test_idx]),
                        "features_json": dumps(list(features)),
                    }
                )

        predictors = cfg["dummy_cols"] + cfg["rf_best_occurrence_covars"]
        for fold_id, (train_local, test_local) in enumerate(LeaveOneOut().split(data_pos), start=1):
            train_idx = data_pos.iloc[train_local]["source_row"].astype(int).to_numpy()
            test_idx = data_pos.iloc[test_local]["source_row"].astype(int).to_numpy()
            rows.append(
                {
                    "task_id": safe_name("xgb", "16A_abundance_rfbest", f"seed{seed}", f"fold{fold_id}"),
                    "analysis": "16A",
                    "task_kind": "conditional_abundance",
                    "scenario": "RFbest_occurrence",
                    "variant": "rfbest",
                    "model": "XGBoost_RF_like",
                    "seed": seed,
                    "fold_id": fold_id,
                    "target_transform": "log1p_positive_abundance",
                    "train_rows_json": dumps([int(i) for i in train_idx]),
                    "test_rows_json": dumps([int(i) for i in test_idx]),
                    "features_json": dumps(list(predictors)),
                }
            )
    return pd.DataFrame(rows)


def select_for_task(df: pd.DataFrame, cfg: dict, row: pd.Series, X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame, y_train: np.ndarray):
    features = list(X_train_raw.columns)
    forced = [c for c in cfg["dummy_cols"] if c in features]
    if row["variant"] == "rfbest":
        return X_train_raw, X_test_raw, features
    core = [c for c in features if c not in forced]
    max_features = cfg["max_selected_features_by_scenario"].get(str(row["scenario"]))
    if max_features is not None:
        max_features = min(int(max_features), len(core))
    task = "classification" if row["task_kind"] == "occurrence" else "regression"
    return select_features_inside_fold(
        X_train_raw,
        y_train,
        X_test_raw,
        task=task,
        max_features=max_features,
        forced_cols=forced,
    )


def common_xgb_params(args: argparse.Namespace, seed: int, n_features: int, device: str) -> dict[str, Any]:
    return {
        "booster": "gbtree",
        "tree_method": "hist",
        "device": device,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": 1.0,
        "colsample_bylevel": 1.0,
        "colsample_bynode": sqrt_colsample(n_features),
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "max_bin": args.max_bin,
        "random_state": seed,
        "n_jobs": args.n_jobs,
        "verbosity": args.verbosity,
        "validate_parameters": True,
    }


def make_classifier(args: argparse.Namespace, seed: int, n_features: int, y_train: np.ndarray, device: str):
    from xgboost import XGBClassifier

    neg = int(np.sum(y_train == 0))
    pos = int(np.sum(y_train == 1))
    scale = float(neg / pos) if pos > 0 else 1.0
    return XGBClassifier(
        **common_xgb_params(args, seed, n_features, device),
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale,
    )


def make_regressor(args: argparse.Namespace, seed: int, n_features: int, device: str):
    from xgboost import XGBRegressor

    return XGBRegressor(
        **common_xgb_params(args, seed, n_features, device),
        objective="reg:squarederror",
        eval_metric="rmse",
    )


def pred_contribs(model: Any, X: np.ndarray, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    import xgboost as xgb

    booster = model.get_booster()
    dm = xgb.DMatrix(X, feature_names=features)
    contrib = np.asarray(booster.predict(dm, pred_contribs=True), dtype=float)
    if contrib.ndim == 3:
        contrib = contrib[:, :, -1]
    if contrib.ndim == 1:
        contrib = contrib.reshape(1, -1)
    shap_values = contrib[:, :-1]
    base_values = contrib[:, -1]
    return shap_values, base_values


def run_one(df: pd.DataFrame, cfg: dict, row: pd.Series, args: argparse.Namespace, out_dir: Path, device: str) -> dict:
    t0 = time.time()
    task_id = str(row["task_id"])
    rec = row.to_dict()
    rec.update(
        {
            "status": "failed",
            "error_message": "",
            "device": device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "min_child_weight": args.min_child_weight,
            "subsample": args.subsample,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
            "max_bin": args.max_bin,
        }
    )
    try:
        features = [str(x) for x in loads_list(row["features_json"])]
        train_rows = [int(x) for x in loads_list(row["train_rows_json"])]
        test_rows = [int(x) for x in loads_list(row["test_rows_json"])]
        X_train_raw = df.iloc[train_rows][features].copy()
        X_test_raw = df.iloc[test_rows][features].copy()
        seed = int(row["seed"])

        partial_dir = out_dir / "partials" / safe_name(task_id)
        partial_dir.mkdir(parents=True, exist_ok=True)

        if row["task_kind"] == "occurrence":
            y_train = df.iloc[train_rows]["Aspergillus_presence"].astype(int).to_numpy()
            y_test = df.iloc[test_rows]["Aspergillus_presence"].astype(int).to_numpy()
            X_train_sel, X_test_sel, selected = select_for_task(df, cfg, row, X_train_raw, X_test_raw, y_train)
            X_train, X_test = impute_train_test(X_train_sel, X_test_sel)
            X_train = np.asarray(X_train, dtype=np.float32)
            X_test = np.asarray(X_test, dtype=np.float32)
            model = make_classifier(args, seed, len(selected), y_train, device)
            model.fit(X_train, y_train)
            prob = np.clip(np.asarray(model.predict_proba(X_test))[:, 1].astype(float), 0.0, 1.0)
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
            shap_y = y_test.astype(float)
            pred_for_shap = prob
            pred_col = "y_score"
        else:
            y_train = np.log1p(df.iloc[train_rows][cfg["target_col"]].astype(float).to_numpy()).astype(np.float32)
            y_test_raw = df.iloc[test_rows][cfg["target_col"]].astype(float).to_numpy()
            y_test_log = np.log1p(y_test_raw).astype(np.float32)
            X_train_sel, X_test_sel, selected = select_for_task(df, cfg, row, X_train_raw, X_test_raw, y_train)
            X_train, X_test = impute_train_test(X_train_sel, X_test_sel)
            X_train = np.asarray(X_train, dtype=np.float32)
            X_test = np.asarray(X_test, dtype=np.float32)
            model = make_regressor(args, seed, len(selected), device)
            model.fit(X_train, y_train)
            pred_log = np.asarray(model.predict(X_test)).ravel().astype(float)
            pred_raw = np.clip(np.expm1(pred_log), 0.0, 1.0)
            metrics = {
                "RMSE": finite(mean_squared_error(y_test_raw, pred_raw) ** 0.5),
                "MAE": finite(mean_absolute_error(y_test_raw, pred_raw)),
                "R2": finite(r2_score(y_test_raw, pred_raw)) if len(y_test_raw) > 1 else None,
                "prediction_mean": finite(np.nanmean(pred_raw)),
            }
            pred_df = pd.DataFrame(
                {
                    "row_index": test_rows,
                    "Sample": df.iloc[test_rows][cfg["id_col"]].to_numpy(),
                    "y_true_raw": y_test_raw,
                    "y_pred_raw": pred_raw,
                    "y_true_log1p": y_test_log,
                    "y_pred_log1p": pred_log,
                }
            )
            shap_y = y_test_raw.astype(float)
            pred_for_shap = pred_raw
            pred_col = "y_pred_raw"

        pred_df["task_id"] = task_id
        pred_df["analysis"] = row["analysis"]
        pred_df["task_kind"] = row["task_kind"]
        pred_df["scenario"] = row["scenario"]
        pred_df["variant"] = row["variant"]
        pred_df["model"] = row["model"]
        pred_df["seed"] = seed
        pred_df["fold_id"] = int(row["fold_id"])
        pred_path = partial_dir / "predictions.csv"
        pred_df.to_csv(pred_path, index=False)

        shap_values, base_values = pred_contribs(model, X_test, list(selected))
        shap_rows = []
        for i, source_row in enumerate(test_rows):
            for j, var in enumerate(selected):
                shap_rows.append(
                    {
                        "task_id": task_id,
                        "analysis": row["analysis"],
                        "task_kind": row["task_kind"],
                        "scenario": row["scenario"],
                        "variant": row["variant"],
                        "model": row["model"],
                        "seed": seed,
                        "fold_id": int(row["fold_id"]),
                        "row_index": int(source_row),
                        "Sample": df.iloc[source_row][cfg["id_col"]],
                        "variable": var,
                        "feature_value": finite(X_test_sel.iloc[i][var]),
                        "shap_value": finite(shap_values[i, j]),
                        "abs_shap_value": finite(abs(shap_values[i, j])),
                        "base_value": finite(base_values[i]),
                        "y_true": finite(shap_y[i]),
                        pred_col: finite(pred_for_shap[i]),
                        "n_features": len(selected),
                    }
                )
        shap_path = partial_dir / "shap_values.csv"
        pd.DataFrame(shap_rows).to_csv(shap_path, index=False)

        summary = {
            "task_id": task_id,
            "status": "ok",
            "metrics": metrics,
            "features": list(selected),
            "n_features": len(selected),
            "n_train": len(train_rows),
            "n_test": len(test_rows),
            "predictions_csv": str(pred_path),
            "shap_csv": str(shap_path),
            "elapsed_sec": round(time.time() - t0, 3),
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
                "selected_features_json": dumps(list(selected)),
                "n_features": len(selected),
            }
        )
    except Exception as exc:
        rec["error_message"] = f"{type(exc).__name__}: {exc}"
    rec["elapsed_sec"] = round(time.time() - t0, 3)
    return rec


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


def read_done_global(monitoring_dir: Path) -> set[str]:
    done: set[str] = set()
    for path in sorted(monitoring_dir.glob("xgboost_shard_*.jsonl")):
        done.update(read_done(path))
    return done


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
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


def summarize_occurrence(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in preds.groupby(["analysis", "scenario", "variant", "model", "seed"], dropna=False):
        y_true = g["y_true"].astype(int).to_numpy()
        y_score = g["y_score"].astype(float).to_numpy()
        y_pred = (y_score >= 0.5).astype(int)
        rows.append(
            {
                "analysis": keys[0],
                "scenario": keys[1],
                "variant": keys[2],
                "model": keys[3],
                "seed": keys[4],
                "AUC": safe_auc(y_true, y_score),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "F1": f1_score(y_true, y_pred, zero_division=0),
                "prediction_mean": float(np.nanmean(y_score)),
                "n_predictions": len(g),
            }
        )
    return pd.DataFrame(rows)


def summarize_regression(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in preds.groupby(["analysis", "scenario", "variant", "model", "seed"], dropna=False):
        y_true = g["y_true_raw"].astype(float).to_numpy()
        y_pred = g["y_pred_raw"].astype(float).to_numpy()
        rows.append(
            {
                "analysis": keys[0],
                "scenario": keys[1],
                "variant": keys[2],
                "model": keys[3],
                "seed": keys[4],
                "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
                "MAE": mean_absolute_error(y_true, y_pred),
                "R2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
                "prediction_mean": float(np.nanmean(y_pred)),
                "n_predictions": len(g),
            }
        )
    return pd.DataFrame(rows)


def aggregate(summary: pd.DataFrame, metric_cols: list[str], sort_col: str, ascending: bool) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    agg = {"n_seeds": ("seed", "nunique"), "n_predictions_total": ("n_predictions", "sum")}
    for col in metric_cols:
        agg[f"{col}_mean"] = (col, "mean")
        agg[f"{col}_sd"] = (col, "std")
    return (
        summary.groupby(["analysis", "scenario", "variant", "model"], as_index=False)
        .agg(**agg)
        .sort_values(f"{sort_col}_mean", ascending=ascending)
    )


def consolidate(run_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mon = run_dir / "monitoring"
    out = run_dir / "consolidated"
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for p in sorted(mon.glob("*.jsonl")):
        records.extend(read_jsonl(p))
    rec_df = pd.DataFrame(records)
    rec_df.to_csv(out / "xgboost_task_records.csv", index=False)
    rec_ok = (
        rec_df[rec_df["status"] == "ok"].drop_duplicates("task_id", keep="last").copy()
        if not rec_df.empty and "task_id" in rec_df.columns
        else pd.DataFrame()
    )
    rec_ok.to_csv(out / "xgboost_task_records_unique_ok.csv", index=False)

    pred_frames = []
    shap_frames = []
    for path in rec_ok.get("predictions_csv", pd.Series(dtype=str)).dropna():
        p = Path(path)
        if p.exists():
            pred_frames.append(pd.read_csv(p))
    for path in rec_ok.get("shap_csv", pd.Series(dtype=str)).dropna():
        p = Path(path)
        if p.exists():
            shap_frames.append(pd.read_csv(p))
    preds = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    shap_values = pd.concat(shap_frames, ignore_index=True) if shap_frames else pd.DataFrame()
    preds.to_csv(out / "xgboost_predictions_all_completed.csv", index=False)
    shap_values.to_csv(out / "xgboost_shap_values_all.csv", index=False)

    if not preds.empty:
        occ = preds[preds["task_kind"] == "occurrence"].copy()
        reg = preds[preds["task_kind"] == "conditional_abundance"].copy()
        occ_seed = summarize_occurrence(occ) if not occ.empty else pd.DataFrame()
        reg_seed = summarize_regression(reg) if not reg.empty else pd.DataFrame()
        occ_agg = aggregate(occ_seed, ["AUC", "balanced_accuracy", "F1", "prediction_mean"], "AUC", ascending=False)
        reg_agg = aggregate(reg_seed, ["RMSE", "MAE", "R2", "prediction_mean"], "RMSE", ascending=True)
        occ_seed.to_csv(out / "xgboost_occurrence_summary_by_seed.csv", index=False)
        occ_agg.to_csv(out / "xgboost_occurrence_summary_across_seeds.csv", index=False)
        reg_seed.to_csv(out / "xgboost_abundance_summary_by_seed.csv", index=False)
        reg_agg.to_csv(out / "xgboost_abundance_summary_across_seeds.csv", index=False)

    if not shap_values.empty:
        summary = (
            shap_values.groupby(["analysis", "task_kind", "scenario", "variant", "model", "variable"], as_index=False)
            .agg(
                mean_abs_shap=("abs_shap_value", "mean"),
                sd_abs_shap=("abs_shap_value", "std"),
                mean_signed_shap=("shap_value", "mean"),
                n_values=("shap_value", "size"),
                n_seeds=("seed", "nunique"),
            )
            .sort_values("mean_abs_shap", ascending=False)
        )
        summary.to_csv(out / "xgboost_shap_importance_summary.csv", index=False)
        plot_dir = out / "plots"
        plot_dir.mkdir(exist_ok=True)
        for keys, g in summary.groupby(["task_kind", "scenario"], dropna=False):
            top = g.sort_values("mean_abs_shap", ascending=False).head(20).iloc[::-1]
            if top.empty:
                continue
            label = safe_name("__".join(map(str, keys)))
            fig_h = max(4.0, 0.28 * len(top) + 1.5)
            fig, ax = plt.subplots(figsize=(9, fig_h))
            ax.barh(top["variable"], top["mean_abs_shap"], xerr=top["sd_abs_shap"].fillna(0.0), color="#9467BD", alpha=0.86)
            ax.set_title(f"XGBoost SHAP: {keys[0]} / {keys[1]}")
            ax.set_xlabel("mean(|SHAP value|)")
            ax.set_ylabel("")
            fig.tight_layout()
            fig.savefig(plot_dir / f"xgboost_shap_importance_{label}.png", dpi=180)
            fig.savefig(plot_dir / f"xgboost_shap_importance_{label}.pdf")
            plt.close(fig)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "jsonl_records": int(len(rec_df)),
                "unique_ok_tasks": int(len(rec_ok)),
                "predictions": int(len(preds)),
                "shap_rows": int(len(shap_values)),
                "consolidated_dir": str(out),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "omar_repro_config.json"))
    ap.add_argument("--output-dir", default=str(ROOT / "outputs" / "32_xgboost_marta" / "xgboost_marta_100seeds_v1"))
    ap.add_argument("--seeds", default=",".join(map(str, range(123, 223))))
    ap.add_argument("--build-tasks-only", action="store_true")
    ap.add_argument("--consolidate-only", action="store_true")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--min-child-weight", type=float, default=1.0)
    ap.add_argument("--subsample", type=float, default=0.632)
    ap.add_argument("--reg-alpha", type=float, default=0.0)
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--max-bin", type=int, default=256)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--verbosity", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.output_dir)
    mon = run_dir / "monitoring"
    mon.mkdir(parents=True, exist_ok=True)
    tasks_csv = mon / "tasks.csv"

    if args.consolidate_only:
        consolidate(run_dir)
        return 0

    cfg = load_config(Path(args.config))
    base = Path(args.config).resolve().parents[1]
    df = load_table((base / cfg["input_csv"]).resolve(), cfg["target_col"], cfg["id_col"])
    seeds = parse_seeds(args.seeds, None, int(cfg["random_state"]))
    if not tasks_csv.exists():
        build_tasks(df, cfg, seeds).to_csv(tasks_csv, index=False)
    if args.build_tasks_only:
        print(json.dumps({"tasks_csv": str(tasks_csv), "n_tasks": int(len(pd.read_csv(tasks_csv)))}, indent=2))
        return 0

    tasks = pd.read_csv(tasks_csv)
    shard = tasks.iloc[args.shard_index :: args.num_shards].copy()
    out_jsonl = mon / f"xgboost_shard_{args.shard_index}_of_{args.num_shards}.jsonl"
    done = read_done_global(mon) if args.resume else set()
    completed = 0
    for _, row in shard.iterrows():
        if args.limit and completed >= args.limit:
            break
        task_id = str(row["task_id"])
        if task_id in done:
            continue
        rec = run_one(df, cfg, row, args, run_dir, args.device)
        append_jsonl(out_jsonl, rec)
        completed += 1
        print(
            json.dumps(
                {
                    "completed": completed,
                    "task_id": task_id,
                    "status": rec.get("status"),
                    "analysis": rec.get("analysis"),
                    "task_kind": rec.get("task_kind"),
                    "scenario": rec.get("scenario"),
                    "seed": rec.get("seed"),
                    "AUC": rec.get("AUC"),
                    "RMSE": rec.get("RMSE"),
                    "elapsed_sec": rec.get("elapsed_sec"),
                    "error": str(rec.get("error_message", ""))[:240],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
