"""Load, validate, and aggregate final and seed-level result tables."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

METRIC_COLUMNS = ("AUC", "F1", "balanced_accuracy", "MAE", "RMSE", "R2")
REQUIRED_LONG_COLUMNS = {"family", "task", "scenario", "model", "seed", "metric", "value"}


def _canonical_task(value: str) -> str:
    value = str(value).strip().lower()
    return {
        "occurrence": "classification",
        "classification": "classification",
        "regression_18": "regression_18_positive",
        "regression_18_positive_samples": "regression_18_positive",
        "regression_18_positive": "regression_18_positive",
        "regression_50": "regression_50_all_samples",
        "regression_50_all_samples": "regression_50_all_samples",
    }.get(value, value)


def load_metrics_table(path: str | Path) -> pd.DataFrame:
    """Read the canonical long metric table and validate its contract."""
    frame = pd.read_csv(path)
    missing = REQUIRED_LONG_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"metrics table is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["task"] = frame["task"].map(_canonical_task)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    if frame.empty or frame["value"].notna().sum() == 0:
        raise ValueError("metrics table contains no numeric values")
    return frame


def _seed_from_path(path: Path) -> int | None:
    match = re.search(r"seed_(\d+)", path.name)
    return int(match.group(1)) if match else None


def _raw_file_to_long(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.copy()
    if "metric" in frame.columns and "value" in frame.columns:
        long = frame
    else:
        metric_cols = [c for c in METRIC_COLUMNS if c in frame.columns]
        if not metric_cols:
            raise ValueError(f"no supported metric columns in {path}")
        id_columns = [c for c in frame.columns if c not in metric_cols]
        long = frame.melt(id_vars=id_columns, value_vars=metric_cols, var_name="metric", value_name="value")
    if "seed" not in long.columns:
        seed = _seed_from_path(path)
        if seed is None:
            raise ValueError(f"seed is missing from {path}")
        long["seed"] = seed
    required_context = {"family", "task", "scenario", "model"}
    missing = required_context - set(long.columns)
    if missing:
        raise ValueError(f"{path} is missing context columns: {sorted(missing)}")
    long["source_file"] = str(path)
    return long


def aggregate_raw_metrics(raw_root: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Aggregate raw ``seed_*_metrics.csv`` files into the canonical table."""
    paths = sorted(Path(raw_root).glob("**/seed_*_metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"no seed metric files found below {raw_root}")
    frames = [_raw_file_to_long(path) for path in paths]
    result = pd.concat(frames, ignore_index=True)
    result["task"] = result["task"].map(_canonical_task)
    key = ["family", "task", "scenario", "model", "seed", "metric"]
    duplicates = result.duplicated(key, keep=False)
    if duplicates.any():
        examples = result.loc[duplicates, key].head(5).to_dict("records")
        raise ValueError(f"duplicate result keys detected: {examples}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def load_importance_table(path: str | Path, model: str | None = None) -> pd.DataFrame:
    """Load either the final SHAP summary or PySR importance summary."""
    frame = pd.read_csv(path)
    if "mean_abs_shap" in frame.columns:
        required = {"family", "task", "scenario", "model", "feature", "mean_abs_shap"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"SHAP table is missing columns: {sorted(missing)}")
        frame = frame.rename(columns={"feature": "variable", "mean_abs_shap": "importance"})
    elif "mean_relative_frequency" in frame.columns:
        required = {"family", "task", "scenario", "feature", "mean_relative_frequency"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"PySR table is missing columns: {sorted(missing)}")
        frame = frame.rename(columns={"feature": "variable", "mean_relative_frequency": "importance"})
        frame["model"] = "pysr"
    elif "importance_mean" in frame.columns:
        required = {"family", "task_type", "scenario", "variable", "importance_mean"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"PySR table is missing columns: {sorted(missing)}")
        frame = frame.rename(columns={"task_type": "task", "importance_mean": "importance"})
        frame["model"] = "pysr"
    else:
        raise ValueError("importance table is neither the final SHAP nor PySR format")
    frame["task"] = frame["task"].map(_canonical_task)
    if model:
        frame = frame[frame["model"].str.lower().eq(model.lower())].copy()
    frame["importance"] = pd.to_numeric(frame["importance"], errors="coerce")
    return frame.dropna(subset=["importance"])
