#!/usr/bin/env python3
"""Seed-level export helpers for the v3 Marta/Omar refactor.

Future model runners should accumulate fold predictions/metrics inside one
seed and write one CSV per seed. Fold-level files may be temporary, but they
should not be part of the public v3 outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def write_seed_metrics_csv(
    rows: Iterable[dict],
    output_csv: str | Path,
    *,
    family: str,
    task: str,
    scenario: str,
    model: str,
    seed: int,
) -> Path:
    """Write one wide metrics CSV for a seed.

    `rows` can be fold-level metric dictionaries or already-averaged metric
    dictionaries. If a `metric`/`value` pair is present, values are averaged
    across folds before writing.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    if df.empty:
        raise ValueError("Cannot write empty seed metrics CSV")
    if {"metric", "value"}.issubset(df.columns):
        wide = df.pivot_table(index=[], columns="metric", values="value", aggfunc="mean").reset_index(drop=True)
    else:
        numeric = df.select_dtypes(include="number")
        wide = pd.DataFrame([numeric.mean(numeric_only=True).to_dict()])
    wide.insert(0, "seed", int(seed))
    wide.insert(0, "model", model)
    wide.insert(0, "scenario", scenario)
    wide.insert(0, "task", task)
    wide.insert(0, "family", family)
    wide.to_csv(output_csv, index=False)
    return output_csv


def write_seed_prediction_csv(
    fold_prediction_frames: Iterable[pd.DataFrame],
    output_csv: str | Path,
    *,
    family: str,
    task: str,
    scenario: str,
    model: str,
    seed: int,
) -> Path:
    """Concatenate fold predictions and write one prediction CSV per seed."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for fold_id, frame in enumerate(fold_prediction_frames, start=1):
        frame = frame.copy()
        frame.insert(0, "fold_id", fold_id)
        frames.append(frame)
    if not frames:
        raise ValueError("Cannot write empty seed prediction CSV")
    out = pd.concat(frames, ignore_index=True, sort=False)
    for col, val in reversed(
        [
            ("family", family),
            ("task", task),
            ("scenario", scenario),
            ("model", model),
            ("seed", int(seed)),
        ]
    ):
        out.insert(0, col, val)
    out.to_csv(output_csv, index=False)
    return output_csv


def write_mean_abs_shap_by_feature(
    shap_frames: Iterable[pd.DataFrame],
    output_csv: str | Path,
    *,
    feature_col: str = "feature",
    shap_value_col: str = "shap_value",
) -> Path:
    """Average absolute SHAP by original predictor name.

    The input feature column must contain the standardized original predictor
    names. For PySR, callers must map symbolic variables back to original names
    before calling this function; do not export X0/X1/etc.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.concat([x.copy() for x in shap_frames], ignore_index=True, sort=False)
    if feature_col not in df or shap_value_col not in df:
        raise ValueError(f"SHAP frames must contain {feature_col!r} and {shap_value_col!r}")
    df["abs_shap"] = df[shap_value_col].abs()
    out = (
        df.groupby(feature_col, as_index=False)
        .agg(mean_abs_shap=("abs_shap", "mean"), sd_abs_shap=("abs_shap", "std"), n_rows=("abs_shap", "size"))
        .sort_values("mean_abs_shap", ascending=False)
    )
    out.to_csv(output_csv, index=False)
    return output_csv
