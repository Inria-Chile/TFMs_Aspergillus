from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


LOG1P_COLS = {
    "built",
    "crops",
    "dist_river_m",
    "dist_sea",
    "dist_urban",
    "elevation_m",
    "flow_len",
    "HBV",
    "NTL",
    "slope_pct",
    "trees",
    "wind_speed_10m",
}
ROBUST_CLIP_COLS = {"curv", "curv_plan", "SOC_0_5cm_pct", "P_PC2"}
ALREADY_LOG_COLS = {"flow_acc_log"}


def environmental_preprocessing_enabled(cfg: dict[str, Any]) -> bool:
    ep = cfg.get("environmental_preprocessing", {})
    return bool(ep.get("enabled", False))


def infer_environmental_cols(full_df: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    id_col = cfg.get("id_col", "Sample")
    if "environmental_cols" in cfg.get("environmental_preprocessing", {}):
        return [str(x) for x in cfg["environmental_preprocessing"]["environmental_cols"]]
    if id_col in full_df.columns:
        return [str(x) for x in full_df.columns[: list(full_df.columns).index(id_col)]]
    return []


def _safe_zscore(train: pd.Series, test: pd.Series) -> tuple[pd.Series, pd.Series]:
    mean = float(train.mean(skipna=True))
    std = float(train.std(skipna=True, ddof=0))
    if not np.isfinite(std) or std <= 0:
        return train * 0.0, test * 0.0
    return (train - mean) / std, (test - mean) / std


def _log1p_or_asinh(train: pd.Series, test: pd.Series) -> tuple[pd.Series, pd.Series]:
    min_train = float(train.min(skipna=True))
    if np.isfinite(min_train) and min_train >= 0:
        return np.log1p(train), np.log1p(test.clip(lower=0))
    return np.arcsinh(train), np.arcsinh(test)


def preprocess_train_test_environmental(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fold-aware environmental preprocessing.

    The transformation parameters are estimated only on X_train_df. Microbiome
    CLR columns and medium dummies are intentionally left unchanged.
    """
    if not environmental_preprocessing_enabled(cfg):
        return X_train_df, X_test_df, {"enabled": False}

    ep = cfg.get("environmental_preprocessing", {})
    dummy_cols = set(map(str, cfg.get("dummy_cols", [])))
    env_cols = set(infer_environmental_cols(full_df, cfg))
    transformed_train = X_train_df.copy()
    transformed_test = X_test_df.copy()
    transformed_cols: list[str] = []
    skipped_cols: list[str] = []
    methods: dict[str, str] = {}

    for col in list(X_train_df.columns):
        if col in dummy_cols or col not in env_cols:
            skipped_cols.append(str(col))
            continue
        train = pd.to_numeric(X_train_df[col], errors="coerce").astype(float)
        test = pd.to_numeric(X_test_df[col], errors="coerce").astype(float)
        method = "zscore"

        if col in ROBUST_CLIP_COLS:
            q_low = float(train.quantile(float(ep.get("winsor_lower_quantile", 0.05))))
            q_high = float(train.quantile(float(ep.get("winsor_upper_quantile", 0.95))))
            if np.isfinite(q_low) and np.isfinite(q_high) and q_high > q_low:
                train = train.clip(lower=q_low, upper=q_high)
                test = test.clip(lower=q_low, upper=q_high)
            method = "winsorize_train_quantiles_then_zscore"
        elif col in LOG1P_COLS:
            train, test = _log1p_or_asinh(train, test)
            method = "log1p_nonnegative_else_asinh_then_zscore"
        elif col in ALREADY_LOG_COLS:
            method = "already_log_then_zscore"

        train, test = _safe_zscore(train, test)
        transformed_train[col] = train
        transformed_test[col] = test
        transformed_cols.append(str(col))
        methods[str(col)] = method

    return (
        transformed_train,
        transformed_test,
        {
            "enabled": True,
            "fit_scope": "training_fold_only",
            "transformed_cols": transformed_cols,
            "skipped_cols": skipped_cols,
            "methods": methods,
        },
    )
