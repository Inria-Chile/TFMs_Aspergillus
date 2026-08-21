from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ShapConfig:
    background_size: int = 20
    nsamples: int | str = "auto"
    random_state: int = 123
    explain_regression_raw_scale: bool = True


def finite(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def choose_background(X_train: np.ndarray, cfg: ShapConfig) -> np.ndarray:
    X_train = np.asarray(X_train, dtype=np.float32)
    if X_train.ndim != 2:
        raise ValueError("X_train must be a 2D array")
    if len(X_train) <= cfg.background_size:
        return X_train
    rng = np.random.default_rng(cfg.random_state)
    idx = rng.choice(len(X_train), size=cfg.background_size, replace=False)
    return X_train[np.sort(idx)]


def normalize_shap_values(values: Any, task_kind: str) -> np.ndarray:
    if isinstance(values, list):
        arr = np.asarray(values[1] if task_kind == "occurrence" and len(values) > 1 else values[0])
    else:
        arr = np.asarray(values)
    if arr.ndim == 3:
        arr = arr[:, :, 1] if task_kind == "occurrence" and arr.shape[2] > 1 else arr[:, :, 0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.astype(float)


def normalize_expected_value(value: Any, n_rows: int, task_kind: str) -> np.ndarray:
    if isinstance(value, (list, tuple)):
        arr = np.asarray(value[1] if task_kind == "occurrence" and len(value) > 1 else value[0], dtype=float)
    else:
        arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.repeat(float(arr), n_rows)
    if arr.ndim == 1 and len(arr) == n_rows:
        return arr.astype(float)
    return np.repeat(float(np.ravel(arr)[0]), n_rows)


def kernel_shap(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    task_kind: str,
    cfg: ShapConfig,
) -> tuple[np.ndarray, np.ndarray]:
    import shap

    background = choose_background(X_train, cfg)
    link = "identity"
    explainer = shap.KernelExplainer(predict_fn, background, feature_names=feature_names, link=link)
    try:
        values = explainer.shap_values(np.asarray(X_test, dtype=np.float32), nsamples=cfg.nsamples, silent=True)
    except TypeError:
        values = explainer.shap_values(np.asarray(X_test, dtype=np.float32), nsamples=cfg.nsamples)
    shap_values = normalize_shap_values(values, task_kind)
    base_values = normalize_expected_value(explainer.expected_value, len(X_test), task_kind)
    return shap_values, base_values


def tree_shap(model: Any, X_test: np.ndarray, task_kind: str) -> tuple[np.ndarray, np.ndarray]:
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_test)
    shap_values = normalize_shap_values(values, task_kind)
    base_values = normalize_expected_value(explainer.expected_value, len(X_test), task_kind)
    return shap_values, base_values


def xgboost_pred_contribs(model: Any, X_test: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    import xgboost as xgb

    booster = model.get_booster()
    dm = xgb.DMatrix(np.asarray(X_test, dtype=np.float32), feature_names=feature_names)
    contrib = np.asarray(booster.predict(dm, pred_contribs=True), dtype=float)
    if contrib.ndim == 3:
        contrib = contrib[:, :, -1]
    if contrib.ndim == 1:
        contrib = contrib.reshape(1, -1)
    return contrib[:, :-1], contrib[:, -1]


def shap_rows(
    *,
    task_id: str,
    analysis: str,
    task_kind: str,
    scenario: str,
    variant: str,
    model_name: str,
    seed: int,
    fold_id: int,
    row_indices: list[int],
    sample_ids: list[Any],
    feature_names: list[str],
    X_test_df: pd.DataFrame,
    shap_values: np.ndarray,
    base_values: np.ndarray,
    y_true: np.ndarray,
    prediction: np.ndarray,
    prediction_col: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, source_row in enumerate(row_indices):
        for j, variable in enumerate(feature_names):
            rows.append(
                {
                    "task_id": task_id,
                    "analysis": analysis,
                    "task_kind": task_kind,
                    "scenario": scenario,
                    "variant": variant,
                    "model": model_name,
                    "seed": int(seed),
                    "fold_id": int(fold_id),
                    "row_index": int(source_row),
                    "Sample": sample_ids[i],
                    "variable": variable,
                    "feature_value": finite(X_test_df.iloc[i][variable]),
                    "shap_value": finite(shap_values[i, j]),
                    "abs_shap_value": finite(abs(shap_values[i, j])),
                    "base_value": finite(base_values[i]),
                    "y_true": finite(y_true[i]),
                    prediction_col: finite(prediction[i]),
                    "n_features": len(feature_names),
                }
            )
    return rows


def summarize_shap(shap_values: pd.DataFrame) -> pd.DataFrame:
    if shap_values.empty:
        return pd.DataFrame()
    return (
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


def safe_plot_name(value: Any, max_len: int = 160) -> str:
    name = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value)).strip("_")
    return (name[:max_len] or "plot")


def _group_instance_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["task_id"].astype(str)
        + "__row"
        + df["row_index"].astype(str)
        + "__"
        + df["Sample"].astype(str)
    )


def _shap_group_to_matrices(group: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    work = group.copy()
    work["instance_id"] = _group_instance_key(work)
    importance = (
        work.groupby("variable", as_index=False)["abs_shap_value"]
        .mean()
        .sort_values("abs_shap_value", ascending=False)
    )
    variables = importance["variable"].head(top_n).tolist()
    if not variables:
        return pd.DataFrame(), pd.DataFrame(), np.array([]), []
    work = work[work["variable"].isin(variables)].copy()
    shap_matrix = work.pivot_table(index="instance_id", columns="variable", values="shap_value", aggfunc="mean")
    value_matrix = work.pivot_table(index="instance_id", columns="variable", values="feature_value", aggfunc="mean")
    ordered_cols = [v for v in variables if v in shap_matrix.columns]
    shap_matrix = shap_matrix.reindex(columns=ordered_cols).fillna(0.0)
    value_matrix = value_matrix.reindex(index=shap_matrix.index, columns=ordered_cols)
    base_values = (
        work.drop_duplicates("instance_id")
        .set_index("instance_id")
        .reindex(shap_matrix.index)["base_value"]
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    return shap_matrix, value_matrix, base_values, ordered_cols


def plot_shap_group_figures(
    group: pd.DataFrame,
    out_dir: Path,
    label: str,
    *,
    top_n: int = 30,
) -> list[str]:
    if group.empty:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import shap
    except Exception:
        shap = None

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    summary = (
        group.groupby("variable", as_index=False)
        .agg(mean_abs_shap=("abs_shap_value", "mean"), sd_abs_shap=("abs_shap_value", "std"))
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .iloc[::-1]
    )
    if not summary.empty:
        height = max(4.0, 0.32 * len(summary) + 1.4)
        fig, ax = plt.subplots(figsize=(9.5, height))
        ax.barh(summary["variable"], summary["mean_abs_shap"], color="#F0064F", alpha=0.9)
        ax.set_xlabel("mean(|SHAP value|)")
        ax.set_ylabel("Predictor")
        ax.set_title(f"Importancia SHAP promedio - {label}")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        for suffix in [".png", ".pdf"]:
            path = out_dir / f"{safe_plot_name(label)}_shap_bar{suffix}"
            fig.savefig(path, dpi=220 if suffix == ".png" else None, bbox_inches="tight")
            paths.append(str(path))
        plt.close(fig)

    shap_matrix, value_matrix, base_values, feature_names = _shap_group_to_matrices(group, top_n)
    if shap_matrix.empty:
        return paths
    heatmap_width = min(16.0, max(8.0, 0.08 * len(shap_matrix) + 6.0))
    heatmap_height = min(14.0, max(5.5, 0.32 * len(feature_names) + 2.5))
    heat_values = shap_matrix.to_numpy(dtype=float).T
    fig, ax = plt.subplots(figsize=(heatmap_width, heatmap_height))
    vmax = float(np.nanmax(np.abs(heat_values))) if np.isfinite(heat_values).any() else 1.0
    vmax = vmax if vmax > 0 else 1.0
    im = ax.imshow(heat_values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel("Muestras / predicciones evaluadas")
    ax.set_ylabel("Predictores")
    ax.set_title(f"Heatmap SHAP - {label}")
    ax.set_xticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("SHAP value")
    fig.tight_layout()
    for suffix in [".png", ".pdf"]:
        path = out_dir / f"{safe_plot_name(label)}_shap_heatmap{suffix}"
        fig.savefig(path, dpi=220 if suffix == ".png" else None, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)

    if shap is None:
        return paths

    explanation = shap.Explanation(
        values=shap_matrix.to_numpy(dtype=float),
        base_values=base_values,
        data=value_matrix.to_numpy(dtype=float),
        feature_names=feature_names,
    )
    try:
        plt.figure(figsize=(heatmap_width, heatmap_height))
        shap.plots.heatmap(explanation, max_display=len(feature_names), show=False)
        fig = plt.gcf()
        fig.suptitle(f"Heatmap SHAP oficial - {label}", y=1.02)
        for suffix in [".png", ".pdf"]:
            path = out_dir / f"{safe_plot_name(label)}_shap_official_heatmap{suffix}"
            fig.savefig(path, dpi=220 if suffix == ".png" else None, bbox_inches="tight")
            paths.append(str(path))
        plt.close(fig)
    except Exception:
        plt.close("all")

    try:
        beeswarm_height = min(13.0, max(5.5, 0.32 * len(feature_names) + 1.8))
        plt.figure(figsize=(9.5, beeswarm_height))
        shap.plots.beeswarm(explanation, max_display=len(feature_names), show=False)
        fig = plt.gcf()
        fig.suptitle(f"Resumen SHAP - {label}", y=1.02)
        for suffix in [".png", ".pdf"]:
            path = out_dir / f"{safe_plot_name(label)}_shap_beeswarm{suffix}"
            fig.savefig(path, dpi=220 if suffix == ".png" else None, bbox_inches="tight")
            paths.append(str(path))
        plt.close(fig)
    except Exception:
        plt.close("all")
    return paths


def plot_shap_figures(shap_values: pd.DataFrame, out_dir: Path, *, top_n: int = 30) -> pd.DataFrame:
    if shap_values.empty:
        return pd.DataFrame(columns=["analysis", "task_kind", "scenario", "variant", "model", "plot_path"])
    records: list[dict[str, Any]] = []
    group_cols = ["analysis", "task_kind", "scenario", "variant", "model"]
    for keys, group in shap_values.groupby(group_cols, dropna=False):
        label = "__".join(str(k) for k in keys)
        paths = plot_shap_group_figures(group, out_dir, label, top_n=top_n)
        for path in paths:
            records.append(dict(zip(group_cols, keys, strict=False)) | {"plot_path": path})
    return pd.DataFrame(records)
