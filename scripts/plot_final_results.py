#!/usr/bin/env python3
"""Generate reproducible boxplots and importance heatmaps from final tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tfms_aspergillus.results import load_importance_table, load_metrics_table


TASK_METRIC = {
    "classification": "AUC",
    "regression_18_positive": "RMSE",
    "regression_50_all_samples": "RMSE",
}


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_boxplots(metrics: pd.DataFrame, output: Path, no_title: bool) -> None:
    for task, metric in TASK_METRIC.items():
        frame = metrics[(metrics["task"] == task) & (metrics["metric"] == metric)].copy()
        if frame.empty:
            continue
        scenarios = list(frame["scenario"].drop_duplicates())
        models = list(frame["model"].drop_duplicates())
        positions, values, labels = [], [], []
        width = 0.72 / max(1, len(models))
        for i, scenario in enumerate(scenarios):
            for j, model in enumerate(models):
                value = frame.loc[(frame["scenario"] == scenario) & (frame["model"] == model), "value"].dropna()
                if value.empty:
                    continue
                positions.append(i + (j - (len(models) - 1) / 2) * width)
                values.append(value.to_numpy())
                labels.append(model)
        fig, ax = plt.subplots(figsize=(max(10, len(scenarios) * 2.4), 6))
        bp = ax.boxplot(values, positions=positions, widths=width * 0.9, patch_artist=True, manage_ticks=False)
        palette = plt.get_cmap("tab10")
        for patch, label in zip(bp["boxes"], labels):
            patch.set_facecolor(palette(models.index(label) % 10))
            patch.set_alpha(0.75)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios)
        ax.set_xlabel("Training scheme")
        ax.set_ylabel(metric)
        if not no_title:
            ax.set_title(f"{task} | {metric}")
        handles = [plt.Line2D([0], [0], color=palette(i % 10), lw=8, alpha=0.75) for i in range(len(models))]
        ax.legend(handles, models, title="Model", loc="best")
        save(fig, output / "boxplots" / f"{task}__{metric}")


def heatmap(matrix: pd.DataFrame, stem: Path, label: str, no_title: bool) -> None:
    if matrix.empty:
        return
    fig_h = max(5.5, 0.28 * len(matrix))
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * matrix.shape[1]), fig_h))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=65, ha="right", fontsize=8)
    ax.set_xlabel("Task | scenario")
    ax.set_ylabel("Predictor")
    if not no_title:
        ax.set_title(label)
    fig.colorbar(image, ax=ax, label=label)
    save(fig, stem)


def plot_importance(table: pd.DataFrame, output: Path, model: str, top_n: int, no_title: bool) -> None:
    model_output = output / "heatmaps" / model
    model_output.mkdir(parents=True, exist_ok=True)
    table = table.copy()
    table["config"] = table["family"].astype(str) + " | " + table["task"].astype(str) + " | " + table["scenario"].astype(str)
    grouped = table.groupby(["config", "variable"], as_index=False)["importance"].mean()
    top_by_config = grouped.sort_values(["config", "importance"], ascending=[True, False]).groupby("config", sort=False).head(top_n)
    variables = top_by_config.groupby("variable")["importance"].agg(["count", "mean"]).sort_values(["count", "mean"], ascending=False).index.tolist()
    matrix = grouped.pivot(index="variable", columns="config", values="importance").reindex(index=variables).fillna(0)
    matrix = matrix.loc[:, sorted(matrix.columns)]
    maxima = matrix.max(axis=0).replace(0, 1)
    matrix = matrix.divide(maxima, axis=1)
    matrix.to_csv(model_output / f"matrix_top{top_n}_column_max.csv")
    heatmap(matrix, output / "heatmaps" / model / f"heatmap_top{top_n}_column_max", f"{model} predictor importance", no_title)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shap-table", type=Path)
    parser.add_argument("--pysr-table", type=Path)
    parser.add_argument("--models", nargs="+", default=["random_forest", "xgboost", "tabpfn", "tabiclv2", "pysr"])
    parser.add_argument("--top-n", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--no-title", action="store_true")
    args = parser.parse_args()
    metrics = load_metrics_table(args.metrics_table)
    plot_boxplots(metrics, args.output_dir, args.no_title)
    for model in args.models:
        source = args.pysr_table if model.lower() == "pysr" else args.shap_table
        if source and source.exists():
            table = load_importance_table(source, model=model)
            for top_n in args.top_n:
                plot_importance(table, args.output_dir, model, top_n, args.no_title)


if __name__ == "__main__":
    main()
