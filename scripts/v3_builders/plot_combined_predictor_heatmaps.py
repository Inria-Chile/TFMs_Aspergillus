#!/usr/bin/env python3
"""Generate column-filtered integrated importance heatmaps for validated V3 Aspergillus workflow.

This runner reuses the project's existing loader and ranking logic, merges the
latest SHAP consolidation over the reference consolidation, and retains only scenario
C (Environment + Microbiome) for each preprocessing family.  The result is
therefore nine columns: three task blocks x three preprocessing families.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIO = "C_environment_microbiome"
TASK_ORDER = ["classification", "regression_18_positive", "regression_50_all_samples"]
TASK_LABEL = {
    "classification": "Classification",
    "regression_18_positive": "Regression 18",
    "regression_50_all_samples": "Regression 50",
}
MODEL_ORDER = ["pysr", "random_forest", "xgboost", "tabpfn", "tabiclv2"]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("project_heatmap", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def merge_shap(reference_path: Path, latest_path: Path, output_path: Path) -> Path:
    reference = pd.read_csv(reference_path)
    latest = pd.read_csv(latest_path)
    if "feature_display" in reference.columns and "feature_display_name" not in reference.columns:
        reference = reference.rename(columns={"feature_display": "feature_display_name"})
    key = ["family", "task", "scenario", "model", "feature"]
    required = key + ["mean_abs_shap", "sd_abs_shap", "n_seeds"]
    missing = [c for c in required if c not in reference.columns or c not in latest.columns]
    if missing:
        raise ValueError(f"SHAP source missing columns: {missing}")
    # Keep the latest consolidated value for a key and use the historical
    # backfill only where the latest consolidation has no row.
    latest_keys = set(map(tuple, latest[key].astype(str).itertuples(index=False, name=None)))
    reference_fallback = reference[
        ~reference[key].astype(str).apply(tuple, axis=1).isin(latest_keys)
    ].copy()
    columns = sorted(set(reference.columns) | set(latest.columns))
    merged = pd.concat(
        [latest.reindex(columns=columns), reference_fallback.reindex(columns=columns)],
        ignore_index=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return output_path


def configs_for_scenario(project, scenario: str):
    # ConfigInfo does not retain scenario as a field; its name contains it.
    return [c for c in project.expected_configs() if f"| {scenario} |" in c.config_name]


def build_matrix(project, aggregated: pd.DataFrame, variables: list[str], configs):
    order = [c.config_name for c in configs]
    matrix = aggregated.pivot_table(
        index="variable", columns="config_name", values="importance_mean", aggfunc="mean"
    )
    return matrix.reindex(index=variables, columns=order).fillna(0.0)


def plot_matrix(project, matrix: pd.DataFrame, configs, path: Path, model: str):
    plotted = project.normalize_matrix(matrix, "column_max")
    plotted = plotted.loc[project.cluster_row_order(plotted)]
    labels = {c.config_name: c.config_label for c in configs}
    xlabels = [labels[c] for c in plotted.columns]
    n_rows, n_cols = plotted.shape
    figsize = (max(14.5, n_cols * 0.72), max(10.0, min(90.0, n_rows * 0.33 + 3.8)))
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    image = ax.imshow(plotted.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(xlabels, rotation=55, ha="right", fontsize=13.6)
    ax.set_yticks(np.arange(n_rows))
    base_y_font = max(4.5, min(7.0, 180 / max(n_rows, 1)))
    ax.set_yticklabels(plotted.index, fontsize=base_y_font * 3.0)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # The retained C columns are ordered as three families within each task.
    # Draw task separators and bottom brackets without assuming 27 columns.
    for x in [2.5, 5.5]:
        ax.axvline(x, color="white", linewidth=1.8, alpha=0.95)
    trans = ax.get_xaxis_transform()
    if n_rows <= 25:
        # Top-3 unions are compact; move the task brackets farther below the
        # single-line column labels to prevent any visual overlap.
        bracket_y = -0.645
    elif n_rows <= 40:
        bracket_y = -0.430
    elif n_rows <= 70:
        bracket_y = -0.245
    else:
        bracket_y = -0.155
    for start, end, label in [(0, 2, "Classification"), (3, 5, "Regression 18"), (6, 8, "Regression 50")]:
        y = bracket_y
        ax.plot([start - 0.45, end + 0.45], [y, y], transform=trans, color="#222222", lw=1.1, clip_on=False)
        ax.plot([start - 0.45, start - 0.45], [y, y + 0.014], transform=trans, color="#222222", lw=1.1, clip_on=False)
        ax.plot([end + 0.45, end + 0.45], [y, y + 0.014], transform=trans, color="#222222", lw=1.1, clip_on=False)
        ax.text((start + end) / 2, y - 0.016, label, transform=trans, ha="center", va="top", fontsize=11, fontweight="bold", clip_on=False)

    cbar = fig.colorbar(image, ax=ax, fraction=0.022, pad=0.015)
    label = "PySR predictor importance" if model == "pysr" else "Mean absolute SHAP value"
    cbar.set_label(f"{label} (column_max)", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(path.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plotted


def process_model(project, model: str, args, shap_summary: Path, output_root: Path):
    taxa = args.taxa_dictionary
    if model == "pysr":
        tidy = project.load_importance_results(
            input_summary=args.pysr_summary,
            reg50_summary=args.pysr_reg50_summary,
            taxa_dictionary=taxa,
            model_name="pysr",
        )
    else:
        tidy = project.load_importance_results(
            input_summary=shap_summary,
            taxa_dictionary=taxa,
            model_name=model,
        )
    tidy = tidy[tidy["scenario"].eq(SCENARIO)].copy()
    if tidy.empty:
        raise ValueError(f"No combined-predictor rows found for {model}")
    aggregated = project.aggregate_importances(tidy, aggregation="mean")
    configs = configs_for_scenario(project, SCENARIO)
    expected = {c.config_name for c in configs}
    found = set(aggregated["config_name"].unique())
    missing = sorted(expected - found)
    model_dir = output_root / model
    tables = model_dir / "tables"
    figures = model_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(tables / "tidy_importance_results_combined_predictor.csv", index=False)
    aggregated.to_csv(tables / "aggregated_importances_combined_predictor.csv", index=False)
    selected_meta = {}
    output_files = []
    for top_k in args.top_k:
        selected = project.select_union_top_variables(aggregated, top_k=top_k, ranking_metric="importance_mean")
        (tables / f"selected_variables_top{top_k}.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
        matrix = build_matrix(project, aggregated, selected, configs)
        raw_path = tables / f"matrix_top{top_k}_raw_values.csv"
        matrix.to_csv(raw_path)
        plotted = project.normalize_matrix(matrix, "column_max")
        plotted.to_csv(tables / f"matrix_top{top_k}_column_max.csv")
        plotted_order = plot_matrix(project, matrix, configs, figures / f"heatmap_top{top_k}_column_max", model)
        plotted_order.to_csv(tables / f"matrix_top{top_k}_column_max_plotted_order.csv")
        selected_meta[str(top_k)] = {"n_variables": len(selected), "selected_variables": selected}
        output_files.extend(str(p) for p in [raw_path, figures / f"heatmap_top{top_k}_column_max.pdf", figures / f"heatmap_top{top_k}_column_max.png", figures / f"heatmap_top{top_k}_column_max.svg"])
    metadata = {
        "created_at": datetime.now().isoformat(),
        "model": model,
        "scenario_filter": SCENARIO,
        "retained_predictor_set": "Environment + Microbiome",
        "retained_columns": 9,
        "top_k": args.top_k,
        "aggregation": "mean",
        "normalization": "column_max",
        "fill_missing": 0,
        "n_configurations_found": int(aggregated["config_name"].nunique()),
        "n_configurations_expected": len(configs),
        "missing_configurations": missing,
        "input_summary": str(args.pysr_summary if model == "pysr" else shap_summary),
        "pysr_reg50_summary": str(args.pysr_reg50_summary) if model == "pysr" else None,
        "selected_variables": selected_meta,
        "output_files": output_files,
    }
    (model_dir / "heatmap_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"model": model, "rows": len(aggregated), "configs": len(found), "missing": missing}, indent=2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--heatmap-module", type=Path, required=True)
    p.add_argument("--reference-shap", type=Path, required=True)
    p.add_argument("--latest-shap", type=Path, required=True)
    p.add_argument("--pysr-summary", type=Path, required=True)
    p.add_argument("--pysr-reg50-summary", type=Path, required=True)
    p.add_argument("--taxa-dictionary", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--top-k", nargs="+", type=int, default=[3, 5, 10])
    return p.parse_args()


def main():
    args = parse_args()
    project = load_module(args.heatmap_module)
    merged = merge_shap(args.reference_shap, args.latest_shap, args.output_dir / "tables" / "merged_latest_shap_mean_abs_by_feature.csv")
    for model in MODEL_ORDER:
        process_model(project, model, args, merged, args.output_dir)


if __name__ == "__main__":
    main()
