#!/usr/bin/env python3
"""Build integrated PySR predictor-usage heatmaps across 27 task configurations.

This script consumes the predictor-usage summaries used by the PySR barplots,
optionally replaces regression-50 summaries with the corrected 100-seed table,
and exports one heatmap per top-k/normalization combination.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAMILY_ORDER = [
    "family1_no_clr_microbiome_raw",
    "family2_full_microbiome_clr_raw_env",
    "family3_predefined_subset_clr_raw_env",
]

FAMILY_TO_SCHEME = {
    "family1_no_clr_microbiome_raw": "without_clr",
    "family2_full_microbiome_clr_raw_env": "with_clr",
    "family3_predefined_subset_clr_raw_env": "subset_with_clr",
}

FAMILY_TITLE = {
    "family1_no_clr_microbiome_raw": "Without CLR",
    "family2_full_microbiome_clr_raw_env": "With CLR",
    "family3_predefined_subset_clr_raw_env": "Subset with CLR",
}

TASK_ORDER = ["classification", "regression_18_positive", "regression_50_all_samples"]

TASK_TITLE = {
    "classification": "Classification",
    "regression_18_positive": "Regression 18 samples",
    "regression_50_all_samples": "Regression 50 samples",
}

SCENARIO_ORDER = ["A_environment", "B_microbiome", "C_environment_microbiome"]

SCENARIO_TO_PREDICTOR = {
    "A_environment": "Environment",
    "B_microbiome": "Microbiome",
    "C_environment_microbiome": "Environment + Microbiome",
}


@dataclass(frozen=True)
class ConfigInfo:
    family: str
    task_type: str
    predictor_type: str
    training_scheme: str
    config_name: str
    config_label: str


def expected_configs() -> list[ConfigInfo]:
    configs: list[ConfigInfo] = []
    for task in TASK_ORDER:
        for family in FAMILY_ORDER:
            for scenario in SCENARIO_ORDER:
                predictor = SCENARIO_TO_PREDICTOR[scenario]
                scheme = FAMILY_TO_SCHEME[family]
                config_name = f"{task} | {scenario} | {scheme}"
                short_task = TASK_TITLE[task].replace(" samples", "")
                # Compact per-column label: task name is shown once per
                # 9-column block above the heatmap.
                config_label = f"{predictor}\n{FAMILY_TITLE[family]}"
                configs.append(
                    ConfigInfo(
                        family=family,
                        task_type=task,
                        predictor_type=predictor,
                        training_scheme=scheme,
                        config_name=config_name,
                        config_label=config_label,
                    )
                )
    return configs


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_taxa_names(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path, sep=None, engine="python")
    mapping: dict[str, str] = {}
    if {"anonymous_name", "original_name"}.issubset(df.columns):
        for _, row in df.dropna(subset=["anonymous_name", "original_name"]).iterrows():
            anon = str(row["anonymous_name"]).strip()
            original = str(row["original_name"]).strip().strip('"')
            if anon and original:
                mapping[anon] = f"{anon}_{original}"
    return mapping


def display_feature(feature: str, feature_display: str | None, taxa: dict[str, str]) -> str:
    if feature_display and str(feature_display).strip() and str(feature_display).strip() != "nan":
        value = str(feature_display).strip()
        if re.match(r"^[BF]\d+_.+", value):
            return value
    else:
        value = str(feature).strip()
    match = re.search(r"(?<![A-Za-z0-9])([BF]\d+)(?![A-Za-z0-9])", value)
    if match and match.group(1) in taxa:
        code = match.group(1)
        value = value[: match.start(1)] + taxa[code] + value[match.end(1) :]
    return value


def load_importance_results(
    input_summary: Path,
    model_name: str = "pysr",
    reg50_summary: Path | None = None,
    taxa_dictionary: Path | None = None,
    task_types: set[str] | None = None,
    predictor_types: set[str] | None = None,
    training_schemes: set[str] | None = None,
) -> pd.DataFrame:
    """Load PySR predictor usage summaries into a standardized tidy table."""
    base = read_table(input_summary)
    if reg50_summary:
        reg50 = read_table(reg50_summary)
        base = base[base["task"].astype(str) != "regression_50_all_samples"].copy()
        base = pd.concat([base, reg50], ignore_index=True)

    required = {
        "family",
        "task",
        "scenario",
        "feature",
        "mean_use_rate",
        "median_use_rate",
        "sd_use_rate",
        "n_seed_equations",
        "n_seeds_using_feature",
    }
    missing = sorted(required - set(base.columns))
    if missing:
        raise ValueError(f"Input summary is missing required columns: {missing}")

    taxa = load_taxa_names(taxa_dictionary)
    configs = {(c.family, c.task_type, c.predictor_type, c.training_scheme): c for c in expected_configs()}
    scenario_to_predictor = SCENARIO_TO_PREDICTOR

    rows = []
    for _, row in base.iterrows():
        family = str(row["family"])
        task = str(row["task"])
        scenario = str(row["scenario"])
        if family not in FAMILY_TO_SCHEME or task not in TASK_TITLE or scenario not in scenario_to_predictor:
            continue
        predictor = scenario_to_predictor[scenario]
        scheme = FAMILY_TO_SCHEME[family]
        if task_types and task not in task_types:
            continue
        if predictor_types and predictor not in predictor_types:
            continue
        if training_schemes and scheme not in training_schemes:
            continue
        cfg = configs[(family, task, predictor, scheme)]
        feature = str(row["feature"])
        feature_display = row.get("feature_display_name", None)
        rows.append(
            {
                "model": model_name,
                "task_type": task,
                "predictor_type": predictor,
                "training_scheme": scheme,
                "family": family,
                "scenario": scenario,
                "config_name": cfg.config_name,
                "config_label": cfg.config_label,
                "replicate": pd.NA,
                "variable": display_feature(feature, feature_display, taxa),
                "variable_original": feature,
                "importance_value": float(row["mean_use_rate"]),
                "importance_mean": float(row["mean_use_rate"]),
                "importance_median": float(row["median_use_rate"]),
                "importance_std": float(row["sd_use_rate"]) if not pd.isna(row["sd_use_rate"]) else np.nan,
                "n_replicates_present": int(row["n_seed_equations"]),
                "n_seeds_using_feature": int(row["n_seeds_using_feature"]),
                "frequency_present": float(row["seed_presence_rate"]) if "seed_presence_rate" in row else np.nan,
                "source": str(row.get("source", "")),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No importance rows remained after filtering.")
    return out


def aggregate_importances(df: pd.DataFrame, aggregation: str = "mean") -> pd.DataFrame:
    """Return one row per configuration-variable.

    PySR summary inputs are already seed-aggregated, but this function preserves
    the requested API and safely re-aggregates if duplicate rows are present.
    """
    if aggregation not in {"mean", "median"}:
        raise ValueError("aggregation must be 'mean' or 'median'")
    group_cols = [
        "model",
        "task_type",
        "predictor_type",
        "training_scheme",
        "family",
        "scenario",
        "config_name",
        "config_label",
        "variable",
        "variable_original",
    ]
    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_median=("importance_median", "median"),
            importance_std=("importance_std", "mean"),
            n_replicates_present=("n_replicates_present", "max"),
            n_seeds_using_feature=("n_seeds_using_feature", "max"),
            frequency_present=("frequency_present", "max"),
            source=("source", lambda x: ";".join(sorted(set(map(str, x))))),
        )
        .reset_index()
    )
    return agg


def select_union_top_variables(
    aggregated_df: pd.DataFrame,
    top_k: int = 30,
    ranking_metric: str = "importance_mean",
) -> list[str]:
    if ranking_metric not in aggregated_df.columns:
        raise ValueError(f"ranking_metric not found: {ranking_metric}")
    selected: set[str] = set()
    rank_rows = []
    for cfg, sub in aggregated_df.groupby("config_name", sort=False):
        ranked = sub.sort_values(ranking_metric, ascending=False).head(top_k).copy()
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            selected.add(row["variable"])
            rank_rows.append(
                {
                    "variable": row["variable"],
                    "config_name": cfg,
                    "rank": rank,
                    ranking_metric: row[ranking_metric],
                }
            )
    rank_df = pd.DataFrame(rank_rows)
    global_df = (
        aggregated_df[aggregated_df["variable"].isin(selected)]
        .groupby("variable")
        .agg(global_importance=(ranking_metric, "mean"))
        .reset_index()
    )
    freq_df = rank_df.groupby("variable").size().reset_index(name="topk_config_frequency")
    ordering = freq_df.merge(global_df, on="variable", how="left")
    ordering = ordering.sort_values(["topk_config_frequency", "global_importance", "variable"], ascending=[False, False, True])
    return ordering["variable"].tolist()


def normalize_matrix(matrix: pd.DataFrame, normalize: str) -> pd.DataFrame:
    if normalize in {"raw", "none", "None", ""}:
        return matrix.copy()
    mat = matrix.astype(float).copy()
    if normalize == "column_max":
        denom = mat.max(axis=0).replace(0, np.nan)
        return mat.div(denom, axis=1).fillna(0)
    if normalize == "column_sum":
        denom = mat.sum(axis=0).replace(0, np.nan)
        return mat.div(denom, axis=1).fillna(0)
    if normalize == "zscore_rows":
        mean = mat.mean(axis=1)
        std = mat.std(axis=1).replace(0, np.nan)
        return mat.sub(mean, axis=0).div(std, axis=0).fillna(0)
    if normalize == "rank":
        ranked = mat.rank(axis=0, method="average", ascending=True)
        ranked = ranked.max(axis=0) - ranked + 1
        return ranked.fillna(0)
    raise ValueError(f"Unknown normalization: {normalize}")


def build_heatmap_matrix(
    aggregated_df: pd.DataFrame,
    selected_variables: list[str],
    value_metric: str = "importance_mean",
    fill_missing: str | float = 0,
) -> pd.DataFrame:
    configs = expected_configs()
    config_order = [c.config_name for c in configs]
    pivot = aggregated_df.pivot_table(
        index="variable",
        columns="config_name",
        values=value_metric,
        aggfunc="mean",
    )
    pivot = pivot.reindex(index=selected_variables, columns=config_order)
    if str(fill_missing).lower() == "nan":
        return pivot
    return pivot.fillna(float(fill_missing))


def cluster_row_order(matrix: pd.DataFrame) -> list[str]:
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import pdist
    except Exception as exc:  # pragma: no cover - depends on env
        warnings.warn(f"scipy unavailable; row clustering skipped: {exc}")
        return list(matrix.index)
    if matrix.shape[0] <= 2:
        return list(matrix.index)
    values = matrix.fillna(0).to_numpy(dtype=float)
    if np.allclose(values, values[0]):
        return list(matrix.index)
    dist = pdist(values, metric="euclidean")
    if np.allclose(dist, 0):
        return list(matrix.index)
    order = leaves_list(linkage(dist, method="average"))
    return list(matrix.index[order])


def plot_importance_heatmap(
    matrix_df: pd.DataFrame,
    output_path: Path,
    title: str,
    normalize: str = "raw",
    cluster_rows: bool = True,
    cluster_cols: bool = False,
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis",
) -> pd.DataFrame:
    plot_matrix = normalize_matrix(matrix_df, normalize)
    if cluster_rows:
        plot_matrix = plot_matrix.loc[cluster_row_order(plot_matrix)]
    if cluster_cols:
        cols_order = cluster_row_order(plot_matrix.T)
        plot_matrix = plot_matrix.loc[:, cols_order]

    labels = {c.config_name: c.config_label for c in expected_configs()}
    xlabels = [labels.get(c, c) for c in plot_matrix.columns]

    n_rows, n_cols = plot_matrix.shape
    if figsize is None:
        # The row labels are intentionally large for readability in exported
        # PDFs, so the figure height scales more aggressively with n_rows.
        figsize = (max(14.5, n_cols * 0.62), max(10.0, min(90.0, n_rows * 0.33 + 3.8)))
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    data = plot_matrix.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(xlabels, rotation=55, ha="right", fontsize=6.8)
    ax.set_yticks(np.arange(n_rows))
    base_y_fontsize = max(4.5, min(7.0, 180 / max(n_rows, 1)))
    ax.set_yticklabels(plot_matrix.index, fontsize=base_y_fontsize * 3.0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    if title:
        ax.set_title(title, fontsize=11)
    # Draw separators after every 3 columns and every 9 columns.
    for x in [2.5, 5.5, 8.5, 11.5, 14.5, 17.5, 20.5, 23.5]:
        ax.axvline(x, color="white", linewidth=0.6, alpha=0.8)
    for x in [8.5, 17.5]:
        ax.axvline(x, color="white", linewidth=1.8, alpha=0.95)
    # Task group labels and bracket-like horizontal lines below the rotated
    # column labels. The vertical ticks point upward, like a 180-degree
    # rotation of the former top brackets.
    task_groups = [
        (0, 8, "Classification"),
        (9, 17, "Regression 18"),
        (18, 26, "Regression 50"),
    ]
    trans = ax.get_xaxis_transform()
    for start, end, label in task_groups:
        y = -0.080
        ax.plot([start - 0.45, end + 0.45], [y, y], transform=trans, color="#222222", lw=1.1, clip_on=False)
        ax.plot([start - 0.45, start - 0.45], [y, y + 0.018], transform=trans, color="#222222", lw=1.1, clip_on=False)
        ax.plot([end + 0.45, end + 0.45], [y, y + 0.018], transform=trans, color="#222222", lw=1.1, clip_on=False)
        ax.text((start + end) / 2, y - 0.014, label, transform=trans, ha="center", va="top", fontsize=11, fontweight="bold", clip_on=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.015)
    cbar.set_label(f"PySR predictor importance ({normalize})", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(output_path.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-summary", required=True, type=Path)
    parser.add_argument("--reg50-summary", type=Path)
    parser.add_argument("--taxa-dictionary", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-name", default="pysr")
    parser.add_argument("--top-k", nargs="+", type=int, default=[30])
    parser.add_argument("--aggregation", choices=["mean", "median"], default="mean")
    parser.add_argument("--normalize", nargs="+", default=["raw"])
    parser.add_argument("--fill-missing", default="0")
    parser.add_argument("--cluster-rows", default="true")
    parser.add_argument("--cluster-cols", default="false")
    parser.add_argument("--ranking-metric", default="importance_mean")
    parser.add_argument("--value-metric", default="importance_mean")
    parser.add_argument("--cmap", default="viridis")
    return parser.parse_args()


def str_to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir / args.model_name
    tables = out_dir / "tables"
    figures = out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    tidy = load_importance_results(
        input_summary=args.input_summary,
        reg50_summary=args.reg50_summary,
        taxa_dictionary=args.taxa_dictionary,
        model_name=args.model_name,
    )
    tidy.to_csv(tables / "tidy_importance_results.csv", index=False)
    aggregated = aggregate_importances(tidy, aggregation=args.aggregation)
    aggregated.to_csv(tables / "aggregated_importances.csv", index=False)

    expected = [c.config_name for c in expected_configs()]
    found = sorted(aggregated["config_name"].unique())
    missing_configs = [c for c in expected if c not in found]
    if missing_configs:
        warnings.warn(f"Missing expected configurations: {missing_configs}")

    output_files: list[str] = []
    selected_meta = {}
    for top_k in args.top_k:
        selected = select_union_top_variables(aggregated, top_k=top_k, ranking_metric=args.ranking_metric)
        selected_path = tables / f"selected_variables_top{top_k}.txt"
        selected_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
        output_files.append(str(selected_path))
        selected_meta[str(top_k)] = {"n_variables": len(selected), "selected_variables": selected}

        matrix = build_heatmap_matrix(
            aggregated,
            selected,
            value_metric=args.value_metric,
            fill_missing=np.nan if str(args.fill_missing).lower() == "nan" else float(args.fill_missing),
        )
        raw_matrix_path = tables / f"matrix_top{top_k}_raw_values.csv"
        matrix.to_csv(raw_matrix_path)
        output_files.append(str(raw_matrix_path))

        for norm in args.normalize:
            norm_label = "raw" if norm in {"none", "None"} else norm
            plot_matrix = normalize_matrix(matrix, norm_label)
            matrix_path = tables / f"matrix_top{top_k}_{norm_label}.csv"
            plot_matrix.to_csv(matrix_path)
            output_files.append(str(matrix_path))
            stem = f"heatmap_top{top_k}_{norm_label}"
            title = ""
            plotted_matrix = plot_importance_heatmap(
                matrix,
                figures / stem,
                title=title,
                normalize=norm_label,
                cluster_rows=str_to_bool(args.cluster_rows),
                cluster_cols=str_to_bool(args.cluster_cols),
                cmap=args.cmap,
            )
            plotted_matrix.to_csv(tables / f"matrix_top{top_k}_{norm_label}_plotted_order.csv")
            output_files.extend(
                str((figures / stem).with_suffix(suffix)) for suffix in [".png", ".pdf", ".svg"]
            )

    metadata = {
        "created_at": datetime.now().isoformat(),
        "model": args.model_name,
        "top_k": args.top_k,
        "aggregation": args.aggregation,
        "normalization": args.normalize,
        "fill_missing": args.fill_missing,
        "ranking_metric": args.ranking_metric,
        "value_metric": args.value_metric,
        "n_configurations_found": int(aggregated["config_name"].nunique()),
        "n_configurations_expected": 27,
        "missing_configurations": missing_configs,
        "task_types": TASK_ORDER,
        "predictor_types": list(SCENARIO_TO_PREDICTOR.values()),
        "training_schemes": [FAMILY_TO_SCHEME[f] for f in FAMILY_ORDER],
        "input_summary": str(args.input_summary),
        "reg50_summary": str(args.reg50_summary) if args.reg50_summary else None,
        "taxa_dictionary": str(args.taxa_dictionary) if args.taxa_dictionary else None,
        "selected_variables": selected_meta,
        "output_files": output_files,
    }
    meta_path = out_dir / "heatmap_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
