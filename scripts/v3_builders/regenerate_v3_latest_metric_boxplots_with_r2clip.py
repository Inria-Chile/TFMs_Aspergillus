#!/usr/bin/env python3
"""Regenerate V3 metric boxplots from current seed-level metric exports.

This script intentionally reads only the expected V3 result-tree contract:

outputs/<family>/<task>/<scenario>/<model>/seed_metrics/seed_XXX_metrics.csv

It writes a fresh consolidation plus two complementary plot sets:

1. Set 99 style: within one family/task, compare A/B/C schemes.
2. Set 100 style: within one task/scheme, compare preprocessing families.

For regression tasks it additionally writes variants without PySR, and for R2
it writes an extra metric where negative R2 values are clipped to zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


FAMILIES = [
    ("family1_no_clr_microbiome_raw", "Family 1: raw microbiome + raw environment", "family_01", "No CLR preprocessing"),
    (
        "family2_full_microbiome_clr_raw_env",
        "Family 2: full microbiome CLR + raw environment",
        "family_02",
        "Full CLR preprocessing",
    ),
    (
        "family3_predefined_subset_clr_raw_env",
        "Family 3: predefined predictor subset with microbiome CLR + raw environment",
        "family_03",
        "Subset CLR preprocessing",
    ),
]

TASKS = [
    ("classification", "Classification", "Classification", "classification", ["classification"], 15),
    (
        "regression_18_positive",
        "Positive-abundance regression",
        "Regression 18 samples",
        "regression_18_samples",
        ["regression_18_positive", "regression_18_positive_samples", "positive_abundance"],
        18,
    ),
    (
        "regression_50_all_samples",
        "Complete-abundance regression",
        "Regression 50 samples",
        "regression_50_samples",
        ["regression_50_all_samples", "all_sample_abundance"],
        15,
    ),
]

SCENARIOS = [
    ("A_environment", "A", "Environment", "Environment"),
    ("B_microbiome", "B", "Microbiome", "Microbiome"),
    ("C_environment_microbiome", "C", "Environment + Microbiome", "Environment_Microbiome"),
]

MODELS = [
    ("random_forest", "Random Forest", ["random_forest", "rf"]),
    ("xgboost", "XGBoost", ["xgboost"]),
    ("tabpfn", "TFN/TabPFN", ["tabpfn", "tfn_tabpfn", "tfn"]),
    ("tabiclv2", "TabICLv2", ["tabiclv2", "tabicl"]),
    ("pysr", "PySR", ["pysr"]),
]

MODEL_ORDER = [x[0] for x in MODELS]
MODEL_LABEL = {x[0]: x[1] for x in MODELS}
MODEL_COLOR = {
    "random_forest": "#4E79A7",
    "xgboost": "#F28E2B",
    "tabpfn": "#B07AA1",
    "tabiclv2": "#59A14F",
    "pysr": "#E15759",
}

METRICS_BY_TASK = {
    "classification": ["AUC", "F1", "balanced_accuracy"],
    "regression_18_positive": ["MAE", "RMSE", "R2"],
    "regression_50_all_samples": ["MAE", "RMSE", "R2"],
}

METRIC_AXIS = {
    "AUC": "AUC",
    "F1": "F1",
    "balanced_accuracy": "Balanced accuracy",
    "MAE": "MAE",
    "RMSE": "RMSE",
    "R2": "R2",
    "R2_negative_to_zero": "R2 (negative values set to 0)",
}

METRIC_FILE = {
    "AUC": "AUC",
    "F1": "F1",
    "balanced_accuracy": "Balanced_accuracy",
    "MAE": "MAE",
    "RMSE": "RMSE",
    "R2": "R2",
    "R2_negative_to_zero": "R2_negative_to_zero",
}

SCHEME_NAME_FOR_FILE = "schemes_Environment__Microbiome__Environment_Microbiome"


@dataclass(frozen=True)
class ExpectedDir:
    family: str
    family_label: str
    family_short: str
    family_comparison_label: str
    task: str
    task_label_es: str
    task_label_en: str
    task_short: str
    task_dir: str
    expected_folds: int
    scenario: str
    scenario_label: str
    scenario_title: str
    scenario_file: str
    model: str
    model_label: str
    model_dir: str
    seed_metrics_dir: Path


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def progress(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_") or "unknown"


def present_order(values: set[str], order: list[str]) -> list[str]:
    return [item for item in order if item in values]


def expected_dirs(v3_root: Path) -> Iterable[ExpectedDir]:
    for family, family_label, family_short, family_comp_label in FAMILIES:
        for task, task_label_es, task_label_en, task_short, task_dirs, expected_folds in TASKS:
            for scenario, scenario_label, scenario_title, scenario_file in SCENARIOS:
                for model, model_label, model_dirs in MODELS:
                    for task_dir in task_dirs:
                        for model_dir in model_dirs:
                            yield ExpectedDir(
                                family=family,
                                family_label=family_label,
                                family_short=family_short,
                                family_comparison_label=family_comp_label,
                                task=task,
                                task_label_es=task_label_es,
                                task_label_en=task_label_en,
                                task_short=task_short,
                                task_dir=task_dir,
                                expected_folds=expected_folds,
                                scenario=scenario,
                                scenario_label=scenario_label,
                                scenario_title=scenario_title,
                                scenario_file=scenario_file,
                                model=model,
                                model_label=model_label,
                                model_dir=model_dir,
                                seed_metrics_dir=v3_root
                                / "outputs"
                                / family
                                / task_dir
                                / scenario
                                / model_dir
                                / "seed_metrics",
                            )


def parse_seed(path: Path) -> int | None:
    match = re.match(r"seed_(\d+)_metrics\.csv$", path.name)
    return int(match.group(1)) if match else None


def truthy(value: object) -> bool | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def is_valid_seed_metric(row: dict[str, object], expected_folds: int) -> bool:
    flag = truthy(row.get("seed_complete_all_folds"))
    if flag is not None:
        return flag
    if "n_folds_tested" in row:
        folds = pd.to_numeric(pd.Series([row["n_folds_tested"]]), errors="coerce").iloc[0]
        if pd.notna(folds):
            return int(folds) >= expected_folds
    return True


def read_seed_csv(path: Path) -> dict[str, object] | None:
    try:
        if path.stat().st_size == 0:
            return None
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df.iloc[0].to_dict()
    except Exception:
        return None


def consolidate_metrics(v3_root: Path, tables_dir: Path) -> pd.DataFrame:
    manifest_rows: list[dict[str, object]] = []
    data_rows: list[dict[str, object]] = []
    metrics_seen = ["AUC", "F1", "balanced_accuracy", "MAE", "RMSE", "R2", "prediction_mean"]

    progress("Consolidating seed-level metric CSV files")
    for spec in expected_dirs(v3_root):
        directory = spec.seed_metrics_dir
        if not directory.is_dir():
            continue
        for csv_path in directory.glob("seed_*_metrics.csv"):
            seed = parse_seed(csv_path)
            if seed is None:
                continue
            try:
                stat = csv_path.stat()
            except OSError:
                continue
            row = read_seed_csv(csv_path)
            valid = bool(row and is_valid_seed_metric(row, spec.expected_folds))
            base = {
                "family": spec.family,
                "family_label": spec.family_label,
                "task": spec.task,
                "task_label": spec.task_label_es,
                "task_label_en": spec.task_label_en,
                "scenario": spec.scenario,
                "scenario_label": spec.scenario_label,
                "scenario_title": spec.scenario_title,
                "scenario_file": spec.scenario_file,
                "model": spec.model,
                "model_label": spec.model_label,
                "seed": seed,
                "expected_folds": spec.expected_folds,
                "source_csv": str(csv_path),
                "source_mtime": stat.st_mtime,
                "source_size": stat.st_size,
                "valid_seed_metrics": valid,
                "task_dir_used": spec.task_dir,
                "model_dir_used": spec.model_dir,
            }
            manifest_rows.append(base)
            if not row or not valid:
                continue
            enriched = dict(row)
            enriched.update(base)
            complete_flag = truthy(row.get("seed_complete_all_folds"))
            enriched["_complete_rank"] = int(bool(complete_flag)) if complete_flag is not None else 0
            data_rows.append(enriched)

    tables_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(tables_dir / "seed_metrics_manifest_all.csv", index=False)

    wide = pd.DataFrame(data_rows)
    if wide.empty:
        wide.to_csv(tables_dir / "seed_metrics_wide.csv", index=False)
        pd.DataFrame().to_csv(tables_dir / "seed_metrics_long_for_boxplots.csv", index=False)
        return pd.DataFrame()

    wide = (
        wide.sort_values(["_complete_rank", "source_mtime", "source_size"], ascending=[False, False, False])
        .drop_duplicates(["family", "task", "scenario", "model", "seed"], keep="first")
        .drop(columns=["_complete_rank"], errors="ignore")
        .sort_values(["family", "task", "scenario", "model", "seed"])
    )
    wide.to_csv(tables_dir / "seed_metrics_wide.csv", index=False)

    metric_cols = [c for c in metrics_seen if c in wide.columns]
    long = wide.melt(
        id_vars=[
            "family",
            "family_label",
            "task",
            "task_label",
            "task_label_en",
            "scenario",
            "scenario_label",
            "scenario_title",
            "scenario_file",
            "model",
            "model_label",
            "seed",
            "source_csv",
        ],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    long.to_csv(tables_dir / "seed_metrics_long_for_boxplots.csv", index=False)

    completion_rows = []
    completed = wide.groupby(["family", "task", "scenario", "model"])["seed"].nunique().to_dict()
    for family, family_label, _, _ in FAMILIES:
        for task, task_label_es, _, _, _, _ in TASKS:
            for scenario, scenario_label, _, _ in SCENARIOS:
                for model, model_label, _ in MODELS:
                    n = int(completed.get((family, task, scenario, model), 0))
                    completion_rows.append(
                        {
                            "family": family,
                            "family_label": family_label,
                            "task": task,
                            "task_label": task_label_es,
                            "scenario": scenario,
                            "scenario_label": scenario_label,
                            "model": model,
                            "model_label": model_label,
                            "expected": 100,
                            "completed": min(n, 100),
                            "pending": max(0, 100 - min(n, 100)),
                            "unique_valid_seed_files": n,
                        }
                    )
    completion = pd.DataFrame(completion_rows)
    completion.to_csv(tables_dir / "completion_by_family_task_scenario_model.csv", index=False)
    completion.groupby(["family", "family_label"], as_index=False).agg(
        expected=("expected", "sum"),
        completed=("completed", "sum"),
        pending=("pending", "sum"),
    ).to_csv(tables_dir / "completion_by_family.csv", index=False)
    completion.groupby(["family", "family_label", "model", "model_label"], as_index=False).agg(
        expected=("expected", "sum"),
        completed=("completed", "sum"),
        pending=("pending", "sum"),
    ).to_csv(tables_dir / "completion_by_family_model.csv", index=False)
    completion.groupby(["family", "family_label", "task", "task_label", "model", "model_label"], as_index=False).agg(
        expected=("expected", "sum"),
        completed=("completed", "sum"),
        pending=("pending", "sum"),
    ).to_csv(tables_dir / "completion_by_family_task_model.csv", index=False)

    progress(f"Consolidated {len(wide)} valid seed rows and {len(long)} metric rows")
    return long


def title_n_text(df: pd.DataFrame) -> str:
    return f"n seeds total={len(df)}"


def boxplot_models_by_group(
    df: pd.DataFrame,
    out_base: Path,
    title: str,
    metric: str,
    x_kind: str,
) -> bool:
    if df.empty:
        return False
    if x_kind == "scenario":
        x_order = [x[0] for x in SCENARIOS]
        x_labels = {x[0]: x[1] for x in SCENARIOS}
        x_col = "scenario"
        xlabel = "Esquema"
        center_values = present_order(set(df[x_col]), x_order)
    else:
        x_order = [x[0] for x in FAMILIES]
        x_labels = {x[0]: x[3] for x in FAMILIES}
        x_col = "family"
        xlabel = "Predictor preprocessing family"
        center_values = present_order(set(df[x_col]), x_order)
    models = present_order(set(df["model"]), MODEL_ORDER)
    if not center_values or not models:
        return False

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14.5, 6.8))
    centers = {value: idx + 1 for idx, value in enumerate(center_values)}
    if len(models) == 1:
        offsets = {models[0]: 0.0}
    else:
        span = 0.64
        step = span / max(len(models) - 1, 1)
        offsets = {model: -span / 2 + i * step for i, model in enumerate(models)}
    width = 0.11 if len(models) >= 5 else 0.15

    data = []
    positions = []
    colors = []
    for value in center_values:
        for model in models:
            values = df[(df[x_col] == value) & (df["model"] == model)]["value"].dropna().to_numpy()
            if len(values) == 0:
                continue
            data.append(values)
            positions.append(centers[value] + offsets[model])
            colors.append(MODEL_COLOR[model])
    if not data:
        plt.close(fig)
        return False

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#2b2b2b", "linewidth": 1.15},
        boxprops={"linewidth": 0.9, "edgecolor": "#333333"},
        whiskerprops={"linewidth": 0.9, "color": "#333333"},
        capprops={"linewidth": 0.9, "color": "#333333"},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.84)

    for value in center_values:
        for model in models:
            values = df[(df[x_col] == value) & (df["model"] == model)]["value"].dropna().to_numpy()
            if len(values) == 0:
                continue
            x = centers[value] + offsets[model]
            jitter = [((i % 9) - 4) * 0.0045 for i in range(len(values))]
            ax.scatter([x + j for j in jitter], values, s=7, alpha=0.24, color=MODEL_COLOR[model], edgecolors="none")

    handles = [plt.Line2D([0], [0], marker="s", linestyle="", color=MODEL_COLOR[m], markersize=9) for m in models]
    ax.legend(handles, [MODEL_LABEL[m] for m in models], title="Model" if x_kind == "family" else "Model", loc="upper left", frameon=True, ncol=3)
    ax.set_xticks([centers[x] for x in center_values])
    ax.set_xticklabels([x_labels[x] for x in center_values], fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(METRIC_AXIS.get(metric, metric))
    ax.set_title(title, fontsize=15, pad=12)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=220)
    plt.close(fig)
    return True


def regression_variants(metric_df: pd.DataFrame, task: str, metric: str) -> list[tuple[str, pd.DataFrame, str]]:
    variants: list[tuple[str, pd.DataFrame, str]] = [("all_models", metric_df, metric)]
    if task != "classification":
        variants.append(("without_PySR", metric_df[metric_df["model"] != "pysr"], metric))
    if task != "classification" and metric == "R2":
        clipped = metric_df.copy()
        clipped["value"] = clipped["value"].clip(lower=0)
        variants.append(("all_models_R2_negative_to_zero", clipped, "R2_negative_to_zero"))
        clipped_no_pysr = clipped[clipped["model"] != "pysr"].copy()
        variants.append(("without_PySR_R2_negative_to_zero", clipped_no_pysr, "R2_negative_to_zero"))
    return [(name, sub, metric_name) for name, sub, metric_name in variants if not sub.empty]


def make_set99(df: pd.DataFrame, output_dir: Path) -> int:
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    logs_dir = output_dir / "logs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    plotted = 0
    family_meta = {x[0]: x for x in FAMILIES}
    task_meta = {x[0]: x for x in TASKS}
    for family in [x[0] for x in FAMILIES]:
        family_df = df[df["family"] == family]
        if family_df.empty:
            continue
        for task in [x[0] for x in TASKS]:
            task_df = family_df[family_df["task"] == task]
            if task_df.empty:
                continue
            for metric in METRICS_BY_TASK[task]:
                metric_df = task_df[task_df["metric"] == metric].copy()
                if metric_df.empty:
                    continue
                for variant, variant_df, plot_metric in regression_variants(metric_df, task, metric):
                    family_short = family_meta[family][2]
                    task_short = task_meta[task][3]
                    filename = (
                        f"{family_short}__{task_short}__{SCHEME_NAME_FOR_FILE}"
                        f"__metric_{METRIC_FILE[plot_metric]}__{variant}"
                    )
                    out_base = figures_dir / family / task / filename
                    title = (
                        f"{family_meta[family][1]} | {task_meta[task][1]} | "
                        f"{METRIC_AXIS.get(plot_metric, plot_metric)} | {title_n_text(variant_df)}"
                    )
                    ok = boxplot_models_by_group(variant_df, out_base, title, plot_metric, x_kind="scenario")
                    rows.append(
                        {
                            "set": "99_by_scheme",
                            "family": family,
                            "task": task,
                            "metric": plot_metric,
                            "source_metric": metric,
                            "variant": variant,
                            "n_seed_metric_rows": len(variant_df),
                            "n_unique_seeds": variant_df["seed"].nunique(),
                            "models": "|".join(present_order(set(variant_df["model"]), MODEL_ORDER)),
                            "scenarios": "|".join(present_order(set(variant_df["scenario"]), [x[0] for x in SCENARIOS])),
                            "plotted": ok,
                            "pdf": str(out_base.with_suffix(".pdf")) if ok else "",
                            "png": str(out_base.with_suffix(".png")) if ok else "",
                        }
                    )
                    plotted += int(ok)
    pd.DataFrame(rows).to_csv(tables_dir / "boxplot_manifest.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps({"created_at": datetime.now().isoformat(), "plot_set": "99_by_scheme", "plots": plotted}, indent=2)
    )
    return plotted


def make_set100(df: pd.DataFrame, output_dir: Path) -> int:
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    logs_dir = output_dir / "logs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "input_metrics_long_filtered.csv", index=False)

    rows: list[dict[str, object]] = []
    plotted = 0
    task_meta = {x[0]: x for x in TASKS}
    scenario_meta = {x[0]: x for x in SCENARIOS}
    for task in [x[0] for x in TASKS]:
        task_df = df[df["task"] == task]
        if task_df.empty:
            continue
        for scenario in [x[0] for x in SCENARIOS]:
            scenario_df = task_df[task_df["scenario"] == scenario]
            if scenario_df.empty:
                continue
            for metric in METRICS_BY_TASK[task]:
                metric_df = scenario_df[scenario_df["metric"] == metric].copy()
                if metric_df.empty:
                    continue
                for variant, variant_df, plot_metric in regression_variants(metric_df, task, metric):
                    filename = (
                        f"{task_meta[task][3]}__scheme_{scenario_meta[scenario][3]}"
                        f"__metric_{METRIC_FILE[plot_metric]}__families_by_model__{variant}"
                    )
                    out_base = figures_dir / task / scenario_meta[scenario][3] / filename
                    title = (
                        f"{task_meta[task][2]} | {scenario_meta[scenario][2]} | "
                        f"{METRIC_AXIS.get(plot_metric, plot_metric)} | {title_n_text(variant_df)}"
                    )
                    ok = boxplot_models_by_group(variant_df, out_base, title, plot_metric, x_kind="family")
                    rows.append(
                        {
                            "set": "100_families_by_scheme",
                            "task": task,
                            "scenario": scenario,
                            "metric": plot_metric,
                            "source_metric": metric,
                            "variant": variant,
                            "n_seed_metric_rows": len(variant_df),
                            "n_unique_seeds": variant_df["seed"].nunique(),
                            "models": "|".join(present_order(set(variant_df["model"]), MODEL_ORDER)),
                            "families": "|".join(present_order(set(variant_df["family"]), [x[0] for x in FAMILIES])),
                            "plotted": ok,
                            "pdf": str(out_base.with_suffix(".pdf")) if ok else "",
                            "png": str(out_base.with_suffix(".png")) if ok else "",
                        }
                    )
                    plotted += int(ok)
    pd.DataFrame(rows).to_csv(tables_dir / "family_comparison_boxplot_manifest.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps({"created_at": datetime.now().isoformat(), "plot_set": "100_families_by_scheme", "plots": plotted}, indent=2)
    )
    return plotted


def make_archive(paths: list[Path], archive_path: Path) -> None:
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in paths:
            tar.add(path, arcname=path.name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--prefix-number", default="193")
    args = parser.parse_args()

    timestamp = args.timestamp or now_tag()
    outputs = args.v3_root / "outputs"
    audit_dir = outputs / f"{args.prefix_number}_latest_metric_consolidation_for_boxplots_{timestamp}"
    set99_dir = outputs / f"{int(args.prefix_number) + 1}_latest_99_boxplots_by_scheme_with_r2clip_{timestamp}"
    set100_dir = outputs / f"{int(args.prefix_number) + 2}_latest_100_boxplots_families_by_scheme_with_r2clip_{timestamp}"
    tables = audit_dir / "tables"
    logs = audit_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    long = consolidate_metrics(args.v3_root, tables)
    if long.empty:
        raise SystemExit("No valid metric rows were found")

    allowed_metrics = set(sum(METRICS_BY_TASK.values(), []))
    long = long[long["metric"].isin(allowed_metrics)].copy()
    long.to_csv(tables / "seed_metrics_long_for_boxplots_filtered.csv", index=False)

    progress(f"Generating set 99 plots in {set99_dir}")
    n99 = make_set99(long, set99_dir)
    progress(f"Generating set 100 plots in {set100_dir}")
    n100 = make_set100(long, set100_dir)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "v3_root": str(args.v3_root),
        "audit_dir": str(audit_dir),
        "set99_dir": str(set99_dir),
        "set100_dir": str(set100_dir),
        "metric_rows": int(len(long)),
        "set99_plots": int(n99),
        "set100_plots": int(n100),
        "r2_negative_to_zero": True,
    }
    (audit_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    archive = outputs / f"{set99_dir.name}__{set100_dir.name}.tar.gz"
    make_archive([audit_dir, set99_dir, set100_dir], archive)
    metadata["archive"] = str(archive)
    (audit_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
