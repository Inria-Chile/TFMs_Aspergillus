#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


FAMILY_SLUG = {
    "family1_no_clr_microbiome_raw": "family_01",
    "family2_full_microbiome_clr_raw_env": "family_02",
    "family3_marta_subset_clr_raw_env": "family_03",
}

FAMILY_TITLE = {
    "family1_no_clr_microbiome_raw": "Family 01: Microbiome without CLR",
    "family2_full_microbiome_clr_raw_env": "Family 02: Complete microbiome with CLR",
    "family3_marta_subset_clr_raw_env": "Family 03: Marta subset with CLR",
}

TASK_SLUG = {
    "classification": "classification",
    "regression_18_positive": "regression_18_samples",
    "regression_18_positive_samples": "regression_18_samples",
    "regression_50_all_samples": "regression_50_samples",
}

TASK_TITLE = {
    "classification": "Classification",
    "regression_18_positive": "Regression 18 positive samples",
    "regression_18_positive_samples": "Regression 18 positive samples",
    "regression_50_all_samples": "Regression 50 samples",
}

SCENARIO_SLUG = {
    "A_environment": "Environment",
    "B_microbiome": "Microbiome",
    "C_environment_microbiome": "Environment_Microbiome",
}

SCENARIO_TITLE = {
    "A_environment": "Environment",
    "B_microbiome": "Microbiome",
    "C_environment_microbiome": "Environment + Microbiome",
}

MODEL_TITLE = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def load_taxa_names(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    required = {"anonymous_name", "original_name"}
    if not required.issubset(df.columns):
        return {}
    mapping = {}
    for _, row in df.dropna(subset=["anonymous_name", "original_name"]).iterrows():
        anon = str(row["anonymous_name"]).strip()
        original = str(row["original_name"]).strip()
        if anon and original:
            mapping[anon] = f"{anon}_{original}"
    return mapping


def display_feature(feature: str, taxa: dict[str, str]) -> str:
    value = str(feature)
    match = re.fullmatch(r"([BF]\d+)", value)
    if match and value in taxa:
        return taxa[value]
    return value


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "NA"


def make_plot(df: pd.DataFrame, out_base: Path, title: str, top_n: int) -> None:
    plot_df = df.sort_values("mean_abs_shap", ascending=False).head(top_n).copy()
    plot_df = plot_df.sort_values("mean_abs_shap", ascending=True)
    height = max(4.8, min(14.0, 0.34 * len(plot_df) + 1.8))
    fig, ax = plt.subplots(figsize=(9.5, height), constrained_layout=True)
    xerr = plot_df["sd_abs_shap"].fillna(0.0).clip(lower=0.0)
    ax.barh(
        plot_df["feature_display"],
        plot_df["mean_abs_shap"],
        xerr=xerr,
        color="#4C78A8",
        edgecolor="#26384f",
        linewidth=0.5,
        error_kw={"elinewidth": 0.8, "capsize": 2.5, "ecolor": "#333333"},
    )
    ax.set_xlabel("Mean absolute SHAP value across seeds")
    ax.set_ylabel("Predictor")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.25, linewidth=0.7)
    ax.tick_params(axis="y", labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-shap", required=True, type=Path)
    parser.add_argument("--taxa-dictionary", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=["random_forest", "xgboost"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-n", default="30,20,10")
    args = parser.parse_args()

    taxa = load_taxa_names(args.taxa_dictionary)
    top_values = [int(x) for x in args.top_n.split(",") if x.strip()]

    usecols = [
        "family",
        "family_label",
        "task",
        "task_label",
        "scenario",
        "scenario_label",
        "model",
        "model_label",
        "seed",
        "feature",
        "mean_abs_shap",
        "source_type",
    ]
    chunks = []
    for chunk in pd.read_csv(args.combined_shap, usecols=usecols, chunksize=250_000):
        part = chunk[chunk["model"].astype(str).eq(args.model)].copy()
        if not part.empty:
            chunks.append(part)
    if not chunks:
        raise SystemExit(f"No rows found for model={args.model}")

    df = pd.concat(chunks, ignore_index=True)
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    df["mean_abs_shap"] = pd.to_numeric(df["mean_abs_shap"], errors="coerce")
    df = df.dropna(subset=["family", "task", "scenario", "seed", "feature", "mean_abs_shap"])
    df = df.drop_duplicates(
        subset=["family", "task", "scenario", "model", "seed", "feature"],
        keep="first",
    )

    group_cols = ["family", "task", "scenario", "model", "feature"]
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "mean"),
            sd_abs_shap=("mean_abs_shap", "std"),
            n_seeds=("seed", "nunique"),
            source_types=("source_type", lambda x: ";".join(sorted(set(map(str, x))))),
        )
        .reset_index()
    )
    summary["feature_display"] = summary["feature"].map(lambda x: display_feature(x, taxa))
    summary = summary.sort_values(
        ["family", "task", "scenario", "model", "mean_abs_shap"],
        ascending=[True, True, True, True, False],
    )

    coverage = (
        df.drop_duplicates(["family", "task", "scenario", "model", "seed"])
        .groupby(["family", "task", "scenario", "model"])
        .agg(n_seeds=("seed", "nunique"))
        .reset_index()
        .sort_values(["family", "task", "scenario", "model"])
    )
    feature_counts = (
        summary.groupby(["family", "task", "scenario", "model"])
        .agg(n_features=("feature", "nunique"))
        .reset_index()
    )
    coverage = coverage.merge(feature_counts, on=["family", "task", "scenario", "model"], how="left")

    tables = args.output_dir / "tables"
    figs = args.output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables / f"{args.model}_shap_predictor_importance_summary.csv", index=False)
    coverage.to_csv(tables / f"{args.model}_shap_seed_coverage_by_task.csv", index=False)

    manifest_rows = []
    for (family, task, scenario, model), sub in summary.groupby(["family", "task", "scenario", "model"], sort=True):
        n_group_seeds = int(coverage.loc[
            (coverage["family"].eq(family))
            & (coverage["task"].eq(task))
            & (coverage["scenario"].eq(scenario))
            & (coverage["model"].eq(model)),
            "n_seeds",
        ].iloc[0])
        family_slug = FAMILY_SLUG.get(family, safe_name(family))
        task_slug = TASK_SLUG.get(task, safe_name(task))
        scenario_slug = SCENARIO_SLUG.get(scenario, safe_name(scenario))
        model_slug = safe_name(model)
        model_title = MODEL_TITLE.get(model, model)
        title = (
            f"{model_title} SHAP importance | {FAMILY_TITLE.get(family, family)} | "
            f"{TASK_TITLE.get(task, task)} | {SCENARIO_TITLE.get(scenario, scenario)} | "
            f"n seeds = {n_group_seeds}"
        )
        for top_n in top_values:
            out_dir = figs / f"top{top_n}" / family / task_slug / scenario_slug
            stem = (
                f"{family_slug}__{task_slug}__scheme_{scenario_slug}"
                f"__model_{model_slug}__top{top_n}__mean_abs_shap"
            )
            out_base = out_dir / stem
            make_plot(sub, out_base, f"{title} | top {top_n}", top_n)
            manifest_rows.append(
                {
                    "family": family,
                    "task": task,
                    "scenario": scenario,
                    "model": model,
                    "top_n": top_n,
                    "n_seeds": n_group_seeds,
                    "pdf": str(out_base.with_suffix(".pdf")),
                    "png": str(out_base.with_suffix(".png")),
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(tables / f"{args.model}_plot_manifest.csv", index=False)
    metadata = {
        "created_at": datetime.now().isoformat(),
        "model": args.model,
        "combined_shap": str(args.combined_shap),
        "taxa_dictionary": str(args.taxa_dictionary),
        "rows_input_model": int(len(df)),
        "summary_rows": int(len(summary)),
        "groups": int(len(coverage)),
        "plots": int(len(manifest)),
        "pdf_count": len(list(figs.rglob("*.pdf"))),
        "png_count": len(list(figs.rglob("*.png"))),
        "top_n": top_values,
    }
    (tables / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
