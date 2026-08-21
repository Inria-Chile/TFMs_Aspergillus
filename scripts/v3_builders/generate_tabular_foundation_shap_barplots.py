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
    "positive_abundance": "regression_18_samples",
    "regression_50_all_samples": "regression_50_samples",
    "all_sample_abundance": "regression_50_samples",
}

TASK_TITLE = {
    "classification": "Classification",
    "regression_18_positive": "Regression 18 positive samples",
    "regression_18_positive_samples": "Regression 18 positive samples",
    "positive_abundance": "Regression 18 positive samples",
    "regression_50_all_samples": "Regression 50 samples",
    "all_sample_abundance": "Regression 50 samples",
}

TASK_NORM = {
    "regression_18_positive_samples": "regression_18_positive",
    "positive_abundance": "regression_18_positive",
    "all_sample_abundance": "regression_50_all_samples",
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
    "tabpfn": "TFN/TabPFN",
    "tabiclv2": "TabICLv2",
}

MODEL_NORM = {
    "tfn/tabpfn": "tabpfn",
    "tfn_tabpfn": "tabpfn",
    "tfn": "tabpfn",
    "tabpfn": "tabpfn",
    "tabicl": "tabiclv2",
    "tabiclv2": "tabiclv2",
    "tabicl_v2": "tabiclv2",
    "tabiclv2": "tabiclv2",
}


def normalize_task(value: str) -> str:
    value = str(value)
    return TASK_NORM.get(value, value)


def normalize_model(value: str) -> str:
    value = str(value).strip().lower()
    return MODEL_NORM.get(value, value.replace(" ", "_"))


def load_taxa_names(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep=None, engine="python")
    mapping = {}
    if {"anonymous_name", "original_name"}.issubset(df.columns):
        for _, row in df.dropna(subset=["anonymous_name", "original_name"]).iterrows():
            anon = str(row["anonymous_name"]).strip()
            original = str(row["original_name"]).strip().strip('"')
            if anon and original:
                mapping[anon] = f"{anon}_{original}"
    return mapping


def display_feature(feature: str, taxa: dict[str, str]) -> str:
    value = str(feature)
    if re.fullmatch(r"[BF]\d+", value) and value in taxa:
        return taxa[value]
    # Some SHAP exports keep preprocessing context in the feature name,
    # e.g. clr_B12, B12_clr, microbiome__F7. Preserve that context while
    # expanding the anonymous microbiome code to a readable taxon label.
    match = re.search(r"(?<![A-Za-z0-9])([BF]\d+)(?![A-Za-z0-9])", value)
    if match and match.group(1) in taxa:
        code = match.group(1)
        return value[: match.start(1)] + taxa[code] + value[match.end(1) :]
    return value


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "NA"


def read_seed_level_summary(path: Path, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000):
        chunk["model"] = chunk["model"].map(normalize_model)
        part = chunk[chunk["model"].eq(model)].copy()
        if not part.empty:
            chunks.append(part)
    if not chunks:
        empty_cols = [
            "family",
            "task",
            "scenario",
            "model",
            "feature",
            "mean_abs_shap",
            "sd_abs_shap",
            "n_seeds",
            "source_layer",
            "source_types",
        ]
        return pd.DataFrame(columns=empty_cols), pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    df["task"] = df["task"].map(normalize_task)
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df["mean_abs_shap"] = pd.to_numeric(df["mean_abs_shap"], errors="coerce")
    df = df.dropna(subset=["family", "task", "scenario", "model", "seed", "feature", "mean_abs_shap"])
    df = df.drop_duplicates(["family", "task", "scenario", "model", "seed", "feature"])
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
    summary["source_layer"] = "seed_level_combined"
    coverage = (
        df.drop_duplicates(["family", "task", "scenario", "model", "seed"])
        .groupby(["family", "task", "scenario", "model"])
        .size()
        .reset_index(name="group_n_seeds")
    )
    return summary, coverage


def read_direct_shap_importance(root: Path, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for p in root.glob(f"outputs/family*/**/{model}/shap_importance/mean_abs_shap_by_feature.csv"):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        lower = {c.lower(): c for c in df.columns}
        required = {"family", "task", "scenario", "model", "feature", "mean_abs_shap"}
        if not required.issubset(lower):
            continue
        out = pd.DataFrame(
            {
                "family": df[lower["family"]].astype(str),
                "task": df[lower["task"]].astype(str).map(normalize_task),
                "scenario": df[lower["scenario"]].astype(str),
                "model": df[lower["model"]].astype(str).map(normalize_model),
                "feature": df[lower["feature"]].astype(str),
                "mean_abs_shap": pd.to_numeric(df[lower["mean_abs_shap"]], errors="coerce"),
                "sd_abs_shap": pd.to_numeric(df[lower["sd_abs_shap"]], errors="coerce")
                if "sd_abs_shap" in lower
                else pd.NA,
                "n_seeds": pd.to_numeric(df[lower["n_seeds"]], errors="coerce") if "n_seeds" in lower else pd.NA,
                "source_types": str(p),
            }
        )
        out = out[out["model"].eq(model)].dropna(subset=["mean_abs_shap"])
        out["source_layer"] = "direct_shap_importance"
        rows.append(out)
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    summary = pd.concat(rows, ignore_index=True)
    summary = summary.drop_duplicates(["family", "task", "scenario", "model", "feature"], keep="first")
    coverage = (
        summary.groupby(["family", "task", "scenario", "model"], dropna=False)
        .agg(
            group_n_seeds=("n_seeds", "median"),
            n_features=("feature", "nunique"),
            source_layer=("source_layer", "first"),
        )
        .reset_index()
    )
    return summary, coverage


def merge_sources(seed_summary: pd.DataFrame, direct_summary: pd.DataFrame) -> pd.DataFrame:
    seed_groups = set(zip(seed_summary.family, seed_summary.task, seed_summary.scenario, seed_summary.model))
    if direct_summary.empty:
        return seed_summary.copy()
    keep = [
        (fam, task, scen, model) not in seed_groups
        for fam, task, scen, model in zip(
            direct_summary.family, direct_summary.task, direct_summary.scenario, direct_summary.model
        )
    ]
    direct_keep = direct_summary.loc[keep].copy()
    return pd.concat([seed_summary, direct_keep], ignore_index=True)


def make_plot(df: pd.DataFrame, out_base: Path, title: str, top_n: int) -> None:
    plot_df = df.sort_values("mean_abs_shap", ascending=False).head(top_n).copy()
    plot_df = plot_df.sort_values("mean_abs_shap", ascending=True)
    height = max(4.8, min(15.0, 0.36 * len(plot_df) + 1.8))
    fig, ax = plt.subplots(figsize=(10.5, height), constrained_layout=True)
    xerr = pd.to_numeric(plot_df["sd_abs_shap"], errors="coerce").fillna(0.0).clip(lower=0.0)
    ax.barh(
        plot_df["feature_display"],
        plot_df["mean_abs_shap"],
        xerr=xerr,
        color="#4C78A8",
        edgecolor="#26384f",
        linewidth=0.5,
        error_kw={"elinewidth": 0.8, "capsize": 2.5, "ecolor": "#333333"},
    )
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("Predictor")
    ax.set_title(title, fontsize=10.5)
    ax.grid(axis="x", alpha=0.25, linewidth=0.7)
    ax.tick_params(axis="y", labelsize=7.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-root", required=True, type=Path)
    parser.add_argument("--combined-shap", required=True, type=Path)
    parser.add_argument("--taxa-dictionary", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=["tabpfn", "tabiclv2"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-n", default="30,20,10")
    args = parser.parse_args()

    taxa = load_taxa_names(args.taxa_dictionary)
    top_values = [int(x) for x in args.top_n.split(",") if x.strip()]
    seed_summary, seed_cov = read_seed_level_summary(args.combined_shap, args.model)
    direct_summary, direct_cov = read_direct_shap_importance(args.v3_root, args.model)
    summary = merge_sources(seed_summary, direct_summary)
    if summary.empty:
        raise SystemExit(f"No SHAP rows found for {args.model}")
    summary["feature_display"] = summary["feature"].map(lambda x: display_feature(x, taxa))
    summary = summary.sort_values(
        ["family", "task", "scenario", "model", "mean_abs_shap"],
        ascending=[True, True, True, True, False],
    )

    tables = args.output_dir / "tables"
    figs = args.output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables / f"{args.model}_seed_replicate_shap_predictor_importance_summary.csv", index=False)
    source_cov = (
        summary.groupby(["family", "task", "scenario", "model", "source_layer"], dropna=False)
        .agg(
            group_n_seeds=("n_seeds", "median"),
            min_feature_n_seeds=("n_seeds", "min"),
            max_feature_n_seeds=("n_seeds", "max"),
            n_features=("feature", "nunique"),
        )
        .reset_index()
        .sort_values(["family", "task", "scenario", "model"])
    )
    source_cov.to_csv(tables / f"{args.model}_seed_replicate_shap_source_coverage.csv", index=False)

    manifest_rows = []
    for (family, task, scenario, model), sub in summary.groupby(["family", "task", "scenario", "model"], sort=True):
        source_layers = ",".join(sorted(sub["source_layer"].dropna().astype(str).unique()))
        n_med = pd.to_numeric(sub["n_seeds"], errors="coerce").median()
        n_min = pd.to_numeric(sub["n_seeds"], errors="coerce").min()
        n_max = pd.to_numeric(sub["n_seeds"], errors="coerce").max()
        if pd.isna(n_med):
            n_text = "n seeds = NA"
        elif n_min == n_max:
            n_text = f"n seeds = {int(n_med)}"
        else:
            n_text = f"feature-level n seeds = {int(n_min)}-{int(n_max)}"
        family_slug = FAMILY_SLUG.get(family, safe_name(family))
        task_slug = TASK_SLUG.get(task, safe_name(task))
        scenario_slug = SCENARIO_SLUG.get(scenario, safe_name(scenario))
        model_slug = safe_name(model)
        model_title = MODEL_TITLE.get(model, model)
        title = (
            f"{model_title} SHAP importance | {FAMILY_TITLE.get(family, family)} | "
            f"{TASK_TITLE.get(task, task)} | {SCENARIO_TITLE.get(scenario, scenario)} | "
            f"{n_text} | source: {source_layers}"
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
                    "n_seeds_median": n_med,
                    "n_seeds_min": n_min,
                    "n_seeds_max": n_max,
                    "source_layer": source_layers,
                    "pdf": str(out_base.with_suffix(".pdf")),
                    "png": str(out_base.with_suffix(".png")),
                }
            )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(tables / f"{args.model}_seed_replicate_shap_plot_manifest.csv", index=False)
    metadata = {
        "created_at": datetime.now().isoformat(),
        "model": args.model,
        "combined_shap": str(args.combined_shap),
        "rows_seed_level_summary": int(len(seed_summary)),
        "rows_direct_summary": int(len(direct_summary)),
        "rows_final_summary": int(len(summary)),
        "groups": int(summary[["family", "task", "scenario", "model"]].drop_duplicates().shape[0]),
        "plots": int(len(manifest)),
        "pdf_count": len(list(figs.rglob("*.pdf"))),
        "png_count": len(list(figs.rglob("*.png"))),
        "note": "Consolidates SHAP mean_abs values by predictor across available seed replicates for TabPFN/TabICLv2; microbiome B/F variables are expanded with anonymous_name_original_name.",
    }
    (tables / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
