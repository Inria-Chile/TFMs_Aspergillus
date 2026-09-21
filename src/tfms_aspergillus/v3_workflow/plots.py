from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def save_fig(fig, out_stem: Path) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def classification_summary_plot(summary: pd.DataFrame, out_stem: Path, title: str) -> None:
    if summary.empty:
        return
    df = summary.copy()
    df["label"] = df["scenario"] + " | " + df["model"]
    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(df)), 6))
    colors = {"RF_full": "#4c78a8", "RF_anova": "#f58518", "RF_same_features": "#54a24b", "TabPFN": "#b279a2"}
    bar_colors = [colors.get(str(m), "#4c78a8") for m in df["model"]]
    ax.bar(range(len(df)), df["AUC_mean"], color=bar_colors)
    ax.errorbar(range(len(df)), df["AUC_mean"], yerr=df["AUC_sd"], fmt="none", ecolor="#333333", capsize=3)
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("AUC media")
    ax.set_title(title)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"], rotation=55, ha="right")
    save_fig(fig, out_stem)


def global_oof_roc_plot(preds: pd.DataFrame, out_stem: Path, title: str) -> None:
    if preds.empty:
        return
    from sklearn.metrics import RocCurveDisplay, roc_auc_score

    fig, ax = plt.subplots(figsize=(7, 7))
    for (scenario, model), g in preds.groupby(["scenario", "model"]):
        if g["y_true"].nunique() < 2:
            continue
        auc = roc_auc_score(g["y_true"], g["y_prob"])
        RocCurveDisplay.from_predictions(
            g["y_true"],
            g["y_prob"],
            name=f"{scenario} | {model} AUC={auc:.3f}",
            ax=ax,
        )
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    ax.set_title(title)
    save_fig(fig, out_stem)


def regression_scatter_plot(preds: pd.DataFrame, out_stem: Path, title: str) -> None:
    if preds.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    for (scenario, model), g in preds.groupby(["scenario", "model"]):
        ax.scatter(g["y_true_raw"], g["y_pred_raw"], alpha=0.75, label=f"{scenario} | {model}")
    lo = min(preds["y_true_raw"].min(), preds["y_pred_raw"].min())
    hi = max(preds["y_true_raw"].max(), preds["y_pred_raw"].max())
    ax.plot([lo, hi], [lo, hi], color="#777777", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Aspergillus observado positivo")
    ax.set_ylabel("Aspergillus predicho")
    ax.legend(fontsize=8)
    save_fig(fig, out_stem)


def importance_plot(importance: pd.DataFrame, value_col: str, out_stem: Path, title: str, top_n: int = 20) -> None:
    if importance.empty or value_col not in importance.columns:
        return
    df = importance.sort_values(value_col, ascending=False).head(top_n).copy()
    fig, ax = plt.subplots(figsize=(8, max(5, 0.35 * len(df))))
    y = range(len(df))
    ax.barh(list(y), df[value_col], color="#4c78a8")
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["variable"])
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.set_ylabel("")
    save_fig(fig, out_stem)
