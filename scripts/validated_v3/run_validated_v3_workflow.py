#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tfms_aspergillus.v3_workflow.data import detect_blocks, load_table, medium_labels, write_json
from tfms_aspergillus.v3_workflow.metrics import (
    classification_metrics,
    regression_metrics,
    safe_auc,
    summarize_classification,
)
from tfms_aspergillus.v3_workflow.models import (
    impute_train_test,
    rf_classifier,
    rf_regressor,
    select_features_inside_fold,
    tabpfn_available,
    tabpfn_classifier,
    tabpfn_regressor,
)
from tfms_aspergillus.v3_workflow.plots import (
    classification_summary_plot,
    global_oof_roc_plot,
    importance_plot,
    regression_scatter_plot,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_seeds(seed_text: str | None, n_seeds: int | None, base_seed: int) -> list[int]:
    if seed_text:
        return [int(x.strip()) for x in seed_text.split(",") if x.strip()]
    if n_seeds and n_seeds > 1:
        return [base_seed + i for i in range(n_seeds)]
    return [base_seed]


def scenario_map(df: pd.DataFrame, blocks, target_col: str) -> dict[str, list[str]]:
    env = blocks.env_cols + blocks.human_cols
    return {
        "A_environment": env + blocks.dummy_cols,
        "B_microbiome": blocks.microbiome_predictors + blocks.dummy_cols,
        "C_environment_microbiome": env + blocks.microbiome_predictors + blocks.dummy_cols,
    }


def max_features_for(cfg: dict, scenario: str, n_candidate: int) -> int | None:
    value = cfg["max_selected_features_by_scenario"].get(scenario)
    if value is None:
        return None
    return min(int(value), n_candidate)


def model_requires_anova(model_name: str) -> bool:
    return model_name in {"RF_anova", "TabPFN"}


def is_rf(model_name: str) -> bool:
    return model_name in {"RF_full", "RF_anova", "RF_same_features"}


def run_classification_cv(df, cfg, scenario_name, feature_cols, model_name, run_tabpfn):
    y = df["Aspergillus_presence"].astype(int).to_numpy()
    X = df[feature_cols].copy()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=cfg["random_state"])
    results, predictions, selected_records, rf_importance = [], [], [], []
    forced_cols = [c for c in cfg["dummy_cols"] if c in feature_cols]
    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train_raw, X_test_raw = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y[train_idx], y[test_idx]
        if model_requires_anova(model_name):
            core = [c for c in feature_cols if c not in forced_cols]
            max_features = max_features_for(cfg, scenario_name, len(core))
            X_train_sel, X_test_sel, selected_cols = select_features_inside_fold(
                X_train_raw,
                y_train,
                X_test_raw,
                task="classification",
                max_features=max_features,
                forced_cols=forced_cols,
            )
        else:
            X_train_sel, X_test_sel, selected_cols = X_train_raw, X_test_raw, list(feature_cols)
        X_train_model, X_test_model = impute_train_test(X_train_sel, X_test_sel)

        if is_rf(model_name):
            model = rf_classifier(cfg["random_state"], cfg["rf_n_estimators"], cfg["rf_n_jobs"])
            model.fit(X_train_model, y_train)
            y_prob = model.predict_proba(X_test_model)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            rf_importance.extend(
                {
                    "task": "classification",
                    "seed": cfg["random_state"],
                    "scenario": scenario_name,
                    "model": model_name,
                    "fold_id": fold_id,
                    "variable": var,
                    "importance": imp,
                }
                for var, imp in zip(selected_cols, model.feature_importances_)
            )
        elif model_name == "TabPFN":
            if not run_tabpfn:
                continue
            model = tabpfn_classifier(cfg["tabpfn_device"])
            model.fit(X_train_model, y_train)
            y_prob = model.predict_proba(X_test_model)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
        else:
            raise ValueError(model_name)

        met = classification_metrics(y_test, y_pred, y_prob)
        met.update(
            {
                "seed": cfg["random_state"],
                "scenario": scenario_name,
                "model": model_name,
                "fold_id": fold_id,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_features_used": len(selected_cols),
            }
        )
        results.append(met)
        predictions.append(
            pd.DataFrame(
                {
                    "scenario": scenario_name,
                    "model": model_name,
                    "seed": cfg["random_state"],
                    "fold_id": fold_id,
                    "Sample": df.iloc[test_idx][cfg["id_col"]].to_numpy(),
                    "y_true": y_test,
                    "y_prob": y_prob,
                    "y_pred": y_pred,
                }
            )
        )
        selected_records.extend(
            {
                "task": "classification",
                "seed": cfg["random_state"],
                "scenario": scenario_name,
                "model": model_name,
                "fold_id": fold_id,
                "variable": v,
            }
            for v in selected_cols
        )
    return (
        pd.DataFrame(results),
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        pd.DataFrame(selected_records),
        pd.DataFrame(rf_importance),
    )


def run_regression_cv(df, cfg, scenario_name, feature_cols, model_name, run_tabpfn):
    data_pos = df[df[cfg["target_col"]] > 0].copy()
    if len(data_pos) < 5:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    y_raw = data_pos[cfg["target_col"]].astype(float).to_numpy()
    y = np.log1p(y_raw)
    X = data_pos[feature_cols].copy()
    results, predictions, selected_records, rf_importance = [], [], [], []
    forced_cols = [c for c in cfg["dummy_cols"] if c in feature_cols]

    from sklearn.model_selection import LeaveOneOut

    for fold_id, (train_idx, test_idx) in enumerate(LeaveOneOut().split(X), start=1):
        X_train_raw, X_test_raw = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y[train_idx], y[test_idx]
        if model_requires_anova(model_name):
            core = [c for c in feature_cols if c not in forced_cols]
            max_features = max_features_for(cfg, scenario_name, len(core))
            X_train_sel, X_test_sel, selected_cols = select_features_inside_fold(
                X_train_raw,
                y_train,
                X_test_raw,
                task="regression",
                max_features=max_features,
                forced_cols=forced_cols,
            )
        else:
            X_train_sel, X_test_sel, selected_cols = X_train_raw, X_test_raw, list(feature_cols)
        X_train_model, X_test_model = impute_train_test(X_train_sel, X_test_sel)

        if is_rf(model_name):
            model = rf_regressor(cfg["random_state"], cfg["rf_n_estimators"], cfg["rf_n_jobs"])
            model.fit(X_train_model, y_train)
            y_pred = model.predict(X_test_model)
            rf_importance.extend(
                {
                    "task": "conditional_abundance",
                    "seed": cfg["random_state"],
                    "scenario": scenario_name,
                    "model": model_name,
                    "fold_id": fold_id,
                    "variable": var,
                    "importance": imp,
                }
                for var, imp in zip(selected_cols, model.feature_importances_)
            )
        elif model_name == "TabPFN":
            if not run_tabpfn:
                continue
            model = tabpfn_regressor(cfg["tabpfn_device"])
            model.fit(X_train_model, y_train)
            y_pred = model.predict(X_test_model)
        else:
            raise ValueError(model_name)

        fold_metrics = regression_metrics(y_test, y_pred)
        fold_metrics.update(
            {
                "seed": cfg["random_state"],
                "scenario": scenario_name,
                "model": model_name,
                "fold_id": fold_id,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_features_used": len(selected_cols),
            }
        )
        results.append(fold_metrics)
        predictions.append(
            pd.DataFrame(
                {
                    "scenario": scenario_name,
                    "model": model_name,
                    "seed": cfg["random_state"],
                    "fold_id": fold_id,
                    "Sample": data_pos.iloc[test_idx][cfg["id_col"]].to_numpy(),
                    "y_true_log1p": y_test,
                    "y_pred_log1p": y_pred,
                    "y_true_raw": y_raw[test_idx],
                    "y_pred_raw": np.expm1(y_pred),
                }
            )
        )
        selected_records.extend(
            {
                "task": "conditional_abundance",
                "seed": cfg["random_state"],
                "scenario": scenario_name,
                "model": model_name,
                "fold_id": fold_id,
                "variable": v,
            }
            for v in selected_cols
        )
    return (
        pd.DataFrame(results),
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        pd.DataFrame(selected_records),
        pd.DataFrame(rf_importance),
    )


def summarize_regression_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df
    rows = []
    for (scenario, model), g in pred_df.groupby(["scenario", "model"]):
        met = regression_metrics(g["y_true_raw"].to_numpy(), g["y_pred_raw"].to_numpy())
        met.update(
            {
                "scenario": scenario,
                "model": model,
                "n_predictions": len(g),
                "n_features_mean": np.nan,
            }
        )
        rows.append(met)
    return pd.DataFrame(rows).sort_values(["RMSE", "MAE"])


def summarize_regression_by_seed(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty or "seed" not in pred_df.columns:
        return pd.DataFrame()
    rows = []
    for (seed, scenario, model), g in pred_df.groupby(["seed", "scenario", "model"]):
        met = regression_metrics(g["y_true_raw"].to_numpy(), g["y_pred_raw"].to_numpy())
        met.update({"seed": seed, "scenario": scenario, "model": model, "n_predictions": len(g)})
        rows.append(met)
    return pd.DataFrame(rows)


def summarize_regression_across_seeds(seed_summary: pd.DataFrame) -> pd.DataFrame:
    if seed_summary.empty:
        return pd.DataFrame()
    return (
        seed_summary.groupby(["scenario", "model"], as_index=False)
        .agg(
            RMSE_mean=("RMSE", "mean"),
            RMSE_sd=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_sd=("MAE", "std"),
            R2_mean=("R2", "mean"),
            R2_sd=("R2", "std"),
            pearson_mean=("pearson_correlation", "mean"),
            spearman_mean=("spearman_correlation", "mean"),
            n_seeds=("seed", "nunique"),
            n_predictions_total=("n_predictions", "sum"),
        )
        .sort_values(["RMSE_mean", "MAE_mean"])
    )


def summarize_classification_by_seed(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    if fold_metrics.empty or "seed" not in fold_metrics.columns:
        return pd.DataFrame()
    return (
        fold_metrics.groupby(["seed", "scenario", "model"], as_index=False)
        .agg(
            AUC_mean=("AUC", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            sensitivity_mean=("sensitivity", "mean"),
            specificity_mean=("specificity", "mean"),
            precision_mean=("precision", "mean"),
            F1_mean=("F1", "mean"),
            n_folds=("fold_id", "count"),
        )
        .sort_values(["scenario", "model", "seed"])
    )


def summarize_classification_across_seeds(seed_summary: pd.DataFrame) -> pd.DataFrame:
    if seed_summary.empty:
        return pd.DataFrame()
    return (
        seed_summary.groupby(["scenario", "model"], as_index=False)
        .agg(
            AUC_mean=("AUC_mean", "mean"),
            AUC_sd_across_seeds=("AUC_mean", "std"),
            balanced_accuracy_mean=("balanced_accuracy_mean", "mean"),
            balanced_accuracy_sd_across_seeds=("balanced_accuracy_mean", "std"),
            sensitivity_mean=("sensitivity_mean", "mean"),
            sensitivity_sd_across_seeds=("sensitivity_mean", "std"),
            specificity_mean=("specificity_mean", "mean"),
            specificity_sd_across_seeds=("specificity_mean", "std"),
            precision_mean=("precision_mean", "mean"),
            F1_mean=("F1_mean", "mean"),
            n_seeds=("seed", "nunique"),
            n_folds_total=("n_folds", "sum"),
        )
        .sort_values(["AUC_mean", "balanced_accuracy_mean"], ascending=[False, False])
    )


def run_15a(df: pd.DataFrame, cfg: dict, out_dir: Path, run_tabpfn: bool):
    blocks = detect_blocks(df, cfg["target_col"], cfg["dummy_cols"])
    scenarios = scenario_map(df, blocks, cfg["target_col"])
    models = ["RF_full", "RF_anova"] + (["TabPFN"] if run_tabpfn else [])
    cls_all, cls_pred_all, cls_sel_all, cls_imp_all = [], [], [], []
    reg_all, reg_pred_all, reg_sel_all, reg_imp_all = [], [], [], []
    for scenario, features in scenarios.items():
        for model_name in models:
            print(f"[15A] classification {scenario} {model_name}", flush=True)
            a, b, c, d = run_classification_cv(df, cfg, scenario, features, model_name, run_tabpfn)
            cls_all.append(a)
            cls_pred_all.append(b)
            cls_sel_all.append(c)
            cls_imp_all.append(d)
            print(f"[15A] regression {scenario} {model_name}", flush=True)
            a, b, c, d = run_regression_cv(df, cfg, scenario, features, model_name, run_tabpfn)
            reg_all.append(a)
            reg_pred_all.append(b)
            reg_sel_all.append(c)
            reg_imp_all.append(d)

    out_dir.mkdir(parents=True, exist_ok=True)
    cls = pd.concat(cls_all, ignore_index=True) if cls_all else pd.DataFrame()
    cls_pred = pd.concat(cls_pred_all, ignore_index=True) if cls_pred_all else pd.DataFrame()
    reg = pd.concat(reg_all, ignore_index=True) if reg_all else pd.DataFrame()
    reg_pred = pd.concat(reg_pred_all, ignore_index=True) if reg_pred_all else pd.DataFrame()
    cls_imp = pd.concat(cls_imp_all, ignore_index=True) if cls_imp_all else pd.DataFrame()
    reg_imp = pd.concat(reg_imp_all, ignore_index=True) if reg_imp_all else pd.DataFrame()
    cls_sel = pd.concat(cls_sel_all, ignore_index=True) if cls_sel_all else pd.DataFrame()
    reg_sel = pd.concat(reg_sel_all, ignore_index=True) if reg_sel_all else pd.DataFrame()

    cls_summary = summarize_classification(cls)
    cls_seed_summary = summarize_classification_by_seed(cls)
    cls_seed_agg = summarize_classification_across_seeds(cls_seed_summary)
    reg_summary = summarize_regression_predictions(reg_pred)
    reg_seed_summary = summarize_regression_by_seed(reg_pred)
    reg_seed_agg = summarize_regression_across_seeds(reg_seed_summary)
    cls.to_csv(out_dir / "15A_classification_fold_metrics.csv", index=False)
    cls_pred.to_csv(out_dir / "15A_classification_predictions.csv", index=False)
    cls_summary.to_csv(out_dir / "15A_classification_summary.csv", index=False)
    cls_seed_summary.to_csv(out_dir / "15A_classification_summary_by_seed.csv", index=False)
    cls_seed_agg.to_csv(out_dir / "15A_classification_summary_across_seeds.csv", index=False)
    reg.to_csv(out_dir / "15A_conditional_abundance_fold_metrics.csv", index=False)
    reg_pred.to_csv(out_dir / "15A_conditional_abundance_predictions.csv", index=False)
    reg_summary.to_csv(out_dir / "15A_conditional_abundance_global_summary.csv", index=False)
    reg_seed_summary.to_csv(out_dir / "15A_conditional_abundance_summary_by_seed.csv", index=False)
    reg_seed_agg.to_csv(out_dir / "15A_conditional_abundance_summary_across_seeds.csv", index=False)
    cls_sel.to_csv(out_dir / "15A_classification_selected_features.csv", index=False)
    reg_sel.to_csv(out_dir / "15A_conditional_abundance_selected_features.csv", index=False)
    cls_imp.to_csv(out_dir / "15A_classification_rf_importance_by_fold.csv", index=False)
    reg_imp.to_csv(out_dir / "15A_conditional_abundance_rf_importance_by_fold.csv", index=False)

    fig_dir = out_dir / "figures"
    classification_summary_plot(cls_summary, fig_dir / "15A_classification_auc_summary", "15A occurrence classification")
    if not cls_seed_agg.empty:
        plot_seed_agg = cls_seed_agg.rename(columns={"AUC_sd_across_seeds": "AUC_sd"}).copy()
        classification_summary_plot(
            plot_seed_agg,
            fig_dir / "15A_classification_auc_summary_across_seeds",
            "15A occurrence classification across seeds",
        )
    global_oof_roc_plot(cls_pred, fig_dir / "15A_classification_global_oof_roc", "15A global out-of-fold ROC")
    regression_scatter_plot(reg_pred, fig_dir / "15A_conditional_abundance_scatter", "15A conditional abundance")
    if not cls_imp.empty:
        imp_summary = cls_imp.groupby(["scenario", "model", "variable"], as_index=False).agg(importance=("importance", "mean"))
        imp_summary.to_csv(out_dir / "15A_classification_rf_importance_summary.csv", index=False)
        for scenario in imp_summary["scenario"].unique():
            g = imp_summary[imp_summary["scenario"] == scenario]
            importance_plot(g, "importance", fig_dir / f"15A_rf_importance_{scenario}", f"15A RF importance {scenario}")
    return cls_summary, reg_summary


def run_16a(df: pd.DataFrame, cfg: dict, out_dir: Path, run_tabpfn: bool):
    predictors = cfg["dummy_cols"] + cfg["rf_best_occurrence_covars"]
    y = df["Aspergillus_presence"].astype(int).to_numpy()
    X = df[predictors].copy()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=cfg["random_state"])
    models = ["RF_same_features"] + (["TabPFN"] if run_tabpfn else [])
    metrics_rows, pred_rows, rf_imp_rows = [], [], []
    for model_name in models:
        for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            X_train, X_test = impute_train_test(X.iloc[train_idx], X.iloc[test_idx])
            y_train, y_test = y[train_idx], y[test_idx]
            if model_name == "RF_same_features":
                model = rf_classifier(cfg["random_state"], cfg["rf_n_estimators"], cfg["rf_n_jobs"])
                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_test)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
                rf_imp_rows.extend(
                    {
                        "seed": cfg["random_state"],
                        "model": model_name,
                        "fold_id": fold_id,
                        "variable": var,
                        "importance": imp,
                    }
                    for var, imp in zip(predictors, model.feature_importances_)
                )
            else:
                model = tabpfn_classifier(cfg["tabpfn_device"])
                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_test)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
            met = classification_metrics(y_test, y_pred, y_prob)
            met.update(
                {
                    "seed": cfg["random_state"],
                    "model": model_name,
                    "fold_id": fold_id,
                    "n_features": len(predictors),
                }
            )
            metrics_rows.append(met)
            pred_rows.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "seed": cfg["random_state"],
                        "fold_id": fold_id,
                        "Sample": df.iloc[test_idx][cfg["id_col"]].to_numpy(),
                        "y_true": y_test,
                        "y_prob": y_prob,
                        "y_pred": y_pred,
                    }
                )
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metrics_rows)
    preds_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    summary = metrics_df.groupby("model", as_index=False).agg(
        AUC_mean=("AUC", "mean"),
        AUC_sd=("AUC", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_sd=("balanced_accuracy", "std"),
        sensitivity_mean=("sensitivity", "mean"),
        sensitivity_sd=("sensitivity", "std"),
        specificity_mean=("specificity", "mean"),
        specificity_sd=("specificity", "std"),
        precision_mean=("precision", "mean"),
        F1_mean=("F1", "mean"),
        n_features_mean=("n_features", "mean"),
    )
    summary_by_seed = metrics_df.groupby(["seed", "model"], as_index=False).agg(
        AUC_mean=("AUC", "mean"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        sensitivity_mean=("sensitivity", "mean"),
        specificity_mean=("specificity", "mean"),
        precision_mean=("precision", "mean"),
        F1_mean=("F1", "mean"),
        n_folds=("fold_id", "count"),
    )
    summary_across_seeds = (
        summary_by_seed.groupby("model", as_index=False)
        .agg(
            AUC_mean=("AUC_mean", "mean"),
            AUC_sd_across_seeds=("AUC_mean", "std"),
            balanced_accuracy_mean=("balanced_accuracy_mean", "mean"),
            balanced_accuracy_sd_across_seeds=("balanced_accuracy_mean", "std"),
            sensitivity_mean=("sensitivity_mean", "mean"),
            specificity_mean=("specificity_mean", "mean"),
            precision_mean=("precision_mean", "mean"),
            F1_mean=("F1_mean", "mean"),
            n_seeds=("seed", "nunique"),
            n_folds_total=("n_folds", "sum"),
        )
        if not summary_by_seed.empty
        else pd.DataFrame()
    )
    global_rows = []
    for model, g in preds_df.groupby("model"):
        global_rows.append(
            {
                "model": model,
                "global_oof_AUC": safe_auc(g["y_true"], g["y_prob"]),
                "global_oof_balanced_accuracy": balanced_accuracy_score(g["y_true"], g["y_pred"]),
                "global_oof_F1": f1_score(g["y_true"], g["y_pred"], zero_division=0),
                "n_predictions": len(g),
            }
        )
    global_summary = pd.DataFrame(global_rows)
    imp = pd.DataFrame(rf_imp_rows)
    imp_summary = imp.groupby("variable", as_index=False).agg(
        mean_importance=("importance", "mean"),
        sd_importance=("importance", "std"),
    ) if not imp.empty else pd.DataFrame()
    metrics_df.to_csv(out_dir / "16A_fold_metrics.csv", index=False)
    preds_df.to_csv(out_dir / "16A_predictions.csv", index=False)
    summary.to_csv(out_dir / "16A_summary.csv", index=False)
    summary_by_seed.to_csv(out_dir / "16A_summary_by_seed.csv", index=False)
    summary_across_seeds.to_csv(out_dir / "16A_summary_across_seeds.csv", index=False)
    global_summary.to_csv(out_dir / "16A_global_oof_summary.csv", index=False)
    pd.DataFrame({"predictor": predictors}).to_csv(out_dir / "16A_predictors_used.csv", index=False)
    imp.to_csv(out_dir / "16A_rf_importance_by_fold.csv", index=False)
    imp_summary.to_csv(out_dir / "16A_rf_importance_summary.csv", index=False)

    fig_dir = out_dir / "figures"
    plot_summary = summary.copy()
    plot_summary["scenario"] = "RFbest_occurrence"
    classification_summary_plot(plot_summary, fig_dir / "16A_auc_summary", "16A RF-best occurrence predictors")
    if not summary_across_seeds.empty:
        plot_seed = summary_across_seeds.rename(columns={"AUC_sd_across_seeds": "AUC_sd"}).copy()
        plot_seed["scenario"] = "RFbest_occurrence"
        classification_summary_plot(plot_seed, fig_dir / "16A_auc_summary_across_seeds", "16A RF-best across seeds")
    if not preds_df.empty:
        preds_plot = preds_df.copy()
        preds_plot["scenario"] = "RFbest_occurrence"
        global_oof_roc_plot(preds_plot, fig_dir / "16A_global_oof_roc", "16A global out-of-fold ROC")
    importance_plot(imp_summary.rename(columns={"mean_importance": "importance"}), "importance", fig_dir / "16A_rf_importance", "16A RF same-features importance")
    return summary


def tabpfn_permutation_ranking(X_train_df, y_train, cfg, candidate_features, forced_features, fold_seed):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=fold_seed)
    inner_train_idx, inner_val_idx = next(splitter.split(X_train_df, y_train))
    all_features = candidate_features + forced_features
    X_inner_train, X_inner_val = impute_train_test(
        X_train_df.iloc[inner_train_idx][all_features],
        X_train_df.iloc[inner_val_idx][all_features],
    )
    y_inner_train = y_train[inner_train_idx]
    y_inner_val = y_train[inner_val_idx]
    model = tabpfn_classifier(cfg["tabpfn_device"])
    model.fit(X_inner_train, y_inner_train)
    baseline = safe_auc(y_inner_val, model.predict_proba(X_inner_val)[:, 1])
    rng = np.random.default_rng(fold_seed)
    rows = []
    pos = {f: i for i, f in enumerate(all_features)}
    for feat in candidate_features:
        drops = []
        for _ in range(5):
            X_perm = X_inner_val.copy()
            X_perm[:, pos[feat]] = rng.permutation(X_perm[:, pos[feat]])
            perm_auc = safe_auc(y_inner_val, model.predict_proba(X_perm)[:, 1])
            drops.append(baseline - perm_auc)
        rows.append(
            {
                "variable": feat,
                "baseline_inner_auc": baseline,
                "mean_auc_drop": np.nanmean(drops),
                "sd_auc_drop": np.nanstd(drops),
                "n_repeats": 5,
            }
        )
    imp = pd.DataFrame(rows).sort_values(["mean_auc_drop", "sd_auc_drop"], ascending=[False, True])
    return imp["variable"].tolist(), imp.reset_index(drop=True)


def run_16b(df: pd.DataFrame, cfg: dict, out_dir: Path, run_tabpfn: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not run_tabpfn:
        write_json(out_dir / "16B_skipped.json", {"status": "skipped", "reason": "TabPFN disabled"})
        return pd.DataFrame()
    blocks = detect_blocks(df, cfg["target_col"], cfg["dummy_cols"])
    candidate_features = blocks.env_cols + blocks.human_cols
    forced_features = blocks.dummy_cols
    all_features = candidate_features + forced_features
    X_all = df[all_features].copy()
    y = df["Aspergillus_presence"].astype(int).to_numpy()
    sample_ids = df[cfg["id_col"]].astype(str).to_numpy()
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=cfg["random_state"])
    metrics_records, pred_records, importance_records, selected_records, ranking_records = [], [], [], [], []
    rf_imp_records = []
    for fold_id, (train_idx, test_idx) in enumerate(outer_cv.split(X_all, y), start=1):
        X_train_raw = X_all.iloc[train_idx].copy()
        X_test_raw = X_all.iloc[test_idx].copy()
        y_train, y_test = y[train_idx], y[test_idx]
        ranked, imp = tabpfn_permutation_ranking(
            X_train_raw, y_train, cfg, candidate_features, forced_features, cfg["random_state"] + fold_id
        )
        imp["outer_fold_id"] = fold_id
        importance_records.append(imp)
        ranking_records.extend({"outer_fold_id": fold_id, "rank": i + 1, "variable": v} for i, v in enumerate(ranked))
        for k in cfg["tabpfn_k_values"]:
            if k == "all":
                selected_covars = ranked
                size_label = "all"
            else:
                selected_covars = ranked[: min(int(k), len(ranked))]
                size_label = str(k)
            selected = selected_covars + forced_features
            selected_records.extend(
                {
                    "outer_fold_id": fold_id,
                    "size_label": size_label,
                    "size_k": len(selected_covars),
                    "rank_within_size": i + 1,
                    "variable": v,
                    "feature_type": "covariate" if v in selected_covars else "forced_dummy",
                }
                for i, v in enumerate(selected)
            )
            X_train, X_test = impute_train_test(X_train_raw[selected], X_test_raw[selected])
            for model_name in ["TabPFN_independent_selection", "RF_on_TabPFN_selected"]:
                if model_name.startswith("TabPFN"):
                    model = tabpfn_classifier(cfg["tabpfn_device"])
                else:
                    model = rf_classifier(cfg["random_state"], cfg["rf_n_estimators"], cfg["rf_n_jobs"])
                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_test)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
                met = classification_metrics(y_test, y_pred, y_prob)
                met.update(
                    {
                        "model": model_name,
                        "outer_fold_id": fold_id,
                        "size_label": size_label,
                        "size_k": len(selected_covars),
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "n_features_total": len(selected),
                    }
                )
                metrics_records.append(met)
                pred_records.append(
                    pd.DataFrame(
                        {
                            "model": model_name,
                            "outer_fold_id": fold_id,
                            "size_label": size_label,
                            "size_k": len(selected_covars),
                            "Sample": sample_ids[test_idx],
                            "y_true": y_test,
                            "y_prob": y_prob,
                            "y_pred": y_pred,
                        }
                    )
                )
                if model_name.startswith("RF"):
                    rf_imp_records.extend(
                        {
                            "outer_fold_id": fold_id,
                            "size_label": size_label,
                            "size_k": len(selected_covars),
                            "variable": var,
                            "importance": val,
                        }
                        for var, val in zip(selected, model.feature_importances_)
                    )
    metrics_df = pd.DataFrame(metrics_records)
    preds_df = pd.concat(pred_records, ignore_index=True)
    importance_df = pd.concat(importance_records, ignore_index=True)
    selected_df = pd.DataFrame(selected_records)
    ranking_df = pd.DataFrame(ranking_records)
    rf_imp_df = pd.DataFrame(rf_imp_records)
    summary = metrics_df.groupby(["model", "size_label", "size_k"], as_index=False).agg(
        AUC_mean=("AUC", "mean"),
        AUC_sd=("AUC", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_sd=("balanced_accuracy", "std"),
        sensitivity_mean=("sensitivity", "mean"),
        specificity_mean=("specificity", "mean"),
        precision_mean=("precision", "mean"),
        F1_mean=("F1", "mean"),
        n_features_total_mean=("n_features_total", "mean"),
    )
    imp_summary = importance_df.groupby("variable", as_index=False).agg(
        mean_auc_drop=("mean_auc_drop", "mean"),
        sd_auc_drop=("mean_auc_drop", "std"),
        times_evaluated=("outer_fold_id", "nunique"),
    ).sort_values("mean_auc_drop", ascending=False)
    metrics_df.to_csv(out_dir / "16B_fold_metrics.csv", index=False)
    preds_df.to_csv(out_dir / "16B_predictions.csv", index=False)
    summary.to_csv(out_dir / "16B_summary_fold_mean.csv", index=False)
    importance_df.to_csv(out_dir / "16B_tabpfn_permutation_importance_by_fold.csv", index=False)
    imp_summary.to_csv(out_dir / "16B_tabpfn_permutation_importance_summary.csv", index=False)
    selected_df.to_csv(out_dir / "16B_selected_features_by_fold_and_k.csv", index=False)
    ranking_df.to_csv(out_dir / "16B_tabpfn_covariate_ranking_by_fold.csv", index=False)
    rf_imp_df.to_csv(out_dir / "16B_rf_reference_importance_by_fold.csv", index=False)
    fig_dir = out_dir / "figures"
    plot_summary = summary.rename(columns={"AUC_mean": "AUC_mean", "AUC_sd": "AUC_sd"}).copy()
    plot_summary["scenario"] = plot_summary["size_label"].map(lambda x: f"k={x}")
    classification_summary_plot(plot_summary, fig_dir / "16B_auc_summary", "16B TabPFN independent covariate selection")
    importance_plot(imp_summary.rename(columns={"mean_auc_drop": "importance"}), "importance", fig_dir / "16B_tabpfn_permutation_importance", "16B TabPFN permutation importance")
    return summary


def read_per_seed_csv(base_dir: Path, seeds: list[int], filename: str) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        path = base_dir / "per_seed" / f"seed_{seed}" / filename
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def regression_seed_dispersion_plot(seed_summary: pd.DataFrame, out_stem: Path, title: str) -> None:
    if seed_summary.empty:
        return
    import matplotlib.pyplot as plt

    df = seed_summary.copy()
    df["label"] = df["scenario"] + " | " + df["model"]
    order = (
        df.groupby("label", as_index=False)
        .agg(RMSE_mean=("RMSE", "mean"))
        .sort_values("RMSE_mean")["label"]
        .tolist()
    )
    fig, ax = plt.subplots(figsize=(max(9, 0.75 * len(order)), 6))
    positions = {label: i for i, label in enumerate(order)}
    for label, g in df.groupby("label"):
        x = np.full(len(g), positions[label])
        ax.scatter(x, g["RMSE"], alpha=0.75, color="#4c78a8", s=30)
        ax.errorbar(
            positions[label],
            g["RMSE"].mean(),
            yerr=g["RMSE"].std(ddof=1) if len(g) > 1 else 0.0,
            fmt="o",
            color="#111111",
            capsize=4,
        )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=55, ha="right")
    ax.set_ylabel("RMSE por semilla")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    from tfms_aspergillus.v3_workflow.plots import save_fig

    save_fig(fig, out_stem)


def consolidate_15a(out_dir: Path, seeds: list[int]) -> None:
    fold_metrics = read_per_seed_csv(out_dir, seeds, "15A_classification_fold_metrics.csv")
    preds = read_per_seed_csv(out_dir, seeds, "15A_classification_predictions.csv")
    reg_preds = read_per_seed_csv(out_dir, seeds, "15A_conditional_abundance_predictions.csv")
    cls_imp = read_per_seed_csv(out_dir, seeds, "15A_classification_rf_importance_by_fold.csv")
    reg_imp = read_per_seed_csv(out_dir, seeds, "15A_conditional_abundance_rf_importance_by_fold.csv")
    if fold_metrics.empty and reg_preds.empty:
        return
    multi_dir = out_dir / "multiseed"
    multi_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = multi_dir / "figures"
    if not fold_metrics.empty:
        cls_by_seed = summarize_classification_by_seed(fold_metrics)
        cls_across = summarize_classification_across_seeds(cls_by_seed)
        fold_metrics.to_csv(multi_dir / "15A_classification_fold_metrics_all_seeds.csv", index=False)
        cls_by_seed.to_csv(multi_dir / "15A_classification_summary_by_seed.csv", index=False)
        cls_across.to_csv(multi_dir / "15A_classification_summary_across_seeds.csv", index=False)
        plot_df = cls_across.rename(columns={"AUC_sd_across_seeds": "AUC_sd"}).copy()
        classification_summary_plot(
            plot_df,
            fig_dir / "15A_classification_auc_across_10_seeds",
            "15A occurrence classification across seeds",
        )
    if not preds.empty:
        preds.to_csv(multi_dir / "15A_classification_predictions_all_seeds.csv", index=False)
    if not reg_preds.empty:
        reg_by_seed = summarize_regression_by_seed(reg_preds)
        reg_across = summarize_regression_across_seeds(reg_by_seed)
        reg_preds.to_csv(multi_dir / "15A_conditional_abundance_predictions_all_seeds.csv", index=False)
        reg_by_seed.to_csv(multi_dir / "15A_conditional_abundance_summary_by_seed.csv", index=False)
        reg_across.to_csv(multi_dir / "15A_conditional_abundance_summary_across_seeds.csv", index=False)
        regression_seed_dispersion_plot(
            reg_by_seed,
            fig_dir / "15A_conditional_abundance_rmse_by_seed",
            "15A conditional abundance RMSE across seeds",
        )
    for name, imp in [
        ("classification", cls_imp),
        ("conditional_abundance", reg_imp),
    ]:
        if imp.empty:
            continue
        imp_summary = (
            imp.groupby(["scenario", "model", "variable"], as_index=False)
            .agg(importance_mean=("importance", "mean"), importance_sd=("importance", "std"), n_folds=("fold_id", "count"))
            .sort_values("importance_mean", ascending=False)
        )
        imp.to_csv(multi_dir / f"15A_{name}_rf_importance_by_fold_all_seeds.csv", index=False)
        imp_summary.to_csv(multi_dir / f"15A_{name}_rf_importance_summary_across_seeds.csv", index=False)
        for scenario in imp_summary["scenario"].unique():
            g = imp_summary[imp_summary["scenario"] == scenario].rename(columns={"importance_mean": "importance"})
            importance_plot(
                g,
                "importance",
                fig_dir / f"15A_{name}_rf_importance_{scenario}_across_seeds",
                f"15A {name} RF importance {scenario} across seeds",
            )


def consolidate_16a(out_dir: Path, seeds: list[int]) -> None:
    metrics = read_per_seed_csv(out_dir, seeds, "16A_fold_metrics.csv")
    preds = read_per_seed_csv(out_dir, seeds, "16A_predictions.csv")
    imp = read_per_seed_csv(out_dir, seeds, "16A_rf_importance_by_fold.csv")
    if metrics.empty:
        return
    multi_dir = out_dir / "multiseed"
    multi_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = multi_dir / "figures"
    by_seed = (
        metrics.groupby(["seed", "model"], as_index=False)
        .agg(
            AUC_mean=("AUC", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            sensitivity_mean=("sensitivity", "mean"),
            specificity_mean=("specificity", "mean"),
            precision_mean=("precision", "mean"),
            F1_mean=("F1", "mean"),
            n_folds=("fold_id", "count"),
        )
        .sort_values(["model", "seed"])
    )
    across = (
        by_seed.groupby("model", as_index=False)
        .agg(
            AUC_mean=("AUC_mean", "mean"),
            AUC_sd_across_seeds=("AUC_mean", "std"),
            balanced_accuracy_mean=("balanced_accuracy_mean", "mean"),
            balanced_accuracy_sd_across_seeds=("balanced_accuracy_mean", "std"),
            sensitivity_mean=("sensitivity_mean", "mean"),
            specificity_mean=("specificity_mean", "mean"),
            precision_mean=("precision_mean", "mean"),
            F1_mean=("F1_mean", "mean"),
            n_seeds=("seed", "nunique"),
            n_folds_total=("n_folds", "sum"),
        )
        .sort_values("AUC_mean", ascending=False)
    )
    metrics.to_csv(multi_dir / "16A_fold_metrics_all_seeds.csv", index=False)
    by_seed.to_csv(multi_dir / "16A_summary_by_seed.csv", index=False)
    across.to_csv(multi_dir / "16A_summary_across_seeds.csv", index=False)
    plot_df = across.rename(columns={"AUC_sd_across_seeds": "AUC_sd"}).copy()
    plot_df["scenario"] = "RFbest_occurrence"
    classification_summary_plot(plot_df, fig_dir / "16A_auc_across_10_seeds", "16A RF-best occurrence across seeds")
    if not preds.empty:
        preds.to_csv(multi_dir / "16A_predictions_all_seeds.csv", index=False)
        global_rows = []
        for (seed, model), g in preds.groupby(["seed", "model"]):
            global_rows.append(
                {
                    "seed": seed,
                    "model": model,
                    "global_oof_AUC": safe_auc(g["y_true"], g["y_prob"]),
                    "global_oof_balanced_accuracy": balanced_accuracy_score(g["y_true"], g["y_pred"]),
                    "global_oof_F1": f1_score(g["y_true"], g["y_pred"], zero_division=0),
                    "n_predictions": len(g),
                }
            )
        pd.DataFrame(global_rows).to_csv(multi_dir / "16A_global_oof_summary_by_seed.csv", index=False)
    if not imp.empty:
        imp_summary = (
            imp.groupby("variable", as_index=False)
            .agg(mean_importance=("importance", "mean"), sd_importance=("importance", "std"), n_folds=("fold_id", "count"))
            .sort_values("mean_importance", ascending=False)
        )
        imp.to_csv(multi_dir / "16A_rf_importance_by_fold_all_seeds.csv", index=False)
        imp_summary.to_csv(multi_dir / "16A_rf_importance_summary_across_seeds.csv", index=False)
        importance_plot(
            imp_summary.rename(columns={"mean_importance": "importance"}),
            "importance",
            fig_dir / "16A_rf_importance_across_seeds",
            "16A RF same-features importance across seeds",
        )


def consolidate_multiseed(outputs: Path, seeds: list[int]) -> None:
    consolidate_15a(outputs / "15A_validated_v3_workflow", seeds)
    consolidate_16a(outputs / "16A_rfbest_occurrence", seeds)
    write_json(outputs / "multiseed_metadata.json", {"seeds": seeds, "n_seeds": len(seeds)})


def descriptive_profile(df: pd.DataFrame, cfg: dict, out_dir: Path) -> None:
    blocks = detect_blocks(df, cfg["target_col"], cfg["dummy_cols"])
    medium = medium_labels(df, blocks.dummy_cols)
    profile = {
        "shape": list(df.shape),
        "target": cfg["target_col"],
        "target_zeros": int((df[cfg["target_col"]] == 0).sum()),
        "target_positive": int((df[cfg["target_col"]] > 0).sum()),
        "env_cols": len(blocks.env_cols),
        "human_cols": len(blocks.human_cols),
        "bacteria_cols": len(blocks.bacteria_cols),
        "fungi_cols": len(blocks.fungi_cols),
        "microbiome_predictors": len(blocks.microbiome_predictors),
    }
    write_json(out_dir / "data_profile.json", profile)
    pd.crosstab(pd.Series(medium, name="medium"), df["Aspergillus_presence"], colnames=["presence"]).to_csv(
        out_dir / "medium_by_presence.csv"
    )
    df[cfg["target_col"]].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_csv(
        out_dir / "target_distribution.csv"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "v3_repro_config.json"))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--run-15a", action="store_true")
    parser.add_argument("--run-16a", action="store_true")
    parser.add_argument("--run-16b", action="store_true")
    parser.add_argument("--run-tabpfn", action="store_true")
    parser.add_argument("--rf-n-jobs", type=int, default=None)
    parser.add_argument("--rf-n-estimators", type=int, default=None)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds, e.g. 123,124,125")
    parser.add_argument("--n-seeds", type=int, default=None, help="Use random_state..random_state+n-1")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.rf_n_jobs is not None:
        cfg["rf_n_jobs"] = args.rf_n_jobs
    if args.rf_n_estimators is not None:
        cfg["rf_n_estimators"] = args.rf_n_estimators
    base = Path(args.config).resolve().parents[1]
    input_csv = (base / cfg["input_csv"]).resolve()
    outputs = (base / cfg["outputs_dir"]).resolve()
    outputs.mkdir(parents=True, exist_ok=True)
    run_tabpfn = bool(args.run_tabpfn)
    if run_tabpfn and not tabpfn_available():
        print("[WARN] --run-tabpfn requested but tabpfn is not importable; TabPFN steps will be skipped.", flush=True)
        run_tabpfn = False
    df = load_table(input_csv, cfg["target_col"], cfg["id_col"])
    seeds = parse_seeds(args.seeds, args.n_seeds, int(cfg["random_state"]))
    descriptive_profile(df, cfg, outputs / "00_data_profile")
    write_json(
        outputs / "run_metadata.json",
        {
            "input_csv": str(input_csv),
            "run_tabpfn": run_tabpfn,
            "rf_n_estimators": cfg["rf_n_estimators"],
            "rf_n_jobs": cfg["rf_n_jobs"],
            "seeds": seeds,
        },
    )
    if len(seeds) == 1:
        cfg_single = copy.deepcopy(cfg)
        cfg_single["random_state"] = int(seeds[0])
        print(f"=== Running seed {seeds[0]} ===", flush=True)
        if args.all or args.run_15a:
            run_15a(df, cfg_single, outputs / "15A_validated_v3_workflow", run_tabpfn)
        if args.all or args.run_16a:
            run_16a(df, cfg_single, outputs / "16A_rfbest_occurrence", run_tabpfn)
        if args.all or args.run_16b:
            run_16b(df, cfg_single, outputs / "16B_tabpfn_independent_selection", run_tabpfn)
    else:
        for seed in seeds:
            cfg_seed = copy.deepcopy(cfg)
            cfg_seed["random_state"] = int(seed)
            seed_suffix = f"seed_{seed}"
            print(f"=== Running seed {seed} ===", flush=True)
            if args.all or args.run_15a:
                run_15a(df, cfg_seed, outputs / "15A_validated_v3_workflow" / "per_seed" / seed_suffix, run_tabpfn)
            if args.all or args.run_16a:
                run_16a(df, cfg_seed, outputs / "16A_rfbest_occurrence" / "per_seed" / seed_suffix, run_tabpfn)
            if args.all or args.run_16b:
                run_16b(df, cfg_seed, outputs / "16B_tabpfn_independent_selection" / "per_seed" / seed_suffix, run_tabpfn)
        consolidate_multiseed(outputs, seeds)
    print(f"Done. Outputs: {outputs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
