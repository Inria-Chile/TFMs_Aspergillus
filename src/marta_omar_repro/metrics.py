from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def safe_auc(y_true, y_prob) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return np.nan


def classification_metrics(y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "AUC": safe_auc(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def regression_metrics(y_true, y_pred) -> dict:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    try:
        r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = np.nan
    try:
        pearson_corr = np.corrcoef(y_true, y_pred)[0, 1]
    except Exception:
        pearson_corr = np.nan
    try:
        spearman_corr, spearman_p = spearmanr(y_true, y_pred)
    except Exception:
        spearman_corr, spearman_p = np.nan, np.nan
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "pearson_correlation": pearson_corr,
        "spearman_correlation": spearman_corr,
        "spearman_p": spearman_p,
    }


def summarize_classification(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    if fold_metrics.empty:
        return fold_metrics
    return (
        fold_metrics.groupby(["scenario", "model"], as_index=False)
        .agg(
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
            n_features_mean=("n_features_used", "mean"),
        )
        .sort_values(["AUC_mean", "balanced_accuracy_mean"], ascending=[False, False])
    )
