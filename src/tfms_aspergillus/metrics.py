import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


def classification_metrics(y_true, y_pred, y_prob):
    return {"AUC": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else np.nan,
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "Balanced accuracy": balanced_accuracy_score(y_true, y_pred)}


def regression_metrics(y_true, y_pred):
    return {"RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
            "MAE": mean_absolute_error(y_true, y_pred), "R2": r2_score(y_true, y_pred)}
