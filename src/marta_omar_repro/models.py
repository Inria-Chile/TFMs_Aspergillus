from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer


def select_features_inside_fold(
    X_train: pd.DataFrame,
    y_train,
    X_test: pd.DataFrame,
    task: str,
    max_features: int | None,
    forced_cols: list[str],
):
    all_cols = list(X_train.columns)
    forced_cols = [c for c in forced_cols if c in all_cols]
    candidate_cols = [c for c in all_cols if c not in forced_cols]
    if not candidate_cols:
        selected = forced_cols
        return X_train[selected], X_test[selected], selected
    k = len(candidate_cols) if max_features is None else min(max_features, len(candidate_cols))
    imputer = SimpleImputer(strategy="median")
    X_candidate = imputer.fit_transform(X_train[candidate_cols])
    selector = SelectKBest(score_func=f_classif if task == "classification" else f_regression, k=k)
    selector.fit(X_candidate, y_train)
    selected_candidate = list(np.array(candidate_cols)[selector.get_support()])
    selected = selected_candidate + forced_cols
    return X_train[selected], X_test[selected], selected


def impute_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame):
    imputer = SimpleImputer(strategy="median")
    return imputer.fit_transform(X_train), imputer.transform(X_test)


def rf_classifier(random_state: int, n_estimators: int, n_jobs: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=n_jobs,
    )


def rf_regressor(random_state: int, n_estimators: int, n_jobs: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=n_jobs,
    )


def tabpfn_available() -> bool:
    try:
        import tabpfn  # noqa: F401

        return True
    except Exception:
        return False


def tabpfn_classifier(device: str):
    from tabpfn import TabPFNClassifier

    try:
        return TabPFNClassifier(device=device)
    except TypeError:
        return TabPFNClassifier()


def tabpfn_regressor(device: str):
    from tabpfn import TabPFNRegressor

    try:
        return TabPFNRegressor(device=device)
    except TypeError:
        return TabPFNRegressor()
