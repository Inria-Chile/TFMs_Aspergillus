from __future__ import annotations

import pandas as pd
from sklearn.model_selection import KFold, LeaveOneOut, RepeatedStratifiedKFold


def classification_cv(n_splits: int, n_repeats: int, random_state: int) -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )


def regression_cv(data_pos: pd.DataFrame, use_loocv: bool = True, n_splits: int = 5):
    if use_loocv:
        return LeaveOneOut().split(data_pos)
    k = min(n_splits, len(data_pos))
    return KFold(n_splits=k, shuffle=True, random_state=123).split(data_pos)
