from sklearn.model_selection import KFold, LeaveOneOut, RepeatedStratifiedKFold


def classification_splits(y, n_splits: int, n_repeats: int, seed: int):
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(y, y)


def regression_splits(n_rows: int, strategy: str, n_splits: int = 5, seed: int = 0):
    if strategy == "leave_one_out":
        return LeaveOneOut().split(range(n_rows))
    return KFold(n_splits=min(n_splits, n_rows), shuffle=True, random_state=seed).split(range(n_rows))
