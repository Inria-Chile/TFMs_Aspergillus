import numpy as np
import pandas as pd


def clr_transform(train: pd.DataFrame, test: pd.DataFrame, pseudocount: float = 1e-6):
    """Fit CLR inputs fold-locally; test rows never determine train parameters."""
    if (train < 0).any().any() or (test < 0).any().any():
        raise ValueError("CLR requires non-negative abundance inputs")
    train_pc = train.astype(float) + pseudocount
    test_pc = test.astype(float) + pseudocount
    train_log = np.log(train_pc)
    test_log = np.log(test_pc)
    return train_log.sub(train_log.mean(axis=1), axis=0), test_log.sub(test_log.mean(axis=1), axis=0)
