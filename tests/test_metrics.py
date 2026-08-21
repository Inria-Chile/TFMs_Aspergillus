import numpy as np
from tfms_aspergillus.metrics import classification_metrics, regression_metrics


def test_metrics_are_finite_for_basic_inputs():
    out = classification_metrics([0, 1], [0, 1], [0.1, 0.9])
    assert np.isclose(out["AUC"], 1.0)
    reg = regression_metrics([1.0, 2.0], [1.0, 3.0])
    assert reg["RMSE"] > 0
