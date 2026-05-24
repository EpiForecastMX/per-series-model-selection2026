"""Forecasting evaluation metrics.

All functions take 1-D numeric arrays and return scalar floats. MASE also
requires the training series to scale by the seasonal-naive benchmark.
These are the metrics defined in the paper's Evaluation Protocol.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Symmetric MAPE in [0, 200] (%). Pairs where both values are zero are
    excluded. Scale-normalised and symmetric in the sign of the error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if not mask.any():
        return 0.0
    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def mase(y_true: ArrayLike, y_pred: ArrayLike, y_train: ArrayLike, season: int = 52) -> float:
    """Mean Absolute Scaled Error against a seasonal-naive benchmark (lag
    `season`). A value below 1 beats seasonal persistence regardless of scale."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    y_train = np.asarray(y_train, float)
    if len(y_train) <= season:
        return float("nan")
    denom = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    if denom == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def poisson_deviance(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Poisson deviance, a count-appropriate error for the weekly counts."""
    y_true = np.asarray(y_true, float)
    y_pred = np.clip(np.asarray(y_pred, float), 1e-9, None)
    term = np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0.0)
    return float(np.mean(2.0 * (term - (y_true - y_pred))))
