"""Minimum-trace (MinT) hierarchical reconciliation sanity check.

The three demographic strata (general = women + men) are forecast
independently during selection; this script reconciles the locked national
forecasts post-hoc with a diagonal weighting built from the per-stratum
cross-validation error variances (MinT WLS / variance scaling, since the full
cross-covariance is not available). It reports the incoherence before/after
and the magnitude of the reconciliation adjustment.

Input (./data/): national_strata_forecast.csv
  columns: week, fc_general, fc_women, fc_men
CV error standard deviations per stratum are passed on the command line or
taken from the defaults below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# per-stratum cross-validation error scale (general, women, men); override as needed
CV_RMSE = np.array([372.23, 239.42, 101.44])


def reconcile(fc: np.ndarray, cv_rmse: np.ndarray = CV_RMSE) -> np.ndarray:
    """fc: 3 x T array of base forecasts [general; women; men]. Returns the
    coherent reconciled 3 x T array (general = women + men)."""
    S = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])   # rows g/w/m; cols women/men
    W_inv = np.linalg.inv(np.diag(cv_rmse ** 2))
    G = np.linalg.inv(S.T @ W_inv @ S) @ S.T @ W_inv      # 2 x 3
    return S @ (G @ fc)


def incoherence(a: np.ndarray) -> float:
    """Mean weekly |women + men - general| / general (%)."""
    return float(np.mean(np.abs(a[1] + a[2] - a[0]) / a[0]) * 100)


if __name__ == "__main__":
    df = pd.read_csv("data/national_strata_forecast.csv")
    base = df[["fc_general", "fc_women", "fc_men"]].to_numpy(float).T
    rec = reconcile(base)
    shift = {s: float(np.mean(np.abs(rec[i] - base[i]) / base[i]) * 100)
             for i, s in enumerate(["general", "women", "men"])}
    print(f"Incoherence  pre={incoherence(base):.2f}%  post={incoherence(rec):.3f}%")
    print("Mean weekly reconciliation shift per stratum (%):",
          {k: round(v, 2) for k, v in shift.items()})
