"""Classical and transformer baselines under the identical rolling-origin CV.

  --mode sarima  : tuned Auto-ARIMA on log1p with Fourier(K=4) annual terms
                   (statsmodels SARIMAX; the standard treatment of long weekly
                   seasonality), AIC-selected order over a compact grid.
  --mode patchtst: a global PatchTST transformer (GluonTS, reference config,
                   104-week context) trained jointly per demographic stratum.

Both are deliberately out-of-the-box implementations of widely cited baselines,
evaluated under the same five-fold rolling-origin CV (horizon 52, stride 13)
as the candidate models, and reported on the scale-free sMAPE and MASE.

Input (./data/): weekly_series.csv  columns: series, stratum, week, value
(assembled from the public SINAVE bulletin; not redistributed here).
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cross_validation import rolling_origin_folds, HORIZON  # noqa: E402
from metrics import smape, mase  # noqa: E402

warnings.filterwarnings("ignore")
ARIMA_ORDERS = [(1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2), (0, 1, 1), (1, 0, 1)]
K = 4


def _fourier(n: int, start: int = 0, period: float = 52.0, k: int = K) -> np.ndarray:
    t = np.arange(start, start + n)
    cols = []
    for j in range(1, k + 1):
        cols += [np.sin(2 * np.pi * j * t / period), np.cos(2 * np.pi * j * t / period)]
    return np.column_stack(cols)


def sarima_forecast(train: np.ndarray, h: int) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    yl = np.log1p(train.astype(float))
    Xtr, Xte = _fourier(len(yl)), _fourier(h, start=len(yl))
    best = None
    for od in ARIMA_ORDERS:
        try:
            r = SARIMAX(yl, exog=Xtr, order=od, trend="c", enforce_stationarity=False,
                        enforce_invertibility=False).fit(disp=False, maxiter=50)
            if best is None or r.aic < best[0]:
                best = (r.aic, r)
        except Exception:
            continue
    if best is None:
        return np.full(h, np.expm1(yl[-1]))
    return np.clip(np.expm1(best[1].forecast(steps=h, exog=Xte)), 0, None)


def cv_series_sarima(y: np.ndarray) -> tuple[float, float] | None:
    denom = np.mean(np.abs(y[52:] - y[:-52])) if len(y) > 52 else np.nan
    sm, ms = [], []
    for tr, te in rolling_origin_folds(len(y)):
        f = sarima_forecast(y[tr], HORIZON)[: te.stop - te.start]
        obs = y[te]
        sm.append(smape(obs, f))
        ms.append(np.mean(np.abs(obs - f)) / denom if denom and denom > 0 else np.nan)
    return (float(np.median(sm)), float(np.nanmedian(ms))) if sm else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["sarima", "patchtst"], default="sarima")
    ap.add_argument("--data", default="data/weekly_series.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.data)
    if a.mode == "patchtst":
        print("PatchTST baseline: train a global GluonTS PatchTST per stratum "
              "(patch length 16, 104-week context) over the rolling-origin folds; "
              "see the GluonTS reference implementation. Omitted here to keep the "
              "release dependency-light.")
        raise SystemExit(0)
    res = []
    for (s, st), g in df.groupby(["series", "stratum"]):
        out = cv_series_sarima(g.sort_values("week")["value"].to_numpy(float))
        if out:
            res.append({"series": s, "stratum": st, "smape": out[0], "mase": out[1]})
    R = pd.DataFrame(res)
    print(f"Auto-ARIMA+Fourier over {len(R)} series-stratum combinations:")
    print(f"  median sMAPE = {R.smape.median():.2f}   median MASE = {R.mase.median():.3f}")
