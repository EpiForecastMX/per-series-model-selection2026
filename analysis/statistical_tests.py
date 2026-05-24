"""Significance tests reported in the paper.

(1) One-sided Wilcoxon signed-rank on the paired per-series cross-validation
    sMAPE values: is the global recurrent model below each rival?
(2) Diebold-Mariano (Harvey-Leybourne-Newbold small-sample correction) on the
    national-general 2026 squared-error losses: locked forecast vs. each rival.

Inputs (place under ./data/, not redistributed; assembled from the public
SINAVE bulletin):
  data/cv_metrics_per_series.csv   columns: series, stratum, model, smape
  data/national_2026_weekly.csv    columns: week, observed, <model columns...>
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from scipy.stats import t as t_dist
from statsmodels.tsa.stattools import acovf

RIVALS = ["prophet", "ensemble", "stacking"]
RECURRENT = "deepar"
LOCKED_NATIONAL = "ensemble"   # selected national-general model


def wilcoxon_cv(path: str = "data/cv_metrics_per_series.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    pivot = df.pivot_table(index=["series", "stratum"], columns="model", values="smape")
    rows = []
    for rival in RIVALS:
        stat, p = wilcoxon(pivot[RECURRENT], pivot[rival], alternative="less")
        rows.append({"comparison": f"{RECURRENT}<{rival}", "W": stat,
                     "p_value": p, "n": len(pivot)})
    return pd.DataFrame(rows)


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    gamma = acovf(d, nlag=max(h - 1, 0), fft=False)
    var_d = (gamma[0] + (2 * sum(gamma[1:h]) if h > 1 else 0)) / n
    dm = d.mean() / np.sqrt(var_d)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * (1 - t_dist.cdf(abs(hln), df=n - 1))
    return float(hln), float(p)


def dm_2026(path: str = "data/national_2026_weekly.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["week"] >= 2]   # drop W01 (structurally incomplete)
    obs = df["observed"].to_numpy(float)
    err = {m: obs - df[m].to_numpy(float) for m in [LOCKED_NATIONAL, *RIVALS] if m in df}
    rows = []
    for rival in [m for m in RIVALS if m in err]:
        hln, p = diebold_mariano(err[LOCKED_NATIONAL], err[rival])
        rows.append({"comparison": f"{LOCKED_NATIONAL}_vs_{rival}",
                     "DM_HLN": round(hln, 3), "p_value": round(p, 3), "n": len(df)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Wilcoxon (cross-validation):")
    print(wilcoxon_cv().to_string(index=False))
    print("\nDiebold-Mariano (2026 W02-W18):")
    print(dm_2026().to_string(index=False))
