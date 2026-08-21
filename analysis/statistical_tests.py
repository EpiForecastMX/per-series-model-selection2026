"""Significance tests reported in the paper.

(1) One-sided Wilcoxon signed-rank on the paired per-series cross-validation
    sMAPE values: is the global recurrent model below each rival?
(2) Diebold-Mariano (Harvey-Leybourne-Newbold small-sample correction) on the
    national-general 2026 squared-error losses: locked forecast vs. each rival.
    The three comparisons share the same weeks, so raw p-values are reported
    alongside a Holm correction for multiplicity; neither should be read alone.

WEEK ALIGNMENT. The preprocessing step shifts the bulletin week down by one, so a
forecast stored under model week N carries the measurement the bulletin published
as week N+1. Scoring against the bulletin therefore requires mapping forecasts back
to bulletin numbering (bulletin_week = model_week + 1). The `week` column of the
input below is already in BULLETIN numbering.

Inputs (place under ./data/, not redistributed; assembled from the public
SINAVE bulletin):
  data/cv_metrics_per_series.csv   columns: series, stratum, model, smape
  data/national_2026_weekly.csv    columns: week, observed, <model columns...>

Also in ./data/: per_series_out_of_sample_2026.csv, the per-entity out-of-sample
table referenced by the paper (sMAPE and MASE per model, 111 series, W03-W18).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from scipy.stats import t as t_dist
from statsmodels.tsa.stattools import acovf

RIVALS = ["prophet", "ensemble", "stacking"]
RECURRENT = "deepar"
LOCKED_NATIONAL = "ensemble"  # selected national-general model


def wilcoxon_cv(path: str = "data/cv_metrics_per_series.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    pivot = df.pivot_table(index=["series", "stratum"], columns="model", values="smape")
    rows = []
    for rival in RIVALS:
        stat, p = wilcoxon(pivot[RECURRENT], pivot[rival], alternative="less")
        rows.append(
            {
                "comparison": f"{RECURRENT}<{rival}",
                "W": stat,
                "p_value": p,
                "n": len(pivot),
            }
        )
    return pd.DataFrame(rows)


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    d = e1**2 - e2**2
    n = len(d)
    gamma = acovf(d, nlag=max(h - 1, 0), fft=False)
    var_d = (gamma[0] + (2 * sum(gamma[1:h]) if h > 1 else 0)) / n
    dm = d.mean() / np.sqrt(var_d)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * (1 - t_dist.cdf(abs(hln), df=n - 1))
    return float(hln), float(p)


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjustment, returned in the original order."""
    m = len(pvals)
    orden = sorted(range(m), key=lambda i: pvals[i])
    ajustados = [0.0] * m
    previo = 0.0
    for rango, i in enumerate(orden):
        previo = max(min(1.0, (m - rango) * pvals[i]), previo)
        ajustados[i] = previo
    return ajustados


def dm_2026(path: str = "data/national_2026_weekly.csv") -> pd.DataFrame:
    """W02-W18 in bulletin numbering; W01 has no locked forecast under the alignment."""
    df = pd.read_csv(path)
    df = df[df["week"] >= 2]
    obs = df["observed"].to_numpy(float)
    err = {
        m: obs - df[m].to_numpy(float) for m in [LOCKED_NATIONAL, *RIVALS] if m in df
    }
    rivales = [m for m in RIVALS if m in err]
    crudos, stats = [], []
    for rival in rivales:
        hln, p = diebold_mariano(err[LOCKED_NATIONAL], err[rival])
        stats.append(hln)
        crudos.append(p)
    ajustados = holm(crudos)
    return pd.DataFrame(
        [
            {
                "comparison": f"{LOCKED_NATIONAL}_vs_{r}",
                "DM_HLN": round(s, 4),
                "p_raw": round(p, 4),
                "p_holm": round(a, 4),
                "n": len(df),
            }
            for r, s, p, a in zip(rivales, stats, crudos, ajustados)
        ]
    )


if __name__ == "__main__":
    print("Wilcoxon (cross-validation):")
    print(wilcoxon_cv().to_string(index=False))
    print("\nDiebold-Mariano (2026 W02-W18, bulletin weeks):")
    print(dm_2026().to_string(index=False))
    print("\np_raw is unadjusted; p_holm corrects for the three comparisons.")
