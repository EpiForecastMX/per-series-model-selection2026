"""Ablation of the candidate pool (Table 3 of the paper).

How much of the result depends on having four candidates? The selection rule is
re-run restricted to each subset of the pool and the resulting assignments are
rescored -- with no refitting, since the rule only adjudicates among forecasts
that were already locked.

Two regimes are separated:

  static   the rule selects once from cross-validated evidence, as in the
           frozen baseline assignment;
  dynamic  the rule reselects on the early 2026 weeks and is scored on the
           later weeks, which the reselection never observes.

WEEK ALIGNMENT. Preprocessing shifts the bulletin week down by one, so a forecast
stored under model week N carries the measurement the bulletin published as week
N+1. Everything here is in BULLETIN numbering; see the README.

Policies are compared PAIRWISE over the same series (Wilcoxon signed-rank).
Comparing each policy's own reassigned subset would compare different sets of
series and is not a valid contrast -- this is the single easiest mistake to make
in this experiment.

Inputs (released under ./data/; assembled from the public SINAVE bulletin and
the frozen pre-2026 selection record):
  data/cv_metrics_per_series.csv    series, stratum, model, smape, mase, rmse
  data/weekly_2026_per_series.csv   series, stratum, week, observed, <model cols>
  data/frozen_assignment.csv        series, stratum, model   (locked before 2026)

Output: data/ablation_results.csv, plus the released copy in this repository.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

MODELS = ["prophet", "deepar", "ensemble", "stacking"]
BAND = 1.05  # selection band on sMAPE, as in the paper
SPLIT = 11  # last week used for reselection in the dynamic regime


def smape(y: np.ndarray, f: np.ndarray) -> float:
    y, f = np.asarray(y, float), np.asarray(f, float)
    denom = (np.abs(y) + np.abs(f)) / 2
    keep = denom > 0
    return float(100 * np.mean(np.abs(y - f)[keep] / denom[keep]))


def mae(y: np.ndarray, f: np.ndarray) -> float:
    """Mean absolute error used to order MASE within one series/window.

    The seasonal-naive denominator is common to every candidate for a given
    series and window.  Dividing every MAE by that same positive constant does
    not change the MASE ordering, so the released 2026 panel is sufficient for
    the rule's MASE tie-break without redistributing the 2025 history.
    """
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(f, float))))


def rmse(y: np.ndarray, f: np.ndarray) -> float:
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.sqrt(np.mean((y - f) ** 2)))


def select(row: pd.Series, pool: tuple[str, ...]) -> str | None:
    """The published rule, restricted to `pool`: sMAPE band, then MASE, then RMSE."""
    s = {m: row[f"{m}_smape"] for m in pool if pd.notna(row.get(f"{m}_smape"))}
    if not s:
        return None
    order = sorted(s, key=lambda m: s[m])
    best = s[order[0]]
    if len(order) == 1 or s[order[1]] > BAND * best:
        return order[0]
    tied = [m for m in s if s[m] <= BAND * best]
    return min(tied, key=lambda m: (row[f"{m}_mase"], row[f"{m}_rmse"]))


def pools() -> list[tuple[str, ...]]:
    full = tuple(MODELS)
    loo = [tuple(m for m in MODELS if m != drop) for drop in MODELS]
    singles = [(m,) for m in MODELS]
    return [full, *loo, *singles]


def static_ablation(cv: pd.DataFrame, oos: pd.DataFrame) -> pd.DataFrame:
    """One row per series and pool: the OOS sMAPE of the model that pool selects."""
    key = oos.set_index(["series", "stratum"])
    rows = []
    for _, row in cv.iterrows():
        k = (row["series"], row["stratum"])
        if k not in key.index:
            continue
        entry = {"series": k[0], "stratum": k[1]}
        for pool in pools():
            m = select(row, pool)
            entry["+".join(pool)] = key.loc[k, f"smape_{m}"] if m else np.nan
        rows.append(entry)
    return pd.DataFrame(rows)


def dynamic_ablation(weekly: pd.DataFrame, frozen: pd.DataFrame) -> pd.DataFrame:
    """Re-run the complete published rule early, then score on later weeks."""
    base = {(r["series"], r["stratum"]): r["model"] for _, r in frozen.iterrows()}
    rows = []
    for (ser, stratum), g in weekly.groupby(["series", "stratum"]):
        early, late = g[g.week <= SPLIT], g[g.week > SPLIT]
        if len(early) < 4 or len(late) < 4:
            continue
        err = {
            m: {
                "early_smape": smape(early.observed, early[m]),
                # MAE and MASE induce the same ordering here; see ``mae``.
                "early_mase_order": mae(early.observed, early[m]),
                "early_rmse": rmse(early.observed, early[m]),
                "late_smape": smape(late.observed, late[m]),
            }
            for m in MODELS
            if m in g
        }
        if len(err) < len(MODELS) or (ser, stratum) not in base:
            continue
        entry = {
            "series": ser,
            "stratum": stratum,
            "frozen": err[base[(ser, stratum)]]["late_smape"],
        }
        evidence = pd.Series(
            {
                key: value
                for m in MODELS
                for key, value in (
                    (f"{m}_smape", err[m]["early_smape"]),
                    (f"{m}_mase", err[m]["early_mase_order"]),
                    (f"{m}_rmse", err[m]["early_rmse"]),
                )
            }
        )
        for pool in pools():
            selected = select(evidence, pool)
            if selected is None:  # all four candidates were checked above
                raise RuntimeError(
                    f"no eligible model for {ser!r}, {stratum!r}, {pool!r}"
                )
            entry["+".join(pool)] = err[selected]["late_smape"]
        rows.append(entry)
    return pd.DataFrame(rows)


def summarize(per_series: pd.DataFrame, label: str) -> pd.DataFrame:
    """Median per policy, with a paired test against the full pool."""
    ref = "+".join(MODELS)
    cols = [c for c in per_series.columns if c not in ("series", "stratum", "frozen")]
    out = []
    for c in cols:
        v = per_series[c].dropna()
        p = (
            np.nan
            if c == ref
            else float(
                wilcoxon(per_series[c], per_series[ref], zero_method="zsplit").pvalue
            )
        )
        out.append(
            {
                "regime": label,
                "policy": c,
                "n": len(v),
                # Sin redondeo agresivo. A tres decimales el CSV guardaba 0.695
                # para un p cuyo valor real es 0.6949...: quien redondee el CSV a
                # dos decimales no recupera con certeza lo que el paper imprime.
                # Seis decimales bastan para reproducir cualquier cifra publicada.
                "median_smape": round(float(v.median()), 6),
                "mean_smape": round(float(v.mean()), 6),
                "p_vs_full_pool": None if np.isnan(p) else round(p, 6),
            }
        )
    return pd.DataFrame(out)


if __name__ == "__main__":
    cv = pd.read_csv("data/cv_metrics_per_series.csv")
    cv = cv.pivot_table(
        index=["series", "stratum"], columns="model", values=["smape", "mase", "rmse"]
    )
    cv.columns = [f"{m}_{stat}" for stat, m in cv.columns]
    cv = cv.reset_index()

    oos = pd.read_csv("data/per_series_out_of_sample_2026.csv").rename(
        columns={"entity": "series"}
    )
    weekly = pd.read_csv("data/weekly_2026_per_series.csv")
    frozen = pd.read_csv("data/frozen_assignment.csv")

    est = summarize(static_ablation(cv, oos), "static")
    dynamic_per_series = dynamic_ablation(weekly, frozen)
    din = summarize(dynamic_per_series, "dynamic")
    res = pd.concat([est, din], ignore_index=True)
    baseline_index = len(res)
    res.loc[baseline_index, "regime"] = "dynamic"
    res.loc[baseline_index, "policy"] = "frozen_baseline"
    res.loc[baseline_index, "n"] = len(dynamic_per_series)
    res.loc[baseline_index, "median_smape"] = round(
        float(dynamic_per_series["frozen"].median()), 6
    )
    res["n"] = res["n"].astype(int)
    res.to_csv("data/ablation_results.csv", index=False)
    print(res.to_string(index=False))
