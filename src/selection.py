"""Deterministic, auditable per-series model selection.

Reference implementation of the rule described in the paper's
"Model Selection Strategy": rank candidates on cross-validated sMAPE with
a 5% relative margin, break ties on MASE then RMSE, and route structurally
low-incidence series to their socio-urban region's model. The rule depends
only on past observations and emits an auditable record per series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

REL_MARGIN = 0.05  # 5% relative-sMAPE band for the primary criterion
LOW_INCIDENCE_CASES = 5  # cumulative-count threshold over the trailing window
TRAILING_WEEKS = 52


@dataclass
class CVMetrics:
    smape: float
    mase: float
    rmse: float


def select_model(candidates: Mapping[str, CVMetrics]) -> tuple[str, str]:
    """Return (selected_model, justification) for one series-stratum.

    Tier 1 (sMAPE): if the best model beats the second best by more than a
        5% relative margin, select it.
    Tier 2 (MASE): otherwise, among models within the 5% sMAPE band of the
        best, select the lowest MASE.
    Tier 3 (RMSE): if MASE is tied, select the lowest RMSE.
    """
    ranked = sorted(candidates.items(), key=lambda kv: kv[1].smape)
    best_name, best = ranked[0]
    if len(ranked) == 1:
        return best_name, "only candidate"

    second = ranked[1][1]
    rel_gap = (
        (second.smape - best.smape) / best.smape if best.smape > 0 else float("inf")
    )
    if rel_gap > REL_MARGIN:
        return best_name, f"sMAPE (relative gap {rel_gap:.1%} > {REL_MARGIN:.0%})"

    # within-band tiebreak on MASE, then RMSE
    band = {
        n: m
        for n, m in candidates.items()
        if best.smape == 0 or (m.smape - best.smape) / best.smape <= REL_MARGIN
    }
    by_mase = sorted(band.items(), key=lambda kv: kv[1].mase)
    if len(by_mase) == 1 or by_mase[0][1].mase < by_mase[1][1].mase:
        return by_mase[0][0], "MASE tiebreaker (within 5% sMAPE band)"
    tied = [n for n, m in band.items() if m.mase == by_mase[0][1].mase]
    winner = min(tied, key=lambda n: band[n].rmse)
    return winner, "RMSE tiebreaker"


def apply_low_incidence_fallback(
    selected: str,
    trailing_counts: list[float],
    regional_model: str,
) -> tuple[str, str]:
    """Reassign structurally low-incidence series to their regional model.

    `trailing_counts` are the observed weekly counts of the TRAILING_WEEKS
    immediately preceding the forecast origin (no look-ahead).
    """
    recent = trailing_counts[-TRAILING_WEEKS:]
    if sum(recent) < LOW_INCIDENCE_CASES:
        return regional_model, "low-incidence regional fallback"
    return selected, "own selected model"
