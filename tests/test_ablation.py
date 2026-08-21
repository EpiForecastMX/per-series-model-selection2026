from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis.ablation import MODELS, dynamic_ablation, select, smape


class SelectionRuleTests(unittest.TestCase):
    def test_tie_band_uses_mase_then_rmse(self) -> None:
        evidence = pd.Series(
            {
                "prophet_smape": 10.0,
                "prophet_mase": 0.8,
                "prophet_rmse": 4.0,
                "deepar_smape": 10.4,
                "deepar_mase": 0.7,
                "deepar_rmse": 5.0,
            }
        )
        self.assertEqual(select(evidence, ("prophet", "deepar")), "deepar")

        evidence["prophet_mase"] = 0.7
        self.assertEqual(select(evidence, ("prophet", "deepar")), "prophet")


class DynamicAblationTests(unittest.TestCase):
    def test_dynamic_regime_calls_complete_rule_with_three_metrics(self) -> None:
        weeks = np.arange(3, 19)
        weekly = pd.DataFrame(
            {
                "series": "Aguascalientes",
                "stratum": "general",
                "week": weeks,
                "observed": np.full(len(weeks), 100.0),
                "prophet": np.r_[np.full(9, 99.0), np.full(7, 95.0)],
                "deepar": np.r_[np.full(9, 98.0), np.full(7, 90.0)],
                "ensemble": np.r_[np.full(9, 97.0), np.full(7, 85.0)],
                "stacking": np.r_[np.full(9, 96.0), np.full(7, 80.0)],
            }
        )
        frozen = pd.DataFrame(
            {"series": ["Aguascalientes"], "stratum": ["general"], "model": ["prophet"]}
        )

        seen: list[set[str]] = []

        def choose_last(evidence: pd.Series, pool: tuple[str, ...]) -> str:
            seen.append(set(evidence.index))
            return pool[-1]

        with patch("analysis.ablation.select", side_effect=choose_last) as mocked:
            result = dynamic_ablation(weekly, frozen)

        self.assertEqual(mocked.call_count, 9)
        expected_fields = {
            f"{model}_{metric}"
            for model in MODELS
            for metric in ("smape", "mase", "rmse")
        }
        self.assertTrue(all(fields == expected_fields for fields in seen))
        expected = smape(np.full(7, 100.0), np.full(7, 80.0))
        self.assertAlmostEqual(
            result.loc[0, "prophet+deepar+ensemble+stacking"], expected
        )


if __name__ == "__main__":
    unittest.main()
