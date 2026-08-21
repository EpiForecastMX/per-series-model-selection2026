from __future__ import annotations

import csv
import pathlib
import subprocess
import sys
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


class TestPublishedTable(unittest.TestCase):
    """Pin the released numbers so a refactor cannot move them silently."""

    CSV = pathlib.Path(__file__).resolve().parents[1] / "data" / "ablation_results.csv"

    # Table 3 of the paper, exactly as printed. sMAPE at one decimal, p at two.
    PRINTED = {
        ("static", "prophet+deepar+ensemble+stacking"): ("27.4", None),
        ("static", "deepar+ensemble+stacking"): ("27.4", "0.84"),
        ("static", "prophet+ensemble+stacking"): ("28.0", "0.49"),
        ("static", "prophet+deepar+stacking"): ("27.4", "0.69"),
        ("static", "prophet+deepar+ensemble"): ("27.4", "1.00"),
        ("static", "prophet"): ("26.7", "0.09"),
        ("dynamic", "prophet+deepar+ensemble+stacking"): ("28.5", None),
        ("dynamic", "deepar+ensemble+stacking"): ("29.4", "0.02"),
        ("dynamic", "prophet+ensemble+stacking"): ("26.4", "0.07"),
        ("dynamic", "prophet+deepar+stacking"): ("28.5", "0.23"),
        ("dynamic", "prophet+deepar+ensemble"): ("28.8", "0.87"),
        ("dynamic", "prophet"): ("27.6", "0.73"),
        ("dynamic", "frozen_baseline"): ("32.7", None),
    }

    def _rows(self):
        with self.CSV.open() as fh:
            return list(csv.DictReader(fh))

    def test_the_release_has_the_nineteen_expected_rows(self):
        rows = self._rows()
        self.assertEqual(len(rows), 19)
        self.assertEqual(sum(r["regime"] == "static" for r in rows), 9)
        self.assertEqual(sum(r["regime"] == "dynamic" for r in rows), 10)
        for r in rows:
            self.assertEqual(int(r["n"]), 99)

    def test_every_printed_cell_derives_from_the_csv_without_double_rounding(self):
        """Formatting the stored value must give the figure the paper prints.

        The CSV used to store three decimals, so a p-value of 0.694911 was saved as
        0.695 and a reader rounding the file could not recover the printed 0.69.
        """
        rows = {(r["regime"], r["policy"]): r for r in self._rows()}
        # OJO: no llamar `smape` a la variable de bucle; `smape` es la funcion
        # importada arriba y quedaria sombreada dentro del test.
        for key, (esperado_smape, p) in self.PRINTED.items():
            r = rows[key]
            self.assertEqual(
                f"{float(r['median_smape']):.1f}", esperado_smape, f"sMAPE moved: {key}"
            )
            if p is None:
                self.assertEqual(
                    r["p_vs_full_pool"], "", f"baseline row has a p: {key}"
                )
            else:
                self.assertEqual(
                    f"{float(r['p_vs_full_pool']):.2f}", p, f"p moved: {key}"
                )


class TestBaselinesCLI(unittest.TestCase):
    """The PatchTST note must work in a fresh clone, where no panel is shipped."""

    def test_patchtst_note_runs_without_the_undistributed_panel(self):
        """It used to read the panel first and die with FileNotFoundError.

        `data/weekly_series.csv` is not redistributed, so the mode the README
        advertises as "prints the configuration and exits" failed instead.
        """
        script = (
            pathlib.Path(__file__).resolve().parents[1] / "analysis" / "baselines.py"
        )
        out = subprocess.run(
            [
                sys.executable,
                str(script),
                "--mode",
                "patchtst",
                "--data",
                "does/not/exist.csv",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("PatchTST", out.stdout)
        self.assertNotIn("FileNotFoundError", out.stderr)
