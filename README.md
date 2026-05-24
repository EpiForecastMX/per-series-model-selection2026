# Per-Series Model Selection for Weekly Depression Incidence Forecasting

Anonymous code release supporting the double-blind submission
*"Forecasting Weekly Depression Incidence in Mexico: A Multi-Model Framework
with Auditable Per-Series Model Selection."*

This repository provides a self-contained, reference implementation of the
methodological core of the paper: four forecasters with complementary
inductive assumptions, a deterministic and auditable **per-series model
selection rule**, the rolling-origin evaluation protocol, and the statistical
tests and hierarchical-reconciliation sanity check reported in the paper.

> Author names, affiliations and the institutional repository are withheld for
> double-blind review. The complete production pipeline is released under the
> authors' institutional account upon acceptance.

## What is here

| File | Paper section | Purpose |
|---|---|---|
| `src/metrics.py` | Evaluation Protocol | sMAPE, MASE, RMSE, MAE and mean Poisson deviance |
| `src/selection.py` | Model Selection Strategy | deterministic 3-tier rule (sMAPE → MASE → RMSE, 5% relative margin) + low-incidence regional fallback |
| `src/cross_validation.py` | Evaluation Protocol | rolling-origin CV (5 folds, horizon 52, stride 13, expanding window) |
| `analysis/statistical_tests.py` | Results / Out-of-sample | one-sided Wilcoxon signed-rank (CV) and Diebold-Mariano with HLN correction (2026) |
| `analysis/reconciliation.py` | Discussion | minimum-trace (MinT, diagonal WLS) hierarchical reconciliation sanity check |
| `analysis/baselines.py` | Table (baselines) | tuned Auto-ARIMA + Fourier and a PatchTST transformer under the identical CV |

## Models

The framework integrates four forecasters behind one interface and selects one
per series:

- a structural additive model (Prophet) on the log-transformed rate;
- a global probabilistic recurrent network (DeepAR, GluonTS/PyTorch);
- a parallel hybrid (Prophet + gradient-boosted trees);
- a stacked hybrid (Prophet + ETS + gradient-boosted trees, non-negative
  ElasticNet meta-learner).

Each forecaster wraps a widely used open-source library; this release contains
the selection, evaluation and analysis code, which constitutes the
methodological contribution. The four model wrappers follow the cited libraries
directly.

## Data

Weekly case counts of depressive episode (CIE-10 F32) for the 32 Mexican
federal entities are published as **open data** by the national epidemiological
surveillance system (SINAVE) bulletin. The code expects pre-assembled weekly
series under `data/` (not redistributed here; see the bulletin's open-data
terms). Expected schemas are documented in each script.

## Reproducing the reported statistics

```bash
pip install -r requirements.txt
# place the assembled panels under data/ (see script docstrings)
python analysis/statistical_tests.py     # Wilcoxon (CV) + Diebold-Mariano (2026)
python analysis/reconciliation.py         # MinT incoherence pre/post
python analysis/baselines.py --help       # Auto-ARIMA / PatchTST baselines
```

## License

Released under the MIT License (see `LICENSE`).
