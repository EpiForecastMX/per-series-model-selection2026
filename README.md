# Per-Series Model Selection for Weekly Depression Incidence Forecasting

Code and supplementary material for

> Rebull-Saucedo, J., Pérez-Nava, J., Sánchez-Salazar, L., Barceló-Alonso, G.,
> Pérez-Hernández, R. *Forecasting Weekly Depression Incidence in Mexico:
> A Multi-Model Framework with Auditable Per-Series Model Selection.*
> 25th Mexican International Conference on Artificial Intelligence (MICAI 2026),
> Chihuahua, Mexico. Springer, Lecture Notes in Artificial Intelligence.

This repository provides a self-contained, reference implementation of the
methodological core of the paper: four forecasters with complementary
inductive assumptions, a deterministic and auditable **per-series model
selection rule**, the rolling-origin evaluation protocol, and the statistical
tests and hierarchical-reconciliation sanity check reported in the paper.

This repository was first released anonymously for double-blind review; the
paper has since been accepted and the author information above is now public.

## What is here

| File | Paper section | Purpose |
|---|---|---|
| `src/metrics.py` | Evaluation Protocol | sMAPE, MASE, RMSE, MAE and mean Poisson deviance |
| `src/selection.py` | Model Selection Strategy | deterministic 3-tier rule (sMAPE → MASE → RMSE, 5% relative margin) + low-incidence regional fallback |
| `src/cross_validation.py` | Evaluation Protocol | rolling-origin CV (5 folds, horizon 52, stride 13, expanding window) |
| `analysis/statistical_tests.py` | Results / Out-of-sample | one-sided Wilcoxon signed-rank (CV) and Diebold-Mariano with HLN correction (2026) |
| `analysis/reconciliation.py` | Discussion | minimum-trace (MinT, diagonal WLS) hierarchical reconciliation sanity check |
| `analysis/baselines.py` | Table (baselines) | tuned Auto-ARIMA + Fourier and a PatchTST transformer under the identical CV |
| `analysis/ablation.py` | §5.1, Table 3 | **ablation of the candidate pool**: the selection rule re-run on every subset, static and dynamic regimes, paired Wilcoxon against the full pool |
| `data/per_series_out_of_sample_2026.csv` | §4.2, Appendix A | **per-entity out-of-sample table**: sMAPE and MASE per model for all 111 series-stratum combinations |
| `data/cv_metrics_per_series.csv` | §5.1, Table 3 | frozen cross-validation evidence used by the static selection rule |
| `data/weekly_2026_per_series.csv` | §5.1, Table 3 | bulletin-aligned W03–W18 observations and locked forecasts used by the dynamic experiment |
| `data/frozen_assignment.csv` | §5.1, Table 3 | 99 directly observed assignments fixed before any 2026 observation existed |
| `data/ablation_results.csv` | §5.1, Table 3 | the published ablation numbers, as reported |

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
surveillance system (SINAVE) bulletin. The three compact, aggregate inputs
needed to reproduce Table 3 are released under `data/`; they contain no
individual-level records. The remaining analysis scripts expect pre-assembled
panels documented in their respective docstrings.

### Week alignment

The preprocessing step shifts the bulletin week down by one, so a forecast
stored under model week *N* carries the measurement the bulletin published as
week *N+1*. **Scoring a forecast against the bulletin therefore requires mapping
it back to bulletin numbering** (`bulletin_week = model_week + 1`). Every figure
in the paper and in the table below is in bulletin numbering. Skipping this step
compares observation and forecast one week apart, which silently inflates the
error.

### Ablation of the candidate pool

`analysis/ablation.py` reproduces Table 3: the selection rule re-run restricted to each
subset of the four candidates, in two regimes — *static*, selecting once from
cross-validated evidence as deployed, and *dynamic*, reselecting on the early 2026 weeks
and scoring on the later weeks it never saw. Policies are compared **pairwise over the
same series**; comparing each policy's own reassigned subset would compare different sets
of series, which is the easiest way to get this experiment wrong.

`data/ablation_results.csv` holds the published numbers. The headline finding is negative
and is reported as such in the paper: no leave-one-out variant separates from the full
pool, and a single-model policy matches it. What does separate from the frozen baseline
is revision.

The command below runs directly from a fresh clone and rewrites
`data/ablation_results.csv` deterministically from the three released inputs.

### Supplementary table

`data/per_series_out_of_sample_2026.csv` is the per-entity out-of-sample table
referenced in §4.2 and Appendix A of the paper. One row per series-stratum
combination (96 federal-entity, 12 socio-urban-region and 3 national series),
with sMAPE and MASE for each of the four candidate models over **W03-W18** —
the sixteen 2026 bulletin weeks for which every series has both an observation
and a locked forecast. MASE is scaled by 52-week seasonal persistence.

The headline national-aggregate evaluation in the paper runs over W02-W18, one
week wider, because the national series has a locked forecast for W02 and the
per-entity series do not. The two windows are reported separately and should not
be pooled.

## Reproducing the reported statistics

```bash
pip install -r requirements.txt
python analysis/ablation.py               # candidate-pool ablation (Table 3)
# The remaining scripts require the assembled panels described in their docstrings:
python analysis/statistical_tests.py     # Wilcoxon (CV) + Diebold-Mariano (2026)
python analysis/reconciliation.py         # MinT incoherence pre/post
python analysis/baselines.py --help       # Auto-ARIMA / PatchTST baselines
```

## License

Released under the MIT License (see `LICENSE`).
