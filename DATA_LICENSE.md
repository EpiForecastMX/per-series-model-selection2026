# Data provenance and reuse terms

The CSV files in `data/` contain aggregate weekly counts, forecasts, metrics,
and model assignments used to reproduce the paper's released analyses. They do
not contain individual-level or identifiable health records.

The observed counts are derived from the **SINAVE Boletín Epidemiológico**,
published by the Dirección General de Epidemiología (DGE), Secretaría de Salud,
Mexico:

- Source: https://www.gob.mx/salud/acciones-y-programas/direccion-general-de-epidemiologia-boletin-epidemiologico
- Official DGE Terms of Free Use for Open Data:
  https://www.gob.mx/salud/documentos/terminos-de-libre-uso-de-datos-abiertos-de-la-direccion-general-de-epidemiologia

The MIT License in `LICENSE` applies to the repository's source code and
documentation, **not** to these data files. Use, reuse, and redistribution of
the DGE-derived observations remain governed by the official terms above. Cite
the Dirección General de Epidemiología, Secretaría de Salud, as the source when
using the observed data.

Forecasts, derived error metrics, and model assignments are analytical outputs
released to support verification of the accompanying paper. They carry no
claim of endorsement by the DGE, Secretaría de Salud, SINAVE, or IMSS.
