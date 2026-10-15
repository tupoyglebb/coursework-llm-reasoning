# Historical behavioral reconciliation: editorial source-of-truth memo

Updated 15 October 2026. Primary input: `new_experiments_behavioral_20261015/historical_reconciliation/HISTORICAL_RECONCILIATION.md`, with supporting `CELL_SUMMARY_RECONCILIATION.csv`, `HISTORICAL_TREATMENT_CONTRASTS.csv`, and `HISTORICAL_MAIN_DF_RECONSTRUCTED.parquet` in the original handoff archive.

## Sample provenance

- Notebook 05 reconstruction: 15,680 raw -> 13,440 clean -> 12,434 analysis-valid -> 11,040 main low-temperature runs (Hassner 6,454; Savary 4,586).
- Earlier aggregate CSV: 12,838 valid rows, not the same sample.
- 168 cells, 50 different n values and 16 different point estimates between versions.
- The unavailable original `attitude/master_table.parquet` cannot be asserted byte-identical to the accessible root-level copy.
- 753 main observations come from a run with `zero_temp` in its name but request metadata indicating T=0.01; provenance remains unsettled. Removing this run preserves the main GLM/Kimi directional contrasts.

## Model-specific no-stimulus treatment contrasts (Notebook 05 reconstruction)

| Study | Contrast | GLM | Kimi |
|---|---|---:|---:|
| Hassner | modest intelligence minus control, P(Yes) | -0.7846 | -0.4419 |
| Hassner | protracted duration minus control, P(Yes) | -0.4385 | -0.5385 |
| Hassner | both modifications minus control, P(Yes) | -1.0000 | -0.7000 |
| Savary | public minus private, mean 1–5 | +0.2376 | +0.4000 |
| Savary | added social benefit minus public, mean 1–5 | +0.4348 | +0.4923 |

Keep the historical *four-model pooled* means distinct from the *matched two-model* table above. Existing reported regression SE and p-values were not recalculated by the reconciliation.

## New none run is separate

The subsequent Hassner/Savary `none` run analyzed in `REPORT.md` has 700 saved runs and 698 strictly valid answers (GLM and Kimi only). It must not be silently pooled into historical Notebook 05. The report contains no corresponding N0/N-/N+ results; those later runs are separate and pending.

## Pending figure and assembly checks

The handoff archive contains the original 15 behavioral figures in PNG/PDF/SVG. Prioritize plots 02 (Bolsen benchmark), 04 (Corbacho nominal outcomes), 05 (stimulus levels) and 09 (Corbacho interactions) for main text, and consider the remaining results for supplementary materials. Binary figures were not committed from this environment; do not claim them integrated until they exist at actual paths and compile.

After copying originals, verify bibliography with Biber, labels/references, figure sizing, front matter, and complete LaTeX compilation. Preserve the original charts instead of visually approximating numerical results.

## Forthcoming additions

- Hassner/Savary N0/N-/N+ generations and contrasts, currently collecting.
- Bolsen/Corbacho reasoning annotation on a restricted set of treatment groups, currently running. Its planned sample is not the full behavioral corpus; report actual cell coverage.
