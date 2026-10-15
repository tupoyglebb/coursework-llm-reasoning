# Approved figures for the final manuscript

The three original PDFs are copied byte for byte from the source analysis folders.
Original paths and SHA256 hashes are in `SOURCE_MANIFEST.json`.

- `05_balanced_stimulus_levels.pdf`: Bolsen/Corbacho, GLM/Kimi; equal-treatment-weight **levels**, not contrasts. Source: `analysis/new_experiments_behavioral_20261015/tables/balanced_marginals.csv`. N0 is neutral realism. Pointwise conditional block-10 reference intervals. Values agree with Results §6.4; for example Kimi/A Corbacho True=0.697959…, and Bolsen A−N0=+0.168 GLM / −0.665 Kimi.
- `04_corbacho_distributions.pdf`: all ten model×stimulus panels; each retains control, inefficiency placebo and corruption, with True/False/refusal and valid n. Source: `analysis/new_experiments_behavioral_20261015/BEHAVIORAL_CELL_SUMMARY.csv`. No uncertainty intervals in this descriptive figure. Values and outcome definitions agree with §6.4.
- `02_bolsen_stimulus_contrasts.pdf`: nine category-presence RD estimates relative to N0, adjusted for treatment/model/annotation version. Source: `analysis/reasoning_2new_p1_20261015/tables/stimulus_regressions.csv`; n=799, four of nine Bolsen treatments, 40 design-cell clusters. Pointwise 95% cell-cluster t intervals; categories selected by effect magnitude, full BH results in source CSV. Not a causal mechanism. Matches §6.7 (e.g. overcorrection A/C=50.7/51.3 pp).

## Page layout, with no re-estimation or redrawing

The wide original charts would have approximately 4–5 pt labels at portrait text width. For readability:

1. The two panels of `05` are selected into `05_bolsen_panel.pdf` and `05_corbacho_panel.pdf` and stacked at 92% text width in Figure 1. Original footer information is supplied in the LaTeX caption; axes and results are retained.
2. `04` is selected into `04_corbacho_N0_A.pdf`, `04_corbacho_C_Nminus.pdf`, and `04_corbacho_Nplus.pdf`. Figure 2 continues over three parts. Every original model/stimulus/treatment panel is retained. The original legend remains in the N+ part; the first caption also states the colors. The common outcome scale remains 0–1. Labels are approximately 9 pt and n approximately 6 pt at final size.
3. Figure 3 uses the unmodified reasoning PDF, rotated within the page. Category labels are approximately 6.6 pt and remain vector text; rotate the page to read normally.

`../prepare_figure_layout.py` reproduces the panel selection using pypdf and Poppler pdftocairo. These operations select existing vector graphics; they do not recalculate, redraw or alter values. Cairo exports only visible clipped content, avoiding duplicated hidden-panel text in the PDF character count. The original three PDFs remain available for verification. Prepared PDFs are committed, so ordinary LaTeX builds do not need to rerun this script.

The five prepared panel PDFs declare PDF 1.7. Tectonic's PDF 1.5 import emits version warnings; the actual vector panels were rendered and visually checked in the assembled PDF. These warnings are recorded in the final audit and are not missing-figure or layout errors.
