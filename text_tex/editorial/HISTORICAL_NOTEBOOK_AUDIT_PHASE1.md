# Historical notebook code audit — phase 1 (14 October 2026)

**Scope:** static inspection of branch `editorial/research-text-2026`, no notebook execution, no numerical recalculation, no manuscript edits. Sources: `notebooks/03_constructing_master_table.ipynb` (5 cells), `notebooks/05_statistical_analysis.ipynb` (134 cells), `notebooks/06_main_analysis.ipynb` (48 cells). Notebook cells indexed 1-based including markdown cells. Large 05/06 files were retrieved using GitHub blob SHAs after contents fetch returned empty.

## Verified code paths

- **03 cells 3–4:** reads run metadata and one final responses table per run, adds source/run metadata, concatenates, assigns incentive labels and writes `master_table.parquet`. This assembles raw data but does not produce historical `MAIN_DF`. Note path uses `Path.cwd()/runs`.
- **05 cell 2:** reads `attitude/master_table.parquet`, defines `load_clean`: drops `no_incentive_zero_temp_2_expos_nonreasoning` run; canonicalizes `__kimi_rerun`, `__glm_only_patch`; rerun rows take priority on `(canonical_run_dir, model, job_id)`; `_p2` kept as extra runs; `parse_status` and answer parser determine validity; `CLEAN_DF`, `ANALYSIS_DF` created. `is_t1` from suffix.
- **05 cell 11:** `MAIN_DF = ANALYSIS_DF[~ANALYSIS_DF["is_t1"]]`. Historical `MAIN_DF` and revised master filtering are not the same estimand.
- **05 cell 43:** `did_test` computes four-cell difference-in-differences of average outcomes, SE from sum of sample variances divided by sample sizes, normal z p-value, Holm adjusted within experiment. It pools models; this requires model-matched and cell-level sensitivity. Note notebook comments that DiD >0 necessarily strengthens treatment, which is incorrect when reference treatment effect is negative.
- **05 cells 39,41:** treatment-within-stimulus and baseline-vs-stimulus tests with correction; **05 cells 45–60:** effect sizes, Newcombe/Wilson, Savary alternative dichotomization; **05 cells 64–72:** interaction GLMs, ordered logit and binary robustness; **05 cells 77–85:** by-model contrasts, refusal sensitivity and stability. Precise numerical claims NOT checked against outputs.
- **06 cells 4,6,8,16:** reasoning annotation feature preparation and comparisons by run_source. **06 cells 17–20:** SHAP/LIME and predictor models on specific sets of `no_p2`, `no_t1`, `incentive_C`, `incentive_X`; these cannot automatically be equated to the Savary SHAP in the source article or the A-vs-C analysis. `06` is one potential source; need inspect `07_markup_analysis.ipynb` for published results.

## Required verification order
1. Verify whether `attitude/master_table.parquet` equals root `master_table.parquet`, hash/version of each and publication-time data.
2. Execute offline count-only waterfall in **new non-overwriting script/output** mirroring `05` and check 15,680 → CLEAN_DF → ANALYSIS_DF → MAIN_DF and T=1.0. Report subset composition by runs, models, treatments, and status.
3. Recompute cell-aggregated/model-matched baseline/stimulus contrasts, then historical DiD with same masks, confidence intervals and uncertainty.
4. Inspect `07_markup_analysis.ipynb` and gold code to identify *published* reasoning direction, A/C, Savary predictive regression, SHAP and temperature comparisons. Evaluate leakage and reliable categories.
5. Reconcile figure/table source to notebook output before copy edits.

**Status:** Code paths inspected; no headline historical percentages, p-values or model fits independently reproduced. Do not delete source results or change manuscript numbers on this basis. MoralLens excluded.
