# Figure production plan — four-experiment manuscript (2026-10-15)

**Purpose.** Prepare figures from reproducible analytic tables without redrawing old plots from rounded prose numbers. Limit the main manuscript to roughly 4–5 figures and avoid confusing historical **no incentive**, fresh **none**, and **N0** (which carries a realism instruction). Print measured outcome units and per-model comparisons. The paper's 60,000-character ceiling includes captions, bibliography and appendices.

## Main-text candidate figures

| Priority | Figure / purpose | Source and restrictions | Status |
|---|---|---|---|
| 1 | **Four-experiment stimulus effects:** small-multiple dot-and-interval display of A/C/N-/N+ versus N0, per experiment and model | Source new Bolsen/Corbacho behavioral analytic tables and Hassner/Savary N-series tables. **A/C cannot be compared with N0 for old cases without matched reruns; do not plot missing combinations.** Use outcome points for Hassner (% endorsement), Savary (1–5), Bolsen (1–7), Corbacho (True and refusal separately). Prefer experiment-specific panels and their own axes. | Needs source-table assembly |
| 2 | **Corbacho outcome compositions:** proportions False/True/refusal by original treatment, stimulus and model; include the inefficiency placebo | Existing source graph `04_corbacho_distributions` or `09_corbacho_stimulus_interactions`; if regenerating, distinguish the three nominal outcomes, and show n | Source figures ready; not yet committed |
| 3 | **Savary N-series contrasted by model:** N-−N0 and N+−N0 with CIs; optional N0−none in a separate panel with routing caveat | 2026-10-15 old-experiment N behavioral archive. Avoid treating sequential none-vs-N0 as a randomized contrast | Needs generated figure |
| 4 | **Reasoning category prevalence across prompts:** adjusted category presence contrasts A/C/N-/N+ against N0, split by experiment/model as space permits | 1,398 preliminary valid annotations with version sensitivity; plot category-specific intervals and q-values only from actual analysis tables. Emphasize that overcorrection_concern is close to A/C instructions | Source first-batch results available; integration pending |
| 5 | **Reasoning–response associations and/or argumentative direction:** coefficient/interval plot for within-cell associations (e.g. Bolsen economic_payback), expanded with direction analysis only if informative | Distinguish within-cell association from causal mechanism; annotation presence binary 0/1 in new cases; do not present historical 0/1/2 signed importance as if the same variable | Association results ready; direction analysis running |

## Reserve figures

- Hassner N-series original treatment × stimulus interactions, with missingness and provider-routing notes.
- Bolsen human-versus-LLM contrasts (old file `02_bolsen_baseline`), potentially more useful than the broad all-four figure if that figure is too busy.
- Balanced five-stimulus levels (old file `05_balanced_stimulus_levels`).
- Corbacho model × stimulus interaction file `09_corbacho_stimulus_interactions`.
- Provider sensitivity / historical H–M coverage diagnostics: repository or presentation, not necessarily main paper.

## Production checks

1. Read authoritative source scripts/tables rather than manual transcriptions from prior chat summaries.
2. Check cell sizes, missingness, provider routing and version-specific sampling.
3. Use appropriate scale and outcome semantics: Corbacho refusal is **not** False.
4. Do not pool GLM and Kimi into a single response-shift estimate without explicitly presenting interaction.
5. Keep chart labels legible at manuscript width, use colorblind-safe distinctions, export vector PDF and preferably PNG previews.
6. Record a source map for each plotted estimate (file, column, filtering, weighting, and CI construction).
7. Ensure TeX compilation succeeds before claiming a figure is integrated.
8. Only insert figures actually committed into the manuscript's figure directory.

**Next action:** export/commit the source graphic binaries from the analysis archives or regenerate directly from the analysis repository. Figure placeholders are **not** yet inserted into the TeX document, to avoid missing-file compilation errors.
