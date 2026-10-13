# Substantive audit — assembled competition manuscript (13 October 2026)

**Scope:** audit of `text_tex/editorial/COMPETITION_MANUSCRIPT_ASSEMBLED.tex`, historical original manuscript, and local project handoff/coverage CSV. **Not** a reproduction of analysis scripts or a verification against the original journal articles. No additional N0/N-negative/N-positive results assumed.

## Overall judgment
A viable standalone paper is already present: two different vignette domains, four API models, prompt-condition comparisons, reasoning annotations, and exploratory hidden-state results. The strongest defensible claim is that prompt sensitivity is observable at multiple levels **without guaranteeing human-effect fidelity**. The Savary private/public reversal is a substantive finding, not an embarrassment to conceal.

## P0 — issues affecting scientific credibility

1. **Observed pooled shifts versus like-for-like stimulus effects.** The displayed stimulus means pool different model compositions; H–M exist on Kimi/GLM only. The table must show at least a balanced Kimi/GLM-only comparison for all relevant conditions, with model- and vignette-treatment-adjusted contrasts and uncertainty. Otherwise use only 'pooled descriptive association' and do not claim universal stimulus effects. Check whether the no-stimulus baseline is comparable in sampling protocol and run counts.

2. **Cluster-based inference needs a precise estimand.** The manuscript says repeated generations within a design cell 'are not independent' and refers to 96/72 clusters, but a cluster-robust sandwich does not magically supply population generalizability when the treatment is fixed at the cell level and the number of unique model/scenario configurations is small. Clarify the sampling target (stochastic completions conditional on these fixed prompts/models), clustering definition and cluster counts for each fitted subset, whether the clustering variable is constant with regressors, and report cell-aggregated or model-stratified sensitivity analysis. Do not imply human-sample inference.

3. **Reconstructed historical prompt text.** A/C and H–M wordings are not yet authenticated as verbatim from the executed generation configs. This limits strict reproducibility and the interpretation of prompt-package contrasts. Separate verified prompt hashes/strings from semantic reconstructions; explicitly mark unknowns.

4. **Human benchmark validity.** The paper compares different outcome formats and experimental stakes across human and LLM samples. Verify exact original treatment definitions, reported human directions, study/year identifiers, and the choice of Savary comparison. Do not numerically rescale binary human donation rates into a 5-point intention score as if equivalent. Retain directional comparisons with clear measurement caveats.

5. **Evidence audit.** Numerical estimates, p-values, model-fit statistics, annotation coefficients, and hidden-state probe metrics have been carried from the original manuscript, not independently recalculated. Create a result-to-code/data traceability table; prioritize headline numbers and report sample denominators after exclusions. Existing 15,680 raw / 12,434 retained / 11,040 T=.01 totals must be reconciled with 10,080 annotated / 8,638 merged sample.

## P1 — substantive strengthening without new experiments

6. **Explicit hypotheses and estimands.** Introduction currently states RQ, goal, objectives, but no compact falsifiable hypotheses or prioritized outcome definitions. Propose 3-4 testable expectations: original vignette effects, stimulus main effects, treatment-by-stimulus heterogeneity, text/representation associations. Mark attribute-valence claims exploratory rather than confirmatory because H–M are incomplete.

7. **Sharper conceptual contrast.** Define 'normatively approved direction' as an operational coding choice, not a measured human SDB scale. Explicitly distinguish steering, human-effect directional replication, and internal representational separability. State what could falsify each.

8. **Reasoning validation.** The 27-trace gold set is tiny and category-specific. Avoid treating kappa thresholds as certification; give coder protocol and sample-selection procedure. The current results rely heavily on categories without human validation. Replace causal process language with descriptive association. Clarify whether annotation features were coded with access to final answers and treatment labels (potential label leakage).

9. **Hidden-state scope.** Local Qwen3-8B is a distinct model and condition set. Avoid interpreting inter-layer Euclidean distances as computation amount or chronology of decision formation. Report last-token extraction position, matching definition, controls for prompt-length/text differences, and probe CV/permutation setup. The reported LOGO AUC of 0.034 is below chance and needs checking for label orientation, multiclass scoring and fold structure; it is not automatically proof that no abstract treatment information exists.

10. **Paper organization.** 'Selected Results Tables' should not be a separate results section: place the stimulus table inside behavioral Results, direction table inside reasoning Results, and kappa table in methods/validation or appendix. Restore only the most informative 1–2 actual figures, if available, and make every display support a claim. The manuscript currently includes no figure environments despite narrative references being removed.

11. **Contribution overreach.** Do not conclude that LLMs are 'trustworthy for contrasts' given Savary reversal. Do not conclude that the prompts 'reduce SDB'. Do not assert that positive framing is the only active attribute. Stronger contribution: experimentally demonstrate that the surrounding instruction layer changes conclusions and that cross-level evidence is non-equivalent.

12. **Bibliographic/source hygiene.** Verify that all cited BibLaTeX keys exist in `Bib additions.bib` and that primary experimental references have correct years and matching claims. Recheck original human study descriptions and cited LLM-methodology papers before submission.

## P2 — submission and completeness

13. The anonymous manuscript omits the old title page and external AI declaration; confirm competition requirements for declarations rather than simply omitting a mandatory form.
14. The assembled source is ~49,986 characters **before rendered bibliography**; 60,000-character eligibility cannot be certified until bibliography and appendix counting rules are applied.
15. Avoid provisional text in final paper such as 'exact template must be preserved' or 'reported historical totals not independently recomputed'. These are appropriate for an editorial draft but should be replaced by verified methods or transparently stated reproducibility limitations before submission.
16. MoralLens must stay excluded throughout; the external annotation pipeline remains in scope. Planned follow-up experiments should not be represented as completed.

## Recommended order of work
**First:** result-to-source audit of headline claims and balanced stimulus contrasts. **Second:** strengthen hypothesis/estimand logic and reorganize tables. **Third:** source/annotation/hidden-state method verification. **Fourth:** copy-edit, figures, compilation, bibliography and final character count.

## Status
This is a critical review, not a claim that the numerical findings have been reproduced. Do not change the main branch or silently alter data/estimates.
