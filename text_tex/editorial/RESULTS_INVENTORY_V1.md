# Results inventory v1 — 14 October 2026

Scope: historical source `text_tex/stiumulus_in_llm_survey.tex`, assembled editorial manuscript, existing substantive audit and local research handoff. This is a **claim inventory, not numerical replication**. No new experiments assumed. MoralLens excluded.

| ID | Analysis/result in original | Current assembled version | Verification needed | Proposed disposition |
|---|---|---|---|---|
| B1 | Per-model Hassner/Savary descriptives; treatment×model heterogeneity | Short prose; original descriptive table absent | Cell counts, sampling imbalance, treatment×model specification | Keep concise table/figure |
| B2 | Hassner four-condition directional replication | Prose | Raw baseline cells; original human treatment definitions | Keep |
| B3 | Savary private/public reversal and social-benefit increment | Prose | Original human measures and effects; treatment definitions | Keep; benchmark directional only |
| B4 | Stimulus A/C/H–M pooled means | Table present | Balanced Kimi/GLM and model/treatment-adjusted contrasts; uncertainty | Central; replace pooled-only headline |
| B5 | H–M attribute regression; OR≈15 for positive packages | Short prose | Prompt package comparability, exact wording, incomplete factorial | Exploratory; not pure valence |
| B6 | Stimulus×treatment interactions | Very short prose | Original interaction table, cell-based estimates and uncertainty | Central; restore |
| R1 | Overcorrection concern 9% to 72–76%; treatment awareness | Short prose; figure absent | Label leakage, kappa, direct wording echoes | Secondary diagnostic |
| R2 | Domain-specific argument prevalence; Savary personal_cost_burden, Hassner emergency/common interests | Short prose | Full prevalence appendix, annotation reliability | Central |
| R3 | Deliberative style and multiple positions | Short prose | Unvalidated style codes | Optional exploratory |
| R4 | A vs C signed presence: goals/means, cruelty, majority, source | Four-row table and short prose | Matched model/treatment cells, category-specific direction reliability | Central |
| R5 | A vs C legality assessment and private interests | Omitted | Original regressions and kappa | Restore if verified |
| R6 | Savary recognition/reward/privacy and motivational ambiguity | Partial regression prose | Full category direction and text examples; theory-aligned manual review | Central, theory-facing |
| R7 | Stimulus-vs-temperature annotation separability AUC .96 vs .68 | Short prose, figure absent | CV split, matched feature sets, exclusion of stimulus-linked codes | Restore as separate method result |
| R8 | Expert validation n=27 category-specific kappas | Table present | Sampling, coder blinding, class prevalence | Keep, emphasize uncertainty |
| P1 | Savary regression ΔR²=.063; privacy/recognition/reward effects | Partial | Exact feature definitions, uncertainty, controls | Keep |
| P2 | Hassner non-tautological ΔR²=.052; argument direction associations | Partial | Direction leakage and model fit | Keep |
| P3 | Savary gradient boosting CV R² .558 vs .475; SHAP | SHAP omitted | Grouped CV, correlated predictors, SHAP stability | Restore conditional on validation |
| P4 | Savary social-benefit×baseline, privacy×baseline interactions | Omitted | Original interaction table; selection/multiple testing | Restore if verified |
| H1 | Qwen3-8B trajectory 3370 vs 3444 | Prose, figure absent | Pairing, representation extraction, prompt-length confounds | Exploratory |
| H2 | Hidden-state layer separation L20–21, CKA>.94 | Prose | Geometry controls, definition of statistic | Exploratory |
| H3 | Representation–behavior dissociation, local saturation | Prose | Local outcomes, stimulus coverage | Exploratory |
| H4 | Treatment probe LOGO AUC .034, Savary ridge R² .665 | Prose | Label orientation, CV/permutation, leakage | Verify before retaining |

Original appendix tables to inspect: tab:attr, tab:homogeneity, tab:prevalence-full, tab:M1-hassner, tab:M1-savary, tab:interactions, tab:hs-behavioral. Original visual assets referenced: fig_treatment_replication.png, fig_stimulus_effects.png, fig_overcorrection.png, fig_benchmark.png, fig_shap_savary.png, fig_hs_trajectory.png, fig_hs_separation.png, fig_hs_decodability.png. Presence in manuscript does not establish image files are present in repo.

## Access / next action
GitHub manuscript access works. GitHub code search returned no matches for several analysis terms; **this is not evidence that analysis files are absent**. Need a branch-aware repository tree or Codex inventory of scripts, raw/processed datasets, notebooks, figure assets and prompt configs, then map each claim to exact file/function/plot. Do not request credentials. A Codex-generated manifest from the user's local workspace is an acceptable alternative.

## Editorial gates
1. Match historical numerical claims to data/code; mark confirmed/reported-only/unresolved.
2. Resolve comparable stimulus estimands and model coverage.
3. Check reasoning category validity and CV leakage.
4. Select primary results and displays only after gates 1–3.
