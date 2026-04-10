---
phase: quick-006
plan: 006
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/fitting/fit_mle.py
  - scripts/14_compare_models.py
  - scripts/16_regress_parameters_on_scales.py
  - scripts/17_analyze_winner_heterogeneity.py
  - .planning/PROJECT.md
  - .planning/STATE.md
  - .planning/quick/006-post-refit-verification-recovery-manuscript/006-SUMMARY.md
  - manuscript/paper.tex
  - output/model_comparison/comparison_results_bic.csv
  - output/model_comparison/winner_heterogeneity.csv
  - output/regressions/*/significance_corrected.csv
autonomous: true

must_haves:
  truths:
    - "All 7 models verified for NaN propagation on both CPU and GPU fit paths"
    - "AIC and BIC rankings both reported; winning model justified against simpler alternatives"
    - "Participant-level winner heterogeneity explained by parameter differences (when does M6b beat M5?)"
    - "All trauma-parameter regressions show uncorrected, FDR, and Bonferroni p-values side by side"
    - "M6b identifiability limitations documented (kappa recoverable, alpha/phi/rho/K/eps not)"
    - "PROJECT.md contains no results/interpretation language - purely infrastructural"
    - "Manuscript Discussion situates M1-M6b lineage against Collins & Frank, Senta, Bishara, Bornstein"
    - "Manuscript Discussion situates trauma-eps finding against Lissek, Myers, Ross, Admon"
    - "Manuscript has participant-level heterogeneity table/figure and stratified-by-trauma comparison"
    - "M2 WMRL 33% non-convergence diagnosed with root cause identified"
  artifacts:
    - path: "scripts/17_analyze_winner_heterogeneity.py"
      provides: "New script: compute participant-level AIC winners and compare param distributions across winner groups"
    - path: "output/model_comparison/comparison_results_bic.csv"
      provides: "BIC-based aggregate model ranking complementing AIC"
    - path: "output/model_comparison/winner_heterogeneity.csv"
      provides: "Per-participant winning model + parameter values by winner group"
    - path: "output/regressions/wmrl_m6b/significance_corrected.csv"
      provides: "M6b parameter-trauma regressions with uncorrected, FDR-BH, Bonferroni p-values"
    - path: ".planning/PROJECT.md"
      provides: "Scrubbed project description with no result contamination"
    - path: "manuscript/paper.tex"
      provides: "Updated manuscript with M6b winning model, lineage, trauma literature, heterogeneity, stratified comparison"
  key_links:
    - from: "scripts/16_regress_parameters_on_scales.py"
      to: "statsmodels.stats.multitest.multipletests"
      via: "fdr_bh and bonferroni correction applied per-parameter across 6 scales"
      pattern: "multipletests.*method="
    - from: "scripts/14_compare_models.py"
      to: "BIC column in comparison_results.csv"
      via: "BIC = -2*logL + k*ln(N_trials_per_ppt * n_ppt) or per-participant BIC summed"
      pattern: "bic"
    - from: "manuscript/paper.tex"
      to: "output/model_comparison/ and output/regressions/"
      via: "All quoted numbers match committed output files"
      pattern: "\\\\input\\{|\\\\includegraphics"
---

<objective>
Complete post-refit verification, identifiability assessment, significance correction, documentation cleanup, and manuscript integration for the N=154 re-fit pipeline. The winning model has changed from M5 to M6b (dAIC=572.89), but M6b has severe identifiability problems for base RLWM parameters (kappa recovers perfectly, but alpha/phi/rho/K/epsilon all fail r<0.80). This task addresses all 12 user items, adds participant-level heterogeneity and stratified-by-trauma analyses to the manuscript, and situates findings against existing RLWM and trauma literature.

Purpose: Ship a scientifically defensible manuscript that correctly reports M6b as the winner while disclosing its limitations, separates infrastructural documentation from time-varying results, and verifies all pipeline scripts handle edge cases correctly.

Output: 6 atomic commits covering verification, comparison extension, heterogeneity analysis, significance correction, documentation cleanup, and manuscript revision. Manuscript paper.tex updated with all required sections. PROJECT.md scrubbed.
</objective>

<execution_context>
@C:\Users\aman0087\.claude\get-shit-done\workflows\execute-plan.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/STATE.md
@CLAUDE.md

# Relevant existing outputs
@output/model_comparison/comparison_results.csv
@output/model_comparison/stratified_results.csv
@output/recovery/wmrl_m6b/recovery_metrics.csv
@output/recovery/wmrl_m5/recovery_metrics.csv
@output/recovery/wmrl_m6a/recovery_metrics.csv
@output/regressions/wmrl_m6b/

# Relevant scripts to modify
@scripts/14_compare_models.py
@scripts/16_regress_parameters_on_scales.py
@scripts/fitting/fit_mle.py
@manuscript/paper.tex
</context>

<preliminary_findings>

## Already Established (from preliminary investigation, do NOT re-discover)

### Pipeline State
- All 7 models fit on N=154 participants (Apr 8 17:54)
- Downstream scripts 14/15/16 re-run Apr 8 18:17-18:20
- Recovery outputs NOW pulled for M5, M6a, M6b (NOT M4)
- PPC outputs exist for wmrl_m4, wmrl_m5, wmrl_m6a, wmrl_m6b under output/ppc/
- Uncommitted change: manuscript/paper.tex (preserve)

### Winning Model: M5 → M6b
Aggregate AIC:
| Rank | Model | AIC | dAIC |
|---|---|---|---|
| 1 | M6b | 143,324.93 | — |
| 2 | M5 | 143,897.82 | +572.89 |
| 3 | M6a | 144,771.59 | +1,446.66 |
| 4 | M3 | 144,865.92 | +1,540.99 |
| 5 | M2 (WMRL) | 147,328.17 | +4,003.24 (51/154 non-conv — HIGH) |
| 6 | M1 (Q-learning) | 152,143.11 | +8,818.18 |
| — | M4 (LBA) | separate track | N/A |

Participant-level AIC wins: M6b=55 (36%), M5=41 (27%), M6a=38 (25%), M3=15, M2=3, M1=2

### M6b Identifiability Crisis
`output/recovery/wmrl_m6b/recovery_metrics.csv`:
- kappa_total: r=0.9971 PASS
- kappa_share: r=0.9311 PASS
- alpha_pos: r=0.5982 FAIL
- alpha_neg: r=0.5161 FAIL
- phi: r=0.4422 FAIL
- rho: r=0.6289 FAIL
- capacity: r=0.2135 FAIL (severe)
- epsilon: r=0.7721 FAIL (close)

Same pattern in M5/M6a. **Capacity is worst in every model**. Implications:
1. TRUST: kappa parameter inferences
2. DO NOT TRUST: K/alpha/phi/rho individual differences
3. STILL VALID: Aggregate AIC/BIC (likelihood-of-data, not params)
4. CAVEAT: Any non-kappa parameter-trauma regression

### NaN argmin Bug Fix Status
`scripts/fitting/fit_mle.py:1447-1448` (GPU path `fit_all_gpu`) has fix:
```python
safe_nlls = jnp.where(jnp.isnan(all_nlls), jnp.inf, all_nlls)
best_idx = jnp.argmin(safe_nlls, axis=1)
```
**MUST VERIFY**: CPU path (`fit_all_participants` ~line 2585+), M4 LBA objective, wmrl (M2) path.

### Trauma-Parameter Results (M6b, `output/regressions/wmrl_m6b/`)
Uncorrected p<0.05:
- epsilon UP with IES-R Hyperarousal: beta=0.003, p=0.020
- epsilon UP with IES-R Intrusion: beta=0.002, p=0.038
- epsilon UP with IES-R Total: beta=0.001, p=0.027
Marginal: alpha_pos DOWN with LEC-5 Total: beta=-0.004, p=0.053
Null: K, alpha_neg, phi, rho, kappa_total, kappa_share

With ~6 scales × 8 params = ~48 tests, Bonferroni threshold is p<0.001. **epsilon associations will likely NOT survive Bonferroni**. FDR-BH may retain them.

### PROJECT.md Contamination
`.planning/PROJECT.md` lines 24, 39 contain: "M5 is the current winning model (dAIC=435.6 over M3)" — STALE and violates infrastructural-only rule.

### User Requests Confirmed
- (c) Add participant-level heterogeneity table/figure
- (d) Add stratified-by-trauma-group model comparison

### Stratified Comparison (already computed)
`output/model_comparison/stratified_results.csv`:
- "No Ongoing Impact" (N=26): M6b dominant (50% wins)
- "Ongoing Impact" (N=19): M6b vs M5 competitive, Fisher p=0.615 M6b vs M6a

</preliminary_findings>

<tasks>

<task type="auto">
  <name>Task 1: Verify NaN/convergence bug fix on all fit code paths + diagnose M2 non-convergence</name>
  <files>
    scripts/fitting/fit_mle.py
    output/mle/wmrl_all_start_nlls.csv
  </files>
  <action>
    Read scripts/fitting/fit_mle.py in full. Identify and audit ALL code paths that compute best_idx from NLL arrays:
    1. `fit_all_gpu` (~line 1447) — already has safe_nlls fix (CONFIRMED)
    2. `fit_all_participants` (CPU path, ~line 2585+) — VERIFY presence/absence of NaN guard
    3. M4 LBA objective path — VERIFY
    4. Any per-participant single-fit fallback paths — VERIFY

    For EACH path missing the guard, add:
    ```python
    safe_nlls = jnp.where(jnp.isnan(all_nlls), jnp.inf, all_nlls)
    best_idx = jnp.argmin(safe_nlls, axis=...)
    ```
    or equivalent numpy version.

    Then diagnose M2 WMRL 33% non-convergence (51/154):
    - Read `output/mle/wmrl_all_start_nlls.csv`
    - For non-converged participants: are ALL starts NaN? Or do some starts finish but hit bounds?
    - Count patterns: (a) all-NaN participants (NaN propagation), (b) all-hit-bound participants (optimizer failure), (c) mixed.
    - Compare against `output/mle/wmrl_m5_all_start_nlls.csv` (M5 has lower non-conv rate) to see if the pattern is M2-specific or general.

    Write findings as inline comments in fit_mle.py next to any fixes. Do NOT re-run fits as part of this task — this is audit+fix only. Note in commit message whether re-fitting is needed for M2 as a follow-up.

    ATOMIC: Even if re-fitting is needed, that's a separate task. Just verify code correctness and document M2 root cause in the SUMMARY.
  </action>
  <verify>
    - `grep -n "argmin" scripts/fitting/fit_mle.py` — every argmin on NLLs should be preceded by an isnan guard OR the NLLs should be pre-filtered
    - Python syntax still valid: `python -m py_compile scripts/fitting/fit_mle.py`
    - M2 non-convergence diagnosis written in task output / SUMMARY with concrete numbers (e.g., "48/51 non-converged ppts had all 10 starts return NaN; 3/51 had at least one finite but hit epsilon=0 bound")
  </verify>
  <done>
    Every NLL argmin/argmax path in fit_mle.py has explicit NaN handling. M2 root cause documented. Any code fixes committed. If M2 re-fit is needed, documented as deferred follow-up in SUMMARY.
  </done>
</task>

<task type="auto">
  <name>Task 2: Extend 14_compare_models.py with BIC computation and AIC/BIC side-by-side output</name>
  <files>
    scripts/14_compare_models.py
    output/model_comparison/comparison_results.csv
    output/model_comparison/comparison_results_bic.csv
  </files>
  <action>
    Extend `scripts/14_compare_models.py` to compute BIC alongside AIC for the choice-only models (M1, M2, M3, M5, M6a, M6b). Keep M4 on its separate track.

    BIC formula: `BIC = -2 * logL + k * ln(N)` where:
    - k = number of free parameters per participant
    - N = effective sample size. Two options — compute BOTH and report:
      (a) per-participant BIC: N = n_trials_for_that_participant (~420), summed across participants
      (b) aggregate BIC: N = total_trials (~64,680), applied to pooled k

    Scientific rationale to write in a comment block at the top of the new function:
    ```
    # AIC vs BIC for this pipeline:
    # - AIC penalty = 2k, BIC penalty = k*ln(N)
    # - With N_trials_per_ppt ~420, ln(N) ~6, so per-ppt BIC penalizes ~3x more than AIC
    # - AIC favors predictive accuracy; BIC favors parsimonious truth recovery
    # - For N=154 with k ranging 3 (M1) to 8 (M6b), BIC may flip winner toward simpler models
    # - Report BOTH; if they disagree, the dominant model is the one that wins both
    ```

    Implementation:
    1. Load per-participant fit CSVs (already have nll, n_params columns)
    2. Compute per-participant n_trials (from task_trials_long.csv group-by participant_id)
    3. Compute per-ppt BIC = 2*nll + k*ln(n_trials_ppt)
    4. Aggregate: sum per-ppt BIC across participants
    5. Write to `output/model_comparison/comparison_results_bic.csv` with columns: model, total_aic, total_bic, k, delta_aic, delta_bic, aic_rank, bic_rank
    6. Also extend the existing `comparison_results.csv` to include a BIC column (don't duplicate — extend in-place if possible, or write both for safety)
    7. Print a summary table to stdout showing whether AIC and BIC agree on the winner

    Do NOT modify participant-level AIC wins logic (that stays in Task 3). This task is aggregate-only.

    Scientific decision to document in SUMMARY: Which criterion is more appropriate here? Given identifiability problems on base params, there's an argument that BIC's parsimony penalty is more honest — it prefers models where the extra parameters buy real explanatory power, not overfit. But AIC is standard in RLWM literature (Collins, Senta). Recommendation: REPORT BOTH in manuscript Table 1, discuss disagreements if any.
  </action>
  <verify>
    - `python scripts/14_compare_models.py` runs without error
    - `output/model_comparison/comparison_results_bic.csv` exists and has all 6 choice-only models
    - stdout shows AIC rank and BIC rank side-by-side
    - BIC values are larger than AIC values (expected — BIC has bigger penalty)
    - M6b's BIC advantage over M5 is SMALLER than its AIC advantage (because M6b has more params)
  </verify>
  <done>
    Both AIC and BIC computed and written to output. Script prints a comparison summary. Scientific justification for which to emphasize is written in the SUMMARY for manuscript citation.
  </done>
</task>

<task type="auto">
  <name>Task 3: Create 17_analyze_winner_heterogeneity.py — explain when M6b vs M5 vs M6a wins</name>
  <files>
    scripts/17_analyze_winner_heterogeneity.py
    output/model_comparison/winner_heterogeneity.csv
    output/model_comparison/winner_heterogeneity_figure.png
  </files>
  <action>
    Create NEW script `scripts/17_analyze_winner_heterogeneity.py` following the numbered pipeline convention. Goal: answer user's Q3 — "when does M6b win vs M5 vs M6a — is it about kappa_share values? capacity? etc."

    Script does:
    1. Load per-participant AIC from all 6 choice-only model fit CSVs (`output/mle/{model}_individual_fits.csv`)
    2. Compute per-participant winning model: `winning_model = argmin(AIC) per participant_id`
    3. Load M6b individual fits (the winner) and extract all parameters per participant
    4. Group participants by their winning model → compare M6b parameter distributions:
       - Participants where M6b wins: what are their kappa_total, kappa_share, K, phi, rho values?
       - Participants where M5 wins: same
       - Participants where M6a wins: same
       - Participants where M3 wins: same
    5. Statistical comparison:
       - Run ANOVA or Kruskal-Wallis on each M6b parameter across winner groups
       - Report effect sizes (eta-squared or similar)
       - Report medians + IQR per winner group per parameter
    6. Write `output/model_comparison/winner_heterogeneity.csv` with columns: participant_id, winning_model, m6b_alpha_pos, m6b_alpha_neg, m6b_phi, m6b_rho, m6b_K, m6b_kappa_total, m6b_kappa_share, m6b_epsilon, m6b_aic, m6b_nll
    7. Create figure `output/model_comparison/winner_heterogeneity_figure.png`: boxplots of M6b parameters (kappa_total, kappa_share, K, phi) grouped by winning_model. Use seaborn/matplotlib. Save at 300 DPI for manuscript inclusion.

    Scientific interpretation to document in SUMMARY (will appear in manuscript):
    - Hypothesis: M6b wins when kappa_share is near 0.5 (participants use both choice- and stimulus-level perseveration roughly equally). M6a wins when kappa_share is near 1 (pure stimulus perseveration). M5 wins when kappa is small overall (perseveration minimal, RL forgetting dominates).
    - Report actual pattern observed in the data.

    IMPORTANT: Use three-layer naming convention — descriptive names at API, math internals can use kappa_s etc, domain English in script comments.
  </action>
  <verify>
    - `python scripts/17_analyze_winner_heterogeneity.py` runs without error
    - `output/model_comparison/winner_heterogeneity.csv` exists and has 154 rows
    - Figure file exists and is non-empty (>100KB, suggests real content)
    - Counts match preliminary: M6b=55, M5=41, M6a=38, M3=15, M2=3, M1=2
    - Statistical tests produce p-values and effect sizes
  </verify>
  <done>
    Script created, follows numbered pipeline pattern. Output CSV and figure generated. Interpretation of when/why each model wins documented in SUMMARY for manuscript Discussion section.
  </done>
</task>

<task type="auto">
  <name>Task 4: Add FDR and Bonferroni correction to 16_regress_parameters_on_scales.py</name>
  <files>
    scripts/16_regress_parameters_on_scales.py
    output/regressions/wmrl_m6b/significance_corrected.csv
    output/regressions/wmrl_m5/significance_corrected.csv
    output/regressions/wmrl_m6a/significance_corrected.csv
    output/regressions/wmrl_m3/significance_corrected.csv
    output/regressions/wmrl/significance_corrected.csv
    output/regressions/qlearning/significance_corrected.csv
    output/regressions/wmrl_m4/significance_corrected.csv
  </files>
  <action>
    Extend `scripts/16_regress_parameters_on_scales.py` to apply multiple-comparison corrections.

    Import at top:
    ```python
    from statsmodels.stats.multitest import multipletests
    ```

    After computing per-parameter regression results (uncorrected p-values), add a correction step:
    1. For each model, collect ALL uncorrected p-values across (parameter × scale) combinations into one vector
    2. Apply FDR Benjamini-Hochberg: `_, p_fdr, _, _ = multipletests(p_vec, method='fdr_bh', alpha=0.05)`
    3. Apply Bonferroni: `_, p_bonf, _, _ = multipletests(p_vec, method='bonferroni', alpha=0.05)`
    4. Map corrected p-values back to (parameter, scale) pairs
    5. Write `output/regressions/{model}/significance_corrected.csv` with columns: parameter, scale, beta, se, t_stat, p_uncorrected, p_fdr_bh, p_bonferroni, sig_uncorrected, sig_fdr, sig_bonferroni
    6. Also create a summary file `output/regressions/{model}/significance_summary.md` with a markdown table showing ONLY the associations that survive each correction level

    Correction family:
    - Within-model: ALL tests for that model (params × scales, ~48 tests for M6b: 8 params × 6 scales)
    - Document choice in script comment: "Family-wise correction is applied within-model, not across-model, because models are alternative explanations of the same data."

    Apply to ALL 7 models (qlearning, wmrl, wmrl_m3, wmrl_m4, wmrl_m5, wmrl_m6a, wmrl_m6b) so the `--model all` CLI flag produces corrected output for every fit.

    Expected result for M6b (from preliminary):
    - 3 uncorrected hits (epsilon × IES-R Hyper/Intrusion/Total) + 1 marginal
    - With ~48 tests, Bonferroni threshold ~0.001 → NONE survive
    - FDR-BH may retain the strongest (Hyperarousal p=0.020 would need p_fdr ≤ 0.05, depends on distribution)

    Do NOT delete the existing per-parameter scatter plots or uncorrected CSVs. This task is PURELY additive.

    Then diagnose in SUMMARY: "The epsilon-trauma associations do not survive Bonferroni correction but [may/may not] survive FDR-BH. Manuscript must report this honestly and frame the epsilon finding as [exploratory/marginal/robust] accordingly."
  </action>
  <verify>
    - `python scripts/16_regress_parameters_on_scales.py --model all` runs without error
    - `output/regressions/wmrl_m6b/significance_corrected.csv` exists
    - For each model: summary.md file exists and shows surviving associations at each correction level
    - Bonferroni column values are always >= FDR column values (math sanity)
    - epsilon × IES-R associations shown with all three p-values side-by-side
  </verify>
  <done>
    All 7 model regression outputs have correction CSVs. Survival pattern at uncorrected/FDR/Bonferroni documented in SUMMARY for manuscript reporting.
  </done>
</task>

<task type="auto">
  <name>Task 5: Clean PROJECT.md of results contamination + update STATE.md with winning model</name>
  <files>
    .planning/PROJECT.md
    .planning/STATE.md
  </files>
  <action>
    **PROJECT.md cleanup**: Read the file in full. Remove ALL results/interpretation language. Specifically:
    1. Line 24: Remove "M5 is the current winning model (dAIC=435.6 over M3)" — replace with an infrastructural statement like "Seven candidate RL/WM/LBA models are implemented and compared via AIC/BIC."
    2. Line 39: Same treatment
    3. Scan for any other numeric claims ("dAIC=...", "r=...", participant counts, p-values, "winning", "best", "currently")
    4. Scan for tense issues — PROJECT.md should describe WHAT the infrastructure does, not WHAT RESULTS emerged
    5. Preserve: infrastructure descriptions, pipeline stages, model names, file paths, conventions, how-to-run instructions

    Acceptance criterion for PROJECT.md: If you search for `winning`, `dAIC`, `p=`, `beta=`, `p<`, `r=0`, the file should have ZERO matches on any of these in the context of specific results. (Occurrences in the model-name column like "M5: WM-RL with RL Forgetting" are fine — that's infrastructure.)

    **STATE.md update**: Reflect the new project state post-refit:
    1. Winning model: M6b (not M5)
    2. Note identifiability caveats for base RLWM parameters
    3. Trauma-parameter findings: epsilon × IES-R uncorrected (correction status TBD from Task 4)
    4. Deferred: M4 LBA recovery (~48h)
    5. Keep as "current state" narrative — this IS where results go, PROJECT.md is where they don't

    Do NOT rewrite STATE.md from scratch — UPDATE the relevant sections (current phase, recent findings, blockers).
  </action>
  <verify>
    - `grep -n "winning\|dAIC\|beta=\|p=0\|p<0\|r=0\." .planning/PROJECT.md` returns no result-contamination matches
    - `.planning/PROJECT.md` still describes all 7 models, pipeline stages, and scripts
    - `.planning/STATE.md` mentions M6b as current winning model
    - `.planning/STATE.md` notes identifiability caveats
  </verify>
  <done>
    PROJECT.md is infrastructural-only. STATE.md reflects post-refit reality. Both files committed together.
  </done>
</task>

<task type="auto">
  <name>Task 6: Literature research — model lineage + trauma-RL citations via bioRxiv MCP and web sources</name>
  <files>
    .planning/quick/006-post-refit-verification-recovery-manuscript/LITERATURE.md
  </files>
  <action>
    Create `.planning/quick/006-post-refit-verification-recovery-manuscript/LITERATURE.md` as a citation gathering document (will be consumed by Task 7 for manuscript edits, then can be preserved or moved to SUMMARY).

    Use the **bioRxiv MCP server** (`search_preprints`, `search_published_preprints`) as primary tool for recent (2018-2026) papers. Use WebFetch for older canonical papers (Collins & Frank 2012).

    **Part A: Model lineage** (user request #7). Find and document with DOI/year:
    1. **Collins & Frank 2012** — "How much of reinforcement learning is working memory, not reinforcement learning?" (European Journal of Neuroscience). The original RLWM decomposition.
    2. **Collins, Brown, Gold, Waltz & Frank 2014** — "Working memory contributions to reinforcement learning impairments in schizophrenia" (J Neurosci). First clinical application of RLWM.
    3. **Collins & Frank 2018** — "Within- and across-trial dynamics of human EEG reveal cooperative interplay between reinforcement learning and working memory" (PNAS). Updated RLWM.
    4. **Senta et al. 2025** — The paper that provides our κ implementation (check bioRxiv preprint first, then published version). Document which figure/equation we're inheriting.
    5. **Bishara & Hawthorne** — find perseveration model variants in RL literature (look for "perseveration parameter reinforcement learning").
    6. **Bornstein & Norman** — LBA variants applied to choice RT (this is our M4 inspiration).
    7. **Brown & Heathcote 2008** — original LBA paper (Cognitive Psychology). Parent of M4.
    8. **Miletić et al. 2020** — RL-LBA joint modeling (if exists, check bioRxiv).

    For each: write 2-3 sentences explaining HOW M1-M6b/M4 specifically inherit from / extend that work. Example format:
    ```
    ### Collins & Frank (2012) — doi: 10.1111/j.1460-9568.2011.07980.x
    Original RLWM decomposition. Working memory component with fast-decaying buffer (capacity K, decay phi)
    complements slow-learning RL (alphas). Our M2 directly implements this two-system architecture.
    M3/M5/M6a/M6b extend by adding perseveration (M3), RL forgetting (M5), or dual perseveration (M6b).
    ```

    **Part B: Trauma attention/RL literature** (user request #8). Find:
    1. **Lissek et al.** — PTSD and fear extinction (multiple papers, find the key 2013-2014 reviews). Relevance: attentional noise in trauma.
    2. **Myers & Gluck** — PTSD and reversal learning (probably 2007-2013 era). Relevance: perseveration + RL flexibility in trauma.
    3. **Ross et al.** — trauma and working memory (check Ross DA or Ross MC — ambiguous without context, search both).
    4. **Admon et al.** — stress-induced cognitive control deficits (Boston-based, around 2013-2018).
    5. **Pizzagalli D. A.** — anhedonia and reward learning (relevant as trauma is comorbid with anhedonia).
    6. **Nestor, Frank, Badre etc.** — any recent RL × PTSD modeling (use bioRxiv search).

    Focus specifically on the epsilon (attentional noise / random responding) finding: what mechanisms in the existing literature would predict trauma → more noisy responding?

    Output structure for LITERATURE.md:
    ```markdown
    # Literature for Quick-006 manuscript updates

    ## Part A: Model Lineage (for manuscript Methods/Discussion)
    [Citations with DOI, 2-3 sentence relevance each]

    ## Part B: Trauma and Attention/RL (for manuscript Discussion)
    [Citations with DOI, 2-3 sentence relevance each]

    ## Epsilon-Trauma Framing Options
    [Draft 2-3 candidate sentences the manuscript could use to situate the finding]
    ```

    IMPORTANT: Do not make up DOIs. If bioRxiv search fails or is ambiguous, mark the entry as `[DOI TBD — ${search query}]` and move on. Task 7 can fill in gaps with additional searches.

    Use bioRxiv MCP first, then fall back to WebFetch for Google Scholar / PubMed. Do NOT use ClinicalTrials.gov for this task — it's clinical trials only, not literature.
  </action>
  <verify>
    - `.planning/quick/006-post-refit-verification-recovery-manuscript/LITERATURE.md` exists
    - At least 6 Part A citations with DOIs (or TBD markers)
    - At least 5 Part B citations with DOIs (or TBD markers)
    - Each entry has 2-3 sentence relevance explanation
    - Epsilon-trauma framing section has 2-3 candidate sentences
  </verify>
  <done>
    LITERATURE.md contains enough citations and framing for Task 7 to write the manuscript Discussion expansions without additional research.
  </done>
</task>

<task type="auto">
  <name>Task 7: Update manuscript/paper.tex — M6b winner, identifiability, heterogeneity, stratified, lineage, trauma literature</name>
  <files>
    manuscript/paper.tex
  </files>
  <action>
    Update manuscript/paper.tex to reflect everything from Tasks 1-6. Preserve the user's uncommitted edits (read the file carefully before editing).

    **Required edits:**

    1. **Abstract** — update winning model from M5 to M6b, update dAIC numbers, update sample size to N=154. Brief mention of identifiability caveats.

    2. **Results — Model Comparison section**:
       - Table 1: Extend to show both AIC and BIC columns. Rank by AIC but show BIC rank too.
       - Add a sentence: "BIC imposes a stronger parsimony penalty (k*ln(N) vs 2k) and [agrees/disagrees] with AIC on the winning model."
       - Update all specific dAIC numbers to N=154 values (from Task 2)

    3. **Results — NEW subsection: Participant-Level Heterogeneity** (user request c):
       - Insert the winner_heterogeneity_figure.png from Task 3
       - Add a table showing counts: M6b=55 (36%), M5=41 (27%), M6a=38 (25%), M3=15, M2=3, M1=2
       - Narrative: "While M6b wins at the aggregate level, only [X]% of participants are best fit by M6b individually. This heterogeneity reflects [interpretation from Task 3 — e.g., M5 wins when participants show minimal perseveration, M6a wins when perseveration is stimulus-locked]."

    4. **Results — NEW subsection: Stratified Model Comparison by Trauma Impact** (user request d):
       - Use data from `output/model_comparison/stratified_results.csv`
       - Table or small figure: win counts per model for "No Ongoing Impact" (N=26) and "Ongoing Impact" (N=19)
       - Fisher's exact tests: M6b vs M5 not significantly different (p=0.615 for M6a comparison)
       - Narrative: "The winning model is consistent across trauma impact groups, suggesting the M6b dual-perseveration architecture captures dynamics common to both trauma-impacted and non-impacted participants."

    5. **Results — Parameter Recovery subsection**:
       - ADD or expand: report that for M6b, kappa_total and kappa_share recover excellently (r=0.997, r=0.931) but base RLWM parameters (alpha_pos r=0.598, alpha_neg r=0.516, phi r=0.442, rho r=0.629, K r=0.213, epsilon r=0.772) do NOT meet the r≥0.80 criterion.
       - Explicit statement: "Consequently, we interpret kappa-level inferences with confidence but treat individual-differences conclusions about base RLWM parameters as exploratory."

    6. **Results — Trauma-Parameter Associations**:
       - Report UNCORRECTED, FDR, and BONFERRONI p-values side-by-side for the three epsilon × IES-R associations
       - Frame robustness: "These associations survive [uncorrected / FDR-BH only / both corrections / neither] multiple comparison correction."
       - Move any claims that don't survive FDR to an "Exploratory Findings" paragraph

    7. **Discussion — NEW paragraph: Model Lineage** (user request #7):
       - Use citations from LITERATURE.md Part A
       - 1-paragraph narrative: "M1 implements classical Q-learning (refs). M2 inherits the RLWM decomposition of Collins & Frank (2012, 2018), adding a capacity-limited WM buffer. M3 extends with a choice-perseveration kernel following Bishara & Hawthorne. M5 adds RL-specific forgetting. M6a and M6b extend the perseveration mechanism: M6a tracks stimulus-level action history, M6b combines choice- and stimulus-level via the kappa_share parameter inherited from Senta et al. (2025). M4 is a joint choice+RT model using the LBA accumulator framework (Brown & Heathcote, 2008) applied to RLWM following Bornstein & Norman."

    8. **Discussion — NEW paragraph: Epsilon-Trauma Finding Situated in Literature** (user request #8):
       - Use citations from LITERATURE.md Part B
       - Narrative: "The observed [uncorrected/FDR] association between IES-R Hyperarousal and the epsilon attention-noise parameter aligns with prior findings of reduced signal-to-noise in PTSD choice behavior (Lissek, Myers & Gluck). Unlike studies focusing on learning-rate impairments (refs), our finding localizes the effect to the response-selection stage — consistent with hypervigilance models where attentional resources are diverted from ongoing task demands. However, the lack of robustness to Bonferroni correction and the identifiability limitations on base RLWM parameters together counsel against strong causal inference."

    9. **Discussion — Limitations**:
       - ADD: parameter recovery failures for base RLWM parameters
       - ADD: M4 LBA recovery not yet performed (deferred — compute cost ~48h)
       - ADD: Sample size N=154 with ~20 scales × 8 params creates a multiple-comparisons burden

    Do NOT rewrite sections that are already correct. Surgical edits only. Preserve user's uncommitted changes — if you see edits that look intentional (e.g., re-worded paragraphs, re-formatted tables), leave them alone.

    If any LaTeX figure \includegraphics paths point to files that don't exist in output/, either (a) change the path to an existing file, or (b) add a TODO comment next to the figure.
  </action>
  <verify>
    - `grep -c "M6b" manuscript/paper.tex` > 5 (M6b is now discussed)
    - `grep -c "identifiability\|recovery" manuscript/paper.tex` > 3 (identifiability is disclosed)
    - `grep -c "Bonferroni\|FDR" manuscript/paper.tex` > 2 (corrections discussed)
    - `grep -c "Collins\|Senta\|Lissek\|Myers" manuscript/paper.tex` > 4 (citations added)
    - LaTeX compiles: `cd manuscript && pdflatex -interaction=nonstopmode paper.tex` should not error on undefined references (unknown citations OK if bib file isn't updated)
    - No stale "M5 is winning" claims: `grep "M5 is.*winning\|M5.*current.*winner" manuscript/paper.tex` returns nothing
  </verify>
  <done>
    paper.tex updated with all 9 sections above. Numbers match committed output files. Identifiability caveats prominently disclosed. Literature citations inserted into Discussion.
  </done>
</task>

<task type="auto">
  <name>Task 8: Write SUMMARY.md + commit all work in atomic logical units</name>
  <files>
    .planning/quick/006-post-refit-verification-recovery-manuscript/006-SUMMARY.md
  </files>
  <action>
    Write a summary of all work done across Tasks 1-7 for future reference. Structure:

    ```markdown
    ---
    phase: quick-006
    plan: 006
    type: execute
    status: complete
    winning_model_before: wmrl_m5
    winning_model_after: wmrl_m6b
    n_participants: 154
    ---

    # Quick-006 Summary: Post-refit verification, recovery, manuscript

    ## What Changed
    - Winning model: M5 → M6b (dAIC=572.89)
    - Added BIC comparison [agreement/disagreement with AIC]
    - Fixed NaN propagation in [specific paths from Task 1]
    - Documented M2 WMRL non-convergence root cause: [from Task 1]
    - Added FDR + Bonferroni corrections to all trauma-parameter regressions
    - epsilon × IES-R findings: [survive/don't survive FDR, don't survive Bonferroni]

    ## Files Created
    [list with paths]

    ## Files Modified
    [list with paths]

    ## Scientific Decisions
    - Why BIC alongside AIC: [rationale]
    - Why disclose identifiability rather than drop M6b: [rationale — aggregate AIC still valid, kappa inferences trustworthy]
    - Family-wise correction within-model not across: [rationale]

    ## Deferred (explicit follow-ups)
    - M4 LBA parameter recovery (~48h compute)
    - M2 WMRL re-fit if Task 1 diagnosis shows NaN propagation was the root cause (Task 1 patches may require re-run to get new M2 fits)
    - Bayesian fitting of M6b for hierarchical posteriors

    ## Commits in This Task
    1. [commit 1 hash] — Task 1: verify + fix NaN guards + diagnose M2
    2. [commit 2 hash] — Task 2: add BIC to 14_compare_models.py
    3. [commit 3 hash] — Task 3: add 17_analyze_winner_heterogeneity.py
    4. [commit 4 hash] — Task 4: FDR+Bonferroni in 16_regress
    5. [commit 5 hash] — Task 5: clean PROJECT.md and update STATE.md
    6. [commit 6 hash] — Task 6: LITERATURE.md research notes
    7. [commit 7 hash] — Task 7: manuscript updates
    ```

    Then commit work in logical units. Suggested commit grouping:
    ```
    commit 1: fix(mle): verify NaN argmin guards on all fit paths; diagnose M2 non-convergence
      - scripts/fitting/fit_mle.py

    commit 2: feat(model_compare): add BIC alongside AIC in 14_compare_models.py
      - scripts/14_compare_models.py
      - output/model_comparison/comparison_results_bic.csv
      - output/model_comparison/comparison_results.csv (if extended)

    commit 3: feat(analysis): add 17_analyze_winner_heterogeneity.py for participant-level winner analysis
      - scripts/17_analyze_winner_heterogeneity.py
      - output/model_comparison/winner_heterogeneity.csv
      - output/model_comparison/winner_heterogeneity_figure.png

    commit 4: feat(regressions): add FDR-BH and Bonferroni correction to 16_regress_parameters_on_scales.py
      - scripts/16_regress_parameters_on_scales.py
      - output/regressions/*/significance_corrected.csv
      - output/regressions/*/significance_summary.md

    commit 5: docs(planning): scrub PROJECT.md of results; update STATE.md with M6b winner
      - .planning/PROJECT.md
      - .planning/STATE.md

    commit 6: docs(quick-006): literature research notes for model lineage and trauma-RL
      - .planning/quick/006-post-refit-verification-recovery-manuscript/LITERATURE.md

    commit 7: docs(manuscript): update paper.tex with M6b winner, identifiability, heterogeneity, lineage, trauma lit
      - manuscript/paper.tex

    commit 8: docs(quick-006): task summary
      - .planning/quick/006-post-refit-verification-recovery-manuscript/006-PLAN.md (if not already committed)
      - .planning/quick/006-post-refit-verification-recovery-manuscript/006-SUMMARY.md
    ```

    Per CLAUDE.md global: NO Co-Authored-By line in commit messages.

    If any task above fails and leaves partial state, commit only the parts that pass and document the failure in SUMMARY.md for manual follow-up.
  </action>
  <verify>
    - `.planning/quick/006-post-refit-verification-recovery-manuscript/006-SUMMARY.md` exists and is populated
    - `git log --oneline -10` shows the 7-8 commits in logical order
    - Each commit builds cleanly (no cross-task contamination)
    - `git status` shows clean working tree (except possibly paper.tex if user adds more while we work)
  </verify>
  <done>
    SUMMARY.md written. All work committed in atomic logical units. User can review the commit history to see the progression.
  </done>
</task>

</tasks>

<verification>

## Post-execution verification gates

**Gate A — Code correctness:**
- [ ] `python -m py_compile scripts/fitting/fit_mle.py` succeeds
- [ ] `python -m py_compile scripts/14_compare_models.py` succeeds
- [ ] `python -m py_compile scripts/16_regress_parameters_on_scales.py` succeeds
- [ ] `python -m py_compile scripts/17_analyze_winner_heterogeneity.py` succeeds

**Gate B — Output artifacts exist:**
- [ ] `output/model_comparison/comparison_results_bic.csv`
- [ ] `output/model_comparison/winner_heterogeneity.csv`
- [ ] `output/model_comparison/winner_heterogeneity_figure.png`
- [ ] `output/regressions/wmrl_m6b/significance_corrected.csv`
- [ ] `output/regressions/wmrl_m6b/significance_summary.md`

**Gate C — Documentation integrity:**
- [ ] PROJECT.md has no result-contamination matches (grep for dAIC, winning, p=, etc.)
- [ ] STATE.md mentions M6b as winning model
- [ ] LITERATURE.md has >=11 citations (6 lineage + 5 trauma)
- [ ] SUMMARY.md exists and cross-references all 7 commits

**Gate D — Manuscript quality:**
- [ ] paper.tex mentions M6b >5 times
- [ ] paper.tex mentions identifiability/recovery >3 times
- [ ] paper.tex mentions Bonferroni or FDR >2 times
- [ ] paper.tex cites Collins, Senta, Lissek, Myers (or marked as TBD)
- [ ] paper.tex has participant-level heterogeneity figure included
- [ ] paper.tex has stratified comparison table/section

**Gate E — Git hygiene:**
- [ ] Commits are atomic (each touches one logical concern)
- [ ] No Co-Authored-By lines
- [ ] Commit messages follow conventional format (feat/fix/docs prefixes)

</verification>

<success_criteria>

This task is complete when:
1. All 12 user requests addressed (or explicitly deferred with justification)
2. All 7 sub-tasks above executed
3. 7-8 atomic commits on main branch
4. Manuscript paper.tex reflects the M6b reality with honest limitations
5. SUMMARY.md cross-references everything
6. User can hand the manuscript to a collaborator and have the story internally consistent

Explicitly deferred (with user acknowledgement):
- M4 LBA parameter recovery (~48h)
- Possible M2 re-fit IF Task 1 diagnosis shows NaN was the only issue
- Any task 1 code fix that requires re-running expensive fits

</success_criteria>

<execution_order>

**Dependency graph:**
```
Task 1 (code audit) ──┐
                      │
Task 2 (BIC)    ──────┤
                      │
Task 3 (heterog) ─────┤──> Task 7 (manuscript)
                      │          │
Task 4 (FDR)    ──────┤          │
                      │          │
Task 5 (docs)   ──────┤          │
                      │          │
Task 6 (literature) ──┘          │
                                 ▼
                          Task 8 (SUMMARY + commits)
```

**Recommended execution order:**
1. Task 1 first (can be done in parallel with 2-6 but if NaN fix needed, informs later tasks)
2. Tasks 2, 3, 4, 5, 6 can be done in any order (all independent)
3. Task 7 LAST before summary — it consumes outputs from all prior tasks
4. Task 8 final — writes SUMMARY and organizes commits

**Single-wave execution is fine** because no task blocks another from STARTING — only Task 7 requires the outputs of 1-6 to COMPLETE. The executor can thread tasks 1-6 linearly, then 7, then 8.

</execution_order>

<deferred>

## Explicitly Deferred for Future Quick Tasks

### M4 LBA Parameter Recovery (~48h)
- STATE.md notes M4 is slow; full recovery on cluster would take ~48h
- Decision: Skip for quick-006. Manuscript will note "M4 parameter recovery not yet performed" in Limitations
- Follow-up: Schedule cluster job, estimated quick-007 or background batch
- Cluster command (for reference): `sbatch cluster/11_recovery.slurm --model wmrl_m4 --n-subjects 50 --n-datasets 10`

### M2 WMRL Re-fit (conditional on Task 1 diagnosis)
- IF Task 1 finds that the NaN propagation bug caused M2's 33% non-convergence → re-fit M2 with patched code
- IF Task 1 finds it's a bounds / optimizer issue → M2 may need bound relaxation (separate investigation)
- Decision: Document the root cause in quick-006 SUMMARY. Actual re-fit happens in a follow-up task if justified.

### Bayesian Hierarchical Fitting of M6b
- Task notes in STATE.md that we have Bayesian infrastructure but haven't run M6b through it
- Would give proper posterior distributions and shrinkage, which could partially address the identifiability problem (hierarchical priors regularize the base RLWM params)
- Decision: Out of scope for quick-006. Schedule for quick-007 or v3.0 phase plan.

### Cross-model Recovery Validation
- Task notes we have `11_run_model_recovery.py --mode cross-model` but haven't re-run it with M5/M6a/M6b included
- Would validate that the new models are distinguishable at the AIC level
- Decision: Out of scope for quick-006.

</deferred>

<output>
After completion, ensure `.planning/quick/006-post-refit-verification-recovery-manuscript/006-SUMMARY.md` is written (per Task 8) and referenced from STATE.md if appropriate.
</output>
