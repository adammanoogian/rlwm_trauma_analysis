---
phase: quick-008-cleanup-pipeline-and-docs
plan: 008
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/04_1_explore_survey_data.py                          # deleted
  - scripts/06_1_plot_task_performance.py                        # deleted
  - scripts/07_1_visualize_by_trauma_group.py                    # deleted
  - scripts/09_1_simulate_model_predictions.py                   # deleted
  - scripts/13_fit_bayesian_m4.py                                # deleted
  - scripts/16b_bayesian_regression.py                           # deleted
  - scripts/18b_mle_vs_bayes_reliability.py                      # deleted
  - scripts/analysis/trauma_scale_distributions.py               # path fix
  - docs/legacy/                                                  # created
  - docs/legacy/JAX_GPU_BAYESIAN_FITTING.md                      # moved in
  - docs/legacy/CONVERGENCE_ASSESSMENT.md                        # moved in
  - docs/legacy/DEER_NONLINEAR_PARALLELIZATION.md                # moved in
  - docs/JAX_GPU_BAYESIAN_FITTING.md                             # moved out
  - docs/CONVERGENCE_ASSESSMENT.md                               # moved out
  - docs/DEER_NONLINEAR_PARALLELIZATION.md                       # moved out
  - docs/README.md                                                # updated
  - docs/02_pipeline_guide/ANALYSIS_PIPELINE.md                  # updated
  - docs/02_pipeline_guide/PLOTTING_REFERENCE.md                 # updated
  - docs/04_methods/                                              # created
  - docs/04_methods/README.md                                     # created
  - docs/04_results/                                              # created
  - docs/04_results/README.md                                     # created
autonomous: true

must_haves:
  truths:
    - "Seven orphaned pipeline scripts no longer exist on disk"
    - "docs/legacy/ exists and contains three superseded documents"
    - "docs/README.md matches the actual docs/ layout after cleanup"
    - "docs/04_methods/ and docs/04_results/ scaffolding exists with short README indexes"
    - "scripts/analysis/trauma_scale_distributions.py writes to figures/scale_distributions.png (the path paper.qmd expects)"
    - "docs/02_pipeline_guide/ANALYSIS_PIPELINE.md Stage 4 lists all 7 models (M1-M6b + M4)"
    - "docs/02_pipeline_guide/PLOTTING_REFERENCE.md covers plot_posterior_diagnostics.py and other scripts/visualization/* tools"
  artifacts:
    - path: "docs/legacy/README.md"
      provides: "Index of archived docs with reason-for-archival"
    - path: "docs/04_methods/README.md"
      provides: "Scaffolding index for methods documentation"
    - path: "docs/04_results/README.md"
      provides: "Scaffolding index for all pipeline results (including orphaned/supplementary)"
    - path: "scripts/analysis/trauma_scale_distributions.py"
      provides: "Scale distributions figure writer with corrected output path"
      contains: "figures/scale_distributions.png"
  key_links:
    - from: "manuscript/paper.qmd line 215"
      to: "figures/scale_distributions.png"
      via: "markdown image include (relative to manuscript/ = manuscript/figures/)"
      pattern: "figures/scale_distributions.png"
    - from: "scripts/analysis/trauma_scale_distributions.py"
      to: "figures/scale_distributions.png"
      via: "plt.savefig on corrected canonical path"
      pattern: "figures/scale_distributions.png"
---

<objective>
Execute the pipeline + docs cleanup dictated by the 2026-04-17 three-part audit. Four concerns: (1) prune seven orphaned/superseded scripts, (2) reorganize docs into a clean layout with a working docs/legacy/, (3) scaffold docs/04_methods/ and docs/04_results/ as empty indexes for future content, (4) fix the scale_distributions.png output-path divergence between paper.qmd and the producing script.

Purpose: The repo accumulated parallel/exploratory scripts and docs over Phases 13-20 that are no longer referenced by the pipeline or paper.qmd. Removing them reduces cognitive load, prevents a future engineer from resurrecting dead code, and eliminates one known path bug (scale_distributions.png). The docs/04 scaffolding establishes a home for results that sit outside the paper (orphaned plots, validation outputs) without contaminating the paper or the existing docs/ structure.

Output: Seven deleted scripts, three moved docs (into docs/legacy/), a fixed path in one analysis script, four updated docs pages (README + two pipeline guide pages + two new index READMEs under docs/04_*), and a clean `git status` with the changes staged into a single commit.
</objective>

<execution_context>
@C:\Users\aman0087\.claude/get-shit-done/workflows/execute-plan.md
@C:\Users\aman0087\.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md

# Audit source of truth (read this first; the table in the conversation that spawned this plan is authoritative)
# The following are the files the audit identified:

# Scripts to delete
@scripts/04_1_explore_survey_data.py
@scripts/06_1_plot_task_performance.py
@scripts/07_1_visualize_by_trauma_group.py
@scripts/09_1_simulate_model_predictions.py
@scripts/13_fit_bayesian_m4.py
@scripts/16b_bayesian_regression.py
@scripts/18b_mle_vs_bayes_reliability.py

# Docs to move to legacy/
@docs/JAX_GPU_BAYESIAN_FITTING.md
@docs/CONVERGENCE_ASSESSMENT.md
@docs/DEER_NONLINEAR_PARALLELIZATION.md

# Docs to update in place
@docs/README.md
@docs/02_pipeline_guide/ANALYSIS_PIPELINE.md
@docs/02_pipeline_guide/PLOTTING_REFERENCE.md

# Path bug source
@scripts/analysis/trauma_scale_distributions.py
@manuscript/paper.qmd
</context>

<tasks>

<task type="auto">
  <name>Task 1: Prune seven orphaned/superseded pipeline scripts</name>
  <files>
    scripts/04_1_explore_survey_data.py (DELETE)
    scripts/06_1_plot_task_performance.py (DELETE)
    scripts/07_1_visualize_by_trauma_group.py (DELETE)
    scripts/09_1_simulate_model_predictions.py (DELETE)
    scripts/13_fit_bayesian_m4.py (DELETE)
    scripts/16b_bayesian_regression.py (DELETE)
    scripts/18b_mle_vs_bayes_reliability.py (DELETE)
  </files>
  <action>
    Delete the seven scripts listed above. Use `git rm` (not `rm`) so deletions stage automatically.

    Important context before deleting `scripts/04_1_explore_survey_data.py`:
    - `manuscript/paper.qmd` line 215 references `figures/scale_distributions.png` (expected resolved location: `manuscript/figures/scale_distributions.png` when Quarto renders).
    - `scripts/04_1_explore_survey_data.py` line 459 produces `figures/behavioral_summary/scale_distributions.png` (wrong subdir for paper).
    - `scripts/analysis/trauma_scale_distributions.py` line 184 produces `figures/trauma_scale_analysis/scale_distributions.png` (also wrong subdir for paper).
    - The audit resolves this by: (a) deleting `04_1_explore_survey_data.py` entirely, and (b) fixing `scripts/analysis/trauma_scale_distributions.py` to write to the canonical `figures/scale_distributions.png`. Task 3 below handles (b). You do not need to inline any code from 04_1 into 04 — the `trauma_scale_distributions.py` script is the canonical producer and already implements the relevant plot.

    Verification first:
    - Before `git rm`, grep the repo (excluding `.planning/`, `.git/`, `__pycache__`) for each filename to confirm no remaining imports or `subprocess.run` call references any of these scripts. Example:
      ```
      git grep -l "04_1_explore_survey_data\|06_1_plot_task_performance\|07_1_visualize_by_trauma_group\|09_1_simulate_model_predictions\|13_fit_bayesian_m4\|16b_bayesian_regression\|18b_mle_vs_bayes_reliability" -- ':!.planning/' ':!__pycache__/'
      ```
    - If grep finds a reference in a script you were NOT told to delete (e.g. a CI/SLURM file), STOP and report. Do not silently leave dangling references. If the only hits are within the files being deleted themselves (e.g. self-reference in docstring), proceed.
    - Do NOT grep `.planning/` — historical plans legitimately reference these filenames, that is expected.

    Deletion commands (run from repo root):
    ```
    git rm scripts/04_1_explore_survey_data.py
    git rm scripts/06_1_plot_task_performance.py
    git rm scripts/07_1_visualize_by_trauma_group.py
    git rm scripts/09_1_simulate_model_predictions.py
    git rm scripts/13_fit_bayesian_m4.py
    git rm scripts/16b_bayesian_regression.py
    git rm scripts/18b_mle_vs_bayes_reliability.py
    ```
  </action>
  <verify>
    1. `ls scripts/ | grep -E "(04_1|06_1|07_1|09_1|13_fit_bayesian_m4|16b_bayesian_regression|18b_mle_vs_bayes_reliability)"` returns nothing.
    2. `git status` shows exactly 7 deletions under `scripts/` (ignore staging state — `git rm` both removes and stages).
    3. `git grep "scripts/04_1_explore_survey_data\|scripts/06_1_plot_task_performance\|scripts/07_1_visualize_by_trauma_group\|scripts/09_1_simulate_model_predictions\|scripts/13_fit_bayesian_m4\|scripts/16b_bayesian_regression\|scripts/18b_mle_vs_bayes_reliability" -- ':!.planning/'` returns no matches (outside .planning/).
  </verify>
  <done>
    Seven scripts removed from the working tree and staged for deletion. No remaining non-`.planning/` code/doc file imports or invokes any of them. `git status` clean except for the seven deletions plus any changes from Tasks 2-4.
  </done>
</task>

<task type="auto">
  <name>Task 2: Reorganize docs — create legacy/, move three superseded docs, update README</name>
  <files>
    docs/legacy/ (NEW DIR)
    docs/legacy/README.md (NEW)
    docs/legacy/JAX_GPU_BAYESIAN_FITTING.md (MOVED from docs/)
    docs/legacy/CONVERGENCE_ASSESSMENT.md (MOVED from docs/)
    docs/legacy/DEER_NONLINEAR_PARALLELIZATION.md (MOVED from docs/)
    docs/README.md (UPDATED)
    docs/02_pipeline_guide/ANALYSIS_PIPELINE.md (UPDATED)
    docs/02_pipeline_guide/PLOTTING_REFERENCE.md (UPDATED)
  </files>
  <action>
    Create `docs/legacy/` and move three docs into it via `git mv` (preserves history). Update three doc files in place.

    **Step 2a — Create docs/legacy/ and move three docs:**
    ```
    mkdir -p docs/legacy
    git mv docs/JAX_GPU_BAYESIAN_FITTING.md docs/legacy/JAX_GPU_BAYESIAN_FITTING.md
    git mv docs/CONVERGENCE_ASSESSMENT.md docs/legacy/CONVERGENCE_ASSESSMENT.md
    git mv docs/DEER_NONLINEAR_PARALLELIZATION.md docs/legacy/DEER_NONLINEAR_PARALLELIZATION.md
    ```

    **Step 2b — Create docs/legacy/README.md:**
    A short index (≤ 40 lines) explaining what's in legacy/ and why. Content:
    ```
    # docs/legacy/ — Archived Documentation

    Superseded or closed-out documents retained for history. Do not use these as
    references for current code.

    | File | Reason for archival | Replacement |
    |---|---|---|
    | JAX_GPU_BAYESIAN_FITTING.md | Early JAX/GPU setup notes from Phase 13-14 — superseded by the operational writeup in CLUSTER_GPU_LESSONS.md | ../CLUSTER_GPU_LESSONS.md |
    | CONVERGENCE_ASSESSMENT.md | Standalone convergence-diagnostics reference — content now covered by HIERARCHICAL_BAYESIAN.md §4.1 | ../HIERARCHICAL_BAYESIAN.md |
    | DEER_NONLINEAR_PARALLELIZATION.md | Phase 20-01 investigation (NO-GO decision locked) | Decision recorded in PARALLEL_SCAN_LIKELIHOOD.md |
    ```

    **Step 2c — Update docs/README.md:**
    Replace the existing tree-diagram block + "Start here" block with a rewritten version reflecting the new layout. Key changes:
    - Drop the three moved files (JAX_GPU_BAYESIAN_FITTING.md, CONVERGENCE_ASSESSMENT.md, DEER_NONLINEAR_PARALLELIZATION.md) from the top-level listing.
    - Add docs/04_methods/ and docs/04_results/ placeholders (created in Task 4 — the README just needs to list them).
    - Keep legacy/ listing but mark it as populated: "Archived superseded docs — see docs/legacy/README.md".
    - Keep existing "Start here" bullets (ANALYSIS_PIPELINE, HIERARCHICAL_BAYESIAN, CLUSTER_GPU_LESSONS, MODEL_REFERENCE, SCALES_AND_FITTING_AUDIT) unchanged.

    **Step 2d — Update docs/02_pipeline_guide/ANALYSIS_PIPELINE.md Stage 4:**
    Read the current Stage 4 section (model fitting). Add entries for M5 (wmrl_m5), M6a (wmrl_m6a), M6b (wmrl_m6b), and M4 (wmrl_m4) — these are currently missing. Mirror the format of the existing M1/M2/M3 entries. Parameter lists come from the project CLAUDE.md "Parameter Summary" table:
    - M5: α₊, α₋, φ, ρ, K, κ, φ_rl, ε
    - M6a: α₊, α₋, φ, ρ, K, κ_s, ε
    - M6b: α₊, α₋, φ, ρ, K, κ_total, κ_share, ε
    - M4: α₊, α₋, φ, ρ, K, κ, v_scale, A, δ, t₀ — note this is the ONLY joint choice+RT model and AIC is NOT comparable to choice-only models.

    Then add Stage 5b (winner heterogeneity — `scripts/17_analyze_winner_heterogeneity.py`) and Stage 5c (Bayesian Level-2 effects — `scripts/18_bayesian_level2_effects.py`) sections below existing Stage 5. One short paragraph each describing inputs, outputs, when to run. Do not write content for artifacts that don't exist yet (e.g. if `output/bayesian/level2/` is empty, mark the outputs as "generated after cluster Bayesian fit completes").

    **Step 2e — Update docs/02_pipeline_guide/PLOTTING_REFERENCE.md:**
    Read the current file. Extend it to document ALL visualization scripts in `scripts/visualization/` — currently only a subset is covered. The full list is:
    - create_modeling_figures.py
    - create_modeling_tables.py
    - create_parameter_behavior_heatmap.py
    - create_publication_figures.py
    - create_supplementary_materials.py
    - create_supplementary_table_s3.py
    - plot_group_parameters.py
    - plot_model_comparison.py
    - **plot_posterior_diagnostics.py** (new today — explicitly call out)
    - plot_wmrl_forest.py
    - quick_arviz_plots.py

    For each, one-sentence purpose + primary input + primary output. Use a table if there isn't already a structured list — match the existing file's style.

    Do NOT merge SCALES_AND_FITTING_AUDIT and HIERARCHICAL_BAYESIAN (explicit audit directive — they are distinct reference docs and should stay separate).
  </action>
  <verify>
    1. `ls docs/legacy/` shows exactly: `CONVERGENCE_ASSESSMENT.md`, `DEER_NONLINEAR_PARALLELIZATION.md`, `JAX_GPU_BAYESIAN_FITTING.md`, `README.md`.
    2. `ls docs/` no longer shows `JAX_GPU_BAYESIAN_FITTING.md`, `CONVERGENCE_ASSESSMENT.md`, or `DEER_NONLINEAR_PARALLELIZATION.md` at the top level.
    3. `grep -c "wmrl_m5\|wmrl_m6a\|wmrl_m6b\|wmrl_m4" docs/02_pipeline_guide/ANALYSIS_PIPELINE.md` returns a count ≥ 4 (each model appears at least once).
    4. `grep "plot_posterior_diagnostics" docs/02_pipeline_guide/PLOTTING_REFERENCE.md` returns at least one hit.
    5. `grep "JAX_GPU_BAYESIAN_FITTING\|CONVERGENCE_ASSESSMENT\|DEER_NONLINEAR_PARALLELIZATION" docs/README.md` — the top-level entries are gone (matches only if the strings appear in a legacy/ pointer line, which is acceptable).
    6. `docs/legacy/README.md` file exists and is ≤ 40 lines.
  </verify>
  <done>
    docs/legacy/ exists with 3 moved docs + README.md. Top-level docs/README.md reflects the new layout. Stage 4 of ANALYSIS_PIPELINE.md documents all 7 models (not just 3). PLOTTING_REFERENCE.md covers all 11 scripts in scripts/visualization/. SCALES_AND_FITTING_AUDIT and HIERARCHICAL_BAYESIAN remain distinct files.
  </done>
</task>

<task type="auto">
  <name>Task 3: Fix scale_distributions.png output path in trauma_scale_distributions.py</name>
  <files>
    scripts/analysis/trauma_scale_distributions.py
  </files>
  <action>
    The script currently writes `figures/trauma_scale_analysis/scale_distributions.png` (line 36: `FIGURES_DIR = Path('figures/trauma_scale_analysis')` + line 184: `plt.savefig(FIGURES_DIR / 'scale_distributions.png', ...)`).

    Paper `manuscript/paper.qmd` line 215 references `figures/scale_distributions.png` (resolves to `manuscript/figures/scale_distributions.png` at render time since Quarto runs from `manuscript/`).

    Audit directive: "change the script to write to `figures/scale_distributions.png`". Interpret as: write to the repo-root-relative canonical path `figures/scale_distributions.png`. The paper.qmd image include on line 215 currently assumes this file sits under `manuscript/figures/`. Since `manuscript/figures/` is effectively empty (contains only `plot_utils.py`), and other figure-path references in paper.qmd (e.g. line 946) already use `../figures/...` for the repo-level figures directory, align line 215 with that convention.

    **Changes:**

    1. In `scripts/analysis/trauma_scale_distributions.py`:
       - Leave the OTHER outputs that live under `figures/trauma_scale_analysis/` alone — this subdirectory is still used for the file's secondary outputs (correlation matrix, etc.). Only the `scale_distributions.png` line needs to move.
       - At line ~184, change the savefig call from `FIGURES_DIR / 'scale_distributions.png'` to an explicit canonical path. The cleanest change:
         - Add near the top of the file (around line 36-37): `CANONICAL_FIGURE_PATH = Path('figures/scale_distributions.png')  # paper.qmd expects this location`
         - Change line 184 savefig to: `plt.savefig(CANONICAL_FIGURE_PATH, dpi=300, bbox_inches='tight')`
         - Change line 187 print message to match: `print(f"\nSaved histogram figure to {CANONICAL_FIGURE_PATH}")`
       - Add an `os.makedirs(CANONICAL_FIGURE_PATH.parent, exist_ok=True)` (or `CANONICAL_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)`) immediately before the savefig so the `figures/` dir is auto-created if missing (it exists today but be defensive).
       - Leave all other outputs (anything else saved to `FIGURES_DIR`) untouched.

    2. In `manuscript/paper.qmd`:
       - Line 215 currently: `![](figures/scale_distributions.png){width=100%}`
       - Change to: `![](../figures/scale_distributions.png){width=100%}` — this is consistent with line 946's `Path("../figures/mle_trauma_analysis/...")` convention and points to the repo-root `figures/` directory.
       - Also update the prose reference on lines 224-225 if it repeats the path literally. Only update the path string; do NOT rewrite surrounding prose.

    **Do NOT create `figures/scale_distributions.png` by running the script.** The audit non-goals are explicit: do not run cluster jobs or re-execute pipeline scripts. This task is a path-wiring fix only. The file will be produced on the next pipeline rerun.
  </action>
  <verify>
    1. `grep -n "scale_distributions" scripts/analysis/trauma_scale_distributions.py` shows the savefig now points to `figures/scale_distributions.png` (via `CANONICAL_FIGURE_PATH`), not `figures/trauma_scale_analysis/scale_distributions.png`.
    2. `grep -n "scale_distributions" manuscript/paper.qmd` shows line 215 now uses `../figures/scale_distributions.png`.
    3. `python -c "import ast; ast.parse(open('scripts/analysis/trauma_scale_distributions.py').read())"` succeeds (no syntax errors introduced).
    4. The script's OTHER saves (any other `FIGURES_DIR / '...'` calls) are unchanged — `grep -c "FIGURES_DIR /" scripts/analysis/trauma_scale_distributions.py` matches the pre-edit count minus 1 (only the scale_distributions.png save was moved off FIGURES_DIR).
  </verify>
  <done>
    trauma_scale_distributions.py writes `scale_distributions.png` to `figures/scale_distributions.png` (canonical repo-root path). paper.qmd line 215 points to `../figures/scale_distributions.png`, consistent with the rest of the manuscript's figure-loading convention. Script still parses. Other outputs under `figures/trauma_scale_analysis/` are untouched.
  </done>
</task>

<task type="auto">
  <name>Task 4: Scaffold docs/04_methods/ and docs/04_results/ with short README indexes</name>
  <files>
    docs/04_methods/ (NEW DIR)
    docs/04_methods/README.md (NEW)
    docs/04_results/ (NEW DIR)
    docs/04_results/README.md (NEW)
  </files>
  <action>
    Create two new empty directories with one README.md each. These are SCAFFOLDING — the READMEs are short indexes with placeholder tables, not full content. Keep each README under ~60 lines.

    **Step 4a — Create docs/04_methods/README.md:**

    Intent (state in the preamble, paraphrased from the audit): "Holds methods documentation for scientifically-relevant analyses AND one-off validations. Scripts 09/10/11 validation outputs are documented here. Distinct from the Quarto paper, which holds only published analyses. Populate entries as the corresponding scripts are run or documented."

    Structure:
    ```
    # docs/04_methods/ — Methods Documentation Index

    Methods notes for published analyses AND supplementary/validation analyses
    that do not appear in manuscript/paper.qmd. Each entry points to the
    producing script and a short method writeup. Populate entries as new
    methods are added or old ones documented.

    ## Published-in-paper methods

    | Topic | Producing script | Method doc |
    |---|---|---|
    | Task structure and environment | scripts/environments/rlwm_env.py | ../03_methods_reference/TASK_AND_ENVIRONMENT.md |
    | Model mathematics | scripts/fitting/jax_likelihoods.py | ../03_methods_reference/MODEL_REFERENCE.md |
    | Hierarchical Bayesian architecture | scripts/fitting/numpyro_models.py | ../HIERARCHICAL_BAYESIAN.md |
    | Scale orthogonalization (IES-R) | scripts/fitting/level2_design.py | ../SCALES_AND_FITTING_AUDIT.md |

    ## Supplementary / validation methods

    | Topic | Producing script | Method doc |
    |---|---|---|
    | Posterior predictive checks | scripts/09_run_ppc.py | _TODO_ |
    | Synthetic-data generation | scripts/09_generate_synthetic_data.py | _TODO_ |
    | Parameter sweep | scripts/10_run_parameter_sweep.py | _TODO_ |
    | Parameter recovery | scripts/11_run_model_recovery.py | _TODO_ |
    | Posterior-vs-MLE sanity check | validation/compare_posterior_to_mle.py | _TODO_ |

    Entries marked _TODO_ are scaffolding. Add short method writeups here
    as results are produced or as reviewers ask for them.
    ```

    **Step 4b — Create docs/04_results/README.md:**

    Intent (paraphrased from the audit): "Holds ALL pipeline results including orphaned/supplementary ones not in paper.qmd. Links to output/* with one-line provenance per result. This is the place to record artifacts that exist on disk but do not appear in the manuscript."

    Structure:
    ```
    # docs/04_results/ — Pipeline Results Index

    Every top-level result category produced by the pipeline, including
    supplementary and orphaned artifacts not shown in manuscript/paper.qmd.
    One row per result with producer + canonical output path + provenance.
    Placeholders are used where artifacts do not yet exist (e.g. Bayesian
    posteriors blocked on cluster runs).

    ## Behavioral

    | Result | Producer | Output path | Status |
    |---|---|---|---|
    | Task performance plots | scripts/06_visualize_task_performance.py | figures/behavioral_analysis/ | available |
    | Trauma-group behavioral stats | scripts/07_analyze_trauma_groups.py | figures/trauma_groups/, output/summary_by_trauma.csv | available |
    | Statistical analyses (ANOVA, descriptives) | scripts/08_run_statistical_analyses.py | output/statistical_analyses/ | available |
    | Scale distributions | scripts/analysis/trauma_scale_distributions.py | figures/scale_distributions.png, figures/trauma_scale_analysis/ | available after pipeline rerun |

    ## Model fitting (MLE)

    | Result | Producer | Output path | Status |
    |---|---|---|---|
    | Individual MLE fits (all 7 models) | scripts/12_fit_mle.py | output/mle/{model}_individual_fits.csv | available |
    | Model comparison (AIC/BIC) | scripts/14_compare_models.py | output/model_comparison/ | available |
    | Winner heterogeneity | scripts/17_analyze_winner_heterogeneity.py | output/model_comparison/winner_heterogeneity*.csv, figures/model_comparison/winner_heterogeneity_figure.png | available |

    ## Trauma associations (MLE)

    | Result | Producer | Output path | Status |
    |---|---|---|---|
    | Parameter-trauma correlations | scripts/15_analyze_mle_by_trauma.py | output/regressions/{model}/ | available |
    | FDR/Bonferroni-corrected regressions | scripts/16_regress_parameters_on_scales.py | output/regressions/{model}/significance_*.{csv,md} | available |

    ## Bayesian (blocked on cluster)

    | Result | Producer | Output path | Status |
    |---|---|---|---|
    | Hierarchical posteriors (6 choice-only models) | scripts/13_fit_bayesian.py | output/bayesian/{model}_posterior.nc | _placeholder — cluster refit pending_ |
    | M4 LBA posterior | scripts/13_fit_bayesian.py --model wmrl_m4 | output/bayesian/wmrl_m4_posterior.nc | _placeholder — cluster refit pending_ |
    | Pscan benchmarks | cluster/13_bayesian_pscan.slurm | output/bayesian/pscan_benchmark.json | available |
    | M6b posterior diagnostics | scripts/visualization/plot_posterior_diagnostics.py | figures/m6b_posterior_diagnostics.png | _placeholder — needs posterior first_ |
    | M6b posterior vs MLE | validation/compare_posterior_to_mle.py | figures/m6b_posterior_vs_mle.png | _placeholder — needs posterior first_ |
    | Level-2 stacking weights | scripts/14_compare_models.py --bayesian-comparison | output/bayesian/level2/stacking_weights.csv | _placeholder — needs posterior first_ |
    | Level-2 forest plots | scripts/18_bayesian_level2_effects.py | output/bayesian/figures/m6b_forest_lec5.png | _placeholder — needs posterior first_ |

    Entries marked _placeholder_ will be filled in after the next cluster
    Bayesian fit completes (see .planning/STATE.md for current blocker).
    ```

    Keep both READMEs tight. Do NOT write full method writeups or full result narratives — the audit explicitly calls these "scaffolding, not full content". The whole point is a navigable index that future-you (or a reviewer) can use to locate any result without grepping the whole repo.
  </action>
  <verify>
    1. `ls docs/04_methods/ docs/04_results/` each shows exactly `README.md`.
    2. `wc -l docs/04_methods/README.md docs/04_results/README.md` — each is ≤ 80 lines (tight).
    3. `grep -c "_TODO_\|_placeholder_" docs/04_methods/README.md docs/04_results/README.md` — placeholders exist, confirming scaffolding nature.
    4. Each README's top paragraph states the intent (methods = published + validation; results = all pipeline outputs incl. orphaned/supplementary).
    5. `grep "plot_posterior_diagnostics" docs/04_results/README.md` returns a hit (the new plotter appears in the index).
    6. `grep "scale_distributions" docs/04_results/README.md` returns a hit (the path-fixed figure is indexed).
  </verify>
  <done>
    docs/04_methods/ and docs/04_results/ exist, each with a README.md index. Both READMEs are short (≤ 80 lines), use placeholder tables for artifacts that don't yet exist, and correctly reference scripts/visualization/plot_posterior_diagnostics.py + figures/scale_distributions.png. No full content written — this is navigable scaffolding.
  </done>
</task>

</tasks>

<verification>
Run after all four tasks complete, from repo root:

1. **Scripts pruned**
   - `ls scripts/ | grep -E '^(04_1|06_1|07_1|09_1|13_fit_bayesian_m4|16b_bayesian_regression|18b_mle_vs_bayes_reliability)'` — returns empty.
   - `git status` shows 7 deletions under `scripts/` plus new/modified files from Tasks 2-4. No untracked orphan files should remain.

2. **Docs reorganized**
   - `ls docs/` no longer contains `JAX_GPU_BAYESIAN_FITTING.md`, `CONVERGENCE_ASSESSMENT.md`, `DEER_NONLINEAR_PARALLELIZATION.md`.
   - `ls docs/legacy/` contains exactly 4 entries: those 3 files + `README.md`.
   - `ls docs/04_methods/ docs/04_results/` each shows `README.md`.
   - `docs/README.md` no longer lists the 3 moved files at top level (or lists them only inside a legacy/ pointer line).
   - `grep -c "wmrl_m5\|wmrl_m6a\|wmrl_m6b\|wmrl_m4" docs/02_pipeline_guide/ANALYSIS_PIPELINE.md` ≥ 4.
   - `grep -c "plot_posterior_diagnostics" docs/02_pipeline_guide/PLOTTING_REFERENCE.md` ≥ 1.

3. **Path fix holds**
   - `grep -c "figures/scale_distributions.png" scripts/analysis/trauma_scale_distributions.py` ≥ 1 (new canonical path).
   - `grep -c "figures/trauma_scale_analysis/scale_distributions.png" scripts/analysis/trauma_scale_distributions.py` == 0 (old bad path gone from that specific filename; other files under trauma_scale_analysis/ remain fine).
   - `grep -c "../figures/scale_distributions.png" manuscript/paper.qmd` ≥ 1 (paper updated to use ../ prefix).
   - `python -c "import ast; ast.parse(open('scripts/analysis/trauma_scale_distributions.py').read())"` succeeds.

4. **No dangling references**
   - `git grep -E "scripts/(04_1_explore_survey_data|06_1_plot_task_performance|07_1_visualize_by_trauma_group|09_1_simulate_model_predictions|13_fit_bayesian_m4|16b_bayesian_regression|18b_mle_vs_bayes_reliability)" -- ':!.planning/'` returns no matches outside `.planning/`.
   - `git grep -E "docs/(JAX_GPU_BAYESIAN_FITTING|CONVERGENCE_ASSESSMENT|DEER_NONLINEAR_PARALLELIZATION)" -- ':!docs/legacy/' ':!.planning/'` returns no matches (all remaining refs, if any, live under docs/legacy/ or .planning/).

5. **Commit staging**
   - `git status` at end shows: 7 deletions under scripts/, 3 renames under docs/, 3-5 modifications (docs/README.md, ANALYSIS_PIPELINE.md, PLOTTING_REFERENCE.md, trauma_scale_distributions.py, manuscript/paper.qmd), and 4 new files (docs/legacy/README.md, docs/04_methods/README.md, docs/04_results/README.md). All cleanly stageable in one commit.

If any check above fails, do not commit — report the failing check and investigate.
</verification>

<success_criteria>
- Seven pruned scripts no longer exist; no non-`.planning/` code references them.
- `docs/legacy/` exists with 3 moved docs + index README.
- `docs/README.md`, `docs/02_pipeline_guide/ANALYSIS_PIPELINE.md`, `docs/02_pipeline_guide/PLOTTING_REFERENCE.md` reflect the new layout and cover all 7 models + all 11 visualization scripts.
- `docs/04_methods/README.md` and `docs/04_results/README.md` exist as short scaffolding indexes (≤ 80 lines each).
- `scripts/analysis/trauma_scale_distributions.py` writes `scale_distributions.png` to `figures/scale_distributions.png`; `manuscript/paper.qmd` line 215 points to `../figures/scale_distributions.png` (consistent with line 946 convention).
- Script still parses (AST-valid Python).
- Single clean commit stageable via `git status` at end.

NON-GOALS (reaffirm — do NOT do):
- Do NOT run cluster jobs or fit Bayesian models.
- Do NOT create `figures/m6b_posterior_diagnostics.png`, `figures/m6b_posterior_vs_mle.png`, or `output/bayesian/level2/stacking_weights.csv`.
- Do NOT refactor numpyro_models.py, jax_likelihoods.py, or any fitting code.
- Do NOT rename numbered pipeline scripts (01-18) — the numbers are a user-facing interface.
- Do NOT merge SCALES_AND_FITTING_AUDIT and HIERARCHICAL_BAYESIAN.
- Do NOT regenerate `figures/scale_distributions.png` — Task 3 is path-wiring only.
</success_criteria>

<output>
After completion, create `.planning/quick/008-cleanup-pipeline-and-docs/008-SUMMARY.md` following the template at `C:\Users\aman0087\.claude/get-shit-done/templates/summary.md`. Key sections to populate:
- Commits (one expected; list the SHA)
- Files touched (deletions, renames, modifications, new files — match the `files_modified` frontmatter)
- Follow-ups (none expected; this plan is self-contained)
- Decisions (record any interpretation choices made during path-fix, e.g. paper.qmd `../figures/` convention adoption)

Then commit all changes in a single commit. Suggested commit message:

```
chore(quick-008): cleanup pipeline scripts, reorganize docs, fix scale_distributions path

- Delete 7 orphaned/superseded scripts (04_1, 06_1, 07_1, 09_1, 13_fit_bayesian_m4,
  16b_bayesian_regression, 18b_mle_vs_bayes_reliability)
- Move 3 superseded docs to docs/legacy/ (JAX_GPU_BAYESIAN_FITTING,
  CONVERGENCE_ASSESSMENT, DEER_NONLINEAR_PARALLELIZATION)
- Update docs/README, ANALYSIS_PIPELINE (add M5/M6a/M6b/M4 + Stage 5b/5c),
  PLOTTING_REFERENCE (cover all scripts/visualization/*)
- Scaffold docs/04_methods/ and docs/04_results/ with short index READMEs
- Fix scripts/analysis/trauma_scale_distributions.py output path to
  figures/scale_distributions.png (canonical); update paper.qmd line 215
  to ../figures/scale_distributions.png for consistency with line 946

Based on 2026-04-17 three-part audit findings.
```
</output>
