---
phase: 29-pipeline-canonical-reorg
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/01_data_preprocessing/                       (new — from scripts/data_processing/ via git mv)
  - scripts/02_behav_analyses/                           (new — from scripts/behavioral/ via git mv)
  - scripts/03_model_prefitting/                         (new — absorbs simulations_recovery/ + bayesian_pipeline/21_run_{prior_predictive,bayesian_recovery}.py)
  - scripts/04_model_fitting/a_mle/                      (new — absorbs scripts/12_fit_mle.py + scripts/fitting/fit_mle.py)
  - scripts/04_model_fitting/b_bayesian/                 (new — absorbs scripts/13_fit_bayesian.py + scripts/fitting/fit_bayesian.py + 21_fit_baseline.py)
  - scripts/04_model_fitting/c_level2/                   (new — absorbs bayesian_pipeline/21_fit_with_l2.py)
  - scripts/05_post_fitting_checks/                      (new — absorbs bayesian_pipeline/21_{baseline_audit,scale_audit}.py)
  - scripts/06_fit_analyses/                             (new — absorbs 14_compare_models.py + post_mle/15-18 + bayesian_pipeline/21_{compute_loo_stacking,model_averaging,manuscript_tables}.py)
  - scripts/data_processing/                             (deleted)
  - scripts/behavioral/                                  (deleted)
  - scripts/simulations_recovery/                        (deleted)
  - scripts/post_mle/                                    (deleted)
  - scripts/bayesian_pipeline/                           (deleted)
  - scripts/12_fit_mle.py                                (deleted — moved to 04/a_mle)
  - scripts/13_fit_bayesian.py                           (deleted — moved to 04/b_bayesian)
  - scripts/14_compare_models.py                         (deleted — moved to 06)
  - scripts/fitting/fit_mle.py                           (moved to 04/a_mle)
  - scripts/fitting/fit_bayesian.py                      (moved to 04/b_bayesian)
  - scripts/**/*.py                                      (internal importer updates across moved tree)
  - validation/compare_posterior_to_mle.py               (docstring references updated)
autonomous: true

must_haves:
  truths:
    - "scripts/ top level contains only: 01_data_preprocessing/, 02_behav_analyses/, 03_model_prefitting/, 04_model_fitting/, 05_post_fitting_checks/, 06_fit_analyses/, utils/, fitting/, __pycache__ (kept), plus analysis|results|simulations|statistical_analyses|visualization (handled in 29-04, not this plan)"  # SC#1, SC#2
    - "Every script moved via git mv (history preserved — git log --follow works)"  # SC#10
    - "Zero callers import from old Phase-28 grouping paths (scripts.data_processing|behavioral|simulations_recovery|post_mle|bayesian_pipeline)"  # SC#10
    - "v4 closure guard tests still pass after the rename (no closure paths broken)"  # SC#9
  artifacts:
    - path: "scripts/01_data_preprocessing/01_parse_raw_data.py"
      provides: "parse raw jsPsych JSON (moved from data_processing/)"
    - path: "scripts/02_behav_analyses/05_summarize_behavioral_data.py"
      provides: "behavioral summary (moved from behavioral/)"
    - path: "scripts/03_model_prefitting/12_run_prior_predictive.py"
      provides: "prior-predictive check (renumbered from 21_run_prior_predictive.py)"
    - path: "scripts/04_model_fitting/a_mle/12_fit_mle.py"
      provides: "MLE entry point (moved from scripts/12_fit_mle.py)"
    - path: "scripts/04_model_fitting/b_bayesian/21_fit_baseline.py"
      provides: "Bayesian baseline fit (moved from bayesian_pipeline/)"
    - path: "scripts/04_model_fitting/c_level2/21_fit_with_l2.py"
      provides: "Level-2 hierarchical fit (moved from bayesian_pipeline/)"
    - path: "scripts/05_post_fitting_checks/baseline_audit.py"
      provides: "Rhat/ESS/divergences audit (moved + renamed from 21_baseline_audit.py)"
    - path: "scripts/06_fit_analyses/compare_models.py"
      provides: "model comparison (moved from scripts/14_compare_models.py)"
  key_links:
    - from: "scripts/06_fit_analyses/manuscript_tables.py"
      to: "scripts/06_fit_analyses/bayesian_level2_effects.py"
      via: "subprocess.run(['python', 'scripts/06_fit_analyses/bayesian_level2_effects.py', ...])"
      pattern: "scripts/06_fit_analyses/bayesian_level2_effects\\.py"
    - from: "scripts/**/*.py"
      to: "rlwm.fitting"
      via: "from rlwm.fitting.{jax_likelihoods,numpyro_models,numpyro_helpers} import ..."
      pattern: "from rlwm\\.fitting\\."
---

<objective>
Execute the big rename wave: transition Phase 28's five-subdir grouping to the canonical paper-directional 01–06 stage layout. Every currently-grouped folder (data_processing/, behavioral/, simulations_recovery/, post_mle/, bayesian_pipeline/) dissolves and its contents migrate (via `git mv`) into the new numbered stage dirs. Top-level entry scripts (12/13/14) and their `scripts/fitting/` implementations fold into `04_model_fitting/{a_mle,b_bayesian,c_level2}/`. Every Python importer across `scripts/`, `validation/`, `tests/`, `manuscript/` is updated in the same commit so the repo never rests in a broken state.

Purpose: Stabilize pipeline paths BEFORE Phase 24 cold-start and Phase 26 manuscript finalization. Paper.qmd Quarto cells, cluster SLURM jobs, and closure guards all ride on these paths — they MUST settle first.

Output: 6 new stage directories populated with moved files + all importers pointing at new paths.
</objective>

<execution_context>
@C:\Users\aman0087\.claude\get-shit-done\workflows\execute-plan.md
@C:\Users\aman0087\.claude\get-shit-done\templates\summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/29-pipeline-canonical-reorg/29-CONTEXT.md
@.planning/phases/28-bayesian-first-restructure-repo-cleanup/28-01-src-consolidation-SUMMARY.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Stage-directory rename — 01, 02 (low-churn; no subprocess coupling)</name>
  <files>
    - scripts/01_data_preprocessing/ (new)
    - scripts/02_behav_analyses/ (new)
    - scripts/data_processing/ (deleted)
    - scripts/behavioral/ (deleted)
  </files>
  <action>
    1. `git mv scripts/data_processing scripts/01_data_preprocessing`
    2. `git mv scripts/behavioral scripts/02_behav_analyses`
    3. Preserve numeric prefixes on every moved file (01_parse_raw_data.py .. 08_run_statistical_analyses.py).
    4. Grep the entire repo EXCLUDING `.planning/` for old paths and update every live reference:
       - `grep -rn "scripts/data_processing\|from scripts.data_processing" . --include="*.py" --include="*.sh" --include="*.slurm" --include="*.md" --include="*.qmd" --include="*.tex" --exclude-dir=.planning --exclude-dir=.git`
       - `grep -rn "scripts/behavioral\|from scripts.behavioral" . --include="*.py" --include="*.sh" --include="*.slurm" --include="*.md" --include="*.qmd" --include="*.tex" --exclude-dir=.planning --exclude-dir=.git`
       - For every hit: rewrite path to the new stage folder. Internal cross-imports (if any — `08_run_statistical_analyses.py` imports `scripts.utils.statistical_tests`) remain untouched since only the folder name changed, not utils.
    5. Known live referrers: `cluster/13_full_pipeline.slurm` (4 hits for `scripts/behavioral/`), `CLAUDE.md` (Quick Reference block), `README.md` (if present), `docs/**/*.md` (a few), `manuscript/paper.qmd` (runtime refs).
    6. Do NOT touch `.planning/` at all — it holds historical plans.
    7. Intermediate `scripts/data_processing/__init__.py` and `scripts/behavioral/__init__.py` move with the folder; verify `git status` shows them under the new path.
  </action>
  <verify>
    - `test -d scripts/01_data_preprocessing && test -d scripts/02_behav_analyses`
    - `test ! -d scripts/data_processing && test ! -d scripts/behavioral`
    - `grep -rn "scripts/data_processing\|from scripts.data_processing" scripts/ tests/ validation/ cluster/ manuscript/ docs/ src/ --include="*.py" --include="*.sh" --include="*.slurm" --include="*.md" --include="*.qmd" --include="*.tex"` returns ZERO matches
    - `grep -rn "scripts/behavioral\|from scripts.behavioral" scripts/ tests/ validation/ cluster/ manuscript/ docs/ src/ --include="*.py" --include="*.sh" --include="*.slurm" --include="*.md" --include="*.qmd" --include="*.tex"` returns ZERO matches
    - `git log --follow --oneline scripts/01_data_preprocessing/01_parse_raw_data.py | head -3` shows history
  </verify>
  <done>Stage 01 and 02 renamed; all live referrers updated; v4 closure (`pytest scripts/fitting/tests/test_v4_closure.py -v`) still PASSES 3/3.</done>
</task>

<task type="auto">
  <name>Task 2: Build stage 03 (pre-fitting) — merge simulations_recovery/ + prior-predictive + recovery from bayesian_pipeline/</name>
  <files>
    - scripts/03_model_prefitting/09_generate_synthetic_data.py  (from simulations_recovery/)
    - scripts/03_model_prefitting/09_run_ppc.py                  (from simulations_recovery/)
    - scripts/03_model_prefitting/10_run_parameter_sweep.py      (from simulations_recovery/)
    - scripts/03_model_prefitting/11_run_model_recovery.py       (from simulations_recovery/)
    - scripts/03_model_prefitting/12_run_prior_predictive.py     (git-mv-renamed from bayesian_pipeline/21_run_prior_predictive.py)
    - scripts/03_model_prefitting/13_run_bayesian_recovery.py    (git-mv-renamed from bayesian_pipeline/21_run_bayesian_recovery.py)
    - scripts/03_model_prefitting/__init__.py                    (new)
    - scripts/simulations_recovery/                              (deleted)
  </files>
  <action>
    1. `mkdir -p scripts/03_model_prefitting` and create empty `__init__.py`.
    2. Move the 4 simulations_recovery files: `git mv scripts/simulations_recovery/09_generate_synthetic_data.py scripts/03_model_prefitting/09_generate_synthetic_data.py` and equivalents for `09_run_ppc.py`, `10_run_parameter_sweep.py`, `11_run_model_recovery.py`.
    3. `git mv scripts/simulations_recovery/__init__.py scripts/03_model_prefitting/__init__.py` (overwriting the empty one you just created, OR skip step 1 and let the git mv create the folder).
    4. Delete the now-empty `scripts/simulations_recovery/` directory (`rmdir` or `git rm -r` if git still shows it).
    5. RENAME prior-predictive and recovery files to align with stage 03 numbering:
       - `git mv scripts/bayesian_pipeline/21_run_prior_predictive.py scripts/03_model_prefitting/12_run_prior_predictive.py`
       - `git mv scripts/bayesian_pipeline/21_run_bayesian_recovery.py scripts/03_model_prefitting/13_run_bayesian_recovery.py`
       (The `21_` prefix came from the Bayesian pipeline ordering; these scripts are conceptually pre-fitting simulation work and belong in stage 03.)
    6. Grep-update all importers and invokers:
       - `grep -rn "scripts/simulations_recovery\|from scripts.simulations_recovery" . --include="*.py" --include="*.sh" --include="*.slurm" --include="*.md" --include="*.qmd" --include="*.tex" --exclude-dir=.planning --exclude-dir=.git`
       - `grep -rn "scripts/bayesian_pipeline/21_run_prior_predictive\|scripts/bayesian_pipeline/21_run_bayesian_recovery\|from scripts.bayesian_pipeline.21_run_prior_predictive\|from scripts.bayesian_pipeline.21_run_bayesian_recovery" . --include="*.py" --include="*.sh" --include="*.slurm" --include="*.md" --include="*.qmd" --include="*.tex" --exclude-dir=.planning --exclude-dir=.git`
       - Known referrers: `cluster/09_ppc_gpu.slurm`, `cluster/11_recovery_gpu.slurm`, `cluster/21_1_prior_predictive.slurm`, `cluster/21_2_recovery.slurm`, `cluster/21_2_recovery_aggregate.slurm`, `cluster/21_submit_pipeline.sh`, `CLAUDE.md` Quick Reference block, `docs/04_methods/README.md`.
       - Rewrite each to the new `scripts/03_model_prefitting/NN_*.py` path.
    7. `scripts/03_model_prefitting/09_generate_synthetic_data.py` line 48 and `scripts/03_model_prefitting/10_run_parameter_sweep.py` line 57 import `from scripts.simulations.{generate_data,parameter_sweep} import main` — LEAVE those imports alone. `scripts/simulations/` is handled by 29-04 (dead-folder audit), not this plan. If 29-04 later decides to delete `scripts/simulations/`, these wrapper imports get rewritten at that time.
  </action>
  <verify>
    - `test -d scripts/03_model_prefitting && test -f scripts/03_model_prefitting/12_run_prior_predictive.py && test -f scripts/03_model_prefitting/13_run_bayesian_recovery.py`
    - `test ! -d scripts/simulations_recovery`
    - `test ! -f scripts/bayesian_pipeline/21_run_prior_predictive.py && test ! -f scripts/bayesian_pipeline/21_run_bayesian_recovery.py`
    - `grep -rn "scripts/simulations_recovery\|from scripts.simulations_recovery" scripts/ tests/ validation/ cluster/ manuscript/ docs/ src/ --include="*.py" --include="*.sh" --include="*.slurm" --include="*.md" --include="*.qmd" --include="*.tex"` returns ZERO
    - `git log --follow --oneline scripts/03_model_prefitting/12_run_prior_predictive.py | head -3` shows the original `21_run_prior_predictive.py` history
  </verify>
  <done>Stage 03 populated with 6 scripts (09, 09_ppc, 10, 11, 12, 13); bayesian_pipeline retains the remaining 7 files for Task 4 pickup; simulations_recovery gone; all importers updated.</done>
</task>

<task type="auto">
  <name>Task 3: Build stage 04 (model_fitting/{a_mle,b_bayesian,c_level2}) — fold top-level 12/13 entries + fitting/ fits + 21_fit_baseline + 21_fit_with_l2</name>
  <files>
    - scripts/04_model_fitting/__init__.py                          (new)
    - scripts/04_model_fitting/a_mle/__init__.py                    (new)
    - scripts/04_model_fitting/a_mle/12_fit_mle.py                  (from scripts/12_fit_mle.py)
    - scripts/04_model_fitting/a_mle/fit_mle.py                     (from scripts/fitting/fit_mle.py — implementation)
    - scripts/04_model_fitting/b_bayesian/__init__.py               (new)
    - scripts/04_model_fitting/b_bayesian/13_fit_bayesian.py        (from scripts/13_fit_bayesian.py)
    - scripts/04_model_fitting/b_bayesian/fit_bayesian.py           (from scripts/fitting/fit_bayesian.py — implementation)
    - scripts/04_model_fitting/b_bayesian/21_fit_baseline.py        (from scripts/bayesian_pipeline/21_fit_baseline.py)
    - scripts/04_model_fitting/c_level2/__init__.py                 (new)
    - scripts/04_model_fitting/c_level2/21_fit_with_l2.py           (from scripts/bayesian_pipeline/21_fit_with_l2.py)
  </files>
  <action>
    1. Create stage directory structure: `mkdir -p scripts/04_model_fitting/{a_mle,b_bayesian,c_level2}` and empty `__init__.py` in each.
    2. Move MLE entry + implementation into a_mle:
       - `git mv scripts/12_fit_mle.py scripts/04_model_fitting/a_mle/12_fit_mle.py`
       - `git mv scripts/fitting/fit_mle.py scripts/04_model_fitting/a_mle/fit_mle.py`
    3. Move Bayesian entry + implementation + baseline into b_bayesian:
       - `git mv scripts/13_fit_bayesian.py scripts/04_model_fitting/b_bayesian/13_fit_bayesian.py`
       - `git mv scripts/fitting/fit_bayesian.py scripts/04_model_fitting/b_bayesian/fit_bayesian.py`
       - `git mv scripts/bayesian_pipeline/21_fit_baseline.py scripts/04_model_fitting/b_bayesian/21_fit_baseline.py`
    4. Move Level-2 fit into c_level2:
       - `git mv scripts/bayesian_pipeline/21_fit_with_l2.py scripts/04_model_fitting/c_level2/21_fit_with_l2.py`
    5. Update `scripts/04_model_fitting/a_mle/12_fit_mle.py` internal import `from scripts.fitting.fit_mle import main` → `from scripts.04_model_fitting.a_mle.fit_mle import main`. Python package names can't start with a digit for `from` imports, so if the dotted-name approach fails at runtime, use a relative import inside the entry script: `from .fit_mle import main` (the entry script is inside the same package), or route via the package's `__init__.py`. PREFER relative imports for intra-stage-package references.
    6. Same treatment for `scripts/04_model_fitting/b_bayesian/13_fit_bayesian.py`: rewrite `from scripts.fitting.fit_bayesian import main` → `from .fit_bayesian import main`.
    7. Grep-update all EXTERNAL callers of the moved files:
       - `grep -rn "scripts/12_fit_mle\|scripts/13_fit_bayesian\|scripts/fitting/fit_mle\|scripts/fitting/fit_bayesian\|scripts/bayesian_pipeline/21_fit_baseline\|scripts/bayesian_pipeline/21_fit_with_l2" . --include="*.py" --include="*.sh" --include="*.slurm" --include="*.md" --include="*.qmd" --include="*.tex" --exclude-dir=.planning --exclude-dir=.git`
       - `grep -rn "from scripts.fitting.fit_mle\|from scripts.fitting.fit_bayesian\|import scripts.fitting.fit_mle\|import scripts.fitting.fit_bayesian" . --include="*.py" --exclude-dir=.planning --exclude-dir=.git`
       - Known referrers: `cluster/12_mle.slurm`, `cluster/12_mle_gpu.slurm`, `cluster/12_mle_single.slurm`, `cluster/12_submit_all.sh`, `cluster/12_submit_all_gpu.sh`, `cluster/13_bayesian_choice_only.slurm`, `cluster/13_bayesian_gpu.slurm`, `cluster/13_bayesian_m4_gpu.slurm`, `cluster/13_bayesian_m6b_subscale.slurm`, `cluster/13_bayesian_multigpu.slurm`, `cluster/13_bayesian_permutation.slurm`, `cluster/13_bayesian_pscan.slurm`, `cluster/13_bayesian_pscan_smoke.slurm`, `cluster/13_bayesian_fullybatched_smoke.slurm`, `cluster/13_full_pipeline.slurm`, `cluster/21_3_fit_baseline.slurm`, `cluster/21_6_fit_with_l2.slurm`, `cluster/21_submit_pipeline.sh`, `cluster/21_dispatch_l2_winners.sh`, `CLAUDE.md`, `README.md`, `validation/compare_posterior_to_mle.py` docstrings (lines 10, 12), `manuscript/paper.qmd` (2 hits for `scripts/14_compare_models.py`, separate from fitting but same wave of path updates — handled in Task 5 below).
       - Rewrite each caller to reference the new path.
    8. Do NOT remove `scripts/fitting/` yet; it still holds bms.py, compare_mle_models.py, mle_utils.py, lba_likelihood.py, level2_design.py, bayesian_diagnostics.py, bayesian_summary_writer.py, model_recovery.py, warmup_jit.py, aggregate_permutation_results.py, and tests/. Plan 29-03 utils consolidation decides the fate of each.
  </action>
  <verify>
    - `test -d scripts/04_model_fitting/a_mle && test -d scripts/04_model_fitting/b_bayesian && test -d scripts/04_model_fitting/c_level2`
    - `test -f scripts/04_model_fitting/a_mle/12_fit_mle.py && test -f scripts/04_model_fitting/a_mle/fit_mle.py`
    - `test -f scripts/04_model_fitting/b_bayesian/13_fit_bayesian.py && test -f scripts/04_model_fitting/b_bayesian/fit_bayesian.py && test -f scripts/04_model_fitting/b_bayesian/21_fit_baseline.py`
    - `test -f scripts/04_model_fitting/c_level2/21_fit_with_l2.py`
    - `test ! -f scripts/12_fit_mle.py && test ! -f scripts/13_fit_bayesian.py && test ! -f scripts/fitting/fit_mle.py && test ! -f scripts/fitting/fit_bayesian.py`
    - `test ! -f scripts/bayesian_pipeline/21_fit_baseline.py && test ! -f scripts/bayesian_pipeline/21_fit_with_l2.py`
    - `grep -rn "scripts/12_fit_mle\|scripts/13_fit_bayesian\|scripts/fitting/fit_mle\|scripts/fitting/fit_bayesian\|scripts/bayesian_pipeline/21_fit_baseline\|scripts/bayesian_pipeline/21_fit_with_l2" scripts/ tests/ validation/ cluster/ manuscript/ docs/ src/ --include="*.py" --include="*.sh" --include="*.slurm" --include="*.md" --include="*.qmd" --include="*.tex"` returns ZERO
    - `python -c "import sys; sys.path.insert(0, 'scripts/04_model_fitting/a_mle'); from fit_mle import main; print('ok')"` prints `ok` (basic-sanity import check; the entry script's relative import style MUST work)
  </verify>
  <done>Stage 04 populated with 3 sub-stages; MLE/Bayesian/Level-2 entry scripts + implementations all under canonical paths; zero external callers point at old locations.</done>
</task>

<task type="auto">
  <name>Task 4: Build stages 05 and 06 — dissolve bayesian_pipeline/ + post_mle/ + top-level 14_compare_models.py</name>
  <files>
    - scripts/05_post_fitting_checks/__init__.py                        (new)
    - scripts/05_post_fitting_checks/baseline_audit.py                  (git-mv-renamed from bayesian_pipeline/21_baseline_audit.py, drops `21_` prefix since 05 reorders)
    - scripts/05_post_fitting_checks/scale_audit.py                     (from bayesian_pipeline/21_scale_audit.py)
    - scripts/06_fit_analyses/__init__.py                               (new)
    - scripts/06_fit_analyses/compare_models.py                         (from scripts/14_compare_models.py)
    - scripts/06_fit_analyses/compute_loo_stacking.py                   (from bayesian_pipeline/21_compute_loo_stacking.py)
    - scripts/06_fit_analyses/model_averaging.py                        (from bayesian_pipeline/21_model_averaging.py)
    - scripts/06_fit_analyses/manuscript_tables.py                      (from bayesian_pipeline/21_manuscript_tables.py)
    - scripts/06_fit_analyses/analyze_mle_by_trauma.py                  (from post_mle/15_analyze_mle_by_trauma.py)
    - scripts/06_fit_analyses/regress_parameters_on_scales.py           (from post_mle/16_regress_parameters_on_scales.py)
    - scripts/06_fit_analyses/analyze_winner_heterogeneity.py           (from post_mle/17_analyze_winner_heterogeneity.py)
    - scripts/06_fit_analyses/bayesian_level2_effects.py                (from post_mle/18_bayesian_level2_effects.py)
    - scripts/bayesian_pipeline/                                        (deleted after all moves)
    - scripts/post_mle/                                                 (deleted after all moves)
  </files>
  <action>
    1. Create directories: `mkdir -p scripts/05_post_fitting_checks scripts/06_fit_analyses` and empty `__init__.py` in each.
    2. Populate stage 05 (2 files from bayesian_pipeline/):
       - `git mv scripts/bayesian_pipeline/21_baseline_audit.py scripts/05_post_fitting_checks/baseline_audit.py`
       - `git mv scripts/bayesian_pipeline/21_scale_audit.py scripts/05_post_fitting_checks/scale_audit.py`
       Note: both scripts drop the `21_` prefix — stage 05 starts its own numbering-less namespace (these are the only two files here for now; a third posterior-PPC script lands here in 29-03 utils consolidation).
    3. Populate stage 06 (8 files):
       - `git mv scripts/14_compare_models.py scripts/06_fit_analyses/compare_models.py`
       - `git mv scripts/bayesian_pipeline/21_compute_loo_stacking.py scripts/06_fit_analyses/compute_loo_stacking.py`
       - `git mv scripts/bayesian_pipeline/21_model_averaging.py scripts/06_fit_analyses/model_averaging.py`
       - `git mv scripts/bayesian_pipeline/21_manuscript_tables.py scripts/06_fit_analyses/manuscript_tables.py`
       - `git mv scripts/post_mle/15_analyze_mle_by_trauma.py scripts/06_fit_analyses/analyze_mle_by_trauma.py` (drops `15_` — stage 06 renumbers internally or drops numbers entirely)
       - `git mv scripts/post_mle/16_regress_parameters_on_scales.py scripts/06_fit_analyses/regress_parameters_on_scales.py`
       - `git mv scripts/post_mle/17_analyze_winner_heterogeneity.py scripts/06_fit_analyses/analyze_winner_heterogeneity.py`
       - `git mv scripts/post_mle/18_bayesian_level2_effects.py scripts/06_fit_analyses/bayesian_level2_effects.py`
    4. Delete now-empty `scripts/bayesian_pipeline/` and `scripts/post_mle/` directories. `bayesian_pipeline/__init__.py` and `post_mle/__init__.py` should `git mv` into the new locations or simply get dropped (empty `__init__.py` recreation in the new stage folders covers them).
    5. CRITICAL UPDATE — subprocess reference in manuscript_tables.py: the moved `scripts/06_fit_analyses/manuscript_tables.py` line ~746 contains `subprocess.run(["python", "scripts/18_bayesian_level2_effects.py", ...])` (per REFAC-06 history — it was updated from `scripts/18_*` → `scripts/post_mle/18_*` in Phase 28). Update it AGAIN to `scripts/06_fit_analyses/bayesian_level2_effects.py`. Use Grep to find the line: `grep -n "18_bayesian_level2_effects\|bayesian_level2_effects" scripts/06_fit_analyses/manuscript_tables.py`.
    6. Grep-update all other callers:
       - `grep -rn "scripts/14_compare_models\|scripts/bayesian_pipeline\|scripts/post_mle\|from scripts.bayesian_pipeline\|from scripts.post_mle" . --include="*.py" --include="*.sh" --include="*.slurm" --include="*.md" --include="*.qmd" --include="*.tex" --exclude-dir=.planning --exclude-dir=.git`
       - Known referrers: `cluster/14_analysis.slurm`, `cluster/21_{4,5,7,8,9}_*.slurm`, `cluster/21_submit_pipeline.sh`, `cluster/21_dispatch_l2_winners.sh`, `CLAUDE.md`, `README.md`, `manuscript/paper.qmd` (2 hits on lines 630, 650 for `scripts/14_compare_models.py`).
       - Rewrite each to new path. Preserve command-line argument structure.
    7. paper.qmd update (2 hits): `scripts/14_compare_models.py` → `scripts/06_fit_analyses/compare_models.py` on lines 630 and 650. Also check for any reference to `scripts/18_bayesian_level2_effects.py` / `scripts/post_mle/18_bayesian_level2_effects.py` and update to `scripts/06_fit_analyses/bayesian_level2_effects.py`. Grep: `grep -n "scripts/" manuscript/paper.qmd`.
  </action>
  <verify>
    - `test -d scripts/05_post_fitting_checks && test -d scripts/06_fit_analyses`
    - `test -f scripts/05_post_fitting_checks/baseline_audit.py && test -f scripts/05_post_fitting_checks/scale_audit.py`
    - `test -f scripts/06_fit_analyses/compare_models.py && test -f scripts/06_fit_analyses/manuscript_tables.py && test -f scripts/06_fit_analyses/bayesian_level2_effects.py`
    - `test ! -d scripts/bayesian_pipeline && test ! -d scripts/post_mle && test ! -f scripts/14_compare_models.py`
    - `grep -rn "scripts/14_compare_models\|scripts/bayesian_pipeline\|scripts/post_mle\|from scripts.bayesian_pipeline\|from scripts.post_mle" scripts/ tests/ validation/ cluster/ manuscript/ docs/ src/ --include="*.py" --include="*.sh" --include="*.slurm" --include="*.md" --include="*.qmd" --include="*.tex"` returns ZERO
    - `grep -n "subprocess" scripts/06_fit_analyses/manuscript_tables.py | grep "bayesian_level2_effects"` shows the subprocess path = `scripts/06_fit_analyses/bayesian_level2_effects.py` (no stale `post_mle/` or bare `scripts/18_*`)
    - `pytest scripts/fitting/tests/test_v4_closure.py -v` still PASSES 3/3
  </verify>
  <done>Stages 05 and 06 populated; bayesian_pipeline/ and post_mle/ deleted; 14_compare_models.py moved; manuscript_tables.py subprocess path updated; paper.qmd 14_compare_models refs updated.</done>
</task>

<task type="auto">
  <name>Task 5: Full-tree grep sweep + atomic commit</name>
  <files>
    - (verification task — touches no files directly)
  </files>
  <action>
    1. Final sweep for any remaining old paths (excluding `.planning/` which is historical):
       ```
       grep -rn \
         "scripts/data_processing\|scripts/behavioral\|scripts/simulations_recovery\|scripts/post_mle\|scripts/bayesian_pipeline\|scripts/12_fit_mle\|scripts/13_fit_bayesian\|scripts/14_compare_models\|scripts/fitting/fit_mle\|scripts/fitting/fit_bayesian\|from scripts\.data_processing\|from scripts\.behavioral\|from scripts\.simulations_recovery\|from scripts\.post_mle\|from scripts\.bayesian_pipeline" \
         scripts/ tests/ validation/ cluster/ manuscript/ docs/ src/ \
         --include="*.py" --include="*.sh" --include="*.slurm" --include="*.md" --include="*.qmd" --include="*.tex"
       ```
       Expected: ZERO matches.
    2. Fix any stragglers by walking each hit and rewriting the path.
    3. Re-verify `validation/check_v4_closure.py --milestone v4.0` exits 0.
    4. Re-verify `pytest scripts/fitting/tests/test_v4_closure.py scripts/fitting/tests/test_load_side_validation.py -v` PASSES.
    5. Atomic commit (ONE commit for the whole rename wave): 
       ```
       refactor(29-01): canonical 01–06 stage layout for scripts/

       Moved via git mv to preserve history:
         - scripts/data_processing/ → scripts/01_data_preprocessing/
         - scripts/behavioral/ → scripts/02_behav_analyses/
         - scripts/simulations_recovery/ → scripts/03_model_prefitting/{09..11}
         - scripts/bayesian_pipeline/21_run_prior_predictive.py → scripts/03_model_prefitting/12_run_prior_predictive.py
         - scripts/bayesian_pipeline/21_run_bayesian_recovery.py → scripts/03_model_prefitting/13_run_bayesian_recovery.py
         - scripts/12_fit_mle.py + scripts/fitting/fit_mle.py → scripts/04_model_fitting/a_mle/
         - scripts/13_fit_bayesian.py + scripts/fitting/fit_bayesian.py + bayesian_pipeline/21_fit_baseline.py → scripts/04_model_fitting/b_bayesian/
         - scripts/bayesian_pipeline/21_fit_with_l2.py → scripts/04_model_fitting/c_level2/
         - scripts/bayesian_pipeline/21_{baseline,scale}_audit.py → scripts/05_post_fitting_checks/
         - scripts/14_compare_models.py + scripts/bayesian_pipeline/21_{compute_loo_stacking,model_averaging,manuscript_tables}.py + scripts/post_mle/{15..18}*.py → scripts/06_fit_analyses/

       Importer updates: all hits under scripts/, tests/, validation/, cluster/, manuscript/, docs/, src/ rewritten.
       subprocess call in manuscript_tables.py updated to 06_fit_analyses/bayesian_level2_effects.py.
       paper.qmd lines 630, 650 updated to 06_fit_analyses/compare_models.py.

       v4 closure guard (validation/check_v4_closure.py + pytest test_v4_closure.py) still PASSES.
       scripts/fitting/ retains library-only remnants (mle_utils, bms, bayesian_diagnostics, bayesian_summary_writer, lba_likelihood, level2_design, model_recovery, warmup_jit, aggregate_permutation_results, compare_mle_models) — 29-03 utils consolidation handles those.
       scripts/{analysis,results,simulations,statistical_analyses,visualization}/ untouched — 29-04 dead-folder audit handles those.
       ```
    6. `git status` after commit: clean.
    7. IF there is a strong reason to split the commit (>2000 files changed, git hooks complain, etc.), split along stage boundaries: one commit per stage (01→02→03→04→05→06) — whichever reduces the review surface while keeping each commit in a compilable/testable state.
  </action>
  <verify>
    - `git log -1 --stat` shows a single "refactor(29-01)" commit (or small sequence if split along stage boundaries)
    - `git status` is clean
    - Final full-tree grep (as in step 1) returns ZERO
    - `validation/check_v4_closure.py --milestone v4.0` exits 0
    - `pytest scripts/fitting/tests/test_v4_closure.py scripts/fitting/tests/test_load_side_validation.py -v` PASSES
  </verify>
  <done>All old-path references eliminated from active codebase (.planning/ exempted); commit(s) landed; v4 closure intact; downstream plans (29-02 through 29-07) unblocked.</done>
</task>

</tasks>

<verification>
```bash
# Stage directories exist
for d in 01_data_preprocessing 02_behav_analyses 03_model_prefitting 04_model_fitting 05_post_fitting_checks 06_fit_analyses; do
  test -d scripts/$d || { echo "MISSING: scripts/$d"; exit 1; }
done
# 04 sub-stages
for d in a_mle b_bayesian c_level2; do
  test -d scripts/04_model_fitting/$d || { echo "MISSING: scripts/04_model_fitting/$d"; exit 1; }
done

# Old groupings gone
for d in data_processing behavioral simulations_recovery post_mle bayesian_pipeline; do
  test ! -d scripts/$d || { echo "STILL EXISTS: scripts/$d"; exit 1; }
done

# Old top-level scripts gone
for f in 12_fit_mle.py 13_fit_bayesian.py 14_compare_models.py; do
  test ! -f scripts/$f || { echo "STILL EXISTS: scripts/$f"; exit 1; }
done

# Zero live imports of old grouping paths
grep -rn \
  "from scripts\.data_processing\|from scripts\.behavioral\|from scripts\.simulations_recovery\|from scripts\.post_mle\|from scripts\.bayesian_pipeline" \
  scripts/ tests/ validation/ cluster/ manuscript/ docs/ src/ \
  --include="*.py" --include="*.sh" --include="*.slurm" --include="*.md" --include="*.qmd" --include="*.tex" \
  || echo "OK: zero matches"

# v4 closure still green
python validation/check_v4_closure.py --milestone v4.0
pytest scripts/fitting/tests/test_v4_closure.py -v
```
</verification>

<success_criteria>
1. Six stage directories (01–06) populated with the correct files, every move via `git mv` (history preserved).
2. All five old grouping folders (data_processing, behavioral, simulations_recovery, post_mle, bayesian_pipeline) gone from `scripts/` top level.
3. Three old top-level entry scripts (12_fit_mle.py, 13_fit_bayesian.py, 14_compare_models.py) gone.
4. `scripts/fitting/fit_mle.py` and `scripts/fitting/fit_bayesian.py` moved under 04_model_fitting; remaining library modules in `scripts/fitting/` untouched (29-03 handles them).
5. `scripts/analysis/, scripts/results/, scripts/simulations/, scripts/statistical_analyses/, scripts/visualization/` UNTOUCHED (29-04 handles).
6. Grep sweep shows zero `from scripts.{data_processing|behavioral|simulations_recovery|post_mle|bayesian_pipeline}.` imports outside `.planning/`.
7. `manuscript/paper.qmd` line 630, 650 updated to `scripts/06_fit_analyses/compare_models.py`.
8. `scripts/06_fit_analyses/manuscript_tables.py` subprocess call targets `scripts/06_fit_analyses/bayesian_level2_effects.py`.
9. `validation/check_v4_closure.py` exits 0.
10. `pytest scripts/fitting/tests/test_v4_closure.py` passes 3/3.
</success_criteria>

<output>
After completion, create `.planning/phases/29-pipeline-canonical-reorg/29-01-SUMMARY.md` with:
- Exact file moves performed (from → to)
- Count of importer updates per consumer directory (cluster/, scripts/, validation/, tests/, manuscript/, docs/)
- Commit SHA(s)
- Verification evidence (v4 closure output, pytest output)
- Any deviations from the plan (e.g., if commit was split along stage boundaries)
- Known remaining old paths in `.planning/` (documented as historical, intentionally not touched)
</output>
