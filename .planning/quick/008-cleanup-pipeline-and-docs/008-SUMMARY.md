---
phase: quick-008-cleanup-pipeline-and-docs
plan: 008
subsystem: infra
tags: [cleanup, docs, pipeline, figures, scale_distributions, legacy]

requires: []
provides:
  - "7 orphaned scripts deleted (04_1, 06_1, 07_1, 09_1, 13_fit_bayesian_m4, 16b_bayesian_regression, 18b_mle_vs_bayes_reliability)"
  - "docs/legacy/ with 3 superseded docs + archival index"
  - "docs/04_methods/README.md and docs/04_results/README.md scaffolding"
  - "ANALYSIS_PIPELINE.md covers all 7 models (M1-M6b + M4) and Stages 5b/5c"
  - "PLOTTING_REFERENCE.md covers all 11 scripts/visualization/* tools"
  - "scale_distributions.png path wired correctly in trauma_scale_distributions.py and paper.qmd"
affects:
  - "Phase 18 manuscript work (paper.qmd figure paths now correct)"
  - "Future pipeline runs (no dead scripts to invoke)"
  - "Onboarding (docs now accurate and navigable)"

tech-stack:
  added: []
  patterns:
    - "docs/legacy/ pattern: superseded docs archived with README.md explaining reason + replacement"
    - "CANONICAL_FIGURE_PATH constant in analysis scripts: single source of truth for paper-expected output path"

key-files:
  created:
    - docs/legacy/README.md
    - docs/04_methods/README.md
    - docs/04_results/README.md
  modified:
    - docs/README.md
    - docs/02_pipeline_guide/ANALYSIS_PIPELINE.md
    - docs/02_pipeline_guide/PLOTTING_REFERENCE.md
    - docs/CLUSTER_GPU_LESSONS.md
    - scripts/analysis/trauma_scale_distributions.py
    - manuscript/paper.qmd
    - run_data_pipeline.py
    - cluster/13_full_pipeline.slurm
    - cluster/13_bayesian_m4_gpu.slurm
    - scripts/fitting/tests/test_m4_integration.py

key-decisions:
  - "paper.qmd uses ../figures/ convention (relative to manuscript/): both line 215 and line 946 now use ../figures/... prefix"
  - "cluster/13_bayesian_m4_gpu.slurm retained but made fail-fast: the SLURM script itself is kept because it contains valid GPU+float64 setup boilerplate, but the python invocation replaced with a clear error pointing to Phase 17 as the replacement"
  - "CANONICAL_FIGURE_PATH added as module-level constant in trauma_scale_distributions.py: separates the canonical output path from FIGURES_DIR (used for secondary outputs), making the paper.qmd link easy to verify by grep"
  - "Dangling refs in 4 additional files cleaned up automatically (Rule 3): run_data_pipeline.py, cluster/13_full_pipeline.slurm, cluster/13_bayesian_m4_gpu.slurm, scripts/fitting/tests/test_m4_integration.py"

patterns-established:
  - "docs/legacy/README.md: each archived doc row includes reason-for-archival + replacement pointer"
  - "docs/04_methods/ and docs/04_results/: short scaffolding indexes, not full content — _TODO_ and _placeholder_ markers for future entries"

duration: 10min
completed: 2026-04-17
---

# Quick Task 008: Cleanup Pipeline and Docs Summary

**7 orphaned scripts pruned, 3 legacy docs archived, docs/ reorganized with 04_methods/04_results scaffolding, and scale_distributions.png path wired correctly between trauma_scale_distributions.py and paper.qmd**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-04-17T16:27:35Z
- **Completed:** 2026-04-17T16:37:46Z
- **Tasks:** 4
- **Files changed:** 23 (7 deleted, 3 renamed/moved, 4 new, 9 modified)

## Accomplishments

- Deleted 7 orphaned/superseded scripts with no remaining non-`.planning/` references
- Moved 3 superseded docs to `docs/legacy/` via `git mv` (history preserved) and created archival index
- Updated `docs/README.md`, `ANALYSIS_PIPELINE.md` (all 7 models + Stage 5b/5c), and `PLOTTING_REFERENCE.md` (all 11 visualization scripts)
- Scaffolded `docs/04_methods/README.md` and `docs/04_results/README.md` as short placeholder indexes
- Fixed `trauma_scale_distributions.py` to write `figures/scale_distributions.png` (canonical path); updated `paper.qmd` line 215 to `../figures/scale_distributions.png` consistent with line 946 convention

## Task Commits

All 4 tasks committed in a single atomic commit:

1. **All tasks (1-4)** - `2c4f12a` (chore(quick-008): cleanup pipeline scripts, reorganize docs, fix scale_distributions path)

## Files Created/Modified

**Deleted (7):**
- `scripts/04_1_explore_survey_data.py` - orphaned survey explorer
- `scripts/06_1_plot_task_performance.py` - superseded by 06_visualize_task_performance.py
- `scripts/07_1_visualize_by_trauma_group.py` - superseded by 07_analyze_trauma_groups.py
- `scripts/09_1_simulate_model_predictions.py` - orphaned simulation script
- `scripts/13_fit_bayesian_m4.py` - M4 hierarchical runner (model lives in numpyro_models.py; Phase 17 will provide a new runner)
- `scripts/16b_bayesian_regression.py` - PyMC-based (PyMC dropped in Phase 13; NumPyro-only backend)
- `scripts/18b_mle_vs_bayes_reliability.py` - orphaned reliability analysis

**Renamed/moved (3):**
- `docs/JAX_GPU_BAYESIAN_FITTING.md` → `docs/legacy/`
- `docs/CONVERGENCE_ASSESSMENT.md` → `docs/legacy/`
- `docs/DEER_NONLINEAR_PARALLELIZATION.md` → `docs/legacy/`

**Created (4):**
- `docs/legacy/README.md` - archival index (3 rows)
- `docs/04_methods/README.md` - methods index (published + validation)
- `docs/04_results/README.md` - pipeline results index (behavioral, MLE, trauma, Bayesian)

**Modified (9):**
- `docs/README.md` - tree updated; moved files removed, 04_methods/04_results added
- `docs/02_pipeline_guide/ANALYSIS_PIPELINE.md` - M5/M6a/M6b/M4 added to Stage 4; Stage 5b/5c added; deleted script refs removed
- `docs/02_pipeline_guide/PLOTTING_REFERENCE.md` - full table of all 11 visualization scripts added
- `docs/CLUSTER_GPU_LESSONS.md` - "See also" pointer updated to docs/legacy/JAX_GPU_BAYESIAN_FITTING.md
- `scripts/analysis/trauma_scale_distributions.py` - CANONICAL_FIGURE_PATH = figures/scale_distributions.png; savefig redirected
- `manuscript/paper.qmd` - line 215 and prose ref on line 224 updated to ../figures/scale_distributions.png
- `run_data_pipeline.py` - 06_1 and 07_1 steps removed
- `cluster/13_full_pipeline.slurm` - 06_1 and 07_1 run_step calls removed
- `cluster/13_bayesian_m4_gpu.slurm` - python invocation replaced with fail-fast error+message
- `scripts/fitting/tests/test_m4_integration.py` - docstring reference to deleted script removed

## Decisions Made

1. **paper.qmd `../figures/` convention adopted (line 215):** Line 215 previously used `figures/scale_distributions.png` (manuscript-local), inconsistent with line 946's `../figures/mle_trauma_analysis/...` (repo-root). Both now use the `../figures/` prefix pointing to the repo-root `figures/` directory.

2. **`cluster/13_bayesian_m4_gpu.slurm` kept as fail-fast stub:** Deleting the SLURM file would lose the carefully configured GPU+float64 setup boilerplate (env activation, JAX cache config, verification step). Retained with a clear error message and Phase 17 pointer instead.

3. **CANONICAL_FIGURE_PATH as a module-level constant:** Makes the canonical figure path greppable and independent of `FIGURES_DIR`. Secondary outputs (correlation_heatmap, exploratory_subscale_forest_plots) remain under `FIGURES_DIR = figures/trauma_scale_analysis/`.

4. **4 additional files cleaned up (deviation Rule 3 — blocking):** The plan's Task 1 deletion check revealed references in `run_data_pipeline.py`, `cluster/13_full_pipeline.slurm`, `cluster/13_bayesian_m4_gpu.slurm`, and `scripts/fitting/tests/test_m4_integration.py`. All cleaned to satisfy the "no dangling references" verification gate.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Cleaned dangling references in 4 additional files**

- **Found during:** Task 1 verification (`git grep` before deletion)
- **Issue:** Plan did not list `run_data_pipeline.py`, `cluster/13_full_pipeline.slurm`, `cluster/13_bayesian_m4_gpu.slurm`, or `scripts/fitting/tests/test_m4_integration.py` as files to modify, but the deletion check revealed each contained references to the scripts being deleted.
- **Fix:** Removed `06_1`/`07_1` steps from pipeline runners; replaced `13_fit_bayesian_m4.py` python call in SLURM with fail-fast error; removed docstring filename reference in test file; updated `docs/CLUSTER_GPU_LESSONS.md` "See also" pointer for moved doc.
- **Files modified:** See "Modified" list above
- **Verification:** `git grep` returned exit code 1 (no matches) for all deleted-script name patterns outside `.planning/`
- **Committed in:** `2c4f12a`

---

**Total deviations:** 1 auto-fixed (Rule 3 - blocking)
**Impact on plan:** Necessary to satisfy the plan's own verification gate. No scope creep — all changes are directly in service of the cleanup objective.

## Issues Encountered

None beyond the dangling references handled above.

## Next Phase Readiness

- Pipeline is clean: no orphaned scripts, no broken references, docs match current code
- `figures/scale_distributions.png` will be written to the correct location on next `python scripts/analysis/trauma_scale_distributions.py` run
- Phase 17 (M4 hierarchical LBA) needs a new runner script before `cluster/13_bayesian_m4_gpu.slurm` can be submitted again
- `docs/04_methods/` and `docs/04_results/` scaffolding is in place; populate with method writeups as reviewers ask for them or as cluster results arrive

---
*Task: quick-008*
*Completed: 2026-04-17*
