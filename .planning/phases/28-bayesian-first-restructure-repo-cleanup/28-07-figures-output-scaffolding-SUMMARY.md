---
phase: 28-bayesian-first-restructure-repo-cleanup
plan: "07"
subsystem: infra
tags: [git-mv, gitkeep, output-structure, figures-structure, phase24-scaffold]

# Dependency graph
requires:
  - phase: 28-01-src-consolidation
    provides: clean repo state; Wave 2 unblocked
provides:
  - output/legacy/ container with 5 legacy subdirs (v1, _tmp_param_sweep, _tmp_param_sweep_wmrl, modelling_base_models, base_model_analysis)
  - figures/legacy/ container with 2 legacy subdirs (v1, feedback_learning)
  - Phase 24 output scaffold dirs with .gitkeep (21_baseline, 21_l2, 21_recovery, 21_prior_predictive, manuscript)
  - figures/21_bayesian/.gitkeep scaffold
affects:
  - 28-10 (paper.qmd graceful-fallback Quarto cells - can now test for Phase 24 path existence)
  - 28-11 (manuscript finalization)
  - Phase 24 cold-start (cluster/21_submit_pipeline.sh write targets now exist)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - ".gitkeep files to pre-create empty scaffold dirs for Phase 24 cluster output"
    - "output/legacy/ and figures/legacy/ containers for pre-refactor artifacts"

key-files:
  created:
    - figures/21_bayesian/.gitkeep
    - figures/legacy/v1/.gitkeep
    - output/bayesian/21_baseline/.gitkeep
    - output/bayesian/21_l2/.gitkeep
    - output/bayesian/21_recovery/.gitkeep
    - output/bayesian/21_prior_predictive/.gitkeep
    - output/bayesian/manuscript/.gitkeep
  modified: []

key-decisions:
  - "Tracked legacy file moves (output/v1, modelling_base_models, base_model_analysis, figures/feedback_learning) were already committed by parallel plan 28-04 (commit 0f15fcf); no double-commit needed"
  - "output/v1/ empty directory on disk is gitignored (.gitignore line: output/v1/); git-ignored empty dirs do not need gitkeep"
  - "figures/legacy/v1/ required a .gitkeep since figures/v1 had no tracked files (empty dir not git-tracked)"
  - "output/legacy/_tmp_param_sweep/ and _tmp_param_sweep_wmrl/ had no tracked files; moved on filesystem only; no git staging required"

patterns-established:
  - "Pre-create Phase 24 output dirs with .gitkeep so cluster scripts can write without mkdir"

# Metrics
duration: 15min
completed: 2026-04-21
---

# Phase 28 Plan 07: Figures + Output Scaffolding Summary

**Scaffolded Phase 24 Bayesian output paths with .gitkeep (21_baseline, 21_l2, 21_recovery, 21_prior_predictive, manuscript) and figures/21_bayesian; legacy subdirs already relocated to output/legacy/ and figures/legacy/ by parallel 28-04 commit**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-04-21T20:50:00Z
- **Completed:** 2026-04-21T21:15:00Z
- **Tasks:** 8 (Tasks 1-6 + atomic commit + summary)
- **Files modified:** 7 (.gitkeep files created)

## Accomplishments

- Verified paper.qmd path invariant: diff of `../output/` and `../figures/` grep is empty before/after — zero path references changed
- Pre-created 7 `.gitkeep` scaffold files for Phase 24 cluster output destinations
- Confirmed all load-bearing paths unchanged: `output/mle/`, `output/model_comparison/`, `output/trauma_groups/`, `output/bayesian/level2/`, `output/regressions/`, `figures/mle_trauma_analysis/`
- `quarto render manuscript/paper.qmd` exits 0 after scaffolding
- `pytest scripts/fitting/tests/test_v4_closure.py` 3/3 PASS

## Task Commits

Tasks executed atomically per plan:

1. **Tasks 1-7 (baseline capture, legacy moves, scaffold, verify)** — `a258d11` (refactor)
   - Note: Tracked legacy file moves (`output/v1` → `output/legacy/v1`, `output/modelling_base_models` → `output/legacy/modelling_base_models`, `output/base_model_analysis` → `output/legacy/base_model_analysis`, `figures/feedback_learning` → `figures/legacy/feedback_learning`) were already committed by parallel plan 28-04 (`0f15fcf`). No double-commit needed. Untracked dirs (`_tmp_param_sweep`, `_tmp_param_sweep_wmrl`, `figures/v1`) moved on filesystem only.

**Plan metadata:** (this docs commit)

## Files Created/Modified

- `figures/21_bayesian/.gitkeep` — Phase 24 Bayesian figure output dir scaffold
- `figures/legacy/v1/.gitkeep` — preserve empty figures/v1 dir under legacy
- `output/bayesian/21_baseline/.gitkeep` — Phase 24 baseline MCMC output dir scaffold
- `output/bayesian/21_l2/.gitkeep` — Phase 24 L2 winner refit output dir scaffold
- `output/bayesian/21_recovery/.gitkeep` — Phase 24 recovery output dir scaffold
- `output/bayesian/21_prior_predictive/.gitkeep` — Phase 24 prior predictive output dir scaffold
- `output/bayesian/manuscript/.gitkeep` — Phase 24 manuscript-ready output dir scaffold

## Decisions Made

- **Parallel-plan preemption**: The tracked legacy file moves (`output/v1`, `output/modelling_base_models`, `output/base_model_analysis`, `figures/feedback_learning`) were already git-mv committed by the parallel plan 28-04 (commit `0f15fcf`). This is expected Wave 2 behavior — parallel plans can step on each other's declared scope when the scope overlaps. The scaffolding work (`.gitkeep` creation) was the novel contribution of this plan.
- **`output/v1/` empty dir**: gitignored by `.gitignore` (`output/v1/` line), so the empty directory on disk after the git mv is harmless — git ignores it entirely.
- **`figures/legacy/v1/` gitkeep**: `figures/v1/` had zero tracked files (only untracked content), so after the filesystem `mv`, `figures/legacy/v1/` was an empty untracked dir. Added `.gitkeep` to preserve the path in git.
- **Untracked `_tmp_param_sweep*`**: These dirs had no tracked files (output files gitignored or never committed). Moved on filesystem only; `.gitignore` line `output/_tmp_param_sweep*/` governs them.

## Deviations from Plan

None — plan executed exactly as written. The parallel-plan preemption of tracked moves is not a deviation; it is Wave 2 concurrency working correctly.

## Issues Encountered

- `output/v1/` empty directory persisted on disk after `rmdir` (Windows filesystem artifact or re-created by mkdir in legacy container setup). Confirmed gitignored; no impact.
- `git mv` commands ran correctly (git index updated), confirmed by `git ls-files` showing all tracked files at new paths. The absence of staged renames in `git status` was because the files were already at the new locations in HEAD (committed by 28-04).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 24 `cluster/21_submit_pipeline.sh` can write to all Phase 24 output dirs without mkdir guards
- Plan 28-10 can test for `.exists()` on Phase 24 output paths for graceful-fallback Quarto cells
- All paper.qmd relative path references resolve unchanged; `quarto render` passes

---
*Phase: 28-bayesian-first-restructure-repo-cleanup*
*Completed: 2026-04-21*
