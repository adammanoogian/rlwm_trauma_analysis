---
phase: 28-bayesian-first-restructure-repo-cleanup
plan: "06"
subsystem: infra
tags: [repo-restructure, slurm, bayesian-pipeline, git-mv, path-bootstrap]

# Dependency graph
requires:
  - phase: 28-05
    provides: subprocess path edit in 21_manuscript_tables.py to post_mle/18_bayesian_level2_effects.py
provides:
  - scripts/bayesian_pipeline/ directory with __init__.py and all 9 Phase 21 scripts
  - 10 SLURM files updated to invoke scripts/bayesian_pipeline/21_*.py
  - 4 stale cluster/13_bayesian_m6b.slurm pattern comments cleaned
affects:
  - 28-08 (grep invariant for deleted 13_bayesian_m*.slurm templates)
  - cluster/21_submit_pipeline.sh (now resolves via updated SLURM files)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Path bootstrap depth: scripts/bayesian_pipeline/ is 2 levels from project root, so _PROJECT_ROOT = Path(__file__).parent.parent.parent"

key-files:
  created:
    - scripts/bayesian_pipeline/__init__.py
    - scripts/bayesian_pipeline/21_run_prior_predictive.py (moved)
    - scripts/bayesian_pipeline/21_run_bayesian_recovery.py (moved)
    - scripts/bayesian_pipeline/21_fit_baseline.py (moved)
    - scripts/bayesian_pipeline/21_baseline_audit.py (moved)
    - scripts/bayesian_pipeline/21_compute_loo_stacking.py (moved)
    - scripts/bayesian_pipeline/21_fit_with_l2.py (moved)
    - scripts/bayesian_pipeline/21_scale_audit.py (moved)
    - scripts/bayesian_pipeline/21_model_averaging.py (moved)
    - scripts/bayesian_pipeline/21_manuscript_tables.py (moved)
  modified:
    - cluster/21_1_prior_predictive.slurm
    - cluster/21_2_recovery.slurm
    - cluster/21_2_recovery_aggregate.slurm
    - cluster/21_3_fit_baseline.slurm
    - cluster/21_4_baseline_audit.slurm
    - cluster/21_5_loo_stacking_bms.slurm
    - cluster/21_6_fit_with_l2.slurm
    - cluster/21_7_scale_audit.slurm
    - cluster/21_8_model_averaging.slurm
    - cluster/21_9_manuscript_tables.slurm
    - scripts/fitting/tests/test_load_side_validation.py

key-decisions:
  - "21_6_dispatch_l2.slurm has no direct python invocation (delegates to cluster/21_dispatch_l2_winners.sh); SLURM count is effectively 10 direct invocations even though 11 SLURM files exist for this phase"
  - "21_8_model_averaging.slurm retains operational references to 13_bayesian_m6b_subscale.slurm — these are functional code (not stale pattern comments) and were not cleaned in Task 4b"

patterns-established:
  - "Path bootstrap depth rule: scripts/<subdir>/ uses .parent.parent.parent to reach project root"

# Metrics
duration: 15min
completed: 2026-04-21
---

# Phase 28 Plan 06: Bayesian Pipeline Group Summary

**Nine Phase-21 Bayesian pipeline scripts moved into scripts/bayesian_pipeline/, 10 SLURM files updated, path bootstraps fixed to .parent.parent.parent depth, and 4 stale m6b pattern comments cleaned**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-04-21T00:00:00Z
- **Completed:** 2026-04-21T00:15:00Z
- **Tasks:** 8 (Tasks 1-7 + 4b)
- **Files modified:** 21

## Accomplishments
- All 9 `scripts/21_*.py` moved to `scripts/bayesian_pipeline/` via git mv with rename detection
- 10 SLURM files updated; `21_6_dispatch_l2.slurm` confirmed clean (delegates to shell wrapper)
- Path bootstrap in 8 scripts updated from `.parent.parent` to `.parent.parent.parent` (Rule 1 bug fix)
- 4 stale `13_bayesian_m6b.slurm` pattern comments rewritten to prevent Plan 28-08 grep false positives
- `test_load_side_validation.py` enumeration updated (5 paths); all 5/5 tests pass

## Task Commits

1. **Tasks 1-8 + 4b (atomic):** `6473f34` (refactor)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `scripts/bayesian_pipeline/__init__.py` - new empty package marker
- `scripts/bayesian_pipeline/21_*.py` (9 files) - moved from `scripts/`; path bootstrap depth corrected
- `cluster/21_1_prior_predictive.slurm` - python path updated + stale m6b comment cleaned
- `cluster/21_2_recovery.slurm` - python path updated + stale m6b comment cleaned
- `cluster/21_2_recovery_aggregate.slurm` - python path updated + stale m6b comment cleaned
- `cluster/21_3_fit_baseline.slurm` - python path updated + stale m6b comment cleaned
- `cluster/21_4_baseline_audit.slurm` - python path updated
- `cluster/21_5_loo_stacking_bms.slurm` - python path updated
- `cluster/21_6_fit_with_l2.slurm` - python path updated
- `cluster/21_7_scale_audit.slurm` - python path updated
- `cluster/21_8_model_averaging.slurm` - python path updated
- `cluster/21_9_manuscript_tables.slurm` - python path updated
- `scripts/fitting/tests/test_load_side_validation.py` - 5 enumerated paths updated to bayesian_pipeline/

## Decisions Made
- `21_6_dispatch_l2.slurm` has no direct Python invocation (delegates to `cluster/21_dispatch_l2_winners.sh`), so only 10 SLURM files required python path updates despite 11 SLURM files existing for Phase 21.
- `21_8_model_averaging.slurm` retains references to `cluster/13_bayesian_m6b_subscale.slurm` — these are live operational code paths (not stale comments), distinct from the `13_bayesian_m6b.slurm` pattern cleaned by Task 4b.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Path bootstrap depth wrong after directory move**
- **Found during:** Task 7 (smoke test)
- **Issue:** 8 scripts used `_PROJECT_ROOT = _THIS_FILE.parent.parent` (or `Path(__file__).parent.parent`), which resolved to `scripts/bayesian_pipeline/` → `scripts/`, not the project root. Running `--help` gave `ModuleNotFoundError: No module named 'config'`.
- **Fix:** Updated all 8 occurrences to `.parent.parent.parent` (scripts/bayesian_pipeline/ is one level deeper than scripts/)
- **Files modified:** 21_baseline_audit.py, 21_compute_loo_stacking.py, 21_fit_baseline.py, 21_fit_with_l2.py, 21_model_averaging.py, 21_run_bayesian_recovery.py, 21_run_prior_predictive.py, 21_scale_audit.py
- **Verification:** All 9 `--help` invocations pass; 5/5 tests pass
- **Committed in:** `6473f34` (part of atomic task commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — bug in path bootstrap depth)
**Impact on plan:** Necessary for correct SLURM execution; no scope creep.

## Issues Encountered
- A `git stash` from a previous session was present on the repo. After the stash pop accidentally unstaged the git mv operations, the staging was repaired by `git add scripts/bayesian_pipeline/` + `git rm` on the old paths to re-establish R (renamed) status.

## Next Phase Readiness
- `scripts/bayesian_pipeline/` is fully operational; all scripts importable and --help tested
- Plan 28-08 grep invariant `grep -rn "13_bayesian_m[1-6]" cluster/21_*.slurm` will still hit `21_8_model_averaging.slurm`'s references to `13_bayesian_m6b_subscale.slurm` (operational code, not stale comments) — Plan 28-08 should account for this distinction
- REFAC-07 closed

---
*Phase: 28-bayesian-first-restructure-repo-cleanup*
*Completed: 2026-04-21*
