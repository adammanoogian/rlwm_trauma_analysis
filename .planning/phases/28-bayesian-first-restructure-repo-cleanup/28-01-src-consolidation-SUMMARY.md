---
phase: 28-bayesian-first-restructure-repo-cleanup
plan: "01"
subsystem: infra
tags: [jax, numpyro, src-layout, import-refactor, shim-deletion, package-structure]

# Dependency graph
requires:
  - phase: 27-closure
    provides: clean repo with shims still present; pip install -e . confirmed working
provides:
  - src/rlwm/fitting/ package with jax_likelihoods, numpyro_models, numpyro_helpers
  - environments/ and models/ shim packages deleted
  - All 44 call sites updated to canonical rlwm.* import paths
  - TestCanonicalPaths replacing TestBackwardCompat in test_rlwm_package.py
affects:
  - 28-02 through 28-09 (all Wave 2+ plans depend on shims being gone and fitting in src/)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pure-library fitting math (JAX likelihoods, NumPyro models) lives in src/rlwm/fitting/"
    - "Orchestrator scripts (fit_mle.py, fit_bayesian.py, etc.) stay in scripts/fitting/"
    - "No backward-compat shims; all consumers import from rlwm.* directly"

key-files:
  created:
    - src/rlwm/fitting/__init__.py
    - src/rlwm/fitting/jax_likelihoods.py (git mv from scripts/fitting/)
    - src/rlwm/fitting/numpyro_models.py (git mv from scripts/fitting/)
    - src/rlwm/fitting/numpyro_helpers.py (git mv from scripts/fitting/)
  modified:
    - scripts/fitting/fit_mle.py
    - scripts/fitting/fit_bayesian.py
    - scripts/fitting/bayesian_diagnostics.py
    - scripts/fitting/warmup_jit.py
    - scripts/utils/remap_mle_ids.py
    - scripts/21_run_bayesian_recovery.py
    - scripts/21_run_prior_predictive.py
    - scripts/21_fit_with_l2.py
    - scripts/fitting/tests/ (11 files)
    - validation/benchmark_parallel_scan.py
    - validation/test_m3_backward_compat.py
    - scripts/simulations/generate_data.py
    - scripts/simulations/parameter_sweep.py
    - scripts/simulations/unified_simulator.py
    - tests/examples/ (3 files)
    - tests/test_rlwm_package.py
    - tests/test_wmrl_exploration.py
    - validation/test_parameter_recovery.py
    - validation/test_model_consistency.py
    - validation/test_unified_simulator.py
    - README.md

key-decisions:
  - "Narrow migration: only jax_likelihoods.py, numpyro_models.py, numpyro_helpers.py moved; 12 orchestrators stay in scripts/fitting/"
  - "git mv used for all 3 module moves to preserve git log --follow history"
  - "scripts/simulations/README.md updated even though it is documentation (grep invariant was over all files, not just .py)"
  - "Stale .pyc bytecache files cleared to ensure grep invariants pass without --include filter"
  - "TestBackwardCompat class rewritten as TestCanonicalPaths testing rlwm.* paths directly"

patterns-established:
  - "src/rlwm/fitting/ is the canonical location for pure-library JAX/NumPyro math"
  - "scripts/fitting/ contains only orchestrator scripts (CLI wrappers, fitting drivers)"
  - "No from environments.* or from models.* imports anywhere in codebase"

# Metrics
duration: 35min
completed: 2026-04-21
---

# Phase 28 Plan 01: src/ Consolidation Summary

**Deleted environments/ and models/ shim packages, git-moved 3 JAX/NumPyro math modules to src/rlwm/fitting/, and rewrote 44 call sites across scripts, tests, and validation to canonical rlwm.* imports in one atomic commit.**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-04-21T00:00:00Z
- **Completed:** 2026-04-21T00:35:00Z
- **Tasks:** 14 (all)
- **Files modified:** 44 (1 new, 3 moved, 6 deleted, 34 edited)

## Accomplishments

- Deleted `environments/` (3 files) and `models/` (4 files) backward-compat shim packages outright
- Migrated 3 pure-library fitting modules to `src/rlwm/fitting/` via `git mv` (history preserved for `git log --follow`)
- Updated all 44 call sites in scripts/, tests/, validation/ to canonical `rlwm.*` import paths
- Replaced shim-testing `TestBackwardCompat` class with `TestCanonicalPaths` testing canonical paths directly
- Resolved pre-existing `ModuleNotFoundError: No module named 'rlwm'` collection error in `tests/test_wmrl_exploration.py`
- All 3 grep invariants return zero, `pytest scripts/fitting/tests/test_v4_closure.py -v` passes 3/3, 166 fitting tests collected cleanly

## Task Commits

All 14 tasks landed in one atomic commit per the plan's explicit directive (broken intermediate states prevented per-task commits):

1. **Tasks 1-14: All work** - `1f06ee7` (refactor)

## Files Created/Modified

- `src/rlwm/fitting/__init__.py` - New package skeleton with module docstring
- `src/rlwm/fitting/jax_likelihoods.py` - Moved from scripts/fitting/ via git mv
- `src/rlwm/fitting/numpyro_models.py` - Moved from scripts/fitting/ via git mv; internal scripts.fitting.* imports rewritten to rlwm.fitting.*
- `src/rlwm/fitting/numpyro_helpers.py` - Moved from scripts/fitting/ via git mv
- `scripts/fitting/fit_mle.py`, `fit_bayesian.py`, `bayesian_diagnostics.py` - Updated to rlwm.fitting.*
- `scripts/fitting/warmup_jit.py`, `scripts/utils/remap_mle_ids.py` - Updated to rlwm.fitting.*
- `scripts/21_run_bayesian_recovery.py`, `scripts/21_run_prior_predictive.py`, `scripts/21_fit_with_l2.py` - Updated to rlwm.fitting.*
- `scripts/fitting/tests/` (11 files) - Updated to rlwm.fitting.*
- `validation/benchmark_parallel_scan.py`, `validation/test_m3_backward_compat.py` - Updated to rlwm.fitting.*
- `scripts/simulations/{generate_data.py,parameter_sweep.py,unified_simulator.py}` - Updated to rlwm.envs.* and rlwm.models.*
- `tests/examples/` (3 files) - Updated to rlwm.envs.* and rlwm.models.*
- `tests/test_rlwm_package.py` - TestBackwardCompat rewritten as TestCanonicalPaths
- `tests/test_wmrl_exploration.py` - Updated to rlwm.envs.* and rlwm.models.*
- `validation/{test_parameter_recovery.py,test_model_consistency.py,test_unified_simulator.py}` - Updated to rlwm.*
- `README.md` - Added `pip install -e .` prerequisite to Setup section
- `scripts/simulations/README.md` - Updated code examples to canonical rlwm.* imports

## Decisions Made

- Narrow migration scope: only `jax_likelihoods.py`, `numpyro_models.py`, `numpyro_helpers.py` moved; 12 orchestrator modules stay in `scripts/fitting/`
- `git mv` for all 3 moves to preserve `git log --follow` history
- `scripts/simulations/README.md` updated (grep invariant covers all files, not just .py) to pass the zero-match invariant
- Stale `.pyc` bytecache files cleared so grep invariants pass without `--include="*.py"` filter
- Single atomic commit covering all 14 tasks (plan explicitly required this to avoid broken intermediate states)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated scripts/simulations/README.md code examples**

- **Found during:** Task 13 (local verification)
- **Issue:** `grep -rn "^from environments\." scripts/` matched `scripts/simulations/README.md` line 18, causing the grep invariant to fail. The plan's invariant does not filter `--include="*.py"`, so markdown files with old import examples also fail.
- **Fix:** Updated two import lines in the README.md code examples to use `rlwm.envs.rlwm_env` and `rlwm.models.q_learning` canonical paths.
- **Files modified:** `scripts/simulations/README.md`
- **Verification:** `grep -rn "^from environments\."` and `grep -rn "^from models\."` now return zero matches.
- **Committed in:** `1f06ee7` (atomic commit)

**2. [Rule 3 - Blocking] Cleared stale .pyc bytecache before verification**

- **Found during:** Task 13 (local verification)
- **Issue:** `grep -rn "scripts\.fitting\.\(jax_likelihoods\|numpyro_models\|numpyro_helpers\)"` matched dozens of `.pyc` binary files from earlier pytest runs containing the old compiled import strings, causing false positives in the invariant check.
- **Fix:** `find ... -name "*.pyc" -delete` to purge stale bytecache.
- **Files modified:** None (deleted binary cache files only)
- **Verification:** Grep invariant now returns zero matches.
- **Committed in:** `1f06ee7` (atomic commit; .pyc files are not tracked by git)

---

**Total deviations:** 2 auto-fixed (1 bug in verification scope, 1 blocking stale cache)
**Impact on plan:** Both fixes were necessary to pass verification invariants exactly as written. No scope creep.

## Issues Encountered

None beyond the two auto-fixed deviations above.

## Next Phase Readiness

Wave 2 (plans 28-02, 28-03, 28-04, 28-05, 28-07, 28-09) can now proceed in parallel. All depend on:
- `environments/` and `models/` shims deleted (DONE)
- `src/rlwm/fitting/` containing the 3 moved modules (DONE)
- All callers updated to canonical `rlwm.*` paths (DONE)

No blockers. The `test_wmrl_exploration.py` collection error is resolved.

---
*Phase: 28-bayesian-first-restructure-repo-cleanup*
*Completed: 2026-04-21*
