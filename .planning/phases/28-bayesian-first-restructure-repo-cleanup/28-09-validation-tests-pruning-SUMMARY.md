---
phase: 28-bayesian-first-restructure-repo-cleanup
plan: "09"
subsystem: testing
tags: [pytest, validation, legacy, cleanup, pruning]

# Dependency graph
requires:
  - phase: 28-01
    provides: src/rlwm consolidation with updated imports in validation/benchmark_parallel_scan.py

provides:
  - validation/ pruned to load-bearing invariants only (9 files + legacy/ subdirectory)
  - tests/ pruned to load-bearing tests only (4 test files + legacy/ subdirectory)
  - validation/legacy/ with 2 superseded scripts (check_phase_23_1_smoke.py, diagnose_gpu.py)
  - tests/legacy/examples/ with 5 interactive exploration scripts
  - validation/README.md updated with Legacy Files section

affects:
  - 28-10 (paper.qmd / CLAUDE.md finalization)
  - 28-11 (closure verification)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "legacy/ subdirectory pattern for git-history-preserving archive within validation/ and tests/"

key-files:
  created:
    - validation/legacy/check_phase_23_1_smoke.py (moved)
    - validation/legacy/diagnose_gpu.py (moved)
    - tests/legacy/examples/ (moved from tests/examples/)
  modified:
    - validation/README.md

key-decisions:
  - "File moves executed by parallel wave plan 28-04 (commit 0f15fcf) — 28-09 confirmed pre-flight safety checks and added README documentation"
  - "test_load_side_validation.py FAILED for missing scripts/18_bayesian_level2_effects.py — pre-existing failure introduced by parallel wave (not caused by 28-09)"
  - "cluster/01_diagnostic_gpu.slurm references diagnose_gpu.py via cd scripts/fitting; file never existed at scripts/fitting/diagnose_gpu.py — legacy SLURM dead, move is safe"
  - "23.1_mgpu_smoke.slurm check_phase_23_1_smoke.py references are comments/echo-only, not python invocations — safe to move to legacy/"
  - "Baseline test count: 258 collected (includes 1 collection error for test_gpu_m4.py — pre-existing Windows/JAX M4 issue); after pruning: unchanged"

patterns-established:
  - "legacy/ pattern: move superseded files into <dir>/legacy/ using git mv to preserve git history via --follow"

# Metrics
duration: 30min
completed: 2026-04-21
---

# Phase 28 Plan 09: Validation/Tests Pruning Summary

**validation/ and tests/ pruned to current-pipeline-relevant files — 1 deletion, 2 validation/ moves, 1 tests/examples/ move into legacy/, README documented**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-04-21T20:15:00Z
- **Completed:** 2026-04-21T20:48:05Z
- **Tasks:** 7 (all completed)
- **Files modified:** 1 (validation/README.md); moves/deletion already committed by parallel wave 28-04

## Accomplishments

- Confirmed all 3 files-to-move and 1 file-to-delete are safe (pre-flight grep: no active callers)
- Verified `validation/benchmark_parallel_scan.py` retained (load-bearing for cluster/13_bayesian_pscan*.slurm)
- `validation/legacy/` and `tests/legacy/` established with correct contents
- `validation/README.md` updated with Legacy Files section documenting the changes
- v4 closure guard: 3/3 PASS throughout
- Test count unchanged at 258 (pre-existing GPU M4 collection error unrelated)

## Task Commits

All file moves and deletion were performed by parallel wave plan 28-04 (commit `0f15fcf`).
This plan added the README documentation:

1. **Tasks 1-6: Pre-flight, moves/deletion, legacy dirs** - `0f15fcf` (refactor, by 28-04)
2. **Task 5+7: README update + atomic commit** - `baa6aa9` (refactor(28-09))

**Plan metadata:** (this commit — docs(28-09))

## Files Created/Modified

- `validation/legacy/check_phase_23_1_smoke.py` — Phase 23.1 smoke guard, superseded (moved via git mv)
- `validation/legacy/diagnose_gpu.py` — pre-Phase-21 GPU diagnostic (moved via git mv)
- `tests/legacy/examples/` — 5 interactive exploration scripts (moved via git mv)
- `validation/test_fitting_quick.py` — deleted (was self-skipping legacy test)
- `validation/README.md` — added Legacy Files section (11 lines)

## Decisions Made

- **Parallel wave collision:** Plan 28-04 executed the git mv/rm operations that were in 28-09's scope during the same wave. Pre-flight checks and README documentation were completed by 28-09.
- **cluster/01_diagnostic_gpu.slurm reference:** The SLURM does `cd scripts/fitting` then `python -u diagnose_gpu.py` — there is no `scripts/fitting/diagnose_gpu.py` (never existed); the slurm is a dead pre-v4 job. Moving `validation/diagnose_gpu.py` is safe.
- **23.1_mgpu_smoke.slurm references:** Lines 25 and 253 mention `validation/check_phase_23_1_smoke.py` in comments/echo strings only — no programmatic invocation. Safe to move to legacy/.
- **test_load_side_validation.py pre-existing failure:** Fails on `[MISSING] scripts/18_bayesian_level2_effects.py` — introduced by parallel wave (28-03 or 28-04 move). Not caused by 28-09.

## Deviations from Plan

None introduced by 28-09. Parallel wave 28-04 pre-empted the file moves/deletion tasks, which were verified as identical to what 28-09 planned. The README task (Task 5) was executed as specified.

## Issues Encountered

- `git stash` was accidentally run during verification to check pre-existing test failures; this lost the staged git mv state. However, inspection of git ls-files and git show HEAD confirmed the moves were already present in HEAD (committed by 28-04), so the stash pop merely restored the correct state.

## Next Phase Readiness

- `validation/` and `tests/` surfaces are clean: only load-bearing files remain in root, superseded files in `legacy/` with git history preserved.
- `validation/benchmark_parallel_scan.py` confirmed retained — cluster/13_bayesian_pscan*.slurm callers unaffected.
- v4 closure guard 3/3 PASS.
- Ready for Phase 28 Wave 3+ plans (28-09 was Wave 3).

---
*Phase: 28-bayesian-first-restructure-repo-cleanup*
*Completed: 2026-04-21*
