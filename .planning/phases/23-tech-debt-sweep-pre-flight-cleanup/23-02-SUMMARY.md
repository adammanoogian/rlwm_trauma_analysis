---
phase: 23-tech-debt-sweep-pre-flight-cleanup
plan: "02"
subsystem: testing
tags: [pytest, mle, mle_utils, k-bounds, config, parameterization, clean02]

# Dependency graph
requires:
  - phase: 22-milestone-closure
    provides: v4.0 closure guard + REQUIREMENTS.md with CLEAN-02 requirement

provides:
  - pytest invariant guard that asserts every MLE bounds dict uses Collins K ∈ [2.0, 6.0]
  - source-text grep invariant ensuring no legacy [1, 7] K-bound substrings in mle_utils.py
  - tightened EXPECTED_PARAMETERIZATION docstring using v4.0+ invariant vocabulary

affects:
  - 23-03 (subsequent tech-debt plans — can assume CLEAN-02 closed)
  - 24-cold-start-pipeline-execution (MLE fits produced under invariant-enforced bounds)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Deprecate-and-guard pattern: invariant already true in code → add pytest guard to keep it true"
    - "Source-text grep test: assert literal substrings absent from a file (matches ROADMAP SC grep command exactly)"

key-files:
  created:
    - scripts/fitting/tests/test_mle_k_bounds_invariant.py
  modified:
    - config.py

key-decisions:
  - "mle_utils.py was already clean before Phase 23; no code edits needed, only the guard was added"
  - "numpyro_models.py legacy [1, 7] references left untouched (out of scope — kept as regression-test target for test_wmrl_model.py)"
  - "test_numpyro_helpers.py:308 'v3.0-legacy' fixture left untouched (tests the rejection path, not an accept path)"

patterns-established:
  - "Source-grep invariant test: use pathlib.Path.read_text() + substring search to enforce source-level cleanness"

# Metrics
duration: 14min
completed: 2026-04-19
---

# Phase 23 Plan 02: K-Bounds Invariant Guard + Config Vocabulary Summary

**pytest K-bounds invariant guard for mle_utils.py + EXPECTED_PARAMETERIZATION docstring rewritten as v4.0+ invariant (CLEAN-02 closed)**

## Performance

- **Duration:** 14 min
- **Started:** 2026-04-19T13:24:42Z
- **Completed:** 2026-04-19T13:38:40Z
- **Tasks:** 2
- **Files modified:** 2 (1 created, 1 edited)

## Accomplishments

- Installed pytest invariant guard (`test_mle_k_bounds_invariant.py`) with two functions: runtime dict inspection + source-text grep, both passing immediately against the already-clean `mle_utils.py`
- Rewrote `config.py` EXPECTED_PARAMETERIZATION module docstring to describe the v4.0+ invariant directly rather than referencing a legacy class as if it were still an accepted parameterization
- Replaced inline error message `"likely a v3.0 legacy fit."` with `"(pre-v4.0 CSVs without this column are rejected)."` — vocabulary now describes the rejection, not the legacy accept path
- v4.0 closure guard (`python validation/check_v4_closure.py --milestone v4.0`) exits 0 after both edits

## Task Commits

Each task was committed atomically:

1. **Task 1: Write K-bounds invariant guard test** - `83fcde5` (chore)
2. **Task 2: Clean legacy vocabulary in config.py EXPECTED_PARAMETERIZATION docstring** - `089f2d0` (chore)

**Plan metadata:** see final commit below

## Files Created/Modified

- `scripts/fitting/tests/test_mle_k_bounds_invariant.py` - New pytest guard (102 lines): `test_mle_capacity_bounds_are_collins` inspects all 6 `*_BOUNDS` dicts at import time; `test_mle_utils_source_has_no_legacy_k_bounds` reads source as text and asserts none of 6 forbidden substrings appear
- `config.py` - EXPECTED_PARAMETERIZATION module docstring (~line 682) rewritten; inline error message (~line 737) updated

## Requirements Closed

- **CLEAN-02**: `mle_utils.py` verified Collins-only K ∈ [2.0, 6.0]; pytest guard in place; config vocabulary aligned with rejection semantics

## Decisions Made

- `mle_utils.py` needed zero code edits — it was already clean pre-Phase-23. The plan's job was to install the guard, not fix code.
- `numpyro_models.py` legacy `[1, 7]` references (lines 326, 382 in `wmrl_hierarchical_model`) left explicitly untouched per plan scope — that deprecated function is kept as a regression-test target for `test_wmrl_model.py`.
- The `"v3.0-legacy"` string in `test_numpyro_helpers.py:308` is a deliberate negative-path fixture (tests that `load_fits_with_validation` correctly rejects it) — left untouched.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

Full `scripts/fitting/tests/` suite run crashed at `test_m3_hierarchical.py::test_smoke_dispatch_with_l2` with a JAX "Fatal Python error: Aborted" due to memory pressure (Windows `C000012D` STATUS_NO_MEMORY) during MCMC compilation. This is a pre-existing environmental issue on this Windows machine, unrelated to the changes in this plan (no JAX code was touched). The directly relevant tests — new invariant guard, `test_numpyro_helpers.py` (including the three `load_fits_with_validation` tests), and `test_mle_quick.py` — all pass (13/13 in 52.97s).

## Next Phase Readiness

- CLEAN-02 closed; Phase 23 plan 02 complete
- `mle_utils.py` invariant guard will catch any future regression in CI
- Phase 24 cold-start MLE fits can proceed with confidence that capacity bounds are locked to Collins K ∈ [2.0, 6.0]

---
*Phase: 23-tech-debt-sweep-pre-flight-cleanup*
*Completed: 2026-04-19*
