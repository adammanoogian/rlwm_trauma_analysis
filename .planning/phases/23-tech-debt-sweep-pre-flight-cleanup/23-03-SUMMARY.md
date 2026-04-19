---
phase: 23-tech-debt-sweep-pre-flight-cleanup
plan: "03"
subsystem: testing
tags: [pytest, guard-test, pycache, clean-up, CLEAN-03, 16b_bayesian_regression]

# Dependency graph
requires:
  - phase: 23-tech-debt-sweep-pre-flight-cleanup/23-01
    provides: "CLEAN-01 closed — legacy qlearning import removed"
  - phase: 23-tech-debt-sweep-pre-flight-cleanup/23-02
    provides: "CLEAN-02 closed — legacy K-bounds removed from mle_utils"
provides:
  - "CLEAN-03 closed: orphaned 16b_bayesian_regression pycache deleted from disk"
  - "pytest guard test_no_16b_references.py: two functions lock out reintroduction"
  - "SC#3 find command interpretation documented and deterministically enforced"
affects:
  - phase: 24-cold-start-pipeline-execution
  - phase: 27-closure

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Self-excluding grep-invariant guard: test skips its own file to avoid self-referential pattern matches"
    - "Archival-policy skip pattern: .planning/ excluded from live-source scans per v4.0 closure protocol"

key-files:
  created:
    - scripts/fitting/tests/test_no_16b_references.py
  modified: []

key-decisions:
  - "MODEL_REFERENCE.md was already clean — pre-flight grep was against docs/ broadly and hit the binary Senta PDF (not the .md file); no edit needed"
  - "pycache file not git-tracked (gitignored) — Task 2 commit is empty but records verified closure state"
  - "Guard test self-excludes via _THIS_FILE constant — necessary because the test file must name the forbidden patterns as string literals"
  - ".planning/ archival tree excluded from both guard functions per v4.0 closure protocol; historical Phase 13/18 plan references to 16b are intentionally preserved"

patterns-established:
  - "Self-excluding guard pattern: filepath == _THIS_FILE check inside text-reference scan loop"
  - "Archival-root exclusion: _is_under_excluded_root() helper for file-existence checks"

# Metrics
duration: 27min
completed: 2026-04-19
---

# Phase 23 Plan 03: CLEAN-03 16b Regression Cleanup Summary

**Pytest grep-invariant guard (test_no_16b_references.py) added to lock out reintroduction of 16b_bayesian_regression; orphaned pycache deleted; MODEL_REFERENCE.md confirmed already clean**

## Performance

- **Duration:** 27 min
- **Started:** 2026-04-19T13:25:34Z
- **Completed:** 2026-04-19T13:52:00Z
- **Tasks:** 2
- **Files modified:** 1 created, 1 deleted (untracked)

## Accomplishments

- Created two-function pytest guard: `test_no_16b_text_references_in_live_source` (grep-invariant for scripts/cluster/docs) and `test_no_16b_files_outside_planning_tree` (file-existence check for SC#3)
- Deleted `scripts/__pycache__/16b_bayesian_regression.cpython-313.pyc` (orphaned bytecode from deleted source file)
- Confirmed `docs/03_methods_reference/MODEL_REFERENCE.md` already had no live 16b reference — no edit required
- `find . -path ./.planning -prune -o -path ./.git -prune -o -name "16b*" -print` returns zero lines
- `python validation/check_v4_closure.py --milestone v4.0` exits 0 — no v4.0 invariants broken

## Task Commits

Each task was committed atomically:

1. **Task 1: Write grep-invariant guard for 16b references** - `f0d4e60` (chore)
2. **Task 2: Remove 16b pycache, update MODEL_REFERENCE.md, verify closure** - `39c2202` (chore, empty — no git-tracked files changed)

## Files Created/Modified

- `scripts/fitting/tests/test_no_16b_references.py` — Two-function CLEAN-03 enforcement guard:
  - `test_no_16b_text_references_in_live_source()`: walks scripts/, cluster/, docs/03_methods_reference/ for .py/.slurm/.sh/.md files; asserts no match for `16b_bayesian_regression` or `scripts/16b`; self-excludes the guard file itself; skips .planning/, .git/, __pycache__/
  - `test_no_16b_files_outside_planning_tree()`: walks entire repo root; asserts no file named `16b*` exists outside .planning/ and .git/; pycache .pyc files count as violations

## Decisions Made

1. **MODEL_REFERENCE.md already clean**: Pre-flight planning grep hit `docs/` broadly and matched the binary Senta PDF (coincidental "16b" substring in paper filename). The actual `.md` file had no live text reference to `16b_bayesian_regression`. No edit was needed.

2. **Guard self-exclusion**: The test file must name the forbidden patterns as string literals to check against them. Added `_THIS_FILE` constant and `filepath == _THIS_FILE` skip inside the text-reference scan to prevent trivial self-detection.

3. **Task 2 empty commit**: The pycache file was not git-tracked (correctly excluded by .gitignore), so its deletion produced no staged change. MODEL_REFERENCE.md required no edit. An empty commit with `--allow-empty` was used to record the verified closure state with the correct commit message.

4. **SC#3 find command interpretation**: ROADMAP.md's raw `find . -name "16b*"` is interpreted with `.planning/` pruned. This policy is now encoded deterministically in the file-existence guard function.

## Deviations from Plan

None — plan executed exactly as written.

The plan's "note" that both tests would likely fail was conditional ("and/or"). In practice, only `test_no_16b_files_outside_planning_tree` failed in Task 1 (pycache violation). `test_no_16b_text_references_in_live_source` passed immediately because MODEL_REFERENCE.md was already clean (pre-flight grep hit binary PDF, not .md file). This is a discovery that makes Task 2 simpler, not a deviation.

## Issues Encountered

- **Self-referential pattern match**: First draft of guard test caused `test_no_16b_text_references_in_live_source` to fail on its own file (the test necessarily contains the forbidden pattern strings). Fixed by adding `_THIS_FILE` self-exclusion constant. This was a predictable implementation challenge, not a plan deviation.

- **Pre-existing test crash**: `test_m3_hierarchical.py::test_smoke_dispatch` crashes with a JAX/XLA compilation error (unrelated to CLEAN-03 changes). This is a pre-existing flaky test not introduced by this plan.

## Next Phase Readiness

- CLEAN-03 is closed. The guard test provides CI protection against reintroduction.
- Phase 24 cold-start pipeline execution can proceed without "why is 16b referenced?" debugging.
- Remaining CLEAN items (CLEAN-04: load-side validation audit) handled in other 23-xx plans.

---
*Phase: 23-tech-debt-sweep-pre-flight-cleanup*
*Completed: 2026-04-19*
