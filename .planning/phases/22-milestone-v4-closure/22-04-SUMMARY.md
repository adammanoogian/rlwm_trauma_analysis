---
phase: 22-milestone-v4-closure
plan: "04"
subsystem: testing
tags: [closure, reproducibility, pytest, determinism, gitignore, traceability]

requires:
  - phase: 22-milestone-v4-closure
    provides: "22-01 STATE.md clean; 22-02 VERIFICATION.md backfill; 22-03 REQUIREMENTS.md 71 rows"

provides:
  - "validation/check_v4_closure.py — deterministic closure-state checker (8 checks, CLI + importable API)"
  - "scripts/fitting/tests/test_v4_closure.py — pytest wrapper (3 tests: passes-on-HEAD, deterministic, rejects-wrong-milestone)"
  - "Phase 22 SC#7 reproducibility guard fully instantiated"
  - "Phase 22 SC#8 (no one-off checks) enforced via banned-phrase scan in check_verification_files_exist"
  - "Phase 22 SC#9 (cluster-freshness framing) enforced via check_cluster_freshness_framing"

affects: [gsd-complete-milestone, gsd-audit-milestone, v5.0-closure]

tech-stack:
  added: []
  patterns:
    - "Closure-state checker pattern: stdlib+PyYAML only, deterministic, exit 0/1, importable + CLI"
    - "Banned-phrase construction via concatenation so checker source does not contain literal banned strings (meta-check)"
    - "YAML frontmatter parsed via _parse_yaml_frontmatter helper; cluster_execution_pending items validated for canonical cold-start framing"

key-files:
  created:
    - validation/check_v4_closure.py
    - scripts/fitting/tests/test_v4_closure.py
  modified: []

key-decisions:
  - "Banned phrases constructed programmatically (string concatenation) so validation/check_v4_closure.py does not contain the literal banned strings that would trigger its own meta-check — 'I confirmed by reading' built as 'I' + ' confirmed by reading'"
  - "check_project_md_active_migration uses full-section extraction (not 500-char window) to find v4.0 reference in the ### Validated section — the subheader '#### v4.0 (milestone)' appears 25 lines below the ### Validated heading"
  - "EXPECTED_COLD_START_ENTRIES includes both bash cluster/21_submit_pipeline.sh AND bash cluster/12_submit_all_gpu.sh — Phase 14 cluster-pending items correctly route to the 12_submit_all_gpu.sh entry (not wired into Phase 21 pipeline, tracked as v5.0 candidate)"
  - "check_cluster_freshness_framing parses cluster_execution_pending YAML list from frontmatter only (not prose grep) to avoid false positives from grep-evidence table cells that reference sbatch commands as verification steps"

patterns-established:
  - "Phase closure checker: deterministic script asserting all SC invariants in one runnable file; future milestones should create check_vX_closure.py following this pattern"
  - "Meta-check: checker source must not contain the literal strings it scans for; use programmatic construction"

duration: 6min
completed: 2026-04-19
---

# Phase 22 Plan 04: Reproducibility Guard + Pytest Wrapper Summary

**Deterministic 8-check closure-state script (stdlib+PyYAML, exit 0 on current HEAD) + 3-test pytest wrapper asserting SC#7/8/9 invariants for milestone v4.0**

## Performance

- **Duration:** 6 min
- **Started:** 2026-04-19T08:18:54Z
- **Completed:** 2026-04-19T08:24:49Z
- **Tasks:** 2
- **Files created:** 2

## Accomplishments

- Wrote `validation/check_v4_closure.py` with 8 check functions, `check_all()` orchestrator, and `main()` CLI; all 8 checks PASS on current HEAD
- Wrote `scripts/fitting/tests/test_v4_closure.py` with 3 tests (passes-on-HEAD, deterministic, rejects-wrong-milestone); all 3 PASS in 2.71s
- Resolved meta-check constraint by building banned-phrase strings via concatenation so the checker source does not contain its own forbidden literals
- Fixed `check_project_md_active_migration` to extract the full `### Validated` section (not a 500-char window) — the `#### v4.0 (milestone)` subheader is 25 lines below the heading

## Task Commits

1. **Task 1: validation/check_v4_closure.py** — `805a58c` (feat)
2. **Task 2: test_v4_closure.py pytest wrapper** — `b818a98` (feat)

## Files Created/Modified

- `validation/check_v4_closure.py` — 8 check functions + check_all + main + CheckResult dataclass; stdlib+PyYAML only
- `scripts/fitting/tests/test_v4_closure.py` — 3 pytest tests; REPO_ROOT sys.path hack mirrors existing test convention

## Decisions Made

- **Banned phrases built programmatically**: `BANNED_EVIDENCE_PHRASES` constant uses string concatenation (`"I" + " confirmed by reading"`) so that `grep -c "I confirmed by reading" validation/check_v4_closure.py` returns 0. The plan's meta-check requires this.
- **Full-section Validated scan**: `check_project_md_active_migration` finds the `### Validated` section boundary and searches the entire section for `v4.0` — not a fixed-char window. The `#### v4.0 (milestone)` subheading is 25 lines below `### Validated`.
- **EXPECTED_COLD_START_ENTRIES includes both canonical scripts**: `bash cluster/21_submit_pipeline.sh` AND `bash cluster/12_submit_all_gpu.sh`. Phase 14's deferred items correctly cite the latter (K-refit is not wired into the Phase 21 pipeline — v5.0 candidate).
- **Frontmatter-only framing check**: `check_cluster_freshness_framing` parses `cluster_execution_pending` YAML list items from frontmatter rather than scanning the prose body with regex. This avoids false positives from `grep` evidence cells in the Observable Truths table (which reference `sbatch cluster/12_mle_gpu.slurm` as a verification command, not as a user instruction).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] check_project_md_active_migration used a 500-char window to find v4.0 in the Validated section**

- **Found during:** Task 1 first test run (exit 1)
- **Issue:** Initial implementation searched `text[validated_start : validated_start + 500]` for "v4.0". The `#### v4.0 (milestone)` subheader in PROJECT.md is 25 lines below `### Validated`, beyond the 500-char window.
- **Fix:** Extract the full Validated section (from `### Validated` to next `### ` or EOF) and search within it.
- **Files modified:** validation/check_v4_closure.py
- **Verification:** Re-ran checker — check_project_md_active_migration PASS.
- **Committed in:** 805a58c (Task 1 commit, fix folded in before push)

**2. [Rule 1 - Bug] Docstring escape warning from raw `\d` in docstring**

- **Found during:** Task 1 first test run
- **Issue:** Python 3.12 SyntaxWarning on invalid escape sequence `\d` inside a plain string docstring.
- **Fix:** Replaced `\d+` with `NN` in the docstring text.
- **Files modified:** validation/check_v4_closure.py
- **Committed in:** 805a58c

---

**Total deviations:** 2 auto-fixed (both Rule 1 — Bug)
**Impact on plan:** Both fixes needed for correctness. No scope creep.

## Issues Encountered

None beyond the auto-fixed bugs above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Phase 22 is now 4/4 plans complete (22-01 through 22-04)
- `python validation/check_v4_closure.py --milestone v4.0` provides the single-command pass/fail verdict for v4.0 closure
- `/gsd:complete-milestone` can treat exit 0 as the final green-light gate
- Future v5.0 closure should create `validation/check_v5_closure.py` following this pattern; consider extracting a shared `check_framework.py` base class if the pattern repeats cleanly (out of v4.0 scope)

---

*Phase: 22-milestone-v4-closure*
*Completed: 2026-04-19*
