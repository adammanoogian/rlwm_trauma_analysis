---
wave: 6
depends_on: [28-10, 28-11]
files_modified:
  - .planning/phases/28-bayesian-first-restructure-repo-cleanup/28-VERIFICATION.md  (new)
autonomous: true
---

# 28-12 End-of-Phase Verification: Grep-Audit + Dual Closure Guards + Quarto Render

## Goal

Run the full post-consolidation verification battery: the three closure guards (pytest suite, `validation/check_v4_closure.py` script, `scripts/fitting/tests/test_v4_closure.py` pytest module), the five grep-audit invariants, and a `quarto render` smoke. Capture the results in `28-VERIFICATION.md` so downstream verification (gsd-plan-checker) has a single artifact to inspect.

This is the LAST plan in Phase 28 — it does not modify any source code, only runs checks and writes a verification log.

## Must Haves

- [ ] `validation/check_v4_closure.py --milestone v4.0` exits 0 (run as a plain script, not via pytest).
- [ ] `pytest scripts/fitting/tests/test_v4_closure.py -v` passes 3/3 (the canonical pytest-collected guard).
- [ ] Full `pytest` suite passes: expect 204 tests, 0 errors (the 1 pre-existing `tests/test_wmrl_exploration.py` collection error is resolved by plan 28-01), 0 new failures introduced by Phase 28. Some tests may skip for cluster-only or env-only reasons — those counts should match the baseline from 28-RESEARCH.md §pytest baseline.
- [ ] Five grep-audit invariants all return zero matches:
  1. `grep -rn "^from environments\\." scripts/ tests/ validation/ src/` — zero matches (shim deletion per REFAC-01)
  2. `grep -rn "^from models\\." scripts/ tests/ validation/ src/` — zero matches (shim deletion per REFAC-01)
  3. `grep -rn "from scripts.fitting.jax_likelihoods" . --include="*.py"` — zero matches (narrow migration per REFAC-02)
  4. `grep -rn "from scripts.fitting.numpyro_models" . --include="*.py"` — zero matches
  5. `grep -rn "from scripts.fitting.numpyro_helpers" . --include="*.py"` — zero matches
- [ ] Two stale-cluster grep invariants return zero matches:
  1. `grep -rn "13_bayesian_m[1-6]\\.slurm" cluster/ --include="*.sh" --include="*.slurm" | grep -v m6b_subscale` — zero matches (per REFAC-09)
  2. `grep -rn "scripts/18_bayesian_level2_effects\\.py" . --include="*.py" --include="*.sh" | grep -v post_mle` — zero matches (per REFAC-06 subprocess update)
- [ ] `quarto render manuscript/paper.qmd` exits 0 and `manuscript/_output/paper.pdf` exists.
- [ ] `28-VERIFICATION.md` written at `.planning/phases/28-bayesian-first-restructure-repo-cleanup/28-VERIFICATION.md` containing:
  - Run date + git HEAD SHA
  - Exit codes and summary stats for each of the 3 closure guards
  - Tabular report of the 7 grep-audit invariants (each with invariant name, expected count, actual count)
  - Quarto render log summary (exit code, PDF path, any warnings)
  - Confirmation table mapping REFAC-01 through REFAC-13 to "closed by plan NN-NN, verified via ..."
- [ ] Atomic commit: `chore(28-12): end-of-phase verification — all REFAC-* invariants hold`.

## Tasks

<tasks>
  <task id="1">
    <title>Run validation/check_v4_closure.py standalone</title>
    <detail>`python validation/check_v4_closure.py --milestone v4.0`. Capture exit code and stdout. Expected: exit 0. If exit != 0, investigate which invariant broke — do NOT commit over a failing closure guard.</detail>
  </task>

  <task id="2">
    <title>Run pytest on the canonical v4 closure guard</title>
    <detail>`pytest scripts/fitting/tests/test_v4_closure.py -v`. Expected: 3 passed, 0 failed.</detail>
  </task>

  <task id="3">
    <title>Run the full pytest suite</title>
    <detail>`pytest --tb=short` (NOT -x; we want to see the full failure surface if any). Capture results. Expected: ~204 passed, some number skipped (baseline from 28-RESEARCH.md was 204/1err/1skip; after plan 28-01 the err becomes a pass or skip, depending on env). If the number of passed tests decreases from baseline, investigate.</detail>
  </task>

  <task id="4">
    <title>Run the five import-hygiene grep invariants</title>
    <detail>For each invariant listed in "Must Haves" items 4.1–4.5, run the grep and capture (pattern, path, expected count, actual count). Each expected count is 0. Record in a markdown table for the verification report.</detail>
  </task>

  <task id="5">
    <title>Run the two stale-cluster grep invariants</title>
    <detail>For each of the two invariants in "Must Haves" item 5, run grep and capture counts.</detail>
  </task>

  <task id="6">
    <title>Run quarto render smoke</title>
    <detail>`quarto render manuscript/paper.qmd 2>&1 | tee /tmp/quarto_render.log`. Confirm exit 0 and `manuscript/_output/paper.pdf` exists. If Quarto isn't locally available (Windows dev env), record "deferred; render verified manually in plan 28-10" in the verification report.</detail>
  </task>

  <task id="7">
    <title>Write 28-VERIFICATION.md</title>
    <detail>Create `.planning/phases/28-bayesian-first-restructure-repo-cleanup/28-VERIFICATION.md` with structure:
      ```
      # Phase 28 Verification — End-of-Phase Audit

      **Date:** <ISO>
      **Git HEAD:** <short SHA>

      ## Closure guards
      | Guard | Command | Exit | Notes |
      |-------|---------|------|-------|
      | v4 script | python validation/check_v4_closure.py --milestone v4.0 | 0 | ... |
      | v4 pytest | pytest scripts/fitting/tests/test_v4_closure.py -v | 0 (3/3) | ... |
      | Full suite | pytest --tb=short | 0 | 204 passed, N skipped |

      ## Grep invariants (all must be zero)
      | # | Pattern | Path | Expected | Actual | Status |
      ...

      ## Quarto render
      | File | Exit | PDF path | Warnings |
      ...

      ## Requirement closure
      | REFAC ID | Closed by plan | Verified via |
      ...
      ```
      Fill with actual run data.</detail>
  </task>

  <task id="8">
    <title>Atomic commit</title>
    <detail>`chore(28-12): end-of-phase verification — all REFAC-* invariants hold`. Body: summarize the 3 closure guards + 7 grep invariants + quarto result.</detail>
  </task>
</tasks>

## Verification

```bash
# All three closure guards
python validation/check_v4_closure.py --milestone v4.0
pytest scripts/fitting/tests/test_v4_closure.py -v
pytest --tb=short

# Seven grep invariants (each must exit 1 with no matches)
! grep -rn "^from environments\\." scripts/ tests/ validation/ src/
! grep -rn "^from models\\." scripts/ tests/ validation/ src/
! grep -rn "from scripts.fitting.jax_likelihoods" . --include="*.py"
! grep -rn "from scripts.fitting.numpyro_models" . --include="*.py"
! grep -rn "from scripts.fitting.numpyro_helpers" . --include="*.py"
! grep -rn "13_bayesian_m[1-6]\\.slurm" cluster/ --include="*.sh" --include="*.slurm" | grep -v m6b_subscale
! grep -rn "scripts/18_bayesian_level2_effects\\.py" . --include="*.py" --include="*.sh" | grep -v post_mle

# Quarto render
quarto render manuscript/paper.qmd
test -f manuscript/_output/paper.pdf

# Verification artifact exists
test -f .planning/phases/28-bayesian-first-restructure-repo-cleanup/28-VERIFICATION.md
```

## Requirement IDs

Closes: **REFAC-13**. Verifies all prior REFAC-01..12 invariants via the verification-report table.
