---
status: gaps_found
phase: "28"
plan: "28-12"
date: "2026-04-22"
git_head: "907fa8c"
---

# Phase 28 Verification — End-of-Phase Audit

**Date:** 2026-04-22T06:49:34Z
**Git HEAD:** 907fa8c (after fix commit)
**Executor:** Plan 28-12 (end-of-phase verification gauntlet)

---

## Executive Summary

Phase 28 verification found **3 bug-class gaps** (REFAC-07 test path regression in
`test_bayesian_recovery.py`, `test_loo_stacking.py`, and `test_prior_predictive.py`) and
**2 informational gaps** (Grep-7 docstring-only hits; Windows JAX backend crash
pre-existing). The REFAC-07 test regressions were fixed inline as Rule-1 bugs in
commit `907fa8c`. After the fix, all addressable tests pass. Both v4.0 closure guards
pass 5/5 and 3/3.

**Final status: `gaps_found`** — 2 informational items remain (documented below);
no blocking gaps remain after the inline fix commit.

---

## Closure Guards

| Guard | Command | Exit | Result | Notes |
|-------|---------|------|--------|-------|
| v4 script | `python validation/check_v4_closure.py --milestone v4.0` | 0 | 5/5 PASS | All Phase 22 archive checks pass |
| v4 pytest | `pytest scripts/fitting/tests/test_v4_closure.py -v` | 0 | 3/3 PASS | passes / deterministic / rejects-wrong-milestone |
| Full suite (non-JAX) | `pytest tests/ + 6 fitting/tests modules` | 0 | 69 passed, 1 skipped | See note on Windows JAX crash below |

### v4 Script Output (all 5 checks)

```
[PASS] check_milestone_archive_complete: All 4 v4.0 archive artifacts present
[PASS] check_verification_files_exist: All 3 VERIFICATION.md files exist, parse, phrase-clean
[PASS] check_thesis_gitignore: All thesis filenames gitignored
[PASS] check_cluster_freshness_framing: All cluster-pending items reference canonical cold-start
[PASS] check_determinism_sentinel: phase=22 milestone=v4.0 checks=8 DETERMINISTIC_CONSTANT
```

### Full Pytest Notes

The full `pytest --tb=short` run is disrupted by a **pre-existing Windows JAX backend
abort** (not a Phase 28 regression). The crash occurs during JAX XLA compilation in:
- `scripts/fitting/tests/test_m3_hierarchical.py` — `prepare_stacked_participant_data` JIT path
- `scripts/fitting/tests/test_bayesian_recovery.py` — `q_learning_block_likelihood` vmap path
- `validation/test_m3_backward_compat.py` — `wmrl_m3_block_likelihood` scan path

These crashes are documented in the v4.0 STATE.md as "pre-existing Windows CPU JAX
M4-integration fatal and GPU-marked M4 tests confirmed environmental (not Phase 23
regressions)". They are cluster-only tests that require the Linux/CUDA environment.

**Non-JAX baseline passing (after 907fa8c fix):** 69 tests pass, 1 skipped (env-only
`test_fit_all_gpu_m4_smoke`). Pre-fix baseline was 51 pass (18 tests were failing or
not collected due to 3 stale-path bugs fixed in commit 907fa8c).

---

## Grep Invariants (all must be zero)

| # | Pattern | Path | Expected | Actual | Status |
|---|---------|------|----------|--------|--------|
| 1 | `^from environments\.` | scripts/ tests/ validation/ src/ | 0 | 0 | PASS |
| 2 | `^from models\.` | scripts/ tests/ validation/ src/ | 0 | 0 | PASS |
| 3 | `from scripts.fitting.jax_likelihoods` | *.py | 0 | 0 | PASS |
| 4 | `from scripts.fitting.numpyro_models` | *.py | 0 | 0 | PASS |
| 5 | `from scripts.fitting.numpyro_helpers` | *.py | 0 | 0 | PASS |
| 6 | `13_bayesian_m[1-6]\.slurm` (excl. m6b_subscale) | cluster/*.sh/*.slurm | 0 | 0 | PASS |
| 7 | `scripts/18_bayesian_level2_effects\.py` (excl. post_mle) | *.py *.sh | 0 | 3 (docstrings) | INFO |

### Grep 7 Detail — Docstring-Only Hits

Invariant 7 returns 3 matches, all in `scripts/bayesian_pipeline/21_manuscript_tables.py`:
- Line 41: docstring reference `plot via ``scripts/18_bayesian_level2_effects.py```
- Line 700: comment `# Figure 1 — forest plot (delegates to scripts/18_bayesian_level2_effects.py)`
- Line 711: docstring `Per plan 21-10, we reuse ``scripts/18_bayesian_level2_effects.py`` rather`

The **actual subprocess call** (line 746) correctly uses
`"scripts/post_mle/18_bayesian_level2_effects.py"` — filtered out by `grep -v post_mle`.
These 3 hits are historical docstring references, not functional stale paths.

**Assessment:** Informational gap only. The grep invariant was designed to catch
functional subprocess/import calls with old paths; these hits are documentation text
only. No behavioral regression.

### Note on `src/rlwm/fitting/numpyro_models.py` late-binding LBA imports

Two `from scripts.fitting.lba_likelihood import` statements at lines 2472 and 2604 of
`src/rlwm/fitting/numpyro_models.py` are **intentionally excluded** from REFAC-02 scope.
Per 28-01-PLAN.md: `lba_likelihood.py` is in the explicit "Do NOT move" list alongside
11 other orchestrator modules. These late-binding imports are inside the M4 LBA model
(separate track) and are by-design. Grep invariants 3-5 correctly do not match these.

---

## REFAC-07 Test Regression — Rule 1 Bug Fix (commit 907fa8c)

**Found during:** Task 3 (full pytest run)
**Issue:** `test_bayesian_recovery.py` line 35, `test_loo_stacking.py` line 52,
`test_prior_predictive.py` line 119 — all three use `importlib.util.spec_from_file_location`
to load Phase 21 scripts by absolute path. The paths referenced `scripts/21_*.py` (old
locations), which REFAC-07 (plan 28-06) moved to `scripts/bayesian_pipeline/21_*.py`.
**Effect:** `test_bayesian_recovery.py` raised `FileNotFoundError` at collection time
(collection error); `test_loo_stacking.py` had 6 test failures; `test_prior_predictive.py`
had 1 test failure.
**Fix:** Updated all 3 `_RUNNER_PATH` / `mod_path` assignments to
`... / "scripts" / "bayesian_pipeline" / "21_*.py"`.
**Files modified:** 3 test files
**Commit:** `907fa8c`

---

## Quarto Render

| File | Exit | PDF path | Warnings |
|------|------|----------|---------|
| manuscript/paper.qmd | 0 | manuscript/_output/paper.pdf | YAML indentation warning (pre-existing; author block line 7) |

Quarto exits 0 and produces `manuscript/_output/paper.pdf` despite the YAML warning.
The warning is a soft parse issue in the YAML front-matter affiliations block (line 7
has extra indentation for `name:` under the first author's affiliation). This is a
pre-existing issue — not introduced by Phase 28 — and does not prevent rendering.

---

## Requirement Closure Table

| REFAC ID | Description | Closed by plan | Verified via |
|----------|-------------|----------------|--------------|
| REFAC-01 | Delete environments/ + models/ shims | 28-01 | grep invariants 1+2 (zero matches); `git ls-files environments/ models/` = empty; pycache-only dirs remain on disk (gitignored) |
| REFAC-02 | Narrow scripts/fitting/ → src/rlwm/fitting/ migration | 28-01 | grep invariants 3+4+5 (zero matches); `ls src/rlwm/fitting/` contains jax_likelihoods.py, numpyro_models.py, numpyro_helpers.py; lba_likelihood.py intentionally excluded |
| REFAC-03 | Group scripts 01-04 under scripts/data_processing/ | 28-02 | `ls scripts/data_processing/` contains 01-04; `ls scripts/*.py` only shows 12/13/14 at top level |
| REFAC-04 | Group scripts 05-08 under scripts/behavioral/ | 28-03 | `ls scripts/behavioral/` contains 05-08 |
| REFAC-05 | Group scripts 09-11 under scripts/simulations_recovery/ | 28-04 | `ls scripts/simulations_recovery/` contains 09/10/11 |
| REFAC-06 | Group scripts 15-18 under scripts/post_mle/ | 28-05 | Functional subprocess call at 21_manuscript_tables.py:746 uses `scripts/post_mle/18_bayesian_level2_effects.py` (PASS) |
| REFAC-07 | Move scripts/21_*.py to scripts/bayesian_pipeline/ | 28-06 | `ls scripts/bayesian_pipeline/21_*.py` = 9 scripts; REFAC-07 test regression fixed in 907fa8c |
| REFAC-08 | figures/ + output/ scaffolding | 28-07 | `.gitkeep` files in figures/21_bayesian, output/bayesian/{21_baseline,21_l2,...} exist |
| REFAC-09 | Cluster SLURM consolidation 6→1 parameterized | 28-08 | grep invariant 6 (zero stale m[1-6].slurm refs); `ls cluster/13_bayesian_choice_only.slurm` exists; `ls cluster/13_bayesian_m{1,2,3,5}.slurm` = absent |
| REFAC-10 | validation/ + tests/ pruning | 28-09 | `ls validation/legacy/` = {check_phase_23_1_smoke.py, diagnose_gpu.py}; `ls tests/legacy/` = examples/ |
| REFAC-11 | paper.qmd Bayesian-first structural scaffolding | 28-10 | Results section order: Summary → Bayesian Model Selection (#sec-bayesian-selection) → Hierarchical L2 (#sec-bayesian-regression) → Subscale (#sec-subscale-breakdown); MLE in Appendix A-K |
| REFAC-12 | Docs refresh | 28-11 | CLAUDE.md Code Organization shows new subdirs; README.md Pipeline block uses new paths; docs/ updated |
| REFAC-13 | End-of-phase verification | 28-12 | This document |

---

## ROADMAP Success Criteria Assessment

| # | Criterion | Status | Evidence |
|---|-----------|--------|---------|
| SC1 | paper.qmd Results follows 5-section Bayesian-first order; MLE in Appendix | PASS | grep shows Results: Summary → Selection → L2 → Subscale; Appendix A-K hold MLE content |
| SC2 | src/ authoritative; no `from scripts.fitting.{jax,numpyro_*}` imports | PASS | grep invariants 3-5 zero; lba_likelihood exclusion by design (28-01-PLAN.md) |
| SC3 | Top-level numbered scripts in scripts/ substantially reduced | PASS | Only 12_fit_mle.py, 13_fit_bayesian.py, 14_compare_models.py remain at top level (3 vs original 16+) |
| SC4 | environments/ consolidated or deleted | PASS | `git ls-files environments/` = empty; pycache-only dir remains (gitignored, benign) |
| SC5 | cluster/ per-model SLURMs ≤ 4 parameterized | PASS | 6 choice-only templates deleted; 3 remain (m4_gpu + m6b_subscale + multigpu = M4 track); 13_bayesian_choice_only.slurm created |
| SC6 | validation/ + tests/ pruned with legacy/ subdirs | PASS | validation/legacy/ has 2 files; tests/legacy/examples/ exists |
| SC7 | quarto render produces paper.pdf without errors | PASS | Exit 0; manuscript/_output/paper.pdf confirmed; YAML warning pre-existing |
| SC8 | README pipeline block ≤ 20 lines | PASS | Pipeline block is 14 lines (counted); uses new script paths |
| SC9 | pytest passes clean | PARTIAL | Non-JAX suite: 69 pass / 1 skip. JAX-MCMC tests crash on Windows XLA backend (pre-existing environmental, not Phase 28 regression). REFAC-07 test regression found and fixed in 907fa8c. |
| SC10 | v4.0 closure invariants still pass | PASS | `check_v4_closure.py` exits 0 (5/5); `pytest test_v4_closure.py` passes 3/3 |

---

## Deviations from Plan

### Rule 1 — Bug: REFAC-07 test path regression (3 test files)

**Found during:** Task 3 (full pytest run)
**Scope:** `test_bayesian_recovery.py`, `test_loo_stacking.py`, `test_prior_predictive.py`
**Root cause:** Plan 28-06 moved 9 Phase 21 scripts via `git mv` to
`scripts/bayesian_pipeline/` but did not update the 3 test files that load those scripts
by absolute path via `importlib.util.spec_from_file_location`.
**Fix applied:** Updated `_RUNNER_PATH` / `mod_path` in all 3 files to include
`"bayesian_pipeline"` in the path. Docstring references updated to match.
**Tests affected (before fix):** 1 collection error + 6 FAIL + 1 FAIL = 8 items broken
**Tests after fix:** 15 pass (confirmed in targeted run)
**Commit:** `907fa8c fix(28-12): update stale 21_* script paths in tests after REFAC-07 move`

### Informational: Grep-7 docstring hits (not a functional gap)

3 docstring lines in `21_manuscript_tables.py` reference the old path
`scripts/18_bayesian_level2_effects.py` as historical text. The actual subprocess call
at line 746 uses the correct new path. No fix required.

### Informational: Windows JAX backend crash (pre-existing environmental)

Several JAX-MCMC tests abort with `Fatal Python error: Aborted` during XLA compilation
on Windows. This is a pre-existing issue documented in v4.0 STATE.md and is unrelated
to Phase 28 changes. Cluster execution is not affected.

---

## Remaining Open Items

1. **Grep-7 docstring hits** (3 lines in 21_manuscript_tables.py): harmless; could be
   cleaned up as a chore but not blocking.
2. **Windows JAX backend crash**: environmental, pre-existing; no action needed in Phase 28.
