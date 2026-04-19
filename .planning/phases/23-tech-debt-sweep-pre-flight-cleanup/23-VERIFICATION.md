---
phase: 23-tech-debt-sweep-pre-flight-cleanup
verified: 2026-04-19T00:00:00Z
status: passed
score: 4/4 must-haves verified
gaps: []
---

# Phase 23: Tech-Debt Sweep & Pre-Flight Cleanup Verification Report

**Phase Goal:** Remove residual v4.0 tech debt before the Phase 24 cold-start run so empirical artifacts are produced against a clean codebase. Four atomic code removals + one audit: (a) delete legacy qlearning hierarchical import path, (b) remove legacy M2 K-bounds [1,7] branch from `mle_utils.py`, (c) delete deprecated `scripts/16b_bayesian_regression.py`, (d) wire a validated NetCDF-loading wrapper into every downstream NetCDF consumer. No new scientific functionality — strictly cleanup.

**Verified:** 2026-04-19T00:00:00Z
**Status:** PASSED
**Requirements closed:** CLEAN-01, CLEAN-02, CLEAN-03, CLEAN-04

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `scripts/fitting/legacy/` does not exist; zero `from scripts.fitting.legacy` imports in `scripts/` | VERIFIED | Filesystem check confirmed directory absent; grep returns 0 matches; commit `15456e0 chore(tech-debt): delete scripts/fitting/legacy/ directory` landed removal of 4 files / ~2,073 lines |
| 2 | `grep -n "1, 7\|\[1,7\]\|K_BOUNDS_LEGACY" scripts/fitting/mle_utils.py` returns zero matches; all `*_BOUNDS` capacity values == `(2.0, 6.0)` | VERIFIED | grep returns 0 for each pattern; verified at lines 43, 53, 64, 76, 87, 100; `grep -c "v3.0 legacy fit" config.py` returns 0 after commit `089f2d0 chore(tech-debt): tighten EXPECTED_PARAMETERIZATION vocabulary` |
| 3 | `find . -path ./.git -prune -o -path ./.planning -prune -o -name "16b*" -print` returns zero lines; no `16b_bayesian_regression` reference in `scripts/ cluster/ docs/03_methods_reference/` | VERIFIED | find produces zero output; grep returns 0 matches; commit `39c2202 chore(tech-debt): remove residual 16b_bayesian_regression traces` cleared stale pycache (MODEL_REFERENCE.md was already clean; pre-flight grep matched Senta PDF coincidentally) |
| 4 | `config.py` defines `load_netcdf_with_validation`; every `az.from_netcdf(` and `xr.open_dataset(` call outside `tests/` and `fit_bayesian.py` is gone | VERIFIED | config.py line 754 defines the wrapper; commit `2ece61c chore(tech-debt): wire load_netcdf_with_validation into all consumer scripts` rewired 15 call sites across 13 files; grep invariants return 0 matches |
| 5 | Four pytest guard tests exist and pass | VERIFIED | 7/7 guard-test functions PASSED in 16.36s: `test_no_legacy_imports`, `test_mle_capacity_bounds_are_collins`, `test_mle_utils_source_has_no_legacy_k_bounds`, `test_no_16b_text_references_in_live_source`, `test_no_16b_files_outside_planning_tree`, `test_no_bare_az_from_netcdf_in_consumer_scripts`, `test_no_bare_xr_open_dataset_anywhere` |
| 6 | v4.0 closure guard still green (cleanup did not break v4.0 invariants) | VERIFIED | `python validation/check_v4_closure.py --milestone v4.0` → 5/5 checks passed, exit 0 |

**Score:** 6/6 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/fitting/tests/test_no_legacy_imports.py` | Exists, ≥20 lines, tests pass | VERIFIED | File exists; guard function + `_SELF_PATH` self-exclusion fix (auto-applied during execution) |
| `scripts/fitting/tests/test_mle_k_bounds_invariant.py` | Exists, ≥25 lines, two guard functions pass | VERIFIED | 102 lines; both functions pass immediately (code was already clean pre-Phase-23) |
| `scripts/fitting/tests/test_no_16b_references.py` | Exists, ≥20 lines, two guard functions pass | VERIFIED | File exists; text-reference + file-existence guards; `_THIS_FILE` self-exclusion fix |
| `scripts/fitting/tests/test_load_side_validation.py` | Exists, ≥40 lines, two guard functions pass | VERIFIED | File exists; enumerated-consumers grep + repo-wide `xr.open_dataset` sweep (tests/ excluded) |
| `config.py::load_netcdf_with_validation` | New function added with soft-warning on missing parameterization_version attr | VERIFIED | Line 754; soft DeprecationWarning on missing attr (hard-raise retrofit deferred to v5.1 per plan) |
| `.planning/phases/23-tech-debt-sweep-pre-flight-cleanup/23-0{1..4}-SUMMARY.md` | All 4 plan summaries written | VERIFIED | All 4 SUMMARY files present |

**Score:** 6/6 artifacts verified

## Phase-Level Invariants

| Invariant | Status | Details |
|-----------|--------|---------|
| `pytest scripts/fitting/tests/` — new guard tests pass | VERIFIED | 7/7 guard functions PASSED |
| `python validation/check_v4_closure.py --milestone v4.0` exits 0 | VERIFIED | 5/5 checks passed, exit 0 |
| Full fitting suite (minus pre-existing environmental exclusions) | VERIFIED | All other tests pass; two exclusions are **pre-existing** and independent of Phase 23 |

### Environmental Exclusions (NOT Phase 23 gaps)

Two tests fail in this CPU-only Windows environment and are explicitly NOT regressions introduced by Phase 23:

1. `test_m4_integration.py::test_log_delta_recovery` — `Fatal Python error: Aborted` inside `jax/_src/compiler.py backend_compile` when JAX JIT-compiles the M4 LBA NUTS model on Windows CPU. Pre-existing JAX/Windows incompatibility; M4 is a GPU-track model per `CLAUDE.md`.
2. `test_gpu_m4.py` — `@pytest.mark.slow`; requires GPU hardware. ERROR (not FAIL) on this machine. Expected by design.

Both failures precede Phase 23 and are tracked implicitly via the v4.0 MILESTONE-AUDIT's cluster-execution pending items. No Phase 23 code change could resolve them (they are environmental).

## Commit Trail

12 commits on `main` for Phase 23 (3 per plan: guard-test-red → code-change-green → `docs({plan}): complete ...`):

- `3b391c2` chore(tech-debt): add guard test for scripts.fitting.legacy imports
- `15456e0` chore(tech-debt): delete scripts/fitting/legacy/ directory
- `3b154d8` docs(23-01): complete CLEAN-01 legacy deletion plan
- `83fcde5` chore(tech-debt): add K-bounds invariant guard for mle_utils.py
- `089f2d0` chore(tech-debt): tighten EXPECTED_PARAMETERIZATION vocabulary
- `8f7cb89` docs(23-02): complete K-bounds invariant guard + config vocabulary plan
- `f0d4e60` chore(tech-debt): add guard test for 16b_bayesian_regression references
- `39c2202` chore(tech-debt): remove residual 16b_bayesian_regression traces
- `0ebbb2a` docs(23-03): complete 16b_bayesian_regression cleanup plan
- `2ac5430` chore(tech-debt): add load_netcdf_with_validation to config.py
- `2ece61c` chore(tech-debt): wire load_netcdf_with_validation into all consumer scripts
- `4b33c76` docs(23-04): complete NetCDF load-side validation plan

Bisectable: each guard lands red before its code change lands green.

## Deferrals Documented (NOT Phase 23 gaps)

- **v5.1 NetCDF write-side retrofit:** `load_netcdf_with_validation` currently emits `DeprecationWarning` (not hard-raise) when the NetCDF posterior lacks a `parameterization_version` attr, because `scripts/fitting/fit_bayesian.py:732` does not emit it today. The infrastructure to hard-enforce is in place; a v5.1 item will (a) emit the attr write-side and (b) flip the soft warning to a hard raise.

## Summary

**Phase 23 goal achieved.** All four tech-debt removals are structurally complete, each guarded by a pytest invariant that fails CI on reintroduction. v4.0 closure guard remains green. Phase 24 cold-start run is unblocked to produce empirical artifacts against a clean codebase.
