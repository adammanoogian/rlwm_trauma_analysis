---
wave: 3
depends_on: [28-01]
files_modified:
  - validation/test_fitting_quick.py  (deleted)
  - validation/legacy/check_phase_23_1_smoke.py  (git mv from validation/check_phase_23_1_smoke.py)
  - validation/legacy/diagnose_gpu.py  (git mv from validation/diagnose_gpu.py)
  - tests/legacy/examples/  (git mv from tests/examples/)
  - validation/README.md  (document legacy/ subdirectory)
autonomous: true
---

# 28-09 Prune validation/ + tests/ Directories

## Goal

Remove dead tests and move legacy-but-referenced files into `legacy/` subdirectories so the active validation/tests surface reflects only load-bearing invariants of the current pipeline.

## Must Haves

- [ ] `validation/test_fitting_quick.py` deleted outright (it self-skips as legacy per 28-RESEARCH.md §Q4; zero test value).
- [ ] `validation/legacy/` directory exists and contains:
  - `validation/legacy/check_phase_23_1_smoke.py` (moved via `git mv` from `validation/`; Phase 23.1 is complete, no active caller)
  - `validation/legacy/diagnose_gpu.py` (moved via `git mv`; pre-Phase 21 diagnostic, no active caller)
- [ ] `tests/legacy/` directory exists and contains:
  - `tests/legacy/examples/` (moved via `git mv` from `tests/examples/`; all 5 files are interactive exploration scripts, not pytest tests — per 28-RESEARCH.md §tests/)
- [ ] **Retained in validation/** (load-bearing per 28-RESEARCH.md §Q4):
  - `check_v4_closure.py`, `benchmark_parallel_scan.py` (called by `13_bayesian_pscan*.slurm`), `compare_posterior_to_mle.py` (referenced by paper.qmd line ~1000), `conftest.py`, `test_m3_backward_compat.py`, `test_model_consistency.py`, `test_parameter_recovery.py`, `test_unified_simulator.py`, `README.md`, `__init__.py`.
- [ ] **Retained in tests/** (load-bearing):
  - `test_period_env.py`, `test_rlwm_package.py`, `test_wmrl_exploration.py`, `test_performance_plots.py`, `__init__.py`.
- [ ] `validation/README.md` updated to document the `legacy/` subdirectory and the deletion of `test_fitting_quick.py`.
- [ ] Full pytest suite passes: expected 204 tests; the pre-existing 1 collection error is resolved by plan 28-01 (shim removal); this plan introduces 0 new failures.
- [ ] `pytest scripts/fitting/tests/test_v4_closure.py` still passes 3/3.
- [ ] Atomic commit: `refactor(28-09): prune validation/ + tests/ — delete test_fitting_quick.py, move legacy files to legacy/`.

## Tasks

<tasks>
  <task id="1">
    <title>Pre-flight: confirm nothing imports the files being moved/deleted</title>
    <detail>Run:
      - `grep -rn "test_fitting_quick\|check_phase_23_1_smoke\|diagnose_gpu" . --include="*.py" --include="*.sh" --include="*.slurm"`
      - `grep -rn "tests.examples\|tests/examples" . --include="*.py" --include="*.sh" --include="*.slurm"`
      Expected: zero in-code refs. `tests/examples/` files are interactive scripts (not imported). `diagnose_gpu.py` is a standalone script. Any match must be evaluated before move.</detail>
  </task>

  <task id="2">
    <title>Delete validation/test_fitting_quick.py</title>
    <detail>`git rm validation/test_fitting_quick.py`. The file contains only `pytest.skip("Legacy fitting test — fit_both_models and numpyro_models modules no longer exist")`.</detail>
  </task>

  <task id="3">
    <title>Create validation/legacy/ and move two legacy scripts</title>
    <detail>
      - `mkdir -p validation/legacy/`
      - `git mv validation/check_phase_23_1_smoke.py validation/legacy/check_phase_23_1_smoke.py`
      - `git mv validation/diagnose_gpu.py validation/legacy/diagnose_gpu.py`</detail>
  </task>

  <task id="4">
    <title>Create tests/legacy/ and move tests/examples/ en bloc</title>
    <detail>
      - `mkdir -p tests/legacy/`
      - `git mv tests/examples tests/legacy/examples`
      This preserves all 5 files in tests/legacy/examples/. No content edits; plan 28-01 already updated their imports to `rlwm.*`.</detail>
  </task>

  <task id="5">
    <title>Update validation/README.md</title>
    <detail>Edit `validation/README.md` — add a short "Legacy files" section noting:
      - `validation/legacy/check_phase_23_1_smoke.py` — Phase 23.1 smoke guard, superseded
      - `validation/legacy/diagnose_gpu.py` — pre-Phase-21 GPU diagnostic
      - `validation/test_fitting_quick.py` deleted (was a self-skipping legacy test)
      Keep it short (3-5 lines).</detail>
  </task>

  <task id="6">
    <title>Run full pytest to verify pruning introduced zero new failures</title>
    <detail>`pytest -x --tb=short`. Expected: 204 passed, 1 skipped, 0 errors. The pre-existing `tests/test_wmrl_exploration.py` collection error is already resolved by plan 28-01. If this plan introduces any new failures, investigate before committing.</detail>
  </task>

  <task id="7">
    <title>Atomic commit</title>
    <detail>`refactor(28-09): prune validation/ + tests/ — delete test_fitting_quick.py, move legacy files to legacy/`. Body lists the 1 deletion, 2 validation moves, 1 tests/examples move, and the README.md update.</detail>
  </task>
</tasks>

## Verification

```bash
# Deletion
test ! -f validation/test_fitting_quick.py

# Legacy moves
test -f validation/legacy/check_phase_23_1_smoke.py
test -f validation/legacy/diagnose_gpu.py
test -d tests/legacy/examples
test -f tests/legacy/examples/example_parameter_sweep.py

# Originals gone
test ! -f validation/check_phase_23_1_smoke.py
test ! -f validation/diagnose_gpu.py
test ! -d tests/examples

# Load-bearing files retained
test -f validation/check_v4_closure.py
test -f validation/benchmark_parallel_scan.py
test -f validation/compare_posterior_to_mle.py
test -f validation/test_m3_backward_compat.py
test -f validation/test_model_consistency.py
test -f validation/test_parameter_recovery.py
test -f validation/test_unified_simulator.py
test -f tests/test_period_env.py
test -f tests/test_rlwm_package.py
test -f tests/test_wmrl_exploration.py
test -f tests/test_performance_plots.py

# Full pytest
pytest -x --tb=short

# v4 closure
pytest scripts/fitting/tests/test_v4_closure.py -v
```

## Requirement IDs

Closes: **REFAC-10**.
