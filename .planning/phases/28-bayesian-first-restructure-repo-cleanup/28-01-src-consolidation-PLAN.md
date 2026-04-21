---
wave: 1
depends_on: []
files_modified:
  - environments/  (deleted)
  - models/  (deleted)
  - src/rlwm/fitting/jax_likelihoods.py  (git mv from scripts/fitting/jax_likelihoods.py)
  - src/rlwm/fitting/numpyro_models.py  (git mv from scripts/fitting/numpyro_models.py)
  - src/rlwm/fitting/numpyro_helpers.py  (git mv from scripts/fitting/numpyro_helpers.py)
  - src/rlwm/fitting/__init__.py  (new)
  - scripts/simulations/generate_data.py
  - scripts/simulations/parameter_sweep.py
  - scripts/simulations/unified_simulator.py
  - scripts/09_run_ppc.py
  - scripts/11_run_model_recovery.py
  - scripts/12_fit_mle.py
  - scripts/13_fit_bayesian.py
  - scripts/21_fit_baseline.py
  - scripts/21_compute_loo_stacking.py
  - scripts/21_run_bayesian_recovery.py
  - scripts/21_run_prior_predictive.py
  - scripts/21_fit_with_l2.py
  - scripts/fitting/fit_mle.py
  - scripts/fitting/fit_bayesian.py
  - scripts/fitting/bayesian_diagnostics.py
  - scripts/fitting/model_recovery.py
  - scripts/fitting/warmup_jit.py
  - scripts/utils/remap_mle_ids.py
  - scripts/fitting/tests/test_mle_quick.py
  - scripts/fitting/tests/test_m3_hierarchical.py
  - scripts/fitting/tests/test_pscan_likelihoods.py
  - scripts/fitting/tests/test_numpyro_helpers.py
  - scripts/fitting/tests/test_compile_gate.py
  - scripts/fitting/tests/test_m4_hierarchical.py
  - scripts/fitting/tests/test_m4_integration.py
  - scripts/fitting/tests/test_numpyro_models_2cov.py
  - scripts/fitting/tests/test_pointwise_loglik.py
  - scripts/fitting/tests/test_prior_predictive.py
  - scripts/fitting/tests/test_wmrl_model.py
  - validation/benchmark_parallel_scan.py
  - validation/test_m3_backward_compat.py
  - tests/examples/example_parameter_sweep.py
  - tests/examples/explore_prior_parameter_space.py
  - tests/examples/interactive_exploration.py
  - tests/test_rlwm_package.py
  - tests/test_wmrl_exploration.py
  - validation/test_parameter_recovery.py
  - validation/test_model_consistency.py
  - validation/test_unified_simulator.py
  - README.md  (dev-setup section only — `pip install -e .`)
autonomous: true
---

# 28-01 src/ Consolidation: Delete Shims + Narrow Fitting Migration

## Goal

Delete the `environments/` and `models/` backward-compat shim packages outright, move pure-library fitting math (`jax_likelihoods.py`, `numpyro_models.py`, `numpyro_helpers.py`) from `scripts/fitting/` to `src/rlwm/fitting/` preserving git history, and update every call site in one atomic commit so the repo never rests in a broken intermediate state.

## Must Haves

- [ ] `environments/` top-level directory deleted entirely (it contained only shims delegating to `rlwm.envs`).
- [ ] `models/` top-level directory deleted entirely (it contained only shims delegating to `rlwm.models`).
- [ ] `src/rlwm/fitting/` directory exists and contains `jax_likelihoods.py`, `numpyro_models.py`, `numpyro_helpers.py` moved via `git mv` (history preserved for `git log --follow`).
- [ ] `src/rlwm/fitting/__init__.py` created (empty or with module docstring — do NOT add re-exports; consumers import specific submodules).
- [ ] Zero files remain in `scripts/fitting/` named `jax_likelihoods.py`, `numpyro_models.py`, or `numpyro_helpers.py`.
- [ ] All 19 call sites enumerated in 28-RESEARCH.md §Q1 "Files importing from top-level shims" updated to import from `rlwm.envs.*` / `rlwm.models.*` directly.
- [ ] All 22 call sites depending on `scripts.fitting.jax_likelihoods`, `scripts.fitting.numpyro_models`, `scripts.fitting.numpyro_helpers` updated to `rlwm.fitting.*` — enumerated below (broader than the original 28-RESEARCH.md §"Import dependency snapshot" which missed tests, validation, warmup_jit, remap_mle_ids, and 21_fit_with_l2.py):
  - Top-level scripts (6): `scripts/09_run_ppc.py`, `scripts/11_run_model_recovery.py`, `scripts/12_fit_mle.py`, `scripts/13_fit_bayesian.py`, `scripts/21_fit_baseline.py`, `scripts/21_compute_loo_stacking.py` — **note: grep confirms these actually have zero matches, kept in list to document the verification sweep**
  - Top-level Phase-21 scripts with live imports (3): `scripts/21_run_bayesian_recovery.py`, `scripts/21_run_prior_predictive.py`, `scripts/21_fit_with_l2.py`
  - `scripts/fitting/` orchestrators (5): `fit_mle.py`, `fit_bayesian.py`, `bayesian_diagnostics.py`, `model_recovery.py` (verify — grep shows zero matches), `warmup_jit.py`
  - `scripts/utils/remap_mle_ids.py` (1)
  - `scripts/fitting/tests/` (11): `test_mle_quick.py`, `test_m3_hierarchical.py`, `test_pscan_likelihoods.py`, `test_numpyro_helpers.py`, `test_compile_gate.py`, `test_m4_hierarchical.py`, `test_m4_integration.py`, `test_numpyro_models_2cov.py`, `test_pointwise_loglik.py`, `test_prior_predictive.py`, `test_wmrl_model.py`
  - `validation/` (2): `benchmark_parallel_scan.py` (LOAD-BEARING — called by `cluster/13_bayesian_pscan.slurm` and `13_bayesian_pscan_smoke.slurm`; Plan 28-09 explicitly retains), `test_m3_backward_compat.py`
- [ ] `tests/test_rlwm_package.py` shim-specific test methods (currently testing `from environments.*` and `from models.*` at lines 82–114) rewritten to test the canonical `rlwm.envs.*` and `rlwm.models.*` paths directly.
- [ ] `README.md` dev-setup section mentions `pip install -e .` as a prerequisite for running tests.
- [ ] `grep -r "from environments\." scripts/ tests/ validation/ src/` returns zero matches.
- [ ] `grep -r "^from models\." scripts/ tests/ validation/ src/` returns zero matches (regex anchored to avoid false positives on variable names).
- [ ] `grep -r "from scripts.fitting.jax_likelihoods" . --include="*.py"` returns zero matches.
- [ ] `grep -r "from scripts.fitting.numpyro_models" . --include="*.py"` returns zero matches.
- [ ] `grep -r "from scripts.fitting.numpyro_helpers" . --include="*.py"` returns zero matches.
- [ ] `pytest scripts/fitting/tests/test_v4_closure.py` still passes 3/3.
- [ ] `tests/test_wmrl_exploration.py` collection error resolved as a side-effect (imports rewritten to `from rlwm.envs import create_rlwm_env`; the pre-existing `ModuleNotFoundError: No module named 'rlwm'` is gone once `pip install -e .` is run in the test env).
- [ ] Atomic single commit: `refactor(28-01): delete environments/ and models/ shims, migrate fitting core to src/rlwm/fitting/`.

## Tasks

<tasks>
  <task id="1">
    <title>Install package in editable mode for local verification</title>
    <detail>Run `pip install -e .` in the active conda env so `import rlwm` resolves. Required for pytest after shim deletion. No commit artifacts from this step.</detail>
  </task>

  <task id="2">
    <title>Create src/rlwm/fitting/ package skeleton</title>
    <detail>Create `src/rlwm/fitting/__init__.py` as an empty file (or with a module docstring only). Do NOT add re-exports — consumers will import specific submodules. Leave `src/rlwm/__init__.py` alone.</detail>
  </task>

  <task id="3">
    <title>git mv pure-library fitting modules from scripts/fitting/ to src/rlwm/fitting/</title>
    <detail>Exactly three moves via `git mv`:
      - `git mv scripts/fitting/jax_likelihoods.py src/rlwm/fitting/jax_likelihoods.py`
      - `git mv scripts/fitting/numpyro_models.py src/rlwm/fitting/numpyro_models.py`
      - `git mv scripts/fitting/numpyro_helpers.py src/rlwm/fitting/numpyro_helpers.py`
      Do NOT move the 12 orchestrator modules (fit_mle.py, fit_bayesian.py, mle_utils.py, bms.py, model_recovery.py, bayesian_diagnostics.py, bayesian_summary_writer.py, lba_likelihood.py, level2_design.py, warmup_jit.py, aggregate_permutation_results.py, compare_mle_models.py) — they stay in scripts/fitting/ per planning_context §1.</detail>
  </task>

  <task id="4">
    <title>Update internal imports inside moved modules</title>
    <detail>`numpyro_models.py` imports from `jax_likelihoods` (per 28-RESEARCH.md §"Within scripts/fitting/ (internal)"). Change `from scripts.fitting.jax_likelihoods import ...` (or `from jax_likelihoods import ...`) to `from rlwm.fitting.jax_likelihoods import ...` or relative `from .jax_likelihoods import ...`. Apply same pattern to `numpyro_helpers.py` if it cross-references the other two. Verify with Grep after editing.</detail>
  </task>

  <task id="5">
    <title>Update scripts/fitting/ orchestrators' imports</title>
    <detail>Change `from scripts.fitting.jax_likelihoods import ...` → `from rlwm.fitting.jax_likelihoods import ...` (and similarly for numpyro_models, numpyro_helpers) in:
      - `scripts/fitting/fit_mle.py` (uses jax_likelihoods)
      - `scripts/fitting/fit_bayesian.py` (uses numpyro_models)
      - `scripts/fitting/bayesian_diagnostics.py` (uses jax_likelihoods)
      - `scripts/fitting/model_recovery.py` (verify)</detail>
  </task>

  <task id="6">
    <title>Update top-level scripts/ call sites for moved fitting modules</title>
    <detail>Per 28-RESEARCH.md §"Import dependency snapshot" top-edges table, rewrite `from scripts.fitting.<module>` → `from rlwm.fitting.<module>` in:
      - `scripts/12_fit_mle.py` (verify)
      - `scripts/13_fit_bayesian.py` (verify)
      - `scripts/09_run_ppc.py` (verify)
      - `scripts/11_run_model_recovery.py` (verify)
      - `scripts/21_fit_baseline.py` (verify)
      - `scripts/21_compute_loo_stacking.py` (verify)
      - `scripts/21_run_bayesian_recovery.py` — imports `scripts.fitting.numpyro_helpers`
      - `scripts/21_run_prior_predictive.py` — imports `scripts.fitting.numpyro_models` AND `scripts.fitting.numpyro_helpers`
      - `scripts/21_fit_with_l2.py` line ~270 — imports `scripts.fitting.numpyro_models` (inside a function; rewrite to `rlwm.fitting.numpyro_models`)
      Final grep sweep: `grep -rn "scripts\.fitting\.\(jax_likelihoods\|numpyro_models\|numpyro_helpers\)" scripts/ tests/ validation/ src/` must return zero matches.</detail>
  </task>

  <task id="6b">
    <title>Update scripts/fitting/warmup_jit.py and scripts/utils/remap_mle_ids.py imports</title>
    <detail>Both files have live `from scripts.fitting.jax_likelihoods import ...` imports that must be rewritten to `from rlwm.fitting.jax_likelihoods import ...`:
      - `scripts/fitting/warmup_jit.py` line 59 (inside a function — preserve surrounding `if` / `try` structure)
      - `scripts/utils/remap_mle_ids.py` line 31
      Both files STAY in their current locations (orchestrator + utility, not pure-library math).</detail>
  </task>

  <task id="6c">
    <title>Update 11 test files under scripts/fitting/tests/</title>
    <detail>Rewrite every `from scripts.fitting.{jax_likelihoods,numpyro_models,numpyro_helpers}` import to `from rlwm.fitting.{...}` in:
      - `scripts/fitting/tests/test_mle_quick.py` (line 14 — jax_likelihoods)
      - `scripts/fitting/tests/test_m3_hierarchical.py` (lines 45, 94, 153, 191, 195, 299, 303, 388, 420, 484, 553, 621 — mix of jax_likelihoods + numpyro_models; do a single grep-replace sweep per module name)
      - `scripts/fitting/tests/test_pscan_likelihoods.py` (lines 29, 909, 966, 1016, 1495)
      - `scripts/fitting/tests/test_numpyro_helpers.py` (line 26)
      - `scripts/fitting/tests/test_compile_gate.py` (lines 84, 122)
      - `scripts/fitting/tests/test_m4_hierarchical.py` (lines 16, 17)
      - `scripts/fitting/tests/test_m4_integration.py` (line 22)
      - `scripts/fitting/tests/test_numpyro_models_2cov.py` (lines 78, 455, 557)
      - `scripts/fitting/tests/test_pointwise_loglik.py` (line 8)
      - `scripts/fitting/tests/test_prior_predictive.py` (lines 17, 36, 79)
      - `scripts/fitting/tests/test_wmrl_model.py` (line 22)
      Approach: Use Grep → Edit per file, or a scripted sed pass on the whole tests/ dir. Verify afterward with `grep -rn "scripts\.fitting\.\(jax_likelihoods\|numpyro_models\|numpyro_helpers\)" scripts/fitting/tests/` → zero matches.
      **Baseline impact:** Without this task, pytest collection of these 11 files fails with ImportError, dropping baseline from 204 passing tests.</detail>
  </task>

  <task id="6d">
    <title>Update 2 validation/ files</title>
    <detail>Rewrite `from scripts.fitting.jax_likelihoods import ...` → `from rlwm.fitting.jax_likelihoods import ...` in:
      - `validation/benchmark_parallel_scan.py` line 50 — **LOAD-BEARING**: invoked by `cluster/13_bayesian_pscan.slurm` and `cluster/13_bayesian_pscan_smoke.slurm`; Plan 28-09 retains this file. Breaking this import breaks the pscan SLURM jobs at submission time.
      - `validation/test_m3_backward_compat.py` line 26.</detail>
  </task>

  <task id="7">
    <title>Update scripts/simulations/ shim call sites (3 files)</title>
    <detail>Per 28-RESEARCH.md §Q1:
      - `scripts/simulations/generate_data.py` lines 38–41: `from environments.rlwm_env import ...` → `from rlwm.envs import ...`; `from models.q_learning import ...` → `from rlwm.models.q_learning import ...`; `from models.wm_rl_hybrid import ...` → `from rlwm.models.wm_rl_hybrid import ...`. Preserve exact imported symbols.
      - `scripts/simulations/parameter_sweep.py` lines 41–43: same pattern.
      - `scripts/simulations/unified_simulator.py` lines 20–22: same pattern.</detail>
  </task>

  <task id="8">
    <title>Update tests/examples/ shim call sites (3 files)</title>
    <detail>Rewrite `from environments.rlwm_env import ...` → `from rlwm.envs import ...` and `from models.* import ...` → `from rlwm.models.* import ...` in:
      - `tests/examples/example_parameter_sweep.py` lines 19–21
      - `tests/examples/explore_prior_parameter_space.py` lines 59–61
      - `tests/examples/interactive_exploration.py` lines 26–28
      NOTE: Plan 28-09 (REFAC-10) will move `tests/examples/` to `tests/legacy/examples/`; leave files in place here — this plan only touches imports.</detail>
  </task>

  <task id="9">
    <title>Update tests/ top-level shim call sites</title>
    <detail>
      - `tests/test_wmrl_exploration.py` lines 14–15: rewrite to `rlwm.envs` / `rlwm.models.*`. Also check for `beta`/`beta_wm` old parameter API usage (per 28-RESEARCH.md §tests/); if present, update to current API or mark xfail with a reason. This resolves the pre-existing collection error.
      - `tests/test_rlwm_package.py` lines 82–114 (the shim test methods): these methods explicitly test `from environments.*` and `from models.*` work. After shim deletion those tests become meaningless. Rewrite each method to test the canonical `rlwm.envs.*` and `rlwm.models.*` paths directly; preserve test semantics (import succeeds, class instantiable, basic method works).</detail>
  </task>

  <task id="10">
    <title>Update validation/ shim call sites (3 files)</title>
    <detail>
      - `validation/test_parameter_recovery.py` lines 26–28: rewrite `from environments.rlwm_env import create_rlwm_env` → `from rlwm.envs import create_rlwm_env`; `from models.*` → `from rlwm.models.*`.
      - `validation/test_model_consistency.py` lines 17–18: rewrite `from models.*` → `from rlwm.models.*`.
      - `validation/test_unified_simulator.py` line 21: rewrite `from environments.rlwm_env import create_rlwm_env` → `from rlwm.envs import create_rlwm_env`.</detail>
  </task>

  <task id="11">
    <title>Delete environments/ and models/ shim packages</title>
    <detail>Only after tasks 7–10 are staged:
      - `git rm -r environments/`
      - `git rm -r models/`
      Do NOT delete before staging all call-site updates; otherwise intermediate pytest state is broken.</detail>
  </task>

  <task id="12">
    <title>Update README.md dev-setup</title>
    <detail>Add a short dev-setup bullet to README.md noting that `pip install -e .` is required to make `rlwm` importable (pytest's `pythonpath = .` only adds repo root; `src/` layout requires an editable install). Keep this minimal — REFAC-12 (plan 28-11) handles the full docs refresh; this plan only adds the one prerequisite line so tests pass post-shim-deletion.</detail>
  </task>

  <task id="13">
    <title>Local verification before commit</title>
    <detail>Run:
      - `grep -rn "^from environments\." scripts/ tests/ validation/ src/` — expect zero matches
      - `grep -rn "^from models\." scripts/ tests/ validation/ src/` — expect zero matches
      - `grep -rn "scripts\.fitting\.\(jax_likelihoods\|numpyro_models\|numpyro_helpers\)" scripts/ tests/ validation/ src/` — expect zero matches (sweeps ALL 22 call sites enumerated above)
      - `pytest scripts/fitting/tests/test_v4_closure.py -v` — expect 3/3 PASS
      - `pytest scripts/fitting/tests/ --collect-only` — expect clean collection of all 11 touched test files (no ImportError from stale `scripts.fitting.jax_likelihoods` etc. paths)
      - `pytest tests/test_wmrl_exploration.py --collect-only` — expect clean collection (no ModuleNotFoundError)</detail>
  </task>

  <task id="14">
    <title>Atomic commit</title>
    <detail>Stage all changes and commit with one coherent message:
      `refactor(28-01): delete environments/ and models/ shims, migrate fitting core to src/rlwm/fitting/`
      Body bullets: list the 3 moved modules, the 2 deleted shim packages, the call-site update count, and that this resolves the pre-existing test_wmrl_exploration.py collection error.</detail>
  </task>
</tasks>

## Verification

```bash
# Shim deletion
test ! -d environments/ && test ! -d models/

# Moved fitting modules
test -f src/rlwm/fitting/jax_likelihoods.py
test -f src/rlwm/fitting/numpyro_models.py
test -f src/rlwm/fitting/numpyro_helpers.py
test ! -f scripts/fitting/jax_likelihoods.py
test ! -f scripts/fitting/numpyro_models.py
test ! -f scripts/fitting/numpyro_helpers.py

# Grep invariants (all must return zero)
grep -rn "^from environments\." scripts/ tests/ validation/ src/
grep -rn "^from models\." scripts/ tests/ validation/ src/
grep -rn "scripts\.fitting\.\(jax_likelihoods\|numpyro_models\|numpyro_helpers\)" scripts/ tests/ validation/ src/

# Git history preserved on the 3 moved files
git log --follow --oneline src/rlwm/fitting/jax_likelihoods.py | head -3
git log --follow --oneline src/rlwm/fitting/numpyro_models.py | head -3
git log --follow --oneline src/rlwm/fitting/numpyro_helpers.py | head -3

# V4 closure guard
pytest scripts/fitting/tests/test_v4_closure.py -v

# Full test collection no longer errors
pytest --collect-only tests/test_wmrl_exploration.py
```

## Requirement IDs

Closes: **REFAC-01**, **REFAC-02** (the narrow-migration slice).
