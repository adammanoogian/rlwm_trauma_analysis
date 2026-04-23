---
status: testing
phase: 29-pipeline-canonical-reorg
source:
  - 29-01-SUMMARY.md
  - 29-02-SUMMARY.md
  - 29-03-SUMMARY.md
  - 29-04-SUMMARY.md
  - 29-04b-SUMMARY.md
  - 29-05-SUMMARY.md
  - 29-06-SUMMARY.md
  - 29-07-SUMMARY.md
  - 29-08-SUMMARY.md
  - 29-VERIFICATION.md
started: 2026-04-23T00:00:00Z
updated: 2026-04-23T00:30:00Z
---

## Current Test

number: 4
name: Docs spare files merged into methods references
expected: |
  `docs/HIERARCHICAL_BAYESIAN.md`, `docs/K_PARAMETERIZATION.md`,
  `docs/SCALES_AND_FITTING_AUDIT.md` are GONE from `docs/` top level.
  Originals present at `docs/legacy/<name>`. `docs/CLUSTER_GPU_LESSONS.md`
  is still at top level and untouched (protected by user directive).
awaiting: user response

## Tests

### 1. Canonical 01-06 scripts/ stage layout
expected: `ls scripts/` shows 01_data_preprocessing through 06_fit_analyses stage dirs plus utils/, fitting/, legacy/. No Phase-28 grouping dirs remain at top level.
result: pass

### 2. Model-fitting sub-letter layout
expected: `scripts/04_model_fitting/` contains `a_mle/`, `b_bayesian/`, `c_level2/`. CLI entries `a_mle/fit_mle.py` and `b_bayesian/fit_bayesian.py` + `b_bayesian/fit_baseline.py` exist.
result: pass

### 3. Dead folders absent from scripts/ top level
expected: None of analysis/, results/, simulations/, statistical_analyses/, visualization/ exist at scripts/ top level. scripts/legacy/ was deleted in commit 5e1da2f (2026-04-23) after user-approved cleanup; git history via commit e574fed (Phase 29-04 archival) preserves the pre-archive file paths. Closure guard test_dead_folder_absent_from_top_level still passes (only asserts top-level absence).
result: pass

### 4. Docs spare files merged into methods references
expected: `docs/HIERARCHICAL_BAYESIAN.md`, `docs/K_PARAMETERIZATION.md`, `docs/SCALES_AND_FITTING_AUDIT.md` are GONE from `docs/` top level. Originals present at `docs/legacy/<name>`. `docs/CLUSTER_GPU_LESSONS.md` is still at top level and untouched (protected by user directive).
result: [pending]

### 5. Cluster SLURM consolidation
expected: `cluster/` contains `submit_all.sh` (master afterok orchestrator) plus stage-numbered entry scripts `0{1..6}*.slurm`. Bayesian per-model variants consolidated into `04b_bayesian_cpu.slurm` (dispatched via `--export=MODEL=<name>`) instead of 6 separate files. `bash cluster/submit_all.sh --dry-run` exits 0.
result: [pending]

### 6. Vertical-by-model library structure (29-08)
expected: `src/rlwm/fitting/` contains `core.py` (shared JAX primitives), `mle.py` + `bayesian.py` (engine entry points), `sampling.py` (MCMC orchestration), `numpyro_helpers.py` (hBayesDM helpers), and `models/{qlearning,wmrl,wmrl_m3,wmrl_m5,wmrl_m6a,wmrl_m6b,wmrl_m4}.py` — one file per Senta 2025 model.
result: [pending]

### 7. Shim files eliminated
expected: `src/rlwm/fitting/jax_likelihoods.py` and `src/rlwm/fitting/numpyro_models.py` DO NOT EXIST. Commit `d20bca6` deleted them after the shim period ended. The canonical import paths are `from rlwm.fitting.core import ...`, `from rlwm.fitting.models.wmrl_m6b import ...`, `from rlwm.fitting.mle import main`, etc.
result: [pending]

### 8. MLE + Bayesian CLIs still work end-to-end
expected: `python scripts/04_model_fitting/a_mle/fit_mle.py --model qlearning --help` exits 0 and prints help text. Same for `python scripts/04_model_fitting/b_bayesian/fit_bayesian.py --model qlearning --help`. No ImportError, no ModuleNotFoundError after the 29-08 relocation.
result: [pending]

### 9. v4 closure guard still green
expected: `python validation/check_v4_closure.py --milestone v4.0` exits 0 (5/5 checks PASS). `pytest scripts/fitting/tests/test_v4_closure.py -v` returns 3/3 PASS (test_v4_closure_passes, test_v4_closure_deterministic, test_v4_closure_rejects_wrong_milestone). Confirms no regression to milestone-v4.0 invariants.
result: [pending]

### 10. Phase 29 structure guard pins the canonical shape
expected: `pytest tests/test_v5_phase29_structure.py -v` returns 31/31 PASS in under 2 seconds. Tests cover: 6 stage-folder existence, sub-letters, 10 dead-folder absences, ppc single-source, 3 docs moved to legacy/, cluster_gpu_lessons sha256 unchanged, 5 no-old-grouping-imports, 3 utils short names.
result: [pending]

## Summary

total: 10
passed: 3
issues: 0
pending: 7
skipped: 0

## Gaps

[none yet]
