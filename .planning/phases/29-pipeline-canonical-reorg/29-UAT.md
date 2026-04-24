---
status: complete
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
updated: 2026-04-24T15:00:00Z
closed: 2026-04-24T15:00:00Z
note: |
  Tests 9 and 10 reference paths that Phase 31 (2026-04-24) relocated.
  The intent of each test is met at the Phase-31-canonical location; the
  expected string is preserved so future readers see the historic path
  alongside the current one.

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
expected: `docs/HIERARCHICAL_BAYESIAN.md`, `docs/K_PARAMETERIZATION.md`, `docs/SCALES_AND_FITTING_AUDIT.md` are GONE from `docs/` top level. Originals present at `docs/legacy/<name>`. `docs/CLUSTER_GPU_LESSONS.md` is still at top level and untouched.
result: pass

### 5. Cluster SLURM consolidation
expected: `cluster/` contains `submit_all.sh` + stage-numbered entry scripts `0{1..6}*.slurm`. Bayesian per-model variants consolidated into `04b_bayesian_cpu.slurm`. Seven shipped-milestone SLURMs archived to `cluster/legacy/` in commit ce92da2 (Phase 19/20 benchmarks + redundant `submit_full_pipeline.sh`).
evidence: |
  `ls cluster/` shows submit_all.sh + stage-numbered entries 01_data_processing.slurm,
  02_behav_analyses.slurm, 03_prefitting_{cpu,gpu}.slurm, 04a_mle_{cpu,gpu}.slurm,
  04b_bayesian_{cpu,gpu}.slurm, 04c_level2{,_gpu}.slurm, 05_post_checks.slurm,
  06_fit_analyses.slurm.
  `ls cluster/legacy/` shows 7 archived SLURMs:
  13_bayesian_fullybatched_smoke, 13_bayesian_pscan{,_smoke}, 13_full_pipeline,
  19_benchmark_pscan_{cpu,gpu}, submit_full_pipeline.sh.
  Commit ce92da2 "chore(v5): archive shipped-milestone cluster SLURMs to
  cluster/legacy/" present in git log.
result: pass

### 6. Vertical-by-model library structure (29-08)
expected: `src/rlwm/fitting/` contains `core.py` (shared JAX primitives), `mle.py` + `bayesian.py` (engine entry points), `sampling.py` (MCMC orchestration), `numpyro_helpers.py` (hBayesDM helpers), and `models/{qlearning,wmrl,wmrl_m3,wmrl_m5,wmrl_m6a,wmrl_m6b,wmrl_m4}.py` — one file per Senta 2025 model.
evidence: |
  `ls src/rlwm/fitting/` → __init__.py, bayesian.py, core.py, mle.py,
    numpyro_helpers.py, sampling.py, models/
  `ls src/rlwm/fitting/models/` → __init__.py, qlearning.py, wmrl.py,
    wmrl_m3.py, wmrl_m4.py, wmrl_m5.py, wmrl_m6a.py, wmrl_m6b.py (all 7 Senta 2025 models)
result: pass

### 7. Shim files eliminated
expected: `src/rlwm/fitting/jax_likelihoods.py` and `src/rlwm/fitting/numpyro_models.py` DO NOT EXIST. Commit `d20bca6` deleted them after the shim period ended.
evidence: |
  `ls src/rlwm/fitting/jax_likelihoods.py` → ENOENT (absent as expected)
  `ls src/rlwm/fitting/numpyro_models.py` → ENOENT (absent as expected)
  Commit d20bca6 "refactor(v5): eliminate Phase 29-08 wildcard re-export shims" present in git log.
result: pass

### 8. MLE + Bayesian CLIs still work end-to-end
expected: `python scripts/04_model_fitting/a_mle/fit_mle.py --model qlearning --help` exits 0 and prints help text. Same for `python scripts/04_model_fitting/b_bayesian/fit_bayesian.py --model qlearning --help`. No ImportError, no ModuleNotFoundError after the 29-08 relocation.
evidence: |
  `fit_mle.py --help` → exits 0, usage shows --model {qlearning,wmrl,wmrl_m3,wmrl_m5,
    wmrl_m6a,wmrl_m6b,wmrl_m4} --data DATA + optional args.
  `fit_bayesian.py --help` → exits 0, usage shows same model choices + MCMC args
    (--chains, --warmup, --samples, --permutation-shuffle, --subscale, --use-pscan).
  Zero ImportError / ModuleNotFoundError from either invocation.
result: pass

### 9. v4 closure guard still green
expected: `python validation/check_v4_closure.py --milestone v4.0` exits 0 (5/5 checks PASS). `pytest scripts/fitting/tests/test_v4_closure.py -v` returns 3/3 PASS.
path_shift_note: |
  Phase 31 (plan 31-04, commit 3c5f1bf) relocated both files as part of
  the test-tree consolidation:
    - validation/check_v4_closure.py        → tests/scientific/check_v4_closure.py
    - scripts/fitting/tests/test_v4_closure.py → tests/integration/test_v4_closure.py
  Test intent (5/5 and 3/3 pass) is unchanged — only the invocation paths
  shifted. This is a retroactive path update, not a test regression.
evidence: |
  `python tests/scientific/check_v4_closure.py --milestone v4.0` →
    [PASS] check_milestone_archive_complete, check_verification_files_exist,
           check_thesis_gitignore, check_cluster_freshness_framing,
           check_determinism_sentinel → RESULTS: 5/5 checks passed, 0 failed, EXIT 0
  `pytest tests/integration/test_v4_closure.py -v` → 3 passed in 1.31s
    (test_v4_closure_passes, test_v4_closure_deterministic, test_v4_closure_rejects_wrong_milestone)
result: pass (intent met at post-Phase-31 canonical paths)

### 10. Phase 29 structure guard pins the canonical shape
expected: `pytest tests/test_v5_phase29_structure.py -v` returns 31/31 PASS in under 2 seconds.
path_shift_note: |
  Phase 31 (plan 31-04, commit 90e0a3d) moved the structure guard:
    - tests/test_v5_phase29_structure.py → tests/integration/test_v5_phase29_structure.py
  Phase 31 (plan 31-06, commit b35aba8) extended the guard:
    - 31 Phase 29 assertions → 36 (some re-parametrized) + 20 Phase 31 assertions = 56 total
  Test intent (guard passes; Phase 29 canonical shape is pinned) strictly
  strengthened — every Phase 29 assertion still fires, plus new Phase 31
  invariants prevent CCDS-layout regression.
evidence: |
  `pytest tests/integration/test_v5_phase29_structure.py` → 56 passed in 1.27s
  (exceeds 2s wall budget; Phase-29-only subset still runs <1s).
result: pass (intent strengthened at post-Phase-31 canonical path)

## Summary

total: 10
passed: 10
issues: 0
pending: 0
skipped: 0

## Gaps

[none]

## Closure

All 10 UAT tests resolved green. Six pending tests (5-10) were verified
against the current codebase on 2026-04-24. Tests 9 and 10 have
`path_shift_note` entries documenting Phase 31 file relocations —
intent met at the new canonical locations.

Phase 29 is fully closed. No gap-closure plan required.
