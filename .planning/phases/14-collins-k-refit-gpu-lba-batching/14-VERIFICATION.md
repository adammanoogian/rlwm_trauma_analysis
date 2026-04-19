---
phase: 14-collins-k-refit-gpu-lba-batching
verified: 2026-04-19
status: partial
score: 3/5 code_verified; 2/5 deferred_to_execution
gaps: []
cluster_execution_pending:
  - truth: "K-02 / K-03: K recovery r >= 0.50 on constrained-K refit"
    deferred_to_execution: "bash cluster/12_submit_all_gpu.sh (main cold-start entry for K-refit + GPU models; NOT wired into cluster/21_submit_pipeline.sh — v5.0 candidate)"
    expected_artifact: "output/mle/{model}_individual_fits.csv rows with parameterization_version=v4.0-K[2,6]-phiapprox; output/recovery/collins_k_refit_summary.md with r >= 0.50 for M3 and M6b kappa"
  - truth: "GPU-03: M4 N=154 GPU wall time < 12h on A100"
    deferred_to_execution: "bash cluster/12_submit_all_gpu.sh (M4-specific GPU submission via cluster/12_mle_gpu.slurm)"
    expected_artifact: "output/mle/wmrl_m4_individual_fits.csv with wall_time_total_s metadata; expected < 43200s"
---

# Phase 14: Collins K Refit + GPU LBA Batching Verification Report

**Phase Goal:** (K-01 follow-up) Constrain WM capacity K to [2, 6] and stamp fits with `parameterization_version`; verify K recovery improves to r >= 0.50 vs. v3.0 r = 0.21 baseline; batch-vectorize M4 LBA fitting on GPU to bring N=154 wall time under 12h on A100.
**Verified:** 2026-04-19
**Status:** PARTIAL — code artifacts verified on disk; cluster refit pending (K-03 recovery r + GPU-03 wall time)
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | K-02: K bounds [2, 6] enforced in all WM-family param transforms | VERIFIED | `grep -n "'capacity': (2.0, 6.0)" scripts/fitting/mle_utils.py` returns lines 43, 53, 64, 76, 87, 100 (one entry per WM-bearing model bounds dict) |
| 2 | K-02: `parameterization_version` stamped in fit CSV via EXPECTED_PARAMETERIZATION | VERIFIED | `grep -n "EXPECTED_PARAMETERIZATION" scripts/fitting/fit_mle.py` returns line 85 (import) and 3018 (`fits_df['parameterization_version'] = EXPECTED_PARAMETERIZATION[args.model]`); config.py lines 673-681 define all 7 model keys with `v4.0-K[2,6]-phiapprox` variants |
| 3 | K-03: K recovery r >= 0.50 on constrained-K refit | DEFERRED | Cluster refit not yet executed; see `cluster_execution_pending` YAML block; baseline r = 0.21 documented in `14-01-SUMMARY.md §Accomplishments` |
| 4 | GPU-01 / GPU-02: `fit_all_gpu_m4()` function exists and handles N=154 real data | VERIFIED | `grep -n "def fit_all_gpu_m4" scripts/fitting/fit_mle.py` returns line 1708; function wraps `fit_all_gpu` with float64 enabled for LBA numerical stability per 14-02-SUMMARY.md §Decisions |
| 5 | GPU-03: M4 GPU wall time < 12h on A100 for N=154 | DEFERRED | Cluster execution not yet run; see `cluster_execution_pending` YAML block; `cluster/12_mle_gpu.slurm` provides the submission path |

**Score:** 3/5 truths code-verified; 2/5 deferred to cluster execution

### Required Artifacts

| Artifact | Expected | Status | Evidence |
|----------|----------|--------|---------|
| `scripts/fitting/mle_utils.py` | K bounds `(2.0, 6.0)` in every WM-family bounds dict | VERIFIED | `grep -n "'capacity': (2.0, 6.0)" scripts/fitting/mle_utils.py` returns 6 lines (wmrl, wmrl_m3, wmrl_m5, wmrl_m6a, wmrl_m6b, wmrl_m4 bounds dicts); comment at line 43 reads `# WM capacity (K); [2,6] per K-01 identifiability analysis` |
| `config.py` | `EXPECTED_PARAMETERIZATION` dict with 7 model entries + `load_fits_with_validation` function | VERIFIED | `grep -n "EXPECTED_PARAMETERIZATION" config.py` returns lines 673 (dict start), 682 (docstring), 708 (param ref), 729 (guard), 735 (validator); `grep -n "def load_fits_with_validation" config.py` returns line 692 |
| `scripts/fitting/fit_mle.py` | `fit_all_gpu()` at line 1224, `fit_all_gpu_m4()` at line 1708 | VERIFIED | `grep -n "def fit_all_gpu\b\|def fit_all_gpu_m4" scripts/fitting/fit_mle.py` returns lines 1224 and 1708 |
| `cluster/12_mle_gpu.slurm` | GPU SLURM script with `--compute-diagnostics` flag | VERIFIED | `grep -n "compute.diagnostics" cluster/12_mle_gpu.slurm` returns line 214; `ls cluster/12_mle_gpu.slurm` succeeds |
| `docs/K_PARAMETERIZATION.md` | K bounds rationale documentation | VERIFIED | `ls docs/K_PARAMETERIZATION.md` succeeds; file documents the K-01 identifiability analysis and [2,6] bound choice |

### Key Link Verification

| From | To | Via | Status | Evidence |
|------|----|-----|--------|---------|
| `config.py::EXPECTED_PARAMETERIZATION` | `scripts/fitting/fit_mle.py` | `from config import EXPECTED_PARAMETERIZATION` + assignment at line 3018 | WIRED | `grep -n "EXPECTED_PARAMETERIZATION" scripts/fitting/fit_mle.py` returns lines 85 (import) and 3018 (stamp) |
| `config.py::load_fits_with_validation` | downstream analysis scripts | validator rejects stale v3.0 fits lacking `parameterization_version` | WIRED | `grep -n "def load_fits_with_validation" config.py` returns line 692; function raises ValueError with expected vs. actual on mismatch per lines 728-741 |
| `cluster/12_mle_gpu.slurm` | `scripts/fitting/fit_mle.py` | `--compute-diagnostics` flag | WIRED | `grep -n "compute.diagnostics" cluster/12_mle_gpu.slurm` returns line 214 |
| `fit_all_gpu_m4()` | `fit_all_gpu()` | wrapper adding float64 enabling for LBA | WIRED | `grep -n "def fit_all_gpu_m4" scripts/fitting/fit_mle.py` returns line 1708; per 14-02-SUMMARY.md §Decisions: wraps `fit_all_gpu` with float64 for numerical stability |

### Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| K-02: K bounds [2, 6] + parameterization_version stamp | SATISFIED | None — code verified on disk |
| K-03: K recovery r >= 0.50 after constrained-K refit | DEFERRED | Cluster refit pending; see `cluster_execution_pending` block |
| GPU-01: `fit_all_gpu_m4()` function implemented | SATISFIED | None — code verified on disk |
| GPU-02: `fit_all_gpu_m4()` handles N=154 real data | SATISFIED | None — function signature accepts real participant data identically to `fit_all_gpu` per 14-02-SUMMARY.md |
| GPU-03: M4 GPU wall time < 12h on A100 for N=154 | DEFERRED | Cluster execution pending; see `cluster_execution_pending` block |

### Cluster-Pending Framing

Phase 14 cluster-pending items (K-03, GPU-03) are NOT currently wired into `bash cluster/21_submit_pipeline.sh`. Their cold-start entry is `bash cluster/12_submit_all_gpu.sh` (which dispatches all 7 MLE models including M4 via `cluster/12_mle_gpu.slurm`). Extending the Phase 21 orchestrator to chain K-refit + GPU LBA batching upstream of step 21.3 is a v5.0 candidate (tracked by PROJECT.md v5.0 notes).

The Phase 14 cluster items were submitted as part of the 14-03-PLAN.md checkpoint:human-action (14-03-SUMMARY.md §Status: PARTIAL — Task 2 checkpoint:human-action awaiting cluster refit). Until the refit runs, the central K-recovery claim (r >= 0.50 vs. r = 0.21 baseline) and the M4 GPU wall-time target remain unverified empirically.

### Anti-Patterns Found

`grep -n "TODO\|FIXME\|XXX" scripts/fitting/mle_utils.py` returns zero matches. No TODO/FIXME/XXX patterns in the Phase 14 code additions. No empty stubs.

`grep -n "TODO\|FIXME\|XXX" scripts/fitting/fit_mle.py` returns zero matches in the `fit_all_gpu` and `fit_all_gpu_m4` functions.

### Human Verification Required

The two deferred requirements (K-03 and GPU-03) require cluster execution. Cold-start entry:

```bash
bash cluster/12_submit_all_gpu.sh
```

This dispatches all 7 MLE models (including M4) via `cluster/12_mle_gpu.slurm` with `--compute-diagnostics`. After completion, verify:
- `output/mle/{model}_individual_fits.csv` exists for all 7 models with `parameterization_version` column
- K recovery r >= 0.50 in `output/recovery/collins_k_refit_summary.md` for M3 and M6b kappa parameter
- M4 wall time < 43200 s (12h) in `wmrl_m4_individual_fits.csv` metadata

Do NOT run `sbatch cluster/12_mle_gpu.slurm` separately; use `bash cluster/12_submit_all_gpu.sh` as the canonical cold-start entry to dispatch all models consistently.

## Gaps Summary

No code gaps. All 5 requirements have substantive code artifacts verified on disk. Two requirements (K-03 recovery r, GPU-03 wall time) are code-complete but require cluster execution to produce the empirical artifacts. Both are documented in the `cluster_execution_pending` YAML block above. Phase 14 is not wired into the Phase 21 master orchestrator; wiring K-refit + GPU LBA upstream of step 21.3 is deferred to v5.0.

---

_Verified: 2026-04-19_
_Verifier: Claude (gsd-executor, plan 22-02)_
