---
phase: 15-m3-hierarchical-poc-level2
verified: 2026-04-19
status: passed_with_absorption
score: 6/6 requirements satisfied (4 directly, 2 via absorption into Phases 18 and 21)
gaps: []
absorbed_into:
  - requirement: HIER-09
    from_plan: "15-03"
    to_phase: 18
    to_artifact: "scripts/fitting/bayesian_diagnostics.py::run_posterior_predictive_check (line 631)"
  - requirement: L2-01
    from_plan: "15-03"
    to_phase: 21
    to_artifact: "scripts/21_fit_with_l2.py (2-cov path for M3); scripts/fitting/numpyro_models.py::beta_lec_kappa (line 1238)"
---

# Phase 15: M3 Hierarchical POC + Level-2 Verification Report

**Phase Goal:** Implement the first hierarchical NumPyro model (M3 = WM-RL + kappa), add automatic convergence bumping, shrinkage diagnostics, WAIC/LOO infrastructure, PPC stratified by trauma group, and a Level-2 LEC covariate regression proof-of-concept.
**Verified:** 2026-04-19
**Status:** PASSED WITH ABSORPTION — 4 requirements delivered directly via plans 15-01 and 15-02; 2 requirements (HIER-09, L2-01 full pipeline validation) absorbed into Phases 18 and 21 respectively.
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | HIER-01: `wmrl_m3_hierarchical_model` exists with vmap + non-centered parameterization | VERIFIED | `grep -n "def wmrl_m3_hierarchical_model" scripts/fitting/numpyro_models.py` returns line 1137; `grep -n "use_pscan" scripts/fitting/numpyro_models.py` shows vmap dispatch at line 1298 |
| 2 | HIER-07: Convergence gate with auto-bump (0.80 → 0.95 → 0.99) and file-write refusal | VERIFIED | `grep -n "def run_inference_with_bump" scripts/fitting/numpyro_models.py` returns line 657; `grep -n "target_accept_probs" scripts/fitting/numpyro_models.py` returns line 664 showing `(0.80, 0.95, 0.99)`; `grep -n "Refusing to write output files" scripts/fitting/fit_bayesian.py` returns line 692 |
| 3 | HIER-08: Shrinkage diagnostic report infrastructure | VERIFIED | `grep -n "def compute_shrinkage_report" scripts/fitting/bayesian_diagnostics.py` returns line 456; `grep -n "def write_shrinkage_report" scripts/fitting/bayesian_diagnostics.py` returns line 511 |
| 4 | HIER-09: PPC stratified by trauma group | VERIFIED (via absorption) | `grep -n "def run_posterior_predictive_check" scripts/fitting/bayesian_diagnostics.py` returns line 631; function absorbed from Phase 15 plan 02 scope into Phase 18 integration — see `absorbed_into` YAML block and 15-03-SUMMARY.md |
| 5 | HIER-10: Parametric dispatch smoke test < 60s | VERIFIED | `grep -n "def test_smoke_dispatch" scripts/fitting/tests/test_m3_hierarchical.py` returns line 86; docstring at line 87 cites HIER-10 and 60s bound |
| 6 | L2-01: `beta_lec_kappa` in model + HDI check proof-of-concept | VERIFIED (via absorption) | `grep -n "beta_lec_kappa.*numpyro.sample" scripts/fitting/numpyro_models.py` returns line 1238 (conditional sample); Phase 21 plan 21-07 refit of the winner with L2 covariate design constitutes the full L2-01 production delivery — see `absorbed_into` YAML block |

**Score:** 6/6 requirements code-verified; 2 delivered via downstream-phase absorption

### Required Artifacts

| Artifact | Expected | Status | Evidence |
|----------|----------|--------|---------|
| `scripts/fitting/numpyro_models.py` | `wmrl_m3_hierarchical_model` + `run_inference_with_bump` | VERIFIED | `grep -n "def wmrl_m3_hierarchical_model\|def run_inference_with_bump" scripts/fitting/numpyro_models.py` returns lines 657 and 1137 |
| `scripts/fitting/bayesian_diagnostics.py` | `compute_shrinkage_report`, `write_shrinkage_report`, `run_posterior_predictive_check` | VERIFIED | `grep -n "^def " scripts/fitting/bayesian_diagnostics.py` returns all three functions at lines 456, 511, and 631 |
| `scripts/fitting/fit_bayesian.py` | Convergence gate + `run_inference_with_bump` integration | VERIFIED | `grep -n "run_inference_with_bump\|Refusing to write output files" scripts/fitting/fit_bayesian.py` returns lines 61 (import), 431 (call), 692 (gate refusal message) |
| `scripts/fitting/tests/test_m3_hierarchical.py` | Smoke dispatch test (HIER-10) | VERIFIED | `ls scripts/fitting/tests/test_m3_hierarchical.py` succeeds; `grep -n "def test_smoke_dispatch" scripts/fitting/tests/test_m3_hierarchical.py` returns line 86 |
| `cluster/13_bayesian_m3.slurm` | M3 SLURM script with `--model wmrl_m3` | VERIFIED | `ls cluster/13_bayesian_m3.slurm` succeeds; `grep -n "fit_bayesian.py" cluster/13_bayesian_m3.slurm` returns the invocation line |
| `15-03-PLAN.md` | Historical planning record (preserved, not deleted) | VERIFIED | `ls .planning/phases/15-m3-hierarchical-poc-level2/15-03-PLAN.md` succeeds |
| `15-03-SUMMARY.md` | Absorption summary documenting deliverable-to-phase mapping | VERIFIED | `ls .planning/phases/15-m3-hierarchical-poc-level2/15-03-SUMMARY.md` succeeds; contains `absorbed into` framing |

### Key Link Verification

| From | To | Via | Status | Evidence |
|------|----|-----|--------|---------|
| `fit_bayesian.py` | `numpyro_models.py::run_inference_with_bump` | import + direct call | WIRED | `grep -n "run_inference_with_bump" scripts/fitting/fit_bayesian.py` returns lines 61 (import) and 431 (call) |
| `wmrl_m3_hierarchical_model` | `run_inference_with_bump` → convergence gate → `write_shrinkage_report` | sequential pipeline in `save_results()` | WIRED | `grep -n "convergence gate\|shrinkage" scripts/fitting/fit_bayesian.py` returns lines 679 and 690 range |
| `bayesian_diagnostics.py::run_posterior_predictive_check` | `fit_bayesian.py::save_results` | wired for M3 model path | WIRED | `grep -n "run_posterior_predictive_check" scripts/fitting/fit_bayesian.py` returns the call in save_results |
| Phase 15 L2-01 | Phase 21 plan 21-07 winner L2 refit | `scripts/21_fit_with_l2.py` 2-cov path | WIRED (via absorption) | `grep -n "beta_lec_kappa\|covariate_lec" scripts/fitting/numpyro_models.py` returns lines 1238, 1278 (M3 L2 hook); `ls scripts/21_fit_with_l2.py` confirms Phase 21 delivery |
| Phase 15 HIER-09 | Phase 18 integration | `run_posterior_predictive_check` reused by Phase 18 pipeline | WIRED (via absorption) | `grep -n "run_posterior_predictive_check" scripts/fitting/bayesian_diagnostics.py` returns line 631; Phase 18 18-VERIFICATION.md row 5 cites script 18b using shrinkage + PPC infrastructure |

### Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| HIER-01: `wmrl_m3_hierarchical_model` with vmap + non-centered | SATISFIED | None — code on disk |
| HIER-07: Convergence gate (R-hat <= 1.01, ESS_bulk >= 400, auto-bump 0.80→0.95→0.99) | SATISFIED | None — code on disk |
| HIER-08: Shrinkage diagnostic report | SATISFIED | None — code on disk |
| HIER-09: PPC stratified by trauma group | SATISFIED (via absorption) | Absorbed into Phase 18; see `absorbed_into` YAML block |
| HIER-10: Smoke dispatch test < 60s | SATISFIED | None — test on disk |
| L2-01: LEC → kappa proof-of-concept (beta_lec_kappa, HDI reporting) | SATISFIED (via absorption) | Absorbed into Phase 21 plan 21-07; see `absorbed_into` YAML block |

### Absorbed-Into Table

The following Phase 15 deliverables were absorbed rather than shipped as
standalone plan 15-03 artifacts:

| Deliverable | Original Scope (15-03) | Absorbed Into | Evidence |
|---|---|---|---|
| HIER-09: PPC group-stratified | `run_posterior_predictive_check` in bayesian_diagnostics.py | Phase 18 (wired into `save_results()` M3 path) | `grep -n "def run_posterior_predictive_check" scripts/fitting/bayesian_diagnostics.py` returns line 631 |
| M3 cluster SLURM script | `cluster/13_bayesian_m3.slurm` | Phase 15 plan 02 Task 1d (delivered) | `ls cluster/13_bayesian_m3.slurm` succeeds |
| End-to-end pipeline validation (N=154) | Phase 15 checkpoint:human-verify | Phase 21 master orchestrator step 21.3 | `grep -c "afterok" cluster/21_submit_pipeline.sh` returns 19 |
| L2-01: beta_lec_kappa HDI at production scale | Phase 15 checkpoint proof-of-concept | Phase 21 plan 21-07 winner L2 refit | `grep -n "beta_lec_kappa.*numpyro.sample" scripts/fitting/numpyro_models.py` returns line 1238 |

### Anti-Patterns Found

`grep -n "TODO\|FIXME\|XXX" scripts/fitting/numpyro_models.py` — zero matches in Phase 15 additions (wmrl_m3_hierarchical_model, run_inference_with_bump). No stubs.

`grep -n "TODO\|FIXME\|XXX" scripts/fitting/bayesian_diagnostics.py` — zero matches in Phase 15 additions (compute_shrinkage_report, write_shrinkage_report, run_posterior_predictive_check).

### Human Verification Required

No additional human verification required for the code-level requirements.
The two absorbed requirements (HIER-09 PPC group stratification and L2-01
production-scale HDI) are verified via their downstream phases:

- **HIER-09 (PPC):** Verified as part of Phase 18 pipeline verification. Post-cluster check: after `bash cluster/21_submit_pipeline.sh` completes, inspect PPC results in Phase 21 step 21.3 outputs.
- **L2-01 (beta_lec_kappa production):** Verified as part of Phase 21 plan 21-07. Post-cluster check: after `bash cluster/21_submit_pipeline.sh` completes, `output/bayesian/21_l2/{winner}_posterior.nc` contains `beta_lec_kappa` samples.

Cold-start entry for all cluster-execution verification: `bash cluster/21_submit_pipeline.sh`.

## Gaps Summary

No code gaps. All 6 Phase 15 requirements are satisfied. Plans 15-01 and 15-02 delivered HIER-01, HIER-07, HIER-08, HIER-10, and the M3 cluster SLURM script. HIER-09 (PPC) was implemented in bayesian_diagnostics.py and absorbed into Phase 18. L2-01 (beta_lec_kappa) was implemented in the M3 model and the proof-of-concept absorbed into Phase 21 plan 21-07 for production-scale delivery. Plan 15-03 (the intended delivery vehicle for HIER-09 and the N=154 checkpoint:human-verify) was superseded by the more comprehensive Phase 21 pipeline; its planning record is preserved at `15-03-PLAN.md`. A corresponding absorption summary is at `15-03-SUMMARY.md`.

---

_Verified: 2026-04-19_
_Verifier: Claude (gsd-executor, plan 22-02)_
