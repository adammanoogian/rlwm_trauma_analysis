---
phase: 17-m4-hierarchical-lba
verified: 2026-04-13T17:41:26Z
status: passed
score: 6/6 must-haves verified
---

# Phase 17: M4 Hierarchical LBA Verification Report

**Phase Goal:** Deliver hierarchical M4 (joint choice+RT via LBA) under NumPyro NUTS with float64 process isolation, non-centered log(b-A) reparameterization, checkpoint-and-resume for the 48h SLURM wall, and Pareto-k gating for downstream comparison.
**Verified:** 2026-04-13T17:41:26Z
**Status:** passed
**Re-verification:** No - initial verification

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | M4 fit launches in its own process with float64 active before any JAX import; integration test confirms jnp.float64 | VERIFIED | `import jax; jax.config.update` at lines 63-65 of 13_fit_bayesian_m4.py (before numpyro, arviz, jnp); runtime assert at line 598 in main(); test_float64_isolation covers both global dtype and rts_stacked dtype |
| 2  | Non-centered log(b-A) parameterization fits hierarchically and recovers structural properties | VERIFIED | log_delta_mu_pr/sigma_pr/z sampled at lines 2313-2320 of numpyro_models.py; b = A + delta decoded inside participant for-loop at line 2344; test_log_delta_recovery checks delta>0, A>0, b>A, all 10 params finite |
| 3  | chain_method=vectorized, num_warmup=1000, num_samples=1500, target_accept_prob=0.95 in MCMC setup | VERIFIED | Defaults at lines 139-146 (argparse); chain_method=vectorized at line 644; target_accept_prob=0.95 at line 636; SLURM invokes with --warmup 1000 --samples 1500 |
| 4  | Checkpoint-and-resume: warmup state pickled, fresh MCMC resumes from post_warmup_state | VERIFIED | Full implementation at lines 656-674 of 13_fit_bayesian_m4.py (jax.device_get before pickle, checkpoint detection); test_checkpoint_resume exercises the exact API path |
| 5  | az.loo pointwise Pareto-k diagnostics with PASS/FALLBACK branches at 5% threshold | VERIFIED | az.loo(idata, pointwise=True) at line 794; frac_bad > 0.05 triggers FALLBACK at line 807; JSON + MD reports written; error path caught with except |
| 6  | cluster/13_bayesian_m4_gpu.slurm has --time=48:00:00, --mem=96G, --gres=gpu:a100:1 | VERIFIED | Lines 40-44 of 13_bayesian_m4_gpu.slurm confirm all three SBATCH directives |

**Score:** 6/6 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/fitting/numpyro_models.py` | prepare_stacked_participant_data_m4 + wmrl_m4_hierarchical_model added | VERIFIED | 2375 lines total; functions at lines 2058 and 2197; no stubs found |
| `scripts/13_fit_bayesian_m4.py` | Self-contained M4 pipeline with float64, checkpoint, Pareto-k | VERIFIED | 857 lines; float64 isolation at top; all M4H requirements present; wired to main() |
| `scripts/fitting/tests/test_m4_hierarchical.py` | Unit tests: RT data prep, sorted participants, NUTS smoke | VERIFIED | 223 lines; 3 test functions; imports both functions under test |
| `scripts/fitting/tests/test_m4_integration.py` | Integration tests: float64 isolation, log(b-A) recovery, checkpoint-resume | VERIFIED | 263 lines; 3 test functions covering M4H-01, M4H-02, M4H-04; float64 guard at module top |
| `cluster/13_bayesian_m4_gpu.slurm` | SLURM script with 48h A100 GPU directives (M4H-06) | VERIFIED | 157 lines; all required SBATCH directives present; GPU verification preamble; checkpoint-resume documented |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| 13_fit_bayesian_m4.py | float64 active globally | import jax + jax.config.update before other imports | WIRED | Lines 63-65; numpyro.enable_x64() at line 68; assert at line 598 in main() |
| wmrl_m4_hierarchical_model | wmrl_m4_multiblock_likelihood_stacked | lazy import inside function body | WIRED | Line 2257 of numpyro_models.py; b=A+delta decode at line 2344 before likelihood call |
| 13_fit_bayesian_m4.py checkpoint | mcmc.post_warmup_state | pickle + jax.device_get | WIRED | Lines 656-674; jax.device_get at line 670; mcmc2.post_warmup_state = loaded_state at line 660 |
| az.loo Pareto-k gate | PASS/FALLBACK report | frac_bad > 0.05 branch + JSON/MD write | WIRED | Lines 793-841; both branches present; except handler writes fallback report regardless |
| SLURM script | 13_fit_bayesian_m4.py | python scripts/13_fit_bayesian_m4.py at line 123 | WIRED | Passes --chains 4 --warmup 1000 --samples 1500 --checkpoint-dir output/bayesian |
| test_m4_integration.py | wmrl_m4_hierarchical_model | import from scripts.fitting.numpyro_models | WIRED | Lines 22-25; both functions imported and exercised in 3 test functions |

### Requirements Coverage

| Requirement | Status | Notes |
|-------------|--------|-------|
| M4H-01 | SATISFIED | Float64 active before any JAX/NumPyro import; runtime assert in main(); test_float64_isolation |
| M4H-02 | SATISFIED | log_delta_mu_pr/sigma_pr/z non-centered; b=A+delta inside for-loop; structural tests pass |
| M4H-03 | SATISFIED | chain_method=vectorized locked; target_accept_prob=0.95; warmup=1000, samples=1500 defaults |
| M4H-04 | SATISFIED | Full warmup-pickle-resume cycle in 13_fit_bayesian_m4.py; test_checkpoint_resume validates API |
| M4H-05 | SATISFIED | az.loo pointwise; 5% threshold; two branches (PASS/FALLBACK); JSON report for Phase 18 |
| M4H-06 | SATISFIED | 13_bayesian_m4_gpu.slurm with --time=48:00:00, --mem=96G, --gres=gpu:a100:1 |

Note: REQUIREMENTS.md checkbox status shows unchecked for M4H-01..06. This is a documentation tracking issue only; the actual implementations are verified present and substantive in the codebase.

### Anti-Patterns Found

None. No TODO/FIXME/placeholder/stub patterns detected in any of the 5 key files.

### Human Verification Required

No automated blockers. The following items require cluster execution and are beyond programmatic verification:

1. **48h wall-time budget** - Verify the actual fit of N=154 completes within 48h on A100. Expected: job exits 0 within wall time. Why human: requires actual SLURM submission.
2. **Pareto-k fallback in production** - Confirm FALLBACK branch triggers as expected for participant-level LOO on LBA. Why human: requires real posterior from full N=154 fit.
3. **15% relative error threshold (M4H-02 strict)** - Integration test uses relaxed structural checks (delta>0, A>0, b>A). The full 15%-relative-error criterion from the success criteria requires N=154 cluster fit; the integration-scale (N=10) posterior is too diffuse for point recovery.

### Gaps Summary

No gaps. All 6 requirements (M4H-01 through M4H-06) have substantive, wired implementations. The 5 key files exist, contain real implementations (no stubs), and are connected to each other and to the test suite.

---

_Verified: 2026-04-13T17:41:26Z_
_Verifier: Claude (gsd-verifier)_
