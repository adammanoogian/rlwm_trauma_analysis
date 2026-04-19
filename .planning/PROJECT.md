# RLWM Trauma Analysis

## What This Is

A computational modeling pipeline for analyzing reinforcement learning and working memory (RLWM) task performance in trauma populations. Implements 7 models (M1-M6b) spanning Q-learning, WM-RL hybrid, perseveration variants, RL forgetting, and joint choice+RT via Linear Ballistic Accumulator, with MLE fitting, cross-model comparison, parameter recovery, and trauma-group analysis.

**Model naming convention:**
- M1: Q-learning (α₊, α₋, ε)
- M2: WM-RL hybrid (α₊, α₋, φ, ρ, K, ε)
- M3: WM-RL + κ perseveration (α₊, α₋, φ, ρ, K, ε, κ)
- M4: RLWM-LBA joint choice+RT (M3 learning + LBA action selection; v_scale, b, A, t₀)
- M5: WM-RL + φ_RL RL forgetting (M3 + Q-value decay; 8 params)
- M6a: WM-RL + κ_s stimulus-specific perseveration (replaces κ; 7 params)
- M6b: WM-RL + κ + κ_s dual perseveration (global + stimulus-specific; 8 params)

**Shipped:** v1 M3 Infrastructure (2026-01-30), v2 Validation (2026-02-06), v3 Model Extensions M4-M6 (2026-04-03)

## Core Value

The model must correctly dissociate perseverative responding from learning-rate effects (α₋), enabling accurate identification of whether post-reversal failures reflect motor perseveration or outcome insensitivity in trauma populations.

## Current State

**v3.0 shipped.** All 7 models (M1-M6b, M4) have JAX likelihoods, MLE fitting, parameter recovery infrastructure, and downstream analysis integration. Seven candidate RL/WM/LBA models are implemented and compared via AIC/BIC. M4 is in a separate choice+RT comparison track because its joint choice+RT likelihood is not commensurable with choice-only information criteria.

**Tech stack:** Python, JAX (float32 + float64 for M4), NumPy, SciPy, pytest

**Infrastructure scope:**
- 7 computational models (M1-M6b, M4) with JAX likelihoods
- MLE fitting pipeline with Latin Hypercube multi-start optimization, per-participant information criteria, Hessian diagnostics
- Parameter recovery infrastructure with r >= 0.80 criterion
- Multi-model comparison (AIC, BIC, Akaike weights) via 14_compare_models.py
- Per-participant heterogeneity diagnostics via 17_analyze_winner_heterogeneity.py
- Trauma-parameter regression pipeline with uncorrected, FDR-BH, and Bonferroni reporting

## Current Milestone: v4.0 Hierarchical Bayesian Pipeline & LBA Acceleration

**Goal:** Move the inference pipeline from MLE point estimates to full hierarchical Bayesian posteriors with trauma subscales as Level-2 predictors, principled Bayesian model comparison (WAIC/LOO), and GPU-accelerated LBA sampling for M4.

**Motivation (from quick-006 verification):**
- Base RLWM parameters (alpha+/alpha-/phi/rho/K/epsilon) fail r>=0.80 recovery across every model. Capacity K recovery is worst (r=0.21). Without shrinkage, individual-differences claims on these parameters are unreliable.
- Post-hoc FDR correction (script 16 + 16b) forces a 48-test within-model family for M6b and washes out the epsilon-IES-R signal. Joint Bayesian inference replaces the correction problem with a single posterior.
- M4 parameter recovery is still unrun because the CPU/serial path needs ~48h per model. GPU-batched vmap with float64 can cut this to a feasible run.

**Target features:**
- GPU-batched LBA fitting path (`fit_all_gpu_m4`) supporting both synthetic recovery and real-data fitting
- Hierarchical Bayesian implementations of all 7 models (M1, M2, M3, M5, M6a, M6b, M4) in `numpyro_models.py`
- Trauma subscales (IES-R intrusion/avoidance/hyperarousal, LEC-5 subcategories) as Level-2 predictors joint with model fit
- Collins K parameterization research + constrained-K refit of all models
- Bayesian model comparison via WAIC and LOO-CV (`arviz.compare`), replacing AIC/BIC as the primary criterion
- 16b script kept as fast-preview supplementary only (no subscale extension)

**Compute expectation:** ~50-96 GPU hours per full NUTS sampling run across all 7 models. Budget 2-3 reruns for convergence diagnostics.

## Requirements

### Validated

- ✓ Q-learning model (M1) with asymmetric learning rates — existing
- ✓ WM-RL hybrid model (M2) with capacity-based weighting — existing
- ✓ JAX-based likelihood functions for MLE fitting — existing
- ✓ Block-aware processing with Q/WM reset at boundaries — existing
- ✓ Fixed β=50 for parameter identifiability — existing
- ✓ Epsilon noise for random responding — existing
- ✓ MLE fitting pipeline with 20 random starts — existing
- ✓ JAX likelihood function `wmrl_m3_block_likelihood()` with κ·Rep(a) term — v1
- ✓ WMRLHybridAgent extended with optional `kappa` parameter — v1
- ✓ κ parameter bounded to [0, 1] in mle_utils.py — v1
- ✓ Global action repetition tracking (Rep(a) = I[a = a_{t-1}]) — v1
- ✓ Last action resets at block boundaries — v1
- ✓ fit_mle.py accepts `--model wmrl_m3` CLI option — v1
- ✓ Backward compatibility: κ=0 produces identical results to M2 — v1
- ✓ N-model comparison with Akaike weights — v1
- ✓ Parameter recovery pipeline with r >= 0.80 criterion — v2
- ✓ M5 RL forgetting (φ_RL Q-value decay toward Q₀=1/3) — v3
- ✓ M6a stimulus-specific perseveration (κ_s per-stimulus tracking) — v3
- ✓ M6b dual perseveration (κ + κ_s with stick-breaking constraint) — v3
- ✓ M4 RLWM-LBA joint choice+RT (LBA decision, float64, separate track) — v3
- ✓ Cross-model recovery infrastructure (confusion matrix, 6 choice-only models) — v3
- ✓ MODEL_REFERENCE.md with complete M1-M6b mathematics — v3

#### v4.0 (milestone)

**INFRA — Infrastructure Repair & Scaffolding (Phase 13)**
- INFRA-01 (Phase 13 — 13-01-SUMMARY.md) — Fix P0 broken import in fit_bayesian.py
- INFRA-02 (Phase 13 — 13-01-SUMMARY.md) — Pin dependency versions in pyproject.toml + environment_gpu.yml
- INFRA-03 (Phase 13 — 13-02-SUMMARY.md) — bayesian_diagnostics.py with compute_pointwise_log_lik()
- INFRA-04 (Phase 13 — 13-02-SUMMARY.md) — bayesian_summary_writer.py with schema-parity CSV output
- INFRA-05 (Phase 13 — 13-02-SUMMARY.md) — numpyro_helpers.py non-centered parameterization
- INFRA-06 (Phase 13 — 13-02-SUMMARY.md) — parameterization_version column convention
- INFRA-07 (Phase 13 — 13-03-SUMMARY.md) — Drop PyMC from 16b_bayesian_regression.py
- INFRA-08 (Phase 13 — 13-03-SUMMARY.md) — JAX compilation cache on cluster + benchmark CI test

**K — Collins K Parameterization Research (Phase 13/14)**
- K-01 (Phase 13 — 13-01-SUMMARY.md) — docs/K_PARAMETERIZATION.md with Collins K convention

**HIER — Hierarchical Model Implementations (Phase 16)**
- HIER-02 (Phase 16 — 16-01-SUMMARY.md) — M1 qlearning_hierarchical_model non-centered
- HIER-03 (Phase 16 — 16-01-SUMMARY.md) — M2 wmrl_hierarchical_model vectorized likelihood
- HIER-04 (Phase 16 — 16-02-SUMMARY.md) — M5 wmrl_m5_hierarchical_model
- HIER-05 (Phase 16 — 16-02-SUMMARY.md) — M6a wmrl_m6a_hierarchical_model
- HIER-06 (Phase 16 — 16-02-SUMMARY.md) — M6b wmrl_m6b_hierarchical_model stick-breaking

**L2 — Level-2 Trauma Regression (Phase 16)**
- L2-02 (Phase 16 — 16-03-SUMMARY.md) — IES-R subscale correlation audit
- L2-03 (Phase 16 — 16-03-SUMMARY.md) — Orthogonalized subscale parameterization (Gram-Schmidt residuals)
- L2-04 (Phase 16 — 16-04-SUMMARY.md) — LEC-5 subcategory Level-2 predictors
- L2-05 (Phase 16 — 16-04-SUMMARY.md) — Full subscale Level-2 fit on M6b (32 coefficients)
- L2-06 (Phase 16 — 16-05-SUMMARY.md) — Permutation null test for Level-2 effects
- L2-07 (Phase 16 — 16-05-SUMMARY.md) — Forest plots via scripts/18_bayesian_level2_effects.py
- L2-08 (Phase 16 — 16-07-SUMMARY.md) — Horseshoe prior deferred; Normal(0,1) confirmed sufficient

**M4H — Hierarchical M4 LBA (Phase 17)**
- M4H-01 (Phase 17 — 17-01-SUMMARY.md) — M4 hierarchical NumPyro model with float64 process isolation
- M4H-02 (Phase 17 — 17-01-SUMMARY.md) — Non-centered log(b-A) parameterization for LBA threshold
- M4H-03 (Phase 17 — 17-01-SUMMARY.md) — chain_method='vectorized' with reduced sampling budget
- M4H-04 (Phase 17 — 17-02-SUMMARY.md) — Checkpoint-and-resume via mcmc.post_warmup_state
- M4H-05 (Phase 17 — 17-02-SUMMARY.md) — Pareto-k gating for M4 vs choice-only comparison
- M4H-06 (Phase 17 — 17-03-SUMMARY.md) — cluster/13_bayesian_m4_gpu.slurm

**CMP — Bayesian Model Comparison (Phase 18)**
- CMP-01 (Phase 18 — 18-01-SUMMARY.md) — arviz.compare stacking weights across 6 choice-only models
- CMP-02 (Phase 18 — 18-01-SUMMARY.md) — M4 separate comparison track
- CMP-03 (Phase 18 — 18-02-SUMMARY.md) — WAIC + LOO as primary criteria
- CMP-04 (Phase 18 — 18-02-SUMMARY.md) — scripts/14_compare_models.py --bayesian-comparison mode

**MIG — Pipeline Migration (Phase 18)**
- MIG-01 (Phase 18 — 18-03-SUMMARY.md) — scripts/15 --source mle|bayesian flag
- MIG-02 (Phase 18 — 18-03-SUMMARY.md) — scripts/16 --source flag
- MIG-03 (Phase 18 — 18-03-SUMMARY.md) — scripts/17 --source flag
- MIG-04 (Phase 18 — 18-04-SUMMARY.md) — scripts/16b deprecated with docstring
- MIG-05 (Phase 18 — 18-04-SUMMARY.md) — MLE-vs-Bayesian reliability scatterplots

**DOC — Documentation & Manuscript (Phase 18)**
- DOC-01 (Phase 18 — 18-05-SUMMARY.md) — MODEL_REFERENCE.md hierarchical Bayesian section
- DOC-02 (Phase 18 — 18-05-SUMMARY.md) — Manuscript methods section NumPyro/WAIC-LOO pipeline
- DOC-03 (Phase 18 — 18-05-SUMMARY.md) — Manuscript results Level-2 forest plots + stacking table
- DOC-04 (Phase 18 — 18-05-SUMMARY.md) — Manuscript limitations Pareto-k/K recovery/M6b shrinkage

**PSCAN — Associative Scan Likelihood Parallelization (Phase 19)**
- PSCAN-01 (Phase 19 — 19-01-SUMMARY.md) — docs/PARALLEL_SCAN_LIKELIHOOD.md
- PSCAN-02 (Phase 19 — 19-01-SUMMARY.md) — associative_scan_q_update() in jax_likelihoods.py
- PSCAN-03 (Phase 19 — 19-02-SUMMARY.md) — associative_scan_wm_update() in jax_likelihoods.py
- PSCAN-04 (Phase 19 — 19-02-SUMMARY.md) — all 6 choice-only pscan likelihood variants
- PSCAN-05 (Phase 19 — 19-03-SUMMARY.md) — GPU benchmark output/bayesian/pscan_benchmark.json
- PSCAN-06 (Phase 19 — 19-03-SUMMARY.md) — A/B comparison sequential vs pscan posteriors

### Active (v4.0)

- [ ] **K-02** — Implement constrained K bounds in mle_utils.py (cluster refit via `bash cluster/21_submit_pipeline.sh` — cold start)
- [ ] **K-03** — Refit all 7 models via MLE with constrained K (cluster refit via `bash cluster/21_submit_pipeline.sh` — cold start)
- [ ] **GPU-01** — fit_all_gpu_m4 function for M4 synthetic path (cluster refit via `bash cluster/21_submit_pipeline.sh` — cold start)
- [ ] **GPU-02** — fit_all_gpu_m4 for real-data M4 fitting (cluster refit via `bash cluster/21_submit_pipeline.sh` — cold start)
- [ ] **GPU-03** — Verify GPU speedup for M4 recovery < 12h on A100 (cluster refit via `bash cluster/21_submit_pipeline.sh` — cold start)
- [ ] **HIER-01** — M3 hierarchical NumPyro model (absorbed into Phases 16/18/21 — see 15-VERIFICATION.md once plan 22-02 lands)
- [ ] **HIER-07** — Convergence gate R-hat/ESS/divergences (absorbed into Phases 16/18/21 — see 15-VERIFICATION.md once plan 22-02 lands)
- [ ] **HIER-08** — Shrinkage diagnostic report (absorbed into Phases 16/18/21 — see 15-VERIFICATION.md once plan 22-02 lands)
- [ ] **HIER-09** — Posterior predictive check infrastructure (absorbed into Phases 16/18/21 — see 15-VERIFICATION.md once plan 22-02 lands)
- [ ] **HIER-10** — Parametric dispatch smoke test (absorbed into Phases 16/18/21 — see 15-VERIFICATION.md once plan 22-02 lands)
- [ ] **L2-01** — M3 POC Level-2 regression LEC → kappa (absorbed into Phases 16/18/21 — see 15-VERIFICATION.md once plan 22-02 lands)

### Out of Scope

- Running actual fits locally — User runs hierarchical fits on Monash M3 GPU cluster; infrastructure only
- Combined M5+M6 models — test mechanisms independently first
- DDM alternative to LBA — task uses 3 choices; LBA is correct framework
- Extending 16b_bayesian_regression.py to subscales — subsumed by hierarchical fits; 16b kept as fast-preview supplementary
- Post-hoc FDR correction pipeline — replaced by joint posterior inference
- Additional M7+ models — v4.0 focuses on inference infrastructure, not new cognitive models

## Context

**Mathematical Formulation:**

The perseveration parameter increases the probability of repeating the immediately preceding action independent of outcome history:

```
Rep_t(a) = I[a = a_{t-1}]  (indicator function: 1 if action matches last action, 0 otherwise)

P(a_t = a | s_t) ∝ exp(V(s_t, a) + κ · Rep_t(a))
```

Where V(s_t, a) is the hybrid value from the WM-RL model (ω·WM + (1-ω)·Q).

**Why This Matters:**

After a contingency reversal, participants may continue selecting the previously correct action. This could reflect:
1. **Reduced α₋**: Failure to learn from negative feedback (outcome insensitivity)
2. **High κ**: Tendency to repeat recent actions regardless of outcome (motor perseveration)

These have different theoretical implications for trauma populations.

**Technical Context:**

- Existing codebase uses JAX for JIT-compiled likelihoods
- MLE fitting uses scipy.optimize.minimize with L-BFGS-B
- Parameters transformed via logit for unconstrained optimization
- Fixed β=50 following Senta et al. (2025) for identifiability

## Constraints

- **Stack**: Must use JAX for likelihood functions (matches existing infrastructure)
- **Parameter bounds**: κ ∈ [0, 1] (same as other parameters per Senta et al.)
- **Fitting**: MLE infrastructure with 20 random starts (user runs on cluster)
- **Block structure**: Last action resets at block boundaries (23 blocks per participant)
- **Integration**: Extend existing code (wm_rl_hybrid.py, mle_utils.py, fit_mle.py) — no duplication

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Name: M3 (not wmrl_kappa) | Follows M1/M2 naming convention | ✓ Good |
| Extend WMRLHybridAgent | Avoid code duplication; kappa=0 gives M2 behavior | ✓ Good |
| Global (not stimulus-specific) perseveration | Captures motor-level response stickiness relevant post-reversal | ✓ Good |
| κ ∈ [0, 1] bounds | Matches Senta et al. parameter constraint convention | ✓ Good |
| Reset last_action at block boundaries | Matches existing Q/WM reset pattern; no carry-over between blocks | ✓ Good |
| Infrastructure only (no fits) | User runs fits on cluster | ✓ Good |
| M3 likelihood branches on kappa=0 | Ensures exact backward compatibility with M2 | ✓ Critical fix |
| Dict-based N-model comparison | Enables flexible comparison without hardcoding | ✓ Good |

---
*Last updated: 2026-04-19 — v4.0 closure audit*
