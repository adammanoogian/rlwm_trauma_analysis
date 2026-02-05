# RLWM Trauma Analysis

## What This Is

A computational modeling pipeline for analyzing reinforcement learning and working memory (RLWM) task performance in trauma populations. Implements Q-learning (M1), WM-RL hybrid (M2), and WM-RL+perseveration (M3) models with MLE fitting, model comparison, and trauma-group analysis. Now entering validation and publication readiness.

**Model naming convention:**
- M1: Q-learning (α₊, α₋, ε)
- M2: WM-RL hybrid (α₊, α₋, φ, ρ, K, ε)
- M3: WM-RL + κ perseveration (α₊, α₋, φ, ρ, K, ε, κ)

**Shipped:** v1 M3 Infrastructure (2026-01-30)

## Core Value

The model must correctly dissociate perseverative responding from learning-rate effects (α₋), enabling accurate identification of whether post-reversal failures reflect motor perseveration or outcome insensitivity in trauma populations.

## Current Milestone: v2 Post-Fitting Validation & Publication Readiness

**Goal:** Validate fitted models through parameter recovery and posterior predictive checks (Senta et al. 2025 requirements), enhance cluster monitoring, and produce publication-ready model comparison and trauma-association outputs.

**Target features:**
- Parameter recovery pipeline (Script 11) — r ≥ 0.80 criterion per Senta et al.
- Posterior predictive checks (Script 09) — synthetic vs. observed comparison
- GPU monitoring and memory checkpoint persistence (cluster SLURM scripts)
- Trauma-group model comparison (`--by-group` in Script 14)
- Combined results summary table (winning model + key associations)

## Current State

**Shipped v1.0** with:
- 4,368 lines across 6 key source files
- 3 phases, 6 plans completed
- 24+ backward compatibility tests
- Full MLE infrastructure for M3 fitting

**Tech stack:** Python, JAX, NumPy, SciPy, pytest

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

### Active

- [ ] Parameter recovery pipeline with MLE fitting loop over synthetic datasets (Script 11)
- [ ] Recovery quality metrics: correlation, RMSE, bias per parameter
- [ ] Parameter recovery scatter plots and CSV output
- [ ] Posterior predictive check comparison pipeline (Script 09)
- [ ] Synthetic vs. observed learning curves, set-size effects, accuracy distributions
- [ ] GPU utilization monitoring in cluster SLURM scripts
- [ ] Memory checkpoint persistence to CSV from `[MEMORY]` log lines
- [ ] Trauma-group model comparison (`--by-group` flag in Script 14)
- [ ] Combined results summary: winning model + key parameter-trauma associations

### Out of Scope

- Running actual fits — User will run on cluster; we build infrastructure only
- Duplicate agent class — Extended existing WMRLHybridAgent
- Stimulus-specific perseveration — Global action repetition captures motor perseveration
- Q-learning+κ variant — Only extended WM-RL for v1

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
*Last updated: 2026-02-05 after v2 milestone started*
