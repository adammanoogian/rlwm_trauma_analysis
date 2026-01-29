# WM-RL+κ Perseveration Model Extension

## What This Is

An extension to the existing WM-RL hybrid model that adds a perseveration parameter (κ) to distinguish outcome-insensitive action repetition from reduced negative learning rate effects. This is particularly important for analyzing post-reversal behavior in the RLWM task, where perseverative responding (repeating the previously correct action regardless of outcome) needs to be separated from learning-rate deficits.

## Core Value

The model must correctly dissociate perseverative responding from learning-rate effects (α₋), enabling accurate identification of whether post-reversal failures reflect motor perseveration or outcome insensitivity in trauma populations.

## Requirements

### Validated

- ✓ Q-learning model with asymmetric learning rates (α₊, α₋) — existing
- ✓ WM-RL hybrid model with capacity-based weighting — existing
- ✓ JAX-based likelihood functions for MLE fitting — existing
- ✓ Block-aware processing with Q/WM reset at boundaries — existing
- ✓ Fixed β=50 for parameter identifiability — existing
- ✓ Epsilon noise for random responding — existing
- ✓ MLE fitting pipeline with 20 random starts — existing

### Active

- [ ] WM-RL+κ JAX likelihood function with perseveration parameter
- [ ] κ parameter constrained to [0, 1] matching existing parameter bounds
- [ ] Global action repetition tracking (Rep(a) = I[a = a_{t-1}])
- [ ] Last action resets at block boundaries
- [ ] Integration with existing MLE fitting infrastructure (mle_utils.py)
- [ ] Model comparison capability (WM-RL vs WM-RL+κ via AIC/BIC)

### Out of Scope

- Stimulus-specific perseveration — Global action repetition captures motor perseveration, which is the theoretically relevant mechanism
- Bayesian hierarchical model (NumPyro) — MLE fitting is the target methodology
- Agent class for simulation — Just the likelihood function for fitting
- Q-learning+κ variant — Only extending WM-RL for now

## Context

**Mathematical Formulation (from user):**

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
- **Fitting**: MLE with 20 random starts (match existing methodology)
- **Block structure**: Last action resets at block boundaries (23 blocks per participant)
- **Compatibility**: Must integrate with existing fit_mle.py and mle_utils.py

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Global (not stimulus-specific) perseveration | Captures motor-level response stickiness relevant post-reversal | — Pending |
| κ ∈ [0, 1] bounds | Matches Senta et al. parameter constraint convention | — Pending |
| WM-RL only (not Q-learning) | WM-RL is the target model for trauma analysis | — Pending |
| Reset last_action at block boundaries | Matches existing Q/WM reset pattern; no carry-over between blocks | — Pending |

---
*Last updated: 2026-01-28 after initialization*
