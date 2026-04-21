# Milestone v1: WM-RL M3 (Perseveration Extension)

**Status:** SHIPPED 2026-01-30
**Phases:** 1-3
**Total Plans:** 6

## Overview

This milestone extended the existing WM-RL model (M2) with a perseveration parameter (κ) to create WM-RL M3. The goal was to dissociate outcome-insensitive action repetition from learning-rate effects.

**Model naming convention:**
- M1: Q-learning (α₊, α₋, ε)
- M2: WM-RL hybrid (α₊, α₋, φ, ρ, K, ε)
- M3: WM-RL + perseveration (α₊, α₋, φ, ρ, K, ε, κ) ← **this milestone**

**Integration approach:** Extend existing code rather than duplicate. The WMRLHybridAgent class and likelihood functions were parameterized to optionally include κ.

## Phases

### Phase 1: Core Implementation

**Goal**: JAX likelihood and agent class support κ perseveration parameter
**Depends on**: Nothing (extends existing code)
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md - JAX M3 likelihood functions with kappa perseveration
- [x] 01-02-PLAN.md - WMRLHybridAgent extension with optional kappa parameter

**Success Criteria Achieved:**
1. ✓ `wmrl_m3_block_likelihood()` function computes log-likelihood with κ·Rep(a) term
2. ✓ Rep(a) = I[a = a_{t-1}] tracks global action repetition (not stimulus-specific)
3. ✓ Last action resets at block start (no carry-over between blocks)
4. ✓ `wmrl_m3_multiblock_likelihood()` sums across blocks
5. ✓ WMRLHybridAgent extended with optional `kappa` parameter (default 0 = M2 behavior)
6. ✓ Agent's `get_hybrid_probs()` includes κ·Rep(a) when kappa > 0

**Completed:** 2026-01-29

### Phase 2: MLE Infrastructure

**Goal**: MLE fitting utilities support wmrl_m3 model type
**Depends on**: Phase 1
**Plans**: 2 plans

Plans:
- [x] 02-01-PLAN.md - Add wmrl_m3 parameter infrastructure to mle_utils.py
- [x] 02-02-PLAN.md - Add wmrl_m3 fitting support to fit_mle.py

**Success Criteria Achieved:**
1. ✓ WMRL_M3_BOUNDS dict includes κ ∈ [0, 1]
2. ✓ WMRL_M3_PARAMS list: [alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon]
3. ✓ `unconstrained_to_params()` and `params_to_unconstrained()` support 'wmrl_m3'
4. ✓ fit_mle.py accepts `--model wmrl_m3` CLI option
5. ✓ `_objective_wmrl_m3()` uses wmrl_m3_multiblock_likelihood
6. ✓ Fitting uses 20 random starts (same methodology as M1/M2)

**Completed:** 2026-01-29

### Phase 3: Validation & Comparison

**Goal**: Researcher can validate M3 and compare against M2 baseline
**Depends on**: Phase 2
**Plans**: 2 plans

Plans:
- [x] 03-01-PLAN.md - Backward compatibility pytest tests for M3 (kappa=0 matches M2)
- [x] 03-02-PLAN.md - Extend compare_mle_models.py for 3-way comparison (M1/M2/M3)

**Success Criteria Achieved:**
1. ✓ AIC/BIC computed for M3 fits (7 free parameters)
2. ✓ Model comparison works: M2 (6 params) vs M3 (7 params)
3. ✓ Test script validates likelihood computation matches expected behavior
4. ✓ κ=0 produces identical results to M2 (backward compatibility)

**Completed:** 2026-01-30

---

## Milestone Summary

### Key Accomplishments

1. Implemented JAX likelihood functions (`wmrl_m3_block_likelihood()`, `wmrl_m3_multiblock_likelihood()`) with κ·Rep(a) perseveration term
2. Extended WMRLHybridAgent with optional kappa parameter, maintaining M2 backward compatibility
3. Added complete MLE infrastructure (WMRL_M3_BOUNDS, WMRL_M3_PARAMS, `--model wmrl_m3` CLI)
4. Created 24+ backward compatibility tests validating M3(κ=0) ≡ M2 to rtol=1e-5
5. Fixed critical bug: M3 likelihood now branches on κ=0 to use M2 probability mixing
6. Extended compare_mle_models.py for N-model comparison with Akaike weights

### Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Name: M3 (not wmrl_kappa) | Follows M1/M2 naming convention | ✓ Good |
| Extend WMRLHybridAgent | Avoid code duplication; kappa=0 gives M2 behavior | ✓ Good |
| Global (not stimulus-specific) perseveration | Captures motor-level response stickiness | ✓ Good |
| κ ∈ [0, 1] bounds | Matches Senta et al. parameter constraint convention | ✓ Good |
| Reset last_action at block boundaries | Matches existing Q/WM reset pattern | ✓ Good |
| Infrastructure only (no fits) | User runs fits on cluster | ✓ Good |
| M3 likelihood branches on kappa=0 | Ensures backward compatibility with M2 | ✓ Critical fix |

### Issues Resolved

- **Critical backward compatibility bug:** M3 was using value mixing while M2 uses probability mixing. Fixed by adding branching logic when κ=0.
- **BETA_WM_DEFAULT missing:** Pre-existing bug in wm_rl_hybrid.py fixed during agent extension.
- **Unicode encoding errors:** Greek characters caused Windows console errors, replaced with ASCII.

### Issues Deferred

None - all v1 requirements completed.

### Technical Debt Incurred

None - clean implementation with no shortcuts.

---

_For current project status, see .planning/PROJECT.md_
