# Roadmap: WM-RL M3 (Perseveration Extension)

## Overview

This roadmap extends the existing WM-RL model (M2) with a perseveration parameter (κ) to create WM-RL M3. The goal is to dissociate outcome-insensitive action repetition from learning-rate effects.

**Model naming convention:**
- M1: Q-learning (α₊, α₋, ε)
- M2: WM-RL hybrid (α₊, α₋, φ, ρ, K, ε)
- M3: WM-RL + perseveration (α₊, α₋, φ, ρ, K, ε, κ) ← **this roadmap**

**Integration approach:** Extend existing code rather than duplicate. The WMRLHybridAgent class and likelihood functions will be parameterized to optionally include κ.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Core Implementation** - Likelihood function and agent extension with κ parameter
- [ ] **Phase 2: MLE Infrastructure** - Parameter bounds, transformations, fitting integration
- [ ] **Phase 3: Validation & Comparison** - Tests and model comparison (M2 vs M3)

## Phase Details

### Phase 1: Core Implementation
**Goal**: JAX likelihood and agent class support κ perseveration parameter
**Depends on**: Nothing (extends existing code)
**Requirements**: LIK-01, LIK-02, LIK-03, LIK-04, LIK-05, AGT-01
**Success Criteria** (what must be TRUE):
  1. `wmrl_m3_block_likelihood()` function computes log-likelihood with κ·Rep(a) term
  2. Rep(a) = I[a = a_{t-1}] tracks global action repetition (not stimulus-specific)
  3. Last action resets at block start (no carry-over between blocks)
  4. `wmrl_m3_multiblock_likelihood()` sums across blocks
  5. WMRLHybridAgent extended with optional `kappa` parameter (default 0 = M2 behavior)
  6. Agent's `get_hybrid_probs()` includes κ·Rep(a) when kappa > 0

Plans:
- [x] 01-01-PLAN.md - JAX M3 likelihood functions with kappa perseveration
- [x] 01-02-PLAN.md - WMRLHybridAgent extension with optional kappa parameter

### Phase 2: MLE Infrastructure
**Goal**: MLE fitting utilities support wmrl_m3 model type
**Depends on**: Phase 1
**Requirements**: PAR-01, PAR-02, PAR-03, FIT-01, FIT-02, FIT-03
**Success Criteria** (what must be TRUE):
  1. WMRL_M3_BOUNDS dict includes κ ∈ [0, 1]
  2. WMRL_M3_PARAMS list: [alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon]
  3. `unconstrained_to_params()` and `params_to_unconstrained()` support 'wmrl_m3'
  4. fit_mle.py accepts `--model wmrl_m3` CLI option
  5. `_objective_wmrl_m3()` uses wmrl_m3_multiblock_likelihood
  6. Fitting uses 20 random starts (same methodology as M1/M2)

**Plans:** 2 plans

Plans:
- [ ] 02-01-PLAN.md - Add wmrl_m3 parameter infrastructure to mle_utils.py
- [ ] 02-02-PLAN.md - Add wmrl_m3 fitting support to fit_mle.py

### Phase 3: Validation & Comparison
**Goal**: Researcher can validate M3 and compare against M2 baseline
**Depends on**: Phase 2
**Requirements**: CMP-01, CMP-02, VAL-01
**Success Criteria** (what must be TRUE):
  1. AIC/BIC computed for M3 fits (7 free parameters)
  2. Model comparison works: M2 (6 params) vs M3 (7 params)
  3. Test script validates likelihood computation matches expected behavior
  4. κ=0 produces identical results to M2 (backward compatibility)

Plans:
- [ ] 03-01: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Core Implementation | 2/2 | Complete | 2026-01-29 |
| 2. MLE Infrastructure | 0/2 | Ready | - |
| 3. Validation & Comparison | 0/TBD | Not started | - |
