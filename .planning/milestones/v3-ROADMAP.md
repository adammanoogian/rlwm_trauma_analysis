# Milestone v3.0: Model Extensions (M4-M6)

**Status:** SHIPPED 2026-04-03
**Phases:** 8-12
**Total Plans:** 12 (including 1 gap closure)

## Overview

Extend the model hierarchy with RL forgetting (M5), stimulus-specific perseveration (M6a/M6b), and LBA joint choice+RT (M4), integrated into the full MLE fitting, comparison, and trauma-analysis pipeline.

## Phases

### Phase 8: M5 RL Forgetting
**Goal**: Users can fit an RL forgetting model where Q-values decay toward baseline each trial before the delta-rule update
**Depends on**: Nothing (additive extension to existing M3; same carry structure)
**Requirements**: M5-01, M5-02, M5-03, M5-04, M5-05, M5-06, M5-07
**Plans**: 2 plans

Plans:
- [x] 08-01-PLAN.md — Core M5 likelihood, parameter registration, and MLE fitting pipeline
- [x] 08-02-PLAN.md — Parameter recovery, model comparison, and trauma analysis integration

### Phase 9: M6a Stimulus-Specific Perseveration
**Goal**: Users can fit a perseveration model that tracks last-action per stimulus independently, rather than using a global scalar
**Depends on**: Phase 8 (pipeline integration pattern validated with M5)
**Requirements**: M6-01, M6-02, M6-03, M6-04, M6-05, M6-06
**Plans**: 2 plans

Plans:
- [x] 09-01-PLAN.md — Core M6a likelihood with per-stimulus carry, parameter registration, and MLE fitting pipeline
- [x] 09-02-PLAN.md — Parameter recovery (per-stimulus synthetic gen), model comparison, and trauma analysis integration

### Phase 10: M6b Dual Perseveration
**Goal**: Users can fit a dual-perseveration model combining global and stimulus-specific kernels with a constraint that their sum stays at or below 1
**Depends on**: Phase 9 (M6a validated in isolation before composing with global kernel)
**Requirements**: M6-07, M6-08, M6-09, M6-10, M6-11
**Plans**: 2 plans

Plans:
- [x] 10-01-PLAN.md — Core M6b likelihood with dual carry (global + per-stimulus), stick-breaking reparameterization, and MLE fitting pipeline
- [x] 10-02-PLAN.md — Parameter recovery (dual-kernel synthetic gen), model comparison, and trauma analysis integration

### Phase 11: M4 LBA Joint Choice+RT
**Goal**: Users can fit a model that accounts for both choice and reaction time via a Linear Ballistic Accumulator process with drift rates derived from the hybrid policy
**Depends on**: Phase 10 (all choice-only models validated; M4 is orthogonal but benefits from stable pipeline)
**Requirements**: M4-01, M4-02, M4-03, M4-04, M4-05, M4-06, M4-07, M4-08, M4-09, M4-10
**Plans**: 3 plans

Plans:
- [x] 11-01-PLAN.md — LBA density functions (float64), RT preprocessing utilities, standalone inline tests
- [x] 11-02-PLAN.md — M4 likelihood (M3 learning + LBA decision), parameter registration, MLE fitting pipeline
- [x] 11-03-PLAN.md — Parameter recovery (RT simulation), separate comparison track, trauma analysis integration

### Phase 12: Cross-Model Integration
**Goal**: All new models are integrated into the downstream comparison and trauma-analysis scripts, model recovery is validated across M4-M6, and documentation is updated
**Depends on**: Phase 11 (all models exist before integration pass)
**Requirements**: INTG-01, INTG-02, INTG-03, INTG-04, INTG-05
**Plans**: 3 plans (including 1 gap closure)

Plans:
- [x] 12-01-PLAN.md — Cross-model AIC recovery validation (extend run_model_recovery_check, add --mode cross-model to script 11)
- [x] 12-02-PLAN.md — Documentation update (MODEL_REFERENCE.md M3-M6 math, CLAUDE.md quick reference)
- [x] 12-03-PLAN.md — Gap closure: fix M3/M5/M6a/M6b synthetic generation to match likelihood formulas

## Milestone Summary

**Key Decisions:**
- Build order M5 -> M6a -> M6b -> M4 (complexity-ordered; M5 validates pipeline integration pattern)
- M4 gets separate comparison track (joint likelihood incommensurable with choice-only AIC)
- M5 phi_rl decay applied BEFORE delta-rule update (Senta et al.)
- M6b uses stick-breaking reparameterization: kappa = kappa_total * kappa_share
- M4 b > A enforced via reparameterization (b = A + delta)
- M4 float64 enabled lazily only when model==wmrl_m4
- All perseveration uses convex combination: (1-kappa)*P_noisy + kappa*Ck (Senta et al.)

**Issues Resolved:**
- M6a/M6b synthetic generation bug: elif branches unreachable (fixed in gap closure 12-03)
- M3/M5 generation formula mismatch: additive vs convex combination (fixed in gap closure 12-03)
- fit_mle.py --output argument: was passing file path instead of directory on Windows (fixed in 12-01)

**Issues Deferred:**
- Full parameter recovery (N=50, r >= 0.80) for M5, M6a, M6b, M4 — requires cluster compute
- Full cross-model recovery — requires cluster compute
- t0 upper bound review for M4 (0.3s exceeds min filtered RT of 0.15s)

**Technical Debt:**
- Dead code: q_learning_step() in jax_likelihoods.py (never called)
- Phase 6 (Cluster Monitoring) and Phase 7 (Publication Polish) deferred from v2.0

---
_Archived: 2026-04-03 as part of v3.0 milestone completion_
