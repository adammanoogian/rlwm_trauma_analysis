---
milestone: v3.0
audited: 2026-04-03T16:30:00Z
status: gaps_found
scores:
  requirements: 31/33
  phases: 5/5
  integration: 5/7
  flows: 4/6
gaps:
  integration:
    - "CRITICAL: M6a/M6b synthetic generation never applies perseveration (if/elif structural bug in model_recovery.py:321-352)"
    - "MEDIUM: M3/M5 synthetic generation uses additive renormalization but likelihood uses convex combination; ordering also differs (perseveration→epsilon vs epsilon→perseveration)"
  requirements:
    - "M6-11: Parameter recovery r >= 0.80 blocked — M6a/M6b synthetic data has zero perseveration"
    - "INTG-04: Cross-model recovery — M6a/M6b will be indistinguishable from M2 in confusion matrix"
tech_debt:
  - phase: 08-m5-rl-forgetting
    items:
      - "M5-07: Full parameter recovery (N=50) pending cluster run"
      - "M3/M5 perseveration formula mismatch in synthetic generation"
  - phase: 10-m6b-dual-perseveration
    items:
      - "M6-11: Parameter recovery blocked by synthetic generation bug"
  - phase: 11-m4-lba-joint-choice-rt
    items:
      - "M4-10: Full parameter recovery (N=50) pending cluster run"
      - "t0 upper bound (0.3s) exceeds min filtered RT (0.15s) — review t0 distribution in real fits"
  - phase: general
    items:
      - "Dead code: q_learning_step() in jax_likelihoods.py:328 — never called"
---

# v3.0 Milestone Audit Report

**Milestone:** v3.0 Model Extensions (M4-M6)
**Audited:** 2026-04-03
**Status:** GAPS FOUND

## Executive Summary

The **fitting pipeline** (scripts 12, 14, 15, 16) is fully and correctly integrated across all 7 models. All dispatch chains are complete, parameter counts match, stick-breaking decodes are consistent, and M4 float64 is properly isolated.

The **parameter recovery pipeline** (model_recovery.py, script 11) has two integration defects that must be fixed:

1. **CRITICAL: M6a/M6b zero perseveration in synthetic data** — structural if/elif bug
2. **MEDIUM: M3/M5 formula mismatch** — additive vs convex combination, ordering difference

## Requirements Coverage

| Requirement | Phase | Status | Notes |
|-------------|-------|--------|-------|
| M5-01..06 | Phase 8 | SATISFIED | All code verified |
| M5-07 | Phase 8 | PENDING | Cluster run needed; code correct |
| M6-01..06 | Phase 9 | SATISFIED | Likelihood code correct |
| M6-07..10 | Phase 10 | SATISFIED | Likelihood code correct |
| M6-11 | Phase 10 | BLOCKED | Synthetic generation bug prevents recovery validation |
| M4-01..09 | Phase 11 | SATISFIED | All code verified |
| M4-10 | Phase 11 | PENDING | Cluster run needed; code correct |
| INTG-01..03 | Phase 12 | SATISFIED | Scripts 14/15/16 handle all models |
| INTG-04 | Phase 12 | PARTIAL | Cross-model infra exists but M6a/M6b results will be wrong |
| INTG-05 | Phase 12 | SATISFIED | Docs updated |

**Score:** 31/33 requirements satisfied (2 blocked by bugs)

## Phase Verifications

| Phase | Status | Score | Gaps |
|-------|--------|-------|------|
| 8. M5 RL Forgetting | human_needed | 4/5 | Recovery pending cluster |
| 9. M6a Stim-Specific | passed | 6/6 | None (but recovery blocked by generation bug) |
| 10. M6b Dual | human_needed | 3/4 | Recovery pending cluster + generation bug |
| 11. M4 LBA | human_needed | 4/5 | Recovery pending cluster |
| 12. Integration | passed | 9/9 | None |

## Integration Check

### Verified Correct (41+ connections)

- All 7 models have complete dispatch chains in mle_utils.py (9+ functions)
- All 3 objective function variants (JAX/bounded/GPU) correctly import and call likelihoods
- M4 float64 isolation: lazy import only in M4 code paths, no contamination
- M4 separate comparison track: popped before choice-only AIC in script 14
- Scripts 15/16: all parameter dispatch branches present, defensive loading handles missing files
- M5 Q-decay ordering: correct in both likelihood and generation (before delta-rule)
- M6b stick-breaking: consistent decode across all 5 locations (3 objectives + generation + warmup)

### Critical Gap: M6a/M6b Synthetic Generation Bug

**File:** `scripts/fitting/model_recovery.py`, lines 321-352

**Bug:** The `elif` branches for M6a (line 333) and M6b (line 340) are siblings of `if last_action is not None:` (line 321), not nested inside it. After the first trial in any block, `last_action` is always an integer (not None), so the outer `if` is always True. The inner chain only matches M4, M3, and M5 — M6a/M6b fall through with no perseveration applied.

**Impact:**
- M6a synthetic data = M2 behavior (kappa_s never applied)
- M6b synthetic data = M2 behavior (neither kappa nor kappa_s applied)
- Parameter recovery for kappa_s, kappa_total, kappa_share will fail
- Cross-model confusion matrix will show M6a/M6b ≈ M2

**Fix:** Move M6a and M6b branches inside the perseveration block with correct logic.

### Medium Gap: M3/M5 Formula Mismatch

**File:** `scripts/fitting/model_recovery.py`, lines 327-331 vs `jax_likelihoods.py`, line 1156

**Mismatch 1 — Formula:**
- Likelihood: `P = (1-kappa)*P_noisy + kappa*Ck` (convex combination)
- Generation: `P[a] += kappa; P /= sum(P)` (additive renormalization)

**Mismatch 2 — Ordering:**
- Likelihood: epsilon → perseveration (noise first, then kappa mixing)
- Generation: perseveration → epsilon (kappa first, then noise)

**Impact:** Systematic bias in kappa recovery for M3/M5. Small for low kappa, grows with kappa magnitude. M4 is NOT affected (correctly uses convex combination).

**Fix:** Change M3/M5 generation to use convex combination with correct ordering.

## E2E Flow Status

| Flow | Status | Notes |
|------|--------|-------|
| Fit model (`12_fit_mle.py --model X`) | COMPLETE | All 7 models |
| Compare models (`14_compare_models.py`) | COMPLETE | M4 separate track |
| Trauma analysis (`15_analyze_mle_by_trauma.py --model all`) | COMPLETE | All 7 models |
| Regression (`16_regress_parameters_on_scales.py --model all`) | COMPLETE | All 7 models |
| Parameter recovery (`11_run_model_recovery.py`) | BROKEN for M6a/M6b | Zero perseveration |
| Cross-model recovery (`11 --mode cross-model`) | BROKEN for M6a/M6b | Inherits generation bug |

## Tech Debt

| Phase | Item | Severity |
|-------|------|----------|
| Phase 8 | M5 full recovery (N=50) pending cluster | Low (code correct) |
| Phase 8 | M3/M5 generation formula mismatch | Medium |
| Phase 10 | M6b full recovery blocked by bug | Critical |
| Phase 11 | M4 full recovery (N=50) pending cluster | Low (code correct) |
| Phase 11 | t0 upper bound review | Low (warning) |
| General | Dead code q_learning_step() | Info |

## Recommended Actions

1. **Fix M6a/M6b generation bug** — restructure if/elif in model_recovery.py
2. **Fix M3/M5 formula** — match likelihood's convex combination and ordering
3. **Re-run smoke tests** — verify M6a/M6b kappa_s now affects synthetic data
4. **Cluster runs** — M5, M6a, M6b, M4 full parameter recovery (N=50)
5. **Cross-model recovery** — full run after fixes

---

_Audited: 2026-04-03_
_Auditor: Claude (gsd-integration-checker + orchestrator)_
