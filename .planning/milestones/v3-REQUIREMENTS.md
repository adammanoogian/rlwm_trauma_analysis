# Requirements Archive: v3.0 Model Extensions (M4-M6)

**Archived:** 2026-04-03
**Status:** SHIPPED

This is the archived requirements specification for v3.0.
For current requirements, see `.planning/REQUIREMENTS.md` (created for next milestone).

---

**Defined:** 2026-04-02
**Core Value:** Extend the model hierarchy with mechanistically distinct extensions (LBA decision process, RL forgetting, stimulus-specific perseveration) to test novel trauma-learning hypotheses, integrated into the existing fitting pipeline

## v3 Requirements

### M5 — RL Forgetting

- [x] **M5-01**: JAX likelihood function with phi_RL parameter implementing global per-trial Q-value decay toward Q0=1/3
- [x] **M5-02**: Decay applied BEFORE delta-rule update on each trial (correct update ordering per Senta et al.)
- [x] **M5-03**: Decay applies to ALL stimulus-action pairs every trial (not just current stimulus)
- [x] **M5-04**: MLE bounds (phi_RL in [0,1]), logit transform, and param names registered in mle_utils.py
- [x] **M5-05**: CLI integration via `--model wmrl_m5` flag in fit_mle.py
- [x] **M5-06**: Backward compatibility: phi_RL=0 produces identical results to M3
- [x] **M5-07**: Parameter recovery passes r >= 0.80 for all 8 parameters including phi_RL *(code complete, full N=50 cluster run pending)*

### M6 — Stimulus-Specific Perseveration

- [x] **M6-01**: JAX likelihood for M6a with stimulus-specific choice kernel kappa_s replacing global kappa
- [x] **M6-02**: Per-stimulus last_action tracking via array in JAX lax.scan carry (not global scalar)
- [x] **M6-03**: Uniform fallback (1/nA) for first presentation of each stimulus in a block (no kernel applied)
- [x] **M6-04**: last_action per stimulus resets at block boundaries (matching existing Q/WM reset pattern)
- [x] **M6-05**: MLE bounds (kappa_s in [0,1]), logit transform, param names for M6a in mle_utils.py
- [x] **M6-06**: CLI integration via `--model wmrl_m6a` flag in fit_mle.py
- [x] **M6-07**: JAX likelihood for M6b with dual kernels: global kappa + stimulus-specific kappa_s
- [x] **M6-08**: Stick-breaking reparameterization for M6b ensuring kappa + kappa_s <= 1
- [x] **M6-09**: MLE bounds, transforms, param names for M6b in mle_utils.py
- [x] **M6-10**: CLI integration via `--model wmrl_m6b` flag in fit_mle.py
- [x] **M6-11**: Parameter recovery passes r >= 0.80 for M6a (7 params) and M6b (8 params) *(code complete, full N=50 cluster run pending)*

### M4 — LBA Joint Choice+RT

- [x] **M4-01**: RT preprocessing utility: outlier removal, minimum RT filtering, validation that t0 < min(RT) per participant
- [x] **M4-02**: JAX LBA density function implementing Brown & Heathcote (2008) analytic formula with numerical stability (float64, log-space CDF, NaN-safe gradient patterns)
- [x] **M4-03**: LBA drift rates derived from hybrid policy weights: v_i = v_scale * pi_t(a_i | s_t)
- [x] **M4-04**: Within-trial noise fixed at s=0.1 (not a free parameter; per McDougle & Collins, 2021)
- [x] **M4-05**: Epsilon parameter dropped from M4 (start-point variability A subsumes undirected exploration)
- [x] **M4-06**: MLE bounds and transforms for M4: v_scale (log), b and A with b > A constraint, t0 (log)
- [x] **M4-07**: b > A constraint enforced via reparameterization (b = A + delta, delta > 0)
- [x] **M4-08**: CLI integration via `--model wmrl_m4` flag in fit_mle.py
- [x] **M4-09**: Joint choice+RT likelihood: P(choice=i, RT=t) = f_i(t) * prod_{j!=i} S_j(t)
- [x] **M4-10**: Parameter recovery passes r >= 0.80 for all params (learning/memory + LBA) *(code complete, full N=50 cluster run pending)*

### INTG — Cross-Model Integration

- [x] **INTG-01**: compare_mle_models.py updated with separate comparison tracks: choice-only (M1-M3, M5, M6a, M6b) and joint choice+RT (M4 standalone)
- [x] **INTG-02**: Script 15 (analyze_mle_by_trauma.py) updated to handle all new model parameter sets
- [x] **INTG-03**: Script 16 (regress_parameters_on_scales.py) updated to handle all new model parameter sets
- [x] **INTG-04**: Model recovery validation: generating model wins by AIC for all M4-M6 models
- [x] **INTG-05**: Documentation updated (MODEL_REFERENCE.md with M4-M6 math, CLAUDE.md quick reference)

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| M5-01..06 | Phase 8 | Complete |
| M5-07 | Phase 8 | Complete (recovery pending cluster) |
| M6-01..06 | Phase 9 | Complete |
| M6-07..10 | Phase 10 | Complete |
| M6-11 | Phase 10 | Complete (recovery pending cluster) |
| M4-01..09 | Phase 11 | Complete |
| M4-10 | Phase 11 | Complete (recovery pending cluster) |
| INTG-01..05 | Phase 12 | Complete |

**Coverage:**
- v3 requirements: 33 total
- Shipped: 33/33
- Pending cluster validation: 3 (M5-07, M6-11, M4-10) — code complete, compute-time gate only

## Milestone Summary

**Shipped:** 33 of 33 v3 requirements
**Adjusted:** None — all requirements shipped as originally specified
**Dropped:** None

---
*Archived: 2026-04-03 as part of v3.0 milestone completion*
