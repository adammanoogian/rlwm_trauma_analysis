# Requirements: RLWM Trauma Analysis

**Defined:** 2026-02-05
**Core Value:** Correctly dissociate perseverative responding from learning-rate effects (alpha_neg) to accurately identify whether post-reversal failures reflect motor perseveration or outcome insensitivity

## v2 Requirements

Requirements for post-fitting validation and publication readiness. Each maps to roadmap phases.

### Continuous Regression & Visualization

- [x] **REGR-01**: Script 16 output is cleanly organized by analysis (each scale x parameter regression grouped and structured)
- [x] **REGR-02**: All scatter/regression plots in Scripts 15-16 accept `--color-by <variable>` to color data points by any categorical grouping variable
- [x] **REGR-03**: Color-by grouping works with trauma group, gender, or any column present in the participant data

### Parameter Recovery

- [x] **RECV-01**: `model_recovery.py` has a `main()` CLI accepting `--model`, `--n-subjects`, `--n-datasets`
- [x] **RECV-02**: Parameter recovery loop: sample true params, generate synthetic data, fit via MLE, collect recovered params
- [x] **RECV-03**: Recovery metrics computed per parameter: Pearson r, RMSE, bias (mean difference)
- [x] **RECV-04**: Scatter plots generated for each parameter (true vs. recovered) with r-squared annotation
- [x] **RECV-05**: Recovery results saved to CSV (true params, recovered params, metrics)
- [x] **RECV-06**: Script 11 invokes parameter recovery pipeline and reports pass/fail per Senta r >= 0.80 criterion

### Posterior Predictive Checks

- [ ] **PPC-01**: `model_recovery.py --mode ppc` loads fitted params from MLE results and generates synthetic data
- [ ] **PPC-02**: Behavioral comparison metrics (accuracy by set-size, learning curves, post-reversal behavior)
- [ ] **PPC-03**: Overlay plots comparing real vs synthetic behavioral patterns
- [ ] **PPC-04**: Model recovery evaluation - fit all models to synthetic data, report if generative model wins
- [ ] **PPC-05**: Script 09 orchestrates full PPC pipeline (generate → analyze → compare → model recovery)

### Cluster Monitoring

- [ ] **MNTR-01**: `run_mle_gpu.slurm` runs background `nvidia-smi` polling at configurable interval, logging to file
- [ ] **MNTR-02**: `fit_mle.py` `[MEMORY]` stdout lines are also written to a persistent CSV file alongside the fit results

### Publication Polish

- [ ] **PUBL-01**: `14_compare_models.py` accepts `--by-group` flag to run AIC/BIC comparison separately per trauma group
- [ ] **PUBL-02**: A combined results summary is generated: winning model per group + key parameter-trauma associations from Scripts 14-16

## Future Requirements

Deferred to later milestones. Tracked but not in current roadmap.

### Bayesian Comparison

- **BAYES-01**: WAIC/LOO Bayesian model comparison in Script 14

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Running actual fits on cluster | User runs fits; we build infrastructure |
| New model variants (M4+) | v2 validates existing M1-M3 |
| Bayesian hierarchical fitting | MLE-focused for v2 |
| Plain-language parameter interpretation | User decided not needed for v2 |
| Stimulus-specific perseveration | Global action repetition sufficient (v1 decision) |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| REGR-01 | Phase 4 | Complete |
| REGR-02 | Phase 4 | Complete |
| REGR-03 | Phase 4 | Complete |
| RECV-01 | Phase 5 | Complete |
| RECV-02 | Phase 5 | Complete |
| RECV-03 | Phase 5 | Complete |
| RECV-04 | Phase 5 | Complete |
| RECV-05 | Phase 5 | Complete |
| RECV-06 | Phase 5 | Complete |
| PPC-01 | Phase 5 | Pending |
| PPC-02 | Phase 5 | Pending |
| PPC-03 | Phase 5 | Pending |
| PPC-04 | Phase 5 | Pending |
| PPC-05 | Phase 5 | Pending |
| MNTR-01 | Phase 6 | Pending |
| MNTR-02 | Phase 6 | Pending |
| PUBL-01 | Phase 7 | Pending |
| PUBL-02 | Phase 7 | Pending |

**Coverage:**
- v2 requirements: 18 total
- Mapped to phases: 18/18 ✓
- Unmapped: 0

---
*Requirements defined: 2026-02-05*
*Last updated: 2026-02-06 — added PPC requirements to Phase 5*
