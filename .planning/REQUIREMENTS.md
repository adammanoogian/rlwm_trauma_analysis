# Requirements: RLWM Trauma Analysis

**Defined:** 2026-02-05
**Core Value:** Correctly dissociate perseverative responding from learning-rate effects (alpha_neg) to accurately identify whether post-reversal failures reflect motor perseveration or outcome insensitivity

## v2 Requirements

Requirements for post-fitting validation and publication readiness. Each maps to roadmap phases.

### Continuous Regression & Visualization

- [ ] **REGR-01**: Script 16 output is cleanly organized by analysis (each scale x parameter regression grouped and structured)
- [ ] **REGR-02**: All scatter/regression plots in Scripts 15-16 accept `--color-by <variable>` to color data points by any categorical grouping variable
- [ ] **REGR-03**: Color-by grouping works with trauma group, gender, or any column present in the participant data

### Parameter Recovery

- [ ] **RECV-01**: `model_recovery.py` has a `main()` CLI accepting `--model`, `--n-subjects`, `--n-datasets`
- [ ] **RECV-02**: Parameter recovery loop: sample true params, generate synthetic data, fit via MLE, collect recovered params
- [ ] **RECV-03**: Recovery metrics computed per parameter: Pearson r, RMSE, bias (mean difference)
- [ ] **RECV-04**: Scatter plots generated for each parameter (true vs. recovered) with r-squared annotation
- [ ] **RECV-05**: Recovery results saved to CSV (true params, recovered params, metrics)
- [ ] **RECV-06**: Script 11 invokes parameter recovery pipeline and reports pass/fail per Senta r >= 0.80 criterion

### Cluster Monitoring

- [ ] **MNTR-01**: `run_mle_gpu.slurm` runs background `nvidia-smi` polling at configurable interval, logging to file
- [ ] **MNTR-02**: `fit_mle.py` `[MEMORY]` stdout lines are also written to a persistent CSV file alongside the fit results

### Publication Polish

- [ ] **PUBL-01**: `14_compare_models.py` accepts `--by-group` flag to run AIC/BIC comparison separately per trauma group
- [ ] **PUBL-02**: A combined results summary is generated: winning model per group + key parameter-trauma associations from Scripts 14-16

## Future Requirements

Deferred to later milestones. Tracked but not in current roadmap.

### Posterior Predictive Checks

- **PPC-01**: Posterior predictive check comparison pipeline (Script 09) comparing synthetic vs. observed data
- **PPC-02**: PPC visualization with overlay plots (learning curves, set-size effects, accuracy distributions)
- **PPC-03**: Goodness-of-fit statistics (KS tests or similar) for synthetic vs. observed distributions

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
| REGR-01 | TBD | Pending |
| REGR-02 | TBD | Pending |
| REGR-03 | TBD | Pending |
| RECV-01 | TBD | Pending |
| RECV-02 | TBD | Pending |
| RECV-03 | TBD | Pending |
| RECV-04 | TBD | Pending |
| RECV-05 | TBD | Pending |
| RECV-06 | TBD | Pending |
| MNTR-01 | TBD | Pending |
| MNTR-02 | TBD | Pending |
| PUBL-01 | TBD | Pending |
| PUBL-02 | TBD | Pending |

**Coverage:**
- v2 requirements: 13 total
- Mapped to phases: 0
- Unmapped: 13 (awaiting roadmap)

---
*Requirements defined: 2026-02-05*
*Last updated: 2026-02-05 after initial definition*
