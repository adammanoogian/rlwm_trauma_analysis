# Requirements: WM-RL+κ Perseveration Model

**Defined:** 2026-01-28
**Core Value:** Dissociate perseverative responding from learning-rate effects for accurate trauma analysis

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Likelihood Function

- [ ] **LIK-01**: JAX likelihood function `wmrl_kappa_block_likelihood()` computes log-likelihood for WM-RL+κ on a single block
- [ ] **LIK-02**: Likelihood includes κ·Rep(a) additive term in softmax: P(a) ∝ exp(V(s,a) + κ·Rep(a))
- [ ] **LIK-03**: Rep(a) tracks whether action matches previous trial's action (global, not stimulus-specific)
- [ ] **LIK-04**: Last action resets to None at start of each block
- [ ] **LIK-05**: Multi-block wrapper `wmrl_kappa_multiblock_likelihood()` sums across blocks

### Parameter Infrastructure

- [ ] **PAR-01**: κ parameter bounded to [0, 1] in mle_utils.py
- [ ] **PAR-02**: WMRL_KAPPA_PARAMS list and WMRL_KAPPA_BOUNDS dict defined
- [ ] **PAR-03**: Parameter transformation functions support 'wmrl_kappa' model type

### MLE Fitting

- [ ] **FIT-01**: `_objective_wmrl_kappa()` objective function in fit_mle.py
- [ ] **FIT-02**: fit_mle.py CLI accepts `--model wmrl_kappa` option
- [ ] **FIT-03**: Fitting uses same 20 random starts methodology as existing models

### Model Comparison

- [ ] **CMP-01**: AIC/BIC computed for WM-RL+κ fits
- [ ] **CMP-02**: Can compare WM-RL vs WM-RL+κ using existing comparison utilities

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Extended Models

- **EXT-01**: Q-learning+κ variant (perseveration on pure Q-learning model)
- **EXT-02**: Stimulus-specific perseveration variant
- **EXT-03**: NumPyro hierarchical Bayesian model for WM-RL+κ

### Analysis

- **ANL-01**: Trauma group comparison of κ parameter
- **ANL-02**: Regression of κ on IES-R/LESS scales
- **ANL-03**: Correlation analysis: κ vs α₋

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Agent class (WMRLKappaAgent) | Only need likelihood for fitting, not simulation |
| Switch-seeking (negative κ) | Theoretical focus is on perseveration; bounds [0,1] |
| Different β for perseveration term | Matches existing model where β=50 is shared |
| Trial-to-trial κ variation | Keep κ static per participant, matching other parameters |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| LIK-01 | Phase 1 | Pending |
| LIK-02 | Phase 1 | Pending |
| LIK-03 | Phase 1 | Pending |
| LIK-04 | Phase 1 | Pending |
| LIK-05 | Phase 1 | Pending |
| PAR-01 | Phase 2 | Pending |
| PAR-02 | Phase 2 | Pending |
| PAR-03 | Phase 2 | Pending |
| FIT-01 | Phase 3 | Pending |
| FIT-02 | Phase 3 | Pending |
| FIT-03 | Phase 3 | Pending |
| CMP-01 | Phase 3 | Pending |
| CMP-02 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 13 total
- Mapped to phases: 13
- Unmapped: 0 ✓

---
*Requirements defined: 2026-01-28*
*Last updated: 2026-01-28 after initial definition*
