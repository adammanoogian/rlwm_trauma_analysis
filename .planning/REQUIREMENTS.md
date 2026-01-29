# Requirements: WM-RL M3 (Perseveration Extension)

**Defined:** 2026-01-28
**Core Value:** Dissociate perseverative responding from learning-rate effects for accurate trauma analysis

**Model naming:**
- M1: Q-learning
- M2: WM-RL hybrid
- M3: WM-RL + κ perseveration ← this project

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Likelihood Function

- [ ] **LIK-01**: JAX likelihood function `wmrl_m3_block_likelihood()` computes log-likelihood for M3 on a single block
- [ ] **LIK-02**: Likelihood includes κ·Rep(a) additive term in softmax: P(a) ∝ exp(V(s,a) + κ·Rep(a))
- [ ] **LIK-03**: Rep(a) tracks whether action matches previous trial's action (global, not stimulus-specific)
- [ ] **LIK-04**: Last action resets to None at start of each block
- [ ] **LIK-05**: Multi-block wrapper `wmrl_m3_multiblock_likelihood()` sums across blocks

### Agent Integration

- [ ] **AGT-01**: WMRLHybridAgent extended with optional `kappa` parameter (default=0 preserves M2 behavior)

### Parameter Infrastructure

- [ ] **PAR-01**: κ parameter bounded to [0, 1] in mle_utils.py
- [ ] **PAR-02**: WMRL_M3_PARAMS list and WMRL_M3_BOUNDS dict defined
- [ ] **PAR-03**: Parameter transformation functions support 'wmrl_m3' model type

### MLE Fitting Infrastructure

- [ ] **FIT-01**: `_objective_wmrl_m3()` objective function in fit_mle.py
- [ ] **FIT-02**: fit_mle.py CLI accepts `--model wmrl_m3` option
- [ ] **FIT-03**: Fitting infrastructure uses same 20 random starts methodology as M1/M2

### Validation & Comparison

- [ ] **VAL-01**: κ=0 produces identical results to M2 (backward compatibility)
- [ ] **CMP-01**: AIC/BIC computed for M3 fits (7 parameters)
- [ ] **CMP-02**: Comparison utilities support M2 vs M3 comparison

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Extended Models

- **EXT-01**: Q-learning+κ variant (M1 + perseveration)
- **EXT-02**: Stimulus-specific perseveration variant
- **EXT-03**: NumPyro hierarchical Bayesian model for M3

### Analysis (after fitting on cluster)

- **ANL-01**: Trauma group comparison of κ parameter
- **ANL-02**: Regression of κ on IES-R/LESS scales
- **ANL-03**: Correlation analysis: κ vs α₋

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Running actual fits | User will run on cluster; we build infrastructure only |
| Duplicate agent class | Extend existing WMRLHybridAgent with kappa parameter |
| Switch-seeking (negative κ) | Theoretical focus is on perseveration; bounds [0,1] |
| Different β for perseveration term | Matches existing model where β=50 is shared |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| LIK-01 | Phase 1 | Pending |
| LIK-02 | Phase 1 | Pending |
| LIK-03 | Phase 1 | Pending |
| LIK-04 | Phase 1 | Pending |
| LIK-05 | Phase 1 | Pending |
| AGT-01 | Phase 1 | Pending |
| PAR-01 | Phase 2 | Pending |
| PAR-02 | Phase 2 | Pending |
| PAR-03 | Phase 2 | Pending |
| FIT-01 | Phase 2 | Pending |
| FIT-02 | Phase 2 | Pending |
| FIT-03 | Phase 2 | Pending |
| VAL-01 | Phase 3 | Pending |
| CMP-01 | Phase 3 | Pending |
| CMP-02 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0 ✓

---
*Requirements defined: 2026-01-28*
*Last updated: 2026-01-29 after roadmap adjustment*
