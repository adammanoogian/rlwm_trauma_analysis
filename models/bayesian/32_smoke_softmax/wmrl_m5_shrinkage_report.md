# Hierarchical Identifiability Report (ICC)

Formula: `ICC = var_between / (var_within + var_between)`

- `var_within`:  mean over participants of var-across-draws (per-participant posterior uncertainty)
- `var_between`: variance across participants of per-participant posterior mean (between-participant spread, with within-MCMC noise averaged out per participant first)

Range [0, 1]: 1.0 = participants well-distinguished by data; 0.0 = within-subject posterior uncertainty dominates.

| Parameter | ICC | Status |
|-----------|-----|--------|
| alpha_pos | 0.3540 | identified |
| alpha_neg | 0.0007 | WARNING: poorly identified |
| phi | 0.8864 | identified |
| rho | 0.1940 | WARNING: poorly identified |
| capacity | 0.0007 | WARNING: poorly identified |
| kappa | 0.8918 | identified |
| phi_rl | 0.2394 | WARNING: poorly identified |
| epsilon | 0.6645 | identified |

**Summary:** 4/8 parameters identified (ICC >= 0.3); 4 poorly identified.

> Parameters with ICC < 0.3 should be treated as descriptive only for downstream inference.