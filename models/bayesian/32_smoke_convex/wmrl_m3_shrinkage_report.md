# Hierarchical Identifiability Report (ICC)

Formula: `ICC = var_between / (var_within + var_between)`

- `var_within`:  mean over participants of var-across-draws (per-participant posterior uncertainty)
- `var_between`: variance across participants of per-participant posterior mean (between-participant spread, with within-MCMC noise averaged out per participant first)

Range [0, 1]: 1.0 = participants well-distinguished by data; 0.0 = within-subject posterior uncertainty dominates.

| Parameter | ICC | Status |
|-----------|-----|--------|
| alpha_pos | 0.6765 | identified |
| alpha_neg | 0.0006 | WARNING: poorly identified |
| phi | 0.8041 | identified |
| rho | 0.7344 | identified |
| capacity | 0.6363 | identified |
| kappa | 0.7681 | identified |
| epsilon | 0.5950 | identified |

**Summary:** 6/7 parameters identified (ICC >= 0.3); 1 poorly identified.

> Parameters with ICC < 0.3 should be treated as descriptive only for downstream inference.