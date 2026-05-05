# Hierarchical Identifiability Report (ICC)

Formula: `ICC = var_between / (var_within + var_between)`

- `var_within`:  mean over participants of var-across-draws (per-participant posterior uncertainty)
- `var_between`: variance across participants of per-participant posterior mean (between-participant spread, with within-MCMC noise averaged out per participant first)

Range [0, 1]: 1.0 = participants well-distinguished by data; 0.0 = within-subject posterior uncertainty dominates.

| Parameter | ICC | Status |
|-----------|-----|--------|
| alpha_pos | 0.6452 | identified |
| alpha_neg | 0.0006 | WARNING: poorly identified |
| phi | 0.8752 | identified |
| rho | 0.3038 | identified |
| capacity | 0.0086 | WARNING: poorly identified |
| kappa | 0.8906 | identified |
| epsilon | 0.6660 | identified |

**Summary:** 5/7 parameters identified (ICC >= 0.3); 2 poorly identified.

> Parameters with ICC < 0.3 should be treated as descriptive only for downstream inference.