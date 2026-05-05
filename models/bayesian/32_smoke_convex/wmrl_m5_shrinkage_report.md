# Hierarchical Identifiability Report (ICC)

Formula: `ICC = var_between / (var_within + var_between)`

- `var_within`:  mean over participants of var-across-draws (per-participant posterior uncertainty)
- `var_between`: variance across participants of per-participant posterior mean (between-participant spread, with within-MCMC noise averaged out per participant first)

Range [0, 1]: 1.0 = participants well-distinguished by data; 0.0 = within-subject posterior uncertainty dominates.

| Parameter | ICC | Status |
|-----------|-----|--------|
| alpha_pos | 0.6000 | identified |
| alpha_neg | 0.0010 | WARNING: poorly identified |
| phi | 0.5135 | identified |
| rho | 0.7450 | identified |
| capacity | 0.3985 | identified |
| kappa | 0.7545 | identified |
| phi_rl | 0.2901 | WARNING: poorly identified |
| epsilon | 0.5915 | identified |

**Summary:** 6/8 parameters identified (ICC >= 0.3); 2 poorly identified.

> Parameters with ICC < 0.3 should be treated as descriptive only for downstream inference.