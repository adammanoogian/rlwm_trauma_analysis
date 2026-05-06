# Hierarchical Identifiability Report (ICC)

Formula: `ICC = var_between / (var_within + var_between)`

- `var_within`:  mean over participants of var-across-draws (per-participant posterior uncertainty)
- `var_between`: variance across participants of per-participant posterior mean (between-participant spread, with within-MCMC noise averaged out per participant first)

Range [0, 1]: 1.0 = participants well-distinguished by data; 0.0 = within-subject posterior uncertainty dominates.

| Parameter | ICC | Status |
|-----------|-----|--------|
| alpha_pos | 0.6743 | identified |
| alpha_neg | 0.0001 | WARNING: poorly identified |
| phi | 0.7680 | identified |
| rho | 0.6241 | identified |
| capacity | 0.5988 | identified |
| kappa_total | 0.8678 | identified |
| kappa_share | 0.6900 | identified |
| epsilon | 0.4533 | identified |

**Summary:** 7/8 parameters identified (ICC >= 0.3); 1 poorly identified.

> Parameters with ICC < 0.3 should be treated as descriptive only for downstream inference.