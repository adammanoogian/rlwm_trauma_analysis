# Hierarchical Identifiability Report (ICC)

Formula: `ICC = var_between / (var_within + var_between)`

- `var_within`:  mean over participants of var-across-draws (per-participant posterior uncertainty)
- `var_between`: mean over draws of var-across-participants (between-participant spread)

Range [0, 1]: 1.0 = participants well-distinguished by data; 0.0 = within-subject posterior uncertainty dominates.

| Parameter | ICC | Status |
|-----------|-----|--------|
| alpha_pos | 0.6487 | identified |
| alpha_neg | 0.0007 | WARNING: poorly identified |
| phi | 0.8253 | identified |
| rho | 0.7342 | identified |
| capacity | 0.6676 | identified |
| kappa_total | 0.2564 | WARNING: poorly identified |
| kappa_share | 0.7310 | identified |
| epsilon | 0.5361 | identified |

**Summary:** 6/8 parameters identified (ICC >= 0.3); 2 poorly identified.

> Parameters with ICC < 0.3 should be treated as descriptive only for downstream inference.