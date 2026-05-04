# Hierarchical Identifiability Report (ICC)

Formula: `ICC = var_between / (var_within + var_between)`

- `var_within`:  mean over participants of var-across-draws (per-participant posterior uncertainty)
- `var_between`: mean over draws of var-across-participants (between-participant spread)

Range [0, 1]: 1.0 = participants well-distinguished by data; 0.0 = within-subject posterior uncertainty dominates.

| Parameter | ICC | Status |
|-----------|-----|--------|
| alpha_pos | 0.7292 | identified |
| alpha_neg | 0.7085 | identified |
| epsilon | 0.5320 | identified |

**Summary:** 3/3 parameters identified (ICC >= 0.3); 0 poorly identified.

> Parameters with ICC < 0.3 should be treated as descriptive only for downstream inference.