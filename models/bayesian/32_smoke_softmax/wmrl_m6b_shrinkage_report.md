# Shrinkage Diagnostic Report

Formula: `Shrinkage = 1 - var(individual draws) / var(per-draw group mean)`

- `var(individual draws)`: variance across all MCMC draws AND all participants
- `var(per-draw group mean)`: variance of the per-draw group mean across iterations

| Parameter | Shrinkage | Status |
|-----------|-----------|--------|
| alpha_pos | -39.3836 | WARNING: poorly identified |
| alpha_neg | -1.4679 | WARNING: poorly identified |
| phi | -124.5050 | WARNING: poorly identified |
| rho | -5.1039 | WARNING: poorly identified |
| capacity | -0.4360 | WARNING: poorly identified |
| kappa_total | -36.6100 | WARNING: poorly identified |
| kappa_share | -109.7117 | WARNING: poorly identified |
| epsilon | -63.1018 | WARNING: poorly identified |

**Summary:** 0/8 parameters identified (shrinkage >= 0.3); 8 poorly identified.

> Parameters with shrinkage < 0.3 should be treated as descriptive only for downstream inference.