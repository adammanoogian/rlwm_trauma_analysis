# Shrinkage Diagnostic Report

Formula: `Shrinkage = 1 - var(individual draws) / var(per-draw group mean)`

- `var(individual draws)`: variance across all MCMC draws AND all participants
- `var(per-draw group mean)`: variance of the per-draw group mean across iterations

| Parameter | Shrinkage | Status |
|-----------|-----------|--------|
| alpha_pos | -41.4375 | WARNING: poorly identified |
| alpha_neg | -1.2406 | WARNING: poorly identified |
| phi | -124.1765 | WARNING: poorly identified |
| rho | -48.1685 | WARNING: poorly identified |
| capacity | -36.8250 | WARNING: poorly identified |
| kappa_total | -13.2169 | WARNING: poorly identified |
| kappa_share | -61.7741 | WARNING: poorly identified |
| epsilon | -29.3020 | WARNING: poorly identified |

**Summary:** 0/8 parameters identified (shrinkage >= 0.3); 8 poorly identified.

> Parameters with shrinkage < 0.3 should be treated as descriptive only for downstream inference.