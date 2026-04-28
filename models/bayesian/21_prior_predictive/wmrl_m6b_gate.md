# Prior predictive gate: wmrl_m6b

- **Verdict:** **HARD_PASS**
- Draws: 500
- Seed: 42
- Real draws simulated: 500

## Gate checks (Baribault & Collins 2023; three-tier policy)

Per-metric band: `hard` = within original B&C hard band; `soft` = within documented soft margin (advisory); `no` = outside both bands.

| Check | Hard threshold | Soft threshold | Value | Band |
|---|---|---|---|---|
| Median accuracy | [0.40, 0.90] | [0.40, 0.92] | 0.617 | hard |
| Sub-chance fraction | < 10% | < 12% | 1.6% | hard |
| At-ceiling fraction | < 5% | < 7% | 0.0% | hard |

## Priors used

```
alpha_neg: {'lower': 0.0, 'upper': 1.0, 'mu_prior_loc': 0.0}
alpha_pos: {'lower': 0.0, 'upper': 1.0, 'mu_prior_loc': 0.0}
capacity: {'lower': 2.0, 'upper': 6.0, 'mu_prior_loc': 0.0}
epsilon: {'lower': 0.0, 'upper': 1.0, 'mu_prior_loc': -2.0}
kappa: {'lower': 0.0, 'upper': 1.0, 'mu_prior_loc': 0.0}
kappa_s: {'lower': 0.0, 'upper': 1.0, 'mu_prior_loc': 0.0}
kappa_share: {'lower': 0.0, 'upper': 1.0, 'mu_prior_loc': 0.0}
kappa_total: {'lower': 0.0, 'upper': 1.0, 'mu_prior_loc': 0.0}
phi: {'lower': 0.0, 'upper': 1.0, 'mu_prior_loc': 0.0}
phi_rl: {'lower': 0.0, 'upper': 1.0, 'mu_prior_loc': 0.0}
rho: {'lower': 0.0, 'upper': 1.0, 'mu_prior_loc': 0.0}
```
