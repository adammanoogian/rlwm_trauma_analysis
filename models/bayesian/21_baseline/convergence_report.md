# Step 21.4 — Baseline Convergence Audit Report

Gate criteria (Baribault & Collins, 2023): R-hat <= 1.05 AND ESS_bulk >= 400.0 AND divergences == 0 AND BFMI >= 0.2.

## Summary

- n_passing: 0
- n_excluded: 6
- models_proceeding_to_loo: (none)
- models_excluded: qlearning, wmrl, wmrl_m3, wmrl_m5, wmrl_m6a, wmrl_m6b

## Per-model audit

### qlearning

- **Gate status:** FAIL
- **Pipeline action:** `EXCLUDED_MISSING_FILE`
- max R-hat: nan
- min ESS_bulk: nan
- divergences: 0
- min BFMI: nan
- PPC coverage: not_available

**Notes:** Posterior NetCDF missing at models/bayesian/21_baseline/qlearning_posterior.nc. Step 21.3 fit either hit the convergence gate inside save_results (which returns None and writes nothing) or the SLURM job itself failed before save_results was called. Check logs/bayesian_21_3_*.{out,err} for the root cause.

### wmrl

- **Gate status:** FAIL
- **Pipeline action:** `EXCLUDED_MISSING_FILE`
- max R-hat: nan
- min ESS_bulk: nan
- divergences: 0
- min BFMI: nan
- PPC coverage: not_available

**Notes:** Posterior NetCDF missing at models/bayesian/21_baseline/wmrl_posterior.nc. Step 21.3 fit either hit the convergence gate inside save_results (which returns None and writes nothing) or the SLURM job itself failed before save_results was called. Check logs/bayesian_21_3_*.{out,err} for the root cause.

### wmrl_m3

- **Gate status:** FAIL
- **Pipeline action:** `EXCLUDED_MISSING_FILE`
- max R-hat: nan
- min ESS_bulk: nan
- divergences: 0
- min BFMI: nan
- PPC coverage: not_available

**Notes:** Posterior NetCDF missing at models/bayesian/21_baseline/wmrl_m3_posterior.nc. Step 21.3 fit either hit the convergence gate inside save_results (which returns None and writes nothing) or the SLURM job itself failed before save_results was called. Check logs/bayesian_21_3_*.{out,err} for the root cause.

### wmrl_m5

- **Gate status:** FAIL
- **Pipeline action:** `EXCLUDED_MISSING_FILE`
- max R-hat: nan
- min ESS_bulk: nan
- divergences: 0
- min BFMI: nan
- PPC coverage: not_available

**Notes:** Posterior NetCDF missing at models/bayesian/21_baseline/wmrl_m5_posterior.nc. Step 21.3 fit either hit the convergence gate inside save_results (which returns None and writes nothing) or the SLURM job itself failed before save_results was called. Check logs/bayesian_21_3_*.{out,err} for the root cause.

### wmrl_m6a

- **Gate status:** FAIL
- **Pipeline action:** `EXCLUDED_MISSING_FILE`
- max R-hat: nan
- min ESS_bulk: nan
- divergences: 0
- min BFMI: nan
- PPC coverage: not_available

**Notes:** Posterior NetCDF missing at models/bayesian/21_baseline/wmrl_m6a_posterior.nc. Step 21.3 fit either hit the convergence gate inside save_results (which returns None and writes nothing) or the SLURM job itself failed before save_results was called. Check logs/bayesian_21_3_*.{out,err} for the root cause.

### wmrl_m6b

- **Gate status:** FAIL
- **Pipeline action:** `EXCLUDED_MISSING_FILE`
- max R-hat: nan
- min ESS_bulk: nan
- divergences: 0
- min BFMI: nan
- PPC coverage: not_available

**Notes:** Posterior NetCDF missing at models/bayesian/21_baseline/wmrl_m6b_posterior.nc. Step 21.3 fit either hit the convergence gate inside save_results (which returns None and writes nothing) or the SLURM job itself failed before save_results was called. Check logs/bayesian_21_3_*.{out,err} for the root cause.
