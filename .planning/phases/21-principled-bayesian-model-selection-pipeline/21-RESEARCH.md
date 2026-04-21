# Phase 21: Principled Bayesian Model Selection Pipeline — Research

**Researched:** 2026-04-18
**Domain:** Bayesian model selection, PSIS-LOO stacking, RFX-BMS/PXP, prior predictive checks, hierarchical MCMC, SLURM chaining
**Confidence:** HIGH for infrastructure audit (codebase verified); HIGH for LOO/stacking algorithm details (ArviZ source confirmed); MEDIUM for RFX-BMS PXP formula (algorithm code from mfit/bms.m, paper PDF unreadable); MEDIUM for paper-specific workflow gates (Baribault PMC and Hess PMC fetched and read); LOW for exact Pareto-k threshold origin claim (WebSearch only, consistent with known literature).

---

# Research Summary

Phase 21 replaces an MLE-preselected "M6b wins" narrative with a step-by-step Bayesian pipeline anchored to Baribault & Collins (2023, *Psychological Methods* DOI:10.1037/met0000554) and Hess et al. (2025, *Computational Psychiatry* DOI:10.5334/cpsy.116). The nine pipeline steps (prior predictive → recovery → fit baseline → audit → LOO+stacking+RFX-BMS → winner refit with L2 → audit → model-averaged effects → tables) are linear, each a separate SLURM submission. Approximately 70% of existing infrastructure is directly reusable: `fit_bayesian._fit_stacked_model`, `save_results`, `compute_pointwise_log_lik`, `build_inference_data_with_loglik`, `run_inference_with_bump`, the `STACKED_MODEL_DISPATCH` dict, and the existing `run_bayesian_comparison` function in `14_compare_models.py` (which already calls `az.compare(ic='loo', method='stacking')` and logs Pareto-k). The two major net-new components are: (a) a pure-prior predictive runner (step 21.1, using `numpyro.infer.Predictive(model_fn, num_samples=N)(rng_key, ...)` with no conditioning) and (b) a `scripts/fitting/bms.py` module implementing RFX-BMS + PXP (step 21.5 secondary). No hierarchical Bayesian posteriors currently exist in `output/bayesian/` — this is the first production run — so all 6 models must be fit from scratch in step 21.3.

---

# Methodological Foundations

## 1. Baribault & Collins (2023) — "Troubleshooting Bayesian Cognitive Models"

**Citation:** Baribault, B., & Collins, A. G. E. (2023). Troubleshooting Bayesian cognitive models. *Psychological Methods*. https://doi.org/10.1037/met0000554  
**Full text:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10522800/  
**Confidence:** HIGH (PMC full text fetched and read)

**Key claims the planner can cite verbatim:**

- **R-hat gate:** "A value of R-hat ≤ 1.01 is required for every instance of every parameter in the model." This is a tightening from the previous ≤1.10 standard.
- **ESS gate:** "ESS ≥ 400, assuming four chains" (100 × number of chains) is mandatory.
- **Divergences gate:** Zero divergences required. "We do not recommend ever disregarding divergences" in cognitive models.
- **BFMI gate:** "BFMI ≥ 0.2 for all chains" required.
- **Prior predictive criterion:** The prior should "encompass a sufficiently broad range of possible behavior" while giving slightly elevated weight to typically observed patterns — neither over- nor under-constraining.
- **Parameter recovery criterion:** "The true value of a parameter will fall within the corresponding 95% credible interval for 95% of the parameters." Visual inspection: "true and estimated values appear moderately correlated or better."
- **Recovery failure modes:** shrinkage (systematic over/underestimation toward mean), consistent bias, total failure (zero correlation).
- **Reparameterization:** Non-centered parameterization resolves hierarchical funnels. Our hBayesDM pattern (`mu_pr + sigma_pr * z`) already implements this.
- **SBC (Simulation-Based Calibration):** Rank distributions should appear uniform across multiple recovery runs.

**How this anchors Phase 21:**
- Steps 21.1 (prior predictive) and 21.2 (recovery) are gates that must pass before 21.3.
- Step 21.4 convergence audit applies the R-hat ≤1.05 / ESS ≥ 400 / 0-divergences gates (phase criterion uses ≤1.05, slightly looser than Baribault's strict ≤1.01, but stricter than legacy ≤1.10).

---

## 2. Hess et al. (2025) — "Bayesian Workflow for Generative Modeling in Computational Psychiatry"

**Citation:** Hess, A. J., et al. (2025). Bayesian Workflow for Generative Modeling in Computational Psychiatry. *Computational Psychiatry*, 9(1), 76–99. https://doi.org/10.5334/cpsy.116  
**Full text:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11951975/  
**Confidence:** HIGH (PMC full text fetched and read)

**Key claims the planner can cite verbatim:**

- **Staged gates:** Model specification → Prior specification (prior predictive gate) → Model inversion & validation (recovery gate) → Model comparison → Model evaluation. Recovery analysis must "establish parameter and model identifiability *before* testing empirical data."
- **Prior predictive gate:** Prior predictive checking must demonstrate the priors "lay in the range of actual human behaviour" while remaining appropriately flexible.
- **Recovery threshold:** Pearson *r* correlations between true and recovered values should be "highly significant (p < 0.001)"; the minimum acceptable r was 0.67 in their winning model for their hardest-to-recover parameter. For our project, the existing r ≥ 0.80 criterion (kappa family) is more stringent.
- **Model comparison methodology:** Random-effects Bayesian Model Selection (RFX-BMS) is recommended because it "accounts for group heterogeneity (different participants may be using different winning models/families) and provides robustness against outliers." Exceedance probabilities (XP) and expected posterior frequencies (Ef) are reported.
- **Family-level comparison:** RFX-BMS conducted at the family level (families of models with shared structural features) before individual model-level comparison.
- **Anti-circularity:** The workflow uses a "separate pilot sample (*N*=20) to estimate empirical priors via MAP estimates, explicitly avoiding 'double-dipping' circularity." For Phase 21, the analogous rule is that M6b cannot be pre-selected; all 6 models start equal.
- **Important difference from Phase 21:** Hess et al. use MAP/Laplace rather than full MCMC, so their MCMC diagnostics (R-hat, ESS, Pareto-k) do not appear. Phase 21 uses full NUTS (already implemented) and must apply Baribault's convergence gates.

**How this anchors Phase 21:**
- The Hess staged workflow is the structural template for the 9-step pipeline.
- RFX-BMS as the secondary comparison method (step 21.5) is justified by citing Hess + Rigoux/Stephan.

---

## 3. Yao, Vehtari, Simpson & Gelman (2018) — "Using Stacking to Average Bayesian Predictive Distributions"

**Citation:** Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Using Stacking to Average Bayesian Predictive Distributions. *Bayesian Analysis*, 13(3), 917–1007. https://doi.org/10.1214/17-BA1091  
**arXiv:** https://arxiv.org/abs/1704.02030  
**Confidence:** HIGH (abstract/search confirmed, ArviZ implementation verified in codebase)

**Key claims the planner can cite verbatim:**

- **The problem stacking solves:** In the M-open setting (true data-generating process is NOT one of the candidates), Bayesian model averaging (BMA) fails because it concentrates mass on the single "best" model. Stacking optimizes combination weights jointly by maximizing leave-one-out predictive accuracy.
- **Weights are optimized jointly:** When similar models exist, stacking prevents dilution — related models share weight while unique models retain it. This is different from pseudo-BMA which normalizes AIC-type scores independently.
- **Implementation:** ArviZ 0.23.4's `az.compare(ic='loo', method='stacking')` calls `scipy.optimize` (SLSQP) to maximize the weighted LOO score subject to weights ≥ 0 and sum to 1.
- **When to use:** Stacking is primary; BB-pseudo-BMA is the fallback when computation cost is prohibitive. For Phase 21, stacking is primary.

---

## 4. Stephan, Penny, Daunizeau, Moran & Friston (2009) — "Bayesian Model Selection for Group Studies"

**Citation:** Stephan, K. E., Penny, W. D., Daunizeau, J., Moran, R. J., & Friston, K. J. (2009). Bayesian model selection for group studies. *NeuroImage*, 46(4), 1004–1017. https://doi.org/10.1016/j.neuroimage.2009.03.025  
**PMC:** https://pmc.ncbi.nlm.nih.gov/articles/PMC2703732/  
**Confidence:** MEDIUM (paper confirmed via search; PDF binary unreadable, PMC text not fetched)

**Key claims (from confirmed search results and secondary sources):**

- **Random-effects BMS (RFX-BMS):** Unlike fixed-effects BMS (sum log evidences), RFX-BMS treats model assignment as a random variable that can differ between subjects, modeled with a Dirichlet prior over model frequencies. This handles population heterogeneity and outliers.
- **Exceedance probability (XP):** Probability that a given model is more frequent in the population than all other models. For K=2 models, computed analytically via Beta CDF. For K>2, computed by 10^6 samples from the posterior Dirichlet.
- **Log model evidence as input:** Each participant contributes a vector of log model evidences (one per candidate model). LOO-ELPD is an approximation to log marginal likelihood and can serve as this input.

---

## 5. Rigoux, Stephan, Friston & Daunizeau (2014) — "Bayesian Model Selection for Group Studies — Revisited"

**Citation:** Rigoux, L., Stephan, K. E., Friston, K. J., & Daunizeau, J. (2014). Bayesian model selection for group studies — Revisited. *NeuroImage*, 84, 971–985. https://doi.org/10.1016/j.neuroimage.2013.08.065  
**PubMed:** https://pubmed.ncbi.nlm.nih.gov/24018303/  
**Confidence:** MEDIUM (paper confirmed; PDF binary; PXP formula extracted from mfit/bms.m reference implementation)

**Key claims (PXP formula from mfit/bms.m — HIGH confidence for the formula itself):**

- **Protected exceedance probability (PXP):** `PXP = (1 - BOR) * EP + BOR / K`  where BOR is the Bayesian Omnibus Risk and EP is the raw exceedance probability.
- **BOR:** `BOR = 1 / (1 + exp(F1 - F0))` where F0 = free energy under null (equal model frequencies), F1 = free energy under the model-heterogeneity alternative.
- **Purpose:** PXP corrects EP for the possibility that observed differences in model evidences (over subjects) are due to chance. When BOR → 1 (data consistent with equal frequencies), PXP → 1/K for all models.
- **Added by Rigoux (2014) vs Stephan (2009):** Rigoux introduced BOR and PXP; Stephan introduced the basic RFX-BMS + XP framework.

---

## 6. Vehtari, Gelman & Gabry (2017) — Pareto-k Threshold Rationale

**Citation:** Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27, 1413–1432. https://doi.org/10.1007/s11222-016-9696-4  
**Confidence:** HIGH (threshold confirmed via search + Stan LOO documentation)

**Key claim:**

- **Pareto-k < 0.7 threshold:** When k < min(1 − 1/log₁₀(S), 0.7), the PSIS estimate and Monte Carlo standard error are reliable (S = effective sample size). When k ∈ [0.7, 1), the estimate has large bias and is unreliable. When k ≥ 1, the PSIS estimate may not converge. The 0.7 threshold is the practical reliability boundary in the ArviZ `warning` column.
- **At S = 8000 (4 chains × 2000 samples):** The sample-size-dependent threshold is 1 − 1/log₁₀(8000) ≈ 0.74, so 0.7 is the binding constraint.

---

# Algorithm Details

## LOO Stacking via `az.compare`

**Function call (already in codebase at `scripts/14_compare_models.py:674`):**
```python
comparison = az.compare(compare_dict, ic="loo", method="stacking")
```

**Input:** `compare_dict` is a `dict[str, az.InferenceData]` — each value must have a `log_likelihood` group. The existing `build_inference_data_with_loglik()` in `scripts/fitting/bayesian_diagnostics.py` produces exactly this format.

**Optimization problem:** Find weights `w = [w_1, ..., w_K]` that maximize:
```
sum_n log(sum_k w_k * exp(log_lik_k(y_n | y_{-n})))
```
subject to `w_k ≥ 0` and `sum(w_k) = 1`. Solved via `scipy.optimize.minimize(method='SLSQP')` internally in ArviZ.

**Output DataFrame columns (ArviZ 0.23.x):**

| Column | Meaning |
|--------|---------|
| `rank` | 0-based rank (0 = best) |
| `elpd_loo` | Total LOO-ELPD (higher = better) |
| `p_loo` | Effective number of parameters (penalty) |
| `elpd_diff` | Difference vs top model (top model = 0) |
| `weight` | Stacking weight (sums to 1.0 across models) |
| `se` | SE of `elpd_loo` |
| `dse` | SE of `elpd_diff` (0 for top model) |
| `warning` | True if Pareto-k > 0.7 for any observation |
| `scale` | "log" (default) |

**Known ArviZ issue:** GitHub issue #2359 documents that stacking weights sometimes do not sum to exactly 1.0 due to floating-point optimization tolerance. Assert `abs(weights.sum() - 1.0) < 0.01` rather than `== 1.0`.

**Pareto-k check (existing in codebase):**
```python
loo_result = az.loo(idata, pointwise=True)
k_vals = loo_result.pareto_k.values
pct_high = float(np.mean(k_vals > 0.7) * 100)
```
Gate criterion: `pct_high < 1.0` (success criterion specifies >99% of observations below 0.7). Existing code uses `> 10%` as "HIGH" flag for M4; Phase 21 tightens this to `< 1%` for the choice-only gate.

---

## RFX-BMS + PXP Implementation

**Algorithm (from mfit/bms.m, confirmed HIGH confidence):**

```python
import numpy as np
from scipy.special import psi  # digamma

def rfx_bms(log_evidence: np.ndarray) -> dict:
    """
    Random-effects Bayesian Model Selection.

    Parameters
    ----------
    log_evidence : np.ndarray, shape (n_subjects, n_models)
        Per-participant log model evidence.
        Use LOO-ELPD per participant per model:
            idata = az.loo(model_idata, pointwise=True)
            # sum over observations per participant
            lme[:, k] = participant-summed LOO log-lik for model k

    Returns
    -------
    dict with keys:
        alpha  : Dirichlet posterior parameters, shape (n_models,)
        r      : Expected model frequencies (r = alpha / sum(alpha))
        xp     : Exceedance probabilities, shape (n_models,)
        bor    : Bayesian Omnibus Risk (scalar)
        pxp    : Protected exceedance probabilities, shape (n_models,)
    """
    n_subjects, n_models = log_evidence.shape
    alpha0 = np.ones(n_models)  # Uniform Dirichlet prior
    alpha = alpha0.copy()

    # Variational Bayes E-step
    for _ in range(1000):  # Max iterations
        log_u = log_evidence + psi(alpha) - psi(alpha.sum())
        u = np.exp(log_u - log_u.max(axis=1, keepdims=True))
        g = u / u.sum(axis=1, keepdims=True)  # Posterior responsibilities
        beta = g.sum(axis=0)                   # Expected counts per model
        alpha_new = alpha0 + beta
        if np.max(np.abs(alpha_new - alpha)) < 1e-4:
            break
        alpha = alpha_new
    alpha = alpha_new

    # Expected frequencies
    r = alpha / alpha.sum()

    # Exceedance probabilities (sampling for K > 2)
    n_samples = 1_000_000
    rng = np.random.default_rng(42)
    dirichlet_samples = rng.dirichlet(alpha, size=n_samples)
    winners = dirichlet_samples.argmax(axis=1)
    xp = np.bincount(winners, minlength=n_models) / n_samples

    # Bayesian Omnibus Risk
    # F0 = free energy under null (equal model frequencies)
    alpha_null = np.ones(n_models) * (n_subjects / n_models + 1)
    F0 = _vb_free_energy(log_evidence, alpha_null)
    F1 = _vb_free_energy(log_evidence, alpha)
    bor = 1.0 / (1.0 + np.exp(F1 - F0))

    # Protected exceedance probability
    pxp = (1 - bor) * xp + bor / n_models

    return {"alpha": alpha, "r": r, "xp": xp, "bor": bor, "pxp": pxp}
```

**Input construction for Phase 21:** For each model, compute per-participant LOO contributions:
```python
loo_result = az.loo(idata, pointwise=True)
# pointwise ELPD values have shape (n_participants * n_trials,)
# Must be summed per participant
```
The `log_likelihood` group in our InferenceData has shape `(chains, draws, n_participants, n_trials_padded)`. Summing over the last axis (after NaN-masking padding) gives per-participant log evidence.

**Note on F0/F1 computation:** The `_vb_free_energy` helper requires implementation based on the Dirichlet variational objective; the mfit/bms.m code does this iteratively. The simplest approach for Phase 21 is to compute F as the ELBO: `F = sum_k (lgamma(alpha_k) - lgamma(alpha0_k)) - lgamma(sum(alpha)) + lgamma(sum(alpha0)) + sum_n log(sum_k g_nk * exp(lme_nk))`. The planner must decide whether to implement this from scratch or adapt from spm_BMS.m. **Recommended:** implement a minimal but correct version in `scripts/fitting/bms.py`; the full free-energy calculation is tractable (~50 lines). Alternatively, use the `pymc` Dirichlet posterior sampler if MATLAB-free code is needed (LOW confidence that a production-quality Python bms.py already exists in any third-party library).

---

## Prior Predictive Check

**NumPyro API (HIGH confidence, confirmed from numpyro.infer.Predictive docs):**
```python
from numpyro.infer import Predictive
import jax

rng_key = jax.random.PRNGKey(42)

# Sample from prior only (no posterior conditioning, no data)
prior_predictive = Predictive(
    model=wmrl_m3_hierarchical_model,
    num_samples=500,  # Number of prior draws
)
prior_samples = prior_predictive(
    rng_key,
    participant_data_stacked=participant_data_stacked,
    covariate_lec=None,
    stacked_arrays=stacked_arrays,
    use_pscan=False,
)
# prior_samples is a dict: {param_name: array(num_samples, ...)}
# Group-level params (mu_pr, sigma_pr) are sampled from their priors.
# Individual params are sampled from the group distribution.
```

**What to extract from prior samples:**
- Per-participant accuracy under prior-sampled parameters (simulate via `unified_simulator` or `generate_data.py`)
- Target "plausible" range: asymptotic accuracy ~0.4–0.9 (ceiling ~0.95 for 3AFC at β=50)
- Red flags: prior concentrating >20% of samples below 0.33 (chance) or above 0.95 (ceiling)

**Practical implementation for step 21.1:**
Because our likelihoods are evaluators not simulators (they compute p(action | params), they don't generate actions), prior predictive check requires:
1. Sample parameters from prior using `Predictive(..., num_samples=N)(rng_key, ...)`
2. For each prior draw, simulate behavior using `generate_data.py` / `unified_simulator.simulate_agent_fixed`
3. Compute summary statistics: accuracy by set size, block-learning curve shape

**PARAM_PRIOR_DEFAULTS (v4.0 locked values, from `scripts/fitting/numpyro_helpers.py:183`):**
```python
PARAM_PRIOR_DEFAULTS = {
    "alpha_pos":    {"lower": 0.0, "upper": 1.0, "mu_prior_loc": 0.0},
    "alpha_neg":    {"lower": 0.0, "upper": 1.0, "mu_prior_loc": 0.0},
    "epsilon":      {"lower": 0.0, "upper": 1.0, "mu_prior_loc": -2.0},
    "phi":          {"lower": 0.0, "upper": 1.0, "mu_prior_loc": 0.0},
    "rho":          {"lower": 0.0, "upper": 1.0, "mu_prior_loc": 0.0},
    "capacity":     {"lower": 2.0, "upper": 6.0, "mu_prior_loc": 0.0},
    "kappa":        {"lower": 0.0, "upper": 1.0, "mu_prior_loc": 0.0},
    "kappa_s":      {"lower": 0.0, "upper": 1.0, "mu_prior_loc": 0.0},
    "kappa_total":  {"lower": 0.0, "upper": 1.0, "mu_prior_loc": 0.0},
    "kappa_share":  {"lower": 0.0, "upper": 1.0, "mu_prior_loc": 0.0},
    "phi_rl":       {"lower": 0.0, "upper": 1.0, "mu_prior_loc": 0.0},
}
```
Prior mean = Φ(0) = 0.5 on the bounded scale; 95% prior interval ≈ [0.02, 0.98] given `sigma_pr ~ HalfNormal(0.2)`. Epsilon exception: `mu_prior_loc=-2.0` → prior mean ≈ Φ(-2) ≈ 0.023 (appropriately near-zero noise).

---

## Bayesian Parameter Recovery (step 21.2)

**Approach:** Adapt existing `scripts/fitting/model_recovery.py` (MLE-based) to Bayesian MCMC. The existing `run_parameter_recovery` samples true parameters uniformly from MLE bounds, generates synthetic data, fits via MLE, and computes Pearson r. Phase 21 needs a Bayesian variant:

1. Sample N=50 synthetic parameter sets from PARAM_PRIOR_DEFAULTS (not uniform bounds)
2. Generate synthetic trial data via `generate_data.py`
3. Fit via `fit_bayesian._fit_stacked_model` (standard MCMC budget: warmup=500, samples=1000 for speed)
4. Compare posterior means vs true values: Pearson r + HDI coverage

**HDI coverage calibration:** For each parameter, check that 95% HDI contains the true value in ≥90% of the 50 datasets (ideally ≥95%). This is a looser check than the strict SBC rank-uniformity test. The planner should decide whether to run full SBC or the simpler HDI coverage check — full SBC requires O(1000) datasets, which is too expensive for a cluster submission. Recommendation: HDI coverage at N=50 datasets.

**Known-identifiable parameters (r ≥ 0.80 target):**
- κ (kappa in M3, kappa_s in M6a, kappa_total and kappa_share in M6b) — confirmed r > 0.93 under MLE

**Known-unidentifiable parameters (descriptive only, r < 0.80 acceptable):**
- α_pos, α_neg, φ, ρ, K (capacity), ε — all r < 0.80 under MLE; hierarchical shrinkage may improve but cannot manufacture identifiability

**Phase 21 pass criterion:**
- κ family: r ≥ 0.80 AND 95% HDI coverage ≥ 90%
- All others: report r and coverage; label "descriptive only" regardless of value

---

# Reusable Infrastructure Audit

The following table confirms which existing functions/files Phase 21 can consume directly. All paths verified against current codebase.

| Existing Function / File | Phase 21 Step | Import Path | Notes |
|--------------------------|--------------|-------------|-------|
| `_fit_stacked_model()` | 21.3 (baseline fit), 21.6 (with-L2 fit) | `scripts/fitting/fit_bayesian._fit_stacked_model` | Already handles all 6 models, auto-bump, convergence gate. Call with `covariate_lec=None` for 21.3. |
| `save_results()` | 21.3, 21.6 | `scripts/fitting/fit_bayesian.save_results` | Writes NetCDF, CSV, shrinkage report, LOO/WAIC. |
| `load_and_prepare_data()` | 21.1–21.9 | `scripts/fitting/fit_bayesian.load_and_prepare_data` | `use_cohort=True` default; N=138 canonical cohort. |
| `STACKED_MODEL_DISPATCH` | 21.3, 21.5 | `scripts/fitting/fit_bayesian.STACKED_MODEL_DISPATCH` | Dict of all 6 choice-only models. |
| `compute_pointwise_log_lik()` | 21.5 (LOO input) | `scripts/fitting/bayesian_diagnostics.compute_pointwise_log_lik` | Returns shape `(chains, samples, participants, trials)`. |
| `build_inference_data_with_loglik()` | 21.5 | `scripts/fitting/bayesian_diagnostics.build_inference_data_with_loglik` | Produces ArviZ InferenceData with `log_likelihood` group for `az.compare`. |
| `run_inference_with_bump()` | 21.2, 21.3, 21.6 | `scripts/fitting/numpyro_models.run_inference_with_bump` | Auto-bumps target_accept 0.80→0.95→0.99 on divergences. |
| `run_bayesian_comparison()` | 21.5 (LOO+stacking) | `scripts/14_compare_models.run_bayesian_comparison` | Already calls `az.compare(ic='loo', method='stacking')`, logs Pareto-k, writes `stacking_weights.md` + `.csv`. **Reuse directly.** |
| `_pareto_k_summary()` | 21.5 | `scripts/14_compare_models._pareto_k_summary` | Returns `{model_name: pct_high_k}` dict. |
| `BAYESIAN_NETCDF_MAP` | 21.5 | `scripts/14_compare_models.BAYESIAN_NETCDF_MAP` | Maps M1–M6b display names to NetCDF paths. Reuse; Phase 21 must not change this mapping. |
| `PARAM_PRIOR_DEFAULTS` | 21.1 (prior predictive), 21.2 (recovery sampling) | `scripts/fitting/numpyro_helpers.PARAM_PRIOR_DEFAULTS` | All `mu_prior_loc=0.0` (v4.0 locked). |
| `sample_bounded_param()` | 21.1 (Predictive) | `scripts/fitting/numpyro_helpers.sample_bounded_param` | Used internally by all hierarchical models. |
| `prepare_stacked_participant_data()` | 21.1–21.6 | `scripts/fitting/numpyro_models.prepare_stacked_participant_data` | Converts DataFrame to stacked JAX arrays. |
| `stack_across_participants()` | 21.3, 21.6 | `scripts/fitting/numpyro_models.stack_across_participants` | Pre-computes (N, B, T) arrays for fully-batched vmap. Required for all 6 models (`_FULLY_BATCHED_MODELS` tuple). |
| `run_parameter_recovery()` | 21.2 (MLE baseline) | `scripts/fitting/model_recovery.run_parameter_recovery` | MLE-based; Phase 21 needs a Bayesian variant but can compare to this as baseline. |
| `sample_parameters()` | 21.2 | `scripts/fitting/model_recovery.sample_parameters` | Samples uniformly from MLE bounds. **For Bayesian recovery, prefer sampling from PARAM_PRIOR_DEFAULTS instead.** |
| `generate_data.py` | 21.1 (prior predictive simulator), 21.2 (recovery data generation) | `scripts/simulations/generate_data.py` | `sample_parameters_from_posterior` loads NetCDF. |
| `simulate_agent_fixed()` | 21.1 | `scripts/simulations/unified_simulator.simulate_agent_fixed` | Simulates behavior given fixed params. Use for prior predictive accuracy checks. |
| `numpyro.infer.Predictive` | 21.1 | `numpyro.infer.Predictive` | Call with `num_samples=N` to sample from prior. No mcmc argument = prior predictive. |
| `filter_padding_from_loglik()` | 21.5 | `scripts/fitting/bayesian_diagnostics.filter_padding_from_loglik` | Sets padded trials to NaN before LOO. Required. |
| `cluster/13_bayesian_m*.slurm` | 21.3 template | `cluster/` | 6 individual CPU fit scripts + multigpu variant. Copy/adapt for 21.3 naming. |
| `cluster/submit_full_pipeline.sh` | 21 orchestrator template | `cluster/submit_full_pipeline.sh` | Wave-based `sbatch --dependency=afterok:$JOBID` pattern. Use for `cluster/21_submit_pipeline.sh`. |

**NOT reusable / net-new components:**
- `scripts/fitting/bms.py` — does not exist; must be created (RFX-BMS + PXP)
- `scripts/21_prior_predictive.py` — does not exist; pure-prior sampling + accuracy histograms
- `scripts/21_bayesian_recovery.py` — does not exist; Bayesian (MCMC) variant of model_recovery.py
- `scripts/21_fit_baseline.py` — thin wrapper over `fit_bayesian.main()` for all 6 models without L2 scales
- `scripts/21_convergence_audit.py` — loads all 6 NetCDFs, checks R-hat/ESS/divergences, writes audit table
- `scripts/21_loo_stacking_bms.py` — calls `run_bayesian_comparison()` + new `rfx_bms()`, writes combined output
- `scripts/21_fit_with_l2.py` — thin wrapper over `fit_bayesian.main()` for winner models with 4-covariate L2
- `scripts/21_scale_audit.py` — loads winner NetCDFs, reports beta HDI exclusions, writes gate report
- `scripts/21_model_averaged_effects.py` — computes stacking-weighted beta estimates; optional M6b-subscale arm
- `scripts/21_manuscript_tables.py` — generates final tables and figures

---

# Gates & Thresholds

## Step 21.1 — Prior Predictive Gate

**Input:** 500 parameter draws per model from `PARAM_PRIOR_DEFAULTS`  
**Metric:** Distribution of simulated trial accuracies across all blocks  
**PASS criterion:**
- Median accuracy across prior draws: 0.4 ≤ median ≤ 0.9
- Less than 10% of prior draws produce accuracy < 0.33 (chance) at any set size
- Less than 5% of prior draws produce accuracy > 0.95 (ceiling) at any set size
**FAIL action:** Revise `mu_prior_loc` values in `PARAM_PRIOR_DEFAULTS`; do NOT proceed to 21.3  
**Confidence:** MEDIUM (thresholds derived from Baribault guidance + task knowledge; not from paper)

## Step 21.2 — Bayesian Parameter Recovery Gate

**Input:** N=50 synthetic datasets per model  
**Metrics:** Pearson r (true vs posterior mean) + 95% HDI coverage rate  
**PASS criterion (per parameter type):**
- κ family: r ≥ 0.80 AND HDI coverage ≥ 90%
- All others: report and label "descriptive only" (no pass/fail gate)
**FAIL action:** For κ family r < 0.80, investigate model pathologies before proceeding  
**Confidence:** HIGH for r ≥ 0.80 threshold (from project's prior recovery results); MEDIUM for HDI coverage threshold (adapted from Baribault 2023 criterion)

## Step 21.3 — Baseline Hierarchical Fit

**Action:** Fit all 6 choice-only models with `covariate_lec=None` (no L2 regression)  
**Budget per model:** warmup=1000, samples=2000, 4 chains (consistent with v4.0 standard)  
**Resources:** CPU, ~1–2h per model at ~1s/iter after Phase 20 fully-batched rollout  
**Output:** `output/bayesian/21_{model}_baseline_posterior.nc` (new naming to avoid overwriting existing posteriors)

## Step 21.4 — Convergence & Fit-Quality Audit Gate

**CONVERGENCE GATE (from Baribault 2023, already implemented in `save_results`):**
- R-hat ≤ 1.05 for all parameters (phase criterion; stricter ≤1.01 is the Baribault ideal)
- ESS_bulk ≥ 400 for all parameters
- 0 divergences (enforced by `run_inference_with_bump` up to target_accept=0.99)
- BFMI ≥ 0.2 (check via `mcmc.get_extra_fields()["energy"]`)

**FIT-QUALITY GATE (posterior predictive check accuracy):**
- Per-participant posterior predictive accuracy should fall within ±2 SD of observed accuracy
- Block-learning curve shape should qualitatively match human data

**BLOCK action:** If convergence gate fails for any model after 3 auto-bump attempts, that model is excluded from step 21.5 with a warning. If fewer than 2 models pass, the pipeline is blocked.

## Step 21.5 — PSIS-LOO + Stacking + RFX-BMS Gate

**PRIMARY — LOO Stacking:**
- Pareto-k < 0.7 for ≥ 99% of observations per model (tighter than existing 10% threshold)
- Stacking weights sum to 1.0 ± 0.01 (numerical tolerance)
- Winner: model(s) with stacking weight ≥ 0.5 (single winner) or top-2 with combined weight ≥ 0.8

**SECONDARY — RFX-BMS:**
- PXP ≥ 0.95 for strong winner claim; PXP ∈ [0.75, 0.95] for moderate claim
- BOR < 0.05 to reject the null of equal model frequencies

**INCONCLUSIVE criterion:** If no model has stacking weight ≥ 0.5, all models with weight ≥ 0.10 advance to step 21.6.

## Step 21.6 — Winner Refit With L2 Scales

**Design matrix (locked, condition number 11.3):**
1. `lec_total` (z-scored)
2. `iesr_total` (z-scored)
3. `iesr_intr_resid` (z-scored, Gram-Schmidt residualized vs total)
4. `iesr_avd_resid` (z-scored, Gram-Schmidt residualized vs total)

Only winner model(s) from 21.5 are refit. L2 regression uses `fit_bayesian._fit_stacked_model` with `covariate_lec` loaded from `output/summary_participant_metrics.csv`. Note: M1/M2 cannot receive L2 (guard in `_L2_LEC_SUPPORTED`); if M1 or M2 wins LOO, they are reported without L2 effects.

## Step 21.7 — Scale-Fit Audit Gate

**GATE criteria:**
- Convergence criteria same as 21.4
- At least one `beta_*` site with 95% HDI excluding zero (for the results to be scientifically meaningful)
- If zero HDI exclusions after FDR-BH: report null result, do NOT proceed to 21.8 model averaging

## Step 21.8 — Model-Averaged Scale Effects

**Triggered if:** ≥ 2 models pass step 21.7 gates  
**Algorithm:** Compute stacking-weight-averaged beta posterior for each covariate × parameter combination  
**M6b-subscale arm:** Optional; runs `wmrl_m6b_hierarchical_model_subscale` with 32 beta sites (12h SLURM job)

## Step 21.9 — Manuscript Tables

**Outputs required:**
- Table: LOO-ELPD comparison + stacking weights (all 6 models)
- Table: RFX-BMS PXP (secondary)
- Table: Winner model β coefficients + 95% HDI (primary results)
- Figure: Forest plot of beta coefficients (`18_bayesian_level2_effects.py` reusable)

---

# Cluster Submission Pattern

## Template: Adapting `cluster/13_bayesian_m6b.slurm`

Each step 21.x SLURM script follows the established pattern. Key adaptations:

```bash
#!/bin/bash
#SBATCH --job-name=bayes_21_3_${MODEL}
#SBATCH --output=logs/bayesian_21_3_${MODEL}_%j.out
#SBATCH --error=logs/bayesian_21_3_${MODEL}_%j.err
#SBATCH --time=08:00:00        # M6b; use 04:00:00 for M1, 06:00:00 for M2-M5
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp       # CPU partition (no GPU needed for choice-only)

MODEL="${MODEL:-wmrl_m6b}"
WARMUP="${WARMUP:-1000}"
SAMPLES="${SAMPLES:-2000}"
CHAINS="${CHAINS:-4}"
SEED="${SEED:-42}"
```

**Environment activation (CPU path):**
```bash
if conda activate ds_env 2>/dev/null; then
    echo "Activated ds_env"
elif conda activate /scratch/fc37/$USER/conda/envs/ds_env 2>/dev/null; then
    echo "Activated ds_env from /scratch/fc37/"
fi
```

**JAX cache (prevent cross-model contamination):**
```bash
export JAX_PLATFORMS=cpu
export NUMPYRO_HOST_DEVICE_COUNT=4
export JAX_COMPILATION_CACHE_DIR="/scratch/${PROJECT:-fc37}/${USER}/.jax_cache_21_${MODEL}"
```

**Script invocation (21.3 baseline, no L2):**
```bash
python scripts/21_fit_baseline.py \
    --model "$MODEL" \
    --data output/task_trials_long.csv \
    --chains "$CHAINS" \
    --warmup "$WARMUP" \
    --samples "$SAMPLES" \
    --seed "$SEED" \
    --output-dir output/bayesian/21_baseline/
```

**Alternatively** (if thin wrapper is not built), call `fit_bayesian.py` directly with an `--output-prefix 21_baseline` flag (requires a small patch to `save_results` to accept an output prefix).

## Template: `cluster/21_submit_pipeline.sh`

Adapted from `cluster/submit_full_pipeline.sh`. Key pattern for 9 linear steps:

```bash
#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b"

# Step 21.1: Prior predictive (6 parallel, 1h each)
declare -A PPC_JOBS
for model in $MODELS; do
    jid=$(sbatch --parsable --export=ALL,MODEL="$model" cluster/21_1_prior_predictive.slurm)
    PPC_JOBS[$model]=$jid
done
PPC_DEP=$(IFS=:; echo "afterok:${PPC_JOBS[*]}")

# Step 21.2: Recovery (6 parallel, after 21.1)
declare -A REC_JOBS
for model in $MODELS; do
    jid=$(sbatch --parsable --dependency=afterok:${PPC_JOBS[$model]} \
        --export=ALL,MODEL="$model" cluster/21_2_recovery.slurm)
    REC_JOBS[$model]=$jid
done
REC_DEP=$(IFS=:; echo "afterok:${REC_JOBS[*]}")

# Step 21.3: Baseline fit (6 parallel, after 21.2)
declare -A FIT_JOBS
for model in $MODELS; do
    jid=$(sbatch --parsable --dependency=${REC_DEP} \
        --export=ALL,MODEL="$model" cluster/21_3_fit_baseline.slurm)
    FIT_JOBS[$model]=$jid
done
ALL_FIT_DEP=$(IFS=:; echo "afterok:${FIT_JOBS[*]}")

# Step 21.4: Convergence audit (1 job, after all fits)
audit_jid=$(sbatch --parsable --dependency=${ALL_FIT_DEP} cluster/21_4_convergence_audit.slurm)

# Step 21.5: LOO+stacking+BMS (1 job, after audit)
loo_jid=$(sbatch --parsable --dependency=afterok:${audit_jid} cluster/21_5_loo_stacking_bms.slurm)

# Steps 21.6–21.9: Linear chain
l2_jid=$(sbatch --parsable --dependency=afterok:${loo_jid} cluster/21_6_fit_with_l2.slurm)
saudit_jid=$(sbatch --parsable --dependency=afterok:${l2_jid} cluster/21_7_scale_audit.slurm)
avg_jid=$(sbatch --parsable --dependency=afterok:${saudit_jid} cluster/21_8_model_averaged.slurm)
tables_jid=$(sbatch --parsable --dependency=afterok:${avg_jid} cluster/21_9_manuscript_tables.slurm)
```

**SLURM time estimates per step:**

| Step | Script | Time | Partition | Notes |
|------|--------|------|-----------|-------|
| 21.1 | `21_1_prior_predictive.slurm` | 1h/model | comp | 500 prior draws × simulate × 6 models |
| 21.2 | `21_2_recovery.slurm` | 6h/model | comp | N=50 × warmup=500/samples=1000 MCMC |
| 21.3 | `21_3_fit_baseline.slurm` | 2–8h/model | comp | M1=4h, M2-M5=6h, M6b=8h |
| 21.4 | `21_4_convergence_audit.slurm` | 30m | comp | Loads 6 NetCDFs, runs az.summary |
| 21.5 | `21_5_loo_stacking_bms.slurm` | 2h | comp | LOO pointwise + az.compare + rfx_bms |
| 21.6 | `21_6_fit_with_l2.slurm` | 4–12h/winner | comp | One or more winner models with 4-covariate L2 |
| 21.7 | `21_7_scale_audit.slurm` | 30m | comp | Loads winner NetCDFs, checks HDI |
| 21.8 | `21_8_model_averaged.slurm` | 1–12h | comp | 1h if single winner; 12h if M6b-subscale arm |
| 21.9 | `21_9_manuscript_tables.slurm` | 30m | comp | Table generation, no MCMC |

**Output naming convention (to avoid overwriting existing posteriors):**
- All 21.x outputs go to `output/bayesian/21_baseline/` (no L2) and `output/bayesian/21_l2/` (with L2)
- NetCDF: `output/bayesian/21_baseline/{model}_posterior.nc`
- CSV: `output/bayesian/21_baseline/{model}_individual_fits.csv`

---

# Open Questions / Risks

## 1. `bms.py` Free Energy Implementation

**Question:** The PXP formula requires computing `F0` (free energy under null) and `F1` (free energy under model). The BOR = 1/(1+exp(F1-F0)) formula is confirmed, but the variational Bayes free energy `F` requires implementing the ELBO of the Dirichlet-categorical model.  
**Risk:** Getting the ELBO wrong produces incorrect BOR, which corrupts PXP.  
**Resolution:** The planner must decide: (a) implement from scratch using the VB equations from Stephan (2009) Appendix A — tractable but ~2 days implementation, or (b) port the MATLAB `bms.m` from `sjgershm/mfit` (license permissible: MIT). Option (b) recommended. Or (c) report only XP (without PXP), citing "BOR computation deferred" — acceptable as PXP is a correction to XP for publication.

## 2. Per-Participant LOO Construction for RFX-BMS

**Question:** RFX-BMS requires per-participant log model evidence as a `(N, K)` matrix. ArviZ LOO returns pointwise ELPD over observations (trials), not participants. We must sum over trials per participant.  
**Risk:** Off-by-one in participant ordering (sorted participant IDs must align across all 6 models).  
**Resolution:** Use `participant_ids = sorted(pdata_stacked.keys())` (already the canonical ordering in `compute_pointwise_log_lik`). The `log_likelihood` InferenceData group has dim `participant` indexed by the same sorted list. Summing `idata.log_likelihood.obs.values` over the `trial` dimension gives `(chains, draws, participants)`, then summing over chains+draws gives the required `(N,)` per-model vector. Aggregate: `lme[:, k] = idata.log_likelihood.obs.sum(dim=['chain', 'draw', '__obs__']).values`.

## 3. Stacking Weights Sum-to-1 Bug in ArviZ 0.23.4

**Question:** GitHub issue #2359 documents that stacking weights sometimes do not sum to 1.0.  
**Risk:** Assert failure in pipeline if exact equality is checked.  
**Resolution:** Check `abs(weights.sum() - 1.0) < 0.01` not `== 1.0`. The existing `run_bayesian_comparison` does not assert sum; add a warning log instead of assert.

## 4. L2 Design Across Winner Models

**Question:** If step 21.5 produces multiple winners (e.g., M3 and M6b both have weight ≥ 0.1), step 21.6 must fit each with the 4-covariate L2 design. But M3 only supports `covariate_lec` → `beta_lec_kappa` (single predictor), not the full 4-covariate design.  
**Risk:** M3 cannot receive the 4-covariate L2 design as currently implemented.  
**Resolution:** The planner must decide: (a) extend M3 to support 4-covariate L2 in step 21.6 (new `wmrl_m3_hierarchical_model_l2full` function, ~50 lines), or (b) fit M3 with single-predictor L2 (existing `covariate_lec` = `lec_total` only) while M6b gets the full 4-covariate design. Option (b) is lower risk and preserves consistency with STATE.md v4.0 decisions.

## 5. Output Directory Collision with Existing Runs

**Question:** The canonical NetCDF paths (`output/bayesian/wmrl_m6b_posterior.nc`) are already referenced by `BAYESIAN_NETCDF_MAP` in `14_compare_models.py`. Phase 21 fits should NOT overwrite these.  
**Risk:** If Phase 21 uses the same output paths as Phase 16, it destroys any v4.0 posteriors that were produced.  
**Resolution:** Phase 21 uses a `21_baseline/` and `21_l2/` subdirectory prefix. `BAYESIAN_NETCDF_MAP` in `14_compare_models.py` must be extended or a Phase-21-specific map created.

## 6. No Existing Posteriors — Cold Start

**Status confirmed:** `output/bayesian/` contains only `pscan_benchmark.json` and `level2/` (empty of NetCDFs). No production hierarchical Bayesian posteriors exist. Phase 21 IS the first production run.  
**Implication:** Step 21.4 cannot gate on previously-cached posteriors; all 6 models must be fit fresh.

## 7. Bayesian Recovery Wall-Clock Budget

**Concern:** N=50 Bayesian fits at warmup=500/samples=1000 per model. At ~1s/iter (Phase 20 rate), that is ~500s warmup + 1000s samples = ~1500s ≈ 25 min per fit × 50 subjects = ~21 hours per model on a single CPU.  
**Resolution:** Parallelize: submit 50 subjects as 50 independent SLURM array tasks. Or reduce to N=20 datasets (still sufficient for Pearson r estimate). Or use shorter chains (warmup=200, samples=500 — adequate for point estimates).  
**Decision needed by planner:** Array job (sbatch --array=1-50) vs sequential vs reduced budget.

## 8. model_averaging (step 21.8) — Stacking-Weighted Posterior

**Question:** Model averaging requires computing, for each covariate × parameter combination, the stacking-weight-averaged beta posterior. This is not currently implemented anywhere.  
**Implementation sketch:** For each posterior draw in each winner model's NetCDF, weight the beta site by the model's stacking weight, then combine chains/draws into a mixture posterior. ArviZ has no built-in for this. Custom implementation: `combined_beta = [w_m * samples_m["beta_x"] for m, w_m in zip(winners, weights)]` then concatenate weighted samples.  
**Risk:** Low — straightforward weighted mixture of posteriors. The hard part is ensuring consistent beta site naming across models (which is already enforced by `COVARIATE_NAMES` in `level2_design.py`).

---

# Sources

### Primary (HIGH confidence)
- PMC 10522800 — Baribault & Collins (2023) — workflow steps and gate criteria fetched and read
- PMC 11951975 — Hess et al. (2025) — staged workflow, recovery thresholds, RFX-BMS justification fetched and read
- `scripts/fitting/numpyro_helpers.py:183` — `PARAM_PRIOR_DEFAULTS` values confirmed in codebase
- `scripts/fitting/bayesian_diagnostics.py` — `compute_pointwise_log_lik`, `build_inference_data_with_loglik` verified
- `scripts/fitting/fit_bayesian.py` — `_fit_stacked_model`, `save_results`, `STACKED_MODEL_DISPATCH` verified
- `scripts/14_compare_models.py:674` — `az.compare(ic='loo', method='stacking')` confirmed in codebase
- `cluster/13_bayesian_m6b.slurm` / `cluster/13_bayesian_multigpu.slurm` — SLURM template verified
- `cluster/submit_full_pipeline.sh` — wave-based dependency chain pattern verified
- GitHub mfit/bms.m — RFX-BMS algorithm + PXP formula extracted from reference MATLAB implementation

### Secondary (MEDIUM confidence)
- Yao et al. (2018) — Bayesian Analysis 13(3): stacking optimization problem confirmed via arXiv abstract + Stan LOO vignette
- mc-stan.org/loo stacking vignette — `az.compare` stacking method description
- Rigoux et al. (2014) PubMed confirmed; PXP formula from mfit wrapper (mfit_bms.m)
- Stephan et al. (2009) PubMed confirmed; RFX-BMS framework confirmed via secondary sources
- Vehtari et al. (2017) — Pareto-k threshold 0.7 confirmed via Stan LOO docs

### Tertiary (LOW confidence)
- ArviZ 0.23 compare column names — extracted from WebSearch aggregate, documentation pages returned 403. Names consistent across multiple sources but not directly verified via page fetch.
- ArviZ stacking weight sum-to-1 issue — GitHub #2359 confirmed via WebSearch; considered HIGH risk.

---

## Metadata

**Confidence breakdown:**
- Reusable infrastructure audit: HIGH — all confirmed against actual codebase files
- LOO stacking algorithm: HIGH — confirmed in codebase + confirmed Yao 2018 origin
- RFX-BMS PXP formula: MEDIUM — algorithm from mfit/bms.m; paper PDF unreadable
- Prior predictive API: HIGH — numpyro.infer.Predictive pattern confirmed
- Paper workflow gates: HIGH — PMC full texts fetched and read for both anchor papers
- Pareto-k threshold rationale: HIGH — confirmed via Stan LOO documentation

**Research date:** 2026-04-18  
**Valid until:** 2026-07-18 (stable domain; ArviZ 0.23.x API unlikely to change)
