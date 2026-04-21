# Feature Research — v4.0 Hierarchical Bayesian RLWM Pipeline

**Domain:** Computational psychiatry / hierarchical Bayesian fitting of RL+WM models with clinical covariates
**Researched:** 2026-04-10
**Confidence:** HIGH for partial-pooling structure, prior choices, convergence diagnostics, WAIC/LOO model comparison (mature literature, hBayesDM patterns are de facto standard); MEDIUM for level-2 trauma covariate injection (multiple competing patterns, no single canonical choice in psychiatric RL); LOW–MEDIUM for hierarchical LBA in clinical populations (exotic — almost no published precedent at N=154 with covariates)

---

## Context: What v4.0 Builds On

The v3.0 pipeline already ships:

- **MLE infrastructure for all 7 models** (M1, M2, M3, M5, M6a, M6b choice-only; M4 LBA joint choice+RT) with Latin Hypercube multistart, Hessian SEs, and r >= 0.80 parameter recovery.
- **Legacy hierarchical Bayesian implementations for M1 and M2 only** (`scripts/fitting/legacy/numpyro_models.py`) using the canonical hBayesDM pattern: weakly informative Beta priors on group means, HalfNormal sigmas, non-centered parameterization via standard normal z-scores plus logit-expit transforms.
- **Post-hoc OLS + Bayesian regression** of MLE point estimates on trauma scales (Scripts 16, 16b).
- **Stratified trauma-group analysis** (Script 15) and per-participant winning-model heterogeneity diagnostics.

v4.0 must extend the existing hBayesDM-style pattern to M3/M5/M6a/M6b/M4, replace the post-hoc trauma regression with **joint** level-2 regression (mu_param = beta_0 + beta_trauma * trauma_subscale) inside the hierarchical fit, switch the primary model-comparison criterion from AIC/BIC to **WAIC + PSIS-LOO**, and decide whether hierarchical LBA for M4 is feasible or should remain MLE-only.

---

## Feature Landscape

### Table Stakes (Must Have for Hierarchical Bayesian RLWM)

These are non-negotiable because hBayesDM, Pedersen & Frank (2020), Sullivan-Toole et al. (2022), and the Wisco et al. (2025) anxiety-RLWM paper all converge on the same minimum viable pattern. Missing any of these would break either convergence, identifiability, or comparability with the field.

| # | Feature | Why Expected | Complexity | Notes / Dependencies |
|---|---------|--------------|------------|----------------------|
| TS-1 | **Two-level hierarchy: group (mu, sigma) -> individual** for every model M1-M6b | Universal pattern in hBayesDM, Ahn et al. 2017, Bruno Nicenboim tutorials, Pedersen & Frank 2020. Single-level (no pooling) reproduces MLE; full pooling discards individual differences which is the whole point of trauma analysis. | LOW (M1/M2 done), MEDIUM (M3/M5/M6a), MEDIUM-HIGH (M6b constraint), HIGH (M4 LBA) | Reuse existing `qlearning_hierarchical_model` template. Add models in dependency order: M3 (adds kappa), M5 (adds phi_rl), M6a (adds kappa_s, requires per-stimulus state), M6b (adds dual constraint). |
| TS-2 | **Non-centered parameterization (Matt trick) on every level-2 latent** | Centered parameterization produces funnel geometry and divergent transitions in hierarchical RL. Stan User Guide, NumPyro `LocScaleReparam`, and BayesCogSci Ch. 9 all flag this as the first-line fix. The legacy M1/M2 code already uses it. | LOW (template exists) | Pattern: `z_param ~ Normal(0,1); param = expit(logit(mu_param) + sigma_param * z_param)`. Use `numpyro.infer.reparam.LocScaleReparam(centered=0)` if a parameter resists manual reparameterization. |
| TS-3 | **Phi_approx / inverse-probit transform for [0,1] parameters** (alpha+, alpha-, phi, rho, epsilon, kappa, kappa_s, phi_rl) | Truncating normals (`Normal(0,1)T[0,1]`) breaks HMC. Inverse-probit / expit-of-logit is the hBayesDM standard for bounded [0,1] params and provides smooth gradient flow. | LOW | hBayesDM `prl_ewa.stan`: `phi[i] = Phi_approx(mu_pr[1] + sigma[1] * phi_pr[i])`. NumPyro equivalent: `jax.scipy.special.expit`. |
| TS-4 | **Truncated-normal prior for working-memory capacity K** (continuous, bounded [1, 7]) | K is the WM-RL-specific parameter; Collins (2018) and Master et al. (2020) treat K as a continuous latent. Discrete K destroys gradients (no HMC). Bounded continuous with TruncatedNormal(4, 1.5, low=1, high=7) is what the legacy M2 code uses and matches Senta et al. (2025) bounds. | LOW | Decision is dependent on the parallel **Collins K parameterization research** finding. If that research recommends a tighter prior (e.g., low=2), update the bound but keep the truncation strategy. |
| TS-5 | **Weakly informative group-mean priors for [0,1] parameters** | Flat priors on bounded params produce poor mixing and unphysical regions. The legacy code uses `Beta(3,2)` for alpha+ (mean 0.6), `Beta(2,3)` for alpha- (mean 0.4), `Beta(2,8)` for phi (mean 0.2), `Beta(5,2)` for rho (mean 0.7), `Beta(1,19)` for epsilon (mean 0.05). These are softly empirical and consistent across hBayesDM models. | LOW | Document each prior choice with citation in `priors.py`. For new params: kappa `Beta(2,5)` (mean 0.29, matching M3 MLE point estimates); kappa_s `Beta(2,5)`; phi_rl `Beta(2,8)`. |
| TS-6 | **Half-Normal priors on group sigmas** (NOT inverse-gamma) | Gelman 2006 + 15 years of Stan practice: HalfNormal or Half-Cauchy on hierarchical SDs is the modern default. Inverse-gamma creates pathological geometry. Legacy code uses `HalfNormal(0.3)` for [0,1] params and `HalfNormal(1.0)` for K. | LOW | Inherit from existing template. Tighter `HalfNormal(0.2)` for epsilon to prevent the noise parameter from absorbing all variance. |
| TS-7 | **Convergence diagnostics: R-hat <= 1.01, ESS_bulk >= 400, ESS_tail >= 400, zero divergent transitions per model fit** | Vehtari et al. (2021) "Rank-normalization, folding, and localization" sets R-hat <= 1.01 as the modern standard (looser 1.05 is now considered too lax). ESS thresholds from Stan workflow guide. Divergences in NUTS = bias, not inefficiency; non-zero divergences invalidate inference. | MEDIUM | Build `validate_mcmc_fit()` helper that fails the run if any diagnostic breaches threshold. Auto-bump `target_accept` from 0.8 -> 0.95 -> 0.99 on divergent failures. Required for every model fit; cannot ship a model whose fit doesn't pass. |
| TS-8 | **WAIC and PSIS-LOO computation per model** (replacing AIC/BIC as primary criterion) | Vehtari, Gelman, Gabry (2017) — modern Bayesian model comparison standard. AIC is a point-estimate approximation to LOO; with full posteriors, computing the real thing is free and strictly more informative. PSIS-LOO also returns per-observation Pareto-k values that flag influential observations. | MEDIUM | Use `arviz.loo` and `arviz.compare`. Required: store pointwise log-likelihood (`log_likelihood` group in InferenceData) for every fit. Pareto k > 0.7 = warning; k > 1.0 = unreliable LOO -> fall back to k-fold CV for that model. |
| TS-9 | **Joint level-2 trauma regression** (mu_param_i = beta_0 + beta_trauma * trauma_subscale_i) for the parameters of scientific interest (kappa, kappa_s, alpha-, phi_rl) | This is the entire scientific motivation for v4.0. Post-hoc regression on MLE point estimates loses uncertainty (Sullivan-Toole et al. 2022 explicitly recommends against). Joint fitting propagates parameter uncertainty into the regression coefficient and shrinks individuals toward the trauma-conditional group mean — exactly what's needed for the 48-test M6b family. | MEDIUM-HIGH | Pattern: `beta_kappa_lec ~ Normal(0, 1); mu_kappa_i = expit(logit(mu_kappa_global) + beta_kappa_lec * lec_z_i)`. Standardize trauma scales to z-scores BEFORE entering the model. Test on M3 first (1 covariate), expand to M6b (multiple subscales). |
| TS-10 | **Trauma covariates standardized (z-scored) before fitting** | Beta coefficients on raw scales make priors meaningless (a `Normal(0,1)` prior on a beta-coefficient applied to a 0-50 LEC sum is effectively a flat prior on [-50, 50]). Z-scoring makes priors interpretable and improves sampler geometry. | LOW | Compute z-scores in data prep, store original scale for back-transforming reported effects. |
| TS-11 | **Per-participant log-likelihood storage in InferenceData** | Required for WAIC/LOO computation; required for posterior predictive checks; required for per-participant winning-model heterogeneity. ArviZ expects `log_likelihood` group with shape `(chain, draw, participant)` or `(chain, draw, observation)`. | LOW | Use `numpyro.handlers.condition` + `log_likelihood` in NumPyro, or `numpyro.deterministic` to record per-participant log-likelihoods. |
| TS-12 | **Posterior predictive checks (PPC) for at least one fit per model** | Standard hBayesDM workflow. PPCs catch model misspecification that diagnostics miss (e.g., the model fits but predicts implausible accuracy curves). The Bayesian Workflow paper (Gelman et al. 2020) treats PPC as mandatory. | MEDIUM | Generate predicted accuracy by trial position and set size; overlay on observed. Already have synthetic-data infrastructure (Script 09); reuse it. |
| TS-13 | **Reproducible random seeds + saved InferenceData per fit** | Bayesian fits are stochastic; without saved seeds + saved posteriors, results aren't reproducible. ArviZ NetCDF format is the standard. | LOW | Save `*.nc` per fit in `output/bayesian/{model}/{date}/`. Include seed, NumPyro version, JAX version, git hash. |

---

### Differentiators (Competitive Advantages for This Pipeline)

These are features that put the pipeline ahead of the published RLWM-clinical literature. As of 2026, the most cited papers (Collins 2018, Yoo & Collins 2022, McDougle & Collins 2021, Wisco et al. PLoS Comp Bio 2025) all use **MLE point estimates** for RLWM models, with the hierarchical Bayesian extensions limited to simple Q-learning. None of them jointly model trauma covariates inside the hierarchical fit. This is genuinely novel territory.

| # | Feature | Value Proposition | Complexity | Notes |
|---|---------|-------------------|------------|-------|
| D-1 | **Joint hierarchical Bayesian fitting of M3/M5/M6a/M6b** (the WM-RL family with perseveration) | Wisco et al. (PLoS Comp Bio 2025) just published the *MLE* version of RLWM+anxiety. Extending to hierarchical Bayes for the perseveration variants (which the Senta et al. 2025 paper also fit only by MLE) would be a publishable methodological advance, not just a re-analysis. | HIGH | M6b is the largest jump (constrained dual perseveration). Requires the existing kappa+kappa_s reparameterization (`kappa_total in [0,1], split in [0,1]`) lifted into the hierarchical framework. Non-trivial: the constraint must be preserved on the latent z-scale. |
| D-2 | **Subscale-level level-2 regressors** (IES-R intrusion / avoidance / hyperarousal; LEC-5 event subcategories) instead of total scores | The Sullivan-Toole et al. (2022) reliability paper specifically calls out that scale TOTAL scores hide variance that the subscales capture. Almost no clinical RL paper uses subscales as level-2 predictors (most use a single composite). Doing so directly with full uncertainty propagation is genuinely novel. | MEDIUM (per-model), MEDIUM-HIGH (overall — the 4 IES-R + ~6 LEC subscales x 4 parameters of interest = 40+ coefficients per model) | Use a multivariate Normal prior on the regression coefficients with shrinkage (horseshoe or regularized horseshoe) to handle multiplicity. This is preferable to FDR correction post-hoc and is the modern Bayesian approach to the multiple-comparison problem. |
| D-3 | **Bayesian multiple-comparison handling via shrinkage priors** (regularized horseshoe on beta_trauma coefficients) | The Carvalho et al. (2010) horseshoe and Piironen & Vehtari (2017) regularized horseshoe are the gold standard for Bayesian variable selection / multiple-comparison handling. With ~48 tests in the M6b family (4 params x 12 subscales), an FDR procedure is fragile and loses power. Shrinkage handles this natively. | HIGH | Requires careful specification (global scale, slab degrees of freedom). PyMC/NumPyro both support it. Optional: start with weakly informative `Normal(0, 0.5)` on standardized betas, add horseshoe in a second iteration if reviewers ask. |
| D-4 | **WAIC + LOO model comparison with stacking weights** | `arviz.compare(method='stacking')` produces optimal predictive weights instead of forcing a single winner. This handles per-participant model heterogeneity (already documented in the v3 winners-table) better than picking a single winning model. | LOW | One function call once log_likelihoods are stored. Report alongside Akaike weights from MLE for backward compatibility. |
| D-5 | **Pareto-k diagnostic flagging for influential participants** | PSIS-LOO returns per-observation Pareto k values. Participants with k > 0.7 are influential outliers — exactly the kind of person you want to flag in a clinical sample. This is a free byproduct of LOO computation. | LOW | Add to the report a "high-influence participants" table with k values and trauma scores. |
| D-6 | **Posterior predictive p-values stratified by trauma group** | Goes beyond the Bayesian fit: tests whether the trauma-conditional model captures group-specific behavior patterns, not just average behavior. Catches the failure mode where "the model fits the average but misses the high-trauma tail." | MEDIUM | PPC simulation already exists (Script 09); add stratification by trauma quartile and report Bayesian p-values. |
| D-7 | **Variational inference (SVI) as fast pre-fit before NUTS** | NumPyro's SVI runs in ~30s and produces a starting point that dramatically reduces NUTS warmup. Used as a sanity check (does SVI find anything sensible?) and as MCMC initialization. Standard in NumPyro tutorials. | MEDIUM | Use `AutoNormal` or `AutoLowRankMultivariateNormal` guides. SVI is *not* the final fit — NUTS is required for valid uncertainty quantification — but SVI catches bugs fast. |
| D-8 | **Comparison: hierarchical Bayes vs MLE point estimates side-by-side per parameter** | Sullivan-Toole et al. (2022) and the Psychonomic Bulletin & Review 2024 paper both ask "does hierarchical Bayes actually improve reliability for *this* dataset?" Reporting both side-by-side answers that question for our N=154 trauma sample. | LOW | Already have MLE estimates; just produce a scatterplot of MLE vs posterior mean and report shrinkage magnitude. |
| D-9 | **Cluster GPU acceleration for hierarchical fits via existing rlwm_gpu env + Monash M3** | NumPyro on JAX-GPU runs hierarchical NUTS roughly 5-20x faster than CPU. Cluster infrastructure already exists (`cluster/12_mle_gpu.slurm`). Hierarchical Bayesian fitting at N=154 with 4 chains x 2000 samples is feasible on GPU but slow on laptop CPU. | MEDIUM | Adapt the existing GPU SLURM script for `13_fit_bayesian.py`. Requires NumPyro to be installed in `rlwm_gpu` env and JAX-CUDA backend selected. |
| D-10 | **Per-model predictive distribution overlay plot** for paper figures | Reviewers will ask "does the hierarchical model actually predict the data better?" A single figure showing observed accuracy curves with shaded posterior predictive intervals per model is the most persuasive evidence. | LOW | Reuse existing accuracy-by-block plotting; add posterior predictive ribbons. |

---

### Anti-Features (Explicitly Do NOT Build)

These are features that look attractive but create problems documented in the literature or in this pipeline's specific context. Each comes with a recommended alternative.

| # | Anti-Feature | Why Tempting | Why Problematic | Alternative |
|---|--------------|--------------|-----------------|-------------|
| AF-1 | **Centered parameterization for any [0,1] RL parameter** | Cleaner-looking model code; matches the math notation directly | Stan + NumPyro forums are full of divergent-transition reports caused by this. The funnel geometry is the canonical hierarchical-model failure mode. The Sullivan-Toole et al. (2022) tutorial code, hBayesDM, and the legacy `numpyro_models.py` all use non-centered. There is no scenario where centered helps for our N=154. | Always use the non-centered z-score template (TS-2). |
| AF-2 | **Full multivariate-normal correlation matrix on the group-level parameters** (LKJ prior on the correlation) | "More principled" — captures correlations between, e.g., alpha+ and alpha-; standard in Pedersen & Frank (2020) RLDDM | At 7+ parameters per RLWM model and N=154, the LKJ correlation matrix has 21+ correlation parameters that are very weakly identified. Sampling time blows up; divergent transitions multiply. Wisco et al. (2025) and the legacy M1/M2 code both use **independent** group priors and report no problems. The correlation gain is empirically tiny for RLWM at this N. | Independent group priors (`mu_param ~ Beta(...); sigma_param ~ HalfNormal(...)`). Report empirical posterior correlations between parameters as a diagnostic, not as part of the model. If a reviewer demands LKJ, add it for the winning model only as a sensitivity analysis. |
| AF-3 | **Hierarchical Bayesian fitting of M4 (LBA) with full PMwG / particle MCMC** | "Symmetry with the choice-only models"; "We should have hierarchical Bayes for everything" | Hierarchical LBA is *exotic* in clinical computational psychiatry as of 2026. The Gunawan et al. (2020) PMwG sampler exists but lives entirely in the R `pmwg` package; no NumPyro/PyMC equivalent. The Annis et al. Stan LBA is documented as having strong assumptions that "cannot be relaxed in a straightforward way." Critically: **zero published clinical-population studies use hierarchical Bayesian LBA with covariates as of 2026 search results.** Implementing it from scratch on JAX would be a 4-6 week side quest with no published precedent to validate against. The float64 + log-space CDF requirements make even MLE M4 the most complex thing in the pipeline. | Keep M4 at MLE. Report hierarchical Bayesian results for the choice-only family (M1, M2, M3, M5, M6a, M6b) as the primary results. Report M4 MLE as a separate "joint choice+RT" track in the paper, exactly as v3.0 does. If reviewers demand hierarchical LBA, the response is "PMwG sampler in R; out of scope for the JAX pipeline." |
| AF-4 | **Discrete prior on K** (categorical 1, 2, 3, 4, 5, 6, 7) | K is conceptually a "number of items" — discrete is the natural choice; some Collins lab papers fit it discrete by grid search | HMC/NUTS requires continuous parameters with smooth gradients. A discrete K kills the sampler. Marginalizing over discrete K via mixture is technically possible but complex and slow. | Continuous bounded `TruncatedNormal(4, 1.5, low=1, high=7)`. Round to nearest integer in reporting only, not in fitting. The Collins K parameterization research that's running in parallel may suggest a tighter range — apply that to the truncation bounds. |
| AF-5 | **AIC/BIC as the primary model-comparison criterion** in v4.0 | Backward compatibility with v3.0; "everyone knows what AIC means" | With full Bayesian posteriors, AIC is strictly worse than WAIC/LOO (it's a point-estimate approximation). The whole point of the v4.0 milestone is to move beyond point-estimate-based inference. Reporting AIC as primary undermines the rationale for the milestone. | Primary: WAIC + PSIS-LOO with stacking weights (TS-8, D-4). Report AIC/BIC as a secondary table for backward comparison with v3.0. State explicitly in the methods that hierarchical fits prefer WAIC. |
| AF-6 | **Post-hoc regression of MLE point estimates on trauma scales as the level-2 inference** | Already implemented in Scripts 16, 16b; "if it works, why redo it?" | Post-hoc regression on point estimates discards the parameter uncertainty (the SE on alpha- per participant is sometimes 30% of the value). This produces inflated significance and biased coefficient estimates. The Sullivan-Toole et al. (2022) paper explicitly warns against it ("fitting the model separately for each individual resulted in substantial and systematic bias"). For a 48-test family in M6b, the bias is non-negligible. | Joint hierarchical regression (TS-9). Keep Scripts 16/16b around for backward compatibility but stop using them as the primary inference once v4.0 ships. |
| AF-7 | **MCMC with very long chains (10k+ samples) "to be safe"** | Defensive programming against undetected non-convergence | With proper non-centered parameterization and target_accept=0.9, NumPyro NUTS converges for these models in ~1000-2000 post-warmup samples per chain. Longer chains waste GPU time and don't fix bias from divergent transitions. The Pedersen & Frank RLDDM paper uses 4 chains x 1000 samples; hBayesDM defaults to 4 chains x 1000 samples. | 4 chains, 1000 warmup, 1000-2000 samples. Increase only if R-hat or ESS fail thresholds — and if they fail, it's a model spec problem, not a sample-count problem. |
| AF-8 | **Single composite trauma score** (e.g., LEC-total OR IES-R-total) as the level-2 predictor | Simplicity — fewer coefficients, easier to report | Specifically what reviewers will push back on: composite scores hide that perseveration tracks LEC events but not IES-R intrusion (or whatever the actual pattern is). The whole motivation for v4.0 is the subscale-level scientific question. | Include both totals AND subscales as separate tables. Subscales as the primary inference; totals reported for cross-paper comparability. Use shrinkage prior (D-3) to handle the multiplicity. |
| AF-9 | **Manual reimplementation of the JAX likelihood functions inside NumPyro models** | "It would be cleaner to have everything in one file" | The existing `jax_likelihoods.py` is JIT-compiled, masked-padding, and validated by 7 model recovery passes. Reimplementing inside NumPyro models risks subtle bugs and breaks the test suite. | Wrap the existing JAX likelihoods with `numpyro.factor(name, log_lik)` exactly as the legacy `qlearning_hierarchical_model` does. The JAX functions are already pure and differentiable; they slot directly into NumPyro. |
| AF-10 | **Save all MCMC samples to CSV** | "More flexible for downstream analysis" | InferenceData NetCDF is ~10x smaller, preserves the chain/draw/dim structure, and is the standard ArviZ format. CSV explodes for 4 chains x 2000 samples x 100+ parameters x N=154 individuals. | `inference_data.to_netcdf()`. Provide a CSV export utility for the small subset of summary statistics (posterior means, 95% HDIs) needed for tables. |
| AF-11 | **Variational inference (SVI) as the primary fit** instead of MCMC | "MCMC is slow; SVI gives me posteriors fast" | SVI returns *approximate* posteriors that systematically underestimate variance, especially for the multimodal/skewed parameters in RL models (learning rates near 0 or 1, capacity K). For a clinical-comparison paper, using SVI as the primary fit will get the paper rejected at any reputable venue. | NUTS is the primary fit. SVI is a pre-fit sanity check + warmup initialization (D-7), nothing more. |

---

## Feature Dependencies

```
[TS-1: 2-level hierarchy template]
    └──extends to──> [TS-2: Non-centered parameterization]
                         └──requires──> [TS-3: Phi_approx transforms]
                                            └──depends on──> [TS-5: Weakly informative priors]
                                            └──depends on──> [TS-6: HalfNormal sigmas]

[TS-4: K parameterization]
    └──depends on──> [parallel Collins K research output]

[TS-9: Joint level-2 trauma regression]
    └──requires──> [TS-1, TS-2, TS-3, TS-10]
    └──enables──> [D-2: Subscale regressors]
                      └──enables──> [D-3: Shrinkage priors for multiplicity]

[TS-8: WAIC + LOO]
    └──requires──> [TS-11: Pointwise log-likelihood storage]
    └──enables──> [D-4: Stacking weights]
    └──enables──> [D-5: Pareto-k flagging]

[TS-7: Convergence diagnostics]
    └──gates──> Every fit. No fit ships without passing.
    └──depends on──> [TS-2: Non-centered] (centered parameterizations break TS-7)

[TS-12: PPCs]
    └──depends on──> existing Script 09 simulation infrastructure

[D-9: GPU acceleration]
    └──depends on──> existing rlwm_gpu env, JAX-CUDA, cluster SLURM scripts

[AF-3: Hierarchical LBA] is EXPLICITLY EXCLUDED
    M4 stays at MLE; reported as separate joint choice+RT track
```

### Dependency Notes

- **TS-1 / TS-2 / TS-3 are a single technique, split for clarity:** the non-centered template + Phi_approx transform IS the hBayesDM pattern. They cannot be implemented independently.
- **TS-9 (joint regression) is the central new feature of v4.0** and depends on the entire foundation being correct. Build M3 first (1 covariate, 1 parameter of interest = kappa) as the proof of concept; expand to M6b only after M3 fits cleanly.
- **D-3 (horseshoe shrinkage)** is optional in iteration 1 (`Normal(0, 0.5)` works); adds value when handling the M6b 48-coefficient family. Defer to phase 2.
- **AF-3 (hierarchical LBA) is the load-bearing exclusion.** Trying to build it would consume 4-6 weeks with no published precedent. The v4.0 scope explicitly excludes it.
- **K parameterization (TS-4) couples to a parallel research thread.** If Collins K research recommends `K in [2, 6]` or a different functional form, update TS-4 only — none of TS-1/2/3/5/6 change.

---

## MVP Definition

### Launch With (v4.0 Phase 1)

Minimum viable hierarchical Bayesian pipeline that delivers the v4.0 scientific claim.

- [ ] **TS-1 through TS-7** for **M3** (the existing winning choice-only model in v3.0 by AIC, before M5/M6 ranked higher)
- [ ] **TS-8** WAIC + LOO computed for the M3 fit
- [ ] **TS-9** joint level-2 regression of `kappa` on `LEC-total` (the v3.0 surviving FDR finding) — proves the joint regression infrastructure works
- [ ] **TS-11, TS-12** log-likelihood storage + posterior predictive check for M3
- [ ] **TS-13** reproducible save/load with seed + git hash
- [ ] **D-9** runs on cluster GPU

This validates the entire hierarchical-Bayesian-with-covariates infrastructure on the simplest non-trivial case (M3 with one covariate). Once it works, scaling up is mechanical.

### Add After Validation (v4.0 Phase 2)

Once M3 is fitting cleanly with joint regression, expand to the rest of the choice-only family.

- [ ] M5 hierarchical fit + joint regression
- [ ] M6a hierarchical fit + joint regression (per-stimulus state — moderate complexity)
- [ ] M6b hierarchical fit + joint regression (constrained dual perseveration — highest complexity in choice-only family)
- [ ] **D-2** Subscale-level regressors (IES-R intrusion/avoidance/hyperarousal; LEC subcategories) for the M6b winning model
- [ ] **D-3** Regularized horseshoe prior for the 48-coefficient M6b regression family
- [ ] **D-4** Stacking weights across all 6 choice-only models
- [ ] **D-5** Pareto-k diagnostic per fit
- [ ] **D-8** MLE-vs-Bayesian side-by-side comparison plots
- [ ] **D-10** Predictive distribution overlay paper figures

### Future Consideration (deferred / out of scope)

- [ ] Hierarchical M4 (LBA) — see AF-3. Out of scope unless reviewers force it. Recommended path if forced: implement in R `pmwg` as a separate compute track, not in JAX.
- [ ] Multivariate normal group prior with LKJ correlation — see AF-2. Adds little for N=154. Only if a reviewer specifically asks.
- [ ] Hierarchical Bayesian variants of M1 and M2 with subscale regressors — already have the M1/M2 templates; mechanical extension. Defer until paper revision phase.
- [ ] Bayesian model averaging across the choice-only family (full BMA, not just stacking) — niche, only useful if no single winner emerges.

---

## Feature Prioritization Matrix

| Feature ID | User Value | Implementation Cost | Priority |
|---|---|---|---|
| TS-1 (2-level hierarchy M3-M6b) | HIGH | MEDIUM | P1 |
| TS-2 (non-centered) | HIGH | LOW (template exists) | P1 |
| TS-3 (Phi_approx) | HIGH | LOW | P1 |
| TS-4 (K parameterization) | HIGH | LOW (after parallel research) | P1 |
| TS-5 (weakly informative priors) | HIGH | LOW | P1 |
| TS-6 (HalfNormal sigmas) | HIGH | LOW | P1 |
| TS-7 (convergence diagnostics) | HIGH | MEDIUM | P1 |
| TS-8 (WAIC + LOO) | HIGH | MEDIUM | P1 |
| TS-9 (joint level-2 regression) | HIGH | MEDIUM-HIGH | P1 |
| TS-10 (z-scored covariates) | HIGH | LOW | P1 |
| TS-11 (log-likelihood storage) | HIGH | LOW | P1 |
| TS-12 (PPCs) | MEDIUM | MEDIUM | P1 |
| TS-13 (reproducibility) | HIGH | LOW | P1 |
| D-1 (M3-M6b hierarchical) | HIGH | HIGH | P1 (this is the milestone) |
| D-2 (subscale regressors) | HIGH | MEDIUM | P1 |
| D-3 (horseshoe shrinkage) | MEDIUM | HIGH | P2 |
| D-4 (stacking weights) | MEDIUM | LOW | P2 |
| D-5 (Pareto-k flagging) | MEDIUM | LOW | P2 |
| D-6 (group-stratified PPC) | MEDIUM | MEDIUM | P2 |
| D-7 (SVI pre-fit) | LOW-MEDIUM | MEDIUM | P3 |
| D-8 (MLE vs Bayes plots) | MEDIUM | LOW | P2 |
| D-9 (GPU acceleration) | HIGH | MEDIUM | P1 |
| D-10 (predictive overlay figures) | MEDIUM | LOW | P2 |
| AF-3 (hierarchical LBA) | LOW (paper risk + cost ratio) | VERY HIGH | EXCLUDED |
| AF-2 (LKJ correlation) | LOW (at this N) | HIGH | EXCLUDED (sensitivity only) |

**Priority key:**
- **P1:** Required for v4.0 launch — without these the milestone is incomplete.
- **P2:** Should ship in v4.0 if time permits. Adds publishable polish.
- **P3:** Nice to have. Defer to paper revision phase.
- **EXCLUDED:** Explicitly NOT in v4.0 scope.

---

## Comparison to State of the Art

| Feature | hBayesDM (Ahn et al. 2017) | Pedersen & Frank RLDDM 2020 | Wisco et al. PLoS Comp Bio 2025 (RLWM+anxiety, MLE) | Senta et al. 2025 (RLWM, MLE, this dataset) | This pipeline v4.0 |
|---|---|---|---|---|---|
| Hierarchical structure | Yes (2-level) | Yes (2-level) | NO (MLE) | NO (MLE) | Yes (2-level) |
| RLWM family | NO (Q-learning + DDM only) | NO (Q-learning + DDM only) | Yes (RLWM only) | Yes (M1-M6b) | **Yes (M1-M6b)** |
| Joint level-2 covariate regression | Limited (group dummies) | Limited (within-subject only) | Post-hoc Spearman with FWER | Post-hoc Spearman with FWER | **Yes (subscale regressors with shrinkage)** |
| Non-centered parameterization | Yes (Phi_approx) | Yes | n/a (MLE) | n/a (MLE) | Yes |
| WAIC/LOO model comparison | Yes | Yes | NO (AIC) | NO (AIC) | Yes |
| Hierarchical LBA | NO | NO | NO | NO (MLE only) | NO (excluded; M4 stays MLE) |
| GPU-accelerated NUTS | NO (Stan CPU) | NO (PyMC2 CPU) | n/a | n/a | **Yes (NumPyro on JAX-CUDA)** |
| Publicly available code | Yes | Yes | Yes | OSF data only | Yes (this repo) |

The differentiators are: (a) **first hierarchical Bayesian fit of the M3-M6b RLWM perseveration family**, (b) **subscale-level trauma regressors with joint regression**, and (c) **GPU-accelerated NumPyro implementation** — none of which exist in the published literature as of April 2026.

---

## Sources

**Hierarchical Bayesian RL methodology (HIGH confidence):**
- [Ahn et al. (2017) hBayesDM Computational Psychiatry](https://pmc.ncbi.nlm.nih.gov/articles/PMC5869013/) — canonical hierarchical RL pattern, prior choices, Phi_approx transforms
- [hBayesDM `prl_ewa.stan` source](https://github.com/CCS-Lab/hBayesDM/blob/master/commons/stan_files/prl_ewa.stan) — verified non-centered parameterization template
- [Hierarchical Bayesian Models of Reinforcement Learning (van Geen & Gerraty 2021, J Math Psych)](https://www.sciencedirect.com/science/article/abs/pii/S0022249621000742) — comparison of HBA vs alternative methods
- [Pedersen & Frank (2020) RLDDM tutorial, Comp Brain Behav](https://link.springer.com/article/10.1007/s42113-020-00084-w) — hierarchical fitting of joint RL+DDM
- [Stan reparameterization guide / Matt trick](https://mc-stan.org/docs/stan-users-guide/reparameterization.html) — non-centered parameterization rationale
- [NumPyro `LocScaleReparam` documentation](https://num.pyro.ai/en/latest/reparam.html) — automatic non-centering tool
- [Bayesian Cognitive Modeling for Cognitive Science Ch.9 (Nicenboim)](https://bruno.nicenboim.me/bayescogsci/ch-complexstan.html) — when to use centered vs non-centered

**Computational psychiatry / clinical covariates (HIGH–MEDIUM confidence):**
- [Sullivan-Toole et al. (2022) Computational Psychiatry on RL parameter reliability](https://cpsyjournal.org/articles/89/) — shrinkage benefits, level-2 regression, anti-patterns for separate fitting
- [Hierarchical Bayesian modeling for experimental psychopathology tutorial (PMC8634778)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8634778/) — sample-size guidance, prior selection, convergence diagnostics for clinical samples
- [Reliability of computational model parameters (Mkrtchian et al. Psychon Bull Rev 2024)](https://link.springer.com/article/10.3758/s13423-024-02490-8) — does hierarchical Bayes actually improve reliability?
- [Wisco et al. (2025) PLoS Comp Bio: Dual-process anxiety + RLWM](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012872) — most-relevant prior work; MLE only, post-hoc covariate analysis. Confirms the gap v4.0 fills.

**RLWM-specific (HIGH confidence):**
- [Collins (2018) Eur J Neuro: WM and RL contributions to learning](https://pubmed.ncbi.nlm.nih.gov/25297101/) — original RLWM model with K parameterization
- [Yoo & Collins (2022) JoCN: WM and RL cognitive and computational perspective](https://ccn.berkeley.edu/pdfs/papers/YooCollins2022JoCN_WMRL.pdf) — RLWM as a computational framework
- [bioRxiv 2024.04: Working memory as representational template for RL](https://www.biorxiv.org/content/10.1101/2024.04.25.591119v2.full) — current RLWM extensions
- [bioRxiv 2024.02: Striatal dopamine and RLWM with hierarchical Bayesian regression](https://www.biorxiv.org/content/10.1101/2024.02.14.580392v1.full) — confirms hierarchical Bayesian regression on RLWM is current SOTA

**Hierarchical LBA (LOW–MEDIUM confidence — almost no clinical literature):**
- [Gunawan et al. (2020) PMwG sampler for hierarchical LBA](https://arxiv.org/abs/1806.10089) — only known mature implementation; R only
- [Newcastle Cognition Lab `pmwg` R package](https://newcastlecl.github.io/pmwg/) — reference implementation, no JAX/Python equivalent
- [Particle-based samplers tutorial (Forstmann LBA chapter)](https://newcastlecl.github.io/samplerDoc/forstmannChapter.html) — confirms PMwG covers within-subject manipulations only; no clinical-covariate examples published
- **No 2023-2025 papers found** that fit hierarchical Bayesian LBA to clinical populations with covariates. This is the empirical justification for AF-3 (excluding hierarchical LBA from v4.0).

**Model comparison / WAIC / LOO (HIGH confidence):**
- [Vehtari, Gelman, Gabry (2017) Practical Bayesian model evaluation using LOO and WAIC](https://link.springer.com/article/10.1007/s11222-016-9696-4) — primary methodology reference
- [`loo` R package + `arviz.compare`](https://mc-stan.org/loo/) — implementation
- [Vehtari et al. (2021) Rank-normalization, folding, localization R-hat](https://arxiv.org/abs/1903.08008) — modern R-hat <= 1.01 standard

**Bayesian workflow (HIGH confidence):**
- [Bayesian Workflow for Generative Modeling in Computational Psychiatry (Yarkoni & Westfall, Comp Psych)](https://cpsyjournal.org/articles/10.5334/cpsy.116) — overall workflow recommendations
- [Wilson & Collins (2019) Ten simple rules for computational modeling, eLife](https://elifesciences.org/articles/49547) — parameter recovery, model recovery

---

*Feature research for: v4.0 hierarchical Bayesian RLWM pipeline with trauma subscale level-2 predictors*
*Researched: 2026-04-10*
*Confidence: HIGH for table stakes, MEDIUM for differentiators (subscale regression with shrinkage is novel), LOW–MEDIUM for the hierarchical LBA exclusion (negative-result confidence — empirical absence of published precedent)*
