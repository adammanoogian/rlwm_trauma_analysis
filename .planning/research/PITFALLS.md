# Pitfalls Research — v4.0 Hierarchical Bayesian Pipeline & LBA Acceleration

**Domain:** Hierarchical Bayesian RLWM + LBA fitting with clinical Level-2 covariates
**Researched:** 2026-04-11
**Confidence:** HIGH (NumPyro patterns, non-centered reparameterization, multicollinearity); MEDIUM (hierarchical LBA under NUTS — sparse literature); MEDIUM (WAIC across heterogeneous observables)

---

## Critical Pitfalls

### Pitfall 1: Centered parameterization for partially-pooled bounded parameters causes funnel pathologies

**What goes wrong:**
NUTS gets stuck on Neal's funnel when individual-level parameters are sampled directly conditional on the group-level scale. Symptoms: divergences concentrated in low-`sigma` regions, R-hat blowing up on `sigma_alpha_pos`/`sigma_K`/`sigma_phi`, ESS bulk dropping below 100 even with 2000 post-warmup samples. For RLWM with 154 participants and 6+ parameters per model, a centered parameterization will almost certainly fail to converge for at least one parameter.

**Why it happens:**
The legacy `numpyro_models.py` does non-centered parameterization for `alpha_pos`/`alpha_neg`/`epsilon` correctly, but adding `phi`, `rho`, `K`, `kappa_total`, `kappa_share` requires repeating that pattern carefully. The pattern is fragile because: (a) bounded parameters need a link function (logit for [0,1], log for positive, softmax/stick-breaking for simplex), (b) the non-centered offset must be added on the *unconstrained* scale, and (c) it's easy to add the offset on the constrained scale by mistake, which silently produces a wrong (but sampleable) model.

**How to avoid:**
- Use the canonical pattern: `theta_unconstrained = mu_unc + sigma_unc * z` then `theta = link^-1(theta_unconstrained)`, never `theta = link^-1(mu) + sigma * z`.
- For K (integer-valued, bounded [1,7]): treat as continuous in unconstrained space, transform with `1 + 6 * sigmoid(...)`. Use `numpyro.deterministic('K', ...)` so posterior has both unconstrained and natural scales.
- For stick-breaking `kappa_total`/`kappa_share`: sample two independent unconstrained Normals, transform `kappa_total = softplus(...)` and `kappa_share = sigmoid(...)`. Do NOT sample on the simplex directly under NUTS — use the unconstrained representation.
- Test on simulated data (n=20, 50, 154 subjects) before fitting real data. If the unconstrained model can't recover known parameters, the prior is wrong.

**Warning signs:**
- `mcmc.print_summary()` shows R-hat > 1.05 on any `sigma_*` parameter.
- `numpyro.diagnostics.summary()` reports divergences > 0.5% of total samples (for 4 chains × 1000 samples = 4000, that's > 20 divergences).
- `arviz.plot_trace()` shows hairy/funnel-shaped traces for individual-level parameters when group `sigma` is small.
- `arviz.plot_pair(divergences=True)` shows divergences clustered in a wedge.

**Phase to address:** Phase 15 (hierarchical framework). Build the parameterization helper module FIRST, with unit tests on simulated data, before wiring any model into NUTS.

---

### Pitfall 2: Hierarchical shrinkage masks genuine non-identifiability

**What goes wrong:**
The MLE-level parameter recovery for M5/M6a/M6b already fails (`r < 0.80` on every parameter). Adding hierarchy will produce *converged* sampling with tight credible intervals — but only because all individuals are pulled to the group mean. The posterior will look beautiful and be scientifically meaningless. Group-level effects (especially trauma covariates) can become artifactually significant because there is no individual variance for them to compete with.

**Why it happens:**
NUTS converges on the *marginal* (group-level) posterior even when the *conditional* (individual-level) likelihood is flat. Researchers see R-hat = 1.00, ESS > 1000, no divergences and conclude the fit is good. The diagnostic that catches this — comparing posterior individual SDs against the prior individual SDs — is not part of standard NumPyro/ArviZ output.

**How to avoid:**
- Run **prior predictive checks** before fitting: simulate data from the prior, fit, and verify recovery.
- Run **posterior predictive checks** at the individual level: do simulated trial sequences from each subject's posterior mean reproduce that subject's actual learning curve?
- Compare per-subject posterior SD against the prior SD. If the ratio is > 0.7 for most subjects, the data is not informing individual parameters — only the group is being learned.
- Compute **posterior shrinkage** statistic: `1 - (var_post_individual / var_post_group)`. Should be > 0.5 for meaningful learning.
- Run a pure-shrinkage null model: same hierarchy but with a permuted likelihood (data shuffled across subjects). If trauma covariate effects survive permutation, they're shrinkage artifacts.

**Warning signs:**
- All 154 individual-level posterior means cluster within ±1σ of group mean.
- `arviz.plot_forest()` of individual parameters shows nearly identical credible intervals for everyone.
- Posterior `sigma_*` is suspiciously close to its prior (e.g., `HalfNormal(0.3)` posterior peaks near 0.3).
- Trauma covariate slopes have credible intervals that shrink as you add subjects (genuine effects should stabilize, shrinkage artifacts get tighter as N grows).

**Phase to address:** Phase 16 (fitting). Build the shrinkage-diagnostic gate as a pass/fail check that runs after every fit, before the model is treated as "valid" for downstream analysis.

---

### Pitfall 3: Multicollinear IES-R subscales as Level-2 predictors give arbitrary slope assignments

**What goes wrong:**
With intrusion/avoidance/hyperarousal correlated at r ≈ 0.6–0.8, the joint posterior for their slopes on `phi` (or any RLWM parameter) is highly degenerate. NUTS will sample a long thin ridge in the joint posterior. R-hat may converge, but the marginal slopes are individually meaningless and unstable across runs. Worse: the posterior credible intervals are wide and centered near zero (so each subscale "looks" non-significant) while the *combined* effect is real but invisible because the analyst is only inspecting marginals.

**Why it happens:**
This is the textbook regression collinearity problem, but Bayesian analysts often think priors will "regularize it away." Weakly informative N(0, 1) priors do not regularize enough when the design matrix has condition number > 30. NumPyro/PyMC do not warn about predictor collinearity automatically.

**How to avoid:**
- Compute the IES-R subscale correlation matrix and condition number BEFORE fitting. If condition number > 10, treat the regression as collinear and act accordingly.
- **Recommended:** Use the IES-R total score plus residualized subscales (Gram-Schmidt orthogonalization). This gives a "global trauma" effect and "subscale-specific deviation from total" effects that are uncorrelated by construction.
- **Alternative:** Use a single composite (PCA component 1) plus orthogonal residuals.
- **Alternative:** Hierarchical horseshoe / regularized horseshoe priors (`numpyro.distributions.HorseshoeNormal`) — these do regularize collinear predictors aggressively.
- Always also report the **joint** test of all subscales' slopes (Bayes factor on `slope_intrusion = slope_avoidance = slope_hyperarousal = 0`), not just the marginals.

**Warning signs:**
- Slope posteriors for the three subscales have correlations > 0.5 in the joint posterior (use `arviz.plot_pair`).
- ESS drops 5–10× compared to the same model without the subscale predictors.
- Slopes flip sign across reruns with different seeds (instability).
- Variance inflation factor (VIF) > 5 on any subscale (compute on the standardized predictor matrix).

**Phase to address:** Phase 17 (Level-2 structure). The subscale parameterization decision must be made before fitting, not after seeing posteriors.

---

### Pitfall 4: NumPyro `lax.scan` inside the trial loop is required — Python loops break NUTS gradient performance

**What goes wrong:**
The legacy `numpyro_models.py` calls `q_learning_multiblock_likelihood` once per participant inside a Python `for` loop, with the underlying likelihood using `jax.lax.scan` over trials. This works but compiles slowly (minutes per model) and produces gradient ops that JAX cannot fully fuse. Adding M5/M6a/M6b/M4 with hierarchical sampling pushes compile time to 10–30 minutes per fit, and any code change re-triggers compilation. Worse, debugging a NaN inside `lax.scan` under `jax.grad` produces unhelpful tracebacks.

**Why it happens:**
NumPyro traces the model function once per chain at compile time. Python-level loops over participants create one node per participant in the trace, which JAX must then fuse. With 154 participants × 5+ parameters × 1000+ trials, the trace becomes huge. The cleaner pattern (`numpyro.plate` + vmap over participants) requires the likelihood to be fully vectorized over participants — which means equal-length sequences (padding/masking) and a single `lax.scan` over the longest block.

**How to avoid:**
- Refactor likelihoods to operate on a participant-batched tensor: `stimuli` of shape `(n_participants, n_trials_max)`, with a `mask` for valid trials.
- Use `jax.vmap` over participants for the per-trial state update, with a single `lax.scan` over trials.
- Wrap the entire likelihood in `numpyro.plate('participants', n_participants)` and call `numpyro.factor` once with the *vector* of per-subject log-likelihoods, not 154 separate `numpyro.factor` calls.
- Cache compiled JIT artifacts using `jax.experimental.compilation_cache` (set `JAX_COMPILATION_CACHE_DIR`) — this saves 10+ minutes on every restart.
- Build a small unit test that fits 5 subjects in < 30 seconds. If that test grows to > 60 seconds at any point, the vectorization broke.

**Warning signs:**
- "Compiling..." message takes > 5 minutes for a model that has compiled before.
- `numpyro.factor` is called inside a Python loop in the model definition.
- Memory usage during compile exceeds 16 GB.
- A single change to a hyperparameter triggers full recompile.

**Phase to address:** Phase 15 (hierarchical framework). The vectorized likelihood pattern is foundational — every model must follow it from the start.

---

### Pitfall 5: Hierarchical LBA under NUTS hits Pareto-k > 0.7 for LOO and gives misleading model comparison

**What goes wrong:**
LBA likelihoods produce extreme log-densities for fast/correct trials (log p ≈ +5) and very negative log-densities for slow/wrong trials (log p ≈ -20). This high variance in pointwise log-likelihood means the importance sampling step in PSIS-LOO fails — Pareto-k diagnostics exceed 0.7 for many trials, and the LOO estimate becomes unreliable. WAIC has similar problems but is harder to diagnose. Naively comparing M4 to M3 by `loo_compare` produces a number with no warning that the comparison is invalid.

**Why it happens:**
LOO assumes the leave-one-out posterior is close to the full posterior. For LBA, dropping a single trial barely moves the posterior (good), but the importance ratio for that trial is extreme because the trial's contribution to the log-density was extreme. PSIS smooths the tail, but only up to Pareto-k ≈ 0.7. Beyond that, you need K-fold CV (expensive) or a refit-LOO (very expensive).

**How to avoid:**
- After fitting M4, IMMEDIATELY run `arviz.loo(idata, pointwise=True)` and inspect the Pareto-k distribution. Plot `az.plot_khat(loo)`.
- If > 5% of points have khat > 0.7: do NOT use LOO. Use either (a) WAIC with explicit warning, (b) refit subset LOO on the bad points, or (c) abandon LOO and use posterior predictive RMSE on RT distribution as the M4 fit quality metric.
- For comparing M4 to M3 (different observables — choice+RT vs choice-only): you cannot use LOO/WAIC at all on the joint observable. Compute the **choice-only** log-likelihood from M4's posterior (marginalize over RT) and compare to M3's choice-only log-likelihood. This is the only apples-to-apples comparison.
- Document explicitly in the comparison output that the choice-only marginal of M4 is what's being compared, not the joint.

**Warning signs:**
- `loo_compare` output shows large `se_diff` (> 4) — LOO is uncertain about the comparison.
- `arviz.loo()` warning: "Estimated shape parameter of Pareto distribution is greater than 0.7 for one or more samples."
- M4 LOO suddenly looks much better than M3 with no theoretical reason — likely a Pareto-k pathology, not a real win.
- Refitting M4 with a different seed changes the LOO comparison by > 2 standard errors.

**Phase to address:** Phase 18 (model comparison). Build Pareto-k gating into the comparison script: any model with > 5% bad-k trials must use the marginal-choice fallback comparison.

---

### Pitfall 6: K parameterization change silently invalidates pre-existing MLE fits without crashing

**What goes wrong:**
The Collins research findings change how K is parameterized (e.g., from `K ∈ [1,7]` continuous to `K ∈ {1,2,3,4,5,6,7}` discrete, or to `K = round(2 + 5*sigmoid(K_unc))`). The old MLE fits stored in `output/wmrl_m3_mle_fits.csv` have the old K column. The new fitting code reads the same column name and silently uses old values as initial conditions or as priors, producing fits that look fine but encode the wrong K. Downstream scripts (15, 16, 17) consume the mixed-K column without knowing.

**Why it happens:**
Pandas CSVs are schema-less. There is no validation that the K in the CSV matches the K convention of the current code. The pipeline has historical "silent bug" failures from exactly this pattern (M6a kappa_s elif branches missed in `fit_all_gpu`).

**How to avoid:**
- Add a `parameterization_version` column to every fit output CSV. Bump it whenever a parameter convention changes. Downstream scripts MUST check it on load and fail loudly if it's wrong.
- Rename the K column when convention changes: `K` → `K_v1` → `K_v2`. Old code using `K` will KeyError.
- Store the parameterization metadata (link function, bounds, prior) in a sidecar JSON next to the CSV, and load+validate it in every consumer.
- Re-run all MLE fits as part of the K parameterization phase. Do NOT mix v3.0 fits with v4.0 fits in the same downstream analysis.
- Add a "parameter audit" script that loads every fit CSV and verifies parameter column ranges match the documented convention.

**Warning signs:**
- A column named `K` exists in two CSVs with different distributions (e.g., one peaks at 3, another at 5).
- Downstream script runs without error but produces results that contradict published M3 findings.
- Parameter recovery suddenly improves or degrades dramatically with no other code changes.
- `15_analyze_mle_by_trauma.py` shows no group differences where v3.0 showed strong differences.

**Phase to address:** Phase 14 (K research). Lock parameterization decisions before Phase 15 so the hierarchical framework is built against the final K convention.

---

### Pitfall 7: Backend detection in `16b_bayesian_regression.py` produces different priors on PyMC vs NumPyro

**What goes wrong:**
The existing dual-backend script uses `pm.HalfNormal('sigma', sigma=y_sd)` on PyMC and `dist.HalfNormal(y_sd)` on NumPyro. These look identical but PyMC's `sigma=` is the *standard deviation of the underlying Normal*, while NumPyro's positional argument is the *scale parameter* — and HalfNormal's scale and SD differ by `sqrt(1 - 2/π) ≈ 0.6`. Result: the same script produces different priors depending on which backend is detected. Posterior comparisons across local (PyMC) and cluster (NumPyro) runs are not directly comparable.

**Why it happens:**
PyMC and NumPyro have inconsistent parameter naming for the same distributions. `Normal` is fine (both use `loc`, `scale`/`sigma`). `HalfNormal` differs as above. `StudentT` has `nu`/`df` mismatches. `Gamma` has `alpha`/`beta` vs `concentration`/`rate`. The backend-detection pattern is convenient but creates a hidden semantic difference that no test catches.

**How to avoid:**
- **Recommended:** Drop PyMC entirely. NumPyro is faster, GPU-compatible, and the cluster path needs it anyway. Maintaining one backend eliminates this entire class of bug.
- If both backends are required: write a thin wrapper layer that exposes a single API (`my_halfnormal(sd=...)`) and handles the conversion to each backend's convention internally. Unit-test that both backends produce the same prior moments (mean, variance) for the same wrapper call.
- Always compare prior predictive distributions across backends before fitting — if they don't match, the priors don't match.
- Use ArviZ's `from_pymc` and `from_numpyro` to convert to a common format and verify identical parameter names.

**Warning signs:**
- Same script gives different posterior means on PyMC vs NumPyro for the same data.
- The cluster Bayesian regression results differ from local results by more than MCMC noise.
- No assertion in the codebase that PyMC and NumPyro priors are equivalent.

**Phase to address:** Phase 15 (hierarchical framework). Make the backend decision (likely: NumPyro only) before any v4.0 model is wired up. Migrate `16b_bayesian_regression.py` to single-backend at the same time.

---

### Pitfall 8: Long GPU runs (12+ hours) crash with no checkpointing, losing entire fit

**What goes wrong:**
A hierarchical M4 LBA fit with 4 chains × 2000 warmup × 2000 samples × 154 subjects on a single GPU takes 12–24 hours. Mid-run failures from: (a) GPU memory leak in JAX caching, (b) cluster preemption / wall-clock limit, (c) NaN appearing in trial 1.7M of 2.1M, (d) network filesystem hiccup, (e) OOM kill. NumPyro's MCMC doesn't checkpoint by default — a crash means restarting from zero.

**Why it happens:**
NUTS state (mass matrix, step size, momentum, position) is held in JAX device memory. NumPyro's `MCMC.run()` doesn't expose intermediate states. The default sampling loop is monolithic.

**How to avoid:**
- Use `MCMC(..., chain_method='sequential')` and run one chain at a time, saving each chain's samples to disk immediately. A crash loses one chain, not all four.
- Use `numpyro.infer.MCMC.last_state` to checkpoint between warmup and sampling — if sampling crashes, restart sampling from the warmed-up state.
- Split sampling into chunks: `n_samples = 2000` becomes 4 calls of `n_samples = 500` with state passed via `init_params=mcmc.last_state`. Save samples to disk after each chunk.
- Set conservative `max_tree_depth` (default 10) — going to 15 doubles per-iteration cost and increases crash exposure.
- Always submit cluster jobs with `--time` set to 1.5× expected runtime, with explicit `trap` handlers to save partial state on SIGTERM.
- Test the checkpointing path BY KILLING A RUN INTENTIONALLY before relying on it for production fits.

**Warning signs:**
- The fit script writes nothing to disk until completion.
- Re-running the script after a crash starts from iteration 0.
- Cluster job logs show "killed" with no stack trace (= OOM).
- GPU memory fragments visibly grow over a long run (`nvidia-smi` showing rising used memory with stable allocated memory).

**Phase to address:** Phase 13 (GPU LBA). Build the checkpoint-and-resume infrastructure as part of the GPU pipeline, before running any production fit.

---

### Pitfall 9: ArviZ InferenceData migration breaks downstream consumers that expect flat parameter CSV

**What goes wrong:**
Scripts 15/16/17 currently consume `output/wmrl_m3_mle_fits.csv` — one row per participant, one column per parameter. After moving to hierarchical fits, "the parameters" become posterior distributions stored in `arviz.InferenceData` (NetCDF). Downstream scripts that compute trauma group means now need to either (a) collapse posterior to point estimate (loses uncertainty), (b) use full posterior (requires rewriting every regression to propagate), or (c) maintain dual outputs (CSV + NetCDF) which doubles I/O and confusion.

**Why it happens:**
The CSV-based pipeline assumes a single parameter value per subject. Hierarchical fits give a posterior with potentially correlated dimensions. There is no natural projection that preserves all the information.

**How to avoid:**
- Decide upfront: posteriors are the source of truth, point-estimate CSVs are *derived artifacts* with a clear convention (posterior median? posterior mean? MAP?).
- Write a `posterior_to_point_estimate.py` utility that loads InferenceData and emits a CSV with the agreed-upon convention. Document the convention in the column header (e.g., `alpha_pos_postmedian`).
- Update scripts 15/16/17 to take *either* a CSV (back-compat) or an InferenceData path. When given InferenceData, propagate uncertainty correctly (e.g., posterior samples of group means, not point estimates).
- Add a `posterior_summary.py` that produces both: per-subject point estimates AND per-subject 95% CrI widths. Downstream regressions can use the CrI width as a measurement-error model input.
- Migrate one downstream script at a time. Don't change the CSV format and the consumer code in the same PR.

**Warning signs:**
- A downstream script silently uses posterior means as if they were point estimates from MLE, with no propagation of uncertainty.
- Trauma group comparisons show smaller p-values after hierarchical fits than after MLE — likely because uncertainty is being thrown away.
- Two scripts compute "the same" parameter but get different values (one uses median, one uses mean).
- `15_analyze_mle_by_trauma.py` and `15_analyze_bayes_by_trauma.py` co-exist with no shared utility.

**Phase to address:** Phase 18 (integration). The downstream migration is the *last* step, but the contract (point-estimate convention) must be decided in Phase 15.

---

### Pitfall 10: Joint covariate fit with hierarchical RLWM creates label-switching / sign-flip across chains

**What goes wrong:**
When trauma covariates enter at Level-2 as `mu_phi = b0 + b1 * IES_total`, and the prior on `phi` is symmetric, multiple chains can converge to opposite sign conventions for the parameter. Specifically: if two model formulations give the same likelihood (e.g., `phi → 1-phi` symmetry in a degenerate model), one chain finds one mode and another chain finds its mirror. R-hat will flag this — but only if you compute it on the right summary. The signed slope `b1` looks bimodal across chains.

**Why it happens:**
RLWM models have weak identifiability between RL and WM contributions — the same data can be explained by "high RL learning rate, low WM weight" or "low RL learning rate, high WM weight." Adding a covariate at Level-2 doesn't fix this; it just propagates the bimodality into the slope.

**How to avoid:**
- Use **strongly informative priors** on the RL/WM mixing parameters, anchored to published Collins values. This breaks symmetry.
- Initialize all chains from the same starting point (a known-good MLE point estimate), with small Gaussian perturbation. Default NumPyro initialization (independent per chain) is the failure path.
- Inspect rank plots (`arviz.plot_rank`) — these catch label-switching where R-hat doesn't.
- If bimodality persists after informative priors: it's a real model identifiability problem, not a sampler problem. Fix the model, not the prior.

**Warning signs:**
- `arviz.plot_trace()` shows two "stripes" at different levels for one chain vs the others.
- R-hat is fine (~ 1.0) but `arviz.plot_pair()` shows two distinct posterior modes.
- Slopes for trauma covariates flip sign when you change the random seed.
- `b1` posterior is bimodal with both modes negative-of-each-other.

**Phase to address:** Phase 17 (Level-2 structure). Test for identifiability on simulated data with known true slopes BEFORE fitting real data with covariates.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Use centered parameterization for "easy" parameters like `epsilon` | Less code, faster to write | When you add a 7th parameter you'll need to refactor everything anyway | Never — start non-centered |
| Skip prior predictive checks "because we trust the priors" | Save 1 hour per model | Find out 12 hours into a fit that the prior predictive is uniform on [0,1] for a parameter that should peak near 0.5 | Never for production fits |
| Hardcode covariate column names (`'IES_intrusion'`) in the model function | Simpler model signature | Adding a new subscale means editing 6 files | Never — pass as argument |
| Use posterior mean as the point estimate without justifying it | Looks like a number for downstream | Hides multimodality, biased on skewed posteriors | When posterior is verified unimodal and symmetric |
| Reuse v3.0 MLE init points for v4.0 hierarchical fits | Faster warmup | If K parameterization changed, init points are wrong | When parameterization is provably unchanged |
| Run a single chain to "test" the model | Saves 4× compute during dev | R-hat needs ≥ 2 chains; can't detect non-convergence | Only for syntax checks, never for results |
| Skip the `arviz.loo()` call and use AIC instead | One less line of code | Cannot compare to other Bayesian models in literature | Only when explicitly comparing MLE artifacts |
| Combine choice-only and choice+RT models in the same `loo_compare` table | One unified table for the paper | Comparison is mathematically invalid; reviewer will catch it | Never |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| NumPyro + JAX compilation cache | Forgetting to set `JAX_COMPILATION_CACHE_DIR` on the cluster | Set in the SLURM script; verify cache is hit on second run |
| NumPyro `numpyro.factor` | Calling once per participant in a Python loop | Call once with vector of per-subject log-likelihoods |
| ArviZ + NumPyro | Using `from_numpyro` without specifying `coords` and `dims` | Always pass `coords={'participant': ids, 'param': names}` for clean plots |
| MCMC + multi-chain GPU | Using `chain_method='parallel'` on a single GPU and OOMing | Use `chain_method='sequential'` on single GPU; `parallel` only with multiple devices |
| Float64 + JAX | Forgetting to set `jax.config.update('jax_enable_x64', True)` BEFORE any JAX import | Set in `__init__.py` of the fitting module, with assertion |
| LBA + NUTS | Using the MLE log-space CDF as-is (correct) but not testing gradient stability | Run `jax.grad(loglik)` on extreme RTs (0.01s, 5.0s) — verify finite gradients |
| ArviZ + NetCDF | Saving InferenceData with object dtype columns (e.g., participant IDs as strings) | Convert participant IDs to int indices, store mapping separately |
| Cluster GPU + checkpointing | Saving samples only at the end of `mcmc.run()` | Wrap the sampler in chunked calls with intermediate disk writes |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Per-participant Python loop in model function | Compile time grows linearly with N | Vectorize with `vmap` + `plate` | N > 30 subjects |
| Recompilation on every script invocation | "Compiling..." appears every run | Persistent `JAX_COMPILATION_CACHE_DIR` | Always; affects iteration speed |
| Excessive `numpyro.deterministic` calls | InferenceData balloons to GB | Only mark final-scale parameters as deterministic, not intermediates | n_samples × n_subjects > 10^6 |
| `max_tree_depth=15` for "more accurate" sampling | 4× slower per iteration with no benefit | Keep at default 10; investigate model if hitting tree depth | Always — no upside |
| Storing full posterior of all 154 individual parameters in memory | OOM kill mid-LOO | Use `arviz.loo(idata, pointwise=True, var_name='log_lik')` and stream | n_subjects × n_samples > 10^6 |
| Float64 throughout the pipeline | 2× memory, 1.5× slower | Float64 only for LBA (M4); float32 for choice-only models | M4 LBA needs it; others don't |
| Re-loading data inside the model function | Disk I/O at every NUTS step | Load once outside, pass as static argument | Always |
| Loading full InferenceData when you only need one parameter | 30s load time | Use `arviz.from_netcdf(..., group='posterior')` and select var_names | InferenceData > 1 GB |

---

## "Looks Done But Isn't" Checklist

- [ ] **Hierarchical model file:** Often missing non-centered parameterization on at least one parameter — verify the unconstrained-space pattern is used uniformly across `alpha_pos`, `alpha_neg`, `phi`, `rho`, `K`, `epsilon`, `kappa_total`, `kappa_share`.
- [ ] **Simulation-based calibration:** Often missing — verify the model can recover known parameters from simulated data at n=20, n=50, n=154 before fitting real data.
- [ ] **Convergence gate:** Often missing — verify the script REFUSES to write outputs if R-hat > 1.01 on any group parameter or divergences > 0.5%.
- [ ] **Pareto-k report for LBA:** Often missing — verify `loo` is called pointwise and khat distribution is summarized in stdout.
- [ ] **Posterior shrinkage diagnostic:** Often missing — verify the script reports per-parameter shrinkage statistic after fitting, with WARN if < 0.3.
- [ ] **Subscale collinearity check:** Often missing — verify the script computes condition number of the L2 design matrix before fitting and refuses if > 30.
- [ ] **Backend assertion:** Often missing — verify there is exactly one Bayesian backend used by `13_fit_bayesian.py` and `16b_bayesian_regression.py`.
- [ ] **Parameterization version stamp:** Often missing — verify every output CSV/NetCDF includes a `parameterization_version` field that downstream consumers check.
- [ ] **Checkpointing test:** Often missing — verify a fit can be resumed from a partial state (kill it intentionally during a dry run).
- [ ] **Choice-only marginal log-likelihood for M4:** Often missing — verify M4 produces a separate log-likelihood that excludes RT, for apples-to-apples comparison with M3/M5/M6.
- [ ] **Per-model fit validation script:** Often missing — verify each model has a corresponding "did this fit work?" script that checks divergences, R-hat, ESS, posterior shape, and LOO/Pareto-k.
- [ ] **Dispatch sweep test:** Often missing — verify ALL models (M1, M2, M3, M5, M6a, M6b, M4) are wired into the new hierarchical fitting CLI (this is the same surface area that broke M6a in v3.0).

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Centered → divergences | LOW | Refactor to non-centered; refit. ~2 hours per model. |
| Hierarchical shrinkage hiding non-identifiability | HIGH | Need to weaken the model (drop parameters), strengthen the data (more trials), or accept non-identifiability and report it explicitly. Days to weeks. |
| Subscale multicollinearity discovered post-hoc | MEDIUM | Rerun with orthogonalized predictors. Old results invalid; figures need regeneration. ~1 day per model. |
| `lax.scan` not vectorized over participants | MEDIUM | Refactor likelihood; affects all models. ~2 days for the refactor + retesting. |
| LOO Pareto-k > 0.7 for M4 | MEDIUM | Switch to choice-only marginal comparison. Re-derive M4 marginal log-likelihood. ~1 day. |
| K parameterization mismatch in old MLE fits | LOW | Refit MLE for all models with the new convention; downstream scripts already work. ~1 day of compute. |
| PyMC vs NumPyro prior mismatch | LOW | Drop PyMC, refit with NumPyro only. ~half day. |
| Long GPU run crash with no checkpoint | HIGH (lost compute) → LOW (next time) | Add checkpointing (Phase 13), accept the lost run. Lost compute can be 12+ hours. |
| ArviZ migration breaks script 15 | MEDIUM | Add compat layer that emits CSV from posterior; gradual migration. ~3 days per script. |
| Label-switching across chains for Level-2 slopes | MEDIUM | Add identifiability constraint (e.g., positive slope on intrusion); refit. ~half day. |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Centered parameterization funnel | Phase 15 (hierarchical framework) | Unit test recovers parameters from simulated data with non-centered code |
| Hierarchical shrinkage hides non-identifiability | Phase 16 (fitting) | Shrinkage diagnostic gate runs after every fit |
| Multicollinear subscale predictors | Phase 17 (Level-2 structure) | Condition number check; orthogonalization unit test |
| Python-loop likelihoods break NUTS perf | Phase 15 (hierarchical framework) | Compile-time benchmark < 60s for any model |
| LBA Pareto-k > 0.7 in LOO | Phase 18 (model comparison) | `arviz.loo` call with pointwise check; choice-only marginal computed for M4 |
| K parameterization silent invalidation | Phase 14 (K research) | `parameterization_version` column in every fit output |
| PyMC/NumPyro prior mismatch | Phase 15 (hierarchical framework) | Single-backend assertion; remove PyMC path |
| GPU run crash without checkpoint | Phase 13 (GPU LBA) | Kill-and-resume integration test in CI |
| ArviZ migration breaks downstream | Phase 18 (integration) | Compat layer; one script at a time; both old and new outputs match for v3.0 data |
| Label-switching with Level-2 covariates | Phase 17 (Level-2 structure) | Rank plots; identifiability test on simulated data |
| Missing model in hierarchical CLI dispatch | Phase 15 / Phase 16 | Parametric test that loops over `ALL_MODELS` and asserts each can be fit (smoke test) |
| Posterior treated as point estimate downstream | Phase 18 (integration) | Convention documented; column names include `_postmedian` etc. |

---

## Sources

- **Stan / Bayesian best practices:** Betancourt's "A Conceptual Introduction to Hamiltonian Monte Carlo" (2018), and the "Hierarchical Modeling" case study — canonical source for non-centered reparameterization and divergence diagnostics. HIGH confidence.
- **NumPyro documentation:** `numpyro.infer.MCMC` with `chain_method` and `init_params` — verified pattern for chained sampling. HIGH confidence (NumPyro docs).
- **ArviZ documentation:** `arviz.loo`, `arviz.plot_khat`, Pareto-k diagnostic threshold of 0.7 — Vehtari et al. (2017) "Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC." HIGH confidence.
- **Collins lab RLWM papers:** Collins (2018), Collins & Frank (2012), Master et al. (2020) — standard practice for RLWM hierarchical fitting and known parameter identifiability issues. HIGH confidence based on published literature.
- **Senta et al. (2025):** Source of fixed-β=50 convention; β identifiability problem is well-known in the lab. HIGH confidence (project's reference paper).
- **Brown & Heathcote (2008) LBA paper + Annis et al. (2017) LBA Bayesian fitting:** Hierarchical LBA exists but is rare; PSIS-LOO problems with LBA are documented in the rtdists / Bayesian RT modeling literature. MEDIUM confidence — rare combination of NumPyro + LBA in literature.
- **IES-R psychometrics:** Creamer et al. (2003), Beck et al. (2008) — subscale intercorrelations r=0.6–0.8 are well-established. HIGH confidence.
- **Project history:** CLAUDE.md and quick-006 records of MLE NaN argmin and M6a dispatch silent bugs. HIGH confidence (in-project knowledge).
- **JAX compilation cache:** `jax.experimental.compilation_cache` — official JAX docs. HIGH confidence.
- **Pareto smoothed importance sampling:** Vehtari, Simpson, Gelman, Yao, Gabry (2024) — the khat > 0.7 cutoff and recommendations for K-fold fallback. HIGH confidence.
