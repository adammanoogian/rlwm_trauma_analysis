# Pitfalls Research

**Domain:** Computational cognitive modeling — adding M4 (LBA), M5 (RL Forgetting), M6 (Stimulus-Specific Kernel) to an existing JAX/MLE pipeline
**Researched:** 2026-04-02
**Confidence:** HIGH (pitfalls derived from existing codebase patterns + well-established numerical analysis for LBA)

---

## Critical Pitfalls

### Pitfall 1: LBA Density Returns NaN/Inf for Inadmissible Parameter Combinations

**What goes wrong:**
The LBA analytic density is `f(t) = (1/A) * [v*Phi((b - A - v*(t-t0))/s) + s*phi((b - A - v*(t-t0))/s) - v*Phi((b - v*(t-t0))/s) - s*phi((b - v*(t-t0))/s)]`. When `t - t0 <= 0` (RT minus non-decision time is zero or negative), the drift argument becomes undefined. When `A >= b` (start-point variability equals or exceeds threshold), the denominator of the survival function collapses. Either condition produces `NaN` or `Inf` that silently propagates through `jax.lax.scan`, contaminating the entire participant's log-likelihood with `NaN`.

**Why it happens:**
The optimizer explores the full parameter space during L-BFGS-B iteration. On some trajectories, trial-level parameter combinations satisfy `t_i - t0 <= 0` or `A >= b` even when the mean parameters are admissible. JAX's `jnp.where` and `jnp.log` do not raise exceptions — they return `NaN` which accumulates silently.

**How to avoid:**
- Clamp `t_obs - t0` to a minimum of `1e-6` inside the density function before computing normal CDF/PDF arguments, using `jnp.maximum(t_obs - t0, 1e-6)`.
- Enforce `A < b` as a hard constraint in the parameter transform: parameterize as `A = sigmoid(a_raw) * b * 0.99` so `A` is always strictly less than `b` regardless of optimizer trajectory.
- Add a log-barrier term: if `A >= b` or `t0 >= min_rt`, return a large positive penalty (e.g., `1e6`) instead of `NaN` so L-BFGS-B has a finite gradient to follow away from the inadmissible region.
- Fix `s = 0.1` (per McDougle & Collins 2021) — this is not optional for identifiability.

**Warning signs:**
- `fit_mle.py` reports `NaN` log-likelihood for a participant on first evaluation.
- Optimizer converges immediately in 1-2 iterations (NaN gradient terminates L-BFGS-B silently in some `jaxopt` versions).
- All parameter estimates for a participant are at their initial starting values after fitting.

**Phase to address:**
M4 implementation phase — build density function, immediately write a test that evaluates it at 100 random parameter samples including boundary cases (`A = b - 0.001`, `t0 = min_rt - 0.001`, `t0 = min_rt + 0.001`) and assert no NaN/Inf output.

---

### Pitfall 2: Comparing M4 (LBA, Joint Choice+RT) Against M1-M3 (Choice-Only) via AIC/BIC

**What goes wrong:**
AIC and BIC penalize the number of parameters relative to log-likelihood. But M4's likelihood is `log p(choice, RT)` while M1-M3's likelihood is `log p(choice)`. Because RT adds information beyond choice, M4 will always have a higher raw log-likelihood even for identical choice behavior. A lower AIC for M4 does not mean it explains choices better — it explains a richer observable. Comparing these AIC values and concluding "M4 is the best model" is a category error.

**Why it happens:**
`fit_mle.py` outputs AIC/BIC in a single results file and `14_compare_models.py` loads all model fits. Unless gated explicitly, the comparison script will include M4 alongside M1-M3.

**How to avoid:**
- Maintain two separate model comparison tables: one for choice-only models (M1, M2, M3, M5, M6) and one for joint models (M4).
- Add an `observable_type: choice | choice_rt` field to each model's result dict and have `14_compare_models.py` refuse to compare models with different `observable_type` values in the same AIC/BIC ranking.
- To evaluate M4's explanatory value for choices specifically, compare its choice marginal likelihood (sum over RT) against M3, not the joint likelihood.

**Warning signs:**
- M4 ranks first in AIC comparison with a margin that far exceeds what parameter count difference would predict.
- `14_compare_models.py` processes M4 result file without special handling.

**Phase to address:**
Before M4 implementation — document the comparison constraint in `14_compare_models.py` as a code comment and in the results section structure.

---

### Pitfall 3: Non-Decision Time t0 Violates Per-Participant Minimum RT

**What goes wrong:**
`t0` must satisfy `t0 < min(RT)` for every participant. If a participant has an unusually fast response (e.g., 150ms) and `t0` is initialized at 200ms, the optimizer starts at an inadmissible point. If the RT data is not preprocessed to remove implausible fast responses (below ~100ms), `min(RT)` may itself be unreliable (a button-mash artifact, not a true response).

**Why it happens:**
The current pipeline fits choice-only and has no RT preprocessing. `min(RT)` is computed naively from raw data. An outlier RT of 50ms forces `t0 < 50ms`, which may not be psychologically plausible and will constrain `t0` estimates across all reasonable participants.

**How to avoid:**
- Apply a minimum RT cutoff of 100ms before computing `min_rt` per participant for the `t0` upper bound.
- Apply a maximum RT cutoff of 3000ms (or 3 standard deviations above per-participant mean) to remove missed/distracted responses.
- Store `min_rt_clean` per participant and pass it into the LBA fitting function as a hard upper bound for `t0`.
- In the parameter transform for `t0`, use `t0 = sigmoid(t0_raw) * (min_rt_clean - 0.05)` so `t0` is always at least 50ms below the clean minimum RT.

**Warning signs:**
- Wide variation in `t0` estimates across participants that correlates with the fastest RT per participant rather than with any psychological construct.
- Many participants have `t0` estimates near zero even with plausible RT distributions.
- `t0` estimates for some participants are at the upper bound of the parameter space.

**Phase to address:**
RT data preprocessing step — write a dedicated `preprocess_rt()` function and test it on the existing `task_trials_long.csv` before writing LBA likelihood.

---

### Pitfall 4: M5 Decay Applied After Update Instead of Before

**What goes wrong:**
In RL forgetting, the intended update order is: (1) decay all Q-values by `phi_RL` on every trial, then (2) update Q(s,a) for the current stimulus-action pair. If the order is reversed — update first, then decay — the just-updated Q-value immediately loses the newly learned information. This produces systematically lower Q-values and mimics a lower effective learning rate, making `phi_RL` and `alpha_pos` confounded in a different way than intended.

**Why it happens:**
The existing `q_learning_block_likelihood` updates first (line 460-468 in `jax_likelihoods.py`). When adding forgetting, a developer will naturally add decay as an additional line after the update, following the existing code structure.

**How to avoid:**
- In the M5 step function inside `lax.scan`, the structure must be:
  1. `Q_decayed = Q_table * phi_RL` (apply global decay to all s,a)
  2. Compute `log_prob` using `Q_decayed[stimulus]` (not original Q)
  3. `Q_updated = Q_decayed.at[stimulus, action].set(Q_decayed[s,a] + alpha * delta)`
- Add an explicit comment in the code: `# ORDER MATTERS: decay THEN update (Senta et al., 2025)`
- Write a unit test that verifies Q-values decay on non-presented stimuli across trials.

**Warning signs:**
- `phi_RL` and `alpha_pos` are highly correlated (r > 0.8) in parameter recovery.
- Simulated data from M5 cannot distinguish it from M1 with lower `alpha_pos`.
- Parameter recovery for `phi_RL` is poor (r < 0.5) even at extreme values.

**Phase to address:**
M5 implementation phase — write the decay-then-update structure from scratch rather than copying the existing Q-learning step function and patching it.

---

### Pitfall 5: M6 Kernel Uses Global Last Action Instead of Per-Stimulus Last Action

**What goes wrong:**
The stimulus-specific perseveration kernel requires tracking the last action taken for each stimulus independently. If the implementation uses a single `last_action` scalar (as a global perseveration kernel would), trials where stimulus B is presented will repeat the last action from stimulus A's trial. This is incorrect and will produce near-zero `kappa_s` estimates because the spurious repetitions dilute the real stimulus-specific signal.

**Why it happens:**
M3 (global kappa) tracks a single scalar `last_action` in the `lax.scan` carry. When writing M6, copy-paste of the M3 carry structure will carry over this scalar. The fix requires a `last_action_per_stimulus` array of shape `(num_stimuli,)` in the carry.

**How to avoid:**
- Define the M6 carry as `(Q_table, WM_table, last_action_per_stim, log_lik_accum)` where `last_action_per_stim` has shape `(num_stimuli,)` initialized to `-1` (sentinel for "never seen").
- At each trial, update `last_action_per_stim = last_action_per_stim.at[stimulus].set(action)` after computing log_prob but before returning the carry.
- On the first presentation of a stimulus in a block, fall back to uniform kernel weight (probability 1/3 for each action) by checking `last_action_per_stim[stimulus] == -1`.

**Warning signs:**
- `kappa_s` estimates cluster near zero for all participants.
- Simulated M6 data cannot be recovered (recovery r < 0.5 for `kappa_s`).
- M6 does not outperform M3 in model comparison despite added parameters.

**Phase to address:**
M6 implementation phase — write carry structure explicitly before coding the step function, and include a unit test that verifies `last_action_per_stim` updates correctly for multi-stimulus sequences.

---

### Pitfall 6: phi_RL and alpha_pos/alpha_neg Are Unidentifiable at Extreme Values

**What goes wrong:**
`phi_RL` (global Q-value decay) and `alpha_pos`/`alpha_neg` (learning rates) are functionally opposed: learning rates build Q-values toward rewards while forgetting decays them back toward zero. At high learning rates (alpha > 0.7), Q-values are updated so strongly each trial that decay has minimal effect. At low learning rates (alpha < 0.1), Q-values change slowly whether or not decay is present. Only in the intermediate range are the two parameters separately identifiable.

**Why it happens:**
Collinearity in the generative model. The joint likelihood surface has a ridge in the `(alpha, phi_RL)` plane where many combinations produce similar fits.

**How to avoid:**
- Run parameter recovery specifically for M5 before fitting real data. Require `r >= 0.80` for all parameters including `phi_RL` before proceeding.
- In the Hessian diagnostics (already implemented in `mle_utils.py`), flag participants where the `(alpha_pos, phi_RL)` correlation exceeds 0.7 as poorly identified.
- Consider fixing `phi_RL` to a value from the literature (e.g., 0.9) and fitting only as a supplementary analysis, reporting it separately from the main M3 comparison.

**Warning signs:**
- `compute_hessian_diagnostics()` reports high condition number (> 100) for M5 fits.
- Parameter recovery for M5 shows `phi_RL` recovery r < 0.80 across the full parameter range.
- `get_high_correlations()` flags `alpha_pos` vs `phi_RL` pair.

**Phase to address:**
M5 parameter recovery phase — run recovery before any real-data fitting and decide whether M5 is sufficiently identified to warrant inclusion.

---

### Pitfall 7: M6b kappa + kappa_s Constraint Not Enforced in Parameter Transform

**What goes wrong:**
M6b is a variant where both global perseveration (`kappa`) and stimulus-specific perseveration (`kappa_s`) are present simultaneously. These are mixture weights and must satisfy `kappa + kappa_s <= 1`. Without an explicit constraint in the parameter transform, the optimizer can find `kappa = 0.6, kappa_s = 0.7` (sum > 1), producing negative probability mass for other action components and causing `log(negative_number) = NaN`.

**Why it happens:**
Standard `logit` transforms enforce each parameter in `[0, 1]` independently but do not enforce joint simplex constraints. The existing M3 parameter transform does not need this because there is only one kappa.

**How to avoid:**
- Use a Dirichlet/stick-breaking parameterization: `kappa = sigmoid(k1_raw)`, `kappa_s = sigmoid(k2_raw) * (1 - kappa)`. This guarantees `kappa + kappa_s <= 1` by construction.
- Alternatively, parameterize as a softmax over three components: `[kappa, kappa_s, 1 - kappa - kappa_s] = softmax([k1_raw, k2_raw, 0])`.
- Add an assertion in the likelihood function: `assert kappa + kappa_s <= 1.0 + 1e-6` (with a debug flag that can be disabled for production).

**Warning signs:**
- `NaN` log-likelihood for participants with high perseveration.
- `kappa + kappa_s > 1` in raw fitted results for some participants.
- Likelihood function returns very negative values (e.g., -1e10) for certain parameter starting points during multi-start optimization.

**Phase to address:**
M6 implementation phase — define the parameter transform for `(kappa, kappa_s)` before writing the likelihood function body.

---

### Pitfall 8: New Models Break Existing Scripts 15 and 16

**What goes wrong:**
Scripts `15_analyze_mle_by_trauma.py` and `16_regress_parameters_on_scales.py` load fitted parameter CSVs and loop over model-specific parameter lists. If M4/M5/M6 results files use different column naming conventions (e.g., `v_scale` vs `drift_scale`), or if the `--model all` flag does not include new models, the scripts either silently skip new models or crash on unexpected columns.

**Why it happens:**
Scripts 15 and 16 were written when only M1-M3 existed. Their model dispatch dictionaries (likely a `if model == 'wmrl_m3':` chain) do not have entries for M4/M5/M6. A developer adding M4 may test `12_fit_mle.py` and `14_compare_models.py` but forget to update downstream analysis scripts.

**How to avoid:**
- Before implementing M4/M5/M6 likelihoods, audit scripts 15 and 16 and add placeholder entries for the new models to their model dispatch tables. This makes the integration requirement explicit and prevents silent omission.
- Define `LBA_PARAMS`, `WMRL_M5_PARAMS`, `WMRL_M6_PARAMS` constant lists in `mle_utils.py` at the same time as defining `LBA_BOUNDS`, etc. Scripts 15 and 16 should import these lists rather than hardcoding parameter names.
- After fitting M4 for the first time, run `python scripts/15_analyze_mle_by_trauma.py --model m4` as a smoke test immediately.

**Warning signs:**
- `python scripts/15_analyze_mle_by_trauma.py --model all` does not produce output for M4/M5/M6.
- KeyError or AttributeError in scripts 15/16 when new model CSV is loaded.
- Script 16 regression output is missing `v_scale`, `phi_rl`, or `kappa_s` columns.

**Phase to address:**
Each new model's integration phase — treat script 15/16 smoke test as a required acceptance criterion before marking any model phase complete.

---

### Pitfall 9: kappa_s and WM Both Produce Stimulus-Specific Repetition — Model Misidentification Risk

**What goes wrong:**
In the M6 model, `kappa_s` captures tendency to repeat the last action for a specific stimulus. The WM component (phi, rho, K) also produces stimulus-specific action tendencies because WM stores exact stimulus-action-reward associations. A participant with strong WM and moderate WM decay will look similar to a participant with weak WM and high `kappa_s`, because both produce consistent stimulus-specific responding. The two mechanisms are confounded at the behavioral level.

**Why it happens:**
Both WM and stimulus-specific kernel are indexed by stimulus identity. Their contributions to action probability both peak after a correct response to a specific stimulus and decay over time (WM via phi, kernel via block resets). The generative model is not jointly identified for all parameter combinations.

**How to avoid:**
- Run model recovery: fit M6 data with M3 and M3 data with M6. If M6 wins for M3-generated data, or if M3 wins for M6-generated data, the comparison is unreliable.
- Interpret `kappa_s` as an "unexplained stimulus-specific repetition beyond WM" rather than a standalone mechanism.
- Consider constraining M6 by fixing WM parameters at M3-fitted values and fitting only `kappa_s` as an additive term, rather than jointly fitting all parameters.

**Warning signs:**
- Model recovery shows M3 and M6 cross-recover (each fits the other's data better than the true model).
- `kappa_s` and WM capacity `K` are negatively correlated across participants.
- BIC advantage for M6 over M3 is less than 6 points (not decisive evidence).

**Phase to address:**
M6 validation phase — run model recovery as part of acceptance criteria before reporting M6 results.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Copy existing Q-learning step function and add forgetting as a postfix line | Fast M5 implementation | Decay-after-update bug (Pitfall 4); incorrect science | Never |
| Use global `last_action` scalar for M6 kernel (copied from M3) | Code reuse | Wrong kernel implementation; near-zero kappa_s estimates | Never |
| Include M4 in the same AIC table as M1-M3 | Single comparison table | Invalid model comparison; wrong model selection conclusions | Never |
| Skip per-participant RT cleaning and use raw `min(RT)` for t0 bound | Simpler preprocessing | t0 constrained by outlier RTs; biased estimates | Never |
| Hardcode new parameter names in scripts 15/16 | Faster initial integration | Scripts break when parameter names change; duplicate constants | Acceptable only for first-pass smoke test; must refactor before analysis |
| Skip model recovery for M5 if parameter recovery r > 0.70 | Time savings | Cannot trust M5 > M3 model comparison | Never — 0.80 threshold is firm |

---

## Integration Gotchas

Common mistakes when connecting new models to the existing pipeline.

| Integration Point | Common Mistake | Correct Approach |
|-------------------|----------------|------------------|
| `fit_mle.py` `--model` flag | Add `elif model == 'm4':` without updating warmup function | Add M4 to `warmup_jax_compilation()` and model dispatch simultaneously |
| `mle_utils.py` parameter lists | Define `LBA_PARAMS` list but forget to define `LBA_BOUNDS` dict | Define bounds and params lists together; both are required for transform functions |
| `14_compare_models.py` | Load M4 result CSV alongside M3 result CSV and compare AIC | Gate on `observable_type` field; M4 goes in separate comparison block |
| Block padding (`pad_block_to_max`) | M4 also needs RT array padded — forget to add `rts` to the padding call | Extend `pad_block_to_max` signature to accept optional `rts` argument at the same time as adding LBA |
| `MAX_BLOCKS` and `MAX_TRIALS_PER_BLOCK` constants | Assume LBA uses the same compiled shapes as choice models | LBA uses the same block structure but may require verifying `MAX_TRIALS_PER_BLOCK = 100` is still sufficient after RT filtering |
| JAX compilation cache | New model adds new scan structure; cached kernels for old models become stale | JAX compilation cache is keyed by function signature — adding new functions does not invalidate old cache entries; safe to add |

---

## Performance Traps

Patterns that work on small samples but fail at scale.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Computing `jnp.log(normal_cdf(...))` directly for LBA survivor function | NaN in tails (CDF near 0 or 1) | Use `jax.scipy.stats.norm.logcdf` and `jax.scipy.stats.norm.logsf` (log-space CDF/SF) instead of `jnp.log(CDF)` | Any time drift argument > 5 or < -5 in standard deviations |
| Materializing per-trial RT arrays inside `lax.scan` for LBA | Excessive memory allocation with large participant counts in parallel fitting | Keep all per-trial computations inside scan; do not return intermediate RT arrays as scan outputs unless needed for debugging | ~50+ participants in parallel on GPU |
| Multi-start optimization with 50 starts for LBA (same as M1-M3) | Fitting time 4-5x longer per participant due to LBA density cost | Use 25 starts for M4 on first pass; increase to 50 only if recovery validates at 25 | Any production fitting run |

---

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **M4 LBA likelihood:** Tests with admissible parameters pass, but boundary tests (`A = b - epsilon`, `t0 = min_rt - epsilon`, `A = 0`) have not been run — verify no NaN at boundaries before fitting real data.
- [ ] **M5 decay order:** Step function produces correct Q-values in unit test with 2-3 trials on 2 stimuli — verify that Q-values for the non-presented stimulus decay while Q-value for presented stimulus reflects the update, not just the decay.
- [ ] **M6 per-stimulus kernel:** `last_action_per_stim` array is in carry — verify that presenting stimulus 0, then stimulus 1, then stimulus 0 again uses the action from the first stimulus 0 presentation, not the stimulus 1 presentation.
- [ ] **M4 RT preprocessing:** `preprocess_rt()` function exists — verify it stores `min_rt_clean` per participant in the result dict and passes it to the LBA fitting function as a bound.
- [ ] **Pipeline scripts 15/16:** New model parameter names are in the dispatch tables — verify `--model all` produces output rows for M4, M5, M6.
- [ ] **Model comparison gating:** M4 AIC is computed — verify it does NOT appear in the same ranking table as M1-M3 in `14_compare_models.py` output.
- [ ] **M6b kappa + kappa_s:** Parameter transform uses stick-breaking — verify `kappa + kappa_s <= 1.0` for 1000 random draws from unconstrained space.

---

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| LBA NaN propagation discovered mid-fitting run | MEDIUM | Add `jnp.where(jnp.isnan(log_lik), 1e6, log_lik)` as guard at end of likelihood; re-run affected participants; diagnose which parameter caused NaN |
| M4 included in AIC comparison and results reported | HIGH | Re-run `14_compare_models.py` with M4 excluded; produce corrected comparison table; note correction in results |
| M5 decay order inverted and fits already run | HIGH | Re-implement step function, re-run full M5 fitting, re-run scripts 15/16; prior M5 results are scientifically invalid and cannot be corrected post-hoc |
| M6 using global last_action discovered after fitting | HIGH | Same as M5 decay order — results are invalid; re-implement and re-run |
| kappa + kappa_s > 1 in some fits | MEDIUM | Post-hoc clip to simplex boundary; flag affected participants; note in methods that a small fraction hit the constraint |
| Scripts 15/16 missing M4/M5/M6 output | LOW | Add model to dispatch table; re-run scripts 15/16 (no re-fitting needed) |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| LBA NaN/Inf at inadmissible parameters | M4 likelihood implementation | Unit tests: 100 random samples including boundary cases, assert `jnp.all(jnp.isfinite(log_lik))` |
| M4 vs M1-M3 invalid AIC comparison | Before M4 implementation (design phase) | `14_compare_models.py` output: M4 appears in separate table section |
| t0 violating per-participant min RT | RT preprocessing step (M4 phase) | `preprocess_rt()` test: verify `min_rt_clean` excludes sub-100ms responses |
| M5 decay-before-update order | M5 likelihood implementation | Unit test: 2-stimulus, 3-trial sequence verifying Q decay on non-presented stimulus |
| M6 global vs per-stimulus last action | M6 likelihood implementation | Unit test: alternate-stimulus sequence verifying `last_action_per_stim` updates correctly |
| phi_RL / alpha_pos collinearity | M5 parameter recovery phase | Recovery r >= 0.80 for phi_RL across full parameter range before real-data fitting |
| kappa + kappa_s > 1 | M6 parameter transform design | Assert `kappa + kappa_s <= 1.0` for 1000 unconstrained draws |
| Scripts 15/16 not updated | Each model's integration phase | Smoke test: `python scripts/15_analyze_mle_by_trauma.py --model [new_model]` runs without error |
| kappa_s / WM confound | M6 validation phase | Model recovery: M6-generated data fits M6 better than M3 with BIC delta > 6 |

---

## Sources

- Existing codebase: `scripts/fitting/jax_likelihoods.py` — established patterns for scan-based likelihood and parameter transform conventions
- Existing codebase: `scripts/fitting/mle_utils.py` — parameter bounds, transform functions, Hessian diagnostics
- Existing codebase: `scripts/fitting/fit_mle.py` — model dispatch, warmup, parallel fitting patterns
- McDougle & Collins (2021): LBA formulation for cognitive modeling; `s = 0.1` identifiability constraint
- Senta et al. (2025): WM-RL model formulation; fixed beta = 50 rationale
- Standard numerical analysis: log-space CDF/SF for heavy-tail stability (`jax.scipy.stats.norm.logcdf`)
- JAX documentation: `jax.lax.scan` carry semantics — all carry elements are materialized at every step

---
*Pitfalls research for: RLWM Trauma Analysis — M4/M5/M6 model extensions*
*Researched: 2026-04-02*
