# Phase 16: Choice-Only Family Extension + Subscale Level-2 — Research

**Researched:** 2026-04-12
**Domain:** NumPyro hierarchical Bayesian inference, JAX likelihoods, Level-2 regression, collinearity audit, permutation null test, ArviZ LOO comparison
**Confidence:** HIGH (all findings from direct codebase inspection of jax_likelihoods.py, numpyro_models.py, numpyro_helpers.py, mle_utils.py, fit_bayesian.py, bayesian_diagnostics.py, bayesian_summary_writer.py, cluster/13_bayesian_m3.slurm, output/summary_participant_metrics.csv)

---

## 1. Stacked Likelihood Function Names (HIER-02 through HIER-06)

All five stacked likelihood functions exist and are fully implemented in
`scripts/fitting/jax_likelihoods.py`. They support both `return_pointwise=False`
(scalar for MCMC `numpyro.factor`) and `return_pointwise=True` (for WAIC/LOO
via `bayesian_diagnostics.py`).

### Q-learning (M1) — HIER-02
```python
q_learning_multiblock_likelihood_stacked(
    stimuli_stacked, actions_stacked, rewards_stacked, masks_stacked,
    alpha_pos, alpha_neg, epsilon,
    ..., *, return_pointwise=False
)
```
Note: Q-learning's stacked signature LACKS `set_sizes_stacked`. This is the
only model without the set-size argument. `bayesian_diagnostics._build_per_participant_fn`
already has a special case that does NOT pass `set_sizes_stacked` for qlearning.
`prepare_stacked_participant_data` always produces `set_sizes_stacked`; the M1
NumPyro model must simply not forward it to the likelihood.

### WM-RL (M2) — HIER-03
```python
wmrl_multiblock_likelihood_stacked(
    stimuli_stacked, actions_stacked, rewards_stacked,
    set_sizes_stacked, masks_stacked,
    alpha_pos, alpha_neg, phi, rho, capacity, epsilon,
    ..., *, return_pointwise=False
)
```
The legacy `wmrl_hierarchical_model` in numpyro_models.py uses the OLD convention
(Beta group priors, `mu_capacity ~ TruncatedNormal(4, 1.5, low=1, high=7)`, unstacked
list-of-blocks data format). HIER-03 requires porting to `sample_bounded_param` + stacked
format. The function must be written fresh; the legacy function must not be modified.

### M5 (WM-RL+phi_rl) — HIER-04
```python
wmrl_m5_multiblock_likelihood_stacked(
    stimuli_stacked, actions_stacked, rewards_stacked,
    set_sizes_stacked, masks_stacked,
    alpha_pos, alpha_neg, phi, rho, capacity, kappa, phi_rl, epsilon,
    ..., *, return_pointwise=False
)
```
phi_rl bounds: `(0.001, 0.999)` (from `WMRL_M5_BOUNDS`). `PARAM_PRIOR_DEFAULTS["phi_rl"]`
is already defined: `{"lower": 0.0, "upper": 1.0, "mu_prior_loc": -0.8}`. M5 adds a
second WM-style decay parameter that operates on RL Q-values (decay toward Q0=1/nA before
applying the delta-rule). Mechanically: 8 parameters (M3's 7 + phi_rl). All 8 are in
`PARAM_PRIOR_DEFAULTS`; all 8 can be sampled via `sample_bounded_param`.

### M6a (WM-RL+kappa_s) — HIER-05
```python
wmrl_m6a_multiblock_likelihood_stacked(
    stimuli_stacked, actions_stacked, rewards_stacked,
    set_sizes_stacked, masks_stacked,
    alpha_pos, alpha_neg, phi, rho, capacity, kappa_s, epsilon,
    ..., *, return_pointwise=False
)
```
`kappa_s` replaces global `kappa` with a per-stimulus perseveration parameter.
`PARAM_PRIOR_DEFAULTS["kappa_s"]`: `{"lower": 0.0, "upper": 1.0, "mu_prior_loc": -2.0}`.
Mechanically identical parameter count to M3 (7 parameters). Same
`sample_bounded_param` pattern as M3's kappa, but name is `kappa_s`. The L2 regression
on the winning model (M6b) applies to `kappa_total` and `kappa_share`, not `kappa_s`,
so M6a's L2 extension (if any) would apply to `kappa_s`.

### M6b (WM-RL+dual) — HIER-06
```python
wmrl_m6b_multiblock_likelihood_stacked(
    stimuli_stacked, actions_stacked, rewards_stacked,
    set_sizes_stacked, masks_stacked,
    alpha_pos, alpha_neg, phi, rho, capacity,
    kappa,    # DECODED = kappa_total * kappa_share
    kappa_s,  # DECODED = kappa_total * (1 - kappa_share)
    epsilon,
    ..., *, return_pointwise=False
)
```
**Critical:** The likelihood expects already-decoded `kappa` and `kappa_s`. The NumPyro
model samples `kappa_total` and `kappa_share` in [0,1], then decodes before calling the
likelihood:
```python
kappa   = kappa_total_i * kappa_share_i
kappa_s = kappa_total_i * (1.0 - kappa_share_i)
```
`PARAM_PRIOR_DEFAULTS["kappa_total"]`: `{"lower": 0.0, "upper": 1.0, "mu_prior_loc": -2.0}`.
`PARAM_PRIOR_DEFAULTS["kappa_share"]`: `{"lower": 0.0, "upper": 1.0, "mu_prior_loc": 0.0}`.
Both are sampled via `sample_bounded_param`. The stick-breaking constraint
(kappa + kappa_s <= kappa_total <= 1) is guaranteed by construction; the decode happens
inside the per-participant loop before passing to the likelihood.

---

## 2. Legacy Model Audit — What Needs Rewriting vs. Preserving

The existing `numpyro_models.py` contains:
1. `qlearning_hierarchical_model` — OLD convention (Beta/logit group priors,
   `numpyro.plate("participants", ...)`, passes list-of-block dicts). Must NOT be
   modified; HIER-02 writes a NEW `qlearning_hierarchical_model_v2` or more likely a
   `qlearning_hierarchical_model_stacked` function.
2. `wmrl_hierarchical_model` — OLD convention (Beta/logit, TruncatedNormal K in [1,7],
   unstacked data). Must NOT be modified; HIER-03 writes new `wmrl_hierarchical_model_stacked`.
3. `wmrl_m3_hierarchical_model` — NEW convention (hBayesDM, K in [2,6], stacked data,
   L2 LEC). This is the Phase 15 template. Already complete per STATE.md.

The `BAYESIAN_IMPLEMENTED` set in `fit_bayesian.py` is currently `{'qlearning', 'wmrl', 'wmrl_m3'}`.
Adding each new model requires:
(a) Writing the hierarchical model function in `numpyro_models.py`
(b) Importing it in `fit_bayesian.py`
(c) Extending `BAYESIAN_IMPLEMENTED` and the dispatch logic in `fit_model()`
(d) Extending `save_results()` to pass `participant_data_stacked` for the new model
(e) Adding a SLURM script by copying and adjusting `13_bayesian_m3.slurm`

---

## 3. M3 Template Pattern — Exact Recipe to Replicate

The `wmrl_m3_hierarchical_model` in `numpyro_models.py` (lines 967–1103) is the canonical
template. Every new model should follow this exact structure:

```
1. n_participants = len(participant_data_stacked)
2. participant_ids = sorted(participant_data_stacked.keys())
3. Sample L2 regression coefficients (one per covariate, only if covariate is not None)
4. For each param in MODEL_REGISTRY[model_name]["params"] except those needing L2 shifts:
       sampled[param] = sample_bounded_param(param, ...)   # uses PARAM_PRIOR_DEFAULTS
   For each param needing L2 shifts (e.g., kappa or kappa_total):
       sample mu_pr, sigma_pr, z manually
       kappa_unc = mu_pr + sigma_pr * z + sum(beta_j * covariate_j)
       kappa = lower + (upper - lower) * phi_approx(kappa_unc)
5. Python for-loop over participant_ids (NOT vmap):
       pdata = participant_data_stacked[pid]
       [decode parameters if needed, e.g. M6b stick-breaking]
       log_lik = <model>_multiblock_likelihood_stacked(pdata, **params, return_pointwise=False)
       numpyro.factor(f"obs_p{pid}", log_lik)
```

Key invariants (all locked in STATE.md):
- `sample_bounded_param` uses `.expand([n_participants])`, not `numpyro.plate`
- `sorted(participant_data_stacked.keys())` for participant ordering
- `phi_approx = jax.scipy.stats.norm.cdf` (from `numpyro_helpers.py`)
- `run_inference_with_bump` with `target_accept_probs=(0.80, 0.95, 0.99)`
- Convergence gate: `max_rhat < 1.01 AND min_ess_bulk > 400 AND num_divergences == 0`

---

## 4. M6b Non-Centered Dual Stick-Breaking Parameterization (HIER-06)

M6b is the only model with two constrained parameters that are linked: `kappa_total`
and `kappa_share` both in [0,1], and their product must not exceed 1 (guaranteed by
construction since `kappa = kappa_total * kappa_share <= kappa_total <= 1`).

For hierarchical inference, BOTH parameters are sampled independently via
`sample_bounded_param` with their respective `PARAM_PRIOR_DEFAULTS`. No special
joint parameterization is needed at the group level because the stick-breaking
constraint is automatically satisfied by the bounded [0,1] range.

The decode happens per-participant inside the for-loop (not in the likelihood itself):
```python
for idx, pid in enumerate(participant_ids):
    kappa_total_i = sampled["kappa_total"][idx]
    kappa_share_i = sampled["kappa_share"][idx]
    kappa   = kappa_total_i * kappa_share_i
    kappa_s = kappa_total_i * (1.0 - kappa_share_i)
    log_lik = wmrl_m6b_multiblock_likelihood_stacked(
        ..., kappa=kappa, kappa_s=kappa_s, ...
    )
    numpyro.factor(f"obs_p{pid}", log_lik)
```

For L2 regression on M6b, the subscale covariates shift `kappa_total` and/or
`kappa_share` on the probit scale (same pattern as M3's `beta_lec_kappa` shifting `kappa`).
The Phase 16 design calls for ~48 coefficients = 8 parameters x 6 predictors. The L2
regression applies only to the "scientifically interesting" parameters. The most likely
implementation: extend the M3 pattern by sampling a coefficient matrix
`beta[n_params_with_L2, n_covariates]` or separate scalar sites per (param, covariate).

The simplest correct implementation uses separate named scalar sites:
```python
beta_lec_kappa_total     = numpyro.sample("beta_lec_kappa_total",     dist.Normal(0, 1))
beta_iesr_kappa_total    = numpyro.sample("beta_iesr_kappa_total",    dist.Normal(0, 1))
# ... etc for each (param, covariate) pair
```
With 8 parameters and 6 covariates this is 48 scalar sites. Alternatively, sample a
matrix `beta_L2 ~ Normal(0,1).expand([8, 6])` and index it. The scalar-site approach
is more readable and ArviZ-compatible; the matrix approach is more compact.

---

## 5. IES-R Subscale Columns and LEC-5 Subcategory Columns

Confirmed from `output/summary_participant_metrics.csv` header (column indices):

| Column | Index | Description |
|--------|-------|-------------|
| `ies_total` | 10 | IES-R total score |
| `ies_intrusion` | 11 | IES-R intrusion subscale |
| `ies_avoidance` | 12 | IES-R avoidance subscale |
| `ies_hyperarousal` | 13 | IES-R hyperarousal subscale |
| `less_total_events` | 14 | LEC-5 total events (confirmed covariate column) |
| `less_personal_events` | 15 | LEC-5 personally experienced events |

**LEC-5 physical/sexual/accident subcategories are NOT currently present in
`summary_participant_metrics.csv`.** The columns `less_total_events` and
`less_personal_events` are the only LEC columns. Success Criterion 3 references
"LEC-5 physical/sexual/accident subcategories" — these must be computed and added to
the metrics CSV before Phase 16 L2 fits can proceed. This is a data-pipeline blocker
that must appear in the Phase 16 plan as task 0 (data audit).

The 6 predictors in the ~48-coefficient design are:
1. `less_total_events` (LEC-5 total)
2. `ies_total` (IES-R total, used as the base term before residualization)
3. IES-R intrusion — Gram-Schmidt residualized against `ies_total`
4. IES-R avoidance — Gram-Schmidt residualized against `ies_total`
5. IES-R hyperarousal — Gram-Schmidt residualized against `ies_total`
6. One of: LEC-5 physical/sexual/accident subcategory (MISSING from current data)

The Gram-Schmidt residualization of subscales against IES-R total is standard OLS:
```python
# Residualize ies_intrusion against ies_total
proj = (ies_intrusion @ ies_total) / (ies_total @ ies_total) * ies_total
ies_intrusion_resid = ies_intrusion - proj
```
This ensures near-zero correlation with the total score by construction. Post-
residualization condition number check with `np.linalg.cond()` confirms acceptability
(target: condition number < 30 on the [intrusion_resid, avoidance_resid, hyperarousal_resid]
design submatrix).

---

## 6. Collinearity Audit — IES-R Subscale (L2-02, L2-03)

**Pattern:** Standard numpy-only operation. No new dependencies.

```python
import numpy as np

# Load N=154 survey data
metrics = pd.read_csv("output/summary_participant_metrics.csv")
X_raw = metrics[["ies_intrusion", "ies_avoidance", "ies_hyperarousal"]].values
cond_raw = np.linalg.cond(X_raw)

# Gram-Schmidt orthogonalization against ies_total
ies_total = metrics["ies_total"].values.astype(float)
X_resid = np.column_stack([
    x - np.dot(x, ies_total) / np.dot(ies_total, ies_total) * ies_total
    for x in X_raw.T
])
cond_resid = np.linalg.cond(X_resid)

# Report to output/bayesian/level2/ies_r_collinearity_audit.md
```

The audit should report: raw correlation matrix, raw condition number, residualized
condition number, and flag if condition number >= 30 post-residualization (requiring
strategy revision).

The orthogonalized regressor set must be checked into `scripts/fitting/level2_design.py`
as a function `build_level2_design_matrix(metrics_df, participant_ids)` returning a
standardized design matrix of shape `(n_participants, n_covariates)`.

---

## 7. Permutation Null Test Infrastructure (L2-06)

**Goal:** 50 shuffles of the trauma label rows refit under M3 + LEC-total; verify
<5% of shuffles produce a "surviving" L2 effect (posterior CI excluding zero).

**Pattern:** The permutation test does NOT require 50 full MCMC runs with the
complete N=154 dataset. The practical approach for 50 shuffles is:
1. For each shuffle: permute the `covariate_lec` array (same size, different alignment
   with participant behavioral data)
2. Run `run_inference_with_bump` on M3 with the shuffled covariate
3. Check whether the 95% HDI of `beta_lec_kappa` excludes zero
4. Count how many shuffles produce HDI-excludes-zero ("surviving")
5. Report: surviving / 50 should be <= 2-3 (<=5% nominal alpha)

This is a compute-intensive task: 50 x full MCMC runs. Each M3 run takes ~2-4 hours on
the cluster (4 chains, 1000 warmup, 2000 samples). At 4 hours each, 50 runs = 200 CPU-hours.
The practical approach is to run fewer iterations for the permutation test (e.g., 500
warmup, 1000 samples) since the null hypothesis requires only a rough HDI-excludes-zero
check, not high-precision posteriors.

The implementation is a loop:
```python
for shuffle_idx in range(n_shuffles):
    rng = np.random.default_rng(shuffle_idx)
    lec_shuffled = rng.permutation(covariate_lec)
    model_args = {"participant_data_stacked": pdata, "covariate_lec": lec_shuffled}
    mcmc_perm = run_inference_with_bump(
        wmrl_m3_hierarchical_model, model_args,
        num_warmup=500, num_samples=1000, num_chains=4
    )
    samples = mcmc_perm.get_samples()
    hdi = az.hdi(np.array(samples["beta_lec_kappa"]), hdi_prob=0.95)
    surviving = hdi[0] > 0 or hdi[1] < 0
    results.append({"shuffle": shuffle_idx, "hdi_low": hdi[0], "hdi_high": hdi[1],
                    "surviving": surviving})
```

SLURM approach: submit as a SLURM array job (50 tasks), each running a single shuffle.
This parallelizes 200 CPU-hours to wall-clock time of ~4-6 hours.

---

## 8. Forest Plot Infrastructure (L2-07, scripts/18_bayesian_level2_effects.py)

**Goal:** Forest plots for the M6b Level-2 regression posterior (~48 coefficients).

**Library pattern:** ArviZ has `az.plot_forest()` which accepts an InferenceData object
and a list of variable names. For 48 named scalar sites (e.g., `beta_lec_alpha_pos`,
`beta_lec_alpha_neg`, ...) this works natively:

```python
import arviz as az
import matplotlib.pyplot as plt

az.plot_forest(
    idata,
    var_names=[v for v in idata.posterior if v.startswith("beta_")],
    combined=True,
    hdi_prob=0.95,
    figsize=(10, 24),
)
plt.tight_layout()
plt.savefig("output/bayesian/figures/m6b_forest_lec5.png", dpi=150)
```

For two separate plots (LEC-5 coefficients vs IES-R residualized subscale coefficients),
filter `var_names` by naming convention (e.g., `beta_lec_*` vs `beta_iesr_*`).

**Naming convention recommendation:** Use a consistent pattern for coefficient sites:
`beta_{covariate_short}_{param_name}` where `covariate_short` is one of:
`lec_total`, `iesr_total`, `iesr_intr_resid`, `iesr_avd_resid`, `iesr_hyp_resid`,
`lec_phys` (or similar for LEC-5 subcategories).

**New script `scripts/18_bayesian_level2_effects.py`** must:
1. Load `output/bayesian/wmrl_m6b_posterior.nc` via `az.from_netcdf()`
2. Filter beta_ coefficient sites
3. Call `az.plot_forest()` for each covariate group
4. Save two PNGs to `output/bayesian/figures/`

No new dependencies: matplotlib and arviz are already in the stack.

---

## 9. az.compare for Stacking Weights (Success Criterion 6)

ArviZ `az.compare()` accepts a dict of InferenceData objects and returns a
comparison table. For stacking weights:

```python
import arviz as az

compare_dict = {
    "M1": idata_qlearning,
    "M2": idata_wmrl,
    "M3": idata_wmrl_m3,
    "M5": idata_wmrl_m5,
    "M6a": idata_wmrl_m6a,
    "M6b": idata_wmrl_m6b,
}
comparison = az.compare(compare_dict, ic="loo", method="stacking")
print(comparison)
```

Each InferenceData must have a `log_likelihood` group (populated by
`build_inference_data_with_loglik`). The comparison returns a DataFrame with columns
including `weight` (stacking weights), `loo`, `p_loo`, `d_loo`, `warning`, `scale`.

**Integration in `scripts/14_compare_models.py`:** The `--bayesian-comparison` flag
must:
1. Load each model's NetCDF from `output/bayesian/{model}_posterior.nc`
2. Verify each has a `log_likelihood` group (fail gracefully if missing)
3. Call `az.compare()` and write output to `output/bayesian/level2/stacking_weights.md`
4. Flag as "inconclusive" if M6b weight < 0.5

The condition number check for Pareto-k > 0.7 (which would indicate LOO instability)
should be logged but not block the stacking output.

---

## 10. fit_bayesian.py Extension Pattern

Current `BAYESIAN_IMPLEMENTED = {'qlearning', 'wmrl', 'wmrl_m3'}`.

The extension pattern for each new model in `fit_model()`:
```python
BAYESIAN_IMPLEMENTED = {'qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b', 'qlearning_stacked', 'wmrl_stacked'}
# ...
if model == 'wmrl_m5':
    # identical to wmrl_m3 block, swap model function + group params list
    participant_data_stacked = prepare_stacked_participant_data(data, ...)
    covariate_lec = _load_lec_covariate(participant_data_stacked)  # helper to extract
    model_args = {"participant_data_stacked": participant_data_stacked, "covariate_lec": covariate_lec}
    mcmc = run_inference_with_bump(wmrl_m5_hierarchical_model, model_args, ...)
    return mcmc, participant_data_stacked
```

The `save_results()` function's dispatch on `model == 'wmrl_m3'` must be broadened to
`model in NEW_CANONICAL_MODELS` where `NEW_CANONICAL_MODELS = {'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b', 'qlearning_stacked', 'wmrl_stacked'}`.

A refactoring opportunity: extract `_fit_canonical_model()` helper that handles the
stacked data preparation + LEC loading + `run_inference_with_bump` + return, parameterized
by model function and model name. The M3 code block in `fit_model()` is ~60 lines and
would be repeated 5 times without this refactor.

---

## 11. SLURM Script Pattern

Template from `cluster/13_bayesian_m3.slurm`:
- `#SBATCH --time=06:00:00` — adequate for M3/M5/M6a; may need 08:00:00 for M6b
- `#SBATCH --mem=32G`
- `#SBATCH --cpus-per-task=4` (4 chains)
- `export JAX_PLATFORMS=cpu`
- `export NUMPYRO_HOST_DEVICE_COUNT=1` (note: 1, not 4; numpyro_set_host_device_count in script)
- conda activation: try `ds_env`, fallback to `rlwm_gpu`
- JAX cache: `JAX_COMPILATION_CACHE_DIR=/scratch/${PROJECT}/${USER}/.jax_cache_bayesian`

For Phase 16, create one SLURM script per new model:
- `cluster/13_bayesian_m1.slurm` — M1 (Q-learning, stacked variant)
- `cluster/13_bayesian_m2.slurm` — M2 (WM-RL, stacked variant)
- `cluster/13_bayesian_m5.slurm` — M5
- `cluster/13_bayesian_m6a.slurm` — M6a
- `cluster/13_bayesian_m6b.slurm` — M6b (expected longer compile; consider `--time=08:00:00`)

For the permutation test, a SLURM array job:
```bash
#SBATCH --array=0-49
python scripts/fitting/fit_bayesian.py --model wmrl_m3 --permutation-shuffle $SLURM_ARRAY_TASK_ID
```
This requires adding a `--permutation-shuffle INT` argument to `fit_bayesian.py` that
loads the LEC covariate, shuffles with `np.random.default_rng(shuffle_idx).permutation()`,
and saves results to `output/bayesian/permutation/shuffle_{idx}_results.json` instead of
the normal output path.

---

## 12. `bayesian_diagnostics.py` Readiness for New Models

`_get_param_names()` and `_get_likelihood_fn()` in `bayesian_diagnostics.py` already
have dispatch cases for ALL six models including M5, M6a, and M6b. The pointwise
log-likelihood computation for M6b correctly decodes the stick-breaking inside
`_build_per_participant_fn` (line ~257):
```python
kappa   = kappa_total * kappa_share
kappa_s = kappa_total * (1.0 - kappa_share)
```

`bayesian_summary_writer.py`'s `_MODEL_PARAMS` dict already contains entries for all
models including M5, M6a, M6b. No changes needed to either diagnostics module for the
new hierarchical models.

The Level-2 subscale coefficients (the 48 `beta_*` sites) will appear automatically in
the ArviZ posterior group. They do NOT need special handling in `write_bayesian_summary`
since that function only writes individual-level parameter posteriors. The `beta_*`
coefficients are group-level parameters and can be accessed directly from
`idata.posterior`.

---

## 13. `compute_pointwise_log_lik` for M1 (Q-learning) — Key Difference

The Q-learning stacked likelihood does not take `set_sizes_stacked`. In
`_build_per_participant_fn`, the `qlearning` branch already handles this correctly:
```python
if model_name == "qlearning":
    def _per_sample(alpha_pos, alpha_neg, epsilon):
        _, pointwise = fn(
            stimuli_stacked=stimuli_stacked,
            actions_stacked=actions_stacked,
            rewards_stacked=rewards_stacked,
            masks_stacked=masks_stacked,   # no set_sizes_stacked
            ...
        )
```
The NEW canonical `qlearning_hierarchical_model_stacked` must therefore NOT pass
`set_sizes_stacked` to the likelihood.

---

## 14. Compute Budget Reality Check

With 6 models to fit and 50 permutation shuffles:

| Task | Est. wall-clock per job | Cluster cores | Total CPU-hours |
|------|------------------------|---------------|-----------------|
| M1 hierarchical | ~1h | 4 | 4 |
| M2 hierarchical | ~2h | 4 | 8 |
| M3 hierarchical (already done in P15) | — | — | 0 |
| M5 hierarchical | ~3h | 4 | 12 |
| M6a hierarchical | ~3h | 4 | 12 |
| M6b hierarchical | ~4-6h | 4 | 24 |
| Permutation test (50 shuffles at 500W/1000S) | ~1h each | 4 | 200 |
| Collinearity audit | <5 min | 1 | <0.1 |
| az.compare (LOO) | ~10 min | 1 | 0.2 |

Total: ~260 CPU-hours. At 4-core jobs on M3 cluster: roughly 65 parallel-equivalent
single-core hours. Practical wall-clock with SLURM array: ~6-8 hours if all jobs run
simultaneously.

The permutation test dominates. Using 500 warmup + 1000 samples (half of standard)
reduces total permutation cost from ~400 to ~200 CPU-hours. The tradeoff is acceptable
since permutation tests only require approximate posteriors to verify HDI-excludes-zero.

---

## 15. Pitfalls and Cautions

### Pitfall 1: M6b compile time
The compound stick-breaking decode inside the per-participant loop adds graph complexity.
From STATE.md: "Compile-gate relaxed to 120s for CPU" — the M6b model may push or exceed
this. If smoke tests show M6b compiles in >120s, increase the compile gate to 180s for
M6b only (parameterize the gate threshold).

### Pitfall 2: 48-coefficient L2 fit — divergences likely
With 48 Level-2 coefficients plus 8x2 group-level parameters, the M6b subscale model has
a very high-dimensional posterior. The `run_inference_with_bump` retry loop handles
divergences but may exhaust all 3 levels (0.80 -> 0.95 -> 0.99). If divergences persist
at 0.99, the horseshoe prior (L2-08) is the explicit fallback path per Success Criterion 7.

### Pitfall 3: Missing LEC-5 subcategory columns
`less_total_events` and `less_personal_events` exist in `summary_participant_metrics.csv`
but LEC-5 physical/sexual/accident subcategories do NOT. These must be added from raw
survey data (script 04_create_summary_csv.py or manual extraction from jsPsych JSON)
before the 6-predictor L2 design matrix can be constructed. This is a hard blocker for
L2-04 and L2-05.

### Pitfall 4: IES-R total as a regressor alongside residualized subscales
If `ies_total` is included as-is in the L2 design alongside residualized subscales, the
residualized columns are by construction orthogonal to `ies_total`, so the design matrix
will be numerically well-conditioned. However, the ordering of Gram-Schmidt projections
matters: always project subscales against `ies_total` first, then check residuals are
orthogonal to `less_total_events` as well (LEC and IES-R total may be correlated).

### Pitfall 5: participant_ids ordering in permutation test
The permutation of `covariate_lec` must permute the rows (participant-level labels),
not the elements within a participant's data. Since `prepare_stacked_participant_data`
sorts participants by `sorted(data_df[participant_col].unique())`, the covariate array
must be aligned to the same sorted order before permutation.

### Pitfall 6: `numpyro.factor` site names must be unique across calls
All factor sites use `f"obs_p{pid}"`. Since SLURM jobs run separate Python processes,
this is fine. But the 50 permutation runs all define `obs_p{pid}` — they run in
separate processes so no collision. Just document this clearly in the permutation script.

### Pitfall 7: Legacy model names in fit_bayesian.py
The current CLI allows `--model qlearning` and `--model wmrl`, which dispatch to the
OLD convention models. Phase 16 adds NEW stacked variants. Two options:
(a) Replace the old dispatch silently (risky: changes existing behavior)
(b) Add a `--use-legacy` flag (complexity)
(c) Name the new functions with `_stacked` suffix and dispatch based on model name check
    (recommended: `'qlearning' -> new stacked`, keeping backward compat since old
    models had separate data prep paths)

The recommended approach: extend `BAYESIAN_IMPLEMENTED` to include `'qlearning'` and
`'wmrl'` using the NEW stacked convention, since HIER-02/03 explicitly say "port from
legacy to new canonical". Update the dispatch in `fit_model()` to use the stacked path
for these model names. The old `qlearning_hierarchical_model` and `wmrl_hierarchical_model`
functions remain in the file but are no longer called by `fit_bayesian.py`.

---

## 16. `scripts/fitting/level2_design.py` — New File Required (L2-03)

This module does not yet exist. It must be created with:
```python
def build_level2_design_matrix(
    metrics_df: pd.DataFrame,
    participant_ids: list,
    include_lec_subcategories: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Return (X, covariate_names) for Level-2 regression.

    X shape: (n_participants, n_covariates), z-scored columns.
    Covariates: less_total, iesr_total, iesr_intr_resid, iesr_avd_resid, iesr_hyp_resid
                + lec subcategories if include_lec_subcategories=True.
    """
```
This function is the single source of truth for the design matrix. Both the NumPyro
model and the collinearity audit call it. Storing the orthogonalization logic here
prevents drift between audit and model.

---

## Summary of Implementation Order

1. **Data audit task** (unblocks everything): Check if LEC-5 subcategory columns can
   be extracted from raw data. If not, restrict to 5 predictors: `less_total_events`,
   `ies_total`, `ies_intrusion_resid`, `ies_avoidance_resid`, `ies_hyperarousal_resid`.
   Report in phase SUMMARY as a deviation from the 6-predictor specification.

2. **Collinearity audit** (L2-02): Standalone script/function, ~50 lines. Produces
   `output/bayesian/level2/ies_r_collinearity_audit.md`. Fast (no MCMC).

3. **`level2_design.py`** (L2-03): New file with `build_level2_design_matrix()`.
   ~80 lines. Checked in before any L2 model fits.

4. **HIER-02 (M1)**: Add `qlearning_hierarchical_model_stacked` to `numpyro_models.py`.
   ~80 lines (M3 template minus kappa L2 logic, different likelihood call).

5. **HIER-03 (M2)**: Add `wmrl_hierarchical_model_stacked`. ~100 lines.

6. **HIER-04 (M5)**: Add `wmrl_m5_hierarchical_model`. ~110 lines (M3 + phi_rl parameter).

7. **HIER-05 (M6a)**: Add `wmrl_m6a_hierarchical_model`. ~100 lines (M3 with kappa_s
   replacing kappa).

8. **HIER-06 (M6b)**: Add `wmrl_m6b_hierarchical_model`. ~120 lines (M3 + kappa_total/
   kappa_share + decode + L2 on both). The most complex new model.

9. **fit_bayesian.py extension**: Add dispatch for 5 new models. Refactor common
   stacked-model logic into helper `_fit_stacked_model()`.

10. **L2-05 M6b subscale fit**: Add multi-covariate variant of M6b model
    (`wmrl_m6b_hierarchical_model_subscale` with 48 beta sites or matrix beta).

11. **L2-06 permutation null test**: Add `--permutation-shuffle INT` to fit_bayesian.py.
    Create permutation SLURM array job script.

12. **L2-07 forest plots**: Create `scripts/18_bayesian_level2_effects.py`.

13. **SLURM scripts** for each new model.

14. **`scripts/14_compare_models.py`** `--bayesian-comparison` flag with `az.compare`.
