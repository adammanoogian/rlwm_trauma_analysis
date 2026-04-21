# Phase 15: M3 Hierarchical POC with Level-2 Regression - Research

**Researched:** 2026-04-12
**Domain:** NumPyro hierarchical Bayesian inference, JAX likelihoods, ArviZ diagnostics
**Confidence:** HIGH (all findings from direct codebase inspection)

## Summary

Phase 15 builds the first complete hierarchical NumPyro model for M3 (WM-RL+kappa, 7 parameters)
on the real N=154 dataset, adds a Level-2 regression for LEC-total -> kappa, and validates that the
resulting posterior reproduces the v3.0 quick-006 FDR-BH survivor `kappa x LEC-5 total_events`
(beta=0.00658, p=0.00187).

The Phase 13 infrastructure is thorough but the two existing hierarchical models in
`numpyro_models.py` — `qlearning_hierarchical_model` and `wmrl_hierarchical_model` — use the OLD
convention (Beta/logit group priors, `mu_capacity ~ TruncatedNormal(4, 1.5, low=1, high=7)`). They
must NOT be modified. Phase 15 writes a **new** `wmrl_m3_hierarchical_model` function in the same
file that uses `sample_model_params()` from `numpyro_helpers.py` (hBayesDM non-centered convention,
K in [2, 6]).

The entire compute pipeline from MCMC -> InferenceData -> schema-parity CSV is already built. The
two missing pieces are: (1) the `wmrl_m3_hierarchical_model` function itself with Level-2 regression,
and (2) the convergence-gate retry loop with auto-bump of `target_accept_prob`.

**Primary recommendation:** Write `wmrl_m3_hierarchical_model` using `sample_model_params()` for all
7 parameters. Add Level-2 regression as a deterministic offset to `kappa_mu_pr` before sampling
individual-level z. Wrap `run_inference` in a retry loop that bumps `target_accept_prob` 0.8 -> 0.95
-> 0.99 on divergences. Use existing `bayesian_diagnostics.py` + `bayesian_summary_writer.py`
unchanged.

## Standard Stack

### Core (already installed and pinned per Phase 13)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpyro | 0.20.1 | NUTS sampler, model syntax | Project standard; pinned Phase 13 |
| jax | 0.4.31 | JAX numerical ops, vmap, jit | Project standard (ds_env) |
| arviz | 0.23.4 | WAIC, LOO, diagnostics | Project standard; pinned Phase 13 |
| pandas | — | DataFrame I/O | Pipeline standard |
| numpy | — | Array prep before JAX | Pipeline standard |

### Key existing modules (no new dependencies)

| Module | Path | What it provides |
|--------|------|-----------------|
| `numpyro_helpers.py` | `scripts/fitting/` | `sample_model_params()`, `sample_bounded_param()`, `phi_approx()`, `PARAM_PRIOR_DEFAULTS` |
| `bayesian_diagnostics.py` | `scripts/fitting/` | `compute_pointwise_log_lik()`, `build_inference_data_with_loglik()` |
| `bayesian_summary_writer.py` | `scripts/fitting/` | `write_bayesian_summary()` with full schema-parity CSV |
| `jax_likelihoods.py` | `scripts/fitting/` | `wmrl_m3_multiblock_likelihood_stacked()` (the likelihood) |
| `numpyro_models.py` | `scripts/fitting/` | Canonical model file; add M3 here |
| `fit_bayesian.py` | `scripts/fitting/` | CLI to extend for M3 + convergence bump |
| `config.py` | project root | `MODEL_REGISTRY`, `PARAM_PRIOR_DEFAULTS`, `EXPECTED_PARAMETERIZATION` |

**Installation:** No new packages needed. All deps pinned in Phase 13.

## Architecture Patterns

### Recommended Project Structure for Phase 15 Changes

```
scripts/fitting/
├── numpyro_models.py          # ADD: wmrl_m3_hierarchical_model, run_inference_with_bump
├── fit_bayesian.py            # EXTEND: support wmrl_m3, call run_inference_with_bump
└── tests/
    ├── test_compile_gate.py   # FIX: synthetic data uses jnp.array(), not .tolist()
    └── test_m3_hierarchical.py  # NEW: smoke test (5 subj, 200 samples)

output/bayesian/
├── wmrl_m3_individual_fits.csv       # schema-parity CSV (write_bayesian_summary)
└── wmrl_m3_shrinkage_report.md       # shrinkage diagnostic (new writer)
```

### Pattern 1: New M3 hierarchical model using sample_model_params

The correct pattern uses `sample_model_params()` from `numpyro_helpers.py`, which internally calls
`sample_bounded_param()` for each of M3's 7 parameters using PARAM_PRIOR_DEFAULTS. This is the
locked Phase 13 convention.

```python
# Source: scripts/fitting/numpyro_helpers.py (sample_model_params) + Phase 13 decisions

def wmrl_m3_hierarchical_model(
    participant_data_stacked: dict,
    covariate_lec: jnp.ndarray | None = None,  # shape (n_participants,), standardized
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
) -> None:
    n_participants = len(participant_data_stacked)

    # --- Level-2: LEC-total -> kappa group-mean regression coefficient ---
    if covariate_lec is not None:
        beta_lec_kappa = numpyro.sample("beta_lec_kappa", dist.Normal(0, 1))
    else:
        beta_lec_kappa = 0.0

    # --- Group priors for all 7 parameters (hBayesDM non-centered, K in [2,6]) ---
    # sample_model_params calls sample_bounded_param for each param in MODEL_REGISTRY["wmrl_m3"]["params"]
    # EXCEPT: for kappa, we need a per-participant shift from the L2 covariate.
    # Strategy: sample all params EXCEPT kappa via sample_model_params, then handle kappa manually.
    params_except_kappa = [p for p in MODEL_REGISTRY["wmrl_m3"]["params"] if p != "kappa"]
    sampled = {}
    for param in params_except_kappa:
        defaults = PARAM_PRIOR_DEFAULTS[param]
        sampled[param] = sample_bounded_param(
            param,
            lower=defaults["lower"],
            upper=defaults["upper"],
            n_participants=n_participants,
            mu_prior_loc=defaults["mu_prior_loc"],
        )

    # Kappa with optional L2 offset on the group mean
    kappa_defaults = PARAM_PRIOR_DEFAULTS["kappa"]
    kappa_mu_pr = numpyro.sample("kappa_mu_pr", dist.Normal(kappa_defaults["mu_prior_loc"], 1.0))
    kappa_sigma_pr = numpyro.sample("kappa_sigma_pr", dist.HalfNormal(0.2))
    with numpyro.plate("participants", n_participants):
        kappa_z = numpyro.sample("kappa_z", dist.Normal(0, 1))
        # L2 shift: add beta_lec_kappa * lec_i to the individual unconstrained value
        lec_shift = beta_lec_kappa * covariate_lec if covariate_lec is not None else 0.0
        kappa_unc = kappa_mu_pr + kappa_sigma_pr * kappa_z + lec_shift
        kappa = numpyro.deterministic("kappa", phi_approx(kappa_unc))
    sampled["kappa"] = kappa

    # --- Likelihood via numpyro.factor ---
    for idx, pid in enumerate(participant_data_stacked.keys()):
        pdata = participant_data_stacked[pid]
        log_lik, _ = wmrl_m3_multiblock_likelihood_stacked(
            stimuli_stacked=pdata["stimuli_stacked"],
            ...
            alpha_pos=sampled["alpha_pos"][idx],
            ...
            kappa=sampled["kappa"][idx],
            return_pointwise=False,  # False for MCMC inner loop performance
        )
        numpyro.factor(f"obs_p{pid}", log_lik)
```

**IMPORTANT NOTE on vmap-over-participants:** The phase description says "vmap-over-participants
likelihood." The existing `qlearning_hierarchical_model` and `wmrl_hierarchical_model` both use a
Python `for` loop over participants (not `vmap`). This is the correct pattern for the current
codebase because `wmrl_m3_multiblock_likelihood_stacked` is not trivially vmappable (it uses
`lax.fori_loop` over variable-length block structures). The for-loop approach is semantically
equivalent and is what the existing infrastructure supports. The planner should use a Python for
loop as in the existing models, NOT try to vmap the outer participant loop.

### Pattern 2: Level-2 regression — where beta_lec_kappa lives

The L2 regression coefficient enters as a **shift on the individual unconstrained kappa value**
(probit scale), NOT as a shift on the group mean mu_pr. This is the standard multilevel approach:

```
kappa_unc_i = kappa_mu_pr + kappa_sigma_pr * z_i + beta_lec_kappa * lec_i
kappa_i = phi_approx(kappa_unc_i)
```

The covariate `covariate_lec` should be standardized (z-scored) before passing to the model.
Standardization happens in the CLI, not inside the model function.

### Pattern 3: Convergence auto-bump (HIER-07)

The auto-bump is a Python retry loop around MCMC.run(), not inside NumPyro. Pattern:

```python
def run_inference_with_bump(
    model,
    model_args,
    num_warmup=1000,
    num_samples=2000,
    num_chains=4,
    seed=42,
    target_accept_probs=(0.80, 0.95, 0.99),
) -> MCMC:
    for tap in target_accept_probs:
        nuts = NUTS(model, target_accept_prob=tap)
        mcmc = MCMC(nuts, num_warmup=num_warmup, num_samples=num_samples,
                    num_chains=num_chains, progress_bar=True)
        mcmc.run(jax.random.PRNGKey(seed), **model_args)
        n_div = int(mcmc.get_extra_fields()["diverging"].sum())
        print(f"  target_accept_prob={tap}: divergences={n_div}")
        if n_div == 0:
            return mcmc
        print(f"  Bumping target_accept_prob from {tap}...")
    # Return last run even if divergences remain (downstream gate flags it)
    return mcmc
```

The divergences count is in `mcmc.get_extra_fields()["diverging"].sum()`.

### Pattern 4: Shrinkage diagnostic (HIER-08)

From `bayesian_summary_writer.py` design: after `az.from_numpyro(mcmc)`, compute:

```python
def compute_shrinkage(idata, param_names):
    # var_post_individual: variance of individual-level draws across participants
    # var_post_group: variance of the group mean across draws (captures how much
    #   the group mean varies, i.e., the expected individual-level spread)
    results = {}
    for param in param_names:
        indiv_draws = idata.posterior[param].values  # (chains, draws, n_participants)
        flat = indiv_draws.reshape(-1, indiv_draws.shape[-1])  # (draws, n_participants)
        var_individual = float(jnp.var(flat))       # variance over all draws AND participants
        var_group = float(jnp.var(flat.mean(axis=1)))  # variance of per-draw group mean
        shrinkage = 1.0 - var_individual / (var_group + 1e-10)
        results[param] = shrinkage
    return results
```

Parameters with shrinkage < 0.3 are flagged as poorly identified; downstream uses them as
descriptive only. The shrinkage report writes to `output/bayesian/wmrl_m3_shrinkage_report.md`.

### Pattern 5: PPC infrastructure (HIER-09)

PPC stratified by trauma group uses `numpyro.infer.Predictive` on the posterior samples.
The existing `09_generate_synthetic_data.py` is NOT the right tool here — it generates forward
simulations from fixed parameters. Phase 15 PPC must:

1. Sample posterior parameter vectors for each participant.
2. For each sample, run the M3 likelihood to generate predicted accuracy by block.
3. Stratify by trauma group (using the group assignment CSV).
4. Compare 95% PPC envelope to observed block-level accuracy.

The simulations module does not need to be modified. PPC is a standalone function in the
Bayesian fitting flow.

### Pattern 6: Schema-parity CSV for Bayesian fits

`write_bayesian_summary()` in `bayesian_summary_writer.py` is the ONLY writer to use. It produces
the correct column order (already tested in `test_bayesian_summary.py`). The M3 param list
`["alpha_pos", "alpha_neg", "phi", "rho", "capacity", "kappa", "epsilon"]` is already in
`_MODEL_PARAMS["wmrl_m3"]`. The `parameterization_version` for M3 Bayesian fits must be
`"v4.0-K[2,6]-phiapprox"` (from `EXPECTED_PARAMETERIZATION["wmrl_m3"]`).

### Pattern 7: WAIC/LOO without Pareto-k warnings

The current `bayesian_diagnostics.py` returns a `(chains, samples, participants, n_blocks*max_trials)`
tensor from `compute_pointwise_log_lik()`. The mask==0 positions carry log_prob=0.0 (from the
likelihood's mask handling). **Before calling `az.waic()` or `az.loo()`, the padding log-probs must
be excluded from the `log_likelihood` group in InferenceData**, otherwise the effective parameter
count is inflated. The `build_inference_data_with_loglik()` function currently stores ALL positions
including padding. Phase 15 must filter or mask the zero log-probs before passing to ArviZ.

### Anti-Patterns to Avoid

- **Using the old `wmrl_hierarchical_model` as a template for M3.** It uses Beta group priors and
  K in [1, 7]. The new M3 model must use `sample_model_params()` / `sample_bounded_param()`.
- **Modifying existing `qlearning_hierarchical_model` or `wmrl_hierarchical_model`.** These are
  working legacy models consumed by existing tests and `fit_bayesian.py`.
- **Calling `wmrl_m3_multiblock_likelihood` (not stacked version).** The non-stacked version
  accepts Python lists and uses Python for-loops. Inside a NumPyro model, always use the stacked
  version (`wmrl_m3_multiblock_likelihood_stacked`) which takes pre-stacked JAX arrays.
- **Registering kappa_mu_pr and kappa_sigma_pr both inside and outside the plate.** The plate
  `participants` wraps only the per-participant z samples. Group-level mu_pr and sigma_pr must be
  outside the plate.
- **Forgetting to standardize the LEC covariate.** The model assumes standardized input (z-score).
  Raw LEC counts (typically 0-17) would make beta_lec_kappa uninterpretable and the Normal(0,1)
  prior severely miscalibrated.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Schema-parity CSV | Custom CSV writer | `write_bayesian_summary()` | Already tested for schema parity; Phase 13 deliverable |
| Non-centered parameterization | Custom phi-transform | `sample_bounded_param()` | hBayesDM convention locked Phase 13; handles edge cases |
| Pointwise log-lik for WAIC/LOO | Custom array reshaping | `compute_pointwise_log_lik()` + `build_inference_data_with_loglik()` | Already handles chains/samples/participants/trials shape |
| ArviZ InferenceData | `az.from_numpyro()` in place | `build_inference_data_with_loglik()` | The bare `az.from_numpyro()` lacks `log_likelihood` group |
| M3 block likelihood | Re-implementing WM+RL+kappa | `wmrl_m3_multiblock_likelihood_stacked()` | Fully tested; locked parameterization |
| HDI computation | Custom percentile | `az.hdi()` inside `write_bayesian_summary()` | Already called correctly |
| Convergence diagnostics | Manual R-hat calc | `az.summary()` with `r_hat` column | Standard; `bayesian_summary_writer.py` already uses it |
| Kappa parameter bounds | Custom clipping | `sample_bounded_param("kappa", lower=0.0, upper=1.0, ...)` | Correct bounds already in PARAM_PRIOR_DEFAULTS |

**Key insight:** Phase 13 built everything except the M3 hierarchical model function itself and the
convergence bump loop. Phase 15 is primarily wiring + one new model function.

## Common Pitfalls

### Pitfall 1: test_compile_gate.py uses Python lists (.tolist()) not JAX arrays

**What goes wrong:** `test_compile_gate.py` fails with `ValueError: scan got value with no leading
axis to scan over`. Root cause confirmed via `conda run -n ds_env python -m pytest` — the synthetic
data helper `_make_minimal_synthetic_data` stores stimuli/actions/rewards as Python lists (result
of `.tolist()`) instead of JAX arrays.

**Root cause:** `lax.scan` in `q_learning_block_likelihood` (line 484 of `jax_likelihoods.py`)
receives `scan_inputs = (stimuli, actions, rewards, mask)`. When `stimuli` is a Python list of
integers, JAX cannot determine the scan axis shape.

**How to fix:** In `_make_minimal_synthetic_data` (test_compile_gate.py lines 56-60), replace
`.tolist()` with direct numpy arrays or wrap in `jnp.array()`:
```python
# OLD (broken):
stimuli_blocks.append(rng.integers(0, 3, n_trials).tolist())
# NEW (fixed):
stimuli_blocks.append(jnp.array(rng.integers(0, 3, n_trials), dtype=jnp.int32))
```

**Warning signs:** The error message says `'int' object has no attribute 'shape'` in JAX scan
internals.

### Pitfall 2: Using sample_model_params() for kappa when L2 covariate is present

**What goes wrong:** `sample_model_params()` samples all params uniformly using default priors.
For the L2 regression, kappa's group mean needs an additive offset from the standardized LEC
covariate. `sample_model_params()` has no mechanism for this.

**How to avoid:** Sample kappa manually (outside `sample_model_params()`), add the `beta_lec_kappa
* lec_i` term to the per-participant unconstrained value, then combine results. Use
`sample_model_params()` for the other 6 parameters.

### Pitfall 3: stacked participant data format differs from numpyro_models.py format

**What goes wrong:** `compute_pointwise_log_lik()` in `bayesian_diagnostics.py` expects
`participant_data_stacked` with keys `stimuli_stacked`, `actions_stacked`, `rewards_stacked`,
`masks_stacked`, `set_sizes_stacked` — pre-stacked JAX arrays of shape `(n_blocks, max_trials)`.

The existing `prepare_data_for_numpyro()` in `numpyro_models.py` returns a DIFFERENT format:
lists of arrays per participant `{'stimuli_blocks': [array1, array2, ...], ...}`.

**How to avoid:** Phase 15 needs a `prepare_stacked_participant_data()` function that converts
the DataFrame into the stacked format. The `pad_block_to_max()` function in `jax_likelihoods.py`
handles the padding. This function does NOT exist yet and must be written in Phase 15.

### Pitfall 4: WAIC/LOO inflated by padding zeros in log_likelihood group

**What goes wrong:** `az.waic()` and `az.loo()` treat every observation position as a real
trial. Padding positions have log_prob=0.0, which biases the WAIC effective parameters upward.

**How to avoid:** Before building InferenceData, filter the pointwise log-lik array to exclude
positions where the mask is 0. Since mask arrays live in the stacked participant data, pass them
through `build_inference_data_with_loglik()` or filter before the call:
```python
# Filter padded positions: only keep positions where mask != 0
# The current build_inference_data_with_loglik does NOT do this automatically
```

### Pitfall 5: Running `fit_bayesian.py` with `--model wmrl_m3` before Phase 15 is done

**What goes wrong:** `fit_bayesian.py:149` has `BAYESIAN_IMPLEMENTED = {'qlearning', 'wmrl'}` and
raises `NotImplementedError` for `wmrl_m3`. The CLI must be extended to include `wmrl_m3` in the
implemented set after the model function exists.

### Pitfall 6: fit_bayesian.py references mu_beta which no longer exists

**What goes wrong:** `fit_bayesian.py` lines 210-216 access `samples['mu_beta']` and
`samples['mu_beta_wm']` which do not exist in the new hBayesDM-convention models. The M3 model
will have `kappa_mu_pr`, `kappa_sigma_pr`, etc. — not `mu_kappa`.

**How to avoid:** The M3-specific printing logic in `fit_bayesian.py`'s `fit_model()` must be
updated to use the new parameter names, or a separate code path added for `wmrl_m3`.

### Pitfall 7: covariate_lec shape must match n_participants plate

**What goes wrong:** If the participant order in `covariate_lec` does not match the order of
keys in `participant_data_stacked`, the regression coefficient is associated with the wrong
participants.

**How to avoid:** Sort both the DataFrame and the covariate array by the same participant_id
list before creating the model arguments. The LEC covariate must be extracted in the same order
as `sorted(participant_data_stacked.keys())`.

## Code Examples

### M3 stacked likelihood call pattern (inside hierarchical model)

```python
# Source: scripts/fitting/jax_likelihoods.py wmrl_m3_multiblock_likelihood_stacked

log_lik = wmrl_m3_multiblock_likelihood_stacked(
    stimuli_stacked=pdata["stimuli_stacked"],    # (n_blocks, max_trials)
    actions_stacked=pdata["actions_stacked"],
    rewards_stacked=pdata["rewards_stacked"],
    set_sizes_stacked=pdata["set_sizes_stacked"],
    masks_stacked=pdata["masks_stacked"],
    alpha_pos=alpha_pos_i,   # scalar float from plate
    alpha_neg=alpha_neg_i,
    phi=phi_i,
    rho=rho_i,
    capacity=capacity_i,
    kappa=kappa_i,
    epsilon=epsilon_i,
    return_pointwise=False,  # False for MCMC (scalar needed by numpyro.factor)
)
numpyro.factor(f"obs_p{pid}", log_lik)
```

### sample_bounded_param call for kappa with L2 shift

```python
# Source: scripts/fitting/numpyro_helpers.py (sample_bounded_param pattern)
# The L2 shift adds beta_lec_kappa * lec_i to the unconstrained value per participant

kappa_mu_pr = numpyro.sample("kappa_mu_pr", dist.Normal(-2.0, 1.0))  # -2.0 from PARAM_PRIOR_DEFAULTS
kappa_sigma_pr = numpyro.sample("kappa_sigma_pr", dist.HalfNormal(0.2))

with numpyro.plate("participants", n_participants):
    kappa_z = numpyro.sample("kappa_z", dist.Normal(0, 1))
    lec_shift = beta_lec_kappa * covariate_lec  # covariate_lec: (n_participants,) jnp array
    kappa_unc = kappa_mu_pr + kappa_sigma_pr * kappa_z + lec_shift
    kappa = numpyro.deterministic("kappa", phi_approx(kappa_unc))
    # phi_approx maps real -> (0,1); kappa in [0.0, 1.0]
```

### Convergence auto-bump loop

```python
# Source: Pattern from run_inference() in numpyro_models.py, extended with bump

from numpyro.infer import MCMC, NUTS

def run_inference_with_bump(model, model_args, num_warmup, num_samples, num_chains, seed):
    for tap in (0.80, 0.95, 0.99):
        nuts = NUTS(model, target_accept_prob=tap, max_tree_depth=10)
        mcmc = MCMC(nuts, num_warmup=num_warmup, num_samples=num_samples,
                    num_chains=num_chains, progress_bar=True)
        mcmc.run(jax.random.PRNGKey(seed), **model_args)
        n_div = int(mcmc.get_extra_fields()["diverging"].sum())
        print(f"[convergence-gate] target_accept_prob={tap:.2f} divergences={n_div}")
        if n_div == 0:
            return mcmc
        print(f"[convergence-gate] bumping to next target_accept_prob level")
    return mcmc  # return last run with divergences; gate flags it downstream
```

### Shrinkage computation

```python
# Source: design pattern for HIER-08; not yet in codebase

import numpy as np

def compute_shrinkage_report(idata, param_names: list[str]) -> dict[str, float]:
    """Compute 1 - var_post_individual / var_post_group for each parameter."""
    results = {}
    posterior = idata.posterior
    for param in param_names:
        arr = posterior[param].values  # (chains, draws, n_participants)
        flat = arr.reshape(-1, arr.shape[-1])  # (total_draws, n_participants)
        # var over all individual samples (across draws AND participants)
        var_indiv = float(np.var(flat))
        # var of per-draw group mean (measures between-draw spread of group average)
        var_group_mean = float(np.var(flat.mean(axis=1)))
        # Shrinkage: how much individual variance is explained by group structure
        shrinkage = 1.0 - var_indiv / (var_group_mean + 1e-10)
        results[param] = shrinkage
    return results
```

### Smoke test dispatch pattern (HIER-10)

```python
# Source: test_compile_gate.py pattern, extended to M3

import pytest

@pytest.mark.slow
def test_smoke_dispatch():
    """M3 hierarchical smoke: 5 subjects, 200 samples, < 60s."""
    import time
    from numpyro.infer import MCMC, NUTS
    from scripts.fitting.numpyro_models import wmrl_m3_hierarchical_model

    model_args = _make_m3_synthetic_stacked(n_ppts=5, n_blocks=3, n_trials=20)
    nuts = NUTS(wmrl_m3_hierarchical_model)
    mcmc = MCMC(nuts, num_warmup=100, num_samples=200, num_chains=1, progress_bar=False)

    t0 = time.monotonic()
    mcmc.run(jax.random.PRNGKey(42), **model_args)
    elapsed = time.monotonic() - t0

    assert elapsed < 60.0, f"Smoke test took {elapsed:.1f}s > 60s gate"
```

The `_make_m3_synthetic_stacked` fixture must produce the STACKED format (not the list format
used by the old models).

### test_compile_gate.py fix

```python
# Fix: lines 57-59 of test_compile_gate.py
# OLD (broken):
stimuli_blocks.append(rng.integers(0, 3, n_trials).tolist())
actions_blocks.append(rng.integers(0, 3, n_trials).tolist())
rewards_blocks.append(rng.integers(0, 2, n_trials).astype(float).tolist())

# NEW (correct):
stimuli_blocks.append(jnp.array(rng.integers(0, 3, n_trials), dtype=jnp.int32))
actions_blocks.append(jnp.array(rng.integers(0, 3, n_trials), dtype=jnp.int32))
rewards_blocks.append(jnp.array(rng.integers(0, 2, n_trials).astype(np.float32)))
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Beta group priors + logit transform | HalfNormal sigma + Normal mu_pr (hBayesDM) | Phase 13 (numpyro_helpers.py) | Better posterior geometry; avoid boundary issues |
| K in [1, 7] | K in [2, 6] | Phase 13 K-01 research, Phase 14 implementation | Structural identifiability fix |
| wmrl_hierarchical_model (M2 only) | new wmrl_m3_hierarchical_model | Phase 15 (to be built) | M3 with kappa + L2 covariate |
| No Level-2 regression | beta_lec_kappa as unconstrained shift on kappa probit scale | Phase 15 (to be built) | Jointly estimates individual shrinkage + LEC effect |
| Python for-loop over participants (slow) | Same for-loop (acceptable; vmap not needed for stacked likelihood) | Phase 13 decision | vmap requires homogeneous inputs; for-loop is correct here |

**Deprecated/outdated:**
- `mu_beta` / `mu_beta_wm` names in `fit_bayesian.py` print block: these group-level names were
  from the old M2 model. Phase 15 model uses `kappa_mu_pr` naming convention.
- The non-stacked `wmrl_m3_multiblock_likelihood()` (takes Python lists): do not use inside MCMC;
  use `_stacked` version.

## Open Questions

1. **stacked data preparation function**
   - What we know: `compute_pointwise_log_lik()` expects stacked format; `prepare_data_for_numpyro()`
     returns list format; no function currently bridges the two.
   - What's unclear: Whether Phase 15 should add `prepare_stacked_participant_data()` to
     `numpyro_models.py` or to `jax_likelihoods.py` or to a new utility module.
   - Recommendation: Add to `numpyro_models.py` alongside `prepare_data_for_numpyro()`.

2. **Padding filter before az.waic()/az.loo()**
   - What we know: `build_inference_data_with_loglik()` stores all positions including
     padding zeros. The docstring says "bayesian_diagnostics.py must filter mask==0 positions
     before az.waic() / az.loo()".
   - What's unclear: Whether to modify `build_inference_data_with_loglik()` to accept a mask
     argument and filter inline, or to filter externally before calling.
   - Recommendation: Filter externally in the fitting script before calling
     `build_inference_data_with_loglik()` — avoids changing a tested API.

3. **Shrinkage formula interpretation**
   - What we know: The requirement says `1 - var_post_individual / var_post_group` > 0.3.
   - What's unclear: The exact definition of `var_post_group`. Options:
     (a) variance of the posterior group mean draws `var(per-draw mean across participants)`,
     (b) variance of the `{param}_mu_pr` posterior.
   - Recommendation: Use option (a) — variance of per-draw mean across participants — as it
     directly measures how much individual differences are shrunk toward the group mean.
     Document the formula explicitly in the shrinkage report.

4. **LEC covariate column name in summary CSV**
   - What we know: `lec_total_events` is the column that survives FDR in quick-006.
   - What's unclear: The exact column name in `output/summary_participant_metrics.csv`.
   - Recommendation: The planner should add a task to verify the column name from the data file
     before writing the CLI data-loading code.

## Sources

### Primary (HIGH confidence)

- Direct code reading: `scripts/fitting/numpyro_models.py` — confirmed legacy model structure
- Direct code reading: `scripts/fitting/numpyro_helpers.py` — confirmed `sample_model_params`,
  `PARAM_PRIOR_DEFAULTS`, hBayesDM convention
- Direct code reading: `scripts/fitting/bayesian_diagnostics.py` — confirmed stacked data format,
  `compute_pointwise_log_lik` signature
- Direct code reading: `scripts/fitting/bayesian_summary_writer.py` — confirmed schema-parity,
  `write_bayesian_summary`, `_MODEL_PARAMS["wmrl_m3"]`
- Direct code reading: `scripts/fitting/jax_likelihoods.py` — confirmed
  `wmrl_m3_multiblock_likelihood_stacked` signature and that the stacked version exists
- Direct test execution: `conda run -n ds_env python -m pytest scripts/fitting/tests/test_compile_gate.py`
  — confirmed failure and root cause (Python lists in _make_minimal_synthetic_data)
- Direct reading: `output/regressions/wmrl_m3/significance_summary.md` — confirmed FDR-BH
  survivors: phi x IES-R Hyperarousal, kappa x lec_total_events, phi x IES-R Total
- Direct reading: `.planning/STATE.md` — confirmed all Phase 13 decisions, locked parameterization
  conventions, and pre-existing test failure documentation
- Direct reading: `config.py` — confirmed MODEL_REGISTRY, EXPECTED_PARAMETERIZATION["wmrl_m3"]
  = "v4.0-K[2,6]-phiapprox", PARAM_PRIOR_DEFAULTS not in config.py (it's in numpyro_helpers.py)

### Secondary (MEDIUM confidence)

- ArviZ documentation pattern: `az.waic()` and `az.loo()` behavior with padding zeros — based on
  ArviZ design; confirmed indirectly by the docstring note in `bayesian_diagnostics.py` that says
  "must filter mask==0 positions before az.waic() / az.loo()"

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all from direct code inspection, no external lookups needed
- Architecture: HIGH — M3 model structure follows existing patterns in numpyro_models.py and
  numpyro_helpers.py exactly
- Pitfalls: HIGH for test failure (confirmed with live test run); HIGH for data format mismatch
  (confirmed from API signatures); MEDIUM for padding/WAIC issue (from docstring, not tested)
- Level-2 regression pattern: HIGH — standard multilevel regression; fits within numpyro_helpers
  non-centered convention naturally

**Research date:** 2026-04-12
**Valid until:** 2026-05-12 (stable codebase; numpyro 0.20.1 pinned)
