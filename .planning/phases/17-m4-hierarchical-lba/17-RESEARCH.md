# Phase 17: M4 Hierarchical LBA - Research

**Researched:** 2026-04-12
**Domain:** NumPyro hierarchical NUTS + JAX float64 LBA likelihood + checkpoint-and-resume
**Confidence:** HIGH (all findings from direct codebase inspection; no training-data guesses)

---

## Summary

Phase 17 adds a hierarchical Bayesian M4 fit to a codebase that already has a working
float64 MLE path (`fit_all_gpu_m4`) and six choice-only hierarchical models following
the hBayesDM non-centered pattern.  The implementation is well-defined by four concrete
dependencies: the stacked LBA likelihood (`wmrl_m4_multiblock_likelihood_stacked` in
`lba_likelihood.py`), the MLE bounds dict (`WMRL_M4_BOUNDS`), the M6b hierarchical
model as a structural template, and the NumPyro `MCMC.post_warmup_state` /
`MCMC.warmup()` API for checkpoint-and-resume.

The largest new work items are: (1) building a `prepare_stacked_participant_data_m4`
variant that also stacks float64 RT arrays and combines the RT-outlier filter with
the padding mask, (2) writing the `wmrl_m4_hierarchical_model` NumPyro function with
log-scale/non-centered reparameterization of LBA params, (3) a separate script
`13_fit_bayesian_m4.py` (not touching `fit_bayesian.py`) with float64 init at the very
top, and (4) a separate `bayesian_diagnostics_m4` path because the existing
`bayesian_diagnostics.py` does not import from `lba_likelihood` and cannot handle M4.

The Pareto-k fallback and choice-only marginal comparison are the novel scientific
components with no existing infrastructure; they require new code from scratch.

**Primary recommendation:** Write M4 as a fully self-contained script
`13_fit_bayesian_m4.py` that imports `lba_likelihood` before anything else. Do NOT add
`wmrl_m4` to `STACKED_MODEL_DISPATCH`, `_get_param_names`, or `_get_likelihood_fn` in
the existing files — float64 isolation is process-wide and must not bleed into
choice-only model paths.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpyro | 0.20.1 (pinned) | NUTS sampler, MCMC | Already in all env specs |
| jax (float64) | installed | LBA likelihood precision | `jax.config.update("jax_enable_x64", True)` |
| arviz | 0.23.4 (pinned) | LOO/Pareto-k diagnostics | `az.loo(idata, pointwise=True)` |
| lba_likelihood.py | project | Stacked LBA likelihood | `wmrl_m4_multiblock_likelihood_stacked` |

### Supporting
| Library | Purpose | When to Use |
|---------|---------|-------------|
| numpyro_helpers.py | `sample_bounded_param`, `phi_approx`, `PARAM_PRIOR_DEFAULTS` | All 6 RLWM params that ARE in PARAM_PRIOR_DEFAULTS |
| bayesian_summary_writer.py | Schema-parity CSV | `_MODEL_PARAMS["wmrl_m4"]` already defined there |
| bayesian_diagnostics.py | Shrinkage, PPC | NOT usable for M4 pointwise loglik — needs M4-specific path |

---

## Architecture Patterns

### Float64 Isolation — CRITICAL

The float64 flag is process-wide. `lba_likelihood.py` sets it at module level:

```python
# lba_likelihood.py line 11
import jax
jax.config.update("jax_enable_x64", True)
```

The M4 hierarchical script MUST follow this exact ordering (from STATE.md locked decision):

```python
# 13_fit_bayesian_m4.py — FIRST two executable lines, no exceptions
import jax
jax.config.update("jax_enable_x64", True)
import numpyro
numpyro.enable_x64()
# Then import lba_likelihood (which re-triggers the same config, idempotent)
from scripts.fitting.lba_likelihood import wmrl_m4_multiblock_likelihood_stacked
```

The existing `fit_bayesian.py` does NOT set float64 and must NOT be modified to do so.
M4 must run in a separate SLURM job from choice-only models.

### Stacked Data Preparation — New Function Needed

`prepare_stacked_participant_data` (in `numpyro_models.py`) does not stack RTs. M4
needs a variant `prepare_stacked_participant_data_m4` that:

- Accepts `rt_col: str = "rt"` parameter
- Reads `block_data["rt"]` as float64 milliseconds
- Calls `preprocess_rt_block(rt_raw)` → `(rt_sec, valid_rt)` to get seconds + outlier mask
- ANDs the RT-outlier mask with the padding mask (`valid_rt & (1.0 for real trial)`)
- Produces an additional `rts_stacked` key of shape `(n_blocks, MAX_TRIALS_PER_BLOCK)` dtype float64
- RT padding positions should be filled with a safe value (e.g., 0.5 seconds) — the mask zeros out their contribution anyway

The MLE path in `fit_mle.py` (lines 2262–2291) is the exact reference implementation.

### Non-Centered LBA Parameter Reparameterization

The existing `PARAM_PRIOR_DEFAULTS` explicitly excludes LBA params (v_scale, A, delta, t0):

```
# numpyro_helpers.py line 208
- LBA-specific parameters (v_scale, A, delta, t0) are NOT listed here
  because they require log-scale or non-standard transforms handled
  separately in the M4 hierarchical model.
```

The required M4H-02 reparameterization for the threshold offset uses:

```python
# Non-centered log(b - A) = log(delta) reparameterization
# delta = b - A must be > 0; sampling log_delta on unconstrained scale
log_delta_mu_pr = numpyro.sample("log_delta_mu_pr", dist.Normal(0.0, 1.0))
log_delta_sigma_pr = numpyro.sample("log_delta_sigma_pr", dist.HalfNormal(0.2))
log_delta_z = numpyro.sample("log_delta_z", dist.Normal(0, 1).expand([n_participants]))
log_delta_i = log_delta_mu_pr + log_delta_sigma_pr * log_delta_z  # unconstrained
delta_i = jnp.exp(log_delta_i)   # strictly positive
b_i = A_i + delta_i              # enforces b > A
```

For the other LBA params (v_scale, A, t0), log-normal non-centered is appropriate:
- `v_scale` in `(0.1, 20.0)` — use log-scale: `log_v_mu_pr`, `log_v_sigma_pr`, `v_scale = exp(...)`
- `A` in `(0.001, 2.0)` — use log-scale (or the same bounded `sample_bounded_param` with explicit lower/upper)
- `t0` in `(0.05, 0.3)` — narrow range; probit-scale like `sample_bounded_param` works

The M6b pattern (kappa_total/kappa_share manual sampling with `mu_pr/sigma_pr/z`) is the
structural template. Use the same naming convention: `{param_name}_mu_pr`,
`{param_name}_sigma_pr`, `{param_name}_z`.

### MCMC.post_warmup_state API — Verified

From direct inspection of NumPyro MCMC source:

```python
# How post_warmup_state checkpoint works:
# --- WARMUP PHASE (first job run, or if job killed before samples collected) ---
mcmc.warmup(rng_key, **model_args)
# After warmup():
#   mcmc.post_warmup_state is set to mcmc._last_state
#   mcmc.post_warmup_state.rng_key holds the RNG state

import pickle
with open("m4_warmup_state.pkl", "wb") as f:
    pickle.dump(mcmc.post_warmup_state, f)

# --- SAMPLING PHASE (second run, skip warmup) ---
nuts_kernel = NUTS(model, target_accept_prob=0.95)
mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=1500, num_chains=4)
with open("m4_warmup_state.pkl", "rb") as f:
    state = pickle.load(f)
mcmc.post_warmup_state = state    # setter wires it to self._warmup_state
mcmc.run(state.rng_key, **model_args)  # warmup is skipped because post_warmup_state is set
```

The `MCMC.run()` source shows that when `_warmup_state` is set, the warmup phase is
skipped. The checkpoint file should be written after `warmup()` completes and before
`run()` starts sampling. In a 48h SLURM job, the practical approach is:

1. Check for `m4_warmup_state.pkl` at job start
2. If not found: run `warmup()`, save state, then run sampling
3. If found: load state, set `mcmc.post_warmup_state`, run sampling directly

This provides crash protection at the warmup/sampling boundary — the most likely
failure point for a 48h job.

### Pareto-k Gating — New Infrastructure

ArviZ `az.loo(idata, pointwise=True)` returns a result where the per-observation
Pareto-k values are in `loo_result.pareto_k` (array of shape matching log_likelihood).
The gating logic:

```python
loo_result = az.loo(idata, pointwise=True)
pareto_k = loo_result.pareto_k.values  # numpy array
frac_bad = (pareto_k > 0.7).mean()
if frac_bad > 0.05:
    print(f"WARNING: {100*frac_bad:.1f}% trials have Pareto-k > 0.7 — LOO unreliable for M4")
    # Fallback: report choice-only marginal log-likelihood
    ...
```

The "choice-only marginal" is obtained by running the M4 pointwise log-lik but summing
over the RT density component:

- The existing `wmrl_m4_block_likelihood` computes the JOINT log P(choice, RT). To
  marginalize over RT, the log-probability of the CHOICE ONLY (integrating out RT)
  equals `log sum_t f_chosen(t) * prod_{j!=chosen} S_j(t) dt`. This is not easily
  computed analytically from the existing code.
- **Practical fallback**: Use the MLE comparison AIC/BIC from `compare_mle_models.py`
  for the M4-vs-choice-only track, and report LOO for M4 as a standalone quality
  metric (not compared cross-model). This matches the STATE.md decision: "M4 cannot
  sit inside a unified az.compare table."
- The "choice-only marginal" in M4H-05 likely means: if Pareto-k fails, report LOO
  on CHOICE OUTCOMES ONLY (i.e., re-run pointwise loglik using only `log f_chosen`
  without the `sum_{j!=chosen} log S_j` survivor terms). This is a simpler fallback
  than true analytical marginalization.

### bayesian_diagnostics.py — M4 NOT Supported

`bayesian_diagnostics.py` imports ONLY from `jax_likelihoods.py` (not `lba_likelihood.py`):

```python
# bayesian_diagnostics.py lines 27-34
from scripts.fitting.jax_likelihoods import (
    q_learning_multiblock_likelihood_stacked,
    wmrl_multiblock_likelihood_stacked,
    wmrl_m3_multiblock_likelihood_stacked,
    wmrl_m5_multiblock_likelihood_stacked,
    wmrl_m6a_multiblock_likelihood_stacked,
    wmrl_m6b_multiblock_likelihood_stacked,
)
```

`_get_param_names` raises ValueError for `wmrl_m4`. `_get_likelihood_fn` raises
ValueError for `wmrl_m4`. `_build_per_participant_fn` raises ValueError for `wmrl_m4`.

Phase 17 must implement M4-specific versions of these functions inline in
`13_fit_bayesian_m4.py` rather than extending `bayesian_diagnostics.py` (float64
isolation again).

### bayesian_summary_writer.py — M4 IS Supported

`_MODEL_PARAMS["wmrl_m4"]` is already defined (lines 66–77) with the correct 10-parameter
list: `["alpha_pos", "alpha_neg", "phi", "rho", "capacity", "kappa", "v_scale", "A", "delta", "t0"]`.

`write_bayesian_summary` can be called directly for M4 after fitting, because it only
reads the idata posterior and parameter list — it does not touch float64 or likelihoods.

### Existing SLURM Pattern

`cluster/13_bayesian_m6b.slurm` is the template. M4 needs:
- `--time=48:00:00` (vs 8h for M6b)
- `--mem=96G` (vs 32G for M6b; LBA float64 NUTS is memory-intensive)
- `--gres=gpu:a100:1` (M4 needs GPU for float64 LBA; choice-only models are CPU-only)
- `--partition=gpu` (vs comp for choice-only)
- `export JAX_PLATFORMS=cuda` instead of `export JAX_PLATFORMS=cpu`
- Remove `export NUMPYRO_HOST_DEVICE_COUNT=1` (GPU, not CPU parallelism)
- Target script: `python scripts/13_fit_bayesian_m4.py` (not `fit_bayesian.py`)

---

## Exact Signatures

### wmrl_m4_multiblock_likelihood_stacked (lba_likelihood.py)

```python
def wmrl_m4_multiblock_likelihood_stacked(
    stimuli_stacked: jnp.ndarray,    # (n_blocks, max_trials), int32
    actions_stacked: jnp.ndarray,    # (n_blocks, max_trials), int32
    rewards_stacked: jnp.ndarray,    # (n_blocks, max_trials), float64
    set_sizes_stacked: jnp.ndarray,  # (n_blocks, max_trials), float32/int32
    rts_stacked: jnp.ndarray,        # (n_blocks, max_trials), float64 IN SECONDS
    masks_stacked: jnp.ndarray,      # (n_blocks, max_trials), float64 0.0/1.0
    alpha_pos: float,
    alpha_neg: float,
    phi: float,
    rho: float,
    capacity: float,
    kappa: float,
    v_scale: float,
    A: float,
    b: float,        # NOT delta! Caller computes b = A + delta
    t0: float,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
) -> float:
    """Returns total NLL (float64)."""
```

**Critical:** The likelihood takes `b` (threshold), not `delta`. The non-centered
hierarchical model samples `log_delta`, computes `delta = exp(log_delta)`, then
`b = A + delta` before passing to the likelihood. This decode happens in the participant
for-loop, same as M6b's kappa decode.

### WMRL_M4_BOUNDS (mle_utils.py)

```python
WMRL_M4_BOUNDS = {
    'alpha_pos':  (0.001, 0.999),
    'alpha_neg':  (0.001, 0.999),
    'phi':        (0.001, 0.999),
    'rho':        (0.001, 0.999),
    'capacity':   (2.0, 6.0),
    'kappa':      (0.0, 1.0),
    'v_scale':    (0.1, 20.0),
    'A':          (0.001, 2.0),
    'delta':      (0.001, 2.0),   # b - A gap; passed as b=A+delta to likelihood
    't0':         (0.05, 0.3),
}
```

These bounds inform the prior choices for the hierarchical model:
- `kappa` in [0,1]: same `sample_bounded_param` as M3 with `mu_prior_loc=-2.0`
- `capacity` in [2,6]: same `sample_capacity` as M3 with K-parameterization
- `v_scale`, `A`, `delta`, `t0`: must be sampled manually (not in PARAM_PRIOR_DEFAULTS)

### PARAM_PRIOR_DEFAULTS Coverage for M4

Parameters IN PARAM_PRIOR_DEFAULTS (can use `sample_bounded_param`):
- `alpha_pos`, `alpha_neg`, `phi`, `rho`, `epsilon` → use standard loop
- `capacity` → use `sample_bounded_param` with `lower=2.0, upper=6.0`
- `kappa` → use `sample_bounded_param` with `mu_prior_loc=-2.0`

Parameters NOT in PARAM_PRIOR_DEFAULTS (must sample manually):
- `v_scale`: log-normal, `log_v_scale ~ Normal(log(3.0), 0.5)` (centered near MLE group mean)
- `A`: log-normal or bounded probit, `A in (0.001, 2.0)`
- `delta`: non-centered log-scale (M4H-02 requirement), `log_delta ~ Normal(mu, sigma)`
- `t0`: bounded probit, `t0 in (0.05, 0.3)`

**Note:** M4 has NO epsilon parameter. The model has 10 params total (alpha_pos, alpha_neg,
phi, rho, capacity, kappa, v_scale, A, delta, t0).

### Data Column: RT

The `output/task_trials_long.csv` has an `rt` column (milliseconds as integer). The
`preprocess_rt_block` function from `lba_likelihood.py` converts to seconds and filters
outliers (150ms < RT < 2000ms). The M4 stacked data preparation must call this.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Non-centered bounded param sampling | Custom probit transform | `sample_bounded_param` from `numpyro_helpers.py` for the 6 RLWM params + kappa |
| RT outlier filtering | Manual thresholds | `preprocess_rt_block` from `lba_likelihood.py` |
| Schema-parity CSV | Custom writer | `write_bayesian_summary` — already handles `wmrl_m4` |
| Convergence gate | Custom rhat check | Copy the existing pattern from `save_results()` in `fit_bayesian.py` |
| Divergence bump loop | Custom retry | Copy `run_inference_with_bump` from `numpyro_models.py` — identical logic |
| Pareto-k computation | Manual | `az.loo(idata, pointwise=True).pareto_k` |

---

## Common Pitfalls

### Pitfall 1: float64 Import Order

**What goes wrong:** Any JAX array materialized before `jax.config.update("jax_enable_x64", True)` silently runs in float32. The LBA density then produces NaN/Inf because thresholds and drift rates need float64 precision.

**Why it happens:** JAX sets the dtype mode at first array creation. Numpyro and its imports create JAX arrays during initialization.

**How to avoid:** `jax.config.update("jax_enable_x64", True)` and `numpyro.enable_x64()` must be literally the first two statements in the M4 script after `import jax`. No `from scripts...import` before this.

**Warning signs:** `jnp.zeros(1).dtype == jnp.float32` in the integration test.

### Pitfall 2: b vs delta in Likelihood

**What goes wrong:** Passing `delta` as the `b` argument to `wmrl_m4_multiblock_likelihood_stacked` — the likelihood takes threshold `b`, not the gap `delta`.

**Why it happens:** MLE objective decodes `b = A + delta` before calling likelihood; hierarchical model must do the same decode in the participant for-loop.

**How to avoid:** Always compute `b_i = A_i + delta_i` inside the for-loop before calling the likelihood. Never pass `delta_i` as `b`.

### Pitfall 3: RTs in Wrong Units or Missing RT Stacking

**What goes wrong:** Passing RTs in milliseconds (not seconds), or failing to include `rts_stacked` in the participant data dict — causes shape errors or nonsensical LBA values.

**How to avoid:** Use `preprocess_rt_block` which returns seconds. Verify dtype is float64 after preprocessing.

### Pitfall 4: Adding wmrl_m4 to STACKED_MODEL_DISPATCH

**What goes wrong:** If `wmrl_m4` is added to `STACKED_MODEL_DISPATCH` in `fit_bayesian.py`, any import of `fit_bayesian.py` will try to dispatch M4 through choice-only infrastructure. The `bayesian_diagnostics.py` import chain does not load float64 and will silently downcast.

**How to avoid:** Keep M4 entirely in `13_fit_bayesian_m4.py`. Do not modify `STACKED_MODEL_DISPATCH`, `_get_param_names`, or `_get_likelihood_fn` in existing files.

### Pitfall 5: Pareto-k > 0.7 is Expected, Not a Bug

**What goes wrong:** Treating LBA Pareto-k failure as a code bug and spending time debugging the model.

**Why it happens:** LBA is a continuous density model. LOO is designed for discrete/bounded likelihoods. The RT density can be large (>> 1), causing extreme log-lik values that inflate Pareto-k.

**How to avoid:** Implement the fallback path as part of the design, not as an error handler. The convergence gate should PASS before the Pareto-k check. Document in output that M4 uses MLE AIC for cross-model comparison.

### Pitfall 6: chain_method='vectorized' vs 'parallel'

**What goes wrong:** Using `chain_method='parallel'` on a GPU node with 4 chains but 1 GPU — each chain tries to claim the GPU, causing OOM or serialization.

**How to avoid:** Use `chain_method='vectorized'` which vectorizes chains on the same device. This is what M4H-03 requires.

```python
mcmc = MCMC(
    nuts_kernel,
    num_warmup=1000,
    num_samples=1500,
    num_chains=4,
    chain_method='vectorized',  # NOT 'parallel'
    progress_bar=True,
)
```

### Pitfall 7: Pickle of post_warmup_state May Not Work Across JAX Versions

**What goes wrong:** `post_warmup_state` is a JAX PyTree; pickle may fail if JAX
version changes between warmup and sampling runs, or if the state contains non-picklable
GPU device arrays.

**How to avoid:** Use `jax.device_get(mcmc.post_warmup_state)` before pickling to move
arrays to CPU/numpy. Alternatively, serialize just the position and step size using
`mcmc.post_warmup_state.z` and reconstruct with known warmup results.

LOW confidence: the exact serialization behavior of `HMCState` is not verified against
NumPyro 0.20.1 specifically. Test with the integration test (M4H-04) before relying on
this pattern in production.

---

## Code Examples

### M4 NumPyro Model Skeleton (from M6b template)

```python
# 13_fit_bayesian_m4.py — must be at top before any other JAX
import jax
jax.config.update("jax_enable_x64", True)
import numpyro
numpyro.enable_x64()

from scripts.fitting.lba_likelihood import (
    wmrl_m4_multiblock_likelihood_stacked,
    preprocess_rt_block,
    FIXED_S,
)
from scripts.fitting.numpyro_helpers import (
    PARAM_PRIOR_DEFAULTS,
    phi_approx,
    sample_bounded_param,
)

def wmrl_m4_hierarchical_model(
    participant_data_stacked: dict,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
) -> None:
    n_participants = len(participant_data_stacked)
    participant_ids = sorted(participant_data_stacked.keys())

    # -- 6 standard params via sample_bounded_param loop --
    sampled = {}
    for param in ["alpha_pos", "alpha_neg", "phi", "rho", "capacity", "kappa"]:
        defaults = PARAM_PRIOR_DEFAULTS[param]
        sampled[param] = sample_bounded_param(
            param,
            lower=defaults["lower"],
            upper=defaults["upper"],
            n_participants=n_participants,
            mu_prior_loc=defaults["mu_prior_loc"],
        )

    # -- v_scale: log-normal non-centered --
    log_v_mu_pr = numpyro.sample("log_v_mu_pr", dist.Normal(jnp.log(3.0), 0.5))
    log_v_sigma_pr = numpyro.sample("log_v_sigma_pr", dist.HalfNormal(0.2))
    log_v_z = numpyro.sample("log_v_z", dist.Normal(0, 1).expand([n_participants]))
    sampled["v_scale"] = numpyro.deterministic(
        "v_scale", jnp.exp(log_v_mu_pr + log_v_sigma_pr * log_v_z)
    )

    # -- A: log-normal non-centered (bounds 0.001, 2.0) --
    log_A_mu_pr = numpyro.sample("log_A_mu_pr", dist.Normal(jnp.log(0.3), 0.5))
    log_A_sigma_pr = numpyro.sample("log_A_sigma_pr", dist.HalfNormal(0.2))
    log_A_z = numpyro.sample("log_A_z", dist.Normal(0, 1).expand([n_participants]))
    sampled["A"] = numpyro.deterministic(
        "A", jnp.exp(log_A_mu_pr + log_A_sigma_pr * log_A_z)
    )

    # -- delta = b - A: M4H-02 non-centered log-scale --
    log_delta_mu_pr = numpyro.sample("log_delta_mu_pr", dist.Normal(0.0, 1.0))
    log_delta_sigma_pr = numpyro.sample("log_delta_sigma_pr", dist.HalfNormal(0.2))
    log_delta_z = numpyro.sample("log_delta_z", dist.Normal(0, 1).expand([n_participants]))
    sampled["delta"] = numpyro.deterministic(
        "delta", jnp.exp(log_delta_mu_pr + log_delta_sigma_pr * log_delta_z)
    )

    # -- t0: bounded probit in [0.05, 0.3] --
    t0_mu_pr = numpyro.sample("t0_mu_pr", dist.Normal(0.0, 1.0))
    t0_sigma_pr = numpyro.sample("t0_sigma_pr", dist.HalfNormal(0.2))
    t0_z = numpyro.sample("t0_z", dist.Normal(0, 1).expand([n_participants]))
    sampled["t0"] = numpyro.deterministic(
        "t0", 0.05 + (0.3 - 0.05) * phi_approx(t0_mu_pr + t0_sigma_pr * t0_z)
    )

    # -- Likelihood: participant for-loop (M6b pattern) --
    for idx, pid in enumerate(participant_ids):
        pdata = participant_data_stacked[pid]
        A_i = sampled["A"][idx]
        delta_i = sampled["delta"][idx]
        b_i = A_i + delta_i   # decode: b > A enforced by delta > 0

        log_lik = wmrl_m4_multiblock_likelihood_stacked(
            stimuli_stacked=pdata["stimuli_stacked"],
            actions_stacked=pdata["actions_stacked"],
            rewards_stacked=pdata["rewards_stacked"],
            set_sizes_stacked=pdata["set_sizes_stacked"],
            rts_stacked=pdata["rts_stacked"],
            masks_stacked=pdata["masks_stacked"],
            alpha_pos=sampled["alpha_pos"][idx],
            alpha_neg=sampled["alpha_neg"][idx],
            phi=sampled["phi"][idx],
            rho=sampled["rho"][idx],
            capacity=sampled["capacity"][idx],
            kappa=sampled["kappa"][idx],
            v_scale=sampled["v_scale"][idx],
            A=A_i,
            b=b_i,
            t0=sampled["t0"][idx],
            num_stimuli=num_stimuli,
            num_actions=num_actions,
            q_init=q_init,
            wm_init=wm_init,
        )
        numpyro.factor(f"obs_p{pid}", log_lik)
```

### Checkpoint-and-Resume Pattern

```python
import pickle
from pathlib import Path

WARMUP_STATE_PATH = Path("output/bayesian/m4_warmup_state.pkl")

nuts_kernel = NUTS(wmrl_m4_hierarchical_model, target_accept_prob=0.95, max_tree_depth=10)
mcmc = MCMC(
    nuts_kernel,
    num_warmup=1000,
    num_samples=1500,
    num_chains=4,
    chain_method='vectorized',
    progress_bar=True,
)

rng_key = jax.random.PRNGKey(42)

if WARMUP_STATE_PATH.exists():
    print("Resuming from warmup checkpoint...")
    with open(WARMUP_STATE_PATH, "rb") as f:
        warmup_state = pickle.load(f)
    mcmc.post_warmup_state = warmup_state
    mcmc.run(warmup_state.rng_key, **model_args)
else:
    print("Running warmup phase...")
    mcmc.warmup(rng_key, **model_args)
    with open(WARMUP_STATE_PATH, "wb") as f:
        pickle.dump(mcmc.post_warmup_state, f)
    print("Warmup state saved. Running sampling...")
    mcmc.run(mcmc.post_warmup_state.rng_key, **model_args)
```

### Pareto-k Gating

```python
loo_result = az.loo(idata, pointwise=True)
pareto_k = loo_result.pareto_k.values
frac_bad = float((pareto_k > 0.7).mean())

if frac_bad > 0.05:
    print(f"M4H-05 FALLBACK: {100*frac_bad:.1f}% trials have Pareto-k > 0.7")
    print("M4 cannot be compared via LOO. Reporting M4 on separate AIC track only.")
    # Write gating metadata to JSON for Phase 18 consumption
    gating_meta = {
        "pareto_k_frac_bad": frac_bad,
        "loo_unreliable": True,
        "fallback": "choice_only_marginal_aic",
    }
else:
    print(f"M4H-05 PASS: {100*frac_bad:.1f}% Pareto-k > 0.7 (threshold: 5%)")
    gating_meta = {"pareto_k_frac_bad": frac_bad, "loo_unreliable": False}
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|-----------------|--------|
| M4 MLE only | M4 hierarchical NUTS | Posterior uncertainty on LBA params |
| Global float32 | Process-isolated float64 | LBA density numerics stable |
| Single long MCMC run | warmup/sampling split with checkpoint | Survives 48h wall time |
| LOO cross-model compare | Separate M4 track (AIC MLE + standalone LOO) | Correct (incommensurable likelihoods) |

---

## Open Questions

1. **Pickle reliability of `post_warmup_state` across GPU nodes**
   - What we know: `MCMC.post_warmup_state` is a JAX PyTree; `pickle` is the simplest
     serialization. `jax.device_get()` before pickling moves arrays to CPU.
   - What's unclear: Whether `HMCState` from NumPyro 0.20.1 is fully picklable when
     arrays live on GPU (A100).
   - Recommendation: Test M4H-04 integration test with `jax.device_get(state)` wrapping
     before pickle. If that fails, fall back to saving step size and position separately.

2. **v_scale prior centering**
   - What we know: MLE v_scale typically 2-8 (group mean ~3-5 from McDougle & Collins 2021
     precedent). `log_v_mu_pr ~ Normal(log(3.0), 0.5)` centers near this.
   - What's unclear: Whether N=154 data with variable RT distributions will favor very
     different v_scale. Wide prior (sigma=0.5 on log scale) should cover the range.
   - Recommendation: Check MLE v_scale posterior from Phase 14 fits before finalizing prior.

3. **chain_method='vectorized' memory footprint with N=154 participants**
   - What we know: M4H-03 specifies vectorized chains + A100 + 96G. Choice-only models
     use 32G on CPU. M4 LBA has 6x more parameters per trial (joint density).
   - What's unclear: Whether 96G is sufficient for 4 vectorized chains with N=154
     participants, 21 blocks, 18 trials each, float64.
   - Recommendation: Add a quick memory estimate to the SLURM script preamble. If OOM
     occurs, reduce to 2 chains (`num_chains=2`) at the cost of longer sampling.

4. **Choice-only marginal for Pareto-k fallback**
   - What we know: The exact definition of "choice-only marginal log-likelihood" for M4
     is ambiguous — could mean (a) log P(choice only) by analytical RT marginalization
     (hard), or (b) dropping survivor terms from the joint log-lik (approximate), or
     (c) simply using the MLE AIC comparison track.
   - What's unclear: What the planner intends by "fallback to choice-only marginal."
   - Recommendation: Default to option (c) — MLE AIC track — and document this clearly.
     Raise a warning in the Pareto-k fallback branch that flags this for Phase 18
     manuscript discussion.

---

## Sources

### Primary (HIGH confidence — direct codebase inspection)
- `scripts/fitting/lba_likelihood.py` — `wmrl_m4_multiblock_likelihood_stacked` exact signature
- `scripts/fitting/mle_utils.py` — `WMRL_M4_BOUNDS`, `WMRL_M4_PARAMS` exact values
- `scripts/fitting/numpyro_helpers.py` — `PARAM_PRIOR_DEFAULTS` content, LBA exclusion comment
- `scripts/fitting/numpyro_models.py` — `wmrl_m6b_hierarchical_model` (template), `prepare_stacked_participant_data`, `run_inference_with_bump`
- `scripts/fitting/bayesian_diagnostics.py` — M4 gap in `_get_param_names`, `_get_likelihood_fn`
- `scripts/fitting/bayesian_summary_writer.py` — `_MODEL_PARAMS["wmrl_m4"]` present
- `scripts/fitting/fit_bayesian.py` — `STACKED_MODEL_DISPATCH` (M4 absent), `save_results` pattern
- `scripts/fitting/fit_mle.py` — RT loading pattern (lines 2262–2291), float64 init in `fit_all_gpu_m4`
- `cluster/13_bayesian_m6b.slurm` — SLURM template for Bayesian jobs
- `cluster/12_mle_gpu.slurm` — GPU SLURM template
- NumPyro MCMC source (via `conda run -n ds_env python`) — `post_warmup_state` property, `warmup()` API
- `.planning/STATE.md` — locked decisions on M4 float64 ordering, Pareto-k expectation

### Secondary (MEDIUM confidence)
- RT column name `"rt"` in milliseconds confirmed from `output/task_trials_long.csv` header

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all from direct code inspection
- Architecture: HIGH — exact signatures verified, M6b template confirmed
- Float64 isolation: HIGH — locked in STATE.md + verified in lba_likelihood.py line 11
- post_warmup_state API: HIGH — verified from NumPyro source via conda env
- Pitfalls: HIGH (float64 order, b vs delta) — MEDIUM (pickle across GPU)
- Pareto-k fallback: MEDIUM — exact fallback semantics ambiguous; option (c) recommended

**Research date:** 2026-04-12
**Valid until:** 2026-05-12 (stable; NumPyro 0.20.1 pinned)
