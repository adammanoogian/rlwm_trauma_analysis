# Stack Research

**Domain:** Hierarchical Bayesian inference for computational cognitive models — extending an existing JAX/NumPyro pipeline to full population-level inference across 7 RLWM variants (M1, M2, M3, M5, M6a, M6b, M4-LBA) with trauma-as-Level-2 predictors, GPU-batched sampling, and Bayesian model comparison.
**Researched:** 2026-04-10
**Confidence:** HIGH (NumPyro / ArviZ versions, lax.scan inside numpyro.factor pattern, SLURM integration), MEDIUM (LBA-under-NUTS float64 numerical stability — pattern is sound but unvalidated for this specific likelihood at scale), HIGH (model comparison restrictions on M4 vs choice-only)

---

## Summary Verdict

**Add nothing JAX-side. Pin NumPyro to 0.20.x and ArviZ to 0.23.4. Do NOT migrate to ArviZ 1.0 in this milestone.** Extend the existing `numpyro_models.py` pattern (numpyro.factor over JAX-JIT'd likelihoods with internal `lax.scan`) to all 7 models. Add a separate `compute_pointwise_log_lik()` post-sampling utility because `numpyro.factor` does not expose pointwise log-likelihoods to ArviZ — required for WAIC/LOO. Reuse the existing GPU SLURM pattern (`cluster/12_mle_gpu.slurm`) as the template for new sampling jobs, with longer wall time and CPU-side multi-chain parallelism via `chain_method="vectorized"`.

The single most important constraint to plan around: **M4 LBA cannot be WAIC/LOO-compared against choice-only models even within the hierarchical framework**, because LBA's likelihood is over (choice, RT) pairs while M1-M3/M5/M6 likelihood is over choice only. This is a measure-theoretic incompatibility, not a software limitation, and is unchanged from the v3 milestone constraint.

---

## Recommended Stack

### Core Technologies (No Changes from v3)

| Technology | Current Version | Purpose | Status |
|------------|-----------------|---------|--------|
| JAX | 0.9.0 (latest stable: 0.9.2, March 18 2026) | JIT-compiled likelihoods, NUTS gradients via autodiff | **Keep at 0.9.0** — safe to upgrade to 0.9.2 if NumPyro 0.20.1 demands it; verified no breaking changes to `lax.scan`, `vmap`, `jit`, or x64 enable in 0.9.x |
| jaxlib | 0.9.0 | XLA backend | **Keep at 0.9.0** |
| NumPy | 2.3.5 | Data marshalling, post-processing | **Keep** — JAX 0.9 requires NumPy >= 2.0 |
| SciPy | 1.16.3 | L-BFGS-B (legacy MLE), normal CDF reference | **Keep** — used by 12_fit_mle.py only |
| Python | 3.11 | Runtime | **Keep** — NumPyro 0.20.x requires Python >= 3.11 |
| pandas | 2.0+ | Trial data loading | **Keep** |

### NEW / UPDATED Bayesian Stack

| Library | Version (Pin) | Purpose | Action |
|---------|---------------|---------|--------|
| **NumPyro** | **`numpyro==0.20.1`** | NUTS sampler + hierarchical model DSL on JAX | **PIN** — currently `>=0.13` in pyproject.toml `[bayesian]` extra; tighten to `==0.20.1` |
| **ArviZ** | **`arviz==0.23.4`** | InferenceData container, `compare`, `waic`, `loo`, `from_numpyro`, plot_trace, plot_posterior | **PIN exact** — current local install is 0.22.0; upgrade to 0.23.4 (Feb 4 2026). **DO NOT install 1.0+** in this milestone (see "What NOT to Use") |
| jax.scipy.stats.norm | bundled with JAX 0.9.0 | LBA density primitives (`pdf`, `cdf`, `sf`) under autodiff | **Already available** |
| `numpyro.contrib.control_flow.scan` | bundled with NumPyro 0.20.x | NOT needed — see "Why we don't use this" below | **N/A** |
| numpyro.infer.util.log_likelihood | bundled with NumPyro 0.20.x | Post-hoc pointwise log-lik computation for WAIC/LOO | **Required new dependency** on existing import (no install change) |
| PyMC | `pymc>=5.28` (already installed locally as 5.28.4) | Fast-preview Bayesian regression in `16b_bayesian_regression.py` only | **Keep as fallback backend**; do NOT use for hierarchical RLWM model fitting in v4.0 |
| pytensor | bundled with PyMC 5.28 | Backend for PyMC (CPU only) | **Keep** — supports the 16b PyMC fallback path only |

### Updated `pyproject.toml` `[bayesian]` Extra

Replace:
```toml
bayesian = [
    "pymc>=5.0",
    "arviz>=0.15",
    "numpyro>=0.13",
]
```

With:
```toml
bayesian = [
    # Hierarchical sampling stack (v4.0 hierarchical Bayesian milestone)
    "numpyro==0.20.1",     # PIN: NUTS + JAX backend; 0.20.1 is the latest as of 2026-03-25
    "arviz==0.23.4",       # PIN EXACTLY: do NOT upgrade to 1.0 (DataTree migration)
    # Fast-preview supplementary regression backend (16b only, NOT used for RLWM hierarchical fitting)
    "pymc>=5.28,<6.0",
    "pytensor>=2.30",      # PyMC 5.28 backend
]
```

Add to GPU env (`environment_gpu.yml` `pip:` section):

```yaml
    - jax[cuda12]>=0.5.0
    - jaxopt>=0.8.0
    - numpyro==0.20.1     # NEW: hierarchical NUTS sampling on GPU
    - arviz==0.23.4       # NEW: model comparison + diagnostics (CPU-side post-processing)
    - netcdf4             # NEW: required by arviz.InferenceData.to_netcdf()
```

PyMC is **deliberately omitted from `environment_gpu.yml`** — its pytensor backend is CPU-only and the GPU job has no use for it. PyMC stays on the CPU `environment.yml` only.

### Development Tools (No Changes)

| Tool | Purpose | Notes |
|------|---------|-------|
| ruff | Linting + formatting | Already ignoring `numpyro.*` typings in mypy override |
| mypy | Static typing | Already configured with `numpyro.*` ignore |
| pytest | Test runner | Reuse existing `requires_pymc` marker pattern for `requires_numpyro` if needed |

---

## Integration with Existing `numpyro_models.py`

The current implementation (located at `scripts/fitting/legacy/numpyro_models.py`) already establishes the correct architectural pattern for v4.0. Verbatim from the existing `qlearning_hierarchical_model`:

```python
log_lik = q_learning_multiblock_likelihood(
    stimuli_blocks=pdata['stimuli_blocks'],
    actions_blocks=pdata['actions_blocks'],
    rewards_blocks=pdata['rewards_blocks'],
    alpha_pos=alpha_pos_i,
    alpha_neg=alpha_neg_i,
    epsilon=epsilon_i,
    ...
)
numpyro.factor(f'obs_p{participant_id}', log_lik)
```

This is the **correct** pattern and should be reused for M3, M5, M6a, M6b, M4. The reasons:

1. **`q_learning_multiblock_likelihood` already uses `jax.lax.scan` internally**, but it does NOT contain `numpyro.sample` calls — only deterministic JAX ops returning a scalar log-likelihood. This sidesteps the well-known incompatibility between `jax.lax.scan` and `numpyro.sample` (see [State-space model thread](https://forum.pyro.ai/t/state-space-model-is-lax-scan-compatible-with-numpyro-sample/1758)) entirely.
2. **`numpyro.factor` correctly contributes to the joint log-density**, so NUTS gradients flow through the JAX likelihood via autodiff. No special handling required.
3. **`numpyro.contrib.control_flow.scan` is NOT needed** because the trial-level RNG is the participant's actual response, not a sampled latent — there are no `numpyro.sample` sites inside the scan to thread.

### What MUST be added (the trap)

**`numpyro.factor` does not register an obs site, so `arviz.from_numpyro` cannot extract pointwise log-likelihoods.** This means out-of-the-box `az.waic(idata)` and `az.loo(idata)` will fail with "no log_likelihood group found" — exactly the case described in [arviz #987](https://github.com/arviz-devs/arviz/issues/987) and [arviz #2196](https://github.com/arviz-devs/arviz/issues/2196).

**Required workaround pattern** (must be added to a new helper module, e.g., `scripts/fitting/bayesian_diagnostics.py`):

```python
import numpyro
from numpyro.infer.util import log_density
import arviz as az
import jax
import jax.numpy as jnp

def compute_pointwise_log_lik(
    model_fn,
    posterior_samples: dict,
    participant_data: dict,
    *,
    likelihood_fn,  # e.g. q_learning_pointwise_likelihood (returns shape [n_trials])
    num_stimuli: int,
    num_actions: int,
) -> dict:
    """
    Post-hoc pointwise log-likelihood for numpyro.factor-style models.

    Returns dict suitable for az.from_dict(log_likelihood={...}) with shape
    (num_chains, num_samples, num_participants, num_trials_padded).
    """
    n_chains, n_samples = next(iter(posterior_samples.values())).shape[:2]
    pids = list(participant_data.keys())

    def _per_sample(alpha_pos_i, alpha_neg_i, epsilon_i, pdata):
        # likelihood_fn must return per-trial log-prob (shape [T])
        return likelihood_fn(
            stimuli=pdata["stimuli_flat"],
            actions=pdata["actions_flat"],
            rewards=pdata["rewards_flat"],
            alpha_pos=alpha_pos_i,
            alpha_neg=alpha_neg_i,
            epsilon=epsilon_i,
            num_stimuli=num_stimuli,
            num_actions=num_actions,
        )

    # vmap over (chains, samples), then loop over participants (avoid axis-misalignment)
    vmapped = jax.vmap(jax.vmap(_per_sample, in_axes=(0, 0, 0, None)),
                       in_axes=(0, 0, 0, None))
    out = {}
    for i, pid in enumerate(pids):
        out[f"p{pid}"] = vmapped(
            posterior_samples["alpha_pos"][..., i],
            posterior_samples["alpha_neg"][..., i],
            posterior_samples["epsilon"][..., i],
            participant_data[pid],
        )
    return out
```

The new likelihood functions must return **per-trial log probabilities** (shape `[T]`), not a summed scalar. The existing `*_multiblock_likelihood` functions return scalars — they need to be either refactored to return per-trial vectors with a `.sum()` wrapper for the `numpyro.factor` site, or duplicated as `*_pointwise_likelihood`. The former is cleaner.

**This is the largest single source of v4.0 implementation risk.** It is not a stack problem, but it dictates the new module shape.

---

## LBA Under NUTS: float64 + vmap Strategy

### The Challenge

M4's LBA likelihood (already implemented in `scripts/fitting/lba_likelihood.py`) was built for MLE with `scipy.optimize.minimize` L-BFGS-B in float64. NUTS adds two requirements MLE didn't have:

1. **Smooth, finite gradients across the entire posterior support**, not just near the optimum.
2. **vmap-compatibility for `chain_method="vectorized"`** if multiple chains are run on a single GPU.

### Strategy

**1. Enable float64 globally before importing JAX.** This is the same strategy used in v3 for L-BFGS-B and is required for LBA's CDF tail precision regardless of optimizer/sampler. In the new `fit_bayesian_*.py` script for M4:

```python
import os
os.environ["JAX_ENABLE_X64"] = "true"
import jax
jax.config.update("jax_enable_x64", True)
import numpyro
numpyro.enable_x64()  # NumPyro-specific setting; mirrors JAX setting and is idempotent
import numpyro.distributions as dist
```

`numpyro.enable_x64()` is the canonical NumPyro-side toggle and is documented in [NumPyro Runtime Utilities](https://num.pyro.ai/en/stable/utilities.html). It must be called before any NumPyro distribution or sampler is constructed.

Confidence: HIGH — verified pattern across NumPyro forum threads and NumPyro source.

**2. Use `chain_method="vectorized"` (vmap) for multi-chain on a single GPU.** From the [NumPyro multi-GPU memory issue](https://github.com/pyro-ppl/numpyro/issues/1115) and forum guidance, vectorized chains run via `vmap` on one device and are the recommended approach for single-GPU clusters (which is what Monash M3 provides per node).

```python
mcmc = MCMC(
    NUTS(model_fn, target_accept_prob=0.95),  # higher than default 0.8 for LBA
    num_warmup=1500,
    num_samples=2000,
    num_chains=4,
    chain_method="vectorized",   # vmap on single GPU
    progress_bar=True,
)
```

The higher `target_accept_prob` (0.95 vs default 0.8) is needed because LBA's curvature varies sharply across the posterior. This is the same precaution used in PyMC LBA implementations and Stan dynamic accumulator models.

Confidence: MEDIUM — pattern is standard but unvalidated for THIS LBA likelihood. Phase 5 (validation) should include a divergence-rate check and a fall-back to `chain_method="parallel"` (jax.pmap across multiple GPUs) if vectorized chains divergence rate exceeds 5%.

**3. Numerical guardrails inherited from v3.** All four guardrails listed in the v3 STACK.md still apply:
- Use `jax.scipy.stats.norm.cdf(-z)` not `norm.sf(z)` (defensive)
- Safe-mask `jnp.where` pattern for `t' = t - t0 <= 0`
- Clamp drift rates with `softplus` or `jnp.maximum(v, 1e-6)`
- Clip log-density with `jnp.maximum(density, 1e-10)` before `jnp.log`

These are all unchanged from v3 and are already implemented in `lba_likelihood.py` for the MLE path.

**4. Sampler tuning for LBA — Critical caveat.** LBA-under-HMC is known to have pathological geometry near the boundary of valid `(t0, A, b)` configurations, where `b - A` approaches zero. Two recommendations:
- Use **non-centered parameterization** for the LBA threshold offset `b - A` (sample `log(b - A)` directly), not centered. This is the standard fix for boundary-driven funnel pathologies.
- Run a **short (200-sample) test fit on 5 participants** before the full run to surface divergences early. Phase 5 of v4.0 should include this as a milestone gate.

Confidence: MEDIUM — non-centered LBA is documented in PyMC LBA blog posts but no specific NumPyro reference exists. The pattern is mathematically standard.

---

## Bayesian Model Comparison: WAIC / LOO Strategy

### The Two Comparison Tracks (Same as MLE — Cannot Be Merged)

**Track A — Choice-only models (M1, M2, M3, M5, M6a, M6b):**
Each model assigns a probability to the observed action conditional on (state, history). Pointwise log-lik is `log P(action_t | state_t, history_<t, theta)`. WAIC and LOO are directly comparable across these 6 models because they share the same observation space (one categorical per trial).

```python
# After fitting all six models hierarchically:
idata_dict = {
    "M1": idata_qlearning,
    "M2": idata_wmrl,
    "M3": idata_wmrl_m3,
    "M5": idata_wmrl_m5,
    "M6a": idata_wmrl_m6a,
    "M6b": idata_wmrl_m6b,
}
comparison = az.compare(idata_dict, ic="loo", method="stacking")
# az.plot_compare(comparison)
```

`az.compare` with `ic="loo"` uses PSIS-LOO-CV (Pareto-smoothed importance sampling) which is preferred over WAIC for hierarchical RL models because the WAIC effective sample size becomes unstable when individual-level parameters are tightly informed by the prior. Both should be reported; LOO is the primary criterion. `method="stacking"` returns Bayesian stacking weights, which are more interpretable than raw WAIC differences for non-nested model families.

**Track B — M4 LBA (separate, never compared with Track A):**
M4's likelihood is over `(choice, RT)` jointly. The pointwise log-lik has the form `log p(choice_t, rt_t | state_t, history_<t, theta)`, which lives in a different measure space than choice-only models. Computing WAIC for M4 is well-defined, but **comparing it to a choice-only WAIC is meaningless** — you would be comparing densities measured over different reference measures, like comparing a discrete log-pmf to a continuous log-pdf.

This is the same constraint that exists in the MLE pipeline (`14_compare_models.py --m4` only compares M4 to itself across initializations), and it does not change under hierarchical Bayes. It is a property of the observation models, not a software limitation.

**The valid M4 comparisons are:**
- M4 vs an alternative joint accumulator model (e.g., a future RDM implementation) — same observation space, same measure
- M4's posterior predictive RT distributions vs observed RT distributions — model adequacy check, not model comparison
- M4's marginal choice predictions vs M5/M6's choice predictions — qualitative cross-track diagnostic only, not a formal information criterion

### What about Bayes Factors / marginal likelihoods?

Skip them. Bayes factors require marginal likelihood computation (bridge sampling, thermodynamic integration, or sequential Monte Carlo), all of which are expensive and notoriously sensitive to prior specification in hierarchical RL models. PSIS-LOO via `az.loo` is the field standard (Vehtari, Gelman & Gabry 2017) and is what the existing arviz API supports natively.

Confidence: HIGH — this is established practice in the computational modeling literature, codified in arviz's design.

### Pareto-k Diagnostic Required

PSIS-LOO returns a Pareto-k diagnostic per observation. Values > 0.7 indicate the importance sampling estimate for that observation is unreliable. **The v4.0 reporting pipeline must include**:

```python
loo_result = az.loo(idata, pointwise=True)
n_bad = (loo_result.pareto_k > 0.7).sum().item()
if n_bad > 0:
    print(f"WARNING: {n_bad}/{loo_result.pareto_k.size} obs have Pareto-k > 0.7")
    print("Consider refitting with reloo or moment-matching importance sampling")
```

Bad Pareto-k values are common in hierarchical RL models with few participants per group. This is not a stack problem — it's a "be prepared to report it honestly" problem. Plan a `reloo` (refit-LOO) fallback path; arviz 0.23.4 supports this natively via `az.reloo`.

Confidence: HIGH — verified in arviz 0.23 documentation.

---

## Cluster / SLURM Integration

### Reuse `cluster/12_mle_gpu.slurm` as the template

The existing GPU SLURM script does most of what's needed already:
- Conda env activation with multi-path fallback
- JAX device verification
- JAX compilation cache configured (`JAX_COMPILATION_CACHE_DIR` on `/scratch`)
- nvidia-smi background monitoring every 5 min
- Per-model job dispatch via `--export=MODEL=...`
- Stale-checkpoint cleanup with `KEEP_CHECKPOINTS` opt-out

The new `cluster/13_bayesian_gpu.slurm` should be a copy of `12_mle_gpu.slurm` with these changes:

```diff
- #SBATCH --time=12:00:00
+ #SBATCH --time=48:00:00          # Hierarchical NUTS is slow per chain
- #SBATCH --mem=64G
+ #SBATCH --mem=96G                 # Pointwise log-lik storage for 154 participants × ~480 trials × 8000 samples
+ #SBATCH --gres=gpu:a100:1         # Pin to A100 (40GB VRAM); LBA float64 will not fit on V100 (16GB)
- python scripts/fitting/fit_mle.py \
+ python scripts/fitting/fit_bayesian.py \
-     --n-starts 50 \
+     --chains 4 --warmup 1500 --samples 2000 \
```

### Multi-Model Dispatch (Already Solved)

The existing dispatch pattern in `12_mle_gpu.slurm` (lines 165-194: "Parallel Dispatch: if multiple models, submit separate jobs") should be replicated unchanged. **One SLURM job per model** is the correct pattern because:
1. NumPyro JIT compilation is per-model and cannot be shared
2. A failed M4 job should not block successful M3/M5 jobs
3. SLURM accounting is per-job, making cost tracking trivial
4. Each model gets its own JAX compilation cache subdirectory automatically (cache key includes model code hash)

### Wall Time Budget Validation

The v4.0 milestone estimates 50-96 GPU hours for a full sampling run. Sanity check with v3 numbers:

| Model | MLE wall time (50 starts, 154 ppts, A100) | Estimated NUTS wall time (4 chains × 3500 samples) |
|-------|-------------------------------------------|----------------------------------------------------|
| M1 (3 params) | ~30 min | ~3-4 hours |
| M2 (6 params) | ~2 hours | ~6-10 hours |
| M3 (7 params) | ~2.5 hours | ~8-12 hours |
| M5 (8 params) | ~3 hours | ~10-14 hours |
| M6a / M6b (7-8 params) | ~3 hours | ~10-14 hours each |
| M4 LBA (10 params, float64) | ~12-24 hours | **~24-48 hours** (LBA gradient is expensive AND float64 doubles memory bandwidth pressure) |

Sum across all 7 models: ~70-110 GPU hours per full run. The 50-96h estimate in the milestone document is **at the optimistic end** — plan for the upper bound, and budget at least one rerun (so ~150-200 GPU hours total). Trauma-as-Level-2 predictors add ~10-20% to runtime per model.

### Long-Job Risks

**Job timeout risk (48h SLURM limit on gpu partition):** M4 LBA may exceed 48h on a single A100. Mitigations in priority order:
1. Start with `num_warmup=1000, num_samples=1500` for M4 specifically (not 1500/2000) — faster while still adequate for diagnostics
2. Use `chain_method="parallel"` across 2 GPUs via `--gres=gpu:a100:2` if vmap-vectorized chains hit memory ceiling
3. Save MCMC state and restart pattern: NumPyro supports `mcmc.post_warmup_state` → save → reload to skip warmup on restart. Use this for M4 if first run hits the wall clock.

**JAX compilation cache invalidation:** The cache key includes JAX version, model bytecode, and shape signatures. Any change to a model function or upgrading JAX 0.9.0 → 0.9.2 invalidates the cache. The existing cache infrastructure handles this correctly; just be aware that cache misses add ~2-5 minutes per model on first run after any code change.

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Sampler | NumPyro NUTS (keep) | Stan via cmdstanpy | Stan would require duplicating all 7 likelihood functions in Stan code, maintaining two implementations, and abandoning JAX autodiff. The existing JAX likelihoods are validated by the MLE pipeline. |
| Sampler | NumPyro NUTS (keep) | PyMC NUTS via JAX backend (`sample_numpyro_nuts`) | Pulls in PyMC + pytensor as a dependency for no benefit; the JAX likelihoods would need to be wrapped in pytensor Op classes. NumPyro is a more direct fit. |
| Sampler | NumPyro NUTS (keep) | Blackjax NUTS | Blackjax is lower-level and would require manual potential function construction. NumPyro's effect-handler-based DSL is the right abstraction level for hierarchical models. |
| Posterior post-processing | ArviZ 0.23.4 (pin) | ArviZ 1.0.0 | 1.0 replaces `InferenceData` with `xarray.DataTree`. This is a breaking change to every script that touches `idata.posterior` access patterns. Migration is a v5.0 task at earliest, after the 1.0 ecosystem stabilizes. |
| Multi-chain strategy | `chain_method="vectorized"` (vmap, single GPU) | `chain_method="parallel"` (pmap, multi-GPU) | Multi-GPU is rarely available on M3 and adds scheduling complexity. Vectorized vmap is the documented pattern for single-GPU multi-chain (NumPyro issue #1115). |
| Multi-chain strategy | `chain_method="vectorized"` | `chain_method="sequential"` | Sequential is 4x slower for 4 chains and provides no advantage on GPU. Only relevant for memory-constrained CPU runs. |
| Pointwise log-lik | Manual `compute_pointwise_log_lik()` post-hoc | `numpyro.sample("obs", ..., obs=actions)` Categorical site | Would require rewriting every likelihood function as a NumPyro distribution, abandoning the existing JAX `lax.scan` likelihoods that the MLE pipeline depends on. The post-hoc approach reuses the existing MLE likelihood code unchanged. |
| Trauma covariates | Joint Level-2 regression in NumPyro (recommended for v4.0) | Two-stage: fit individual params, then regress (current 16b approach) | Two-stage ignores parameter uncertainty and underestimates standard errors. Joint hierarchical regression propagates uncertainty correctly and is the field standard for computational psychiatry (Daw 2011, Wilson & Collins 2019). Keep 16b as a fast supplementary check, not the primary inference. |
| float64 toggle | `numpyro.enable_x64()` + JAX env var | JAX-only `jax.config.update("jax_enable_x64", True)` | Both work but NumPyro internally constructs distributions during sampling. `numpyro.enable_x64()` ensures NumPyro's distribution constructors respect the setting. Use both as belt-and-braces. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **ArviZ 1.0.0** | Replaces `InferenceData` with `xarray.DataTree`. Breaks every `az.from_numpyro`, `az.compare`, `az.plot_*` call pattern. Migration guide exists but is a v5.0 task. The existing `samples_to_arviz()` function in `numpyro_models.py` returns `InferenceData` and would need full rewrite. | Pin `arviz==0.23.4` |
| `numpyro.contrib.control_flow.scan` inside the trial loop | Solves a problem you don't have. The existing pattern (JAX `lax.scan` inside a numpyro.factor) works because there are no `numpyro.sample` sites at the trial level — trial-level "randomness" is the participant's actual response, fed in as data. | Keep the existing `numpyro.factor` over JIT'd JAX likelihood pattern |
| `numpyro.sample("obs_p1", dist.Categorical(probs), obs=actions_p1)` per-participant | Would need a custom Categorical wrapper that runs the full RL update sequence to compute `probs` for each trial. This is what the JAX likelihood already does, but in JAX-native form. Wrapping it as a NumPyro distribution adds complexity for no benefit and breaks JIT compilation. | `numpyro.factor` + post-hoc pointwise computation |
| Two-stage trauma analysis as primary inference | Ignores individual-level parameter uncertainty. Standard errors are anti-conservative. | Joint hierarchical regression with trauma scales as Level-2 predictors |
| Stan / cmdstanpy | Would require duplicating 7 likelihood implementations in Stan and maintaining both | Reuse the JAX likelihoods via NumPyro |
| WAIC/LOO comparison of M4 vs choice-only models | M4 lives in `(choice, RT)` measure space; choice-only models live in choice measure space. Comparing log-densities across measure spaces is mathematically meaningless. | Run M4 in its own `az.compare()` group (currently a singleton); compare M4 to itself via Pareto-k diagnostics and posterior predictive checks |
| `chain_method="parallel"` on Monash M3 single-GPU partition | Requires multiple GPUs in the same job. M3's `gpu` partition typically allocates one GPU per job. Will silently fall back to sequential if pmap can't find enough devices. | `chain_method="vectorized"` (vmap on one GPU) |
| Pinning JAX > 0.9.2 in this milestone | Risk of NumPyro 0.20.x incompatibility with JAX > 0.9.2. NumPyro 0.20 minimum is JAX 0.7, but upper bound is not always tested. Lock to a known-good combo. | `jax==0.9.0` (current) or upgrade to `jax==0.9.2` only after a smoke test of the existing MLE pipeline on 0.9.2 |
| `numpyro.factor(..., obs=...)` (does not exist) | A common confusion. `numpyro.factor` does NOT have an `obs` argument. It always contributes to the joint, never to a separate observation site. | Use `numpyro.sample("obs", dist.Distribution(...), obs=data)` for observation sites OR `numpyro.factor("name", log_lik_value)` for direct log-density contribution — they are different primitives |

---

## Stack Patterns by Model

**M1 (Q-learning, 3 params), M2 (WM-RL, 6 params), M3 (M2 + κ, 7 params):**
- Reuse `numpyro_models.py` patterns directly
- `chain_method="vectorized"`, `num_warmup=1000`, `num_samples=2000`
- float32 OK (existing default) — these are the fastest models

**M5 (M3 + φ_rl, 8 params), M6a (M3 + κ_s, 7 params), M6b (M3 + dual κ, 8 params):**
- Same stack as M2/M3 but new model functions
- Higher dimensional priors require slightly longer warmup: `num_warmup=1500, num_samples=2000`
- Watch for funnel pathologies in `φ_rl` and `κ_s` — both are bounded scaled parameters that may need non-centered parameterization
- float32 OK

**M4 (RLWM-LBA, 10 params, joint choice+RT):**
- **MUST set float64 BEFORE first JAX import**
- `chain_method="vectorized"` with `num_chains=4`, `num_warmup=1500`, `num_samples=2000`, `target_accept_prob=0.95`
- Non-centered parameterization for `b - A` (the LBA threshold offset)
- Separate SLURM job, separate `compare()` track
- A100 GPU only (40GB VRAM minimum)

---

## Version Compatibility Matrix

| Package | Pinned Version | Compatible With | Notes |
|---------|----------------|-----------------|-------|
| Python 3.11 | environment_gpu.yml | NumPyro 0.20 (requires >=3.11), ArviZ 0.23 (requires >=3.10) | OK |
| JAX 0.9.0 | requirements.txt | NumPyro 0.20.1 (requires JAX >=0.7), jaxlib 0.9.0 | OK |
| jaxlib 0.9.0 | requirements.txt | JAX 0.9.0 | Must match JAX version |
| jaxopt 0.8.5 | environment_gpu.yml (deprecated but functional) | JAX 0.9.0 | Used by MLE only; NUTS path doesn't touch it |
| NumPyro 0.20.1 | NEW PIN | JAX >=0.7, Python >=3.11 | Released 2026-03-25 |
| ArviZ 0.23.4 | NEW PIN | numpy>=2.0, xarray, scipy>=1.13 | Released 2026-02-04. **Do NOT upgrade to 1.0** |
| PyMC 5.28.4 | locally installed | pytensor, NumPy 2.x | CPU only; used by 16b only |
| NumPy 2.3.5 | requirements.txt | JAX 0.9.0, ArviZ 0.23.4, NumPyro 0.20.1 | OK across the stack |
| netcdf4 | NEW (add to env) | ArviZ 0.23.4 InferenceData.to_netcdf() | Required for saving posterior samples to disk |

### Known compatibility gotchas

- **NumPyro 0.20.0 → 0.20.1 release notes mention "FIX: JAX pxla is deprecated from 0.8.2 onwards"** — confirms NumPyro 0.20.1 has been updated for JAX 0.9.x compatibility. Safe to use with JAX 0.9.0.
- **JAX 0.9.x is the first version where `jax.random.PRNGKey` is fully deprecated in favor of `jax.random.key`.** NumPyro 0.20 has been updated to use `jax.random.key` internally. Any user code in `numpyro_models.py` that calls `jax.random.PRNGKey(seed)` should be updated to `jax.random.key(seed)` — this is a one-line audit per model.
- **PyMC 5.28 + ArviZ 0.23 compatibility is verified locally** (the existing 16b script works against this combo). No changes needed for the PyMC fallback path.

---

## Installation

### Local CPU (development, fast preview)

```bash
# Existing environment.yml — add to pip section:
pip install numpyro==0.20.1 arviz==0.23.4 netcdf4
```

### Cluster GPU (full hierarchical sampling)

```bash
# Update environment_gpu.yml pip section to:
#   - jax[cuda12]>=0.5.0
#   - jaxopt>=0.8.0
#   - numpyro==0.20.1
#   - arviz==0.23.4
#   - netcdf4

# Recreate the env on M3:
mamba env remove -n rlwm_gpu
mamba env create -f environment_gpu.yml
conda activate rlwm_gpu

# Verify the new stack:
python -c "
import jax
jax.config.update('jax_enable_x64', True)
import numpyro
numpyro.enable_x64()
import arviz as az
print(f'JAX: {jax.__version__} (x64={jax.config.x64_enabled})')
print(f'NumPyro: {numpyro.__version__}')
print(f'ArviZ: {az.__version__}')
print(f'JAX devices: {jax.devices()}')
"

# Expected output:
# JAX: 0.9.0 (x64=True)
# NumPyro: 0.20.1
# ArviZ: 0.23.4
# JAX devices: [CudaDevice(id=0)]
```

### Verify NUTS + lax.scan + numpyro.factor pattern works

```bash
# Run existing test suite which already exercises this pattern via fit_bayesian.py:
python -m pytest scripts/fitting/tests/test_wmrl_model.py -v --tb=short
```

If this test passes against the new pin, the integration is sound and you can proceed to Phase 2 (extending to M3, M5, M6a, M6b, M4).

---

## Sources

**Verified via Context7-equivalent (PyPI direct):**
- `pypi.org/project/numpyro/` — NumPyro 0.20.1 released 2026-03-25, requires Python >=3.11, JAX >=0.7 (HIGH confidence)
- `pypi.org/project/arviz/` — ArviZ 0.23.4 released 2026-02-04, ArviZ 1.0.0 released 2026-03-02 with breaking changes (HIGH confidence)
- `pypi.org/project/jax/` (verified via [JAX changelog](https://github.com/jax-ml/jax/blob/main/CHANGELOG.md)) — JAX 0.9.2 latest stable (March 18 2026), no breaking changes to lax.scan/vmap/jit/x64 in 0.9.x (HIGH confidence)

**NumPyro behavior verified via official docs / GitHub:**
- [NumPyro 0.20.0 release notes](https://github.com/pyro-ppl/numpyro/releases) — confirmed minimum JAX 0.7, deprecation fixes for JAX pxla (HIGH confidence)
- [NumPyro Pyro forum: lax.scan compatibility](https://forum.pyro.ai/t/state-space-model-is-lax-scan-compatible-with-numpyro-sample/1758) — confirmed `numpyro.contrib.control_flow.scan` is needed only when `numpyro.sample` sites are inside the scan; pure JAX `lax.scan` inside a `numpyro.factor` is fine (HIGH confidence)
- [NumPyro issue #1115: multi-GPU memory](https://github.com/pyro-ppl/numpyro/issues/1115) — confirms `chain_method="vectorized"` is the documented pattern for single-GPU multi-chain; `post_warmup_state` workaround for memory pressure (HIGH confidence)

**ArviZ behavior verified via official docs / GitHub:**
- [arviz issue #2196: from_numpyro log_likelihood ignored](https://github.com/arviz-devs/arviz/issues/2196) — confirmed `arviz.from_numpyro(log_likelihood=...)` accepts only boolean, not custom dict; bug remains open. Workaround: post-hoc `add_groups` (HIGH confidence)
- [arviz issue #987: WAIC for multiple observed variables](https://github.com/arviz-devs/arviz/issues/987) — confirmed pointwise log-lik group structure required for `az.waic(pointwise=True)` (HIGH confidence)
- [ArviZ 0.23 docs: arviz.waic, arviz.loo, arviz.compare](https://python.arviz.org/en/stable/api/generated/arviz.waic.html) — confirmed API stability through 0.23.4; replaced by arviz-stats package in 1.0 (HIGH confidence)
- [ArviZ 1.0 migration notes](https://python.arviz.org/en/latest/user_guide/migration_guide.html) — confirmed `InferenceData` → `xarray.DataTree` breaking change, splitting into arviz-base/arviz-stats/arviz-plots (HIGH confidence based on release notes; full migration guide is 403 from automated fetcher)

**LBA-under-NUTS pattern (carried from v3 STACK.md):**
- [JAX issue #17199](https://github.com/jax-ml/jax/issues/17199) — `norm.sf` precision (HIGH confidence)
- [JAX discussions #6778, #5039](https://github.com/jax-ml/jax/discussions/5039) — `jnp.where` NaN gradient pattern (HIGH confidence)
- McDougle & Collins (2021) — LBA parameters and MATLAB fmincon precedent (MEDIUM confidence for NUTS adaptation)

**Inferred / contextual (lower confidence):**
- 50-200 GPU hour budget estimate — extrapolated from v3 MLE wall times and ratio of MLE-vs-NUTS in similar published models. **MEDIUM confidence**, plan for upper bound.
- Non-centered parameterization for LBA `b - A` boundary funnel — standard practice in PyMC LBA blog posts but no specific NumPyro reference. **MEDIUM confidence**, validate empirically in Phase 5.
- A100 vs V100 GPU choice — based on 40GB VRAM requirement for 154-participant pointwise log-lik storage in float64. **HIGH confidence** for memory math, **MEDIUM confidence** that V100 will OOM (untested).

---

*Stack research for: v4.0 Hierarchical Bayesian Pipeline & LBA Acceleration*
*Researched: 2026-04-10*
*Building on: v3 STACK.md (M4-M6 LBA implementation) and existing numpyro_models.py M1/M2 hierarchical pattern*
