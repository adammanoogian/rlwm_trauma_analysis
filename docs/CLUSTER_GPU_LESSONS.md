# Cluster, GPU, and JAX Performance — Lessons from v4.0

Operational writeup for running JAX + NumPyro hierarchical Bayesian fits on
the Monash M3 cluster.  Captures what actually caused slowness, how we
fixed it, and the pitfalls that were subtle enough to bite again if
forgotten.

Scope: the rlwm_trauma_analysis v4.0 pipeline (choice-only models
M1/M2/M3/M5/M6a/M6b and joint choice+RT model M4).  Many of the
patterns generalize to any hierarchical NumPyro project.

> **How to read this document.**  Each pitfall is paired with its
> **context** — the reason the failing pattern looked correct at the
> time.  Apply the rules to situations matching the context, not
> summarily.  JAX/NumPyro has enough sharp edges that "always do X"
> rules are misleading without the *why*.

---

## 1. The 1000× qlearning speedup — what actually happened

Production smoke tests on 2026-04-16 timed out after 6 hours at
iteration 65/100 of qlearning warmup.  Per-iteration cost: ~6 minutes.
After Issue 1 fix (commit `c54ee6c`), the same MCMC iterations run in
~0.5 seconds.  Root cause and resolution are the single most important
lesson in this document.

### Root cause: participant-level for-loop emits N trace sites

The pre-refactor hierarchical model looked like this:

```python
# scripts/fitting/numpyro_models.py, legacy pattern (DO NOT REVIVE)
for idx, pid in enumerate(sorted(participant_data_stacked.keys())):
    pdata = participant_data_stacked[pid]
    log_lik = wmrl_m3_multiblock_likelihood_stacked(
        stimuli_stacked=pdata["stimuli_stacked"],
        actions_stacked=pdata["actions_stacked"],
        ...
        alpha_pos=sampled["alpha_pos"][idx],
        ...
    )
    numpyro.factor(f"obs_p{pid}", log_lik)
```

For N=154 participants, this emits **154 separate trace sites** at
`numpyro.factor(f"obs_p{pid}")`.  During NUTS leapfrog integration,
each tree-doubling step evaluates the joint log-density, which
requires reading *all* 154 factor values.  With tree depth 6, a single
NUTS iteration performs up to `2^6 - 1 = 63` leapfrog steps, each
touching 154 factor sites → 9,700+ GPU kernel dispatches per MCMC
iteration.  On the L40S, each dispatch takes ~5–10 ms of overhead
(synchronize, launch, copy result).  That's 1–2 minutes of dispatch
time per iteration before any math happens.

**Why this looked correct.**  The pattern is idiomatic NumPyro: each
participant's likelihood gets its own `factor`.  It runs fine on CPU
(dispatch overhead amortized across larger kernels) and for small
models.  The bug only appears under two conditions: (a) GPU execution
where dispatch overhead dominates, and (b) many trace sites (N ≳ 20).
It passed unit tests because unit tests use N=3–5 synthetic data.

### Fix: collapse N factors into one via nested vmap

```python
# scripts/fitting/jax_likelihoods.py — fully-batched pattern
def wmrl_m3_fully_batched_likelihood(
    stimuli,       # (N, B, T)
    actions,       # (N, B, T)
    ...
    alpha_pos,     # (N,) per-participant scalar
    ...
):
    def _block_ll(stim, act, rew, ss, mask, ap, an, ph, rh, cap, k, e):
        return wmrl_m3_block_likelihood(
            stimuli=stim, actions=act, ...
            mask=mask, return_pointwise=False,
        )
    # Inner vmap: over blocks within a participant.  Data axes: 0, params: None.
    _over_blocks = jax.vmap(
        _block_ll,
        in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None, None),
        out_axes=0,
    )
    def _participant_ll(stim, act, rew, ss, mask, ap, an, ph, rh, cap, k, e):
        return _over_blocks(stim, act, rew, ss, mask, ap, an, ph, rh, cap, k, e).sum()
    # Outer vmap: over participants.  Everything on axis 0.
    _over_participants = jax.vmap(
        _participant_ll,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        out_axes=0,
    )
    return _over_participants(stimuli, actions, rewards, set_sizes, masks,
                              alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon)
```

Caller uses *one* trace site:

```python
# scripts/fitting/numpyro_models.py, current pattern
per_participant_ll = wmrl_m3_fully_batched_likelihood(...)
numpyro.factor("obs", per_participant_ll.sum())
```

### Why the speedup was 1000×, not 154×

Hypothesis going in: 154 kernel dispatches → 1 kernel dispatch should
give ~154× speedup.  Actual speedup: ~700–1000× on qlearning.

The extra gain comes from XLA fusion.  With 154 separate factor
closures, XLA compiles 154 independent sub-graphs and cannot reorder
or fuse operations across them.  With a single fused vmap call, XLA
sees one big graph and can:

- Hoist loop-invariant computations out of the participant loop
- Share intermediate memory buffers
- Schedule kernel execution to saturate SMs instead of launching
  one-shot kernels

The GPU was also idle 99% of wall time in the old code, waiting for
Python dispatches.  In the new code, kernels run back-to-back.  Peak
memory went from 0.1 MB → tens of MB, which is the GPU doing actual
work.

### Takeaway rule

**In NumPyro, every Python loop in the model body creates N trace
sites.**  If N ≳ 10 and you run on GPU, collapse the loop into one
`numpyro.factor("obs", batched_fn(...).sum())` call via vmap.  This is
not optional for production-scale hierarchical models.

---

## 2. The chain_method decision tree

NumPyro NUTS MCMC has three `chain_method` options:

| Method | Execution | Cost model |
|---|---|---|
| `"parallel"` | pmap across devices (one chain per device) | 1× per-device time, ≥ n_chains devices required |
| `"vectorized"` | vmap across chains on the same device | ~1× per-chain time × n_chains per device |
| `"sequential"` | Python for-loop over chains | n_chains × per-chain time |

Our `_select_chain_method` helper (`numpyro_models.py:67–94`) picks:

```python
def _select_chain_method(num_chains: int) -> str:
    backend = jax.default_backend()           # "cpu", "gpu", "tpu"
    n_devices = jax.local_device_count()
    if backend == "gpu":
        if n_devices >= num_chains:
            return "parallel"    # true pmap
        return "vectorized"      # chains time-multiplexed on 1 GPU
    if backend == "tpu":
        return "parallel" if n_devices >= num_chains else "vectorized"
    # CPU: need set_host_device_count() to see >1 device
    return "parallel" if n_devices >= num_chains else "sequential"
```

### Pitfall: `numpyro.set_host_device_count(4)` is CPU-only

**Context.**  The call tells XLA to expose 4 virtual host devices so
that CPU pmap works.  On GPU it is a **no-op** — `n_devices` is
determined by SLURM `--gres=gpu:N` allocation.

**Consequence.**  If you write:

```python
numpyro.set_host_device_count(args.chains)   # chains=4
# ... later:
chain_method = _select_chain_method(4)
# On a single-GPU SLURM job:
# backend=gpu, n_devices=1 (not 4!), chain_method="vectorized"
```

you do NOT get pmap parallelism.  You get 4 chains time-multiplexed on
one GPU.  Wall time = 4× per-chain time.  You only see true parallel
chains on GPU by requesting `--gres=gpu:4` in the SLURM script.

**Fix.**  `cluster/13_bayesian_multigpu.slurm` requests 4 GPUs and sets
`CHAINS=4`.  For single-GPU runs, accept that chains are vectorized and
budget time accordingly.

### Pitfall: `chain_method="parallel"` on Windows/WSL

`"parallel"` uses process forking.  On Windows and some WSL
configurations, `multiprocessing.fork` is not supported and NumPyro
silently falls back to sequential or crashes.  For local testing on
Windows, use `"vectorized"` explicitly.

### `NUMPYRO_HOST_DEVICE_COUNT` env var

Setting `NUMPYRO_HOST_DEVICE_COUNT=4` at the shell level before
`python ...` is equivalent to calling `numpyro.set_host_device_count(4)`
early in the process.  Required in SLURM scripts where you want CPU
pmap:

```bash
export NUMPYRO_HOST_DEVICE_COUNT=4
python scripts/04_model_fitting/b_bayesian/fit_bayesian.py --chains 4 --model wmrl_m6b
```

---

## 3. Parallel scan (pscan) — when it helps, when it hurts

Phase 19–20 work added `associative_scan`-based O(log T) variants of
the block likelihood functions.  The standard pattern in sequential
code is `lax.scan` with O(T) depth; pscan replaces that with
`lax.associative_scan` which has O(T) *work* but O(log T) *depth*.

### Benchmark results (pre-Issue-1-fix, L40S, N=154)

| Model | Sequential | PScan | Speedup |
|---|---:|---:|---:|
| qlearning | 165.76 ms | 597.13 ms | **0.28×** (slower) |
| wmrl | 230.97 ms | 125.18 ms | 1.85× |
| wmrl_m3 | 255.27 ms | 148.98 ms | 1.71× |
| wmrl_m5 | 262.20 ms | 157.89 ms | 1.66× |
| wmrl_m6a | 281.08 ms | 163.96 ms | 1.71× |
| wmrl_m6b | 293.29 ms | 185.55 ms | 1.58× |

### Why qlearning is slower under pscan

Qlearning's per-trial math is tiny — three Q-value updates per
trial.  Pscan's overhead (2× work, more complex index math) exceeds
the benefit when T is small (T=100 blocks).  Pscan pays off when the
per-trial math is large enough that the log-depth benefit wins.

### Why pscan runs on CPU are 3.7× **slower**

CPU has no depth parallelism.  Pscan does O(2T) work; sequential scan
does O(T) work.  On CPU, more work = more time.  Pscan is a
GPU/TPU-only optimization.

### Pitfall: pscan compile-time memory explosion

**Context.**  Pscan generates large XLA intermediate representations.
The compile graph is much larger than the runtime tensors.  Runtime
peak memory: 0.1 MB.  Compile-time peak host memory: ~50–80 GB for
the full N=154 model.

**Consequence.**  Our first pscan smoke SLURM used `--mem=32G` and
hit OOM during compilation (not runtime).  Fix: bumped to `--mem=96G`
for pscan-containing jobs.  Runtime memory is tiny; host memory for
compile is the real constraint.

### Pitfall: pscan + nested vmap is not supported in our stack

**Context.**  The fully-batched vmap path (outer over participants,
inner over blocks) does not compose with the pscan variant of the
block likelihood.  Attempting to vmap a function that internally calls
`lax.associative_scan` with mask/length-dependent work sometimes
produces obscure tracer errors.

**Consequence.**  All `*_fully_batched_likelihood` functions raise
`NotImplementedError` when `use_pscan=True`.  If you need pscan, you
fall back to the sequential-participant path (the pre-Issue-1 pattern,
but per participant the block likelihood uses O(log T) associative
scan).  In practice this is rarely needed because the Issue 1 fix
already closes the per-iter-cost gap.

### Rule

**Pscan is opt-in via `--use-pscan` and only makes sense on GPU for
models with non-trivial per-trial math (WM-RL family).**  For CPU or
qlearning, the sequential `lax.scan` path is faster.

---

## 4. The return_pointwise JIT quirk

Block likelihood functions have a dual return signature:

```python
def wmrl_m3_block_likelihood(
    ...,
    *,
    return_pointwise: bool = False,
) -> float | tuple[float, jnp.ndarray]:
    ...
    if return_pointwise:
        return total_log_lik, per_trial_log_probs
    return total_log_lik
```

### Pitfall: "is this JIT-compatible?"

**Context.**  JAX tracers flow through Python code; any `if` on a
traced value raises `ConcretizationTypeError` at trace time.  A
reasonable instinct is to avoid all `if`s in likelihood code.

**Reality.**  `if return_pointwise` is fine.  `return_pointwise` is
a Python `bool`, **not a traced value**.  It is a compile-time
constant: JIT traces the function once per value of that boolean and
caches the resulting graph.  The `if` is evaluated at trace time, not
run time.

**Compare:**

```python
def bad(x, threshold):
    if x > threshold:          # ConcretizationTypeError — x is traced
        return x * 2
    return x

def good_1(x, threshold):
    return jnp.where(x > threshold, x * 2, x)    # in-graph conditional

def good_2(x, is_double):
    if is_double:               # is_double is bool, not traced
        return x * 2
    return x
```

The second `good_` form is identical to our `return_pointwise` branch.

### Rule

**Branching on Python booleans or other static values is fine inside
JIT.  Branching on traced arrays requires `jnp.where`, `lax.cond`,
or similar.**  When in doubt, add `static_argnames=("flag",)` to the
JIT decorator or let the caller pass two different function objects.

---

## 5. JIT cache management

### Cache is keyed on graph structure + types

**Pitfall.**  Sharing `JAX_COMPILATION_CACHE_DIR` across runs that use
different dtypes (e.g., float32 choice-only models and float64 M4 LBA)
silently re-compiles everything because the cache keys don't match.
Worse, if you force-reuse a cache with mismatched dtypes, you can get
subtly wrong results.

**Fix.**  Separate cache dirs per job type:

```bash
# Choice-only
export JAX_COMPILATION_CACHE_DIR="/scratch/${PROJECT}/${USER}/.jax_cache_bayesian"
# M4 LBA (float64)
export JAX_COMPILATION_CACHE_DIR="/scratch/${PROJECT}/${USER}/.jax_cache_m4_bayesian"
# pscan smoke
export JAX_COMPILATION_CACHE_DIR="/scratch/${PROJECT}/${USER}/.jax_cache_pscan_smoke"
```

### `JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES`

By default, JAX only caches compilations above ~1 MB.  Many smaller
compilations (helper functions, validation tests) get re-compiled
every run.  Setting this to 0 caches everything:

```bash
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
```

Worth it for long-running MCMC jobs where even small re-compiles
accumulate.

### Pitfall: `jax.config.jax_enable_x64 = True` is global

**Context.**  M4 LBA requires float64 to avoid numerical underflow in
the LBA density.  `jax.config.update("jax_enable_x64", True)` is a
process-global switch.

**Consequence.**  If `lba_likelihood` is imported at module load time,
every JAX operation in the process runs at float64 — including
choice-only models that are calibrated for float32.

**Fix.**  Lazy imports inside function bodies:

```python
# scripts/fitting/numpyro_models.py, M4 hierarchical model
def wmrl_m4_hierarchical_model(...):
    from scripts.fitting.lba_likelihood import wmrl_m4_multiblock_likelihood_stacked
    # ... rest of model
```

Choice-only models never trigger the LBA import and stay float32.

---

## 6. SLURM templates — GPU vs CPU

### Template A — CPU, 4 chains, parallel via set_host_device_count

Used by `cluster/13_bayesian_m{1,2,3,5,6a,6b}.slurm`.

```bash
#SBATCH --time=36:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

module load miniforge3
conda activate ds_env

export JAX_PLATFORMS=cpu
export NUMPYRO_HOST_DEVICE_COUNT=4
export JAX_COMPILATION_CACHE_DIR="/scratch/${PROJECT}/${USER}/.jax_cache_bayesian"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export PYTHONUNBUFFERED=1

python scripts/04_model_fitting/b_bayesian/fit_bayesian.py \
    --model wmrl_m6b --chains 4 --warmup 1000 --samples 2000 --max-tree-depth 8

source cluster/autopush.sh
```

Key decisions:

- `--cpus-per-task=4` — one core per chain (parallel pmap)
- `--mem=64G` — enough for warm JIT + MCMC samples
- `NUMPYRO_HOST_DEVICE_COUNT=4` — exposes 4 host devices so
  `jax.local_device_count()` returns 4 and `_select_chain_method`
  returns "parallel"
- `JAX_PLATFORMS=cpu` — forces CPU even if GPU is visible (prevents
  accidental GPU selection on shared nodes)

### Template B — Single GPU, chains vectorized

Used by `cluster/13_bayesian_pscan_smoke.slurm` and the new
`cluster/13_bayesian_fullybatched_smoke.slurm`.

```bash
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load miniforge3
conda activate rlwm_gpu

export NUMPYRO_HOST_DEVICE_COUNT=1
export JAX_COMPILATION_CACHE_DIR="/scratch/${PROJECT}/${USER}/.jax_cache_fb_smoke"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export PYTHONUNBUFFERED=1

python scripts/04_model_fitting/b_bayesian/fit_bayesian.py \
    --model wmrl_m6b --chains 4 --warmup 500 --samples 500
```

Key decisions:

- `--gres=gpu:1` — single GPU; `chain_method` auto-resolves to
  "vectorized"
- `NUMPYRO_HOST_DEVICE_COUNT=1` — no CPU host devices needed
- `--mem=32G` — smaller than CPU template because runtime memory is
  tiny
- Environment: `rlwm_gpu` (not `ds_env`) has CUDA dependencies

### Template C — Multi-GPU, chains parallel via pmap

New, `cluster/13_bayesian_multigpu.slurm`.  Untested in production.

```bash
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4

# ... activate rlwm_gpu, cache, etc. ...

python scripts/04_model_fitting/b_bayesian/fit_bayesian.py \
    --model wmrl_m6b --chains 4 --warmup 1000 --samples 2000
```

`_select_chain_method(4)` returns `"parallel"` when `local_device_count=4`.
Untested: whether NumPyro's pmap path correctly handles our custom
single-factor model.  Phase 20 did not verify this end-to-end.

### The autopush pattern

Every SLURM script ends with:

```bash
source cluster/autopush.sh
```

That script:

1. `git add logs/*` and output artifacts produced by the run
2. Creates a commit with the job ID in the message
3. Pushes to `origin/main`

**Why this matters.**  SLURM jobs run while you are offline.  Without
autopush, you have to SSH back, `git pull` from the cluster, then
`git push` to origin — a four-step manual dance.  With autopush, logs
appear on GitHub shortly after the job completes.  You can diagnose
failures, compare to past runs, and load results into local tools
without touching the cluster.

### Pitfall: `PYTHONUNBUFFERED=1`

**Context.**  Python buffers stdout by default.  Under SLURM, output
lines may never reach `logs/*.out` if the job is killed, or only
appear at job completion.

**Fix.**  `export PYTHONUNBUFFERED=1` forces line-buffering.  Progress
bars and `[convergence-gate]` markers appear in real time.  Essential
for debugging stuck jobs.

### Pitfall: per-model timeout in smoke tests

**Context.**  The Apr 16 pscan smoke SLURM (job 54845743) had a single
6-hour wall clock for all 6 models.  qlearning's pre-fix 6-min/iter
rate burned the entire wall running just qlearning — we got zero
data on the other 5 models.

**Fix.**  Wrap each model invocation in `timeout 600`:

```bash
for MODEL in $MODELS; do
    timeout "$PER_MODEL_TIMEOUT" python scripts/04_model_fitting/b_bayesian/fit_bayesian.py \
        --model "$MODEL" ...
    EXIT=$?
    if [[ $EXIT -eq 124 ]]; then
        echo "  $MODEL: TIMEOUT"  # keep going to next model
    fi
done
```

Worst-case wall = N_models × per_model_timeout instead of single
runaway model.  Implemented in `cluster/13_bayesian_fullybatched_smoke.slurm`.

---

## 7. Common JAX pitfalls with context

### Pitfall 1: `.item()`, `.numpy()`, `float(x)` inside JIT

**Context.**  You want to print or log a scalar value from inside a
JIT'd function.  Calling `.item()` on a JAX array is the natural
instinct.

**What breaks.**  JIT traces the function symbolically; there is no
concrete value to call `.item()` on.  You get `ConcretizationTypeError`.

**Fix.**  Either:

- Move the `.item()` call outside the JIT boundary:
  ```python
  result = jit_fn(x)
  print(result.item())           # outside JIT, fine
  ```
- Use `jax.debug.print` inside JIT:
  ```python
  jax.debug.print("value: {x}", x=x)    # in-graph, runtime printing
  ```

### Pitfall 2: Python `for` loop over traced length

**Context.**  You have `num_blocks = stimuli.shape[0]` inside a JIT'd
function and want to iterate.

**What breaks.**  If `num_blocks` is static (known at trace time, e.g.,
from a Python tuple or `static_argnums`), Python `for` works.  If it
comes from the shape of a dynamically-shaped input, it fails because
JAX traces a single graph and the loop length must be known at trace
time.

**Fix.**  Use `lax.fori_loop` for known-at-runtime length:

```python
def body(i, carry):
    return carry + stimuli[i]
total = lax.fori_loop(0, num_blocks, body, 0.0)
```

or pad to a fixed max length and use `vmap` with masking.

### Pitfall 3: Creating new arrays in a loop body

**Context.**  A function like:

```python
def process_each(x):
    results = []
    for i in range(x.shape[0]):
        results.append(expensive_fn(x[i]))
    return jnp.stack(results)
```

**What breaks.**  Under JIT, this traces `expensive_fn` `x.shape[0]`
times and creates that many sub-graphs.  For large loop counts,
compile time balloons and memory explodes.

**Fix.**  Use `vmap`:

```python
def process_each(x):
    return jax.vmap(expensive_fn)(x)
```

One graph, one compilation, efficient execution.

### Pitfall 4: `in_axes=(0, 0, 0, ...)` vs `in_axes=(0, 0, 0, None, None, ...)`

**Context.**  You have a function with 7 arguments where the first 4
are batch-varying and the last 3 are shared across the batch.

**What breaks.**  Marking everything as `0`:

```python
vmap(f, in_axes=(0, 0, 0, 0, 0, 0, 0))   # expects batch axis on all 7 args
# f(stim, act, rew, ss, mask, alpha_pos_scalar, alpha_neg_scalar) fails —
# the last two are scalars, not (N,) vectors
```

raises a shape error, or worse, silently broadcasts wrong.

**Fix.**  Use `None` for shared args:

```python
vmap(f, in_axes=(0, 0, 0, 0, 0, None, None))
# Now the last two args are not vmapped — they're passed as-is to each
# invocation.
```

In our fully-batched likelihood, the inner vmap (over blocks) has
`in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None, None)`
because within one participant, the parameters are shared across that
participant's blocks.

### Pitfall 5: Using Python `print` inside JIT

**Context.**  Debugging a JIT'd function.

**What happens.**  `print("x =", x)` runs at **trace time**, not at
runtime.  You see one print per compilation (not per call), and the
traced value is a Tracer object with no concrete data.

**Fix.**  Use `jax.debug.print`:

```python
jax.debug.print("x = {x}", x=x)   # runs at runtime, inside the graph
```

Works with JIT, vmap, pmap.  Does add a small runtime cost, so remove
before production.

### Pitfall 6: `jax.jit` on function with optional kwargs

**Context.**  You JIT a likelihood function that takes `return_pointwise=False`
as a default kwarg.

**What breaks.**  JIT assumes all arguments are traced arrays by
default.  It tries to trace the `bool` and fails.

**Fix.**  Mark static arguments:

```python
@partial(jax.jit, static_argnames=("return_pointwise",))
def likelihood_fn(x, *, return_pointwise=False):
    ...
```

Two separate compilations (one for `True`, one for `False`), each
correct for its branch.

### Pitfall 7: `group_by_chain=False` for convergence diagnostics

**Context.**  Reading MCMC samples to compute R-hat or ESS.

**What breaks.**  `mcmc.get_samples()` defaults to
`group_by_chain=False`, flattening the chains dimension.  ArviZ's
`az.rhat` and `az.ess_bulk` need the chain dimension to compute
between-chain variance.  Silently returns nonsense values.

**Fix.**  Always pass `group_by_chain=True` for diagnostics:

```python
samples_for_diag = mcmc.get_samples(group_by_chain=True)  # (chains, draws, ...)
samples_for_post = mcmc.get_samples()                     # (draws, ...)
```

---

## 8. JIT-optimal coding patterns

### Pattern 1: Precompute data-dependent quantities outside the MCMC trace

**Principle.**  Anything that doesn't depend on sampled parameters
should be precomputed before the MCMC call, not inside the model
function.

**Example: perseveration carry.**

Pre-refactor:

```python
# Inside the block likelihood, during each MCMC iteration:
def block_ll(stim, act, rew, mask, ...):
    def carry_body(prev_last_action, t):
        ...
        new_last_action = jnp.where(mask[t], act[t], prev_last_action)
        return new_last_action, ...
    _, last_actions = lax.scan(carry_body, -1, jnp.arange(T))
    # ... use last_actions in the likelihood
```

This recomputes `last_actions` on every likelihood evaluation even
though it depends only on observed actions (data), not on parameters.

Post-refactor (Phase 20-01):

```python
# Once, before MCMC:
last_actions_precomputed = precompute_last_action_global(actions, mask)
# During MCMC, likelihood receives it as a non-traced input:
def block_ll(stim, act, rew, mask, last_actions, ...):
    # Use last_actions directly, no scan needed
```

Saved ~30% of likelihood evaluation time on M3 and M6a.

### Pattern 2: Stack before vmap

**Principle.**  Build one big (N, B, T) tensor once, then vmap over it,
rather than iterating over a dict of per-participant arrays.

```python
# config.py / numpyro_models.py
def stack_across_participants(participant_data_stacked):
    """Produce (N, B_max, T_max) tensors."""
    max_n_blocks = max(p["stimuli_stacked"].shape[0] for p in participant_data_stacked.values())
    ...
    return {
        "stimuli":   jnp.stack([...]),    # (N, B_max, T_max)
        "actions":   jnp.stack([...]),
        "masks":     jnp.stack([...]),    # pad blocks have masks entirely 0
        ...
    }
```

The caller passes these to the fully-batched likelihood, which vmaps
over axis 0.  Padding blocks contribute 0 to the likelihood because
every likelihood term is multiplied by `mask[t]`.

### Pattern 3: Associative combiners for O(log T) depth

**When to reach for pscan.**  If your scan body has the form

```
carry_new = f(carry_old, data[t])
```

and `f` is associative (i.e., `f(f(a, x), y) = f(a, f_combined(x, y))`
for some combiner), then you can replace `lax.scan` with
`lax.associative_scan`, which has O(T) work but O(log T) depth.

**Example: affine Q-update.**

The asymmetric Q-learning update

```
Q[t+1] = Q[t] + alpha * (r[t] - Q[t])
       = (1 - alpha) * Q[t] + alpha * r[t]
```

is affine in `Q[t]` with coefficient `a = (1 - alpha)` and intercept
`b = alpha * r[t]`.  Affine maps compose:

```
(a1, b1) ∘ (a2, b2) = (a1 * a2, a1 * b2 + b1)
```

so pscan over `(a, b)` pairs gives O(log T) depth.  Implemented as
`associative_scan_q_update` in `scripts/fitting/jax_likelihoods.py`.
Same trick works for WM decay.

**Limits.**  The perseveration kernel carry (last-action update) is
not associative in the obvious way — so we precompute it outside the
scan instead (Pattern 1).

### Pattern 4: `jax.vmap` over `lax.scan` instead of Python loop over `vmap`

```python
# Slow: one JIT compile per participant
def slow_way(data_list, param_list):
    results = []
    for data, params in zip(data_list, param_list):
        results.append(jax.vmap(block_fn)(data, params))   # compiled per call
    return jnp.stack(results)

# Fast: one JIT compile total
def fast_way(data_stacked, param_stacked):
    return jax.vmap(lambda d, p: jax.vmap(block_fn)(d, p))(data_stacked, param_stacked)
```

### Pattern 5: Pad to power-of-2 lengths when feasible

Modern GPUs dispatch at warp granularity (32 threads on NVIDIA,
64 threads on AMD).  Kernels on arrays with length 128 or 256 run at
the same wall time as length 100 but with better occupancy.  For small
scan lengths, padding to a power of 2 can give free speedups.

In our pipeline, `MAX_TRIALS_PER_BLOCK = 100`.  Rounding to 128 would
be a micro-optimization — worth considering if we ever revisit the
block-size constraint.

### Pattern 6: JIT warmup before timing

```python
# Cold timing (includes compile time):
t0 = time.perf_counter()
result = jit_fn(x)
result.block_until_ready()
t_cold = time.perf_counter() - t0

# Warm timing:
t0 = time.perf_counter()
result = jit_fn(x)
result.block_until_ready()
t_warm = time.perf_counter() - t0
```

Always report both.  Cold timing is what a first-run cluster job sees;
warm timing is what MCMC sees on every subsequent NUTS step.

The `.block_until_ready()` call is essential — JAX dispatches
asynchronously, so without it you're timing the dispatch, not the
computation.

---

## 9. NumPyro-specific gotchas

### Gotcha 1: `numpyro.sample` site names must be unique per iteration

**Context.**  A model function is called once per MCMC iteration.  Each
`numpyro.sample("name", dist)` call registers a site.  Two sites with
the same name in the same iteration cause an error.

```python
# Wrong
for i in range(num_groups):
    x = numpyro.sample("theta", dist.Normal(0, 1))
# "theta" registered num_groups times — error

# Right
theta = numpyro.sample("theta", dist.Normal(0, 1).expand([num_groups]))
```

### Gotcha 2: `deterministic` sites vs `sample` sites

```python
kappa_mu = numpyro.sample("kappa_mu", dist.Normal(0, 1))
kappa_z = numpyro.sample("kappa_z", dist.Normal(0, 1).expand([N]))
kappa_unc = kappa_mu + sigma * kappa_z
kappa = numpyro.deterministic("kappa", lower + (upper - lower) * Phi(kappa_unc))
```

`deterministic` sites are recorded in samples but not sampled — they
are functions of other sites.  Useful for recording transformed
parameters (the bounded `kappa`) that are functions of unbounded
ones (`kappa_unc`).

### Gotcha 3: `chain_method="vectorized"` and `progress_bar=True`

With `progress_bar=True`, NumPyro prints one progress bar per chain.
On `vectorized` chain method, progress bars **lie** — the bar reaches
100% for each chain simultaneously, not sequentially.  The true
progress is `max(bars)`, which is equal to each bar.

Harmless but confusing.  For genuine progress, log `t_warm` after
each iteration manually.

### Gotcha 4: Convergence auto-bump

Our `run_inference_with_bump` escalates `target_accept_prob` through
`(0.80, 0.95, 0.99)` if divergences occur.  Each escalation is a full
re-run.  Expected wall time = 1× (best case, no divergences) to 3×
(worst case, all three levels needed).

Budget SLURM `--time` accordingly.  For M6b with
`target_accept=0.99` as worst case: 3 × 3–4h ≈ 12 h.

---

## 10. Debugging checklist

When a cluster job misbehaves, work through these in order:

1. **Read the SLURM log first.**  Grep for `Traceback`, `Error`,
   `Killed`, `OOM`, `DUE TO TIME LIMIT`.  Don't skim — pitfalls often
   hide in warnings.
2. **Check `[convergence-gate]` lines.**  Divergences on first
   `target_accept_prob=0.80` often just mean NUTS is still adapting.
   Divergences at 0.99 indicate true geometric pathology.
3. **Check `R-hat` values in `{model}_convergence_summary.csv`.**  Any
   site with R-hat > 1.05 means the chains haven't mixed.  Two
   chains disagree more than they agree with themselves.
4. **Check trace-site count.**  If `mcmc.get_samples()` has
   `obs_p{pid}` entries for every participant, you're running the
   pre-Issue-1 model code by mistake.
5. **Check `jax.devices()`.**  If you expected GPU and got CPU, the
   SLURM `--gres=gpu:1` didn't stick or the conda environment is
   wrong.  The diagnostic banner prints this at the top of every
   hierarchical fit.
6. **Check `chain_method`.**  Print in the diagnostic banner.
   "parallel" on a single-GPU job is a misconfiguration.
7. **Check free memory at compile time.**  `nvidia-smi` during
   compile is misleading (memory usage is low).  `free -g` on the host
   during compile reveals whether pscan compile blew the host RAM.
8. **Check `JAX_COMPILATION_CACHE_DIR`.**  If running fresh because
   the cache dir wasn't set, the first MCMC iter includes ~30s of
   compile time.  If running fresh when you expected cache hits, you
   changed a JIT'd function's signature or dtype.

---

## 11. What would I do differently next time

Retrospective from the v4.0 development:

1. **Start with the vmap'd likelihood pattern from day one.**  The
   per-participant for-loop idiom works fine for MLE and small-N unit
   tests, and hides the dispatch bottleneck until you try to scale.
   Design the hierarchical model as fully-batched from the start.
2. **Write the real-data N=154 agreement test before the hierarchical
   model.**  The quick-006 MLE fits were a useful cross-check; a
   real-data numerical agreement test against stored MLE NLLs catches
   refactor bugs at float32 precision (1e-7) before they become
   posterior-level bugs.
3. **Budget SLURM wall time with explicit per-model timeouts.**
   Single-wall-clock SLURM jobs are a trap — one slow model eats the
   entire budget.  Use `timeout` wrappers and per-model status
   reporting from the start.
4. **Run the diagnostic banner (jax config + env vars) as the first
   thing the script does.**  90% of "why is it slow" questions answer
   themselves from the banner alone.
5. **Keep the JIT compilation cache per-job-type.**  Don't try to
   share.
6. **Use `PYTHONUNBUFFERED=1` everywhere, always.**

---

## 12. Glossary

- **JIT**: just-in-time compilation.  JAX traces a Python function and
  compiles a static graph at first call.
- **vmap**: `jax.vmap` — auto-vectorize a function over an added batch
  axis.  Pure Python, composes with JIT.
- **pmap**: `jax.pmap` — parallelize a function across devices.
  Requires `num_devices` >= batch size.
- **chain_method**: how NumPyro distributes MCMC chains across
  devices.  `"parallel"` = pmap, `"vectorized"` = vmap on one device,
  `"sequential"` = Python loop.
- **pscan**: shorthand for `lax.associative_scan`.  O(log T) depth
  parallel reduction for associative combiners.
- **Trace site**: a named random variable registered in a NumPyro
  model via `numpyro.sample`, `numpyro.factor`, or
  `numpyro.deterministic`.
- **Trace time vs runtime**: trace time is the first invocation when
  JAX builds the computation graph; runtime is every subsequent
  invocation that reuses the cached graph.

---

## 13. References

- `scripts/fitting/jax_likelihoods.py` — all fully-batched likelihood
  implementations
- `scripts/fitting/numpyro_models.py` — `_select_chain_method`,
  `run_inference_with_bump`, all hierarchical model bodies
- `cluster/13_bayesian_*.slurm` — SLURM templates for CPU, single-GPU,
  multi-GPU
- `docs/PARALLEL_SCAN_LIKELIHOOD.md` — pscan architecture deep-dive
- `docs/legacy/JAX_GPU_BAYESIAN_FITTING.md` — earlier lessons, partial
  overlap with this document (archived; superseded by this file)
- `docs/HIERARCHICAL_BAYESIAN.md` — Bayesian architecture and
  validation checklist
- `.planning/STATE.md` — chronological decision records
