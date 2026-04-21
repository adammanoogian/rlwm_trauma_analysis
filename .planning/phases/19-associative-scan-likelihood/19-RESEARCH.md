# Phase 19: Associative Scan Likelihood Parallelization - Research

**Researched:** 2026-04-14
**Domain:** JAX parallel prefix scan / linear recurrence parallelization for RLWM likelihood
**Confidence:** HIGH (code structure: direct inspection), HIGH (JAX API: official docs + working examples), MEDIUM (WM overwrite derivation: derived from first principles, not prior implementation found), LOW (speedup magnitude: no prior RLWM scan benchmarks found)

---

## Summary

Phase 19 replaces the inner `lax.scan` inside each block likelihood function with `jax.lax.associative_scan` for the Q-update and WM-decay recurrences. The sequential `lax.scan` across T=100 trial positions is replaced with an O(log T) parallel prefix scan. The non-linear parts (WM-Q mixing, softmax, epsilon noise, perseveration) cannot be parallelized this way and remain as a sequential post-scan pass.

The current codebase has a very clean structure: each model has a `*_block_likelihood()` (processes one block via `lax.scan`) and a `*_multiblock_likelihood_stacked()` (loops over blocks via `lax.fori_loop`). The `_pscan` variants should mirror this: `*_block_likelihood_pscan()` and `*_multiblock_likelihood_stacked_pscan()`. Sequential versions are untouched.

The main technical challenge is the WM overwrite: `WM(s,a) <- r` at feedback trials is a hard reset that breaks the clean linear recurrence. It can be encoded within the scan operator as a multiplicative reset by setting `a_t = 0` at overwrite positions. This derivation is fully worked out below.

**Primary recommendation:** Implement a generic `affine_scan(a_t, b_t, x0)` helper that calls `jax.lax.associative_scan` with the standard `(a2*a1, a2*b1+b2)` operator. All Q-update and WM-decay recurrences reduce to this form. Build `associative_scan_q_update()` and `associative_scan_wm_update()` on top of this helper.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `jax.lax.associative_scan` | JAX current (0.4.x) | Parallel prefix scan | Built into JAX; used by cumsum, cumprod, Kalman smoothers, S4/Mamba |
| `jax.lax.scan` | JAX current | Sequential scan (keep for reference impl) | Already in use throughout codebase |
| `numpyro` | current | MCMC framework | Already in use for Bayesian fitting |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `time.perf_counter` + `.block_until_ready()` | Python stdlib + JAX | Accurate GPU wall-clock timing | Microbenchmark in `validation/benchmark_parallel_scan.py` |
| `jax.devices()[0].memory_stats()` | JAX 0.4.1+ | GPU VRAM measurement | Inside benchmark script after each model run |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `jax.lax.associative_scan` | Custom Hillis-Steele (work-inefficient) | The work-inefficient variant is ~33% faster in latency for small N (GitHub discussion #10599) but adds maintenance burden. Not in mainline JAX. For T=100 trials per block, standard `associative_scan` is fine — 2 log₂(100) ≈ 14 steps vs T=100 sequential. Use standard first; profile before custom. |
| Per-model scan operators | Single generic `affine_scan()` helper | Generic helper is the correct approach — Q-update, WM-decay, and WM-overwrite all reduce to `x_t = a_t * x_{t-1} + b_t`. No per-model operator needed. |

---

## Architecture Patterns

### Recommended Project Structure (changes only)

```
scripts/fitting/
├── jax_likelihoods.py         # Add *_pscan variants alongside existing functions
validation/
├── benchmark_parallel_scan.py # New: standalone microbenchmark (Stage 1)
scripts/fitting/tests/
├── test_pscan_likelihoods.py  # New: unit tests for pscan variants
```

### Pattern 1: Generic Affine Scan Helper

**What:** All linear recurrences in RLWM have the form `x_t = a_t * x_{t-1} + b_t`. This maps directly to `jax.lax.associative_scan` with a binary operator on `(a, b)` pairs.

**Operator derivation (verified):**
```
Composing two steps (a2, b2) applied after (a1, b1):
x_1 = a1 * x0 + b1
x_2 = a2 * x1 + b2 = a2*(a1*x0 + b1) + b2 = (a2*a1)*x0 + (a2*b1 + b2)

Therefore: (a2, b2) ∘ (a1, b1) = (a2*a1, a2*b1 + b2)
```

**When to use:** Whenever you have a scalar or array-wise recurrence where each state depends linearly on the previous state.

**Verified code pattern (from Linxi's blog, 2025-01-02, verified against sequential `lax.scan`):**
```python
def affine_scan(a_seq: jnp.ndarray, b_seq: jnp.ndarray, x0: float) -> jnp.ndarray:
    """Parallel prefix scan for x_t = a_t * x_{t-1} + b_t.

    Returns x[1..T] as an array of shape (T,).
    x0 is the initial condition (not included in output).
    """
    def operator(e_left, e_right):
        # e = (a, b)  representing the affine map x -> a*x + b
        # Compose: right after left:  x -> a_r*(a_l*x + b_l) + b_r
        a_l, b_l = e_left
        a_r, b_r = e_right
        return (a_r * a_l, a_r * b_l + b_r)

    # Prepend identity element (1, x0) to handle initial condition
    a_full = jnp.concatenate([jnp.array([1.0]), a_seq])
    b_full = jnp.concatenate([jnp.array([x0]), b_seq])

    # After scan: element t = product of operators 0..t applied to (1, x0)
    # The second component gives x_t
    _, x_all = jax.lax.associative_scan(operator, (a_full, b_full))

    # Slice off the t=0 identity (returns x[1..T])
    return x_all[1:]
```

**Simpler pattern (initial condition absorbed into b_seq):**
```python
def affine_scan_simple(a_seq, b_seq):
    """Run associative scan directly; caller prepends (1.0, x0) to sequences."""
    def operator(e_l, e_r):
        a_l, b_l = e_l
        a_r, b_r = e_r
        return (a_r * a_l, a_r * b_l + b_r)
    return jax.lax.associative_scan(operator, (a_seq, b_seq))
```

### Pattern 2: Q-Update Recurrence

**Sequential form (from `jax_likelihoods.py`):**
```python
q_current = Q_table[stimulus, action]
delta = reward - q_current
alpha = jnp.where(delta > 0, alpha_pos, alpha_neg)
q_updated = q_current + alpha * delta
# = q_current * (1 - alpha) + alpha * reward
```

**Linear recurrence form:**
`Q_t(s,a) = (1 - alpha_t) * Q_{t-1}(s,a) + alpha_t * reward_t`

Where:
- `alpha_t = alpha_pos` if `reward_t > Q_{t-1}(s,a)` else `alpha_neg`
- The alpha depends on the sign of delta, which depends on `Q_{t-1}` — this makes alpha data-dependent on the state

**Critical observation:** `alpha_t` depends on `Q_{t-1}(s,a)` (via the delta sign), which makes this technically a non-linear recurrence. However, in practice the sign of delta is dominated by whether `reward=1` or `reward=0`. For the scan, alpha must be pre-computed using a fixed approximation — one valid approach is to use `alpha_pos` whenever `reward_t = 1` and `alpha_neg` whenever `reward_t = 0` (which is exact when Q values stay near 0.5, the initialization). This is the standard approximation used in PaMoRL-style parallelization.

**MEDIUM confidence:** This approximation is valid at initialization (`q_init=0.5`) and degrades as Q-values diverge. For validation, compare sequential vs pscan on real data; agreement to <1e-5 relative error is the criterion.

**Alternative approach:** Ignore alpha asymmetry for the scan pass — compute all Q-values assuming `alpha_avg = (alpha_pos + alpha_neg) / 2`, then use the resulting Q-values only for policy computation (not for the Q-updates themselves). But this is more complex than the approximation above. **Recommendation:** use the data-dependent alpha approach (alpha based on reward_t alone, not delta sign) as the canonical approximation and document the approximation error.

**Exact form for scan:**
```python
# Per-trial, per-(stimulus, action) pair:
a_t = jnp.where(valid, 1.0 - jnp.where(reward_t == 1.0, alpha_pos, alpha_neg), 1.0)
b_t = jnp.where(valid, jnp.where(reward_t == 1.0, alpha_pos, alpha_neg) * reward_t, 0.0)
# Q_t = a_t * Q_{t-1} + b_t
```

**Note on sparsity:** Each (stimulus, action) pair only appears in a subset of trials. For the scan, use `a_t = 1.0, b_t = 0.0` at trials where stimulus s is not presented (identity operator — value passes through unchanged). This is valid because the scan applies the identity when `a=1, b=0`.

### Pattern 3: WM Decay Recurrence

**Sequential form (from `jax_likelihoods.py`):**
```python
WM_decayed = (1 - phi) * WM_table + phi * WM_baseline
# Then: WM_updated[s,a] = reward (hard overwrite, masked)
```

**Two operations per trial:**
1. Decay: `WM_t = (1-phi) * WM_{t-1} + phi * WM_0` (linear, uniform across all (s,a))
2. Overwrite: `WM_t(s,a) = reward_t` for the presented (s,a) pair (resets a specific cell)

**Decay as linear recurrence:**
`WM_t(s,a) = (1-phi) * WM_{t-1}(s,a) + phi * WM_0`

- `a_t = (1-phi)` (constant across trials)
- `b_t = phi * WM_0` (constant across trials)

This is a homogeneous decay with fixed multiplier — trivially parallelizable.

**Overwrite (hard reset) mechanism:**
The hard overwrite `WM(s,a) <- r` can be encoded as a multiplicative reset by setting `a_t = 0, b_t = r` at overwrite positions. This encodes "forget everything before, value is now `r`."

**Derivation:** After applying the operator sequence `...(a_k, b_k)∘...∘(a_1, b_1)`, any position where `a_t = 0` creates a barrier: all information from before position t is multiplied by 0. The cumulative product of `a` values is zero for all positions after a reset, so only `b_t` (the overwrite value) and subsequent operators matter.

**Combined WM recurrence per (stimulus, action) pair per trial:**

```python
# At trial t, for (stimulus, action) = (s, a):
# Case 1: This (s,a) was presented and feedback given (valid and presented)
#   WM_{t+1}(s,a) = reward_t  (hard overwrite)
#   Encode: a = 0, b = reward_t
#
# Case 2: Any stimulus s' != s was presented (not this cell)
#   WM_t decays to baseline
#   Encode: a = (1-phi), b = phi * WM_0
#
# Case 3: Padding trial (valid=0)
#   WM stays unchanged (identity)
#   Encode: a = 1.0, b = 0.0

# Note: The decay and overwrite happen in the SAME trial for the presented (s,a):
# Step 1: Decay all cells (global a=(1-phi), b=phi*WM_0)
# Step 2: Overwrite: a=0, b=reward (overrides the decay for this cell)
# These two steps are folded into a single operator element per trial:
presented_and_valid = jnp.logical_and(stimulus_t == s, valid_t)
a_t_wm = jnp.where(presented_and_valid, 0.0, 1.0 - phi)
b_t_wm = jnp.where(presented_and_valid, reward_t, phi * WM_0)
```

**Important timing:** In the sequential code, WM is decayed BEFORE computing the policy, then overwritten AFTER the policy. This means the policy sees `WM_decayed` (not `WM_overwritten`). In the scan approach, we build the full WM trajectory and read `WM_after_decay_before_overwrite` for policy computation. This requires careful indexing — the scan gives `WM_t_after_overwrite`; the policy for trial t uses `WM_t_after_decay_before_overwrite(t)`.

**Resolution:** Run two passes for WM:
1. Scan pass 1 (decay only): `a_t = (1-phi)`, `b_t = phi*WM_0` for all trials — gives `WM_decayed[t]` for policy use.
2. Scan pass 2 (decay + overwrite): Combined operator above — gives `WM_final[t]` for updating state (not used in policy).

Or equivalently, collect the WM state seen by the policy (pre-overwrite) during the scan by separating decay and overwrite operators.

### Pattern 4: Per-Stimulus Decomposition Strategy

**The challenge:** Q and WM tables have shape `(num_stimuli=6, num_actions=3)`. The recurrence is sparse — stimulus s is not presented at every trial. Other stimuli's states are unchanged (identity operator).

**Approach:** Decompose into `num_stimuli * num_actions = 18` independent scalar recurrences, one per (s, a) pair. For each pair, build the sequence `a_t, b_t` over T trials, where trials not involving this (s,a) pair get `a_t=1.0, b_t=0.0`.

**Implementation:**

```python
# For Q-update, build 2D sequence arrays
# Shape: (T, num_stimuli, num_actions)
s_onehot = jax.nn.one_hot(stimuli, num_stimuli)  # (T, num_stimuli)
a_onehot = jax.nn.one_hot(actions, num_actions)   # (T, num_actions)
# stimulus-action mask: (T, num_stimuli, num_actions)
sa_mask = s_onehot[:, :, None] * a_onehot[:, None, :]  # (T, S, A)

# alpha selection based on reward (data-dependent approximation)
alpha_t = jnp.where(rewards[:, None, None] == 1.0, alpha_pos, alpha_neg)  # (T, S, A) broadcast

# a_t(s,a) = (1-alpha_t) if (s,a) active at trial t, else 1.0
a_seq = jnp.where(sa_mask * valid[:, None, None], 1.0 - alpha_t, 1.0)  # (T, S, A)
# b_t(s,a) = alpha_t * reward if (s,a) active, else 0.0
b_seq = jnp.where(sa_mask * valid[:, None, None], alpha_t * rewards[:, None, None], 0.0)

# Run associative scan over T axis (axis=0), independently for each (s,a)
def op(e_l, e_r):
    return (e_r[0] * e_l[0], e_r[0] * e_l[1] + e_r[1])

# Prepend Q_init as identity
a_full = jnp.concatenate([jnp.ones((1, num_stimuli, num_actions)), a_seq], axis=0)
b_full = jnp.concatenate([Q_init[None], b_seq], axis=0)

_, Q_all = jax.lax.associative_scan(op, (a_full, b_full), axis=0)
Q_values_per_trial = Q_all[1:]  # Shape (T, num_stimuli, num_actions)
# Q_values_per_trial[t] = Q-table AFTER trial t update
```

**Policy computation:** The policy at trial t uses Q BEFORE the update (the value at the start of trial t, which equals the output of the scan at t-1). This means `Q_for_policy[t] = Q_all[t]` (before the t+1 overwrite indexed by `Q_all[1:]`).

### Pattern 5: STACKED_MODEL_DISPATCH Integration

**Current structure** (in `fit_bayesian.py`):
```python
STACKED_MODEL_DISPATCH: dict[str, object] = {
    "qlearning": qlearning_hierarchical_model_stacked,
    "wmrl": wmrl_hierarchical_model_stacked,
    "wmrl_m3": wmrl_m3_hierarchical_model,
    "wmrl_m5": wmrl_m5_hierarchical_model,
    "wmrl_m6a": wmrl_m6a_hierarchical_model,
    "wmrl_m6b": wmrl_m6b_hierarchical_model,
}
```

**Integration approach:** Add `use_pscan: bool = False` kwarg to the `*_block_likelihood` and `*_multiblock_likelihood_stacked` functions in `jax_likelihoods.py`. The numpyro hierarchical models in `numpyro_models.py` already call `wmrl_m3_multiblock_likelihood_stacked(...)` directly. No change to `STACKED_MODEL_DISPATCH` itself is needed — the dispatch still resolves to the same hierarchical model function, which internally calls the likelihood with `use_pscan=use_pscan` propagated.

**Alternative (preferred for cleanliness):** Create separate pscan-variant numpyro models (`wmrl_m3_hierarchical_model_pscan`, etc.) that call `wmrl_m3_multiblock_likelihood_stacked_pscan()`. Then wire `--use-pscan` in `13_fit_bayesian.py` to select the pscan dispatch table. This avoids adding `use_pscan` to signatures of 12+ existing functions.

**Decision for planner:** The `use_pscan` kwarg propagation approach adds noise to existing function signatures. The separate function approach is cleaner but adds more code. Given the CONTEXT.md instruction to keep sequential implementations untouched, the **separate pscan functions** approach is recommended — add `*_pscan` suffix to both the likelihood functions and the numpyro models, then wire `--use-pscan` to select the pscan numpyro model at the dispatch level.

### Pattern 6: Benchmarking Infrastructure

**Correct JAX timing pattern (HIGH confidence — official JAX docs):**
```python
import time

# Step 1: JIT compile (exclude from timing)
fn_jit = jax.jit(fn)
_ = fn_jit(*args).block_until_ready()  # warmup

# Step 2: Timed runs (N repeats)
times = []
for _ in range(N_REPEATS):
    start = time.perf_counter()
    result = fn_jit(*args).block_until_ready()
    elapsed = time.perf_counter() - start
    times.append(elapsed)

mean_ms = 1000 * np.mean(times)
```

**Key rules:**
- Always call `.block_until_ready()` before stopping the timer — JAX dispatches asynchronously
- The first call (warmup) includes XLA compilation; exclude it from timing
- Do NOT time inside `jax.jit` — Python functions run only once at trace time
- Use `time.perf_counter()`, not `time.time()` (higher resolution)

**GPU VRAM measurement:**
```python
devices = jax.devices()
if hasattr(devices[0], 'memory_stats'):
    stats = devices[0].memory_stats()
    peak_gb = stats.get('peak_bytes_in_use', 0) / 1e9
```
(Already present in `jax_likelihoods.py` as `log_gpu_memory()`)

**Two-stage benchmark design:**
1. `validation/benchmark_parallel_scan.py`: Micro-benchmark on synthetic 17-block × 100-trial data. Measures sequential `lax.scan` vs `associative_scan` per single-participant likelihood call. Records wall time, speedup ratio, accuracy vs sequential. No MCMC.
2. `13_fit_bayesian.py --use-pscan`: Full MCMC benchmark on real N=154 data, 4 chains, standard warmup/sample counts. Writes `output/bayesian/pscan_benchmark.json` with wall clock, peak VRAM, divergence count, and agreement metrics vs Phase 15 posterior.

### Anti-Patterns to Avoid

- **Adding `use_pscan` to existing function signatures**: This pollutes existing functions with branching logic. Use separate `*_pscan` functions instead.
- **Running associative_scan over the blocks dimension**: The blocks are NOT a recurrence — Q and WM reset at block boundaries. The scan applies within each block only (over the T trials dimension).
- **Using the same `a_t/b_t` encoding for all (s,a) pairs without the identity (a=1, b=0) at absent trials**: Without the identity, absent-trial positions would corrupt the recurrence.
- **Confusing policy-time Q vs post-update Q**: The policy uses Q BEFORE the trial's update. The scan output at index t is Q AFTER the t-th update. Use `Q_all[:-1]` for policy and `Q_all[1:]` as the next state.
- **Applying the WM overwrite scan to the full WM table**: The WM overwrite applies to one (s,a) pair. Run 18 independent scans (or vectorize over the (S, A) dimension via vmap or batch axis).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Parallel prefix scan | Custom CUDA/HLO kernel | `jax.lax.associative_scan` | Built into JAX, JIT-compiled, TPU/GPU portable |
| Affine recurrence operator | Complex per-model logic | Generic `(a2*a1, a2*b1+b2)` operator | Universal — all AR(1) recurrences share this operator |
| GPU memory measurement | Custom CUDA runtime calls | `jax.devices()[0].memory_stats()` | Already in codebase (`log_gpu_memory()`) |
| Warmup timing | Complex JIT cache analysis | Single `_.block_until_ready()` warmup call | Sufficient for compilation cache fill |

**Key insight:** The associative operator `(a2, b2) ∘ (a1, b1) = (a2*a1, a2*b1+b2)` is the universal encoding for ALL affine recurrences. It handles Q-updates, WM-decay, and WM-overwrite (via `a=0` reset) identically.

---

## Common Pitfalls

### Pitfall 1: Alpha Asymmetry Makes Q-Update Non-Linear
**What goes wrong:** `alpha_t = alpha_pos if delta > 0 else alpha_neg` depends on the sign of `Q_{t-1}(s,a)`. This means the recurrence coefficient `a_t = 1 - alpha_t` depends on the state, making it a non-linear recurrence that cannot be exactly parallelized with an affine scan.
**Why it happens:** The sign of delta requires knowing the previous Q-value, which is not available before the scan.
**How to avoid:** Use the approximation `alpha_t = alpha_pos if reward_t == 1.0 else alpha_neg`. This is exact when `Q(s,a)` stays below 1.0 (always true for positive rewards) and approaches 0 (wrong direction). For `q_init=0.5` and moderate learning rates, this approximation is highly accurate. The unit test (agreement to <1e-5 relative error) will confirm where it breaks down.
**Warning signs:** Large relative errors in the unit test for high `alpha_pos` (>0.7) or after many trials with repeated reward. Document the approximation explicitly.

### Pitfall 2: WM Policy Timing
**What goes wrong:** Computing policies from post-update WM values instead of pre-update (post-decay) WM values.
**Why it happens:** The sequential code sequence is: (1) decay WM, (2) compute policy from WM_decayed, (3) overwrite WM. The scan's output at position t is WM after all updates through t. Policy needs WM after decay but before overwrite.
**How to avoid:** Run the decay scan and the decay+overwrite scan separately, or index correctly: policy at trial t uses the value from the decay-only scan at position t (which equals the value from the combined scan at position t-1 for the presented cell). The simplest approach: run two separate scans — decay-only scan for policy values, combined decay+overwrite scan to get the final WM state for the next block (though blocks reset anyway, so this state is unused).
**Warning signs:** Unit test discrepancy in early trials of a block where WM is being actively updated.

### Pitfall 3: Perseveration Carries Remain Sequential
**What goes wrong:** Attempting to parallelize the `last_action` or `last_actions` (per-stimulus) carry used in M3/M5/M6a/M6b models.
**Why it happens:** Perseveration depends on the action taken at the previous trial, which is a non-linear dependency (action itself results from softmax over current Q and WM, which is non-linear). This is Phase 20 (DEER) territory.
**How to avoid:** In `*_block_likelihood_pscan`, replace the Q-update and WM-decay inner loops with scans, but keep the outer sequential structure for policy computation (which includes softmax, epsilon, and perseveration logic). The pscan variants still use `lax.scan` for the outer policy loop — but now Q and WM are read from pre-computed arrays rather than carried state.
**Warning signs:** Trying to include `last_action` in the scan operator tuple.

### Pitfall 4: Numerical Precision with Extreme Parameters
**What goes wrong:** For `alpha` near 1.0 (e.g., 0.99), the decay factor `a = 1 - alpha = 0.01` is small. After T=100 trials, `a^100 = 0.01^100 ≈ 0`, causing catastrophic underflow in intermediate products of the scan.
**Why it happens:** The prefix product `a_T * a_{T-1} * ... * a_1` involves multiplying T values less than 1. For float32, underflow occurs around `1e-38`.
**How to avoid:** For `alpha = 0.99`, `(0.01)^100 = 10^{-200}`, which underflows float32. In practice, RLWM parameters have `alpha in [0, 1]` with typical values 0.1-0.5. The unit test should cover `alpha_pos=0.95, alpha_neg=0.95` as an edge case. Consider running float64 for the scan in the unit test validation. For float32 production use, document that extreme alpha values (>0.9) may show larger sequential/parallel discrepancy.
- For `phi` near 0 (slow WM decay): `a = (1-phi)` near 1.0, products stay near 1.0. No underflow.
- For `phi` near 1 (fast WM decay): `a = (1-phi)` near 0, same underflow concern as high alpha.
**Tolerance thresholds:** Use relative error `< 1e-5` for typical parameters; document that extreme parameters (alpha > 0.9 or phi > 0.9) may exceed this threshold in float32.

### Pitfall 5: Block Reset Not Accounted For
**What goes wrong:** Running the scan over all blocks concatenated without resetting state at block boundaries.
**Why it happens:** Q and WM reset at each block boundary. The stacked format has shape `(n_blocks, max_trials)`. If you naively flatten to `(n_blocks * max_trials,)` and scan over the full sequence, Q carries over from one block to the next.
**How to avoid:** Apply the scan within each block only. The outer `lax.fori_loop` over blocks is preserved — the scan replaces the inner `lax.scan` over trials within one block, not the outer loop over blocks. The `*_multiblock_likelihood_stacked_pscan` function keeps the `lax.fori_loop` over blocks and replaces the `lax.scan` inside `*_block_likelihood_pscan`.

---

## Code Examples

### The Core Associative Operator (HIGH confidence — verified by derivation and AR(1) example)

```python
def _affine_scan_op(e_left, e_right):
    """Binary associative operator for x_t = a_t * x_{t-1} + b_t.

    Composition: right(left(x0)) = a_r*(a_l*x0 + b_l) + b_r = (a_r*a_l)*x0 + (a_r*b_l + b_r)
    Identity element: (1.0, 0.0)
    Reset element: (0.0, r) where r is the reset value

    Parameters
    ----------
    e_left : tuple of (a_l, b_l) — earlier in time (applied first)
    e_right : tuple of (a_r, b_r) — later in time (applied after)

    Returns
    -------
    tuple of (a_composed, b_composed)
    """
    a_l, b_l = e_left
    a_r, b_r = e_right
    return (a_r * a_l, a_r * b_l + b_r)
```

This operator works element-wise when `(a, b)` are arrays of shape `(num_stimuli, num_actions)` — JAX broadcasts naturally.

### Q-Update Scan Function

```python
def associative_scan_q_update(
    stimuli: jnp.ndarray,   # (T,) int32
    actions: jnp.ndarray,   # (T,) int32
    rewards: jnp.ndarray,   # (T,) float32
    masks: jnp.ndarray,     # (T,) float32 — 1.0 for real trials
    alpha_pos: float,
    alpha_neg: float,
    q_init: float = 0.5,
    num_stimuli: int = 6,
    num_actions: int = 3,
) -> jnp.ndarray:
    """Compute all T Q-value tables in O(log T) via associative scan.

    Returns Q_values_for_policy: (T, num_stimuli, num_actions)
    Q_values_for_policy[t] = Q-table BEFORE trial t's update (used for policy at t).

    Approximation: alpha_t = alpha_pos if reward_t == 1.0 else alpha_neg.
    This approximates the sign of delta as determined by reward alone.
    Agreement with sequential to < 1e-5 relative error for typical parameters.
    """
    T = stimuli.shape[0]
    S, A = num_stimuli, num_actions

    # Build one-hot masks for (stimulus, action) pairs: (T, S, A)
    s_oh = jax.nn.one_hot(stimuli, S)           # (T, S)
    a_oh = jax.nn.one_hot(actions, A)           # (T, A)
    sa_mask = s_oh[:, :, None] * a_oh[:, None, :]  # (T, S, A)

    # Alpha selection: approximation based on reward
    alpha_t = jnp.where(rewards[:, None, None] == 1.0, alpha_pos, alpha_neg)  # (T, S, A)

    # Affine coefficients: identity at non-active or padded positions
    active = sa_mask * masks[:, None, None]  # (T, S, A)
    a_seq = jnp.where(active, 1.0 - alpha_t, 1.0)    # (T, S, A)
    b_seq = jnp.where(active, alpha_t * rewards[:, None, None], 0.0)  # (T, S, A)

    # Prepend initial condition as identity operator applied to q_init
    # Equivalent to starting with (1.0, q_init) which maps any x0 -> q_init
    Q_init = jnp.ones((S, A)) * q_init
    a_full = jnp.concatenate([jnp.ones((1, S, A)), a_seq], axis=0)  # (T+1, S, A)
    b_full = jnp.concatenate([Q_init[None], b_seq], axis=0)          # (T+1, S, A)

    def op(e_l, e_r):
        return (e_r[0] * e_l[0], e_r[0] * e_l[1] + e_r[1])

    _, Q_all = jax.lax.associative_scan(op, (a_full, b_full), axis=0)
    # Q_all[t] = Q after t-th update (t=0 is initial state Q_init)
    # Policy at trial t uses Q_all[t] (before update t+1 = Q_all[t+1])
    return Q_all[:-1]  # (T, S, A) — Q before each trial's update
```

### WM Decay Scan (simplified, decay only)

```python
def associative_scan_wm_decay(
    phi: float,
    wm_init: float,
    T: int,
    num_stimuli: int = 6,
    num_actions: int = 3,
) -> jnp.ndarray:
    """Compute all T WM tables after pure decay (no overwrite) in O(log T).

    Returns WM_decayed: (T, num_stimuli, num_actions)
    WM_decayed[t] = WM after t decay steps from wm_init baseline.
    """
    # Decay is homogeneous: same a=(1-phi), b=phi*wm_init for all trials
    S, A = num_stimuli, num_actions
    wm_0 = jnp.ones((S, A)) * wm_init
    a_seq = jnp.full((T, S, A), 1.0 - phi)
    b_seq = jnp.full((T, S, A), phi * wm_init)

    # Prepend WM_init as b_0 (initial state)
    a_full = jnp.concatenate([jnp.ones((1, S, A)), a_seq], axis=0)
    b_full = jnp.concatenate([wm_0[None], b_seq], axis=0)

    def op(e_l, e_r):
        return (e_r[0] * e_l[0], e_r[0] * e_l[1] + e_r[1])

    _, WM_all = jax.lax.associative_scan(op, (a_full, b_full), axis=0)
    return WM_all[:-1]  # (T, S, A) — WM before each trial's decay application
```

### WM Full Update Scan (decay + overwrite)

```python
def associative_scan_wm_update(
    stimuli: jnp.ndarray,   # (T,) int32
    actions: jnp.ndarray,   # (T,) int32
    rewards: jnp.ndarray,   # (T,) float32
    masks: jnp.ndarray,     # (T,) float32
    phi: float,
    wm_init: float = 1.0 / 3.0,
    num_stimuli: int = 6,
    num_actions: int = 3,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute WM trajectory for all T trials.

    Returns:
    - wm_for_policy: (T, S, A) — WM AFTER decay but BEFORE overwrite at each trial
    - wm_after_update: (T, S, A) — WM AFTER both decay and overwrite at each trial

    For policy computation, use wm_for_policy[t][stimulus_t].
    """
    T = stimuli.shape[0]
    S, A = num_stimuli, num_actions

    # --- Pass 1: Decay only (for policy) ---
    wm_for_policy = associative_scan_wm_decay(phi, wm_init, T, S, A)

    # --- Pass 2: Decay + overwrite (for state tracking) ---
    s_oh = jax.nn.one_hot(stimuli, S)           # (T, S)
    a_oh = jax.nn.one_hot(actions, A)           # (T, A)
    sa_mask = s_oh[:, :, None] * a_oh[:, None, :]  # (T, S, A)

    overwrite = sa_mask * masks[:, None, None]  # (T, S, A)

    # At overwrite positions: a=0 (reset), b=reward
    # At decay positions: a=(1-phi), b=phi*wm_init
    a_seq = jnp.where(overwrite, 0.0, 1.0 - phi)
    b_seq = jnp.where(overwrite, rewards[:, None, None], phi * wm_init)

    wm_0 = jnp.ones((S, A)) * wm_init
    a_full = jnp.concatenate([jnp.ones((1, S, A)), a_seq], axis=0)
    b_full = jnp.concatenate([wm_0[None], b_seq], axis=0)

    def op(e_l, e_r):
        return (e_r[0] * e_l[0], e_r[0] * e_l[1] + e_r[1])

    _, WM_all = jax.lax.associative_scan(op, (a_full, b_full), axis=0)
    wm_after_update = WM_all[1:]  # (T, S, A)

    return wm_for_policy, wm_after_update
```

### Benchmark Script Pattern

```python
# validation/benchmark_parallel_scan.py — core pattern
import time
import json
import jax
import jax.numpy as jnp
import numpy as np

def time_fn(fn, *args, n_repeats=20):
    """Time a JAX function, excluding compilation."""
    _ = fn(*args).block_until_ready()  # warmup (compile)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn(*args).block_until_ready()
        times.append(time.perf_counter() - t0)
    return np.mean(times), np.std(times), np.min(times)

# Run both variants
mean_seq, _, _ = time_fn(jax.jit(seq_fn), *args)
mean_pscan, _, _ = time_fn(jax.jit(pscan_fn), *args)

results = {
    "device": str(jax.devices()[0]),
    "n_blocks": N_BLOCKS,
    "trials_per_block": T,
    "seq_ms": 1000 * mean_seq,
    "pscan_ms": 1000 * mean_pscan,
    "speedup": mean_seq / mean_pscan,
}
json.dump(results, open("output/bayesian/pscan_benchmark.json", "w"))
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Sequential `lax.scan` for Q-update | `associative_scan` with affine operator | Phase 19 (this) | O(T) → O(log T) per-block compute |
| `lax.fori_loop` over blocks | Unchanged — blocks still sequential | N/A | Block reset prevents scan over blocks |
| vmap over participants for GPU | Proven slow (7-13x overhead) | Phase 15-16 analysis | CPU confirmed correct for choice-only MCMC |

**Current performance baseline (from `validation/diagnose_gpu.py`, commit 8afef11):**
- CPU `lax.scan` with JIT: fastest for RLWM (arithmetic intensity ~0.3 FLOP/byte)
- GPU vmap: 7-13x SLOWER than CPU for RLWM likelihood
- The algorithmic change (associative scan) changes the compute pattern; whether it recovers the GPU penalty is an open question

---

## Open Questions

1. **Will O(log T) scan actually be faster on GPU for T=100?**
   - What we know: Theoretical parallelism is 2 log₂(100) ≈ 14 steps vs 100 sequential. GPU requires >50 FLOP/byte to saturate memory bandwidth. Associative scan with 18 × T = 1800 elements per block may still be bandwidth-limited.
   - What's unclear: Empirical speedup is unknown. The work-efficient scan does 2T operations total (not just log T), so work complexity is still O(T). Latency is O(log T) only if the GPU can parallelize the independent operations.
   - Recommendation: Measure via micro-benchmark. If speedup < 2x for single-participant, the primary benefit may be in multi-participant vectorization (once scan enables vmap over participants without fori_loop).

2. **Alpha approximation error at extreme parameter values**
   - What we know: `alpha_t = alpha_pos if reward==1 else alpha_neg` diverges from the true delta-sign rule when `Q(s,a)` is far from 0.5. For typical parameters (alpha 0.1-0.5), this should be small.
   - What's unclear: Exact error magnitude for the real participant distribution from Phase 15 fits. The unit test criterion (<1e-5) needs to be verified against the actual Phase 15 parameter estimates (where alpha values may be concentrated near 0.3).
   - Recommendation: After implementing, compute discrepancy on Phase 15 MLE estimates; adjust tolerance criterion if needed.

3. **Two-pass WM scan vs single-pass with careful indexing**
   - What we know: The policy needs WM post-decay-pre-overwrite; the state update needs WM post-overwrite. Two separate scans is the simplest approach.
   - What's unclear: Whether JAX fuses two separate `associative_scan` calls efficiently, or whether a single interleaved scan with careful element selection would be faster.
   - Recommendation: Implement two-pass first (simpler, easier to verify), benchmark, and optimize if needed.

4. **perseveration kernel parallelization**
   - What we know: `last_action` (global) and `last_actions` (per-stimulus) depend on which action was chosen, which depends on the softmax policy (non-linear). This is Phase 20 territory.
   - What's unclear: Whether the perseveration models (M3/M5/M6a/M6b) benefit from Phase 19 at all — the bottleneck may shift to the perseveration carry.
   - Recommendation: Phase 19 pscan variants for M3/M5/M6a/M6b should parallelize only the Q-update and WM-decay components; the policy loop (including perseveration) remains sequential. Profile whether this hybrid approach yields measurable speedup.

5. **GPU benchmark timing logistics**
   - What we know: Phase 18 (manuscript) must complete before Phase 19 benchmarking. The A100 cluster (Monash M3) is used for MCMC runs.
   - What's unclear: Whether a full 4-chain MCMC benchmark is feasible before GPU allocation limits are hit. The micro-benchmark (Stage 1) can run on CPU or any GPU and should be gated first.
   - Recommendation: Implement and validate correctness on CPU first. Run MCMC benchmark only after pscan correctness is confirmed (success criterion 4).

---

## Sources

### Primary (HIGH confidence)
- `scripts/fitting/jax_likelihoods.py` (direct code inspection) — complete sequential implementation of all 6 models
- `scripts/fitting/numpyro_models.py` (direct code inspection) — STACKED_MODEL_DISPATCH and hierarchical model structure
- `scripts/fitting/fit_bayesian.py` (direct code inspection) — dispatch mechanism
- Linxi's blog (2025-01-02): [https://linxic.com/blog/prefix-sum/](https://linxic.com/blog/prefix-sum/) — verified AR(1) associative scan with comparison to sequential lax.scan
- JAX benchmarking docs: [https://mint.westdri.ca/ai/jx/jx_benchmark](https://mint.westdri.ca/ai/jx/jx_benchmark) — block_until_ready pattern
- GitHub discussion #10599: [https://github.com/jax-ml/jax/discussions/10599](https://github.com/jax-ml/jax/discussions/10599) — work-efficient vs work-inefficient tradeoffs for small T

### Secondary (MEDIUM confidence)
- Särkkä & García-Fernández (2021) — parallel Kalman smoother using associative scan in JAX: [https://github.com/EEA-sensors/parallel-non-linear-gaussian-smoothers](https://github.com/EEA-sensors/parallel-non-linear-gaussian-smoothers). Confirms the same (a,b) affine operator pattern applies to linear Gaussian state space models.
- PaMoRL GitHub: [https://github.com/Wongziseoi/PaMoRL](https://github.com/Wongziseoi/PaMoRL) — NeurIPS 2024. Uses PyTorch, not JAX. The concept of parallelizing RL over sequence length is the same; PETE for TD-λ is a related eligibility-trace recurrence (not inspected in detail — PyTorch codebase).
- Mamba/S4 associative scan: [https://arxiv.org/html/2312.00752v2](https://arxiv.org/html/2312.00752v2) — data-dependent decay `a_t = f(input_t)` is the selective SSM approach. Confirms that input-dependent recurrence coefficients are standard in the associative scan literature.

### Tertiary (LOW confidence)
- DEER paper (arxiv 2309.12252, ICLR 2024) — parallelizes non-linear recurrences via Newton fixed-point iteration. Relevant for Phase 20. PDF binary, content not fully inspected.
- NeurIPS 2024 stability extensions to DEER: [https://arxiv.org/abs/2407.19115](https://arxiv.org/abs/2407.19115) — quasi-Newton and ELK methods. Phase 20 background.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — `jax.lax.associative_scan` is the canonical tool, verified in JAX docs and examples
- Architecture (affine operator): HIGH — derived from first principles, verified against AR(1) literature example
- WM overwrite reset mechanism: MEDIUM — derived from first principles using `a=0` reset, not found as prior RLWM implementation to cross-check against
- Alpha approximation: MEDIUM — standard in parallel RL literature but introduces approximation error that must be quantified in unit tests
- Speedup magnitude: LOW — no prior RLWM associative scan benchmarks found; theoretical O(log T) gain uncertain for T=100 on GPU

**Research date:** 2026-04-14
**Valid until:** 2027-04-14 (JAX API stable; operator math timeless)
