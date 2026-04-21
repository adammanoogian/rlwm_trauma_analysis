# Phase 10 Research: M6b Dual Perseveration

**Researched:** 2026-04-02
**Confidence:** HIGH (all findings from direct codebase inspection of M3 and M6a implementations)

---

## Executive Summary

M6b combines both the global perseveration kernel from M3 (`kappa`) and the stimulus-specific kernel from M6a (`kappa_s`) in a single model. The constraint `kappa + kappa_s <= 1` is enforced via a stick-breaking reparameterization. The carry must hold BOTH a scalar `last_action` (for the global kernel) AND a `(num_stimuli,)` `last_actions` array (for the stimulus-specific kernel). The choice equation is a three-way weighted sum: `(1 - kappa - kappa_s) * P_noisy + kappa * Ck_global + kappa_s * Ck_stim`. M6b has 8 free parameters: same as M5, but the new params are `kappa_total` and `kappa_share` (unconstrained, decode to `kappa` and `kappa_s`) rather than `kappa` and `phi_rl`. The downstream extension pattern is identical to M5/M6a: add `elif model == 'wmrl_m6b':` blocks everywhere.

---

## Q1: The Dual Carry Structure

M3 carry:
```
(Q_table, WM_table, WM_baseline, log_lik_accum, last_action)
#                                               scalar int32
```

M6a carry:
```
(Q_table, WM_table, WM_baseline, log_lik_accum, last_actions)
#                                               shape (num_stimuli,) int32
```

M6b carry must hold BOTH:
```
(Q_table, WM_table, WM_baseline, log_lik_accum, last_action, last_actions)
#                                               scalar int32  shape (S,) int32
```

**Why both?** The global kernel (M3-style) applies based on the most recent action taken anywhere in the block (`last_action`). The stimulus-specific kernel (M6a-style) applies based on the last action taken for THIS stimulus in the block (`last_actions[stimulus]`). They are structurally independent and must both be tracked.

**Alternative rejected:** One might think to derive the global `last_action` from the `last_actions` array (e.g., track a separate "most recent overall" index). But that adds complexity without benefit. Simply carrying both, as M3 and M6a already do independently, is cleaner and avoids any ambiguity in JAX's static typing for scan carries.

**Initialization:**
```python
last_action_init = -1  # Python int, same as M3
last_actions_init = jnp.full((num_stimuli,), -1, dtype=jnp.int32)
init_carry = (Q_init, WM_init, WM_0, 0.0, last_action_init, last_actions_init)
```

**Block boundary reset:** Free, as with M3/M6a. Each call to `wmrl_m6b_block_likelihood` reinitializes the carry from scratch. The multiblock loop calls per block, so both `last_action` and `last_actions` reset at each block boundary automatically.

---

## Q2: The Stick-Breaking Reparameterization

### Why it's needed

`kappa` and `kappa_s` are both probabilities in `[0, 1]`, but they appear together in:
```
P_M6b = (1 - kappa - kappa_s) * P_noisy + kappa * Ck_global + kappa_s * Ck_stim
```

For this to be a valid probability mixture, `kappa + kappa_s <= 1` is required. If this is violated during optimization, probabilities go negative (or sum beyond 1), producing `log(negative)` = NaN in JAX. This crashes optimization silently.

### The reparameterization

Introduce two free parameters in `[0, 1]`:
- `kappa_total` (in `[0, 1]`): total amount of perseveration budget
- `kappa_share` (in `[0, 1]`): fraction of that budget allocated to the global kernel

Then:
```
kappa   = kappa_total * kappa_share
kappa_s = kappa_total * (1 - kappa_share)
```

This guarantees `kappa + kappa_s = kappa_total <= 1` by construction, and both `kappa >= 0`, `kappa_s >= 0`.

**Bounds for new parameters:**
- `kappa_total`: `(0.0, 1.0)` — same as M3's `kappa`
- `kappa_share`: `(0.0, 1.0)` — fraction, no prior reason to restrict

**Interpretive meaning:**
- `kappa_total = 0.0`: no perseveration at all (M2-equivalent)
- `kappa_share = 0.0`: all perseveration is stimulus-specific (reduces to M6a)
- `kappa_share = 1.0`: all perseveration is global (reduces to M3)
- `kappa_total = 0.3, kappa_share = 0.5`: equal split (0.15 global, 0.15 stimulus-specific)

### How transforms work

The two new parameters (`kappa_total`, `kappa_share`) both use the same `jax_unbounded_to_bounded(x, 0.0, 1.0)` transform as M3's `kappa`. The stick-breaking decode happens INSIDE the likelihood function (or at the boundary of the objective function), not in the transform layer.

There are two valid design choices:

**Option A: Decode inside likelihood function**
- `wmrl_m6b_block_likelihood` takes `kappa_total, kappa_share` directly and decodes internally
- Transform layer returns `(alpha_pos, alpha_neg, phi, rho, capacity, kappa_total, kappa_share, epsilon)`
- Decode: `kappa = kappa_total * kappa_share; kappa_s = kappa_total * (1 - kappa_share)`

**Option B: Decode in objective function**
- `wmrl_m6b_block_likelihood` takes `kappa, kappa_s` as decoded values (simpler signature)
- The objective function calls the transform, decodes to `kappa`/`kappa_s`, then calls likelihood
- Consistent with M3 where `kappa` is directly passed

**Recommendation: Option B** (decode in objective, pass decoded values to likelihood). This keeps the likelihood function interface consistent (uses `kappa` and `kappa_s` directly, matching M3/M6a conventions) and makes the likelihood easier to test. The objective functions (`_make_jax_objective_wmrl_m6b`, `_gpu_objective_wmrl_m6b`) decode before calling.

```python
# In the objective function:
alpha_pos, alpha_neg, phi, rho, capacity, kappa_total, kappa_share, epsilon = \
    jax_unconstrained_to_params_wmrl_m6b(x)
kappa   = kappa_total * kappa_share
kappa_s = kappa_total * (1 - kappa_share)
log_lik = wmrl_m6b_multiblock_likelihood_stacked(
    ..., kappa=kappa, kappa_s=kappa_s, ...
)
```

The likelihood function signature is:
```python
def wmrl_m6b_block_likelihood(..., kappa: float, kappa_s: float, ...):
```

---

## Q3: The Choice Kernel Combination Formula

Based on the M3 and M6a implementations:

**M3:** `P_M3 = (1 - kappa) * P_noisy + kappa * Ck_global`
**M6a:** `P_M6a = (1 - kappa_s) * P_noisy + kappa_s * Ck_stim(stimulus)`

**M6b combines both:**
```
P_M6b = (1 - kappa - kappa_s) * P_noisy + kappa * Ck_global + kappa_s * Ck_stim(stimulus)
```

Where:
- `Ck_global = jnp.eye(num_actions)[jnp.maximum(last_action, 0)]` — one-hot of the most recent action overall
- `Ck_stim = jnp.eye(num_actions)[jnp.maximum(last_actions[stimulus], 0)]` — one-hot of last action for this stimulus

### Gating logic

The existing models use a gate to avoid applying the kernel on first presentation:

- M3: `use_m2_path = jnp.logical_or(kappa == 0.0, last_action < 0)`
- M6a: `use_m2_path = jnp.logical_or(kappa_s == 0.0, last_action_s < 0)`

M6b is more complex: each kernel has its own availability condition:
- Global kernel available: `last_action >= 0`
- Stimulus-specific kernel available: `last_actions[stimulus] >= 0`

**Concrete gating logic for M6b:**

```python
# Global kernel: apply only if last_action exists AND kappa > 0
has_global_kernel = jnp.logical_and(kappa > 0.0, last_action >= 0)
Ck_global = jnp.eye(num_actions)[jnp.maximum(last_action, 0)]

# Stimulus-specific kernel: apply only if this stimulus was seen AND kappa_s > 0
last_action_s = last_actions[stimulus]
has_stim_kernel = jnp.logical_and(kappa_s > 0.0, last_action_s >= 0)
Ck_stim = jnp.eye(num_actions)[jnp.maximum(last_action_s, 0)]

# Effective kernel weights (zero out if condition not met)
eff_kappa   = jnp.where(has_global_kernel, kappa, 0.0)
eff_kappa_s = jnp.where(has_stim_kernel, kappa_s, 0.0)

# Blend
base_weight = 1.0 - eff_kappa - eff_kappa_s
noisy_probs = (
    base_weight * P_noisy +
    eff_kappa   * Ck_global +
    eff_kappa_s * Ck_stim
)
```

This is correct and handles all four cases:
1. Neither kernel available (first trial overall, first trial for this stimulus): `noisy_probs = P_noisy`
2. Only global kernel available (first time seeing this stimulus, but some other action was taken before): global-only mixing
3. Only stim kernel available (this stimulus was seen before, but it's the first trial in block): stim-only mixing
4. Both kernels available: full M6b mixing

Note that after stick-breaking, `kappa + kappa_s = kappa_total <= 1`, so `base_weight = 1 - eff_kappa - eff_kappa_s >= 0` is guaranteed when both kernels are active.

---

## Q4: Parameter List and Count

**M6b has 8 free parameters:**

```
alpha_pos, alpha_neg, phi, rho, capacity, kappa_total, kappa_share, epsilon
```

This is the same count as M5. The parameter positions:
- `[0]` alpha_pos
- `[1]` alpha_neg
- `[2]` phi
- `[3]` rho
- `[4]` capacity
- `[5]` kappa_total  (new, replaces M3's kappa at same index)
- `[6]` kappa_share  (new, replaces M5's phi_rl at same index)
- `[7]` epsilon

**Critical: Parameter ordering must match the transform function signature exactly.**

WMRL_M6B_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa_total', 'kappa_share', 'epsilon']

WMRL_M6B_BOUNDS:
```python
WMRL_M6B_BOUNDS = {
    'alpha_pos':   (0.001, 0.999),
    'alpha_neg':   (0.001, 0.999),
    'phi':         (0.001, 0.999),
    'rho':         (0.001, 0.999),
    'capacity':    (1.0, 7.0),
    'kappa_total': (0.0, 1.0),   # Total perseveration budget
    'kappa_share': (0.0, 1.0),   # Fraction allocated to global kernel
    'epsilon':     (0.001, 0.999),
}
```

Both `kappa_total` and `kappa_share` use 0.0 as lower bound (same as M3's `kappa`) — allowing full reduction to simpler models.

**NOTE on model reduction:** With stick-breaking, M6b reduces to:
- M3 when `kappa_share = 1.0` (all budget to global)
- M6a when `kappa_share = 0.0` (all budget to stimulus-specific)
- M2 when `kappa_total = 0.0` (no perseveration)

---

## Q5: First-Presentation and Block Boundary Handling

### Global kernel (`last_action`)

Identical to M3:
- Block start: `last_action = -1`
- Gate: `last_action >= 0` means at least one trial has been completed in the block
- Update: `jnp.where(valid, action, last_action).astype(jnp.int32)` (unconditional on valid)

### Stimulus-specific kernel (`last_actions`)

Identical to M6a:
- Block start: `last_actions = jnp.full((num_stimuli,), -1, dtype=jnp.int32)`
- Gate per trial: `last_actions[stimulus] >= 0` means this stimulus was seen before in block
- Update: `last_actions.at[stimulus].set(jnp.where(valid, action, last_action_s).astype(jnp.int32))`

### Block boundary reset

Both resets are implicit (same as M3/M6a). The `wmrl_m6b_block_likelihood` function is called fresh per block with `init_carry = (Q_init, WM_init, WM_0, 0.0, -1, last_actions_init)`. No explicit reset code needed in the multiblock loop.

---

## Q6: Full step_fn Structure for M6b

The step_fn integrates M3 and M6a logic:

```python
def step_fn(carry, inputs):
    Q_table, WM_table, WM_baseline, log_lik_accum, last_action, last_actions = carry
    stimulus, action, reward, set_size, valid = inputs

    # 1. DECAY WM
    WM_decayed = (1 - phi) * WM_table + phi * WM_baseline

    # 2. COMPUTE HYBRID POLICY
    q_vals   = Q_table[stimulus]
    wm_vals  = WM_decayed[stimulus]
    omega    = rho * jnp.minimum(1.0, capacity / set_size)
    rl_probs = softmax_policy(q_vals, FIXED_BETA)
    wm_probs = softmax_policy(wm_vals, FIXED_BETA)
    base_probs = omega * wm_probs + (1 - omega) * rl_probs
    base_probs = base_probs / jnp.sum(base_probs)
    P_noisy = apply_epsilon_noise(base_probs, epsilon, num_actions)

    # Global kernel (M3 component)
    has_global = jnp.logical_and(kappa > 0.0, last_action >= 0)
    Ck_global  = jnp.eye(num_actions)[jnp.maximum(last_action, 0)]
    eff_kappa  = jnp.where(has_global, kappa, 0.0)

    # Stimulus-specific kernel (M6a component)
    last_action_s = last_actions[stimulus]
    has_stim       = jnp.logical_and(kappa_s > 0.0, last_action_s >= 0)
    Ck_stim        = jnp.eye(num_actions)[jnp.maximum(last_action_s, 0)]
    eff_kappa_s    = jnp.where(has_stim, kappa_s, 0.0)

    # Blend
    noisy_probs = (
        (1.0 - eff_kappa - eff_kappa_s) * P_noisy
        + eff_kappa   * Ck_global
        + eff_kappa_s * Ck_stim
    )

    log_prob = jnp.log(noisy_probs[action] + 1e-8)
    log_prob_masked = log_prob * valid

    # 3. UPDATE WM
    wm_current = WM_decayed[stimulus, action]
    WM_updated = WM_decayed.at[stimulus, action].set(
        jnp.where(valid, reward, wm_current)
    )

    # 4. UPDATE Q
    q_current = Q_table[stimulus, action]
    delta     = reward - q_current
    alpha     = jnp.where(delta > 0, alpha_pos, alpha_neg)
    q_updated = q_current + alpha * delta
    Q_updated = Q_table.at[stimulus, action].set(
        jnp.where(valid, q_updated, q_current)
    )

    log_lik_new = log_lik_accum + log_prob_masked

    # 5. UPDATE PERSEVERATION STATES
    new_last_action  = jnp.where(valid, action, last_action).astype(jnp.int32)
    new_last_actions = last_actions.at[stimulus].set(
        jnp.where(valid, action, last_action_s).astype(jnp.int32)
    )

    return (Q_updated, WM_updated, WM_baseline, log_lik_new,
            new_last_action, new_last_actions), log_prob_masked
```

---

## Q7: Transform Functions Needed in mle_utils.py

**New bounds dict:** `WMRL_M6B_BOUNDS` (see Q4)
**New param list:** `WMRL_M6B_PARAMS` (see Q4)

**New transform functions** (same pattern as M5/M6a):

```python
def jax_unconstrained_to_params_wmrl_m6b(x: jnp.ndarray) -> tuple:
    """Returns (alpha_pos, alpha_neg, phi, rho, capacity, kappa_total, kappa_share, epsilon)."""
    bounds = WMRL_M6B_BOUNDS
    alpha_pos   = jax_unbounded_to_bounded(x[0], *bounds['alpha_pos'])
    alpha_neg   = jax_unbounded_to_bounded(x[1], *bounds['alpha_neg'])
    phi         = jax_unbounded_to_bounded(x[2], *bounds['phi'])
    rho         = jax_unbounded_to_bounded(x[3], *bounds['rho'])
    capacity    = jax_unbounded_to_bounded(x[4], *bounds['capacity'])
    kappa_total = jax_unbounded_to_bounded(x[5], *bounds['kappa_total'])
    kappa_share = jax_unbounded_to_bounded(x[6], *bounds['kappa_share'])
    epsilon     = jax_unbounded_to_bounded(x[7], *bounds['epsilon'])
    return alpha_pos, alpha_neg, phi, rho, capacity, kappa_total, kappa_share, epsilon

def jax_bounded_to_unconstrained_wmrl_m6b(x: jnp.ndarray) -> jnp.ndarray:
    bounds = WMRL_M6B_BOUNDS
    return jnp.array([
        jax_bounded_to_unbounded(x[0], *bounds['alpha_pos']),
        jax_bounded_to_unbounded(x[1], *bounds['alpha_neg']),
        jax_bounded_to_unbounded(x[2], *bounds['phi']),
        jax_bounded_to_unbounded(x[3], *bounds['rho']),
        jax_bounded_to_unbounded(x[4], *bounds['capacity']),
        jax_bounded_to_unbounded(x[5], *bounds['kappa_total']),
        jax_bounded_to_unbounded(x[6], *bounds['kappa_share']),
        jax_bounded_to_unbounded(x[7], *bounds['epsilon']),
    ])
```

**All switch-case locations in mle_utils.py** that have `elif model == 'wmrl_m5':` and `elif model == 'wmrl_m6a':` need a corresponding `elif model == 'wmrl_m6b':` block. Since M6b has 8 params (same as M5), the structure of each block is closest to M5.

---

## Q8: Objective Functions in fit_mle.py

**Three new objective functions** (parallel to M5/M6a):

```python
def _make_jax_objective_wmrl_m6b(
    stimuli_blocks, actions_blocks, rewards_blocks, set_sizes_blocks, masks_blocks=None
):
    """JIT-compiled unbounded objective; 8 params including kappa_total, kappa_share."""
    ...
    def objective(x):
        alpha_pos, alpha_neg, phi, rho, capacity, kappa_total, kappa_share, epsilon = \
            jax_unconstrained_to_params_wmrl_m6b(x)
        kappa   = kappa_total * kappa_share
        kappa_s = kappa_total * (1 - kappa_share)
        log_lik = wmrl_m6b_multiblock_likelihood_stacked(
            ..., kappa=kappa, kappa_s=kappa_s, ...
        )
        return -log_lik
    return jax.jit(objective)

def _make_bounded_objective_wmrl_m6b(...):
    # Uses bounded params[5]=kappa_total, params[6]=kappa_share
    # Decodes before calling likelihood

def _gpu_objective_wmrl_m6b(x, stimuli, actions, rewards, masks, set_sizes):
    # Unconstrained objective for GPU vmap
    alpha_pos, alpha_neg, phi, rho, capacity, kappa_total, kappa_share, epsilon = \
        jax_unconstrained_to_params_wmrl_m6b(x)
    kappa   = kappa_total * kappa_share
    kappa_s = kappa_total * (1 - kappa_share)
    ...
```

**All `elif model == 'wmrl_m6a':` dispatch blocks in fit_mle.py** need a corresponding `elif model == 'wmrl_m6b':` block. Locations from existing M6a research:
- Warm-up call block (~line 207)
- `_make_jax_objective_*` selection (~line 370)
- `_make_bounded_objective_*` selection (~line 600)
- `_gpu_objective_*` selection (~line 977)
- `to_unc` selection for LHS starts (~line 960)
- `fit_all_gpu` Stage 3 objective selection (~line 983)
- `fit_all_gpu` Stage 5 param transformation (~line 1075)
- Stage 1 set_sizes model list (~line 931)
- `compute_diagnostics` objective_fn selection (~line 1172)

The data signature for M6b is the same as M5/M6a (includes `set_sizes`), so the `else:` path in `_run_one` vmap lambda already handles it.

**Argparse:** add `'wmrl_m6b'` to `--model` choices.

---

## Q9: model_recovery.py Changes

### generate_synthetic_participant

M6b simulation loop needs BOTH global and per-stimulus tracking. The M6a pattern already exists — M6b extends it:

```python
# Block initialization:
last_action  = None          # global (M3-style)
last_actions = {}            # per-stimulus dict (M6a-style)

# Inside trial loop, after computing hybrid_probs:
if model == 'wmrl_m6b':
    kappa_total = params['kappa_total']
    kappa_share = params['kappa_share']
    kappa   = kappa_total * kappa_share
    kappa_s = kappa_total * (1 - kappa_share)
    modified = False
    if last_action is not None and kappa > 0.0:
        hybrid_probs[last_action] += kappa
        modified = True
    if last_actions.get(stimulus) is not None and kappa_s > 0.0:
        hybrid_probs[last_actions[stimulus]] += kappa_s
        modified = True
    if modified:
        hybrid_probs = hybrid_probs.copy() if not modified else hybrid_probs
        hybrid_probs = hybrid_probs / np.sum(hybrid_probs)

# After action is taken:
last_action = action
if model == 'wmrl_m6b':
    last_actions[stimulus] = action
```

Note: The existing code for M3/M5 modifies `hybrid_probs[last_action] += kappa` in-place (without `.copy()` first). For M6b this is fine since the dict/scalar is re-initialized per block.

### Param extraction at top of block loop

When `model == 'wmrl_m6b'`:
- Extract `kappa_total = params['kappa_total']` and `kappa_share = params['kappa_share']`
- Decode: `kappa = kappa_total * kappa_share; kappa_s = kappa_total * (1 - kappa_share)`
- Initialize BOTH `last_action = None` and `last_actions = {}`

### All switch-case locations in model_recovery.py

Follow the M6a pattern — every `elif model == 'wmrl_m6a':` gets a corresponding `elif model == 'wmrl_m6b':` block:
- `get_param_names()`: return `WMRL_M6B_PARAMS`
- `sample_random_params()`: use `WMRL_M6B_BOUNDS`
- `generate_synthetic_participant()` model list for phi/rho/capacity extraction: add `'wmrl_m6b'`
- All other elif chains

---

## Q10: Downstream Scripts (14, 15, 16, 11)

All follow the M5/M6a extension pattern exactly.

### scripts/14_compare_models.py

Add to `patterns` dict:
```python
'M6b': ['wmrl_m6b_individual_fits.csv', 'wmrl_m6b_mle_results.csv'],
```
Add `--m6b` argparse argument parallel to `--m5` and `--m6a`.

### scripts/15_analyze_mle_by_trauma.py

- Add display names: `'kappa_total': r'$\kappa_{total}$'` and `'kappa_share': r'$\kappa_{share}$'`
- Add M6b load path detection
- Add `elif model == 'wmrl_m6b':` to `MODEL_CONFIG` dict
- Add `'wmrl_m6b'` to `--model` choices
- Column rename: detect `kappa_total` and `kappa_share` columns

### scripts/16_regress_parameters_on_scales.py

- Add `'wmrl_m6b'` to `--model` choices
- Add `elif model == 'wmrl_m6b':` param column list:
  ```python
  ['alpha_pos_mean', 'alpha_neg_mean', 'phi_mean', 'rho_mean',
   'wm_capacity_mean', 'kappa_total_mean', 'kappa_share_mean', 'epsilon_mean']
  ```
- Add dynamic column rename for `kappa_total_mean` and `kappa_share_mean`

### scripts/11_run_model_recovery.py

Add `'wmrl_m6b'` to `choices` and to the `'all'` models list.

---

## Q11: Functions Required in jax_likelihoods.py

Three new functions (parallel to M3/M5/M6a):

1. `wmrl_m6b_block_likelihood(stimuli, actions, rewards, set_sizes, alpha_pos, alpha_neg, phi, rho, capacity, kappa, kappa_s, epsilon, num_stimuli, num_actions, q_init, wm_init, mask)`
   - Carries: `(Q, WM, WM_0, log_lik, last_action, last_actions)` — dual carry
   - `kappa` and `kappa_s` are decoded values, passed directly (not `kappa_total`/`kappa_share`)
   - Full step_fn as shown in Q6

2. `wmrl_m6b_multiblock_likelihood(...)` — fori_loop fast path + Python loop fallback

3. `wmrl_m6b_multiblock_likelihood_stacked(...)` — fast stacked version for JIT/GPU

---

## Q12: AIC Table Membership

M6b has 8 parameters — same as M5. It belongs in the **choice-only** AIC/BIC comparison table alongside M1 (3), M2 (6), M3 (7), M5 (8), M6a (7). M6b does NOT belong in the M4 (LBA joint choice+RT) comparison track, which is a separate table.

The success criterion "M6b appears in the choice-only AIC/BIC comparison table alongside M1-M3, M5, and M6a" is met by adding the M6b fits file to `14_compare_models.py`.

---

## Architecture Decisions

### Stick-breaking design choice: `kappa_total` + `kappa_share` vs `kappa` + `kappa_s` with simplex constraint

**Alternative approach:** Keep `kappa` and `kappa_s` as direct free parameters and add a `jnp.minimum(kappa + kappa_s, 1.0)` clip. This is simpler but has a pathological optimization landscape: the constraint is active over a large region, creating a flat ridge where gradients are zero.

**Chosen approach:** `kappa_total * kappa_share` parameterization. The unconstrained optimization space is smooth (no constraint boundary to worry about), and both new parameters have the same `[0,1]` bounds as M3's `kappa`. The reparameterization is mathematically clean.

### Decoded `kappa`/`kappa_s` in likelihood vs raw `kappa_total`/`kappa_share`

The likelihood function (`wmrl_m6b_block_likelihood`) takes decoded `kappa` and `kappa_s`. The decoding (`kappa = kappa_total * kappa_share`) happens in the objective functions. This keeps the likelihood interface interpretable and testable independently (can pass exact `kappa=0.2, kappa_s=0.3` without going through the reparameterization).

### Carry dtype stability

JAX `lax.scan` requires static carry structure. The M6b carry is:
```
(jnp.float32 (S,A), jnp.float32 (S,A), jnp.float32 (S,A), jnp.float32 scalar, jnp.int32 scalar, jnp.int32 (S,))
```

The Python int `-1` for `last_action` is fine (JAX traces it as a concrete static value at trace time and immediately `.astype(jnp.int32)` on update, same as M3). The `jnp.full((num_stimuli,), -1, dtype=jnp.int32)` for `last_actions` is a static-shaped array, satisfying scan.

### Verification test for M6b likelihood

Following M6a's structural verification (NLL diff=0.693147 for 2-stimulus sequence), M6b should be tested:
- `kappa_share = 1.0` → should match M3 exactly (NLL diff = 0.0)
- `kappa_share = 0.0` → should match M6a exactly (NLL diff = 0.0)
- `kappa_total = 0.0` → should match M2 exactly
- Random params → verify `kappa + kappa_s <= 1` always holds by construction

---

## Pitfalls to Avoid

### Pitfall 1: Wrong place for stick-breaking decode

The decode (`kappa = kappa_total * kappa_share`) must happen in the objective/gpu_objective functions, NOT inside `jax_unconstrained_to_params_wmrl_m6b`. The transform function should return `(alpha_pos, ..., kappa_total, kappa_share, epsilon)`. If decoding happens in the transform, the CSV output columns would show derived `kappa`/`kappa_s` rather than the actual free parameters `kappa_total`/`kappa_share`, making result files misleading.

### Pitfall 2: Forgetting to copy hybrid_probs before additive modification in model_recovery.py

The existing M3/M5 generation code does `hybrid_probs[last_action] += kappa` without `.copy()`. This is a NumPy mutation. For M6b it still works (numpy arrays are mutable), but both modifications to the same array are sequential — confirm that the second modification (for `kappa_s`) applies to the already-modified array from the first modification (for `kappa`). The normalization `/ np.sum(hybrid_probs)` at the end handles both combined.

### Pitfall 3: Missing the `jnp.maximum(..., 0)` clamps for both kernels

Just like M3 (line 1153) and M6a (line 2132), both `Ck_global` and `Ck_stim` need the clamp:
```python
Ck_global = jnp.eye(num_actions)[jnp.maximum(last_action, 0)]
Ck_stim   = jnp.eye(num_actions)[jnp.maximum(last_action_s, 0)]
```
Without the clamp, `jnp.eye(num_actions)[-1]` in JAX wraps to the LAST row (index 2 for 3 actions), not an error — giving a silently wrong result.

### Pitfall 4: `base_weight = 1 - eff_kappa - eff_kappa_s` can still underflow

Even with stick-breaking, if gating zeros out `eff_kappa_s` but not `eff_kappa`, the base weight correctly adjusts. But floating-point imprecision near boundaries could make `base_weight` slightly negative. Add `jnp.maximum(base_weight, 0.0)` before computing the blend, or re-normalize after blending.

### Pitfall 5: 14_compare_models.py patterns dict vs argparse — same as M6a

Must add `'M6b'` to BOTH the `patterns` dict (for file auto-detection) AND the argparse (for explicit path). Missing either causes silent omission from AIC table.

### Pitfall 6: fit_all_gpu missing one of the four dispatch points

M6a required all four GPU dispatch points to be explicit `elif`. M6b must do the same. The four points in `fit_all_gpu` are: (1) set_sizes model list, (2) `to_unc` selection, (3) `objective` selection, (4) param transformation in Stage 5. Missing any causes either wrong parameter counts or wrong likelihood function.

### Pitfall 7: Parameter recovery reports `kappa_total`/`kappa_share` not `kappa`/`kappa_s`

The CSV output from MLE fitting stores the fitted free parameters. For M6b, these are `kappa_total` and `kappa_share`. Parameter recovery tests should check recovery of `kappa_total` and `kappa_share` (the actual free parameters), NOT `kappa` and `kappa_s` (derived). The `run_parameter_recovery` function stores `true_{param}` and `recovered_{param}` for each param in `WMRL_M6B_PARAMS`, so this is automatic if the param list is correct.

---

## Summary of Files to Modify

| File | Changes |
|------|---------|
| `scripts/fitting/jax_likelihoods.py` | Add `wmrl_m6b_block_likelihood`, `wmrl_m6b_multiblock_likelihood`, `wmrl_m6b_multiblock_likelihood_stacked` |
| `scripts/fitting/mle_utils.py` | Add `WMRL_M6B_BOUNDS`, `WMRL_M6B_PARAMS`, `jax_unconstrained_to_params_wmrl_m6b`, `jax_bounded_to_unconstrained_wmrl_m6b`, extend all switch-case blocks |
| `scripts/fitting/fit_mle.py` | Add imports, 3 new objective functions, extend all 9+ dispatch blocks, extend argparse; stick-breaking decode in objective functions |
| `scripts/fitting/model_recovery.py` | Extend `get_param_names`, `sample_random_params`, `generate_synthetic_participant` (dual-kernel), all switch-case blocks |
| `scripts/11_run_model_recovery.py` | Add `'wmrl_m6b'` to choices and `all` list |
| `scripts/14_compare_models.py` | Add `'M6b'` to patterns dict, add `--m6b` argparse, add load block |
| `scripts/15_analyze_mle_by_trauma.py` | Add display names for `kappa_total`/`kappa_share`, load path, MODEL_CONFIG entry, extend choices |
| `scripts/16_regress_parameters_on_scales.py` | Extend choices, add M6b param column list, add column rename |

---

## Confidence Assessment

| Area | Confidence | Basis |
|------|------------|-------|
| Dual carry structure (both last_action + last_actions) | HIGH | Direct reading of M3 (lines 1101-1197) and M6a (lines 2074-2182) |
| Stick-breaking formula and reparameterization | HIGH | Mathematical consequence of probability simplex constraint |
| Choice kernel blend formula | HIGH | Algebraic extension of M3 + M6a |
| Gating logic (per-kernel availability) | HIGH | Structural extension of M3/M6a gating patterns |
| Param count (8) and param names | HIGH | Stick-breaking adds kappa_total + kappa_share; epsilon last |
| Transform functions (same bounds as M3 kappa) | HIGH | kappa_total and kappa_share are both in [0,1] |
| Decode placement (in objective, not transform) | HIGH | Design decision rationale: CSV output must show free params |
| Downstream script extension pattern | HIGH | M5 and M6a both established the pattern identically |
| Verification tests (kappa_share=0/1 equivalence) | HIGH | Mathematical identity from stick-breaking construction |
| `jnp.maximum(..., 0)` requirement for both clamps | HIGH | Same JAX indexing behavior as M3/M6a |
