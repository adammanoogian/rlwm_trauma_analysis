# Phase 9 Research: M6a Stimulus-Specific Perseveration

**Researched:** 2026-04-02
**Confidence:** HIGH (all findings from direct codebase inspection)

---

## Executive Summary

M6a replaces the global `last_action` scalar (type `int32`, initialized to `-1`) in the M3 carry with a per-stimulus `last_actions` array of shape `(num_stimuli,)`. The block boundary reset pattern is already established by Q/WM initialization: M6a simply initializes `last_actions = jnp.full((num_stimuli,), -1, dtype=jnp.int32)` at block start. First-presentation handling follows from the existing sentinel: when `last_actions[stimulus] < 0`, no kernel is applied (identical logic to M3's `last_action < 0` gate). The parameter `kappa_s` is semantically identical to `kappa` in M3, same bounds `[0.0, 1.0]`, same transform. M6a branches from M3, not M5: no `phi_rl`. The downstream scripts (14, 15, 16, 11) each follow a rigid `elif model == 'wmrl_m5'` pattern that M6a must extend with `elif model == 'wmrl_m6a'`.

---

## Q1: M3 `last_action` in the lax.scan Carry

### Type and Initialization

```python
# jax_likelihoods.py line 1103
init_carry = (Q_init, WM_init, WM_0, 0.0, -1)
#                                          ^ last_action: Python int -1
```

Inside `step_fn`, it is immediately cast via `.astype(jnp.int32)` on update (line 1191). The sentinel `-1` means "no previous action exists for this block."

### How it is used

```python
# line 1135
use_m2_path = jnp.logical_or(kappa == 0.0, last_action < 0)

# line 1153
choice_kernel = jnp.eye(num_actions)[jnp.maximum(last_action, 0)]  # Clamp for safe indexing

# line 1156
hybrid_probs_m3 = (1 - kappa) * noisy_base + kappa * choice_kernel

# line 1159
noisy_probs = jnp.where(use_m2_path, noisy_base, hybrid_probs_m3)
```

The `jnp.maximum(last_action, 0)` clamp prevents index-out-of-bounds when `last_action == -1` — only safe because `use_m2_path` masks the result.

### Update per trial

```python
# line 1191
new_last_action = jnp.where(valid, action, last_action).astype(jnp.int32)
```

Padding trials (`valid == 0.0`) keep the previous `last_action` unchanged.

### Block boundary reset

The block boundary reset is implicit: `wmrl_m3_block_likelihood` is called fresh per block with `init_carry = (..., -1)`. There is no explicit reset code in the multiblock loop — each call to `wmrl_m3_block_likelihood` reinitializes carry from scratch.

---

## Q2: Changes Needed for `last_actions` Array

### Carry structure change

M3 carry tuple:
```
(Q_table, WM_table, WM_baseline, log_lik_accum, last_action)
#         shape (S,A)                           scalar int32
```

M6a carry tuple:
```
(Q_table, WM_table, WM_baseline, log_lik_accum, last_actions)
#         shape (S,A)                           shape (S,) int32
```

### Initialization

```python
# M6a init_carry
last_actions_init = jnp.full((num_stimuli,), -1, dtype=jnp.int32)
init_carry = (Q_init, WM_init, WM_0, 0.0, last_actions_init)
```

### step_fn changes

```python
# Unpack
Q_table, WM_table, WM_baseline, log_lik_accum, last_actions = carry

# Get stimulus-specific last action
last_action_s = last_actions[stimulus]

# Gate: no kernel if kappa_s == 0 OR stimulus never seen in this block
use_m2_path = jnp.logical_or(kappa_s == 0.0, last_action_s < 0)

# Choice kernel for this stimulus
choice_kernel = jnp.eye(num_actions)[jnp.maximum(last_action_s, 0)]

# Mixing
hybrid_probs_m6a = (1 - kappa_s) * noisy_base + kappa_s * choice_kernel
noisy_probs = jnp.where(use_m2_path, noisy_base, hybrid_probs_m6a)

# Update only this stimulus's slot
new_last_actions = last_actions.at[stimulus].set(
    jnp.where(valid, action, last_action_s).astype(jnp.int32)
)

return (Q_updated, WM_updated, WM_baseline, log_lik_new, new_last_actions), log_prob_masked
```

**Critical JAX note:** `last_actions.at[stimulus].set(...)` is the JAX functional update idiom. `stimulus` is a traced integer during `lax.scan`, so this works with XLA dynamic indexing without issues.

---

## Q3: Block Boundary Reset Pattern

The reset is free: each call to `wmrl_m6a_block_likelihood` reinitializes the carry with `last_actions = jnp.full((num_stimuli,), -1, dtype=jnp.int32)`. The multiblock loop calls `wmrl_m6a_block_likelihood` per block (via `lax.fori_loop` on stacked data or Python loop for variable-sized blocks) — identical structure to M3/M5.

No explicit reset code needed in `wmrl_m6a_multiblock_likelihood`. The pattern is already established.

---

## Q4: First Presentation of Each Stimulus (M6-03: Uniform Fallback)

The existing sentinel `-1` naturally handles this:

- At block start: `last_actions = jnp.full((num_stimuli,), -1, dtype=jnp.int32)`
- When `last_actions[stimulus] < 0`: `use_m2_path = True` → `noisy_probs = noisy_base` (no kernel)
- After first encounter: `last_actions[stimulus]` is set to the action taken

This means each stimulus independently gets the "no kernel on first presentation" behavior. The global `last_action` in M3 applies kernel if *any* action was taken previously in the block — even for a stimulus seen for the first time. M6a corrects this: each stimulus starts fresh within the block.

**No special casing needed** beyond what the `-1` sentinel already provides.

---

## Q5: Parameter Difference Between `kappa` and `kappa_s`

From `mle_utils.py`:
```python
# M3 bounds
'kappa': (0.0, 1.0)  # NOTE: 0.0 allowed (M2 equivalence)
```

`kappa_s` (stimulus-specific perseveration) is semantically a drop-in replacement:
- Same bounds: `[0.0, 1.0]`
- Same transform: `jax_unbounded_to_bounded(x, 0.0, 1.0)` — i.e., logit/sigmoid over `[0, 1]`
- Same default (for recovery starting values): `0.0`

Name the parameter `kappa_s` in param lists and `kappa_s` in the bounds dict.

M6a parameter count: 7 (same as M3: `alpha_pos, alpha_neg, phi, rho, capacity, kappa_s, epsilon`)
M6a parameter order: `alpha_pos, alpha_neg, phi, rho, capacity, kappa_s, epsilon`

**No extra parameters vs M3.** The change is behavioral (per-stimulus tracking), not parametric.

---

## Q6: Does M6a Include `phi_rl`?

No. M6a branches from M3, not M5.

M5 = M3 + phi_rl (RL forgetting)
M6a = M3 + per-stimulus last_actions (replaces global last_action)

M6a does not inherit phi_rl. This is confirmed by the objective: M6a is an alternative perseveration architecture vs M3, competing on the same 7-parameter count. Adding phi_rl would create M6a+M5 = an 8-parameter model, which is a different model.

---

## Q7: Downstream Script Changes (14, 15, 16)

### Pattern established by M5

All downstream scripts follow the identical `elif model == 'wmrl_m5'` extension pattern. M6a adds `elif model == 'wmrl_m6a'` blocks in the same locations.

### `scripts/14_compare_models.py`

File patterns dict at line 545:
```python
patterns = {
    'M1': ['qlearning_individual_fits.csv', ...],
    'M2': ['wmrl_individual_fits.csv', ...],
    'M3': ['wmrl_m3_individual_fits.csv', ...],
    'M5': ['wmrl_m5_individual_fits.csv', ...],
    # ADD:
    'M6a': ['wmrl_m6a_individual_fits.csv', 'wmrl_m6a_mle_results.csv'],
}
```

Argparse at line 576+: add `--m6a` argument parallel to `--m5`.

Load block at line 625+:
```python
if args.m6a:
    fits_dict['M6a'] = load_fits(args.m6a)
```

The comparison table logic is data-driven (iterates `fits_dict`), so no other changes needed for AIC/BIC table generation once the key is added.

### `scripts/15_analyze_mle_by_trauma.py`

Param lists at lines 99-100:
```python
WMRL_M6A_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa_s', 'epsilon']
```

Display name dict at line 116+:
```python
'kappa_s': r'$\kappa_s$',
```

Load path at line 137+: add M6a path detection.
MODEL_CONFIG dict at line 760+: add `wmrl_m6a` entry.
`--model` choices at line 737: add `'wmrl_m6a'`.
`models_to_analyze` list: add `'wmrl_m6a'`.

Column name mapping in `load_data()`: same as M5 pattern — detect `kappa_s` column dynamically (already done generically at lines 163-166 for `kappa` and `phi_rl`, so `kappa_s` just needs adding).

### `scripts/16_regress_parameters_on_scales.py`

`--model` choices at line 672: add `'wmrl_m6a'`.
Param column list at line 775+ (`elif model == 'wmrl_m6a'`):
```python
['alpha_pos_mean', 'alpha_neg_mean', 'phi_mean', 'rho_mean',
 'wm_capacity_mean', 'kappa_s_mean', 'epsilon_mean']
```

Dynamic rename at line 163-166: add `'kappa_s': 'kappa_s_mean'` alongside existing mappings.

---

## Q8: `model_recovery.py` Changes for M6a

### `get_param_names()` (line 78+)

```python
elif model == 'wmrl_m6a':
    return WMRL_M6A_PARAMS  # ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa_s', 'epsilon']
```

### `sample_random_params()` (line 92+)

```python
elif model == 'wmrl_m6a':
    bounds = WMRL_M6A_BOUNDS
    param_names = WMRL_M6A_PARAMS
```

### `generate_synthetic_participant()` (line 170+)

The simulation loop is the key change. Currently M3/M5 use global `last_action`:
```python
last_action = None
...
if model in ('wmrl_m3', 'wmrl_m5') and last_action is not None:
    hybrid_probs[last_action] += kappa
    hybrid_probs = hybrid_probs / np.sum(hybrid_probs)
...
last_action = action
```

M6a needs per-stimulus tracking:
```python
# Block initialization
last_actions = {}  # stimulus -> last action (None = never seen in this block)

# Within trial loop, after computing hybrid_probs:
if model == 'wmrl_m6a' and last_actions.get(stimulus) is not None:
    kappa_s = params['kappa_s']
    hybrid_probs = hybrid_probs.copy()
    hybrid_probs[last_actions[stimulus]] += kappa_s
    hybrid_probs = hybrid_probs / np.sum(hybrid_probs)

# After action is taken:
last_actions[stimulus] = action
```

The `last_actions` dict resets to `{}` at the top of each block iteration (inside `for block_idx in range(NUM_BLOCKS)`).

### `run_parameter_recovery()` and `compare_behavior()` (lines 527+, 629+, 720+, 819+)

Extend `elif model == 'wmrl_m5'` blocks with `elif model == 'wmrl_m6a'` in all switch-case locations.

### `model_recovery.py` argparse (line 919)

Currently: `choices=['qlearning', 'wmrl', 'wmrl_m3']` — this is already out of date (M5 not listed). Add `'wmrl_m6a'`.

### `scripts/11_run_model_recovery.py` (line 129)

```python
choices=['qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'all'],
```

---

## Q9: `mle_utils.py` Changes

### Bounds dict

```python
WMRL_M6A_BOUNDS = {
    'alpha_pos': (0.001, 0.999),
    'alpha_neg': (0.001, 0.999),
    'phi': (0.001, 0.999),
    'rho': (0.001, 0.999),
    'capacity': (1.0, 7.0),
    'kappa_s': (0.0, 1.0),     # Stimulus-specific perseveration (same bounds as kappa)
    'epsilon': (0.001, 0.999),
}
```

### Param list

```python
WMRL_M6A_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa_s', 'epsilon']
```

### Transform functions

Two new functions (parallel to `jax_unconstrained_to_params_wmrl_m3`):

```python
def jax_unconstrained_to_params_wmrl_m6a(x: jnp.ndarray) -> tuple:
    """x[0..4] same as M3. x[5] = kappa_s. x[6] = epsilon."""
    bounds = WMRL_M6A_BOUNDS
    alpha_pos = jax_unbounded_to_bounded(x[0], *bounds['alpha_pos'])
    alpha_neg = jax_unbounded_to_bounded(x[1], *bounds['alpha_neg'])
    phi       = jax_unbounded_to_bounded(x[2], *bounds['phi'])
    rho       = jax_unbounded_to_bounded(x[3], *bounds['rho'])
    capacity  = jax_unbounded_to_bounded(x[4], *bounds['capacity'])
    kappa_s   = jax_unbounded_to_bounded(x[5], *bounds['kappa_s'])
    epsilon   = jax_unbounded_to_bounded(x[6], *bounds['epsilon'])
    return alpha_pos, alpha_neg, phi, rho, capacity, kappa_s, epsilon

def jax_bounded_to_unconstrained_wmrl_m6a(x: jnp.ndarray) -> jnp.ndarray:
    bounds = WMRL_M6A_BOUNDS
    return jnp.array([
        jax_bounded_to_unbounded(x[0], *bounds['alpha_pos']),
        jax_bounded_to_unbounded(x[1], *bounds['alpha_neg']),
        jax_bounded_to_unbounded(x[2], *bounds['phi']),
        jax_bounded_to_unbounded(x[3], *bounds['rho']),
        jax_bounded_to_unbounded(x[4], *bounds['capacity']),
        jax_bounded_to_unbounded(x[5], *bounds['kappa_s']),
        jax_bounded_to_unbounded(x[6], *bounds['epsilon']),
    ])
```

### Switch-case extensions

Every `elif model == 'wmrl_m5':` block in `mle_utils.py` must get a corresponding `elif model == 'wmrl_m6a':` block. From the grep, the affected lines are:
- `params_to_unconstrained()` line 283+
- `unconstrained_to_params()` line 316+
- `get_default_starting_point()` line 349+
- `get_param_names()` (via model switch, line 391+)
- `get_bounds_list()` line 410+
- `get_n_params()` line 508+
- `get_display_names()` line 563+
- `get_param_bounds_dict()` line 690+
- Hessian-related functions line 847+, 994+

---

## Q10: `fit_mle.py` Changes

### Imports (line 88+)

```python
from scripts.fitting.jax_likelihoods import (
    ...
    wmrl_m6a_multiblock_likelihood,
    wmrl_m6a_multiblock_likelihood_stacked,
)
from scripts.fitting.mle_utils import (
    ...
    jax_bounded_to_unconstrained_wmrl_m6a,
    jax_unconstrained_to_params_wmrl_m6a,
)
```

### New objective function

```python
def _make_jax_objective_wmrl_m6a(...):
    # Identical structure to _make_jax_objective_wmrl_m3
    # except: jax_unconstrained_to_params_wmrl_m6a unpacks kappa_s
    # and calls wmrl_m6a_multiblock_likelihood_stacked
```

```python
def _make_bounded_objective_wmrl_m6a(...):
    # Identical structure to _make_bounded_objective_wmrl_m3
    # except calls wmrl_m6a_multiblock_likelihood_stacked
```

```python
def _gpu_objective_wmrl_m6a(x, stimuli, actions, rewards, masks, set_sizes):
    # Identical structure to _gpu_objective_wmrl_m3
```

### Argparse (line 2121)

```python
choices=['qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a'],
help='Model to fit (..., wmrl_m6a=M6a)'
```

### All `elif model == 'wmrl_m5'` dispatch blocks

Locations (from grep): lines 195, 820, 841, 931, 1024, 1259, 1266, 2334. Each needs `elif model == 'wmrl_m6a':` added.

The pattern is mechanical: `wmrl_m6a` slots in exactly like `wmrl_m3` but uses M6a functions and transforms.

---

## Q11: `jax_likelihoods.py` — New Functions Required

Three new functions required (matching M3/M5 pattern):

1. `wmrl_m6a_block_likelihood(stimuli, actions, rewards, set_sizes, alpha_pos, alpha_neg, phi, rho, capacity, kappa_s, epsilon, num_stimuli, num_actions, q_init, wm_init, mask)` — core lax.scan with per-stimulus carry

2. `wmrl_m6a_multiblock_likelihood(...)` — Python-loop / fori_loop multiblock wrapper

3. `wmrl_m6a_multiblock_likelihood_stacked(...)` — fast stacked version for JIT

---

## Architecture Decisions

### `kappa_s` naming

The `kappa_s` suffix distinguishes stimulus-specific from global. Use `kappa_s` everywhere (param names, bounds dict, function arguments, CSV column headers). This avoids collision with M3's `kappa` column in comparison tables.

### Carry dtype stability

JAX `lax.scan` requires carry structure to be stable (same dtypes/shapes across iterations). Initializing `last_actions = jnp.full((num_stimuli,), -1, dtype=jnp.int32)` and always returning `new_last_actions` from `.at[].set()` maintains a static `(num_stimuli,) int32` array — safe for XLA.

### `lax.fori_loop` compatibility

The stacked multiblock version uses `lax.fori_loop`. The carry of the outer loop is just `total_ll: float` — the inner `wmrl_m6a_block_likelihood` call is a pure function with its own `lax.scan`. This nesting works fine (M3 and M5 already use it).

### Backward compatibility with M3

M6a is NOT meant to reduce to M3 when `kappa_s == 0` (it would, mathematically, but per-stimulus tracking is a structural change). No backward compatibility requirement stated in M6-01 through M6-06.

### AIC table placement

M6a has 7 parameters (same as M3), making it directly comparable in the choice-only AIC/BIC table alongside M1 (3 params), M2 (6 params), M3 (7 params), M5 (8 params). M6a belongs in the choice-only table — confirmed by success criteria item 4.

---

## Pitfalls to Avoid

### Pitfall 1: Forgetting `jnp.maximum(last_action_s, 0)` clamp

When `last_action_s == -1`, `jnp.eye(num_actions)[last_action_s]` would index with `-1` — valid Python but wrong in JAX (wraps to last row). The clamp `jnp.maximum(last_action_s, 0)` is required, paired with the `use_m2_path` gate that nullifies the result when `last_action_s < 0`. Copy this pattern exactly from M3 (line 1153).

### Pitfall 2: `last_actions.at[stimulus].set(...)` update must stay outside the `use_m2_path` branch

The `last_actions` array must be updated for every valid trial, regardless of whether the choice kernel was applied. Update it unconditionally (after action sampling), gated only on `valid`.

### Pitfall 3: model_recovery.py `last_actions` dict vs `last_action` scalar

In `generate_synthetic_participant`, the current code uses `last_action = None` (global) and resets it at block start implicitly (variable goes out of scope). For M6a, use `last_actions = {}` (dict keyed by stimulus) initialized at the top of each block loop iteration. Don't accidentally share it across blocks.

### Pitfall 4: model_recovery.py argparse already outdated

The `model_recovery.py` argparse at line 919 only lists `['qlearning', 'wmrl', 'wmrl_m3']` — M5 is already missing. Add both `'wmrl_m5'` and `'wmrl_m6a'` when updating.

### Pitfall 5: 14_compare_models.py patterns dict vs argparse

`find_mle_files()` uses a `patterns` dict for auto-detection. If `'M6a'` is not added there, `--m6a` via argparse works but auto-detection silently misses it.

### Pitfall 6: kappa_s bounds allow 0.0

Same as M3's `kappa`: the lower bound is `0.0` not `0.001`. This allows M6a to reduce to M2-equivalent behavior (no perseveration). The `bounded_to_unbounded` transform handles 0.0 via the `logit(p)` of normalized value — but `p = (0.0 - 0.0) / (1.0 - 0.0) = 0.0` means `logit(0.0) = -inf`. In practice, starting points should use a small positive value (e.g., `0.1`) as the M3 default does.

---

## Summary of Files to Modify

| File | Changes |
|------|---------|
| `scripts/fitting/jax_likelihoods.py` | Add `wmrl_m6a_block_likelihood`, `wmrl_m6a_multiblock_likelihood`, `wmrl_m6a_multiblock_likelihood_stacked`, plus test functions |
| `scripts/fitting/mle_utils.py` | Add `WMRL_M6A_BOUNDS`, `WMRL_M6A_PARAMS`, `jax_unconstrained_to_params_wmrl_m6a`, `jax_bounded_to_unconstrained_wmrl_m6a`, extend all switch-case blocks |
| `scripts/fitting/fit_mle.py` | Add imports, `_make_jax_objective_wmrl_m6a`, `_make_bounded_objective_wmrl_m6a`, `_gpu_objective_wmrl_m6a`, extend all dispatch blocks, extend argparse |
| `scripts/fitting/model_recovery.py` | Extend `get_param_names`, `sample_random_params`, `generate_synthetic_participant` (per-stimulus dict), all switch-case blocks, fix argparse choices |
| `scripts/11_run_model_recovery.py` | Add `'wmrl_m6a'` to choices, add to `all` list |
| `scripts/14_compare_models.py` | Add `'M6a'` to patterns dict, add `--m6a` argparse, add load block |
| `scripts/15_analyze_mle_by_trauma.py` | Add `WMRL_M6A_PARAMS`, `kappa_s` display name, M6a load path, MODEL_CONFIG entry, extend choices |
| `scripts/16_regress_parameters_on_scales.py` | Extend choices, add M6a param column list, add `kappa_s_mean` rename |

---

## Confidence Assessment

| Area | Confidence | Basis |
|------|------------|-------|
| M3 carry structure | HIGH | Direct code reading, lines 1101-1197 |
| Block boundary reset mechanism | HIGH | Direct code reading, multiblock functions |
| First-presentation sentinel logic | HIGH | `-1` sentinel pattern clear at line 1135 |
| kappa_s bounds and transform | HIGH | M3 bounds at mle_utils.py lines 47-56 |
| M6a param count (7, same as M3) | HIGH | No new params vs M3, just structural change |
| Downstream script change locations | HIGH | Grep results show all `elif model == 'wmrl_m5'` locations |
| model_recovery.py synthetic gen pattern | HIGH | Direct code reading, lines 170-345 |
| JAX `.at[].set()` for dynamic indexing | HIGH | Standard JAX functional update idiom |
| lax.scan carry dtype stability requirement | HIGH | JAX documentation, confirmed by existing pattern |
