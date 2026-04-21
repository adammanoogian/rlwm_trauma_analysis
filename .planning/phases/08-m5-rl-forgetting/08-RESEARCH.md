# Phase 8: M5 RL Forgetting - Research

**Researched:** 2026-04-02
**Domain:** JAX computational modeling pipeline extension (codebase archaeology)
**Confidence:** HIGH — all findings are from direct source inspection, no external lookup required

---

## Summary

This phase adds one parameter (phi_RL) to the existing M3 model (wmrl_m3). The implementation pattern is
completely established by how M3 extended M2 (kappa perseveration). All seven touch points have been
mapped from source. No external libraries are needed and no new architectural decisions are required.

The carry structure of M3's `lax.scan` body passes `(Q_table, WM_table, WM_baseline, log_lik, last_action)`
through trials. M5 inserts a global Q-value decay step at the top of that body (before the delta-rule
update), mirroring exactly how WM decay (`WM_decayed = (1-phi)*WM + phi*WM_baseline`) already works.

Parameter registration follows a rigid, fully parallel pattern across mle_utils.py and fit_mle.py.
Every new model requires adding to the same set of dicts/conditionals in both files — no registry
object, just explicit `elif model == 'wmrl_m5':` branches. Scripts 15 and 16 already handle new models
via their own `elif model == 'wmrl_m5':` branches in the param-column definitions; they do NOT need
global architectural changes, only those branch additions.

**Primary recommendation:** Implement exactly the M3-to-M5 delta. Copy M3 code blocks, substitute
`wmrl_m5` for `wmrl_m3` throughout, add the phi_RL decay step at position 1a in the scan body, and
add phi_RL to every parallel registration site.

---

## Standard Stack

All libraries are already present in the environment. No new dependencies.

### Core

| Library | Purpose | Notes |
|---------|---------|-------|
| JAX / jax.lax.scan | Sequential trial processing, JIT | M3 pattern directly reused |
| jaxopt.ScipyBoundedMinimize | L-BFGS-B optimization | Already used for M3 |
| jax.jit / jax.vmap | JIT and GPU vectorization | Already used for M3 |
| numpy / scipy.stats | Recovery metrics, AIC/BIC | Already present |
| pandas | Data loading, result tables | Already present |

### No New Dependencies

M5 adds one float parameter to an existing scan body. Zero new library calls.

---

## Architecture Patterns

### Pattern: M3 carry structure

The M3 `lax.scan` carry is a 5-tuple:

```python
init_carry = (Q_init, WM_init, WM_baseline, 0.0, -1)
# fields:     Q_table  WM_table  WM_0      log_lik  last_action
```

The scan body signature:

```python
def step_fn(carry, inputs):
    Q_table, WM_table, WM_baseline, log_lik_accum, last_action = carry
    stimulus, action, reward, set_size, valid = inputs
```

**M5 carry is identical** — no new carry fields needed. phi_RL is accessed from closure
(same as phi, rho, alpha_pos, etc.).

### Pattern: The exact update ordering in M3

```python
# Step 1: WM decay (BEFORE policy)
WM_decayed = (1 - phi) * WM_table + phi * WM_baseline

# Step 2: Policy computation (uses decayed WM)
...

# Step 3: WM update (immediate overwrite)
WM_updated = WM_decayed.at[stimulus, action].set(jnp.where(valid, reward, wm_current))

# Step 4: Q update (delta-rule, AFTER policy)
delta = reward - q_current
alpha = jnp.where(delta > 0, alpha_pos, alpha_neg)
q_updated = q_current + alpha * delta
Q_updated = Q_table.at[stimulus, action].set(jnp.where(valid, q_updated, q_current))
```

### M5 update ordering (required by M5-02)

M5 inserts a GLOBAL Q decay step between Step 1 and Step 2 — call it Step 1a:

```python
# Step 1: WM decay (same as M3)
WM_decayed = (1 - phi) * WM_table + phi * WM_baseline

# Step 1a: RL forgetting (NEW — global decay toward Q0=1/3 BEFORE delta-rule)
Q0 = 1.0 / 3.0  # = 0.333... matches wm_init baseline convention
Q_decayed = (1 - phi_rl) * Q_table + phi_rl * Q0

# Step 2: Policy computation (uses Q_decayed instead of Q_table)
q_vals = Q_decayed[stimulus]          # NOT Q_table[stimulus]
...

# Step 3: WM update (unchanged)
...

# Step 4: Q update (applies delta-rule to Q_decayed, not Q_table)
q_current = Q_decayed[stimulus, action]   # NOT Q_table[stimulus, action]
delta = reward - q_current
alpha = jnp.where(delta > 0, alpha_pos, alpha_neg)
q_updated = q_current + alpha * delta
Q_updated = Q_decayed.at[stimulus, action].set(jnp.where(valid, q_updated, q_current))
```

**Critical detail:** After Step 1a, all subsequent Q references within the scan body
must use `Q_decayed`, not `Q_table`. The carry update returns `Q_updated` (which is
derived from `Q_decayed`).

### Pattern: Backward compatibility (phi_RL = 0)

When `phi_rl = 0.0`:
```
Q_decayed = (1 - 0) * Q_table + 0 * Q0 = Q_table
```
Mathematically identical to M3 for all subsequent steps. This satisfies M5-06.

The M3 analogue is `kappa=0` bypassing the choice kernel. M5 uses a simpler identity
(pure algebraic equivalence at phi_rl=0), not a conditional branch.

### Pattern: Global all-pairs decay (M5-03)

`Q_decayed = (1 - phi_rl) * Q_table + phi_rl * Q0` operates on the full
`(num_stimuli, num_actions)` matrix. This is the same shape as `Q_table` and WM_table.
No indexing by stimulus needed — the entire matrix decays every trial.

This matches how WM decay works: `WM_decayed = (1 - phi) * WM_table + phi * WM_baseline`
also operates on the full matrix. M5-03 is satisfied automatically by the same idiom.

### Recommended Project Structure for new files

```
scripts/fitting/jax_likelihoods.py    # ADD: wmrl_m5_block_likelihood,
                                      #      wmrl_m5_multiblock_likelihood,
                                      #      wmrl_m5_multiblock_likelihood_stacked

scripts/fitting/mle_utils.py          # ADD: WMRL_M5_BOUNDS, WMRL_M5_PARAMS,
                                      #      jax_unconstrained_to_params_wmrl_m5,
                                      #      jax_bounded_to_unconstrained_wmrl_m5

scripts/fitting/fit_mle.py            # ADD: _make_bounded_objective_wmrl_m5,
                                      #      _make_jax_objective_wmrl_m5,
                                      #      _gpu_objective_wmrl_m5,
                                      #      elif wmrl_m5 branches in:
                                      #        warmup_jax_compilation()
                                      #        fit_participant_mle()
                                      #        fit_all_gpu()
                                      #        main() argparse choices

scripts/fitting/model_recovery.py     # ADD: elif wmrl_m5 branches in:
                                      #        get_param_names()
                                      #        sample_parameters()
                                      #        generate_synthetic_participant()

scripts/15_analyze_mle_by_trauma.py   # ADD: WMRL_M5_PARAMS list,
                                      #      load_data() loads wmrl_m5 CSV,
                                      #      MODEL_CONFIG entry

scripts/16_regress_parameters_on_scales.py  # ADD: elif wmrl_m5 branch
                                            #      in load_integrated_data()
                                            #      param_rename block,
                                            #      elif wmrl_m5 in param_cols
                                            #      argparse choices update

scripts/14_compare_models.py          # Needs --m5 argument OR user passes
                                      # wmrl_m5_individual_fits.csv via --m3
                                      # (existing fits_dict pattern handles it)
```

---

## Complete Touch-Point Map

### 1. `jax_likelihoods.py`

**What to add:**

a) `wmrl_m5_block_likelihood(stimuli, actions, rewards, set_sizes, alpha_pos, alpha_neg,
   phi, rho, capacity, kappa, phi_rl, epsilon, ..., mask)` — same signature as M3 +
   one new `phi_rl: float` parameter. Inject Step 1a inside scan body.

b) `wmrl_m5_multiblock_likelihood(...)` — copy of `wmrl_m3_multiblock_likelihood`,
   forwarding `phi_rl` down to block function. Fast path uses `fori_loop`.

c) `wmrl_m5_multiblock_likelihood_stacked(...)` — copy of `wmrl_m3_multiblock_likelihood_stacked`,
   forwarding `phi_rl`. Used by bounded/GPU objectives in fit_mle.

**Source to copy-extend:** lines 1001–1199 (block_likelihood) and 1396–1600 (multi/stacked).

**Inline tests to add at bottom** (matching M3 test pattern at lines 1715–1930):
- `test_wmrl_m5_single_block()` — smoke test, NLL is finite
- `test_wmrl_m5_backward_compatibility()` — phi_rl=0 produces identical NLL to M3
- `test_padding_equivalence_wmrl_m5()` — padded vs unpadded gives same NLL

### 2. `mle_utils.py`

**What to add:**

a) `WMRL_M5_BOUNDS` dict — copy of `WMRL_M3_BOUNDS` plus:
```python
'phi_rl': (0.001, 0.999),   # same bounds as phi (per user decision)
```

b) `WMRL_M5_PARAMS` list — `WMRL_M3_PARAMS` + `'phi_rl'` inserted before `'epsilon'`:
```python
WMRL_M5_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa', 'phi_rl', 'epsilon']
```
Order must match signature: `(alpha_pos, alpha_neg, phi, rho, capacity, kappa, phi_rl, epsilon)`.

c) `jax_unconstrained_to_params_wmrl_m5(x)` — 8-element version of
   `jax_unconstrained_to_params_wmrl_m3`, adds `phi_rl = jax_unbounded_to_bounded(x[6], *bounds['phi_rl'])`,
   pushes epsilon to `x[7]`.

d) `jax_bounded_to_unconstrained_wmrl_m5(x)` — 8-element inverse.

e) `elif model == 'wmrl_m5':` branches in all dispatch functions:
   - `params_to_unconstrained()`
   - `unconstrained_to_params()`
   - `get_default_params()` — phi_rl default = 0.0 (matches M3 kappa convention: default to simpler model)
   - `sample_random_start()`
   - `sample_lhs_starts()`
   - `get_n_params()` — returns 8
   - `check_at_bounds()`
   - `summarize_all_parameters()`

**Default start value for phi_rl:** Use 0.0 per M3 kappa analogue (default to simpler
model behavior). This matches the user decision "match phi starting value" — phi itself
defaults to 0.1, but phi_rl at 0.0 follows the kappa convention of defaulting to the
nested model. Clarify with user which convention to use (see Open Questions).

### 3. `fit_mle.py`

**What to add:**

a) Add to imports: `wmrl_m5_multiblock_likelihood`, `wmrl_m5_multiblock_likelihood_stacked`
   from jax_likelihoods; `WMRL_M5_BOUNDS`, `WMRL_M5_PARAMS`, and transform functions
   from mle_utils.

b) `_make_bounded_objective_wmrl_m5(...)` — 8-param version of `_make_bounded_objective_wmrl_m3`.
   Params are: `alpha_pos, alpha_neg, phi, rho, capacity, kappa, phi_rl, epsilon` (indices 0–7).

c) `_make_jax_objective_wmrl_m5(...)` — unconstrained version.

d) `_gpu_objective_wmrl_m5(x, stimuli, actions, rewards, masks, set_sizes)` — 8-element x.

e) `warmup_jax_compilation()`: add `elif model == 'wmrl_m5':` block calling
   `wmrl_m5_multiblock_likelihood(...)` with `phi_rl=0.1` (or any valid value).

f) `fit_participant_mle()`: add `elif model == 'wmrl_m5':` block (parallel to wmrl_m3 block,
   lines 1086–1098). This creates bounded and jax objectives, sets `n_params=8`,
   `param_names=WMRL_M5_PARAMS`, `bounds_dict=WMRL_M5_BOUNDS`.

g) `fit_all_gpu()`: add `elif model == 'wmrl_m5':` in the objective selection block
   (line ~690) and the `to_unc` transform selection block (line ~672).

h) `prepare_participant_data()`: add `'wmrl_m5'` to the model list checks where
   `'wmrl_m3'` already appears (line 1381, 1418, 1430). Pattern: `model in ('wmrl', 'wmrl_m3', 'wmrl_m5')`.

i) `main()`: add `'wmrl_m5'` to `choices` in `--model` argparse argument (line 1946).

j) Output file naming: `wmrl_m5_individual_fits.csv` and `wmrl_m5_group_summary.csv` will be
   produced automatically since the filename is derived from `args.model`.

### 4. `model_recovery.py`

**What to add:**

a) `get_param_names()`: add `elif model == 'wmrl_m5': return WMRL_M5_PARAMS`.
   Must import `WMRL_M5_BOUNDS`, `WMRL_M5_PARAMS`.

b) `sample_parameters()`: add `elif model == 'wmrl_m5': bounds = WMRL_M5_BOUNDS; param_names = WMRL_M5_PARAMS`.

c) `generate_synthetic_participant()`: add `elif model == 'wmrl_m5':` handling.
   **Key:** extract `phi_rl = params['phi_rl']`, apply Q decay inside the block loop
   BEFORE computing action probabilities. The decay:
   ```python
   Q = (1 - phi_rl) * Q + phi_rl * (1.0 / NUM_ACTIONS)
   ```
   is applied every trial to the entire Q matrix (all stimuli, all actions).
   Position: immediately after `WM = (1 - phi) * WM + phi * wm_baseline` for wmrl_m3.

### 5. `scripts/15_analyze_mle_by_trauma.py`

**What to add:**

a) Add constant: `WMRL_M5_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa', 'phi_rl', 'epsilon']`

b) `load_data()`: load wmrl_m5 CSV alongside existing models:
   ```python
   wmrl_m5 = pd.read_csv(OUTPUT_DIR / "wmrl_m5_individual_fits.csv")
   ```
   and merge with surveys/groups the same way as wmrl_m3. Return as 5th element or
   integrate into existing return tuple.

c) `MODEL_CONFIG` dict: add entry:
   ```python
   'wmrl_m5': {'name': 'WM-RL M5', 'params': WMRL_M5_PARAMS, 'data': wmrl_m5},
   ```

d) Update argparse `choices`: add `'wmrl_m5'` and update `'all'` expansion to include it.

e) Add `'phi_rl'` to `PARAM_NAMES` display dict: `'phi_rl': r'$\phi_{RL}$'`

**Important:** `load_data()` currently hard-codes 3 models. If wmrl_m5 file doesn't exist,
it will raise an error. Add a try/except or `exists()` check when loading wmrl_m5,
consistent with how script 16 already uses `if not params_path.exists(): continue`.

### 6. `scripts/16_regress_parameters_on_scales.py`

**What to add:**

a) In `load_integrated_data()`, the `elif model_type != 'qlearning'` branch already
   handles `kappa` by key-existence check. Add `phi_rl` to the same block:
   ```python
   if 'phi_rl' in params_df.columns:
       param_rename['phi_rl'] = 'phi_rl_mean'
   ```

b) In `main()`, in the param_cols section for wmrl_m3 (lines ~769–771):
   ```python
   elif model == 'wmrl_m5':
       param_cols = ['alpha_pos_mean', 'alpha_neg_mean', 'phi_mean', 'rho_mean',
                     'wm_capacity_mean', 'kappa_mean', 'phi_rl_mean', 'epsilon_mean']
   ```

c) Update argparse `choices`: add `'wmrl_m5'` to `['qlearning', 'wmrl', 'wmrl_m3', 'all']`
   and update `'all'` expansion.

### 7. `scripts/14_compare_models.py`

The model comparison script (script 14) does NOT register specific model names. It operates on
a `fits_dict` with arbitrary keys (M1, M2, M3, etc.). M5 can be included by:

Option A (recommended): add `--m5` argument and auto-detection:
```python
parser.add_argument('--m5', type=str, default=None, help='Path to M5 fits')
# In auto-detection: look for wmrl_m5_individual_fits.csv
```

Option B: user passes wmrl_m5 fits via `--m3` as an override (no code change, but confusing).

Option A keeps the table clean. The `find_mle_files()` function (already present) handles
auto-detection if it's updated to look for `wmrl_m5_individual_fits.csv`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Bounded logit transform for phi_rl | Custom sigmoid/logit | `jax_bounded_to_unbounded` / `jax_unbounded_to_bounded` already in mle_utils | Already handles (0.001, 0.999) bounds; JAX-compatible |
| Global Q matrix decay | Custom per-stimulus loop | `Q_decayed = (1 - phi_rl) * Q_table + phi_rl * Q0` — single JAX broadcast | Same idiom as existing WM decay; JIT-friendly |
| Backward compat check | if/else branch in scan | Algebraic identity at phi_rl=0 | Branch introduces `jnp.where` overhead; identity is free |

**Key insight:** Every "new" operation in M5 already exists identically in M3 for the WM
component. WM decay: `(1-phi)*WM + phi*WM_0`. Q decay: `(1-phi_rl)*Q + phi_rl*Q0`.
Copy the WM decay line, substitute Q table and Q0=1/3 baseline.

---

## Common Pitfalls

### Pitfall 1: Wrong Q0 baseline

**What goes wrong:** Using `q_init=0.5` instead of `Q0=1/3` for the decay target.
**Why it happens:** q_init=0.5 is the initialization value; Q0=1/3 is the semantic baseline
(uniform probability, matching wm_init=1/nA=0.333).
**How to avoid:** Hardcode `Q0 = 1.0 / num_actions` inside the block function, matching
`wm_init = 1.0 / 3.0` which is explicitly set in M3's signature.
**Warning signs:** phi_rl=0 backward compat test passes but NLL at phi_rl=1.0 looks
like Q-learning with different initialization.

### Pitfall 2: Decaying the wrong Q variable

**What goes wrong:** Applying decay to `Q_table` but then using `Q_table` (not `Q_decayed`)
for policy and Q-update — decay has no effect.
**Why it happens:** M3 scan body uses `WM_decayed` consistently after the WM decay step.
If you add `Q_decayed = ...` but forget to replace `Q_table` with `Q_decayed` in
`q_vals = Q_table[stimulus]` and `q_current = Q_table[stimulus, action]`, the parameter
is silently disconnected from the computation graph.
**How to avoid:** Search the scan body for every `Q_table[` occurrence and verify each
one should be `Q_table` (carry update only) or `Q_decayed` (computation within trial).
**Warning signs:** phi_rl=0 passes but phi_rl=0.9 produces same NLL as phi_rl=0 (decay
not connected).

### Pitfall 3: Returning Q_table instead of Q_updated in carry

**What goes wrong:** The carry update returns `Q_table` instead of `Q_updated`, so
Q-values never accumulate across trials.
**Why it happens:** Typo when renaming variables from M3 copy-paste.
**How to avoid:** The carry return must be `(Q_updated, WM_updated, WM_baseline, log_lik_new, new_last_action)`,
same as M3 but `Q_updated` is now derived from `Q_decayed`.
**Warning signs:** All NLL values look too high (like chance); recovery fails badly.

### Pitfall 4: Parameter order mismatch

**What goes wrong:** `WMRL_M5_PARAMS` order doesn't match function signature order.
**Why it happens:** Inserting phi_rl in a different position in the params list vs function signature.
M3 comment explicitly warns about this: `# CRITICAL: Order must match wmrl_m3_multiblock_likelihood() signature`.
**How to avoid:** Define function signature and WMRL_M5_PARAMS list in lockstep.
Recommended order: `alpha_pos, alpha_neg, phi, rho, capacity, kappa, phi_rl, epsilon`
(phi_rl inserted before epsilon, just as kappa was inserted before epsilon in M3).
**Warning signs:** Parameters appear to cross-contaminate (e.g., epsilon value assigned to phi_rl).

### Pitfall 5: M3 backward compat test uses wrong comparison

**What goes wrong:** The backward compat test (M5-06) compares M5 at phi_rl=0 against M3,
but uses different data or seeds, producing false failures.
**Why it happens:** Constructing test data independently instead of reusing same arrays.
**How to avoid:** Use literally identical inputs to both likelihood calls, same as the
existing `test_wmrl_m3_backward_compatibility` at line 1750.

### Pitfall 6: Script 15 crashes when wmrl_m5 file is absent

**What goes wrong:** `load_data()` in script 15 hard-codes CSV reads for all 3 models.
When called with `--model wmrl_m5`, it still tries to load wmrl_m5 AND all others.
If any are missing, it crashes before analysis.
**How to avoid:** Load wmrl_m5 CSV only when requested, using `try/except` or
`if path.exists():`. Match the pattern already used in script 16 (auto-skip missing models).

### Pitfall 7: Recovery simulation applies phi_rl decay AFTER Q-update

**What goes wrong:** In `generate_synthetic_participant()`, the Q decay is applied after
the delta-rule update instead of before. This produces inconsistent behavior vs the
likelihood function.
**Why it happens:** Forgetting that M5-02 mandates decay BEFORE delta-rule update.
**How to avoid:** In the simulation loop, apply Q decay immediately after WM decay and
BEFORE computing policy. The comment in model_recovery.py at line ~244 shows WM decay
applied first — Q decay should follow immediately.

---

## Code Examples

### Example 1: Decay step inside scan body (from M3 WM analogue)

```python
# Source: scripts/fitting/jax_likelihoods.py, line 1119
# WM decay pattern (to be mirrored for Q decay in M5):
WM_decayed = (1 - phi) * WM_table + phi * WM_baseline

# M5 Q decay — insert immediately after the WM decay line:
Q0 = 1.0 / num_actions   # = 0.333... (compile-time constant, not array)
Q_decayed = (1 - phi_rl) * Q_table + phi_rl * Q0
```

### Example 2: WMRL_M5_BOUNDS pattern (from mle_utils.py lines 47–56)

```python
# Source: scripts/fitting/mle_utils.py, WMRL_M3_BOUNDS pattern
WMRL_M5_BOUNDS = {
    'alpha_pos': (0.001, 0.999),
    'alpha_neg': (0.001, 0.999),
    'phi':       (0.001, 0.999),
    'rho':       (0.001, 0.999),
    'capacity':  (1.0, 7.0),
    'kappa':     (0.0, 1.0),       # 0.0 allowed (same as M3)
    'phi_rl':    (0.001, 0.999),   # Match phi bounds exactly (user decision)
    'epsilon':   (0.001, 0.999),
}
```

### Example 3: jax_unconstrained_to_params_wmrl_m5 (from mle_utils.py lines 131–145)

```python
# Source: extended from jax_unconstrained_to_params_wmrl_m3
def jax_unconstrained_to_params_wmrl_m5(x: jnp.ndarray) -> tuple:
    bounds = WMRL_M5_BOUNDS
    alpha_pos = jax_unbounded_to_bounded(x[0], *bounds['alpha_pos'])
    alpha_neg = jax_unbounded_to_bounded(x[1], *bounds['alpha_neg'])
    phi       = jax_unbounded_to_bounded(x[2], *bounds['phi'])
    rho       = jax_unbounded_to_bounded(x[3], *bounds['rho'])
    capacity  = jax_unbounded_to_bounded(x[4], *bounds['capacity'])
    kappa     = jax_unbounded_to_bounded(x[5], *bounds['kappa'])
    phi_rl    = jax_unbounded_to_bounded(x[6], *bounds['phi_rl'])
    epsilon   = jax_unbounded_to_bounded(x[7], *bounds['epsilon'])
    return alpha_pos, alpha_neg, phi, rho, capacity, kappa, phi_rl, epsilon
```

### Example 4: bounded_objective pattern (from fit_mle.py lines 436–482)

```python
# Source: extended from _make_bounded_objective_wmrl_m3
@jax.jit
def objective(params: jnp.ndarray) -> float:
    alpha_pos = params[0]
    alpha_neg = params[1]
    phi       = params[2]
    rho       = params[3]
    capacity  = params[4]
    kappa     = params[5]
    phi_rl    = params[6]
    epsilon   = params[7]
    log_lik = wmrl_m5_multiblock_likelihood_stacked(
        ..., kappa=kappa, phi_rl=phi_rl, epsilon=epsilon
    )
    return -log_lik
```

### Example 5: inline backward compat test (from jax_likelihoods.py lines 1750–1800)

```python
def test_wmrl_m5_backward_compatibility():
    """phi_rl=0 must produce NLL identical to M3."""
    # ... build test block data ...
    kappa = 0.3   # use non-zero kappa so we test M3 parity, not M2 parity

    log_lik_m3 = wmrl_m3_block_likelihood(
        stimuli, actions, rewards, set_sizes,
        alpha_pos=0.3, alpha_neg=0.1, phi=0.2, rho=0.7, capacity=4.0,
        kappa=kappa, epsilon=0.05
    )
    log_lik_m5 = wmrl_m5_block_likelihood(
        stimuli, actions, rewards, set_sizes,
        alpha_pos=0.3, alpha_neg=0.1, phi=0.2, rho=0.7, capacity=4.0,
        kappa=kappa, phi_rl=0.0, epsilon=0.05
    )
    assert jnp.allclose(log_lik_m3, log_lik_m5, atol=1e-5), \
        f"M5 phi_rl=0 should match M3: {log_lik_m3} vs {log_lik_m5}"
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Script 15 loads models by hardcoded filenames | Script 15 needs wmrl_m5 added to load_data() | Low — simple file addition |
| Script 14 has --m1/--m2/--m3 | Script 14 needs --m5 (or user passes manually) | Low — user can pass via --m3 as workaround |
| model_recovery has 3-way dispatch | needs 4-way dispatch for wmrl_m5 | Low — single elif block |

No deprecated approaches to avoid. The M3 pattern is current and validated.

---

## Open Questions

1. **phi_rl default start value**
   - What we know: User said "match phi's starting value" (phi default = 0.1) but also
     M3 uses kappa default = 0.0 (default to simpler model)
   - What's unclear: Should phi_rl default to 0.0 (matching kappa convention) or 0.1
     (matching phi convention)?
   - Recommendation: Default to 0.0 to match kappa precedent (simpler model default).
     If the user explicitly said "match phi's starting value" they may mean 0.1. Confirm
     at planning time.

2. **Script 14 --m5 argument**
   - What we know: script 14 uses a fits_dict with arbitrary labels
   - What's unclear: Should we add a formal --m5 argument now, or defer to user
     passing results manually via --m3 override?
   - Recommendation: Add --m5 argument formally. It's trivial (one `add_argument` call)
     and makes the comparison table include M5 correctly labeled.

3. **Script 15 load_data() defensive loading**
   - What we know: load_data() currently crashes if any expected CSV is missing
   - What's unclear: Should all 4 models be loaded defensively, or only when explicitly
     requested?
   - Recommendation: Add `if (OUTPUT_DIR / "wmrl_m5_individual_fits.csv").exists():` guard
     when loading wmrl_m5. Return None for that model if absent, skip in MODEL_CONFIG.

---

## Sources

### Primary (HIGH confidence — direct source inspection)

- `scripts/fitting/jax_likelihoods.py` lines 1001–1199, 1396–1600, 1715–1930
  — M3 block likelihood, multiblock, stacked, inline tests
- `scripts/fitting/mle_utils.py` lines 30–56, 60–145, 193–211, 224–435
  — All bounds dicts, transform functions, get_n_params, check_at_bounds
- `scripts/fitting/fit_mle.py` lines 88–130, 139–192, 296–482, 491–572, 578–700,
  979–1100, 1227–1293, 1339–1433, 1940–2050
  — Imports, warmup, objectives (bounded/jax/gpu), GPU fitting, participant fitting,
  result dict, prepare_participant_data, CLI
- `scripts/fitting/model_recovery.py` lines 1–330
  — get_param_names, sample_parameters, generate_synthetic_participant
- `scripts/15_analyze_mle_by_trauma.py` lines 97–167, 710–800
  — PARAM constants, load_data, MODEL_CONFIG, main dispatch
- `scripts/16_regress_parameters_on_scales.py` lines 118–200, 660–775
  — load_integrated_data param_rename block, main param_cols dispatch
- `scripts/14_compare_models.py` lines 558–626
  — argparse --m1/--m2/--m3 structure, fits_dict pattern
- `scripts/fitting/tests/conftest.py` — fixture patterns for new test fixtures
- `scripts/fitting/tests/test_mle_quick.py` — unit test patterns

---

## Metadata

**Confidence breakdown:**
- Carry structure and decay insertion: HIGH — read exact M3 scan body
- Parameter registration pattern: HIGH — all branches enumerated from source
- Backward compatibility: HIGH — algebraic identity verified from first principles
- Script 15/16 dispatch: HIGH — read exact elif chains
- phi_rl default value: MEDIUM — two plausible conventions; needs user confirmation

**Research date:** 2026-04-02
**Valid until:** 2026-05-02 (stable codebase; only stale if pipeline scripts are refactored)
