# Architecture Research

**Domain:** Computational psychiatry — adding RL model variants to an existing fitting pipeline
**Researched:** 2026-04-02
**Confidence:** HIGH (based on direct codebase inspection)

## Standard Architecture

### System Overview

The pipeline is layered: numbered scripts at the top consume library modules below. Models
live entirely in the library layer.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Numbered Pipeline Scripts (CLI layer)               │
│  12_fit_mle.py  13_fit_bayesian.py  14_compare_models.py  15_...  16_.. │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ imports
┌───────────────────────────────▼─────────────────────────────────────────┐
│                     scripts/fitting/ (library layer)                    │
│                                                                         │
│  jax_likelihoods.py        mle_utils.py        fit_mle.py               │
│  (core math: per-model     (BOUNDS dicts,      (objectives, prepare_    │
│  block + multiblock        param name lists,   participant_data,        │
│  likelihood functions)     transforms,         fit_participant_mle,     │
│                            LHS sampling)       fit_all_participants_    │
│                                                gpu)                     │
│                                                                         │
│  fit_bayesian.py           compare_mle_models.py   model_recovery.py   │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ reads
┌───────────────────────────────▼─────────────────────────────────────────┐
│                          Data layer                                     │
│  output/task_trials_long.csv          output/mle/<model>_individual_   │
│  output/task_trials_long_all.csv      fits.csv                          │
│  (stimulus, action, reward, rt,       (fitted params + AIC/BIC per      │
│   set_size already present)           participant)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Key constraint |
|-----------|----------------|----------------|
| `jax_likelihoods.py` | Block-level and multiblock likelihood math | Pure JAX, JIT-compiled; `lax.scan` for trials, `lax.fori_loop` for blocks |
| `mle_utils.py` | Parameter bounds dicts, param name lists, logit/inv-logit transforms, LHS sampler | Order of entries in `*_PARAMS` lists must match likelihood function signature |
| `fit_mle.py` | Objective closures, `prepare_participant_data`, GPU vmap fitting, CLI entry-point | Dispatches on `--model` string; new models require new objective functions and warmup branch |
| `compare_mle_models.py` | AIC/BIC aggregation, Akaike weights, delta IC | Currently designed for choice-only likelihoods; mixed comparison (choice+RT vs choice-only) requires explicit separation |
| `model_recovery.py` | Generates synthetic data, runs MLE on it, computes recovery metrics | Calls `fit_participant_mle` directly; must register new model names in `get_param_names` |
| `scripts/15, 16` | Trauma correlations and regressions on fitted parameters | Hard-code `*_PARAMS` lists locally — must be extended for M4, M5, M6 |

## Integration Points per New Model

### M5 (RL forgetting) and M6a/M6b (stimulus-specific perseveration)

These are variations of `wmrl_m3_block_likelihood`. The integration is additive at every layer.

**jax_likelihoods.py — new functions:**
- `wmrl_m5_block_likelihood(...)` — identical signature to `wmrl_m3_block_likelihood` plus
  `gamma_rl` (forgetting rate for Q-values). Add forgetting step: after Q-update, decay
  non-chosen Q-values: `Q(s, a') ← (1 - gamma_rl) * Q(s, a')` for all `a' != a`.
- `wmrl_m5_multiblock_likelihood(...)` — wraps M5 block fn, same structure as existing
  multiblock wrappers.
- `wmrl_m5_multiblock_likelihood_stacked(...)` — fast stacked version for GPU vmap.
- `wmrl_m6a_block_likelihood(...)` — M3 + stimulus-specific perseveration: the choice
  kernel `Ck` is per-stimulus (shape `(num_stimuli, num_actions)`) not global. Last action
  tracked per stimulus in carry.
- `wmrl_m6b_block_likelihood(...)` — combines M5 forgetting + M6a stimulus-specific
  perseveration.
- Multiblock and stacked variants for each.

**Code sharing decision:** Keep M5 and M6 as separate functions rather than a single
parameterized supermodel. Reason: JAX's `lax.scan` carry structure changes per model
(M6 needs a `(num_stimuli,)` last-action array vs M3's scalar). Trying to unify into one
function would require padding the carry to the superset and conditionally zeroing parts
out via `jnp.where` — adding complexity with no performance benefit. Separate functions
are cleaner, independently testable, and follow the existing M2/M3 precedent.

**mle_utils.py — additions:**
```python
WMRL_M5_BOUNDS = {
    **WMRL_M3_BOUNDS,
    'gamma_rl': (0.0, 1.0),  # RL forgetting rate
}
WMRL_M5_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa', 'gamma_rl', 'epsilon']

WMRL_M6A_BOUNDS = {
    **WMRL_M3_BOUNDS,  # same params as M3, different carry structure
}
WMRL_M6A_PARAMS = WMRL_M3_PARAMS  # identical param names, but stimulus-scoped internally

WMRL_M6B_BOUNDS = {
    **WMRL_M5_BOUNDS,  # M5 + M6a = all params
}
WMRL_M6B_PARAMS = WMRL_M5_PARAMS
```

Add `jax_unconstrained_to_params_wmrl_m5`, `jax_bounded_to_unconstrained_wmrl_m5`, etc.
for each new model. The pattern is already well-established in `mle_utils.py`.

**fit_mle.py — additions:**
- One new `_make_bounded_objective_wmrl_m5(...)` closure function per model.
- One new `_gpu_objective_wmrl_m5(...)` data-as-args function per model.
- Extend `warmup_jax_compilation` with new `elif model == 'wmrl_m5':` branches.
- Extend `prepare_participant_data` to recognise new model names (they need `set_sizes`
  just like M2/M3, so only the string dispatch needs extending).
- Extend the GPU vmap dispatch block (`_run_all_starts`, `_run_all` construction) with
  `elif` branches for the new model names.
- Extend the `--model` argparse choices.

**compare_mle_models.py — additions:**
- M5, M6a, M6b all produce choice-only likelihoods. They can be directly compared with
  M1–M3 using existing AIC/BIC. The `compare_models` function already accepts an
  arbitrary `fits_dict`; just pass new model fits under new keys (e.g., `'M5'`, `'M6a'`).
- No structural changes needed for the choice-only comparison pathway.

**model_recovery.py — additions:**
- Extend `get_param_names` with new model names.
- Extend `sample_parameters` to recognise new bounds dicts.
- The synthetic data generation already cycles through set sizes and calls
  `fit_participant_mle` — no structural changes required.

**Scripts 15 and 16 — additions:**
- Both scripts hard-code `WMRL_M3_PARAMS` locally. Add:
  ```python
  WMRL_M5_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa', 'gamma_rl', 'epsilon']
  WMRL_M6A_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa', 'epsilon']
  WMRL_M6B_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa', 'gamma_rl', 'epsilon']
  ```
  and extend the `--model` dispatching.

### M4 (LBA joint choice + RT)

M4 is architecturally different from M5/M6. It requires a second density on top of the
choice probability: a reaction time likelihood from the Linear Ballistic Accumulator model.
This introduces four new concerns.

**1. RT data — already present in CSV, no Script 03 changes needed.**

Inspecting `output/task_trials_long.csv` column headers confirms `rt` is already a column
in the output. Script 03 already reads `rt` from `parsed_task_trials.csv` and passes it
through (it creates `rt_category` derived from `rt`). The `prepare_participant_data`
function in `fit_mle.py` does not currently pass `rt` arrays to likelihoods because M1–M3
do not need them. M4 preparation must extract `rt` per block alongside the other arrays.

**2. New carry state in the JAX scan.**

The LBA accumulator parameters (`v_correct`, `v_error`, `A`, `b`, `tau`) operate on each
trial independently given the choice probabilities from the cognitive model. The log-
likelihood at each trial is:
```
log p(a, RT | state) = log p(a | state)_WM-RL + log f_LBA(RT | v_a, v_{-a}, A, b, tau)
```
The LBA density `f_LBA` is available analytically (Navarro & Fuss 2009). JAX implements
this as:
```python
def lba_log_density(rt, v_winner, v_losers, A, b, tau):
    # See Navarro & Fuss (2009), eq. 7
    ...
```
This adds to the scan body without changing the carry structure, because LBA parameters
are constant across trials (no within-trial state update for LBA).

**3. New functions in jax_likelihoods.py:**
- `lba_log_density(rt, v_winner, v_losers, A, b, tau)` — analytic LBA density.
- `wmrl_m4_block_likelihood(stimuli, actions, rewards, rts, set_sizes, ..., v_scale, A, b, tau)` —
  extends the M3 scan body with `rt` in inputs and adds `lba_log_density` call.
- `wmrl_m4_multiblock_likelihood(...)` and `wmrl_m4_multiblock_likelihood_stacked(...)`.

**4. New bounds and params in mle_utils.py:**
```python
WMRL_M4_BOUNDS = {
    **WMRL_M3_BOUNDS,
    'v_scale': (0.1, 5.0),  # scales RL/WM Q-values to drift rates
    'A':       (0.01, 2.0),  # start-point noise
    'b':       (0.1, 3.0),   # response threshold (b > A required)
    'tau':     (0.05, 0.5),  # non-decision time (seconds)
}
WMRL_M4_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa',
                  'v_scale', 'A', 'b', 'tau', 'epsilon']
```

**5. RT in prepare_participant_data:**
Add an `rts_blocks` extraction path, enabled when `model == 'wmrl_m4'`:
```python
if model == 'wmrl_m4' and 'rt' in block_data.columns:
    rts = jnp.array(block_data['rt'].values / 1000.0, dtype=jnp.float32)  # ms → seconds
```
RT values need to be converted from milliseconds (as stored) to seconds (LBA convention).
Timeout trials (`rt` is NaN or `key_press` is NaN) must be excluded or masked.

**6. Model comparison constraint — M4 cannot be directly AIC-compared with M1–M3.**

M4 uses a joint `log p(a, RT)` likelihood. M1–M3 use `log p(a)` only. The two are not
comparable via AIC because:
- M4 NLL will be lower (it has more information to explain RT as well as choice).
- The penalty term `2k` from additional LBA parameters does not account for this.

The correct comparison approach is to evaluate M4 on both the choice marginal
(integrating RT out, or equivalently using only the choice component) and the full joint
density. `compare_mle_models.py` should be extended to flag M4 as a `joint_likelihood`
model and exclude it from direct AIC comparisons with choice-only models. A note column
in the comparison table is the minimum viable implementation.

**7. fit_mle.py GPU objective for M4:**

M4's GPU vmap objective must pass `rts` as an additional batched argument:
```python
def _gpu_objective_wmrl_m4(x, stimuli, actions, rewards, rts, masks, set_sizes):
    ...
```
The `_run_all` vmap call must add `rts` with `in_axes=0` for the participant dimension.

## Recommended File Change Summary

| File | Change type | What changes |
|------|-------------|--------------|
| `scripts/fitting/jax_likelihoods.py` | Addition | New `_block_likelihood`, `_multiblock_likelihood`, `_multiblock_likelihood_stacked` functions for M4, M5, M6a, M6b; `lba_log_density` helper |
| `scripts/fitting/mle_utils.py` | Addition | New `*_BOUNDS`, `*_PARAMS`, `jax_unconstrained_to_params_*`, `jax_bounded_to_unconstrained_*`, `params_to_unconstrained` branches |
| `scripts/fitting/fit_mle.py` | Modification | New objective closures, GPU objectives, `warmup_jax_compilation` branches, `prepare_participant_data` rt extraction, argparse choices |
| `scripts/fitting/compare_mle_models.py` | Modification | Pass new models as additional keys to `fits_dict`; flag M4 as joint-likelihood model |
| `scripts/fitting/model_recovery.py` | Modification | Extend `get_param_names`, `sample_parameters` bounds dispatch |
| `scripts/15_analyze_mle_by_trauma.py` | Modification | Add new `*_PARAMS` lists, extend `--model` dispatch |
| `scripts/16_regress_parameters_on_scales.py` | Modification | Same as 15 |
| `scripts/12_fit_mle.py` | No change | CLI entry-point already delegates to `fit_mle.py`; no change needed |
| `scripts/03_create_task_trials_csv.py` | No change | `rt` column is already present in output |

No new library modules are needed. All new code lives inside existing files.

## Recommended Build Order

**1. M5 (RL forgetting) — build first.**

Rationale: M5 adds one new parameter (`gamma_rl`) to an already-working carry structure
(scalar `last_action` from M3). The forgetting step is a single `Q.at[stimulus].set(...)` 
operation applied after the standard Q-update. This is the smallest incremental change and
will validate the full pipeline integration pattern (likelihoods → bounds → objective →
GPU vmap → compare → recovery → trauma analysis) before tackling more complex models.

**2. M6a (stimulus-specific perseveration) — build second.**

Rationale: M6a changes the carry structure (scalar `last_action` in M3 becomes a
`(num_stimuli,)` array in M6a). This is the novel architectural concern for M6. Isolating
it in M6a before combining with forgetting in M6b makes debugging cleaner.

**3. M6b (M5 + M6a combined) — build third.**

Rationale: M6b is mechanical composition of M5 and M6a. Once both work independently,
M6b is a straightforward merge of their carry structures and step functions.

**4. M4 (LBA joint choice+RT) — build last.**

Rationale: M4 requires the LBA density implementation, RT data extraction, and comparison
methodology changes. It is the highest-risk model and benefits from the pipeline integration
patterns having been validated by M5 and M6.

## Data Flow for M4 (RT path)

```
output/task_trials_long.csv
    ↓ (prepare_participant_data, model='wmrl_m4')
rts_blocks: list of arrays (rt in seconds, NaN-masked)
    ↓ (pad_block_to_max — rt padded with 0.0, mask zeros out contribution)
rts_padded: shape (MAX_TRIALS_PER_BLOCK,)
    ↓ (stacked into batch)
rts_stacked: shape (n_participants, MAX_BLOCKS, MAX_TRIALS_PER_BLOCK)
    ↓ (GPU vmap objective)
wmrl_m4_multiblock_likelihood_stacked(stimuli, actions, rewards, rts, set_sizes, masks, ...)
    ↓ (per trial in lax.scan)
log p(a_t) + log f_LBA(rt_t | ...) * mask_t
    ↓ (summed across blocks and trials)
NLL → AIC/BIC
    ↓ (stored)
output/mle/wmrl_m4_individual_fits.csv
    ↓ (compared separately from M1–M3)
compare_mle_models.py (note: joint-likelihood flag, separate AIC table)
```

## Patterns to Follow

### Pattern 1: Paired block + multiblock + stacked functions

Every model in the codebase follows this three-function pattern:
- `_block_likelihood(stimuli, actions, rewards, ..., mask)` — single block via `lax.scan`
- `_multiblock_likelihood(stimuli_blocks, actions_blocks, rewards_blocks, ...)` — Python
  list loop that dispatches to the block function (includes fallback path)
- `_multiblock_likelihood_stacked(stimuli_stacked, actions_stacked, ...)` — takes
  pre-stacked `(n_blocks, max_trials)` arrays, uses `lax.fori_loop` for GPU

All new models must implement all three. The stacked version is what the GPU vmap path
uses; omitting it causes a fallback to Python loops and loses most of the GPU speedup.

### Pattern 2: Masked padding in lax.scan carry

Every scan body follows this masking protocol for padding trials:
- Likelihood: `log_prob_masked = log_prob * valid`
- State updates: `Q.at[s,a].set(jnp.where(valid, q_updated, q_current))`
- carry state updates (last_action, etc.): `jnp.where(valid, new_value, old_value)`

New models must apply `valid` masking to every carry update, not just the likelihood term.
Forgetting in M5 is an example where it is tempting to skip masking: the non-chosen
Q-value decay should also be gated on `valid` to avoid spurious Q-decay on padding trials.

### Pattern 3: Bounds dict drives everything else

The pattern in `mle_utils.py` is:
1. Define `MODEL_BOUNDS` dict (parameter name → (lower, upper)).
2. Define `MODEL_PARAMS` list (names in the exact order the likelihood function expects them).
3. Implement `jax_unconstrained_to_params_MODEL` and `jax_bounded_to_unconstrained_MODEL`
   using the same bounds dict.

The order in `MODEL_PARAMS` is a load-bearing contract — it must match the order of
positional arguments in the likelihood function. Mismatches are silent bugs.

## Anti-Patterns

### Anti-Pattern 1: Unified supermodel function

**What people do:** Try to write one `wmrl_super_block_likelihood(kappa, gamma_rl,
stimulus_specific, ...)` that handles all model variants via boolean flags.

**Why it's wrong:** JAX's `lax.scan` requires carry shape to be statically known at
compile time. A stimulus-specific perseveration array `(num_stimuli,)` and a scalar
`last_action` have different shapes — you cannot switch between them with a runtime flag.
JAX would force the carry to always be the superset shape, adding silent overhead.
Additionally, `jnp.where(use_forgetting, ...)` inside the scan is fine for scalar params
but not for conditional carry structure changes.

**Do this instead:** Write separate functions per model. The duplication is minor (~40
lines per variant); the clarity and testability gain is significant.

### Anti-Pattern 2: Directly AIC-comparing M4 with M1–M3

**What people do:** Run `compare_mle_models.py` with M4 included alongside M1–M3 and
interpret M4's lower AIC as evidence it is a better cognitive model.

**Why it's wrong:** M4's NLL is computed over `log p(a, RT)`, which is a two-dimensional
density. M1–M3 NLLs are computed over `log p(a)` only. They are not on the same scale.
A lower M4 AIC says only that M4 predicts both choices and RTs better than M1–M3 predict
choices — an unfair comparison.

**Do this instead:** Compare M4 vs M1–M3 only on the choice-marginal NLL (run the WM-RL
component of M4 without the LBA term, store separately). Compare M4 against a
choice-only baseline LBA model to ask whether the RT component adds information.

### Anti-Pattern 3: Skipping the stacked likelihood variant

**What people do:** Implement `_block_likelihood` and `_multiblock_likelihood` for a new
model but skip `_multiblock_likelihood_stacked`.

**Why it's wrong:** The GPU vmap path in `fit_mle.py` requires the stacked variant. The
objective closures (`_make_bounded_objective_*`) pre-stack data and call the stacked
function directly. Without it the GPU path silently falls back to Python loops, inflating
fitting time by 10-50x.

**Do this instead:** Always implement all three functions as a unit. Add them together,
test them together.

### Anti-Pattern 4: Converting RT inside the likelihood function

**What people do:** Store RT in milliseconds in the JAX arrays and convert to seconds
inside `lba_log_density`.

**Why it's wrong:** Division by 1000 inside a JIT-compiled scan body is not wrong per se,
but it adds unnecessary operations on every trial of every block for every participant
across every starting point. More importantly, it makes the expected input units of the
function non-obvious.

**Do this instead:** Convert ms → seconds in `prepare_participant_data` (CPU, once),
before the arrays enter JAX. Document the unit assumption in `wmrl_m4_block_likelihood`'s
docstring.

## Sources

- Direct inspection of `scripts/fitting/jax_likelihoods.py` (M2 and M3 block likelihood
  implementations, scan body patterns, masking protocol)
- Direct inspection of `scripts/fitting/mle_utils.py` (BOUNDS dicts, PARAMS lists,
  transform functions — confirmed structure for all three existing models)
- Direct inspection of `scripts/fitting/fit_mle.py` (GPU vmap dispatch, objective closure
  pattern, `prepare_participant_data`, `warmup_jax_compilation` branches)
- Direct inspection of `scripts/fitting/compare_mle_models.py` (`compare_models` function,
  `fits_dict` API)
- Direct inspection of `scripts/fitting/model_recovery.py` (`get_param_names` dispatch)
- Direct inspection of `scripts/15_analyze_mle_by_trauma.py` and
  `scripts/16_regress_parameters_on_scales.py` (local `*_PARAMS` hard-coding pattern)
- Direct inspection of `output/task_trials_long.csv` column headers (confirmed `rt` is
  already present; Script 03 does not need modification for M4)
- Navarro & Fuss (2009) for LBA density — confidence LOW on exact implementation details,
  needs verification against published formula during M4 phase

---
*Architecture research for: M4-M6 model extension into rlwm_trauma_analysis pipeline*
*Researched: 2026-04-02*
