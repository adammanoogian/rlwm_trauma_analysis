# Phase 11: M4 LBA Joint Choice+RT - Research

**Researched:** 2026-04-02
**Domain:** Linear Ballistic Accumulator, JAX numerical computing, joint choice+RT likelihood
**Confidence:** HIGH (codebase read directly; LBA formula verified from MATLAB source and McDougle & Collins 2021)

---

## Summary

M4 replaces the softmax choice mechanism in M3 with a Linear Ballistic Accumulator (LBA; Brown & Heathcote, 2008) that produces a joint choice+RT density. The learning dynamics (Q-values, WM matrix, omega weighting, perseveration) are **reused unchanged from M3**. The only change is the decision stage: instead of softmax producing a choice probability, the hybrid policy weights feed into LBA drift rates that produce both a choice and an RT.

The key reference implementation is McDougle & Collins (2021), which integrates WM-RL hybrid weights into LBA drift rates for this exact task design. Their formulas (Equations 9, 10, 14) are verified and match the classical Brown & Heathcote (2008) analytic density.

**Primary recommendation:** New file `scripts/fitting/lba_likelihood.py` containing the LBA density/survivor functions and the `wmrl_m4_block_likelihood()` function. Import into `jax_likelihoods.py` or call directly from `fit_mle.py`. The LBA density must use float64 throughout; `jax.config.update("jax_enable_x64", True)` is required at module load.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| JAX | 0.9.0 (installed) | LBA density, gradients, JIT | Already the fitting stack |
| jax.scipy.stats.norm | same | Phi, phi, logsf, logpdf | Built-in; has `logsf` for log-space survivor |
| jaxopt.ScipyBoundedMinimize | same | L-BFGS-B with b>A constraint | Same optimizer used for all models |
| numpy/pandas | same | RT preprocessing, data pipeline | Same as rest of codebase |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| jax.lax.scan | same | Sequential trial loop | Same carry pattern as M3 |
| jax.config | same | Enable float64 | Required once at module import |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom LBA in new file | Embed in jax_likelihoods.py | New file keeps separation clean; jax_likelihoods.py is already large (>2000 lines) |
| log-space CDF via logsf | Direct 1-F_i(t) | logsf is numerically superior for small survivor values (tail precision) |
| float32 | float64 | float32 causes cancellation errors in LBA CDF formula; float64 mandatory |

---

## Architecture Patterns

### Recommended Project Structure

New file for M4:
```
scripts/fitting/
├── jax_likelihoods.py          # Existing (M1-M3, M5, M6a, M6b) — NOT modified for LBA
├── lba_likelihood.py           # NEW: LBA density + wmrl_m4_block/multiblock likelihood
├── mle_utils.py                # Add WMRL_M4_BOUNDS, WMRL_M4_PARAMS, get_n_params update
├── fit_mle.py                  # Add wmrl_m4 dispatch branch
└── model_recovery.py           # Add wmrl_m4 simulation branch (generate RT + choice)
```

New utility:
```
scripts/utils/rt_preprocessing.py  # NEW: outlier removal, t0 validation
```

Or embed RT preprocessing in `fit_mle.py`'s `prepare_participant_data()` as a new code path for M4.

### Pattern 1: LBA Carry Structure in lax.scan

The M4 `lax.scan` carry extends M3's carry by adding the RT array per block and removing epsilon:

```python
# M3 carry: (Q, WM, WM_baseline, log_lik_accum, last_action)
# M4 carry: (Q, WM, WM_baseline, log_lik_accum, last_action)
# Inputs extended: (stimuli, actions, rewards, set_sizes, rts, mask)
```

The learning updates inside the scan body are **identical to M3**. Only the probability-to-log-likelihood step changes:
- M3: `log_prob = log(noisy_probs[action] + 1e-8)`
- M4: `log_lik = log(lba_pdf(t_star, b, A, v[action], s)) + sum(log(lba_sf(t_star, b, A, v[j], s)) for j != action)`

### Pattern 2: LBA Density Functions

Verified from MATLAB source (LBA_tpdf.m, smfleming/LBA GitHub) and McDougle & Collins (2021) Eq. 10:

```python
# Source: LBA_tpdf.m (Fleming 2012), Brown & Heathcote (2008)
# Requires float64 and jax.config.update("jax_enable_x64", True) at module load

import jax.scipy.stats as jss

def lba_pdf(t, b, A, v_i, s):
    """Single-accumulator LBA density (defective PDF).

    Parameters
    ----------
    t : float  — time since stimulus onset minus t0 (seconds)
    b : float  — decision threshold (same units as A)
    A : float  — max starting point; b > A enforced
    v_i : float — drift rate for this accumulator
    s : float  — within-trial noise (fixed at 0.1)
    """
    g = (b - A - t * v_i) / (t * s)
    h = (b - t * v_i) / (t * s)
    return (-v_i * jss.norm.cdf(g) + s * jss.norm.pdf(g)
            + v_i * jss.norm.cdf(h) - s * jss.norm.pdf(h)) / A

def lba_cdf(t, b, A, v_i, s):
    """Single-accumulator LBA CDF. Brown & Heathcote (2008) Eq. 3."""
    g = (b - A - t * v_i) / (t * s)
    h = (b - t * v_i) / (t * s)
    return (1.0
            + (b - A - t * v_i) / A * jss.norm.cdf(g)
            - (b - t * v_i) / A * jss.norm.cdf(h)
            + (t * s) / A * (jss.norm.pdf(g) - jss.norm.pdf(h)))

def lba_sf(t, b, A, v_i, s):
    """Survivor function S_i(t) = 1 - F_i(t)."""
    return 1.0 - lba_cdf(t, b, A, v_i, s)
```

### Pattern 3: Joint Choice+RT Log-Likelihood

Per-trial log-likelihood from McDougle & Collins (2021) Eq. 9:

```python
# For chosen action i at adjusted time t_star = RT - t0:
# log P(choice=i, RT) = log f_i(t*) + sum_{j != i} log S_j(t*)

def lba_joint_log_lik(t_star, chosen, b, A, v_all, s, n_actions=3):
    """
    v_all : shape (n_actions,) — drift rates for each accumulator
    Returns scalar log-likelihood for one trial
    """
    # Density of chosen accumulator
    f_chosen = lba_pdf(t_star, b, A, v_all[chosen], s)
    log_lik = jnp.log(jnp.maximum(f_chosen, 1e-300))

    # Survivor functions for non-chosen accumulators
    for j in range(n_actions):
        if j != chosen:
            sf_j = lba_sf(t_star, b, A, v_all[j], s)
            log_lik += jnp.log(jnp.maximum(sf_j, 1e-300))

    return log_lik
```

For JAX/scan compatibility, replace the Python loop with `jax.vmap` or manual unroll:

```python
# JAX-friendly version (no Python loop in JIT):
sf_all = jax.vmap(lambda vi: lba_sf(t_star, b, A, vi, s))(v_all)
f_chosen = lba_pdf(t_star, b, A, v_all[chosen], s)
log_lik = (jnp.log(jnp.maximum(f_chosen, 1e-300))
           + jnp.sum(jnp.log(jnp.maximum(sf_all, 1e-300)))
           - jnp.log(jnp.maximum(sf_all[chosen], 1e-300)))
# (subtracts the log-SF of chosen since chosen should contribute pdf, not sf)
```

### Pattern 4: Drift Rates from Hybrid Policy

Per M4-03 (McDougle & Collins 2021, Eq. 14 simplified variant):

```python
# After computing hybrid policy probs pi_t (before epsilon is applied in M3)
# For M4, epsilon is DROPPED (requirement M4-05)
# Drift rates:
v_all = v_scale * pi_hybrid  # shape: (n_actions,)
```

Where `pi_hybrid` is the M3 hybrid policy (omega*wm_probs + (1-omega)*rl_probs, with perseveration if applicable), without epsilon noise. v_scale is a free parameter (positive, log-transformed).

**Critical:** Since epsilon is dropped in M4, the `noisy_probs` step from M3 is skipped entirely. The hybrid policy probabilities feed directly into drift rates.

### Pattern 5: b > A Constraint

Reparameterize as `b = A + softplus(delta_raw)` where `delta_raw` is unconstrained:

```python
# In the bounded objective: A and b are fitted directly with L-BFGS-B bounds
# The b > A constraint is enforced by setting:
#   lower bound on b = lower bound on A (e.g., 0.001)
#   Constraint added via reparameterization in the objective:
#     b_actual = A + jax.nn.softplus(b_raw)
# OR simply bound A in [A_min, A_max] and b in [A_min + delta_min, b_max]
# with the objective computing delta = b - A and asserting delta > 0

# Recommended: use b_raw in objective, decoded as:
#   b = A + softplus(b_raw)
# This ensures b > A always, with b_raw unconstrained
```

Alternatively: two bounded params `A` and `delta` (both > 0), then `b = A + delta` in the objective. This matches existing stick-breaking patterns in M6b.

### Pattern 6: RT Preprocessing

RT data is in milliseconds (confirmed from CSV). Filter before fitting:

```python
def preprocess_rts(rt_array_ms, min_rt_ms=150.0, max_rt_ms=2000.0, t0_ms=150.0):
    """
    Returns filtered mask and validates t0 < min(RT_filtered).
    min_rt_ms=150: removes anticipatory responses (matches McDougle 2021)
    max_rt_ms=2000: removes outliers (2 trials in full dataset = 0.006%)
    """
    valid = (rt_array_ms >= min_rt_ms) & (rt_array_ms <= max_rt_ms)
    if t0_ms >= rt_array_ms[valid].min():
        raise ValueError(f"t0={t0_ms}ms >= min(RT)={rt_array_ms[valid].min():.1f}ms")
    return valid
```

**RT scale for LBA:** Convert ms to seconds before passing to LBA (`rt_sec = rt_ms / 1000`). This matches McDougle & Collins (2021) scale (t0=0.15s, A/b in the range 0-2, drift rates ~1-5 per second).

### Pattern 7: M4 Parameter Space

```python
WMRL_M4_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa',
                   'v_scale', 'A', 'delta', 't0']
# Where b = A + delta (reparameterization for b > A)
# OR: use b directly with b > A enforced in objective decode

WMRL_M4_BOUNDS = {
    'alpha_pos':  (0.001, 0.999),
    'alpha_neg':  (0.001, 0.999),
    'phi':        (0.001, 0.999),
    'rho':        (0.001, 0.999),
    'capacity':   (1.0, 7.0),
    'kappa':      (0.0, 1.0),
    # LBA params (all in seconds scale):
    'v_scale':    (0.1, 20.0),     # log-transform in unconstrained space
    'A':          (0.001, 2.0),    # max starting point (seconds)
    'delta':      (0.001, 2.0),    # b - A > 0 (softplus or bounded)
    't0':         (0.05, 0.4),     # non-decision time (seconds); t0 < min(RT)
}
# n_params = 10 for AIC/BIC
```

**Note:** McDougle fixed t0=150ms and fit A/b freely. The requirement says to fit t0 with validation t0 < min(RT). Recommended: fit t0 with log-transform in (0.05, 0.4) seconds, validate at data-prep time.

### Anti-Patterns to Avoid

- **Float32 for LBA:** The CDF formula `(b - A - t*v)/A` involves cancellation. Float32 gives catastrophic precision loss. Float64 is mandatory.
- **Negative density in log:** `lba_pdf` can return small negative values due to floating-point. Use `jnp.maximum(f, 1e-300)` before log. Verified: at extreme parameter values (negative drift, very short t), pdf approaches 0 but can go slightly negative.
- **Missing jax_enable_x64:** Without this, JAX silently runs float64 ops in float32 on some backends. Call `jax.config.update("jax_enable_x64", True)` at module load of `lba_likelihood.py`.
- **Python loops in scan body:** `lba_joint_log_lik` must not have Python `for` loops; use `jax.vmap` or manual unroll for 3-accumulator fixed case.
- **Mixing RT scales:** RT in ms must be converted to seconds before calling `lba_pdf`/`lba_sf`. Drift rates have units threshold/second, and A/b/t0 are also in seconds.
- **Comparing M4 AIC to M1-M3:** Joint log-likelihood includes RT density which has different units from choice log-prob. Per STATE.md decision: M4 gets a separate comparison track in compare_mle_models.py. Do NOT include M4 in the main AIC table.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Normal CDF/PDF | Custom implementation | `jax.scipy.stats.norm.cdf/pdf` | Already differentiable, numerically stable, in JAX |
| Log survivor function | `log(1-cdf)` | `jax.scipy.stats.norm.logsf` (confirmed available in JAX 0.9.0) | Better tail precision (avoids catastrophic cancellation for small survivor) |
| b > A enforcement | Clipping in objective | Reparameterize: `b = A + softplus(delta_raw)` | Clipping breaks gradient; reparameterization is clean |
| RT distribution simulation | Custom random walk | Inverse CDF sampling: `t_sim = (b - U*A) / v_i` where `U ~ Uniform(0,1)` | LBA simulation is analytically simple due to no within-trial noise |

**Key insight:** JAX's `jax.scipy.stats.norm` has `logsf` (log of survival function). Use `logsf` for log-space survivor computation; do NOT compute `log(1 - cdf)` which is numerically unstable in the tails.

---

## Common Pitfalls

### Pitfall 1: LBA Density Can Return Negative Values

**What goes wrong:** For certain parameter combinations (especially negative drift rates), `lba_pdf` returns a slightly negative value due to floating-point cancellation in `(-v*Phi(g) + s*phi(g) + v*Phi(h) - s*phi(h)) / A`.

**Why it happens:** When drift rate is very negative or t is very small, `Phi(g)` and `Phi(h)` are both near 1, causing catastrophic cancellation.

**How to avoid:** Clamp before log: `log_f = jnp.log(jnp.maximum(f_i, 1e-300))`. The density should logically be >= 0; the negativity is a numerical artifact. This is safe and widely used in LBA implementations.

**Warning signs:** NaN gradients during optimization; converging to boundary.

### Pitfall 2: t0 Constraint Not Enforced at Data-Prep Time

**What goes wrong:** The optimizer can propose t0 values larger than the minimum observed RT, causing `t_star = RT - t0 < 0` which makes `g` and `h` undefined.

**How to avoid:** In `preprocess_rts()`, validate `t0_upper_bound < min(RT_filtered)`. Set the t0 upper bound in `WMRL_M4_BOUNDS` to be at most 0.15s initially (conservative). During fitting, `t_star = max(RT - t0, 1e-6)` as a safety floor — though this should never trigger if bounds are set correctly.

**Warning signs:** Participant has very fast RTs after filtering; NaN likelihood on first optimization step.

### Pitfall 3: Float32 Mode Breaks CDF Precision

**What goes wrong:** JAX defaults to float32. LBA CDF formula involves computing differences of near-equal quantities. In float32, the result has ~7 digits of precision; in float64, ~15 digits.

**How to avoid:** `jax.config.update("jax_enable_x64", True)` at the TOP of `lba_likelihood.py`, before any JAX imports are used. Verify all arrays passed to LBA functions are `jnp.float64`.

**Warning signs:** LBA likelihoods that are numerically identical across very different parameters (underflow masking).

### Pitfall 4: M4 AIC Compared to Choice-Only Models

**What goes wrong:** M4's log-likelihood includes RT density (which sums over continuous time), while M1-M3 likelihoods are pure choice probabilities. Their numerical ranges are not comparable; AIC comparison is invalid.

**How to avoid:** M4 gets its own comparison section in `compare_mle_models.py`. Add a `--track m4` or `--m4` argument that computes M4's own AIC/BIC in isolation. Do NOT merge M4 into the existing `--m1/--m2/--m3` comparison table.

### Pitfall 5: Model Recovery Needs RT Simulation

**What goes wrong:** `generate_synthetic_participant()` only outputs `key_press` and `reward`, not `rt`. M4 parameter recovery requires synthetic RT data.

**How to avoid:** Add RT simulation in `model_recovery.py` for `model == 'wmrl_m4'`:

```python
# LBA simulation: RT for winning accumulator i
# Start points: k_i ~ Uniform(0, A)
# Time to threshold: t_i = (b - k_i) / v_i  (if v_i > 0)
# Winner = argmin t_i; RT = t_winner + t0
k = rng.uniform(0, A, size=n_actions)
t_race = (b - k) / np.maximum(v_all, 1e-6)  # avoid div-by-zero for negative drift
winner = np.argmin(t_race)
rt_sim = t_race[winner] + t0
```

If drift is negative for all accumulators (very rare for reasonable parameters), the race is numerically unstable — re-run or assign a large RT.

### Pitfall 6: Epsilon Removal vs M3 Backward Compatibility

**What goes wrong:** M4 drops epsilon (M4-05) but the scan body is copied from M3 which includes epsilon. Forgetting to remove it gives a model with both LBA noise AND epsilon — double-counting randomness.

**How to avoid:** In M4 scan body, the hybrid policy `pi_hybrid` feeds DIRECTLY into `v_all = v_scale * pi_hybrid`. Do NOT call `apply_epsilon_noise()`. Start-point variability A already models undirected exploration (per McDougle & Collins 2021).

---

## Code Examples

Verified patterns ready for implementation:

### LBA Module Header (lba_likelihood.py)

```python
"""JAX LBA likelihood for M4. Requires float64."""
import jax
jax.config.update("jax_enable_x64", True)  # MUST be before other JAX imports

import jax.numpy as jnp
import jax.scipy.stats as jss
from jax import lax

FIXED_S = 0.1    # Within-trial noise (fixed; McDougle & Collins 2021)
NUM_ACTIONS = 3
```

### RT Preprocessing

```python
def preprocess_rt_block(rt_ms, min_rt_ms=150.0, max_rt_ms=2000.0):
    """
    Returns valid mask (bool array) for RT outlier removal.
    Converts to seconds for LBA.
    """
    valid = (rt_ms >= min_rt_ms) & (rt_ms <= max_rt_ms)
    rt_sec = rt_ms / 1000.0
    return rt_sec, valid

def validate_t0_constraint(rt_sec_all_blocks, t0_sec):
    """Raises if t0 >= min(RT) after filtering."""
    min_rt = min(rt.min() for rt in rt_sec_all_blocks)
    if t0_sec >= min_rt:
        raise ValueError(f"t0={t0_sec:.3f}s >= min(RT)={min_rt:.3f}s")
```

### Checking JAX logsf

```python
# Confirmed available (JAX 0.9.0):
# jax.scipy.stats.norm.logsf(x)  — returns log(1 - Phi(x))
# Use this instead of jnp.log(1 - jss.norm.cdf(x)) for tail precision
import jax.scipy.stats as jss
log_sf = jss.norm.logsf(x)  # numerically stable log-survivor
```

### Prepare Participant Data for M4

In `prepare_participant_data()`, add M4 branch:

```python
if model == 'wmrl_m4':
    rt_raw = jnp.array(block_data['rt'].values, dtype=jnp.float64)
    rt_sec, valid_rt = preprocess_rt_block(rt_raw)
    # Combine with block padding mask: both must be 1 for trial to contribute
    combined_mask = mask * valid_rt.astype(jnp.float32)
    rts_blocks.append(rt_sec_padded)  # padded to max_trials
    masks_blocks.append(combined_mask)  # RT-filtered mask
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Softmax choice only | LBA joint density | M4 (this phase) | Accounts for RT; requires float64 |
| Epsilon noise for exploration | LBA start-point variability A | M4 per M4-05 | Removes epsilon; replaces with A parameter |
| AIC comparison across all models | Separate M4 comparison track | STATE.md decision | M4 AIC not directly comparable to M1-M3 |

---

## Open Questions

1. **v_scale parameterization: direct linear or information-theoretic?**
   - What we know: M4-03 says `v_i = v_scale * pi_t(a_i | s_t)` (linear scaling). McDougle & Collins used an entropy-normalized version (Eq. 14: `v_i = eta * pi_i / H_prior`).
   - What's unclear: Whether the information-theoretic formulation (dividing by entropy) is necessary, or whether simple linear scaling (`v_scale * pi_i`) is sufficient and simpler.
   - Recommendation: Use linear scaling (`v_scale * pi_i`) per M4-03 requirement. The entropy normalization can be tried if parameter recovery is poor.

2. **t0 as free parameter vs fixed at 150ms**
   - What we know: McDougle & Collins fixed t0=150ms. M4-01 requires fitting t0 with validation. With the 150ms RT filter, min(RT_filtered) ≈ 150-310ms across participants, leaving very little room for t0.
   - What's unclear: Whether fitting t0 in (0.05, 0.15)s is identifiable given data quality.
   - Recommendation: Fit t0 with bounds (0.05, 0.3) seconds, log-transformed. Flag for review in planning — may need to fix t0=0.1 if recovery is poor (r < 0.80).

3. **Reparameterization: b = A + softplus(delta) vs b = A + delta with b bounded**
   - What we know: Both enforce b > A. Stick-breaking is already used in M6b.
   - What's unclear: Whether L-BFGS-B with two bounded params [A, delta] is equivalent in practice to a single unconstrained b_raw via softplus decode.
   - Recommendation: Use bounded [A, delta] with delta in (0.001, 2.0). Simpler to implement, matches existing bounded-objective pattern in `_make_bounded_objective_*`.

4. **Where does prepare_participant_data extract RT?**
   - Confirmed: `task_trials_long.csv` has `rt` column (float64, milliseconds, range 0.1-2009.5ms).
   - Action needed: `prepare_participant_data()` must be extended to extract and preprocess `rt` for `model == 'wmrl_m4'`. This is currently only extracting `stimulus`, `key_press`, `reward`, `set_size`.

---

## Sources

### Primary (HIGH confidence)

- Codebase direct read: `scripts/fitting/jax_likelihoods.py`, `scripts/fitting/mle_utils.py`, `scripts/fitting/fit_mle.py`, `scripts/fitting/model_recovery.py`, `scripts/14_compare_models.py`
- McDougle & Collins (2021): PMC7854965 — LBA formula Eqs. 9, 10, 14; parameter bounds; s=0.1 fixed; t0=150ms fixed; b>A constraint; RT exclusion criteria
- Fleming (2012) MATLAB source: `github.com/smfleming/LBA/blob/master/LBA_tpdf.m` — exact formula: `(-v*normcdf(g) + sv*normpdf(g) + v*normcdf(h) - sv*normpdf(h)) / A` where `g=(b-A-t*v)/(t*sv)`, `h=(b-t*v)/(t*sv)`
- JAX 0.9.0 (installed): confirmed `jax.scipy.stats.norm` has `cdf`, `pdf`, `logsf`, `logpdf`, `sf`; float64 via `jax_enable_x64`

### Secondary (MEDIUM confidence)

- Brown & Heathcote (2008) — original paper; formula verified via MATLAB implementation above
- Task RT data analysis (direct Python run): rt column confirmed in ms, range 0.1-2009.5ms; 3.0% trials < 150ms; 0.006% > 2000ms; 39/48 participants have min RT < 150ms before filtering

### Tertiary (LOW confidence)

- WebSearch: ecosystem context for LBA; not load-bearing for implementation

---

## Metadata

**Confidence breakdown:**
- LBA formula: HIGH — verified from MATLAB source + McDougle & Collins 2021 PMC paper
- Codebase integration: HIGH — code read directly; patterns traced through full stack
- Parameter bounds: MEDIUM — McDougle bounds verified; t0 as free param is new (McDougle fixed it)
- RT data: HIGH — direct CSV inspection
- b>A reparameterization choice: MEDIUM — multiple valid options; recommendation based on existing patterns

**Research date:** 2026-04-02
**Valid until:** 2026-05-02 (stable domain; LBA formulas are 17 years old and unchanged)
