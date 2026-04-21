# Stack Research

**Domain:** Computational cognitive modeling — LBA (Linear Ballistic Accumulator) likelihood extension to existing JAX MLE pipeline
**Researched:** 2026-04-02
**Confidence:** HIGH (core JAX stack), MEDIUM (optimizer migration risk), HIGH (no external LBA library needed)

---

## Summary Verdict

**Implement LBA density in pure JAX. No external LBA library is needed or recommended.**

The Brown & Heathcote (2008) analytic LBA likelihood requires only `jax.scipy.stats.norm.pdf` and `jax.scipy.stats.norm.cdf` / `jax.scipy.stats.norm.sf`, all of which are already available in the installed JAX 0.9.0. The only genuine stack question for M4 is numerical stability strategy (addressed below), plus a flag that `jaxopt` is deprecated and the optimizer layer may need attention before or during this milestone.

---

## Recommended Stack

### Core Technologies (No Changes for M4-M6)

| Technology | Current Version | Purpose | Status |
|------------|-----------------|---------|--------|
| JAX | 0.9.0 (latest: 0.9.2) | JIT-compiled likelihood, autodiff for MLE gradients | Keep as-is |
| jaxlib | 0.9.0 | XLA backend for JAX | Keep as-is |
| NumPy | 2.3.5 | Data manipulation outside JIT | Keep as-is |
| SciPy | 1.16.3 | L-BFGS-B optimizer (via jaxopt wrapper), normal CDF reference | Keep as-is |
| pandas | 2.0+ | Trial data loading | Keep as-is |

### Supporting Libraries for LBA (M4)

| Library | Version | Purpose | Action |
|---------|---------|---------|--------|
| `jax.scipy.stats.norm` | bundled with JAX 0.9.0 | `pdf(t)` and `sf(t)` for LBA density terms | Already available — no install needed |
| `jax.lax.scan` | bundled with JAX 0.9.0 | Sequential accumulation over trial sequence | Already used in M1-M3 — no changes needed |
| `jaxopt` | 0.8.5 (deprecated) | `ScipyBoundedMinimize` wrapper for L-BFGS-B with bounds | See migration note below |

### No New Libraries Required for M5 or M6

- **M5 (RL Forgetting):** Adds a scalar decay parameter `lambda_forget` with existing logit-transform pattern. Zero new dependencies.
- **M6 (Stimulus-Specific Perseveration):** Changes kernel indexing logic. Zero new dependencies.

---

## LBA Implementation: Pure JAX Approach

### Why pure JAX (not an external library)

The two Python LBA libraries found in the ecosystem are both inappropriate:

| Library | Problem | Verdict |
|---------|---------|---------|
| `psireact` 0.2.15 (mortonne/psireact) | Last release November 2020, targets Python 3.8, NumPy-based (not JAX), not JIT-compilable, not GPU-compatible | Do not use |
| `rtdists` (R package) | R only — no Python version | Not applicable |

The LBA analytic density (Brown & Heathcote 2008) is a closed-form expression involving only normal PDF and CDF. It is short to implement, well-understood, and maps directly to JAX primitives already in the project. The McDougle & Collins (2021) implementation used MATLAB `fmincon` — no reference Python code exists. Implement from scratch in `scripts/fitting/jax_likelihoods.py`.

### The LBA Density in JAX

The standard Brown & Heathcote (2008) single-accumulator density for accumulator `i` finishing at time `t` (corrected for non-decision time `t0`) is:

```
t' = t - t0                          # decision time
f_i(t') = (1/A) * [-v_i * Phi(z1) + s_v * phi(z1) + v_i * Phi(z2) - s_v * phi(z2)]
S_i(t') = 1 - (1/A) * [(b - A - v_i*t') * Phi(z1)/t' + ... ]   # survivor
```

where `z1 = (b - A - v_i*t') / (s_v*t')`, `z2 = (b - v_i*t') / (s_v*t')`, `A` = start-point range, `b` = threshold, `v_i` = drift rate, `s_v` = drift SD (fixed), `t0` = non-decision time.

The joint log-likelihood for choice `c` at RT `t` is:

```python
log_lik = log(f_c(t')) + sum_{j != c} log(S_j(t'))
```

JAX primitives needed:
- `jax.scipy.stats.norm.pdf(z, loc=0, scale=1)` — for `phi(z)` terms
- `jax.scipy.stats.norm.sf(z)` — for `Phi(-z)` / survivor terms — **use `cdf(-z)` form** (see stability note)

### Numerical Stability Strategy

**Critical:** LBA density is numerically fragile at edge cases. All four strategies below are required.

**1. Use `sf(z) = cdf(-z)` not `1 - cdf(z)`**

JAX's `norm.sf` was previously inaccurate (JAX issue #17199, fixed in a patch released before JAX 0.4). The current implementation already uses `cdf(-x, -loc, scale)` internally. However, explicitly calling `jax.scipy.stats.norm.cdf(-z)` rather than `norm.sf(z)` is defensive and self-documenting.

Confidence: HIGH — verified from JAX source at `jax/_src/scipy/stats/norm.py`.

**2. Enable float64 globally**

L-BFGS-B (FORTRAN-backed) requires float64. The existing `ScipyBoundedMinimize` pipeline already casts to float64 for this reason. Extend the same pattern to M4 likelihood. Add at module init:

```python
jax.config.update("jax_enable_x64", True)
```

This is already done in `jax_likelihoods.py` for the existing models — verify it covers the LBA path.

Confidence: HIGH — verified from JAXopt docs and JAX x64 documentation.

**3. Safe operations in `jnp.where` branches**

JAX evaluates both branches of `jnp.where` before selecting, so `NaN` in the unselected branch propagates through gradients. For the LBA density at `t <= t0` (negative decision time), use the safe-mask pattern:

```python
# Compute density for t' > 0 only; mask out t' <= 0
t_prime = jnp.where(rt > t0, rt - t0, 1.0)  # safe dummy (1.0) prevents NaN in computation
density = _lba_single_density(t_prime, v, A, b, s_v)
log_density = jnp.where(rt > t0, jnp.log(jnp.maximum(density, 1e-10)), -30.0)
```

This pattern is documented in the JAX FAQ and is the standard approach for conditionally undefined computations.

Confidence: HIGH — pattern documented in JAX GitHub discussions #6778, #5039.

**4. Clamp drift rates to prevent division-by-zero**

When `v_i` is near zero, `z1` and `z2` blow up. Clamp drift rates before entering the density:

```python
v_safe = jnp.maximum(v_i, 1e-6)  # or use softplus: jax.nn.softplus(v_raw) + 1e-6
```

Use `jax.nn.softplus` for the parameter transformation if drift rates are unconstrained during optimization, since softplus is smooth and differentiable everywhere. This is the same category of fix used in `mle_utils.py` for logit-bounded parameters.

Confidence: MEDIUM — standard practice in RT modeling, no specific JAX source, but follows directly from the density formula.

---

## Optimizer Layer: jaxopt Deprecation

### Status

`jaxopt` (last release 0.8.5, April 2025) is officially deprecated. The project's `pyproject.toml` and `environment_gpu.yml` both reference `jaxopt>=0.8`. The library still works — the final 0.8.5 release is a maintenance release — but it will not receive updates.

### Recommendation for This Milestone

**Do not migrate the optimizer during M4-M6.** `jaxopt.ScipyBoundedMinimize` works correctly and is the most direct path to maintaining `scipy.optimize.minimize` L-BFGS-B with box constraints, which is what this project needs. The deprecation does not mean the library breaks — it means no new features.

**Flag for the next major milestone.** The natural successor for bounded SciPy-backed optimization is:
- A thin manual wrapper around `scipy.optimize.minimize` with JAX value-and-gradient (straightforward ~20 lines of code, documented in community gists)
- Or `optimistix` (the jaxopt-recommended successor) — but note its bounded minimize support is currently limited, and migrating would require validation work

Confidence: HIGH — verified from jaxopt PyPI page and optax GitHub issue #977.

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| LBA density | Pure JAX implementation | `psireact` library | Last released 2020, Python 3.8 only, NumPy-based, not JIT-compatible |
| LBA density | Pure JAX implementation | `rtdists` (R) | R-only, no Python port |
| LBA density | Pure JAX implementation | DDM (drift-diffusion) via `PyDDM` | DDM is for 2-choice paradigms only; this task has 3 responses (J/K/L) |
| RT accumulator model | LBA | Racing diffusion model (RDM) | Higher complexity, no proven advantage for this paradigm; LBA is the McDougle & Collins (2021) precedent |
| Optimizer | `jaxopt.ScipyBoundedMinimize` (keep) | `optimistix` migration | Optimistix bounded minimize is limited in current version; migration risk not worth it for M4-M6 |
| Optimizer | `jaxopt.ScipyBoundedMinimize` (keep) | `jax.scipy.optimize.minimize` | No box-constraint support in JAX's native wrapper |
| Precision | float64 (keep) | float32 | L-BFGS-B FORTRAN code requires float64; CDF tails need precision for extreme RT values |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `PyDDM` | Drift-diffusion model for 2-choice only; 3-choice LBA requires racing accumulators | Pure JAX LBA density |
| `psireact` | Unmaintained (2020), NumPy-only, incompatible with JAX JIT | Pure JAX LBA density |
| `jax.scipy.stats.norm.sf(z)` via naive `1 - cdf(z)` | Loses precision in extreme tails (though fixed in JAX's own `sf` implementation, being explicit is safer) | Use `cdf(-z)` or verify `sf` implementation before each JAX upgrade |
| `jnp.where(cond, risky_expr, safe_val)` without safe dummy in risky branch | NaN propagation through gradients in unselected branch | Double `jnp.where` pattern (see JAX FAQ) |
| `float32` for LBA likelihood | LBA CDF evaluated near zero loses precision; optimizer crashes | Enable `jax_enable_x64 = True` globally |

---

## Installation

No new packages to install. All required primitives are in the existing environment.

```bash
# Verify existing JAX has required scipy stats
python -c "from jax.scipy.stats import norm; print('norm.pdf:', norm.pdf); print('norm.cdf:', norm.cdf); print('norm.sf:', norm.sf)"

# Verify float64 is available
python -c "import jax; jax.config.update('jax_enable_x64', True); import jax.numpy as jnp; print(jnp.array(1.0, dtype=jnp.float64).dtype)"
```

---

## Version Compatibility

| Package | Current | Compatible With | Notes |
|---------|---------|-----------------|-------|
| JAX 0.9.0 | installed | NumPy 2.3.5, SciPy 1.16.3 | JAX 0.9 requires NumPy >= 2.0 and SciPy >= 1.13 — both satisfied |
| jaxopt 0.8.5 | pinned in pyproject.toml | JAX 0.9.0 | Deprecated but functional; `ScipyBoundedMinimize` still works |
| Python 3.11 | environment_gpu.yml | JAX 0.9.2 officially supports 3.11–3.14 | Compatible |
| JAX 0.9.2 (upgrade path) | not installed | Same as 0.9.0 | Safe to upgrade; no breaking changes to `scipy.stats.norm` or `lax.scan` in 0.9.x |

---

## Sources

- `jax/_src/scipy/stats/norm.py` at jax-ml/jax — verified `sf` uses `cdf(-x)` pattern (HIGH confidence)
- `jaxopt` PyPI page — confirmed deprecated, latest 0.8.5, April 2025 (HIGH confidence)
- JAX GitHub issue #17199 — `norm.sf` precision bug, confirmed fixed (HIGH confidence)
- JAX GitHub discussions #6778, #5039 — `jnp.where` NaN gradient pattern (HIGH confidence)
- JAXopt `ScipyBoundedMinimize` docs — confirmed float64 requirement for L-BFGS-B (HIGH confidence)
- McDougle & Collins (2021) PMC article — confirmed LBA parameters (A, b, v_i, s_v, t0), MATLAB fmincon used, no Python reference code (MEDIUM confidence for parameter conventions)
- `psireact` GitHub (mortonne/psireact) — confirmed LBA implementation but unmaintained, Python 3.8 (HIGH confidence for avoid recommendation)
- WebSearch: jaxopt migration landscape — confirmed Optimistix as successor but bounded support limited (MEDIUM confidence — ecosystem still in flux)
- JAX PyPI page — confirmed JAX 0.9.2 latest stable as of March 18, 2026 (HIGH confidence)

---

*Stack research for: RLWM-LBA model extension (M4), RL Forgetting (M5), Stimulus-Specific Perseveration (M6)*
*Researched: 2026-04-02*
