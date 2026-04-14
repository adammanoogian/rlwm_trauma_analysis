# DEER Non-Linear Parallelization: Research and Recommendation

**Phase:** 20 (DEER Non-Linear Parallelization)
**Status:** Research complete -- NO-GO on DEER, GO on vectorized policy
**Cross-reference:** [docs/PARALLEL_SCAN_LIKELIHOOD.md](PARALLEL_SCAN_LIKELIHOOD.md) for Phase 19 linear scan implementation

---

## Overview

Phase 19 implemented parallel scan (associative scan) for the **linear**
components of the RLWM likelihood: Q-value updates and WM decay/overwrite.
These are AR(1) recurrences solvable in O(log T) depth via
`jax.lax.associative_scan`. Phase 19 left a sequential `lax.scan` for the
**non-linear** Phase 2 components: WM-Q mixing, softmax, epsilon noise,
perseveration kernels, and log-probability computation.

Phase 20 investigated whether DEER (Deep Equilibrium Recurrences) could
parallelize this remaining sequential pass. The conclusion is that DEER is
unnecessary -- the Phase 2 sequential dependency is a phantom caused by the
`lax.scan` implementation, not an intrinsic property of the likelihood.

---

## Table of Contents

1. [DEER Algorithm](#deer-algorithm)
2. [Worked Example: RLWM Perseveration](#worked-example-rlwm-perseveration)
3. [Convergence Analysis](#convergence-analysis)
4. [Alternative Approaches Comparison](#alternative-approaches-comparison)
5. [Unifying Framework Perspective](#unifying-framework-perspective)
6. [Go/No-Go Recommendation](#gono-go-recommendation)
7. [Precomputation Strategy](#precomputation-strategy)
8. [References](#references)

---

## DEER Algorithm

### Background

DEER (Lim et al., ICLR 2024) parallelizes non-linear recurrences of the form:

```
h_t = f(h_{t-1}, x_t)    for t = 1, ..., T
```

where `f` is a non-linear transition function, `h_t` is the hidden state, and
`x_t` is the input at step t. Sequential evaluation requires O(T) depth.

### Fixed-Point Iteration via Newton's Method

DEER reformulates the recurrence as a root-finding problem. Define the
residual over the full trajectory:

```
r(h_{1:T}) = [ h_1 - f(h_0, x_1),
               h_2 - f(h_1, x_2),
               ...
               h_T - f(h_{T-1}, x_T) ]
```

The true state trajectory h* satisfies r(h*) = 0. Newton's method iterates:

```
h^{(i+1)} = h^{(i)} - J(h^{(i)})^{-1} r(h^{(i)})
```

where J is the block-bidiagonal Jacobian:

```
J = [ I_D          0           0       ...    0          ]
    [ -df/dh_0     I_D         0       ...    0          ]
    [ 0            -df/dh_1    I_D     ...    0          ]
    [ ...          ...         ...     ...    ...        ]
    [ 0            0           0   -df/dh_{T-1}   I_D   ]
```

### The Key Insight: Linear Recurrence in the Update

The Newton update `Dh = J^{-1} r` satisfies a linear recurrence:

```
Dh_t = A_t * Dh_{t-1} + b_t
```

where `A_t = df/dh_{t-1}` (Jacobian of f) and `b_t = -r_t` (residual). Being
a linear recurrence, this can be solved via `jax.lax.associative_scan` in
O(log T) depth -- the same technique Phase 19 uses for Q-value updates.

### Computational Complexity

| Variant | Work per Iteration | Memory per Iteration | Convergence |
|---|---|---|---|
| Full DEER | O(T * D^3) | O(T * D^2) | Quadratic (Newton) |
| Quasi-DEER | O(T * D) | O(T * D) | Linear (guaranteed in <= T iterations) |
| ELK (damped) | O(T * D^3) | O(T * D^2) | Damped Newton (stable) |
| Quasi-ELK (damped) | O(T * D) | O(T * D) | Damped linear (stable) |

For K Newton iterations with state dimension D and sequence length T, the
total work is O(K * T * D^3) for full DEER. Practical convergence typically
requires 2-5 iterations for smooth dynamics with stable eigenvalues.

---

## Worked Example: RLWM Perseveration

### The Apparent Sequential Dependency

In the Phase 2 `lax.scan` of M3/M5 pscan variants, the carry tracks
`last_action` (scalar int) for the global perseveration kernel:

```python
def policy_step(carry, t_inputs):
    log_lik_accum, last_action = carry
    ...
    # Perseveration depends on last_action
    choice_kernel = jnp.eye(num_actions)[last_action]
    noisy_probs = (1 - kappa) * noisy_base + kappa * choice_kernel
    ...
    # Update carry with OBSERVED action
    new_last_action = jnp.where(valid, action, last_action)
    return (log_lik_accum + log_prob, new_last_action), log_prob
```

### How DEER Would Apply (Hypothetically)

If this were a true non-linear recurrence, the DEER formulation would be:

**State:** `h_t = last_action_t` (integer in {0, 1, 2})
**Transition:** `h_t = f(h_{t-1}, x_t)` where `x_t = (stimulus_t, action_t, ...)`
**Residual:** `r_t = h_t - f(h_{t-1}, x_t)`

The Jacobian `df/dh_{t-1}` would need to be computed. However:

1. **h_t is discrete** (integer action index), not continuous
2. **f involves argmax** (implicitly: "which action was taken"), which is
   non-differentiable
3. **The Jacobian is undefined** at the decision boundary

This means DEER's Newton iteration cannot be applied -- the linearization
step fails at the fundamental level.

### Why It Does Not Matter: The Dependency is a Phantom

The critical line in the `lax.scan` is:

```python
new_last_action = jnp.where(valid, action, last_action)
```

Here, `action` is the **observed action from the data**, not a sample from the
model's policy. In likelihood evaluation, the observed data sequence
`(stimuli, actions, rewards)` is fixed and known. Therefore:

- `last_action[0] = -1` (sentinel: no previous action)
- `last_action[t] = actions[t-1]` for valid trials
- For masked/padded trials: last valid action propagates forward

The entire `last_action` trajectory is determined by the observed data alone.
There is no parameter dependence and no recurrence to solve. The `lax.scan`
carry is merely a convenient implementation pattern, not a mathematical
necessity.

---

## Convergence Analysis

### Contraction Mapping Question

The Phase 20 success criteria ask: "Is the RLWM softmax-mixing a contraction
mapping?" This question has two answers depending on context.

### Simulation Context (Not Our Use Case)

In simulation (generating synthetic data), the action at trial t is *sampled*
from the model's policy, creating a true recurrence:

```
h_t = argmax(softmax(beta * policy(Q_t, WM_t, h_{t-1})))
```

This is **NOT a contraction mapping** because:

1. **Discrete state space:** `h_t` takes values in {0, 1, 2} -- there is no
   meaningful notion of "distance shrinking" in a discrete set
2. **Non-differentiable dynamics:** `argmax` is not smooth; small changes in
   policy logits can flip the selected action entirely
3. **Non-contractiveness:** A perturbation to `h_{t-1}` that shifts the
   perseveration kernel can cause a different action to be selected,
   amplifying the perturbation rather than shrinking it
4. **No Lipschitz constant < 1:** The Lipschitz constant of the mapping is
   not well-defined for discrete-to-discrete functions through a
   non-differentiable intermediary

DEER's convergence theory (Proposition 1 of Lim et al.) requires the
dynamics to have bounded, finite Jacobians everywhere. The RLWM perseveration
recurrence violates this assumption.

### Likelihood Evaluation Context (Our Actual Use Case)

In likelihood evaluation, the question is **moot**. There is no recurrence:

- Actions are observed data, not model outputs
- `last_action[t]` is determined entirely by the observed action sequence
- Each trial's log-probability is an independent function of known inputs
- The "dynamics" have Lipschitz constant 0 (no actual sequential dependency)

The contraction mapping analysis is therefore irrelevant to our use case.
The Phase 2 computation is embarrassingly parallel once the `last_action`
array is precomputed from observed data.

---

## Alternative Approaches Comparison

### Comparison Table

| Approach | Applicable to RLWM? | Depth Complexity | Implementation Effort | Numerical Stability | Speedup Potential |
|---|---|---|---|---|---|
| **Vectorized policy** | Yes | O(1) | Low (array ops) | Exact (no iteration) | Eliminates Phase 2 scan |
| DEER (Newton) | No | O(K * log T) | High | Requires smooth f | Negative (overhead) |
| Picard iteration | No | O(K * T) | Medium | Slow convergence | Negative (same issues) |
| Newton-Raphson (direct) | No | O(K * T * D^3) | High | Requires Jacobian | Negative (D=1 overhead) |
| Direct linearization | No | N/A | N/A | N/A (no recurrence) | N/A |

### Why Vectorized Policy Wins

The vectorized policy approach eliminates the Phase 2 `lax.scan` entirely by
precomputing the `last_action` array from observed data and then computing all
T trial log-probabilities simultaneously via array broadcasting. This approach:

- **Has O(1) depth:** All T trials computed in parallel
- **Is numerically exact:** No iteration, no approximation, no convergence
  concerns
- **Is trivial to implement:** Standard JAX array operations
- **Works on both CPU and GPU:** No hardware-specific assumptions
- **Requires no new dependencies:** Uses existing JAX primitives

### Why DEER/Picard/Newton-Raphson Do Not Apply

All iterative fixed-point methods assume a true sequential dependency that
must be resolved through iteration. In the RLWM likelihood:

1. The Q/WM updates (Phase 1) are already handled by parallel scan
2. The policy computation (Phase 2) has no sequential dependency once
   `last_action` is precomputed
3. There is nothing left to iterate on

Even hypothetically, the RLWM perseveration state (discrete integer) violates
the smoothness assumptions required by Newton-based methods. And for the tiny
state dimension (D=1 for M3/M5, D=6 for M6a) and short sequences (T=100),
the per-iteration overhead of DEER would exceed the cost of sequential
evaluation.

---

## Unifying Framework Perspective

### The Taxonomy

The paper "A Unifying Framework for Parallelizing Sequential Models with
Linear Dynamical Systems" (arXiv 2509.21716, Linderman Lab, 2025) provides a
unified taxonomy showing that all fixed-point parallel methods (Newton, Picard,
Jacobi iteration) are special cases of solving a linearized dynamical system
via parallel scan.

The framework categorizes sequential models into layers:

1. **Linear recurrence layers:** State evolves as `h_t = A_t h_{t-1} + b_t`.
   Solvable via parallel scan in O(log T) depth. This is the simplest case.

2. **Non-linear recurrence layers:** State evolves as `h_t = f(h_{t-1}, x_t)`
   with non-linear f. Requires fixed-point iteration (DEER, Picard, etc.) or
   model-specific reformulation.

3. **Observation layers:** Output is a pointwise function of state and input,
   `y_t = g(h_t, x_t)`, with no sequential dependency on previous outputs.
   Trivially parallel.

### Where RLWM Fits

The RLWM likelihood decomposes cleanly into two layers:

**Phase 1 -- Linear recurrence layer (parallel scan):**
- Q-value update: `Q_t(s,a) = (1-alpha) * Q_{t-1}(s,a) + alpha * r_t`
- WM update: decay `(1-phi) * WM + phi * wm_init` then overwrite
- Both are AR(1) recurrences, handled by `affine_scan` in O(log T) depth
- Phase 19 implementation: `associative_scan_q_update`, `associative_scan_wm_update`

**Phase 2 -- Observation layer (zero sequential dependency):**
- Policy computation: softmax, WM-Q mixing, epsilon noise, perseveration
- Given `(Q_for_policy[t], wm_for_policy[t], last_action[t], stimulus[t], action[t])`,
  the log-probability at trial t is a pointwise function
- `last_action[t]` is precomputable from observed data (not a recurrence)
- Falls into the "observation layer" category -- trivially parallel

In the Unifying Framework's terminology, the RLWM Phase 2 has **zero
contractivity** (Lipschitz constant = 0) because there is no actual state
propagation. The framework's convergence analysis for fixed-point iteration
is vacuous here -- convergence is trivial because there is nothing to
converge.

### Implication for Model Architecture

This analysis confirms that the RLWM likelihood is a well-structured model
for parallel evaluation:

- Phase 1 is "embarrassingly scannable" (linear AR(1) recurrence)
- Phase 2 is "embarrassingly parallel" (pointwise observation function)
- No component requires fixed-point iteration

---

## Go/No-Go Recommendation

### DEER: NO-GO

**Recommendation:** Do not implement DEER, quasi-DEER, ELK, or any
fixed-point iteration method for the RLWM likelihood.

**Supporting evidence:**

1. **Code inspection (HIGH confidence):** Direct inspection of all 12 pscan
   likelihood variants in `scripts/fitting/jax_likelihoods.py` (lines
   3466-4427) confirms that `last_action` / `last_actions` carries are
   updated with OBSERVED actions from the data array, not model-computed
   actions. The sequential dependency is an implementation artifact, not a
   mathematical property.

2. **Discrete state (HIGH confidence):** The perseveration carry is a
   discrete integer (action index 0, 1, or 2), not a continuous state.
   DEER's Newton iteration linearizes around continuous trajectories --
   linearizing a discrete variable is mathematically ill-defined.

3. **Complexity overhead (HIGH confidence):** For D=1 (scalar last_action in
   M3/M5) and T=100, DEER's overhead per iteration exceeds sequential cost.
   Even one Newton iteration of O(T * D^3) = O(100) with the associative scan
   constant factor (~10x for log_2(100) = 6.6 steps plus overhead) makes
   parallel evaluation slower than the O(100) sequential scan.

4. **CPU is the target platform (HIGH confidence):** Phase 19 benchmark data
   (`output/bayesian/pscan_benchmark.json`) shows that the pscan approach
   achieves only 0.26x speedup on CPU (i.e., 3.8x slower than sequential:
   624ms vs 164ms). The RLWM likelihood has arithmetic intensity of ~0.3
   FLOP/byte, well below the threshold where GPU parallelism provides
   benefit. Adding DEER iteration overhead would further degrade performance.

### Vectorized Policy: GO

**Recommendation:** Replace the Phase 2 sequential `lax.scan` with vectorized
array operations, using precomputed `last_action` arrays.

**Justification:**

- Precomputation of `last_action` arrays is parameter-independent (run once
  before MCMC)
- Vectorized policy computation is O(1) depth, numerically exact, and
  trivially implementable
- Eliminates the last sequential bottleneck in the likelihood, enabling
  future GPU acceleration if arithmetic intensity improves
- Simplifies the code (array broadcasts vs. `lax.scan` closures)

### Phase 19 Benchmark Context

The Phase 19 CPU benchmark established baseline performance:

- Sequential (lax.scan): 164ms per likelihood evaluation (17 blocks x 100 trials)
- Parallel scan (Phase 1 pscan + Phase 2 sequential): 624ms (0.26x speedup)
- Compilation: 646ms (pscan) vs 257ms (sequential)

The 0.26x speedup (i.e., slowdown) demonstrates that for T=100 on CPU, the
overhead of `jax.lax.associative_scan` exceeds the benefit of O(log T) depth.
Vectorizing Phase 2 will not change this fundamental dynamic on CPU -- the
primary benefit is enabling GPU parallelism and code simplification for when
the full likelihood is compiled as a single vectorized function.

---

## Precomputation Strategy

### Global Perseveration (M3, M5)

For models with global perseveration (`kappa` parameter), `last_action[t]`
is the most recent valid action before trial t, regardless of stimulus:

```python
def precompute_last_action_global(actions, mask):
    # result[0] = -1 (no previous action)
    # result[t] = last valid action before trial t
    # Handles masked/padded trials correctly
    ...
```

For sequences with no masked trials, this reduces to a simple shift:
`last_action = concat([-1], actions[:-1])`. With masked trials, a scan
propagates the last valid action through padding.

### Per-Stimulus Perseveration (M6a, M6b)

For models with stimulus-specific perseveration (`kappa_s` parameter),
`last_action_per_stimulus[t]` is the last action taken for `stimulus[t]`,
considering only trials 0..t-1:

```python
def precompute_last_actions_per_stimulus(stimuli, actions, mask, num_stimuli):
    # Uses lax.scan to track per-stimulus last actions
    # Parameter-independent: depends only on observed data
    ...
```

This requires a sequential scan over the data (O(T) per block), but is run
only once before MCMC begins -- not at every likelihood evaluation.

### M6b: Dual Perseveration

M6b uses both global and per-stimulus perseveration. Both precomputation
functions are called, and both arrays are passed to the vectorized likelihood.

---

## References

### Primary Sources

1. **DEER:** Lim, S., Linsley, D., & Serre, T. (2024). Parallelizing
   non-linear sequential models over the sequence length. *ICLR 2024*.
   [arXiv:2309.12252](https://arxiv.org/abs/2309.12252)

2. **Quasi-DEER / ELK:** Gonzalez, A., et al. (2024). Towards Scalable and
   Stable Parallelization of Nonlinear RNNs. *NeurIPS 2024*.
   [arXiv:2407.19115](https://arxiv.org/abs/2407.19115)

3. **Unifying Framework:** Linderman Lab (2025). A Unifying Framework for
   Parallelizing Sequential Models with Linear Dynamical Systems.
   [arXiv:2509.21716](https://arxiv.org/abs/2509.21716)

### Implementation References

4. DEER reference: [github.com/machine-discovery/deer](https://github.com/machine-discovery/deer)
5. ELK reference: [github.com/lindermanlab/elk](https://github.com/lindermanlab/elk)
6. Unifying Framework code: [github.com/lindermanlab/parallelizing_with_lds](https://github.com/lindermanlab/parallelizing_with_lds)

### Project References

7. Phase 19 parallel scan implementation: [docs/PARALLEL_SCAN_LIKELIHOOD.md](PARALLEL_SCAN_LIKELIHOOD.md)
8. Phase 19 benchmark: `output/bayesian/pscan_benchmark.json`
   (CPU, JAX 0.4.31, 0.26x speedup = 3.8x slowdown)
9. Pscan likelihood variants: `scripts/fitting/jax_likelihoods.py` (lines 3303-4530)
