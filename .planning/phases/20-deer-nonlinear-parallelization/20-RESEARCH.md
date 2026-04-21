# Phase 20: DEER Non-Linear Parallelization - Research

**Researched:** 2026-04-14
**Domain:** Parallel non-linear recurrence evaluation for RLWM likelihood
**Confidence:** HIGH (code analysis: direct inspection of all 12 pscan variants), HIGH (DEER algorithm: multiple papers cross-referenced), HIGH (critical finding: derived from first principles with code verification)

---

## Executive Summary

Phase 20 was designed to investigate DEER-style fixed-point iteration for
parallelizing the "non-linear" components of the RLWM likelihood that Phase 19
left sequential: WM-Q mixing, softmax, epsilon noise, perseveration kernels,
and log-probability computation.

**The central finding of this research is that DEER is not needed.**

Careful analysis of the existing Phase 2 sequential pass reveals that none of
the "non-linear" components actually have sequential state dependencies in the
likelihood evaluation context. The apparent sequential dependency through
`last_action` (perseveration carry) is an artifact of the `lax.scan`
implementation -- in likelihood evaluation, actions are observed data, so
`last_action[t]` is simply `actions[t-1]`, which is precomputable from the
data array with a single shift operation. Once `last_action` is precomputed,
every trial's log-probability is an independent function of `(Q_for_policy[t],
wm_for_policy[t], stimulus[t], action[t], set_size[t], last_action[t])`,
making the entire Phase 2 pass trivially vectorizable with `jax.vmap` or
array broadcasting -- no fixed-point iteration required.

**Recommendation: NO-GO on DEER. Instead, implement a simpler and more
effective "vectorized policy" approach that replaces the sequential `lax.scan`
Phase 2 with pure array operations, achieving full O(1)-depth parallelism
for the non-linear components.**

---

## Key Finding: The Perseveration Carry is NOT a Sequential Dependency

### The Mischaracterization

Phase 19 research (Pitfall 3, lines 338-342 of 19-RESEARCH.md) states:

> "Perseveration depends on the action taken at the previous trial, which is a
> non-linear dependency (action itself results from softmax over current Q and
> WM, which is non-linear). This is Phase 20 (DEER) territory."

This characterization is correct for **simulation** (generating synthetic data)
where the action at trial t is sampled from the model's policy and therefore
depends on all previous states. But it is **incorrect** for **likelihood
evaluation** (the actual use case for MCMC fitting).

### Why Likelihood Evaluation is Different

In likelihood evaluation, the observed data sequence `(stimuli, actions,
rewards)` is fixed and known. The likelihood function computes the probability
of the OBSERVED action sequence given the model parameters. The key operations
in Phase 2 are:

```python
# From wmrl_m3_block_likelihood_pscan, Phase 2 policy_step:
def policy_step(carry, t_inputs):
    log_lik_accum, last_action = carry
    t_idx, stimulus, action, set_size, valid = t_inputs

    # Read pre-computed Q and WM (from Phase 1 parallel scan)
    q_vals = Q_for_policy[t_idx, stimulus]      # independent of carry
    wm_vals = wm_for_policy[t_idx, stimulus]    # independent of carry

    # Compute policy (independent of carry)
    omega = rho * jnp.minimum(1.0, capacity / set_size)
    rl_probs = softmax_policy(q_vals, FIXED_BETA)
    wm_probs = softmax_policy(wm_vals, FIXED_BETA)
    base_probs = omega * wm_probs + (1 - omega) * rl_probs
    noisy_base = apply_epsilon_noise(base_probs, epsilon, num_actions)

    # Perseveration: depends on last_action (the only carry)
    choice_kernel = jnp.eye(num_actions)[last_action]
    noisy_probs = (1 - kappa) * noisy_base + kappa * choice_kernel

    # Log-probability of OBSERVED action
    log_prob = jnp.log(noisy_probs[action] + 1e-8)

    # Update carry: last_action = OBSERVED action (from data, not model!)
    new_last_action = jnp.where(valid, action, last_action)
    return (log_lik_accum + log_prob, new_last_action), log_prob
```

The critical line is:
```python
new_last_action = jnp.where(valid, action, last_action)
```

Here, `action` is the **observed action from the data**, not a sample from the
model's policy. This means `last_action[t]` is entirely determined by the
observed data sequence, not by any model computation. Specifically:

- `last_action[0] = -1` (initial, no previous action)
- `last_action[t] = actions[t-1]` for t > 0 (assuming all trials are valid)
- For padded/invalid trials: `last_action[t] = actions[t_last_valid]`

### Precomputation is Trivial

For M3/M5 (global perseveration):
```python
# Precompute last_action from observed data
last_action_global = jnp.concatenate([jnp.array([-1]), actions[:-1]])
```

For M6a (stimulus-specific perseveration):
```python
# Precompute last_action_per_stimulus from observed data
# This requires tracking per-stimulus last action, but still depends only on
# observed (stimuli, actions) sequences -- a simple sequential scan over DATA,
# run once outside the likelihood function (not during MCMC).
```

For M6b (dual perseveration): combine both of the above.

The per-stimulus case (M6a/M6b) requires a scan to track which action was last
taken for each stimulus, but this scan depends only on observed data
`(stimuli, actions, mask)` -- it is parameter-independent and can be
precomputed once before MCMC begins.

### Implication: Phase 2 is Embarrassingly Parallel

Once `last_action` (or `last_actions` for M6a/M6b) is precomputed:

1. `Q_for_policy[t]` -- precomputed by Phase 1 associative scan
2. `wm_for_policy[t]` -- precomputed by Phase 1 associative scan
3. `last_action[t]` -- precomputed from observed data
4. `stimulus[t], action[t], set_size[t], mask[t]` -- observed data

Every input to the policy computation at trial t is available without
sequential dependency. The entire Phase 2 pass can be replaced with vectorized
array operations:

```python
# Vectorized Phase 2 (all T trials simultaneously)
q_vals = Q_for_policy[jnp.arange(T), stimuli]           # (T, A)
wm_vals = wm_for_policy[jnp.arange(T), stimuli]         # (T, A)

omega = rho * jnp.minimum(1.0, capacity / set_sizes)    # (T,)
rl_probs = jax.vmap(softmax_policy, in_axes=(0, None))(q_vals, FIXED_BETA)
wm_probs = jax.vmap(softmax_policy, in_axes=(0, None))(wm_vals, FIXED_BETA)
base_probs = omega[:, None] * wm_probs + (1 - omega[:, None]) * rl_probs
noisy_base = jax.vmap(apply_epsilon_noise, in_axes=(0, None, None))(
    base_probs, epsilon, num_actions
)

# Perseveration (precomputed last_action)
choice_kernels = jnp.eye(num_actions)[last_action_precomputed]  # (T, A)
noisy_probs = (1 - kappa) * noisy_base + kappa * choice_kernels

# Log-probabilities (all T trials at once)
log_probs = jnp.log(noisy_probs[jnp.arange(T), actions] + 1e-8)
total_ll = jnp.sum(log_probs * mask)
```

This is O(1) depth (fully parallel), no Newton iterations, no convergence
concerns, no numerical instability. The only constraint is that it requires
the Q and WM trajectories from Phase 1, which Phase 19 already provides.

---

## DEER Algorithm Analysis (for Completeness)

Even though DEER is not needed for the RLWM case, the roadmap requires a
thorough analysis. Here is the DEER deep-dive as specified by the success
criteria.

### Algorithm Overview

DEER (Lim et al., ICLR 2024) parallelizes non-linear recurrences of the form:
```
h_t = f(h_{t-1}, x_t)
```
by reformulating as a fixed-point problem. Define the residual:
```
r(h_{1:T}) = [h_1 - f(h_0, x_1), h_2 - f(h_1, x_2), ..., h_T - f(h_{T-1}, x_T)]
```
The true state trajectory satisfies `r(h*) = 0`. Newton's method iterates:
```
h^(i+1) = h^(i) - J(h^(i))^{-1} r(h^(i))
```
where J is the block-bidiagonal Jacobian:
```
J = [I_D        0       ...    0  ]
    [-df/dh_0   I_D     ...    0  ]
    [0          -df/dh_1 ...    0  ]
    [...        ...      ...   ... ]
    [0          0       -df/dh_{T-1} I_D]
```

The Newton update Dh_t = A_t * Dh_{t-1} + b_t is a linear recurrence
(with A_t = df/dh_{t-1} and b_t = -r_t), which can be solved via
`jax.lax.associative_scan` in O(log T) depth.

### Computational Complexity

| Variant | Work per Iteration | Memory per Iteration | Convergence |
|---------|--------------------|----------------------|-------------|
| Full DEER | O(T * D^3) | O(T * D^2) | Quadratic (Newton) |
| Quasi-DEER | O(T * D) | O(T * D) | Linear (guaranteed in <= T iters) |
| ELK | O(T * D^3) | O(T * D^2) | Damped Newton (stable) |
| Quasi-ELK | O(T * D) | O(T * D) | Damped linear (stable) |

### Convergence Guarantees

Proposition 1 (from Lim et al., extended by NeurIPS 2024 paper): If the
Jacobians df/dh are finite everywhere, undamped Newton converges in at most
T iterations for any initial condition. In practice, convergence is much
faster (2-5 iterations) when the dynamics are stable (eigenvalues of df/dh
have magnitude < 1).

### Empirical Performance

From the NeurIPS 2024 paper (quasi-DEER / ELK):
- Tested on V100 GPU with D in {32, 64, 128, 256} and T up to 17,984
- Speedup: up to 20x over sequential evaluation
- Quasi-DEER: ~10x less memory than full DEER
- **Failure case:** On trained autoregressive GRU, all parallel methods were
  SLOWER than sequential (221ms vs 96ms) due to convergence overhead

### Why DEER Would Not Help RLWM (Even if Needed)

Even if the RLWM non-linear components had true sequential dependencies,
DEER would be a poor fit for several reasons:

1. **Tiny state dimension:** RLWM perseveration state is D=1 (scalar
   `last_action` for M3/M5) or D=6 (per-stimulus `last_actions` for M6a).
   DEER's overhead per iteration (computing Jacobians, running linear scans)
   would dominate for such small D. The DEER papers test D >= 32.

2. **Short sequences:** T = 100 trials per block. DEER's O(log T) advantage
   over O(T) sequential is log_2(100) = 6.6x at best, minus Newton iteration
   overhead. For T=100, 3 Newton iterations would make the parallel version
   slower than sequential.

3. **Discrete state:** The `last_action` carry is a discrete integer (action
   index 0, 1, or 2), not a continuous state. DEER's Newton iteration
   linearizes around continuous trajectories. Linearizing a discrete
   variable is mathematically ill-defined.

4. **Non-smooth dynamics:** The argmax implicit in "which action was chosen"
   is not differentiable, so the Jacobian df/dh is undefined at the
   non-linearity. DEER requires smooth f for convergence.

5. **CPU is the target platform:** Phase 19 benchmarks show CPU sequential
   is faster than GPU parallel for RLWM (arithmetic intensity 0.3 FLOP/byte).
   DEER parallelism is designed for GPU throughput on large models.

---

## The Unifying Framework (TMLR 2025 / Lindermanlab 2025)

The paper "A Unifying Framework for Parallelizing Sequential Models with
Linear Dynamical Systems" (arXiv 2509.21716, Linderman Lab) provides a
taxonomy of parallel sequence methods. All fixed-point methods (Newton,
Picard, Jacobi) are shown to be special cases of solving a linearized
dynamical system via parallel scan.

### Where RLWM Fits in the Taxonomy

The RLWM likelihood has two structural layers:

1. **Linear layer (Q-update, WM-decay):** Pure AR(1) recurrences. These
   are the simplest case in the taxonomy: direct parallel scan with no
   iteration. Phase 19 handles this completely.

2. **Observation layer (policy computation):** Given the state trajectory
   (Q, WM) and observed data (stimuli, actions, rewards), compute
   log-probabilities. In likelihood evaluation, this layer has NO sequential
   dependencies -- it is a pointwise function of the state trajectory and
   observed data. This falls outside the taxonomy entirely because it is
   not a recurrence at all.

The framework's key insight is that parallelization difficulty scales with
the contractivity of the dynamics: systems with small Lipschitz constants
converge quickly. The RLWM "non-linearity" has Lipschitz constant = 0
(no actual sequential dependency), so it is trivially parallelizable.

---

## Alternative Approaches Comparison

| Approach | Applicable? | Complexity | Implementation Effort | Speedup |
|----------|-------------|------------|----------------------|---------|
| **Vectorized policy (recommended)** | Yes | O(1) depth | Low (array ops) | Eliminates Phase 2 scan entirely |
| DEER / Newton fixed-point | No | O(K * T * D^3) | High | Negative (overhead > benefit) |
| Picard iteration | No | O(K * T * D) | Medium | Negative (same issues as DEER) |
| Direct linearization | No | Not applicable | N/A | N/A (no recurrence to linearize) |

### Vectorized Policy (Recommended)

Replace the `lax.scan` Phase 2 with pure array operations:

- Precompute `last_action` arrays from observed data (once, outside MCMC)
- Use broadcasting/vmap to compute all T trial log-probs simultaneously
- No iteration, no convergence concerns, no numerical instability
- Works identically on CPU and GPU

### Why Not DEER

See analysis above. The RLWM non-linearity is a phantom: it appears
sequential in the `lax.scan` implementation but is actually embarrassingly
parallel in the likelihood evaluation context.

---

## Contraction Mapping Analysis

The roadmap asks: "Is the RLWM softmax-mixing a contraction mapping?"

This question is moot because the softmax-mixing is not a recurrence in
likelihood evaluation. But for completeness:

### If It Were a Recurrence (Simulation Context)

In simulation, the state `h_t = last_action_t` evolves as:
```
h_t = argmax(softmax(beta * policy(Q_t, WM_t, h_{t-1})))
```

The "dynamics" f maps integer -> probability distribution -> integer. This is:
- **Not continuous** (discrete state space {0, 1, 2})
- **Not differentiable** (argmax is not smooth)
- **Not a contraction** (small changes in policy can flip the argmax)

DEER's convergence theory requires smooth, continuously-differentiable
dynamics. The RLWM perseveration recurrence violates this assumption.

### In Likelihood Evaluation (Actual Use Case)

There is no recurrence. Each trial's log-probability is an independent
function of known inputs. The contraction mapping question does not apply.

---

## Implementation Plan

### What to Build

1. **Precompute perseveration arrays** from observed data:
   - `precompute_last_action_global(actions, mask)` -> array shape (T,)
   - `precompute_last_actions_per_stimulus(stimuli, actions, mask, S)` -> array shape (T, S) or (T,)
   - Run once before MCMC, store as additional data arrays

2. **Vectorized policy functions** for each model:
   - M1 (Q-learning): already no Phase 2 carry; vectorize directly
   - M2 (WM-RL): already no perseveration carry; vectorize directly
   - M3 (WM-RL+kappa): precompute global last_action, vectorize
   - M5 (WM-RL+phi_rl): same as M3
   - M6a (WM-RL+kappa_s): precompute per-stimulus last_actions, vectorize
   - M6b (WM-RL+dual): precompute both, vectorize

3. **Benchmark against Phase 19 hybrid** (pscan Phase 1 + sequential Phase 2)
   using existing `validation/benchmark_parallel_scan.py` infrastructure

4. **Update documentation** with findings

### What NOT to Build

- No DEER implementation
- No Newton iteration
- No fixed-point solver
- No quasi-Newton / ELK machinery

### Expected Speedup

The Phase 2 sequential scan currently runs O(T) steps per block. With
vectorized operations, this becomes O(1) on GPU (all T trials computed
simultaneously) or a constant-factor improvement on CPU (SIMD vectorization
of the array operations). The main benefit is:

- **Eliminates the sequential scan bottleneck** for perseveration models
- **Enables full GPU parallelism** if the Q/WM scan + vectorized policy
  combined have enough arithmetic intensity
- **Simplifies the code** (array ops are easier to read than lax.scan)

For T=100 per block, the speedup from vectorizing Phase 2 alone is modest
(Phase 2 is already fast per step). The more significant benefit is enabling
future GPU acceleration by making the entire likelihood fully parallel.

---

## Per-Stimulus Perseveration: The One Complication

For M6a and M6b, the perseveration carry is `last_actions` (shape
`(num_stimuli,)` = `(6,)`), tracking the last action taken for EACH stimulus.
This requires knowing which stimulus was presented at each trial and what
action was taken.

### Precomputation Approach

Since stimuli and actions are observed data, this can be precomputed:

```python
def precompute_last_actions_per_stimulus(
    stimuli: jnp.ndarray,      # (T,) int
    actions: jnp.ndarray,      # (T,) int
    mask: jnp.ndarray,         # (T,) float
    num_stimuli: int = 6,
) -> jnp.ndarray:
    """Precompute per-stimulus last_action for all T trials.

    Returns shape (T,) where result[t] is the last action taken
    for stimulus[t], considering trials 0..t-1.
    """
    def scan_fn(last_actions, t_inputs):
        stimulus, action, valid = t_inputs
        # Output: last action for THIS stimulus before THIS trial
        last_action_s = last_actions[stimulus]
        # Update: record this trial's action for this stimulus
        new_last_actions = last_actions.at[stimulus].set(
            jnp.where(valid, action, last_action_s).astype(jnp.int32)
        )
        return new_last_actions, last_action_s

    init = jnp.full((num_stimuli,), -1, dtype=jnp.int32)
    _, last_action_per_trial = jax.lax.scan(
        scan_fn, init, (stimuli, actions, mask)
    )
    return last_action_per_trial  # (T,) int, last action for each trial's stimulus
```

This scan is O(T) but runs ONCE (outside the likelihood function, before MCMC).
It depends only on observed data, not model parameters. During MCMC, the
precomputed array is passed as additional data, and the likelihood function
uses it as a simple array index -- no sequential carry needed.

### Alternative: Pure Array Construction

For the GLOBAL perseveration (M3/M5), the precomputation is even simpler:
```python
last_action_global = jnp.concatenate([jnp.array([-1]), actions[:-1]])
# Handle mask: where mask[t-1]==0, propagate from earlier valid trial
# (This matches the existing lax.scan behavior)
```

With valid masking, this may need a scan for correctness (masked trials
don't update last_action). But again, this is a one-time data precomputation.

---

## Go/No-Go Recommendation

### DEER: NO-GO

**Reason:** The RLWM non-linear components do not constitute a sequential
recurrence in the likelihood evaluation context. The only apparent sequential
dependency (perseveration carry) is actually determined by observed data alone.
DEER solves a problem that does not exist here.

**Supporting evidence:**
1. Code inspection of all 12 pscan likelihood variants confirms that
   `last_action` / `last_actions` carries are updated with OBSERVED actions,
   not model-computed actions (HIGH confidence)
2. DEER requires continuous, differentiable dynamics; perseveration uses
   discrete integer state (HIGH confidence)
3. DEER complexity O(K * T * D^3) exceeds sequential O(T) for D=1, T=100
   (HIGH confidence, from DEER paper complexity analysis)
4. CPU benchmarks show sequential is already faster than parallel scan for
   RLWM (HIGH confidence, from pscan_benchmark.json)

### Vectorized Policy: GO

**Reason:** Replacing the Phase 2 sequential `lax.scan` with vectorized array
operations is straightforward, correct, and beneficial. It eliminates the last
sequential bottleneck in the likelihood, enabling future GPU acceleration.

---

## Pitfalls

### Pitfall 1: Mask Handling in Precomputed Perseveration

**What goes wrong:** Invalid/padded trials should not update `last_action`.
The existing `lax.scan` handles this with `jnp.where(valid, action,
last_action)`. The precomputation must replicate this logic exactly.

**Prevention:** Use a data-precomputation scan (shown above) that respects
the mask, and verify agreement with sequential implementation to < 1e-8
(exact agreement expected since no floating-point approximation is involved).

### Pitfall 2: First Trial Edge Case

**What goes wrong:** At trial t=0, `last_action = -1` (no previous action).
The vectorized code must handle this sentinel value correctly: when
`last_action < 0`, use the M2 path (no perseveration).

**Prevention:** The precomputed array should contain -1 at positions where
no valid previous action exists. The vectorized policy should use the same
`jnp.where(use_m2_path, noisy_base, hybrid_probs)` logic.

### Pitfall 3: Assuming Vectorization Always Speeds Up

**What goes wrong:** On CPU with T=100, the vectorized Phase 2 may not be
faster than the sequential `lax.scan` Phase 2. The scan overhead is small,
and CPU SIMD may not provide meaningful speedup for T=100 trials of 3-element
softmax computations.

**Prevention:** Benchmark both approaches. The primary motivation is
correctness and enabling GPU parallelism, not necessarily CPU speedup.
Phase 19 benchmark showed pscan was 0.26x (slower) on CPU, so the same
dynamic may apply to vectorized Phase 2.

### Pitfall 4: Conflating Simulation and Likelihood

**What goes wrong:** Future developers may read the Phase 19 research
(which says perseveration is non-linear / sequential) and revert to sequential
implementation, not understanding that it only applies to simulation.

**Prevention:** Document clearly in the code and in
`docs/PARALLEL_SCAN_LIKELIHOOD.md` that in likelihood evaluation, actions are
observed data, making the perseveration carry precomputable.

---

## Sources

### HIGH Confidence (Official / Direct Inspection)

- Phase 19 pscan implementations in `scripts/fitting/jax_likelihoods.py`
  (lines 3466-4427): Direct code inspection showing `last_action` is
  updated with observed `action` from data, not model output
- Phase 19 benchmark results: `output/bayesian/pscan_benchmark.json`
  (CPU, JAX 0.4.31, pscan 0.26x speedup = 4x slowdown)

### MEDIUM Confidence (Peer-Reviewed Papers)

- DEER: Lim et al., "Parallelizing non-linear sequential models over the
  sequence length," ICLR 2024.
  [arXiv:2309.12252](https://arxiv.org/abs/2309.12252)
- Quasi-DEER / ELK: "Towards Scalable and Stable Parallelization of
  Nonlinear RNNs," NeurIPS 2024.
  [arXiv:2407.19115](https://arxiv.org/abs/2407.19115)
- Unifying Framework: "A Unifying Framework for Parallelizing Sequential
  Models with Linear Dynamical Systems."
  [arXiv:2509.21716](https://arxiv.org/abs/2509.21716)
- DEER reference implementation: [github.com/machine-discovery/deer](https://github.com/machine-discovery/deer)
  (JAX, V100 benchmark: 68x speedup on T=10000 GRU, D=?)
- ELK reference implementation: [github.com/lindermanlab/elk](https://github.com/lindermanlab/elk)
  (JAX, NeurIPS 2024)
- Unifying Framework code: [github.com/lindermanlab/parallelizing_with_lds](https://github.com/lindermanlab/parallelizing_with_lds)

### LOW Confidence (WebSearch Only)

- No LOW confidence findings needed; all conclusions derived from code
  inspection and verified paper content.

---

## Implications for Roadmap

### Phase Structure Recommendation

The Phase 20 scope should be reframed from "DEER research" to "vectorized
policy pass" implementation. The research component (this document) is
complete with a clear NO-GO on DEER and GO on vectorized policy.

Suggested plan structure:

1. **Plan 20-01: Research document + precomputation functions**
   - Write `docs/DEER_NONLINEAR_PARALLELIZATION.md` (the main deliverable)
   - Implement `precompute_last_action_global()` and
     `precompute_last_actions_per_stimulus()` utility functions
   - Unit tests verifying agreement with sequential carry
   - ~1 plan, small scope

2. **Plan 20-02: Vectorized policy likelihood variants**
   - Replace Phase 2 `lax.scan` with vectorized array ops in all 12 pscan
     variants (or create new `*_pscan_vec` variants)
   - Benchmark against Phase 19 hybrid (pscan + sequential Phase 2)
   - Numerical agreement tests (< 1e-6 relative error expected: exact)
   - ~1 plan, medium scope

3. **Plan 20-03: Documentation and benchmark update**
   - Update `docs/PARALLEL_SCAN_LIKELIHOOD.md` with Phase 20 findings
   - Update `docs/JAX_GPU_BAYESIAN_FITTING.md` (if exists) or create
   - Update benchmark script to include vectorized variants
   - ~1 plan, small scope

### Phase Ordering Rationale

- Plan 20-01 (research doc + precomputation) has no dependencies beyond
  Phase 19 completion, which is already done
- Plan 20-02 (vectorized variants) depends on 20-01 precomputation functions
- Plan 20-03 (docs) depends on 20-02 benchmark results

### Research Flags

- **No deeper research needed:** The DEER analysis is conclusive. The
  vectorized approach is standard JAX array programming.
- **Benchmark results may surprise:** CPU vectorized may not be faster than
  CPU sequential for T=100. This is acceptable -- the goal is enabling GPU
  parallelism and code simplification.
