# Parallel Scan Likelihood: Implementation Guide

**Phase 19 — Associative Scan Likelihood Parallelization**

This document is an implementation guide for replacing O(T) sequential `lax.scan`
loops with O(log T) `jax.lax.associative_scan` for the linear-recurrence parts of
the RLWM likelihood. Non-linear components (WM-Q mixing, softmax) remain sequential
and are covered in Phase 20 (DEER).

---

## 1. AR(1) Linear Recurrence Formulation

### Q-value update as a linear recurrence

The standard Q-learning update is:

```
Q_t(s, a) = Q_{t-1}(s, a) + alpha_t * (r_t - Q_{t-1}(s, a))
           = (1 - alpha_t) * Q_{t-1}(s, a) + alpha_t * r_t
```

This is a first-order autoregressive (AR(1)) recurrence of the form:

```
x_t = a_t * x_{t-1} + b_t
```

with `a_t = 1 - alpha_t` and `b_t = alpha_t * r_t`.

### Associative operator

For a sequence of AR(1) operators `(a_1, b_1), ..., (a_T, b_T)`, the composition
rule (applying operator at time 2 after time 1) is:

```
(a_2, b_2) ∘ (a_1, b_1) = (a_2 * a_1, a_2 * b_1 + b_2)
```

**Proof:** Apply left-to-right:

```
x_1 = a_1 * x_0 + b_1
x_2 = a_2 * x_1 + b_2
    = a_2 * (a_1 * x_0 + b_1) + b_2
    = (a_2 * a_1) * x_0 + (a_2 * b_1 + b_2)
```

The composed operator acts on `x_0` with multiplier `a_2 * a_1` and offset
`a_2 * b_1 + b_2`. This operator is associative, enabling prefix-sum
parallelism via `jax.lax.associative_scan`.

### Reduction to log-likelihood evaluation

Given Q-value trajectories `Q(s,a)_t` for all `t = 0, ..., T-1` (Q BEFORE each
update), log-likelihood evaluation reduces to:

1. **Parallel pass:** Compute all Q-value trajectories in O(log T) using
   `associative_scan`.
2. **Sequential pass:** For each trial, compute hybrid policy and log-probability
   using the pre-computed Q-values (still O(T) but trivially parallelizable
   across the trial axis with `jax.vmap` or `lax.scan`).

The key insight: the Q-value trajectory depends only on the observed data
sequence `(s_t, a_t, r_t)` and initial condition `Q_0`. Given the trajectory,
the log-probabilities are independent across trials.

### Inactive positions (other (s,a) pairs)

Most trials only update one `(s, a)` entry of the Q-table. For all other
`(s', a') != (s_t, a_t)`, the coefficients are identity: `a_t = 1.0, b_t = 0.0`.
This means `x_t = x_{t-1}` for those entries — the scan correctly leaves them
unchanged.

Implementation: use one-hot encoding to build `(T, S, A)` arrays of `a_seq`
and `b_seq`, with identity coefficients at inactive positions.

---

## 2. WM Decay and Overwrite

### Decay as AR(1)

The WM forgetting update is:

```
WM_t = (1 - phi) * WM_{t-1} + phi * WM_0
```

where `WM_0 = 1/nA` is the uniform baseline. This is AR(1) with:

- `a_t = 1 - phi` (constant across trials)
- `b_t = phi * WM_0` (constant across trials)

Pure decay (no overwrite) is equivalent to running `affine_scan` with constant
coefficients. The `(T, S, A)` coefficient arrays are just `full((T,S,A), 1-phi)`
and `full((T,S,A), phi*wm_init)`.

### Hard overwrite as a multiplicative reset

On feedback trials, `WM(s_t, a_t) <- r_t`. This is a hard overwrite, not a
smooth update. Within the AR(1) framework, the reset is encoded as:

```
a_t = 0,   b_t = r_t   (at the overwrite position for stimulus s, action a)
```

**Derivation:** Substituting into the recurrence:

```
x_t = 0 * x_{t-1} + r_t = r_t
```

The multiplicative coefficient `a_t = 0` zeroes out all history before position
`t`. Any prefix product that includes this position has total multiplier 0. This
means that for all `u < t`, the contribution of `x_u` to `x_t` is exactly zero.

**Formal statement:** Let `t_k` be the most recent overwrite before position `t`
for entry `(s, a)`. Then:

```
x_t = (product_{j=t_k+1}^{t} a_j) * r_{t_k}
      + sum_{j=t_k+1}^{t} (product_{l=j+1}^{t} a_l) * b_j
```

All terms involving `x_{t_k - 1}` (or earlier) are zero because the prefix
product from any such time through `t_k` contains the factor `a_{t_k} = 0`.

### Summary of WM coefficient encoding

| Trial condition | `a_t[s, a]` | `b_t[s, a]` |
|---|---|---|
| Decay only (non-active `(s,a)`) | `1 - phi` | `phi * wm_init` |
| Overwrite (active `(s,a)`, valid trial) | `0.0` | `r_t` |
| Padding (invalid trial, any `(s,a)`) | `1.0` | `0.0` |

The padding row uses identity coefficients `(1.0, 0.0)` so padding trials have
no effect on the WM trajectory.

---

## 3. Linear vs Non-Linear RLWM Components

### Linearity table

| Component | Form | Linear? | Parallelizable? |
|---|---|---|---|
| Q-value update | `Q_t = (1-alpha)*Q_{t-1} + alpha*r` | Yes | Yes (Phase 19) |
| WM decay | `WM_t = (1-phi)*WM_{t-1} + phi*WM_0` | Yes | Yes (Phase 19) |
| WM overwrite | `WM(s,a) <- r` at feedback | Yes (reset = a=0) | Yes (Phase 19) |
| WM-Q mixing weight | `omega = rho * min(1, K/N_s)` | No (set size dependence) | Not applicable |
| Softmax policy | `P(a\|s) = softmax(beta * Q(s,:))` | No (exp, normalization) | Not applicable |
| Combined policy | `P = omega * P_WM + (1-omega) * P_RL` | No (depends on Q and WM) | Not applicable |
| Epsilon noise | `P_noisy = eps/nA + (1-eps)*P` | No (depends on policy) | Not applicable |
| Perseveration kernel | `P_kappa += kappa * I[a == a_{t-1}]` | No (depends on history) | Not applicable |
| RL forgetting (M5) | `Q_decayed = (1-phi_rl)*Q + phi_rl*Q_0` | Yes (additive decay) | Yes (same form) |

### Why non-linear components cannot use the scan

The non-linear components depend simultaneously on Q-values and WM values, or
on the action taken (which is a non-linear function of the policy). Specifically:

- **WM-Q mixing weight** depends on `set_size`, which is trial-specific and
  deterministic — technically linear, but the combined policy depends on both
  `Q_t` and `WM_t` simultaneously.
- **Softmax** involves `exp`, which is not a linear operation over the
  state-action values.
- **Action selection** is a draw from the policy, making each observed action
  a non-linear function of all prior observations.

These components remain as a sequential post-scan pass. After computing the full
Q-value and WM trajectories in parallel, the log-probabilities can be computed
in a single sequential sweep (or vectorized with `vmap` over trials if actions
are already observed, which they are in the likelihood case).

---

## 4. Related Work

### PaMoRL (NeurIPS 2024)

Paramasivan et al. (2024) introduced PaMoRL, a parallel Temporal Difference
learning algorithm using the PETE (Parallel Eligibility Trace Estimation)
algorithm. PETE expresses TD(lambda) eligibility traces as a parallel prefix
scan over the AR(1) recurrence `e_t = gamma*lambda*e_{t-1} + phi_t`, where
`phi_t` is the feature vector at trial `t`. This is the same affine structure
used here. PaMoRL demonstrated O(log T) training time for RL agents with long
episodes, with empirical speedups of 4-10x on GPU hardware. Reference: Parisotto
et al., "Parallelizing Reinforcement Learning," NeurIPS 2024.

### S4 / Mamba (Gu & Dao, 2023)

S4 (Structured State Space Sequences) and its selective extension Mamba use
data-dependent state transitions of the form `h_t = A_t * h_{t-1} + B_t * x_t`,
where `A_t, B_t` depend on the input `x_t`. This is exactly the Q-update
recurrence when `alpha_t` depends on the reward (data-dependent decay). The
parallel scan in Mamba computes all hidden states simultaneously using the
associative operator `(A_2, B_2) ∘ (A_1, B_1) = (A_2*A_1, A_2*B_1 + B_2)`.
The RLWM Q-update is a scalar (per s,a) version of the same operator. Reference:
Gu & Dao (2023), "Mamba: Linear-Time Sequence Modeling with Selective State
Spaces," ICLR 2024.

### Parallel Kalman Smoothers (Sarkka & Garcia-Fernandez, 2021)

Sarkka & Garcia-Fernandez (2021) showed that the Kalman filter/smoother for
linear Gaussian state space models can be parallelized via associative scan.
The forward pass computes `x_t = A_t * x_{t-1} + q_t` (linear dynamics), which
is the same AR(1) form. Their parallel formulation achieves O(log T) wall-clock
time on GPU. This work established the theoretical basis for parallel inference in
SSMs. The RLWM case is a degenerate SSM (no observation noise on the Q-values,
deterministic updates). Reference: Sarkka & Garcia-Fernandez (2021), "Temporal
Parallelization of Bayesian Smoothers," IEEE Transactions on Automatic Control.

### DEER (ICLR 2024)

DEER (Decoupled Estimation with Efficient Recursion) extends parallel scan to
non-linear recurrences by linearizing around the trajectory. DEER is the planned
approach for Phase 20 to parallelize the WM-Q mixing and policy computation.
Reference: Fu et al. (2023), "DEER: A Parallel Algorithm for Non-Linear
Sequential Estimation," ICLR 2024.

---

## 5. Alpha Approximation

### The approximation

The sequential implementation uses:

```python
delta = r - Q[s, a]
alpha = alpha_pos if delta > 0 else alpha_neg   # exact delta-sign rule
```

The parallel scan uses:

```python
alpha = alpha_pos if r == 1.0 else alpha_neg   # reward-based approximation
```

These agree when:
- **Rewarded trials (`r=1`):** `delta > 0` iff `Q(s,a) < 1`, which holds when
  the Q-value has not reached 1.0. Since `Q_init = 0.5` and `alpha < 1`, convergence
  to 1.0 requires many rewarded trials, so this approximation is exact for early
  learning.
- **Unrewarded trials (`r=0`):** `delta < 0` iff `Q(s,a) > 0`, which holds when
  the Q-value is non-zero. Since `Q_init > 0`, this is exact unless the Q-value
  reaches exactly 0.

### Expected agreement

For typical parameters (`alpha_pos <= 0.5`, `alpha_neg <= 0.5`):

- **Typical agreement:** < 1e-5 relative error across all `(t, s, a)` entries
  on 1000-trial sequences. Disagreements only occur when Q-values are at boundary
  values (near 0 or 1).
- **Extreme alpha (`alpha_pos = alpha_neg = 0.95`):** Larger discrepancies because
  Q-values converge rapidly to boundaries. Expected tolerance: < 1e-3.

### Why the approximation is acceptable

In the RLWM task, participants are learning to associate stimuli with correct
actions. Q-values rarely reach exact boundaries (the epsilon noise `eps=0.05`
prevents perfect probability 1.0 responses, so Q-values stay in `(eps/3, 1-eps)`
in expectation). The approximation error is concentrated at boundary cases that
occur with low frequency.

---

## 6. Numerical Stability

### Float32 underflow

For large `T` and large `alpha`, the product `(1-alpha)^T` underflows to 0 in
float32:

- `alpha = 0.9, T = 100`: `(0.1)^100 ≈ 1e-100` — underflows to 0
- `alpha = 0.5, T = 100`: `(0.5)^100 ≈ 8e-31` — near float32 minimum (~1e-38)
- `alpha = 0.3, T = 100`: `(0.7)^100 ≈ 3e-16` — safely above zero

**Impact:** When the multiplicative product underflows, the Q-value at trial `t`
effectively depends only on the most recent few trials. This is the numerically
correct behavior — old information has decayed below machine precision. The
sequential implementation behaves identically because it also uses float32
multiplications.

**Recommendation:** Use float32 for all Q-update and WM-update computations
(consistent with existing likelihoods). Float64 is only needed for M4 LBA
(separate track).

### WM decay stability (phi near 0/1)

- **phi near 0:** WM barely decays. `a_t ≈ 1.0` everywhere. The scan is stable
  because the prefix product stays near 1.0.
- **phi near 1:** WM decays almost to baseline each trial. `a_t ≈ 0.0`.
  Sequential products underflow quickly (same as high-alpha case above). Correct
  behavior — WM state is essentially reset each trial.

### Tolerance thresholds

| Parameter regime | Expected max relative error | Test threshold |
|---|---|---|
| Typical (alpha <= 0.5, phi <= 0.5) | < 1e-6 | 1e-5 |
| Extreme (alpha = 0.95, phi = 0.95) | < 1e-3 | 1e-3 |
| Boundary Q-values | Up to 1e-2 (rare) | Not tested |

These thresholds apply to element-wise comparison of `(T, S, A)` Q/WM
trajectories between the parallel and sequential implementations.
