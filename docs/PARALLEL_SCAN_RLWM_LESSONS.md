---
title: Parallel / Associative Scan for RLWM — Methods, Maths, and Where It Scaled
aliases:
  - parallel-scan-rlwm
  - associative-scan-lessons
  - pscan-rlwm
tags:
  - jax
  - parallel-scan
  - associative-scan
  - reinforcement-learning
  - working-memory
  - mcmc
  - deer
  - state-space-models
  - ar1
created: 2026-04-27
status: lessons-learned
related:
  - "[[PARALLEL_SCAN_LIKELIHOOD]]"
  - "[[CLUSTER_GPU_LESSONS]]"
  - "[[legacy/DEER_NONLINEAR_PARALLELIZATION]]"
project: rlwm_trauma_analysis
phases: [19, 20]
---

# Parallel / Associative Scan for RLWM — Methods, Maths, and Where It Scaled

> [!abstract] TL;DR
> We rewrote the per-trial Q-learning and working-memory updates of a 6-model RLWM family as **AR(1) recurrences** and evaluated them with [`jax.lax.associative_scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.associative_scan.html), giving **O(log T) depth** in theory. In practice, for T=100 trials × 17 blocks the associative scan was **~3.7× slower on CPU** (the log-depth benefit doesn't pay off until T is large). The real wall-clock win came from the *complementary* refactor: realizing the perseveration "carry" in our likelihood was a **phantom recurrence** — actions are observed data during likelihood evaluation, so `last_action[t]` is precomputable once before MCMC and the entire policy phase becomes embarrassingly parallel via `vmap`. We also looked at [DEER](https://arxiv.org/abs/2309.12252) for the non-linear policy step and rejected it; this writeup explains *why* and what other settings would actually benefit.

This is a **retrospective** companion to [[PARALLEL_SCAN_LIKELIHOOD]] (the implementation guide). Read this if you want the lessons; read that if you want the API contract.

---

## 1. The Recurrence That We Tried to Parallelise

A single trial of the RLWM model (Senta et al., 2025; [doi:10.1371/journal.pcbi.1012872](https://doi.org/10.1371/journal.pcbi.1012872)) updates two state tensors — a Q-table and a working-memory table — and emits a choice probability through a softmax mixture.

For one $(s, a)$ entry, both updates are **first-order autoregressive (AR(1))**:

$$
\underbrace{Q_t(s,a)}_{\text{state}} = \underbrace{(1-\alpha_t)}_{a_t}\,Q_{t-1}(s,a) + \underbrace{\alpha_t r_t}_{b_t}
$$

$$
\underbrace{\mathrm{WM}_t(s,a)}_{\text{state}} = \underbrace{(1-\phi)}_{a_t}\,\mathrm{WM}_{t-1}(s,a) + \underbrace{\phi \cdot \mathrm{WM}_0}_{b_t}
$$

Both have the form $x_t = a_t x_{t-1} + b_t$. That's exactly the family that admits a parallel prefix scan via the affine composition operator:

$$
(a_2, b_2) \circ (a_1, b_1) = (a_2 a_1,\; a_2 b_1 + b_2)
$$

**Why this matters:** if you can express your update as an AR(1) and you can supply $a_t$ and $b_t$ as data-independent (or trivially data-dependent) sequences, you get a $O(\log T)$-depth, $O(T)$-work parallel evaluation for free using [Blelloch's prefix-sum algorithm](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf) — which is what JAX's `associative_scan` implements.

> [!math] Associativity proof
> $x_2 = a_2(a_1 x_0 + b_1) + b_2 = (a_2 a_1)x_0 + (a_2 b_1 + b_2)$ — the composed operator $(a_2 a_1,\, a_2 b_1 + b_2)$ acts on $x_0$ exactly the same way the two-step recurrence does. Composition is associative, so any tree-reduction order produces the same answer.

---

## 2. The Two Tricks That Made It Work

The textbook AR(1) parallel scan handles smooth, scalar updates. The RLWM updates have two warts that needed encoding tricks.

### 2.1 Encoding hard overwrite as a multiplicative reset

The WM update is **not** smooth: on a feedback trial, $\mathrm{WM}(s_t, a_t) \leftarrow r_t$ — a hard overwrite, not a convex combination. Sequential code writes that as `WM = WM.at[s, a].set(r)`. That looks fundamentally non-linear.

It's not. Inside the AR(1) framework, a hard reset to value $r$ at trial $t$ is encoded as:

$$
a_t = 0,\qquad b_t = r
$$

Substituting: $x_t = 0 \cdot x_{t-1} + r = r$ ✓.

The $a_t = 0$ does the heavy lifting: it **zeroes out all history before position $t$**. Any prefix product crossing this point gets the factor 0, so the contribution of every $x_u$ for $u < t$ collapses to zero — exactly what an overwrite means.

| Trial condition for entry $(s,a)$ | $a_t$ | $b_t$ |
|---|---|---|
| Inactive (some other $(s',a')$ presented) | $1 - \phi$ | $\phi \cdot \mathrm{WM}_0$ |
| Active, valid trial (overwrite) | $0$ | $r_t$ |
| Active, *padding* trial | $1 - \phi$ | $\phi \cdot \mathrm{WM}_0$ |

> [!warning] Subtle padding asymmetry
> For Q-updates, padding trials use **identity coefficients** $(1, 0)$ — they don't change state. For WM, padding trials use **decay** $(1-\phi, \phi \mathrm{WM}_0)$, because the sequential reference applies WM decay every trial regardless of validity (the mask only gates the *overwrite*, not the *decay*). Get this wrong and the parallel scan disagrees with the sequential reference at $\sim 10^{-2}$ relative error instead of $10^{-6}$.

See [`src/rlwm/fitting/core.py:439`](../src/rlwm/fitting/core.py#L439) (`associative_scan_wm_update`) for the implementation.

### 2.2 The reward-based α approximation

The asymmetric Q-learning update uses $\alpha = \alpha_+$ for positive prediction errors and $\alpha_-$ for negative ones:

$$
\alpha_t = \begin{cases} \alpha_+ & \text{if } \delta_t = r_t - Q_{t-1}(s,a) > 0 \\ \alpha_- & \text{otherwise}\end{cases}
$$

This is a problem: $\delta_t$ depends on $Q_{t-1}$, which is the *output* of the scan we're trying to compute. Data-dependent coefficients make the scan no longer pure-AR(1).

**The trick:** approximate the prediction-error sign by the reward sign.

$$
\alpha_t \approx \begin{cases}\alpha_+ & \text{if } r_t = 1 \\ \alpha_- & \text{if } r_t = 0\end{cases}
$$

Both $\alpha_t$ and $b_t$ are now functions of $r_t$ alone, so the coefficient arrays can be built upfront from observed data. The approximation is **exact** unless $Q$ has converged to the boundaries of $[0, 1]$, which barely happens in practice because $\epsilon$-noise keeps $Q$ in $(\epsilon/3, 1-\epsilon)$.

| Parameter regime | Max relative error vs. exact rule |
|---|---|
| Typical ($\alpha \le 0.5$) | $< 10^{-5}$ |
| Extreme ($\alpha = 0.95$) | $< 10^{-3}$ |

This is the cleanest case of the more general principle: **if your "data-dependent decay" only depends on the *observed* portion of the data (rewards, stimuli) and not on the *latent* state being computed, the scan is fully linear.** Mamba ([Gu & Dao, 2023](https://arxiv.org/abs/2312.00752)) makes the same observation: $h_t = A_t h_{t-1} + B_t x_t$ is linear-in-state even when $A_t, B_t$ depend on the input $x_t$.

---

## 3. The Reusable Primitive

Here's the AR(1) scan as a 30-line transferable primitive. Drop this into any project where you have a linear recurrence over a long sequence.

```python
# Adapted from src/rlwm/fitting/core.py
import jax
import jax.numpy as jnp
from jax import lax

def affine_scan(a_seq: jnp.ndarray, b_seq: jnp.ndarray, x0: jnp.ndarray) -> jnp.ndarray:
    """Parallel prefix scan for x_t = a_t * x_{t-1} + b_t.

    Computes the full trajectory in O(log T) depth on parallel hardware
    using jax.lax.associative_scan with the affine composition operator.

    Parameters
    ----------
    a_seq : (T, ...) multiplicative coefficients
    b_seq : (T, ...) additive coefficients
    x0    : (...)    initial state, broadcastable to a_seq[0]

    Returns
    -------
    x_all : (T, ...) where x_all[t] is the state AFTER applying step t
    """
    trailing = a_seq.shape[1:]
    x0_b = jnp.broadcast_to(x0, trailing)

    # Prepend (a=1, b=x0) as the identity element so the scan absorbs x0
    a_full = jnp.concatenate([jnp.ones(trailing)[None], a_seq], axis=0)
    b_full = jnp.concatenate([x0_b[None],            b_seq], axis=0)

    def affine_op(left, right):
        a_l, b_l = left
        a_r, b_r = right
        return a_r * a_l, a_r * b_l + b_r           # right ∘ left

    _, b_acc = lax.associative_scan(affine_op, (a_full, b_full))
    return b_acc[1:]                                 # drop the prepended init
```

**Three things worth knowing about this primitive:**

1. **`trailing = a_seq.shape[1:]`** lets the scan operate over arbitrary tensor shapes — for RLWM we pass `(T, S, A) = (100, 6, 3)` arrays. Each `(s, a)` entry is its own AR(1) running in parallel; one scan handles all 18 channels.
2. **The prepended identity element** is the standard trick to fold $x_0$ into a prefix scan. JAX's `associative_scan` operates inclusively starting from index 0, so we add a sentinel $(1, x_0)$ at position $-1$ and drop the result.
3. **Composition direction matters.** `lax.associative_scan` accumulates left-to-right by default, so `affine_op(left, right)` must compute $\text{right} \circ \text{left}$, not $\text{left} \circ \text{right}$. The asymmetric formula $(a_r a_l, a_r b_l + b_r)$ encodes that direction. Swap the args by accident and you'll get a silently-wrong scan that *looks* right on $T=2$ but disagrees on $T=3$.

---

## 4. Worked Example: One-block Q-update via affine_scan

This is the heart of `associative_scan_q_update` — encode the recurrence, then call the primitive.

```python
import jax
import jax.numpy as jnp

def q_update_pscan(stimuli, actions, rewards, masks,
                   alpha_pos: float, alpha_neg: float,
                   q_init: float = 0.5,
                   num_stimuli: int = 6, num_actions: int = 3):
    T = stimuli.shape[0]
    S, A = num_stimuli, num_actions

    # One-hot the active (s, a) pair at each trial
    stim_oh = jax.nn.one_hot(stimuli, S)            # (T, S)
    act_oh  = jax.nn.one_hot(actions,  A)           # (T, A)
    sa_mask = stim_oh[:, :, None] * act_oh[:, None, :]   # (T, S, A) — 1 only at active cell
    active  = sa_mask * masks[:, None, None]              # gate by validity mask

    # Reward-based alpha (Section 2.2 trick)
    alpha_t = jnp.where(rewards[:, None, None] == 1.0, alpha_pos, alpha_neg)

    # AR(1) coefficients: active → (1-alpha, alpha*r), inactive → (1, 0)
    a_seq = jnp.where(active, 1.0 - alpha_t, 1.0)
    b_seq = jnp.where(active, alpha_t * rewards[:, None, None], 0.0)

    # Run the scan — same primitive as Section 3
    Q_after = affine_scan(a_seq, b_seq, x0=jnp.full((S, A), q_init))      # (T, S, A)

    # For likelihood we want Q BEFORE each update (the policy at trial t reads Q_{t-1})
    Q_before = jnp.concatenate([jnp.full((1, S, A), q_init), Q_after[:-1]], axis=0)
    return Q_before
```

That's it. The same template covers WM with one substitution: replace the "inactive → identity" branch with "inactive → decay" and the "active → reset" coefficients $(0, r_t)$ from Section 2.1.

---

## 5. The Phantom Recurrence — and Why Phase 20 Was the Real Win

After Phase 19, the Q and WM tables compute in $O(\log T)$. But the *policy* step still ran inside a `lax.scan`, because models M3/M5/M6a/M6b have a perseveration kernel:

$$
P(a_t = a) \mathrel{+}= \kappa \cdot \mathbf{1}[a = a_{t-1}]
$$

That looks like a sequential dependency — you can't compute $P(a_t)$ without knowing $a_{t-1}$. So we tried to parallelise the policy phase with [DEER](https://arxiv.org/abs/2309.12252) (Lim, Schoenholz & Sussillo, 2023), which extends parallel scan to non-linear recurrences via Newton iteration on the sequential dynamics.

**DEER was a NO-GO. Three reasons that generalise:**

1. **There was no actual recurrence.** During *likelihood evaluation*, actions are observed data. $a_{t-1}$ is just `actions[t-1]` — a known integer, fully determined by the dataset. The "carry" in our `lax.scan` was tracking something that *the data already tells us*. It was a phantom dependency: an implementation artifact, not a mathematical one.
2. **Discrete state breaks DEER's premise.** DEER linearises continuous trajectories around an initial guess. Our perseveration carry is a categorical action index $\in \{0, 1, 2\}$. Linearising a discrete variable is mathematically ill-defined, and even if you smooth it, Newton iterations don't converge meaningfully on integer-valued fixed points.
3. **The horizon is too short.** For $D = 1$ (one carry variable) and $T = 100$, DEER's per-iteration overhead of an inner Jacobian solve dominates the depth saving. DEER pays off when $D$ is small *and* $T$ is in the thousands or more. Our regime sits below the break-even point.

The fix was to **precompute the perseveration array from observed data, once, before MCMC** — a parameter-independent operation amortised over thousands of likelihood evaluations:

```python
def precompute_last_action_global(actions, mask):
    """For each trial t, returns the most recent valid action before t.
    Parameter-independent — call once before MCMC, not inside the likelihood."""
    def step(last_act, inputs):
        action, valid = inputs
        out = last_act
        new_last_act = jnp.where(valid, action, last_act).astype(jnp.int32)
        return new_last_act, out

    _, arr = lax.scan(step, jnp.array(-1, jnp.int32), (actions, mask))
    return arr
```

Yes, this is itself a `lax.scan` — but it runs **once per dataset**, not on every MCMC gradient evaluation. The likelihood now reads `last_action[t]` from a precomputed array and the entire policy step becomes a vectorised broadcast:

```python
# Phase 20 vectorised policy — all T trials at once, no scan
trial_idx = jnp.arange(T)
q_vals  = q_traj[trial_idx, stimuli]               # (T, A)
wm_vals = wm_traj[trial_idx, stimuli]              # (T, A)

omega = rho * jnp.minimum(1.0, capacity / set_sizes)            # (T,)
base  = omega[:, None] * wm_vals + (1 - omega[:, None]) * softmax(beta * q_vals)
probs = epsilon / num_actions + (1 - epsilon) * base

# Perseveration: vectorised .at[].add on all T trials simultaneously
has_prev   = (last_actions >= 0)
kappa_arr  = jnp.where(has_prev, kappa, 0.0)
probs      = probs.at[trial_idx, last_actions].add(kappa_arr)
probs      = probs / probs.sum(axis=-1, keepdims=True)

log_probs = jnp.log(probs[trial_idx, actions] + 1e-8) * mask
nll       = -jnp.sum(log_probs)
```

> [!tip] Generalised lesson
> Before reaching for DEER / Newton-on-trajectory / fixed-point parallelisation, audit your "sequential dependency" carefully. If the carry is a function of *observed data only*, it's a phantom recurrence and the right answer is precomputation + vectorisation, not non-linear parallelisation.

---

## 6. Where It Scaled — and Where It Didn't

> [!important] Headline finding
> On CPU at our problem size ($T=100$ trials, $S=6$ stimuli, $A=3$ actions, 17 blocks), Phase 19 associative scan was **0.26× speedup (= 3.85× slowdown)** vs. the sequential `lax.scan` reference.

Why? At small T, **work matters more than depth**. Associative scan does roughly $2T$ multiplications (the up-sweep + down-sweep tree), versus $T$ for a sequential scan. The depth advantage ($\log_2 100 \approx 7$ vs. $100$ sequential steps) only helps when you have parallel hardware *and* the per-step cost is small enough that the constant factor dominates. On CPU, neither holds.

### What we'd expect on different hardware / problem sizes

| Regime | T | Hardware | Predicted outcome |
|---|---|---|---|
| RLWM (this work) | 100 | CPU | **Slowdown** (3-4×). Confirmed. |
| RLWM | 100 | A100 GPU | **~Break-even or mild speedup**. Phase 19 plan estimated 1.5-3× but the empirical GPU benchmark hasn't been run; planned but never executed because the *real* MCMC bottleneck shifted (see below). |
| Long-horizon RL (e.g. Atari, language-model rollouts) | 1k–100k | GPU/TPU | **Strong speedup** (10×+). This is the regime [PaMoRL](https://arxiv.org/abs/2402.05290) and Mamba-style SSMs were designed for. |
| Hierarchical MCMC over chains | — | GPU | **Different axis — vmap over chains, not scan over trials.** This is what actually unblocked our cluster runs (see [[CLUSTER_GPU_LESSONS]]). |

### What actually unblocked MCMC throughput in our project

The real win for the v4.0 hierarchical Bayesian pipeline wasn't the associative scan over $T$. It was:

1. **Phase 20's vectorised policy + precomputed perseveration** — eliminated the $O(T)$ Phase-2 sequential scan entirely, dropping the per-likelihood-eval cost by ~30% on CPU and roughly the same on GPU.
2. **`pmap` / `chain_method='parallel'` over MCMC chains, not scan over trials.** A 4-chain NUTS run on a 4-GPU A100 node parallelises *across chains*, which is far higher-leverage than parallelising within a single chain's likelihood eval. See [[CLUSTER_GPU_LESSONS]] for the empirical numbers.
3. **`vmap` over participants in the hierarchical model.** With $N=178$ participants in the cohort, the per-likelihood batch axis is the dominant source of GPU utilisation, not the trial axis.

> [!note] The deeper lesson
> "Parallelise the longest dimension" is glib but useful. For RLWM the ranking is: **chains (4) ≪ participants (178) ≫ blocks (17) ≫ trials (100)**. We initially assumed trials were the axis to crack because that's the *recurrent* axis — but recurrent ≠ bottleneck. Always profile before parallelising; the axis that "looks" hardest mathematically often isn't the wall-clock blocker.

---

## 7. Other Scaling Scenarios Worth Considering

The associative-scan + AR(1) toolkit transfers cleanly to several adjacent settings. If you're working in any of these regimes, the techniques in Sections 2-3 of this doc (and especially the [Sasha Rush "Annotated S4"](https://srush.github.io/annotated-s4/) tutorial for the broader SSM picture) will pay off.

**1. Long-horizon RL with TD(λ) eligibility traces.** The trace recurrence $e_t = \gamma\lambda e_{t-1} + \phi_t$ is exactly AR(1) with constant coefficients. PaMoRL parallelises this for multi-step return estimation in deep RL ([Parisotto et al., 2024](https://arxiv.org/abs/2402.05290)). The speedup is real when episode length runs into the thousands.

**2. Linear-Gaussian state-space models / Kalman filters.** [Särkkä & García-Fernández (2021)](https://arxiv.org/abs/2006.04369) showed that the Kalman forward and smoother passes are AR(1) over (mean, covariance) and admit $O(\log T)$ parallel scan. JAX implementations (e.g., `dynamax`) use exactly this pattern. If your generative model has linear-Gaussian dynamics, you get the speedup essentially for free.

**3. Selective state-space models (Mamba, S5, Hyena).** [S4](https://arxiv.org/abs/2111.00396) and [Mamba](https://arxiv.org/abs/2312.00752) made the SSM-as-parallel-scan idea mainstream for sequence modelling. The Mamba selective scan composes the same AR(1) operator, just with diagonal $A_t$ matrices instead of scalars. Anyone training long-context language models with these architectures is using the same primitive in Section 3.

**4. Cumulative discounted sums and exponential moving averages.** Anything of the form $y_t = \beta y_{t-1} + (1-\beta) x_t$ is a special case ($a_t = \beta$ constant, $b_t = (1-\beta) x_t$). EMA-style normalisers, exponential-decay attention biases, and discounted return calculations all fit this template.

**5. *When NOT to reach for associative scan*:**

- **$T < ~256$ on CPU.** Constant factors win.
- **Genuinely non-linear recurrences with long horizons.** DEER ([Lim et al., 2023](https://arxiv.org/abs/2309.12252)) is the right tool here — but only when $D$ (state dim) is small relative to $T$, and only when your state is continuous and differentiable.
- **Discrete-state recurrences** (HMM Viterbi, beam search). The associative operator for max-product semirings exists but `jax.lax.associative_scan` doesn't accept arbitrary semirings; you'd need a custom XLA lowering.
- **Recurrences whose "carry" is a function of observed data.** Precompute and `vmap` instead — that's the [[#5. The Phantom Recurrence — and Why Phase 20 Was the Real Win|Phase 20 lesson]].

---

## 8. Pulling It All Together: A Decision Checklist

Use this checklist when you're staring at a `lax.scan` and wondering whether to parallelise it.

> [!check] Step 1 — Audit the carry
> What variables are actually in the scan's carry? For each one:
> - Is it a function only of *observed* data (actions, rewards, stimuli)? → **Precompute it once outside MCMC**, eliminate it from the carry.
> - Is it a deterministic function of latent parameters via a *linear* recurrence? → Continue to Step 2.
> - Is it a non-linear function of latent state? → Skip to Step 5.

> [!check] Step 2 — Try to express the linear recurrence as AR(1)
> Can you write the update as $x_t = a_t x_{t-1} + b_t$, possibly per-channel?
> - Q-learning update: yes, $a_t = 1-\alpha$, $b_t = \alpha r$. ✓
> - Soft WM decay: yes, $a_t = 1-\phi$, $b_t = \phi \mathrm{WM}_0$. ✓
> - Hard overwrite at trial $t$: yes, encoded as $(a_t, b_t) = (0, r_t)$ — multiplicative reset. ✓
> - Categorical / discrete update: usually no — see Step 5.

> [!check] Step 3 — Are the coefficients data-dependent?
> If $a_t, b_t$ depend on the *latent state* you're solving for, you're outside pure AR(1). But:
> - If they only depend on *observed* data (rewards, stimuli) — fully linear, scan works. ✓
> - If the dependency is approximately on observed data (our $\alpha$ approximation in [[#2.2 The reward-based α approximation|§2.2]]) — quantify the error; if $< 10^{-3}$ relative on your typical parameter range, often acceptable.

> [!check] Step 4 — Check problem size
> - $T \gtrsim 1000$ on GPU → scan likely wins.
> - $T \lesssim 200$ on CPU → scan likely *loses*. Use sequential and vmap over a different axis (chains, participants, batches).
> - In doubt: write both, benchmark, decide. Our [`tests/scientific/benchmark_parallel_scan.py`](../tests/scientific/benchmark_parallel_scan.py) is a transferable harness.

> [!check] Step 5 — If the recurrence is genuinely non-linear
> - Continuous state, differentiable dynamics, $T \gtrsim 1000$ → consider [DEER](https://arxiv.org/abs/2309.12252).
> - Discrete state or short horizon → DEER won't help. Look for other axes to parallelise (chains, batches, participants).
> - Or: ask whether the non-linearity is *necessary* in the likelihood path. We thought ours was; it wasn't. (See [[#5. The Phantom Recurrence — and Why Phase 20 Was the Real Win|§5]].)

---

## 9. References

### Methods papers
- Senta, F. et al. (2025). *RLWM with asymmetric learning rates and capacity constraints*. PLoS Comp. Biol. 21(9): e1012872. [doi:10.1371/journal.pcbi.1012872](https://doi.org/10.1371/journal.pcbi.1012872)
- Blelloch, G. E. (1990). *Prefix sums and their applications*. Tech. Report CMU-CS-90-190. [PDF](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)
- Särkkä, S. & García-Fernández, Á. F. (2021). *Temporal Parallelization of Bayesian Smoothers*. IEEE Trans. Auto. Control. [arXiv:2006.04369](https://arxiv.org/abs/2006.04369)
- Gu, A., Goel, K., & Ré, C. (2022). *Efficiently Modeling Long Sequences with Structured State Spaces (S4)*. ICLR. [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)
- Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- Lim, Y. H., Schoenholz, S. S., & Sussillo, D. (2023). *Parallelizing Non-Linear Sequential Models over the Sequence Length (DEER)*. [arXiv:2309.12252](https://arxiv.org/abs/2309.12252)
- Parisotto, E. et al. (2024). *PaMoRL: Parallelizing Reinforcement Learning*. [arXiv:2402.05290](https://arxiv.org/abs/2402.05290)

### Tutorials and blog posts
- Rush, S. *The Annotated S4*. [srush.github.io/annotated-s4](https://srush.github.io/annotated-s4/) — best line-by-line walk-through of the SSM-as-scan picture.
- JAX docs: [`jax.lax.associative_scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.associative_scan.html)

### Local artifacts in this repo
- [[PARALLEL_SCAN_LIKELIHOOD]] — implementation guide (API contract, encoding tables, alpha-approximation bounds)
- [[legacy/DEER_NONLINEAR_PARALLELIZATION]] — the original DEER NO-GO research document
- [[CLUSTER_GPU_LESSONS]] — where chain-parallelism (vs. trial-parallelism) actually paid off
- [`src/rlwm/fitting/core.py`](../src/rlwm/fitting/core.py) — `affine_scan`, `associative_scan_q_update`, `associative_scan_wm_update`, the precompute helpers
- [`tests/scientific/benchmark_parallel_scan.py`](../tests/scientific/benchmark_parallel_scan.py) — transferable benchmark harness (sequential vs. pscan, all 6 models)
- [`tests/integration/test_pscan_likelihoods.py`](../tests/integration/test_pscan_likelihoods.py) — agreement tests vs. sequential reference (< 1e-5 typical, < 1e-3 extreme)
