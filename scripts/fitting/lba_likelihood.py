"""JAX LBA likelihood for M4 (joint choice+RT).

Brown & Heathcote (2008) analytic density, verified against
Fleming (2012) MATLAB LBA_tpdf.m and McDougle & Collins (2021).

IMPORTANT: This module enables float64 globally via jax_enable_x64.
Import this module BEFORE jax_likelihoods.py if both are needed,
or import it in the M4-specific code path only.
"""
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.scipy.stats as jss
from jax import lax

FIXED_S = 0.1    # Within-trial noise (fixed; McDougle & Collins 2021)
NUM_ACTIONS = 3


# =============================================================================
# LBA DENSITY, CDF, AND SURVIVOR FUNCTIONS
# =============================================================================

def lba_pdf(t, b, A, v_i, s=FIXED_S):
    """Single-accumulator LBA defective PDF.

    Brown & Heathcote (2008) analytic density, verified from
    Fleming (2012) MATLAB LBA_tpdf.m:
      g = (b - A - t * v_i) / (t * s)
      h = (b - t * v_i) / (t * s)
      f = (-v_i * Phi(g) + s * phi(g) + v_i * Phi(h) - s * phi(h)) / A

    Parameters
    ----------
    t : float or array
        Adjusted time (RT - t0) in seconds. Must be positive.
    b : float
        Decision threshold (same units as A). Must satisfy b > A.
    A : float
        Maximum starting point (uniform distribution on [0, A]).
    v_i : float
        Drift rate for this accumulator (in threshold units per second).
    s : float
        Within-trial noise standard deviation (fixed at FIXED_S = 0.1).

    Returns
    -------
    float or array
        Defective PDF value, clamped to >= 1e-300 to prevent log(0).
        All outputs are float64.
    """
    t = jnp.asarray(t, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    A = jnp.asarray(A, dtype=jnp.float64)
    v_i = jnp.asarray(v_i, dtype=jnp.float64)
    s = jnp.asarray(s, dtype=jnp.float64)

    g = (b - A - t * v_i) / (t * s)
    h = (b - t * v_i) / (t * s)
    f = (-v_i * jss.norm.cdf(g) + s * jss.norm.pdf(g)
         + v_i * jss.norm.cdf(h) - s * jss.norm.pdf(h)) / A
    return jnp.maximum(f, 1e-300)


def lba_cdf(t, b, A, v_i, s=FIXED_S):
    """Single-accumulator LBA CDF.

    Brown & Heathcote (2008) Equation 3:
      F(t) = 1 + (b - A - t*v)/A * Phi(g) - (b - t*v)/A * Phi(h)
               + (t*s)/A * (phi(g) - phi(h))
    where g = (b - A - t*v)/(t*s), h = (b - t*v)/(t*s).

    Parameters
    ----------
    t : float or array
        Adjusted time (RT - t0) in seconds.
    b : float
        Decision threshold.
    A : float
        Maximum starting point.
    v_i : float
        Drift rate.
    s : float
        Within-trial noise standard deviation.

    Returns
    -------
    float or array
        CDF value clipped to [0, 1]. All outputs are float64.
    """
    t = jnp.asarray(t, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    A = jnp.asarray(A, dtype=jnp.float64)
    v_i = jnp.asarray(v_i, dtype=jnp.float64)
    s = jnp.asarray(s, dtype=jnp.float64)

    g = (b - A - t * v_i) / (t * s)
    h = (b - t * v_i) / (t * s)
    F = (1.0
         + (b - A - t * v_i) / A * jss.norm.cdf(g)
         - (b - t * v_i) / A * jss.norm.cdf(h)
         + (t * s) / A * (jss.norm.pdf(g) - jss.norm.pdf(h)))
    return jnp.clip(F, 0.0, 1.0)


def lba_sf(t, b, A, v_i, s=FIXED_S):
    """Single-accumulator LBA survivor function S_i(t) = 1 - F_i(t).

    Parameters
    ----------
    t : float or array
        Adjusted time (RT - t0) in seconds.
    b : float
        Decision threshold.
    A : float
        Maximum starting point.
    v_i : float
        Drift rate.
    s : float
        Within-trial noise standard deviation.

    Returns
    -------
    float or array
        Survivor value clamped to >= 1e-300. All outputs are float64.
    """
    return jnp.maximum(1.0 - lba_cdf(t, b, A, v_i, s), 1e-300)


def lba_log_sf(t, b, A, v_i, s=FIXED_S):
    """Log-survivor function for numerical stability.

    Uses clamp-then-log pattern since LBA survivor is a composite of
    multiple normal CDF/PDF terms, not a single normal survivor.
    (Cannot use jss.norm.logsf directly here.)

    Parameters
    ----------
    t : float or array
        Adjusted time (RT - t0) in seconds.
    b : float
        Decision threshold.
    A : float
        Maximum starting point.
    v_i : float
        Drift rate.
    s : float
        Within-trial noise standard deviation.

    Returns
    -------
    float or array
        log(S_i(t)), with clamp to avoid log(0). All outputs are float64.
    """
    return jnp.log(jnp.maximum(1.0 - lba_cdf(t, b, A, v_i, s), 1e-300))


# =============================================================================
# JOINT CHOICE+RT LOG-LIKELIHOOD
# =============================================================================

def lba_joint_log_lik(t_star, chosen, b, A, v_all, s=FIXED_S):
    """Per-trial joint log-likelihood for LBA race model.

    McDougle & Collins (2021) Equation 9:
      log P(choice=i, RT=t) = log f_i(t*) + sum_{j != i} log S_j(t*)
    where t* = RT - t0 (adjusted time, must be positive).

    Uses jax.vmap for JAX/JIT compatibility (no Python for loops).

    Parameters
    ----------
    t_star : float
        Adjusted RT in seconds (RT - t0). Must be > 0.
    chosen : int
        Index of the chosen accumulator (0 to NUM_ACTIONS-1).
    b : float
        Decision threshold.
    A : float
        Maximum starting point.
    v_all : array, shape (NUM_ACTIONS,)
        Drift rates for each accumulator.
    s : float
        Within-trial noise standard deviation.

    Returns
    -------
    float
        Scalar log-likelihood for this trial. Will be negative (log density).
        Float64 precision.
    """
    t_star = jnp.asarray(t_star, dtype=jnp.float64)
    v_all = jnp.asarray(v_all, dtype=jnp.float64)

    # Compute PDF and SF for all accumulators using vmap (no Python loops)
    f_all = jax.vmap(lambda vi: lba_pdf(t_star, b, A, vi, s))(v_all)
    sf_all = jax.vmap(lambda vi: lba_sf(t_star, b, A, vi, s))(v_all)

    # Log density of chosen accumulator
    log_f_chosen = jnp.log(jnp.maximum(f_all[chosen], 1e-300))

    # Log survivor of ALL accumulators (then subtract chosen's contribution)
    # to get sum_{j != chosen} log S_j
    log_sf_sum = jnp.sum(jnp.log(jnp.maximum(sf_all, 1e-300)))
    log_sf_chosen = jnp.log(jnp.maximum(sf_all[chosen], 1e-300))

    # Joint: f_chosen * prod_{j != chosen} S_j
    return log_f_chosen + log_sf_sum - log_sf_chosen


# =============================================================================
# RT PREPROCESSING UTILITIES
# =============================================================================

def preprocess_rt_block(rt_ms, min_rt_ms=150.0, max_rt_ms=2000.0):
    """Filter RT outliers and convert to seconds.

    Parameters
    ----------
    rt_ms : array-like
        Reaction times in milliseconds.
    min_rt_ms : float
        Minimum RT threshold (anticipatory response filter).
        Default 150ms matches McDougle & Collins (2021).
    max_rt_ms : float
        Maximum RT threshold (outlier filter).
        Default 2000ms; ~0.006% of trials in this dataset.

    Returns
    -------
    rt_sec : ndarray
        RTs converted to seconds (float64).
    valid : ndarray (bool)
        Mask: True for trials within [min_rt_ms, max_rt_ms].
    """
    rt_ms = jnp.asarray(rt_ms, dtype=jnp.float64)
    valid = (rt_ms >= min_rt_ms) & (rt_ms <= max_rt_ms)
    rt_sec = rt_ms / 1000.0
    return rt_sec, valid


def validate_t0_constraint(rt_sec_filtered, t0_sec):
    """Check that t0 < min(RT) after filtering.

    Standalone diagnostic utility. Not wired into the fitting loop;
    structural protection is provided by WMRL_M4_BOUNDS in mle_utils.py.
    Use this for ad-hoc checks or debugging.

    Parameters
    ----------
    rt_sec_filtered : array
        Filtered RTs in seconds (only valid trials).
    t0_sec : float
        Non-decision time in seconds.

    Raises
    ------
    ValueError
        If t0 >= min(RT_filtered), which would produce t_star <= 0.
    """
    import numpy as np
    min_rt = float(np.min(rt_sec_filtered))
    if t0_sec >= min_rt:
        raise ValueError(
            f"t0={t0_sec:.4f}s >= min(RT)={min_rt:.4f}s. "
            f"Reduce t0 upper bound or check RT filtering."
        )


# =============================================================================
# INLINE SMOKE TESTS
# =============================================================================

def test_lba_pdf_basic():
    """Smoke test: PDF is positive and finite for typical params."""
    t, b, A, v, s = 0.3, 1.0, 0.5, 2.0, 0.1
    f = lba_pdf(t, b, A, v, s)
    assert jnp.isfinite(f) and f > 0, f"lba_pdf failed: {f}"
    assert f.dtype == jnp.float64, f"Expected float64, got {f.dtype}"
    print(f"  lba_pdf(t=0.3, b=1.0, A=0.5, v=2.0, s=0.1) = {float(f):.6f} [PASS]")


def test_lba_cdf_bounds():
    """CDF should be in [0, 1]."""
    for t in [0.1, 0.3, 0.5, 1.0, 2.0]:
        F = lba_cdf(t, 1.0, 0.5, 2.0, 0.1)
        assert 0.0 <= float(F) <= 1.0, f"CDF out of range at t={t}: {F}"
    print("  lba_cdf in [0,1] for t in {0.1, 0.3, 0.5, 1.0, 2.0} [PASS]")


def test_lba_sf_complement():
    """SF = 1 - CDF."""
    t, b, A, v, s = 0.3, 1.0, 0.5, 2.0, 0.1
    sf = lba_sf(t, b, A, v, s)
    cdf = lba_cdf(t, b, A, v, s)
    diff = abs(float(sf) + float(cdf) - 1.0)
    assert diff < 1e-10, f"SF + CDF != 1: diff={diff}"
    print(f"  lba_sf + lba_cdf = 1.0 (diff={diff:.2e}) [PASS]")


def test_lba_joint_log_lik():
    """Joint log-likelihood is finite for typical LBA parameters.

    Note: Log-density CAN be positive when the defective PDF > 1 (e.g., for
    fast drift rates where most probability mass is at short times). The
    joint log-likelihood is a sum of log(defective PDF) and log(survivors).
    Finiteness is the correctness criterion; sign is not constrained.
    """
    v_all = jnp.array([3.0, 1.0, 0.5])  # drift rates for 3 accumulators
    ll = lba_joint_log_lik(0.3, 0, 1.0, 0.5, v_all, 0.1)
    assert jnp.isfinite(ll), f"Joint LL not finite: {ll}"
    # Verify chosen accumulator has highest drift (highest probability)
    v_all_slow = jnp.array([0.5, 3.0, 3.0])  # chosen=0 is slow
    ll_slow = lba_joint_log_lik(0.3, 0, 1.0, 0.5, v_all_slow, 0.1)
    assert jnp.isfinite(ll_slow), f"Joint LL not finite (slow chosen): {ll_slow}"
    # Faster chosen should have higher joint LL than slower chosen
    assert float(ll) > float(ll_slow), (
        f"Fast chosen LL={float(ll):.4f} should exceed slow chosen LL={float(ll_slow):.4f}"
    )
    print(f"  lba_joint_log_lik = {float(ll):.4f} (finite, fast>slow) [PASS]")


def test_lba_float64_dtype():
    """All outputs must be float64."""
    t, b, A, v, s = 0.3, 1.0, 0.5, 2.0, 0.1
    assert lba_pdf(t, b, A, v, s).dtype == jnp.float64
    assert lba_cdf(t, b, A, v, s).dtype == jnp.float64
    assert lba_sf(t, b, A, v, s).dtype == jnp.float64
    v_all = jnp.array([3.0, 1.0, 0.5])
    assert lba_joint_log_lik(0.3, 0, 1.0, 0.5, v_all, 0.1).dtype == jnp.float64
    print("  All outputs are float64 [PASS]")


def test_lba_negative_drift():
    """PDF should handle near-zero/negative drift gracefully (no NaN)."""
    f = lba_pdf(0.5, 1.0, 0.5, -0.1, 0.1)
    assert jnp.isfinite(f), f"NaN/Inf for negative drift: {f}"
    assert float(f) >= 0, f"Negative PDF clamped: {f}"
    print(f"  lba_pdf with v=-0.1: {float(f):.6e} (clamped, finite) [PASS]")


def test_rt_preprocessing():
    """RT filter removes outliers and converts to seconds."""
    rt_ms = jnp.array([50.0, 150.0, 500.0, 1500.0, 2500.0])
    rt_sec, valid = preprocess_rt_block(rt_ms)
    expected_valid = jnp.array([False, True, True, True, False])
    assert jnp.all(valid == expected_valid), f"Valid mask wrong: {valid}"
    assert float(rt_sec[2]) == 0.5, f"Conversion wrong: {rt_sec[2]}"
    print("  RT preprocessing: filter + conversion [PASS]")


def test_t0_validation():
    """t0 validation raises on violation."""
    import numpy as np
    try:
        validate_t0_constraint(np.array([0.15, 0.20, 0.30]), 0.16)
        print("  t0 validation: FAIL (should have raised)")
    except ValueError:
        print("  t0 validation: correctly raised for t0 > min(RT) [PASS]")
    # Should NOT raise
    validate_t0_constraint(np.array([0.15, 0.20, 0.30]), 0.10)
    print("  t0 validation: correctly passed for t0 < min(RT) [PASS]")


# =============================================================================
# M4 BLOCK AND MULTIBLOCK LIKELIHOOD FUNCTIONS
# =============================================================================
# M4 reuses M3 learning dynamics (Q, WM, omega, kappa perseveration) but
# replaces the softmax log-probability with LBA joint choice+RT density.
# No epsilon parameter. Drift rates: v_i = v_scale * pi_hybrid.

def wmrl_m4_block_likelihood(
    stimuli: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    set_sizes: jnp.ndarray,
    rts: jnp.ndarray,
    alpha_pos: float,
    alpha_neg: float,
    phi: float,
    rho: float,
    capacity: float,
    kappa: float,
    v_scale: float,
    A: float,
    b: float,
    t0: float,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    mask: jnp.ndarray = None,
) -> float:
    """Compute NLL for WM-RL M4 (LBA decision) on a SINGLE BLOCK.

    M4 extends M3 by replacing the softmax choice probability with a Linear
    Ballistic Accumulator (LBA) that produces a joint choice+RT density.
    Learning dynamics (Q, WM, omega, kappa perseveration) are IDENTICAL to M3.
    No epsilon parameter -- start-point variability A models undirected exploration.

    Parameters
    ----------
    stimuli : array, shape (n_trials,)
        Stimulus sequence (int32).
    actions : array, shape (n_trials,)
        Action sequence (int32).
    rewards : array, shape (n_trials,)
        Reward sequence (float64).
    set_sizes : array, shape (n_trials,)
        Set size for adaptive weighting (int32).
    rts : array, shape (n_trials,)
        Reaction times in seconds (float64). Outliers already filtered by mask.
    alpha_pos : float
        RL learning rate for positive prediction error.
    alpha_neg : float
        RL learning rate for negative prediction error.
    phi : float
        WM decay rate (0-1).
    rho : float
        Base WM reliance (0-1).
    capacity : float
        WM capacity (for adaptive weighting).
    kappa : float
        Perseveration parameter (0-1).
    v_scale : float
        Drift rate scaling (positive); v_i = v_scale * pi_hybrid[i].
    A : float
        Maximum starting point (seconds, uniform on [0, A]).
    b : float
        Decision threshold (seconds). Caller ensures b > A via b = A + delta.
    t0 : float
        Non-decision time (seconds).
    num_stimuli : int
        Number of possible stimuli (default 6).
    num_actions : int
        Number of possible actions (default 3).
    q_init : float
        Initial Q-values (default 0.5).
    wm_init : float
        Initial WM values (default 1/nA = uniform).
    mask : array, shape (n_trials,), optional
        1.0 for real trials, 0.0 for padding/RT-outlier-filtered trials.

    Returns
    -------
    float
        Negative log-likelihood (NLL) for this block. Float64.
    """
    FIXED_BETA = 50.0

    # Initialize carry: (Q, WM, WM_baseline, log_lik_accum, last_action)
    Q_init_mat  = jnp.full((num_stimuli, num_actions), q_init,  dtype=jnp.float64)
    WM_init_mat = jnp.full((num_stimuli, num_actions), wm_init, dtype=jnp.float64)
    WM_0        = jnp.full((num_stimuli, num_actions), wm_init, dtype=jnp.float64)
    init_carry  = (Q_init_mat, WM_init_mat, WM_0, jnp.float64(0.0), jnp.int32(-1))

    if mask is None:
        mask = jnp.ones(stimuli.shape[0], dtype=jnp.float64)

    scan_inputs = (stimuli, actions, rewards, set_sizes, rts, mask)

    def step_fn(carry, inputs):
        Q_table, WM_table, WM_baseline, log_lik_accum, last_action = carry
        stimulus, action, reward, set_size, rt, valid = inputs

        # ------------------------------------------------------------------
        # 1. DECAY WM toward baseline
        # ------------------------------------------------------------------
        WM_decayed = (1.0 - phi) * WM_table + phi * WM_baseline

        # ------------------------------------------------------------------
        # 2. COMPUTE HYBRID POLICY (identical to M3, no epsilon)
        # ------------------------------------------------------------------
        q_vals  = Q_table[stimulus].astype(jnp.float64)
        wm_vals = WM_decayed[stimulus].astype(jnp.float64)

        # Softmax (beta=50) for RL and WM paths
        q_scaled  = FIXED_BETA * (q_vals  - jnp.max(q_vals))
        wm_scaled = FIXED_BETA * (wm_vals - jnp.max(wm_vals))
        rl_probs  = jnp.exp(q_scaled)  / jnp.sum(jnp.exp(q_scaled))
        wm_probs  = jnp.exp(wm_scaled) / jnp.sum(jnp.exp(wm_scaled))

        # Adaptive weighting: omega = rho * min(1, capacity/set_size)
        omega = rho * jnp.minimum(1.0, capacity / set_size.astype(jnp.float64))
        hybrid_probs = omega * wm_probs + (1.0 - omega) * rl_probs
        hybrid_probs = hybrid_probs / jnp.sum(hybrid_probs)  # renormalize

        # Perseveration kernel (M3 pattern, effective-weight gating)
        has_prev    = last_action >= 0
        Ck          = jnp.eye(num_actions, dtype=jnp.float64)[jnp.maximum(last_action, 0)]
        eff_kappa   = jnp.where(has_prev, kappa, jnp.float64(0.0))
        pi_hybrid   = (1.0 - eff_kappa) * hybrid_probs + eff_kappa * Ck

        # ------------------------------------------------------------------
        # 3. M4 DECISION: LBA drift rates from hybrid policy (no epsilon)
        # ------------------------------------------------------------------
        v_all = jnp.asarray(v_scale, dtype=jnp.float64) * pi_hybrid  # (num_actions,)

        # Adjusted time: t* = RT - t0, clamped for numerical safety
        t_star = jnp.maximum(rt - jnp.asarray(t0, dtype=jnp.float64), jnp.float64(1e-6))

        # Joint LBA log-likelihood: log f_{chosen}(t*) + sum_{j != chosen} log S_j(t*)
        log_prob = lba_joint_log_lik(t_star, action, b, A, v_all, FIXED_S)

        # Mask: padding/RT-outlier trials contribute 0 to NLL
        log_prob_masked = jnp.where(valid, log_prob, jnp.float64(0.0))

        # ------------------------------------------------------------------
        # 4. UPDATE WM: immediate overwrite (masked)
        # ------------------------------------------------------------------
        wm_current = WM_decayed[stimulus, action]
        WM_updated = WM_decayed.at[stimulus, action].set(
            jnp.where(valid, reward.astype(jnp.float64), wm_current)
        )

        # ------------------------------------------------------------------
        # 5. UPDATE Q-TABLE: asymmetric delta rule (masked)
        # ------------------------------------------------------------------
        q_current = Q_table[stimulus, action]
        delta     = reward.astype(jnp.float64) - q_current
        alpha_lr  = jnp.where(delta > 0, alpha_pos, alpha_neg)
        q_updated = q_current + alpha_lr * delta
        Q_updated = Q_table.at[stimulus, action].set(
            jnp.where(valid, q_updated, q_current)
        )

        # Accumulate log-likelihood
        log_lik_new = log_lik_accum + log_prob_masked

        # Update last_action only for valid trials
        new_last_action = jnp.where(valid, action, last_action).astype(jnp.int32)

        return (Q_updated, WM_updated, WM_baseline, log_lik_new, new_last_action), log_prob_masked

    # Run scan over trials
    (_, _, _, log_lik_total, _), _ = lax.scan(step_fn, init_carry, scan_inputs)

    # Return NLL (negative log-likelihood)
    return -log_lik_total


def wmrl_m4_multiblock_likelihood(
    stimuli_blocks: list,
    actions_blocks: list,
    rewards_blocks: list,
    set_sizes_blocks: list,
    rts_blocks: list,
    alpha_pos: float,
    alpha_neg: float,
    phi: float,
    rho: float,
    capacity: float,
    kappa: float,
    v_scale: float,
    A: float,
    b: float,
    t0: float,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    masks_blocks: list = None,
) -> float:
    """Compute NLL for M4 across multiple blocks.

    Q-values, WM, and last_action reset at each block boundary (independent blocks).

    Parameters
    ----------
    stimuli_blocks, actions_blocks, rewards_blocks, set_sizes_blocks : lists
        Per-block data arrays.
    rts_blocks : list
        Per-block RT arrays in seconds (float64). Outliers filtered by mask.
    A : float
        Max starting point (seconds).
    b : float
        Decision threshold (seconds). Caller encodes b = A + delta.
    (other params same as wmrl_m4_block_likelihood)

    Returns
    -------
    float
        Total NLL summed across blocks.
    """
    num_blocks = len(stimuli_blocks)

    if masks_blocks is None:
        masks_blocks = [None] * num_blocks

    total_nll = 0.0
    for stim, act, rew, ss, rt, mask in zip(
        stimuli_blocks, actions_blocks, rewards_blocks,
        set_sizes_blocks, rts_blocks, masks_blocks
    ):
        block_nll = wmrl_m4_block_likelihood(
            stimuli=stim, actions=act, rewards=rew, set_sizes=ss, rts=rt,
            alpha_pos=alpha_pos, alpha_neg=alpha_neg, phi=phi, rho=rho,
            capacity=capacity, kappa=kappa, v_scale=v_scale, A=A, b=b, t0=t0,
            num_stimuli=num_stimuli, num_actions=num_actions,
            q_init=q_init, wm_init=wm_init, mask=mask,
        )
        total_nll = total_nll + block_nll

    return total_nll


def wmrl_m4_multiblock_likelihood_stacked(
    stimuli_stacked: jnp.ndarray,
    actions_stacked: jnp.ndarray,
    rewards_stacked: jnp.ndarray,
    set_sizes_stacked: jnp.ndarray,
    rts_stacked: jnp.ndarray,
    masks_stacked: jnp.ndarray,
    alpha_pos: float,
    alpha_neg: float,
    phi: float,
    rho: float,
    capacity: float,
    kappa: float,
    v_scale: float,
    A: float,
    b: float,
    t0: float,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
) -> float:
    """Stacked (fori_loop) version of M4 multiblock likelihood.

    Takes pre-stacked arrays of shape (MAX_BLOCKS, MAX_TRIALS_PER_BLOCK).
    Used by the GPU/JIT-compiled objective functions in fit_mle.py.

    Parameters
    ----------
    stimuli_stacked : array, shape (n_blocks, max_trials)
    actions_stacked : array, shape (n_blocks, max_trials)
    rewards_stacked : array, shape (n_blocks, max_trials)
    set_sizes_stacked : array, shape (n_blocks, max_trials)
    rts_stacked : array, shape (n_blocks, max_trials) -- float64, seconds
    masks_stacked : array, shape (n_blocks, max_trials) -- 1.0 real, 0.0 pad

    Returns
    -------
    float
        Total NLL (float64).
    """
    num_blocks = stimuli_stacked.shape[0]

    def body_fn(block_idx, total_nll):
        block_nll = wmrl_m4_block_likelihood(
            stimuli=stimuli_stacked[block_idx],
            actions=actions_stacked[block_idx],
            rewards=rewards_stacked[block_idx],
            set_sizes=set_sizes_stacked[block_idx],
            rts=rts_stacked[block_idx],
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            phi=phi,
            rho=rho,
            capacity=capacity,
            kappa=kappa,
            v_scale=v_scale,
            A=A,
            b=b,
            t0=t0,
            num_stimuli=num_stimuli,
            num_actions=num_actions,
            q_init=q_init,
            wm_init=wm_init,
            mask=masks_stacked[block_idx],
        )
        return total_nll + block_nll

    return lax.fori_loop(0, num_blocks, body_fn, jnp.float64(0.0))


# =============================================================================
# M4 INLINE SMOKE TESTS
# =============================================================================

def test_wmrl_m4_single_block():
    """Smoke test: M4 NLL is finite and positive for typical params."""
    import numpy as np

    n_trials = 10
    rng = np.random.default_rng(0)

    # Synthetic data
    stimuli  = jnp.array(rng.integers(0, 4, n_trials), dtype=jnp.int32)
    actions  = jnp.array(rng.integers(0, 3, n_trials), dtype=jnp.int32)
    rewards  = jnp.array(rng.binomial(1, 0.7, n_trials), dtype=jnp.float64)
    set_sizes = jnp.full(n_trials, 4, dtype=jnp.int32)
    rts      = jnp.array(rng.uniform(0.3, 0.8, n_trials), dtype=jnp.float64)

    nll = wmrl_m4_block_likelihood(
        stimuli=stimuli, actions=actions, rewards=rewards,
        set_sizes=set_sizes, rts=rts,
        alpha_pos=0.3, alpha_neg=0.1, phi=0.5, rho=0.8,
        capacity=4.0, kappa=0.1,
        v_scale=3.0, A=0.3, b=0.8, t0=0.1,
    )
    assert jnp.isfinite(nll), f"M4 single block NLL not finite: {nll}"
    assert float(nll) > 0, f"M4 NLL should be positive (NLL=-log-lik): {nll}"
    print(f"  test_wmrl_m4_single_block: NLL={float(nll):.4f} (finite, positive) [PASS]")


def test_wmrl_m4_multiblock():
    """Smoke test: M4 multiblock NLL is finite and positive."""
    import numpy as np

    n_trials = 12
    n_blocks = 3
    rng = np.random.default_rng(1)

    stimuli_blocks  = []
    actions_blocks  = []
    rewards_blocks  = []
    set_sizes_blocks = []
    rts_blocks      = []

    for _ in range(n_blocks):
        stimuli_blocks.append(jnp.array(rng.integers(0, 4, n_trials), dtype=jnp.int32))
        actions_blocks.append(jnp.array(rng.integers(0, 3, n_trials), dtype=jnp.int32))
        rewards_blocks.append(jnp.array(rng.binomial(1, 0.7, n_trials), dtype=jnp.float64))
        set_sizes_blocks.append(jnp.full(n_trials, 4, dtype=jnp.int32))
        rts_blocks.append(jnp.array(rng.uniform(0.3, 0.8, n_trials), dtype=jnp.float64))

    nll = wmrl_m4_multiblock_likelihood(
        stimuli_blocks, actions_blocks, rewards_blocks, set_sizes_blocks, rts_blocks,
        alpha_pos=0.3, alpha_neg=0.1, phi=0.5, rho=0.8,
        capacity=4.0, kappa=0.1,
        v_scale=3.0, A=0.3, b=0.8, t0=0.1,
    )
    assert jnp.isfinite(nll), f"M4 multiblock NLL not finite: {nll}"
    assert float(nll) > 0, f"M4 NLL should be positive: {nll}"
    print(f"  test_wmrl_m4_multiblock: NLL={float(nll):.4f} (finite, positive) [PASS]")


def test_wmrl_m4_no_epsilon():
    """Verify M4 likelihood has NO epsilon parameter in its signature."""
    import inspect
    sig = inspect.signature(wmrl_m4_block_likelihood)
    assert 'epsilon' not in sig.parameters, (
        f"M4 should NOT have epsilon parameter, but found it in signature: {list(sig.parameters.keys())}"
    )
    print(f"  test_wmrl_m4_no_epsilon: epsilon absent from signature {list(sig.parameters.keys())} [PASS]")


if __name__ == '__main__':
    print("LBA Likelihood Tests (float64)")
    print("=" * 50)
    test_lba_pdf_basic()
    test_lba_cdf_bounds()
    test_lba_sf_complement()
    test_lba_joint_log_lik()
    test_lba_float64_dtype()
    test_lba_negative_drift()
    test_rt_preprocessing()
    test_t0_validation()
    print()
    print("M4 Likelihood Tests")
    print("-" * 50)
    test_wmrl_m4_single_block()
    test_wmrl_m4_multiblock()
    test_wmrl_m4_no_epsilon()
    print("=" * 50)
    print("All LBA tests passed!")
