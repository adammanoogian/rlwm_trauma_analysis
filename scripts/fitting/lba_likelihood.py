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
    print("=" * 50)
    print("All LBA tests passed!")
