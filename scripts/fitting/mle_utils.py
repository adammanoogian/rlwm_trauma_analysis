"""
MLE Utility Functions for RLWM Model Fitting

Parameter transformations, information criteria, and helper functions
for Maximum Likelihood Estimation following Senta et al. (2025) methodology.

Supports both numpy (for result handling) and JAX (for optimization).

Extended with Hessian-based diagnostics:
- Standard errors via inverse Hessian
- Parameter correlations
- Condition number for identifiability
- Pseudo-R² for model fit quality
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats
from scipy.stats import qmc

# =============================================================================
# Parameter Bounds
# =============================================================================

# Q-Learning parameter bounds
QLEARNING_BOUNDS = {
    'alpha_pos': (0.001, 0.999),  # Positive learning rate
    'alpha_neg': (0.001, 0.999),  # Negative learning rate
    'epsilon': (0.001, 0.999),    # Noise parameter
}

# WM-RL parameter bounds
WMRL_BOUNDS = {
    'alpha_pos': (0.001, 0.999),
    'alpha_neg': (0.001, 0.999),
    'phi': (0.001, 0.999),        # WM decay rate
    'rho': (0.001, 0.999),        # Base WM reliance
    'capacity': (1.0, 7.0),       # WM capacity (K)
    'epsilon': (0.001, 0.999),
}

# WM-RL M3 parameter bounds (includes kappa perseveration)
WMRL_M3_BOUNDS = {
    'alpha_pos': (0.001, 0.999),
    'alpha_neg': (0.001, 0.999),
    'phi': (0.001, 0.999),
    'rho': (0.001, 0.999),
    'capacity': (1.0, 7.0),
    'kappa': (0.0, 1.0),      # Perseveration parameter - NOTE: 0.0 allowed (M2 equivalence)
    'epsilon': (0.001, 0.999),
}

# WM-RL M5 parameter bounds (M3 + phi_rl RL forgetting)
WMRL_M5_BOUNDS = {
    'alpha_pos': (0.001, 0.999),
    'alpha_neg': (0.001, 0.999),
    'phi': (0.001, 0.999),
    'rho': (0.001, 0.999),
    'capacity': (1.0, 7.0),
    'kappa': (0.0, 1.0),      # Perseveration parameter
    'phi_rl': (0.001, 0.999),  # RL forgetting rate (decay toward Q0=1/nA before delta-rule)
    'epsilon': (0.001, 0.999),
}

# WM-RL M6a parameter bounds (M3 with per-stimulus perseveration; kappa_s replaces kappa)
WMRL_M6A_BOUNDS = {
    'alpha_pos': (0.001, 0.999),
    'alpha_neg': (0.001, 0.999),
    'phi': (0.001, 0.999),
    'rho': (0.001, 0.999),
    'capacity': (1.0, 7.0),
    'kappa_s': (0.0, 1.0),     # Stimulus-specific perseveration (same bounds as kappa)
    'epsilon': (0.001, 0.999),
}

# WM-RL M6b parameter bounds (dual perseveration: global + stimulus-specific via stick-breaking)
WMRL_M6B_BOUNDS = {
    'alpha_pos':   (0.001, 0.999),
    'alpha_neg':   (0.001, 0.999),
    'phi':         (0.001, 0.999),
    'rho':         (0.001, 0.999),
    'capacity':    (1.0, 7.0),
    'kappa_total': (0.0, 1.0),    # Total perseveration budget (kappa + kappa_s <= 1 by construction)
    'kappa_share': (0.0, 1.0),    # Fraction allocated to global kernel; remainder goes to stim-specific
    'epsilon':     (0.001, 0.999),
}

# WM-RL M4 parameter bounds (M3 learning + LBA decision; NO epsilon)
# b = A + delta reparameterization enforced in objective functions (not here)
WMRL_M4_BOUNDS = {
    'alpha_pos':  (0.001, 0.999),
    'alpha_neg':  (0.001, 0.999),
    'phi':        (0.001, 0.999),
    'rho':        (0.001, 0.999),
    'capacity':   (1.0, 7.0),
    'kappa':      (0.0, 1.0),
    'v_scale':    (0.1, 20.0),   # Drift rate scaling (log-transform recommended)
    'A':          (0.001, 2.0),  # Max start point (seconds)
    'delta':      (0.001, 2.0),  # b - A gap; b = A + delta (decoded in objectives)
    't0':         (0.05, 0.3),   # Non-decision time (seconds); conservative upper bound
}

# Parameter names in order (for array-dict conversion)
QLEARNING_PARAMS = ['alpha_pos', 'alpha_neg', 'epsilon']
WMRL_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'epsilon']
# CRITICAL: Order must match wmrl_m3_multiblock_likelihood() signature
# Signature: alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon
WMRL_M3_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa', 'epsilon']
# CRITICAL: Order must match wmrl_m5_multiblock_likelihood() signature
# Signature: alpha_pos, alpha_neg, phi, rho, capacity, kappa, phi_rl, epsilon
WMRL_M5_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa', 'phi_rl', 'epsilon']
# CRITICAL: Order must match wmrl_m6a_multiblock_likelihood() signature
# Signature: alpha_pos, alpha_neg, phi, rho, capacity, kappa_s, epsilon
WMRL_M6A_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa_s', 'epsilon']
# CRITICAL: Order must match wmrl_m6b_multiblock_likelihood() objective decode order
# Signature (decoded): alpha_pos, alpha_neg, phi, rho, capacity, kappa_total, kappa_share, epsilon
# Objective decodes: kappa = kappa_total * kappa_share; kappa_s = kappa_total * (1 - kappa_share)
WMRL_M6B_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa_total', 'kappa_share', 'epsilon']
# CRITICAL: Order must match wmrl_m4_block_likelihood() signature
# kappa at index 5; v_scale at index 6; A at index 7; delta at index 8; t0 at index 9
# NO epsilon. b = A + delta decoded in objective functions.
WMRL_M4_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa',
                   'v_scale', 'A', 'delta', 't0']

# =============================================================================
# Parameter Transformations
# =============================================================================

def logit(p: float) -> float:
    """Transform probability p in (0,1) to unbounded space (-inf, inf)."""
    return np.log(p / (1 - p))

def inv_logit(x: float) -> float:
    """Transform unbounded x to probability space (0, 1)."""
    return 1 / (1 + np.exp(-x))

def bounded_to_unbounded(value: float, lower: float, upper: float) -> float:
    """Transform value from [lower, upper] to unbounded space."""
    # First normalize to (0, 1)
    p = (value - lower) / (upper - lower)
    # Clamp to prevent logit(0) = -inf or logit(1) = +inf
    # This handles optimizer solutions that land exactly on bounds
    p = np.clip(p, 1e-8, 1 - 1e-8)
    # Then apply logit
    return logit(p)

def unbounded_to_bounded(x: float, lower: float, upper: float) -> float:
    """Transform unbounded x to [lower, upper]."""
    # Apply inverse logit to get (0, 1)
    p = inv_logit(x)
    # Then scale to bounds
    return lower + p * (upper - lower)

# =============================================================================
# JAX-Compatible Transformations (for jaxopt optimization)
# =============================================================================

def jax_inv_logit(x):
    """JAX-compatible inverse logit transformation."""
    return 1 / (1 + jnp.exp(-x))

def jax_unbounded_to_bounded(x, lower: float, upper: float):
    """JAX-compatible unbounded to bounded transformation."""
    p = jax_inv_logit(x)
    return lower + p * (upper - lower)

def jax_unconstrained_to_params_qlearning(x: jnp.ndarray) -> tuple:
    """
    JAX-compatible parameter transformation for Q-learning.

    Returns tuple (alpha_pos, alpha_neg, epsilon) for direct use in likelihood.
    """
    bounds = QLEARNING_BOUNDS
    alpha_pos = jax_unbounded_to_bounded(x[0], *bounds['alpha_pos'])
    alpha_neg = jax_unbounded_to_bounded(x[1], *bounds['alpha_neg'])
    epsilon = jax_unbounded_to_bounded(x[2], *bounds['epsilon'])
    return alpha_pos, alpha_neg, epsilon

def jax_unconstrained_to_params_wmrl(x: jnp.ndarray) -> tuple:
    """
    JAX-compatible parameter transformation for WM-RL.

    Returns tuple (alpha_pos, alpha_neg, phi, rho, capacity, epsilon) for direct use.
    """
    bounds = WMRL_BOUNDS
    alpha_pos = jax_unbounded_to_bounded(x[0], *bounds['alpha_pos'])
    alpha_neg = jax_unbounded_to_bounded(x[1], *bounds['alpha_neg'])
    phi = jax_unbounded_to_bounded(x[2], *bounds['phi'])
    rho = jax_unbounded_to_bounded(x[3], *bounds['rho'])
    capacity = jax_unbounded_to_bounded(x[4], *bounds['capacity'])
    epsilon = jax_unbounded_to_bounded(x[5], *bounds['epsilon'])
    return alpha_pos, alpha_neg, phi, rho, capacity, epsilon

def jax_unconstrained_to_params_wmrl_m3(x: jnp.ndarray) -> tuple:
    """
    JAX-compatible parameter transformation for WM-RL M3.

    Returns tuple (alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon) for direct use.
    """
    bounds = WMRL_M3_BOUNDS
    alpha_pos = jax_unbounded_to_bounded(x[0], *bounds['alpha_pos'])
    alpha_neg = jax_unbounded_to_bounded(x[1], *bounds['alpha_neg'])
    phi = jax_unbounded_to_bounded(x[2], *bounds['phi'])
    rho = jax_unbounded_to_bounded(x[3], *bounds['rho'])
    capacity = jax_unbounded_to_bounded(x[4], *bounds['capacity'])
    kappa = jax_unbounded_to_bounded(x[5], *bounds['kappa'])
    epsilon = jax_unbounded_to_bounded(x[6], *bounds['epsilon'])
    return alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon

# =============================================================================
# JAX-Compatible Inverse Transformations (bounded → unconstrained)
# Used for transforming LHS starting points before jaxopt.LBFGS optimization
# =============================================================================

def jax_logit(p):
    """JAX-compatible logit transformation: (0,1) → (-inf, inf)."""
    return jnp.log(p / (1 - p))

def jax_bounded_to_unbounded(x, lower: float, upper: float):
    """JAX-compatible bounded to unbounded transformation."""
    p = (x - lower) / (upper - lower)
    return jax_logit(p)

def jax_bounded_to_unconstrained_qlearning(x: jnp.ndarray) -> jnp.ndarray:
    """
    Transform bounded Q-learning params to unconstrained space (JAX-compatible).

    Inverse of jax_unconstrained_to_params_qlearning.
    Input: array of shape (3,) with [alpha_pos, alpha_neg, epsilon] in bounded space.
    Output: array of shape (3,) in unconstrained space.
    """
    bounds = QLEARNING_BOUNDS
    return jnp.array([
        jax_bounded_to_unbounded(x[0], *bounds['alpha_pos']),
        jax_bounded_to_unbounded(x[1], *bounds['alpha_neg']),
        jax_bounded_to_unbounded(x[2], *bounds['epsilon']),
    ])

def jax_bounded_to_unconstrained_wmrl(x: jnp.ndarray) -> jnp.ndarray:
    """
    Transform bounded WM-RL params to unconstrained space (JAX-compatible).

    Inverse of jax_unconstrained_to_params_wmrl.
    Input: array of shape (6,) in bounded space.
    Output: array of shape (6,) in unconstrained space.
    """
    bounds = WMRL_BOUNDS
    return jnp.array([
        jax_bounded_to_unbounded(x[0], *bounds['alpha_pos']),
        jax_bounded_to_unbounded(x[1], *bounds['alpha_neg']),
        jax_bounded_to_unbounded(x[2], *bounds['phi']),
        jax_bounded_to_unbounded(x[3], *bounds['rho']),
        jax_bounded_to_unbounded(x[4], *bounds['capacity']),
        jax_bounded_to_unbounded(x[5], *bounds['epsilon']),
    ])

def jax_bounded_to_unconstrained_wmrl_m3(x: jnp.ndarray) -> jnp.ndarray:
    """
    Transform bounded WM-RL M3 params to unconstrained space (JAX-compatible).

    Inverse of jax_unconstrained_to_params_wmrl_m3.
    Input: array of shape (7,) in bounded space.
    Output: array of shape (7,) in unconstrained space.
    """
    bounds = WMRL_M3_BOUNDS
    return jnp.array([
        jax_bounded_to_unbounded(x[0], *bounds['alpha_pos']),
        jax_bounded_to_unbounded(x[1], *bounds['alpha_neg']),
        jax_bounded_to_unbounded(x[2], *bounds['phi']),
        jax_bounded_to_unbounded(x[3], *bounds['rho']),
        jax_bounded_to_unbounded(x[4], *bounds['capacity']),
        jax_bounded_to_unbounded(x[5], *bounds['kappa']),
        jax_bounded_to_unbounded(x[6], *bounds['epsilon']),
    ])

def jax_unconstrained_to_params_wmrl_m5(x: jnp.ndarray) -> tuple:
    """
    JAX-compatible parameter transformation for WM-RL M5.

    Returns tuple (alpha_pos, alpha_neg, phi, rho, capacity, kappa, phi_rl, epsilon) for direct use.
    x[0..5] same as M3. x[6] = phi_rl. x[7] = epsilon.
    """
    bounds = WMRL_M5_BOUNDS
    alpha_pos = jax_unbounded_to_bounded(x[0], *bounds['alpha_pos'])
    alpha_neg = jax_unbounded_to_bounded(x[1], *bounds['alpha_neg'])
    phi = jax_unbounded_to_bounded(x[2], *bounds['phi'])
    rho = jax_unbounded_to_bounded(x[3], *bounds['rho'])
    capacity = jax_unbounded_to_bounded(x[4], *bounds['capacity'])
    kappa = jax_unbounded_to_bounded(x[5], *bounds['kappa'])
    phi_rl = jax_unbounded_to_bounded(x[6], *bounds['phi_rl'])
    epsilon = jax_unbounded_to_bounded(x[7], *bounds['epsilon'])
    return alpha_pos, alpha_neg, phi, rho, capacity, kappa, phi_rl, epsilon

def jax_bounded_to_unconstrained_wmrl_m5(x: jnp.ndarray) -> jnp.ndarray:
    """
    Transform bounded WM-RL M5 params to unconstrained space (JAX-compatible).

    Inverse of jax_unconstrained_to_params_wmrl_m5.
    Input: array of shape (8,) in bounded space.
    Output: array of shape (8,) in unconstrained space.
    """
    bounds = WMRL_M5_BOUNDS
    return jnp.array([
        jax_bounded_to_unbounded(x[0], *bounds['alpha_pos']),
        jax_bounded_to_unbounded(x[1], *bounds['alpha_neg']),
        jax_bounded_to_unbounded(x[2], *bounds['phi']),
        jax_bounded_to_unbounded(x[3], *bounds['rho']),
        jax_bounded_to_unbounded(x[4], *bounds['capacity']),
        jax_bounded_to_unbounded(x[5], *bounds['kappa']),
        jax_bounded_to_unbounded(x[6], *bounds['phi_rl']),
        jax_bounded_to_unbounded(x[7], *bounds['epsilon']),
    ])

def jax_unconstrained_to_params_wmrl_m6a(x: jnp.ndarray) -> tuple:
    """
    JAX-compatible parameter transformation for WM-RL M6a.

    Returns tuple (alpha_pos, alpha_neg, phi, rho, capacity, kappa_s, epsilon).
    x[0..4] same as M3. x[5] = kappa_s. x[6] = epsilon.
    7 parameters total (same count as M3; kappa_s replaces kappa).
    """
    bounds = WMRL_M6A_BOUNDS
    alpha_pos = jax_unbounded_to_bounded(x[0], *bounds['alpha_pos'])
    alpha_neg = jax_unbounded_to_bounded(x[1], *bounds['alpha_neg'])
    phi       = jax_unbounded_to_bounded(x[2], *bounds['phi'])
    rho       = jax_unbounded_to_bounded(x[3], *bounds['rho'])
    capacity  = jax_unbounded_to_bounded(x[4], *bounds['capacity'])
    kappa_s   = jax_unbounded_to_bounded(x[5], *bounds['kappa_s'])
    epsilon   = jax_unbounded_to_bounded(x[6], *bounds['epsilon'])
    return alpha_pos, alpha_neg, phi, rho, capacity, kappa_s, epsilon

def jax_bounded_to_unconstrained_wmrl_m6a(x: jnp.ndarray) -> jnp.ndarray:
    """
    Transform bounded WM-RL M6a params to unconstrained space (JAX-compatible).

    Inverse of jax_unconstrained_to_params_wmrl_m6a.
    Input: array of shape (7,) in bounded space.
    Output: array of shape (7,) in unconstrained space.
    """
    bounds = WMRL_M6A_BOUNDS
    return jnp.array([
        jax_bounded_to_unbounded(x[0], *bounds['alpha_pos']),
        jax_bounded_to_unbounded(x[1], *bounds['alpha_neg']),
        jax_bounded_to_unbounded(x[2], *bounds['phi']),
        jax_bounded_to_unbounded(x[3], *bounds['rho']),
        jax_bounded_to_unbounded(x[4], *bounds['capacity']),
        jax_bounded_to_unbounded(x[5], *bounds['kappa_s']),
        jax_bounded_to_unbounded(x[6], *bounds['epsilon']),
    ])

def jax_unconstrained_to_params_wmrl_m6b(x: jnp.ndarray) -> tuple:
    """
    JAX-compatible parameter transformation for WM-RL M6b.

    Returns tuple (alpha_pos, alpha_neg, phi, rho, capacity, kappa_total, kappa_share, epsilon).
    x[0..4] same as M3/M6a. x[5] = kappa_total (total perseveration budget).
    x[6] = kappa_share (fraction allocated to global kernel). x[7] = epsilon.
    8 parameters total.

    CRITICAL: Returns kappa_total and kappa_share, NOT decoded kappa/kappa_s.
    The stick-breaking decode (kappa = kappa_total * kappa_share) happens in
    objective functions only, not here.
    """
    bounds = WMRL_M6B_BOUNDS
    alpha_pos   = jax_unbounded_to_bounded(x[0], *bounds['alpha_pos'])
    alpha_neg   = jax_unbounded_to_bounded(x[1], *bounds['alpha_neg'])
    phi         = jax_unbounded_to_bounded(x[2], *bounds['phi'])
    rho         = jax_unbounded_to_bounded(x[3], *bounds['rho'])
    capacity    = jax_unbounded_to_bounded(x[4], *bounds['capacity'])
    kappa_total = jax_unbounded_to_bounded(x[5], *bounds['kappa_total'])
    kappa_share = jax_unbounded_to_bounded(x[6], *bounds['kappa_share'])
    epsilon     = jax_unbounded_to_bounded(x[7], *bounds['epsilon'])
    return alpha_pos, alpha_neg, phi, rho, capacity, kappa_total, kappa_share, epsilon

def jax_bounded_to_unconstrained_wmrl_m6b(x: jnp.ndarray) -> jnp.ndarray:
    """
    Transform bounded WM-RL M6b params to unconstrained space (JAX-compatible).

    Inverse of jax_unconstrained_to_params_wmrl_m6b.
    Input: array of shape (8,) in bounded space [kappa_total, kappa_share at index 5, 6].
    Output: array of shape (8,) in unconstrained space.
    """
    bounds = WMRL_M6B_BOUNDS
    return jnp.array([
        jax_bounded_to_unbounded(x[0], *bounds['alpha_pos']),
        jax_bounded_to_unbounded(x[1], *bounds['alpha_neg']),
        jax_bounded_to_unbounded(x[2], *bounds['phi']),
        jax_bounded_to_unbounded(x[3], *bounds['rho']),
        jax_bounded_to_unbounded(x[4], *bounds['capacity']),
        jax_bounded_to_unbounded(x[5], *bounds['kappa_total']),
        jax_bounded_to_unbounded(x[6], *bounds['kappa_share']),
        jax_bounded_to_unbounded(x[7], *bounds['epsilon']),
    ])

def jax_unconstrained_to_params_wmrl_m4(x: jnp.ndarray) -> tuple:
    """
    JAX-compatible parameter transformation for WM-RL M4.

    Returns tuple (alpha_pos, alpha_neg, phi, rho, capacity, kappa, v_scale, A, delta, t0).
    x[0..4] same as M3. x[5] = kappa. x[6] = v_scale. x[7] = A. x[8] = delta. x[9] = t0.
    10 parameters total. NO epsilon.

    CRITICAL: Returns A and delta (NOT decoded b). The decode b = A + delta happens
    in objective functions only, not here.
    """
    bounds = WMRL_M4_BOUNDS
    alpha_pos = jax_unbounded_to_bounded(x[0], *bounds['alpha_pos'])
    alpha_neg = jax_unbounded_to_bounded(x[1], *bounds['alpha_neg'])
    phi       = jax_unbounded_to_bounded(x[2], *bounds['phi'])
    rho       = jax_unbounded_to_bounded(x[3], *bounds['rho'])
    capacity  = jax_unbounded_to_bounded(x[4], *bounds['capacity'])
    kappa     = jax_unbounded_to_bounded(x[5], *bounds['kappa'])
    v_scale   = jax_unbounded_to_bounded(x[6], *bounds['v_scale'])
    A         = jax_unbounded_to_bounded(x[7], *bounds['A'])
    delta     = jax_unbounded_to_bounded(x[8], *bounds['delta'])
    t0        = jax_unbounded_to_bounded(x[9], *bounds['t0'])
    return alpha_pos, alpha_neg, phi, rho, capacity, kappa, v_scale, A, delta, t0

def jax_bounded_to_unconstrained_wmrl_m4(x: jnp.ndarray) -> jnp.ndarray:
    """
    Transform bounded WM-RL M4 params to unconstrained space (JAX-compatible).

    Inverse of jax_unconstrained_to_params_wmrl_m4.
    Input: array of shape (10,) in bounded space [kappa at 5, v_scale at 6, A at 7, delta at 8, t0 at 9].
    Output: array of shape (10,) in unconstrained space.
    """
    bounds = WMRL_M4_BOUNDS
    return jnp.array([
        jax_bounded_to_unbounded(x[0], *bounds['alpha_pos']),
        jax_bounded_to_unbounded(x[1], *bounds['alpha_neg']),
        jax_bounded_to_unbounded(x[2], *bounds['phi']),
        jax_bounded_to_unbounded(x[3], *bounds['rho']),
        jax_bounded_to_unbounded(x[4], *bounds['capacity']),
        jax_bounded_to_unbounded(x[5], *bounds['kappa']),
        jax_bounded_to_unbounded(x[6], *bounds['v_scale']),
        jax_bounded_to_unbounded(x[7], *bounds['A']),
        jax_bounded_to_unbounded(x[8], *bounds['delta']),
        jax_bounded_to_unbounded(x[9], *bounds['t0']),
    ])

def params_to_unconstrained(params: dict[str, float], model: str) -> np.ndarray:
    """
    Transform bounded parameter dict to unconstrained numpy array.

    Args:
        params: Dictionary of parameter values
        model: 'qlearning', 'wmrl', 'wmrl_m3', or 'wmrl_m5'

    Returns:
        Unconstrained parameter array
    """
    if model == 'qlearning':
        bounds = QLEARNING_BOUNDS
        param_names = QLEARNING_PARAMS
    elif model == 'wmrl':
        bounds = WMRL_BOUNDS
        param_names = WMRL_PARAMS
    elif model == 'wmrl_m3':
        bounds = WMRL_M3_BOUNDS
        param_names = WMRL_M3_PARAMS
    elif model == 'wmrl_m5':
        bounds = WMRL_M5_BOUNDS
        param_names = WMRL_M5_PARAMS
    elif model == 'wmrl_m6a':
        bounds = WMRL_M6A_BOUNDS
        param_names = WMRL_M6A_PARAMS
    elif model == 'wmrl_m6b':
        bounds = WMRL_M6B_BOUNDS
        param_names = WMRL_M6B_PARAMS
    elif model == 'wmrl_m4':
        bounds = WMRL_M4_BOUNDS
        param_names = WMRL_M4_PARAMS
    else:
        raise ValueError(f"Unknown model: {model}")

    x = []
    for name in param_names:
        lower, upper = bounds[name]
        x.append(bounded_to_unbounded(params[name], lower, upper))

    return np.array(x)

def unconstrained_to_params(x: np.ndarray, model: str) -> dict[str, float]:
    """
    Transform unconstrained array back to bounded parameter dict.

    Args:
        x: Unconstrained parameter array
        model: 'qlearning', 'wmrl', 'wmrl_m3', or 'wmrl_m5'

    Returns:
        Dictionary of bounded parameter values
    """
    if model == 'qlearning':
        bounds = QLEARNING_BOUNDS
        param_names = QLEARNING_PARAMS
    elif model == 'wmrl':
        bounds = WMRL_BOUNDS
        param_names = WMRL_PARAMS
    elif model == 'wmrl_m3':
        bounds = WMRL_M3_BOUNDS
        param_names = WMRL_M3_PARAMS
    elif model == 'wmrl_m5':
        bounds = WMRL_M5_BOUNDS
        param_names = WMRL_M5_PARAMS
    elif model == 'wmrl_m6a':
        bounds = WMRL_M6A_BOUNDS
        param_names = WMRL_M6A_PARAMS
    elif model == 'wmrl_m6b':
        bounds = WMRL_M6B_BOUNDS
        param_names = WMRL_M6B_PARAMS
    elif model == 'wmrl_m4':
        bounds = WMRL_M4_BOUNDS
        param_names = WMRL_M4_PARAMS
    else:
        raise ValueError(f"Unknown model: {model}")

    params = {}
    for i, name in enumerate(param_names):
        lower, upper = bounds[name]
        params[name] = unbounded_to_bounded(x[i], lower, upper)

    return params

def get_default_params(model: str) -> dict[str, float]:
    """Get default starting parameters for a model."""
    if model == 'qlearning':
        return {
            'alpha_pos': 0.3,
            'alpha_neg': 0.1,
            'epsilon': 0.05
        }
    elif model == 'wmrl':
        return {
            'alpha_pos': 0.3,
            'alpha_neg': 0.1,
            'phi': 0.1,
            'rho': 0.7,
            'capacity': 4.0,
            'epsilon': 0.05
        }
    elif model == 'wmrl_m3':
        return {
            'alpha_pos': 0.3,
            'alpha_neg': 0.1,
            'phi': 0.1,
            'rho': 0.7,
            'capacity': 4.0,
            'kappa': 0.0,  # Default to M2 behavior (no perseveration)
            'epsilon': 0.05
        }
    elif model == 'wmrl_m5':
        return {
            'alpha_pos': 0.3,
            'alpha_neg': 0.1,
            'phi': 0.1,
            'rho': 0.7,
            'capacity': 4.0,
            'kappa': 0.0,   # Default to no perseveration
            'phi_rl': 0.1,  # Default: match phi's starting value
            'epsilon': 0.05
        }
    elif model == 'wmrl_m6a':
        return {
            'alpha_pos': 0.3,
            'alpha_neg': 0.1,
            'phi': 0.1,
            'rho': 0.7,
            'capacity': 4.0,
            'kappa_s': 0.1,  # Default: small positive (like M3's kappa default)
            'epsilon': 0.05
        }
    elif model == 'wmrl_m6b':
        return {
            'alpha_pos': 0.3,
            'alpha_neg': 0.1,
            'phi': 0.1,
            'rho': 0.7,
            'capacity': 4.0,
            'kappa_total': 0.2,  # Moderate total perseveration budget
            'kappa_share': 0.5,  # Equal split: 0.1 global, 0.1 stim-specific
            'epsilon': 0.05
        }
    elif model == 'wmrl_m4':
        return {
            'alpha_pos': 0.3,
            'alpha_neg': 0.1,
            'phi': 0.1,
            'rho': 0.7,
            'capacity': 4.0,
            'kappa':   0.1,   # Moderate perseveration (no epsilon)
            'v_scale': 3.0,   # Typical drift rate scale
            'A':       0.3,   # Max start point (seconds)
            'delta':   0.5,   # b - A gap; b = 0.3 + 0.5 = 0.8
            't0':      0.15,  # Non-decision time (seconds)
        }
    else:
        raise ValueError(f"Unknown model: {model}")

def sample_random_start(model: str, rng: np.random.Generator) -> np.ndarray:
    """
    Sample random starting point in unconstrained space.

    Uses Normal(0, 1) which maps to roughly uniform in bounded space
    (centered around 0.5 for (0,1) bounds).

    Args:
        model: 'qlearning', 'wmrl', or 'wmrl_m3'
        rng: NumPy random generator

    Returns:
        Unconstrained parameter array
    """
    if model == 'qlearning':
        n_params = len(QLEARNING_PARAMS)
    elif model == 'wmrl':
        n_params = len(WMRL_PARAMS)
    elif model == 'wmrl_m3':
        n_params = len(WMRL_M3_PARAMS)
    elif model == 'wmrl_m5':
        n_params = len(WMRL_M5_PARAMS)
    elif model == 'wmrl_m6a':
        n_params = len(WMRL_M6A_PARAMS)
    elif model == 'wmrl_m6b':
        n_params = len(WMRL_M6B_PARAMS)
    elif model == 'wmrl_m4':
        n_params = len(WMRL_M4_PARAMS)
    else:
        raise ValueError(f"Unknown model: {model}")

    return rng.normal(0, 1.5, size=n_params)  # SD=1.5 gives reasonable spread

def sample_lhs_starts(model: str, n_starts: int, seed: int = None) -> np.ndarray:
    """
    Generate starting points using Latin Hypercube Sampling in bounded space.

    LHS ensures even coverage of parameter space by dividing each dimension
    into n_starts equal strata and sampling exactly once per stratum. This
    improves the chance of finding the global optimum compared to random
    sampling, which can leave gaps or have clusters.

    Args:
        model: 'qlearning', 'wmrl', or 'wmrl_m3'
        n_starts: Number of starting points
        seed: Random seed for reproducibility

    Returns:
        Array of shape (n_starts, n_params) in BOUNDED parameter space
    """
    if model == 'qlearning':
        bounds_dict = QLEARNING_BOUNDS
        param_names = QLEARNING_PARAMS
    elif model == 'wmrl':
        bounds_dict = WMRL_BOUNDS
        param_names = WMRL_PARAMS
    elif model == 'wmrl_m3':
        bounds_dict = WMRL_M3_BOUNDS
        param_names = WMRL_M3_PARAMS
    elif model == 'wmrl_m5':
        bounds_dict = WMRL_M5_BOUNDS
        param_names = WMRL_M5_PARAMS
    elif model == 'wmrl_m6a':
        bounds_dict = WMRL_M6A_BOUNDS
        param_names = WMRL_M6A_PARAMS
    elif model == 'wmrl_m6b':
        bounds_dict = WMRL_M6B_BOUNDS
        param_names = WMRL_M6B_PARAMS
    elif model == 'wmrl_m4':
        bounds_dict = WMRL_M4_BOUNDS
        param_names = WMRL_M4_PARAMS
    else:
        raise ValueError(f"Unknown model: {model}")

    n_params = len(param_names)

    # Generate LHS samples in [0, 1]^d
    sampler = qmc.LatinHypercube(d=n_params, seed=seed)
    samples = sampler.random(n=n_starts)

    # Scale to parameter bounds
    lower = np.array([bounds_dict[p][0] for p in param_names])
    upper = np.array([bounds_dict[p][1] for p in param_names])

    scaled_samples = qmc.scale(samples, lower, upper)

    return scaled_samples

# =============================================================================
# Information Criteria
# =============================================================================

def compute_aic(nll: float, k: int) -> float:
    """
    Compute Akaike Information Criterion.

    AIC = 2k - 2*log(L) = 2k + 2*NLL

    Args:
        nll: Negative log-likelihood at MLE
        k: Number of free parameters

    Returns:
        AIC value (lower is better)
    """
    return 2 * k + 2 * nll

def compute_bic(nll: float, k: int, n: int) -> float:
    """
    Compute Bayesian Information Criterion.

    BIC = k*log(n) - 2*log(L) = k*log(n) + 2*NLL

    Args:
        nll: Negative log-likelihood at MLE
        k: Number of free parameters
        n: Number of observations (trials)

    Returns:
        BIC value (lower is better)
    """
    return k * np.log(n) + 2 * nll

def compute_aicc(nll: float, k: int, n: int) -> float:
    """
    Compute corrected AIC (for small sample sizes).

    AICc = AIC + (2k^2 + 2k) / (n - k - 1)

    Args:
        nll: Negative log-likelihood at MLE
        k: Number of free parameters
        n: Number of observations (trials)

    Returns:
        AICc value (lower is better)
    """
    aic = compute_aic(nll, k)
    if n - k - 1 > 0:
        correction = (2 * k * k + 2 * k) / (n - k - 1)
    else:
        correction = np.inf  # Not enough data
    return aic + correction

def get_n_params(model: str) -> int:
    """Get number of free parameters for a model."""
    if model == 'qlearning':
        return 3  # alpha_pos, alpha_neg, epsilon
    elif model == 'wmrl':
        return 6  # alpha_pos, alpha_neg, phi, rho, capacity, epsilon
    elif model == 'wmrl_m3':
        return 7  # alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon
    elif model == 'wmrl_m5':
        return 8  # alpha_pos, alpha_neg, phi, rho, capacity, kappa, phi_rl, epsilon
    elif model == 'wmrl_m6a':
        return 7  # alpha_pos, alpha_neg, phi, rho, capacity, kappa_s, epsilon
    elif model == 'wmrl_m6b':
        return 8  # alpha_pos, alpha_neg, phi, rho, capacity, kappa_total, kappa_share, epsilon
    elif model == 'wmrl_m4':
        return 10  # alpha_pos, alpha_neg, phi, rho, capacity, kappa, v_scale, A, delta, t0
    else:
        raise ValueError(f"Unknown model: {model}")

# =============================================================================
# Group Statistics (Senta et al. style)
# =============================================================================

def compute_group_statistics(
    param_values: np.ndarray,
    param_name: str
) -> dict[str, float]:
    """
    Compute group-level statistics for a parameter.

    Following Senta et al. (2025): report mean +/- SEM across participants.

    Args:
        param_values: Array of parameter values across participants
        param_name: Name of the parameter

    Returns:
        Dictionary with mean, sd, se, ci_lower, ci_upper
    """
    n = len(param_values)
    mean = np.mean(param_values)
    sd = np.std(param_values, ddof=1)  # Sample SD
    se = sd / np.sqrt(n)  # Standard error of the mean

    # 95% CI using t-distribution
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_lower = mean - t_crit * se
    ci_upper = mean + t_crit * se

    return {
        'mean': mean,
        'sd': sd,
        'se': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n': n
    }

def summarize_all_parameters(
    fits_df,  # pd.DataFrame
    model: str
) -> dict[str, dict[str, float]]:
    """
    Compute group statistics for all parameters.

    Args:
        fits_df: DataFrame with individual fit results
        model: 'qlearning', 'wmrl', or 'wmrl_m3'

    Returns:
        Nested dict: {param_name: {mean, sd, se, ci_lower, ci_upper}}
    """
    if model == 'qlearning':
        param_names = QLEARNING_PARAMS
    elif model == 'wmrl':
        param_names = WMRL_PARAMS
    elif model == 'wmrl_m3':
        param_names = WMRL_M3_PARAMS
    elif model == 'wmrl_m5':
        param_names = WMRL_M5_PARAMS
    elif model == 'wmrl_m6a':
        param_names = WMRL_M6A_PARAMS
    elif model == 'wmrl_m6b':
        param_names = WMRL_M6B_PARAMS
    elif model == 'wmrl_m4':
        param_names = WMRL_M4_PARAMS
    else:
        raise ValueError(f"Unknown model: {model}")

    summary = {}
    for param in param_names:
        if param in fits_df.columns:
            values = fits_df[param].values
            summary[param] = compute_group_statistics(values, param)

    return summary

# =============================================================================
# Convergence Diagnostics
# =============================================================================

def check_convergence(
    results: list,  # list of scipy OptimizeResult (or _JaxoptResult wrappers)
    iteration_stats: list[dict] = None,
    tolerance: float = 1.0
) -> dict[str, any]:
    """
    Check convergence based on optimizer success flags.

    A participant is "converged" if EITHER:
    1. The best start's optimizer reported scipy convergence, OR
    2. Any scipy-converged start found an NLL within tolerance of the best.

    The fallback (2) handles the edge case where the best start hit maxiter
    while still making marginal progress, but other converged starts
    independently confirmed essentially the same optimum.

    Args:
        results: list of optimization results from different starts
        iteration_stats: list of per-start dicts with 'scipy_converged', etc.
                        When provided, uses the best start's scipy_converged flag.
        tolerance: NLL tolerance for counting "near best" starts (default: 1.0)

    Returns:
        Dictionary with convergence diagnostics including:
        - converged: bool (primary criterion OR fallback)
        - best_scipy_converged: bool (primary criterion only)
        - any_converged_near_best: bool (fallback criterion only)
    """
    if not results:
        return {
            'n_successful': 0,
            'n_near_best': 0,
            'best_nll': np.inf,
            'nll_spread': np.inf,
            'best_scipy_converged': False,
            'any_converged_near_best': False,
            'converged': False,
        }

    nlls = [r.fun for r in results if r.success]

    if not nlls:
        return {
            'n_successful': 0,
            'n_near_best': 0,
            'best_nll': np.inf,
            'nll_spread': np.inf,
            'best_scipy_converged': False,
            'any_converged_near_best': False,
            'converged': False,
        }

    best_nll = min(nlls)
    n_near_best = sum(1 for nll in nlls if abs(nll - best_nll) < tolerance)

    # Find the best start's index among successful results
    best_idx = next(i for i, r in enumerate(results) if r.success and r.fun == best_nll)

    # Determine scipy convergence of the best start
    if iteration_stats and best_idx < len(iteration_stats):
        best_scipy_converged = iteration_stats[best_idx].get('scipy_converged', True)
    else:
        # No iteration_stats provided — trust r.success (finite NLL)
        best_scipy_converged = True

    # Fallback: a scipy-converged start confirmed a near-identical NLL
    # This handles the edge case where the best start hit maxiter while still
    # making marginal progress, but other converged starts independently found
    # essentially the same optimum — strong evidence the solution is correct.
    any_converged_near_best = False
    if not best_scipy_converged and iteration_stats:
        for i, r in enumerate(results):
            if r.success and i < len(iteration_stats):
                is_near_best = abs(r.fun - best_nll) < tolerance
                is_scipy_conv = iteration_stats[i].get('scipy_converged', False)
                if is_near_best and is_scipy_conv:
                    any_converged_near_best = True
                    break

    return {
        'n_successful': len(nlls),
        'n_near_best': n_near_best,
        'best_nll': best_nll,
        'nll_spread': max(nlls) - min(nlls),
        'best_scipy_converged': best_scipy_converged,
        'any_converged_near_best': any_converged_near_best,
        'converged': best_scipy_converged or any_converged_near_best,
    }

def check_at_bounds(
    params: dict[str, float],
    model: str,
    tolerance: float = 0.01
) -> list[str]:
    """
    Check if any parameters are at their bounds.

    Args:
        params: Parameter dictionary
        model: 'qlearning', 'wmrl', or 'wmrl_m3'
        tolerance: Distance from bound to consider "at bound"

    Returns:
        list of parameter names that hit bounds.
    """
    if model == 'qlearning':
        bounds = QLEARNING_BOUNDS
    elif model == 'wmrl':
        bounds = WMRL_BOUNDS
    elif model == 'wmrl_m3':
        bounds = WMRL_M3_BOUNDS
    elif model == 'wmrl_m5':
        bounds = WMRL_M5_BOUNDS
    elif model == 'wmrl_m6a':
        bounds = WMRL_M6A_BOUNDS
    elif model == 'wmrl_m6b':
        bounds = WMRL_M6B_BOUNDS
    elif model == 'wmrl_m4':
        bounds = WMRL_M4_BOUNDS
    else:
        raise ValueError(f"Unknown model: {model}")

    at_bounds = []
    for name, value in params.items():
        if name in bounds:
            lower, upper = bounds[name]
            if abs(value - lower) < tolerance or abs(value - upper) < tolerance:
                at_bounds.append(name)

    return at_bounds

# =============================================================================
# Model Comparison
# =============================================================================

def compare_models_aic(
    aic_model1: float,
    aic_model2: float,
    model1_name: str = 'model1',
    model2_name: str = 'model2'
) -> dict[str, any]:
    """
    Compare two models using AIC.

    Returns interpretation following Burnham & Anderson guidelines.
    """
    delta_aic = aic_model1 - aic_model2  # Positive means model2 is better

    # Akaike weights
    min_aic = min(aic_model1, aic_model2)
    w1 = np.exp(-0.5 * (aic_model1 - min_aic))
    w2 = np.exp(-0.5 * (aic_model2 - min_aic))
    total_w = w1 + w2

    weight_model1 = w1 / total_w
    weight_model2 = w2 / total_w

    # Evidence strength
    abs_delta = abs(delta_aic)
    if abs_delta < 2:
        evidence = 'Weak (models essentially equivalent)'
    elif abs_delta < 4:
        evidence = 'Moderate'
    elif abs_delta < 10:
        evidence = 'Strong'
    else:
        evidence = 'Very strong'

    preferred = model2_name if delta_aic > 0 else model1_name

    return {
        'delta_aic': delta_aic,
        f'weight_{model1_name}': weight_model1,
        f'weight_{model2_name}': weight_model2,
        'evidence_strength': evidence,
        'preferred_model': preferred
    }

# =============================================================================
# Hessian-Based Diagnostics
# =============================================================================

def compute_pseudo_r2(nll: float, n_trials: int, n_actions: int = 3) -> float:
    """
    Compute McFadden's pseudo-R² (variance explained).

    R² = 1 - (NLL_model / NLL_null)
    where NLL_null = -n * log(1/n_actions) = n * log(n_actions)

    This gives an intuitive measure of how much better the model is than
    chance-level performance (random responding).

    Args:
        nll: Negative log-likelihood of the fitted model
        n_trials: Number of trials
        n_actions: Number of possible actions (default: 3)

    Returns:
        Pseudo-R² value (0 = chance, 1 = perfect fit)

    Example:
        >>> # Chance-level NLL for 988 trials = 988 * log(3) ≈ 1085
        >>> # If model NLL = 763, pseudo-R² ≈ 0.30 (30% variance explained)
        >>> pseudo_r2 = compute_pseudo_r2(763, 988)
    """
    nll_null = n_trials * np.log(n_actions)  # Chance-level NLL
    if nll_null == 0:
        return np.nan
    pseudo_r2 = 1 - (nll / nll_null)
    return float(pseudo_r2)

def check_gradient_norm(
    objective_fn,
    x_opt: np.ndarray,
    tolerance: float = 1e-4
) -> tuple[float, bool]:
    """
    Check gradient norm at optimum using JAX autodiff.

    At a true optimum, the gradient should be approximately zero.
    A large gradient indicates the optimizer didn't fully converge.

    Args:
        objective_fn: JAX-compatible objective function
        x_opt: Optimal parameter values (unconstrained)
        tolerance: Maximum gradient norm to consider "converged"

    Returns:
        tuple of (gradient_norm, is_converged)
    """
    try:
        # JIT-compile for faster execution (especially on GPU)
        grad_fn = jax.jit(jax.grad(objective_fn))
        x_jax = jnp.array(x_opt)
        g = grad_fn(x_jax)
        grad_norm = float(jnp.linalg.norm(g))
        is_converged = grad_norm < tolerance
        return grad_norm, is_converged
    except Exception:
        return np.nan, False

def compute_hessian_diagnostics(
    objective_fn,
    x_opt: np.ndarray,
    model: str,
    param_names: list[str] = None
) -> dict[str, Any]:
    """
    Compute Hessian-based diagnostics for MLE fit quality.

    At the MLE optimum, the Fisher Information matrix I = E[H(NLL)]
    The covariance of parameter estimates is approximately: Var(θ̂) ≈ I⁻¹
    Standard errors are the square roots of the diagonal: SE(θᵢ) = √(I⁻¹ᵢᵢ)

    This function computes:
    1. Standard errors for all parameters (in unconstrained space)
    2. Parameter correlation matrix
    3. Condition number (for identifiability assessment)

    Args:
        objective_fn: JAX-compatible objective function (returns NLL)
        x_opt: Optimal parameter values (unconstrained space)
        model: Model name ('qlearning', 'wmrl', 'wmrl_m3')
        param_names: list of parameter names (optional, will use defaults)

    Returns:
        Dictionary with:
            'success': bool - whether computation succeeded
            'se_unconstrained': dict - SEs in unconstrained space
            'se_bounded': dict - SEs in bounded space (via delta method)
            'correlations': dict - parameter correlation matrix
            'condition_number': float - Hessian condition number
            'hessian_invertible': bool - whether Hessian was invertible

    Note:
        High condition number (>1000) indicates poor parameter identifiability.
        This often occurs when parameters are highly correlated (e.g., α₊ ≈ α₋).
    """
    # Get parameter names
    if param_names is None:
        if model == 'qlearning':
            param_names = QLEARNING_PARAMS
        elif model == 'wmrl':
            param_names = WMRL_PARAMS
        elif model == 'wmrl_m3':
            param_names = WMRL_M3_PARAMS
        elif model == 'wmrl_m5':
            param_names = WMRL_M5_PARAMS
        elif model == 'wmrl_m6a':
            param_names = WMRL_M6A_PARAMS
        elif model == 'wmrl_m6b':
            param_names = WMRL_M6B_PARAMS
        elif model == 'wmrl_m4':
            param_names = WMRL_M4_PARAMS
        else:
            return {'success': False, 'error': f'Unknown model: {model}'}

    n_params = len(param_names)

    try:
        # Compute Hessian at optimum using JAX
        # JIT-compile for faster execution (especially on GPU)
        hess_fn = jax.jit(jax.hessian(objective_fn))
        x_jax = jnp.array(x_opt)
        H = hess_fn(x_jax)

        # Convert to numpy for numerical operations
        H_np = np.array(H)

        # Check for NaN/Inf in Hessian
        if not np.all(np.isfinite(H_np)):
            return {
                'success': False,
                'error': 'Hessian contains NaN/Inf values',
                'hessian_invertible': False
            }

        # Compute eigenvalues for condition number
        eigenvalues = np.linalg.eigvalsh(H_np)
        min_eig = np.min(eigenvalues)
        max_eig = np.max(eigenvalues)

        # Condition number (ratio of max to min eigenvalue)
        # High values indicate near-singularity
        if min_eig <= 0:
            condition_number = np.inf
            hessian_positive_definite = False
        else:
            condition_number = max_eig / min_eig
            hessian_positive_definite = True

        # Try to invert Hessian for covariance matrix
        try:
            cov_matrix = np.linalg.inv(H_np)
            hessian_invertible = True
        except np.linalg.LinAlgError:
            # Hessian is singular - use pseudo-inverse
            cov_matrix = np.linalg.pinv(H_np)
            hessian_invertible = False

        # Extract standard errors (sqrt of diagonal)
        variances = np.diag(cov_matrix)

        # Handle negative variances (numerical issues)
        se_unconstrained = {}
        for i, param in enumerate(param_names):
            if variances[i] > 0:
                se_unconstrained[param] = float(np.sqrt(variances[i]))
            else:
                se_unconstrained[param] = np.nan

        # Compute correlation matrix
        correlations = {}
        std_devs = np.sqrt(np.maximum(variances, 1e-10))
        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)

        # Store as nested dict for easy access
        for i, param_i in enumerate(param_names):
            correlations[param_i] = {}
            for j, param_j in enumerate(param_names):
                correlations[param_i][param_j] = float(corr_matrix[i, j])

        # Transform SEs to bounded space using delta method
        se_bounded = _transform_se_to_bounded(
            se_unconstrained, x_opt, model, param_names
        )

        return {
            'success': True,
            'se_unconstrained': se_unconstrained,
            'se_bounded': se_bounded,
            'correlations': correlations,
            'condition_number': float(condition_number),
            'hessian_invertible': hessian_invertible,
            'hessian_positive_definite': hessian_positive_definite,
            'min_eigenvalue': float(min_eig),
            'max_eigenvalue': float(max_eig)
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'hessian_invertible': False
        }

def _transform_se_to_bounded(
    se_unconstrained: dict[str, float],
    x_opt: np.ndarray,
    model: str,
    param_names: list[str]
) -> dict[str, float]:
    """
    Transform standard errors from unconstrained to bounded space using delta method.

    The delta method approximation:
    SE_bounded ≈ SE_unconstrained * |∂(transform)/∂x|

    For logit transformation: p = 1/(1+exp(-x))
    The derivative is: dp/dx = p * (1-p)

    Args:
        se_unconstrained: SEs in unconstrained space
        x_opt: Optimal values in unconstrained space
        model: Model name for bounds lookup
        param_names: Parameter names

    Returns:
        Dictionary of SEs in bounded parameter space
    """
    # Get bounds for model
    if model == 'qlearning':
        bounds = QLEARNING_BOUNDS
    elif model == 'wmrl':
        bounds = WMRL_BOUNDS
    elif model == 'wmrl_m3':
        bounds = WMRL_M3_BOUNDS
    elif model == 'wmrl_m5':
        bounds = WMRL_M5_BOUNDS
    elif model == 'wmrl_m6a':
        bounds = WMRL_M6A_BOUNDS
    elif model == 'wmrl_m6b':
        bounds = WMRL_M6B_BOUNDS
    elif model == 'wmrl_m4':
        bounds = WMRL_M4_BOUNDS
    else:
        return {}

    se_bounded = {}

    for i, param in enumerate(param_names):
        se_unc = se_unconstrained.get(param, np.nan)
        if np.isnan(se_unc):
            se_bounded[param] = np.nan
            continue

        # Get bounds
        lower, upper = bounds[param]

        # Current value in unconstrained space
        x = x_opt[i]

        # Compute derivative of inverse logit (sigmoid)
        # p = 1 / (1 + exp(-x)) = sigmoid(x)
        # dp/dx = p * (1-p)
        p = 1 / (1 + np.exp(-x))
        dp_dx = p * (1 - p)

        # Scale by range
        scale = upper - lower
        jacobian = scale * dp_dx

        # Delta method: SE_bounded = |jacobian| * SE_unconstrained
        se_bounded[param] = float(np.abs(jacobian) * se_unc)

    return se_bounded

def compute_confidence_intervals(
    params: dict[str, float],
    se_bounded: dict[str, float],
    alpha: float = 0.05
) -> dict[str, tuple[float, float]]:
    """
    Compute confidence intervals for fitted parameters.

    Uses normal approximation: CI = θ̂ ± z_{α/2} * SE

    Args:
        params: Fitted parameter values
        se_bounded: Standard errors in bounded space
        alpha: Significance level (default: 0.05 for 95% CI)

    Returns:
        Dictionary of (lower, upper) tuples for each parameter
    """
    z_crit = stats.norm.ppf(1 - alpha / 2)

    ci = {}
    for param, value in params.items():
        se = se_bounded.get(param, np.nan)
        if np.isnan(se):
            ci[param] = (np.nan, np.nan)
        else:
            ci[param] = (value - z_crit * se, value + z_crit * se)

    return ci

def get_high_correlations(
    correlations: dict[str, dict[str, float]],
    threshold: float = 0.9
) -> list[tuple[str, str, float]]:
    """
    Find pairs of parameters with high correlations.

    High correlations (|r| > 0.9) indicate potential identifiability issues
    where parameters trade off against each other.

    Args:
        correlations: Correlation matrix as nested dict
        threshold: Absolute correlation threshold (default: 0.9)

    Returns:
        list of (param1, param2, correlation) tuples for high correlations
    """
    high_corr = []
    seen_pairs = set()

    for param1, corr_dict in correlations.items():
        for param2, corr in corr_dict.items():
            if param1 == param2:
                continue
            pair = tuple(sorted([param1, param2]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            if abs(corr) > threshold:
                high_corr.append((param1, param2, corr))

    return high_corr

# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("Testing MLE utilities...")

    # Test transformations
    print("\n1. Testing parameter transformations:")
    test_params_ql = {'alpha_pos': 0.3, 'alpha_neg': 0.1, 'epsilon': 0.05}
    x = params_to_unconstrained(test_params_ql, 'qlearning')
    recovered = unconstrained_to_params(x, 'qlearning')
    print(f"   Original: {test_params_ql}")
    print(f"   Unconstrained: {x}")
    print(f"   Recovered: {recovered}")

    # Check round-trip
    for key in test_params_ql:
        assert abs(test_params_ql[key] - recovered[key]) < 1e-10, f"Round-trip failed for {key}"
    print("   Round-trip: PASSED")

    # Test information criteria
    print("\n2. Testing information criteria:")
    nll = 100.0
    n_trials = 500
    k_ql = 3
    k_wmrl = 6

    aic_ql = compute_aic(nll, k_ql)
    bic_ql = compute_bic(nll, k_ql, n_trials)
    aic_wmrl = compute_aic(nll, k_wmrl)
    bic_wmrl = compute_bic(nll, k_wmrl, n_trials)

    print(f"   Q-Learning (k=3): AIC={aic_ql:.2f}, BIC={bic_ql:.2f}")
    print(f"   WM-RL (k=6): AIC={aic_wmrl:.2f}, BIC={bic_wmrl:.2f}")
    print(f"   Delta AIC (WMRL - QL): {aic_wmrl - aic_ql:.2f}")

    # Test group statistics
    print("\n3. Testing group statistics:")
    fake_alpha_pos = np.array([0.25, 0.30, 0.35, 0.28, 0.32, 0.40, 0.22])
    stats_result = compute_group_statistics(fake_alpha_pos, 'alpha_pos')
    print(f"   Mean: {stats_result['mean']:.3f}")
    print(f"   SD: {stats_result['sd']:.3f}")
    print(f"   SE: {stats_result['se']:.3f}")
    print(f"   95% CI: [{stats_result['ci_lower']:.3f}, {stats_result['ci_upper']:.3f}]")

    # Test model comparison
    print("\n4. Testing model comparison:")
    comparison = compare_models_aic(210, 200, 'qlearning', 'wmrl')
    print(f"   Delta AIC: {comparison['delta_aic']:.2f}")
    print(f"   Preferred: {comparison['preferred_model']}")
    print(f"   Evidence: {comparison['evidence_strength']}")

    print("\nAll tests passed!")
