"""Non-centered parameterization helpers for NumPyro hierarchical models.

Implements the hBayesDM convention (Ahn, Haines, Zhang 2017):
    theta_unc = mu_pr + sigma_pr * z
    theta = Phi_approx(theta_unc) * (upper - lower) + lower

Where Phi_approx is the standard normal CDF (jax.scipy.stats.norm.cdf).

K is parameterized in [2, 6] following Senta, Bishop, Collins (2025)
PLoS Computational Biology 21(9):e1012872 (the project reference paper):
    K_i = 2.0 + 4.0 * Phi_approx(mu_K_pr + sigma_K_pr * z_K_i)

Group priors (hBayesDM defaults):
    mu_pr  ~ Normal(0, 1)
    sigma_pr ~ HalfNormal(0.2)

v4.0 INFRA-05.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from config import MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Core transform
# ---------------------------------------------------------------------------


def phi_approx(x: jnp.ndarray) -> jnp.ndarray:
    """Standard normal CDF — the Phi_approx used in hBayesDM.

    Uses ``jax.scipy.stats.norm.cdf`` directly (no polynomial approximation).
    Named for API clarity and grep-ability.

    Parameters
    ----------
    x : jnp.ndarray
        Input values on the unconstrained real line.

    Returns
    -------
    jnp.ndarray
        Values in (0, 1), monotonically increasing in x.
    """
    import jax.scipy.stats as jss

    return jss.norm.cdf(x)


# ---------------------------------------------------------------------------
# Non-centered parameter sampling
# ---------------------------------------------------------------------------


def sample_bounded_param(
    name: str,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
    n_participants: int,
    mu_prior_loc: float = 0.0,
    mu_prior_scale: float = 1.0,
    sigma_prior_scale: float = 0.2,
) -> jnp.ndarray:
    """Sample a non-centered bounded parameter for all participants.

    Implements the hBayesDM non-centered convention:
        mu_pr   ~ Normal(mu_prior_loc, mu_prior_scale)
        sigma_pr ~ HalfNormal(sigma_prior_scale)
        z_i    ~ Normal(0, 1)  for i = 1 .. n_participants
        theta_unc_i = mu_pr + sigma_pr * z_i
        theta_i     = lower + (upper - lower) * Phi_approx(theta_unc_i)

    The resulting individual-level parameters are registered as a
    NumPyro deterministic site named ``name`` for downstream use in
    ``numpyro.infer.Predictive`` and ArviZ diagnostics.

    Parameters
    ----------
    name : str
        Base name for all NumPyro sites.  Sites created:
        ``{name}_mu_pr``, ``{name}_sigma_pr``, ``{name}_z``, ``{name}``.
    lower : float
        Lower bound of the parameter.  Default 0.0.
    upper : float
        Upper bound of the parameter.  Default 1.0.
    n_participants : int
        Number of participants (length of the individual-level array).
    mu_prior_loc : float
        Location (mean) of the Normal prior on ``{name}_mu_pr``.
        Shift this to bias the group mean; e.g. -2.5 for epsilon
        (pushes most mass toward small noise values).  Default 0.0.
    mu_prior_scale : float
        Scale (std) of the Normal prior on ``{name}_mu_pr``.  Default 1.0.
    sigma_prior_scale : float
        Scale of the HalfNormal prior on ``{name}_sigma_pr``.  Controls
        how much inter-individual variability is expected.  Default 0.2.

    Returns
    -------
    jnp.ndarray
        Array of shape ``(n_participants,)`` with values in
        ``[lower, upper]``.
    """
    mu_pr = numpyro.sample(
        f"{name}_mu_pr",
        dist.Normal(mu_prior_loc, mu_prior_scale),
    )
    sigma_pr = numpyro.sample(
        f"{name}_sigma_pr",
        dist.HalfNormal(sigma_prior_scale),
    )
    z = numpyro.sample(
        f"{name}_z",
        dist.Normal(0, 1).expand([n_participants]),
    )
    theta_unc = mu_pr + sigma_pr * z
    theta = lower + (upper - lower) * phi_approx(theta_unc)
    numpyro.deterministic(name, theta)
    return theta


def sample_capacity(
    name: str = "capacity",
    *,
    n_participants: int,
    lower: float = 2.0,
    upper: float = 6.0,
    mu_prior_loc: float = 0.0,
    mu_prior_scale: float = 1.0,
    sigma_prior_scale: float = 0.2,
) -> jnp.ndarray:
    """Sample K (WM capacity) for all participants in [2, 6].

    Convenience wrapper around :func:`sample_bounded_param` that
    enforces the Senta, Bishop, Collins (2025) capacity bounds.

    Formula (from research doc K_PARAMETERIZATION.md):
        K_i = 2.0 + 4.0 * Phi_approx(mu_K_pr + sigma_K_pr * z_K_i)

    Parameters
    ----------
    name : str
        Base name for NumPyro sites.  Default ``"capacity"``.
    n_participants : int
        Number of participants.
    lower : float
        Lower capacity bound.  Must be 2.0 (Senta 2025 convention).
    upper : float
        Upper capacity bound.  Must be 6.0 (task max set size).
    mu_prior_loc : float
        Location of group-mean prior.  Default 0.0.
    mu_prior_scale : float
        Scale of group-mean prior.  Default 1.0.
    sigma_prior_scale : float
        Scale of HalfNormal prior on sigma.  Default 0.2.

    Returns
    -------
    jnp.ndarray
        Array of shape ``(n_participants,)`` with values in ``[2.0, 6.0]``.
    """
    return sample_bounded_param(
        name,
        lower=lower,
        upper=upper,
        n_participants=n_participants,
        mu_prior_loc=mu_prior_loc,
        mu_prior_scale=mu_prior_scale,
        sigma_prior_scale=sigma_prior_scale,
    )


# ---------------------------------------------------------------------------
# Prior defaults catalogue
# ---------------------------------------------------------------------------

PARAM_PRIOR_DEFAULTS: dict[str, dict] = {
    "alpha_pos": {"lower": 0.0, "upper": 1.0, "mu_prior_loc": 0.0},
    "alpha_neg": {"lower": 0.0, "upper": 1.0, "mu_prior_loc": 0.0},
    "epsilon": {"lower": 0.0, "upper": 1.0, "mu_prior_loc": -2.5},
    "phi": {"lower": 0.0, "upper": 1.0, "mu_prior_loc": -0.8},
    "rho": {"lower": 0.0, "upper": 1.0, "mu_prior_loc": 0.8},
    "capacity": {"lower": 2.0, "upper": 6.0, "mu_prior_loc": 0.0},
    "kappa": {"lower": 0.0, "upper": 1.0, "mu_prior_loc": -2.0},
    "kappa_s": {"lower": 0.0, "upper": 1.0, "mu_prior_loc": -2.0},
    "kappa_total": {"lower": 0.0, "upper": 1.0, "mu_prior_loc": -2.0},
    "kappa_share": {"lower": 0.0, "upper": 1.0, "mu_prior_loc": 0.0},
    "phi_rl": {"lower": 0.0, "upper": 1.0, "mu_prior_loc": -0.8},
}
"""Default prior hyperparameters for all bounded RLWM parameters.

Keys are parameter names matching ``MODEL_REGISTRY['params']`` lists.
Each value dict is passed as ``**kwargs`` to :func:`sample_bounded_param`.

Notes
-----
- ``mu_prior_loc`` shifts the group mean on the probit scale.
  Negative values push the group mean toward small values; e.g.,
  ``epsilon: -2.5`` implies Phi(-2.5) ≈ 0.006 group-mean noise.
- ``lower`` / ``upper`` match Senta, Bishop, Collins (2025) for
  ``capacity`` and [0, 1] for all other parameters.
- LBA-specific parameters (v_scale, A, delta, t0) are NOT listed here
  because they require log-scale or non-standard transforms handled
  separately in the M4 hierarchical model.
"""


# ---------------------------------------------------------------------------
# Model-level parameter sampling
# ---------------------------------------------------------------------------


def sample_model_params(
    model_name: str,
    n_participants: int,
) -> dict[str, jnp.ndarray]:
    """Sample all bounded parameters for a given model.

    Iterates over the parameter list for ``model_name`` from
    ``MODEL_REGISTRY``, looks up each parameter's prior defaults from
    :data:`PARAM_PRIOR_DEFAULTS`, and calls :func:`sample_bounded_param`
    (or :func:`sample_capacity` for ``"capacity"``).

    Parameters not in ``PARAM_PRIOR_DEFAULTS`` (e.g., LBA parameters)
    are skipped with a warning and must be sampled manually.

    Parameters
    ----------
    model_name : str
        Key into ``config.MODEL_REGISTRY`` (e.g. ``"wmrl_m3"``).
    n_participants : int
        Number of participants.

    Returns
    -------
    dict[str, jnp.ndarray]
        Mapping from parameter name to sampled array of shape
        ``(n_participants,)``.
    """
    params_list: list[str] = MODEL_REGISTRY[model_name]["params"]
    sampled: dict[str, jnp.ndarray] = {}

    for param_name in params_list:
        if param_name not in PARAM_PRIOR_DEFAULTS:
            # LBA-only or future params — skip gracefully
            continue

        defaults = PARAM_PRIOR_DEFAULTS[param_name]
        sampled[param_name] = sample_bounded_param(
            param_name,
            lower=defaults["lower"],
            upper=defaults["upper"],
            n_participants=n_participants,
            mu_prior_loc=defaults["mu_prior_loc"],
        )

    return sampled
