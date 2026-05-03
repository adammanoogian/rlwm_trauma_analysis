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


def sample_unbounded_normal_param(
    name: str,
    *,
    n_participants: int,
    mu_prior_loc: float = 0.0,
    mu_prior_scale: float = 0.5,
    sigma_prior_scale: float = 0.5,
    clip_lower: float | None = None,
    clip_upper: float | None = None,
) -> jnp.ndarray:
    """Sample a non-centered unbounded parameter on the real line.

    Used for softmax-bias kappa (Collins 2025) where kappa lives on the
    real line (or clipped to [-1, 1] for paper conformance). No
    Phi_approx transform is applied — the sampled value IS the raw
    parameter value.

    Implements::

        mu_pr   ~ Normal(mu_prior_loc, mu_prior_scale)
        sigma_pr ~ HalfNormal(sigma_prior_scale)
        z_i    ~ Normal(0, 1)  for i = 1 .. n_participants
        theta_i = mu_pr + sigma_pr * z_i
        (optionally clipped to [clip_lower, clip_upper])

    Parameters
    ----------
    name : str
        Base name for NumPyro sites: ``{name}_mu_pr``,
        ``{name}_sigma_pr``, ``{name}_z``, ``{name}``.
    n_participants : int
        Number of participants.
    mu_prior_loc : float
        Location of the Normal prior on the group mean. Default 0.0.
    mu_prior_scale : float
        Scale of the Normal prior on the group mean. Default 0.5
        (tighter than the bounded helper's default of 1.0 because
        unbounded kappa in [-1, 1] needs less group-level wiggle).
    sigma_prior_scale : float
        Scale of HalfNormal on inter-individual SD. Default 0.5.
    clip_lower, clip_upper : float | None
        Optional hard bounds applied via :func:`jnp.clip`. Use for paper
        conformance to Collins 2025 kappa in [-1, 1]. Default None
        (no clip applied; sample lives on the unconstrained real line).

    Returns
    -------
    jnp.ndarray
        Array of shape ``(n_participants,)`` on the real line (or
        clipped to ``[clip_lower, clip_upper]``).
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
    theta = mu_pr + sigma_pr * z
    if clip_lower is not None or clip_upper is not None:
        theta = jnp.clip(
            theta,
            min=clip_lower if clip_lower is not None else -jnp.inf,
            max=clip_upper if clip_upper is not None else jnp.inf,
        )
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

    Formula (from docs/03_methods_reference/MODEL_REFERENCE.md section 12):
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
    "epsilon": {"lower": 0.0, "upper": 1.0, "mu_prior_loc": -2.0},
    "phi": {"lower": 0.0, "upper": 1.0, "mu_prior_loc": 0.0},
    "rho": {"lower": 0.0, "upper": 1.0, "mu_prior_loc": 0.0},
    "capacity": {"lower": 2.0, "upper": 6.0, "mu_prior_loc": 0.0},
    "kappa": {"lower": 0.0, "upper": 1.0, "mu_prior_loc": -1.0},
    "kappa_s": {"lower": 0.0, "upper": 1.0, "mu_prior_loc": -1.0},
    "kappa_total": {"lower": 0.0, "upper": 1.0, "mu_prior_loc": -1.0},
    "kappa_share": {"lower": 0.0, "upper": 1.0, "mu_prior_loc": 0.0},
    "phi_rl": {"lower": 0.0, "upper": 1.0, "mu_prior_loc": 0.0},
}
"""Default prior hyperparameters for all bounded RLWM parameters.

Keys are parameter names matching ``MODEL_REGISTRY['params']`` lists.
Each value dict is passed as ``**kwargs`` to :func:`sample_bounded_param`.

Design: principled weakly-informative priors (v4.0 Phase 16 revision)
---------------------------------------------------------------------
As of 2026-04-17, all ``mu_prior_loc`` values default to 0.0, which
corresponds to a group-mean prior of Phi(0) = 0.5 on the bounded scale
for all [0, 1] parameters.  The 95% prior interval on the bounded scale,
given ``sigma_pr ~ HalfNormal(0.2)``, is approximately [0.02, 0.98].

Rationale for moving to 0.0 from the prior MLE-calibrated values:

- The previous priors (``epsilon = -2.5``, ``phi = -0.8``, ``rho = +0.8``,
  ``kappa-family = -2.0``) were informed by MLE point estimates from
  quick-006.  This is a mild form of empirical Bayes that creates a
  subtle circularity — the MLE fit already shrinks parameters toward
  their modal values for under-identified participants, and we then
  tell the hierarchical model "these parameters are usually small" on
  the basis of those MLE results.
- An informative prior toward zero perseveration (``kappa = -2.0``) is
  **conservative for L2 null testing**: if trauma is associated with
  *increased* perseveration, the prior pulling kappa down reduces the
  chance of the HDI excluding zero.  The published trauma-parameter
  literature does not universally support that direction of effect,
  so a neutral prior avoids building the hypothesis into the prior.
- Principled priors make the L2 HDI interpretable as "data-driven"
  rather than "prior-shifted" conclusions.

Phase 32 update (2026-05-03): The κ-family ``mu_prior_loc`` is shifted
from 0.0 back to −1.0.  This anchors the group-mean prior at Phi(−1) ≈
0.16 on the bounded scale, matching Rmus 2023's hierarchical
empirical-Bayes pattern (Beta(1+a, 1+b) with Gamma(1,1) / Gamma(12,1)
hyperpriors) without re-introducing the MLE-circularity of the
pre-2026-04-17 priors.  Justification: Baribault & Collins 2023 (their
"more informative priors" pathology recipe for multimodal R-hat) plus
the model-misspecification finding that M3 / M5 / M6a hit identical
R-hat = 1.60 with ESS = 7 under the previous ``mu_prior_loc = 0.0``
setup.  ``kappa_share`` is left at 0.0 because the channel-share is
genuinely uncertain a priori — Collins 2025's WMH motor-vs-stimulus
split does not anchor it to either corner.

Consequences:

- ``sigma_pr ~ HalfNormal(0.2)`` is retained — this gives the data
  enough leverage to pull the group mean away from 0.5 toward wherever
  the MLE cluster sits.  In practice the posterior group mean will
  match the MLE-calibrated prior mean closely for well-identified
  parameters, so only L2 HDIs change meaningfully.
- For informal sensitivity analysis, set ``mu_prior_loc`` back to the
  prior values (`_PRIOR_LEGACY_MLE_CALIBRATED` below).

LBA-specific parameters (v_scale, A, delta, t0) are NOT listed here
because they require log-scale or non-standard transforms handled
separately in the M4 hierarchical model.
"""

# Phase 32-04: kappa-family parameter names. These params receive
# softmax-bias treatment (sample_unbounded_normal_param, kappa in
# [-1, 1]) when kappa_parameterization == "softmax", and convex-mixture
# treatment (sample_bounded_param, kappa in [0, 1]) when "convex".
# kappa_share is intentionally EXCLUDED — it is a channel-share weight
# (mixture mass) not a softmax additive bias, so it stays bounded
# [0, 1] under both parameterizations.
KAPPA_FAMILY_PARAMS: set[str] = {"kappa", "kappa_s", "kappa_total"}


# Legacy MLE-calibrated defaults (kept for sensitivity analysis).
# Use via `sample_bounded_param(..., mu_prior_loc=_PRIOR_LEGACY_MLE_CALIBRATED[p])`
# when reproducing pre-v4.0-refactor fits.
_PRIOR_LEGACY_MLE_CALIBRATED: dict[str, float] = {
    "alpha_pos": 0.0,
    "alpha_neg": 0.0,
    "epsilon": -2.5,
    "phi": -0.8,
    "rho": 0.8,
    "capacity": 0.0,
    "kappa": -2.0,
    "kappa_s": -2.0,
    "kappa_total": -2.0,
    "kappa_share": 0.0,
    "phi_rl": -0.8,
}


# ---------------------------------------------------------------------------
# Model-level parameter sampling
# ---------------------------------------------------------------------------


def sample_model_params(
    model_name: str,
    n_participants: int,
    *,
    kappa_parameterization: str = "softmax",
) -> dict[str, jnp.ndarray]:
    """Sample all parameters for a given model.

    Iterates over the parameter list for ``model_name`` from
    ``MODEL_REGISTRY``, looks up each parameter's prior defaults from
    :data:`PARAM_PRIOR_DEFAULTS`, and dispatches to the appropriate
    sampler:

    - kappa-family parameters (:data:`KAPPA_FAMILY_PARAMS`) under
      ``kappa_parameterization="softmax"``: sampled via
      :func:`sample_unbounded_normal_param` clipped to ``[-1, 1]``
      (Collins 2025 softmax-bias formulation).
    - kappa-family parameters under ``kappa_parameterization="convex"``:
      sampled via :func:`sample_bounded_param` in ``[0, 1]``
      (Senta 2025 convex-mixture formulation, legacy v5.0 behavior).
    - ``"capacity"``: dispatched to :func:`sample_capacity` (K in
      ``[2, 6]``).
    - All other bounded params: :func:`sample_bounded_param` with
      defaults from :data:`PARAM_PRIOR_DEFAULTS`.

    Parameters not in ``PARAM_PRIOR_DEFAULTS`` (e.g., LBA parameters)
    are skipped with a warning and must be sampled manually.

    Parameters
    ----------
    model_name : str
        Key into ``config.MODEL_REGISTRY`` (e.g. ``"wmrl_m3"``).
    n_participants : int
        Number of participants.
    kappa_parameterization : str, optional
        One of ``{"softmax", "convex"}``. Default ``"softmax"`` per
        Phase 32-04 user decision (Collins 2025 alignment).
        ``"convex"`` reproduces v5.0 pre-Phase-32 behavior.

    Returns
    -------
    dict[str, jnp.ndarray]
        Mapping from parameter name to sampled array of shape
        ``(n_participants,)``.

    Raises
    ------
    ValueError
        If ``kappa_parameterization`` is not in ``{"softmax", "convex"}``.
    """
    if kappa_parameterization not in {"softmax", "convex"}:
        raise ValueError(
            f"kappa_parameterization must be 'softmax' or 'convex', "
            f"got {kappa_parameterization!r}"
        )

    params_list: list[str] = MODEL_REGISTRY[model_name]["params"]
    sampled: dict[str, jnp.ndarray] = {}

    for param_name in params_list:
        if param_name not in PARAM_PRIOR_DEFAULTS:
            # LBA-only or future params — skip gracefully
            continue

        defaults = PARAM_PRIOR_DEFAULTS[param_name]

        # Phase 32-04: dispatch kappa-family params on parameterization.
        if (
            kappa_parameterization == "softmax"
            and param_name in KAPPA_FAMILY_PARAMS
        ):
            sampled[param_name] = sample_unbounded_normal_param(
                param_name,
                n_participants=n_participants,
                mu_prior_loc=0.0,
                mu_prior_scale=0.5,
                sigma_prior_scale=0.5,
                clip_lower=-1.0,
                clip_upper=1.0,
            )
            continue

        sampled[param_name] = sample_bounded_param(
            param_name,
            lower=defaults["lower"],
            upper=defaults["upper"],
            n_participants=n_participants,
            mu_prior_loc=defaults["mu_prior_loc"],
        )

    return sampled
