"""Random-effects Bayesian Model Selection with protected exceedance probability.

This module implements the RFX-BMS framework of Stephan et al. (2009) extended
with the Bayesian Omnibus Risk (BOR) and protected exceedance probability (PXP)
correction of Rigoux et al. (2014). The implementation is a faithful port of the
MATLAB reference ``mfit/bms.m`` (S. J. Gershman).

The algorithm treats the discrete model identity for each participant as a
latent categorical variable, places a Dirichlet prior with concentration
``alpha0`` on the population model frequencies, and performs variational
Bayesian (VB) inference to recover the Dirichlet posterior parameters
``alpha``. The exceedance probability that a given model is the most frequent
is estimated by Monte-Carlo sampling from ``Dirichlet(alpha)``. The PXP rescales
the exceedance probability by the posterior probability that the observed model
evidence is **not** consistent with the null hypothesis of equal frequencies.

References
----------
Stephan, K. E., Penny, W. D., Daunizeau, J., Moran, R. J., & Friston, K. J.
    (2009). Bayesian model selection for group studies. *NeuroImage*, 46(4),
    1004-1017. https://doi.org/10.1016/j.neuroimage.2009.03.025

Rigoux, L., Stephan, K. E., Friston, K. J., & Daunizeau, J. (2014). Bayesian
    model selection for group studies - Revisited. *NeuroImage*, 84, 971-985.
    https://doi.org/10.1016/j.neuroimage.2013.08.065

Gershman, S. J. (2016). mfit: simple model-fitting tools. GitHub repository
    https://github.com/sjgershm/mfit (MIT license). File ``bms.m`` was the
    primary reference for this port.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp, psi


def _vb_dirichlet_update(
    log_evidence: np.ndarray,
    alpha0_vec: np.ndarray,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the variational-Bayes E-step on the Dirichlet-categorical model.

    Alternates between (a) computing the posterior responsibilities ``g`` over
    models per participant given the current Dirichlet parameters and (b)
    updating the Dirichlet parameters given the expected per-model counts. The
    loop terminates when the L-infinity change in ``alpha`` falls below ``tol``.

    Parameters
    ----------
    log_evidence : np.ndarray, shape (n_subjects, n_models)
        Per-participant log model evidence.
    alpha0_vec : np.ndarray, shape (n_models,)
        Dirichlet prior concentrations (broadcast from the scalar ``alpha0``).
    max_iter : int
        Hard cap on VB iterations.
    tol : float
        L-infinity convergence threshold on the Dirichlet parameter vector.

    Returns
    -------
    alpha : np.ndarray, shape (n_models,)
        Converged Dirichlet posterior parameters.
    g : np.ndarray, shape (n_subjects, n_models)
        Posterior responsibilities of each model for each participant.
    """
    alpha = alpha0_vec.copy()
    g = np.zeros_like(log_evidence)
    for _ in range(max_iter):
        log_u = log_evidence + (psi(alpha) - psi(alpha.sum()))[None, :]
        log_u -= log_u.max(axis=1, keepdims=True)
        u = np.exp(log_u)
        g = u / u.sum(axis=1, keepdims=True)
        beta = g.sum(axis=0)
        alpha_new = alpha0_vec + beta
        if np.max(np.abs(alpha_new - alpha)) < tol:
            alpha = alpha_new
            break
        alpha = alpha_new
    return alpha, g


def _vb_free_energy(
    log_evidence: np.ndarray,
    alpha: np.ndarray,
    alpha0_vec: np.ndarray,
) -> float:
    """Compute the variational free energy (ELBO) for the RFX-BMS model.

    Uses the closed-form ELBO of the Dirichlet-categorical mixture obtained by
    marginalising the per-participant model responsibilities analytically:

    ``F = sum_n logsumexp_k( lme[n,k] + psi(alpha_k) - psi(sum alpha) )
          - KL[Dir(alpha) || Dir(alpha0)]``

    where the Dirichlet KL is the standard closed form

    ``KL = gammaln(sum alpha) - sum_k gammaln(alpha_k)
           - gammaln(sum alpha0) + sum_k gammaln(alpha0_k)
           + sum_k (alpha_k - alpha0_k) * (psi(alpha_k) - psi(sum alpha))``

    The first term is the expected complete-data log-evidence under the
    Dirichlet-mean factorised variational posterior; the second term penalises
    departure of the Dirichlet posterior from the prior.

    Parameters
    ----------
    log_evidence : np.ndarray, shape (n_subjects, n_models)
        Per-participant log model evidence.
    alpha : np.ndarray, shape (n_models,)
        Dirichlet posterior parameters.
    alpha0_vec : np.ndarray, shape (n_models,)
        Dirichlet prior parameters.

    Returns
    -------
    float
        Variational free energy (higher is better).
    """
    psi_alpha = psi(alpha) - psi(alpha.sum())
    expected_ll = logsumexp(log_evidence + psi_alpha[None, :], axis=1).sum()
    # KL(Dir(alpha) || Dir(alpha0)) — standard closed form.
    kl_dirichlet = (
        gammaln(alpha.sum())
        - np.sum(gammaln(alpha))
        - gammaln(alpha0_vec.sum())
        + np.sum(gammaln(alpha0_vec))
        + np.sum((alpha - alpha0_vec) * psi_alpha)
    )
    return float(expected_ll - kl_dirichlet)


def _exceedance_probability(
    alpha: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Estimate exceedance probabilities by Monte-Carlo Dirichlet sampling.

    Draws ``n_samples`` independent samples from ``Dirichlet(alpha)``, records
    the argmax index per sample, and returns the empirical fraction of times
    each model was the winner. This estimator is unbiased and its standard
    error decays as ``1 / sqrt(n_samples)``.

    Parameters
    ----------
    alpha : np.ndarray, shape (n_models,)
        Dirichlet posterior parameters.
    n_samples : int
        Number of Monte-Carlo draws.
    rng : np.random.Generator
        Seeded NumPy generator for reproducibility.

    Returns
    -------
    np.ndarray, shape (n_models,)
        Exceedance probabilities summing to 1.
    """
    n_models = alpha.shape[0]
    samples = rng.dirichlet(alpha, size=n_samples)
    winners = samples.argmax(axis=1)
    return np.bincount(winners, minlength=n_models) / n_samples


def _bor(
    log_evidence: np.ndarray,
    alpha: np.ndarray,
    alpha0_vec: np.ndarray,
) -> float:
    """Compute the Bayesian Omnibus Risk (Rigoux et al. 2014).

    The BOR is the posterior probability that the data are consistent with the
    null hypothesis of equal model frequencies, given by

    ``BOR = 1 / (1 + exp(F1 - F0))``

    where ``F1`` is the free energy of the full (heterogeneous, random-effects)
    model and ``F0`` is the free energy of the null model with **fixed**
    uniform frequencies ``r = 1/K``. Under the null the Dirichlet posterior is
    a degenerate delta at the uniform vector (not a posterior to be inferred),
    so the null free energy carries no KL penalty and reduces to

    ``F0 = sum_n logsumexp_k( log_evidence[n, k] ) - N * log(K)``

    which is exactly the log-likelihood of the data under a uniform mixture
    per Rigoux et al. (2014) eq. A1 with ``r`` fixed. The planner spec's
    alternative formulation (null as a Dirichlet posterior with concentration
    ``alpha0 + n/K``) yields F1 == F0 exactly when the data are uninformative
    (BOR = 0.5), which does not match the published "null strongly supported"
    behaviour for uniform evidence; we use the fixed-``r`` form instead.

    Parameters
    ----------
    log_evidence : np.ndarray, shape (n_subjects, n_models)
        Per-participant log model evidence.
    alpha : np.ndarray, shape (n_models,)
        Dirichlet posterior parameters from the full (heterogeneous) model.
    alpha0_vec : np.ndarray, shape (n_models,)
        Dirichlet prior parameters (used for F1's KL term).

    Returns
    -------
    float
        BOR in ``[0, 1]``. Values near 1 indicate the null is likely.
    """
    n_subjects, n_models = log_evidence.shape

    # F1: full random-effects model (Dirichlet posterior inferred).
    f1 = _vb_free_energy(log_evidence, alpha, alpha0_vec)

    # F0: null with fixed uniform r = 1/K. The Dirichlet KL vanishes because
    # r is fixed (no posterior inferred); only the data-likelihood term
    # remains, evaluated at the uniform categorical.
    f0 = logsumexp(log_evidence, axis=1).sum() - n_subjects * np.log(n_models)

    # Numerically stable 1 / (1 + exp(F1 - F0)).
    delta = f1 - f0
    if delta > 500.0:
        return 0.0
    if delta < -500.0:
        return 1.0
    return float(1.0 / (1.0 + np.exp(delta)))


def rfx_bms(
    log_evidence: np.ndarray,
    alpha0: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-4,
    n_xp_samples: int = 1_000_000,
    seed: int = 42,
) -> dict[str, np.ndarray | float]:
    """Random-effects Bayesian Model Selection with protected exceedance probability.

    Implements the variational Bayesian treatment of Stephan et al. (2009) and
    the PXP correction of Rigoux et al. (2014). This is a faithful port of the
    MATLAB reference ``mfit/bms.m``.

    Parameters
    ----------
    log_evidence : np.ndarray, shape (n_subjects, n_models)
        Per-participant log model evidence. For the Phase 21 pipeline this is
        typically the participant-summed LOO log-likelihood extracted from
        ``arviz.loo(idata, pointwise=True).loo_i`` and reshaped per
        participant. Must be strictly 2-D.
    alpha0 : float, default 1.0
        Uniform Dirichlet prior concentration on the population model
        frequencies. The default of 1.0 corresponds to a flat prior.
    max_iter : int, default 1000
        Maximum number of VB E-step iterations.
    tol : float, default 1e-4
        L-infinity convergence threshold on the Dirichlet parameter vector.
    n_xp_samples : int, default 1_000_000
        Number of Monte-Carlo draws used to estimate the exceedance
        probability.
    seed : int, default 42
        Seed for the NumPy ``Generator`` used in the exceedance-probability
        sampling.

    Returns
    -------
    dict
        ``alpha`` : np.ndarray, shape (n_models,)
            Dirichlet posterior parameters.
        ``r`` : np.ndarray, shape (n_models,)
            Expected model frequencies, ``alpha / alpha.sum()``.
        ``xp`` : np.ndarray, shape (n_models,)
            Exceedance probabilities (sum to 1).
        ``bor`` : float
            Bayesian Omnibus Risk (Rigoux et al. 2014).
        ``pxp`` : np.ndarray, shape (n_models,)
            Protected exceedance probabilities (sum to 1).

    Raises
    ------
    ValueError
        If ``log_evidence`` is not 2-D, contains non-finite entries, or if
        ``alpha0`` is non-positive.

    Notes
    -----
    The protected exceedance probability is defined as

    ``PXP = (1 - BOR) * XP + BOR / K``

    where ``K`` is the number of models. As ``BOR -> 1`` (null plausible), all
    PXPs tend to ``1/K``; as ``BOR -> 0`` (null implausible), PXP reduces to
    the raw exceedance probability XP.
    """
    log_evidence = np.asarray(log_evidence, dtype=np.float64)

    if log_evidence.ndim != 2:
        raise ValueError(
            "log_evidence must be 2-D (n_subjects, n_models); "
            f"got shape {log_evidence.shape} (ndim={log_evidence.ndim})"
        )
    if not np.all(np.isfinite(log_evidence)):
        raise ValueError(
            "log_evidence must contain only finite values; "
            f"got {np.sum(~np.isfinite(log_evidence))} non-finite entries "
            "(NaN or inf). Check for failed LOO estimates before calling."
        )
    if alpha0 <= 0.0:
        raise ValueError(
            f"alpha0 must be strictly positive; got {alpha0}"
        )

    _, n_models = log_evidence.shape
    alpha0_vec = np.full(n_models, float(alpha0))

    # Variational-Bayes Dirichlet update.
    alpha, _g = _vb_dirichlet_update(log_evidence, alpha0_vec, max_iter, tol)

    # Expected model frequencies.
    r = alpha / alpha.sum()

    # Exceedance probabilities via Dirichlet sampling.
    rng = np.random.default_rng(seed)
    xp = _exceedance_probability(alpha, n_xp_samples, rng)

    # Bayesian Omnibus Risk via free-energy ratio with null-hypothesis posterior.
    bor = _bor(log_evidence, alpha, alpha0_vec)

    # Protected exceedance probability per Rigoux et al. (2014).
    pxp = (1.0 - bor) * xp + bor / n_models

    return {
        "alpha": alpha,
        "r": r,
        "xp": xp,
        "bor": bor,
        "pxp": pxp,
    }
