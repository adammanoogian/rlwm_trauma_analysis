"""Post-sampling Bayesian diagnostics for RLWM hierarchical models.

Provides:
- compute_pointwise_log_lik(): per-trial log-likelihood for WAIC/LOO
- build_inference_data_with_loglik(): ArviZ InferenceData with log_likelihood group

After calling build_inference_data_with_loglik(), az.waic(idata) and
az.loo(idata) work natively without 'log_likelihood group missing' warnings.

v4.0 INFRA-03.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    import arviz as az
    from numpyro.infer import MCMC

from scripts.fitting.jax_likelihoods import (
    q_learning_multiblock_likelihood_stacked,
    wmrl_multiblock_likelihood_stacked,
    wmrl_m3_multiblock_likelihood_stacked,
    wmrl_m5_multiblock_likelihood_stacked,
    wmrl_m6a_multiblock_likelihood_stacked,
    wmrl_m6b_multiblock_likelihood_stacked,
)


# ---------------------------------------------------------------------------
# Internal dispatch helpers
# ---------------------------------------------------------------------------

def _get_param_names(model_name: str) -> list[str]:
    """Return the individual-level parameter names stored in MCMC samples.

    Parameters
    ----------
    model_name : str
        One of 'qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b'.

    Returns
    -------
    list[str]
        Parameter name keys as stored in ``mcmc.get_samples(group_by_chain=True)``.
    """
    _param_map: dict[str, list[str]] = {
        "qlearning": ["alpha_pos", "alpha_neg", "epsilon"],
        "wmrl": ["alpha_pos", "alpha_neg", "phi", "rho", "capacity", "epsilon"],
        "wmrl_m3": ["alpha_pos", "alpha_neg", "phi", "rho", "capacity", "kappa", "epsilon"],
        "wmrl_m5": ["alpha_pos", "alpha_neg", "phi", "rho", "capacity", "kappa", "phi_rl", "epsilon"],
        "wmrl_m6a": ["alpha_pos", "alpha_neg", "phi", "rho", "capacity", "kappa_s", "epsilon"],
        "wmrl_m6b": [
            "alpha_pos",
            "alpha_neg",
            "phi",
            "rho",
            "capacity",
            "kappa_total",
            "kappa_share",
            "epsilon",
        ],
    }
    if model_name not in _param_map:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Expected one of {sorted(_param_map)}; got '{model_name}'."
        )
    return _param_map[model_name]


def _get_likelihood_fn(model_name: str):
    """Dispatch model name to the corresponding stacked likelihood function.

    Parameters
    ----------
    model_name : str
        One of 'qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b'.

    Returns
    -------
    callable
        Stacked likelihood function accepting ``return_pointwise=True``.
    """
    _fn_map = {
        "qlearning": q_learning_multiblock_likelihood_stacked,
        "wmrl": wmrl_multiblock_likelihood_stacked,
        "wmrl_m3": wmrl_m3_multiblock_likelihood_stacked,
        "wmrl_m5": wmrl_m5_multiblock_likelihood_stacked,
        "wmrl_m6a": wmrl_m6a_multiblock_likelihood_stacked,
        "wmrl_m6b": wmrl_m6b_multiblock_likelihood_stacked,
    }
    if model_name not in _fn_map:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Expected one of {sorted(_fn_map)}; got '{model_name}'."
        )
    return _fn_map[model_name]


def _build_per_participant_fn(model_name: str, pdata: dict, num_stimuli: int, num_actions: int, q_init: float):
    """Build a vmappable per-participant pointwise log-lik function.

    The returned function maps a dict of per-participant parameter vectors
    (shape: ``(n_params,)`` — each element is scalar for that participant)
    to a flat pointwise log-lik array of shape ``(n_blocks * max_trials,)``.

    Parameters
    ----------
    model_name : str
        Model identifier.
    pdata : dict
        Stacked arrays for ONE participant with keys:
        - ``stimuli_stacked``, ``actions_stacked``, ``rewards_stacked``, ``masks_stacked``
        - ``set_sizes_stacked`` (for WMRL models; absent for qlearning)
    num_stimuli : int
    num_actions : int
    q_init : float

    Returns
    -------
    callable
        A function ``fn(*param_scalars) -> jnp.ndarray`` of shape
        ``(n_blocks * max_trials,)`` that can be vmapped over samples.
    """
    fn = _get_likelihood_fn(model_name)

    stimuli_stacked = pdata["stimuli_stacked"]
    actions_stacked = pdata["actions_stacked"]
    rewards_stacked = pdata["rewards_stacked"]
    masks_stacked = pdata["masks_stacked"]

    if model_name == "qlearning":
        def _per_sample(alpha_pos, alpha_neg, epsilon):
            _, pointwise = fn(
                stimuli_stacked=stimuli_stacked,
                actions_stacked=actions_stacked,
                rewards_stacked=rewards_stacked,
                masks_stacked=masks_stacked,
                alpha_pos=alpha_pos,
                alpha_neg=alpha_neg,
                epsilon=epsilon,
                num_stimuli=num_stimuli,
                num_actions=num_actions,
                q_init=q_init,
                return_pointwise=True,
            )
            return pointwise

    elif model_name == "wmrl":
        set_sizes_stacked = pdata["set_sizes_stacked"]

        def _per_sample(alpha_pos, alpha_neg, phi, rho, capacity, epsilon):
            _, pointwise = fn(
                stimuli_stacked=stimuli_stacked,
                actions_stacked=actions_stacked,
                rewards_stacked=rewards_stacked,
                set_sizes_stacked=set_sizes_stacked,
                masks_stacked=masks_stacked,
                alpha_pos=alpha_pos,
                alpha_neg=alpha_neg,
                phi=phi,
                rho=rho,
                capacity=capacity,
                epsilon=epsilon,
                num_stimuli=num_stimuli,
                num_actions=num_actions,
                q_init=q_init,
                return_pointwise=True,
            )
            return pointwise

    elif model_name == "wmrl_m3":
        set_sizes_stacked = pdata["set_sizes_stacked"]

        def _per_sample(alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon):
            _, pointwise = fn(
                stimuli_stacked=stimuli_stacked,
                actions_stacked=actions_stacked,
                rewards_stacked=rewards_stacked,
                set_sizes_stacked=set_sizes_stacked,
                masks_stacked=masks_stacked,
                alpha_pos=alpha_pos,
                alpha_neg=alpha_neg,
                phi=phi,
                rho=rho,
                capacity=capacity,
                kappa=kappa,
                epsilon=epsilon,
                num_stimuli=num_stimuli,
                num_actions=num_actions,
                q_init=q_init,
                return_pointwise=True,
            )
            return pointwise

    elif model_name == "wmrl_m5":
        set_sizes_stacked = pdata["set_sizes_stacked"]

        def _per_sample(alpha_pos, alpha_neg, phi, rho, capacity, kappa, phi_rl, epsilon):
            _, pointwise = fn(
                stimuli_stacked=stimuli_stacked,
                actions_stacked=actions_stacked,
                rewards_stacked=rewards_stacked,
                set_sizes_stacked=set_sizes_stacked,
                masks_stacked=masks_stacked,
                alpha_pos=alpha_pos,
                alpha_neg=alpha_neg,
                phi=phi,
                rho=rho,
                capacity=capacity,
                kappa=kappa,
                phi_rl=phi_rl,
                epsilon=epsilon,
                num_stimuli=num_stimuli,
                num_actions=num_actions,
                q_init=q_init,
                return_pointwise=True,
            )
            return pointwise

    elif model_name == "wmrl_m6a":
        set_sizes_stacked = pdata["set_sizes_stacked"]

        def _per_sample(alpha_pos, alpha_neg, phi, rho, capacity, kappa_s, epsilon):
            _, pointwise = fn(
                stimuli_stacked=stimuli_stacked,
                actions_stacked=actions_stacked,
                rewards_stacked=rewards_stacked,
                set_sizes_stacked=set_sizes_stacked,
                masks_stacked=masks_stacked,
                alpha_pos=alpha_pos,
                alpha_neg=alpha_neg,
                phi=phi,
                rho=rho,
                capacity=capacity,
                kappa_s=kappa_s,
                epsilon=epsilon,
                num_stimuli=num_stimuli,
                num_actions=num_actions,
                q_init=q_init,
                return_pointwise=True,
            )
            return pointwise

    elif model_name == "wmrl_m6b":
        set_sizes_stacked = pdata["set_sizes_stacked"]

        def _per_sample(alpha_pos, alpha_neg, phi, rho, capacity, kappa_total, kappa_share, epsilon):
            # Decode stick-breaking: kappa = kappa_total * kappa_share, kappa_s = kappa_total * (1 - kappa_share)
            kappa = kappa_total * kappa_share
            kappa_s = kappa_total * (1.0 - kappa_share)
            _, pointwise = fn(
                stimuli_stacked=stimuli_stacked,
                actions_stacked=actions_stacked,
                rewards_stacked=rewards_stacked,
                set_sizes_stacked=set_sizes_stacked,
                masks_stacked=masks_stacked,
                alpha_pos=alpha_pos,
                alpha_neg=alpha_neg,
                phi=phi,
                rho=rho,
                capacity=capacity,
                kappa=kappa,
                kappa_s=kappa_s,
                epsilon=epsilon,
                num_stimuli=num_stimuli,
                num_actions=num_actions,
                q_init=q_init,
                return_pointwise=True,
            )
            return pointwise

    else:
        raise ValueError(f"No per-sample function defined for model '{model_name}'.")

    return _per_sample


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_pointwise_log_lik(
    mcmc: "MCMC",
    participant_data_stacked: dict,
    model_name: str,
    *,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
) -> jnp.ndarray:
    """Compute per-trial pointwise log-likelihoods for WAIC/LOO computation.

    Calls the appropriate stacked likelihood function with
    ``return_pointwise=True`` for each participant, vectorized over MCMC
    samples using ``jax.vmap``.

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        A fitted MCMC object (after ``mcmc.run()``).
    participant_data_stacked : dict
        Dict mapping participant_id to a dict of pre-stacked arrays.
        Required keys per participant:

        - ``stimuli_stacked`` : ``(n_blocks, max_trials)``
        - ``actions_stacked`` : ``(n_blocks, max_trials)``
        - ``rewards_stacked`` : ``(n_blocks, max_trials)``
        - ``masks_stacked`` : ``(n_blocks, max_trials)``
        - ``set_sizes_stacked`` : ``(n_blocks, max_trials)`` (WMRL models only)

    model_name : str
        One of 'qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b'.
    num_stimuli : int, optional
        Number of unique stimuli per block. Default 6.
    num_actions : int, optional
        Number of possible actions. Default 3.
    q_init : float, optional
        Initial Q-value. Default 0.5.

    Returns
    -------
    jnp.ndarray
        Shape ``(chains, samples_per_chain, n_participants, n_blocks * max_trials)``.
        Padded trials carry log_prob = 0.0 (inherited from mask in likelihood).

    Notes
    -----
    Feed the output to :func:`build_inference_data_with_loglik` to get an
    ArviZ ``InferenceData`` object that supports ``az.waic()`` and ``az.loo()``.
    """
    # samples[param] has shape (chains, samples_per_chain, n_participants)
    samples = mcmc.get_samples(group_by_chain=True)
    param_names = _get_param_names(model_name)
    participant_ids = list(participant_data_stacked.keys())
    n_participants = len(participant_ids)

    # Build per-participant pointwise arrays, one participant at a time.
    # Each call returns shape (chains, samples_per_chain, n_blocks * max_trials).
    per_participant_outputs = []

    for idx, pid in enumerate(participant_ids):
        pdata = participant_data_stacked[pid]

        # Extract per-participant parameter arrays: each has shape (chains, samples_per_chain)
        param_arrays = [samples[name][..., idx] for name in param_names]

        # Build the per-sample function for this participant (closes over pdata)
        per_sample_fn = _build_per_participant_fn(
            model_name, pdata, num_stimuli, num_actions, q_init
        )

        # vmap over samples_per_chain dimension (axis 0 after chain slice)
        vmapped_over_samples = jax.vmap(per_sample_fn)
        # vmap over chains dimension (axis 0)
        vmapped_over_chains = jax.vmap(vmapped_over_samples)

        # JIT for performance
        jitted = jax.jit(vmapped_over_chains)

        # Result: (chains, samples_per_chain, n_blocks * max_trials)
        result = jitted(*param_arrays)
        per_participant_outputs.append(result)

    # Stack: (chains, samples_per_chain, n_participants, n_blocks * max_trials)
    return jnp.stack(per_participant_outputs, axis=2)


def build_inference_data_with_loglik(
    mcmc: "MCMC",
    pointwise_log_lik: jnp.ndarray,
    *,
    participant_ids: list | None = None,
) -> "az.InferenceData":
    """Convert MCMC to ArviZ InferenceData and attach the log_likelihood group.

    After this, ``az.waic(idata)`` and ``az.loo(idata)`` work natively without
    raising 'log_likelihood group missing' warnings.

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        Fitted MCMC object.
    pointwise_log_lik : jnp.ndarray
        Output of :func:`compute_pointwise_log_lik` with shape
        ``(chains, samples_per_chain, n_participants, n_trials)``.
    participant_ids : list, optional
        Participant IDs for coord labels. Defaults to integer indices if None.

    Returns
    -------
    az.InferenceData
        InferenceData with ``posterior``, ``sample_stats``, and
        ``log_likelihood`` groups.

    Examples
    --------
    >>> loglik = compute_pointwise_log_lik(mcmc, pdata, "qlearning")
    >>> idata = build_inference_data_with_loglik(mcmc, loglik)
    >>> waic = az.waic(idata)
    >>> loo = az.loo(idata, pointwise=True)
    """
    import arviz as az

    n_participants = pointwise_log_lik.shape[2]
    n_trials = pointwise_log_lik.shape[3]

    if participant_ids is None:
        participant_ids = list(range(n_participants))

    idata = az.from_numpyro(mcmc)

    idata.add_groups(
        log_likelihood={"obs": pointwise_log_lik},
        coords={
            "participant": participant_ids,
            "trial": list(range(n_trials)),
        },
        dims={"obs": ["participant", "trial"]},
    )

    return idata


# ---------------------------------------------------------------------------
# Shrinkage diagnostics (HIER-08)
# ---------------------------------------------------------------------------


def compute_shrinkage_report(
    idata: "az.InferenceData",
    param_names: list[str],
) -> dict[str, float]:
    """Compute shrinkage for each individual-level parameter.

    Shrinkage measures how much the posterior individual differences are
    pulled toward the group mean relative to the total posterior variance.
    Values close to 1.0 indicate strong shrinkage (the model identifies the
    parameter well at the group level); values below 0.3 flag the parameter
    as poorly identified.

    Formula::

        shrinkage = 1 - var_indiv / (var_group_mean + 1e-10)

    Where:

    - ``var_indiv`` = variance of ALL individual-level draws
      (across all MCMC draws AND all participants)
    - ``var_group_mean`` = variance of the per-draw group mean
      (how much the group mean shifts across MCMC iterations)

    Parameters
    ----------
    idata : az.InferenceData
        ArviZ InferenceData with a ``posterior`` group containing individual-
        level parameter arrays of shape ``(chains, draws, n_participants)``.
    param_names : list[str]
        Names of individual-level parameters to compute shrinkage for.
        Each must be a key in ``idata.posterior``.

    Returns
    -------
    dict[str, float]
        Mapping from parameter name to shrinkage value in ``(-inf, 1.0]``.
        Negative values are theoretically possible but indicate a poorly
        specified model (individual variance exceeds group variance).

    Notes
    -----
    Uses ``numpy`` (not ``jax.numpy``) because ArviZ arrays are NumPy-backed.
    """
    results: dict[str, float] = {}
    posterior = idata.posterior
    for param in param_names:
        arr = posterior[param].values  # (chains, draws, n_participants)
        flat = arr.reshape(-1, arr.shape[-1])  # (total_draws, n_participants)
        var_indiv = float(np.var(flat))
        var_group_mean = float(np.var(flat.mean(axis=1)))
        shrinkage = 1.0 - var_indiv / (var_group_mean + 1e-10)
        results[param] = shrinkage
    return results


def write_shrinkage_report(
    shrinkage: dict[str, float],
    output_path: Path,
    *,
    threshold: float = 0.3,
) -> Path:
    """Write a markdown shrinkage diagnostic report.

    Parameters
    ----------
    shrinkage : dict[str, float]
        Output of :func:`compute_shrinkage_report`.
    output_path : Path
        File path for the markdown report.
    threshold : float
        Shrinkage threshold below which a parameter is flagged as poorly
        identified.  Default 0.3.

    Returns
    -------
    Path
        The path to the written report (same as ``output_path``).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_identified = sum(1 for v in shrinkage.values() if v >= threshold)
    n_poor = len(shrinkage) - n_identified

    lines = [
        "# Shrinkage Diagnostic Report\n",
        "Formula: `Shrinkage = 1 - var(individual draws) / var(per-draw group mean)`\n",
        "- `var(individual draws)`: variance across all MCMC draws AND all participants",
        "- `var(per-draw group mean)`: variance of the per-draw group mean across iterations\n",
        "| Parameter | Shrinkage | Status |",
        "|-----------|-----------|--------|",
    ]
    for param, val in shrinkage.items():
        status = "identified" if val >= threshold else "WARNING: poorly identified"
        lines.append(f"| {param} | {val:.4f} | {status} |")

    lines += [
        "",
        f"**Summary:** {n_identified}/{len(shrinkage)} parameters identified "
        f"(shrinkage >= {threshold}); {n_poor} poorly identified.",
        "",
        "> Parameters with shrinkage < 0.3 should be treated as descriptive only "
        "for downstream inference.",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Padding filter for WAIC/LOO (fixes Pitfall 4 from RESEARCH.md)
# ---------------------------------------------------------------------------


def filter_padding_from_loglik(
    pointwise_loglik: jnp.ndarray,
    participant_data_stacked: dict,
) -> jnp.ndarray:
    """Replace padded trial positions with NaN so WAIC/LOO ignores them.

    Padded positions in the stacked likelihood output carry
    ``log_prob = 0.0`` (inherited from the mask in the JAX likelihood
    functions).  Passing these zeros to ``az.waic()`` or ``az.loo()``
    inflates the effective parameter count.  This function sets masked
    positions to ``NaN``, which ArviZ 0.23.4 silently drops.

    Parameters
    ----------
    pointwise_loglik : jnp.ndarray
        Shape ``(chains, samples, n_participants, n_blocks * max_trials)``.
        Output of :func:`compute_pointwise_log_lik`.
    participant_data_stacked : dict
        Mapping from participant_id to stacked arrays.  The ``masks_stacked``
        key holds a float32 array of shape ``(n_blocks, max_trials)`` where
        1.0 = real trial and 0.0 = padding.

    Returns
    -------
    jnp.ndarray
        Same shape as ``pointwise_loglik`` but with padding positions set to
        ``NaN``.  Real-trial positions are unchanged.

    Notes
    -----
    The returned array is a NumPy array (not a JAX DeviceArray) because
    downstream ArviZ operations work on NumPy-backed data.
    """
    result = np.array(pointwise_loglik, dtype=np.float32)

    for idx, pid in enumerate(sorted(participant_data_stacked.keys())):
        mask = np.array(
            participant_data_stacked[pid]["masks_stacked"], dtype=np.float32
        )  # (n_blocks, max_trials)
        flat_mask = mask.reshape(-1)  # (n_blocks * max_trials,)

        # Where mask == 0, set to NaN
        padding_indices = np.where(flat_mask == 0.0)[0]
        result[:, :, idx, padding_indices] = np.nan

    return result
