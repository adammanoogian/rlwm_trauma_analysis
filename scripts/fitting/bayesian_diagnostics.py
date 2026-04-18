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
    import pandas as pd
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
    # Each call returns shape (chains, samples_per_chain, n_blocks_i * max_trials)
    # where n_blocks_i may differ across participants.
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

        # Result: (chains, samples_per_chain, n_blocks_i * max_trials)
        result = jitted(*param_arrays)
        per_participant_outputs.append(result)

    # Participants may have different n_blocks (e.g., 12 vs 17), producing
    # different trial-dimension lengths.  Pad shorter arrays with 0.0
    # (masked trials already contribute 0.0) so jnp.stack succeeds.
    max_trials_dim = max(arr.shape[-1] for arr in per_participant_outputs)
    padded_outputs = []
    for arr in per_participant_outputs:
        pad_width = max_trials_dim - arr.shape[-1]
        if pad_width > 0:
            arr = jnp.pad(arr, ((0, 0), (0, 0), (0, pad_width)), constant_values=0.0)
        padded_outputs.append(arr)

    # Stack: (chains, samples_per_chain, n_participants, max_trials_dim)
    return jnp.stack(padded_outputs, axis=2)


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

    # az.from_numpyro may create a log_likelihood group from numpyro.factor sites
    # (per-participant scalar log-probs).  Those have the wrong shape for WAIC/LOO.
    # Overwrite with the proper pointwise array computed by compute_pointwise_log_lik.
    if "log_likelihood" in idata._groups:
        del idata["log_likelihood"]

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
    max_trials_dim = result.shape[-1]

    for idx, pid in enumerate(sorted(participant_data_stacked.keys())):
        mask = np.array(
            participant_data_stacked[pid]["masks_stacked"], dtype=np.float32
        )  # (n_blocks_i, max_trials)
        flat_mask = mask.reshape(-1)  # (n_blocks_i * max_trials,)

        # Pad flat_mask to match the padded trial dimension (participants
        # with fewer blocks have shorter masks than max_trials_dim).
        if flat_mask.shape[0] < max_trials_dim:
            flat_mask = np.pad(
                flat_mask, (0, max_trials_dim - flat_mask.shape[0]), constant_values=0.0
            )

        # Where mask == 0, set to NaN
        padding_indices = np.where(flat_mask == 0.0)[0]
        result[:, :, idx, padding_indices] = np.nan

    return result


# ---------------------------------------------------------------------------
# Posterior predictive check (HIER-09)
# ---------------------------------------------------------------------------


def run_posterior_predictive_check(
    mcmc: "MCMC",
    participant_data_stacked: dict,
    model_name: str,
    data_df: "pd.DataFrame",
    *,
    group_col: str = "hypothesis_group",
    participant_col: str = "sona_id",
    block_col: str = "block",
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    n_ppc_samples: int = 200,
    output_dir: Path | None = None,
    ppc_output_dir: Path | None = None,
) -> dict:
    """Compute group-stratified posterior predictive checks for block accuracy.

    For each posterior draw, computes the model's per-trial predicted probability
    (i.e. ``exp(log_prob)`` of the observed action), then averages within blocks
    to obtain predicted accuracy.  This is the *calibration* PPC: does the model's
    average predicted probability for the observed action match the observed accuracy?

    Observed accuracy is computed from ``data_df`` as the mean of the ``correct``
    column (1 = correct, 0 = incorrect) per block per trauma group.

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        A fitted MCMC object after ``mcmc.run()``.
    participant_data_stacked : dict
        Dict mapping participant_id to stacked arrays (output of
        ``prepare_stacked_participant_data``).
    model_name : str
        One of ``'qlearning'``, ``'wmrl'``, ``'wmrl_m3'``, etc.
    data_df : pd.DataFrame
        Original trial-level DataFrame containing at minimum the columns
        ``participant_col``, ``block_col``, ``group_col``, and ``'correct'``.
    group_col : str
        Column in ``data_df`` identifying the trauma group label.
    participant_col : str
        Column in ``data_df`` for participant IDs.
    block_col : str
        Column in ``data_df`` for block number.
    num_stimuli : int
        Number of unique stimuli per block.
    num_actions : int
        Number of possible actions.
    q_init : float
        Initial Q-value.
    n_ppc_samples : int
        Number of posterior draws to use for the PPC (subsampled from the full
        posterior for efficiency).
    output_dir : Path or None
        If provided (and ``ppc_output_dir`` is None), saves
        ``output_dir / "bayesian" / "{model_name}_ppc_results.csv"``. Legacy
        path — Phase 16 callers still work.
    ppc_output_dir : Path or None
        If provided, saves ``ppc_output_dir / "{model_name}_ppc_results.csv"``
        verbatim (no ``"bayesian"`` prefix appended). Takes precedence over
        ``output_dir``. Used by Phase 21 (``save_results`` with
        ``output_subdir``) so the PPC CSV lands in the same subdirectory as the
        posterior NetCDF (e.g. ``output/bayesian/21_baseline/``) and does NOT
        leak into the legacy ``output/bayesian/`` root.

    Returns
    -------
    dict
        Keys:

        - ``"covered_count"`` : int — number of blocks (out of ``total_blocks``) where
          observed accuracy falls within the [2.5, 97.5] PPC envelope.
        - ``"total_blocks"`` : int — total blocks evaluated (main task only, >= block 3).
        - ``"ppc_results_df"`` : pd.DataFrame — one row per (group, block) with columns
          ``group``, ``block``, ``observed_accuracy``, ``ppc_median``, ``ppc_2.5``,
          ``ppc_97.5``, ``covered``.

    Notes
    -----
    Uses the *pragmatic* PPC approach: ``exp(log_prob_trial)`` is the model's
    probability of the *observed* action on that trial.  Averaging within a block
    gives the model's expected accuracy under the assumption that the observed data
    is a typical sample.  This is a standard calibration check — it measures whether
    the model's softmax probabilities are well-calibrated against observed choices.
    """
    import pandas as pd  # local import — keep TYPE_CHECKING clean

    samples = mcmc.get_samples(group_by_chain=True)
    param_names = _get_param_names(model_name)
    participant_ids = sorted(participant_data_stacked.keys())
    n_participants = len(participant_ids)

    # Shape: (chains, draws_per_chain, ...) → flatten to (total_draws, ...)
    # We subsample n_ppc_samples from the flattened draws for efficiency.
    n_chains = samples[param_names[0]].shape[0]
    n_draws = samples[param_names[0]].shape[1]
    total_draws = n_chains * n_draws
    rng = np.random.default_rng(seed=0)
    draw_indices = rng.choice(total_draws, size=min(n_ppc_samples, total_draws), replace=False)

    # Flatten samples to (total_draws, n_participants)
    flat_params: dict[str, np.ndarray] = {}
    for name in param_names:
        arr = np.array(samples[name])  # (chains, draws, n_participants)
        flat_params[name] = arr.reshape(total_draws, n_participants)

    # Get block-level stacked structure — participants may have different n_blocks.
    # Use the max across all participants for the predicted_acc array.
    max_trials = participant_data_stacked[participant_ids[0]]["masks_stacked"].shape[1]
    max_n_blocks = max(
        participant_data_stacked[pid]["masks_stacked"].shape[0]
        for pid in participant_ids
    )

    # predicted_acc[draw_idx, participant_idx, block_idx] = mean exp(logprob) in block
    predicted_acc = np.full(
        (len(draw_indices), n_participants, max_n_blocks), np.nan, dtype=np.float32
    )

    for draw_i, draw_idx in enumerate(draw_indices):
        for p_idx, pid in enumerate(participant_ids):
            pdata = participant_data_stacked[pid]
            n_blocks_i = pdata["masks_stacked"].shape[0]

            # Extract scalar params for this draw and participant
            param_scalars = [
                float(flat_params[name][draw_idx, p_idx]) for name in param_names
            ]

            per_sample_fn = _build_per_participant_fn(
                model_name, pdata, num_stimuli, num_actions, q_init
            )
            # pointwise: (n_blocks_i * max_trials,) — log P(observed_action | params)
            pointwise = per_sample_fn(*param_scalars)  # type: ignore[arg-type]
            pointwise_np = np.array(pointwise, dtype=np.float32)

            # Reshape back to (n_blocks_i, max_trials)
            pointwise_blocks = pointwise_np.reshape(n_blocks_i, max_trials)
            mask_blocks = np.array(
                pdata["masks_stacked"], dtype=np.float32
            )  # (n_blocks_i, max_trials)

            # exp(log_prob) = P(observed action | params)
            prob_blocks = np.exp(pointwise_blocks)

            # Mean over real (non-padded) trials per block
            for b_idx in range(n_blocks_i):
                real_mask = mask_blocks[b_idx] == 1.0
                if real_mask.sum() > 0:
                    predicted_acc[draw_i, p_idx, b_idx] = float(
                        prob_blocks[b_idx][real_mask].mean()
                    )

    # Build group assignment map: participant_id -> group label
    group_map: dict = {}
    if group_col in data_df.columns:
        for pid in participant_ids:
            rows = data_df[data_df[participant_col] == pid]
            if len(rows) > 0:
                group_map[pid] = rows[group_col].iloc[0]
            else:
                group_map[pid] = "unknown"
    else:
        for pid in participant_ids:
            group_map[pid] = "all"

    # Compute observed accuracy per block per group from data_df
    main_blocks = sorted(data_df[block_col].unique())
    # We only report blocks that are present in the data (main task)
    all_groups = sorted(set(group_map.values()))

    rows_ppc: list[dict] = []
    covered_count = 0
    total_blocks_evaluated = 0

    for group in all_groups:
        group_pids = [pid for pid, g in group_map.items() if g == group]
        group_pid_indices = [participant_ids.index(pid) for pid in group_pids if pid in participant_ids]

        for block in main_blocks:
            # Observed accuracy for this group, block
            mask_obs = (
                (data_df[participant_col].isin(group_pids))
                & (data_df[block_col] == block)
            )
            block_rows = data_df[mask_obs]
            if len(block_rows) == 0:
                continue

            if "correct" not in data_df.columns:
                continue

            observed_accuracy = float(block_rows["correct"].mean())

            # PPC: predicted_acc[:, group_pid_indices, block_idx_in_stacked]
            # We need to map block number to stacked block index.
            # The stacked data was built from the same sorted participant order.
            # Collect predicted accuracy across draws and group participants
            # Block index in stacked: blocks are 0-indexed from the minimum block in stacked data.
            # We use the block number directly as an offset relative to block 3 (main task start).
            # More robustly: find which stacked block index corresponds to this block number.
            # The stacked format uses ALL blocks present per participant; we identify the block
            # by finding its index in the sorted unique blocks of the first group participant.
            if len(group_pid_indices) == 0:
                continue

            # Find block index in stacked: iterate stacked blocks for one participant
            # The stacked data preserves block order from prepare_stacked_participant_data.
            # We use the observation that the block number order in stacked == sorted unique blocks.
            ref_pid = participant_ids[group_pid_indices[0]]
            ref_pdata = participant_data_stacked[ref_pid]
            ref_n_blocks = ref_pdata["masks_stacked"].shape[0]
            # Find the 0-based block index from sorted unique blocks for that participant
            ref_blocks_sorted = sorted(
                data_df[data_df[participant_col] == ref_pid][block_col].unique()
            )
            if block not in ref_blocks_sorted:
                continue
            b_stacked_idx = ref_blocks_sorted.index(block)

            if b_stacked_idx >= ref_n_blocks:
                continue

            # Gather predicted accuracy for this group and block across draws
            # Shape: (n_ppc_samples, len(group_pid_indices))
            group_ppc = predicted_acc[:, group_pid_indices, b_stacked_idx]  # (draws, n_group)
            # Flatten draws × group-participants
            group_ppc_flat = group_ppc.flatten()
            group_ppc_flat = group_ppc_flat[~np.isnan(group_ppc_flat)]

            if len(group_ppc_flat) == 0:
                continue

            ppc_2_5 = float(np.percentile(group_ppc_flat, 2.5))
            ppc_50 = float(np.percentile(group_ppc_flat, 50.0))
            ppc_97_5 = float(np.percentile(group_ppc_flat, 97.5))
            covered = bool(ppc_2_5 <= observed_accuracy <= ppc_97_5)

            rows_ppc.append({
                "group": group,
                "block": block,
                "observed_accuracy": observed_accuracy,
                "ppc_median": ppc_50,
                "ppc_2.5": ppc_2_5,
                "ppc_97.5": ppc_97_5,
                "covered": covered,
            })

            covered_count += int(covered)
            total_blocks_evaluated += 1

    ppc_df = pd.DataFrame(rows_ppc)
    print(
        f"\n[PPC] {covered_count}/{total_blocks_evaluated} blocks covered "
        f"by 95% PPC envelope"
    )

    if ppc_output_dir is not None:
        bayesian_out = Path(ppc_output_dir)
        bayesian_out.mkdir(parents=True, exist_ok=True)
        ppc_csv = bayesian_out / f"{model_name}_ppc_results.csv"
        ppc_df.to_csv(ppc_csv, index=False)
        print(f"[PPC] Results saved: {ppc_csv}")
    elif output_dir is not None:
        bayesian_out = Path(output_dir) / "bayesian"
        bayesian_out.mkdir(parents=True, exist_ok=True)
        ppc_csv = bayesian_out / f"{model_name}_ppc_results.csv"
        ppc_df.to_csv(ppc_csv, index=False)
        print(f"[PPC] Results saved: {ppc_csv}")

    return {
        "covered_count": covered_count,
        "total_blocks": total_blocks_evaluated,
        "ppc_results_df": ppc_df,
    }
