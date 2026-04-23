"""M4 RLWM-LBA: joint choice+RT model using the Linear Ballistic Accumulator.

Canonical home for M4's NumPyro hierarchical wrapper. Callers should
import directly from this module; the legacy
``rlwm.fitting.numpyro_models`` re-export shim was deleted in the v5.0
shim cleanup.

M4 is the only joint choice+RT model in the project; its AIC is NOT comparable
to choice-only models (M1, M2, M3, M5, M6a, M6b).  There is no JAX likelihood
function for M4 — the LBA log-density is computed directly inside the NumPyro
hierarchical model.
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd

from ..core import MAX_TRIALS_PER_BLOCK, pad_block_to_max, prepare_stacked_participant_data

__all__ = [
    "prepare_stacked_participant_data_m4",
    "wmrl_m4_hierarchical_model",
]


def prepare_stacked_participant_data_m4(
    data_df: pd.DataFrame,
    participant_col: str = "sona_id",
    block_col: str = "block",
    stimulus_col: str = "stimulus",
    action_col: str = "key_press",
    reward_col: str = "reward",
    set_size_col: str = "set_size",
    rt_col: str = "rt",
) -> dict[Any, dict[str, jnp.ndarray]]:
    """Prepare stacked participant data for the M4 hierarchical LBA model.

    Extends :func:`prepare_stacked_participant_data` with RT extraction, outlier
    masking, and float64 RT stacking required by ``wmrl_m4_hierarchical_model``.

    For each trial block the function:

    1. Extracts RTs from ``rt_col`` (assumed milliseconds).
    2. Calls ``preprocess_rt_block`` to convert to seconds and flag outliers.
    3. Pads RTs to ``MAX_TRIALS_PER_BLOCK`` with 0.5 (safe masked-out value).
    4. ANDs the padding mask with the RT-outlier mask so that padding trials
       **and** RT outliers both contribute 0 to the likelihood.

    Parameters
    ----------
    data_df : pd.DataFrame
        Trial-level data with participant, block, stimulus, action, reward,
        set_size, and rt columns.
    participant_col : str
        Column name for participant identifier.  Default ``"sona_id"``.
    block_col : str
        Column name for block number.  Default ``"block"``.
    stimulus_col : str
        Column name for stimulus index.  Default ``"stimulus"``.
    action_col : str
        Column name for action taken.  Default ``"key_press"``.
    reward_col : str
        Column name for reward received.  Default ``"reward"``.
    set_size_col : str
        Column name for set size.  Default ``"set_size"``.
    rt_col : str
        Column name for reaction time in milliseconds.  Default ``"rt"``.

    Returns
    -------
    dict[Any, dict[str, jnp.ndarray]]
        Mapping from participant_id to a dict with keys:

        * ``stimuli_stacked``   -- shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``, int32
        * ``actions_stacked``   -- shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``, int32
        * ``rewards_stacked``   -- shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``, float32
        * ``set_sizes_stacked`` -- shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``, float32
        * ``masks_stacked``     -- shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``, float32
          (padding AND RT-outlier mask combined)
        * ``rts_stacked``       -- shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``, float64
          (seconds; padding positions filled with 0.5)

    Notes
    -----
    Lazy-imports ``preprocess_rt_block`` from ``lba_likelihood`` to avoid
    triggering float64 in choice-only import paths.

    Participant keys are sorted (same ordering as
    :func:`prepare_stacked_participant_data`) so that downstream arrays align
    with ``sorted(result.keys())``.
    """
    # Lazy import to avoid float64 contamination in choice-only paths
    from scripts.fitting.lba_likelihood import preprocess_rt_block

    participant_data: dict[Any, dict[str, jnp.ndarray]] = {}

    for participant_id in sorted(data_df[participant_col].unique()):
        ppt_df = data_df[data_df[participant_col] == participant_id]

        stimuli_blocks = []
        actions_blocks = []
        rewards_blocks = []
        set_sizes_blocks = []
        masks_blocks = []
        rts_blocks = []

        for block_num in sorted(ppt_df[block_col].unique()):
            block_df = ppt_df[ppt_df[block_col] == block_num]

            stim = jnp.array(block_df[stimulus_col].values, dtype=jnp.int32)
            act = jnp.array(block_df[action_col].values, dtype=jnp.int32)
            rew = jnp.array(block_df[reward_col].values, dtype=jnp.float32)

            if set_size_col in block_df.columns:
                ss = jnp.array(block_df[set_size_col].values, dtype=jnp.float32)
            else:
                ss = jnp.ones(len(stim), dtype=jnp.float32) * 6.0

            # pad_block_to_max returns (stim, act, rew, set_sizes_padded, mask)
            p_stim, p_act, p_rew, p_ss, p_mask = pad_block_to_max(
                stim, act, rew, set_sizes=ss
            )
            n_real = len(stim)

            # Extract and preprocess RT (milliseconds -> seconds + outlier mask)
            rt_raw = jnp.array(block_df[rt_col].values, dtype=jnp.float64)
            rt_sec, valid_rt = preprocess_rt_block(rt_raw)

            # Pad RT to MAX_TRIALS_PER_BLOCK with 0.5 (safe masked-out value)
            rt_padded = jnp.zeros(MAX_TRIALS_PER_BLOCK, dtype=jnp.float64)
            rt_padded = rt_padded.at[:n_real].set(rt_sec)
            # Fill padding positions with 0.5 to avoid t_star <= 0 in masked calls
            padding_fill = jnp.zeros(MAX_TRIALS_PER_BLOCK, dtype=jnp.float64)
            padding_fill = padding_fill.at[n_real:].set(0.5)
            rt_padded = rt_padded + padding_fill

            # Pad valid_rt mask to MAX_TRIALS_PER_BLOCK (False for padding positions)
            valid_rt_padded = jnp.zeros(MAX_TRIALS_PER_BLOCK, dtype=jnp.float32)
            valid_rt_padded = valid_rt_padded.at[:n_real].set(
                valid_rt.astype(jnp.float32)
            )

            # AND padding mask with RT-outlier mask
            combined_mask = p_mask * valid_rt_padded

            stimuli_blocks.append(p_stim)
            actions_blocks.append(p_act)
            rewards_blocks.append(p_rew)
            set_sizes_blocks.append(p_ss)
            masks_blocks.append(combined_mask)
            rts_blocks.append(rt_padded)

        participant_data[participant_id] = {
            "stimuli_stacked": jnp.stack(stimuli_blocks),
            "actions_stacked": jnp.stack(actions_blocks),
            "rewards_stacked": jnp.stack(rewards_blocks),
            "set_sizes_stacked": jnp.stack(set_sizes_blocks),
            "masks_stacked": jnp.stack(masks_blocks),
            "rts_stacked": jnp.stack(rts_blocks),
        }

    return participant_data

def wmrl_m4_hierarchical_model(
    participant_data_stacked: dict,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
) -> None:
    """Hierarchical Bayesian M4 (WM-RL+LBA) model.

    Implements a joint choice+RT hierarchical model combining the WM-RL
    hybrid model with Linear Ballistic Accumulator (LBA) decision dynamics.

    Uses the hBayesDM non-centered parameterization convention (Ahn, Haines, Zhang 2017):
    ``theta_unc = mu_pr + sigma_pr * z``,
    ``theta = lower + (upper - lower) * Phi_approx(theta_unc)``,
    where ``Phi_approx = jax.scipy.stats.norm.cdf``.

    K (capacity) is parameterized in [2, 6] following Senta, Bishop, Collins (2025).

    LBA parameters use log-normal non-centered parameterization:

    - ``v_scale``: drift-rate scale, log-normal in (0, inf).
    - ``A``: start-point noise width, log-normal in (0, inf).
    - ``delta`` (M4H-02): b - A gap, log-normal ensuring b > A by construction.
      Inside the participant loop: ``b = A + delta`` (decoded, NOT sampled directly).
    - ``t0``: non-decision time, bounded in [0.05, 0.3] via probit transform.

    Parameters
    ----------
    participant_data_stacked : dict
        Mapping from participant_id to stacked-format arrays (from
        ``prepare_stacked_participant_data_m4``).  Keys per participant:
        ``stimuli_stacked``, ``actions_stacked``, ``rewards_stacked``,
        ``set_sizes_stacked``, ``masks_stacked`` (RT-outlier combined),
        ``rts_stacked`` — each shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``.
    num_stimuli : int
        Number of distinct stimuli in the task.  Default 6.
    num_actions : int
        Number of possible actions.  Default 3.
    q_init : float
        Initial Q-value for all state-action pairs.  Default 0.5.
    wm_init : float
        Initial WM values (uniform baseline ``1/nA``).  Default ``1/3``.

    Notes
    -----
    - Six RLWM params (alpha_pos, alpha_neg, phi, rho, capacity, kappa) are
      sampled via ``sample_bounded_param`` from ``numpyro_helpers``.
    - No ``epsilon`` parameter: M4 is the joint choice+RT model and uses the
      LBA decision process directly (not the noisy softmax).
    - No Level-2 regression: M4 hierarchical model is unconditional.
    - ``b = A + delta`` decode happens INSIDE the participant for-loop, not in
      the likelihood.  This guarantees ``b > A`` by construction (M4H-02).
    - ``wmrl_m4_multiblock_likelihood_stacked`` is lazy-imported inside the
      function body to avoid triggering float64 in choice-only import paths.
    - Likelihood is accumulated via ``numpyro.factor`` in a Python for-loop.
    - Do NOT modify this function's API: ``fit_bayesian.py`` dispatches to it
      by name.
    """
    # Lazy imports to avoid float64 contamination in choice-only import paths
    from scripts.fitting.lba_likelihood import wmrl_m4_multiblock_likelihood_stacked
    from rlwm.fitting.numpyro_helpers import (
        PARAM_PRIOR_DEFAULTS,
        phi_approx,
        sample_bounded_param,
    )

    n_participants = len(participant_data_stacked)
    participant_ids = sorted(participant_data_stacked.keys())

    # ------------------------------------------------------------------
    # Group priors for 6 RLWM parameters via sample_bounded_param
    # Note: no epsilon in M4 (LBA handles decision noise directly)
    # ------------------------------------------------------------------
    sampled: dict[str, jnp.ndarray] = {}
    for param in ["alpha_pos", "alpha_neg", "phi", "rho", "capacity", "kappa"]:
        defaults = PARAM_PRIOR_DEFAULTS[param]
        sampled[param] = sample_bounded_param(
            param,
            lower=defaults["lower"],
            upper=defaults["upper"],
            n_participants=n_participants,
            mu_prior_loc=defaults["mu_prior_loc"],
        )

    # ------------------------------------------------------------------
    # v_scale: log-normal non-centered
    # mu ~ Normal(log(3.0), 0.5) -> group mean drift ~ 3.0 Hz
    # ------------------------------------------------------------------
    log_v_mu_pr = numpyro.sample("log_v_mu_pr", dist.Normal(jnp.log(3.0), 0.5))
    log_v_sigma_pr = numpyro.sample("log_v_sigma_pr", dist.HalfNormal(0.2))
    log_v_z = numpyro.sample(
        "log_v_z", dist.Normal(0, 1).expand([n_participants])
    )
    sampled["v_scale"] = numpyro.deterministic(
        "v_scale", jnp.exp(log_v_mu_pr + log_v_sigma_pr * log_v_z)
    )

    # ------------------------------------------------------------------
    # A: start-point noise width, log-normal non-centered
    # mu ~ Normal(log(0.3), 0.5) -> group mean A ~ 0.3 s
    # ------------------------------------------------------------------
    log_A_mu_pr = numpyro.sample("log_A_mu_pr", dist.Normal(jnp.log(0.3), 0.5))
    log_A_sigma_pr = numpyro.sample("log_A_sigma_pr", dist.HalfNormal(0.2))
    log_A_z = numpyro.sample(
        "log_A_z", dist.Normal(0, 1).expand([n_participants])
    )
    sampled["A"] = numpyro.deterministic(
        "A", jnp.exp(log_A_mu_pr + log_A_sigma_pr * log_A_z)
    )

    # ------------------------------------------------------------------
    # delta: b - A gap, log-normal non-centered (M4H-02)
    # b = A + delta guarantees b > A by construction.
    # mu ~ Normal(0.0, 1.0) -> median delta ~ 1.0
    # ------------------------------------------------------------------
    log_delta_mu_pr = numpyro.sample("log_delta_mu_pr", dist.Normal(0.0, 1.0))
    log_delta_sigma_pr = numpyro.sample("log_delta_sigma_pr", dist.HalfNormal(0.2))
    log_delta_z = numpyro.sample(
        "log_delta_z", dist.Normal(0, 1).expand([n_participants])
    )
    sampled["delta"] = numpyro.deterministic(
        "delta", jnp.exp(log_delta_mu_pr + log_delta_sigma_pr * log_delta_z)
    )

    # ------------------------------------------------------------------
    # t0: non-decision time, bounded in [0.05, 0.3] via probit transform
    # mu ~ Normal(0.0, 1.0) -> group mean t0 near midpoint of [0.05, 0.3]
    # ------------------------------------------------------------------
    t0_mu_pr = numpyro.sample("t0_mu_pr", dist.Normal(0.0, 1.0))
    t0_sigma_pr = numpyro.sample("t0_sigma_pr", dist.HalfNormal(0.2))
    t0_z = numpyro.sample(
        "t0_z", dist.Normal(0, 1).expand([n_participants])
    )
    sampled["t0"] = numpyro.deterministic(
        "t0", 0.05 + 0.25 * phi_approx(t0_mu_pr + t0_sigma_pr * t0_z)
    )

    # ------------------------------------------------------------------
    # Likelihood via numpyro.factor — Python for-loop over participants
    # CRITICAL: decode b = A + delta INSIDE the loop (not in likelihood)
    # This guarantees b > A by construction (M4H-02).
    # ------------------------------------------------------------------
    for idx, pid in enumerate(participant_ids):
        pdata = participant_data_stacked[pid]
        A_i = sampled["A"][idx]
        delta_i = sampled["delta"][idx]
        b_i = A_i + delta_i  # b > A guaranteed by delta > 0

        log_lik = wmrl_m4_multiblock_likelihood_stacked(
            stimuli_stacked=pdata["stimuli_stacked"],
            actions_stacked=pdata["actions_stacked"],
            rewards_stacked=pdata["rewards_stacked"],
            set_sizes_stacked=pdata["set_sizes_stacked"],
            rts_stacked=pdata["rts_stacked"],
            masks_stacked=pdata["masks_stacked"],
            alpha_pos=sampled["alpha_pos"][idx],
            alpha_neg=sampled["alpha_neg"][idx],
            phi=sampled["phi"][idx],
            rho=sampled["rho"][idx],
            capacity=sampled["capacity"][idx],
            kappa=sampled["kappa"][idx],
            v_scale=sampled["v_scale"][idx],
            A=A_i,
            b=b_i,
            t0=sampled["t0"][idx],
            num_stimuli=num_stimuli,
            num_actions=num_actions,
            q_init=q_init,
            wm_init=wm_init,
        )
        numpyro.factor(f"obs_p{pid}", log_lik)
