"""M6b WM-RL + dual perseveration (kappa_total, kappa_share): JAX likelihoods + NumPyro wrappers.

Canonical home for M6b's JAX likelihoods and NumPyro wrappers. Callers
should import directly from this module; the legacy
``rlwm.fitting.jax_likelihoods`` / ``rlwm.fitting.numpyro_models``
re-export shims were deleted in the v5.0 shim cleanup.

M6b parameterizes perseveration as ``kappa_total`` (total magnitude) and
``kappa_share`` (fraction routed to global vs. stimulus-specific tracks).
At ``kappa_share=0`` M6b reduces to M6a; at ``kappa_share=1`` to M3.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import lax

from ..core import (
    DEFAULT_EPSILON,
    FIXED_BETA,
    MAX_BLOCKS,
    MAX_TRIALS_PER_BLOCK,
    NUM_ACTIONS,
    affine_scan,
    apply_epsilon_noise,
    associative_scan_q_update,
    associative_scan_wm_update,
    pad_block_to_max,
    precompute_last_action_global,
    precompute_last_actions_per_stimulus,
    prepare_stacked_participant_data,
    softmax_policy,
    stack_across_participants,
)

__all__ = [
    "wmrl_m6b_fully_batched_likelihood",
    "wmrl_m6b_block_likelihood",
    "wmrl_m6b_multiblock_likelihood",
    "wmrl_m6b_multiblock_likelihood_stacked",
    "wmrl_m6b_block_likelihood_pscan",
    "wmrl_m6b_multiblock_likelihood_stacked_pscan",
    "test_wmrl_m6b_single_block",
    "test_wmrl_m6b_kappa_share_one_matches_m3",
    "test_wmrl_m6b_kappa_share_zero_matches_m6a",
    "test_padding_equivalence_wmrl_m6b",
    "wmrl_m6b_hierarchical_model",
    "wmrl_m6b_hierarchical_model_subscale",
]


def wmrl_m6b_fully_batched_likelihood(
    stimuli: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    set_sizes: jnp.ndarray,
    masks: jnp.ndarray,
    alpha_pos: jnp.ndarray,
    alpha_neg: jnp.ndarray,
    phi: jnp.ndarray,
    rho: jnp.ndarray,
    capacity: jnp.ndarray,
    kappa: jnp.ndarray,
    kappa_s: jnp.ndarray,
    epsilon: jnp.ndarray,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    use_pscan: bool = False,
) -> jnp.ndarray:
    """Fully-batched WM-RL+dual-perseveration (M6b) log-likelihood via nested vmap.

    Takes DECODED ``kappa`` and ``kappa_s`` (shape (N,) each).  Stick-breaking
    reparameterization (kappa = kappa_total * kappa_share,
    kappa_s = kappa_total * (1 - kappa_share)) must happen in the caller,
    matching the block-likelihood convention.

    Parameters
    ----------
    stimuli, actions, rewards, set_sizes, masks : jnp.ndarray
        Shape (N, B, T).
    alpha_pos, alpha_neg, phi, rho, capacity, kappa, kappa_s, epsilon : jnp.ndarray
        Shape (N,) per-participant parameter vectors.  ``kappa`` and
        ``kappa_s`` are already decoded from the stick-breaking variables.

    Returns
    -------
    jnp.ndarray
        Shape (N,) — total log-likelihood per participant.
    """
    if use_pscan:
        raise NotImplementedError(
            "wmrl_m6b_fully_batched_likelihood: use_pscan=True is not supported."
        )

    def _block_ll(stim, act, rew, ss, mask, ap, an, ph, rh, cap, k, ks, e):
        return wmrl_m6b_block_likelihood(
            stimuli=stim,
            actions=act,
            rewards=rew,
            set_sizes=ss,
            alpha_pos=ap,
            alpha_neg=an,
            phi=ph,
            rho=rh,
            capacity=cap,
            kappa=k,
            kappa_s=ks,
            epsilon=e,
            num_stimuli=num_stimuli,
            num_actions=num_actions,
            q_init=q_init,
            wm_init=wm_init,
            mask=mask,
            return_pointwise=False,
        )

    _over_blocks = jax.vmap(
        _block_ll,
        in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None, None, None),
        out_axes=0,
    )

    def _participant_ll(stim, act, rew, ss, mask, ap, an, ph, rh, cap, k, ks, e):
        return _over_blocks(
            stim, act, rew, ss, mask, ap, an, ph, rh, cap, k, ks, e,
        ).sum()

    _over_participants = jax.vmap(
        _participant_ll,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        out_axes=0,
    )
    return _over_participants(
        stimuli, actions, rewards, set_sizes, masks,
        alpha_pos, alpha_neg, phi, rho, capacity, kappa, kappa_s, epsilon,
    )

def wmrl_m6b_block_likelihood(
    stimuli: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    set_sizes: jnp.ndarray,
    alpha_pos: float,
    alpha_neg: float,
    phi: float,
    rho: float,
    capacity: float,
    kappa: float,   # DECODED global perseveration (= kappa_total * kappa_share)
    kappa_s: float, # DECODED stimulus-specific perseveration (= kappa_total * (1 - kappa_share))
    epsilon: float = DEFAULT_EPSILON,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,  # WM baseline = 1/nA (uniform)
    mask: jnp.ndarray = None,
    *,
    return_pointwise: bool = False,
) -> float | tuple[float, jnp.ndarray]:
    """
    Compute log-likelihood for WM-RL M6b model with DUAL perseveration.

    M6b combines M3's global kernel (kappa) and M6a's stimulus-specific kernel (kappa_s)
    in a single model. The constraint kappa + kappa_s <= 1 is enforced externally via
    stick-breaking reparameterization in the objective functions (kappa = kappa_total *
    kappa_share; kappa_s = kappa_total * (1 - kappa_share)).

    CRITICAL: This function takes DECODED kappa and kappa_s directly. The stick-breaking
    decode happens in the objective functions (_make_jax_objective_wmrl_m6b, etc.),
    NOT in this function.

    DUAL CARRY: Tracks both global last_action (scalar, M3-style) and per-stimulus
    last_actions (array shape num_stimuli, M6a-style) independently.

    Choice equation:
        P = (1 - eff_kappa - eff_kappa_s) * P_noisy
            + eff_kappa * Ck_global
            + eff_kappa_s * Ck_stim

    Where eff_kappa/eff_kappa_s are zero-gated when the respective kernel is
    unavailable (first trial, or kappa=0/kappa_s=0).

    When kappa_s=0, reduces to M3 (global only).
    When kappa=0, reduces to M6a (stimulus-specific only).
    When kappa_total=0 (both=0), reduces to M2 (no perseveration).

    Parameters
    ----------
    stimuli : array, shape (n_trials,)
    actions : array, shape (n_trials,)
    rewards : array, shape (n_trials,)
    set_sizes : array, shape (n_trials,)
    alpha_pos : float
    alpha_neg : float
    phi : float (WM decay)
    rho : float (WM reliance)
    capacity : float (WM capacity)
    kappa : float
        DECODED global perseveration weight (= kappa_total * kappa_share)
    kappa_s : float
        DECODED stimulus-specific perseveration weight (= kappa_total * (1 - kappa_share))
    epsilon : float
    num_stimuli : int
    num_actions : int
    q_init : float
    wm_init : float
    mask : array, shape (n_trials,), optional
    return_pointwise : bool, optional
        If True, return (total_log_lik, per_trial_log_probs) tuple instead of
        scalar. per_trial_log_probs has shape (max_trials,) where
        max_trials = MAX_TRIALS_PER_BLOCK (100). Padding trials (where mask=0)
        have log_prob = 0.0. The sum of per_trial_log_probs equals total_log_lik.
        Default False for backward compatibility with MLE callers.

    Returns
    -------
    float or tuple[float, jnp.ndarray]
        If return_pointwise=False (default): total log-likelihood (scalar).
        If return_pointwise=True: (total_log_lik, per_trial_log_probs) where
        per_trial_log_probs has shape (max_trials,) with 0.0 for padding entries.
    """
    # Initialize Q-table and WM matrix
    Q_init = jnp.ones((num_stimuli, num_actions)) * q_init
    WM_init = jnp.ones((num_stimuli, num_actions)) * wm_init
    WM_0 = jnp.ones((num_stimuli, num_actions)) * wm_init  # Baseline for decay

    # DUAL CARRY: Both global (M3-style) and per-stimulus (M6a-style) tracking
    last_action_init = -1  # scalar int, global (resets to -1 at block start)
    last_actions_init = jnp.full((num_stimuli,), -1, dtype=jnp.int32)  # (num_stimuli,)

    # Initial carry: (Q, WM, WM_0, log_likelihood, last_action_scalar, last_actions_array)
    init_carry = (Q_init, WM_init, WM_0, 0.0, last_action_init, last_actions_init)

    # If no mask provided, all trials are valid (backward compatibility)
    if mask is None:
        mask = jnp.ones(len(stimuli))

    scan_inputs = (stimuli, actions, rewards, set_sizes, mask)

    def step_fn(carry, inputs):
        Q_table, WM_table, WM_baseline, log_lik_accum, last_action, last_actions = carry
        stimulus, action, reward, set_size, valid = inputs

        # =================================================================
        # 1. DECAY WM
        # =================================================================
        WM_decayed = (1 - phi) * WM_table + phi * WM_baseline

        # =================================================================
        # 2. COMPUTE HYBRID POLICY (M2 base)
        # =================================================================
        q_vals = Q_table[stimulus]
        wm_vals = WM_decayed[stimulus]

        omega = rho * jnp.minimum(1.0, capacity / set_size)

        rl_probs = softmax_policy(q_vals, FIXED_BETA)
        wm_probs = softmax_policy(wm_vals, FIXED_BETA)
        base_probs = omega * wm_probs + (1 - omega) * rl_probs
        base_probs = base_probs / jnp.sum(base_probs)  # Normalize

        # Apply epsilon noise to base policy
        P_noisy = apply_epsilon_noise(base_probs, epsilon, num_actions)

        # =================================================================
        # GLOBAL KERNEL (M3 component)
        # Apply only if last_action >= 0 (any action was taken in block) AND kappa > 0
        # =================================================================
        has_global = jnp.logical_and(kappa > 0.0, last_action >= 0)
        # Clamp: jnp.maximum prevents -1 from wrapping to last row of eye matrix
        Ck_global = jnp.eye(num_actions)[jnp.maximum(last_action, 0)]
        eff_kappa = jnp.where(has_global, kappa, 0.0)

        # =================================================================
        # STIMULUS-SPECIFIC KERNEL (M6a component)
        # Apply only if this stimulus was seen before in block AND kappa_s > 0
        # =================================================================
        last_action_s = last_actions[stimulus]
        has_stim = jnp.logical_and(kappa_s > 0.0, last_action_s >= 0)
        # Clamp prevents -1 wrapping when sentinel
        Ck_stim = jnp.eye(num_actions)[jnp.maximum(last_action_s, 0)]
        eff_kappa_s = jnp.where(has_stim, kappa_s, 0.0)

        # =================================================================
        # THREE-WAY BLEND: M2 base + global kernel + stim-specific kernel
        # After stick-breaking: kappa + kappa_s = kappa_total <= 1
        # So base_weight = 1 - eff_kappa - eff_kappa_s >= 0 always
        # =================================================================
        noisy_probs = (
            (1.0 - eff_kappa - eff_kappa_s) * P_noisy
            + eff_kappa * Ck_global
            + eff_kappa_s * Ck_stim
        )

        # Log probability of observed action
        log_prob = jnp.log(noisy_probs[action] + 1e-8)

        # Mask log probability: padding trials contribute 0 to likelihood
        log_prob_masked = log_prob * valid

        # =================================================================
        # 3. UPDATE WM: Immediate overwrite (masked)
        # =================================================================
        wm_current = WM_decayed[stimulus, action]
        WM_updated = WM_decayed.at[stimulus, action].set(
            jnp.where(valid, reward, wm_current)
        )

        # =================================================================
        # 4. UPDATE Q-TABLE: Asymmetric learning (masked)
        # =================================================================
        q_current = Q_table[stimulus, action]
        delta = reward - q_current
        alpha = jnp.where(delta > 0, alpha_pos, alpha_neg)
        q_updated = q_current + alpha * delta
        Q_updated = Q_table.at[stimulus, action].set(
            jnp.where(valid, q_updated, q_current)
        )

        # Accumulate log-likelihood
        log_lik_new = log_lik_accum + log_prob_masked

        # =================================================================
        # 5. UPDATE BOTH PERSEVERATION STATES
        # Global: update on every valid trial (same as M3)
        # Per-stimulus: update unconditionally on valid (same as M6a)
        # =================================================================
        new_last_action = jnp.where(valid, action, last_action).astype(jnp.int32)
        new_last_actions = last_actions.at[stimulus].set(
            jnp.where(valid, action, last_action_s).astype(jnp.int32)
        )

        return (Q_updated, WM_updated, WM_baseline, log_lik_new,
                new_last_action, new_last_actions), log_prob_masked

    # Run scan over trials
    (Q_final, WM_final, _, log_lik_total, _, _), log_probs = lax.scan(
        step_fn, init_carry, scan_inputs
    )

    if return_pointwise:
        return log_lik_total, log_probs
    return log_lik_total

def wmrl_m6b_multiblock_likelihood(
    stimuli_blocks: list,
    actions_blocks: list,
    rewards_blocks: list,
    set_sizes_blocks: list,
    alpha_pos: float,
    alpha_neg: float,
    phi: float,
    rho: float,
    capacity: float,
    kappa: float,   # DECODED global perseveration (= kappa_total * kappa_share)
    kappa_s: float, # DECODED stimulus-specific perseveration (= kappa_total * (1 - kappa_share))
    epsilon: float = DEFAULT_EPSILON,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    masks_blocks: list = None,
    verbose: bool = False,
    participant_id: str = None
) -> float:
    """
    Compute log-likelihood for WM-RL M6b (dual perseveration) across MULTIPLE BLOCKS.

    Q-values, WM, last_action, and last_actions all reset at each block boundary.
    Uses fori_loop fast path for uniformly-sized padded blocks, Python fallback otherwise.

    CRITICAL: kappa and kappa_s are DECODED values (not kappa_total/kappa_share).
    Caller must decode: kappa = kappa_total * kappa_share; kappa_s = kappa_total * (1 - kappa_share).
    """
    num_blocks = len(stimuli_blocks)

    if verbose:
        print(f"\n  >> Processing {num_blocks} blocks for participant {participant_id}...")

    # Check if blocks are uniformly sized (for JAX-native fori_loop)
    block_sizes = [len(b) for b in stimuli_blocks]
    blocks_uniform = len(set(block_sizes)) == 1

    if blocks_uniform and masks_blocks is not None:
        # FAST PATH: Use JAX-native fori_loop for padded data
        stimuli_stacked = jnp.stack(stimuli_blocks)
        actions_stacked = jnp.stack(actions_blocks)
        rewards_stacked = jnp.stack(rewards_blocks)
        set_sizes_stacked = jnp.stack(set_sizes_blocks)
        masks_stacked = jnp.stack(masks_blocks)

        def body_fn(block_idx, total_ll):
            block_ll = wmrl_m6b_block_likelihood(
                stimuli=stimuli_stacked[block_idx],
                actions=actions_stacked[block_idx],
                rewards=rewards_stacked[block_idx],
                set_sizes=set_sizes_stacked[block_idx],
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
                wm_init=wm_init,
                mask=masks_stacked[block_idx]
            )
            return total_ll + block_ll

        total_log_lik = lax.fori_loop(0, num_blocks, body_fn, 0.0)

    else:
        # FALLBACK PATH: Python loop for variable-sized blocks
        total_log_lik = 0.0

        if masks_blocks is None:
            masks_blocks = [None] * num_blocks

        for block_idx, (stim_block, act_block, rew_block, set_block, mask_block) in enumerate(
            zip(stimuli_blocks, actions_blocks, rewards_blocks, set_sizes_blocks, masks_blocks)
        ):
            block_log_lik = wmrl_m6b_block_likelihood(
                stimuli=stim_block,
                actions=act_block,
                rewards=rew_block,
                set_sizes=set_block,
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
                wm_init=wm_init,
                mask=mask_block
            )
            total_log_lik += block_log_lik

    if verbose:
        print(f"  >> Total log-likelihood: {float(total_log_lik):.2f} ({num_blocks} blocks)\n", flush=True)

    return total_log_lik

def wmrl_m6b_multiblock_likelihood_stacked(
    stimuli_stacked: jnp.ndarray,
    actions_stacked: jnp.ndarray,
    rewards_stacked: jnp.ndarray,
    set_sizes_stacked: jnp.ndarray,
    masks_stacked: jnp.ndarray,
    alpha_pos: float,
    alpha_neg: float,
    phi: float,
    rho: float,
    capacity: float,
    kappa: float,   # DECODED global perseveration (= kappa_total * kappa_share)
    kappa_s: float, # DECODED stimulus-specific perseveration (= kappa_total * (1 - kappa_share))
    epsilon: float = DEFAULT_EPSILON,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    *,
    return_pointwise: bool = False,
) -> float | tuple[float, jnp.ndarray]:
    """
    FAST WM-RL M6b multiblock likelihood that takes pre-stacked arrays directly.

    See wmrl_m6b_multiblock_likelihood for full documentation.
    This version avoids list/restack overhead inside JIT.
    kappa and kappa_s are DECODED values (kappa = kappa_total * kappa_share;
    kappa_s = kappa_total * (1 - kappa_share)).

    Parameters
    ----------
    return_pointwise : bool, optional
        If True, return (total_log_lik, per_trial_log_probs) tuple instead of
        scalar. per_trial_log_probs has shape (n_blocks * max_trials,) flattened.
        Default False for backward compatibility with MLE callers.
    """
    num_blocks = stimuli_stacked.shape[0]

    if return_pointwise:
        def scan_body(total_ll, block_idx):
            block_ll, block_probs = wmrl_m6b_block_likelihood(
                stimuli=stimuli_stacked[block_idx],
                actions=actions_stacked[block_idx],
                rewards=rewards_stacked[block_idx],
                set_sizes=set_sizes_stacked[block_idx],
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
                wm_init=wm_init,
                mask=masks_stacked[block_idx],
                return_pointwise=True,
            )
            return total_ll + block_ll, block_probs

        total_ll, all_block_probs = lax.scan(
            scan_body, 0.0, jnp.arange(num_blocks)
        )
        return total_ll, all_block_probs.reshape(-1)
    else:
        def body_fn(block_idx, total_ll):
            block_ll = wmrl_m6b_block_likelihood(
                stimuli=stimuli_stacked[block_idx],
                actions=actions_stacked[block_idx],
                rewards=rewards_stacked[block_idx],
                set_sizes=set_sizes_stacked[block_idx],
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
                wm_init=wm_init,
                mask=masks_stacked[block_idx]
            )
            return total_ll + block_ll

        return lax.fori_loop(0, num_blocks, body_fn, 0.0)

def wmrl_m6b_block_likelihood_pscan(
    stimuli: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    set_sizes: jnp.ndarray,
    alpha_pos: float,
    alpha_neg: float,
    phi: float,
    rho: float,
    capacity: float,
    kappa: float,
    kappa_s: float,
    epsilon: float = DEFAULT_EPSILON,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    mask: jnp.ndarray = None,
    *,
    return_pointwise: bool = False,
) -> float | tuple[float, jnp.ndarray]:
    """
    WM-RL M6b block likelihood using parallel scan (dual perseveration).

    Drop-in replacement for ``wmrl_m6b_block_likelihood``.

    Phase 1 (parallel): Pre-compute Q_for_policy[t] and wm_for_policy[t].
    Phase 2 (vectorized): Precompute both global and per-stimulus last_action
    arrays via ``precompute_last_action_global`` and
    ``precompute_last_actions_per_stimulus``, then compute hybrid WM-Q
    policy with dual perseveration kernels for all trials simultaneously.

    Parameters
    ----------
    stimuli, actions, rewards, set_sizes : arrays, shape (n_trials,)
    alpha_pos, alpha_neg, phi, rho, capacity, kappa, kappa_s, epsilon : float
    num_stimuli, num_actions : int
    q_init, wm_init : float
    mask : array, optional
    return_pointwise : bool, optional

    Returns
    -------
    float or tuple[float, jnp.ndarray]
        Same as ``wmrl_m6b_block_likelihood``.
    """
    if mask is None:
        mask = jnp.ones(len(stimuli))

    # Phase 1: parallel Q and WM trajectories
    T = stimuli.shape[0]
    Q_for_policy = associative_scan_q_update(
        stimuli, actions, rewards, mask,
        alpha_pos, alpha_neg, q_init,
        num_stimuli, num_actions,
    )

    wm_for_policy, _ = associative_scan_wm_update(
        stimuli, actions, rewards, mask,
        phi, wm_init, num_stimuli, num_actions,
    )

    # ------------------------------------------------------------------
    # Phase 2 (vectorized): hybrid policy + dual perseveration
    # ------------------------------------------------------------------
    t_idx = jnp.arange(T)
    q_vals = Q_for_policy[t_idx, stimuli]      # (T, A)
    wm_vals = wm_for_policy[t_idx, stimuli]    # (T, A)

    omega = rho * jnp.minimum(1.0, capacity / set_sizes)  # (T,)
    rl_probs = jax.vmap(softmax_policy, in_axes=(0, None))(q_vals, FIXED_BETA)
    wm_probs = jax.vmap(softmax_policy, in_axes=(0, None))(wm_vals, FIXED_BETA)
    base_probs = omega[:, None] * wm_probs + (1 - omega[:, None]) * rl_probs
    base_probs = base_probs / jnp.sum(base_probs, axis=-1, keepdims=True)

    P_noisy = jax.vmap(apply_epsilon_noise, in_axes=(0, None, None))(
        base_probs, epsilon, num_actions
    )  # (T, A)

    # Precompute BOTH global and per-stimulus last_action arrays
    last_action_global = precompute_last_action_global(actions, mask)  # (T,)
    last_action_stim = precompute_last_actions_per_stimulus(
        stimuli, actions, mask, num_stimuli
    )  # (T,)

    # Global kernel (M3 component)
    has_global = jnp.logical_and(kappa > 0.0, last_action_global >= 0)  # (T,)
    Ck_global = jnp.eye(num_actions)[jnp.maximum(last_action_global, 0)]  # (T, A)
    eff_kappa = jnp.where(has_global, kappa, 0.0)  # (T,)

    # Stimulus-specific kernel (M6a component)
    has_stim = jnp.logical_and(kappa_s > 0.0, last_action_stim >= 0)  # (T,)
    Ck_stim = jnp.eye(num_actions)[jnp.maximum(last_action_stim, 0)]  # (T, A)
    eff_kappa_s = jnp.where(has_stim, kappa_s, 0.0)  # (T,)

    noisy_probs = (
        (1.0 - eff_kappa[:, None] - eff_kappa_s[:, None]) * P_noisy
        + eff_kappa[:, None] * Ck_global
        + eff_kappa_s[:, None] * Ck_stim
    )  # (T, A)

    log_probs = jnp.log(noisy_probs[t_idx, actions] + 1e-8) * mask

    if return_pointwise:
        return jnp.sum(log_probs), log_probs
    return jnp.sum(log_probs)

def wmrl_m6b_multiblock_likelihood_stacked_pscan(
    stimuli_stacked: jnp.ndarray,
    actions_stacked: jnp.ndarray,
    rewards_stacked: jnp.ndarray,
    set_sizes_stacked: jnp.ndarray,
    masks_stacked: jnp.ndarray,
    alpha_pos: float,
    alpha_neg: float,
    phi: float,
    rho: float,
    capacity: float,
    kappa: float,
    kappa_s: float,
    epsilon: float = DEFAULT_EPSILON,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    *,
    return_pointwise: bool = False,
) -> float | tuple[float, jnp.ndarray]:
    """
    WM-RL M6b multiblock likelihood using parallel scan.

    Drop-in replacement for ``wmrl_m6b_multiblock_likelihood_stacked``.

    Parameters
    ----------
    stimuli_stacked, actions_stacked, rewards_stacked,
    set_sizes_stacked, masks_stacked : arrays, shape (n_blocks, max_trials)
    alpha_pos, alpha_neg, phi, rho, capacity, kappa, kappa_s, epsilon : float
    num_stimuli, num_actions : int
    q_init, wm_init : float
    return_pointwise : bool, optional

    Returns
    -------
    float or tuple[float, jnp.ndarray]
        Same as ``wmrl_m6b_multiblock_likelihood_stacked``.
    """
    num_blocks = stimuli_stacked.shape[0]

    if return_pointwise:
        def scan_body(total_ll, block_idx):
            block_ll, block_probs = wmrl_m6b_block_likelihood_pscan(
                stimuli=stimuli_stacked[block_idx],
                actions=actions_stacked[block_idx],
                rewards=rewards_stacked[block_idx],
                set_sizes=set_sizes_stacked[block_idx],
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
                wm_init=wm_init,
                mask=masks_stacked[block_idx],
                return_pointwise=True,
            )
            return total_ll + block_ll, block_probs

        total_ll, all_block_probs = lax.scan(
            scan_body, 0.0, jnp.arange(num_blocks)
        )
        return total_ll, all_block_probs.reshape(-1)
    else:
        def body_fn(block_idx, total_ll):
            block_ll = wmrl_m6b_block_likelihood_pscan(
                stimuli=stimuli_stacked[block_idx],
                actions=actions_stacked[block_idx],
                rewards=rewards_stacked[block_idx],
                set_sizes=set_sizes_stacked[block_idx],
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
                wm_init=wm_init,
                mask=masks_stacked[block_idx],
            )
            return total_ll + block_ll

        return lax.fori_loop(0, num_blocks, body_fn, 0.0)

def test_wmrl_m6b_single_block():
    """Smoke test for WM-RL M6b single block likelihood (dual perseveration)."""
    print("\nTesting WM-RL M6b single block likelihood (smoke test)...")

    key = jax.random.PRNGKey(42)
    n_trials = 30

    stimuli = jax.random.randint(key, (n_trials,), 0, 6)
    key, subkey = jax.random.split(key)
    actions = jax.random.randint(subkey, (n_trials,), 0, 3)
    key, subkey = jax.random.split(key)
    rewards = jax.random.bernoulli(subkey, 0.7, (n_trials,)).astype(jnp.float32)
    set_sizes = jnp.ones((n_trials,)) * 5

    # kappa_total=0.3, kappa_share=0.667 => kappa=0.2, kappa_s=0.1
    params = {
        'alpha_pos': 0.3,
        'alpha_neg': 0.1,
        'phi': 0.1,
        'rho': 0.7,
        'capacity': 4.0,
        'kappa': 0.2,   # decoded global
        'kappa_s': 0.1, # decoded stim-specific
        'epsilon': 0.05,
    }

    log_lik = wmrl_m6b_block_likelihood(
        stimuli, actions, rewards, set_sizes, **params
    )

    nll = -float(log_lik)
    is_finite = jnp.isfinite(log_lik)
    is_negative = log_lik < 0

    print(f"  WM-RL M6b log-likelihood: {float(log_lik):.4f}")
    print(f"  NLL: {nll:.4f}")
    print(f"  Is finite: {bool(is_finite)}")
    print(f"  Is negative (expected): {bool(is_negative)}")

    assert bool(is_finite), "M6b log-likelihood must be finite!"
    print("[OK] WM-RL M6b single block smoke test passed")
    return log_lik

def test_wmrl_m6b_kappa_share_one_matches_m3():
    """Verify M6b with kappa_share=1.0 reduces exactly to M3 (all budget to global)."""
    print("\nTesting WM-RL M6b kappa_share=1.0 matches M3...")

    key = jax.random.PRNGKey(123)
    n_trials = 50

    stimuli = jax.random.randint(key, (n_trials,), 0, 6)
    key, subkey = jax.random.split(key)
    actions = jax.random.randint(subkey, (n_trials,), 0, 3)
    key, subkey = jax.random.split(key)
    rewards = jax.random.bernoulli(subkey, 0.7, (n_trials,)).astype(jnp.float32)
    set_sizes = jnp.ones((n_trials,)) * 5

    shared_params = {
        'alpha_pos': 0.3,
        'alpha_neg': 0.1,
        'phi': 0.1,
        'rho': 0.7,
        'capacity': 4.0,
        'epsilon': 0.05,
    }

    # kappa_total=0.3, kappa_share=1.0 => kappa=0.3, kappa_s=0.0
    kappa_total = 0.3
    kappa_share = 1.0
    kappa = kappa_total * kappa_share        # = 0.3
    kappa_s = kappa_total * (1 - kappa_share)  # = 0.0

    log_lik_m6b = wmrl_m6b_block_likelihood(
        stimuli, actions, rewards, set_sizes,
        **shared_params, kappa=kappa, kappa_s=kappa_s
    )

    # M3 with same kappa=0.3
    log_lik_m3 = wmrl_m3_block_likelihood(
        stimuli, actions, rewards, set_sizes,
        **shared_params, kappa=kappa
    )

    diff = abs(float(log_lik_m6b) - float(log_lik_m3))
    print(f"  M6b (kappa_share=1.0): log-lik = {float(log_lik_m6b):.8f}")
    print(f"  M3 (kappa=0.3):        log-lik = {float(log_lik_m3):.8f}")
    print(f"  Difference: {diff:.2e}")

    assert diff < 1e-6, f"M6b kappa_share=1.0 must match M3! Diff={diff}"
    print("[OK] M6b kappa_share=1.0 matches M3 exactly (diff < 1e-6)")
    return diff

def test_wmrl_m6b_kappa_share_zero_matches_m6a():
    """Verify M6b with kappa_share=0.0 reduces exactly to M6a (all budget to stim-specific)."""
    print("\nTesting WM-RL M6b kappa_share=0.0 matches M6a...")

    key = jax.random.PRNGKey(456)
    n_trials = 50

    stimuli = jax.random.randint(key, (n_trials,), 0, 6)
    key, subkey = jax.random.split(key)
    actions = jax.random.randint(subkey, (n_trials,), 0, 3)
    key, subkey = jax.random.split(key)
    rewards = jax.random.bernoulli(subkey, 0.7, (n_trials,)).astype(jnp.float32)
    set_sizes = jnp.ones((n_trials,)) * 5

    shared_params = {
        'alpha_pos': 0.3,
        'alpha_neg': 0.1,
        'phi': 0.1,
        'rho': 0.7,
        'capacity': 4.0,
        'epsilon': 0.05,
    }

    # kappa_total=0.3, kappa_share=0.0 => kappa=0.0, kappa_s=0.3
    kappa_total = 0.3
    kappa_share = 0.0
    kappa = kappa_total * kappa_share        # = 0.0
    kappa_s = kappa_total * (1 - kappa_share)  # = 0.3

    log_lik_m6b = wmrl_m6b_block_likelihood(
        stimuli, actions, rewards, set_sizes,
        **shared_params, kappa=kappa, kappa_s=kappa_s
    )

    # M6a with same kappa_s=0.3
    log_lik_m6a = wmrl_m6a_block_likelihood(
        stimuli, actions, rewards, set_sizes,
        **shared_params, kappa_s=kappa_s
    )

    diff = abs(float(log_lik_m6b) - float(log_lik_m6a))
    print(f"  M6b (kappa_share=0.0): log-lik = {float(log_lik_m6b):.8f}")
    print(f"  M6a (kappa_s=0.3):     log-lik = {float(log_lik_m6a):.8f}")
    print(f"  Difference: {diff:.2e}")

    assert diff < 1e-6, f"M6b kappa_share=0.0 must match M6a! Diff={diff}"
    print("[OK] M6b kappa_share=0.0 matches M6a exactly (diff < 1e-6)")
    return diff

def test_padding_equivalence_wmrl_m6b():
    """
    Verify padded and unpadded WM-RL M6b likelihoods are mathematically equivalent.
    """
    print("\nTesting WM-RL M6b padding equivalence (CRITICAL)...")

    key = jax.random.PRNGKey(789)
    n_real_trials = 48

    # Generate real trial data
    stimuli = jax.random.randint(key, (n_real_trials,), 0, 6)
    key, subkey = jax.random.split(key)
    actions = jax.random.randint(subkey, (n_real_trials,), 0, 3)
    key, subkey = jax.random.split(key)
    rewards = jax.random.bernoulli(subkey, 0.7, (n_real_trials,)).astype(jnp.float32)
    set_sizes = jnp.full((n_real_trials,), 5, dtype=jnp.int32)

    params = {
        'alpha_pos': 0.3, 'alpha_neg': 0.1, 'phi': 0.1,
        'rho': 0.7, 'capacity': 4.0,
        'kappa': 0.2, 'kappa_s': 0.1, 'epsilon': 0.05
    }

    # Unpadded likelihood (original)
    log_lik_original = wmrl_m6b_block_likelihood(
        stimuli, actions, rewards, set_sizes, **params
    )

    # Padded likelihood (with mask)
    stimuli_pad, actions_pad, rewards_pad, set_sizes_pad, mask = pad_block_to_max(
        stimuli, actions, rewards, max_trials=100, set_sizes=set_sizes
    )
    log_lik_padded = wmrl_m6b_block_likelihood(
        stimuli_pad, actions_pad, rewards_pad, set_sizes_pad, **params, mask=mask
    )

    # Verify equivalence
    match = jnp.allclose(log_lik_original, log_lik_padded, rtol=1e-6)
    print(f"  Original (unpadded) log-lik: {float(log_lik_original):.8f}")
    print(f"  Padded (with mask) log-lik:  {float(log_lik_padded):.8f}")
    print(f"  Difference: {abs(float(log_lik_original - log_lik_padded)):.2e}")
    print(f"[OK] WM-RL M6b padding equivalence: {'PASSED' if match else 'FAILED'}")

    assert bool(match), "WM-RL M6b padded/unpadded must be IDENTICAL!"
    return match

def wmrl_m6b_hierarchical_model(
    participant_data_stacked: dict,
    covariate_lec: jnp.ndarray | None = None,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    use_pscan: bool = False,
    stacked_arrays: dict | None = None,
) -> None:
    """Hierarchical Bayesian M6b (WM-RL+dual perseveration) model with optional L2 regression.

    Uses the hBayesDM non-centered parameterization convention (Ahn, Haines, Zhang 2017):
    ``theta_unc = mu_pr + sigma_pr * z``,
    ``theta = lower + (upper - lower) * Phi_approx(theta_unc)``,
    where ``Phi_approx = jax.scipy.stats.norm.cdf``.

    K (capacity) is parameterized in [2, 6] following Senta, Bishop, Collins (2025).

    M6b uses a stick-breaking parameterization for dual perseveration:

    - ``kappa_total`` in [0, 1]: total perseveration budget (sampled manually)
    - ``kappa_share`` in [0, 1]: fraction allocated to global perseveration (sampled
      manually)

    Per participant, decoded values are:

    - ``kappa   = kappa_total * kappa_share``       (global perseveration)
    - ``kappa_s = kappa_total * (1 - kappa_share)`` (stimulus-specific perseveration)

    The decode happens INSIDE the participant for-loop, not in the likelihood function.
    This guarantees ``kappa + kappa_s == kappa_total <= 1`` by construction (HIER-06).

    Level-2 regression: if ``covariate_lec`` is provided, independent coefficients
    ``beta_lec_kappa_total`` and ``beta_lec_kappa_share`` shift their respective
    unconstrained parameters on the probit scale per participant.

    Parameters
    ----------
    participant_data_stacked : dict
        Mapping from participant_id to stacked-format arrays (from
        ``prepare_stacked_participant_data``). Keys per participant:
        ``stimuli_stacked``, ``actions_stacked``, ``rewards_stacked``,
        ``set_sizes_stacked``, ``masks_stacked`` — each shape
        ``(n_blocks, MAX_TRIALS_PER_BLOCK)``.
    covariate_lec : jnp.ndarray or None
        Shape ``(n_participants,)`` standardized LEC-total covariate.
        Participants must be in the same order as
        ``sorted(participant_data_stacked.keys())``.  If ``None``,
        no Level-2 regression is applied and neither beta coefficient is sampled.
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
    - Six parameters (alpha_pos, alpha_neg, phi, rho, capacity, epsilon) are sampled
      via ``sample_bounded_param`` from ``numpyro_helpers``.
    - ``kappa_total`` and ``kappa_share`` are each sampled manually (with optional L2
      shift) using the same probit-scale pattern as M3's kappa.
    - ``kappa_total`` prior: ``mu_prior_loc=-2.0`` (pushes group mean toward small
      total perseveration budget).
    - ``kappa_share`` prior: ``mu_prior_loc=0.0`` (group-mean share near 0.5 a priori).
    - The decoded values ``kappa`` and ``kappa_s`` are plain JAX scalars computed
      per participant; they are NOT NumPyro named sites.
    - Likelihood is accumulated via ``numpyro.factor`` in a Python for-loop over
      participants.  This implements HIER-06.
    - Do NOT modify this function's API: ``fit_bayesian.py`` dispatches to it by
      name.
    """
    from rlwm.fitting.numpyro_helpers import (
        PARAM_PRIOR_DEFAULTS,
        phi_approx,
        sample_bounded_param,
    )

    if use_pscan:
        raise NotImplementedError(
            "wmrl_m6b_hierarchical_model: use_pscan + fully-batched vmap "
            "path is not implemented.  Pass use_pscan=False."
        )

    n_participants = len(participant_data_stacked)

    # ------------------------------------------------------------------
    # Level-2: LEC-total -> kappa_total and kappa_share regression
    # Two independent beta coefficients, one per perseveration component.
    # ------------------------------------------------------------------
    if covariate_lec is not None:
        beta_lec_kappa_total = numpyro.sample(
            "beta_lec_kappa_total", dist.Normal(0.0, 1.0)
        )
        beta_lec_kappa_share = numpyro.sample(
            "beta_lec_kappa_share", dist.Normal(0.0, 1.0)
        )
    else:
        beta_lec_kappa_total = 0.0
        beta_lec_kappa_share = 0.0

    # ------------------------------------------------------------------
    # Group priors for 6 standard parameters (all except kappa_total/kappa_share)
    # Uses hBayesDM non-centered convention locked in Phase 13.
    # ------------------------------------------------------------------
    sampled: dict[str, jnp.ndarray] = {}
    for param in ["alpha_pos", "alpha_neg", "phi", "rho", "capacity", "epsilon"]:
        defaults = PARAM_PRIOR_DEFAULTS[param]
        sampled[param] = sample_bounded_param(
            param,
            lower=defaults["lower"],
            upper=defaults["upper"],
            n_participants=n_participants,
            mu_prior_loc=defaults["mu_prior_loc"],
        )

    # ------------------------------------------------------------------
    # kappa_total with optional L2 shift on the probit scale
    # mu_prior_loc=-2.0 pushes group mean toward small total budgets.
    # ------------------------------------------------------------------
    kt_defaults = PARAM_PRIOR_DEFAULTS["kappa_total"]
    kappa_total_mu_pr = numpyro.sample(
        "kappa_total_mu_pr",
        dist.Normal(kt_defaults["mu_prior_loc"], 1.0),
    )
    kappa_total_sigma_pr = numpyro.sample(
        "kappa_total_sigma_pr", dist.HalfNormal(0.2)
    )
    kappa_total_z = numpyro.sample(
        "kappa_total_z",
        dist.Normal(0, 1).expand([n_participants]),
    )
    lec_shift_total = (
        beta_lec_kappa_total * covariate_lec if covariate_lec is not None else 0.0
    )
    kappa_total_unc = (
        kappa_total_mu_pr + kappa_total_sigma_pr * kappa_total_z + lec_shift_total
    )
    kappa_total = numpyro.deterministic(
        "kappa_total",
        kt_defaults["lower"]
        + (kt_defaults["upper"] - kt_defaults["lower"])
        * phi_approx(kappa_total_unc),
    )
    sampled["kappa_total"] = kappa_total

    # ------------------------------------------------------------------
    # kappa_share with optional L2 shift on the probit scale
    # mu_prior_loc=0.0 -> group-mean share near 0.5 a priori.
    # ------------------------------------------------------------------
    ks_defaults = PARAM_PRIOR_DEFAULTS["kappa_share"]
    kappa_share_mu_pr = numpyro.sample(
        "kappa_share_mu_pr",
        dist.Normal(ks_defaults["mu_prior_loc"], 1.0),
    )
    kappa_share_sigma_pr = numpyro.sample(
        "kappa_share_sigma_pr", dist.HalfNormal(0.2)
    )
    kappa_share_z = numpyro.sample(
        "kappa_share_z",
        dist.Normal(0, 1).expand([n_participants]),
    )
    lec_shift_share = (
        beta_lec_kappa_share * covariate_lec if covariate_lec is not None else 0.0
    )
    kappa_share_unc = (
        kappa_share_mu_pr + kappa_share_sigma_pr * kappa_share_z + lec_shift_share
    )
    kappa_share = numpyro.deterministic(
        "kappa_share",
        ks_defaults["lower"]
        + (ks_defaults["upper"] - ks_defaults["lower"])
        * phi_approx(kappa_share_unc),
    )
    sampled["kappa_share"] = kappa_share

    # ------------------------------------------------------------------
    # Decode kappa and kappa_s on (N,) vectors BEFORE vmap (STATE.md locked):
    #   kappa   = kappa_total * kappa_share
    #   kappa_s = kappa_total * (1 - kappa_share)
    # This preserves the stick-breaking invariant kappa + kappa_s = kappa_total.
    # ------------------------------------------------------------------
    kappa = sampled["kappa_total"] * sampled["kappa_share"]
    kappa_s = sampled["kappa_total"] * (1.0 - sampled["kappa_share"])

    # ------------------------------------------------------------------
    # Likelihood via single numpyro.factor("obs", ...) — fully-batched
    # vmap over participants and blocks.
    # ------------------------------------------------------------------
    if stacked_arrays is None:
        stacked_arrays = stack_across_participants(participant_data_stacked)

    from rlwm.fitting.models.wmrl_m6b import wmrl_m6b_fully_batched_likelihood

    per_participant_ll = wmrl_m6b_fully_batched_likelihood(
        stimuli=stacked_arrays["stimuli"],
        actions=stacked_arrays["actions"],
        rewards=stacked_arrays["rewards"],
        set_sizes=stacked_arrays["set_sizes"],
        masks=stacked_arrays["masks"],
        alpha_pos=sampled["alpha_pos"],
        alpha_neg=sampled["alpha_neg"],
        phi=sampled["phi"],
        rho=sampled["rho"],
        capacity=sampled["capacity"],
        kappa=kappa,
        kappa_s=kappa_s,
        epsilon=sampled["epsilon"],
        num_stimuli=num_stimuli,
        num_actions=num_actions,
        q_init=q_init,
        wm_init=wm_init,
    )
    numpyro.factor("obs", per_participant_ll.sum())

def wmrl_m6b_hierarchical_model_subscale(
    participant_data_stacked: dict,
    covariate_matrix: jnp.ndarray | None = None,
    covariate_names: list[str] | None = None,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    use_pscan: bool = False,
) -> None:
    """Hierarchical M6b model with full subscale Level-2 regression (L2-05).

    Extends :func:`wmrl_m6b_hierarchical_model` by accepting a full covariate
    matrix instead of a single LEC vector, enabling Level-2 regression on ALL
    8 M6b parameters from the 4-predictor subscale design
    [lec_total, iesr_total, iesr_intr_resid, iesr_avd_resid].

    This produces **32 beta coefficient sites** (8 parameters x 4 covariates)
    matching ``COVARIATE_NAMES`` from ``scripts/fitting/level2_design.py``.
    (Plan references "~40 beta sites" — the actual count is 32 because the
    hyperarousal residual was dropped due to exact linear dependence.)

    All 8 parameters are sampled **manually** (bypassing ``sample_bounded_param``)
    so that multi-covariate L2 shifts can be applied uniformly on the unconstrained
    probit scale before the Phi_approx transform.

    Model structure (non-centered, hBayesDM convention):
    ::

        # Level-2 beta coefficients (if covariate_matrix provided)
        beta_{cov}_{param} ~ Normal(0, 1)   # for each covariate x parameter

        # Group priors
        {param}_mu_pr    ~ Normal(mu_prior_loc, 1.0)
        {param}_sigma_pr ~ HalfNormal(0.2)
        {param}_z        ~ Normal(0, 1)  shape (n_participants,)

        # Individual unconstrained
        theta_unc_i = mu_pr + sigma_pr * z_i + sum_j(beta_j * X_ij)

        # Constrained
        theta_i = lower + (upper - lower) * Phi_approx(theta_unc_i)

    M6b stick-breaking decode (per participant, inside for-loop):
    ::

        kappa   = kappa_total_i * kappa_share_i
        kappa_s = kappa_total_i * (1 - kappa_share_i)

    Parameters
    ----------
    participant_data_stacked : dict
        Mapping from participant_id to stacked-format arrays from
        ``prepare_stacked_participant_data``. Keys per participant:
        ``stimuli_stacked``, ``actions_stacked``, ``rewards_stacked``,
        ``set_sizes_stacked``, ``masks_stacked`` — each shape
        ``(n_blocks, MAX_TRIALS_PER_BLOCK)``.
    covariate_matrix : jnp.ndarray or None
        Shape ``(n_participants, n_covariates)`` standardized design matrix.
        Participants must be in the same order as
        ``sorted(participant_data_stacked.keys())``.  Built by
        ``scripts.fitting.level2_design.build_level2_design_matrix``.
        If ``None``, no Level-2 shifts are applied (equivalent to the
        single-covariate model with ``covariate_lec=None``).
    covariate_names : list[str] or None
        Names for each column of ``covariate_matrix``, used to name
        ``beta_{cov}_{param}`` sites.  Must have length matching
        ``covariate_matrix.shape[1]``.  Expected value (from
        ``COVARIATE_NAMES`` in ``level2_design.py``):
        ``["lec_total", "iesr_total", "iesr_intr_resid", "iesr_avd_resid"]``.
        If ``None`` and ``covariate_matrix`` is not ``None``, columns are
        named ``cov0``, ``cov1``, etc.
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
    - Beta coefficient naming: ``beta_{covariate_name}_{param_name}``.
      Example sites: ``beta_lec_total_kappa_total``,
      ``beta_iesr_intr_resid_alpha_pos``, etc.
    - This model has a high-dimensional posterior (group priors + 32 beta
      sites + 8 x n_participants individual parameters).  Use
      ``run_inference_with_bump`` with target_accept_prob up to 0.99 to
      handle potential divergences (L2-08 horseshoe prior upgrade is the
      fallback if divergences persist).
    - Beta site order: outer loop over param_names, inner loop over
      covariate_names.  This matches the naming convention for downstream
      ArviZ extraction.
    - Do NOT modify this function's API: ``fit_bayesian.py`` dispatches to
      it via the ``--subscale`` flag.
    """
    from rlwm.fitting.numpyro_helpers import PARAM_PRIOR_DEFAULTS, phi_approx

    n_participants = len(participant_data_stacked)
    participant_ids = sorted(participant_data_stacked.keys())

    # All 8 M6b parameters in canonical order
    param_names = [
        "alpha_pos",
        "alpha_neg",
        "phi",
        "rho",
        "capacity",
        "epsilon",
        "kappa_total",
        "kappa_share",
    ]

    # ------------------------------------------------------------------
    # Resolve covariate column names
    # ------------------------------------------------------------------
    if covariate_matrix is not None:
        n_covariates = covariate_matrix.shape[1]
        if covariate_names is None:
            covariate_names = [f"cov{j}" for j in range(n_covariates)]
        if len(covariate_names) != n_covariates:
            raise ValueError(
                f"wmrl_m6b_hierarchical_model_subscale: covariate_names has "
                f"{len(covariate_names)} entries but covariate_matrix has "
                f"{n_covariates} columns. They must match."
            )
    else:
        covariate_names = []

    # ------------------------------------------------------------------
    # Level-2 beta coefficients — one per (param, covariate) pair
    # Naming: beta_{covariate_name}_{param_name}
    # 8 params x 4 covariates = 32 sites when using the default design.
    # ------------------------------------------------------------------
    betas: dict[tuple[str, str], object] = {}
    if covariate_matrix is not None:
        for pname in param_names:
            for cov_name in covariate_names:
                site_name = f"beta_{cov_name}_{pname}"
                betas[(pname, cov_name)] = numpyro.sample(
                    site_name, dist.Normal(0.0, 1.0)
                )

    # ------------------------------------------------------------------
    # Manual non-centered sampling for all 8 parameters
    # Bypasses sample_bounded_param to support per-parameter L2 shifts.
    # Pattern: mu_pr + sigma_pr * z + sum_j(beta_j * X[:, j])
    # ------------------------------------------------------------------
    sampled: dict[str, jnp.ndarray] = {}
    for pname in param_names:
        defaults = PARAM_PRIOR_DEFAULTS[pname]
        lower = defaults["lower"]
        upper = defaults["upper"]
        mu_prior_loc = defaults["mu_prior_loc"]

        mu_pr = numpyro.sample(
            f"{pname}_mu_pr",
            dist.Normal(mu_prior_loc, 1.0),
        )
        sigma_pr = numpyro.sample(
            f"{pname}_sigma_pr",
            dist.HalfNormal(0.2),
        )
        z = numpyro.sample(
            f"{pname}_z",
            dist.Normal(0, 1).expand([n_participants]),
        )

        # Unconstrained individual-level values
        theta_unc = mu_pr + sigma_pr * z

        # Add L2 shifts from each covariate column
        if covariate_matrix is not None:
            for j, cov_name in enumerate(covariate_names):
                theta_unc = theta_unc + betas[(pname, cov_name)] * covariate_matrix[:, j]

        # Transform to constrained space via Phi_approx
        theta = lower + (upper - lower) * phi_approx(theta_unc)
        sampled[pname] = numpyro.deterministic(pname, theta)

    # ------------------------------------------------------------------
    # Likelihood via numpyro.factor — Python for-loop over participants
    # Decode kappa and kappa_s per-participant (STATE.md locked decision):
    #   kappa   = kappa_total * kappa_share
    #   kappa_s = kappa_total * (1 - kappa_share)
    # ------------------------------------------------------------------
    _m6b_sub_lik_fn = (
        wmrl_m6b_multiblock_likelihood_stacked_pscan
        if use_pscan
        else wmrl_m6b_multiblock_likelihood_stacked
    )
    for idx, pid in enumerate(participant_ids):
        pdata = participant_data_stacked[pid]
        kappa_total_i = sampled["kappa_total"][idx]
        kappa_share_i = sampled["kappa_share"][idx]
        kappa = kappa_total_i * kappa_share_i
        kappa_s = kappa_total_i * (1.0 - kappa_share_i)
        log_lik = _m6b_sub_lik_fn(
            stimuli_stacked=pdata["stimuli_stacked"],
            actions_stacked=pdata["actions_stacked"],
            rewards_stacked=pdata["rewards_stacked"],
            set_sizes_stacked=pdata["set_sizes_stacked"],
            masks_stacked=pdata["masks_stacked"],
            alpha_pos=sampled["alpha_pos"][idx],
            alpha_neg=sampled["alpha_neg"][idx],
            phi=sampled["phi"][idx],
            rho=sampled["rho"][idx],
            capacity=sampled["capacity"][idx],
            kappa=kappa,
            kappa_s=kappa_s,
            epsilon=sampled["epsilon"][idx],
            num_stimuli=num_stimuli,
            num_actions=num_actions,
            q_init=q_init,
            wm_init=wm_init,
            return_pointwise=False,
        )
        numpyro.factor(f"obs_p{pid}", log_lik)
