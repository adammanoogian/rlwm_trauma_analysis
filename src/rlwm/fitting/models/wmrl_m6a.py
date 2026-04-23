"""M6a WM-RL + kappa_s (stimulus-specific perseveration): JAX likelihoods + NumPyro wrapper.

Canonical home for M6a's JAX likelihoods and NumPyro wrappers. Callers
should import directly from this module; the legacy
``rlwm.fitting.jax_likelihoods`` / ``rlwm.fitting.numpyro_models``
re-export shims were deleted in the v5.0 shim cleanup.

M6a replaces M3's global perseveration with a per-stimulus ``kappa_s`` bonus
applied to the last action taken for THIS stimulus (not globally).
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
    precompute_last_actions_per_stimulus,
    prepare_stacked_participant_data,
    softmax_policy,
    stack_across_participants,
)

__all__ = [
    "wmrl_m6a_fully_batched_likelihood",
    "wmrl_m6a_block_likelihood",
    "wmrl_m6a_multiblock_likelihood",
    "wmrl_m6a_multiblock_likelihood_stacked",
    "wmrl_m6a_block_likelihood_pscan",
    "wmrl_m6a_multiblock_likelihood_stacked_pscan",
    "test_wmrl_m6a_single_block",
    "test_wmrl_m6a_per_stimulus_tracking",
    "test_padding_equivalence_wmrl_m6a",
    "wmrl_m6a_hierarchical_model",
]


def wmrl_m6a_fully_batched_likelihood(
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
    kappa_s: jnp.ndarray,
    epsilon: jnp.ndarray,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    use_pscan: bool = False,
) -> jnp.ndarray:
    """Fully-batched WM-RL+kappa_s (M6a) log-likelihood via nested vmap.

    Replaces M3's global kappa with stimulus-specific kappa_s.  See
    ``wmrl_m6a_block_likelihood`` for the per-stimulus last-action carry.

    Parameters
    ----------
    stimuli, actions, rewards, set_sizes, masks : jnp.ndarray
        Shape (N, B, T).
    alpha_pos, alpha_neg, phi, rho, capacity, kappa_s, epsilon : jnp.ndarray
        Shape (N,) per-participant parameter vectors.

    Returns
    -------
    jnp.ndarray
        Shape (N,) — total log-likelihood per participant.
    """
    if use_pscan:
        raise NotImplementedError(
            "wmrl_m6a_fully_batched_likelihood: use_pscan=True is not supported."
        )

    def _block_ll(stim, act, rew, ss, mask, ap, an, ph, rh, cap, ks, e):
        return wmrl_m6a_block_likelihood(
            stimuli=stim,
            actions=act,
            rewards=rew,
            set_sizes=ss,
            alpha_pos=ap,
            alpha_neg=an,
            phi=ph,
            rho=rh,
            capacity=cap,
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
        in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None, None),
        out_axes=0,
    )

    def _participant_ll(stim, act, rew, ss, mask, ap, an, ph, rh, cap, ks, e):
        return _over_blocks(
            stim, act, rew, ss, mask, ap, an, ph, rh, cap, ks, e,
        ).sum()

    _over_participants = jax.vmap(
        _participant_ll,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        out_axes=0,
    )
    return _over_participants(
        stimuli, actions, rewards, set_sizes, masks,
        alpha_pos, alpha_neg, phi, rho, capacity, kappa_s, epsilon,
    )

def wmrl_m6a_block_likelihood(
    stimuli: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    set_sizes: jnp.ndarray,
    alpha_pos: float,
    alpha_neg: float,
    phi: float,
    rho: float,
    capacity: float,
    kappa_s: float,  # Stimulus-specific perseveration (replaces global kappa)
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
    Compute log-likelihood for WM-RL M6a model with stimulus-specific perseveration.

    M6a replaces M3's global perseveration kernel (which applies to any stimulus
    after any action was taken in the block) with a stimulus-specific kernel that
    only applies to a stimulus after IT has been acted upon. This tests whether
    perseveration is stimulus-bound rather than globally motor-based.

    CRITICAL CHANGE from M3: The carry changes from a single scalar last_action
    to a per-stimulus array last_actions of shape (num_stimuli,). Each stimulus
    independently tracks the last action taken for that specific stimulus within
    the current block.

    When kappa_s=0, this reduces to M2-equivalent behavior (no perseveration).
    First presentation of any stimulus in a block uses uniform fallback (sentinel -1).

    Model is otherwise identical to M3 -- no phi_rl, same 7 parameters.

    Parameters
    ----------
    stimuli : array, shape (n_trials,)
        Stimulus sequence
    actions : array, shape (n_trials,)
        Action sequence
    rewards : array, shape (n_trials,)
        Reward sequence (0 or 1)
    set_sizes : array, shape (n_trials,)
        Set size for each trial (for adaptive weighting)
    alpha_pos : float
        RL learning rate for positive PE
    alpha_neg : float
        RL learning rate for negative PE
    phi : float
        WM decay rate (0-1)
    rho : float
        Base WM reliance (0-1)
    capacity : float
        WM capacity (for adaptive weighting)
    kappa_s : float
        Stimulus-specific perseveration (0-1)
    epsilon : float
        Epsilon noise parameter (probability of random action)
    num_stimuli : int
        Number of possible stimuli
    num_actions : int
        Number of possible actions
    q_init : float
        Initial Q-values
    wm_init : float
        Initial WM values (baseline = 1/nA for uniform)
    mask : array, shape (n_trials,), optional
        Mask for padded blocks: 1.0 for real trials, 0.0 for padding.
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

    # M6a carry: per-stimulus last_actions array, initialized to -1 (never seen)
    # Shape: (num_stimuli,) int32 — each stimulus independently tracks last action
    last_actions_init = jnp.full((num_stimuli,), -1, dtype=jnp.int32)

    # Initial carry: (Q, WM, WM_0, log_likelihood, last_actions)
    init_carry = (Q_init, WM_init, WM_0, 0.0, last_actions_init)

    # If no mask provided, all trials are valid (backward compatibility)
    if mask is None:
        mask = jnp.ones(len(stimuli))

    # Prepare inputs (include mask)
    scan_inputs = (stimuli, actions, rewards, set_sizes, mask)

    def step_fn(carry, inputs):
        Q_table, WM_table, WM_baseline, log_lik_accum, last_actions = carry
        stimulus, action, reward, set_size, valid = inputs

        # =================================================================
        # 1. DECAY WM: All values move toward baseline
        # =================================================================
        WM_decayed = (1 - phi) * WM_table + phi * WM_baseline

        # =================================================================
        # 2. COMPUTE HYBRID POLICY
        # =================================================================
        q_vals = Q_table[stimulus]
        wm_vals = WM_decayed[stimulus]

        # Adaptive weight: omega = rho * min(1, K/N_s)
        omega = rho * jnp.minimum(1.0, capacity / set_size)

        # Get stimulus-specific last action (sentinel -1 = never seen in block)
        last_action_s = last_actions[stimulus]

        # Gate: no kernel if kappa_s == 0 OR this stimulus never seen in this block
        use_m2_path = jnp.logical_or(kappa_s == 0.0, last_action_s < 0)

        # Both paths start with M2 probability mixing
        rl_probs = softmax_policy(q_vals, FIXED_BETA)
        wm_probs = softmax_policy(wm_vals, FIXED_BETA)
        base_probs = omega * wm_probs + (1 - omega) * rl_probs
        base_probs = base_probs / jnp.sum(base_probs)  # Normalize

        # Apply epsilon noise to base policy
        noisy_base = apply_epsilon_noise(base_probs, epsilon, num_actions)

        # =================================================================
        # M6a path: Stimulus-specific choice kernel
        # P_M6a = (1-kappa_s)*P_noisy + kappa_s*Ck(stimulus)
        # where Ck(stimulus) = one-hot(last_actions[stimulus])
        # =================================================================
        # Clamp prevents bad indexing when sentinel is -1 (safe because masked by use_m2_path)
        choice_kernel = jnp.eye(num_actions)[jnp.maximum(last_action_s, 0)]

        # Probability mixing: (1-kappa_s)*noisy_base + kappa_s*choice_kernel
        hybrid_probs_m6a = (1 - kappa_s) * noisy_base + kappa_s * choice_kernel

        # Select correct path: M2 uses noisy_base, M6a uses probability mixing
        noisy_probs = jnp.where(use_m2_path, noisy_base, hybrid_probs_m6a)

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
        # 5. UPDATE PER-STIMULUS last_actions (unconditionally on valid trials)
        # NOTE: Update happens regardless of whether kernel was applied.
        # For padding trials (valid=0), keep previous last_action for this stimulus.
        # =================================================================
        new_last_actions = last_actions.at[stimulus].set(
            jnp.where(valid, action, last_action_s).astype(jnp.int32)
        )

        return (Q_updated, WM_updated, WM_baseline, log_lik_new, new_last_actions), log_prob_masked

    # Run scan over trials
    (Q_final, WM_final, _, log_lik_total, _), log_probs = lax.scan(step_fn, init_carry, scan_inputs)

    if return_pointwise:
        return log_lik_total, log_probs
    return log_lik_total

def wmrl_m6a_multiblock_likelihood(
    stimuli_blocks: list,
    actions_blocks: list,
    rewards_blocks: list,
    set_sizes_blocks: list,
    alpha_pos: float,
    alpha_neg: float,
    phi: float,
    rho: float,
    capacity: float,
    kappa_s: float,  # Stimulus-specific perseveration
    epsilon: float = DEFAULT_EPSILON,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,  # WM baseline = 1/nA (uniform)
    masks_blocks: list = None,
    verbose: bool = False,
    participant_id: str = None
) -> float:
    """
    Compute log-likelihood for WM-RL M6a (stimulus-specific perseveration) across MULTIPLE BLOCKS.

    Q-values, WM, and per-stimulus last_actions reset at each block boundary.
    Uses fori_loop fast path for uniformly-sized padded blocks, Python fallback otherwise.
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
            block_ll = wmrl_m6a_block_likelihood(
                stimuli=stimuli_stacked[block_idx],
                actions=actions_stacked[block_idx],
                rewards=rewards_stacked[block_idx],
                set_sizes=set_sizes_stacked[block_idx],
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
            block_log_lik = wmrl_m6a_block_likelihood(
                stimuli=stim_block,
                actions=act_block,
                rewards=rew_block,
                set_sizes=set_block,
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
                wm_init=wm_init,
                mask=mask_block
            )
            total_log_lik += block_log_lik

    if verbose:
        print(f"  >> Total log-likelihood: {float(total_log_lik):.2f} ({num_blocks} blocks)\n", flush=True)

    return total_log_lik

def wmrl_m6a_multiblock_likelihood_stacked(
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
    FAST WM-RL M6a multiblock likelihood that takes pre-stacked arrays directly.

    See wmrl_m6a_multiblock_likelihood for full documentation.
    This version avoids list/restack overhead inside JIT.
    kappa_s controls stimulus-specific perseveration (replaces global kappa from M3).

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
            block_ll, block_probs = wmrl_m6a_block_likelihood(
                stimuli=stimuli_stacked[block_idx],
                actions=actions_stacked[block_idx],
                rewards=rewards_stacked[block_idx],
                set_sizes=set_sizes_stacked[block_idx],
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
            block_ll = wmrl_m6a_block_likelihood(
                stimuli=stimuli_stacked[block_idx],
                actions=actions_stacked[block_idx],
                rewards=rewards_stacked[block_idx],
                set_sizes=set_sizes_stacked[block_idx],
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
                wm_init=wm_init,
                mask=masks_stacked[block_idx]
            )
            return total_ll + block_ll

        return lax.fori_loop(0, num_blocks, body_fn, 0.0)

def wmrl_m6a_block_likelihood_pscan(
    stimuli: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    set_sizes: jnp.ndarray,
    alpha_pos: float,
    alpha_neg: float,
    phi: float,
    rho: float,
    capacity: float,
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
    WM-RL M6a block likelihood using parallel scan (stimulus-specific perseveration).

    Drop-in replacement for ``wmrl_m6a_block_likelihood``.

    Phase 1 (parallel): Pre-compute Q_for_policy[t] and wm_for_policy[t].
    Phase 2 (vectorized): Precompute per-stimulus last_action array via
    ``precompute_last_actions_per_stimulus``, then compute hybrid WM-Q
    policy with per-stimulus perseveration kernel for all trials
    simultaneously.

    Parameters
    ----------
    stimuli, actions, rewards, set_sizes : arrays, shape (n_trials,)
    alpha_pos, alpha_neg, phi, rho, capacity, kappa_s, epsilon : float
    num_stimuli, num_actions : int
    q_init, wm_init : float
    mask : array, optional
    return_pointwise : bool, optional

    Returns
    -------
    float or tuple[float, jnp.ndarray]
        Same as ``wmrl_m6a_block_likelihood``.
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
    # Phase 2 (vectorized): hybrid policy + per-stimulus perseveration
    # ------------------------------------------------------------------
    t_idx = jnp.arange(T)
    q_vals = Q_for_policy[t_idx, stimuli]      # (T, A)
    wm_vals = wm_for_policy[t_idx, stimuli]    # (T, A)

    omega = rho * jnp.minimum(1.0, capacity / set_sizes)  # (T,)
    rl_probs = jax.vmap(softmax_policy, in_axes=(0, None))(q_vals, FIXED_BETA)
    wm_probs = jax.vmap(softmax_policy, in_axes=(0, None))(wm_vals, FIXED_BETA)
    base_probs = omega[:, None] * wm_probs + (1 - omega[:, None]) * rl_probs
    base_probs = base_probs / jnp.sum(base_probs, axis=-1, keepdims=True)

    noisy_base = jax.vmap(apply_epsilon_noise, in_axes=(0, None, None))(
        base_probs, epsilon, num_actions
    )  # (T, A)

    # Precompute per-stimulus last_action for perseveration
    last_action_stim = precompute_last_actions_per_stimulus(
        stimuli, actions, mask, num_stimuli
    )  # (T,)
    use_m2_path = jnp.logical_or(kappa_s == 0.0, last_action_stim < 0)  # (T,)
    choice_kernels = jnp.eye(num_actions)[jnp.maximum(last_action_stim, 0)]  # (T, A)
    hybrid_probs = (1 - kappa_s) * noisy_base + kappa_s * choice_kernels
    noisy_probs = jnp.where(use_m2_path[:, None], noisy_base, hybrid_probs)  # (T, A)

    log_probs = jnp.log(noisy_probs[t_idx, actions] + 1e-8) * mask

    if return_pointwise:
        return jnp.sum(log_probs), log_probs
    return jnp.sum(log_probs)

def wmrl_m6a_multiblock_likelihood_stacked_pscan(
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
    WM-RL M6a multiblock likelihood using parallel scan.

    Drop-in replacement for ``wmrl_m6a_multiblock_likelihood_stacked``.

    Parameters
    ----------
    stimuli_stacked, actions_stacked, rewards_stacked,
    set_sizes_stacked, masks_stacked : arrays, shape (n_blocks, max_trials)
    alpha_pos, alpha_neg, phi, rho, capacity, kappa_s, epsilon : float
    num_stimuli, num_actions : int
    q_init, wm_init : float
    return_pointwise : bool, optional

    Returns
    -------
    float or tuple[float, jnp.ndarray]
        Same as ``wmrl_m6a_multiblock_likelihood_stacked``.
    """
    num_blocks = stimuli_stacked.shape[0]

    if return_pointwise:
        def scan_body(total_ll, block_idx):
            block_ll, block_probs = wmrl_m6a_block_likelihood_pscan(
                stimuli=stimuli_stacked[block_idx],
                actions=actions_stacked[block_idx],
                rewards=rewards_stacked[block_idx],
                set_sizes=set_sizes_stacked[block_idx],
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
            block_ll = wmrl_m6a_block_likelihood_pscan(
                stimuli=stimuli_stacked[block_idx],
                actions=actions_stacked[block_idx],
                rewards=rewards_stacked[block_idx],
                set_sizes=set_sizes_stacked[block_idx],
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
                wm_init=wm_init,
                mask=masks_stacked[block_idx],
            )
            return total_ll + block_ll

        return lax.fori_loop(0, num_blocks, body_fn, 0.0)

def test_wmrl_m6a_single_block():
    """Smoke test for WM-RL M6a single block likelihood."""
    print("\nTesting WM-RL M6a single block likelihood (smoke test)...")

    key = jax.random.PRNGKey(42)
    n_trials = 30

    stimuli = jax.random.randint(key, (n_trials,), 0, 6)
    key, subkey = jax.random.split(key)
    actions = jax.random.randint(subkey, (n_trials,), 0, 3)
    key, subkey = jax.random.split(key)
    rewards = jax.random.bernoulli(subkey, 0.7, (n_trials,)).astype(jnp.float32)
    set_sizes = jnp.ones((n_trials,)) * 5

    params = {
        'alpha_pos': 0.3,
        'alpha_neg': 0.1,
        'phi': 0.1,
        'rho': 0.7,
        'capacity': 4.0,
        'kappa_s': 0.3,  # Moderate stimulus-specific perseveration
        'epsilon': 0.05,
    }

    log_lik = wmrl_m6a_block_likelihood(
        stimuli, actions, rewards, set_sizes, **params
    )

    nll = -float(log_lik)
    is_finite = jnp.isfinite(log_lik)
    is_negative = log_lik < 0  # log-likelihood should be negative

    print(f"  WM-RL M6a log-likelihood: {float(log_lik):.4f}")
    print(f"  NLL: {nll:.4f}")
    print(f"  Is finite: {bool(is_finite)}")
    print(f"  Is negative (expected): {bool(is_negative)}")

    assert bool(is_finite), "M6a log-likelihood must be finite!"
    print("[OK] WM-RL M6a single block smoke test passed")
    return log_lik

def test_wmrl_m6a_per_stimulus_tracking():
    """
    Verify that M6a per-stimulus tracking differs from M3 global tracking.

    Construct a scenario where:
    - Stimulus 0 is presented first, action taken
    - Stimulus 1 is presented next (first time in block)

    In M3 (global): last_action from stimulus 0 would be applied to stimulus 1's kernel.
    In M6a (per-stimulus): stimulus 1 has never been seen, so no kernel applied.
    Therefore M6a and M3 should produce DIFFERENT NLLs for this sequence.
    """
    print("\nTesting WM-RL M6a per-stimulus tracking (vs M3 global tracking)...")

    # Construct minimal 2-stimulus sequence:
    # Trial 1: Stimulus 0 presented, action 1 taken
    # Trial 2: Stimulus 1 presented (first time) -- M3 uses kernel from trial 1, M6a does not
    stimuli = jnp.array([0, 1], dtype=jnp.int32)
    actions = jnp.array([1, 2], dtype=jnp.int32)
    rewards = jnp.array([1.0, 0.0], dtype=jnp.float32)
    set_sizes = jnp.array([3, 3], dtype=jnp.int32)

    shared_params = {
        'alpha_pos': 0.3,
        'alpha_neg': 0.1,
        'phi': 0.1,
        'rho': 0.7,
        'capacity': 4.0,
        'epsilon': 0.05,
    }

    # M3 with kappa=0.5 (uses global last_action)
    log_lik_m3 = wmrl_m3_block_likelihood(
        stimuli, actions, rewards, set_sizes, **shared_params, kappa=0.5
    )

    # M6a with kappa_s=0.5 (uses per-stimulus tracking)
    log_lik_m6a = wmrl_m6a_block_likelihood(
        stimuli, actions, rewards, set_sizes, **shared_params, kappa_s=0.5
    )

    m3_val = float(log_lik_m3)
    m6a_val = float(log_lik_m6a)
    diff = abs(m3_val - m6a_val)

    print(f"  M3 (global last_action) log-lik: {m3_val:.6f}")
    print(f"  M6a (per-stimulus last_actions) log-lik: {m6a_val:.6f}")
    print(f"  Difference: {diff:.6f}")

    # They must differ: M3 applies kernel on trial 2 (last_action=1 from trial 1),
    # M6a does NOT (stimulus 1 never seen before in this block)
    different = diff > 1e-6
    print(f"  M6a differs from M3 (per-stimulus tracking verified): {different}")
    assert different, (
        "M6a and M3 should produce DIFFERENT NLLs when stimulus 1 is first seen "
        "after stimulus 0 was acted on. If identical, per-stimulus tracking is NOT working."
    )
    print("[OK] WM-RL M6a per-stimulus tracking verified")
    return m3_val, m6a_val

def test_padding_equivalence_wmrl_m6a():
    """
    Verify padded and unpadded WM-RL M6a likelihoods are mathematically equivalent.
    """
    print("\nTesting WM-RL M6a padding equivalence (CRITICAL)...")

    key = jax.random.PRNGKey(567)
    n_real_trials = 50

    # Generate real trial data
    stimuli = jax.random.randint(key, (n_real_trials,), 0, 6)
    key, subkey = jax.random.split(key)
    actions = jax.random.randint(subkey, (n_real_trials,), 0, 3)
    key, subkey = jax.random.split(key)
    rewards = jax.random.bernoulli(subkey, 0.7, (n_real_trials,)).astype(jnp.float32)
    set_sizes = jnp.full((n_real_trials,), 5, dtype=jnp.int32)

    params = {
        'alpha_pos': 0.3, 'alpha_neg': 0.1, 'phi': 0.1,
        'rho': 0.7, 'capacity': 4.0, 'kappa_s': 0.3, 'epsilon': 0.05
    }

    # Unpadded likelihood (original)
    log_lik_original = wmrl_m6a_block_likelihood(
        stimuli, actions, rewards, set_sizes, **params
    )

    # Padded likelihood (with mask)
    stimuli_pad, actions_pad, rewards_pad, set_sizes_pad, mask = pad_block_to_max(
        stimuli, actions, rewards, max_trials=100, set_sizes=set_sizes
    )
    log_lik_padded = wmrl_m6a_block_likelihood(
        stimuli_pad, actions_pad, rewards_pad, set_sizes_pad, **params, mask=mask
    )

    # Verify equivalence
    match = jnp.allclose(log_lik_original, log_lik_padded, rtol=1e-6)
    print(f"  Original (unpadded) log-lik: {float(log_lik_original):.8f}")
    print(f"  Padded (with mask) log-lik:  {float(log_lik_padded):.8f}")
    print(f"  Difference: {abs(float(log_lik_original - log_lik_padded)):.2e}")
    print(f"[OK] WM-RL M6a padding equivalence: {'PASSED' if match else 'FAILED'}")

    assert bool(match), "WM-RL M6a padded/unpadded must be IDENTICAL!"
    return match

def wmrl_m6a_hierarchical_model(
    participant_data_stacked: dict,
    covariate_lec: jnp.ndarray | None = None,
    covariate_iesr: jnp.ndarray | None = None,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    use_pscan: bool = False,
    stacked_arrays: dict | None = None,
) -> None:
    """Hierarchical Bayesian M6a (WM-RL+kappa_s) model with optional L2 regression.

    Uses the hBayesDM non-centered parameterization convention (Ahn, Haines, Zhang 2017):
    ``theta_unc = mu_pr + sigma_pr * z``,
    ``theta = lower + (upper - lower) * Phi_approx(theta_unc)``,
    where ``Phi_approx = jax.scipy.stats.norm.cdf``.

    K (capacity) is parameterized in [2, 6] following Senta, Bishop, Collins (2025).

    M6a replaces global perseveration ``kappa`` (M3) with stimulus-specific
    perseveration ``kappa_s``.  The 7 model parameters match M3 in count:
    alpha_pos, alpha_neg, phi, rho, capacity, epsilon (6, sampled via
    ``sample_bounded_param``), and kappa_s (1, sampled manually with optional
    L2 shift using the same pattern as M3's kappa).

    Level-2 regression (single-covariate legacy Phase 16 mode): if
    ``covariate_lec`` is provided, a coefficient ``beta_lec_kappa_s`` is
    sampled and shifts ``kappa_s`` on the probit scale:
    ``kappa_s_unc_i = kappa_s_mu_pr + kappa_s_sigma_pr * z_i + beta_lec_kappa_s * lec_i``.

    Level-2 regression (2-covariate Phase 21 Option C mode): if BOTH
    ``covariate_lec`` and ``covariate_iesr`` are provided, an additional
    coefficient ``beta_iesr_kappa_s`` is sampled with the same Normal(0, 1)
    prior and both shifts are summed on the probit scale:
    ``kappa_s_unc_i = kappa_s_mu_pr + kappa_s_sigma_pr * z_i
                     + beta_lec_kappa_s * lec_i + beta_iesr_kappa_s * iesr_i``.

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
        no Level-2 regression is applied and ``beta_lec_kappa_s`` is
        not sampled.
    covariate_iesr : jnp.ndarray or None
        Shape ``(n_participants,)`` standardized IES-R total covariate.
        When both ``covariate_lec`` and ``covariate_iesr`` are provided, the
        2-covariate L2 design is applied: both beta sites are sampled with
        ``Normal(0, 1)`` priors and their contributions are summed on the
        probit scale before the Phi_approx transform. ``covariate_iesr``
        must not be passed without ``covariate_lec`` (ValueError otherwise).
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
    - kappa_s is sampled manually with the optional L2 shift applied on the probit
      scale before the Phi_approx transform (OUTSIDE ``sample_bounded_param``).
    - kappa_s uses the same bounds and prior as M3's kappa: both in [0, 1] with
      ``mu_prior_loc=-2.0`` from ``PARAM_PRIOR_DEFAULTS``.
    - Likelihood is accumulated via ``numpyro.factor`` in a Python for-loop over
      participants.  This implements HIER-05.
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
            "wmrl_m6a_hierarchical_model: use_pscan + fully-batched vmap "
            "path is not implemented.  Pass use_pscan=False."
        )

    # ------------------------------------------------------------------
    # Phase 21 Option C guard: IES-R covariate requires LEC covariate.
    # Prevents silently dropping LEC when only IES-R is passed.
    # ------------------------------------------------------------------
    if covariate_iesr is not None and covariate_lec is None:
        raise ValueError(
            "covariate_iesr provided without covariate_lec. To use 2-covariate "
            "L2 design, pass both covariate_lec and covariate_iesr. Expected: "
            "both None (no L2), or both vectors (2-cov L2), or covariate_lec "
            "only (single-cov L2 — legacy Phase 16 mode)."
        )

    n_participants = len(participant_data_stacked)

    # ------------------------------------------------------------------
    # Level-2: LEC-total + IES-R-total -> kappa_s regression coefficients
    # (2-covariate design, Phase 21 Option C)
    # ------------------------------------------------------------------
    if covariate_lec is not None:
        beta_lec_kappa_s = numpyro.sample("beta_lec_kappa_s", dist.Normal(0.0, 1.0))
    else:
        beta_lec_kappa_s = 0.0

    if covariate_iesr is not None:
        beta_iesr_kappa_s = numpyro.sample(
            "beta_iesr_kappa_s", dist.Normal(0.0, 1.0)
        )
    else:
        beta_iesr_kappa_s = 0.0

    # ------------------------------------------------------------------
    # Group priors for 6 parameters (all except kappa_s)
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
    # kappa_s with optional L2 shift on the probit scale
    # Sampled manually to allow per-participant LEC + IES-R offset.
    # Same bounds and prior as M3's kappa (both in PARAM_PRIOR_DEFAULTS).
    # ------------------------------------------------------------------
    kappa_s_defaults = PARAM_PRIOR_DEFAULTS["kappa_s"]
    kappa_s_mu_pr = numpyro.sample(
        "kappa_s_mu_pr",
        dist.Normal(kappa_s_defaults["mu_prior_loc"], 1.0),
    )
    kappa_s_sigma_pr = numpyro.sample("kappa_s_sigma_pr", dist.HalfNormal(0.2))
    kappa_s_z = numpyro.sample(
        "kappa_s_z",
        dist.Normal(0, 1).expand([n_participants]),
    )
    lec_shift = (
        beta_lec_kappa_s * covariate_lec if covariate_lec is not None else 0.0
    )
    iesr_shift = (
        beta_iesr_kappa_s * covariate_iesr if covariate_iesr is not None else 0.0
    )
    kappa_s_unc = (
        kappa_s_mu_pr + kappa_s_sigma_pr * kappa_s_z + lec_shift + iesr_shift
    )
    kappa_s = numpyro.deterministic(
        "kappa_s",
        kappa_s_defaults["lower"]
        + (kappa_s_defaults["upper"] - kappa_s_defaults["lower"])
        * phi_approx(kappa_s_unc),
    )
    sampled["kappa_s"] = kappa_s

    # ------------------------------------------------------------------
    # Likelihood via single numpyro.factor("obs", ...) — fully-batched
    # vmap over participants and blocks.
    # ------------------------------------------------------------------
    if stacked_arrays is None:
        stacked_arrays = stack_across_participants(participant_data_stacked)

    from rlwm.fitting.models.wmrl_m6a import wmrl_m6a_fully_batched_likelihood

    per_participant_ll = wmrl_m6a_fully_batched_likelihood(
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
        kappa_s=sampled["kappa_s"],
        epsilon=sampled["epsilon"],
        num_stimuli=num_stimuli,
        num_actions=num_actions,
        q_init=q_init,
        wm_init=wm_init,
    )
    numpyro.factor("obs", per_participant_ll.sum())
