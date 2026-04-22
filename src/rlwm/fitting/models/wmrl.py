"""M2 WM-RL hybrid: JAX likelihoods (sequential + pscan variants) + NumPyro hierarchical wrappers.

Relocated here in Phase 29-08 from :mod:`rlwm.fitting.jax_likelihoods` and
:mod:`rlwm.fitting.numpyro_models`. Old import paths remain available via
wildcard re-export shims.

Senta et al. (2025) M2: WM mixes with RL via weight
``omega = rho * min(1, K/nS)``; WM decays toward 1/nA baseline with rate ``phi``.
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import lax
from numpyro.infer import MCMC, NUTS

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
    softmax_policy,
    stack_across_participants,
)

__all__ = [
    "wmrl_block_likelihood",
    "wmrl_multiblock_likelihood",
    "wmrl_multiblock_likelihood_stacked",
    "wmrl_fully_batched_likelihood",
    "wmrl_block_likelihood_jit",
    "wmrl_block_likelihood_pscan",
    "wmrl_multiblock_likelihood_stacked_pscan",
    "test_wmrl_single_block",
    "test_wmrl_multiblock",
    "test_padding_equivalence_wmrl",
    "test_multiblock_padding_equivalence",
    "wmrl_hierarchical_model",
    "wmrl_hierarchical_model_stacked",
]


def wmrl_block_likelihood(
    stimuli: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    set_sizes: jnp.ndarray,
    alpha_pos: float,
    alpha_neg: float,
    phi: float,
    rho: float,
    capacity: float,
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
    Compute log-likelihood for WM-RL hybrid model on a SINGLE BLOCK.

    Following Senta et al. (2025):
    - Beta is fixed at 50 for both WM and RL (parameter identifiability)
    - Epsilon noise is applied to the final hybrid policy
    - WM baseline = 1/nA (uniform probability)

    Model combines:
    1. Working Memory (WM): Immediate encoding with decay
    2. Q-Learning (RL): Gradual learning with asymmetric rates
    3. Hybrid decision: Adaptive weighting based on capacity

    Update sequence per trial:
    1. Decay WM: WM ← (1-φ)WM + φ·WM_0
    2. Compute hybrid policy: p = ω·p_WM + (1-ω)·p_RL
    3. Apply epsilon noise: p_noisy = ε/nA + (1-ε)·p
    4. Update WM: WM(s,a) ← r (immediate overwrite)
    5. Update Q: Q(s,a) ← Q(s,a) + α·(r - Q(s,a))

    Supports masked padding for JAX compilation efficiency:
    - When mask is provided, only trials with mask[t]=1 contribute to likelihood
    - Padding trials (mask[t]=0) are ignored in likelihood, Q-updates, and WM-updates

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
        If None, all trials are treated as real (backward compatible).
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

    # Initial carry: (Q, WM, WM_0, log_likelihood)
    init_carry = (Q_init, WM_init, WM_0, 0.0)

    # If no mask provided, all trials are valid (backward compatibility)
    if mask is None:
        mask = jnp.ones(len(stimuli))

    # Prepare inputs (include mask)
    scan_inputs = (stimuli, actions, rewards, set_sizes, mask)

    def step_fn(carry, inputs):
        Q_table, WM_table, WM_baseline, log_lik_accum = carry
        stimulus, action, reward, set_size, valid = inputs

        # =================================================================
        # 1. DECAY WM: All values move toward baseline
        # Note: Decay happens for all trials (valid or not) to maintain
        # consistent WM state, but WM updates are masked below.
        # =================================================================
        WM_decayed = (1 - phi) * WM_table + phi * WM_baseline

        # =================================================================
        # 2. COMPUTE HYBRID POLICY (with fixed beta)
        # =================================================================
        # RL policy: softmax(β * Q(s,:)) with β=50
        q_vals = Q_table[stimulus]
        rl_probs = softmax_policy(q_vals, FIXED_BETA)

        # WM policy: softmax(β * WM(s,:)) with β=50
        wm_vals = WM_decayed[stimulus]
        wm_probs = softmax_policy(wm_vals, FIXED_BETA)

        # Adaptive weight: ω = ρ * min(1, K/N_s)
        omega = rho * jnp.minimum(1.0, capacity / set_size)

        # Hybrid policy: p = ω·p_WM + (1-ω)·p_RL
        hybrid_probs = omega * wm_probs + (1 - omega) * rl_probs

        # Normalize (numerical stability)
        hybrid_probs = hybrid_probs / jnp.sum(hybrid_probs)

        # =================================================================
        # 3. APPLY EPSILON NOISE: p_noisy = ε/nA + (1-ε)*p
        # =================================================================
        noisy_probs = apply_epsilon_noise(hybrid_probs, epsilon, num_actions)

        # Log probability of observed action
        log_prob = jnp.log(noisy_probs[action] + 1e-8)

        # Mask log probability: padding trials contribute 0 to likelihood
        log_prob_masked = log_prob * valid

        # =================================================================
        # 4. UPDATE WM: Immediate overwrite (masked)
        # =================================================================
        wm_current = WM_decayed[stimulus, action]
        WM_updated = WM_decayed.at[stimulus, action].set(
            jnp.where(valid, reward, wm_current)
        )

        # =================================================================
        # 5. UPDATE Q-TABLE: Asymmetric learning (masked)
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

        return (Q_updated, WM_updated, WM_baseline, log_lik_new), log_prob_masked

    # Run scan over trials
    (Q_final, WM_final, _, log_lik_total), log_probs = lax.scan(step_fn, init_carry, scan_inputs)

    if return_pointwise:
        return log_lik_total, log_probs
    return log_lik_total

def wmrl_multiblock_likelihood(
    stimuli_blocks: list,
    actions_blocks: list,
    rewards_blocks: list,
    set_sizes_blocks: list,
    alpha_pos: float,
    alpha_neg: float,
    phi: float,
    rho: float,
    capacity: float,
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
    Compute log-likelihood for WM-RL across MULTIPLE BLOCKS.

    Q-values and WM reset at each block boundary.

    Following Senta et al. (2025):
    - Beta is fixed at 50 (not a parameter)
    - Epsilon noise captures random responding

    PERFORMANCE NOTE: This function uses jax.lax.fori_loop instead of a Python
    for-loop. This is critical for GPU performance because:
    1. Python loops launch separate GPU kernels per iteration (17,000+ launches)
    2. JAX cannot fuse operations across Python loop boundaries
    3. fori_loop compiles the entire loop into ONE XLA computation

    Parameters
    ----------
    stimuli_blocks : list of arrays
        Stimulus sequences per block
    actions_blocks : list of arrays
        Action sequences per block
    rewards_blocks : list of arrays
        Reward sequences per block
    set_sizes_blocks : list of arrays
        Set sizes per trial per block
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
    epsilon : float
        Epsilon noise parameter (probability of random action)
    num_stimuli, num_actions : int
        State/action space dimensions
    q_init : float
        Initial Q-values
    wm_init : float
        Initial WM values (baseline = 1/nA for uniform)
    masks_blocks : list of arrays, optional
        List of mask arrays, one per block. Each mask has 1.0 for real
        trials and 0.0 for padding. If None, no masking is applied.
    verbose : bool
        Print progress
    participant_id : str
        For verbose output

    Returns
    -------
    float
        Total log-likelihood summed across blocks
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
            block_ll = wmrl_block_likelihood(
                stimuli=stimuli_stacked[block_idx],
                actions=actions_stacked[block_idx],
                rewards=rewards_stacked[block_idx],
                set_sizes=set_sizes_stacked[block_idx],
                alpha_pos=alpha_pos,
                alpha_neg=alpha_neg,
                phi=phi,
                rho=rho,
                capacity=capacity,
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
            block_log_lik = wmrl_block_likelihood(
                stimuli=stim_block,
                actions=act_block,
                rewards=rew_block,
                set_sizes=set_block,
                alpha_pos=alpha_pos,
                alpha_neg=alpha_neg,
                phi=phi,
                rho=rho,
                capacity=capacity,
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

def wmrl_multiblock_likelihood_stacked(
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
    epsilon: float = DEFAULT_EPSILON,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    *,
    return_pointwise: bool = False,
) -> float | tuple[float, jnp.ndarray]:
    """
    FAST WM-RL multiblock likelihood that takes pre-stacked arrays directly.

    See wmrl_multiblock_likelihood for full documentation.
    This version avoids list/restack overhead inside JIT.

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
            block_ll, block_probs = wmrl_block_likelihood(
                stimuli=stimuli_stacked[block_idx],
                actions=actions_stacked[block_idx],
                rewards=rewards_stacked[block_idx],
                set_sizes=set_sizes_stacked[block_idx],
                alpha_pos=alpha_pos,
                alpha_neg=alpha_neg,
                phi=phi,
                rho=rho,
                capacity=capacity,
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
            block_ll = wmrl_block_likelihood(
                stimuli=stimuli_stacked[block_idx],
                actions=actions_stacked[block_idx],
                rewards=rewards_stacked[block_idx],
                set_sizes=set_sizes_stacked[block_idx],
                alpha_pos=alpha_pos,
                alpha_neg=alpha_neg,
                phi=phi,
                rho=rho,
                capacity=capacity,
                epsilon=epsilon,
                num_stimuli=num_stimuli,
                num_actions=num_actions,
                q_init=q_init,
                wm_init=wm_init,
                mask=masks_stacked[block_idx]
            )
            return total_ll + block_ll

        return lax.fori_loop(0, num_blocks, body_fn, 0.0)

def wmrl_fully_batched_likelihood(
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
    epsilon: jnp.ndarray,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    use_pscan: bool = False,
) -> jnp.ndarray:
    """Fully-batched WM-RL (M2) log-likelihood via nested vmap.

    Participants and blocks are independent (Q, WM reset at block entry per
    Senta 2025), so outer=participants × inner=blocks vmap is correct.
    Padded blocks (mask entirely 0.0) contribute 0.0.

    Parameters
    ----------
    stimuli, actions, rewards, set_sizes, masks : jnp.ndarray
        Shape (N, B, T).  Participants × blocks × trials.
    alpha_pos, alpha_neg, phi, rho, capacity, epsilon : jnp.ndarray
        Shape (N,) per-participant parameter vectors.
    num_stimuli, num_actions, q_init, wm_init : int / float
        Static parameters.
    use_pscan : bool
        Must be False (pscan + vmap composition out of scope).

    Returns
    -------
    jnp.ndarray
        Shape (N,) — total log-likelihood per participant.
    """
    if use_pscan:
        raise NotImplementedError(
            "wmrl_fully_batched_likelihood: use_pscan=True is not supported."
        )

    def _block_ll(stim, act, rew, ss, mask, ap, an, ph, rh, cap, e):
        return wmrl_block_likelihood(
            stimuli=stim,
            actions=act,
            rewards=rew,
            set_sizes=ss,
            alpha_pos=ap,
            alpha_neg=an,
            phi=ph,
            rho=rh,
            capacity=cap,
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
        in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None),
        out_axes=0,
    )

    def _participant_ll(stim, act, rew, ss, mask, ap, an, ph, rh, cap, e):
        return _over_blocks(stim, act, rew, ss, mask, ap, an, ph, rh, cap, e).sum()

    _over_participants = jax.vmap(
        _participant_ll,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        out_axes=0,
    )
    return _over_participants(
        stimuli, actions, rewards, set_sizes, masks,
        alpha_pos, alpha_neg, phi, rho, capacity, epsilon,
    )

wmrl_block_likelihood_jit = jax.jit(
    wmrl_block_likelihood,
    static_argnums=(10, 11, 12, 13),  # num_stimuli, num_actions, q_init, wm_init are static
    static_argnames=("return_pointwise",),
)

def wmrl_block_likelihood_pscan(
    stimuli: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    set_sizes: jnp.ndarray,
    alpha_pos: float,
    alpha_neg: float,
    phi: float,
    rho: float,
    capacity: float,
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
    WM-RL block likelihood using parallel scan (M2 — no perseveration).

    Drop-in replacement for ``wmrl_block_likelihood``.

    Phase 1 (parallel): Pre-compute Q_for_policy[t] and wm_for_policy[t]
    via associative scans.
    Phase 2 (vectorized): Compute hybrid WM-Q policy, epsilon noise, and
    log-probs for all trials simultaneously. No perseveration carry needed
    (M2 has no choice kernel).

    Parameters
    ----------
    stimuli, actions, rewards, set_sizes : arrays, shape (n_trials,)
    alpha_pos, alpha_neg, phi, rho, capacity, epsilon : float
    num_stimuli, num_actions : int
    q_init, wm_init : float
    mask : array, shape (n_trials,), optional
    return_pointwise : bool, optional

    Returns
    -------
    float or tuple[float, jnp.ndarray]
        Same as ``wmrl_block_likelihood``.
    """
    if mask is None:
        mask = jnp.ones(len(stimuli))

    # Phase 1: parallel Q and WM trajectories
    T = stimuli.shape[0]
    Q_for_policy = associative_scan_q_update(
        stimuli, actions, rewards, mask,
        alpha_pos, alpha_neg, q_init,
        num_stimuli, num_actions,
    )  # (T, S, A)

    wm_for_policy, _ = associative_scan_wm_update(
        stimuli, actions, rewards, mask,
        phi, wm_init, num_stimuli, num_actions,
    )  # (T, S, A)

    # ------------------------------------------------------------------
    # Phase 2 (vectorized): hybrid WM-Q policy for all trials at once
    # ------------------------------------------------------------------
    t_idx = jnp.arange(T)
    q_vals = Q_for_policy[t_idx, stimuli]      # (T, A)
    wm_vals = wm_for_policy[t_idx, stimuli]    # (T, A)

    omega = rho * jnp.minimum(1.0, capacity / set_sizes)  # (T,)
    rl_probs = jax.vmap(softmax_policy, in_axes=(0, None))(q_vals, FIXED_BETA)
    wm_probs = jax.vmap(softmax_policy, in_axes=(0, None))(wm_vals, FIXED_BETA)
    base_probs = omega[:, None] * wm_probs + (1 - omega[:, None]) * rl_probs
    base_probs = base_probs / jnp.sum(base_probs, axis=-1, keepdims=True)
    noisy_probs = jax.vmap(apply_epsilon_noise, in_axes=(0, None, None))(
        base_probs, epsilon, num_actions
    )  # (T, A)
    log_probs = jnp.log(noisy_probs[t_idx, actions] + 1e-8) * mask

    if return_pointwise:
        return jnp.sum(log_probs), log_probs
    return jnp.sum(log_probs)

def wmrl_multiblock_likelihood_stacked_pscan(
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
    epsilon: float = DEFAULT_EPSILON,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    *,
    return_pointwise: bool = False,
) -> float | tuple[float, jnp.ndarray]:
    """
    WM-RL multiblock likelihood using parallel scan (M2).

    Drop-in replacement for ``wmrl_multiblock_likelihood_stacked``.

    Parameters
    ----------
    stimuli_stacked, actions_stacked, rewards_stacked,
    set_sizes_stacked, masks_stacked : arrays, shape (n_blocks, max_trials)
    alpha_pos, alpha_neg, phi, rho, capacity, epsilon : float
    num_stimuli, num_actions : int
    q_init, wm_init : float
    return_pointwise : bool, optional

    Returns
    -------
    float or tuple[float, jnp.ndarray]
        Same as ``wmrl_multiblock_likelihood_stacked``.
    """
    num_blocks = stimuli_stacked.shape[0]

    if return_pointwise:
        def scan_body(total_ll, block_idx):
            block_ll, block_probs = wmrl_block_likelihood_pscan(
                stimuli=stimuli_stacked[block_idx],
                actions=actions_stacked[block_idx],
                rewards=rewards_stacked[block_idx],
                set_sizes=set_sizes_stacked[block_idx],
                alpha_pos=alpha_pos,
                alpha_neg=alpha_neg,
                phi=phi,
                rho=rho,
                capacity=capacity,
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
            block_ll = wmrl_block_likelihood_pscan(
                stimuli=stimuli_stacked[block_idx],
                actions=actions_stacked[block_idx],
                rewards=rewards_stacked[block_idx],
                set_sizes=set_sizes_stacked[block_idx],
                alpha_pos=alpha_pos,
                alpha_neg=alpha_neg,
                phi=phi,
                rho=rho,
                capacity=capacity,
                epsilon=epsilon,
                num_stimuli=num_stimuli,
                num_actions=num_actions,
                q_init=q_init,
                wm_init=wm_init,
                mask=masks_stacked[block_idx],
            )
            return total_ll + block_ll

        return lax.fori_loop(0, num_blocks, body_fn, 0.0)

def test_wmrl_single_block():
    """Test WM-RL likelihood on a single block."""
    print("\nTesting WM-RL single block likelihood...")
    print(f"  (Using fixed beta={FIXED_BETA}, epsilon noise enabled)")

    # Create synthetic block (30 trials)
    key = jax.random.PRNGKey(42)
    n_trials = 30

    stimuli = jax.random.randint(key, (n_trials,), 0, 6)
    key, subkey = jax.random.split(key)
    actions = jax.random.randint(subkey, (n_trials,), 0, 3)
    key, subkey = jax.random.split(key)
    rewards = jax.random.bernoulli(subkey, 0.7, (n_trials,)).astype(jnp.float32)
    set_sizes = jnp.ones((n_trials,)) * 5  # Set size of 5

    # Test parameters (no beta/beta_wm - they're fixed at 50)
    params = {
        'alpha_pos': 0.3,
        'alpha_neg': 0.1,
        'phi': 0.1,
        'rho': 0.7,
        'capacity': 4.0,
        'epsilon': 0.05
    }

    # Compute likelihood
    log_lik = wmrl_block_likelihood(
        stimuli, actions, rewards, set_sizes, **params
    )

    print(f"[OK] WM-RL single block log-likelihood: {log_lik:.2f}")
    print(f"  Average log-prob per trial: {log_lik / n_trials:.3f}")

    # Test JIT compilation
    log_lik_jit = wmrl_block_likelihood_jit(
        stimuli, actions, rewards, set_sizes, **params
    )

    print(f"[OK] JIT-compiled result matches: {jnp.allclose(log_lik, log_lik_jit)}")

    return log_lik

def test_wmrl_multiblock():
    """Test WM-RL likelihood on multiple blocks."""
    print("\nTesting WM-RL multi-block likelihood...")
    print(f"  (Using fixed beta={FIXED_BETA}, epsilon noise enabled)")

    key = jax.random.PRNGKey(42)

    # Create 3 blocks of varying sizes
    block_sizes = [30, 60, 45]
    stimuli_blocks = []
    actions_blocks = []
    rewards_blocks = []
    set_sizes_blocks = []

    for i, size in enumerate(block_sizes):
        key, subkey = jax.random.split(key)
        stimuli_blocks.append(jax.random.randint(subkey, (size,), 0, 6))

        key, subkey = jax.random.split(key)
        actions_blocks.append(jax.random.randint(subkey, (size,), 0, 3))

        key, subkey = jax.random.split(key)
        rewards_blocks.append(jax.random.bernoulli(subkey, 0.7, (size,)).astype(jnp.float32))

        set_sizes_blocks.append(jnp.ones((size,)) * 5)

    # Test parameters (no beta/beta_wm - they're fixed at 50)
    params = {
        'alpha_pos': 0.3,
        'alpha_neg': 0.1,
        'phi': 0.1,
        'rho': 0.7,
        'capacity': 4.0,
        'epsilon': 0.05
    }

    # Compute likelihood
    log_lik = wmrl_multiblock_likelihood(
        stimuli_blocks, actions_blocks, rewards_blocks, set_sizes_blocks,
        **params
    )

    total_trials = sum(block_sizes)
    print(f"[OK] WM-RL multi-block log-likelihood: {log_lik:.2f}")
    print(f"  Total trials: {total_trials}")
    print(f"  Average log-prob per trial: {log_lik / total_trials:.3f}")

    # Verify it equals sum of individual blocks
    manual_sum = sum([
        wmrl_block_likelihood(stim, act, rew, sets, **params)
        for stim, act, rew, sets in zip(stimuli_blocks, actions_blocks, rewards_blocks, set_sizes_blocks)
    ])
    print(f"[OK] Matches manual block summation: {jnp.allclose(log_lik, manual_sum)}")

    return log_lik

def test_padding_equivalence_wmrl():
    """
    Verify padded and unpadded WM-RL likelihoods are mathematically equivalent.
    """
    print("\nTesting WM-RL padding equivalence (CRITICAL)...")

    key = jax.random.PRNGKey(123)
    n_real_trials = 45

    # Generate real trial data
    stimuli = jax.random.randint(key, (n_real_trials,), 0, 6)
    key, subkey = jax.random.split(key)
    actions = jax.random.randint(subkey, (n_real_trials,), 0, 3)
    key, subkey = jax.random.split(key)
    rewards = jax.random.bernoulli(subkey, 0.7, (n_real_trials,)).astype(jnp.float32)
    set_sizes = jnp.full((n_real_trials,), 5, dtype=jnp.int32)

    params = {
        'alpha_pos': 0.3, 'alpha_neg': 0.1, 'phi': 0.1,
        'rho': 0.7, 'capacity': 4.0, 'epsilon': 0.05
    }

    # Unpadded likelihood (original)
    log_lik_original = wmrl_block_likelihood(
        stimuli, actions, rewards, set_sizes, **params
    )

    # Padded likelihood (with mask)
    stimuli_pad, actions_pad, rewards_pad, set_sizes_pad, mask = pad_block_to_max(
        stimuli, actions, rewards, max_trials=100, set_sizes=set_sizes
    )
    log_lik_padded = wmrl_block_likelihood(
        stimuli_pad, actions_pad, rewards_pad, set_sizes_pad, **params, mask=mask
    )

    # Verify equivalence
    match = jnp.allclose(log_lik_original, log_lik_padded, rtol=1e-6)
    print(f"  Original (unpadded) log-lik: {log_lik_original:.8f}")
    print(f"  Padded (with mask) log-lik:  {log_lik_padded:.8f}")
    print(f"  Difference: {abs(float(log_lik_original - log_lik_padded)):.2e}")
    print(f"[OK] WM-RL padding equivalence: {'PASSED' if match else 'FAILED'}")

    assert match, "WM-RL padded/unpadded must be IDENTICAL!"
    return match

def test_multiblock_padding_equivalence():
    """
    Verify padding equivalence works across multiple blocks (full participant).
    """
    print("\nTesting multiblock padding equivalence...")

    key = jax.random.PRNGKey(789)

    # Create blocks of varying sizes (like real data: 30, 45, 75, 88, 90)
    block_sizes = [30, 45, 75]
    stimuli_blocks = []
    actions_blocks = []
    rewards_blocks = []

    for size in block_sizes:
        key, k1, k2, k3 = jax.random.split(key, 4)
        stimuli_blocks.append(jax.random.randint(k1, (size,), 0, 6))
        actions_blocks.append(jax.random.randint(k2, (size,), 0, 3))
        rewards_blocks.append(jax.random.bernoulli(k3, 0.7, (size,)).astype(jnp.float32))

    params = {'alpha_pos': 0.3, 'alpha_neg': 0.1, 'epsilon': 0.05}

    # Unpadded multiblock likelihood
    log_lik_original = q_learning_multiblock_likelihood(
        stimuli_blocks, actions_blocks, rewards_blocks, **params
    )

    # Padded multiblock likelihood
    stimuli_padded = []
    actions_padded = []
    rewards_padded = []
    masks = []

    for stim, act, rew in zip(stimuli_blocks, actions_blocks, rewards_blocks):
        s_pad, a_pad, r_pad, mask = pad_block_to_max(stim, act, rew, max_trials=100)
        stimuli_padded.append(s_pad)
        actions_padded.append(a_pad)
        rewards_padded.append(r_pad)
        masks.append(mask)

    log_lik_padded = q_learning_multiblock_likelihood(
        stimuli_padded, actions_padded, rewards_padded,
        masks_blocks=masks, **params
    )

    match = jnp.allclose(log_lik_original, log_lik_padded, rtol=1e-6)
    print(f"  Original (variable sizes) log-lik: {log_lik_original:.8f}")
    print(f"  Padded (fixed size 100) log-lik:   {log_lik_padded:.8f}")
    print(f"  Difference: {abs(float(log_lik_original - log_lik_padded)):.2e}")
    print(f"[OK] Multiblock padding equivalence: {'PASSED' if match else 'FAILED'}")

    assert match, "Multiblock padded/unpadded must be IDENTICAL!"
    return match

def wmrl_hierarchical_model(
    participant_data: dict[Any, dict[str, list]],
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,  # WM baseline = 1/nA (uniform)
) -> None:
    """
    .. deprecated:: v4.0

        Uses pre-Collins K bounds [1, 7] and Python for-loop per-participant
        factor sites. Superseded by :func:`wmrl_hierarchical_model_stacked`
        (line 1446) which uses Collins K bounds [2, 6] via
        ``numpyro_helpers.sample_bounded_param`` and fully-batched ``jax.vmap``.
        This legacy function is NOT wired into ``STACKED_MODEL_DISPATCH`` and
        is retained only as a regression-test target for
        ``scripts/fitting/tests/test_wmrl_model.py`` (compilation smoke test).
        Any production fit must go through the stacked variant.

    Hierarchical Bayesian WM-RL hybrid model for multiple participants.

    Following Senta et al. (2025):
    - Beta is FIXED at 50 for both WM and RL (not estimated)
    - Epsilon noise parameter is estimated to capture random responding

    Model Structure:
    ---------------
    # Group-level (population) parameters
    mu_alpha_pos ~ Beta(3, 2)         # Mean positive learning rate ~ 0.6
    sigma_alpha_pos ~ HalfNormal(0.3) # Variability in alpha_pos
    mu_alpha_neg ~ Beta(2, 3)         # Mean negative learning rate ~ 0.4
    sigma_alpha_neg ~ HalfNormal(0.3) # Variability in alpha_neg
    mu_phi ~ Beta(2, 8)               # Mean WM decay rate ~ 0.2
    sigma_phi ~ HalfNormal(0.3)       # Variability in phi
    mu_rho ~ Beta(5, 2)               # Mean WM reliance ~ 0.7
    sigma_rho ~ HalfNormal(0.3)       # Variability in rho
    mu_K ~ TruncatedNormal(4, 1.5, low=1, high=7)  # Mean WM capacity ~ 4
    sigma_K ~ HalfNormal(1.0)         # Variability in K
    mu_epsilon ~ Beta(1, 19)          # Mean epsilon noise ~ 0.05
    sigma_epsilon ~ HalfNormal(0.1)   # Variability in epsilon

    # Individual-level parameters (non-centered)
    All parameters transformed via logit/log for Beta/bounded distributions

    # Likelihood (beta=50 fixed)
    actions_i ~ Hybrid(WM, RL; all parameters, beta=50)

    Parameters
    ----------
    participant_data : dict
        Nested dict: {participant_id: {
            'stimuli_blocks': list of arrays,
            'actions_blocks': list of arrays,
            'rewards_blocks': list of arrays,
            'set_sizes_blocks': list of arrays  # Set sizes for adaptive weighting
        }}
    num_stimuli : int
        Number of possible stimuli
    num_actions : int
        Number of possible actions
    q_init : float
        Initial Q-value for all state-action pairs
    wm_init : float
        Initial WM values (baseline = 1/nA for uniform)

    Notes
    -----
    This function is used with numpyro.infer.MCMC for sampling.
    """
    num_participants = len(participant_data)
    participant_ids = list(participant_data.keys())

    # ========================================================================
    # GROUP-LEVEL (POPULATION) PRIORS
    # ========================================================================

    # Positive learning rate: bounded [0, 1]
    mu_alpha_pos = numpyro.sample("mu_alpha_pos", dist.Beta(3, 2))
    sigma_alpha_pos = numpyro.sample("sigma_alpha_pos", dist.HalfNormal(0.3))

    # Negative learning rate: bounded [0, 1]
    mu_alpha_neg = numpyro.sample("mu_alpha_neg", dist.Beta(2, 3))
    sigma_alpha_neg = numpyro.sample("sigma_alpha_neg", dist.HalfNormal(0.3))

    # WM decay rate: bounded [0, 1]
    mu_phi = numpyro.sample("mu_phi", dist.Beta(2, 8))
    sigma_phi = numpyro.sample("sigma_phi", dist.HalfNormal(0.3))

    # WM reliance: bounded [0, 1]
    mu_rho = numpyro.sample("mu_rho", dist.Beta(5, 2))
    sigma_rho = numpyro.sample("sigma_rho", dist.HalfNormal(0.3))

    # WM capacity: bounded [1, 7] (using truncated normal)
    mu_capacity = numpyro.sample(
        "mu_capacity", dist.TruncatedNormal(4.0, 1.5, low=1.0, high=7.0)
    )
    sigma_capacity = numpyro.sample("sigma_capacity", dist.HalfNormal(1.0))

    # Epsilon noise: bounded [0, 1], prior centered around 0.05
    mu_epsilon = numpyro.sample("mu_epsilon", dist.Beta(1, 19))
    sigma_epsilon = numpyro.sample("sigma_epsilon", dist.HalfNormal(0.1))

    # ========================================================================
    # INDIVIDUAL-LEVEL PARAMETERS (NON-CENTERED)
    # ========================================================================

    with numpyro.plate("participants", num_participants):
        # Sample standard normal offsets
        z_alpha_pos = numpyro.sample("z_alpha_pos", dist.Normal(0, 1))
        z_alpha_neg = numpyro.sample("z_alpha_neg", dist.Normal(0, 1))
        z_phi = numpyro.sample("z_phi", dist.Normal(0, 1))
        z_rho = numpyro.sample("z_rho", dist.Normal(0, 1))
        z_capacity = numpyro.sample("z_capacity", dist.Normal(0, 1))
        z_epsilon = numpyro.sample("z_epsilon", dist.Normal(0, 1))

        # Transform to constrained space via logit transformation
        alpha_pos = numpyro.deterministic(
            "alpha_pos",
            jax.scipy.special.expit(
                jax.scipy.special.logit(mu_alpha_pos) + sigma_alpha_pos * z_alpha_pos
            ),
        )
        alpha_neg = numpyro.deterministic(
            "alpha_neg",
            jax.scipy.special.expit(
                jax.scipy.special.logit(mu_alpha_neg) + sigma_alpha_neg * z_alpha_neg
            ),
        )
        phi = numpyro.deterministic(
            "phi",
            jax.scipy.special.expit(
                jax.scipy.special.logit(mu_phi) + sigma_phi * z_phi
            ),
        )
        rho = numpyro.deterministic(
            "rho",
            jax.scipy.special.expit(
                jax.scipy.special.logit(mu_rho) + sigma_rho * z_rho
            ),
        )
        epsilon = numpyro.deterministic(
            "epsilon",
            jax.scipy.special.expit(
                jax.scipy.special.logit(mu_epsilon) + sigma_epsilon * z_epsilon
            ),
        )

        # For capacity: use clipped normal transformation
        capacity = numpyro.deterministic(
            "capacity",
            jnp.clip(mu_capacity + sigma_capacity * z_capacity, 1.0, 7.0),
        )

    # ========================================================================
    # LIKELIHOOD: Compute for each participant
    # ========================================================================

    for i, participant_id in enumerate(participant_ids):
        pdata = participant_data[participant_id]

        # Get individual parameters
        alpha_pos_i = alpha_pos[i]
        alpha_neg_i = alpha_neg[i]
        phi_i = phi[i]
        rho_i = rho[i]
        capacity_i = capacity[i]
        epsilon_i = epsilon[i]

        # Compute log-likelihood across all blocks for this participant
        # Note: beta is fixed at 50 inside the likelihood function
        log_lik = wmrl_multiblock_likelihood(
            stimuli_blocks=pdata["stimuli_blocks"],
            actions_blocks=pdata["actions_blocks"],
            rewards_blocks=pdata["rewards_blocks"],
            set_sizes_blocks=pdata["set_sizes_blocks"],
            alpha_pos=alpha_pos_i,
            alpha_neg=alpha_neg_i,
            phi=phi_i,
            rho=rho_i,
            capacity=capacity_i,
            epsilon=epsilon_i,
            num_stimuli=num_stimuli,
            num_actions=num_actions,
            q_init=q_init,
            wm_init=wm_init,
        )

        # Add to model via factor (log probability)
        numpyro.factor(f"obs_p{participant_id}", log_lik)

def wmrl_hierarchical_model_stacked(
    participant_data_stacked: dict,
    covariate_lec: jnp.ndarray | None = None,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    use_pscan: bool = False,
    stacked_arrays: dict | None = None,
) -> None:
    """Hierarchical Bayesian M2 (WM-RL) model using fully-batched vmap likelihood.

    Uses the hBayesDM non-centered parameterization convention (Ahn, Haines,
    Zhang 2017):
    ``theta_unc = mu_pr + sigma_pr * z``,
    ``theta = lower + (upper - lower) * Phi_approx(theta_unc)``,
    where ``Phi_approx = jax.scipy.stats.norm.cdf``.

    Six parameters (alpha_pos, alpha_neg, phi, rho, capacity, epsilon) are
    sampled via :func:`sample_bounded_param` from
    :mod:`rlwm.fitting.numpyro_helpers`.

    Likelihood is accumulated via a single ``numpyro.factor("obs", ...)`` call
    using ``wmrl_fully_batched_likelihood`` (nested vmap over participants and
    blocks).  This removes the 154-factor Python for-loop that was the main
    NUTS leapfrog bottleneck.

    Parameters
    ----------
    participant_data_stacked : dict
        Mapping from participant_id to stacked-format arrays.
    covariate_lec : jnp.ndarray or None
        Must be ``None``; M2 has no perseveration parameter as L2 target.
    num_stimuli, num_actions, q_init, wm_init : int / float
        Static model parameters.
    use_pscan : bool
        Must be ``False``.  The fully-batched vmap path does not compose with
        the O(log T) associative scan variants.
    stacked_arrays : dict or None
        Pre-computed output of ``stack_across_participants``.  If ``None``,
        computed here.  ``fit_bayesian.py`` passes this to avoid recomputing
        (N, B, T) tensors on every MCMC trace call.
    """
    if covariate_lec is not None:
        raise NotImplementedError(
            "wmrl_hierarchical_model_stacked: covariate_lec is not supported "
            "for M2 WM-RL (no perseveration parameter as L2 target). "
            "Pass covariate_lec=None."
        )
    if use_pscan:
        raise NotImplementedError(
            "wmrl_hierarchical_model_stacked: use_pscan + fully-batched vmap "
            "path is not implemented.  Pass use_pscan=False."
        )

    from rlwm.fitting.numpyro_helpers import (
        PARAM_PRIOR_DEFAULTS,
        sample_bounded_param,
    )

    n_participants = len(participant_data_stacked)

    # ------------------------------------------------------------------
    # Group priors for 6 parameters via hBayesDM non-centered convention
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
    # Likelihood via single numpyro.factor("obs", ...) — fully-batched
    # vmap over participants and blocks.
    # ------------------------------------------------------------------
    if stacked_arrays is None:
        stacked_arrays = stack_across_participants(participant_data_stacked)

    from rlwm.fitting.jax_likelihoods import wmrl_fully_batched_likelihood

    per_participant_ll = wmrl_fully_batched_likelihood(
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
        epsilon=sampled["epsilon"],
        num_stimuli=num_stimuli,
        num_actions=num_actions,
        q_init=q_init,
        wm_init=wm_init,
    )
    numpyro.factor("obs", per_participant_ll.sum())
