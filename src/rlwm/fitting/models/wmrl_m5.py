"""M5 WM-RL + kappa + phi_rl (RL forgetting): JAX likelihoods + NumPyro hierarchical wrapper.

Relocated here in Phase 29-08 from :mod:`rlwm.fitting.jax_likelihoods` and
:mod:`rlwm.fitting.numpyro_models`. Old import paths remain available via
wildcard re-export shims.

M5 extends M3 with an additional ``phi_rl`` RL-forgetting parameter (Q-values
decay toward the chance baseline between updates). Current winning choice-only
model per 29-CONTEXT.md (dAIC=435.6 vs. M3).
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
    prepare_stacked_participant_data,
    softmax_policy,
    stack_across_participants,
)

__all__ = [
    "wmrl_m5_fully_batched_likelihood",
    "wmrl_m5_block_likelihood",
    "wmrl_m5_multiblock_likelihood",
    "wmrl_m5_multiblock_likelihood_stacked",
    "wmrl_m5_block_likelihood_pscan",
    "wmrl_m5_multiblock_likelihood_stacked_pscan",
    "test_wmrl_m5_single_block",
    "test_wmrl_m5_backward_compatibility",
    "test_padding_equivalence_wmrl_m5",
    "wmrl_m5_hierarchical_model",
]


def wmrl_m5_fully_batched_likelihood(
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
    phi_rl: jnp.ndarray,
    epsilon: jnp.ndarray,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    use_pscan: bool = False,
) -> jnp.ndarray:
    """Fully-batched WM-RL+phi_rl (M5) log-likelihood via nested vmap.

    Extends M3 by adding phi_rl (RL forgetting rate) as an additional
    per-participant parameter.  When phi_rl=0, reduces to M3.  See
    ``wmrl_m5_block_likelihood`` for details.

    Parameters
    ----------
    stimuli, actions, rewards, set_sizes, masks : jnp.ndarray
        Shape (N, B, T).
    alpha_pos, alpha_neg, phi, rho, capacity, kappa, phi_rl, epsilon : jnp.ndarray
        Shape (N,) per-participant parameter vectors.

    Returns
    -------
    jnp.ndarray
        Shape (N,) — total log-likelihood per participant.
    """
    if use_pscan:
        raise NotImplementedError(
            "wmrl_m5_fully_batched_likelihood: use_pscan=True is not supported."
        )

    def _block_ll(stim, act, rew, ss, mask, ap, an, ph, rh, cap, k, prl, e):
        return wmrl_m5_block_likelihood(
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
            phi_rl=prl,
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

    def _participant_ll(stim, act, rew, ss, mask, ap, an, ph, rh, cap, k, prl, e):
        return _over_blocks(
            stim, act, rew, ss, mask, ap, an, ph, rh, cap, k, prl, e,
        ).sum()

    _over_participants = jax.vmap(
        _participant_ll,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        out_axes=0,
    )
    return _over_participants(
        stimuli, actions, rewards, set_sizes, masks,
        alpha_pos, alpha_neg, phi, rho, capacity, kappa, phi_rl, epsilon,
    )

def wmrl_m5_block_likelihood(
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
    phi_rl: float,  # RL forgetting rate: decay Q-values toward Q0=1/nA before delta-rule
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
    Compute log-likelihood for WM-RL M5 model with RL forgetting on a SINGLE BLOCK.

    M5 extends M3 by adding global per-trial Q-value decay toward baseline (Q0=1/nA)
    BEFORE the delta-rule update. This dissociates RL forgetting from WM decay (phi),
    enabling the model to capture participants who forget learned Q-values between trials.

    Following Senta et al. (2025) M3 model, with additional Step 1a:
    - Beta is fixed at 50 for both WM and RL (parameter identifiability)
    - Epsilon noise is applied to the base policy before perseveration mixing
    - WM baseline = 1/nA (uniform probability)
    - phi_rl decay target is Q0 = 1/nA = 0.333 (NOT q_init=0.5)

    When phi_rl=0, Q_decayed = Q_table, so M5 reduces exactly to M3.

    Update sequence per trial:
    1. Decay WM: WM <- (1-phi)WM + phi*WM_0
    1a. RL forgetting: Q_decayed = (1-phi_rl)*Q_table + phi_rl*Q0
    2. Compute hybrid policy using Q_decayed (not Q_table):
       - P_base = omega*softmax(WM) + (1-omega)*softmax(Q_decayed)
       - Apply epsilon: P_noisy = eps/nA + (1-eps)*P_base
       - If kappa>0: P_M5 = (1-kappa)*P_noisy + kappa*Ck
    3. Update WM: WM(s,a) <- r (immediate overwrite)
    4. Update Q: Q_decayed(s,a) <- Q_decayed(s,a) + alpha*(r - Q_decayed(s,a))

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
    kappa : float
        Perseveration parameter (0-1) - captures motor-level action stickiness
    phi_rl : float
        RL forgetting rate (0-1) - per-trial Q-value decay toward Q0=1/nA
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

    # Q0 = uniform prior (decay target for RL forgetting)
    # CRITICAL: This is 1/nA = 0.333, NOT q_init = 0.5
    Q0 = 1.0 / num_actions

    # Initial carry: (Q, WM, WM_0, log_likelihood, last_action)
    # last_action = -1 (no previous action at block start)
    init_carry = (Q_init, WM_init, WM_0, 0.0, -1)

    # If no mask provided, all trials are valid (backward compatibility)
    if mask is None:
        mask = jnp.ones(len(stimuli))

    # Prepare inputs (include mask)
    scan_inputs = (stimuli, actions, rewards, set_sizes, mask)

    def step_fn(carry, inputs):
        Q_table, WM_table, WM_baseline, log_lik_accum, last_action = carry
        stimulus, action, reward, set_size, valid = inputs

        # =================================================================
        # 1. DECAY WM: All values move toward baseline
        # =================================================================
        WM_decayed = (1 - phi) * WM_table + phi * WM_baseline

        # =================================================================
        # 1a. RL FORGETTING: Q-values decay toward Q0 BEFORE delta-rule
        # When phi_rl=0: Q_decayed = Q_table (algebraic identity, exact M3)
        # =================================================================
        Q_decayed = (1 - phi_rl) * Q_table + phi_rl * Q0

        # =================================================================
        # 2. COMPUTE HYBRID POLICY (using Q_decayed, not Q_table)
        # =================================================================
        q_vals = Q_decayed[stimulus]
        wm_vals = WM_decayed[stimulus]

        # Adaptive weight: omega = rho * min(1, K/N_s)
        omega = rho * jnp.minimum(1.0, capacity / set_size)

        # Branch: M2 (probability mixing) vs M5 (probability mixing + perseveration)
        use_m2_path = jnp.logical_or(kappa == 0.0, last_action < 0)

        # =================================================================
        # Both paths start with M2 probability mixing
        # =================================================================
        rl_probs = softmax_policy(q_vals, FIXED_BETA)
        wm_probs = softmax_policy(wm_vals, FIXED_BETA)
        base_probs = omega * wm_probs + (1 - omega) * rl_probs
        base_probs = base_probs / jnp.sum(base_probs)  # Normalize

        # Apply epsilon noise to base policy
        noisy_base = apply_epsilon_noise(base_probs, epsilon, num_actions)

        # =================================================================
        # M5 path: Probability mixing with choice kernel (same as M3)
        # P_M5 = (1-kappa)*P_noisy + kappa*Ck where Ck = one-hot(last_action)
        # =================================================================
        # Choice kernel = one-hot of last action (tau=1 simplification)
        choice_kernel = jnp.eye(num_actions)[jnp.maximum(last_action, 0)]  # Clamp for indexing

        # Probability mixing: (1-kappa)*noisy_base + kappa*choice_kernel
        hybrid_probs_m5 = (1 - kappa) * noisy_base + kappa * choice_kernel

        # Select correct path: M2 uses noisy_base, M5 uses probability mixing
        noisy_probs = jnp.where(use_m2_path, noisy_base, hybrid_probs_m5)

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
        # 4. UPDATE Q-TABLE: Asymmetric learning on Q_decayed (masked)
        # =================================================================
        q_current = Q_decayed[stimulus, action]
        delta = reward - q_current
        alpha = jnp.where(delta > 0, alpha_pos, alpha_neg)
        q_updated = q_current + alpha * delta
        Q_updated = Q_decayed.at[stimulus, action].set(
            jnp.where(valid, q_updated, q_current)
        )

        # Accumulate log-likelihood
        log_lik_new = log_lik_accum + log_prob_masked

        # Update last_action only for valid trials (masked perseveration)
        # For padding trials, keep the previous last_action
        new_last_action = jnp.where(valid, action, last_action).astype(jnp.int32)

        # Return updated carry (Q_updated derived from Q_decayed)
        return (Q_updated, WM_updated, WM_baseline, log_lik_new, new_last_action), log_prob_masked

    # Run scan over trials
    (Q_final, WM_final, _, log_lik_total, _), log_probs = lax.scan(step_fn, init_carry, scan_inputs)

    if return_pointwise:
        return log_lik_total, log_probs
    return log_lik_total

def wmrl_m5_multiblock_likelihood(
    stimuli_blocks: list,
    actions_blocks: list,
    rewards_blocks: list,
    set_sizes_blocks: list,
    alpha_pos: float,
    alpha_neg: float,
    phi: float,
    rho: float,
    capacity: float,
    kappa: float,
    phi_rl: float,  # RL forgetting rate
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
    Compute log-likelihood for WM-RL M5 (RL forgetting) across MULTIPLE BLOCKS.

    Q-values, WM, and last_action reset at each block boundary.

    When phi_rl=0, results match wmrl_m3_multiblock_likelihood exactly (M3 model).

    PERFORMANCE NOTE: Uses jax.lax.fori_loop for GPU-efficient computation.

    Parameters
    ----------
    stimuli_blocks : list of arrays
    actions_blocks : list of arrays
    rewards_blocks : list of arrays
    set_sizes_blocks : list of arrays
    alpha_pos, alpha_neg : float
    phi : float (WM decay)
    rho : float (WM reliance)
    capacity : float (WM capacity)
    kappa : float (perseveration)
    phi_rl : float (RL forgetting rate)
    epsilon : float
    masks_blocks : list of arrays, optional

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
            block_ll = wmrl_m5_block_likelihood(
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
                phi_rl=phi_rl,
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
            block_log_lik = wmrl_m5_block_likelihood(
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
                phi_rl=phi_rl,
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

def wmrl_m5_multiblock_likelihood_stacked(
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
    phi_rl: float,
    epsilon: float = DEFAULT_EPSILON,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    *,
    return_pointwise: bool = False,
) -> float | tuple[float, jnp.ndarray]:
    """
    FAST WM-RL M5 multiblock likelihood that takes pre-stacked arrays directly.

    See wmrl_m5_multiblock_likelihood for full documentation.
    This version avoids list/restack overhead inside JIT.
    When phi_rl=0, results match wmrl_m3_multiblock_likelihood_stacked exactly.

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
            block_ll, block_probs = wmrl_m5_block_likelihood(
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
                phi_rl=phi_rl,
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
            block_ll = wmrl_m5_block_likelihood(
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
                phi_rl=phi_rl,
                epsilon=epsilon,
                num_stimuli=num_stimuli,
                num_actions=num_actions,
                q_init=q_init,
                wm_init=wm_init,
                mask=masks_stacked[block_idx]
            )
            return total_ll + block_ll

        return lax.fori_loop(0, num_blocks, body_fn, 0.0)

def wmrl_m5_block_likelihood_pscan(
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
    phi_rl: float,
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
    WM-RL M5 block likelihood using parallel scan (RL forgetting + perseveration).

    Drop-in replacement for ``wmrl_m5_block_likelihood``.

    Phase 1 (parallel): Q scan uses a composed affine operator that combines
    per-trial Q-forgetting (phi_rl) and the delta-rule update.  WM scan uses
    the standard ``associative_scan_wm_update``.

    **M5 Q-forgetting composition:**
    For each trial t at the active (s, a) pair the sequential update is:
      Q_decayed = (1-phi_rl)*Q + phi_rl*Q0          [decay toward Q0=1/nA]
      Q_updated = Q_decayed + alpha*(r - Q_decayed)  [delta-rule on decayed]
    Combined: Q_updated = (1-alpha)*(1-phi_rl)*Q + (1-alpha)*phi_rl*Q0 + alpha*r
    As AR(1): a_t = (1-alpha)*(1-phi_rl),  b_t = (1-alpha)*phi_rl*Q0 + alpha*r

    For inactive (s', a') pairs at trial t:
      Q_decayed = (1-phi_rl)*Q + phi_rl*Q0           [decay only]
    As AR(1): a_t = 1-phi_rl, b_t = phi_rl*Q0

    For padding trials (mask=0): same as inactive — decay only, no update.

    Phase 2 (vectorized): Precompute global last_action array via
    ``precompute_last_action_global``, then compute hybrid WM-Q policy
    with global perseveration kernel for all trials simultaneously.
    Identical to M3 Phase 2 since M5 only differs in Phase 1.

    Parameters
    ----------
    stimuli, actions, rewards, set_sizes : arrays, shape (n_trials,)
    alpha_pos, alpha_neg, phi, rho, capacity, kappa, phi_rl, epsilon : float
    num_stimuli, num_actions : int
    q_init, wm_init : float
    mask : array, optional
    return_pointwise : bool, optional

    Returns
    -------
    float or tuple[float, jnp.ndarray]
        Same as ``wmrl_m5_block_likelihood``.
    """
    if mask is None:
        mask = jnp.ones(len(stimuli))

    T = stimuli.shape[0]
    S, A = num_stimuli, num_actions
    Q0 = 1.0 / A  # RL forgetting target (1/nA)

    # ------------------------------------------------------------------
    # Phase 1a: Build composed affine coefficients for M5 Q-update
    # ------------------------------------------------------------------
    stim_oh = jax.nn.one_hot(stimuli, S)    # (T, S)
    act_oh = jax.nn.one_hot(actions, A)     # (T, A)
    sa_mask = stim_oh[:, :, None] * act_oh[:, None, :]  # (T, S, A)
    active = sa_mask * mask[:, None, None]  # (T, S, A)

    # Reward-based alpha approximation (same as standard Q scan)
    alpha_t = jnp.where(
        rewards[:, None, None] == 1.0,
        alpha_pos,
        alpha_neg,
    )  # (T, S, A)

    # Composed coefficients for active (learning) positions:
    #   a = (1-alpha)*(1-phi_rl),  b = (1-alpha)*phi_rl*Q0 + alpha*r
    a_active = (1.0 - alpha_t) * (1.0 - phi_rl)
    b_active = (1.0 - alpha_t) * phi_rl * Q0 + alpha_t * rewards[:, None, None]

    # Coefficients for inactive / padding positions (decay only):
    #   a = 1-phi_rl,  b = phi_rl*Q0
    a_decay = 1.0 - phi_rl
    b_decay = phi_rl * Q0

    a_seq = jnp.where(active, a_active, a_decay)
    b_seq = jnp.where(active, b_active, b_decay)

    q_init_table = jnp.ones((S, A)) * q_init

    # affine_scan returns Q AFTER update at each trial
    Q_all = affine_scan(a_seq, b_seq, x0=q_init_table)  # (T, S, A)
    # Q_for_policy[t] = Q BEFORE update at trial t
    Q_for_policy = jnp.concatenate([q_init_table[None], Q_all[:-1]], axis=0)

    # Phase 1b: WM trajectories (same as M3)
    wm_for_policy, _ = associative_scan_wm_update(
        stimuli, actions, rewards, mask,
        phi, wm_init, num_stimuli, num_actions,
    )  # (T, S, A)

    # Phase 1c: Derive Q_decayed_for_policy
    # The policy at trial t uses Q_decayed = (1-phi_rl)*Q_carry_in + phi_rl*Q0,
    # where Q_carry_in[t] = Q BEFORE the combined (decay+update) at trial t.
    # Our Q_for_policy contains the carry-in values (= Q_all[t-1] prepended with
    # Q_init). Apply one phi_rl decay step to recover Q_decayed as seen by policy.
    # This mirrors wm_for_policy recovery in associative_scan_wm_update.
    Q_decayed_for_policy = (1.0 - phi_rl) * Q_for_policy + phi_rl * Q0  # (T, S, A)

    # ------------------------------------------------------------------
    # Phase 2 (vectorized): hybrid policy + global perseveration
    # ------------------------------------------------------------------
    t_idx = jnp.arange(T)
    q_vals = Q_decayed_for_policy[t_idx, stimuli]  # (T, A)
    wm_vals = wm_for_policy[t_idx, stimuli]        # (T, A)

    omega = rho * jnp.minimum(1.0, capacity / set_sizes)  # (T,)
    rl_probs = jax.vmap(softmax_policy, in_axes=(0, None))(q_vals, FIXED_BETA)
    wm_probs = jax.vmap(softmax_policy, in_axes=(0, None))(wm_vals, FIXED_BETA)
    base_probs = omega[:, None] * wm_probs + (1 - omega[:, None]) * rl_probs
    base_probs = base_probs / jnp.sum(base_probs, axis=-1, keepdims=True)

    noisy_base = jax.vmap(apply_epsilon_noise, in_axes=(0, None, None))(
        base_probs, epsilon, num_actions
    )  # (T, A)

    # Precompute global last_action for perseveration
    last_action_pre = precompute_last_action_global(actions, mask)  # (T,)
    use_m2_path = jnp.logical_or(kappa == 0.0, last_action_pre < 0)  # (T,)
    choice_kernels = jnp.eye(num_actions)[jnp.maximum(last_action_pre, 0)]  # (T, A)
    hybrid_probs = (1 - kappa) * noisy_base + kappa * choice_kernels
    noisy_probs = jnp.where(use_m2_path[:, None], noisy_base, hybrid_probs)  # (T, A)

    log_probs = jnp.log(noisy_probs[t_idx, actions] + 1e-8) * mask

    if return_pointwise:
        return jnp.sum(log_probs), log_probs
    return jnp.sum(log_probs)

def wmrl_m5_multiblock_likelihood_stacked_pscan(
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
    phi_rl: float,
    epsilon: float = DEFAULT_EPSILON,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    *,
    return_pointwise: bool = False,
) -> float | tuple[float, jnp.ndarray]:
    """
    WM-RL M5 multiblock likelihood using parallel scan.

    Drop-in replacement for ``wmrl_m5_multiblock_likelihood_stacked``.

    Parameters
    ----------
    stimuli_stacked, actions_stacked, rewards_stacked,
    set_sizes_stacked, masks_stacked : arrays, shape (n_blocks, max_trials)
    alpha_pos, alpha_neg, phi, rho, capacity, kappa, phi_rl, epsilon : float
    num_stimuli, num_actions : int
    q_init, wm_init : float
    return_pointwise : bool, optional

    Returns
    -------
    float or tuple[float, jnp.ndarray]
        Same as ``wmrl_m5_multiblock_likelihood_stacked``.
    """
    num_blocks = stimuli_stacked.shape[0]

    if return_pointwise:
        def scan_body(total_ll, block_idx):
            block_ll, block_probs = wmrl_m5_block_likelihood_pscan(
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
                phi_rl=phi_rl,
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
            block_ll = wmrl_m5_block_likelihood_pscan(
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
                phi_rl=phi_rl,
                epsilon=epsilon,
                num_stimuli=num_stimuli,
                num_actions=num_actions,
                q_init=q_init,
                wm_init=wm_init,
                mask=masks_stacked[block_idx],
            )
            return total_ll + block_ll

        return lax.fori_loop(0, num_blocks, body_fn, 0.0)

def test_wmrl_m5_single_block():
    """Smoke test for WM-RL M5 single block likelihood."""
    print("\nTesting WM-RL M5 single block likelihood (smoke test)...")

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
        'kappa': 0.3,
        'phi_rl': 0.2,  # Non-zero RL forgetting
        'epsilon': 0.05,
    }

    log_lik = wmrl_m5_block_likelihood(
        stimuli, actions, rewards, set_sizes, **params
    )

    nll = -float(log_lik)
    is_finite = jnp.isfinite(log_lik)
    is_negative = log_lik < 0  # log-likelihood should be negative

    print(f"  WM-RL M5 log-likelihood: {float(log_lik):.4f}")
    print(f"  NLL: {nll:.4f}")
    print(f"  Is finite: {bool(is_finite)}")
    print(f"  Is negative (expected): {bool(is_negative)}")

    assert bool(is_finite), "M5 log-likelihood must be finite!"
    print("[OK] WM-RL M5 single block smoke test passed")
    return log_lik

def test_wmrl_m5_backward_compatibility():
    """Verify M5 with phi_rl=0 matches M3 exactly."""
    print("\nTesting WM-RL M5 backward compatibility (phi_rl=0 == M3)...")

    key = jax.random.PRNGKey(123)
    n_trials = 50

    # IDENTICAL test data for both M3 and M5 calls
    stimuli = jax.random.randint(key, (n_trials,), 0, 6)
    key, subkey = jax.random.split(key)
    actions = jax.random.randint(subkey, (n_trials,), 0, 3)
    key, subkey = jax.random.split(key)
    rewards = jax.random.bernoulli(subkey, 0.7, (n_trials,)).astype(jnp.float32)
    set_sizes = jnp.ones((n_trials,)) * 5

    # kappa=0.3 (non-zero to exercise perseveration path in both models)
    params_m3 = {
        'alpha_pos': 0.3,
        'alpha_neg': 0.1,
        'phi': 0.1,
        'rho': 0.7,
        'capacity': 4.0,
        'kappa': 0.3,
        'epsilon': 0.05,
    }

    # M3 likelihood
    log_lik_m3 = wmrl_m3_block_likelihood(
        stimuli, actions, rewards, set_sizes, **params_m3
    )

    # M5 with phi_rl=0.0 (should match M3 exactly)
    log_lik_m5 = wmrl_m5_block_likelihood(
        stimuli, actions, rewards, set_sizes,
        **params_m3, phi_rl=0.0
    )

    match = jnp.allclose(log_lik_m3, log_lik_m5, atol=1e-5)
    print(f"  M3 log-likelihood:          {float(log_lik_m3):.8f}")
    print(f"  M5 (phi_rl=0) log-likelihood: {float(log_lik_m5):.8f}")
    print(f"  Difference: {abs(float(log_lik_m3 - log_lik_m5)):.2e}")
    print(f"  Backward compatibility verified: {bool(match)}")

    assert bool(match), "M5 with phi_rl=0 should match M3 exactly!"
    print("[OK] WM-RL M5 backward compatibility test passed")
    return match

def test_padding_equivalence_wmrl_m5():
    """
    Verify padded and unpadded WM-RL M5 likelihoods are mathematically equivalent.
    """
    print("\nTesting WM-RL M5 padding equivalence (CRITICAL)...")

    key = jax.random.PRNGKey(789)
    n_real_trials = 55

    # Generate real trial data
    stimuli = jax.random.randint(key, (n_real_trials,), 0, 6)
    key, subkey = jax.random.split(key)
    actions = jax.random.randint(subkey, (n_real_trials,), 0, 3)
    key, subkey = jax.random.split(key)
    rewards = jax.random.bernoulli(subkey, 0.7, (n_real_trials,)).astype(jnp.float32)
    set_sizes = jnp.full((n_real_trials,), 5, dtype=jnp.int32)

    params = {
        'alpha_pos': 0.3, 'alpha_neg': 0.1, 'phi': 0.1,
        'rho': 0.7, 'capacity': 4.0, 'kappa': 0.3,
        'phi_rl': 0.15, 'epsilon': 0.05
    }

    # Unpadded likelihood (original)
    log_lik_original = wmrl_m5_block_likelihood(
        stimuli, actions, rewards, set_sizes, **params
    )

    # Padded likelihood (with mask)
    stimuli_pad, actions_pad, rewards_pad, set_sizes_pad, mask = pad_block_to_max(
        stimuli, actions, rewards, max_trials=100, set_sizes=set_sizes
    )
    log_lik_padded = wmrl_m5_block_likelihood(
        stimuli_pad, actions_pad, rewards_pad, set_sizes_pad, **params, mask=mask
    )

    # Verify equivalence
    match = jnp.allclose(log_lik_original, log_lik_padded, rtol=1e-6)
    print(f"  Original (unpadded) log-lik: {float(log_lik_original):.8f}")
    print(f"  Padded (with mask) log-lik:  {float(log_lik_padded):.8f}")
    print(f"  Difference: {abs(float(log_lik_original - log_lik_padded)):.2e}")
    print(f"[OK] WM-RL M5 padding equivalence: {'PASSED' if match else 'FAILED'}")

    assert bool(match), "WM-RL M5 padded/unpadded must be IDENTICAL!"
    return match

def wmrl_m5_hierarchical_model(
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
    """Hierarchical Bayesian M5 (WM-RL+phi_rl) model with optional Level-2 regression.

    Uses the hBayesDM non-centered parameterization convention (Ahn, Haines, Zhang 2017):
    ``theta_unc = mu_pr + sigma_pr * z``,
    ``theta = lower + (upper - lower) * Phi_approx(theta_unc)``,
    where ``Phi_approx = jax.scipy.stats.norm.cdf``.

    K (capacity) is parameterized in [2, 6] following Senta, Bishop, Collins (2025).

    Level-2 regression (single-covariate legacy Phase 16 mode): if
    ``covariate_lec`` is provided (standardized LEC-total score), a coefficient
    ``beta_lec_kappa`` is sampled and added as a per-participant shift on the
    unconstrained kappa scale before the Phi_approx transform:
    ``kappa_unc_i = kappa_mu_pr + kappa_sigma_pr * z_i + beta_lec_kappa * lec_i``.

    Level-2 regression (2-covariate Phase 21 Option C mode): if BOTH
    ``covariate_lec`` and ``covariate_iesr`` are provided, an additional
    coefficient ``beta_iesr_kappa`` is sampled with the same Normal(0, 1)
    prior and both shifts are summed on the probit scale:
    ``kappa_unc_i = kappa_mu_pr + kappa_sigma_pr * z_i
                   + beta_lec_kappa * lec_i + beta_iesr_kappa * iesr_i``.

    M5 adds ``phi_rl`` (RL forgetting rate) relative to M3.  The 8 model parameters
    are: alpha_pos, alpha_neg, phi, rho, capacity, epsilon, phi_rl (7, sampled via
    ``sample_bounded_param``) and kappa (1, sampled manually with optional L2 shift).

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
        no Level-2 regression is applied and ``beta_lec_kappa`` is
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
    - Seven parameters (alpha_pos, alpha_neg, phi, rho, capacity, epsilon, phi_rl)
      are sampled via ``sample_bounded_param`` from ``numpyro_helpers``.
    - Kappa is sampled manually with the optional L2 shift applied on the probit
      scale before the Phi_approx transform (OUTSIDE ``sample_bounded_param``).
    - phi_rl uses ``mu_prior_loc=-0.8`` (same as phi) from ``PARAM_PRIOR_DEFAULTS``.
    - Likelihood is accumulated via ``numpyro.factor`` in a Python for-loop over
      participants.  This implements HIER-04.
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
            "wmrl_m5_hierarchical_model: use_pscan + fully-batched vmap "
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
    # Level-2: LEC-total + IES-R-total -> kappa regression coefficients
    # (2-covariate design, Phase 21 Option C)
    # ------------------------------------------------------------------
    if covariate_lec is not None:
        beta_lec_kappa = numpyro.sample("beta_lec_kappa", dist.Normal(0.0, 1.0))
    else:
        beta_lec_kappa = 0.0

    if covariate_iesr is not None:
        beta_iesr_kappa = numpyro.sample(
            "beta_iesr_kappa", dist.Normal(0.0, 1.0)
        )
    else:
        beta_iesr_kappa = 0.0

    # ------------------------------------------------------------------
    # Group priors for 7 parameters (all except kappa)
    # Uses hBayesDM non-centered convention locked in Phase 13.
    # phi_rl: mu_prior_loc=-0.8 from PARAM_PRIOR_DEFAULTS.
    # ------------------------------------------------------------------
    sampled: dict[str, jnp.ndarray] = {}
    for param in [
        "alpha_pos",
        "alpha_neg",
        "phi",
        "rho",
        "capacity",
        "epsilon",
        "phi_rl",
    ]:
        defaults = PARAM_PRIOR_DEFAULTS[param]
        sampled[param] = sample_bounded_param(
            param,
            lower=defaults["lower"],
            upper=defaults["upper"],
            n_participants=n_participants,
            mu_prior_loc=defaults["mu_prior_loc"],
        )

    # ------------------------------------------------------------------
    # Kappa with optional L2 shift on the probit scale
    # Sampled manually to allow per-participant LEC + IES-R offset.
    # ------------------------------------------------------------------
    kappa_defaults = PARAM_PRIOR_DEFAULTS["kappa"]
    kappa_mu_pr = numpyro.sample(
        "kappa_mu_pr",
        dist.Normal(kappa_defaults["mu_prior_loc"], 1.0),
    )
    kappa_sigma_pr = numpyro.sample("kappa_sigma_pr", dist.HalfNormal(0.2))
    kappa_z = numpyro.sample(
        "kappa_z",
        dist.Normal(0, 1).expand([n_participants]),
    )
    lec_shift = beta_lec_kappa * covariate_lec if covariate_lec is not None else 0.0
    iesr_shift = (
        beta_iesr_kappa * covariate_iesr if covariate_iesr is not None else 0.0
    )
    kappa_unc = kappa_mu_pr + kappa_sigma_pr * kappa_z + lec_shift + iesr_shift
    kappa = numpyro.deterministic(
        "kappa",
        kappa_defaults["lower"]
        + (kappa_defaults["upper"] - kappa_defaults["lower"])
        * phi_approx(kappa_unc),
    )
    sampled["kappa"] = kappa

    # ------------------------------------------------------------------
    # Likelihood via single numpyro.factor("obs", ...) — fully-batched
    # vmap over participants and blocks.
    # ------------------------------------------------------------------
    if stacked_arrays is None:
        stacked_arrays = stack_across_participants(participant_data_stacked)

    from rlwm.fitting.jax_likelihoods import wmrl_m5_fully_batched_likelihood

    per_participant_ll = wmrl_m5_fully_batched_likelihood(
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
        kappa=sampled["kappa"],
        phi_rl=sampled["phi_rl"],
        epsilon=sampled["epsilon"],
        num_stimuli=num_stimuli,
        num_actions=num_actions,
        q_init=q_init,
        wm_init=wm_init,
    )
    numpyro.factor("obs", per_participant_ll.sum())
