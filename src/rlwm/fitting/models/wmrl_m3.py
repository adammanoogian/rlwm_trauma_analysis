"""M3 WM-RL + kappa perseveration: JAX likelihoods + NumPyro hierarchical wrapper.

Relocated here in Phase 29-08 from :mod:`rlwm.fitting.jax_likelihoods` and
:mod:`rlwm.fitting.numpyro_models`. Old import paths remain available via
wildcard re-export shims.

Senta et al. (2025) M3 extends M2 with a global perseveration bonus ``kappa``
added to the last chosen action's log-probability (pre-softmax).
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
    softmax_policy,
)

__all__ = [
    "wmrl_m3_block_likelihood",
    "wmrl_m3_multiblock_likelihood",
    "wmrl_m3_multiblock_likelihood_stacked",
    "wmrl_m3_fully_batched_likelihood",
    "wmrl_m3_block_likelihood_pscan",
    "wmrl_m3_multiblock_likelihood_stacked_pscan",
    "test_wmrl_m3_single_block",
    "test_wmrl_m3_backward_compatibility",
    "test_padding_equivalence_wmrl_m3",
    "wmrl_m3_hierarchical_model",
]


def wmrl_m3_block_likelihood(
    stimuli: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    set_sizes: jnp.ndarray,
    alpha_pos: float,
    alpha_neg: float,
    phi: float,
    rho: float,
    capacity: float,
    kappa: float,  # NEW: perseveration parameter
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
    Compute log-likelihood for WM-RL M3 model with perseveration on a SINGLE BLOCK.

    This extends the WM-RL M2 model by adding a perseveration parameter (kappa) that
    captures outcome-insensitive action repetition (motor-level response stickiness).

    Following Senta et al. (2025):
    - Beta is fixed at 50 for both WM and RL (parameter identifiability)
    - Epsilon noise is applied to the base policy before perseveration mixing
    - WM baseline = 1/nA (uniform probability)
    - Perseveration uses PROBABILITY MIXING: P_M3 = (1-κ)*P_noisy + κ*Ck
    - Ck = one-hot(a_{t-1}) is the choice kernel (global, not stimulus-specific)
    - last_action resets to -1 at block start (no previous action)

    When kappa=0, this reduces exactly to wmrl_block_likelihood (M2 model).

    Model combines:
    1. Working Memory (WM): Immediate encoding with decay
    2. Q-Learning (RL): Gradual learning with asymmetric rates
    3. Hybrid decision: Adaptive weighting based on capacity
    4. Perseveration: Motor-level response stickiness via probability mixing

    Update sequence per trial:
    1. Decay WM: WM ← (1-φ)WM + φ·WM_0
    2. Compute hybrid policy:
       - Both paths: P_base = ω·softmax(WM) + (1-ω)·softmax(Q)
       - Apply epsilon: P_noisy = ε/nA + (1-ε)·P_base
       - If κ=0 OR no last_action: return P_noisy (M2 backward compat)
       - If κ>0 AND last_action exists: P_M3 = (1-κ)*P_noisy + κ*Ck
    3. Update WM: WM(s,a) ← r (immediate overwrite)
    4. Update Q: Q(s,a) ← Q(s,a) + α·(r - Q(s,a))

    Supports masked padding for JAX compilation efficiency:
    - When mask is provided, only trials with mask[t]=1 contribute to likelihood
    - Padding trials (mask[t]=0) are ignored in likelihood, Q/WM/perseveration updates

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
        # 2. COMPUTE HYBRID POLICY
        # =================================================================
        # Two branches for backward compatibility:
        # - kappa=0 OR no last_action: Use M2 probability mixing (exact backward compat)
        # - kappa>0 AND last_action exists: Use M3 value mixing with perseveration

        q_vals = Q_table[stimulus]
        wm_vals = WM_decayed[stimulus]

        # Adaptive weight: ω = ρ * min(1, K/N_s)
        omega = rho * jnp.minimum(1.0, capacity / set_size)

        # Branch: M2 (probability mixing) vs M3 (probability mixing + perseveration)
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
        # M3 path: Probability mixing with choice kernel (Senta et al.)
        # P_M3 = (1-κ)*P_noisy + κ*Ck where Ck = one-hot(last_action)
        # =================================================================
        # Choice kernel = one-hot of last action (τ=1 simplification)
        choice_kernel = jnp.eye(num_actions)[jnp.maximum(last_action, 0)]  # Clamp for indexing

        # Probability mixing: (1-κ)*noisy_base + κ*choice_kernel
        hybrid_probs_m3 = (1 - kappa) * noisy_base + kappa * choice_kernel

        # Select correct path: M2 uses noisy_base, M3 uses probability mixing
        noisy_probs = jnp.where(use_m2_path, noisy_base, hybrid_probs_m3)

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

        # Update last_action only for valid trials (masked perseveration)
        # For padding trials, keep the previous last_action
        new_last_action = jnp.where(valid, action, last_action).astype(jnp.int32)

        # Return updated carry with current action as last_action for next trial
        return (Q_updated, WM_updated, WM_baseline, log_lik_new, new_last_action), log_prob_masked

    # Run scan over trials
    (Q_final, WM_final, _, log_lik_total, _), log_probs = lax.scan(step_fn, init_carry, scan_inputs)

    if return_pointwise:
        return log_lik_total, log_probs
    return log_lik_total

def wmrl_m3_multiblock_likelihood(
    stimuli_blocks: list,
    actions_blocks: list,
    rewards_blocks: list,
    set_sizes_blocks: list,
    alpha_pos: float,
    alpha_neg: float,
    phi: float,
    rho: float,
    capacity: float,
    kappa: float,  # NEW: perseveration parameter
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
    Compute log-likelihood for WM-RL M3 (with perseveration) across MULTIPLE BLOCKS.

    Q-values, WM, and last_action reset at each block boundary.

    Following Senta et al. (2025):
    - Beta is fixed at 50 (not a parameter)
    - Epsilon noise captures random responding
    - Kappa captures motor-level perseveration (global action stickiness)

    Each block is independent:
    - Q-values reset to q_init
    - WM resets to wm_init
    - last_action resets to -1 (no previous action)

    When kappa=0, results match wmrl_multiblock_likelihood exactly (M2 model).

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
    kappa : float
        Perseveration parameter (0-1) - captures motor-level action stickiness
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
            block_ll = wmrl_m3_block_likelihood(
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
            block_log_lik = wmrl_m3_block_likelihood(
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

def wmrl_m3_multiblock_likelihood_stacked(
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
    epsilon: float = DEFAULT_EPSILON,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    *,
    return_pointwise: bool = False,
) -> float | tuple[float, jnp.ndarray]:
    """
    FAST WM-RL M3 multiblock likelihood that takes pre-stacked arrays directly.

    See wmrl_m3_multiblock_likelihood for full documentation.
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
            block_ll, block_probs = wmrl_m3_block_likelihood(
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
            block_ll = wmrl_m3_block_likelihood(
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
                epsilon=epsilon,
                num_stimuli=num_stimuli,
                num_actions=num_actions,
                q_init=q_init,
                wm_init=wm_init,
                mask=masks_stacked[block_idx]
            )
            return total_ll + block_ll

        return lax.fori_loop(0, num_blocks, body_fn, 0.0)

def wmrl_m3_fully_batched_likelihood(
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
    epsilon: jnp.ndarray,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    use_pscan: bool = False,
) -> jnp.ndarray:
    """Fully-batched WM-RL M3 log-likelihood via nested vmap.

    Pattern::

        outer vmap over participants (axis 0 of every input)
          -> inner vmap over blocks (axis 0 of per-participant data)
            -> wmrl_m3_block_likelihood on (T,) slices and scalar params
          -> sum over blocks -> scalar per participant
        -> (N,) vector returned

    CRITICAL: this uses the SAME block likelihood as the sequential
    path (wmrl_m3_block_likelihood). Q, WM, and perseveration state
    all reset at block entry, so blocks are independent and vmap is
    correct per Senta 2025 (MODEL_REFERENCE.md §2.2, §3.1).

    Padded blocks (mask entirely 0.0) contribute 0.0 because the
    inner scan gates every likelihood and state update on mask[t].

    Parameters
    ----------
    stimuli : jnp.ndarray
        Shape (N, B, T) int32. Dimension 0 is participant, 1 is block,
        2 is trial. B = max_n_blocks (padded to uniform size).
    actions : jnp.ndarray
        Shape (N, B, T) int32.
    rewards : jnp.ndarray
        Shape (N, B, T) float32.
    set_sizes : jnp.ndarray
        Shape (N, B, T) float32.
    masks : jnp.ndarray
        Shape (N, B, T) float32. Padded blocks have mask entirely 0.0.
    alpha_pos : jnp.ndarray
        Shape (N,) float32 per-participant positive learning rates.
    alpha_neg : jnp.ndarray
        Shape (N,) float32 per-participant negative learning rates.
    phi : jnp.ndarray
        Shape (N,) float32 per-participant WM forgetting rates.
    rho : jnp.ndarray
        Shape (N,) float32 per-participant WM mixing weights.
    capacity : jnp.ndarray
        Shape (N,) float32 per-participant WM capacity values.
    kappa : jnp.ndarray
        Shape (N,) float32 per-participant perseveration weights.
    epsilon : jnp.ndarray
        Shape (N,) float32 per-participant random-response rates.
    num_stimuli : int
        Number of distinct stimuli.  Default 6.
    num_actions : int
        Number of possible actions.  Default 3.
    q_init : float
        Initial Q-value.  Default 0.5.
    wm_init : float
        Initial WM value (uniform baseline 1/nA).  Default 1/3.
    use_pscan : bool
        Must be False. Raises NotImplementedError if True (pscan + vmap
        composition is out of scope for quick-007).

    Returns
    -------
    jnp.ndarray
        Shape (N,) float — total log-likelihood per participant.

    Raises
    ------
    NotImplementedError
        If use_pscan=True.
    """
    if use_pscan:
        raise NotImplementedError(
            "wmrl_m3_fully_batched_likelihood: use_pscan=True is not "
            "supported. pscan + vmap composition is out of scope for "
            "quick-007. Pass use_pscan=False."
        )

    def _block_ll(
        stim, act, rew, ss, mask,
        ap, an, ph, rh, cap, k, e,
    ):
        # Scalar log-lik for a single (participant, block).
        return wmrl_m3_block_likelihood(
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
            epsilon=e,
            num_stimuli=num_stimuli,
            num_actions=num_actions,
            q_init=q_init,
            wm_init=wm_init,
            mask=mask,
            return_pointwise=False,
        )

    # Inner vmap: over blocks. Data args on axis 0, params broadcast (None).
    _over_blocks = jax.vmap(
        _block_ll,
        in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None, None),
        out_axes=0,
    )

    def _participant_ll(
        stim, act, rew, ss, mask,
        ap, an, ph, rh, cap, k, e,
    ):
        block_lls = _over_blocks(
            stim, act, rew, ss, mask, ap, an, ph, rh, cap, k, e,
        )
        return block_lls.sum()

    # Outer vmap: over participants. Everything on axis 0.
    _over_participants = jax.vmap(
        _participant_ll,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        out_axes=0,
    )
    return _over_participants(
        stimuli, actions, rewards, set_sizes, masks,
        alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon,
    )

def wmrl_m3_block_likelihood_pscan(
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
    WM-RL M3 block likelihood using parallel scan (global perseveration).

    Drop-in replacement for ``wmrl_m3_block_likelihood``.

    Phase 1 (parallel): Pre-compute Q_for_policy[t] and wm_for_policy[t].
    Phase 2 (vectorized): Precompute global last_action array via
    ``precompute_last_action_global``, then compute hybrid WM-Q policy
    with global perseveration kernel for all trials simultaneously.

    Parameters
    ----------
    stimuli, actions, rewards, set_sizes : arrays, shape (n_trials,)
    alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon : float
    num_stimuli, num_actions : int
    q_init, wm_init : float
    mask : array, optional
    return_pointwise : bool, optional

    Returns
    -------
    float or tuple[float, jnp.ndarray]
        Same as ``wmrl_m3_block_likelihood``.
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
    # Phase 2 (vectorized): hybrid policy + global perseveration
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

def wmrl_m3_multiblock_likelihood_stacked_pscan(
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
    epsilon: float = DEFAULT_EPSILON,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    *,
    return_pointwise: bool = False,
) -> float | tuple[float, jnp.ndarray]:
    """
    WM-RL M3 multiblock likelihood using parallel scan.

    Drop-in replacement for ``wmrl_m3_multiblock_likelihood_stacked``.

    Parameters
    ----------
    stimuli_stacked, actions_stacked, rewards_stacked,
    set_sizes_stacked, masks_stacked : arrays, shape (n_blocks, max_trials)
    alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon : float
    num_stimuli, num_actions : int
    q_init, wm_init : float
    return_pointwise : bool, optional

    Returns
    -------
    float or tuple[float, jnp.ndarray]
        Same as ``wmrl_m3_multiblock_likelihood_stacked``.
    """
    num_blocks = stimuli_stacked.shape[0]

    if return_pointwise:
        def scan_body(total_ll, block_idx):
            block_ll, block_probs = wmrl_m3_block_likelihood_pscan(
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
            block_ll = wmrl_m3_block_likelihood_pscan(
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
                epsilon=epsilon,
                num_stimuli=num_stimuli,
                num_actions=num_actions,
                q_init=q_init,
                wm_init=wm_init,
                mask=masks_stacked[block_idx],
            )
            return total_ll + block_ll

        return lax.fori_loop(0, num_blocks, body_fn, 0.0)

def test_wmrl_m3_single_block():
    """Test WM-RL M3 likelihood on a single block."""
    print("\nTesting WM-RL M3 single block likelihood...")
    print(f"  (Using fixed beta={FIXED_BETA}, with perseveration kappa)")

    # Create synthetic block (30 trials)
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
        'epsilon': 0.05,
        'kappa': 0.3  # Moderate perseveration
    }

    log_lik = wmrl_m3_block_likelihood(
        stimuli, actions, rewards, set_sizes, **params
    )

    print(f"  WM-RL M3 single block log-likelihood: {log_lik:.2f}")
    print(f"  Average log-prob per trial: {log_lik / n_trials:.3f}")

    return log_lik

def test_wmrl_m3_backward_compatibility():
    """Verify M3 with kappa=0 matches M2 exactly."""
    print("\nTesting WM-RL M3 backward compatibility (kappa=0 == M2)...")

    key = jax.random.PRNGKey(123)
    n_trials = 50

    stimuli = jax.random.randint(key, (n_trials,), 0, 6)
    key, subkey = jax.random.split(key)
    actions = jax.random.randint(subkey, (n_trials,), 0, 3)
    key, subkey = jax.random.split(key)
    rewards = jax.random.bernoulli(subkey, 0.7, (n_trials,)).astype(jnp.float32)
    set_sizes = jnp.ones((n_trials,)) * 5

    params_m2 = {
        'alpha_pos': 0.3,
        'alpha_neg': 0.1,
        'phi': 0.1,
        'rho': 0.7,
        'capacity': 4.0,
        'epsilon': 0.05
    }

    # M2 likelihood
    log_lik_m2 = wmrl_block_likelihood(
        stimuli, actions, rewards, set_sizes, **params_m2
    )

    # M3 with kappa=0 (should match M2)
    log_lik_m3 = wmrl_m3_block_likelihood(
        stimuli, actions, rewards, set_sizes, **params_m2, kappa=0.0
    )

    match = jnp.allclose(log_lik_m2, log_lik_m3, rtol=1e-5)
    print(f"  M2 log-likelihood: {log_lik_m2:.6f}")
    print(f"  M3 (kappa=0) log-likelihood: {log_lik_m3:.6f}")
    print(f"  Backward compatibility verified: {match}")

    assert match, "M3 with kappa=0 should match M2 exactly!"
    return match

def test_padding_equivalence_wmrl_m3():
    """
    Verify padded and unpadded WM-RL M3 likelihoods are mathematically equivalent.
    """
    print("\nTesting WM-RL M3 padding equivalence (CRITICAL)...")

    key = jax.random.PRNGKey(456)
    n_real_trials = 60

    # Generate real trial data
    stimuli = jax.random.randint(key, (n_real_trials,), 0, 6)
    key, subkey = jax.random.split(key)
    actions = jax.random.randint(subkey, (n_real_trials,), 0, 3)
    key, subkey = jax.random.split(key)
    rewards = jax.random.bernoulli(subkey, 0.7, (n_real_trials,)).astype(jnp.float32)
    set_sizes = jnp.full((n_real_trials,), 5, dtype=jnp.int32)

    params = {
        'alpha_pos': 0.3, 'alpha_neg': 0.1, 'phi': 0.1,
        'rho': 0.7, 'capacity': 4.0, 'kappa': 0.3, 'epsilon': 0.05
    }

    # Unpadded likelihood (original)
    log_lik_original = wmrl_m3_block_likelihood(
        stimuli, actions, rewards, set_sizes, **params
    )

    # Padded likelihood (with mask)
    stimuli_pad, actions_pad, rewards_pad, set_sizes_pad, mask = pad_block_to_max(
        stimuli, actions, rewards, max_trials=100, set_sizes=set_sizes
    )
    log_lik_padded = wmrl_m3_block_likelihood(
        stimuli_pad, actions_pad, rewards_pad, set_sizes_pad, **params, mask=mask
    )

    # Verify equivalence
    match = jnp.allclose(log_lik_original, log_lik_padded, rtol=1e-6)
    print(f"  Original (unpadded) log-lik: {log_lik_original:.8f}")
    print(f"  Padded (with mask) log-lik:  {log_lik_padded:.8f}")
    print(f"  Difference: {abs(float(log_lik_original - log_lik_padded)):.2e}")
    print(f"[OK] WM-RL M3 padding equivalence: {'PASSED' if match else 'FAILED'}")

    assert match, "WM-RL M3 padded/unpadded must be IDENTICAL!"
    return match

def wmrl_m3_hierarchical_model(
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
    """Hierarchical Bayesian M3 (WM-RL+kappa) model with optional Level-2 regression.

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
    - Six parameters (alpha_pos, alpha_neg, phi, rho, capacity, epsilon) are sampled
      via ``sample_bounded_param`` from ``numpyro_helpers``.
    - Kappa is sampled manually with the optional L2 shift applied on the probit scale
      before the Phi_approx transform (OUTSIDE ``sample_bounded_param``).
    - Likelihood is accumulated via a single ``numpyro.factor("obs", ...)`` call
      using ``wmrl_m3_fully_batched_likelihood`` (nested vmap over participants
      and blocks). Blocks are independent per Senta 2025 (Q/WM/perseveration
      reset at block boundaries).
    - Do NOT modify this function's API: ``fit_bayesian.py`` dispatches to it by name.
    """
    from rlwm.fitting.numpyro_helpers import (
        PARAM_PRIOR_DEFAULTS,
        phi_approx,
        sample_bounded_param,
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
    participant_ids = sorted(participant_data_stacked.keys())

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
    # Group priors for 6 parameters (all except kappa)
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
        + (kappa_defaults["upper"] - kappa_defaults["lower"]) * phi_approx(kappa_unc),
    )
    sampled["kappa"] = kappa

    # ------------------------------------------------------------------
    # Likelihood via single numpyro.factor("obs", ...) — fully-batched
    # vmap over participants and blocks.
    # use_pscan=True raises NotImplementedError (pscan + vmap composition
    # is out of scope for quick-007).
    # ------------------------------------------------------------------
    if use_pscan:
        raise NotImplementedError(
            "wmrl_m3_hierarchical_model: use_pscan + fully-batched vmap "
            "path is not implemented in quick-007. Use use_pscan=False, "
            "or revert to the sequential for-loop model (not exposed)."
        )

    if stacked_arrays is None:
        stacked_arrays = stack_across_participants(participant_data_stacked)

    from rlwm.fitting.jax_likelihoods import wmrl_m3_fully_batched_likelihood

    per_participant_ll = wmrl_m3_fully_batched_likelihood(
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
        epsilon=sampled["epsilon"],
        num_stimuli=num_stimuli,
        num_actions=num_actions,
        q_init=q_init,
        wm_init=wm_init,
    )
    numpyro.factor("obs", per_participant_ll.sum())
