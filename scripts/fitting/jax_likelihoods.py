"""
JAX-based Likelihood Functions for RL Models

This module implements Q-learning and WM-RL likelihoods using pure JAX operations.
All functions are JIT-compilable and support automatic differentiation.

Key differences from PyTensor version:
- Uses jax.lax.scan() for sequential operations (like pytensor.scan)
- Native fast compilation via XLA (no C compiler needed)
- Cleaner functional API
- Works seamlessly with NumPyro for Bayesian inference

Mathematical Background (following Senta et al., 2025):
------------------------------------------------------
Q-learning update: Q(s,a) ← Q(s,a) + α * (r - Q(s,a))
Softmax policy: P(a|s) = exp(β*Q(s,a)) / Σ exp(β*Q(s,a'))
Epsilon noise: P_noisy(a|s) = ε/nA + (1-ε)*P(a|s)
Asymmetric learning: α = α+ if δ > 0 else α-
Fixed β = 50 during learning (for parameter identifiability)

Block-aware processing:
- Q-values reset at each block boundary
- Likelihoods summed across independent blocks

Author: Generated for RLWM trauma analysis project
Date: 2025-11-22
Updated: 2026-01-20 - Added epsilon noise, fixed beta=50
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

# =============================================================================
# TIMING AND DEBUGGING UTILITIES
# =============================================================================

def log_gpu_memory(label: str) -> dict | None:
    """
    Log GPU memory usage for debugging.

    This is useful for identifying memory leaks or excessive allocation
    during JAX optimization. Only works with JAX 0.4.1+ and CUDA.

    Args:
        label: Description of checkpoint (e.g., "PRE-OPTIMIZATION")

    Returns:
        Dictionary with memory stats, or None if unavailable
    """
    try:
        # Force pending computations to complete
        devices = jax.devices()
        if devices and hasattr(devices[0], 'synchronize'):
            devices[0].synchronize()

        # Get memory stats (JAX 0.4.1+)
        if hasattr(devices[0], 'memory_stats'):
            stats = devices[0].memory_stats()
            print(f"[GPU-MEM] {label}: "
                  f"used={stats.get('bytes_in_use', 0)/1e9:.2f}GB, "
                  f"peak={stats.get('peak_bytes_in_use', 0)/1e9:.2f}GB")
            return stats
    except Exception:
        # Silently skip if not available
        pass
    return None

# =============================================================================
# CONSTANTS (following Senta et al., 2025)
# =============================================================================
FIXED_BETA = 50.0  # Fixed inverse temperature during learning for identifiability
DEFAULT_EPSILON = 0.05  # Default epsilon noise
NUM_ACTIONS = 3  # Number of possible actions
MAX_TRIALS_PER_BLOCK = 100  # Fixed block size for JAX compilation efficiency
MAX_BLOCKS = 17  # Fixed number of blocks for JAX compilation efficiency (actual max in data)

# =============================================================================
# BLOCK PADDING UTILITIES (for JAX compilation efficiency)
# =============================================================================

def pad_block_to_max(
    stimuli: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    max_trials: int = MAX_TRIALS_PER_BLOCK,
    set_sizes: jnp.ndarray = None
) -> tuple:
    """
    Pad block arrays to fixed length with mask for JAX compilation efficiency.

    JAX's XLA compiler generates different machine code for different array shapes.
    By padding all blocks to the same size, we ensure JAX compiles only ONE kernel
    that gets reused across all blocks and participants. This reduces compilation
    from ~5 per participant (one per unique block size) to 1 total.

    The mask ensures mathematical equivalence:
    - mask[t] = 1.0 for real trials → included in likelihood
    - mask[t] = 0.0 for padding → zeroed out, no effect on result

    Parameters
    ----------
    stimuli : array, shape (n_trials,)
        Stimulus indices for real trials
    actions : array, shape (n_trials,)
        Action indices for real trials
    rewards : array, shape (n_trials,)
        Rewards for real trials
    max_trials : int
        Target padded length (default: 100)
    set_sizes : array, shape (n_trials,), optional
        Set sizes for WM-RL models (will be padded if provided)

    Returns
    -------
    tuple
        If set_sizes is None:
            (stimuli_padded, actions_padded, rewards_padded, mask)
        If set_sizes provided:
            (stimuli_padded, actions_padded, rewards_padded, set_sizes_padded, mask)
        All arrays have shape (max_trials,)
        mask[t] = 1.0 for real trials, 0.0 for padding
    """
    n_real = len(stimuli)
    n_pad = max_trials - n_real

    if n_pad < 0:
        raise ValueError(
            f"Block has {n_real} trials, exceeding max_trials={max_trials}. "
            f"Increase MAX_TRIALS_PER_BLOCK constant."
        )

    # Create mask: 1 for real trials, 0 for padding
    mask = jnp.concatenate([jnp.ones(n_real), jnp.zeros(n_pad)])

    # Pad arrays with zeros (values don't matter since mask will zero them out)
    stimuli_padded = jnp.concatenate([stimuli, jnp.zeros(n_pad, dtype=stimuli.dtype)])
    actions_padded = jnp.concatenate([actions, jnp.zeros(n_pad, dtype=actions.dtype)])
    rewards_padded = jnp.concatenate([rewards, jnp.zeros(n_pad, dtype=rewards.dtype)])

    if set_sizes is not None:
        set_sizes_padded = jnp.concatenate([set_sizes, jnp.ones(n_pad, dtype=set_sizes.dtype)])
        return stimuli_padded, actions_padded, rewards_padded, set_sizes_padded, mask

    return stimuli_padded, actions_padded, rewards_padded, mask

def pad_blocks_to_max(
    stimuli_blocks: list,
    actions_blocks: list,
    rewards_blocks: list,
    masks_blocks: list,
    max_blocks: int = MAX_BLOCKS,
    set_sizes_blocks: list = None
) -> tuple:
    """
    Pad block lists to fixed length for consistent JAX compilation.

    JAX's XLA compiler generates different machine code for different array shapes.
    When participants have different numbers of blocks (e.g., 3 vs 21), JAX recompiles
    for each unique shape. This can cause:
    - Excessive compilation time (~30s per unique shape)
    - Memory fragmentation when compiling in parallel
    - LLVM compilation errors under memory pressure

    By padding all participants to the same number of blocks, we ensure JAX compiles
    ONE kernel that gets reused across all participants. Empty blocks use mask=0
    and contribute nothing to the likelihood calculation.

    Parameters
    ----------
    stimuli_blocks : list of arrays
        List of stimulus arrays per block (each shape: max_trials,)
    actions_blocks : list of arrays
        List of action arrays per block
    rewards_blocks : list of arrays
        List of reward arrays per block
    masks_blocks : list of arrays
        List of mask arrays per block (1.0 for real trials, 0.0 for padding)
    max_blocks : int
        Target number of blocks (default: MAX_BLOCKS=25)
    set_sizes_blocks : list of arrays, optional
        List of set size arrays per block (for WM-RL models)

    Returns
    -------
    tuple
        (stimuli_padded, actions_padded, rewards_padded, masks_padded, set_sizes_padded)
        All lists have length max_blocks. Empty blocks have all-zero arrays with mask=0.

    Notes
    -----
    - Empty (padding) blocks have mask=0 for all trials, so they contribute 0 to likelihood
    - This is mathematically equivalent to not having those blocks at all
    - The mask ensures Q-updates and WM-updates are also skipped for padding blocks
    """
    n_current = len(stimuli_blocks)
    n_pad = max_blocks - n_current

    if n_pad <= 0:
        # Already at or over max, return as-is
        return (stimuli_blocks, actions_blocks, rewards_blocks,
                masks_blocks, set_sizes_blocks)

    # Get the shape of existing blocks (should be MAX_TRIALS_PER_BLOCK=100)
    trials_per_block = stimuli_blocks[0].shape[0]

    # Create empty padded block (all zeros, all masked out)
    empty_stimuli = jnp.zeros(trials_per_block, dtype=jnp.int32)
    empty_actions = jnp.zeros(trials_per_block, dtype=jnp.int32)
    empty_rewards = jnp.zeros(trials_per_block, dtype=jnp.float32)
    empty_mask = jnp.zeros(trials_per_block, dtype=jnp.float32)  # All masked out

    # Extend lists with empty blocks
    stimuli_padded = list(stimuli_blocks) + [empty_stimuli] * n_pad
    actions_padded = list(actions_blocks) + [empty_actions] * n_pad
    rewards_padded = list(rewards_blocks) + [empty_rewards] * n_pad
    masks_padded = list(masks_blocks) + [empty_mask] * n_pad

    if set_sizes_blocks is not None:
        # Use set_size=1 for padding to avoid division by zero in omega calculation
        empty_set_sizes = jnp.ones(trials_per_block, dtype=jnp.int32)
        set_sizes_padded = list(set_sizes_blocks) + [empty_set_sizes] * n_pad
    else:
        set_sizes_padded = None

    return (stimuli_padded, actions_padded, rewards_padded,
            masks_padded, set_sizes_padded)

def softmax_policy(q_values: jnp.ndarray, beta: float) -> jnp.ndarray:
    """
    Compute softmax action probabilities with numerical stability.

    Parameters
    ----------
    q_values : array, shape (num_actions,)
        Q-values for current state
    beta : float
        Inverse temperature

    Returns
    -------
    array, shape (num_actions,)
        Action probabilities
    """
    # Numerical stability: subtract max before exp
    q_scaled = beta * (q_values - jnp.max(q_values))
    exp_q = jnp.exp(q_scaled)
    return exp_q / jnp.sum(exp_q)

def apply_epsilon_noise(probs: jnp.ndarray, epsilon: float, num_actions: int = NUM_ACTIONS) -> jnp.ndarray:
    """
    Apply epsilon-greedy noise to action probabilities.

    Following Senta et al. (2025): p_noisy(a|s) = ε/nA + (1-ε)*p(a|s)

    This adds a small probability of random action selection to account for:
    - Motor noise
    - Attentional lapses
    - Other sources of response variability

    Parameters
    ----------
    probs : array, shape (num_actions,)
        Base action probabilities (from softmax)
    epsilon : float
        Noise parameter (0-1). Probability of random action.
    num_actions : int
        Number of possible actions (default: 3)

    Returns
    -------
    array, shape (num_actions,)
        Noisy action probabilities
    """
    uniform_prob = 1.0 / num_actions
    noisy_probs = epsilon * uniform_prob + (1 - epsilon) * probs
    return noisy_probs

def q_learning_step(
    carry: tuple[jnp.ndarray, float],
    inputs: tuple[int, int, float]
) -> tuple[tuple[jnp.ndarray, float], float]:
    """
    Single Q-learning trial (functional, JIT-compilable).

    This implements one trial of Q-learning using pure functional operations.
    The Q-table is passed through the carry and updated immutably.

    Parameters
    ----------
    carry : tuple of (Q_table, log_likelihood)
        Q_table : array, shape (num_stimuli, num_actions)
            Current Q-value table
        log_likelihood : float
            Accumulated log-likelihood so far
    inputs : tuple of (stimulus, action, reward)
        stimulus : int
            Current stimulus index
        action : int
            Observed action index
        reward : float
            Observed reward (0 or 1)

    Returns
    -------
    carry : tuple
        Updated (Q_table, log_likelihood)
    output : float
        Log probability of observed action (for debugging/monitoring)

    Notes
    -----
    - Alpha, beta parameters accessed from closure (will be passed via partial)
    - Uses .at[].set() for immutable updates (JAX functional array updates)
    """
    Q_table, log_lik_accum = carry
    stimulus, action, reward = inputs

    # Get Q-values for current stimulus
    q_vals = Q_table[stimulus]

    # Compute action probabilities (softmax policy)
    # Note: alpha_pos, alpha_neg, beta will be passed via functools.partial
    # For now, using placeholder values - will be fixed when integrated with model
    probs = softmax_policy(q_vals, beta=2.0)  # Will be parameterized

    # Log probability of observed action
    log_prob = jnp.log(probs[action] + 1e-8)

    # Compute prediction error
    q_current = Q_table[stimulus, action]
    delta = reward - q_current

    # Asymmetric learning rate (will be parameterized)
    alpha = jnp.where(delta > 0, 0.3, 0.1)  # Placeholder: alpha_pos, alpha_neg

    # Q-value update (functional - creates new value)
    q_updated = q_current + alpha * delta

    # Create new Q-table with updated value (immutable update)
    Q_table_new = Q_table.at[stimulus, action].set(q_updated)

    # Accumulate log-likelihood
    log_lik_new = log_lik_accum + log_prob

    return (Q_table_new, log_lik_new), log_prob

def q_learning_block_likelihood(
    stimuli: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    alpha_pos: float,
    alpha_neg: float,
    epsilon: float = DEFAULT_EPSILON,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    mask: jnp.ndarray = None,
    *,
    return_pointwise: bool = False,
) -> float | tuple[float, jnp.ndarray]:
    """
    Compute log-likelihood for Q-learning on a SINGLE BLOCK.

    This processes one block of trials (typically 30-90 trials) with
    Q-values initialized at the start of the block.

    Following Senta et al. (2025):
    - Beta is fixed at 50 for parameter identifiability
    - Epsilon noise is applied to capture random responding

    Supports masked padding for JAX compilation efficiency:
    - When mask is provided, only trials with mask[t]=1 contribute to likelihood
    - Padding trials (mask[t]=0) are ignored in both likelihood and Q-updates
    - This allows fixed-size compilation while preserving mathematical equivalence

    Parameters
    ----------
    stimuli : array, shape (n_trials,)
        Stimulus sequence for this block
    actions : array, shape (n_trials,)
        Action sequence for this block
    rewards : array, shape (n_trials,)
        Reward sequence for this block
    alpha_pos : float
        Learning rate for positive prediction errors
    alpha_neg : float
        Learning rate for negative prediction errors
    epsilon : float
        Epsilon noise parameter (probability of random action)
    num_stimuli : int
        Number of possible stimuli
    num_actions : int
        Number of possible actions
    q_init : float
        Initial Q-value for all state-action pairs
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

    Examples
    --------
    >>> stimuli = jnp.array([0, 1, 0, 2, 1])
    >>> actions = jnp.array([0, 1, 0, 2, 1])
    >>> rewards = jnp.array([1.0, 0.0, 1.0, 1.0, 0.0])
    >>> log_lik = q_learning_block_likelihood(
    ...     stimuli, actions, rewards,
    ...     alpha_pos=0.3, alpha_neg=0.1, epsilon=0.05
    ... )

    # With padding (same result for real trials):
    >>> mask = jnp.array([1., 1., 1., 1., 1., 0., 0., 0.])  # 5 real + 3 padding
    >>> stimuli_pad = jnp.concatenate([stimuli, jnp.zeros(3, dtype=jnp.int32)])
    >>> # ... pad other arrays similarly
    >>> log_lik_padded = q_learning_block_likelihood(
    ...     stimuli_pad, actions_pad, rewards_pad,
    ...     alpha_pos=0.3, alpha_neg=0.1, epsilon=0.05, mask=mask
    ... )
    >>> assert jnp.allclose(log_lik, log_lik_padded)  # Mathematical equivalence
    """
    # Initialize Q-table
    Q_init = jnp.ones((num_stimuli, num_actions)) * q_init

    # Initial carry: (Q_table, accumulated_log_likelihood)
    init_carry = (Q_init, 0.0)

    # If no mask provided, all trials are valid (backward compatibility)
    if mask is None:
        mask = jnp.ones(len(stimuli))

    # Prepare inputs for scan (include mask)
    scan_inputs = (stimuli, actions, rewards, mask)

    # Create step function with parameters bound
    def step_fn(carry, inputs):
        Q_table, log_lik_accum = carry
        stimulus, action, reward, valid = inputs

        # Get Q-values and compute probabilities with fixed beta
        q_vals = Q_table[stimulus]
        base_probs = softmax_policy(q_vals, FIXED_BETA)

        # Apply epsilon noise: p_noisy = ε/nA + (1-ε)*p
        noisy_probs = apply_epsilon_noise(base_probs, epsilon, num_actions)

        log_prob = jnp.log(noisy_probs[action] + 1e-8)

        # Mask log probability: padding trials contribute 0 to likelihood
        log_prob_masked = log_prob * valid

        # Compute prediction error and update
        q_current = Q_table[stimulus, action]
        delta = reward - q_current
        alpha = jnp.where(delta > 0, alpha_pos, alpha_neg)
        q_updated = q_current + alpha * delta

        # Conditional Q-update: only update for valid trials
        # For padding trials, keep Q-table unchanged
        Q_table_new = Q_table.at[stimulus, action].set(
            jnp.where(valid, q_updated, q_current)
        )
        log_lik_new = log_lik_accum + log_prob_masked

        return (Q_table_new, log_lik_new), log_prob_masked

    # Run scan over trials
    (Q_final, log_lik_total), log_probs = lax.scan(step_fn, init_carry, scan_inputs)

    if return_pointwise:
        return log_lik_total, log_probs
    return log_lik_total

def q_learning_multiblock_likelihood(
    stimuli_blocks: list,
    actions_blocks: list,
    rewards_blocks: list,
    alpha_pos: float,
    alpha_neg: float,
    epsilon: float = DEFAULT_EPSILON,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    masks_blocks: list = None,
    verbose: bool = False,
    participant_id: str = None
) -> float:
    """
    Compute log-likelihood across MULTIPLE BLOCKS.

    This is the main likelihood function for a single participant.
    Q-values reset at each block boundary (matches experimental design).

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
        List of stimulus sequences, one per block
    actions_blocks : list of arrays
        List of action sequences, one per block
    rewards_blocks : list of arrays
        List of reward sequences, one per block
    alpha_pos : float
        Learning rate for positive prediction errors
    alpha_neg : float
        Learning rate for negative prediction errors
    epsilon : float
        Epsilon noise parameter (probability of random action)
    num_stimuli, num_actions : int
        State/action space dimensions
    q_init : float
        Initial Q-value for all state-action pairs
    masks_blocks : list of arrays, optional
        List of mask arrays, one per block. Each mask has 1.0 for real
        trials and 0.0 for padding. If None, no masking is applied.
    verbose : bool
        Print block-by-block progress
    participant_id : str
        Participant ID for verbose output

    Returns
    -------
    float
        Total log-likelihood summed across all blocks

    Examples
    --------
    >>> # Two blocks of data
    >>> stimuli_blocks = [jnp.array([0,1,0]), jnp.array([2,1,2])]
    >>> actions_blocks = [jnp.array([0,1,0]), jnp.array([1,1,2])]
    >>> rewards_blocks = [jnp.array([1,0,1]), jnp.array([0,1,1])]
    >>> log_lik = q_learning_multiblock_likelihood(
    ...     stimuli_blocks, actions_blocks, rewards_blocks,
    ...     alpha_pos=0.3, alpha_neg=0.1, epsilon=0.05
    ... )
    """
    num_blocks = len(stimuli_blocks)

    if verbose:
        print(f"\n  >> Processing {num_blocks} blocks for participant {participant_id}...")

    # Check if blocks are uniformly sized (for JAX-native fori_loop)
    # This is the case when data comes from prepare_participant_data with padding
    block_sizes = [len(b) for b in stimuli_blocks]
    blocks_uniform = len(set(block_sizes)) == 1

    if blocks_uniform and masks_blocks is not None:
        # FAST PATH: Use JAX-native fori_loop for padded data
        # This compiles into a single XLA computation for massive GPU speedup

        # Stack all blocks into arrays for JAX-native loop
        stimuli_stacked = jnp.stack(stimuli_blocks)    # Shape: (n_blocks, max_trials)
        actions_stacked = jnp.stack(actions_blocks)    # Shape: (n_blocks, max_trials)
        rewards_stacked = jnp.stack(rewards_blocks)    # Shape: (n_blocks, max_trials)
        masks_stacked = jnp.stack(masks_blocks)        # Shape: (n_blocks, max_trials)

        # Define the loop body for fori_loop
        def body_fn(block_idx, total_ll):
            block_ll = q_learning_block_likelihood(
                stimuli=stimuli_stacked[block_idx],
                actions=actions_stacked[block_idx],
                rewards=rewards_stacked[block_idx],
                alpha_pos=alpha_pos,
                alpha_neg=alpha_neg,
                epsilon=epsilon,
                num_stimuli=num_stimuli,
                num_actions=num_actions,
                q_init=q_init,
                mask=masks_stacked[block_idx]
            )
            return total_ll + block_ll

        # Use JAX-native fori_loop
        total_log_lik = lax.fori_loop(0, num_blocks, body_fn, 0.0)

    else:
        # FALLBACK PATH: Python loop for variable-sized blocks (backward compatibility)
        # This is slower on GPU but handles arbitrary block sizes
        total_log_lik = 0.0

        # Handle case where masks_blocks is not provided
        if masks_blocks is None:
            masks_blocks = [None] * num_blocks

        for block_idx, (stim_block, act_block, rew_block, mask_block) in enumerate(
            zip(stimuli_blocks, actions_blocks, rewards_blocks, masks_blocks)
        ):
            block_log_lik = q_learning_block_likelihood(
                stimuli=stim_block,
                actions=act_block,
                rewards=rew_block,
                alpha_pos=alpha_pos,
                alpha_neg=alpha_neg,
                epsilon=epsilon,
                num_stimuli=num_stimuli,
                num_actions=num_actions,
                q_init=q_init,
                mask=mask_block
            )
            total_log_lik += block_log_lik

    if verbose:
        print(f"  >> Total log-likelihood: {float(total_log_lik):.2f} ({num_blocks} blocks)\n", flush=True)

    return total_log_lik

# JIT-compile for performance
q_learning_block_likelihood_jit = jax.jit(
    q_learning_block_likelihood,
    static_argnums=(6, 7, 8),  # num_stimuli, num_actions, q_init are static (epsilon is at index 5)
    static_argnames=("return_pointwise",),
)

def q_learning_multiblock_likelihood_stacked(
    stimuli_stacked: jnp.ndarray,
    actions_stacked: jnp.ndarray,
    rewards_stacked: jnp.ndarray,
    masks_stacked: jnp.ndarray,
    alpha_pos: float,
    alpha_neg: float,
    epsilon: float = DEFAULT_EPSILON,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
) -> float:
    """
    FAST multiblock likelihood that takes pre-stacked arrays directly.

    This version:
    - Takes stacked arrays (n_blocks, max_trials) instead of lists
    - Skips uniformity checks (assumes padded data)
    - Avoids list/restack overhead inside JIT

    Use this for GPU optimization when data is already padded to fixed shapes.

    Parameters
    ----------
    stimuli_stacked : array, shape (n_blocks, max_trials)
    actions_stacked : array, shape (n_blocks, max_trials)
    rewards_stacked : array, shape (n_blocks, max_trials)
    masks_stacked : array, shape (n_blocks, max_trials)
        Mask with 1.0 for real trials, 0.0 for padding
    alpha_pos, alpha_neg, epsilon : float
        Model parameters
    num_stimuli, num_actions, q_init : int/float
        Static parameters

    Returns
    -------
    float
        Total log-likelihood summed across all blocks
    """
    num_blocks = stimuli_stacked.shape[0]

    def body_fn(block_idx, total_ll):
        block_ll = q_learning_block_likelihood(
            stimuli=stimuli_stacked[block_idx],
            actions=actions_stacked[block_idx],
            rewards=rewards_stacked[block_idx],
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            epsilon=epsilon,
            num_stimuli=num_stimuli,
            num_actions=num_actions,
            q_init=q_init,
            mask=masks_stacked[block_idx]
        )
        return total_ll + block_ll

    return lax.fori_loop(0, num_blocks, body_fn, 0.0)

def prepare_block_data(
    data_df,
    participant_col: str = 'sona_id',
    block_col: str = 'block',
    stimulus_col: str = 'stimulus',
    action_col: str = 'key_press',
    reward_col: str = 'reward'
) -> dict[Any, dict[int, dict[str, jnp.ndarray]]]:
    """
    Prepare data in block-structured format for JAX likelihoods.

    Parameters
    ----------
    data_df : pd.DataFrame
        Trial-level data
    participant_col : str
        Column name for participant IDs
    block_col : str
        Column name for block numbers
    stimulus_col, action_col, reward_col : str
        Column names for trial data

    Returns
    -------
    dict
        Nested dict: {participant_id: {block_num: {'stimuli': array, 'actions': array, 'rewards': array}}}

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'sona_id': [1, 1, 1, 1, 2, 2],
    ...     'block': [3, 3, 4, 4, 3, 3],
    ...     'stimulus': [0, 1, 0, 2, 1, 0],
    ...     'key_press': [0, 1, 0, 1, 1, 0],
    ...     'reward': [1, 0, 1, 0, 1, 1]
    ... })
    >>> block_data = prepare_block_data(df)
    >>> print(block_data[1][3]['stimuli'])  # Participant 1, Block 3
    """
    block_data = {}

    for participant_id in data_df[participant_col].unique():
        participant_data = data_df[data_df[participant_col] == participant_id]

        block_data[participant_id] = {}

        for block_num in participant_data[block_col].unique():
            block_trials = participant_data[participant_data[block_col] == block_num]

            block_data[participant_id][int(block_num)] = {
                'stimuli': jnp.array(block_trials[stimulus_col].values, dtype=jnp.int32),
                'actions': jnp.array(block_trials[action_col].values, dtype=jnp.int32),
                'rewards': jnp.array(block_trials[reward_col].values, dtype=jnp.float32)
            }

    return block_data

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_single_block():
    """Test Q-learning likelihood on a single block."""
    print("Testing single block Q-learning likelihood...")
    print(f"  (Using fixed beta={FIXED_BETA}, epsilon noise enabled)")

    # Create synthetic block (30 trials)
    key = jax.random.PRNGKey(42)
    n_trials = 30

    stimuli = jax.random.randint(key, (n_trials,), 0, 6)
    actions = jax.random.randint(key, (n_trials,), 0, 3)
    rewards = jax.random.bernoulli(key, 0.7, (n_trials,)).astype(jnp.float32)

    # Test parameters (no beta - it's fixed at 50)
    alpha_pos = 0.3
    alpha_neg = 0.1
    epsilon = 0.05

    # Compute likelihood
    log_lik = q_learning_block_likelihood(
        stimuli, actions, rewards,
        alpha_pos, alpha_neg, epsilon
    )

    print(f"[OK] Single block log-likelihood: {log_lik:.2f}")
    print(f"  Average log-prob per trial: {log_lik / n_trials:.3f}")

    # Test JIT compilation
    log_lik_jit = q_learning_block_likelihood_jit(
        stimuli, actions, rewards,
        alpha_pos, alpha_neg, epsilon
    )

    print(f"[OK] JIT-compiled result matches: {jnp.allclose(log_lik, log_lik_jit)}")

    return log_lik

def test_multiblock():
    """Test Q-learning likelihood on multiple blocks."""
    print("\nTesting multi-block Q-learning likelihood...")
    print(f"  (Using fixed beta={FIXED_BETA}, epsilon noise enabled)")

    key = jax.random.PRNGKey(42)

    # Create 3 blocks of varying sizes
    block_sizes = [30, 60, 45]
    stimuli_blocks = []
    actions_blocks = []
    rewards_blocks = []

    for i, size in enumerate(block_sizes):
        key, subkey = jax.random.split(key)
        stimuli_blocks.append(jax.random.randint(subkey, (size,), 0, 6))

        key, subkey = jax.random.split(key)
        actions_blocks.append(jax.random.randint(subkey, (size,), 0, 3))

        key, subkey = jax.random.split(key)
        rewards_blocks.append(jax.random.bernoulli(subkey, 0.7, (size,)).astype(jnp.float32))

    # Test parameters (no beta - it's fixed at 50)
    alpha_pos = 0.3
    alpha_neg = 0.1
    epsilon = 0.05

    # Compute likelihood
    log_lik = q_learning_multiblock_likelihood(
        stimuli_blocks, actions_blocks, rewards_blocks,
        alpha_pos, alpha_neg, epsilon
    )

    total_trials = sum(block_sizes)
    print(f"[OK] Multi-block log-likelihood: {log_lik:.2f}")
    print(f"  Total trials: {total_trials}")
    print(f"  Average log-prob per trial: {log_lik / total_trials:.3f}")

    # Verify it equals sum of individual blocks
    manual_sum = sum([
        q_learning_block_likelihood(stim, act, rew, alpha_pos, alpha_neg, epsilon)
        for stim, act, rew in zip(stimuli_blocks, actions_blocks, rewards_blocks)
    ])
    print(f"[OK] Matches manual block summation: {jnp.allclose(log_lik, manual_sum)}")

    return log_lik

# ============================================================================
# WM-RL HYBRID MODEL LIKELIHOODS
# ============================================================================

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
) -> float:
    """
    FAST WM-RL multiblock likelihood that takes pre-stacked arrays directly.

    See wmrl_multiblock_likelihood for full documentation.
    This version avoids list/restack overhead inside JIT.
    """
    num_blocks = stimuli_stacked.shape[0]

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
) -> float:
    """
    FAST WM-RL M3 multiblock likelihood that takes pre-stacked arrays directly.

    See wmrl_m3_multiblock_likelihood for full documentation.
    This version avoids list/restack overhead inside JIT.
    """
    num_blocks = stimuli_stacked.shape[0]

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

# JIT-compile WM-RL for performance
wmrl_block_likelihood_jit = jax.jit(
    wmrl_block_likelihood,
    static_argnums=(10, 11, 12, 13),  # num_stimuli, num_actions, q_init, wm_init are static
    static_argnames=("return_pointwise",),
)

# ============================================================================
# WM-RL M5: RL FORGETTING MODEL (M3 + phi_rl Q-value decay)
# ============================================================================

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
) -> float:
    """
    FAST WM-RL M5 multiblock likelihood that takes pre-stacked arrays directly.

    See wmrl_m5_multiblock_likelihood for full documentation.
    This version avoids list/restack overhead inside JIT.
    When phi_rl=0, results match wmrl_m3_multiblock_likelihood_stacked exactly.
    """
    num_blocks = stimuli_stacked.shape[0]

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

# ============================================================================
# WM-RL M6a: STIMULUS-SPECIFIC PERSEVERATION MODEL (M3 with per-stimulus carry)
# ============================================================================

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
) -> float:
    """
    FAST WM-RL M6a multiblock likelihood that takes pre-stacked arrays directly.

    See wmrl_m6a_multiblock_likelihood for full documentation.
    This version avoids list/restack overhead inside JIT.
    kappa_s controls stimulus-specific perseveration (replaces global kappa from M3).
    """
    num_blocks = stimuli_stacked.shape[0]

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

# ============================================================================
# WM-RL M6b: DUAL PERSEVERATION MODEL (global + stimulus-specific, stick-breaking)
# ============================================================================

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
) -> float:
    """
    FAST WM-RL M6b multiblock likelihood that takes pre-stacked arrays directly.

    See wmrl_m6b_multiblock_likelihood for full documentation.
    This version avoids list/restack overhead inside JIT.
    kappa and kappa_s are DECODED values (kappa = kappa_total * kappa_share;
    kappa_s = kappa_total * (1 - kappa_share)).
    """
    num_blocks = stimuli_stacked.shape[0]

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

# ============================================================================
# WM-RL TEST FUNCTIONS
# ============================================================================

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

# ============================================================================
# PADDING EQUIVALENCE TESTS (Critical for verifying mask correctness)
# ============================================================================

def test_padding_equivalence_qlearning():
    """
    Verify padded and unpadded Q-learning likelihoods are mathematically equivalent.

    This is a CRITICAL test: the mask must correctly zero out padding contributions
    so that the likelihood is IDENTICAL regardless of padding.

    Mathematical proof:
    Original: L = Σₜ₌₁ⁿ log P(aₜ|sₜ)
    Padded:   L = Σₜ₌₁¹⁰⁰ mask[t] × log P(aₜ|sₜ)
                = Σₜ₌₁ⁿ 1×log P(...) + Σₜ₌ₙ₊₁¹⁰⁰ 0×log P(...)
                = Σₜ₌₁ⁿ log P(aₜ|sₜ)  [OK] Identical
    """
    print("\nTesting Q-learning padding equivalence (CRITICAL)...")

    key = jax.random.PRNGKey(42)
    n_real_trials = 30

    # Generate real trial data
    stimuli = jax.random.randint(key, (n_real_trials,), 0, 6)
    key, subkey = jax.random.split(key)
    actions = jax.random.randint(subkey, (n_real_trials,), 0, 3)
    key, subkey = jax.random.split(key)
    rewards = jax.random.bernoulli(subkey, 0.7, (n_real_trials,)).astype(jnp.float32)

    params = {'alpha_pos': 0.3, 'alpha_neg': 0.1, 'epsilon': 0.05}

    # Unpadded likelihood (original)
    log_lik_original = q_learning_block_likelihood(
        stimuli, actions, rewards, **params
    )

    # Padded likelihood (with mask)
    stimuli_pad, actions_pad, rewards_pad, mask = pad_block_to_max(
        stimuli, actions, rewards, max_trials=100
    )
    log_lik_padded = q_learning_block_likelihood(
        stimuli_pad, actions_pad, rewards_pad, **params, mask=mask
    )

    # Verify equivalence
    match = jnp.allclose(log_lik_original, log_lik_padded, rtol=1e-6)
    print(f"  Original (unpadded) log-lik: {log_lik_original:.8f}")
    print(f"  Padded (with mask) log-lik:  {log_lik_padded:.8f}")
    print(f"  Difference: {abs(float(log_lik_original - log_lik_padded)):.2e}")
    print(f"[OK] Q-learning padding equivalence: {'PASSED' if match else 'FAILED'}")

    assert match, "Q-learning padded/unpadded must be IDENTICAL!"
    return match

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


if __name__ == "__main__":
    print("=" * 80)
    print("JAX Q-LEARNING LIKELIHOOD TESTS")
    print("=" * 80)

    test_single_block()
    test_multiblock()

    print("\n" + "=" * 80)
    print("JAX WM-RL LIKELIHOOD TESTS")
    print("=" * 80)

    test_wmrl_single_block()
    test_wmrl_multiblock()

    print("\n" + "=" * 80)
    print("JAX WM-RL M3 (PERSEVERATION) LIKELIHOOD TESTS")
    print("=" * 80)

    test_wmrl_m3_single_block()
    test_wmrl_m3_backward_compatibility()

    print("\n" + "=" * 80)
    print("JAX WM-RL M5 (RL FORGETTING) LIKELIHOOD TESTS")
    print("=" * 80)

    test_wmrl_m5_single_block()
    test_wmrl_m5_backward_compatibility()

    print("\n" + "=" * 80)
    print("JAX WM-RL M6a (STIMULUS-SPECIFIC PERSEVERATION) LIKELIHOOD TESTS")
    print("=" * 80)

    test_wmrl_m6a_single_block()
    test_wmrl_m6a_per_stimulus_tracking()

    print("\n" + "=" * 80)
    print("JAX WM-RL M6b (DUAL PERSEVERATION) LIKELIHOOD TESTS")
    print("=" * 80)

    test_wmrl_m6b_single_block()
    test_wmrl_m6b_kappa_share_one_matches_m3()
    test_wmrl_m6b_kappa_share_zero_matches_m6a()

    print("\n" + "=" * 80)
    print("PADDING EQUIVALENCE TESTS (CRITICAL)")
    print("=" * 80)

    test_padding_equivalence_qlearning()
    test_padding_equivalence_wmrl()
    test_padding_equivalence_wmrl_m3()
    test_padding_equivalence_wmrl_m5()
    test_padding_equivalence_wmrl_m6a()
    test_padding_equivalence_wmrl_m6b()
    test_multiblock_padding_equivalence()

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
