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

import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Dict, Any
import numpy as np

# =============================================================================
# CONSTANTS (following Senta et al., 2025)
# =============================================================================
FIXED_BETA = 50.0  # Fixed inverse temperature during learning for identifiability
DEFAULT_EPSILON = 0.05  # Default epsilon noise
NUM_ACTIONS = 3  # Number of possible actions
MAX_TRIALS_PER_BLOCK = 100  # Fixed block size for JAX compilation efficiency


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
    carry: Tuple[jnp.ndarray, float],
    inputs: Tuple[int, int, float]
) -> Tuple[Tuple[jnp.ndarray, float], float]:
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
    mask: jnp.ndarray = None
) -> float:
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

    Returns
    -------
    float
        Total log-likelihood for this block

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
    total_log_lik = 0.0
    num_blocks = len(stimuli_blocks)

    if verbose:
        print(f"\n  >> Processing {num_blocks} blocks for participant {participant_id}...")

    # Handle case where masks_blocks is not provided
    if masks_blocks is None:
        masks_blocks = [None] * num_blocks

    for block_idx, (stim_block, act_block, rew_block, mask_block) in enumerate(
        zip(stimuli_blocks, actions_blocks, rewards_blocks, masks_blocks)
    ):
        if verbose and block_idx == 0:
            print(f"     [Block 1/{num_blocks}] Compiling JAX likelihood (first time only)...", flush=True)
        elif verbose and block_idx % 5 == 0:
            print(f"     [Block {block_idx+1}/{num_blocks}] Processing...", flush=True)

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

        if verbose and block_idx == 0:
            print(f"     [Block 1/{num_blocks}] Compiled! Log-lik: {float(block_log_lik):.2f}", flush=True)

    if verbose:
        print(f"  >> Total log-likelihood: {float(total_log_lik):.2f} ({num_blocks} blocks)\n", flush=True)

    return total_log_lik


# JIT-compile for performance
q_learning_block_likelihood_jit = jax.jit(
    q_learning_block_likelihood,
    static_argnums=(6, 7, 8)  # num_stimuli, num_actions, q_init are static (epsilon is at index 5)
)


def prepare_block_data(
    data_df,
    participant_col: str = 'sona_id',
    block_col: str = 'block',
    stimulus_col: str = 'stimulus',
    action_col: str = 'key_press',
    reward_col: str = 'reward'
) -> Dict[Any, Dict[int, Dict[str, jnp.ndarray]]]:
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
    mask: jnp.ndarray = None
) -> float:
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

    Returns
    -------
    float
        Total log-likelihood for this block
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
    mask: jnp.ndarray = None
) -> float:
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

    Returns
    -------
    float
        Total log-likelihood for this block
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
    total_log_lik = 0.0
    num_blocks = len(stimuli_blocks)

    if verbose:
        print(f"\n  >> Processing {num_blocks} blocks for participant {participant_id}...")

    # Handle case where masks_blocks is not provided
    if masks_blocks is None:
        masks_blocks = [None] * num_blocks

    for block_idx, (stim_block, act_block, rew_block, set_block, mask_block) in enumerate(
        zip(stimuli_blocks, actions_blocks, rewards_blocks, set_sizes_blocks, masks_blocks)
    ):
        if verbose and block_idx == 0:
            print(f"     [Block 1/{num_blocks}] Compiling WM-RL JAX likelihood (first time only)...", flush=True)
        elif verbose and block_idx % 5 == 0:
            print(f"     [Block {block_idx+1}/{num_blocks}] Processing...", flush=True)

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

        if verbose and block_idx == 0:
            print(f"     [Block 1/{num_blocks}] Compiled! Log-lik: {float(block_log_lik):.2f}", flush=True)

    if verbose:
        print(f"  >> Total log-likelihood: {float(total_log_lik):.2f} ({num_blocks} blocks)\n", flush=True)

    return total_log_lik


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
    total_log_lik = 0.0
    num_blocks = len(stimuli_blocks)

    if verbose:
        print(f"\n  >> Processing {num_blocks} blocks for participant {participant_id}...")

    # Handle case where masks_blocks is not provided
    if masks_blocks is None:
        masks_blocks = [None] * num_blocks

    for block_idx, (stim_block, act_block, rew_block, set_block, mask_block) in enumerate(
        zip(stimuli_blocks, actions_blocks, rewards_blocks, set_sizes_blocks, masks_blocks)
    ):
        if verbose and block_idx == 0:
            print(f"     [Block 1/{num_blocks}] Compiling WM-RL M3 JAX likelihood (first time only)...", flush=True)
        elif verbose and block_idx % 5 == 0:
            print(f"     [Block {block_idx+1}/{num_blocks}] Processing...", flush=True)

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

        if verbose and block_idx == 0:
            print(f"     [Block 1/{num_blocks}] Compiled! Log-lik: {float(block_log_lik):.2f}", flush=True)

    if verbose:
        print(f"  >> Total log-likelihood: {float(total_log_lik):.2f} ({num_blocks} blocks)\n", flush=True)

    return total_log_lik


# JIT-compile WM-RL for performance
wmrl_block_likelihood_jit = jax.jit(
    wmrl_block_likelihood,
    static_argnums=(10, 11, 12, 13)  # num_stimuli, num_actions, q_init, wm_init are static
)


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
    print("PADDING EQUIVALENCE TESTS (CRITICAL)")
    print("=" * 80)

    test_padding_equivalence_qlearning()
    test_padding_equivalence_wmrl()
    test_padding_equivalence_wmrl_m3()
    test_multiblock_padding_equivalence()

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
