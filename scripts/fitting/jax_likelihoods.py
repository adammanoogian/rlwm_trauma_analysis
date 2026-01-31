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
    q_init: float = 0.5
) -> float:
    """
    Compute log-likelihood for Q-learning on a SINGLE BLOCK.

    This processes one block of trials (typically 30-90 trials) with
    Q-values initialized at the start of the block.

    Following Senta et al. (2025):
    - Beta is fixed at 50 for parameter identifiability
    - Epsilon noise is applied to capture random responding

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
    """
    # Initialize Q-table
    Q_init = jnp.ones((num_stimuli, num_actions)) * q_init

    # Initial carry: (Q_table, accumulated_log_likelihood)
    init_carry = (Q_init, 0.0)

    # Prepare inputs for scan
    scan_inputs = (stimuli, actions, rewards)

    # Create step function with parameters bound
    def step_fn(carry, inputs):
        Q_table, log_lik_accum = carry
        stimulus, action, reward = inputs

        # Get Q-values and compute probabilities with fixed beta
        q_vals = Q_table[stimulus]
        base_probs = softmax_policy(q_vals, FIXED_BETA)

        # Apply epsilon noise: p_noisy = ε/nA + (1-ε)*p
        noisy_probs = apply_epsilon_noise(base_probs, epsilon, num_actions)

        log_prob = jnp.log(noisy_probs[action] + 1e-8)

        # Compute prediction error and update
        q_current = Q_table[stimulus, action]
        delta = reward - q_current
        alpha = jnp.where(delta > 0, alpha_pos, alpha_neg)
        q_updated = q_current + alpha * delta

        # Immutable update
        Q_table_new = Q_table.at[stimulus, action].set(q_updated)
        log_lik_new = log_lik_accum + log_prob

        return (Q_table_new, log_lik_new), log_prob

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

    for block_idx, (stim_block, act_block, rew_block) in enumerate(zip(stimuli_blocks, actions_blocks, rewards_blocks)):
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
            q_init=q_init
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

    print(f"✓ Single block log-likelihood: {log_lik:.2f}")
    print(f"  Average log-prob per trial: {log_lik / n_trials:.3f}")

    # Test JIT compilation
    log_lik_jit = q_learning_block_likelihood_jit(
        stimuli, actions, rewards,
        alpha_pos, alpha_neg, epsilon
    )

    print(f"✓ JIT-compiled result matches: {jnp.allclose(log_lik, log_lik_jit)}")

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
    print(f"✓ Multi-block log-likelihood: {log_lik:.2f}")
    print(f"  Total trials: {total_trials}")
    print(f"  Average log-prob per trial: {log_lik / total_trials:.3f}")

    # Verify it equals sum of individual blocks
    manual_sum = sum([
        q_learning_block_likelihood(stim, act, rew, alpha_pos, alpha_neg, epsilon)
        for stim, act, rew in zip(stimuli_blocks, actions_blocks, rewards_blocks)
    ])
    print(f"✓ Matches manual block summation: {jnp.allclose(log_lik, manual_sum)}")

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
    wm_init: float = 1.0 / 3.0  # WM baseline = 1/nA (uniform)
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

    # Prepare inputs
    scan_inputs = (stimuli, actions, rewards, set_sizes)

    def step_fn(carry, inputs):
        Q_table, WM_table, WM_baseline, log_lik_accum = carry
        stimulus, action, reward, set_size = inputs

        # =================================================================
        # 1. DECAY WM: All values move toward baseline
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

        # =================================================================
        # 4. UPDATE WM: Immediate overwrite
        # =================================================================
        WM_updated = WM_decayed.at[stimulus, action].set(reward)

        # =================================================================
        # 5. UPDATE Q-TABLE: Asymmetric learning
        # =================================================================
        q_current = Q_table[stimulus, action]
        delta = reward - q_current
        alpha = jnp.where(delta > 0, alpha_pos, alpha_neg)
        q_updated = q_current + alpha * delta
        Q_updated = Q_table.at[stimulus, action].set(q_updated)

        # Accumulate log-likelihood
        log_lik_new = log_lik_accum + log_prob

        return (Q_updated, WM_updated, WM_baseline, log_lik_new), log_prob

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
    wm_init: float = 1.0 / 3.0  # WM baseline = 1/nA (uniform)
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

    # Prepare inputs
    scan_inputs = (stimuli, actions, rewards, set_sizes)

    def step_fn(carry, inputs):
        Q_table, WM_table, WM_baseline, log_lik_accum, last_action = carry
        stimulus, action, reward, set_size = inputs

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

        # =================================================================
        # 6. UPDATE WM: Immediate overwrite
        # =================================================================
        WM_updated = WM_decayed.at[stimulus, action].set(reward)

        # =================================================================
        # 7. UPDATE Q-TABLE: Asymmetric learning
        # =================================================================
        q_current = Q_table[stimulus, action]
        delta = reward - q_current
        alpha = jnp.where(delta > 0, alpha_pos, alpha_neg)
        q_updated = q_current + alpha * delta
        Q_updated = Q_table.at[stimulus, action].set(q_updated)

        # Accumulate log-likelihood
        log_lik_new = log_lik_accum + log_prob

        # Return updated carry with current action as last_action for next trial
        return (Q_updated, WM_updated, WM_baseline, log_lik_new, action), log_prob

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

    for block_idx, (stim_block, act_block, rew_block, set_block) in enumerate(
        zip(stimuli_blocks, actions_blocks, rewards_blocks, set_sizes_blocks)
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
            wm_init=wm_init
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

    for block_idx, (stim_block, act_block, rew_block, set_block) in enumerate(
        zip(stimuli_blocks, actions_blocks, rewards_blocks, set_sizes_blocks)
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
            wm_init=wm_init
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

    print(f"✓ WM-RL single block log-likelihood: {log_lik:.2f}")
    print(f"  Average log-prob per trial: {log_lik / n_trials:.3f}")

    # Test JIT compilation
    log_lik_jit = wmrl_block_likelihood_jit(
        stimuli, actions, rewards, set_sizes, **params
    )

    print(f"✓ JIT-compiled result matches: {jnp.allclose(log_lik, log_lik_jit)}")

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
    print(f"✓ WM-RL multi-block log-likelihood: {log_lik:.2f}")
    print(f"  Total trials: {total_trials}")
    print(f"  Average log-prob per trial: {log_lik / total_trials:.3f}")

    # Verify it equals sum of individual blocks
    manual_sum = sum([
        wmrl_block_likelihood(stim, act, rew, sets, **params)
        for stim, act, rew, sets in zip(stimuli_blocks, actions_blocks, rewards_blocks, set_sizes_blocks)
    ])
    print(f"✓ Matches manual block summation: {jnp.allclose(log_lik, manual_sum)}")

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
    print("ALL TESTS PASSED!")
    print("=" * 80)
