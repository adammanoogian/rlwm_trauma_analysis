"""M1 Q-learning: JAX likelihoods (sequential + pscan variants) + NumPyro hierarchical wrappers.

Canonical home for M1's JAX likelihoods and NumPyro wrappers. Callers
should import directly from this module; the legacy
``rlwm.fitting.jax_likelihoods`` / ``rlwm.fitting.numpyro_models``
re-export shims were deleted in the v5.0 shim cleanup.

Senta et al. (2025) M1: Q-learning with asymmetric learning rates
``(alpha_pos, alpha_neg)`` and epsilon noise.
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
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
    pad_block_to_max,
    pad_blocks_to_max,
    prepare_stacked_participant_data,
    softmax_policy,
    stack_across_participants,
)

__all__ = [
    "q_learning_step",
    "q_learning_block_likelihood",
    "q_learning_multiblock_likelihood",
    "q_learning_block_likelihood_jit",
    "q_learning_multiblock_likelihood_stacked",
    "prepare_block_data",
    "q_learning_fully_batched_likelihood",
    "q_learning_block_likelihood_pscan",
    "q_learning_multiblock_likelihood_stacked_pscan",
    "test_single_block",
    "test_multiblock",
    "test_padding_equivalence_qlearning",
    "qlearning_hierarchical_model",
    "qlearning_hierarchical_model_stacked",
]


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
    *,
    return_pointwise: bool = False,
) -> float | tuple[float, jnp.ndarray]:
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
    return_pointwise : bool, optional
        If True, return (total_log_lik, per_trial_log_probs) tuple instead of
        scalar. per_trial_log_probs has shape (n_blocks * max_trials,) flattened.
        Padding trials have log_prob = 0.0.
        Default False for backward compatibility with MLE callers.

    Returns
    -------
    float or tuple[float, jnp.ndarray]
        If return_pointwise=False (default): total log-likelihood (scalar).
        If return_pointwise=True: (total_log_lik, per_trial_log_probs) where
        per_trial_log_probs has shape (n_blocks * max_trials,).
    """
    num_blocks = stimuli_stacked.shape[0]

    if return_pointwise:
        def scan_body(total_ll, block_idx):
            block_ll, block_probs = q_learning_block_likelihood(
                stimuli=stimuli_stacked[block_idx],
                actions=actions_stacked[block_idx],
                rewards=rewards_stacked[block_idx],
                alpha_pos=alpha_pos,
                alpha_neg=alpha_neg,
                epsilon=epsilon,
                num_stimuli=num_stimuli,
                num_actions=num_actions,
                q_init=q_init,
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

def q_learning_fully_batched_likelihood(
    stimuli: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    masks: jnp.ndarray,
    alpha_pos: jnp.ndarray,
    alpha_neg: jnp.ndarray,
    epsilon: jnp.ndarray,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    use_pscan: bool = False,
) -> jnp.ndarray:
    """Fully-batched Q-learning log-likelihood via nested vmap.

    Pattern::

        outer vmap over participants (axis 0 of every input)
          -> inner vmap over blocks (axis 0 of per-participant data)
            -> q_learning_block_likelihood on (T,) slices and scalar params
          -> sum over blocks -> scalar per participant
        -> (N,) vector returned

    CRITICAL: this uses the SAME block likelihood as the sequential
    path (``q_learning_block_likelihood``). Q resets at block entry,
    so blocks are independent and vmap is correct per Senta 2025
    (MODEL_REFERENCE.md §2.2, §3.1).

    Padded blocks (mask entirely 0.0) contribute 0.0 because the
    inner scan gates every likelihood and Q update on mask[t].

    Parameters
    ----------
    stimuli : jnp.ndarray
        Shape (N, B, T) int32.  Dimension 0 is participant, 1 is block,
        2 is trial.  B = max_n_blocks (padded to uniform size).
    actions : jnp.ndarray
        Shape (N, B, T) int32.
    rewards : jnp.ndarray
        Shape (N, B, T) float32.
    masks : jnp.ndarray
        Shape (N, B, T) float32.  Padded blocks have mask entirely 0.0.
    alpha_pos : jnp.ndarray
        Shape (N,) float32 per-participant positive learning rates.
    alpha_neg : jnp.ndarray
        Shape (N,) float32 per-participant negative learning rates.
    epsilon : jnp.ndarray
        Shape (N,) float32 per-participant random-response rates.
    num_stimuli : int
        Number of distinct stimuli.  Default 6.
    num_actions : int
        Number of possible actions.  Default 3.
    q_init : float
        Initial Q-value.  Default 0.5.
    use_pscan : bool
        Must be False.  Raises NotImplementedError if True (pscan + vmap
        composition is out of scope for Issue 1 rollout).

    Returns
    -------
    jnp.ndarray
        Shape (N,) float — total log-likelihood per participant.

    Raises
    ------
    NotImplementedError
        If ``use_pscan=True``.
    """
    if use_pscan:
        raise NotImplementedError(
            "q_learning_fully_batched_likelihood: use_pscan=True is not "
            "supported.  pscan + vmap composition is out of scope for the "
            "Issue 1 rollout.  Pass use_pscan=False."
        )

    def _block_ll(stim, act, rew, mask, ap, an, e):
        # Scalar log-lik for a single (participant, block).
        return q_learning_block_likelihood(
            stimuli=stim,
            actions=act,
            rewards=rew,
            alpha_pos=ap,
            alpha_neg=an,
            epsilon=e,
            num_stimuli=num_stimuli,
            num_actions=num_actions,
            q_init=q_init,
            mask=mask,
            return_pointwise=False,
        )

    # Inner vmap: over blocks.  Data args on axis 0, params broadcast (None).
    _over_blocks = jax.vmap(
        _block_ll,
        in_axes=(0, 0, 0, 0, None, None, None),
        out_axes=0,
    )

    def _participant_ll(stim, act, rew, mask, ap, an, e):
        block_lls = _over_blocks(stim, act, rew, mask, ap, an, e)
        return block_lls.sum()

    # Outer vmap: over participants.  Everything on axis 0.
    _over_participants = jax.vmap(
        _participant_ll,
        in_axes=(0, 0, 0, 0, 0, 0, 0),
        out_axes=0,
    )
    return _over_participants(
        stimuli, actions, rewards, masks,
        alpha_pos, alpha_neg, epsilon,
    )

def q_learning_block_likelihood_pscan(
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
    Q-learning block likelihood using parallel scan for Q-value trajectories.

    Drop-in replacement for ``q_learning_block_likelihood``.

    Phase 1 (parallel): Pre-compute Q_for_policy[t] for all t via
    ``associative_scan_q_update`` (O(log T) depth on parallel hardware).
    Phase 2 (vectorized): Compute softmax + epsilon + log_prob for all
    trials simultaneously using ``jax.vmap``. No sequential carry needed
    (M1 has no perseveration).

    Parameters
    ----------
    stimuli : array, shape (n_trials,)
    actions : array, shape (n_trials,)
    rewards : array, shape (n_trials,)
    alpha_pos, alpha_neg : float
    epsilon : float
    num_stimuli, num_actions : int
    q_init : float
    mask : array, shape (n_trials,), optional
    return_pointwise : bool, optional

    Returns
    -------
    float or tuple[float, jnp.ndarray]
        Same as ``q_learning_block_likelihood``.
    """
    if mask is None:
        mask = jnp.ones(len(stimuli))

    # ------------------------------------------------------------------
    # Phase 1 (parallel): Q-value trajectories for the whole block
    # Q_for_policy[t] = Q-table BEFORE update at trial t
    # ------------------------------------------------------------------
    T = stimuli.shape[0]
    Q_for_policy = associative_scan_q_update(
        stimuli, actions, rewards, mask,
        alpha_pos, alpha_neg, q_init,
        num_stimuli, num_actions,
    )  # (T, S, A)

    # ------------------------------------------------------------------
    # Phase 2 (vectorized): policy computation for all trials at once
    # No perseveration carry — fully embarrassingly parallel.
    # ------------------------------------------------------------------
    q_vals = Q_for_policy[jnp.arange(T), stimuli]  # (T, A)
    base_probs = jax.vmap(softmax_policy, in_axes=(0, None))(q_vals, FIXED_BETA)
    noisy_probs = jax.vmap(apply_epsilon_noise, in_axes=(0, None, None))(
        base_probs, epsilon, num_actions
    )  # (T, A)
    log_probs = jnp.log(noisy_probs[jnp.arange(T), actions] + 1e-8) * mask

    if return_pointwise:
        return jnp.sum(log_probs), log_probs
    return jnp.sum(log_probs)

def q_learning_multiblock_likelihood_stacked_pscan(
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
    *,
    return_pointwise: bool = False,
) -> float | tuple[float, jnp.ndarray]:
    """
    Q-learning multiblock likelihood using parallel scan.

    Drop-in replacement for ``q_learning_multiblock_likelihood_stacked``.

    Parameters
    ----------
    stimuli_stacked : array, shape (n_blocks, max_trials)
    actions_stacked : array, shape (n_blocks, max_trials)
    rewards_stacked : array, shape (n_blocks, max_trials)
    masks_stacked : array, shape (n_blocks, max_trials)
    alpha_pos, alpha_neg, epsilon : float
    num_stimuli, num_actions, q_init : int / float
    return_pointwise : bool, optional

    Returns
    -------
    float or tuple[float, jnp.ndarray]
        Same as ``q_learning_multiblock_likelihood_stacked``.
    """
    num_blocks = stimuli_stacked.shape[0]

    if return_pointwise:
        def scan_body(total_ll, block_idx):
            block_ll, block_probs = q_learning_block_likelihood_pscan(
                stimuli=stimuli_stacked[block_idx],
                actions=actions_stacked[block_idx],
                rewards=rewards_stacked[block_idx],
                alpha_pos=alpha_pos,
                alpha_neg=alpha_neg,
                epsilon=epsilon,
                num_stimuli=num_stimuli,
                num_actions=num_actions,
                q_init=q_init,
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
            block_ll = q_learning_block_likelihood_pscan(
                stimuli=stimuli_stacked[block_idx],
                actions=actions_stacked[block_idx],
                rewards=rewards_stacked[block_idx],
                alpha_pos=alpha_pos,
                alpha_neg=alpha_neg,
                epsilon=epsilon,
                num_stimuli=num_stimuli,
                num_actions=num_actions,
                q_init=q_init,
                mask=masks_stacked[block_idx],
            )
            return total_ll + block_ll

        return lax.fori_loop(0, num_blocks, body_fn, 0.0)

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

def qlearning_hierarchical_model(
    participant_data: dict[Any, dict[str, list]],
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
) -> None:
    """
    Hierarchical Bayesian Q-learning model for multiple participants.

    Following Senta et al. (2025):
    - Beta is FIXED at 50 (not estimated) for parameter identifiability
    - Epsilon noise parameter is estimated to capture random responding

    Model Structure:
    ---------------
    # Group-level (population) parameters
    mu_alpha_pos ~ Beta(3, 2)      # Mean positive learning rate ~ 0.6
    sigma_alpha_pos ~ HalfNormal(0.3) # Variability in alpha_pos
    mu_alpha_neg ~ Beta(2, 3)      # Mean negative learning rate ~ 0.4
    sigma_alpha_neg ~ HalfNormal(0.3) # Variability in alpha_neg
    mu_epsilon ~ Beta(1, 19)       # Mean epsilon noise ~ 0.05
    sigma_epsilon ~ HalfNormal(0.1)  # Variability in epsilon

    # Individual-level parameters (non-centered)
    z_alpha_pos_i ~ Normal(0, 1)
    alpha_pos_i = expit(logit(mu_alpha_pos) + sigma_alpha_pos * z_alpha_pos_i)

    z_alpha_neg_i ~ Normal(0, 1)
    alpha_neg_i = expit(logit(mu_alpha_neg) + sigma_alpha_neg * z_alpha_neg_i)

    z_epsilon_i ~ Normal(0, 1)
    epsilon_i = expit(logit(mu_epsilon) + sigma_epsilon * z_epsilon_i)

    # Likelihood (beta=50 fixed)
    actions_i ~ Softmax(Q-values; alpha_pos_i, alpha_neg_i, beta=50, epsilon_i)

    Parameters
    ----------
    participant_data : dict
        Nested dict: {participant_id: {
            'stimuli_blocks': list of arrays,
            'actions_blocks': list of arrays,
            'rewards_blocks': list of arrays
        }}
    num_stimuli : int
        Number of possible stimuli
    num_actions : int
        Number of possible actions
    q_init : float
        Initial Q-value for all state-action pairs

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

    # Epsilon noise: bounded [0, 1], prior centered around 0.05
    # Beta(1, 19) gives mean of 1/20 = 0.05
    mu_epsilon = numpyro.sample("mu_epsilon", dist.Beta(1, 19))
    sigma_epsilon = numpyro.sample("sigma_epsilon", dist.HalfNormal(0.1))

    # ========================================================================
    # INDIVIDUAL-LEVEL PARAMETERS (NON-CENTERED)
    # ========================================================================

    with numpyro.plate("participants", num_participants):
        # Sample standard normal offsets
        z_alpha_pos = numpyro.sample("z_alpha_pos", dist.Normal(0, 1))
        z_alpha_neg = numpyro.sample("z_alpha_neg", dist.Normal(0, 1))
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
        epsilon = numpyro.deterministic(
            "epsilon",
            jax.scipy.special.expit(
                jax.scipy.special.logit(mu_epsilon) + sigma_epsilon * z_epsilon
            ),
        )

    # ========================================================================
    # LIKELIHOOD: Compute for each participant
    # ========================================================================

    for i, participant_id in enumerate(participant_ids):
        pdata = participant_data[participant_id]

        # Get individual parameters
        alpha_pos_i = alpha_pos[i]
        alpha_neg_i = alpha_neg[i]
        epsilon_i = epsilon[i]

        # Compute log-likelihood across all blocks for this participant
        # Note: beta is fixed at 50 inside the likelihood function
        log_lik = q_learning_multiblock_likelihood(
            stimuli_blocks=pdata["stimuli_blocks"],
            actions_blocks=pdata["actions_blocks"],
            rewards_blocks=pdata["rewards_blocks"],
            alpha_pos=alpha_pos_i,
            alpha_neg=alpha_neg_i,
            epsilon=epsilon_i,
            num_stimuli=num_stimuli,
            num_actions=num_actions,
            q_init=q_init,
        )

        # Add to model via factor (log probability)
        numpyro.factor(f"obs_p{participant_id}", log_lik)

def qlearning_hierarchical_model_stacked(
    participant_data_stacked: dict,
    covariate_lec: jnp.ndarray | None = None,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    use_pscan: bool = False,
    stacked_arrays: dict | None = None,
) -> None:
    """Hierarchical Bayesian M1 (Q-learning) model using stacked pre-padded arrays.

    Uses the hBayesDM non-centered parameterization convention (Ahn, Haines,
    Zhang 2017):
    ``theta_unc = mu_pr + sigma_pr * z``,
    ``theta = lower + (upper - lower) * Phi_approx(theta_unc)``,
    where ``Phi_approx = jax.scipy.stats.norm.cdf``.

    Three parameters (alpha_pos, alpha_neg, epsilon) are sampled via
    :func:`sample_bounded_param` from :mod:`rlwm.fitting.numpyro_helpers`.

    Likelihood is accumulated via a single ``numpyro.factor("obs", ...)`` call
    using ``q_learning_fully_batched_likelihood`` (nested vmap over participants
    and blocks).  This mirrors the M3 refactor from commit 6403c72 and removes
    the per-participant Python for-loop that was the main NUTS leapfrog
    bottleneck.

    Parameters
    ----------
    participant_data_stacked : dict
        Mapping from participant_id to stacked-format arrays (from
        ``prepare_stacked_participant_data``).  Keys per participant:
        ``stimuli_stacked``, ``actions_stacked``, ``rewards_stacked``,
        ``masks_stacked`` — each shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``.
        NOTE: ``set_sizes_stacked`` is NOT required by the Q-learning
        likelihood and is ignored even if present.
    covariate_lec : jnp.ndarray or None
        Reserved for forward compatibility.  Must be ``None``; passing a
        non-None value raises ``NotImplementedError`` because Q-learning
        has no natural Level-2 target parameter in this release.
    num_stimuli : int
        Number of distinct stimuli in the task.  Default 6.
    num_actions : int
        Number of possible actions.  Default 3.
    q_init : float
        Initial Q-value for all state-action pairs.  Default 0.5.
    use_pscan : bool
        Must be ``False``.  The fully-batched vmap path does not compose
        with the O(log T) associative scan variants.
    stacked_arrays : dict or None
        Pre-computed output of ``stack_across_participants``.  If ``None``,
        computed here.  ``fit_bayesian.py`` passes this to avoid recomputing
        (N, B, T) tensors on every MCMC trace call.

    Notes
    -----
    - Participant ordering follows ``sorted(participant_data_stacked.keys())``
      to align with covariate arrays prepared by downstream scripts.
    - Do NOT modify this function's API: ``fit_bayesian.py`` dispatches to it
      by name.
    """
    if covariate_lec is not None:
        raise NotImplementedError(
            "qlearning_hierarchical_model_stacked: covariate_lec is not "
            "supported for Q-learning (no natural L2 target parameter). "
            "Pass covariate_lec=None."
        )
    if use_pscan:
        raise NotImplementedError(
            "qlearning_hierarchical_model_stacked: use_pscan + fully-batched "
            "vmap path is not implemented.  Pass use_pscan=False."
        )

    from rlwm.fitting.numpyro_helpers import (
        PARAM_PRIOR_DEFAULTS,
        sample_bounded_param,
    )

    n_participants = len(participant_data_stacked)

    # ------------------------------------------------------------------
    # Group priors for 3 parameters via hBayesDM non-centered convention
    # ------------------------------------------------------------------
    sampled: dict[str, jnp.ndarray] = {}
    for param in ["alpha_pos", "alpha_neg", "epsilon"]:
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

    from rlwm.fitting.models.qlearning import q_learning_fully_batched_likelihood

    per_participant_ll = q_learning_fully_batched_likelihood(
        stimuli=stacked_arrays["stimuli"],
        actions=stacked_arrays["actions"],
        rewards=stacked_arrays["rewards"],
        masks=stacked_arrays["masks"],
        alpha_pos=sampled["alpha_pos"],
        alpha_neg=sampled["alpha_neg"],
        epsilon=sampled["epsilon"],
        num_stimuli=num_stimuli,
        num_actions=num_actions,
        q_init=q_init,
    )
    numpyro.factor("obs", per_participant_ll.sum())
