"""Shared JAX primitives for RLWM fitting: padding, softmax, scans, perseveration precompute, stacking.

Relocated here in Phase 29-08 from :mod:`rlwm.fitting.jax_likelihoods` and
:mod:`rlwm.fitting.numpyro_models` (the stacking helpers).  The canonical
home for functions listed in ``__all__`` below is now this module.  The
``rlwm.fitting.jax_likelihoods`` and ``rlwm.fitting.numpyro_models``
modules are retained as thin wildcard re-export shims for backward
compatibility with v4.0 closure invariants and external callers.

Follows Senta et al. (2025) PLoS Comp. Biol. 21(9):e1012872 math conventions:
``beta = 50`` fixed during learning, epsilon-noise action probabilities, and
asymmetric Q-learning with ``alpha_pos`` / ``alpha_neg``.
"""
from __future__ import annotations

from typing import Any  # noqa: F401 — kept for downstream signatures

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

__all__ = [
    "log_gpu_memory",
    "FIXED_BETA",
    "DEFAULT_EPSILON",
    "NUM_ACTIONS",
    "MAX_TRIALS_PER_BLOCK",
    "MAX_BLOCKS",
    "pad_block_to_max",
    "pad_blocks_to_max",
    "softmax_policy",
    "apply_epsilon_noise",
    "affine_scan",
    "associative_scan_q_update",
    "associative_scan_wm_update",
    "precompute_last_action_global",
    "precompute_last_actions_per_stimulus",
    "prepare_stacked_participant_data",
    "stack_across_participants",
]


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

FIXED_BETA = 50.0

DEFAULT_EPSILON = 0.05

NUM_ACTIONS = 3

MAX_TRIALS_PER_BLOCK = 100

MAX_BLOCKS = 17

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

def affine_scan(
    a_seq: jnp.ndarray,
    b_seq: jnp.ndarray,
    x0: jnp.ndarray,
) -> jnp.ndarray:
    """
    Parallel prefix scan for the AR(1) recurrence x_t = a_t * x_{t-1} + b_t.

    Uses ``jax.lax.associative_scan`` with the affine operator
    ``(a_r, b_r) ∘ (a_l, b_l) = (a_r*a_l, a_r*b_l + b_r)`` to compute all
    output states in O(log T) steps (wall-clock on parallel hardware).

    The associativity proof:
    ``x_2 = a_2*(a_1*x_0 + b_1) + b_2 = (a_2*a_1)*x_0 + (a_2*b_1 + b_2)``

    Parameters
    ----------
    a_seq : array, shape (T, ...)
        Multiplicative coefficients.
    b_seq : array, shape (T, ...)
        Additive coefficients.
    x0 : array, shape (...)
        Initial value, broadcastable to the trailing dimensions of a_seq.

    Returns
    -------
    x_all : array, shape (T, ...)
        ``x_all[t]`` is the state AFTER applying operator at position t.
        ``x_all[0] = a_seq[0]*x0 + b_seq[0]``.

    Notes
    -----
    Inactive positions use identity coefficients ``(1.0, 0.0)`` so their
    state passes through unchanged. Hard resets use ``(0.0, r)`` to zero
    out all history before that position.
    """
    # Prepend (1.0, x0) as the identity/init element.
    # This makes the prefix scan include x0 without treating it as a step.
    T = a_seq.shape[0]
    trailing = a_seq.shape[1:]

    # Broadcast x0 to trailing shape
    x0_broadcast = jnp.broadcast_to(x0, trailing)

    # Prepend: a=1.0 (identity multiplier), b=x0 (initial value)
    a_init = jnp.ones(trailing)
    b_init = x0_broadcast

    # Stack: shape (T+1, ...)
    a_full = jnp.concatenate([a_init[None], a_seq], axis=0)
    b_full = jnp.concatenate([b_init[None], b_seq], axis=0)

    def _affine_op(
        left: tuple[jnp.ndarray, jnp.ndarray],
        right: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compose two AR(1) operators: right ∘ left."""
        a_l, b_l = left
        a_r, b_r = right
        return a_r * a_l, a_r * b_l + b_r

    # associative_scan accumulates left-to-right: result[t] = op_t ∘ ... ∘ op_0
    _, b_accumulated = lax.associative_scan(_affine_op, (a_full, b_full))

    # Drop the prepended init element; result[t] corresponds to trial t
    return b_accumulated[1:]

def associative_scan_q_update(
    stimuli: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    masks: jnp.ndarray,
    alpha_pos: float,
    alpha_neg: float,
    q_init: float,
    num_stimuli: int,
    num_actions: int,
) -> jnp.ndarray:
    """
    Compute Q-value trajectories for a single block via parallel scan.

    Uses the AR(1) reformulation of the Q-learning update:
    ``Q_t(s,a) = (1-alpha)*Q_{t-1}(s,a) + alpha*r_t``
    for the active (s,a) pair at trial t; identity elsewhere.

    **Alpha approximation:** Uses ``alpha = alpha_pos if r==1 else alpha_neg``
    instead of the exact ``alpha = alpha_pos if delta>0 else alpha_neg``.
    Agreement with the exact sequential rule is < 1e-5 for typical parameters
    (alpha <= 0.5) and < 1e-3 for extreme parameters (alpha ~ 0.95).
    See ``docs/PARALLEL_SCAN_LIKELIHOOD.md`` for derivation.

    Parameters
    ----------
    stimuli : array, shape (T,)
        Stimulus indices (may include padding).
    actions : array, shape (T,)
        Action indices (may include padding).
    rewards : array, shape (T,)
        Rewards, 0 or 1 (may include padding).
    masks : array, shape (T,)
        1.0 for real trials, 0.0 for padding.
    alpha_pos : float
        Learning rate for positive outcomes (r=1).
    alpha_neg : float
        Learning rate for negative outcomes (r=0).
    q_init : float
        Initial Q-value for all (s, a) pairs.
    num_stimuli : int
        Number of stimuli (S).
    num_actions : int
        Number of actions (A).

    Returns
    -------
    Q_for_policy : array, shape (T, num_stimuli, num_actions)
        ``Q_for_policy[t]`` is the Q-table BEFORE the update at trial t.
        Use this array for computing the policy at trial t.
    """
    T = stimuli.shape[0]
    S, A = num_stimuli, num_actions

    # One-hot encode stimuli and actions: shapes (T, S) and (T, A)
    stim_oh = jax.nn.one_hot(stimuli, S)    # (T, S)
    act_oh = jax.nn.one_hot(actions, A)     # (T, A)

    # Outer product -> (T, S, A): 1 only at the active (s, a) pair
    sa_mask = stim_oh[:, :, None] * act_oh[:, None, :]  # (T, S, A)

    # Apply trial validity mask: inactive for padding trials
    active = sa_mask * masks[:, None, None]  # (T, S, A)

    # Data-dependent alpha: reward-based approximation
    # alpha_t = alpha_pos if r==1, else alpha_neg
    alpha_t = jnp.where(
        rewards[:, None, None] == 1.0,
        alpha_pos,
        alpha_neg,
    )  # (T, S, A) broadcast

    # AR(1) coefficients
    # Active position: a = 1 - alpha, b = alpha * r
    # Inactive position: a = 1.0 (identity), b = 0.0
    a_seq = jnp.where(active, 1.0 - alpha_t, 1.0)
    b_seq = jnp.where(active, alpha_t * rewards[:, None, None], 0.0)

    # Initial Q-table: shape (S, A)
    q_init_table = jnp.ones((S, A)) * q_init

    # Run parallel scan: Q_all[t] = Q-table AFTER update at trial t
    Q_all = affine_scan(a_seq, b_seq, x0=q_init_table)  # (T, S, A)

    # For policy at trial t, we need Q BEFORE update t.
    # Prepend Q_init and drop the last element.
    Q_for_policy = jnp.concatenate(
        [q_init_table[None], Q_all[:-1]], axis=0
    )  # (T, S, A)

    return Q_for_policy

def associative_scan_wm_update(
    stimuli: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    masks: jnp.ndarray,
    phi: float,
    wm_init: float,
    num_stimuli: int,
    num_actions: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute WM trajectories for a single block via parallel scan.

    Implements a single AR(1) associative scan that encodes the combined
    decay + conditional overwrite update:

    1. Decay (always, including padding trials):
       ``WM_decayed = (1-phi)*WM + phi*wm_init``
    2. Overwrite (only on valid trials):
       ``WM[s,a] <- r``

    AR(1) coefficients per (s,a) entry at trial t:
    - Inactive (not the presented s,a): ``a_t=1-phi, b_t=phi*wm_init``
    - Active + valid (overwrite): ``a_t=0, b_t=r``
    - Active + padding (decay only, no overwrite): ``a_t=1-phi, b_t=phi*wm_init``

    The ``wm_for_policy`` output is derived from ``wm_after_update`` via a
    vectorized post-scan computation: ``wm_for_policy[t] = (1-phi)*carry[t] + phi*wm_init``
    where ``carry[t]`` is the WM state entering trial t.

    The WM decay formula from the sequential implementation (confirmed from
    ``wmrl_m3_block_likelihood`` in jax_likelihoods.py, line ~993):
    ``WM_decayed = (1 - phi) * WM_table + phi * WM_baseline``
    maps to ``a_t = 1-phi, b_t = phi*wm_init``.

    Parameters
    ----------
    stimuli : array, shape (T,)
        Stimulus indices (may include padding).
    actions : array, shape (T,)
        Action indices (may include padding).
    rewards : array, shape (T,)
        Rewards, 0 or 1 (may include padding).
    masks : array, shape (T,)
        1.0 for real trials, 0.0 for padding.
    phi : float
        WM decay rate (0 = no decay, 1 = full reset to baseline each trial).
    wm_init : float
        WM baseline value (= 1/nA for uniform prior).
    num_stimuli : int
        Number of stimuli (S).
    num_actions : int
        Number of actions (A).

    Returns
    -------
    wm_for_policy : array, shape (T, num_stimuli, num_actions)
        ``wm_for_policy[t]`` is the WM table AFTER decay but BEFORE overwrite
        at trial t. Used for policy computation at trial t.
    wm_after_update : array, shape (T, num_stimuli, num_actions)
        ``wm_after_update[t]`` is the WM table AFTER both decay and overwrite
        at trial t. Used for propagation to trial t+1.
    """
    T = stimuli.shape[0]
    S, A = num_stimuli, num_actions

    # One-hot encodings: (T, S), (T, A)
    stim_oh = jax.nn.one_hot(stimuli, S)
    act_oh = jax.nn.one_hot(actions, A)

    # Active (s, a) mask: (T, S, A)
    sa_mask = stim_oh[:, :, None] * act_oh[:, None, :]
    active = sa_mask * masks[:, None, None]  # 1 only at valid, active positions

    # Initial WM table
    wm_init_table = jnp.ones((S, A)) * wm_init

    # -------------------------------------------------------------------------
    # Single pass: Decay + overwrite together.
    #
    # The scan encodes the full sequential update per trial t:
    #   WM_decayed = (1-phi)*carry + phi*wm_init     [decay, ALWAYS happens]
    #   WM_updated = WM_decayed.at[s,a].set(r)       [overwrite ONLY if valid]
    #
    # In AR(1) form for each (s, a) entry at trial t:
    #   - All inactive positions (not the presented s,a), any mask:
    #       a_t = 1-phi, b_t = phi*wm_init   (decay only, no overwrite)
    #   - Active + valid (presented s,a, real trial, mask=1):
    #       Combined decay+overwrite: x_t = 0 * x_{t-1} + r
    #       => a_t=0, b_t=r   (zero multiplier erases history; r is new value)
    #   - Active + padding (presented s,a index, but mask=0):
    #       Decay only: a_t=1-phi, b_t=phi*wm_init  (no overwrite for padding)
    #
    # IMPORTANT: Decay happens on ALL trials, even padding trials. The mask
    # only gates the overwrite step, not the decay step. This matches the
    # sequential implementation in wmrl_m3_block_likelihood where:
    #   WM_decayed = (1-phi)*WM + phi*baseline   (always, comment: "Decay
    #   happens for all trials (valid or not) to maintain consistent WM state")
    # -------------------------------------------------------------------------
    a_seq = jnp.full((T, S, A), 1.0 - phi)   # base: decay everywhere
    b_seq = jnp.full((T, S, A), phi * wm_init)

    # Override ONLY active+valid positions: reset encodes (decay then overwrite)
    # Note: we do NOT override padding positions — they correctly use decay.
    a_seq = jnp.where(active, 0.0, a_seq)
    b_seq = jnp.where(active, rewards[:, None, None], b_seq)

    # WM_all[t] = WM AFTER (decay+overwrite) at trial t
    WM_all = affine_scan(a_seq, b_seq, x0=wm_init_table)  # (T, S, A)

    # wm_after_update[t] = WM after both decay and overwrite at trial t.
    # This is directly WM_all[t].
    wm_after_update = WM_all  # (T, S, A)

    # wm_for_policy[t] = WM after decay at trial t, BEFORE overwrite.
    # Sequential model: WM entering trial t is WM_all[t-1] (= carry from t-1).
    # Decay is applied to that carry: (1-phi)*WM_all[t-1] + phi*wm_init.
    # For t=0, the carry is wm_init_table.
    # We construct the "carry entering t" array by prepending wm_init and
    # dropping the last element.
    wm_carry_in = jnp.concatenate(
        [wm_init_table[None], WM_all[:-1]], axis=0
    )  # (T, S, A): wm_carry_in[t] = WM carry entering trial t

    # Apply decay: wm_for_policy[t] = (1-phi)*carry_in[t] + phi*wm_init
    wm_for_policy = (1.0 - phi) * wm_carry_in + phi * wm_init  # (T, S, A)

    return wm_for_policy, wm_after_update

def precompute_last_action_global(
    actions: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Precompute global last_action array from observed data.

    For each trial t, ``result[t]`` is the most recent valid action before
    trial t. This eliminates the ``last_action`` carry in the Phase 2
    sequential ``lax.scan`` for M3/M5 models, making the policy computation
    embarrassingly parallel.

    This function is **parameter-independent** and should be called once
    before MCMC begins, not at every likelihood evaluation.

    Parameters
    ----------
    actions : array, shape (T,)
        Observed action indices (int).
    mask : array, shape (T,)
        1.0 for valid trials, 0.0 for padding/invalid.

    Returns
    -------
    last_action : array, shape (T,), dtype int32
        ``last_action[0] = -1`` (sentinel: no previous action).
        ``last_action[t]`` = most recent valid action before trial t.
        For masked trials, the last valid action propagates forward.

    Notes
    -----
    Matches the sequential carry behavior in ``wmrl_m3_block_likelihood_pscan``
    where ``new_last_action = jnp.where(valid, action, last_action)``.
    """

    def _scan_fn(
        last_act: jnp.ndarray,
        inputs: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        action, valid = inputs
        # Output: last_action BEFORE this trial (for the policy at this trial)
        out = last_act
        # Update: if this trial is valid, record its action for future trials
        new_last_act = jnp.where(valid, action, last_act).astype(jnp.int32)
        return new_last_act, out

    _, last_action_arr = lax.scan(
        _scan_fn,
        jnp.array(-1, dtype=jnp.int32),
        (actions, mask),
    )
    return last_action_arr

def precompute_last_actions_per_stimulus(
    stimuli: jnp.ndarray,
    actions: jnp.ndarray,
    mask: jnp.ndarray,
    num_stimuli: int = 6,
) -> jnp.ndarray:
    """
    Precompute per-stimulus last_action array from observed data.

    For each trial t, ``result[t]`` is the last action taken for the stimulus
    presented at trial t, considering only trials 0..t-1. This eliminates the
    ``last_actions`` carry (shape ``(num_stimuli,)``) in the Phase 2
    sequential ``lax.scan`` for M6a/M6b models.

    This function is **parameter-independent** and should be called once
    before MCMC begins, not at every likelihood evaluation.

    Parameters
    ----------
    stimuli : array, shape (T,)
        Stimulus indices (int).
    actions : array, shape (T,)
        Observed action indices (int).
    mask : array, shape (T,)
        1.0 for valid trials, 0.0 for padding/invalid.
    num_stimuli : int, optional
        Number of distinct stimuli. Default 6.

    Returns
    -------
    last_action_per_trial : array, shape (T,), dtype int32
        ``last_action_per_trial[t]`` = last action taken for ``stimuli[t]``,
        considering only valid trials before t. Returns -1 if stimulus has
        not been seen before.

    Notes
    -----
    Uses ``lax.scan`` with carry ``last_actions`` shape ``(num_stimuli,)``
    initialized to -1. The scan is O(T) but runs once outside the likelihood
    function. Matches the sequential carry behavior in
    ``wmrl_m6a_block_likelihood_pscan``.
    """

    def _scan_fn(
        last_actions: jnp.ndarray,
        inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        stimulus, action, valid = inputs
        # Output: last action for THIS stimulus before THIS trial
        last_action_s = last_actions[stimulus]
        # Update: record this trial's action for this stimulus (if valid)
        new_last_actions = last_actions.at[stimulus].set(
            jnp.where(valid, action, last_action_s).astype(jnp.int32)
        )
        return new_last_actions, last_action_s

    init = jnp.full((num_stimuli,), -1, dtype=jnp.int32)
    _, last_action_per_trial = lax.scan(
        _scan_fn,
        init,
        (stimuli, actions, mask),
    )
    return last_action_per_trial

def prepare_stacked_participant_data(
    data_df: pd.DataFrame,
    participant_col: str = "sona_id",
    block_col: str = "block",
    stimulus_col: str = "stimulus",
    action_col: str = "key_press",
    reward_col: str = "reward",
    set_size_col: str = "set_size",
) -> dict[Any, dict[str, jnp.ndarray]]:
    """Prepare stacked participant data for the M3 hierarchical model.

    Converts a trial-level DataFrame into the pre-stacked JAX array format
    expected by ``wmrl_m3_hierarchical_model`` and ``compute_pointwise_log_lik``.
    Each participant's blocks are padded to ``MAX_TRIALS_PER_BLOCK`` using
    ``pad_block_to_max``, then stacked into 2-D arrays of shape
    ``(n_blocks, MAX_TRIALS_PER_BLOCK)``.

    Parameters
    ----------
    data_df : pd.DataFrame
        Trial-level data with participant, block, stimulus, action, reward
        and set_size columns.
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

    Returns
    -------
    dict[Any, dict[str, jnp.ndarray]]
        Mapping from participant_id to a dict with keys:

        * ``stimuli_stacked``   -- shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``, int32
        * ``actions_stacked``   -- shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``, int32
        * ``rewards_stacked``   -- shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``, float32
        * ``set_sizes_stacked`` -- shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``, float32
        * ``masks_stacked``     -- shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``, float32

    Notes
    -----
    This function is the bridge between the DataFrame pipeline and the stacked
    format consumed by ``compute_pointwise_log_lik`` in ``bayesian_diagnostics.py``.
    The existing ``prepare_data_for_numpyro`` returns lists of arrays (old format);
    this function returns pre-stacked 2-D JAX arrays (new format for Phase 15+).

    Participant keys are sorted before processing so that downstream covariate
    arrays (e.g., ``covariate_lec``) align with ``sorted(result.keys())``.
    """
    participant_data: dict[Any, dict[str, jnp.ndarray]] = {}

    for participant_id in sorted(data_df[participant_col].unique()):
        ppt_df = data_df[data_df[participant_col] == participant_id]

        stimuli_blocks = []
        actions_blocks = []
        rewards_blocks = []
        set_sizes_blocks = []
        masks_blocks = []

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
            # when set_sizes is provided -- mask is LAST, set_sizes is fourth.
            p_stim, p_act, p_rew, p_ss, p_mask = pad_block_to_max(
                stim, act, rew, set_sizes=ss
            )
            stimuli_blocks.append(p_stim)
            actions_blocks.append(p_act)
            rewards_blocks.append(p_rew)
            set_sizes_blocks.append(p_ss)
            masks_blocks.append(p_mask)

        participant_data[participant_id] = {
            "stimuli_stacked": jnp.stack(stimuli_blocks),
            "actions_stacked": jnp.stack(actions_blocks),
            "rewards_stacked": jnp.stack(rewards_blocks),
            "set_sizes_stacked": jnp.stack(set_sizes_blocks),
            "masks_stacked": jnp.stack(masks_blocks),
        }

    return participant_data

def stack_across_participants(
    participant_data_stacked: dict[Any, dict[str, jnp.ndarray]],
) -> dict[str, Any]:
    """Stack per-participant arrays into (N, max_n_blocks, max_trials) tensors.

    Pads participants with fewer blocks to max_n_blocks by appending
    zero-mask blocks. Because mask=0 contributes exactly 0.0 to the
    block likelihood (both the log-prob term and the Q/WM updates are
    gated on mask), padded blocks leave total_ll invariant.

    Participant order follows sorted(participant_data_stacked.keys())
    — same order used by covariate_lec downstream.

    Parameters
    ----------
    participant_data_stacked : dict
        Output of prepare_stacked_participant_data. Per-participant
        arrays have shape (n_blocks_i, MAX_TRIALS_PER_BLOCK=100).

    Returns
    -------
    dict
        Keys (all shape (N, max_n_blocks, 100)):

        * ``stimuli``           -- int32
        * ``actions``           -- int32
        * ``rewards``           -- float32
        * ``set_sizes``         -- float32
        * ``masks``             -- float32 (padded blocks are entirely 0.0)

        Plus:

        * ``participant_ids``   -- list, ordered
        * ``n_blocks_per_ppt``  -- jnp.ndarray shape (N,) int32
    """
    ppt_ids = sorted(participant_data_stacked.keys())
    max_n_blocks = max(
        participant_data_stacked[pid]["stimuli_stacked"].shape[0]
        for pid in ppt_ids
    )
    max_trials = MAX_TRIALS_PER_BLOCK  # 100

    def _pad_blocks(arr: jnp.ndarray, fill_value: float) -> jnp.ndarray:
        n_blocks_i = arr.shape[0]
        pad_blocks = max_n_blocks - n_blocks_i
        if pad_blocks == 0:
            return arr
        pad_shape = (pad_blocks, max_trials)
        pad_arr = jnp.full(pad_shape, fill_value, dtype=arr.dtype)
        return jnp.concatenate([arr, pad_arr], axis=0)

    stacked: dict[str, Any] = {
        "stimuli": jnp.stack([
            _pad_blocks(participant_data_stacked[pid]["stimuli_stacked"], 0)
            for pid in ppt_ids
        ]),
        "actions": jnp.stack([
            _pad_blocks(participant_data_stacked[pid]["actions_stacked"], 0)
            for pid in ppt_ids
        ]),
        "rewards": jnp.stack([
            _pad_blocks(participant_data_stacked[pid]["rewards_stacked"], 0.0)
            for pid in ppt_ids
        ]),
        "set_sizes": jnp.stack([
            _pad_blocks(participant_data_stacked[pid]["set_sizes_stacked"], 6.0)
            for pid in ppt_ids
        ]),
        "masks": jnp.stack([
            _pad_blocks(participant_data_stacked[pid]["masks_stacked"], 0.0)
            for pid in ppt_ids
        ]),
    }
    stacked["participant_ids"] = ppt_ids
    stacked["n_blocks_per_ppt"] = jnp.array(
        [participant_data_stacked[pid]["stimuli_stacked"].shape[0]
         for pid in ppt_ids],
        dtype=jnp.int32,
    )
    return stacked
