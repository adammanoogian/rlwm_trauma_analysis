"""
Shared fixtures for fitting module tests.

Provides synthetic data generators for Q-learning and WM-RL model testing.
"""

import numpy as np
import jax.numpy as jnp
import jax
import pytest


def simulate_qlearning_block(
    alpha_pos: float,
    alpha_neg: float,
    epsilon: float,
    beta: float,
    n_trials: int,
    n_stim: int,
    n_act: int,
    seed: int = 42
):
    """
    Simulate a single block of Q-learning behavior.

    Args:
        alpha_pos: Positive learning rate
        alpha_neg: Negative learning rate
        epsilon: Noise parameter
        beta: Inverse temperature
        n_trials: Number of trials to simulate
        n_stim: Number of unique stimuli
        n_act: Number of actions
        seed: Random seed for reproducibility

    Returns:
        Tuple of (stimuli, actions, rewards) as numpy arrays
    """
    Q = np.ones((n_stim, n_act)) * 0.5
    np.random.seed(seed)
    correct_actions = np.random.randint(0, n_act, n_stim)

    stimuli, actions, rewards = [], [], []

    for t in range(n_trials):
        s = np.random.randint(0, n_stim)
        q_s = Q[s, :]
        exp_q = np.exp(beta * (q_s - q_s.max()))
        p = exp_q / exp_q.sum()
        p = epsilon / n_act + (1 - epsilon) * p

        a = np.random.choice(n_act, p=p)
        r = 1.0 if a == correct_actions[s] else 0.0

        stimuli.append(s)
        actions.append(a)
        rewards.append(r)

        delta = r - Q[s, a]
        alpha = alpha_pos if delta > 0 else alpha_neg
        Q[s, a] += alpha * delta

    return (
        np.array(stimuli, dtype=np.int32),
        np.array(actions, dtype=np.int32),
        np.array(rewards, dtype=np.float32)
    )


def simulate_wmrl_block(
    alpha_pos: float,
    alpha_neg: float,
    phi: float,
    rho: float,
    capacity: float,
    epsilon: float,
    beta: float,
    n_trials: int,
    n_stim: int,
    n_act: int,
    set_size: int = 6,
    seed: int = 42
):
    """
    Simulate a single block of WM-RL behavior.

    Simplified WM-RL simulation that combines RL and WM components.

    Args:
        alpha_pos: Positive learning rate (RL)
        alpha_neg: Negative learning rate (RL)
        phi: WM decay rate
        rho: Base WM reliance
        capacity: WM capacity (K)
        epsilon: Noise parameter
        beta: Inverse temperature
        n_trials: Number of trials
        n_stim: Number of unique stimuli
        n_act: Number of actions
        set_size: Set size for capacity computation
        seed: Random seed

    Returns:
        Tuple of (stimuli, actions, rewards, set_sizes) as numpy arrays
    """
    np.random.seed(seed)

    # Initialize Q-values and WM
    Q = np.ones((n_stim, n_act)) * 0.5
    WM = np.ones((n_stim, n_act)) * (1.0 / n_act)

    correct_actions = np.random.randint(0, n_act, n_stim)

    stimuli, actions, rewards = [], [], []

    for t in range(n_trials):
        s = np.random.randint(0, n_stim)

        # Compute WM weight
        w = rho * min(1.0, capacity / set_size)

        # Combine RL and WM
        q_s = Q[s, :]
        wm_s = WM[s, :]

        # Softmax on RL
        exp_q = np.exp(beta * (q_s - q_s.max()))
        p_rl = exp_q / exp_q.sum()

        # Combined probability
        p = w * wm_s + (1 - w) * p_rl

        # Add noise
        p = epsilon / n_act + (1 - epsilon) * p

        a = np.random.choice(n_act, p=p)
        r = 1.0 if a == correct_actions[s] else 0.0

        stimuli.append(s)
        actions.append(a)
        rewards.append(r)

        # Update RL
        delta = r - Q[s, a]
        alpha = alpha_pos if delta > 0 else alpha_neg
        Q[s, a] += alpha * delta

        # Update WM (immediate overwrite)
        WM[s, a] = r

        # Decay WM
        WM = WM * (1 - phi) + phi * (1.0 / n_act)

    return (
        np.array(stimuli, dtype=np.int32),
        np.array(actions, dtype=np.int32),
        np.array(rewards, dtype=np.float32),
        np.full(n_trials, set_size, dtype=np.int32)
    )


@pytest.fixture
def qlearning_synthetic_data():
    """
    Generate synthetic Q-learning data for testing.

    Returns 3 blocks of data with known parameters.
    """
    true_params = {
        'alpha_pos': 0.4,
        'alpha_neg': 0.15,
        'epsilon': 0.05,
        'beta': 50.0
    }

    stimuli_blocks, actions_blocks, rewards_blocks = [], [], []

    for i in range(3):
        s, a, r = simulate_qlearning_block(
            alpha_pos=true_params['alpha_pos'],
            alpha_neg=true_params['alpha_neg'],
            epsilon=true_params['epsilon'],
            beta=true_params['beta'],
            n_trials=30,
            n_stim=3,
            n_act=3,
            seed=42 + i
        )
        stimuli_blocks.append(s)
        actions_blocks.append(a)
        rewards_blocks.append(r)

    return {
        'stimuli_blocks': stimuli_blocks,
        'actions_blocks': actions_blocks,
        'rewards_blocks': rewards_blocks,
        'true_params': true_params
    }


@pytest.fixture
def wmrl_synthetic_data():
    """
    Generate synthetic WM-RL data for testing.

    Returns 2 blocks of data with known parameters.
    """
    true_params = {
        'alpha_pos': 0.3,
        'alpha_neg': 0.1,
        'phi': 0.1,
        'rho': 0.7,
        'capacity': 4.0,
        'epsilon': 0.05,
        'beta': 50.0
    }

    stimuli_blocks, actions_blocks, rewards_blocks, set_sizes_blocks = [], [], [], []

    for i in range(2):
        s, a, r, ss = simulate_wmrl_block(
            alpha_pos=true_params['alpha_pos'],
            alpha_neg=true_params['alpha_neg'],
            phi=true_params['phi'],
            rho=true_params['rho'],
            capacity=true_params['capacity'],
            epsilon=true_params['epsilon'],
            beta=true_params['beta'],
            n_trials=30,
            n_stim=6,
            n_act=3,
            set_size=6,
            seed=42 + i
        )
        stimuli_blocks.append(s)
        actions_blocks.append(a)
        rewards_blocks.append(r)
        set_sizes_blocks.append(ss)

    return {
        'stimuli_blocks': stimuli_blocks,
        'actions_blocks': actions_blocks,
        'rewards_blocks': rewards_blocks,
        'set_sizes_blocks': set_sizes_blocks,
        'true_params': true_params
    }


@pytest.fixture
def wmrl_participant_data():
    """
    Generate WM-RL data in the format expected by numpyro models.

    Returns participant_data dictionary for 2 participants.
    """
    key = jax.random.PRNGKey(42)
    participant_data = {}

    for i in range(2):
        stimuli_blocks = []
        actions_blocks = []
        rewards_blocks = []
        set_sizes_blocks = []

        # 2 blocks of 30 trials each
        for block in range(2):
            key, subkey = jax.random.split(key)
            stimuli = jax.random.randint(subkey, (30,), 0, 6)

            key, subkey = jax.random.split(key)
            actions = jax.random.randint(subkey, (30,), 0, 3)

            key, subkey = jax.random.split(key)
            rewards = jax.random.bernoulli(subkey, 0.7, (30,)).astype(jnp.float32)

            set_sizes = jnp.ones((30,)) * 5

            stimuli_blocks.append(stimuli)
            actions_blocks.append(actions)
            rewards_blocks.append(rewards)
            set_sizes_blocks.append(set_sizes)

        participant_data[i] = {
            'stimuli_blocks': stimuli_blocks,
            'actions_blocks': actions_blocks,
            'rewards_blocks': rewards_blocks,
            'set_sizes_blocks': set_sizes_blocks
        }

    return participant_data
