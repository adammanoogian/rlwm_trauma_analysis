"""
Test M3 Backward Compatibility

Validates that M3 model with kappa=0 produces numerically identical results to M2 model.
This is critical scientific validation before M3 can be used in analysis.

Tests cover:
- Single block compatibility with multiple seeds
- Multi-block compatibility (3 and 23 blocks)
- Parameter variation tests
- Sanity checks that kappa > 0 produces different likelihood
- Sanity check that high kappa increases likelihood for repetitive actions
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rlwm.fitting.jax_likelihoods import (
    wmrl_block_likelihood,
    wmrl_m3_block_likelihood,
    wmrl_multiblock_likelihood,
    wmrl_m3_multiblock_likelihood
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def wmrl_params():
    """Standard WM-RL parameters for testing."""
    return {
        'alpha_pos': 0.3,
        'alpha_neg': 0.1,
        'phi': 0.2,
        'rho': 0.7,
        'capacity': 4.0,
        'epsilon': 0.05
    }


@pytest.fixture
def parameter_variations():
    """Multiple parameter combinations to test edge cases."""
    return [
        # Standard params
        {'alpha_pos': 0.3, 'alpha_neg': 0.1, 'phi': 0.2, 'rho': 0.7, 'capacity': 4.0, 'epsilon': 0.05},
        # High learning rates
        {'alpha_pos': 0.9, 'alpha_neg': 0.8, 'phi': 0.5, 'rho': 0.9, 'capacity': 6.0, 'epsilon': 0.01},
        # Low learning rates
        {'alpha_pos': 0.1, 'alpha_neg': 0.05, 'phi': 0.05, 'rho': 0.3, 'capacity': 2.0, 'epsilon': 0.1},
        # Asymmetric learning
        {'alpha_pos': 0.8, 'alpha_neg': 0.1, 'phi': 0.3, 'rho': 0.5, 'capacity': 3.0, 'epsilon': 0.02},
        # Edge case: high epsilon noise
        {'alpha_pos': 0.5, 'alpha_neg': 0.5, 'phi': 0.1, 'rho': 0.6, 'capacity': 5.0, 'epsilon': 0.2}
    ]


def generate_block_data(seed, n_trials=40, num_stimuli=6, num_actions=3):
    """
    Generate synthetic block data for testing.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility
    n_trials : int
        Number of trials in block
    num_stimuli : int
        Number of possible stimuli
    num_actions : int
        Number of possible actions

    Returns
    -------
    dict
        Dictionary with stimuli, actions, rewards, set_sizes arrays
    """
    key = jax.random.PRNGKey(seed)

    # Split keys for different random draws
    key_stim, key_act, key_rew, key_set = jax.random.split(key, 4)

    # Generate trial data
    stimuli = jax.random.randint(key_stim, (n_trials,), 0, num_stimuli)
    actions = jax.random.randint(key_act, (n_trials,), 0, num_actions)
    rewards = jax.random.bernoulli(key_rew, 0.7, (n_trials,)).astype(jnp.float32)

    # Generate set sizes (realistic distribution: 2, 3, 5, 6)
    set_size_options = jnp.array([2, 3, 5, 6])
    set_size_indices = jax.random.randint(key_set, (n_trials,), 0, len(set_size_options))
    set_sizes = set_size_options[set_size_indices]

    return {
        'stimuli': stimuli,
        'actions': actions,
        'rewards': rewards,
        'set_sizes': set_sizes
    }


def generate_multiblock_data(seed, n_blocks=3, n_trials_per_block=40):
    """
    Generate multi-block synthetic data.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility
    n_blocks : int
        Number of blocks
    n_trials_per_block : int
        Trials per block

    Returns
    -------
    dict
        Dictionary with lists of arrays for each block
    """
    blocks = {
        'stimuli_blocks': [],
        'actions_blocks': [],
        'rewards_blocks': [],
        'set_sizes_blocks': []
    }

    for block_idx in range(n_blocks):
        block_data = generate_block_data(seed + block_idx * 1000, n_trials_per_block)
        blocks['stimuli_blocks'].append(block_data['stimuli'])
        blocks['actions_blocks'].append(block_data['actions'])
        blocks['rewards_blocks'].append(block_data['rewards'])
        blocks['set_sizes_blocks'].append(block_data['set_sizes'])

    return blocks


def generate_repetitive_actions(seed, n_trials=40, repetition_prob=0.7):
    """
    Generate trial data with high action repetition to test perseveration effect.

    Parameters
    ----------
    seed : int
        Random seed
    n_trials : int
        Number of trials
    repetition_prob : float
        Probability of repeating previous action

    Returns
    -------
    dict
        Dictionary with stimuli, actions, rewards, set_sizes
    """
    key = jax.random.PRNGKey(seed)
    key_stim, key_act, key_rew, key_set, key_repeat = jax.random.split(key, 5)

    # Generate stimuli, rewards, set_sizes normally
    stimuli = jax.random.randint(key_stim, (n_trials,), 0, 6)
    rewards = jax.random.bernoulli(key_rew, 0.7, (n_trials,)).astype(jnp.float32)
    set_size_options = jnp.array([2, 3, 5, 6])
    set_size_indices = jax.random.randint(key_set, (n_trials,), 0, len(set_size_options))
    set_sizes = set_size_options[set_size_indices]

    # Generate actions with repetition tendency
    actions = np.zeros(n_trials, dtype=np.int32)
    actions[0] = jax.random.randint(key_act, (), 0, 3)

    repeat_decisions = jax.random.bernoulli(key_repeat, repetition_prob, (n_trials - 1,))
    random_actions = jax.random.randint(key_act, (n_trials - 1,), 0, 3)

    for t in range(1, n_trials):
        if repeat_decisions[t - 1]:
            actions[t] = actions[t - 1]  # Repeat previous action
        else:
            actions[t] = random_actions[t - 1]  # Random action

    return {
        'stimuli': jnp.array(stimuli),
        'actions': jnp.array(actions),
        'rewards': jnp.array(rewards),
        'set_sizes': jnp.array(set_sizes)
    }


# ============================================================================
# SINGLE BLOCK BACKWARD COMPATIBILITY TESTS
# ============================================================================

class TestSingleBlockBackwardCompatibility:
    """Test that M3(kappa=0) matches M2 exactly for single blocks."""

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1011])
    def test_single_block_multiple_seeds(self, wmrl_params, seed):
        """M3 with kappa=0 matches M2 for different random seeds."""
        # Generate block data
        data = generate_block_data(seed, n_trials=40)

        # Compute M2 likelihood
        m2_log_lik = wmrl_block_likelihood(
            stimuli=data['stimuli'],
            actions=data['actions'],
            rewards=data['rewards'],
            set_sizes=data['set_sizes'],
            **wmrl_params
        )

        # Compute M3 likelihood with kappa=0
        m3_log_lik = wmrl_m3_block_likelihood(
            stimuli=data['stimuli'],
            actions=data['actions'],
            rewards=data['rewards'],
            set_sizes=data['set_sizes'],
            kappa=0.0,
            **wmrl_params
        )

        # Should be numerically identical
        assert jnp.allclose(m2_log_lik, m3_log_lik, rtol=1e-5, atol=1e-8), \
            f"Seed {seed}: M2={m2_log_lik:.6f}, M3(κ=0)={m3_log_lik:.6f}, diff={abs(m2_log_lik - m3_log_lik):.2e}"

    @pytest.mark.parametrize("params", [
        pytest.param(p, id=f"params_{i}") for i, p in enumerate([
            {'alpha_pos': 0.3, 'alpha_neg': 0.1, 'phi': 0.2, 'rho': 0.7, 'capacity': 4.0, 'epsilon': 0.05},
            {'alpha_pos': 0.9, 'alpha_neg': 0.8, 'phi': 0.5, 'rho': 0.9, 'capacity': 6.0, 'epsilon': 0.01},
            {'alpha_pos': 0.1, 'alpha_neg': 0.05, 'phi': 0.05, 'rho': 0.3, 'capacity': 2.0, 'epsilon': 0.1},
            {'alpha_pos': 0.8, 'alpha_neg': 0.1, 'phi': 0.3, 'rho': 0.5, 'capacity': 3.0, 'epsilon': 0.02}
        ])
    ])
    def test_single_block_parameter_variations(self, params):
        """M3 with kappa=0 matches M2 across different parameter combinations."""
        # Generate block data
        data = generate_block_data(seed=42, n_trials=50)

        # Compute M2 likelihood
        m2_log_lik = wmrl_block_likelihood(
            stimuli=data['stimuli'],
            actions=data['actions'],
            rewards=data['rewards'],
            set_sizes=data['set_sizes'],
            **params
        )

        # Compute M3 likelihood with kappa=0
        m3_log_lik = wmrl_m3_block_likelihood(
            stimuli=data['stimuli'],
            actions=data['actions'],
            rewards=data['rewards'],
            set_sizes=data['set_sizes'],
            kappa=0.0,
            **params
        )

        # Should be numerically identical
        assert jnp.allclose(m2_log_lik, m3_log_lik, rtol=1e-5, atol=1e-8), \
            f"M2={m2_log_lik:.6f}, M3(κ=0)={m3_log_lik:.6f}, diff={abs(m2_log_lik - m3_log_lik):.2e}"

    def test_single_block_long_sequence(self, wmrl_params):
        """M3 with kappa=0 matches M2 for longer trial sequences."""
        # Generate longer block (100 trials)
        data = generate_block_data(seed=999, n_trials=100)

        # Compute M2 likelihood
        m2_log_lik = wmrl_block_likelihood(
            stimuli=data['stimuli'],
            actions=data['actions'],
            rewards=data['rewards'],
            set_sizes=data['set_sizes'],
            **wmrl_params
        )

        # Compute M3 likelihood with kappa=0
        m3_log_lik = wmrl_m3_block_likelihood(
            stimuli=data['stimuli'],
            actions=data['actions'],
            rewards=data['rewards'],
            set_sizes=data['set_sizes'],
            kappa=0.0,
            **wmrl_params
        )

        # Should be numerically identical even with 100 trials
        assert jnp.allclose(m2_log_lik, m3_log_lik, rtol=1e-5, atol=1e-8), \
            f"M2={m2_log_lik:.6f}, M3(κ=0)={m3_log_lik:.6f}, diff={abs(m2_log_lik - m3_log_lik):.2e}"


# ============================================================================
# MULTI-BLOCK BACKWARD COMPATIBILITY TESTS
# ============================================================================

class TestMultiBlockBackwardCompatibility:
    """Test that M3(kappa=0) matches M2 exactly for multiple blocks."""

    def test_three_blocks(self, wmrl_params):
        """M3 with kappa=0 matches M2 for 3 blocks."""
        # Generate 3-block data
        blocks = generate_multiblock_data(seed=100, n_blocks=3, n_trials_per_block=40)

        # Compute M2 likelihood
        m2_log_lik = wmrl_multiblock_likelihood(
            stimuli_blocks=blocks['stimuli_blocks'],
            actions_blocks=blocks['actions_blocks'],
            rewards_blocks=blocks['rewards_blocks'],
            set_sizes_blocks=blocks['set_sizes_blocks'],
            **wmrl_params
        )

        # Compute M3 likelihood with kappa=0
        m3_log_lik = wmrl_m3_multiblock_likelihood(
            stimuli_blocks=blocks['stimuli_blocks'],
            actions_blocks=blocks['actions_blocks'],
            rewards_blocks=blocks['rewards_blocks'],
            set_sizes_blocks=blocks['set_sizes_blocks'],
            kappa=0.0,
            **wmrl_params
        )

        # Should be numerically identical
        assert jnp.allclose(m2_log_lik, m3_log_lik, rtol=1e-5, atol=1e-8), \
            f"3 blocks: M2={m2_log_lik:.6f}, M3(κ=0)={m3_log_lik:.6f}, diff={abs(m2_log_lik - m3_log_lik):.2e}"

    def test_realistic_23_blocks(self, wmrl_params):
        """M3 with kappa=0 matches M2 for realistic 23-block experiment."""
        # Generate 23-block data (realistic experiment length)
        blocks = generate_multiblock_data(seed=200, n_blocks=23, n_trials_per_block=45)

        # Compute M2 likelihood
        m2_log_lik = wmrl_multiblock_likelihood(
            stimuli_blocks=blocks['stimuli_blocks'],
            actions_blocks=blocks['actions_blocks'],
            rewards_blocks=blocks['rewards_blocks'],
            set_sizes_blocks=blocks['set_sizes_blocks'],
            **wmrl_params
        )

        # Compute M3 likelihood with kappa=0
        m3_log_lik = wmrl_m3_multiblock_likelihood(
            stimuli_blocks=blocks['stimuli_blocks'],
            actions_blocks=blocks['actions_blocks'],
            rewards_blocks=blocks['rewards_blocks'],
            set_sizes_blocks=blocks['set_sizes_blocks'],
            kappa=0.0,
            **wmrl_params
        )

        # Should be numerically identical even across 23 blocks
        assert jnp.allclose(m2_log_lik, m3_log_lik, rtol=1e-5, atol=1e-8), \
            f"23 blocks: M2={m2_log_lik:.6f}, M3(κ=0)={m3_log_lik:.6f}, diff={abs(m2_log_lik - m3_log_lik):.2e}"

    @pytest.mark.parametrize("n_blocks", [1, 5, 10, 21])
    def test_varying_block_counts(self, wmrl_params, n_blocks):
        """M3 with kappa=0 matches M2 for different numbers of blocks."""
        # Generate variable-block data
        blocks = generate_multiblock_data(seed=300, n_blocks=n_blocks, n_trials_per_block=35)

        # Compute M2 likelihood
        m2_log_lik = wmrl_multiblock_likelihood(
            stimuli_blocks=blocks['stimuli_blocks'],
            actions_blocks=blocks['actions_blocks'],
            rewards_blocks=blocks['rewards_blocks'],
            set_sizes_blocks=blocks['set_sizes_blocks'],
            **wmrl_params
        )

        # Compute M3 likelihood with kappa=0
        m3_log_lik = wmrl_m3_multiblock_likelihood(
            stimuli_blocks=blocks['stimuli_blocks'],
            actions_blocks=blocks['actions_blocks'],
            rewards_blocks=blocks['rewards_blocks'],
            set_sizes_blocks=blocks['set_sizes_blocks'],
            kappa=0.0,
            **wmrl_params
        )

        # Should be numerically identical
        assert jnp.allclose(m2_log_lik, m3_log_lik, rtol=1e-5, atol=1e-8), \
            f"{n_blocks} blocks: M2={m2_log_lik:.6f}, M3(κ=0)={m3_log_lik:.6f}, diff={abs(m2_log_lik - m3_log_lik):.2e}"


# ============================================================================
# KAPPA EFFECT SANITY CHECKS
# ============================================================================

class TestKappaEffect:
    """Sanity checks that kappa parameter has the intended effect."""

    def test_kappa_nonzero_differs_from_zero(self, wmrl_params):
        """M3 with kappa > 0 produces different likelihood than kappa = 0."""
        # Generate block data
        data = generate_block_data(seed=42, n_trials=40)

        # Compute M3 with kappa=0
        m3_kappa0 = wmrl_m3_block_likelihood(
            stimuli=data['stimuli'],
            actions=data['actions'],
            rewards=data['rewards'],
            set_sizes=data['set_sizes'],
            kappa=0.0,
            **wmrl_params
        )

        # Compute M3 with kappa=0.5
        m3_kappa05 = wmrl_m3_block_likelihood(
            stimuli=data['stimuli'],
            actions=data['actions'],
            rewards=data['rewards'],
            set_sizes=data['set_sizes'],
            kappa=0.5,
            **wmrl_params
        )

        # Should be different
        assert not jnp.allclose(m3_kappa0, m3_kappa05, rtol=1e-3), \
            f"κ=0 and κ=0.5 should differ! Got κ=0: {m3_kappa0:.6f}, κ=0.5: {m3_kappa05:.6f}"

    def test_high_kappa_increases_likelihood_for_repetitive_actions(self, wmrl_params):
        """High kappa increases likelihood when actions are repetitive."""
        # Generate data with high action repetition
        data = generate_repetitive_actions(seed=42, n_trials=50, repetition_prob=0.8)

        # Compute M3 with kappa=0 (no perseveration bonus)
        m3_kappa0 = wmrl_m3_block_likelihood(
            stimuli=data['stimuli'],
            actions=data['actions'],
            rewards=data['rewards'],
            set_sizes=data['set_sizes'],
            kappa=0.0,
            **wmrl_params
        )

        # Compute M3 with kappa=0.7 (high perseveration)
        m3_kappa07 = wmrl_m3_block_likelihood(
            stimuli=data['stimuli'],
            actions=data['actions'],
            rewards=data['rewards'],
            set_sizes=data['set_sizes'],
            kappa=0.7,
            **wmrl_params
        )

        # High kappa should increase likelihood for repetitive actions
        # (because it increases probability of observed repeated actions)
        assert m3_kappa07 > m3_kappa0, \
            f"High κ should increase likelihood for repetitive actions! κ=0: {m3_kappa0:.6f}, κ=0.7: {m3_kappa07:.6f}"

    @pytest.mark.parametrize("kappa", [0.0, 0.2, 0.5, 0.8, 1.0])
    def test_kappa_range_produces_valid_likelihoods(self, wmrl_params, kappa):
        """All kappa values in [0, 1] produce valid (finite, non-NaN) likelihoods."""
        # Generate block data
        data = generate_block_data(seed=42, n_trials=40)

        # Compute M3 likelihood
        m3_log_lik = wmrl_m3_block_likelihood(
            stimuli=data['stimuli'],
            actions=data['actions'],
            rewards=data['rewards'],
            set_sizes=data['set_sizes'],
            kappa=kappa,
            **wmrl_params
        )

        # Should be finite and not NaN
        assert jnp.isfinite(m3_log_lik), f"κ={kappa} produced non-finite likelihood: {m3_log_lik}"
        assert not jnp.isnan(m3_log_lik), f"κ={kappa} produced NaN likelihood"

    def test_kappa_monotonic_with_repetition(self, wmrl_params):
        """Higher kappa monotonically increases likelihood for highly repetitive actions."""
        # Generate very repetitive actions
        data = generate_repetitive_actions(seed=555, n_trials=50, repetition_prob=0.9)

        # Test increasing kappa values
        kappa_values = [0.0, 0.2, 0.4, 0.6, 0.8]
        likelihoods = []

        for kappa in kappa_values:
            log_lik = wmrl_m3_block_likelihood(
                stimuli=data['stimuli'],
                actions=data['actions'],
                rewards=data['rewards'],
                set_sizes=data['set_sizes'],
                kappa=kappa,
                **wmrl_params
            )
            likelihoods.append(float(log_lik))

        # Check monotonic increase
        for i in range(len(likelihoods) - 1):
            assert likelihoods[i+1] > likelihoods[i], \
                f"Likelihood not monotonic: κ={kappa_values[i]}: {likelihoods[i]:.4f}, κ={kappa_values[i+1]}: {likelihoods[i+1]:.4f}"
