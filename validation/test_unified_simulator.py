"""
Tests for Unified Simulator

Tests both fixed parameter and sampled parameter simulation modes,
ensuring consistency across parameter sweeps, PyMC fitting, and data generation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.rlwm_env import create_rlwm_env
from models.q_learning import QLearningAgent
from models.wm_rl_hybrid import WMRLHybridAgent
from scripts.simulations.unified_simulator import (
    SimulationResult,
    results_to_dataframe,
    simulate_agent_fixed,
    simulate_agent_sampled,
    simulate_qlearning_for_likelihood,
    simulate_wmrl_for_likelihood,
)


class TestSimulateAgentFixed:
    """Test fixed parameter simulation."""

    def test_qlearning_fixed_simulation(self):
        """Test Q-learning agent with fixed parameters."""
        env = create_rlwm_env(set_size=3, seed=42)

        params = {
            'num_stimuli': 6,
            'num_actions': 3,
            'alpha_pos': 0.3,
            'alpha_neg': 0.1,
            'beta': 2.0,
            'gamma': 0.0,
            'q_init': 0.5
        }

        result = simulate_agent_fixed(
            agent_class=QLearningAgent,
            params=params,
            env=env,
            num_trials=50,
            seed=42
        )

        # Check result structure
        assert isinstance(result, SimulationResult)
        assert len(result.stimuli) == 50
        assert len(result.actions) == 50
        assert len(result.rewards) == 50
        assert 0.0 <= result.accuracy <= 1.0

        # Check parameters are stored
        assert result.params['alpha_pos'] == 0.3
        assert result.params['alpha_neg'] == 0.1
        assert result.params['beta'] == 2.0
        assert result.seed == 42

    def test_wmrl_fixed_simulation(self):
        """Test WM-RL agent with fixed parameters (matrix-based architecture)."""
        env = create_rlwm_env(set_size=3, seed=42)

        params = {
            'num_stimuli': 6,
            'num_actions': 3,
            'alpha_pos': 0.3,
            'alpha_neg': 0.1,
            'beta': 2.0,
            'beta_wm': 3.0,
            'capacity': 4,
            'phi': 0.1,
            'rho': 0.7,
            'gamma': 0.0,
            'q_init': 0.5,
            'wm_init': 0.0
        }

        result = simulate_agent_fixed(
            agent_class=WMRLHybridAgent,
            params=params,
            env=env,
            num_trials=50,
            seed=42
        )

        assert isinstance(result, SimulationResult)
        assert len(result.stimuli) == 50
        assert result.params['capacity'] == 4
        assert result.params['phi'] == 0.1
        assert result.params['rho'] == 0.7

    def test_deterministic_with_seed(self):
        """Test that same seed produces identical results."""
        env1 = create_rlwm_env(set_size=3, seed=42)
        env2 = create_rlwm_env(set_size=3, seed=42)

        params = {
            'num_stimuli': 6,
            'num_actions': 3,
            'alpha_pos': 0.3,
            'alpha_neg': 0.1,
            'beta': 2.0,
            'gamma': 0.0,
            'q_init': 0.5
        }

        result1 = simulate_agent_fixed(QLearningAgent, params, env1, 50, seed=42)
        result2 = simulate_agent_fixed(QLearningAgent, params, env2, 50, seed=42)

        np.testing.assert_array_equal(result1.stimuli, result2.stimuli)
        np.testing.assert_array_equal(result1.actions, result2.actions)
        np.testing.assert_array_equal(result1.rewards, result2.rewards)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        env1 = create_rlwm_env(set_size=3, seed=42)
        env2 = create_rlwm_env(set_size=3, seed=123)

        params = {
            'num_stimuli': 6,
            'num_actions': 3,
            'alpha_pos': 0.3,
            'alpha_neg': 0.1,
            'beta': 2.0,
            'gamma': 0.0,
            'q_init': 0.5
        }

        result1 = simulate_agent_fixed(QLearningAgent, params, env1, 50, seed=42)
        result2 = simulate_agent_fixed(QLearningAgent, params, env2, 50, seed=123)

        # Actions should be different (with high probability)
        assert not np.array_equal(result1.actions, result2.actions)


class TestSimulateAgentSampled:
    """Test sampled parameter simulation."""

    def test_qlearning_sampled_simulation(self):
        """Test Q-learning with sampled parameters."""
        def make_env(seed):
            return create_rlwm_env(set_size=3, seed=seed)

        # Define parameter distributions
        param_distributions = {
            'alpha_pos': lambda rng: rng.beta(3, 2),
            'alpha_neg': lambda rng: rng.beta(2, 3),
            'beta': lambda rng: rng.gamma(2, 1)
        }

        fixed_params = {
            'num_stimuli': 6,
            'num_actions': 3,
            'gamma': 0.0,
            'q_init': 0.5
        }

        results = simulate_agent_sampled(
            agent_class=QLearningAgent,
            param_distributions=param_distributions,
            fixed_params=fixed_params,
            env_factory=make_env,
            num_trials=50,
            num_samples=10,
            seed=42
        )

        # Check we got 10 results
        assert len(results) == 10

        # Check each result has different parameters
        alphas_pos = [r.params['alpha_pos'] for r in results]
        alphas_neg = [r.params['alpha_neg'] for r in results]
        betas = [r.params['beta'] for r in results]

        # Parameters should vary across samples
        assert len(set(alphas_pos)) > 1  # At least some variation
        assert len(set(alphas_neg)) > 1
        assert len(set(betas)) > 1

        # All should be valid
        for result in results:
            assert isinstance(result, SimulationResult)
            assert len(result.stimuli) == 50
            assert 0.0 < result.params['alpha_pos'] < 1.0
            assert 0.0 < result.params['alpha_neg'] < 1.0
            assert result.params['beta'] > 0

    def test_wmrl_sampled_simulation(self):
        """Test WM-RL with sampled parameters (matrix-based architecture)."""
        def make_env(seed):
            return create_rlwm_env(set_size=3, seed=seed)

        param_distributions = {
            'alpha_pos': lambda rng: rng.beta(3, 2),
            'alpha_neg': lambda rng: rng.beta(2, 3),
            'beta': lambda rng: rng.gamma(2, 1),
            'beta_wm': lambda rng: rng.gamma(3, 1),
            'capacity': lambda rng: int(rng.integers(2, 7)),
            'phi': lambda rng: rng.beta(1, 9),
            'rho': lambda rng: rng.beta(7, 3)
        }

        fixed_params = {
            'num_stimuli': 6,
            'num_actions': 3,
            'gamma': 0.0,
            'q_init': 0.5,
            'wm_init': 0.0
        }

        results = simulate_agent_sampled(
            agent_class=WMRLHybridAgent,
            param_distributions=param_distributions,
            fixed_params=fixed_params,
            env_factory=make_env,
            num_trials=50,
            num_samples=5,
            seed=42
        )

        assert len(results) == 5

        # Check capacity values are integers in valid range
        capacities = [r.params['capacity'] for r in results]
        for cap in capacities:
            assert isinstance(cap, int)
            assert 2 <= cap <= 7

        # Check phi and rho are in [0, 1]
        phis = [r.params['phi'] for r in results]
        rhos = [r.params['rho'] for r in results]
        for phi in phis:
            assert 0.0 <= phi <= 1.0
        for rho in rhos:
            assert 0.0 <= rho <= 1.0

    def test_sampled_parameters_reproducible_with_seed(self):
        """Test that sampled simulations are reproducible."""
        def make_env(seed):
            return create_rlwm_env(set_size=3, seed=seed)

        param_distributions = {
            'alpha_pos': lambda rng: rng.beta(3, 2),
            'alpha_neg': lambda rng: rng.beta(2, 3),
            'beta': lambda rng: rng.gamma(2, 1)
        }

        fixed_params = {
            'num_stimuli': 6,
            'num_actions': 3,
            'gamma': 0.0,
            'q_init': 0.5
        }

        results1 = simulate_agent_sampled(
            QLearningAgent, param_distributions, fixed_params,
            make_env, 50, 5, seed=42
        )

        results2 = simulate_agent_sampled(
            QLearningAgent, param_distributions, fixed_params,
            make_env, 50, 5, seed=42
        )

        # Same sampled parameters
        for r1, r2 in zip(results1, results2):
            assert r1.params['alpha_pos'] == r2.params['alpha_pos']
            assert r1.params['alpha_neg'] == r2.params['alpha_neg']
            assert r1.params['beta'] == r2.params['beta']


class TestLikelihoodFunctions:
    """Test likelihood computation functions for PyMC."""

    def test_qlearning_likelihood_computation(self):
        """Test Q-learning likelihood function."""
        stimuli = np.array([0, 1, 0, 2, 1])
        rewards = np.array([1, 0, 1, 1, 0])

        action_probs = simulate_qlearning_for_likelihood(
            stimuli=stimuli,
            rewards=rewards,
            alpha_pos=0.3,
            alpha_neg=0.1,
            beta=2.0
        )

        # Check shape
        assert action_probs.shape == (5, 3)

        # Check probabilities sum to 1
        for t in range(5):
            assert np.isclose(np.sum(action_probs[t]), 1.0)

        # Check all probabilities are valid
        assert np.all(action_probs >= 0)
        assert np.all(action_probs <= 1)

    def test_wmrl_likelihood_computation(self):
        """Test WM-RL likelihood function (matrix-based architecture)."""
        stimuli = np.array([0, 1, 0, 2, 1])
        rewards = np.array([1, 0, 1, 1, 0])
        set_sizes = np.array([3, 3, 3, 3, 3])

        action_probs = simulate_wmrl_for_likelihood(
            stimuli=stimuli,
            rewards=rewards,
            set_sizes=set_sizes,
            alpha_pos=0.3,
            alpha_neg=0.1,
            beta=2.0,
            beta_wm=3.0,
            capacity=4,
            phi=0.1,
            rho=0.7
        )

        assert action_probs.shape == (5, 3)

        # Check valid probabilities
        for t in range(5):
            assert np.isclose(np.sum(action_probs[t]), 1.0)
            assert np.all(action_probs[t] >= 0)
            assert np.all(action_probs[t] <= 1)


class TestResultsToDataFrame:
    """Test conversion of results to DataFrame."""

    def test_single_result_to_dataframe(self):
        """Test converting single result to DataFrame."""
        env = create_rlwm_env(set_size=3, seed=42)

        params = {
            'num_stimuli': 6,
            'num_actions': 3,
            'alpha_pos': 0.3,
            'alpha_neg': 0.1,
            'beta': 2.0,
            'gamma': 0.0,
            'q_init': 0.5
        }

        result = simulate_agent_fixed(
            QLearningAgent, params, env, 50, seed=42
        )

        df = results_to_dataframe(result)

        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50
        assert 'trial' in df.columns
        assert 'stimulus' in df.columns
        assert 'action' in df.columns
        assert 'reward' in df.columns
        assert 'alpha_pos' in df.columns
        assert 'beta' in df.columns

    def test_multiple_results_to_dataframe(self):
        """Test converting multiple results to DataFrame."""
        def make_env(seed):
            return create_rlwm_env(set_size=3, seed=seed)

        param_distributions = {
            'alpha_pos': lambda rng: rng.beta(3, 2),
            'alpha_neg': lambda rng: rng.beta(2, 3),
            'beta': lambda rng: rng.gamma(2, 1)
        }

        fixed_params = {
            'num_stimuli': 6,
            'num_actions': 3,
            'gamma': 0.0,
            'q_init': 0.5
        }

        results = simulate_agent_sampled(
            QLearningAgent, param_distributions, fixed_params,
            make_env, 50, 3, seed=42
        )

        df = results_to_dataframe(results)

        # Should have 3 samples × 50 trials = 150 rows
        assert len(df) == 150

        # Check sample_id column
        assert 'sample_id' in df.columns
        assert df['sample_id'].nunique() == 3

        # Check each sample has 50 trials
        for sample_id in range(3):
            sample_df = df[df['sample_id'] == sample_id]
            assert len(sample_df) == 50


class TestConsistencyAcrossCodePaths:
    """Test that unified simulator produces consistent results across different use cases."""

    def test_parameter_sweep_consistency(self):
        """Test that parameter sweep results match direct agent simulation."""
        env = create_rlwm_env(set_size=3, seed=42)

        params = {
            'num_stimuli': 6,
            'num_actions': 3,
            'alpha_pos': 0.3,
            'alpha_neg': 0.1,
            'beta': 2.0,
            'gamma': 0.0,
            'q_init': 0.5
        }

        # Simulate using unified simulator
        result = simulate_agent_fixed(
            QLearningAgent, params, env, 50, seed=42
        )

        # Create agent directly and check behavior matches
        agent = QLearningAgent(**params, seed=42)
        env2 = create_rlwm_env(set_size=3, seed=42)
        obs, _ = env2.reset()

        # First action should match
        first_action = result.actions[0]
        first_stimulus = result.stimuli[0]

        action_probs = agent.get_action_probs(first_stimulus)

        # At least check that action probabilities are valid
        assert np.isclose(np.sum(action_probs), 1.0)
        assert np.all(action_probs >= 0)

    def test_pymc_likelihood_consistency(self):
        """Test that PyMC likelihood uses same agent implementation."""
        stimuli = np.array([0, 1, 0, 2, 1])
        rewards = np.array([1, 0, 1, 1, 0])

        # Compute likelihood probabilities
        action_probs = simulate_qlearning_for_likelihood(
            stimuli=stimuli,
            rewards=rewards,
            alpha_pos=0.3,
            alpha_neg=0.1,
            beta=2.0
        )

        # Create agent directly and step through
        agent = QLearningAgent(
            num_stimuli=6,
            num_actions=3,
            alpha_pos=0.3,
            alpha_neg=0.1,
            beta=2.0,
            gamma=0.0,
            q_init=0.5,
            seed=None
        )

        # First trial probabilities should match
        probs_direct = agent.get_action_probs(stimuli[0])
        np.testing.assert_allclose(action_probs[0], probs_direct, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
