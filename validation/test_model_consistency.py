"""
Test Model Consistency

Ensures that agent classes produce identical, deterministic results
across different contexts (simulation, fitting, testing).
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.q_learning import QLearningAgent
from models.wm_rl_hybrid import WMRLHybridAgent


class TestQLearningConsistency:
    """Test Q-learning agent consistency and determinism."""

    def test_deterministic_behavior(self, sample_trial_data, sample_agent_params):
        """Agent with same params and seed produces identical results."""
        params = sample_agent_params['qlearning']

        # Run 1
        agent1 = QLearningAgent(
            **params,
            num_stimuli=6,
            num_actions=3,
            seed=42
        )
        probs1, q_values1 = self._run_agent(agent1, sample_trial_data)

        # Run 2 (same seed)
        agent2 = QLearningAgent(
            **params,
            num_stimuli=6,
            num_actions=3,
            seed=42
        )
        probs2, q_values2 = self._run_agent(agent2, sample_trial_data)

        # Should be exactly identical
        np.testing.assert_allclose(probs1, probs2, rtol=1e-10)
        np.testing.assert_allclose(q_values1, q_values2, rtol=1e-10)

    def test_reset_behavior(self, sample_trial_data, sample_agent_params):
        """Agent reset produces identical starting state."""
        params = sample_agent_params['qlearning']
        agent = QLearningAgent(
            **params,
            num_stimuli=6,
            num_actions=3,
            seed=42
        )

        # Run once
        probs1, q1 = self._run_agent(agent, sample_trial_data)

        # Reset and run again (same seed preserved)
        agent.reset()
        agent.rng = np.random.RandomState(42)  # Reset random state too
        probs2, q2 = self._run_agent(agent, sample_trial_data)

        # Should be identical
        np.testing.assert_allclose(probs1, probs2, rtol=1e-10)

    def test_parameter_effects_alpha(self, sample_trial_data):
        """Different learning rates produce different behavior."""
        data = sample_trial_data

        # High learning rates (both positive and negative)
        agent_high = QLearningAgent(
            alpha_pos=0.9, alpha_neg=0.9, beta=3.0,
            num_stimuli=6, num_actions=3,
            seed=42
        )
        probs_high, q_high = self._run_agent(agent_high, data)

        # Low learning rates (both positive and negative)
        agent_low = QLearningAgent(
            alpha_pos=0.1, alpha_neg=0.1, beta=3.0,
            num_stimuli=6, num_actions=3,
            seed=42
        )
        probs_low, q_low = self._run_agent(agent_low, data)

        # Q-values should differ significantly
        assert not np.allclose(q_high, q_low, rtol=0.1)

    def test_asymmetric_learning(self, sample_trial_data):
        """Asymmetric learning rates affect behavior differently."""
        data = sample_trial_data

        # High positive, low negative (optimistic learner)
        agent_optimistic = QLearningAgent(
            alpha_pos=0.9, alpha_neg=0.1, beta=3.0,
            num_stimuli=6, num_actions=3,
            seed=42
        )
        _, q_optimistic = self._run_agent(agent_optimistic, data)

        # Low positive, high negative (pessimistic learner)
        agent_pessimistic = QLearningAgent(
            alpha_pos=0.1, alpha_neg=0.9, beta=3.0,
            num_stimuli=6, num_actions=3,
            seed=42
        )
        _, q_pessimistic = self._run_agent(agent_pessimistic, data)

        # Q-values should differ
        assert not np.allclose(q_optimistic, q_pessimistic, rtol=0.1)

    def test_parameter_effects_beta(self, sample_trial_data):
        """Different inverse temperatures affect exploration."""
        data = sample_trial_data

        # High beta (exploitation)
        agent_exploit = QLearningAgent(
            alpha_pos=0.3, alpha_neg=0.1, beta=10.0,
            num_stimuli=6, num_actions=3,
            seed=42
        )
        probs_exploit, _ = self._run_agent(agent_exploit, data)

        # Low beta (exploration)
        agent_explore = QLearningAgent(
            alpha_pos=0.3, alpha_neg=0.1, beta=0.5,
            num_stimuli=6, num_actions=3,
            seed=42
        )
        probs_explore, _ = self._run_agent(agent_explore, data)

        # Action probabilities should differ
        assert not np.allclose(probs_exploit, probs_explore, rtol=0.1)

        # High beta should have more peaked distributions
        entropy_exploit = -np.sum(probs_exploit * np.log(probs_exploit + 1e-10), axis=1).mean()
        entropy_explore = -np.sum(probs_explore * np.log(probs_explore + 1e-10), axis=1).mean()
        assert entropy_exploit < entropy_explore

    def test_q_value_bounds(self, sample_trial_data, sample_agent_params):
        """Q-values stay within reasonable bounds."""
        params = sample_agent_params['qlearning']
        agent = QLearningAgent(
            **params,
            num_stimuli=6,
            num_actions=3,
            q_init=0.5
        )

        _, q_values = self._run_agent(agent, sample_trial_data)

        # With rewards in [0, 1] and gamma=0, Q-values should be in [0, 1]
        assert np.all(q_values >= 0)
        assert np.all(q_values <= 1)

    def test_action_prob_validity(self, sample_trial_data, sample_agent_params):
        """Action probabilities are valid probability distributions."""
        params = sample_agent_params['qlearning']
        agent = QLearningAgent(
            **params,
            num_stimuli=6,
            num_actions=3
        )

        probs, _ = self._run_agent(agent, sample_trial_data)

        # All probabilities non-negative
        assert np.all(probs >= 0)

        # Each row sums to 1
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-6)

    @staticmethod
    def _run_agent(agent, trial_data):
        """
        Helper to run agent through trials.

        Parameters
        ----------
        agent : QLearningAgent
            Agent to run
        trial_data : dict
            Dictionary with 'stimuli', 'actions', 'rewards'

        Returns
        -------
        probs : np.ndarray
            Action probabilities for each trial
        q_table : np.ndarray
            Final Q-table
        """
        probs_list = []

        for s, a, r in zip(trial_data['stimuli'],
                          trial_data['actions'],
                          trial_data['rewards']):
            probs = agent.get_action_probs(s)
            probs_list.append(probs.copy())
            agent.update(s, a, r)

        return np.array(probs_list), agent.get_q_table()


class TestWMRLConsistency:
    """Test WM-RL hybrid agent consistency and determinism."""

    def test_deterministic_behavior(self, sample_trial_data, sample_agent_params):
        """WM-RL agent with same params produces identical results."""
        params = sample_agent_params['wmrl']

        agent1 = WMRLHybridAgent(
            **params,
            num_stimuli=6,
            num_actions=3,
            seed=42
        )
        probs1 = self._run_agent(agent1, sample_trial_data)

        agent2 = WMRLHybridAgent(
            **params,
            num_stimuli=6,
            num_actions=3,
            seed=42
        )
        probs2 = self._run_agent(agent2, sample_trial_data)

        np.testing.assert_allclose(probs1, probs2, rtol=1e-10)

    def test_wm_capacity_effects(self, sample_trial_data):
        """Different WM capacities affect behavior (via adaptive weighting)."""
        data = sample_trial_data
        set_size = 3  # Test set size

        # High capacity (omega will be higher)
        agent_high = WMRLHybridAgent(
            alpha_pos=0.3, alpha_neg=0.1, beta=2.0, beta_wm=3.0,
            capacity=7, phi=0.1, rho=0.7,
            num_stimuli=6, num_actions=3, seed=42
        )
        probs_high = self._run_agent(agent_high, data, set_size)

        # Low capacity (omega will be lower)
        agent_low = WMRLHybridAgent(
            alpha_pos=0.3, alpha_neg=0.1, beta=2.0, beta_wm=3.0,
            capacity=2, phi=0.1, rho=0.7,
            num_stimuli=6, num_actions=3, seed=42
        )
        probs_low = self._run_agent(agent_low, data, set_size)

        # Should differ due to adaptive weighting
        assert not np.allclose(probs_high, probs_low, rtol=0.1)

    def test_wm_weight_effects(self):
        """Base WM reliance (rho) affects reliance on WM vs RL."""
        data = {
            'stimuli': np.array([0, 0, 0]),
            'actions': np.array([0, 0, 0]),
            'rewards': np.array([1.0, 1.0, 1.0])
        }
        set_size = 3

        # Low WM reliance (rho = 0.1)
        agent_low_rho = WMRLHybridAgent(
            alpha_pos=0.5, alpha_neg=0.2, beta=2.0, beta_wm=3.0,
            capacity=4, phi=0.1, rho=0.1,
            num_stimuli=6, num_actions=3, seed=42
        )
        probs_low_rho = self._run_agent(agent_low_rho, data, set_size)

        # High WM reliance (rho = 0.9)
        agent_high_rho = WMRLHybridAgent(
            alpha_pos=0.5, alpha_neg=0.2, beta=2.0, beta_wm=3.0,
            capacity=4, phi=0.1, rho=0.9,
            num_stimuli=6, num_actions=3, seed=42
        )
        probs_high_rho = self._run_agent(agent_high_rho, data, set_size)

        # Should differ (high rho should rely more on WM one-shot learning)
        assert not np.allclose(probs_low_rho, probs_high_rho, rtol=0.1)

    def test_wm_matrix_structure(self, sample_trial_data, sample_agent_params):
        """WM matrix maintains correct structure (matrix-based architecture)."""
        params = sample_agent_params['wmrl']
        set_size = 3

        agent = WMRLHybridAgent(
            **params,
            num_stimuli=6,
            num_actions=3
        )

        # Run trials
        for s, a, r in zip(sample_trial_data['stimuli'],
                          sample_trial_data['actions'],
                          sample_trial_data['rewards']):
            agent.get_hybrid_probs(s, set_size)
            agent.update(s, a, r)

        # WM should be a matrix of shape (num_stimuli, num_actions)
        wm_matrix = agent.get_wm_matrix()
        assert wm_matrix.shape == (6, 3)

        # All WM values should be in valid range [0, 1] for binary rewards
        assert np.all(wm_matrix >= 0)
        assert np.all(wm_matrix <= 1)

    def test_hybrid_probs_validity(self, sample_trial_data, sample_agent_params):
        """Hybrid probabilities are valid probability distributions."""
        params = sample_agent_params['wmrl']
        set_size = 3

        agent = WMRLHybridAgent(
            **params,
            num_stimuli=6,
            num_actions=3
        )

        probs = self._run_agent(agent, sample_trial_data, set_size)

        # All probabilities non-negative
        assert np.all(probs >= 0)

        # Each row sums to 1
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-6)

    @staticmethod
    def _run_agent(agent, trial_data, set_size=3):
        """
        Helper to run WM-RL agent through trials.

        Parameters
        ----------
        agent : WMRLHybridAgent
            Agent to run
        trial_data : dict
            Trial data
        set_size : int
            Set size for adaptive weighting (default: 3)

        Returns
        -------
        np.ndarray
            Hybrid action probabilities for each trial
        """
        probs_list = []

        for s, a, r in zip(trial_data['stimuli'],
                          trial_data['actions'],
                          trial_data['rewards']):
            hybrid_info = agent.get_hybrid_probs(s, set_size)
            probs_list.append(hybrid_info['probs'].copy())
            agent.update(s, a, r)

        return np.array(probs_list)


class TestCrossModelComparison:
    """Test comparisons between Q-learning and WM-RL models."""

    def test_wmrl_with_zero_rho_approaches_qlearning(self):
        """WM-RL with rho=0 should behave more like Q-learning (though not identical)."""
        data = {
            'stimuli': np.array([0, 1, 2, 0, 1]),
            'actions': np.array([0, 1, 2, 1, 0]),
            'rewards': np.array([1.0, 0.0, 1.0, 1.0, 0.0])
        }
        set_size = 3

        # Q-learning
        agent_q = QLearningAgent(
            alpha_pos=0.3, alpha_neg=0.3, beta=3.0,
            num_stimuli=6, num_actions=3,
            seed=42
        )
        probs_q = []
        for s, a, r in zip(data['stimuli'], data['actions'], data['rewards']):
            probs_q.append(agent_q.get_action_probs(s))
            agent_q.update(s, a, r)
        probs_q = np.array(probs_q)

        # WM-RL with rho≈0 (minimal WM influence, omega ≈ 0)
        agent_wmrl = WMRLHybridAgent(
            alpha_pos=0.3, alpha_neg=0.3, beta=3.0, beta_wm=3.0,
            capacity=4, phi=0.1, rho=0.01,  # Very low rho
            num_stimuli=6, num_actions=3,
            seed=42
        )
        probs_wmrl = []
        for s, a, r in zip(data['stimuli'], data['actions'], data['rewards']):
            hybrid_info = agent_wmrl.get_hybrid_probs(s, set_size)
            probs_wmrl.append(hybrid_info['probs'])
            agent_wmrl.update(s, a, r)
        probs_wmrl = np.array(probs_wmrl)

        # Should be similar but not identical (due to WM baseline and decay)
        # Check that correlation is high
        correlation = np.corrcoef(probs_q.flatten(), probs_wmrl.flatten())[0, 1]
        assert correlation > 0.95  # High correlation when rho is near zero
