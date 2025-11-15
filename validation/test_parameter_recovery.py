"""
Test Parameter Recovery

Verifies that fitting procedures can recover known parameters
from synthetic data. This is critical for validating the model
fitting pipeline.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.q_learning import QLearningAgent
from models.wm_rl_hybrid import WMRLHybridAgent
from environments.rlwm_env import create_rlwm_env


class TestQLearningParameterRecovery:
    """Test parameter recovery for Q-learning model."""

    @pytest.mark.slow
    def test_recovery_with_scipy_mle(self):
        """
        Recover Q-learning parameters using scipy MLE.

        This is faster than PyMC for testing purposes.
        """
        from scipy.optimize import minimize

        # Ground truth parameters
        true_alpha = 0.3
        true_beta = 4.0

        # Generate synthetic data
        env = create_rlwm_env(set_size=3, seed=42)
        agent = QLearningAgent(
            alpha=true_alpha,
            beta=true_beta,
            num_stimuli=6,
            num_actions=3,
            seed=42
        )

        data = self._generate_synthetic_data(env, agent, n_trials=100)

        # Fit using MLE
        def neg_loglik(params):
            alpha, beta = params
            test_agent = QLearningAgent(
                alpha=alpha,
                beta=beta,
                num_stimuli=6,
                num_actions=3
            )

            loglik = 0
            for _, trial in data.iterrows():
                probs = test_agent.get_action_probs(trial['stimulus'])
                loglik += np.log(probs[trial['key_press']] + 1e-10)
                test_agent.update(
                    trial['stimulus'],
                    trial['key_press'],
                    trial['correct']
                )

            return -loglik

        result = minimize(
            neg_loglik,
            x0=[0.1, 2.0],
            bounds=[(0.01, 0.99), (0.1, 20)],
            method='L-BFGS-B'
        )

        recovered_alpha, recovered_beta = result.x

        # Check recovery (allow 20% error for 100 trials)
        alpha_error = abs(recovered_alpha - true_alpha) / true_alpha
        beta_error = abs(recovered_beta - true_beta) / true_beta

        assert alpha_error < 0.2, f"Alpha recovery error {alpha_error:.2%} > 20%"
        assert beta_error < 0.25, f"Beta recovery error {beta_error:.2%} > 25%"

        print(f"\nParameter Recovery Results:")
        print(f"  True α: {true_alpha:.3f}, Recovered: {recovered_alpha:.3f}")
        print(f"  True β: {true_beta:.3f}, Recovered: {recovered_beta:.3f}")

    @pytest.mark.slow
    def test_recovery_multiple_datasets(self):
        """
        Test recovery across multiple synthetic datasets.

        Ensures recovery is robust across different random seeds.
        """
        from scipy.optimize import minimize

        true_alpha = 0.3
        true_beta = 3.0

        recoveries = []

        for seed in range(5):  # 5 different datasets
            env = create_rlwm_env(set_size=3, seed=seed)
            agent = QLearningAgent(
                alpha=true_alpha,
                beta=true_beta,
                num_stimuli=6,
                num_actions=3,
                seed=seed
            )

            data = self._generate_synthetic_data(env, agent, n_trials=150)

            # Fit
            def neg_loglik(params):
                alpha, beta = params
                test_agent = QLearningAgent(
                    alpha=alpha, beta=beta,
                    num_stimuli=6, num_actions=3
                )

                loglik = 0
                for _, trial in data.iterrows():
                    probs = test_agent.get_action_probs(trial['stimulus'])
                    loglik += np.log(probs[trial['key_press']] + 1e-10)
                    test_agent.update(trial['stimulus'], trial['key_press'], trial['correct'])

                return -loglik

            result = minimize(
                neg_loglik,
                x0=[0.15, 2.5],
                bounds=[(0.01, 0.99), (0.1, 20)],
                method='L-BFGS-B'
            )

            recoveries.append(result.x)

        recoveries = np.array(recoveries)

        # Mean recovered values should be close to true
        mean_alpha = recoveries[:, 0].mean()
        mean_beta = recoveries[:, 1].mean()

        assert abs(mean_alpha - true_alpha) < 0.1
        assert abs(mean_beta - true_beta) < 1.0

    @staticmethod
    def _generate_synthetic_data(env, agent, n_trials):
        """
        Generate synthetic trial data.

        Parameters
        ----------
        env : RLWMEnv
            Environment
        agent : QLearningAgent
            Agent
        n_trials : int
            Number of trials

        Returns
        -------
        pd.DataFrame
            Synthetic data
        """
        obs, _ = env.reset()
        data = []

        for trial in range(n_trials):
            stimulus = obs['stimulus']
            action = agent.choose_action(stimulus)
            obs, reward, terminated, truncated, info = env.step(action)

            data.append({
                'stimulus': stimulus,
                'key_press': action,
                'correct': info['is_correct']
            })

            agent.update(stimulus, action, reward)

            if terminated or truncated:
                break

        return pd.DataFrame(data)


class TestWMRLParameterRecovery:
    """Test parameter recovery for WM-RL model."""

    @pytest.mark.slow
    def test_recovery_with_fixed_capacity(self):
        """
        Recover WM-RL parameters with capacity fixed to true value.

        Simplifies optimization by fixing capacity.
        """
        from scipy.optimize import minimize

        # Ground truth
        true_alpha = 0.2
        true_beta = 3.0
        true_capacity = 4
        true_w_wm = 0.6

        # Generate data
        env = create_rlwm_env(set_size=4, seed=42)
        agent = WMRLHybridAgent(
            alpha=true_alpha,
            beta=true_beta,
            capacity=true_capacity,
            lambda_decay=0.1,
            w_wm=true_w_wm,
            num_stimuli=6,
            num_actions=3,
            seed=42
        )

        data = self._generate_synthetic_data(env, agent, n_trials=150)

        # Fit (fix capacity, lambda_decay)
        def neg_loglik(params):
            alpha, beta, w_wm = params
            test_agent = WMRLHybridAgent(
                alpha=alpha,
                beta=beta,
                capacity=true_capacity,  # Fixed
                lambda_decay=0.1,  # Fixed
                w_wm=w_wm,
                num_stimuli=6,
                num_actions=3
            )

            loglik = 0
            for _, trial in data.iterrows():
                hybrid_info = test_agent.get_hybrid_probs(trial['stimulus'])
                probs = hybrid_info['probs']
                loglik += np.log(probs[trial['key_press']] + 1e-10)
                test_agent.update(trial['stimulus'], trial['key_press'], trial['correct'])

            return -loglik

        result = minimize(
            neg_loglik,
            x0=[0.15, 2.5, 0.5],
            bounds=[(0.01, 0.99), (0.1, 20), (0.0, 1.0)],
            method='L-BFGS-B'
        )

        recovered_alpha, recovered_beta, recovered_w_wm = result.x

        # Check recovery (allow more error for complex model)
        assert abs(recovered_alpha - true_alpha) < 0.15
        assert abs(recovered_beta - true_beta) < 1.5
        assert abs(recovered_w_wm - true_w_wm) < 0.3

        print(f"\nWM-RL Parameter Recovery:")
        print(f"  True α: {true_alpha:.3f}, Recovered: {recovered_alpha:.3f}")
        print(f"  True β: {true_beta:.3f}, Recovered: {recovered_beta:.3f}")
        print(f"  True w_wm: {true_w_wm:.3f}, Recovered: {recovered_w_wm:.3f}")

    @staticmethod
    def _generate_synthetic_data(env, agent, n_trials):
        """Generate synthetic data from WM-RL agent."""
        obs, _ = env.reset()
        data = []

        for trial in range(n_trials):
            stimulus = obs['stimulus']
            action, _ = agent.choose_action(stimulus)
            obs, reward, terminated, truncated, info = env.step(action)

            data.append({
                'stimulus': stimulus,
                'key_press': action,
                'correct': info['is_correct']
            })

            agent.update(stimulus, action, reward)

            if terminated or truncated:
                break

        return pd.DataFrame(data)


class TestRecoveryDiagnostics:
    """Test parameter recovery diagnostics and edge cases."""

    def test_identifiability_beta_with_uniform_q(self):
        """
        Beta is not identifiable when Q-values are uniform.

        This tests our understanding of model identifiability.
        """
        from scipy.optimize import minimize

        # Create data where agent never learns (alpha=0)
        env = create_rlwm_env(set_size=2, seed=42)
        agent = QLearningAgent(
            alpha=0.0,  # No learning
            beta=5.0,  # Should be unidentifiable
            num_stimuli=6,
            num_actions=3,
            seed=42
        )

        obs, _ = env.reset()
        data = []

        for _ in range(50):
            stimulus = obs['stimulus']
            action = agent.choose_action(stimulus)
            obs, reward, terminated, truncated, info = env.step(action)

            data.append({
                'stimulus': stimulus,
                'key_press': action,
                'correct': info['is_correct']
            })

            agent.update(stimulus, action, reward)

            if terminated or truncated:
                break

        data = pd.DataFrame(data)

        # Try to fit - beta should be poorly constrained
        def neg_loglik(params):
            alpha, beta = params
            test_agent = QLearningAgent(
                alpha=alpha, beta=beta,
                num_stimuli=6, num_actions=3
            )

            loglik = 0
            for _, trial in data.iterrows():
                probs = test_agent.get_action_probs(trial['stimulus'])
                loglik += np.log(probs[trial['key_press']] + 1e-10)
                test_agent.update(trial['stimulus'], trial['key_press'], trial['correct'])

            return -loglik

        result = minimize(
            neg_loglik,
            x0=[0.1, 2.0],
            bounds=[(0.001, 0.99), (0.1, 20)],
            method='L-BFGS-B'
        )

        recovered_alpha = result.x[0]

        # Alpha should be close to 0 (well identified)
        assert recovered_alpha < 0.1, "Alpha should be near 0 when no learning"

        # Note: Beta is poorly identified in this case (expected)
        print(f"\nIdentifiability test:")
        print(f"  Recovered α: {recovered_alpha:.4f} (true: 0.0)")
        print(f"  Recovered β: {result.x[1]:.2f} (true: 5.0, poorly identified)")
