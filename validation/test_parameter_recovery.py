"""
Test Parameter Recovery

Verifies that fitting procedures can recover known parameters
from synthetic data. This is critical for validating the model
fitting pipeline.

Updated to use current asymmetric learning rate API (alpha_pos, alpha_neg).
For these recovery tests we use symmetric rates (alpha_pos == alpha_neg)
to test single-alpha recovery.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rlwm.envs.rlwm_env import create_rlwm_env
from rlwm.models.q_learning import QLearningAgent
from rlwm.models.wm_rl_hybrid import WMRLHybridAgent


class TestQLearningParameterRecovery:
    """Test parameter recovery for Q-learning model."""

    @pytest.mark.slow
    def test_recovery_with_scipy_mle(self):
        """
        Recover Q-learning parameters using scipy MLE.

        This is faster than PyMC for testing purposes.
        """
        from scipy.optimize import minimize

        # Ground truth parameters (symmetric alpha for simplicity)
        true_alpha = 0.3
        true_beta = 4.0

        # Generate synthetic data
        env = create_rlwm_env(set_size=3, seed=42)
        agent = QLearningAgent(
            alpha_pos=true_alpha,
            alpha_neg=true_alpha,
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
                alpha_pos=alpha,
                alpha_neg=alpha,
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

        # Check recovery (allow 50% error for 100 trials with symmetric alpha)
        alpha_error = abs(recovered_alpha - true_alpha) / true_alpha
        beta_error = abs(recovered_beta - true_beta) / true_beta

        assert alpha_error < 0.5, f"Alpha recovery error {alpha_error:.2%} > 50%"
        assert beta_error < 0.5, f"Beta recovery error {beta_error:.2%} > 50%"

        print("\nParameter Recovery Results:")
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
                alpha_pos=true_alpha,
                alpha_neg=true_alpha,
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
                    alpha_pos=alpha, alpha_neg=alpha, beta=beta,
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
        """Generate synthetic trial data."""
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

        # Ground truth (using current parameter names)
        true_alpha = 0.2
        true_beta = 3.0
        true_capacity = 4
        true_rho = 0.6

        # Generate data
        env = create_rlwm_env(set_size=4, seed=42)
        agent = WMRLHybridAgent(
            alpha_pos=true_alpha,
            alpha_neg=true_alpha,
            beta=true_beta,
            capacity=true_capacity,
            phi=0.1,
            rho=true_rho,
            num_stimuli=6,
            num_actions=3,
            seed=42
        )

        data = self._generate_synthetic_data(env, agent, set_size=4, n_trials=150)

        # Fit (fix capacity, phi)
        def neg_loglik(params):
            alpha, beta, rho = params
            test_agent = WMRLHybridAgent(
                alpha_pos=alpha,
                alpha_neg=alpha,
                beta=beta,
                capacity=true_capacity,  # Fixed
                phi=0.1,  # Fixed
                rho=rho,
                num_stimuli=6,
                num_actions=3
            )

            loglik = 0
            for _, trial in data.iterrows():
                hybrid_info = test_agent.get_hybrid_probs(
                    trial['stimulus'], trial['set_size']
                )
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

        recovered_alpha, recovered_beta, recovered_rho = result.x

        # Check recovery (allow generous tolerance for complex model with 150 trials)
        assert abs(recovered_alpha - true_alpha) < 0.3
        assert abs(recovered_beta - true_beta) < 3.0
        assert abs(recovered_rho - true_rho) < 0.4

        print("\nWM-RL Parameter Recovery:")
        print(f"  True α: {true_alpha:.3f}, Recovered: {recovered_alpha:.3f}")
        print(f"  True β: {true_beta:.3f}, Recovered: {recovered_beta:.3f}")
        print(f"  True ρ: {true_rho:.3f}, Recovered: {recovered_rho:.3f}")

    @staticmethod
    def _generate_synthetic_data(env, agent, set_size, n_trials):
        """Generate synthetic data from WM-RL agent."""
        obs, _ = env.reset()
        data = []

        for trial in range(n_trials):
            stimulus = obs['stimulus']
            action, _ = agent.choose_action(stimulus, set_size)
            obs, reward, terminated, truncated, info = env.step(action)

            data.append({
                'stimulus': stimulus,
                'key_press': action,
                'correct': info['is_correct'],
                'set_size': set_size,
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

        When alpha=0, Q-values stay at q_init so the agent always chooses
        uniformly — beta and alpha are both non-identifiable. We verify
        that the negative log-likelihood surface is flat (multiple optima
        give similar loss), confirming non-identifiability.
        """
        from scipy.optimize import minimize

        # Create data where agent never learns (alpha=0)
        env = create_rlwm_env(set_size=2, seed=42)
        agent = QLearningAgent(
            alpha_pos=0.0,
            alpha_neg=0.0,
            beta=5.0,
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

        def neg_loglik(params):
            alpha, beta = params
            test_agent = QLearningAgent(
                alpha_pos=alpha, alpha_neg=alpha, beta=beta,
                num_stimuli=6, num_actions=3
            )

            loglik = 0
            for _, trial in data.iterrows():
                probs = test_agent.get_action_probs(trial['stimulus'])
                loglik += np.log(probs[trial['key_press']] + 1e-10)
                test_agent.update(trial['stimulus'], trial['key_press'], trial['correct'])

            return -loglik

        # Fit from two very different starting points
        result1 = minimize(
            neg_loglik, x0=[0.01, 1.0],
            bounds=[(0.001, 0.99), (0.1, 20)], method='L-BFGS-B'
        )
        result2 = minimize(
            neg_loglik, x0=[0.5, 10.0],
            bounds=[(0.001, 0.99), (0.1, 20)], method='L-BFGS-B'
        )

        # Both should achieve similar log-likelihood (flat surface = non-identifiable)
        nll_diff = abs(result1.fun - result2.fun)
        assert nll_diff < 2.0, (
            f"NLL difference {nll_diff:.2f} too large — "
            f"surface should be flat when data is near-random"
        )

        print("\nIdentifiability test:")
        print(f"  Fit 1: α={result1.x[0]:.4f}, β={result1.x[1]:.2f}, NLL={result1.fun:.2f}")
        print(f"  Fit 2: α={result2.x[0]:.4f}, β={result2.x[1]:.2f}, NLL={result2.fun:.2f}")
        print(f"  NLL difference: {nll_diff:.4f} (confirms non-identifiability)")
