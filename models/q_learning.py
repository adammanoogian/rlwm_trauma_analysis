"""
Q-Learning Model for RLWM Task

Standard model-free reinforcement learning agent that learns stimulus-response
mappings through trial-and-error with temporal difference learning.

Model Equations
---------------
Q-value update:
    Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',·) - Q(s,a)]

Action selection (softmax):
    P(a|s) = exp(β·Q(s,a)) / Σ_a' exp(β·Q(s,a'))

Parameters
----------
α (alpha) : learning rate (0-1)
    How much new information overrides old information
β (beta) : inverse temperature (>0)
    Controls exploration vs exploitation (higher = more exploitation)
γ (gamma) : discount factor (0-1)
    How much future rewards are valued (often 0 for immediate rewards)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import TaskParams, ModelParams


class QLearningAgent:
    """
    Q-Learning agent for the RLWM task.

    Learns stimulus-response mappings using temporal difference learning.
    Maintains a Q-table Q[stimulus, action] representing expected rewards.
    """

    def __init__(
        self,
        num_stimuli: int = TaskParams.MAX_STIMULI,
        num_actions: int = TaskParams.NUM_ACTIONS,
        alpha: float = ModelParams.ALPHA_DEFAULT,
        beta: float = ModelParams.BETA_DEFAULT,
        gamma: float = ModelParams.GAMMA_DEFAULT,
        q_init: float = ModelParams.Q_INIT_VALUE,
        seed: Optional[int] = None,
    ):
        """
        Initialize Q-learning agent.

        Parameters
        ----------
        num_stimuli : int
            Number of possible stimuli
        num_actions : int
            Number of possible actions
        alpha : float
            Learning rate (0-1)
        beta : float
            Inverse temperature for softmax (>0)
        gamma : float
            Discount factor (0-1)
        q_init : float
            Initial Q-values (optimistic initialization if > 0)
        seed : int, optional
            Random seed for reproducibility
        """
        self.num_stimuli = num_stimuli
        self.num_actions = num_actions
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.q_init = q_init

        # Random state
        self.rng = np.random.RandomState(seed)

        # Initialize Q-table
        self.Q = np.full((num_stimuli, num_actions), q_init, dtype=np.float64)

        # History tracking
        self.history = {
            'stimuli': [],
            'actions': [],
            'rewards': [],
            'q_values': [],
            'action_probs': [],
        }

    def reset(self, q_init: Optional[float] = None):
        """
        Reset the Q-table and history.

        Parameters
        ----------
        q_init : float, optional
            Initial Q-values. If None, uses self.q_init.
        """
        if q_init is not None:
            self.q_init = q_init

        self.Q.fill(self.q_init)

        # Clear history
        self.history = {
            'stimuli': [],
            'actions': [],
            'rewards': [],
            'q_values': [],
            'action_probs': [],
        }

    def get_action_probs(self, stimulus: int) -> np.ndarray:
        """
        Compute action probabilities using softmax.

        P(a|s) = exp(β·Q(s,a)) / Σ_a' exp(β·Q(s,a'))

        Parameters
        ----------
        stimulus : int
            Current stimulus index

        Returns
        -------
        np.ndarray
            Probability distribution over actions
        """
        q_vals = self.Q[stimulus, :]

        # Numerical stability: subtract max before exp
        q_vals_scaled = self.beta * (q_vals - np.max(q_vals))
        exp_q = np.exp(q_vals_scaled)
        probs = exp_q / np.sum(exp_q)

        return probs

    def choose_action(self, stimulus: int, return_probs: bool = False) -> int:
        """
        Choose an action based on softmax policy.

        Parameters
        ----------
        stimulus : int
            Current stimulus index
        return_probs : bool
            If True, also return action probabilities

        Returns
        -------
        action : int
            Chosen action
        probs : np.ndarray (optional)
            Action probabilities (if return_probs=True)
        """
        probs = self.get_action_probs(stimulus)
        action = self.rng.choice(self.num_actions, p=probs)

        if return_probs:
            return action, probs
        return action

    def update(
        self,
        stimulus: int,
        action: int,
        reward: float,
        next_stimulus: Optional[int] = None,
    ):
        """
        Update Q-value based on observed reward.

        Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',·) - Q(s,a)]

        For episodic tasks or when γ=0, simplifies to:
        Q(s,a) ← Q(s,a) + α[r - Q(s,a)]

        Parameters
        ----------
        stimulus : int
            Current stimulus
        action : int
            Action taken
        reward : float
            Reward received
        next_stimulus : int, optional
            Next stimulus (for γ > 0). If None and γ > 0, assumes terminal state.
        """
        # Current Q-value
        q_current = self.Q[stimulus, action]

        # TD target
        if self.gamma > 0 and next_stimulus is not None:
            # Value of next state (max over actions)
            v_next = np.max(self.Q[next_stimulus, :])
            td_target = reward + self.gamma * v_next
        else:
            # No bootstrapping (immediate reward only)
            td_target = reward

        # TD error
        td_error = td_target - q_current

        # Q-value update
        self.Q[stimulus, action] += self.alpha * td_error

    def predict_action_probs(self, stimulus: int) -> np.ndarray:
        """
        Predict action probabilities for a given stimulus.

        Same as get_action_probs, but more explicit for external use.

        Parameters
        ----------
        stimulus : int
            Stimulus index

        Returns
        -------
        np.ndarray
            Probability distribution over actions
        """
        return self.get_action_probs(stimulus)

    def get_max_q_action(self, stimulus: int) -> int:
        """
        Get the action with highest Q-value (greedy action).

        Parameters
        ----------
        stimulus : int
            Stimulus index

        Returns
        -------
        int
            Action with highest Q-value
        """
        return int(np.argmax(self.Q[stimulus, :]))

    def log_trial(
        self,
        stimulus: int,
        action: int,
        reward: float,
        action_probs: Optional[np.ndarray] = None
    ):
        """
        Log trial data for analysis.

        Parameters
        ----------
        stimulus : int
            Stimulus presented
        action : int
            Action taken
        reward : float
            Reward received
        action_probs : np.ndarray, optional
            Action probabilities at time of choice
        """
        self.history['stimuli'].append(stimulus)
        self.history['actions'].append(action)
        self.history['rewards'].append(reward)
        self.history['q_values'].append(self.Q[stimulus, :].copy())

        if action_probs is not None:
            self.history['action_probs'].append(action_probs)
        else:
            self.history['action_probs'].append(self.get_action_probs(stimulus))

    def get_history(self) -> Dict[str, List]:
        """
        Get logged trial history.

        Returns
        -------
        dict
            Dictionary containing trial history
        """
        return self.history

    def get_q_table(self) -> np.ndarray:
        """
        Get current Q-table.

        Returns
        -------
        np.ndarray
            Q-table of shape (num_stimuli, num_actions)
        """
        return self.Q.copy()

    def set_parameters(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None
    ):
        """
        Set model parameters.

        Parameters
        ----------
        alpha : float, optional
            Learning rate
        beta : float, optional
            Inverse temperature
        gamma : float, optional
            Discount factor
        """
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma

    def get_parameters(self) -> Dict[str, float]:
        """
        Get current model parameters.

        Returns
        -------
        dict
            Dictionary of parameters
        """
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'q_init': self.q_init,
        }


# ============================================================================
# AGENT FACTORY FUNCTION
# ============================================================================

def create_q_learning_agent(
    alpha: float = ModelParams.ALPHA_DEFAULT,
    beta: float = ModelParams.BETA_DEFAULT,
    gamma: float = ModelParams.GAMMA_DEFAULT,
    seed: Optional[int] = None,
    **kwargs
) -> QLearningAgent:
    """
    Factory function to create Q-learning agent with common configurations.

    Parameters
    ----------
    alpha : float
        Learning rate
    beta : float
        Inverse temperature
    gamma : float
        Discount factor
    seed : int, optional
        Random seed
    **kwargs
        Additional arguments passed to QLearningAgent

    Returns
    -------
    QLearningAgent
        Configured agent
    """
    return QLearningAgent(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        seed=seed,
        **kwargs
    )


# ============================================================================
# SIMULATION UTILITIES
# ============================================================================

def simulate_agent_on_env(
    agent: QLearningAgent,
    env,
    num_trials: int = 100,
    log_history: bool = True
) -> Dict[str, any]:
    """
    Simulate an agent interacting with the environment.

    Parameters
    ----------
    agent : QLearningAgent
        Agent to simulate
    env : RLWMEnv
        Environment to interact with
    num_trials : int
        Number of trials to simulate
    log_history : bool
        Whether to log trial history in agent

    Returns
    -------
    dict
        Simulation results including rewards, actions, accuracy
    """
    # Reset environment and agent
    obs, info = env.reset()
    agent.reset()

    rewards = []
    actions = []
    correct = []
    stimuli = []

    for trial in range(num_trials):
        # Get current stimulus
        stimulus = obs['stimulus']
        stimuli.append(stimulus)

        # Choose action
        action, probs = agent.choose_action(stimulus, return_probs=True)
        actions.append(action)

        # Take step in environment
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        correct.append(info['is_correct'])

        # Update agent
        next_stimulus = obs['stimulus'] if not (terminated or truncated) else None
        agent.update(stimulus, action, reward, next_stimulus)

        # Log if requested
        if log_history:
            agent.log_trial(stimulus, action, reward, probs)

        # Check if done
        if terminated or truncated:
            break

    return {
        'stimuli': stimuli,
        'actions': actions,
        'rewards': rewards,
        'correct': correct,
        'accuracy': np.mean(correct),
        'total_reward': np.sum(rewards),
        'num_trials': len(correct),
    }


def test_q_learning():
    """Test Q-learning agent on RLWM environment."""
    print("Testing Q-Learning Agent")
    print("=" * 80)

    # Import environment
    from environments.rlwm_env import create_rlwm_env

    # Create environment and agent
    env = create_rlwm_env(set_size=3, phase_type='main_task', seed=42)
    agent = create_q_learning_agent(alpha=0.3, beta=3.0, gamma=0.0, seed=42)

    print(f"\nAgent parameters: {agent.get_parameters()}")

    # Run simulation
    print("\nRunning 100-trial simulation...")
    results = simulate_agent_on_env(agent, env, num_trials=100, log_history=True)

    print(f"\nResults:")
    print(f"  Accuracy: {results['accuracy']:.3f}")
    print(f"  Total reward: {results['total_reward']:.0f}")
    print(f"  Trials completed: {results['num_trials']}")

    # Show final Q-table
    print(f"\nFinal Q-table (first 3 stimuli):")
    print(agent.get_q_table()[:3, :])

    # Compute learning curve (moving average)
    window = 10
    correct_ma = np.convolve(
        results['correct'],
        np.ones(window) / window,
        mode='valid'
    )
    print(f"\nLearning curve (moving average, window={window}):")
    print(f"  First {window} trials: {correct_ma[0]:.3f}")
    print(f"  Last {window} trials: {correct_ma[-1]:.3f}")
    print(f"  Improvement: {correct_ma[-1] - correct_ma[0]:.3f}")

    print("\n" + "=" * 80)
    print("Test complete!")


if __name__ == "__main__":
    test_q_learning()
