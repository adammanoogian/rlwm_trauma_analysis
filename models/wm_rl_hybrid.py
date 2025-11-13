"""
Working Memory + RL Hybrid Model for RLWM Task

Combines episodic working memory with model-free reinforcement learning.
The model maintains:
1. Working memory buffer storing recent (stimulus, action, reward) experiences
2. Q-learning system for gradual learning
3. Hybrid decision system weighting WM retrieval vs RL

This model captures set-size effects: performance degrades when WM capacity
is exceeded, forcing reliance on slower RL learning.

Model Components
----------------
Working Memory:
- Capacity: K items (typically 2-7)
- Storage: Recent (stimulus, action, reward) tuples
- Decay: Older memories fade with decay rate λ
- Retrieval: If stimulus in WM, retrieve last action-reward

Q-Learning:
- Same as standard Q-learning model
- Q(s,a) ← Q(s,a) + α[r - Q(s,a)]

Hybrid Decision:
- P(a|s) = w_wm * P_wm(a|s) + (1 - w_wm) * P_rl(a|s)
- where w_wm ∈ [0,1] is the WM weight

Parameters
----------
α (alpha) : learning rate for Q-learning (0-1)
β (beta) : inverse temperature for softmax (>0)
γ (gamma) : discount factor (0-1)
K (capacity) : WM capacity (1-7 items)
λ (lambda_decay) : memory decay rate (0-1)
w_wm : weight for WM vs RL (0-1)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from collections import deque
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import TaskParams, ModelParams


class WMMemoryItem:
    """Single working memory item storing stimulus-action-reward association."""

    def __init__(self, stimulus: int, action: int, reward: float, age: int = 0):
        """
        Initialize memory item.

        Parameters
        ----------
        stimulus : int
            Stimulus ID
        action : int
            Action taken
        reward : float
            Reward received
        age : int
            Age of memory (in trials)
        """
        self.stimulus = stimulus
        self.action = action
        self.reward = reward
        self.age = age

    def get_strength(self, lambda_decay: float) -> float:
        """
        Compute memory strength based on age and decay rate.

        strength = exp(-λ * age)

        Parameters
        ----------
        lambda_decay : float
            Decay rate

        Returns
        -------
        float
            Memory strength (0-1)
        """
        return np.exp(-lambda_decay * self.age)


class WMRLHybridAgent:
    """
    Working Memory + RL Hybrid agent for RLWM task.

    Combines episodic WM with incremental Q-learning for robust learning
    across different set sizes.
    """

    def __init__(
        self,
        num_stimuli: int = TaskParams.MAX_STIMULI,
        num_actions: int = TaskParams.NUM_ACTIONS,
        alpha: float = ModelParams.ALPHA_DEFAULT,
        beta: float = ModelParams.BETA_DEFAULT,
        gamma: float = ModelParams.GAMMA_DEFAULT,
        capacity: int = ModelParams.WM_CAPACITY_DEFAULT,
        lambda_decay: float = ModelParams.LAMBDA_DECAY_DEFAULT,
        w_wm: float = ModelParams.W_WM_DEFAULT,
        q_init: float = ModelParams.Q_INIT_VALUE,
        seed: Optional[int] = None,
    ):
        """
        Initialize WM-RL hybrid agent.

        Parameters
        ----------
        num_stimuli : int
            Number of possible stimuli
        num_actions : int
            Number of possible actions
        alpha : float
            Learning rate for Q-learning (0-1)
        beta : float
            Inverse temperature for softmax (>0)
        gamma : float
            Discount factor (0-1)
        capacity : int
            Working memory capacity (items)
        lambda_decay : float
            Memory decay rate (0-1)
        w_wm : float
            Weight for WM vs RL (0-1)
        q_init : float
            Initial Q-values
        seed : int, optional
            Random seed
        """
        self.num_stimuli = num_stimuli
        self.num_actions = num_actions
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.capacity = capacity
        self.lambda_decay = lambda_decay
        self.w_wm = w_wm
        self.q_init = q_init

        # Random state
        self.rng = np.random.RandomState(seed)

        # Q-learning component
        self.Q = np.full((num_stimuli, num_actions), q_init, dtype=np.float64)

        # Working memory buffer (FIFO with capacity limit)
        self.wm_buffer: deque = deque(maxlen=capacity)

        # Trial counter
        self.trial_count = 0

        # History tracking
        self.history = {
            'stimuli': [],
            'actions': [],
            'rewards': [],
            'q_values': [],
            'wm_probs': [],
            'rl_probs': [],
            'hybrid_probs': [],
            'wm_retrieved': [],
            'wm_strength': [],
        }

    def reset(self, q_init: Optional[float] = None):
        """
        Reset the agent (Q-table, WM, history).

        Parameters
        ----------
        q_init : float, optional
            Initial Q-values
        """
        if q_init is not None:
            self.q_init = q_init

        self.Q.fill(self.q_init)
        self.wm_buffer.clear()
        self.trial_count = 0

        # Clear history
        self.history = {
            'stimuli': [],
            'actions': [],
            'rewards': [],
            'q_values': [],
            'wm_probs': [],
            'rl_probs': [],
            'hybrid_probs': [],
            'wm_retrieved': [],
            'wm_strength': [],
        }

    def get_rl_probs(self, stimulus: int) -> np.ndarray:
        """
        Get action probabilities from Q-learning component.

        P_rl(a|s) = exp(β·Q(s,a)) / Σ_a' exp(β·Q(s,a'))

        Parameters
        ----------
        stimulus : int
            Stimulus index

        Returns
        -------
        np.ndarray
            Probability distribution over actions
        """
        q_vals = self.Q[stimulus, :]

        # Numerical stability
        q_vals_scaled = self.beta * (q_vals - np.max(q_vals))
        exp_q = np.exp(q_vals_scaled)
        probs = exp_q / np.sum(exp_q)

        return probs

    def get_wm_probs(self, stimulus: int) -> Tuple[np.ndarray, bool, float]:
        """
        Get action probabilities from working memory retrieval.

        If stimulus found in WM, return action associated with most recent
        high-reward memory (weighted by strength). Otherwise, return uniform.

        Parameters
        ----------
        stimulus : int
            Stimulus index

        Returns
        -------
        probs : np.ndarray
            Probability distribution over actions
        retrieved : bool
            Whether a memory was successfully retrieved
        strength : float
            Strength of retrieved memory (0 if not retrieved)
        """
        # Age all memories
        for mem in self.wm_buffer:
            mem.age += 1

        # Find memories matching stimulus
        matching_memories = [mem for mem in self.wm_buffer if mem.stimulus == stimulus]

        if not matching_memories:
            # No memory found - uniform distribution
            probs = np.ones(self.num_actions) / self.num_actions
            return probs, False, 0.0

        # Get most recent memory with positive reward
        best_mem = None
        best_strength = 0.0

        for mem in matching_memories:
            strength = mem.get_strength(self.lambda_decay)
            if mem.reward > 0 and strength > best_strength:
                best_mem = mem
                best_strength = strength

        if best_mem is None:
            # No positive reward memory - use most recent
            best_mem = matching_memories[-1]
            best_strength = best_mem.get_strength(self.lambda_decay)

        # Create probability distribution (peaked on retrieved action)
        probs = np.ones(self.num_actions) * (1 - best_strength) / self.num_actions
        probs[best_mem.action] += best_strength

        # Normalize (just to be safe)
        probs /= np.sum(probs)

        return probs, True, best_strength

    def get_hybrid_probs(self, stimulus: int) -> Dict[str, Any]:
        """
        Get hybrid action probabilities combining WM and RL.

        P(a|s) = w_wm * P_wm(a|s) + (1 - w_wm) * P_rl(a|s)

        Parameters
        ----------
        stimulus : int
            Stimulus index

        Returns
        -------
        dict
            Dictionary containing:
            - 'probs': hybrid probability distribution
            - 'wm_probs': WM probabilities
            - 'rl_probs': RL probabilities
            - 'wm_retrieved': whether WM retrieval succeeded
            - 'wm_strength': strength of retrieved memory
        """
        # Get component probabilities
        rl_probs = self.get_rl_probs(stimulus)
        wm_probs, wm_retrieved, wm_strength = self.get_wm_probs(stimulus)

        # Hybrid combination
        hybrid_probs = self.w_wm * wm_probs + (1 - self.w_wm) * rl_probs

        # Normalize (numerical safety)
        hybrid_probs /= np.sum(hybrid_probs)

        return {
            'probs': hybrid_probs,
            'wm_probs': wm_probs,
            'rl_probs': rl_probs,
            'wm_retrieved': wm_retrieved,
            'wm_strength': wm_strength,
        }

    def choose_action(self, stimulus: int) -> Tuple[int, Dict[str, Any]]:
        """
        Choose action based on hybrid policy.

        Parameters
        ----------
        stimulus : int
            Current stimulus

        Returns
        -------
        action : int
            Chosen action
        info : dict
            Information about decision process
        """
        # Get hybrid probabilities
        hybrid_info = self.get_hybrid_probs(stimulus)
        probs = hybrid_info['probs']

        # Sample action
        action = self.rng.choice(self.num_actions, p=probs)

        # Package info
        info = {
            'action': action,
            'probs': probs,
            'wm_probs': hybrid_info['wm_probs'],
            'rl_probs': hybrid_info['rl_probs'],
            'wm_retrieved': hybrid_info['wm_retrieved'],
            'wm_strength': hybrid_info['wm_strength'],
        }

        return action, info

    def update(
        self,
        stimulus: int,
        action: int,
        reward: float,
        next_stimulus: Optional[int] = None,
    ):
        """
        Update both WM and Q-learning components.

        Parameters
        ----------
        stimulus : int
            Current stimulus
        action : int
            Action taken
        reward : float
            Reward received
        next_stimulus : int, optional
            Next stimulus (for Q-learning)
        """
        # Update Q-learning component
        q_current = self.Q[stimulus, action]

        if self.gamma > 0 and next_stimulus is not None:
            v_next = np.max(self.Q[next_stimulus, :])
            td_target = reward + self.gamma * v_next
        else:
            td_target = reward

        td_error = td_target - q_current
        self.Q[stimulus, action] += self.alpha * td_error

        # Update working memory
        # Store new memory (will automatically evict oldest if at capacity)
        new_memory = WMMemoryItem(stimulus, action, reward, age=0)
        self.wm_buffer.append(new_memory)

        # Increment trial counter
        self.trial_count += 1

    def log_trial(
        self,
        stimulus: int,
        action: int,
        reward: float,
        decision_info: Dict[str, Any]
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
        decision_info : dict
            Information from choose_action
        """
        self.history['stimuli'].append(stimulus)
        self.history['actions'].append(action)
        self.history['rewards'].append(reward)
        self.history['q_values'].append(self.Q[stimulus, :].copy())
        self.history['wm_probs'].append(decision_info['wm_probs'])
        self.history['rl_probs'].append(decision_info['rl_probs'])
        self.history['hybrid_probs'].append(decision_info['probs'])
        self.history['wm_retrieved'].append(decision_info['wm_retrieved'])
        self.history['wm_strength'].append(decision_info['wm_strength'])

    def get_history(self) -> Dict[str, List]:
        """Get logged trial history."""
        return self.history

    def get_q_table(self) -> np.ndarray:
        """Get current Q-table."""
        return self.Q.copy()

    def get_wm_contents(self) -> List[Dict[str, Any]]:
        """
        Get current working memory contents.

        Returns
        -------
        list
            List of memory items with stimulus, action, reward, age, strength
        """
        contents = []
        for mem in self.wm_buffer:
            contents.append({
                'stimulus': mem.stimulus,
                'action': mem.action,
                'reward': mem.reward,
                'age': mem.age,
                'strength': mem.get_strength(self.lambda_decay),
            })
        return contents

    def set_parameters(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        capacity: Optional[int] = None,
        lambda_decay: Optional[float] = None,
        w_wm: Optional[float] = None,
    ):
        """Set model parameters."""
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        if capacity is not None:
            self.capacity = capacity
            # Resize buffer
            new_buffer = deque(maxlen=capacity)
            new_buffer.extend(list(self.wm_buffer)[-capacity:])
            self.wm_buffer = new_buffer
        if lambda_decay is not None:
            self.lambda_decay = lambda_decay
        if w_wm is not None:
            self.w_wm = w_wm

    def get_parameters(self) -> Dict[str, float]:
        """Get current model parameters."""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'capacity': self.capacity,
            'lambda_decay': self.lambda_decay,
            'w_wm': self.w_wm,
            'q_init': self.q_init,
        }


# ============================================================================
# AGENT FACTORY FUNCTION
# ============================================================================

def create_wm_rl_agent(
    alpha: float = ModelParams.ALPHA_DEFAULT,
    beta: float = ModelParams.BETA_DEFAULT,
    capacity: int = ModelParams.WM_CAPACITY_DEFAULT,
    w_wm: float = ModelParams.W_WM_DEFAULT,
    seed: Optional[int] = None,
    **kwargs
) -> WMRLHybridAgent:
    """
    Factory function to create WM-RL hybrid agent.

    Parameters
    ----------
    alpha : float
        Learning rate
    beta : float
        Inverse temperature
    capacity : int
        WM capacity
    w_wm : float
        WM weight
    seed : int, optional
        Random seed
    **kwargs
        Additional arguments

    Returns
    -------
    WMRLHybridAgent
        Configured agent
    """
    return WMRLHybridAgent(
        alpha=alpha,
        beta=beta,
        capacity=capacity,
        w_wm=w_wm,
        seed=seed,
        **kwargs
    )


# ============================================================================
# SIMULATION UTILITIES
# ============================================================================

def simulate_wm_rl_on_env(
    agent: WMRLHybridAgent,
    env,
    num_trials: int = 100,
    log_history: bool = True
) -> Dict[str, Any]:
    """
    Simulate WM-RL agent interacting with environment.

    Parameters
    ----------
    agent : WMRLHybridAgent
        Agent to simulate
    env : RLWMEnv
        Environment
    num_trials : int
        Number of trials
    log_history : bool
        Whether to log history

    Returns
    -------
    dict
        Simulation results
    """
    # Reset
    obs, info = env.reset()
    agent.reset()

    rewards = []
    actions = []
    correct = []
    stimuli = []
    wm_retrieved = []

    for trial in range(num_trials):
        # Get stimulus
        stimulus = obs['stimulus']
        stimuli.append(stimulus)

        # Choose action
        action, decision_info = agent.choose_action(stimulus)
        actions.append(action)
        wm_retrieved.append(decision_info['wm_retrieved'])

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        correct.append(info['is_correct'])

        # Update agent
        next_stimulus = obs['stimulus'] if not (terminated or truncated) else None
        agent.update(stimulus, action, reward, next_stimulus)

        # Log
        if log_history:
            agent.log_trial(stimulus, action, reward, decision_info)

        if terminated or truncated:
            break

    return {
        'stimuli': stimuli,
        'actions': actions,
        'rewards': rewards,
        'correct': correct,
        'wm_retrieved': wm_retrieved,
        'accuracy': np.mean(correct),
        'wm_retrieval_rate': np.mean(wm_retrieved),
        'total_reward': np.sum(rewards),
        'num_trials': len(correct),
    }


def test_wm_rl_hybrid():
    """Test WM-RL hybrid agent."""
    print("Testing WM-RL Hybrid Agent")
    print("=" * 80)

    # Import environment
    from environments.rlwm_env import create_rlwm_env

    # Create environment and agent
    env = create_rlwm_env(set_size=5, phase_type='main_task', seed=42)
    agent = create_wm_rl_agent(
        alpha=0.2,
        beta=3.0,
        capacity=4,
        w_wm=0.6,
        lambda_decay=0.1,
        seed=42
    )

    print(f"\nAgent parameters: {agent.get_parameters()}")

    # Run simulation
    print("\nRunning 100-trial simulation (set size = 5)...")
    results = simulate_wm_rl_on_env(agent, env, num_trials=100, log_history=True)

    print(f"\nResults:")
    print(f"  Accuracy: {results['accuracy']:.3f}")
    print(f"  WM retrieval rate: {results['wm_retrieval_rate']:.3f}")
    print(f"  Total reward: {results['total_reward']:.0f}")

    # Show WM contents
    print(f"\nFinal WM contents (capacity={agent.capacity}):")
    for i, mem in enumerate(agent.get_wm_contents()):
        print(f"  [{i}] stim={mem['stimulus']}, action={mem['action']}, "
              f"reward={mem['reward']:.0f}, age={mem['age']}, "
              f"strength={mem['strength']:.3f}")

    # Learning curve
    window = 10
    correct_ma = np.convolve(
        results['correct'],
        np.ones(window) / window,
        mode='valid'
    )
    print(f"\nLearning curve (moving average, window={window}):")
    print(f"  First {window} trials: {correct_ma[0]:.3f}")
    print(f"  Last {window} trials: {correct_ma[-1]:.3f}")

    print("\n" + "=" * 80)
    print("Test complete!")


if __name__ == "__main__":
    test_wm_rl_hybrid()
