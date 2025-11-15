"""
Working Memory + RL Hybrid Model for RLWM Task

Matrix-based WM-RL model combining distributed working memory with
model-free reinforcement learning. The model maintains:
1. WM value matrix: immediate one-shot encoding with global decay
2. Q-learning system: gradual learning with asymmetric learning rates
3. Adaptive hybrid decision: capacity-based weighting of WM vs RL

Model Architecture
------------------
Working Memory (WM):
- Representation: State-action value matrix WM(s,a)
- Update: WM(s,a) ← r (α=1, immediate overwrite)
- Decay: WM ← (1-φ)WM + φ·WM_0 (global, every trial)
- Policy: p_WM(a|s) = softmax(β_WM · WM(s,:))

Q-Learning (RL):
- Representation: State-action value matrix Q(s,a)
- Update: δ = r - Q(s,a); α = α_pos if δ>0 else α_neg; Q(s,a) ← Q(s,a) + α·δ
- Policy: p_RL(a|s) = softmax(β · Q(s,:))

Hybrid Decision:
- Adaptive weight: ω = ρ * min(1, K/N_s)
- Policy: p(a|s) = ω·p_WM(a|s) + (1-ω)·p_RL(a|s)

Parameters
----------
α_pos (alpha_pos) : learning rate for positive PE in RL (0-1)
α_neg (alpha_neg) : learning rate for negative PE in RL (0-1)
β (beta) : inverse temperature for RL softmax (>0)
β_WM (beta_wm) : inverse temperature for WM softmax (>0)
γ (gamma) : discount factor, fixed at 0
φ (phi) : WM decay rate toward baseline (0-1)
ρ (rho) : base WM reliance parameter (0-1)
K (capacity) : WM capacity for adaptive weighting

Key Differences from Buffer-Based Model
----------------------------------------
1. WM is a matrix, not a FIFO buffer of discrete items
2. Capacity affects reliance (soft constraint), not storage (hard constraint)
3. Decay is global and trial-based, not age-based per item
4. Retrieval is direct lookup + softmax, not search + selection
5. Hybrid weight adapts to set size via ω = ρ * min(1, K/N_s)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import TaskParams, ModelParams


class WMRLHybridAgent:
    """
    Matrix-based Working Memory + RL Hybrid agent.

    Combines distributed WM value matrix with incremental Q-learning for
    robust learning across different set sizes.
    """

    def __init__(
        self,
        num_stimuli: int = TaskParams.MAX_STIMULI,
        num_actions: int = TaskParams.NUM_ACTIONS,
        alpha_pos: float = ModelParams.ALPHA_POS_DEFAULT,
        alpha_neg: float = ModelParams.ALPHA_NEG_DEFAULT,
        beta: float = ModelParams.BETA_DEFAULT,
        beta_wm: float = ModelParams.BETA_WM_DEFAULT,
        gamma: float = 0.0,
        capacity: int = ModelParams.WM_CAPACITY_DEFAULT,
        phi: float = ModelParams.PHI_DEFAULT,
        rho: float = ModelParams.RHO_DEFAULT,
        q_init: float = ModelParams.Q_INIT_VALUE,
        wm_init: float = ModelParams.WM_INIT_VALUE,
        seed: Optional[int] = None,
    ):
        """
        Initialize WM-RL hybrid agent with matrix-based architecture.

        Parameters
        ----------
        num_stimuli : int
            Number of possible stimuli
        num_actions : int
            Number of possible actions
        alpha_pos : float
            Learning rate for positive PE in RL (0-1)
        alpha_neg : float
            Learning rate for negative PE in RL (0-1)
        beta : float
            Inverse temperature for RL softmax (>0)
        beta_wm : float
            Inverse temperature for WM softmax (>0)
        gamma : float
            Discount factor (fixed at 0.0)
        capacity : int
            WM capacity for adaptive weighting (1-7)
        phi : float
            WM decay rate toward baseline (0-1)
        rho : float
            Base WM reliance parameter (0-1)
        q_init : float
            Initial Q-values
        wm_init : float
            Initial WM values (baseline)
        seed : int, optional
            Random seed
        """
        self.num_stimuli = num_stimuli
        self.num_actions = num_actions
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.beta = beta
        self.beta_wm = beta_wm
        self.gamma = gamma
        self.capacity = capacity
        self.phi = phi
        self.rho = rho
        self.q_init = q_init
        self.wm_init = wm_init

        # Random state
        self.rng = np.random.RandomState(seed)

        # RL component: Q-table
        self.Q = np.full((num_stimuli, num_actions), q_init, dtype=np.float64)

        # WM component: value matrix
        self.WM = np.full((num_stimuli, num_actions), wm_init, dtype=np.float64)
        self.WM_0 = np.full((num_stimuli, num_actions), wm_init, dtype=np.float64)

        # History tracking
        self.history = {
            'stimuli': [],
            'actions': [],
            'rewards': [],
            'q_values': [],
            'wm_values': [],
            'rl_probs': [],
            'wm_probs': [],
            'hybrid_probs': [],
            'omega': [],  # Adaptive weight
            'prediction_errors': [],
        }

    def reset(self, q_init: Optional[float] = None, wm_init: Optional[float] = None):
        """
        Reset the agent to initial state.

        Parameters
        ----------
        q_init : float, optional
            Initial Q-values. If None, uses self.q_init.
        wm_init : float, optional
            Initial WM values. If None, uses self.wm_init.
        """
        if q_init is not None:
            self.q_init = q_init
        if wm_init is not None:
            self.wm_init = wm_init

        # Reset Q-table
        self.Q.fill(self.q_init)

        # Reset WM matrix and baseline
        self.WM.fill(self.wm_init)
        self.WM_0.fill(self.wm_init)

        # Clear history
        self.history = {
            'stimuli': [],
            'actions': [],
            'rewards': [],
            'q_values': [],
            'wm_values': [],
            'rl_probs': [],
            'wm_probs': [],
            'hybrid_probs': [],
            'omega': [],
            'prediction_errors': [],
        }

    def get_rl_probs(self, stimulus: int) -> np.ndarray:
        """
        Compute RL action probabilities using softmax over Q-values.

        p_RL(a|s) = exp(β·Q(s,a)) / Σ_a' exp(β·Q(s,a'))

        Parameters
        ----------
        stimulus : int
            Current stimulus index

        Returns
        -------
        np.ndarray
            RL probability distribution over actions
        """
        q_vals = self.Q[stimulus, :]

        # Numerical stability: subtract max before exp
        q_vals_scaled = self.beta * (q_vals - np.max(q_vals))
        exp_q = np.exp(q_vals_scaled)
        probs = exp_q / np.sum(exp_q)

        return probs

    def get_wm_probs(self, stimulus: int) -> np.ndarray:
        """
        Compute WM action probabilities using softmax over WM values.

        p_WM(a|s) = exp(β_WM·WM(s,a)) / Σ_a' exp(β_WM·WM(s,a'))

        Parameters
        ----------
        stimulus : int
            Current stimulus index

        Returns
        -------
        np.ndarray
            WM probability distribution over actions
        """
        wm_vals = self.WM[stimulus, :]

        # Numerical stability: subtract max before exp
        wm_vals_scaled = self.beta_wm * (wm_vals - np.max(wm_vals))
        exp_wm = np.exp(wm_vals_scaled)
        probs = exp_wm / np.sum(exp_wm)

        return probs

    def get_adaptive_weight(self, set_size: int) -> float:
        """
        Compute adaptive WM weight based on capacity and set size.

        ω = ρ * min(1, K/N_s)

        Where:
        - ρ is base WM reliance
        - K is WM capacity
        - N_s is current set size

        When set size ≤ capacity, full WM reliance (ω = ρ)
        When set size > capacity, reduced reliance (ω < ρ)

        Parameters
        ----------
        set_size : int
            Number of stimuli in current context

        Returns
        -------
        float
            Adaptive WM weight (0-1)
        """
        omega = self.rho * min(1.0, self.capacity / set_size)
        return omega

    def get_hybrid_probs(self, stimulus: int, set_size: int) -> Dict[str, Any]:
        """
        Compute hybrid action probabilities combining WM and RL.

        p(a|s) = ω·p_WM(a|s) + (1-ω)·p_RL(a|s)

        Where ω = ρ * min(1, K/N_s) adapts to task demands.

        Parameters
        ----------
        stimulus : int
            Current stimulus index
        set_size : int
            Number of stimuli in current context

        Returns
        -------
        dict
            Dictionary with:
            - 'probs': hybrid probabilities
            - 'wm_probs': WM-only probabilities
            - 'rl_probs': RL-only probabilities
            - 'omega': adaptive weight used
        """
        # Get component probabilities
        rl_probs = self.get_rl_probs(stimulus)
        wm_probs = self.get_wm_probs(stimulus)

        # Adaptive weight
        omega = self.get_adaptive_weight(set_size)

        # Weighted combination
        hybrid_probs = omega * wm_probs + (1 - omega) * rl_probs

        # Normalize (should already sum to 1, but ensure numerical stability)
        hybrid_probs /= np.sum(hybrid_probs)

        return {
            'probs': hybrid_probs,
            'wm_probs': wm_probs,
            'rl_probs': rl_probs,
            'omega': omega,
        }

    def choose_action(
        self,
        stimulus: int,
        set_size: int,
        return_info: bool = False
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        """
        Choose an action based on hybrid WM-RL policy.

        Parameters
        ----------
        stimulus : int
            Current stimulus index
        set_size : int
            Number of stimuli in current context
        return_info : bool
            If True, return full probability info

        Returns
        -------
        action : int
            Chosen action
        info : dict, optional
            Full probability breakdown (if return_info=True)
        """
        hybrid_info = self.get_hybrid_probs(stimulus, set_size)
        action = self.rng.choice(self.num_actions, p=hybrid_info['probs'])

        if return_info:
            return action, hybrid_info
        return action, None

    def update(
        self,
        stimulus: int,
        action: int,
        reward: float,
        next_stimulus: Optional[int] = None,
    ):
        """
        Update WM and Q-table based on observed reward.

        Update sequence:
        1. Decay all WM values toward baseline: WM ← (1-φ)WM + φ·WM_0
        2. Overwrite WM cell with current reward: WM(s,a) ← r
        3. Update Q-table with asymmetric learning: Q(s,a) ← Q(s,a) + α·δ

        Parameters
        ----------
        stimulus : int
            Current stimulus
        action : int
            Action taken
        reward : float
            Reward received
        next_stimulus : int, optional
            Next stimulus (unused when γ=0)
        """
        # =================================================================
        # 1. DECAY WM: All values move toward baseline
        # =================================================================
        self.WM = (1 - self.phi) * self.WM + self.phi * self.WM_0

        # =================================================================
        # 2. UPDATE WM: Immediate overwrite (α=1)
        # =================================================================
        self.WM[stimulus, action] = reward

        # =================================================================
        # 3. UPDATE Q-TABLE: Asymmetric learning rates
        # =================================================================
        q_current = self.Q[stimulus, action]

        # TD target (simplified since γ=0)
        if self.gamma > 0 and next_stimulus is not None:
            v_next = np.max(self.Q[next_stimulus, :])
            td_target = reward + self.gamma * v_next
        else:
            td_target = reward

        # Prediction error
        prediction_error = td_target - q_current

        # Select learning rate based on PE sign
        if prediction_error > 0:
            alpha = self.alpha_pos  # Positive PE (correct trial)
        else:
            alpha = self.alpha_neg  # Negative PE (incorrect trial)

        # Q-value update
        self.Q[stimulus, action] += alpha * prediction_error

    def log_trial(
        self,
        stimulus: int,
        action: int,
        reward: float,
        set_size: int,
        hybrid_info: Optional[Dict[str, Any]] = None
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
        set_size : int
            Current set size
        hybrid_info : dict, optional
            Pre-computed probability info from choose_action
        """
        self.history['stimuli'].append(stimulus)
        self.history['actions'].append(action)
        self.history['rewards'].append(reward)
        self.history['q_values'].append(self.Q[stimulus, :].copy())
        self.history['wm_values'].append(self.WM[stimulus, :].copy())

        # Get or recompute probabilities
        if hybrid_info is not None:
            self.history['rl_probs'].append(hybrid_info['rl_probs'])
            self.history['wm_probs'].append(hybrid_info['wm_probs'])
            self.history['hybrid_probs'].append(hybrid_info['probs'])
            self.history['omega'].append(hybrid_info['omega'])
        else:
            info = self.get_hybrid_probs(stimulus, set_size)
            self.history['rl_probs'].append(info['rl_probs'])
            self.history['wm_probs'].append(info['wm_probs'])
            self.history['hybrid_probs'].append(info['probs'])
            self.history['omega'].append(info['omega'])

        # Compute prediction error
        pe = reward - self.Q[stimulus, action]
        self.history['prediction_errors'].append(pe)

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

    def get_wm_matrix(self) -> np.ndarray:
        """
        Get current WM value matrix.

        Returns
        -------
        np.ndarray
            WM matrix of shape (num_stimuli, num_actions)
        """
        return self.WM.copy()

    def set_parameters(
        self,
        alpha_pos: Optional[float] = None,
        alpha_neg: Optional[float] = None,
        beta: Optional[float] = None,
        beta_wm: Optional[float] = None,
        phi: Optional[float] = None,
        rho: Optional[float] = None,
        capacity: Optional[int] = None,
    ):
        """
        Set model parameters.

        Parameters
        ----------
        alpha_pos : float, optional
            Learning rate for positive PE
        alpha_neg : float, optional
            Learning rate for negative PE
        beta : float, optional
            RL inverse temperature
        beta_wm : float, optional
            WM inverse temperature
        phi : float, optional
            WM decay rate
        rho : float, optional
            Base WM reliance
        capacity : int, optional
            WM capacity
        """
        if alpha_pos is not None:
            self.alpha_pos = alpha_pos
        if alpha_neg is not None:
            self.alpha_neg = alpha_neg
        if beta is not None:
            self.beta = beta
        if beta_wm is not None:
            self.beta_wm = beta_wm
        if phi is not None:
            self.phi = phi
        if rho is not None:
            self.rho = rho
        if capacity is not None:
            self.capacity = capacity

    def get_parameters(self) -> Dict[str, float]:
        """
        Get current model parameters.

        Returns
        -------
        dict
            Dictionary of parameters
        """
        return {
            'alpha_pos': self.alpha_pos,
            'alpha_neg': self.alpha_neg,
            'beta': self.beta,
            'beta_wm': self.beta_wm,
            'gamma': self.gamma,
            'capacity': self.capacity,
            'phi': self.phi,
            'rho': self.rho,
            'q_init': self.q_init,
            'wm_init': self.wm_init,
        }


# ============================================================================
# AGENT FACTORY FUNCTION
# ============================================================================

def create_wm_rl_agent(
    alpha_pos: float = ModelParams.ALPHA_POS_DEFAULT,
    alpha_neg: float = ModelParams.ALPHA_NEG_DEFAULT,
    beta: float = ModelParams.BETA_DEFAULT,
    beta_wm: float = ModelParams.BETA_WM_DEFAULT,
    capacity: int = ModelParams.WM_CAPACITY_DEFAULT,
    phi: float = ModelParams.PHI_DEFAULT,
    rho: float = ModelParams.RHO_DEFAULT,
    gamma: float = 0.0,
    seed: Optional[int] = None,
    **kwargs
) -> WMRLHybridAgent:
    """
    Factory function to create WM-RL agent with common configurations.

    Parameters
    ----------
    alpha_pos : float
        Learning rate for positive prediction errors
    alpha_neg : float
        Learning rate for negative prediction errors
    beta : float
        RL inverse temperature
    beta_wm : float
        WM inverse temperature
    capacity : int
        WM capacity
    phi : float
        WM decay rate
    rho : float
        Base WM reliance
    gamma : float
        Discount factor (typically 0.0)
    seed : int, optional
        Random seed
    **kwargs
        Additional arguments passed to WMRLHybridAgent

    Returns
    -------
    WMRLHybridAgent
        Configured agent
    """
    return WMRLHybridAgent(
        alpha_pos=alpha_pos,
        alpha_neg=alpha_neg,
        beta=beta,
        beta_wm=beta_wm,
        capacity=capacity,
        phi=phi,
        rho=rho,
        gamma=gamma,
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
) -> Dict[str, any]:
    """
    Simulate WM-RL agent interacting with the environment.

    Parameters
    ----------
    agent : WMRLHybridAgent
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
    set_sizes = []
    omegas = []

    for trial in range(num_trials):
        # Get current stimulus and set size
        stimulus = obs['stimulus']
        set_size = obs['set_size'].item() if hasattr(obs['set_size'], 'item') else obs['set_size']

        stimuli.append(stimulus)
        set_sizes.append(set_size)

        # Choose action
        action, hybrid_info = agent.choose_action(stimulus, set_size, return_info=True)
        actions.append(action)

        # Take step in environment
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        correct.append(info.get('correct', info.get('is_correct', reward > 0)))
        omegas.append(hybrid_info['omega'])

        # Update agent
        next_stimulus = obs['stimulus'] if not (terminated or truncated) else None
        agent.update(stimulus, action, reward, next_stimulus)

        # Log if requested
        if log_history:
            agent.log_trial(stimulus, action, reward, set_size, hybrid_info)

        # Check if done
        if terminated or truncated:
            break

    return {
        'stimuli': stimuli,
        'actions': actions,
        'rewards': rewards,
        'correct': correct,
        'set_sizes': set_sizes,
        'omegas': omegas,
        'accuracy': np.mean(correct),
        'total_reward': np.sum(rewards),
        'num_trials': len(correct),
        'mean_omega': np.mean(omegas),
    }


def test_wm_rl_agent():
    """Test WM-RL agent on RLWM environment."""
    print("Testing WM-RL Hybrid Agent (Matrix-Based)")
    print("=" * 80)

    # Import environment
    from environments.rlwm_env import create_rlwm_env

    # Create environment and agent
    env = create_rlwm_env(set_size=5, phase_type='main_task', seed=42)
    agent = create_wm_rl_agent(
        alpha_pos=0.3,
        alpha_neg=0.1,
        beta=2.0,
        beta_wm=3.0,
        capacity=4,
        phi=0.1,
        rho=0.7,
        seed=42
    )

    print(f"\nAgent parameters: {agent.get_parameters()}")
    print(f"  WM capacity K = {agent.capacity}")
    print(f"  Base WM reliance ρ = {agent.rho:.2f}")
    print(f"  Set size N_s = 5")
    print(f"  Expected ω = ρ * min(1, K/N_s) = {agent.rho:.2f} * min(1, 4/5) = {agent.rho * 0.8:.2f}")

    # Run simulation
    print("\nRunning 100-trial simulation...")
    results = simulate_wm_rl_on_env(agent, env, num_trials=100, log_history=True)

    print(f"\nResults:")
    print(f"  Accuracy: {results['accuracy']:.3f}")
    print(f"  Total reward: {results['total_reward']:.0f}")
    print(f"  Trials completed: {results['num_trials']}")
    print(f"  Mean ω (WM weight): {results['mean_omega']:.3f}")

    # Show final matrices
    print(f"\nFinal Q-table (first 3 stimuli):")
    print(agent.get_q_table()[:3, :])

    print(f"\nFinal WM matrix (first 3 stimuli):")
    print(agent.get_wm_matrix()[:3, :])

    # Analyze omega over time
    history = agent.get_history()
    if len(history['omega']) > 0:
        omegas = np.array(history['omega'])
        print(f"\nOmega (WM weight) statistics:")
        print(f"  Mean: {np.mean(omegas):.3f}")
        print(f"  Std: {np.std(omegas):.3f}")
        print(f"  Min: {np.min(omegas):.3f}")
        print(f"  Max: {np.max(omegas):.3f}")

    # Compute learning curve
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
    test_wm_rl_agent()
