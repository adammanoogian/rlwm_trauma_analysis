"""
Working Memory + RL Hybrid Model for RLWM Task

Matrix-based WM-RL model combining distributed working memory with
model-free reinforcement learning.

Model Architecture
------------------
Working Memory (WM):
- Representation: State-action value matrix WM(s,a)
- Update: WM(s,a) <- r (alpha=1, immediate overwrite)
- Decay: WM <- (1-phi)*WM + phi*WM_0 (global, every trial)
- Policy: p_WM(a|s) = softmax(beta_WM * WM(s,:))

Q-Learning (RL):
- Representation: State-action value matrix Q(s,a)
- Update: delta = r - Q(s,a); alpha = alpha_pos if delta>0 else alpha_neg
- Policy: p_RL(a|s) = softmax(beta * Q(s,:))

Hybrid Decision:
- Adaptive weight: omega = rho * min(1, K/N_s)
- Policy: p(a|s) = omega*p_WM(a|s) + (1-omega)*p_RL(a|s)

Parameters
----------
alpha_pos : learning rate for positive PE in RL (0-1)
alpha_neg : learning rate for negative PE in RL (0-1)
beta : inverse temperature for RL softmax (>0)
beta_wm : inverse temperature for WM softmax (>0)
phi : WM decay rate toward baseline (0-1)
rho : base WM reliance parameter (0-1)
K (capacity) : WM capacity for adaptive weighting
kappa : perseveration parameter (0-1), M3 extension
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rlwm.config import ModelParams, TaskParams


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
        beta_wm: float = ModelParams.BETA_DEFAULT,
        gamma: float = 0.0,
        capacity: int = ModelParams.WM_CAPACITY_DEFAULT,
        phi: float = ModelParams.PHI_DEFAULT,
        rho: float = ModelParams.RHO_DEFAULT,
        q_init: float = ModelParams.Q_INIT_VALUE,
        wm_init: float = ModelParams.WM_INIT_VALUE,
        kappa: float = 0.0,
        seed: int | None = None,
    ):
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
        self.kappa = kappa
        self.last_action: int | None = None

        self.rng = np.random.RandomState(seed)

        self.Q = np.full((num_stimuli, num_actions), q_init, dtype=np.float64)
        self.WM = np.full((num_stimuli, num_actions), wm_init, dtype=np.float64)
        self.WM_0 = np.full((num_stimuli, num_actions), wm_init, dtype=np.float64)

        self.history: dict[str, list] = {
            "stimuli": [],
            "actions": [],
            "rewards": [],
            "q_values": [],
            "wm_values": [],
            "rl_probs": [],
            "wm_probs": [],
            "hybrid_probs": [],
            "omega": [],
            "prediction_errors": [],
            "last_actions": [],
        }

    def reset(self, q_init: float | None = None, wm_init: float | None = None):
        if q_init is not None:
            self.q_init = q_init
        if wm_init is not None:
            self.wm_init = wm_init

        self.Q.fill(self.q_init)
        self.WM.fill(self.wm_init)
        self.WM_0.fill(self.wm_init)
        self.last_action = None

        self.history = {
            "stimuli": [],
            "actions": [],
            "rewards": [],
            "q_values": [],
            "wm_values": [],
            "rl_probs": [],
            "wm_probs": [],
            "hybrid_probs": [],
            "omega": [],
            "prediction_errors": [],
        }

    def get_rl_probs(self, stimulus: int) -> np.ndarray:
        q_vals = self.Q[stimulus, :]
        q_vals_scaled = self.beta * (q_vals - np.max(q_vals))
        exp_q = np.exp(q_vals_scaled)
        return exp_q / exp_q.sum()

    def get_wm_probs(self, stimulus: int) -> np.ndarray:
        wm_vals = self.WM[stimulus, :]
        wm_vals_scaled = self.beta_wm * (wm_vals - np.max(wm_vals))
        exp_wm = np.exp(wm_vals_scaled)
        return exp_wm / exp_wm.sum()

    def get_adaptive_weight(self, set_size: int) -> float:
        return self.rho * min(1.0, self.capacity / set_size)

    def get_hybrid_probs(self, stimulus: int, set_size: int) -> dict[str, Any]:
        q_vals = self.Q[stimulus, :]
        wm_vals = self.WM[stimulus, :]

        rl_probs = self.get_rl_probs(stimulus)
        wm_probs = self.get_wm_probs(stimulus)

        omega = self.get_adaptive_weight(set_size)

        base_probs = omega * wm_probs + (1 - omega) * rl_probs
        base_probs /= base_probs.sum()
        hybrid_vals = omega * wm_vals + (1 - omega) * q_vals

        if self.kappa == 0 or self.last_action is None:
            hybrid_probs = base_probs
        else:
            choice_kernel = np.zeros(self.num_actions)
            choice_kernel[self.last_action] = 1.0
            hybrid_probs = (1 - self.kappa) * base_probs + self.kappa * choice_kernel

        return {
            "probs": hybrid_probs,
            "wm_probs": wm_probs,
            "rl_probs": rl_probs,
            "omega": omega,
            "hybrid_vals": hybrid_vals,
        }

    def choose_action(
        self,
        stimulus: int,
        set_size: int,
        return_info: bool = False,
    ) -> tuple[int, dict[str, Any] | None]:
        hybrid_info = self.get_hybrid_probs(stimulus, set_size)
        action = self.rng.choice(self.num_actions, p=hybrid_info["probs"])

        if return_info:
            return action, hybrid_info
        return action, None

    def update(
        self,
        stimulus: int,
        action: int,
        reward: float,
        next_stimulus: int | None = None,
    ):
        # 1. Decay WM
        self.WM = (1 - self.phi) * self.WM + self.phi * self.WM_0

        # 2. Update WM: immediate overwrite
        self.WM[stimulus, action] = reward

        # 3. Update Q-table: asymmetric learning
        q_current = self.Q[stimulus, action]

        if self.gamma > 0 and next_stimulus is not None:
            v_next = np.max(self.Q[next_stimulus, :])
            td_target = reward + self.gamma * v_next
        else:
            td_target = reward

        prediction_error = td_target - q_current
        alpha = self.alpha_pos if prediction_error > 0 else self.alpha_neg
        self.Q[stimulus, action] += alpha * prediction_error

        # 4. Track action for perseveration
        self.last_action = action

    def log_trial(
        self,
        stimulus: int,
        action: int,
        reward: float,
        set_size: int,
        hybrid_info: dict[str, Any] | None = None,
    ):
        self.history["stimuli"].append(stimulus)
        self.history["actions"].append(action)
        self.history["rewards"].append(reward)
        self.history["q_values"].append(self.Q[stimulus, :].copy())
        self.history["wm_values"].append(self.WM[stimulus, :].copy())

        if hybrid_info is not None:
            self.history["rl_probs"].append(hybrid_info["rl_probs"])
            self.history["wm_probs"].append(hybrid_info["wm_probs"])
            self.history["hybrid_probs"].append(hybrid_info["probs"])
            self.history["omega"].append(hybrid_info["omega"])
        else:
            info = self.get_hybrid_probs(stimulus, set_size)
            self.history["rl_probs"].append(info["rl_probs"])
            self.history["wm_probs"].append(info["wm_probs"])
            self.history["hybrid_probs"].append(info["probs"])
            self.history["omega"].append(info["omega"])

        pe = reward - self.Q[stimulus, action]
        self.history["prediction_errors"].append(pe)

    def get_history(self) -> dict[str, list]:
        return self.history

    def get_q_table(self) -> np.ndarray:
        return self.Q.copy()

    def get_wm_matrix(self) -> np.ndarray:
        return self.WM.copy()

    def set_parameters(
        self,
        alpha_pos: float | None = None,
        alpha_neg: float | None = None,
        beta: float | None = None,
        beta_wm: float | None = None,
        phi: float | None = None,
        rho: float | None = None,
        capacity: int | None = None,
        kappa: float | None = None,
    ):
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
        if kappa is not None:
            self.kappa = kappa

    def get_parameters(self) -> dict[str, float]:
        return {
            "alpha_pos": self.alpha_pos,
            "alpha_neg": self.alpha_neg,
            "beta": self.beta,
            "beta_wm": self.beta_wm,
            "gamma": self.gamma,
            "capacity": self.capacity,
            "phi": self.phi,
            "rho": self.rho,
            "q_init": self.q_init,
            "wm_init": self.wm_init,
            "kappa": self.kappa,
        }


def create_wm_rl_agent(
    alpha_pos: float = ModelParams.ALPHA_POS_DEFAULT,
    alpha_neg: float = ModelParams.ALPHA_NEG_DEFAULT,
    beta: float = ModelParams.BETA_DEFAULT,
    beta_wm: float = ModelParams.BETA_DEFAULT,
    capacity: int = ModelParams.WM_CAPACITY_DEFAULT,
    phi: float = ModelParams.PHI_DEFAULT,
    rho: float = ModelParams.RHO_DEFAULT,
    gamma: float = 0.0,
    kappa: float = 0.0,
    seed: int | None = None,
    **kwargs,
) -> WMRLHybridAgent:
    """Factory function to create WM-RL agent."""
    return WMRLHybridAgent(
        alpha_pos=alpha_pos,
        alpha_neg=alpha_neg,
        beta=beta,
        beta_wm=beta_wm,
        capacity=capacity,
        phi=phi,
        rho=rho,
        gamma=gamma,
        kappa=kappa,
        seed=seed,
        **kwargs,
    )


def simulate_wm_rl_on_env(
    agent: WMRLHybridAgent,
    env,
    num_trials: int = 100,
    log_history: bool = True,
) -> dict:
    """Simulate WM-RL agent interacting with the environment."""
    obs, info = env.reset()
    agent.reset()

    rewards = []
    actions = []
    correct = []
    stimuli = []
    set_sizes = []
    omegas = []

    for trial in range(num_trials):
        stimulus = obs["stimulus"]
        set_size = (
            obs["set_size"].item()
            if hasattr(obs["set_size"], "item")
            else obs["set_size"]
        )

        stimuli.append(stimulus)
        set_sizes.append(set_size)

        action, hybrid_info = agent.choose_action(
            stimulus, set_size, return_info=True
        )
        actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        correct.append(info.get("correct", info.get("is_correct", reward > 0)))
        omegas.append(hybrid_info["omega"])

        next_stimulus = obs["stimulus"] if not (terminated or truncated) else None
        agent.update(stimulus, action, reward, next_stimulus)

        if log_history:
            agent.log_trial(stimulus, action, reward, set_size, hybrid_info)

        if terminated or truncated:
            break

    return {
        "stimuli": stimuli,
        "actions": actions,
        "rewards": rewards,
        "correct": correct,
        "set_sizes": set_sizes,
        "omegas": omegas,
        "accuracy": np.mean(correct),
        "total_reward": np.sum(rewards),
        "num_trials": len(correct),
        "mean_omega": np.mean(omegas),
    }
