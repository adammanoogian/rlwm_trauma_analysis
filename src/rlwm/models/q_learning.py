"""
Q-Learning Model for RLWM Task

Model-free reinforcement learning agent with asymmetric learning rates for
positive and negative prediction errors.

Model Equations
---------------
Q-value update (asymmetric learning):
    delta = r - Q(s,a)                          [prediction error]
    alpha = alpha_pos if delta > 0 else alpha_neg  [select learning rate]
    Q(s,a) <- Q(s,a) + alpha * delta

Action selection (softmax):
    P(a|s) = exp(beta * Q(s,a)) / sum_a' exp(beta * Q(s,a'))

Parameters
----------
alpha_pos : learning rate for positive PE (0-1)
alpha_neg : learning rate for negative PE (0-1)
beta : inverse temperature (>0)
gamma : discount factor (0-1), fixed at 0 for this task
"""

from __future__ import annotations

import numpy as np

from rlwm.config import ModelParams, TaskParams


class QLearningAgent:
    """
    Q-Learning agent with asymmetric learning rates.

    Learns stimulus-response mappings using temporal difference learning with
    separate learning rates for positive and negative prediction errors.
    """

    def __init__(
        self,
        num_stimuli: int = TaskParams.MAX_STIMULI,
        num_actions: int = TaskParams.NUM_ACTIONS,
        alpha_pos: float = ModelParams.ALPHA_POS_DEFAULT,
        alpha_neg: float = ModelParams.ALPHA_NEG_DEFAULT,
        beta: float = ModelParams.BETA_DEFAULT,
        gamma: float = 0.0,
        q_init: float = ModelParams.Q_INIT_VALUE,
        seed: int | None = None,
    ):
        self.num_stimuli = num_stimuli
        self.num_actions = num_actions
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.beta = beta
        self.gamma = gamma
        self.q_init = q_init

        self.rng = np.random.RandomState(seed)
        self.Q = np.full((num_stimuli, num_actions), q_init, dtype=np.float64)

        self.history: dict[str, list] = {
            "stimuli": [],
            "actions": [],
            "rewards": [],
            "q_values": [],
            "action_probs": [],
            "prediction_errors": [],
        }

    def reset(self, q_init: float | None = None):
        if q_init is not None:
            self.q_init = q_init
        self.Q.fill(self.q_init)
        self.history = {
            "stimuli": [],
            "actions": [],
            "rewards": [],
            "q_values": [],
            "action_probs": [],
            "prediction_errors": [],
        }

    def get_action_probs(self, stimulus: int) -> np.ndarray:
        q_vals = self.Q[stimulus, :]
        q_vals_scaled = self.beta * (q_vals - np.max(q_vals))
        exp_q = np.exp(q_vals_scaled)
        return exp_q / exp_q.sum()

    def choose_action(self, stimulus: int, return_probs: bool = False) -> int:
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
        next_stimulus: int | None = None,
    ):
        q_current = self.Q[stimulus, action]

        if self.gamma > 0 and next_stimulus is not None:
            v_next = np.max(self.Q[next_stimulus, :])
            td_target = reward + self.gamma * v_next
        else:
            td_target = reward

        prediction_error = td_target - q_current
        alpha = self.alpha_pos if prediction_error > 0 else self.alpha_neg
        self.Q[stimulus, action] += alpha * prediction_error

    def predict_action_probs(self, stimulus: int) -> np.ndarray:
        return self.get_action_probs(stimulus)

    def get_max_q_action(self, stimulus: int) -> int:
        return int(np.argmax(self.Q[stimulus, :]))

    def log_trial(
        self,
        stimulus: int,
        action: int,
        reward: float,
        action_probs: np.ndarray | None = None,
        prediction_error: float | None = None,
    ):
        self.history["stimuli"].append(stimulus)
        self.history["actions"].append(action)
        self.history["rewards"].append(reward)
        self.history["q_values"].append(self.Q[stimulus, :].copy())

        if action_probs is not None:
            self.history["action_probs"].append(action_probs)
        else:
            self.history["action_probs"].append(self.get_action_probs(stimulus))

        if prediction_error is not None:
            self.history["prediction_errors"].append(prediction_error)
        else:
            pe = reward - self.Q[stimulus, action]
            self.history["prediction_errors"].append(pe)

    def get_history(self) -> dict[str, list]:
        return self.history

    def get_q_table(self) -> np.ndarray:
        return self.Q.copy()

    def set_parameters(
        self,
        alpha_pos: float | None = None,
        alpha_neg: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
    ):
        if alpha_pos is not None:
            self.alpha_pos = alpha_pos
        if alpha_neg is not None:
            self.alpha_neg = alpha_neg
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma

    def get_parameters(self) -> dict[str, float]:
        return {
            "alpha_pos": self.alpha_pos,
            "alpha_neg": self.alpha_neg,
            "beta": self.beta,
            "gamma": self.gamma,
            "q_init": self.q_init,
        }


def create_q_learning_agent(
    alpha_pos: float = ModelParams.ALPHA_POS_DEFAULT,
    alpha_neg: float = ModelParams.ALPHA_NEG_DEFAULT,
    beta: float = ModelParams.BETA_DEFAULT,
    gamma: float = 0.0,
    seed: int | None = None,
    **kwargs,
) -> QLearningAgent:
    """Factory function to create Q-learning agent."""
    return QLearningAgent(
        alpha_pos=alpha_pos,
        alpha_neg=alpha_neg,
        beta=beta,
        gamma=gamma,
        seed=seed,
        **kwargs,
    )


def simulate_agent_on_env(
    agent: QLearningAgent,
    env,
    num_trials: int = 100,
    log_history: bool = True,
) -> dict:
    """Simulate an agent interacting with the environment."""
    obs, info = env.reset()
    agent.reset()

    rewards = []
    actions = []
    correct = []
    stimuli = []

    for trial in range(num_trials):
        stimulus = obs["stimulus"]
        stimuli.append(stimulus)

        action, probs = agent.choose_action(stimulus, return_probs=True)
        actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        correct.append(info["is_correct"])

        next_stimulus = obs["stimulus"] if not (terminated or truncated) else None
        agent.update(stimulus, action, reward, next_stimulus)

        if log_history:
            agent.log_trial(stimulus, action, reward, probs)

        if terminated or truncated:
            break

    return {
        "stimuli": stimuli,
        "actions": actions,
        "rewards": rewards,
        "correct": correct,
        "accuracy": np.mean(correct),
        "total_reward": np.sum(rewards),
        "num_trials": len(correct),
    }
