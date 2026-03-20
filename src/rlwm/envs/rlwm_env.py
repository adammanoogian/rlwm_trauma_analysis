"""
RLWM Environment: Reinforcement Learning + Working Memory Task

Trial-based gymnasium environment for the RLWM task. Each env.step() represents
one complete trial: stimulus presentation -> response -> feedback.

Based on the jsPsych implementation with:
- Variable set sizes (2, 3, 5, 6 stimuli)
- 3-choice responses (J/K/L -> actions 0/1/2)
- Rare, late reversals (12-18 consecutive correct)
- Binary rewards (+1 correct, 0 incorrect)
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlwm.config import TaskParams


class RLWMEnv(gym.Env):
    """
    Reinforcement Learning + Working Memory Environment.

    This is a trial-based environment where each step represents a complete trial.
    The agent must learn stimulus-response mappings that occasionally reverse.

    Observation Space
    -----------------
    dict with keys:
        - 'stimulus': Discrete(6) - which stimulus (0-5)
        - 'set_size': Box(1,) - current set size (2, 3, 5, or 6)
        - 'block': Discrete(24) - current block (0-23)
        - 'phase': Discrete(3) - phase type (0=practice_static, 1=practice_dynamic, 2=main)

    Action Space
    ------------
    Discrete(3) - which key to press (0=J, 1=K, 2=L)

    Rewards
    -------
    +1 for correct response, 0 for incorrect

    Episode Termination
    -------------------
    Episode ends when max_trials_per_block is reached or when truncated.

    Parameters
    ----------
    set_size : int, optional
        Fixed set size for all blocks. If None, will vary according to task design.
    block_sequence : list, optional
        Sequence of set sizes per block. If None, uses default task structure.
    max_trials_per_block : int, optional
        Maximum trials per block. Default is 100.
    phase_type : str, optional
        Task phase: 'practice_static', 'practice_dynamic', or 'main_task'.
        Default is 'main_task'.
    seed : int, optional
        Random seed for reproducibility.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        set_size: int | None = None,
        block_sequence: list | None = None,
        max_trials_per_block: int = 100,
        phase_type: str = "main_task",
        seed: int | None = None,
    ):
        super().__init__()

        # Set random seed
        self.seed_value = seed
        self.rng = np.random.RandomState(seed)

        # Task parameters
        self.set_size = set_size
        self.block_sequence = block_sequence
        self.max_trials_per_block = max_trials_per_block
        self.phase_type = phase_type

        # Define observation space
        self.observation_space = spaces.Dict(
            {
                "stimulus": spaces.Discrete(TaskParams.MAX_STIMULI),
                "set_size": spaces.Box(
                    low=min(TaskParams.SET_SIZES),
                    high=max(TaskParams.SET_SIZES),
                    shape=(1,),
                    dtype=np.int32,
                ),
                "block": spaces.Discrete(TaskParams.TOTAL_BLOCKS),
                "phase": spaces.Discrete(3),
            }
        )

        # Define action space
        self.action_space = spaces.Discrete(TaskParams.NUM_ACTIONS)

        # Initialize state variables
        self.current_block = 0
        self.trial_in_block = 0
        self.current_stimulus = 0
        self.current_set_size = 2

        # Stimulus-response mappings
        self.correct_responses = np.zeros(TaskParams.MAX_STIMULI, dtype=np.int32)

        # Reversal tracking
        self.reversal_points = np.full(TaskParams.MAX_STIMULI, np.inf)
        self.correct_counters = np.zeros(TaskParams.MAX_STIMULI, dtype=np.int32)
        self.reversals_done = np.zeros(TaskParams.MAX_STIMULI, dtype=np.int32)

        # Trial history
        self.trial_count = 0
        self.episode_rewards: list[float] = []
        self.episode_stimuli: list[int] = []
        self.episode_actions: list[int] = []
        self.episode_correct: list[bool] = []

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict, dict]:
        """Reset the environment for a new block."""
        if seed is not None:
            self.seed_value = seed
            self.rng = np.random.RandomState(seed)

        if options:
            if "block" in options:
                self.current_block = options["block"]
            if "set_size" in options:
                self.current_set_size = options["set_size"]
            if "phase_type" in options:
                self.phase_type = options["phase_type"]

        self.trial_in_block = 0
        self.trial_count = 0

        if self.set_size is not None:
            self.current_set_size = self.set_size
        elif self.block_sequence is not None and len(self.block_sequence) > self.current_block:
            self.current_set_size = self.block_sequence[self.current_block]
        else:
            self.current_set_size = self.rng.choice(TaskParams.SET_SIZES)

        for stim in range(self.current_set_size):
            self.correct_responses[stim] = self.rng.choice(TaskParams.ACTIONS)

        self._initialize_reversals()
        self.correct_counters.fill(0)
        self.reversals_done.fill(0)

        self.episode_rewards = []
        self.episode_stimuli = []
        self.episode_actions = []
        self.episode_correct = []

        self.current_stimulus = self._sample_stimulus()
        obs = self._get_observation()

        info = {
            "block": self.current_block,
            "trial": self.trial_in_block,
            "set_size": self.current_set_size,
            "phase_type": self.phase_type,
            "correct_response": self.correct_responses[self.current_stimulus],
        }

        return obs, info

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        """Execute one trial: agent responds to stimulus, receives feedback."""
        correct_action = self.correct_responses[self.current_stimulus]
        is_correct = action == correct_action
        reward = TaskParams.REWARD_CORRECT if is_correct else TaskParams.REWARD_INCORRECT

        if is_correct:
            self.correct_counters[self.current_stimulus] += 1
        else:
            self.correct_counters[self.current_stimulus] = 0

        reversal_occurred = False
        if (
            self.correct_counters[self.current_stimulus]
            >= self.reversal_points[self.current_stimulus]
            and self.reversals_done[self.current_stimulus]
            < TaskParams.MAX_REVERSALS_PER_STIM
        ):
            reversal_occurred = True
            self._execute_reversal(self.current_stimulus)

        self.episode_rewards.append(reward)
        self.episode_stimuli.append(self.current_stimulus)
        self.episode_actions.append(action)
        self.episode_correct.append(is_correct)

        self.trial_in_block += 1
        self.trial_count += 1

        truncated = self.trial_in_block >= self.max_trials_per_block
        terminated = False

        if not truncated:
            self.current_stimulus = self._sample_stimulus()

        obs = self._get_observation()

        info = {
            "block": self.current_block,
            "trial": self.trial_in_block,
            "set_size": self.current_set_size,
            "phase_type": self.phase_type,
            "stimulus": self.episode_stimuli[-1],
            "action": action,
            "correct_response": correct_action,
            "is_correct": is_correct,
            "reversal_occurred": reversal_occurred,
            "correct_counter": self.correct_counters[self.episode_stimuli[-1]],
            "accuracy": np.mean(self.episode_correct),
            "total_reward": sum(self.episode_rewards),
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> dict:
        phase_map = {
            "practice_static": 0,
            "practice_dynamic": 1,
            "main_task": 2,
        }
        phase_int = phase_map.get(self.phase_type, 2)

        return {
            "stimulus": self.current_stimulus,
            "set_size": np.array([self.current_set_size], dtype=np.int32),
            "block": self.current_block,
            "phase": phase_int,
        }

    def _sample_stimulus(self) -> int:
        return self.rng.choice(self.current_set_size)

    def _initialize_reversals(self):
        if self.phase_type == "practice_static":
            self.reversal_points.fill(np.inf)
        elif self.phase_type == "practice_dynamic":
            for stim in range(self.current_set_size):
                self.reversal_points[stim] = (
                    TaskParams.PRACTICE_DYNAMIC_REVERSAL_CRITERION
                )
        elif self.phase_type == "main_task":
            if TaskParams.RARE_REVERSALS:
                for stim in range(self.current_set_size):
                    self.reversal_points[stim] = self.rng.randint(
                        TaskParams.REVERSAL_MIN, TaskParams.REVERSAL_MAX + 1
                    )
            else:
                self.reversal_points.fill(np.inf)
        else:
            self.reversal_points.fill(np.inf)

    def _execute_reversal(self, stimulus: int):
        self.correct_counters[stimulus] = 0
        self.reversals_done[stimulus] += 1

        current_response = self.correct_responses[stimulus]
        possible_responses = [a for a in TaskParams.ACTIONS if a != current_response]
        self.correct_responses[stimulus] = self.rng.choice(possible_responses)

        if self.phase_type == "main_task":
            self.reversal_points[stimulus] = np.inf
        elif self.phase_type == "practice_dynamic":
            self.reversal_points[stimulus] = (
                TaskParams.PRACTICE_DYNAMIC_REVERSAL_CRITERION
            )

    def get_performance_metrics(self) -> dict[str, float]:
        if len(self.episode_correct) == 0:
            return {
                "accuracy": 0.0,
                "total_reward": 0.0,
                "num_trials": 0,
                "num_reversals": 0,
            }

        return {
            "accuracy": np.mean(self.episode_correct),
            "total_reward": sum(self.episode_rewards),
            "num_trials": len(self.episode_correct),
            "num_reversals": int(np.sum(self.reversals_done)),
        }

    def seed(self, seed: int):
        self.seed_value = seed
        self.rng = np.random.RandomState(seed)

    def render(self):
        pass

    def close(self):
        pass


def create_rlwm_env(
    set_size: int | None = None,
    phase_type: str = "main_task",
    seed: int | None = None,
    **kwargs,
) -> RLWMEnv:
    """Factory function to create RLWM environment."""
    return RLWMEnv(
        set_size=set_size,
        phase_type=phase_type,
        seed=seed,
        **kwargs,
    )
