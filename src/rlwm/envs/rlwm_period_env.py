"""
RLWMPeriodEnv — Timestep-level RLWM environment for RNN training.

Follows the NeuroGym convention of splitting each trial into temporal
periods (fixation, stimulus, feedback) with configurable timing.

Each step advances by ``dt`` milliseconds. Within a trial the agent sees
different observations and is expected to produce different actions
depending on the current period.

Design
------
Periods per trial:
    fixation  (500 ms default, 5 steps at dt=100)
    stimulus  (2000 ms default, 20 steps at dt=100)
    feedback  (500 ms default, 5 steps at dt=100)
    → 30 steps per trial at dt=100 ms

Observation: Box(8,) = [fixation_cue, stimulus_onehot(6), feedback_signal]
    fixation   : 1.0 during fixation period, else 0.0
    stim[0..5] : one-hot stimulus encoding during stimulus period ONLY
                 (hidden during feedback, matching show_stim_with_feedback=false)
    feedback   : reward value (1.0 or 0.0) during feedback period, else 0.0

Action: Discrete(4) = [fixation/no-op, choice_0, choice_1, choice_2]
    During fixation period the correct action is 0 (fixation).
    During stimulus period the agent must respond with 1/2/3.
    During feedback period ground-truth is fixation again.

Reward:
    +1.0 on the *first step* of feedback if the agent's response during
    stimulus was correct, else 0.0. All other steps yield 0.0.

info["new_trial"] is True on the first step of each new trial.

NeuroGym metadata
-----------------
paper_link : https://doi.org/10.1016/j.neuron.2025.XX  (Senta et al., 2025)
tags       : ["learning", "working memory", "decision making"]
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlwm.config import TaskParams


class RLWMPeriodEnv(gym.Env):
    """
    Timestep-level RLWM environment with period structure.

    Parameters
    ----------
    timing : dict, optional
        Duration (ms) for each period. Keys: 'fixation', 'stimulus', 'feedback'.
    dt : int, optional
        Timestep duration in ms. Default 100.
    set_size : int, optional
        Fixed set size. If None, sampled from TaskParams.SET_SIZES per block.
    max_trials : int, optional
        Maximum trials per episode. Default 100.
    seed : int, optional
        Random seed.
    """

    metadata = {
        "render_modes": [],
        "paper_link": "https://doi.org/10.1016/j.neuron.2025.XX",
        "tags": ["learning", "working memory", "decision making"],
    }

    # Period names and order
    PERIODS = ("fixation", "stimulus", "feedback")

    def __init__(
        self,
        timing: dict[str, int] | None = None,
        dt: int = 100,
        set_size: int | None = None,
        max_trials: int = 100,
        seed: int | None = None,
    ):
        super().__init__()

        self.dt = dt

        # Default timing from TaskParams
        default_timing = {
            "fixation": TaskParams.FIXATION_DURATION,
            "stimulus": TaskParams.TRIAL_DURATION,
            "feedback": TaskParams.FEEDBACK_DURATION,
        }
        self.timing = {**default_timing, **(timing or {})}

        # Steps per period
        self._period_steps = {
            p: max(1, self.timing[p] // self.dt) for p in self.PERIODS
        }
        self.steps_per_trial = sum(self._period_steps.values())

        self.set_size_setting = set_size
        self.max_trials = max_trials

        # Observation: [fixation_cue, stim_onehot(6), feedback_signal]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )
        # Actions: 0=fixation, 1/2/3 = choice_0/1/2
        self.action_space = spaces.Discrete(4)

        self.rng = np.random.RandomState(seed)

        # State (populated by reset)
        self.current_set_size = 2
        self.correct_responses = np.zeros(TaskParams.MAX_STIMULI, dtype=np.int32)
        self.trial_count = 0
        self._step_in_trial = 0
        self._current_period_idx = 0
        self._current_stimulus = 0
        self._agent_response: int | None = None  # action chosen during stimulus
        self._trial_reward = 0.0

        # Pre-allocated observation / ground-truth arrays for this trial
        self.ob = np.zeros((self.steps_per_trial, 8), dtype=np.float32)
        self.gt = np.zeros(self.steps_per_trial, dtype=np.int64)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # Determine set size
        if self.set_size_setting is not None:
            self.current_set_size = self.set_size_setting
        else:
            self.current_set_size = int(self.rng.choice(TaskParams.SET_SIZES))

        # Randomise stimulus-response mappings
        for s in range(self.current_set_size):
            self.correct_responses[s] = self.rng.choice(TaskParams.ACTIONS)

        self.trial_count = 0
        self._new_trial()

        obs = self.ob[0].copy()
        info = {
            "new_trial": True,
            "trial": self.trial_count,
            "set_size": self.current_set_size,
        }
        return obs, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        reward = 0.0

        period_name = self._current_period()

        # Record agent response during stimulus period (first response wins)
        if period_name == "stimulus" and self._agent_response is None:
            if action in (1, 2, 3):
                self._agent_response = action - 1  # map to 0/1/2

        # Deliver reward on *first* step of feedback and populate feedback channel
        if period_name == "feedback" and self._step_in_trial == self._feedback_start:
            if self._agent_response is not None:
                correct_action = self.correct_responses[self._current_stimulus]
                if self._agent_response == correct_action:
                    reward = 1.0
            self._trial_reward = reward
            # Set feedback signal (obs[7]) for all remaining feedback steps
            feed_end = self._feedback_start + self._period_steps["feedback"]
            self.ob[self._feedback_start : feed_end, 7] = self._trial_reward

        self._step_in_trial += 1

        # Check if trial is finished
        new_trial = False
        if self._step_in_trial >= self.steps_per_trial:
            self.trial_count += 1
            if self.trial_count >= self.max_trials:
                # Episode done
                obs = np.zeros(8, dtype=np.float32)
                info = {
                    "new_trial": False,
                    "trial": self.trial_count,
                    "set_size": self.current_set_size,
                }
                return obs, reward, False, True, info

            self._new_trial()
            new_trial = True

        obs = self.ob[self._step_in_trial].copy()
        info = {
            "new_trial": new_trial,
            "trial": self.trial_count,
            "set_size": self.current_set_size,
        }
        return obs, reward, False, False, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _new_trial(self):
        """Set up observation and ground-truth arrays for a new trial."""
        self._step_in_trial = 0
        self._agent_response = None
        self._trial_reward = 0.0

        # Sample stimulus
        self._current_stimulus = self.rng.choice(self.current_set_size)
        correct_action = self.correct_responses[self._current_stimulus]

        # Build ob and gt for this trial
        self.ob[:] = 0.0
        self.gt[:] = 0  # fixation by default

        fix_steps = self._period_steps["fixation"]
        stim_steps = self._period_steps["stimulus"]
        feed_steps = self._period_steps["feedback"]

        stim_start = fix_steps
        self._feedback_start = fix_steps + stim_steps

        # Fixation period: fixation cue ON
        self.ob[:fix_steps, 0] = 1.0
        self.gt[:fix_steps] = 0  # fixation action

        # Stimulus period: stimulus one-hot ON
        stim_idx = self._current_stimulus
        self.ob[stim_start : stim_start + stim_steps, 1 + stim_idx] = 1.0
        # Ground truth: correct choice (1-indexed for action space)
        self.gt[stim_start : stim_start + stim_steps] = correct_action + 1

        # Feedback period: stimulus HIDDEN (show_stim_with_feedback=false in jsPsych)
        # obs[7] (feedback signal) is filled dynamically in step() once reward is known
        self.gt[self._feedback_start : self._feedback_start + feed_steps] = 0  # fixation

    def _current_period(self) -> str:
        """Return which period the current step belongs to."""
        t = self._step_in_trial
        cumsum = 0
        for p in self.PERIODS:
            cumsum += self._period_steps[p]
            if t < cumsum:
                return p
        return self.PERIODS[-1]

    def render(self):
        pass

    def close(self):
        pass
