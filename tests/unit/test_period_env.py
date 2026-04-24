"""Tests for RLWMPeriodEnv — timestep-level environment with period structure."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest


@pytest.fixture
def period_env():
    from rlwm.envs import RLWMPeriodEnv

    env = RLWMPeriodEnv(set_size=3, max_trials=5, seed=42)
    return env


@pytest.fixture
def default_timing():
    return {"fixation": 500, "stimulus": 2000, "feedback": 500}


# ──────────────────────────────────────────────────────────────────
# Observation shape and space
# ──────────────────────────────────────────────────────────────────


class TestObservationSpace:
    def test_observation_shape(self, period_env):
        obs, info = period_env.reset()
        assert obs.shape == (8,), f"Expected (8,), got {obs.shape}"

    def test_observation_dtype(self, period_env):
        obs, info = period_env.reset()
        assert obs.dtype == np.float32

    def test_observation_in_bounds(self, period_env):
        obs, info = period_env.reset()
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_fixation_cue_on_at_reset(self, period_env):
        obs, info = period_env.reset()
        # First step is fixation period → fixation cue should be on
        assert obs[0] == 1.0, "Fixation cue should be on at trial start"
        # No stimulus should be visible yet
        assert np.sum(obs[1:7]) == 0.0, "No stimulus during fixation"


# ──────────────────────────────────────────────────────────────────
# Action space
# ──────────────────────────────────────────────────────────────────


class TestActionSpace:
    def test_action_space_size(self, period_env):
        assert period_env.action_space.n == 4

    def test_fixation_action(self, period_env):
        obs, _ = period_env.reset()
        # Fixation action (0) during fixation period should be valid
        obs2, reward, terminated, truncated, info = period_env.step(0)
        assert reward == 0.0  # No reward during fixation


# ──────────────────────────────────────────────────────────────────
# Period transitions
# ──────────────────────────────────────────────────────────────────


class TestPeriodTransitions:
    def test_fixation_to_stimulus(self, period_env):
        """After fixation steps, stimulus should appear."""
        obs, _ = period_env.reset()
        fix_steps = period_env._period_steps["fixation"]

        # Step through fixation
        for _ in range(fix_steps):
            obs, _, _, _, _ = period_env.step(0)

        # Now in stimulus period — one stimulus channel should be on
        stim_channels = obs[1:7]
        assert np.sum(stim_channels) == 1.0, (
            f"Exactly one stimulus should be active, got {stim_channels}"
        )

    def test_stimulus_to_feedback(self, period_env):
        """After fixation + stimulus steps, enter feedback with stimulus hidden."""
        obs, _ = period_env.reset()
        fix_steps = period_env._period_steps["fixation"]
        stim_steps = period_env._period_steps["stimulus"]

        # Step through fixation + stimulus
        for i in range(fix_steps + stim_steps):
            action = 1 if i >= fix_steps else 0  # respond during stimulus
            obs, _, _, _, _ = period_env.step(action)

        # Now in feedback period — stimulus should be HIDDEN
        # (matching show_stim_with_feedback=false in jsPsych)
        stim_channels = obs[1:7]
        assert np.sum(stim_channels) == 0.0, (
            f"Stimulus should be hidden during feedback, got {stim_channels}"
        )

    def test_steps_per_trial(self, period_env):
        """Total steps per trial matches timing configuration."""
        expected = (
            period_env._period_steps["fixation"]
            + period_env._period_steps["stimulus"]
            + period_env._period_steps["feedback"]
        )
        assert period_env.steps_per_trial == expected

    def test_default_timing_30_steps(self):
        """Default timing (500/2000/500 ms at dt=100) gives 30 steps."""
        from rlwm.envs import RLWMPeriodEnv

        env = RLWMPeriodEnv(dt=100, seed=42)
        assert env.steps_per_trial == 30


# ──────────────────────────────────────────────────────────────────
# Ground truth
# ──────────────────────────────────────────────────────────────────


class TestGroundTruth:
    def test_gt_fixation_is_zero(self, period_env):
        """Ground truth during fixation should be 0 (fixation action)."""
        period_env.reset()
        fix_steps = period_env._period_steps["fixation"]
        assert np.all(period_env.gt[:fix_steps] == 0)

    def test_gt_stimulus_is_correct_action(self, period_env):
        """Ground truth during stimulus should be correct action + 1."""
        period_env.reset()
        fix_steps = period_env._period_steps["fixation"]
        stim_steps = period_env._period_steps["stimulus"]

        gt_stim = period_env.gt[fix_steps : fix_steps + stim_steps]
        # Should be 1, 2, or 3 (1-indexed action)
        assert np.all(gt_stim >= 1)
        assert np.all(gt_stim <= 3)
        # All same value within trial
        assert len(np.unique(gt_stim)) == 1

    def test_gt_feedback_is_zero(self, period_env):
        """Ground truth during feedback should be 0 (fixation)."""
        period_env.reset()
        fb_start = period_env._feedback_start
        fb_steps = period_env._period_steps["feedback"]
        assert np.all(period_env.gt[fb_start : fb_start + fb_steps] == 0)


# ──────────────────────────────────────────────────────────────────
# Reward and trial boundaries
# ──────────────────────────────────────────────────────────────────


class TestFeedbackSignal:
    def test_feedback_channel_on_correct(self, period_env):
        """Feedback signal (obs[7]) should be 1.0 during feedback after correct response."""
        period_env.reset()
        fix_steps = period_env._period_steps["fixation"]
        stim_steps = period_env._period_steps["stimulus"]

        gt_action = int(period_env.gt[fix_steps])

        # Step through fixation
        for _ in range(fix_steps):
            period_env.step(0)
        # Respond correctly during stimulus
        for i in range(stim_steps):
            period_env.step(gt_action if i == 0 else 0)
        # First feedback step — should have feedback signal = 1.0
        obs, _, _, _, _ = period_env.step(0)
        assert obs[7] == 1.0, f"Feedback signal should be 1.0 after correct, got {obs[7]}"

    def test_feedback_channel_on_incorrect(self, period_env):
        """Feedback signal (obs[7]) should be 0.0 during feedback after incorrect response."""
        period_env.reset()
        fix_steps = period_env._period_steps["fixation"]
        stim_steps = period_env._period_steps["stimulus"]

        gt_action = int(period_env.gt[fix_steps])
        wrong_action = (gt_action % 3) + 1

        for _ in range(fix_steps):
            period_env.step(0)
        for i in range(stim_steps):
            period_env.step(wrong_action if i == 0 else 0)
        obs, _, _, _, _ = period_env.step(0)
        assert obs[7] == 0.0, f"Feedback signal should be 0.0 after incorrect, got {obs[7]}"

    def test_no_feedback_during_stimulus(self, period_env):
        """Feedback signal should be 0.0 during fixation and stimulus."""
        obs, _ = period_env.reset()
        fix_steps = period_env._period_steps["fixation"]
        stim_steps = period_env._period_steps["stimulus"]

        # Check all fixation + stimulus steps
        assert obs[7] == 0.0  # first fixation step
        for i in range(fix_steps + stim_steps - 1):
            obs, _, _, _, _ = period_env.step(0)
            assert obs[7] == 0.0, f"Feedback should be 0 at step {i+1}"


class TestRewardAndTrialBoundaries:
    def test_correct_response_gives_reward(self, period_env):
        """Correct response during stimulus should yield reward=1 at feedback."""
        period_env.reset()
        fix_steps = period_env._period_steps["fixation"]
        stim_steps = period_env._period_steps["stimulus"]
        feed_steps = period_env._period_steps["feedback"]

        # Get ground truth correct action
        gt_action = int(period_env.gt[fix_steps])  # 1-indexed

        total_reward = 0.0

        # Step through fixation
        for _ in range(fix_steps):
            _, r, _, _, _ = period_env.step(0)
            total_reward += r

        # Step through stimulus — respond correctly on first step
        for i in range(stim_steps):
            action = gt_action if i == 0 else 0
            _, r, _, _, _ = period_env.step(action)
            total_reward += r

        # Step through feedback
        for _ in range(feed_steps):
            _, r, _, _, _ = period_env.step(0)
            total_reward += r

        assert total_reward == 1.0

    def test_incorrect_response_gives_no_reward(self, period_env):
        """Incorrect response during stimulus should yield reward=0."""
        period_env.reset()
        fix_steps = period_env._period_steps["fixation"]
        stim_steps = period_env._period_steps["stimulus"]
        feed_steps = period_env._period_steps["feedback"]

        gt_action = int(period_env.gt[fix_steps])
        # Choose wrong action
        wrong_action = (gt_action % 3) + 1  # different from gt_action, still in 1-3

        total_reward = 0.0
        for _ in range(fix_steps):
            _, r, _, _, _ = period_env.step(0)
            total_reward += r
        for i in range(stim_steps):
            action = wrong_action if i == 0 else 0
            _, r, _, _, _ = period_env.step(action)
            total_reward += r
        for _ in range(feed_steps):
            _, r, _, _, _ = period_env.step(0)
            total_reward += r

        assert total_reward == 0.0

    def test_new_trial_flag(self, period_env):
        """info['new_trial'] should be True at trial boundaries."""
        obs, info = period_env.reset()
        assert info["new_trial"] is True

        # Step through an entire trial
        new_trial_seen = False
        for step in range(period_env.steps_per_trial):
            obs, _, _, _, info = period_env.step(0)
            if info["new_trial"]:
                new_trial_seen = True
                break

        assert new_trial_seen, "new_trial should be True after first trial ends"

    def test_episode_truncation(self):
        """Episode should truncate after max_trials."""
        from rlwm.envs import RLWMPeriodEnv

        env = RLWMPeriodEnv(set_size=2, max_trials=2, seed=42)
        env.reset()

        truncated = False
        for _ in range(env.steps_per_trial * 3):  # more than enough
            _, _, _, truncated, _ = env.step(0)
            if truncated:
                break

        assert truncated, "Should truncate after max_trials"


# ──────────────────────────────────────────────────────────────────
# Gymnasium registration
# ──────────────────────────────────────────────────────────────────


class TestGymnasiumRegistration:
    def test_make_period_env(self):
        import rlwm  # noqa: F401

        env = gym.make("rlwm/RLWM-Period-v0")
        obs, info = env.reset(seed=42)
        assert obs.shape == (8,)
        env.close()


# ──────────────────────────────────────────────────────────────────
# Custom timing
# ──────────────────────────────────────────────────────────────────


class TestCustomTiming:
    def test_custom_dt(self):
        from rlwm.envs import RLWMPeriodEnv

        env = RLWMPeriodEnv(dt=50, seed=42)
        # 500/50=10 + 2000/50=40 + 500/50=10 = 60 steps
        assert env.steps_per_trial == 60

    def test_custom_timing_dict(self):
        from rlwm.envs import RLWMPeriodEnv

        env = RLWMPeriodEnv(
            timing={"fixation": 200, "stimulus": 1000, "feedback": 300},
            dt=100,
            seed=42,
        )
        # 200/100=2 + 1000/100=10 + 300/100=3 = 15 steps
        assert env.steps_per_trial == 15
