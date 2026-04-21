"""Tests for the rlwm package: imports, gymnasium registration, canonical paths."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────
# Package imports
# ──────────────────────────────────────────────────────────────────


class TestPackageImports:
    def test_import_version(self):
        import rlwm

        assert hasattr(rlwm, "__version__")

    def test_import_config(self):
        from rlwm.config import ModelParams, TaskParams

        assert TaskParams.NUM_ACTIONS == 3
        assert ModelParams.BETA_FIXED == 50.0

    def test_import_envs(self):
        from rlwm.envs import RLWMEnv, RLWMPeriodEnv

        assert RLWMEnv is not None
        assert RLWMPeriodEnv is not None

    def test_import_models(self):
        from rlwm.models import QLearningAgent, WMRLHybridAgent

        assert QLearningAgent is not None
        assert WMRLHybridAgent is not None


# ──────────────────────────────────────────────────────────────────
# Gymnasium registration
# ──────────────────────────────────────────────────────────────────


class TestGymnasiumRegistration:
    def test_make_rlwm_v0(self):
        import rlwm  # noqa: F401  triggers registration

        env = gym.make("rlwm/RLWM-v0")
        obs, info = env.reset(seed=42)
        assert "stimulus" in obs
        assert "set_size" in obs
        env.close()

    def test_rlwm_v0_step(self):
        import rlwm  # noqa: F401

        env = gym.make("rlwm/RLWM-v0")
        obs, info = env.reset(seed=42)
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info2 = env.step(action)
        assert reward in (0.0, 1.0)
        assert isinstance(terminated, bool)
        env.close()

    def test_make_period_v0(self):
        import rlwm  # noqa: F401

        env = gym.make("rlwm/RLWM-Period-v0")
        obs, info = env.reset(seed=42)
        assert obs.shape == (8,)
        assert info["new_trial"] is True
        env.close()


# ──────────────────────────────────────────────────────────────────
# Canonical path tests (shims deleted — test rlwm.* directly)
# ──────────────────────────────────────────────────────────────────


class TestCanonicalPaths:
    def test_envs_rlwm_env_importable(self):
        from rlwm.envs.rlwm_env import RLWMEnv, create_rlwm_env

        assert RLWMEnv is not None
        assert create_rlwm_env is not None

    def test_envs_task_config_importable(self):
        from rlwm.envs.task_config import TaskSequenceLoader

        assert TaskSequenceLoader is not None

    def test_models_q_learning_importable(self):
        from rlwm.models.q_learning import QLearningAgent

        assert QLearningAgent is not None

    def test_models_wm_rl_hybrid_importable(self):
        from rlwm.models.wm_rl_hybrid import WMRLHybridAgent

        assert WMRLHybridAgent is not None

    def test_create_env_via_canonical_import(self):
        from rlwm.envs.rlwm_env import create_rlwm_env

        env = create_rlwm_env(set_size=3, seed=42)
        obs, info = env.reset()
        assert info["set_size"] == 3

    def test_create_agent_via_canonical_import(self):
        from rlwm.models.q_learning import create_q_learning_agent

        agent = create_q_learning_agent(alpha_pos=0.5, seed=42)
        assert agent.alpha_pos == 0.5


# ──────────────────────────────────────────────────────────────────
# Smoke test: agent + env interaction via package
# ──────────────────────────────────────────────────────────────────


class TestAgentEnvInteraction:
    def test_q_learning_on_rlwm(self):
        from rlwm.envs import create_rlwm_env
        from rlwm.models import QLearningAgent

        env = create_rlwm_env(set_size=3, seed=42)
        agent = QLearningAgent(alpha_pos=0.3, alpha_neg=0.1, beta=3.0, seed=42)

        obs, info = env.reset()
        for _ in range(20):
            stim = obs["stimulus"]
            action = agent.choose_action(stim)
            obs, reward, terminated, truncated, info = env.step(action)
            agent.update(stim, action, reward)
            if terminated or truncated:
                break

        assert len(agent.history["actions"]) == 0  # didn't call log_trial

    def test_wmrl_on_rlwm(self):
        from rlwm.envs import create_rlwm_env
        from rlwm.models import WMRLHybridAgent

        env = create_rlwm_env(set_size=5, seed=42)
        agent = WMRLHybridAgent(
            alpha_pos=0.3, alpha_neg=0.1, beta=2.0, capacity=4, seed=42
        )

        obs, info = env.reset()
        for _ in range(20):
            stim = obs["stimulus"]
            ss = obs["set_size"].item()
            action, _ = agent.choose_action(stim, ss)
            obs, reward, terminated, truncated, info = env.step(action)
            agent.update(stim, action, reward)
            if terminated or truncated:
                break
