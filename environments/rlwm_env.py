"""Backward-compatibility shim — delegates to rlwm.envs.rlwm_env."""

from __future__ import annotations

from rlwm.envs.rlwm_env import RLWMEnv, create_rlwm_env  # noqa: F401


def test_environment():
    """Test the RLWM environment with random actions."""
    env = create_rlwm_env(set_size=3, phase_type="main_task", seed=42)
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    print("Test complete!")


if __name__ == "__main__":
    test_environment()
