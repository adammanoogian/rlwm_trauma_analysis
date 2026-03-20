"""Backward-compatibility shim — delegates to rlwm.models."""

from __future__ import annotations

from rlwm.models import (  # noqa: F401
    QLearningAgent,
    WMRLHybridAgent,
    create_q_learning_agent,
    create_wm_rl_agent,
    simulate_agent_on_env,
    simulate_wm_rl_on_env,
)
