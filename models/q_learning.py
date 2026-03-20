"""Backward-compatibility shim — delegates to rlwm.models.q_learning."""

from __future__ import annotations

from rlwm.models.q_learning import (  # noqa: F401
    QLearningAgent,
    create_q_learning_agent,
    simulate_agent_on_env,
)
