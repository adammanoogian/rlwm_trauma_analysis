"""
rlwm — Reinforcement Learning + Working Memory task environments and models.

Provides gymnasium-compatible environments and computational models for the
RLWM task (Senta et al., 2025).
"""

from __future__ import annotations

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Gymnasium environment registration
# ---------------------------------------------------------------------------
from gymnasium.envs.registration import register

register(
    id="rlwm/RLWM-v0",
    entry_point="rlwm.envs.rlwm_env:RLWMEnv",
)

register(
    id="rlwm/RLWM-Period-v0",
    entry_point="rlwm.envs.rlwm_period_env:RLWMPeriodEnv",
)
