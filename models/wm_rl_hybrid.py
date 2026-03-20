"""Backward-compatibility shim — delegates to rlwm.models.wm_rl_hybrid."""

from __future__ import annotations

from rlwm.models.wm_rl_hybrid import (  # noqa: F401
    WMRLHybridAgent,
    create_wm_rl_agent,
    simulate_wm_rl_on_env,
)
