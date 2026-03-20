"""Backward-compatibility shim — delegates to rlwm.envs.task_config."""

from __future__ import annotations

from rlwm.envs.task_config import (  # noqa: F401
    TaskConfigGenerator,
    TaskSequenceLoader,
    generate_synthetic_config,
    load_task_sequence,
)
