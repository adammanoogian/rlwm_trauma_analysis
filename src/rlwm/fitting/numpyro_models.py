"""Legacy import path — canonical home moved in Phase 29-08.

This module is retained as a thin wildcard re-export shim so that every
pre-refactor consumer that imports via::

    from rlwm.fitting.numpyro_models import <symbol>

continues to work unchanged. The v4.0 closure invariants pinned to this
import path therefore remain green.

Canonical homes (new in Phase 29-08):

- Per-model NumPyro hierarchical wrappers live in
  :mod:`rlwm.fitting.models.<model>` (the same file as the model's JAX
  likelihood — vertical-by-model layout).
- Sampling orchestration (``run_inference``, ``samples_to_arviz``,
  chain-method selector, data-prep utilities, stacking helpers) lives
  in :mod:`rlwm.fitting.sampling`.

All public symbols from those modules (defined via their ``__all__`` lists)
are re-exported here via wildcard import.
"""
from __future__ import annotations

# Sampling orchestration
from .sampling import *  # noqa: F401,F403

# Per-model numpyro wrappers (and their JAX likelihoods — harmless double-export)
from .models.qlearning import *  # noqa: F401,F403
from .models.wmrl import *  # noqa: F401,F403
from .models.wmrl_m3 import *  # noqa: F401,F403
from .models.wmrl_m5 import *  # noqa: F401,F403
from .models.wmrl_m6a import *  # noqa: F401,F403
from .models.wmrl_m6b import *  # noqa: F401,F403
from .models.wmrl_m4 import *  # noqa: F401,F403
