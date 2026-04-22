"""Legacy import path — canonical home moved in Phase 29-08.

This module is retained as a thin wildcard re-export shim so that every
pre-refactor consumer that imports via::

    from rlwm.fitting.jax_likelihoods import <symbol>

continues to work unchanged. The v4.0 closure invariants pinned to this
import path therefore remain green.

Canonical homes (new in Phase 29-08):

- Shared primitives (padding, softmax, epsilon, scans, perseveration
  precompute, module constants) live in :mod:`rlwm.fitting.core`.
- Per-model JAX likelihood variants (sequential + ``pscan``) live in
  :mod:`rlwm.fitting.models.<model>`.

All public symbols from those modules (defined via their ``__all__`` lists)
are re-exported here via wildcard import.
"""
from __future__ import annotations

# Shared primitives: padding/softmax/scans + module constants
from .core import *  # noqa: F401,F403

# Per-model JAX likelihoods (and their numpyro wrappers — harmless double-export)
from .models.qlearning import *  # noqa: F401,F403
from .models.wmrl import *  # noqa: F401,F403
from .models.wmrl_m3 import *  # noqa: F401,F403
from .models.wmrl_m5 import *  # noqa: F401,F403
from .models.wmrl_m6a import *  # noqa: F401,F403
from .models.wmrl_m6b import *  # noqa: F401,F403


if __name__ == "__main__":
    # Preserved from the pre-refactor ``python src/rlwm/fitting/jax_likelihoods.py``
    # smoke-test driver; the individual ``test_*`` functions now live in their
    # per-model files but are all re-exported here.
    from .models.qlearning import (
        test_multiblock,
        test_padding_equivalence_qlearning,
        test_single_block,
    )
    from .models.wmrl import (
        test_multiblock_padding_equivalence,
        test_padding_equivalence_wmrl,
        test_wmrl_multiblock,
        test_wmrl_single_block,
    )
    from .models.wmrl_m3 import (
        test_padding_equivalence_wmrl_m3,
        test_wmrl_m3_backward_compatibility,
        test_wmrl_m3_single_block,
    )
    from .models.wmrl_m5 import (
        test_padding_equivalence_wmrl_m5,
        test_wmrl_m5_backward_compatibility,
        test_wmrl_m5_single_block,
    )
    from .models.wmrl_m6a import (
        test_padding_equivalence_wmrl_m6a,
        test_wmrl_m6a_per_stimulus_tracking,
        test_wmrl_m6a_single_block,
    )
    from .models.wmrl_m6b import (
        test_padding_equivalence_wmrl_m6b,
        test_wmrl_m6b_kappa_share_one_matches_m3,
        test_wmrl_m6b_kappa_share_zero_matches_m6a,
        test_wmrl_m6b_single_block,
    )

    print("=" * 80)
    print("JAX Q-LEARNING LIKELIHOOD TESTS")
    print("=" * 80)

    test_single_block()
    test_multiblock()

    print("\n" + "=" * 80)
    print("JAX WM-RL LIKELIHOOD TESTS")
    print("=" * 80)

    test_wmrl_single_block()
    test_wmrl_multiblock()

    print("\n" + "=" * 80)
    print("JAX WM-RL M3 (PERSEVERATION) LIKELIHOOD TESTS")
    print("=" * 80)

    test_wmrl_m3_single_block()
    test_wmrl_m3_backward_compatibility()

    print("\n" + "=" * 80)
    print("JAX WM-RL M5 (RL FORGETTING) LIKELIHOOD TESTS")
    print("=" * 80)

    test_wmrl_m5_single_block()
    test_wmrl_m5_backward_compatibility()

    print("\n" + "=" * 80)
    print("JAX WM-RL M6a (STIMULUS-SPECIFIC PERSEVERATION) LIKELIHOOD TESTS")
    print("=" * 80)

    test_wmrl_m6a_single_block()
    test_wmrl_m6a_per_stimulus_tracking()

    print("\n" + "=" * 80)
    print("JAX WM-RL M6b (DUAL PERSEVERATION) LIKELIHOOD TESTS")
    print("=" * 80)

    test_wmrl_m6b_single_block()
    test_wmrl_m6b_kappa_share_one_matches_m3()
    test_wmrl_m6b_kappa_share_zero_matches_m6a()

    print("\n" + "=" * 80)
    print("PADDING EQUIVALENCE TESTS (CRITICAL)")
    print("=" * 80)

    test_padding_equivalence_qlearning()
    test_padding_equivalence_wmrl()
    test_padding_equivalence_wmrl_m3()
    test_padding_equivalence_wmrl_m5()
    test_padding_equivalence_wmrl_m6a()
    test_padding_equivalence_wmrl_m6b()
    test_multiblock_padding_equivalence()

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
