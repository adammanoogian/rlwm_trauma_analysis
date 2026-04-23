"""Per-model JAX likelihoods + NumPyro hierarchical wrappers (Phase 29-08 vertical layout).

Each submodule here is self-contained for one Senta et al. (2025) model:

- :mod:`.qlearning` — M1 asymmetric Q-learning
- :mod:`.wmrl`     — M2 WM-RL hybrid
- :mod:`.wmrl_m3`  — M3 = M2 + global perseveration (kappa)
- :mod:`.wmrl_m5`  — M5 = M3 + RL forgetting (phi_rl)
- :mod:`.wmrl_m6a` — M6a = M2 + stimulus-specific perseveration (kappa_s)
- :mod:`.wmrl_m6b` — M6b = M2 + dual perseveration (kappa_total, kappa_share)
- :mod:`.wmrl_m4`  — M4 RLWM-LBA joint choice+RT (numpyro-only, no JAX likelihood)

Import each symbol directly from its per-model submodule. The legacy
``rlwm.fitting.jax_likelihoods`` and ``rlwm.fitting.numpyro_models``
re-export shims were deleted in the v5.0 shim cleanup.
"""
