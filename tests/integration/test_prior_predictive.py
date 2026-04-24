"""Smoke test for ``scripts/21_run_prior_predictive.py``.

Validates that ``numpyro.infer.Predictive`` can be instantiated against the
stacked hierarchical model dispatch and produces samples of the expected
individual-level parameter sites.  Kept minimal (N=3 ppts, 2 blocks, 20
trials, num_samples=10) so the whole test runs in under a few seconds on
CPU.  Part of the v4.0 Phase 21 prior-predictive gate.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rlwm.fitting.core import pad_block_to_max


def _make_synthetic_stacked(
    n_ppts: int = 3,
    n_blocks: int = 2,
    n_trials: int = 20,
    seed: int = 7,
) -> tuple[dict, dict]:
    """Create a minimal stacked-participant dict plus the (N,B,T) tensors.

    Returns
    -------
    (participant_data_stacked, stacked_arrays) : tuple of dicts
        Both are required by the hierarchical model signatures —
        ``participant_data_stacked`` supplies the per-participant shape,
        ``stacked_arrays`` supplies the (N, B, T) tensors consumed by
        the fully-batched likelihoods.
    """
    from rlwm.fitting.core import stack_across_participants

    rng = np.random.default_rng(seed)
    participant_data_stacked: dict = {}

    for pid in range(n_ppts):
        stim_blocks, act_blocks, rew_blocks, ss_blocks, mask_blocks = [], [], [], [], []
        for _ in range(n_blocks):
            stim = jnp.array(rng.integers(0, 6, n_trials), dtype=jnp.int32)
            act = jnp.array(rng.integers(0, 3, n_trials), dtype=jnp.int32)
            rew = jnp.array(rng.integers(0, 2, n_trials).astype(np.float32))
            ss = jnp.ones(n_trials, dtype=jnp.float32) * 3.0
            p_stim, p_act, p_rew, p_ss, p_mask = pad_block_to_max(
                stim, act, rew, set_sizes=ss
            )
            stim_blocks.append(p_stim)
            act_blocks.append(p_act)
            rew_blocks.append(p_rew)
            ss_blocks.append(p_ss)
            mask_blocks.append(p_mask)

        participant_data_stacked[pid] = {
            "stimuli_stacked": jnp.stack(stim_blocks),
            "actions_stacked": jnp.stack(act_blocks),
            "rewards_stacked": jnp.stack(rew_blocks),
            "set_sizes_stacked": jnp.stack(ss_blocks),
            "masks_stacked": jnp.stack(mask_blocks),
        }

    stacked_arrays = stack_across_participants(participant_data_stacked)
    return participant_data_stacked, stacked_arrays


def test_prior_predictive_wmrl_m3_smoke():
    """Predictive runs on wmrl_m3 and produces expected sample keys.

    Gate: ``alpha_pos`` and ``kappa`` both appear in the returned dict;
    first axis of ``alpha_pos`` equals ``num_samples``.  This protects
    against regressions in the NumPyro dispatch table (M3 was the model
    that triggered the v3 dispatch bug).
    """
    from numpyro.infer import Predictive

    from rlwm.fitting.models.wmrl_m3 import wmrl_m3_hierarchical_model

    ppt_data, stacked = _make_synthetic_stacked(n_ppts=3, n_blocks=2, n_trials=20)

    predictive = Predictive(wmrl_m3_hierarchical_model, num_samples=10)
    samples = predictive(
        jax.random.PRNGKey(0),
        participant_data_stacked=ppt_data,
        covariate_lec=None,
        stacked_arrays=stacked,
        use_pscan=False,
    )

    assert "alpha_pos" in samples, (
        f"Expected 'alpha_pos' in prior samples; got {sorted(samples.keys())}"
    )
    assert "kappa" in samples, (
        f"Expected 'kappa' in prior samples; got {sorted(samples.keys())}"
    )
    alpha_pos = np.asarray(samples["alpha_pos"])
    assert alpha_pos.shape[0] == 10, (
        f"Expected num_samples=10 on axis 0; got shape {alpha_pos.shape}"
    )
    assert alpha_pos.shape[1] == 3, (
        f"Expected n_participants=3 on axis 1; got shape {alpha_pos.shape}"
    )
    # Bounded in [0, 1]
    assert float(np.min(alpha_pos)) >= 0.0
    assert float(np.max(alpha_pos)) <= 1.0


def test_prior_predictive_gate_helper():
    """Pure unit test of the three-part gate evaluator."""
    from scripts import __init__  # noqa: F401 ensure pkg init
    import importlib.util
    from pathlib import Path

    # Load the 21_run_prior_predictive module by filepath (leading digit makes
    # normal imports illegal).
    mod_path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "bayesian_pipeline"
        / "21_run_prior_predictive.py"
    )
    spec = importlib.util.spec_from_file_location("prior_pred_mod", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # All plausible
    good = np.array([0.55, 0.60, 0.58, 0.62, 0.70])
    passed, metrics = mod._evaluate_gate(good)
    assert passed
    assert 0.40 <= metrics["median"] <= 0.90
    assert metrics["frac_below_chance"] == pytest.approx(0.0)
    assert metrics["frac_at_ceiling"] == pytest.approx(0.0)

    # Mostly ceiling
    bad_ceiling = np.array([0.99, 0.98, 0.97, 0.96, 0.50])
    passed_bad, metrics_bad = mod._evaluate_gate(bad_ceiling)
    assert not passed_bad
    assert metrics_bad["frac_at_ceiling"] >= 0.05
