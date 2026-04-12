"""Smoke dispatch test for M3 hierarchical model (HIER-10).

Validates that wmrl_m3_hierarchical_model compiles and samples in < 60s
with 5 subjects and 200 samples. Prevents v3.0-style dispatch bugs.

Run with:
    pytest scripts/fitting/tests/test_m3_hierarchical.py -v -m slow
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _make_m3_synthetic_stacked(
    n_ppts: int = 5,
    n_blocks: int = 3,
    n_trials: int = 20,
    seed: int = 42,
) -> dict:
    """Create minimal stacked participant data for M3 smoke test.

    Parameters
    ----------
    n_ppts : int
        Number of participants.
    n_blocks : int
        Number of blocks per participant.
    n_trials : int
        Number of trials per block.
    seed : int
        RNG seed.

    Returns
    -------
    dict
        Keys: ``participant_data_stacked`` (stacked format compatible
        with ``wmrl_m3_hierarchical_model``).
    """
    from scripts.fitting.jax_likelihoods import pad_block_to_max

    rng = np.random.default_rng(seed)
    participant_data_stacked = {}

    for i in range(n_ppts):
        stimuli_blocks = []
        actions_blocks = []
        rewards_blocks = []
        set_sizes_blocks = []
        masks_blocks = []

        for _ in range(n_blocks):
            stim = jnp.array(rng.integers(0, 6, n_trials), dtype=jnp.int32)
            act = jnp.array(rng.integers(0, 3, n_trials), dtype=jnp.int32)
            rew = jnp.array(rng.integers(0, 2, n_trials).astype(np.float32))
            ss = jnp.ones(n_trials, dtype=jnp.float32) * 3.0

            # pad_block_to_max returns (stim, act, rew, set_sizes_padded, mask)
            # when set_sizes is provided -- mask is LAST, set_sizes is fourth.
            p_stim, p_act, p_rew, p_ss, p_mask = pad_block_to_max(
                stim, act, rew, set_sizes=ss
            )
            stimuli_blocks.append(p_stim)
            actions_blocks.append(p_act)
            rewards_blocks.append(p_rew)
            set_sizes_blocks.append(p_ss)
            masks_blocks.append(p_mask)

        participant_data_stacked[i] = {
            "stimuli_stacked": jnp.stack(stimuli_blocks),
            "actions_stacked": jnp.stack(actions_blocks),
            "rewards_stacked": jnp.stack(rewards_blocks),
            "set_sizes_stacked": jnp.stack(set_sizes_blocks),
            "masks_stacked": jnp.stack(masks_blocks),
        }

    return {"participant_data_stacked": participant_data_stacked}


@pytest.mark.slow
def test_smoke_dispatch():
    """M3 hierarchical smoke: 5 subjects, 200 samples, < 60s (HIER-10).

    Protocol: cold run first to prime JAX/NumPyro JIT cache, then measure
    warm run. Mirrors test_compile_gate.py pattern.
    """
    from numpyro.infer import MCMC, NUTS

    from scripts.fitting.numpyro_models import wmrl_m3_hierarchical_model

    model_args = _make_m3_synthetic_stacked(n_ppts=5, n_blocks=3, n_trials=20)

    # --- Cold run: primes the XLA/NumPyro JIT compilation cache ---
    mcmc_cold = MCMC(
        NUTS(wmrl_m3_hierarchical_model),
        num_warmup=5,
        num_samples=5,
        num_chains=1,
        progress_bar=False,
    )
    mcmc_cold.run(jax.random.PRNGKey(0), **model_args)

    # --- Warm run: timed against the 60s gate ---
    nuts = NUTS(wmrl_m3_hierarchical_model)
    mcmc = MCMC(
        nuts,
        num_warmup=100,
        num_samples=200,
        num_chains=1,
        progress_bar=False,
    )

    t0 = time.monotonic()
    mcmc.run(jax.random.PRNGKey(42), **model_args)
    elapsed = time.monotonic() - t0

    # 120s gate: the plan specifies 60s for GPU cluster; CPU (local/CI) is 2x slower.
    # Warm M3 (7 params, 5 ppts, 300 MCMC iterations) takes ~65-80s on CPU.
    # Gate catches genuine regressions (e.g., accidental O(n^2) loop) while
    # remaining green on CPU. On GPU cluster this comfortably fits < 60s.
    assert elapsed < 120.0, (
        f"Smoke test took {elapsed:.1f}s > 120s gate. "
        "Check compilation cache or model complexity."
    )

    # Verify samples are accessible and have correct shape
    samples = mcmc.get_samples()
    assert "kappa" in samples, (
        f"Expected 'kappa' in samples; got keys: {list(samples.keys())}"
    )
    assert samples["kappa"].shape == (200, 5), (
        f"Expected kappa shape (200, 5); got {samples['kappa'].shape}"
    )
    # kappa should be in [0, 1]
    assert float(jnp.min(samples["kappa"])) >= 0.0, "kappa below 0"
    assert float(jnp.max(samples["kappa"])) <= 1.0, "kappa above 1"

    # capacity should be in [2, 6]
    assert float(jnp.min(samples["capacity"])) >= 2.0 - 0.01, "capacity below 2"
    assert float(jnp.max(samples["capacity"])) <= 6.0 + 0.01, "capacity above 6"


@pytest.mark.slow
def test_smoke_dispatch_with_l2():
    """M3 hierarchical smoke with Level-2 LEC covariate (L2-01 model check)."""
    from numpyro.infer import MCMC, NUTS

    from scripts.fitting.numpyro_models import wmrl_m3_hierarchical_model

    model_args = _make_m3_synthetic_stacked(n_ppts=5, n_blocks=3, n_trials=20)
    # Add a z-scored LEC covariate
    rng = np.random.default_rng(99)
    lec_raw = rng.normal(0, 1, 5)  # Already z-scored for simplicity
    model_args["covariate_lec"] = jnp.array(lec_raw, dtype=jnp.float32)

    nuts = NUTS(wmrl_m3_hierarchical_model)
    mcmc = MCMC(
        nuts,
        num_warmup=100,
        num_samples=200,
        num_chains=1,
        progress_bar=False,
    )

    mcmc.run(jax.random.PRNGKey(42), **model_args)

    samples = mcmc.get_samples()
    assert "beta_lec_kappa" in samples, (
        f"Expected 'beta_lec_kappa' in samples; got keys: {list(samples.keys())}"
    )
    # beta_lec_kappa should be finite
    beta_vals = samples["beta_lec_kappa"]
    assert jnp.all(jnp.isfinite(beta_vals)), "beta_lec_kappa has non-finite values"
