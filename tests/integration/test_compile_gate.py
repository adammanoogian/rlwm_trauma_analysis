"""Compile-time gate test for hierarchical Bayesian model compilation.

Tests that the warm (cached) invocation of the Q-learning hierarchical model
completes in under 60 seconds, validating INFRA-08.

Marked @pytest.mark.slow — excluded from fast CI runs. Run with:
    pytest scripts/fitting/tests/test_compile_gate.py -v -m slow

Cold compile on CPU is 30-90s; warm compile is typically < 15s + setup.
"""

from __future__ import annotations

import time

import numpy as np
import jax
import jax.numpy as jnp
import pytest

# ---------------------------------------------------------------------------
# Synthetic data helper
# ---------------------------------------------------------------------------

def _make_minimal_synthetic_data(
    n_ppts: int = 2,
    n_blocks: int = 2,
    n_trials: int = 20,
    seed: int = 42,
) -> dict:
    """Create minimal synthetic participant_data dict for model compilation.

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
        ``participant_data`` dict compatible with ``qlearning_hierarchical_model``.
    """
    rng = np.random.default_rng(seed)
    participant_data = {}

    for i in range(n_ppts):
        stimuli_blocks = []
        actions_blocks = []
        rewards_blocks = []

        for _ in range(n_blocks):
            stimuli_blocks.append(jnp.array(rng.integers(0, 3, n_trials), dtype=jnp.int32))
            actions_blocks.append(jnp.array(rng.integers(0, 3, n_trials), dtype=jnp.int32))
            rewards_blocks.append(jnp.array(rng.integers(0, 2, n_trials).astype(np.float32)))

        participant_data[i] = {
            "stimuli_blocks": stimuli_blocks,
            "actions_blocks": actions_blocks,
            "rewards_blocks": rewards_blocks,
        }

    return {"participant_data": participant_data}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_compile_gate():
    """Warm invocation of qlearning_hierarchical_model must complete in < 60s.

    Protocol:
    1. Cold run: first MCMC call triggers JAX/NumPyro JIT compilation.
    2. Warm run: second MCMC call reuses compiled kernels; must be < 60s.
    """
    from numpyro.infer import MCMC, NUTS
    from rlwm.fitting.models.qlearning import qlearning_hierarchical_model

    model_args = _make_minimal_synthetic_data(n_ppts=2, n_blocks=2, n_trials=20)

    # --- Cold run (primes the XLA/NumPyro compilation cache) ---
    mcmc_cold = MCMC(
        NUTS(qlearning_hierarchical_model),
        num_warmup=5,
        num_samples=5,
        num_chains=1,
        progress_bar=False,
    )
    mcmc_cold.run(jax.random.PRNGKey(0), **model_args)

    # --- Warm run (measures compiled execution time) ---
    t0 = time.monotonic()
    mcmc_warm = MCMC(
        NUTS(qlearning_hierarchical_model),
        num_warmup=5,
        num_samples=5,
        num_chains=1,
        progress_bar=False,
    )
    mcmc_warm.run(jax.random.PRNGKey(1), **model_args)
    elapsed = time.monotonic() - t0

    assert elapsed < 60.0, (
        f"Warm invocation took {elapsed:.1f}s, exceeding the 60s gate. "
        "This may indicate a JAX compilation cache miss or regression in model complexity. "
        "Check if the compilation cache directory is accessible and "
        "consider profiling with jax.profiler."
    )


@pytest.mark.slow
def test_compile_gate_samples_accessible():
    """After warm run, MCMC samples must be accessible and have correct shape."""
    from numpyro.infer import MCMC, NUTS
    from rlwm.fitting.models.qlearning import qlearning_hierarchical_model

    model_args = _make_minimal_synthetic_data(n_ppts=2, n_blocks=2, n_trials=20)

    mcmc = MCMC(
        NUTS(qlearning_hierarchical_model),
        num_warmup=5,
        num_samples=10,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(42), **model_args)

    samples = mcmc.get_samples()
    assert "alpha_pos" in samples, (
        f"Expected 'alpha_pos' in samples; got keys: {list(samples.keys())}"
    )
    alpha_pos = samples["alpha_pos"]
    # shape: (n_samples, n_participants) = (10, 2)
    assert alpha_pos.shape == (10, 2), (
        f"Expected alpha_pos shape (10, 2); got {alpha_pos.shape}"
    )
    # Values in [0, 1]
    assert float(jnp.min(alpha_pos)) >= 0.0, "alpha_pos below 0"
    assert float(jnp.max(alpha_pos)) <= 1.0, "alpha_pos above 1"
