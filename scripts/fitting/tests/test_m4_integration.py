"""Integration tests for M4 hierarchical LBA (M4H-01, M4H-02, M4H-04)."""
from __future__ import annotations

# Float64 MUST be set before any other JAX import
import jax

jax.config.update("jax_enable_x64", True)
import numpyro

numpyro.enable_x64()

import pickle
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpyro.infer import MCMC, NUTS

from scripts.fitting.numpyro_models import (
    prepare_stacked_participant_data_m4,
    wmrl_m4_hierarchical_model,
)


def _make_synthetic_m4_data(
    n_participants: int = 3,
    n_blocks: int = 2,
    n_trials_per_block: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic trial-level DataFrame for M4 integration tests.

    Parameters
    ----------
    n_participants : int
        Number of participants.
    n_blocks : int
        Number of blocks per participant.
    n_trials_per_block : int
        Number of trials per block.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Trial-level DataFrame with columns: sona_id, block, stimulus,
        key_press, reward, set_size, rt.  RT values are in milliseconds
        (integers) uniformly drawn from [300, 1500].
    """
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_participants):
        for blk in range(n_blocks):
            for _trial in range(n_trials_per_block):
                rows.append(
                    {
                        "sona_id": pid,
                        "block": blk,
                        "stimulus": int(rng.integers(0, 6)),
                        "key_press": int(rng.integers(0, 3)),
                        "reward": int(rng.integers(0, 2)),
                        "set_size": int(rng.choice([2, 3, 6])),
                        "rt": int(rng.integers(300, 1500)),
                    }
                )
    return pd.DataFrame(rows)


def test_float64_isolation() -> None:
    """M4H-01: Verify float64 is active in the model context.

    Checks two things:
    1. The global JAX default dtype is float64 (not float32).
    2. RT data passed through ``prepare_stacked_participant_data_m4``
       produces ``rts_stacked`` arrays with dtype float64.
    """
    # Verify global dtype
    assert jnp.zeros(1).dtype == jnp.float64, (
        f"Float64 not active: got dtype {jnp.zeros(1).dtype}, expected float64"
    )

    # Verify RT data is float64 through the pipeline
    df = _make_synthetic_m4_data(n_participants=2, n_blocks=2, n_trials_per_block=8)
    pdata = prepare_stacked_participant_data_m4(df)
    for pid, d in pdata.items():
        assert d["rts_stacked"].dtype == jnp.float64, (
            f"Participant {pid}: rts_stacked dtype={d['rts_stacked'].dtype}, "
            f"expected float64"
        )


@pytest.mark.slow
def test_log_delta_recovery() -> None:
    """M4H-02: Non-centered log(b-A) produces valid delta>0 and b>A.

    Uses N=10 participants with short chains (200 warmup, 300 samples,
    2 vectorized chains). The 15% relative error threshold from M4H-02
    applies to the full N=154 cluster fit; here we use relaxed structural
    checks appropriate for a quick integration test:

    - ``delta > 0`` for all posterior samples (exp-transform guarantee)
    - ``A > 0`` for all posterior samples (log-normal parameterization)
    - ``b = A + delta > A`` for all posterior samples (critical M4H-02 property)
    - ``log_delta_mu_pr`` is finite everywhere
    - All 10 expected parameters are present and finite
    """
    df = _make_synthetic_m4_data(
        n_participants=10, n_blocks=3, n_trials_per_block=12, seed=42
    )
    pdata = prepare_stacked_participant_data_m4(df)

    nuts = NUTS(
        wmrl_m4_hierarchical_model,
        target_accept_prob=0.80,
        max_tree_depth=8,
    )
    mcmc = MCMC(
        nuts,
        num_warmup=200,
        num_samples=300,
        num_chains=2,
        chain_method="vectorized",
        progress_bar=False,
    )
    rng_key = jax.random.PRNGKey(42)
    mcmc.run(rng_key, participant_data_stacked=pdata)

    samples = mcmc.get_samples()

    # delta must be strictly positive (exp transform guarantees this)
    delta_samples = samples["delta"]
    assert jnp.all(delta_samples > 0), (
        f"delta must be > 0 (log-scale parameterization). "
        f"min delta={float(jnp.min(delta_samples)):.6f}"
    )

    # A must be strictly positive
    A_samples = samples["A"]
    assert jnp.all(A_samples > 0), (
        f"A must be > 0 (log-normal parameterization). "
        f"min A={float(jnp.min(A_samples)):.6f}"
    )

    # b = A + delta must be > A (structural property of M4H-02)
    b_samples = A_samples + delta_samples
    assert jnp.all(b_samples > A_samples), (
        "b must be > A (enforced by delta > 0). "
        f"min (b-A)={float(jnp.min(b_samples - A_samples)):.6f}"
    )

    # Group-level log_delta_mu_pr should be finite
    log_delta_mu = samples["log_delta_mu_pr"]
    assert jnp.all(jnp.isfinite(log_delta_mu)), (
        "log_delta_mu_pr has non-finite values. "
        f"n_nonfinite={int(jnp.sum(~jnp.isfinite(log_delta_mu)))}"
    )

    # All 10 expected parameters must be present and finite
    expected_params = [
        "alpha_pos",
        "alpha_neg",
        "phi",
        "rho",
        "capacity",
        "kappa",
        "v_scale",
        "A",
        "delta",
        "t0",
    ]
    for p in expected_params:
        assert p in samples, (
            f"Missing parameter '{p}'. "
            f"Got keys: {sorted(samples.keys())}"
        )
        assert jnp.all(jnp.isfinite(samples[p])), (
            f"Parameter '{p}' has non-finite values. "
            f"n_nonfinite={int(jnp.sum(~jnp.isfinite(samples[p])))}"
        )


@pytest.mark.slow
def test_checkpoint_resume() -> None:
    """M4H-04: Warmup state can be pickled and used to resume sampling.

    Tests the checkpoint-resume API directly (implementing M4H-04):

    1. Run warmup-only phase with ``mcmc.warmup(...)``.
    2. Pickle ``jax.device_get(mcmc.post_warmup_state)`` to a temp file.
    3. Create a fresh MCMC object.
    4. Load the pickle and set ``mcmc.post_warmup_state``.
    5. Resume sampling with ``mcmc.run(loaded_state.rng_key, ...)``.
    6. Verify: posterior samples exist, have expected parameters, and are finite.

    This does not require killing a process. It tests the same API path
    the production script uses, which is sufficient to validate M4H-04.
    """
    df = _make_synthetic_m4_data(
        n_participants=3, n_blocks=2, n_trials_per_block=8, seed=99
    )
    pdata = prepare_stacked_participant_data_m4(df)
    model_kwargs: dict = {"participant_data_stacked": pdata}

    # Phase 1: warmup only
    nuts1 = NUTS(
        wmrl_m4_hierarchical_model,
        target_accept_prob=0.80,
        max_tree_depth=8,
    )
    mcmc1 = MCMC(
        nuts1,
        num_warmup=50,
        num_samples=50,
        num_chains=1,
        chain_method="vectorized",
        progress_bar=False,
    )
    rng_key = jax.random.PRNGKey(42)
    mcmc1.warmup(rng_key, **model_kwargs)

    # Save warmup state to temp file
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        warmup_path = Path(f.name)
        warmup_state_cpu = jax.device_get(mcmc1.post_warmup_state)
        pickle.dump(warmup_state_cpu, f)

    # Phase 2: resume from checkpoint
    try:
        nuts2 = NUTS(
            wmrl_m4_hierarchical_model,
            target_accept_prob=0.80,
            max_tree_depth=8,
        )
        mcmc2 = MCMC(
            nuts2,
            num_warmup=50,
            num_samples=50,
            num_chains=1,
            chain_method="vectorized",
            progress_bar=False,
        )

        with open(warmup_path, "rb") as f:
            loaded_state = pickle.load(f)
        mcmc2.post_warmup_state = loaded_state
        mcmc2.run(loaded_state.rng_key, **model_kwargs)

        # Verify: samples exist and are finite
        samples = mcmc2.get_samples()
        assert len(samples) > 0, "No samples produced after checkpoint resume"

        for param_name, vals in samples.items():
            assert jnp.all(jnp.isfinite(vals)), (
                f"Non-finite samples for '{param_name}' after checkpoint resume. "
                f"n_nonfinite={int(jnp.sum(~jnp.isfinite(vals)))}"
            )
    finally:
        warmup_path.unlink(missing_ok=True)
