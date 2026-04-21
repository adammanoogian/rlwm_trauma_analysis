"""Tests for M4 hierarchical model components."""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import numpyro

numpyro.enable_x64()

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from rlwm.fitting.jax_likelihoods import MAX_TRIALS_PER_BLOCK
from rlwm.fitting.numpyro_models import (
    prepare_stacked_participant_data_m4,
    wmrl_m4_hierarchical_model,
)


def _make_synthetic_df(
    n_ppts: int = 3,
    n_blocks: int = 2,
    n_trials: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic trial-level DataFrame for M4 tests.

    Parameters
    ----------
    n_ppts : int
        Number of participants.
    n_blocks : int
        Number of blocks per participant.
    n_trials : int
        Number of trials per block.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Trial-level DataFrame with columns: sona_id, block, stimulus,
        key_press, reward, set_size, rt.  One RT outlier (100ms) is placed
        on the first trial of the first block for each participant.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_ppts):
        for blk in range(n_blocks):
            for trial in range(n_trials):
                # Place one outlier RT (100ms) on trial 0 of block 0
                if blk == 0 and trial == 0:
                    rt_val = 100  # below 150ms threshold -> outlier
                else:
                    rt_val = int(rng.integers(400, 1200))
                rows.append(
                    {
                        "sona_id": pid,
                        "block": blk,
                        "stimulus": int(rng.integers(0, 6)),
                        "key_press": int(rng.integers(0, 3)),
                        "reward": int(rng.integers(0, 2)),
                        "set_size": int(rng.choice([2, 3, 6])),
                        "rt": rt_val,
                    }
                )
    return pd.DataFrame(rows)


def test_prepare_stacked_data_m4_rts():
    """RT stacking: float64 seconds, correct shape, outlier mask applied.

    Verifies:
    - ``rts_stacked`` key present in each participant dict
    - dtype is float64
    - shape is (n_blocks, MAX_TRIALS_PER_BLOCK)
    - all RT values are in seconds (max < 3.0)
    - ``masks_stacked`` has a zero where the RT outlier occurred
    - ``masks_stacked`` has zeros for padding positions beyond n_trials
    """
    n_blocks = 2
    n_trials = 10
    df = _make_synthetic_df(n_ppts=3, n_blocks=n_blocks, n_trials=n_trials)
    result = prepare_stacked_participant_data_m4(df)

    assert len(result) == 3, f"Expected 3 participants, got {len(result)}"

    for pid, pdata in result.items():
        # rts_stacked key must exist
        assert "rts_stacked" in pdata, (
            f"Participant {pid}: missing 'rts_stacked' key. "
            f"Got keys: {list(pdata.keys())}"
        )

        rts = pdata["rts_stacked"]
        masks = pdata["masks_stacked"]

        # dtype must be float64
        assert rts.dtype == jnp.float64, (
            f"Participant {pid}: rts_stacked.dtype={rts.dtype}, "
            f"expected float64"
        )

        # shape must be (n_blocks, MAX_TRIALS_PER_BLOCK)
        assert rts.shape == (n_blocks, MAX_TRIALS_PER_BLOCK), (
            f"Participant {pid}: rts_stacked.shape={rts.shape}, "
            f"expected ({n_blocks}, {MAX_TRIALS_PER_BLOCK})"
        )

        # RT values in seconds: real trials should all be < 3.0 s
        real_rts = rts[:, :n_trials]
        assert float(real_rts.max()) < 3.0, (
            f"Participant {pid}: RT max={float(real_rts.max()):.3f}s >= 3.0s. "
            "Expected values in seconds, not milliseconds."
        )

        # The RT outlier is on trial 0 of block 0 (100ms < 150ms threshold)
        # masks_stacked must be 0.0 at that position
        outlier_mask_val = float(masks[0, 0])
        assert outlier_mask_val == 0.0, (
            f"Participant {pid}: mask[0,0]={outlier_mask_val}, "
            f"expected 0.0 (RT outlier at 100ms should be masked out)"
        )

        # Padding positions (beyond n_trials) must be 0.0
        if MAX_TRIALS_PER_BLOCK > n_trials:
            pad_vals = masks[:, n_trials:]
            assert float(pad_vals.max()) == 0.0, (
                f"Participant {pid}: padding mask positions have non-zero values. "
                f"max={float(pad_vals.max())}"
            )


def test_prepare_stacked_data_m4_sorted_participants():
    """Participant keys are returned in sorted order.

    Verifies that result.keys() equals sorted(result.keys()) so downstream
    covariate arrays (participants in sorted order) align correctly.
    """
    # Create DataFrame with participant IDs in reverse order
    rng = np.random.default_rng(7)
    rows = []
    for pid in [5, 3, 1, 4, 2]:
        for blk in range(2):
            for trial in range(8):
                rows.append(
                    {
                        "sona_id": pid,
                        "block": blk,
                        "stimulus": int(rng.integers(0, 6)),
                        "key_press": int(rng.integers(0, 3)),
                        "reward": int(rng.integers(0, 2)),
                        "set_size": 3,
                        "rt": int(rng.integers(400, 1200)),
                    }
                )
    df = pd.DataFrame(rows)

    result = prepare_stacked_participant_data_m4(df)

    keys = list(result.keys())
    assert keys == sorted(keys), (
        f"Participant keys are not sorted. Got: {keys}, "
        f"expected: {sorted(keys)}"
    )


@pytest.mark.slow
def test_m4_model_smoke():
    """M4 hierarchical model compiles under NUTS and produces finite samples.

    Smoke test: 3 participants, 2 blocks, 10 trials, 5 warmup/samples.
    Verifies:
    - No exceptions raised during NUTS compilation + sampling
    - All 10 parameter keys present in posterior samples
    - All sample values are finite
    """
    from numpyro.infer import MCMC, NUTS

    df = _make_synthetic_df(n_ppts=3, n_blocks=2, n_trials=10, seed=0)
    participant_data_stacked = prepare_stacked_participant_data_m4(df)

    nuts = NUTS(wmrl_m4_hierarchical_model)
    mcmc = MCMC(
        nuts,
        num_warmup=5,
        num_samples=5,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(
        jax.random.PRNGKey(42),
        participant_data_stacked=participant_data_stacked,
    )

    samples = mcmc.get_samples()

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
    for param in expected_params:
        assert param in samples, (
            f"Expected parameter '{param}' in samples. "
            f"Got keys: {sorted(samples.keys())}"
        )
        param_samples = samples[param]
        assert jnp.all(jnp.isfinite(param_samples)), (
            f"Parameter '{param}' has non-finite samples. "
            f"min={float(param_samples.min()):.4f}, "
            f"max={float(param_samples.max()):.4f}"
        )
