"""Smoke test for GPU M4 (RLWM-LBA) fitting path."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.slow
def test_fit_all_gpu_m4_smoke(m4_synthetic_data_small):
    """Verify fit_all_gpu_m4 returns finite NLL without NaN or dtype errors.

    This is a GPU-01 acceptance test. It runs M4 MLE fitting on 5 synthetic
    participants with 5 random starts each. The test validates:

    - No NaN in NLL column
    - All NLLs are finite
    - Capacity estimates are within the [2.0, 6.0] Collins K bounds
    """
    import jax

    jax.config.update("jax_enable_x64", True)

    from rlwm.fitting.mle import fit_all_gpu_m4

    df, timing_info, timing_records = fit_all_gpu_m4(
        m4_synthetic_data_small,
        n_starts=5,
        seed=42,
        verbose=False,
    )

    # All 5 participants should be in the output
    assert len(df) == 5, (
        f"Expected 5 participants in output, got {len(df)}"
    )

    # NLL should be finite for all participants
    assert df["nll"].notna().all(), (
        f"NLL contains NaN: {df['nll'].tolist()}"
    )
    assert np.isfinite(df["nll"].values.astype(float)).all(), (
        f"NLL contains non-finite values: {df['nll'].tolist()}"
    )

    # Capacity should respect Collins K bounds [2.0, 6.0]
    caps = df["capacity"].values.astype(float)
    assert (caps >= 2.0).all(), (
        f"Capacity below lower bound 2.0: min={caps.min()}"
    )
    assert (caps <= 6.0).all(), (
        f"Capacity above upper bound 6.0: max={caps.max()}"
    )
