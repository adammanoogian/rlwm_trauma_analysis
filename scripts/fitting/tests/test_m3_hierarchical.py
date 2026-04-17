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


def test_wmrl_m3_fully_batched_matches_sequential():
    """Fully-batched vmap'd M3 likelihood agrees with sequential.

    Generates N=5 synthetic participants with variable n_blocks
    (12 or 17) and 100-trial blocks. For 3 random parameter draws,
    computes total NLL both ways and asserts relative error < 1e-4.

    This is the correctness gate for the vmap refactor (Task 4/5).
    """
    import pandas as pd
    from scripts.fitting.numpyro_models import (
        prepare_stacked_participant_data,
        stack_across_participants,
    )
    from scripts.fitting.jax_likelihoods import (
        wmrl_m3_fully_batched_likelihood,
        wmrl_m3_multiblock_likelihood_stacked,
    )

    rng = np.random.default_rng(42)

    # N=5 participants; n_blocks in {12, 17}; T = full MAX_TRIALS_PER_BLOCK=100
    ppt_configs = [("P0", 12), ("P1", 17), ("P2", 12), ("P3", 17), ("P4", 13)]
    rows = []
    for pid, n_blocks in ppt_configs:
        for b in range(n_blocks):
            trials_in_block = int(rng.integers(60, 100))
            for t in range(trials_in_block):
                rows.append({
                    "sona_id": pid,
                    "block": b,
                    "stimulus": int(rng.integers(0, 6)),
                    "key_press": int(rng.integers(0, 3)),
                    "reward": float(rng.integers(0, 2)),
                    "set_size": float(rng.choice([2.0, 3.0, 6.0])),
                })
    df = pd.DataFrame(rows)

    pdata = prepare_stacked_participant_data(df)
    stacked = stack_across_participants(pdata)
    N = len(stacked["participant_ids"])
    assert N == 5

    for draw_idx in range(3):
        # Random but reasonable parameter draws
        key = jax.random.PRNGKey(1000 + draw_idx)
        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
        alpha_pos = jax.random.uniform(k1, (N,), minval=0.1, maxval=0.9)
        alpha_neg = jax.random.uniform(k2, (N,), minval=0.1, maxval=0.9)
        phi       = jax.random.uniform(k3, (N,), minval=0.05, maxval=0.5)
        rho       = jax.random.uniform(k4, (N,), minval=0.2, maxval=0.9)
        capacity  = jax.random.uniform(k5, (N,), minval=2.0, maxval=6.0)
        kappa     = jax.random.uniform(k6, (N,), minval=0.0, maxval=0.5)
        epsilon   = jax.random.uniform(k7, (N,), minval=0.01, maxval=0.1)

        # Path A: fully-batched
        batched_ll = wmrl_m3_fully_batched_likelihood(
            stimuli=stacked["stimuli"],
            actions=stacked["actions"],
            rewards=stacked["rewards"],
            set_sizes=stacked["set_sizes"],
            masks=stacked["masks"],
            alpha_pos=alpha_pos, alpha_neg=alpha_neg, phi=phi,
            rho=rho, capacity=capacity, kappa=kappa, epsilon=epsilon,
        )
        # Shape check
        assert batched_ll.shape == (N,), (
            f"Expected shape (N,)=({N},); got {batched_ll.shape}"
        )

        # Path B: sequential per-participant
        seq_lls = []
        for idx, pid in enumerate(stacked["participant_ids"]):
            pp = pdata[pid]
            ll_i = wmrl_m3_multiblock_likelihood_stacked(
                stimuli_stacked=pp["stimuli_stacked"],
                actions_stacked=pp["actions_stacked"],
                rewards_stacked=pp["rewards_stacked"],
                set_sizes_stacked=pp["set_sizes_stacked"],
                masks_stacked=pp["masks_stacked"],
                alpha_pos=float(alpha_pos[idx]),
                alpha_neg=float(alpha_neg[idx]),
                phi=float(phi[idx]),
                rho=float(rho[idx]),
                capacity=float(capacity[idx]),
                kappa=float(kappa[idx]),
                epsilon=float(epsilon[idx]),
            )
            seq_lls.append(float(ll_i))

        batched_np = np.array(batched_ll)
        seq_np = np.array(seq_lls, dtype=np.float32)

        # Per-participant relative error
        rel_err = np.abs(batched_np - seq_np) / (np.abs(seq_np) + 1e-9)
        max_rel_err = float(rel_err.max())
        print(f"draw {draw_idx}: max_rel_err = {max_rel_err:.2e}")
        assert max_rel_err < 1e-4, (
            f"Agreement failed: draw={draw_idx}, "
            f"expected max_rel_err < 1e-4, got {max_rel_err:.2e}. "
            f"batched={batched_np}, seq={seq_np}"
        )


def test_qlearning_fully_batched_matches_sequential():
    """Fully-batched vmap'd Q-learning likelihood agrees with sequential.

    Mirrors ``test_wmrl_m3_fully_batched_matches_sequential`` but for M1.
    Q-learning has only 3 free parameters (alpha_pos, alpha_neg, epsilon)
    and does not use set_sizes.

    Generates N=5 synthetic participants with variable n_blocks (12 or 17)
    and variable block lengths.  For 3 random parameter draws, computes
    total log-lik both ways and asserts per-participant relative error
    < 1e-4.  This is the correctness gate for the Issue 1 qlearning
    rollout.
    """
    import pandas as pd
    from scripts.fitting.numpyro_models import (
        prepare_stacked_participant_data,
        stack_across_participants,
    )
    from scripts.fitting.jax_likelihoods import (
        q_learning_fully_batched_likelihood,
        q_learning_multiblock_likelihood_stacked,
    )

    rng = np.random.default_rng(42)

    # Same synthetic design as M3 test for consistency.
    ppt_configs = [("P0", 12), ("P1", 17), ("P2", 12), ("P3", 17), ("P4", 13)]
    rows = []
    for pid, n_blocks in ppt_configs:
        for b in range(n_blocks):
            trials_in_block = int(rng.integers(60, 100))
            for t in range(trials_in_block):
                rows.append({
                    "sona_id": pid,
                    "block": b,
                    "stimulus": int(rng.integers(0, 6)),
                    "key_press": int(rng.integers(0, 3)),
                    "reward": float(rng.integers(0, 2)),
                    "set_size": float(rng.choice([2.0, 3.0, 6.0])),
                })
    df = pd.DataFrame(rows)

    pdata = prepare_stacked_participant_data(df)
    stacked = stack_across_participants(pdata)
    N = len(stacked["participant_ids"])
    assert N == 5

    for draw_idx in range(3):
        key = jax.random.PRNGKey(2000 + draw_idx)
        k1, k2, k3 = jax.random.split(key, 3)
        alpha_pos = jax.random.uniform(k1, (N,), minval=0.1, maxval=0.9)
        alpha_neg = jax.random.uniform(k2, (N,), minval=0.1, maxval=0.9)
        epsilon   = jax.random.uniform(k3, (N,), minval=0.01, maxval=0.1)

        # Path A: fully-batched
        batched_ll = q_learning_fully_batched_likelihood(
            stimuli=stacked["stimuli"],
            actions=stacked["actions"],
            rewards=stacked["rewards"],
            masks=stacked["masks"],
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            epsilon=epsilon,
        )
        assert batched_ll.shape == (N,), (
            f"Expected shape (N,)=({N},); got {batched_ll.shape}"
        )

        # Path B: sequential per-participant
        seq_lls = []
        for idx, pid in enumerate(stacked["participant_ids"]):
            pp = pdata[pid]
            ll_i = q_learning_multiblock_likelihood_stacked(
                stimuli_stacked=pp["stimuli_stacked"],
                actions_stacked=pp["actions_stacked"],
                rewards_stacked=pp["rewards_stacked"],
                masks_stacked=pp["masks_stacked"],
                alpha_pos=float(alpha_pos[idx]),
                alpha_neg=float(alpha_neg[idx]),
                epsilon=float(epsilon[idx]),
            )
            seq_lls.append(float(ll_i))

        batched_np = np.array(batched_ll)
        seq_np = np.array(seq_lls, dtype=np.float32)

        rel_err = np.abs(batched_np - seq_np) / (np.abs(seq_np) + 1e-9)
        max_rel_err = float(rel_err.max())
        print(f"qlearning draw {draw_idx}: max_rel_err = {max_rel_err:.2e}")
        assert max_rel_err < 1e-4, (
            f"Agreement failed: draw={draw_idx}, "
            f"expected max_rel_err < 1e-4, got {max_rel_err:.2e}. "
            f"batched={batched_np}, seq={seq_np}"
        )


def _make_wmrl_family_synthetic():
    """Shared synthetic data builder for wmrl family (M2/M5/M6a/M6b) tests.

    Returns (pdata, stacked, N) where stacked has (N, B, T) arrays with N=5,
    B in {12, 17}, and variable block lengths.  Matches the M3 test design.
    """
    import pandas as pd
    from scripts.fitting.numpyro_models import (
        prepare_stacked_participant_data,
        stack_across_participants,
    )

    rng = np.random.default_rng(42)
    ppt_configs = [("P0", 12), ("P1", 17), ("P2", 12), ("P3", 17), ("P4", 13)]
    rows = []
    for pid, n_blocks in ppt_configs:
        for b in range(n_blocks):
            trials_in_block = int(rng.integers(60, 100))
            for t in range(trials_in_block):
                rows.append({
                    "sona_id": pid,
                    "block": b,
                    "stimulus": int(rng.integers(0, 6)),
                    "key_press": int(rng.integers(0, 3)),
                    "reward": float(rng.integers(0, 2)),
                    "set_size": float(rng.choice([2.0, 3.0, 6.0])),
                })
    df = pd.DataFrame(rows)
    pdata = prepare_stacked_participant_data(df)
    stacked = stack_across_participants(pdata)
    return pdata, stacked, len(stacked["participant_ids"])


def test_wmrl_fully_batched_matches_sequential():
    """Fully-batched vmap'd M2 (WM-RL) likelihood agrees with sequential.

    6 per-participant params: alpha_pos, alpha_neg, phi, rho, capacity, epsilon.
    No kappa (M2 has no perseveration).
    """
    from scripts.fitting.jax_likelihoods import (
        wmrl_fully_batched_likelihood,
        wmrl_multiblock_likelihood_stacked,
    )

    pdata, stacked, N = _make_wmrl_family_synthetic()
    assert N == 5

    for draw_idx in range(3):
        key = jax.random.PRNGKey(3000 + draw_idx)
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        alpha_pos = jax.random.uniform(k1, (N,), minval=0.1, maxval=0.9)
        alpha_neg = jax.random.uniform(k2, (N,), minval=0.1, maxval=0.9)
        phi       = jax.random.uniform(k3, (N,), minval=0.05, maxval=0.5)
        rho       = jax.random.uniform(k4, (N,), minval=0.2, maxval=0.9)
        capacity  = jax.random.uniform(k5, (N,), minval=2.0, maxval=6.0)
        epsilon   = jax.random.uniform(k6, (N,), minval=0.01, maxval=0.1)

        batched_ll = wmrl_fully_batched_likelihood(
            stimuli=stacked["stimuli"],
            actions=stacked["actions"],
            rewards=stacked["rewards"],
            set_sizes=stacked["set_sizes"],
            masks=stacked["masks"],
            alpha_pos=alpha_pos, alpha_neg=alpha_neg, phi=phi,
            rho=rho, capacity=capacity, epsilon=epsilon,
        )
        assert batched_ll.shape == (N,)

        seq_lls = []
        for idx, pid in enumerate(stacked["participant_ids"]):
            pp = pdata[pid]
            ll_i = wmrl_multiblock_likelihood_stacked(
                stimuli_stacked=pp["stimuli_stacked"],
                actions_stacked=pp["actions_stacked"],
                rewards_stacked=pp["rewards_stacked"],
                set_sizes_stacked=pp["set_sizes_stacked"],
                masks_stacked=pp["masks_stacked"],
                alpha_pos=float(alpha_pos[idx]),
                alpha_neg=float(alpha_neg[idx]),
                phi=float(phi[idx]),
                rho=float(rho[idx]),
                capacity=float(capacity[idx]),
                epsilon=float(epsilon[idx]),
            )
            seq_lls.append(float(ll_i))

        batched_np = np.array(batched_ll)
        seq_np = np.array(seq_lls, dtype=np.float32)
        rel_err = np.abs(batched_np - seq_np) / (np.abs(seq_np) + 1e-9)
        max_rel_err = float(rel_err.max())
        print(f"wmrl draw {draw_idx}: max_rel_err = {max_rel_err:.2e}")
        assert max_rel_err < 1e-4, (
            f"wmrl agreement failed: draw={draw_idx}, max_rel_err={max_rel_err:.2e}. "
            f"batched={batched_np}, seq={seq_np}"
        )


def test_wmrl_m5_fully_batched_matches_sequential():
    """Fully-batched vmap'd M5 (WM-RL+phi_rl) likelihood agrees with sequential.

    8 per-participant params: alpha_pos, alpha_neg, phi, rho, capacity,
    kappa, phi_rl, epsilon.
    """
    from scripts.fitting.jax_likelihoods import (
        wmrl_m5_fully_batched_likelihood,
        wmrl_m5_multiblock_likelihood_stacked,
    )

    pdata, stacked, N = _make_wmrl_family_synthetic()
    assert N == 5

    for draw_idx in range(3):
        key = jax.random.PRNGKey(4000 + draw_idx)
        k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)
        alpha_pos = jax.random.uniform(k1, (N,), minval=0.1, maxval=0.9)
        alpha_neg = jax.random.uniform(k2, (N,), minval=0.1, maxval=0.9)
        phi       = jax.random.uniform(k3, (N,), minval=0.05, maxval=0.5)
        rho       = jax.random.uniform(k4, (N,), minval=0.2, maxval=0.9)
        capacity  = jax.random.uniform(k5, (N,), minval=2.0, maxval=6.0)
        kappa     = jax.random.uniform(k6, (N,), minval=0.0, maxval=0.5)
        phi_rl    = jax.random.uniform(k7, (N,), minval=0.0, maxval=0.3)
        epsilon   = jax.random.uniform(k8, (N,), minval=0.01, maxval=0.1)

        batched_ll = wmrl_m5_fully_batched_likelihood(
            stimuli=stacked["stimuli"],
            actions=stacked["actions"],
            rewards=stacked["rewards"],
            set_sizes=stacked["set_sizes"],
            masks=stacked["masks"],
            alpha_pos=alpha_pos, alpha_neg=alpha_neg, phi=phi,
            rho=rho, capacity=capacity, kappa=kappa, phi_rl=phi_rl,
            epsilon=epsilon,
        )
        assert batched_ll.shape == (N,)

        seq_lls = []
        for idx, pid in enumerate(stacked["participant_ids"]):
            pp = pdata[pid]
            ll_i = wmrl_m5_multiblock_likelihood_stacked(
                stimuli_stacked=pp["stimuli_stacked"],
                actions_stacked=pp["actions_stacked"],
                rewards_stacked=pp["rewards_stacked"],
                set_sizes_stacked=pp["set_sizes_stacked"],
                masks_stacked=pp["masks_stacked"],
                alpha_pos=float(alpha_pos[idx]),
                alpha_neg=float(alpha_neg[idx]),
                phi=float(phi[idx]),
                rho=float(rho[idx]),
                capacity=float(capacity[idx]),
                kappa=float(kappa[idx]),
                phi_rl=float(phi_rl[idx]),
                epsilon=float(epsilon[idx]),
            )
            seq_lls.append(float(ll_i))

        batched_np = np.array(batched_ll)
        seq_np = np.array(seq_lls, dtype=np.float32)
        rel_err = np.abs(batched_np - seq_np) / (np.abs(seq_np) + 1e-9)
        max_rel_err = float(rel_err.max())
        print(f"wmrl_m5 draw {draw_idx}: max_rel_err = {max_rel_err:.2e}")
        assert max_rel_err < 1e-4, (
            f"wmrl_m5 agreement failed: draw={draw_idx}, "
            f"max_rel_err={max_rel_err:.2e}. batched={batched_np}, seq={seq_np}"
        )


def test_wmrl_m6a_fully_batched_matches_sequential():
    """Fully-batched vmap'd M6a (WM-RL+kappa_s) likelihood agrees with sequential.

    7 per-participant params: alpha_pos, alpha_neg, phi, rho, capacity,
    kappa_s, epsilon.
    """
    from scripts.fitting.jax_likelihoods import (
        wmrl_m6a_fully_batched_likelihood,
        wmrl_m6a_multiblock_likelihood_stacked,
    )

    pdata, stacked, N = _make_wmrl_family_synthetic()
    assert N == 5

    for draw_idx in range(3):
        key = jax.random.PRNGKey(5000 + draw_idx)
        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
        alpha_pos = jax.random.uniform(k1, (N,), minval=0.1, maxval=0.9)
        alpha_neg = jax.random.uniform(k2, (N,), minval=0.1, maxval=0.9)
        phi       = jax.random.uniform(k3, (N,), minval=0.05, maxval=0.5)
        rho       = jax.random.uniform(k4, (N,), minval=0.2, maxval=0.9)
        capacity  = jax.random.uniform(k5, (N,), minval=2.0, maxval=6.0)
        kappa_s   = jax.random.uniform(k6, (N,), minval=0.0, maxval=0.5)
        epsilon   = jax.random.uniform(k7, (N,), minval=0.01, maxval=0.1)

        batched_ll = wmrl_m6a_fully_batched_likelihood(
            stimuli=stacked["stimuli"],
            actions=stacked["actions"],
            rewards=stacked["rewards"],
            set_sizes=stacked["set_sizes"],
            masks=stacked["masks"],
            alpha_pos=alpha_pos, alpha_neg=alpha_neg, phi=phi,
            rho=rho, capacity=capacity, kappa_s=kappa_s, epsilon=epsilon,
        )
        assert batched_ll.shape == (N,)

        seq_lls = []
        for idx, pid in enumerate(stacked["participant_ids"]):
            pp = pdata[pid]
            ll_i = wmrl_m6a_multiblock_likelihood_stacked(
                stimuli_stacked=pp["stimuli_stacked"],
                actions_stacked=pp["actions_stacked"],
                rewards_stacked=pp["rewards_stacked"],
                set_sizes_stacked=pp["set_sizes_stacked"],
                masks_stacked=pp["masks_stacked"],
                alpha_pos=float(alpha_pos[idx]),
                alpha_neg=float(alpha_neg[idx]),
                phi=float(phi[idx]),
                rho=float(rho[idx]),
                capacity=float(capacity[idx]),
                kappa_s=float(kappa_s[idx]),
                epsilon=float(epsilon[idx]),
            )
            seq_lls.append(float(ll_i))

        batched_np = np.array(batched_ll)
        seq_np = np.array(seq_lls, dtype=np.float32)
        rel_err = np.abs(batched_np - seq_np) / (np.abs(seq_np) + 1e-9)
        max_rel_err = float(rel_err.max())
        print(f"wmrl_m6a draw {draw_idx}: max_rel_err = {max_rel_err:.2e}")
        assert max_rel_err < 1e-4, (
            f"wmrl_m6a agreement failed: draw={draw_idx}, "
            f"max_rel_err={max_rel_err:.2e}. batched={batched_np}, seq={seq_np}"
        )


def test_wmrl_m6b_fully_batched_matches_sequential():
    """Fully-batched vmap'd M6b (WM-RL+dual) likelihood agrees with sequential.

    8 per-participant params: alpha_pos, alpha_neg, phi, rho, capacity,
    kappa, kappa_s, epsilon.  kappa/kappa_s are decoded from the
    stick-breaking parameters (kappa_total, kappa_share); this test
    draws them so that kappa + kappa_s <= 1 by construction.
    """
    from scripts.fitting.jax_likelihoods import (
        wmrl_m6b_fully_batched_likelihood,
        wmrl_m6b_multiblock_likelihood_stacked,
    )

    pdata, stacked, N = _make_wmrl_family_synthetic()
    assert N == 5

    for draw_idx in range(3):
        key = jax.random.PRNGKey(6000 + draw_idx)
        k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)
        alpha_pos   = jax.random.uniform(k1, (N,), minval=0.1, maxval=0.9)
        alpha_neg   = jax.random.uniform(k2, (N,), minval=0.1, maxval=0.9)
        phi         = jax.random.uniform(k3, (N,), minval=0.05, maxval=0.5)
        rho         = jax.random.uniform(k4, (N,), minval=0.2, maxval=0.9)
        capacity    = jax.random.uniform(k5, (N,), minval=2.0, maxval=6.0)
        # Stick-breaking: draw kappa_total in [0, 0.9], kappa_share in [0, 1]
        kappa_total = jax.random.uniform(k6, (N,), minval=0.0, maxval=0.9)
        kappa_share = jax.random.uniform(k7, (N,), minval=0.0, maxval=1.0)
        kappa   = kappa_total * kappa_share
        kappa_s = kappa_total * (1.0 - kappa_share)
        epsilon = jax.random.uniform(k8, (N,), minval=0.01, maxval=0.1)

        batched_ll = wmrl_m6b_fully_batched_likelihood(
            stimuli=stacked["stimuli"],
            actions=stacked["actions"],
            rewards=stacked["rewards"],
            set_sizes=stacked["set_sizes"],
            masks=stacked["masks"],
            alpha_pos=alpha_pos, alpha_neg=alpha_neg, phi=phi,
            rho=rho, capacity=capacity, kappa=kappa, kappa_s=kappa_s,
            epsilon=epsilon,
        )
        assert batched_ll.shape == (N,)

        seq_lls = []
        for idx, pid in enumerate(stacked["participant_ids"]):
            pp = pdata[pid]
            ll_i = wmrl_m6b_multiblock_likelihood_stacked(
                stimuli_stacked=pp["stimuli_stacked"],
                actions_stacked=pp["actions_stacked"],
                rewards_stacked=pp["rewards_stacked"],
                set_sizes_stacked=pp["set_sizes_stacked"],
                masks_stacked=pp["masks_stacked"],
                alpha_pos=float(alpha_pos[idx]),
                alpha_neg=float(alpha_neg[idx]),
                phi=float(phi[idx]),
                rho=float(rho[idx]),
                capacity=float(capacity[idx]),
                kappa=float(kappa[idx]),
                kappa_s=float(kappa_s[idx]),
                epsilon=float(epsilon[idx]),
            )
            seq_lls.append(float(ll_i))

        batched_np = np.array(batched_ll)
        seq_np = np.array(seq_lls, dtype=np.float32)
        rel_err = np.abs(batched_np - seq_np) / (np.abs(seq_np) + 1e-9)
        max_rel_err = float(rel_err.max())
        print(f"wmrl_m6b draw {draw_idx}: max_rel_err = {max_rel_err:.2e}")
        assert max_rel_err < 1e-4, (
            f"wmrl_m6b agreement failed: draw={draw_idx}, "
            f"max_rel_err={max_rel_err:.2e}. batched={batched_np}, seq={seq_np}"
        )
