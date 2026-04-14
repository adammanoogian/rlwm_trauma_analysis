"""
Unit tests for parallel scan (pscan) likelihood primitives and model variants.

Tests:
- affine_scan: basic correctness, identity, reset handling
- associative_scan_q_update: agreement with sequential Q-learning
- associative_scan_wm_update: decay-only and decay+overwrite against sequential
- Model pscan variants: agreement with sequential *_multiblock_likelihood_stacked
  for all 6 choice-only models (M1/M2/M3/M5/M6a/M6b) on synthetic data
- Real-data agreement tests (marked slow, skipped if data files unavailable)

All comparisons use element-wise relative error thresholds documented in
``docs/PARALLEL_SCAN_LIKELIHOOD.md``:
- Typical parameters (alpha <= 0.5): < 1e-5 relative error
- Extreme alpha (~0.95): < 1e-3 relative error (alpha approximation degrades)
- Multiblock NLL agreement: < 1e-4 absolute or relative error
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from scripts.fitting.jax_likelihoods import (
    MAX_TRIALS_PER_BLOCK,
    affine_scan,
    associative_scan_q_update,
    associative_scan_wm_update,
    # Sequential multiblock stacked variants
    q_learning_multiblock_likelihood_stacked,
    wmrl_multiblock_likelihood_stacked,
    wmrl_m3_multiblock_likelihood_stacked,
    wmrl_m5_multiblock_likelihood_stacked,
    wmrl_m6a_multiblock_likelihood_stacked,
    wmrl_m6b_multiblock_likelihood_stacked,
    # Pscan multiblock stacked variants
    q_learning_multiblock_likelihood_stacked_pscan,
    wmrl_multiblock_likelihood_stacked_pscan,
    wmrl_m3_multiblock_likelihood_stacked_pscan,
    wmrl_m5_multiblock_likelihood_stacked_pscan,
    wmrl_m6a_multiblock_likelihood_stacked_pscan,
    wmrl_m6b_multiblock_likelihood_stacked_pscan,
)

# Path to project root (3 levels up from scripts/fitting/tests/)
_PROJECT_ROOT = Path(__file__).parents[3]
_MLE_DIR = _PROJECT_ROOT / "output" / "mle"
_DATA_PATH = _PROJECT_ROOT / "output" / "task_trials_long.csv"


# =============================================================================
# HELPERS
# =============================================================================


def _sequential_q_update(
    stimuli: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    masks: np.ndarray,
    alpha_pos: float,
    alpha_neg: float,
    q_init: float,
    num_stimuli: int,
    num_actions: int,
) -> np.ndarray:
    """
    Sequential Q-value trajectory using exact delta-sign alpha selection.

    This is the ground truth reference. The pscan version uses the
    reward-based alpha approximation, so agreement will be close but not exact.

    Parameters
    ----------
    stimuli, actions, rewards, masks : arrays, shape (T,)
    alpha_pos, alpha_neg : float
    q_init : float
    num_stimuli, num_actions : int

    Returns
    -------
    Q_history : array, shape (T, num_stimuli, num_actions)
        Q_history[t] = Q-table BEFORE update at trial t.
    """
    T = len(stimuli)
    Q = np.ones((num_stimuli, num_actions)) * q_init
    Q_history = []

    for t in range(T):
        Q_history.append(Q.copy())
        if masks[t] > 0:
            s, a, r = int(stimuli[t]), int(actions[t]), float(rewards[t])
            q_old = Q[s, a]
            delta = r - q_old
            alpha = alpha_pos if delta > 0 else alpha_neg
            Q[s, a] = q_old + alpha * delta

    return np.array(Q_history)


def _sequential_wm_decay_only(
    T: int,
    phi: float,
    wm_init: float,
    num_stimuli: int,
    num_actions: int,
) -> np.ndarray:
    """
    Sequential WM decay-only trajectory (no overwrites).

    Matches the sequential model order: decay FIRST, then use for policy.
    Returns the decayed WM (= wm_for_policy) for each trial.

    Parameters
    ----------
    T : int
        Number of trials.
    phi : float
        WM decay rate.
    wm_init : float
        WM baseline.
    num_stimuli, num_actions : int

    Returns
    -------
    WM_decayed_history : array, shape (T, S, A)
        WM_decayed_history[t] = WM-table AFTER decay at trial t (used for policy).
    """
    WM = np.ones((num_stimuli, num_actions)) * wm_init
    WM_decayed_history = []

    for t in range(T):
        WM_decayed = (1 - phi) * WM + phi * wm_init  # decay first
        WM_decayed_history.append(WM_decayed.copy())
        WM = WM_decayed  # no overwrite (decay only)

    return np.array(WM_decayed_history)


def _sequential_wm_with_overwrite(
    stimuli: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    masks: np.ndarray,
    phi: float,
    wm_init: float,
    num_stimuli: int,
    num_actions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sequential WM trajectory with decay and overwrite.

    Matches the sequential implementation in wmrl_m3_block_likelihood:
    1. Decay: WM_decayed = (1-phi)*WM + phi*wm_init
    2. Policy uses WM_decayed
    3. Overwrite: WM_decayed[s, a] = r (on valid trials)

    Parameters
    ----------
    stimuli, actions, rewards, masks : arrays, shape (T,)
    phi : float
    wm_init : float
    num_stimuli, num_actions : int

    Returns
    -------
    wm_for_policy : array, shape (T, S, A)
        WM after decay, before overwrite at each trial.
    wm_after_update : array, shape (T, S, A)
        WM after both decay and overwrite at each trial.
    """
    T = len(stimuli)
    WM = np.ones((num_stimuli, num_actions)) * wm_init
    wm_for_policy_list = []
    wm_after_update_list = []

    for t in range(T):
        # Step 1: decay
        WM_decayed = (1 - phi) * WM + phi * wm_init
        wm_for_policy_list.append(WM_decayed.copy())

        # Step 2: overwrite (only on valid trials)
        WM_updated = WM_decayed.copy()
        if masks[t] > 0:
            s, a, r = int(stimuli[t]), int(actions[t]), float(rewards[t])
            WM_updated[s, a] = r

        wm_after_update_list.append(WM_updated.copy())
        WM = WM_updated

    return np.array(wm_for_policy_list), np.array(wm_after_update_list)


def _rel_error(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Max element-wise relative error, avoiding division by zero."""
    return float(np.max(np.abs(a - b) / (np.abs(b) + eps)))


# =============================================================================
# TEST 1: affine_scan basic AR(1) correctness
# =============================================================================


def test_affine_scan_ar1():
    """
    affine_scan must reproduce a random AR(1) sequence to < 1e-6 relative error.
    """
    rng = np.random.default_rng(42)
    T = 50

    a_np = rng.uniform(0.1, 0.9, size=T).astype(np.float32)
    b_np = rng.uniform(-1.0, 1.0, size=T).astype(np.float32)
    x0 = np.float32(0.5)

    # Sequential reference
    x_seq = np.zeros(T, dtype=np.float32)
    x = x0
    for t in range(T):
        x = a_np[t] * x + b_np[t]
        x_seq[t] = x

    # Parallel scan
    a_jax = jnp.array(a_np)
    b_jax = jnp.array(b_np)
    x_par = np.array(affine_scan(a_jax, b_jax, x0=jnp.array(x0)))

    rel_err = _rel_error(x_par, x_seq)
    assert rel_err < 1e-6, (
        f"affine_scan AR(1) relative error {rel_err:.2e} exceeds 1e-6"
    )


# =============================================================================
# TEST 2: affine_scan identity (a=1, b=0 everywhere -> output == x0)
# =============================================================================


def test_affine_scan_identity():
    """
    With a_t=1 and b_t=0 everywhere, all outputs should equal x0.
    """
    T = 20
    x0 = jnp.array(0.42)
    a_seq = jnp.ones(T)
    b_seq = jnp.zeros(T)

    result = affine_scan(a_seq, b_seq, x0=x0)

    expected = jnp.full(T, 0.42)
    assert jnp.allclose(result, expected, atol=1e-7), (
        f"Identity scan failed: max deviation = {float(jnp.max(jnp.abs(result - expected))):.2e}"
    )


# =============================================================================
# TEST 3: affine_scan reset (a=0 at position k -> output == b_k at position k)
# =============================================================================


def test_affine_scan_reset():
    """
    Inserting a=0, b=r at position k (0-indexed) forces output[k] == r,
    and all subsequent outputs reflect only the state after the reset.
    """
    T = 30
    reset_pos = 10
    reset_val = 0.75
    x0 = jnp.array(0.5)

    a_np = np.full(T, 0.8, dtype=np.float32)
    b_np = np.full(T, 0.1, dtype=np.float32)
    a_np[reset_pos] = 0.0
    b_np[reset_pos] = reset_val

    a_jax = jnp.array(a_np)
    b_jax = jnp.array(b_np)
    result = np.array(affine_scan(a_jax, b_jax, x0=x0))

    # At reset position: output must equal reset_val
    assert abs(result[reset_pos] - reset_val) < 1e-7, (
        f"Reset value wrong: got {result[reset_pos]:.6f}, expected {reset_val}"
    )

    # All positions BEFORE reset: should match a standard AR(1) from x0
    x = float(x0)
    for t in range(reset_pos):
        x = a_np[t] * x + b_np[t]
        assert abs(result[t] - x) < 1e-6, (
            f"Pre-reset position {t}: got {result[t]:.6f}, expected {x:.6f}"
        )

    # After reset: no contribution from pre-reset history
    x = reset_val
    for t in range(reset_pos + 1, T):
        x = a_np[t] * x + b_np[t]
        assert abs(result[t] - x) < 1e-5, (
            f"Post-reset position {t}: got {result[t]:.6f}, expected {x:.6f}"
        )


# =============================================================================
# TEST 4: Q-update agreement with sequential (typical alpha)
# =============================================================================


def test_q_update_agreement_synthetic():
    """
    associative_scan_q_update must agree with sequential Q-learning to < 1e-5
    relative error on a 1000-trial synthetic sequence (typical parameters).
    """
    rng = np.random.default_rng(123)
    T = 1000
    S, A = 6, 3
    alpha_pos = 0.3
    alpha_neg = 0.2
    q_init = 0.5

    stimuli = rng.integers(0, S, T).astype(np.int32)
    actions = rng.integers(0, A, T).astype(np.int32)
    rewards = rng.choice([0.0, 1.0], T).astype(np.float32)
    masks = np.ones(T, dtype=np.float32)

    # Sequential ground truth
    Q_seq = _sequential_q_update(
        stimuli, actions, rewards, masks,
        alpha_pos, alpha_neg, q_init, S, A,
    )

    # Parallel scan
    Q_par = np.array(associative_scan_q_update(
        jnp.array(stimuli), jnp.array(actions), jnp.array(rewards),
        jnp.array(masks), alpha_pos, alpha_neg, q_init, S, A,
    ))

    rel_err = _rel_error(Q_par, Q_seq)
    assert rel_err < 1e-5, (
        f"Q-update agreement failed: max relative error {rel_err:.2e} > 1e-5 "
        f"(typical alpha regime)"
    )


# =============================================================================
# TEST 5: Q-update agreement with sequential (extreme alpha)
# =============================================================================


def test_q_update_agreement_extreme_alpha():
    """
    associative_scan_q_update with extreme alpha (0.95) agrees to < 1e-3.

    The alpha approximation (reward-based vs delta-sign) degrades when alpha is
    large because Q-values converge quickly to 0 or 1, making the boundary
    condition (Q near 0 or 1) occur more frequently. The 1e-3 threshold
    accounts for this well-understood approximation error. See
    ``docs/PARALLEL_SCAN_LIKELIHOOD.md`` Section 5 for derivation.
    """
    rng = np.random.default_rng(456)
    T = 1000
    S, A = 6, 3
    alpha_pos = 0.95
    alpha_neg = 0.95  # extreme: fast convergence to boundary
    q_init = 0.5

    stimuli = rng.integers(0, S, T).astype(np.int32)
    actions = rng.integers(0, A, T).astype(np.int32)
    rewards = rng.choice([0.0, 1.0], T).astype(np.float32)
    masks = np.ones(T, dtype=np.float32)

    Q_seq = _sequential_q_update(
        stimuli, actions, rewards, masks,
        alpha_pos, alpha_neg, q_init, S, A,
    )

    Q_par = np.array(associative_scan_q_update(
        jnp.array(stimuli), jnp.array(actions), jnp.array(rewards),
        jnp.array(masks), alpha_pos, alpha_neg, q_init, S, A,
    ))

    rel_err = _rel_error(Q_par, Q_seq)
    # Relaxed tolerance for extreme alpha: alpha approximation breaks down
    # near Q-value boundaries (Q near 0 or 1)
    assert rel_err < 1e-3, (
        f"Q-update extreme-alpha agreement failed: max relative error "
        f"{rel_err:.2e} > 1e-3"
    )


# =============================================================================
# TEST 6: WM decay-only correctness
# =============================================================================


def test_wm_update_decay_only():
    """
    wm_for_policy from associative_scan_wm_update must match pure sequential
    decay to < 1e-5 relative error (no overwrites, all-zero masks).
    """
    T = 100
    S, A = 6, 3
    phi = 0.3
    wm_init = 1.0 / A

    # All-masked (no overwrites will occur): masks=0 means padding
    # But decay still happens — use masks=1 with no updates for a clean test
    # Actually use a sequence where stimuli/actions don't matter (decay passes
    # use constant coefficients regardless of stimulus/action)
    rng = np.random.default_rng(789)
    stimuli = rng.integers(0, S, T).astype(np.int32)
    actions = rng.integers(0, A, T).astype(np.int32)
    rewards = np.zeros(T, dtype=np.float32)
    masks = np.zeros(T, dtype=np.float32)  # all padding -> no overwrites

    # Sequential decay (T steps, pure decay)
    WM_seq = _sequential_wm_decay_only(T, phi, wm_init, S, A)

    # Parallel scan
    wm_for_policy_par, _ = associative_scan_wm_update(
        jnp.array(stimuli), jnp.array(actions), jnp.array(rewards),
        jnp.array(masks), phi, wm_init, S, A,
    )
    wm_for_policy_par = np.array(wm_for_policy_par)

    rel_err = _rel_error(wm_for_policy_par, WM_seq)
    assert rel_err < 1e-5, (
        f"WM decay-only relative error {rel_err:.2e} > 1e-5"
    )


# =============================================================================
# TEST 7: WM overwrite correctness at reset position
# =============================================================================


def test_wm_update_with_overwrite():
    """
    wm_after_update at overwrite positions must equal the reward exactly.

    After a reset at trial k for (s_k, a_k):
    wm_after_update[k, s_k, a_k] == reward_k
    """
    T = 20
    S, A = 6, 3
    phi = 0.3
    wm_init = 1.0 / A

    # Fixed trial data with known overwrites at positions 5 and 12
    stimuli = np.zeros(T, dtype=np.int32)
    actions = np.zeros(T, dtype=np.int32)
    rewards = np.zeros(T, dtype=np.float32)
    masks = np.zeros(T, dtype=np.float32)

    # Overwrite at trial 5: s=2, a=1, r=1.0
    stimuli[5] = 2; actions[5] = 1; rewards[5] = 1.0; masks[5] = 1.0

    # Overwrite at trial 12: s=4, a=2, r=0.0
    stimuli[12] = 4; actions[12] = 2; rewards[12] = 0.0; masks[12] = 1.0

    _, wm_after = associative_scan_wm_update(
        jnp.array(stimuli), jnp.array(actions), jnp.array(rewards),
        jnp.array(masks), phi, wm_init, S, A,
    )
    wm_after = np.array(wm_after)

    # At trial 5 (after decay + overwrite): WM[2, 1] should equal 1.0
    # Note: wm_after_update[t] is the state AFTER trial t's decay+overwrite
    # wm_after_update as returned has wm_after_update[t] = WM after t-1's update
    # (it's prepend-init, drop-last). So wm_after_update[5] is WM going into
    # trial 5's policy (before decay at t=5).
    # Wait -- wm_after_update[t] is what state WM is ENTERING trial t with.
    # Let's verify: after trial 5, state WM has WM[2,1]=1.0.
    # That state feeds INTO trial 6. So wm_after[6, 2, 1] should be decayed
    # from 1.0.

    # Direct check: compute sequential ground truth
    _, wm_after_seq = _sequential_wm_with_overwrite(
        stimuli, actions, rewards, masks, phi, wm_init, S, A,
    )

    # wm_after from pscan should match sequential
    rel_err = _rel_error(wm_after, wm_after_seq)
    assert rel_err < 1e-5, (
        f"WM after-overwrite relative error {rel_err:.2e} > 1e-5"
    )

    # Explicit spot-check: at trial 5 (wm_for_policy[5] = WM before overwrite at t=5)
    # After overwrite: WM[2,1] = 1.0. This is wm_after_seq[5, 2, 1].
    wm_for_policy_par, wm_after_par = associative_scan_wm_update(
        jnp.array(stimuli), jnp.array(actions), jnp.array(rewards),
        jnp.array(masks), phi, wm_init, S, A,
    )
    wm_for_policy_par = np.array(wm_for_policy_par)
    wm_after_par = np.array(wm_after_par)

    # wm_after_update[t] is WM entering trial t (after t-1's update).
    # WM after trial 5's overwrite (WM[2,1]=1.0) feeds into trial 6.
    # Therefore wm_after_par[6, 2, 1] = decayed(1.0) = (1-phi)*1.0 + phi*wm_init
    # Actually wm_after_update is indexed differently -- it's the state AFTER
    # update at that trial, prepended with init and dropped last.
    # wm_after_update[t] = WM AFTER update at trial t-1 = WM entering trial t.

    # At trial 6, the value is (1-phi)*1.0 + phi*wm_init (from next-trial's decay)
    # The overwrite happened in trial 5, so wm_after_par[5] is entering trial 5
    # and wm_after_par[6] is entering trial 6 (= WM after trial 5's update).

    # Let's check wm_after_seq[5] is before overwrite at t=5 (after t=4's update)
    # and wm_after_seq[6] is after overwrite at t=5 (entering t=6)
    # Actually _sequential_wm_with_overwrite returns wm_after_update where
    # wm_after_update[t] = WM after step t (including overwrite at t).
    # And wm_after_par[t] = WM entering trial t = WM after trial t-1.

    # Checking consistency: after trial 5, WM[2,1] should be 1.0
    # This is reflected in wm_after_seq[5, 2, 1] (wm after step 5)
    expected_after_5 = 1.0
    assert abs(float(wm_after_seq[5, 2, 1]) - expected_after_5) < 1e-7, (
        f"Sequential WM after overwrite at t=5: got {wm_after_seq[5, 2, 1]:.6f}, "
        f"expected {expected_after_5}"
    )


# =============================================================================
# TEST 8: WM full agreement with sequential (1000 trials)
# =============================================================================


def test_wm_update_agreement_with_sequential():
    """
    Both wm_for_policy and wm_after_update from associative_scan_wm_update
    must agree with sequential to < 1e-5 relative error on a 1000-trial block
    with frequent valid overwrites.
    """
    rng = np.random.default_rng(999)
    T = 1000
    S, A = 6, 3
    phi = 0.3
    wm_init = 1.0 / A

    stimuli = rng.integers(0, S, T).astype(np.int32)
    actions = rng.integers(0, A, T).astype(np.int32)
    rewards = rng.choice([0.0, 1.0], T).astype(np.float32)
    # ~60% valid trials to exercise overwrites frequently
    masks = rng.choice([0.0, 1.0], T, p=[0.4, 0.6]).astype(np.float32)

    # Sequential ground truth
    wm_for_policy_seq, wm_after_seq = _sequential_wm_with_overwrite(
        stimuli, actions, rewards, masks, phi, wm_init, S, A,
    )

    # Parallel scan
    wm_for_policy_par, wm_after_par = associative_scan_wm_update(
        jnp.array(stimuli), jnp.array(actions), jnp.array(rewards),
        jnp.array(masks), phi, wm_init, S, A,
    )
    wm_for_policy_par = np.array(wm_for_policy_par)
    wm_after_par = np.array(wm_after_par)

    rel_err_policy = _rel_error(wm_for_policy_par, wm_for_policy_seq)
    rel_err_after = _rel_error(wm_after_par, wm_after_seq)

    assert rel_err_policy < 1e-5, (
        f"wm_for_policy relative error {rel_err_policy:.2e} > 1e-5"
    )
    assert rel_err_after < 1e-5, (
        f"wm_after_update relative error {rel_err_after:.2e} > 1e-5"
    )


# =============================================================================
# FIXTURE: synthetic multi-block stacked data
# =============================================================================


@pytest.fixture
def synthetic_stacked_data():
    """
    Generate synthetic 3-block stacked data in the format expected by
    ``*_multiblock_likelihood_stacked`` functions.

    Returns a dict with:
        stimuli_stacked   (3, MAX_TRIALS_PER_BLOCK) int32
        actions_stacked   (3, MAX_TRIALS_PER_BLOCK) int32
        rewards_stacked   (3, MAX_TRIALS_PER_BLOCK) float32
        set_sizes_stacked (3, MAX_TRIALS_PER_BLOCK) float32
        masks_stacked     (3, MAX_TRIALS_PER_BLOCK) float32
        n_blocks          3
    """
    rng = np.random.default_rng(20240414)
    n_blocks = 3
    real_trials = [50, 50, 50]  # same for each block
    S, A = 6, 3
    T = MAX_TRIALS_PER_BLOCK  # 100

    stimuli_stacked = np.zeros((n_blocks, T), dtype=np.int32)
    actions_stacked = np.zeros((n_blocks, T), dtype=np.int32)
    rewards_stacked = np.zeros((n_blocks, T), dtype=np.float32)
    set_sizes_stacked = np.ones((n_blocks, T), dtype=np.float32) * 2.0
    masks_stacked = np.zeros((n_blocks, T), dtype=np.float32)

    # Different set sizes per block to exercise omega computation
    block_set_sizes = [2, 3, 6]

    for b in range(n_blocks):
        n = real_trials[b]
        stimuli_stacked[b, :n] = rng.integers(0, S, n)
        actions_stacked[b, :n] = rng.integers(0, A, n)
        rewards_stacked[b, :n] = rng.choice([0.0, 1.0], n).astype(np.float32)
        set_sizes_stacked[b, :n] = block_set_sizes[b]
        masks_stacked[b, :n] = 1.0

    return {
        "stimuli_stacked": jnp.array(stimuli_stacked),
        "actions_stacked": jnp.array(actions_stacked),
        "rewards_stacked": jnp.array(rewards_stacked),
        "set_sizes_stacked": jnp.array(set_sizes_stacked),
        "masks_stacked": jnp.array(masks_stacked),
        "n_blocks": n_blocks,
    }


# =============================================================================
# HELPER: load MLE parameters from CSV
# =============================================================================


def _load_mle_params(model: str, participant_id: int) -> dict:
    """
    Load MLE parameters for a single participant from the CSV file.

    Parameters
    ----------
    model : str
        One of 'qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b'.
    participant_id : int
        Participant ID to look up in the CSV (matches 'participant_id' column).

    Returns
    -------
    dict
        Parameter dict matching the function signature of the model's
        sequential likelihood.
    """
    import pandas as pd

    csv_path = _MLE_DIR / f"{model}_individual_fits.csv"
    df = pd.read_csv(csv_path)
    matches = df[df["participant_id"] == participant_id]
    if len(matches) == 0:
        raise KeyError(
            f"Participant {participant_id} not found in {csv_path.name} "
            f"(available: {df['participant_id'].tolist()[:5]}...)"
        )
    row = matches.iloc[0]

    # Model-specific parameter extraction
    base = {
        "alpha_pos": float(row["alpha_pos"]),
        "alpha_neg": float(row["alpha_neg"]),
        "epsilon": float(row["epsilon"]),
    }
    if model == "qlearning":
        return base

    wm_base = {
        **base,
        "phi": float(row["phi"]),
        "rho": float(row["rho"]),
        "capacity": float(row["capacity"]),
    }
    if model == "wmrl":
        return wm_base

    if model == "wmrl_m3":
        return {**wm_base, "kappa": float(row["kappa"])}

    if model == "wmrl_m5":
        return {**wm_base, "kappa": float(row["kappa"]), "phi_rl": float(row["phi_rl"])}

    if model == "wmrl_m6a":
        return {**wm_base, "kappa_s": float(row["kappa_s"])}

    if model == "wmrl_m6b":
        # M6b stores kappa_total / kappa_share in CSV; decode to kappa / kappa_s
        kappa_total = float(row["kappa_total"])
        kappa_share = float(row["kappa_share"])
        return {
            **wm_base,
            "kappa": kappa_total * kappa_share,
            "kappa_s": kappa_total * (1.0 - kappa_share),
        }

    raise ValueError(f"Unknown model: {model}")


# =============================================================================
# HELPER: call sequential and pscan multiblock functions for a given model
# =============================================================================


def _call_seq_and_pscan(model: str, data: dict, params: dict) -> tuple[float, float]:
    """
    Call the sequential and pscan multiblock stacked functions for a model.

    Returns
    -------
    (nll_seq, nll_pscan) : tuple of float
        Negative log-likelihoods (actually raw log-likelihoods; compare as-is).
    """
    stim = data["stimuli_stacked"]
    act = data["actions_stacked"]
    rew = data["rewards_stacked"]
    mask = data["masks_stacked"]

    if model == "qlearning":
        nll_seq = float(q_learning_multiblock_likelihood_stacked(
            stim, act, rew, mask, **params
        ))
        nll_pscan = float(q_learning_multiblock_likelihood_stacked_pscan(
            stim, act, rew, mask, **params
        ))
        return nll_seq, nll_pscan

    ss = data["set_sizes_stacked"]

    if model == "wmrl":
        nll_seq = float(wmrl_multiblock_likelihood_stacked(
            stim, act, rew, ss, mask, **params
        ))
        nll_pscan = float(wmrl_multiblock_likelihood_stacked_pscan(
            stim, act, rew, ss, mask, **params
        ))

    elif model == "wmrl_m3":
        nll_seq = float(wmrl_m3_multiblock_likelihood_stacked(
            stim, act, rew, ss, mask, **params
        ))
        nll_pscan = float(wmrl_m3_multiblock_likelihood_stacked_pscan(
            stim, act, rew, ss, mask, **params
        ))

    elif model == "wmrl_m5":
        nll_seq = float(wmrl_m5_multiblock_likelihood_stacked(
            stim, act, rew, ss, mask, **params
        ))
        nll_pscan = float(wmrl_m5_multiblock_likelihood_stacked_pscan(
            stim, act, rew, ss, mask, **params
        ))

    elif model == "wmrl_m6a":
        nll_seq = float(wmrl_m6a_multiblock_likelihood_stacked(
            stim, act, rew, ss, mask, **params
        ))
        nll_pscan = float(wmrl_m6a_multiblock_likelihood_stacked_pscan(
            stim, act, rew, ss, mask, **params
        ))

    elif model == "wmrl_m6b":
        nll_seq = float(wmrl_m6b_multiblock_likelihood_stacked(
            stim, act, rew, ss, mask, **params
        ))
        nll_pscan = float(wmrl_m6b_multiblock_likelihood_stacked_pscan(
            stim, act, rew, ss, mask, **params
        ))

    else:
        raise ValueError(f"Unknown model: {model}")

    return nll_seq, nll_pscan


# =============================================================================
# TYPICAL PARAMETERS FOR EACH MODEL (used in synthetic tests)
# =============================================================================

_TYPICAL_PARAMS = {
    "qlearning": {"alpha_pos": 0.3, "alpha_neg": 0.1, "epsilon": 0.05},
    "wmrl": {
        "alpha_pos": 0.3, "alpha_neg": 0.1,
        "phi": 0.1, "rho": 0.7, "capacity": 3.0, "epsilon": 0.05,
    },
    "wmrl_m3": {
        "alpha_pos": 0.3, "alpha_neg": 0.1,
        "phi": 0.1, "rho": 0.7, "capacity": 3.0, "kappa": 0.2, "epsilon": 0.05,
    },
    "wmrl_m5": {
        "alpha_pos": 0.3, "alpha_neg": 0.1,
        "phi": 0.1, "rho": 0.7, "capacity": 3.0, "kappa": 0.2,
        "phi_rl": 0.05, "epsilon": 0.05,
    },
    "wmrl_m6a": {
        "alpha_pos": 0.3, "alpha_neg": 0.1,
        "phi": 0.1, "rho": 0.7, "capacity": 3.0, "kappa_s": 0.2, "epsilon": 0.05,
    },
    "wmrl_m6b": {
        "alpha_pos": 0.3, "alpha_neg": 0.1,
        "phi": 0.1, "rho": 0.7, "capacity": 3.0,
        "kappa": 0.15, "kappa_s": 0.10, "epsilon": 0.05,
    },
}


# =============================================================================
# TESTS 9-14: Synthetic agreement tests for all 6 models
# =============================================================================


@pytest.mark.parametrize("model", [
    "qlearning", "wmrl", "wmrl_m3", "wmrl_m5", "wmrl_m6a", "wmrl_m6b",
])
def test_pscan_agreement_synthetic(model, synthetic_stacked_data):
    """
    Each pscan multiblock likelihood must agree with its sequential counterpart
    to < 1e-4 absolute error on synthetic 3-block data.

    Also tests return_pointwise=True: per-trial log-probs agree to < 1e-4.
    """
    data = synthetic_stacked_data
    params = _TYPICAL_PARAMS[model]

    nll_seq, nll_pscan = _call_seq_and_pscan(model, data, params)

    abs_err = abs(nll_seq - nll_pscan)
    max_nll = max(abs(nll_seq), 1e-8)
    rel_err = abs_err / max_nll

    assert abs_err < 1e-4 or rel_err < 1e-4, (
        f"[{model}] synthetic NLL agreement failed: "
        f"seq={nll_seq:.6f}, pscan={nll_pscan:.6f}, "
        f"abs_err={abs_err:.2e}, rel_err={rel_err:.2e}"
    )

    # Also test return_pointwise=True
    stim = data["stimuli_stacked"]
    act = data["actions_stacked"]
    rew = data["rewards_stacked"]
    mask = data["masks_stacked"]

    if model == "qlearning":
        _, probs_seq = q_learning_multiblock_likelihood_stacked(
            stim, act, rew, mask, return_pointwise=True, **params
        )
        _, probs_pscan = q_learning_multiblock_likelihood_stacked_pscan(
            stim, act, rew, mask, return_pointwise=True, **params
        )
    else:
        ss = data["set_sizes_stacked"]
        seq_fn_map = {
            "wmrl": wmrl_multiblock_likelihood_stacked,
            "wmrl_m3": wmrl_m3_multiblock_likelihood_stacked,
            "wmrl_m5": wmrl_m5_multiblock_likelihood_stacked,
            "wmrl_m6a": wmrl_m6a_multiblock_likelihood_stacked,
            "wmrl_m6b": wmrl_m6b_multiblock_likelihood_stacked,
        }
        pscan_fn_map = {
            "wmrl": wmrl_multiblock_likelihood_stacked_pscan,
            "wmrl_m3": wmrl_m3_multiblock_likelihood_stacked_pscan,
            "wmrl_m5": wmrl_m5_multiblock_likelihood_stacked_pscan,
            "wmrl_m6a": wmrl_m6a_multiblock_likelihood_stacked_pscan,
            "wmrl_m6b": wmrl_m6b_multiblock_likelihood_stacked_pscan,
        }
        _, probs_seq = seq_fn_map[model](
            stim, act, rew, ss, mask, return_pointwise=True, **params
        )
        _, probs_pscan = pscan_fn_map[model](
            stim, act, rew, ss, mask, return_pointwise=True, **params
        )

    probs_seq_np = np.array(probs_seq)
    probs_pscan_np = np.array(probs_pscan)
    max_pointwise_err = float(np.max(np.abs(probs_seq_np - probs_pscan_np)))
    assert max_pointwise_err < 1e-4, (
        f"[{model}] per-trial log-prob max deviation: {max_pointwise_err:.2e} > 1e-4"
    )


# =============================================================================
# TESTS 15-20: Real-data single-participant smoke tests (marked slow)
# =============================================================================


@pytest.mark.slow
@pytest.mark.parametrize("model", [
    "qlearning", "wmrl", "wmrl_m3", "wmrl_m5", "wmrl_m6a", "wmrl_m6b",
])
def test_pscan_agreement_real_data(model):
    """
    Pscan variant must agree with sequential to < 1e-4 relative error on the
    first participant's real data using that participant's MLE parameters.

    Skipped if the MLE CSV or trial data files are not available.
    """
    csv_path = _MLE_DIR / f"{model}_individual_fits.csv"
    if not csv_path.exists():
        pytest.skip(f"MLE CSV not found: {csv_path}")
    if not _DATA_PATH.exists():
        pytest.skip(f"Trial data not found: {_DATA_PATH}")

    import pandas as pd
    from scripts.fitting.numpyro_models import prepare_stacked_participant_data

    # Load trial data
    data_df = pd.read_csv(_DATA_PATH)
    participant_data = prepare_stacked_participant_data(data_df)

    # Use first participant that has MLE fits
    mle_df = pd.read_csv(csv_path)
    mle_pids = set(mle_df["participant_id"].tolist())
    data_pids = sorted(participant_data.keys())
    # Find first PID that exists in both data and MLE fits
    pid = None
    for _p in data_pids:
        if int(_p) in mle_pids:
            pid = _p
            break
    if pid is None:
        pytest.skip(f"No overlapping participants between data and {csv_path.name}")
    pdata = participant_data[pid]
    params = _load_mle_params(model, participant_id=int(pid))

    nll_seq, nll_pscan = _call_seq_and_pscan(model, pdata, params)

    abs_err = abs(nll_seq - nll_pscan)
    max_nll = max(abs(nll_seq), 1e-8)
    rel_err = abs_err / max_nll

    assert rel_err < 1e-4, (
        f"[{model}] real-data (participant {pid}) agreement failed: "
        f"seq={nll_seq:.4f}, pscan={nll_pscan:.4f}, rel_err={rel_err:.2e}"
    )


# =============================================================================
# TESTS 21-26: Full N=154 agreement (marked slow, PSCAN-04 validation)
# =============================================================================


@pytest.mark.slow
@pytest.mark.parametrize("model", [
    "qlearning", "wmrl", "wmrl_m3", "wmrl_m5", "wmrl_m6a", "wmrl_m6b",
])
def test_pscan_full_n154_agreement(model):
    """
    PSCAN-04: Pscan multiblock likelihood must agree with sequential to
    < 1e-4 relative error for ALL 154 real participants.

    Logs the maximum relative error observed across all participants per model.
    Skipped if data files are unavailable.
    """
    csv_path = _MLE_DIR / f"{model}_individual_fits.csv"
    if not csv_path.exists():
        pytest.skip(f"MLE CSV not found: {csv_path}")
    if not _DATA_PATH.exists():
        pytest.skip(f"Trial data not found: {_DATA_PATH}")

    import pandas as pd
    from scripts.fitting.numpyro_models import prepare_stacked_participant_data

    data_df = pd.read_csv(_DATA_PATH)
    participant_data = prepare_stacked_participant_data(data_df)
    mle_df = pd.read_csv(csv_path)

    # Only test participants that exist in BOTH data and MLE fits
    mle_pids = set(mle_df["participant_id"].tolist())
    data_pids = sorted(participant_data.keys())
    participants = [p for p in data_pids if int(p) in mle_pids]

    if not participants:
        pytest.skip(f"No overlapping participants between data and {csv_path.name}")

    max_rel_err = 0.0
    failures = []

    for pid in participants:
        pdata = participant_data[pid]
        params = _load_mle_params(model, participant_id=int(pid))

        nll_seq, nll_pscan = _call_seq_and_pscan(model, pdata, params)

        abs_err = abs(nll_seq - nll_pscan)
        max_nll = max(abs(nll_seq), 1e-8)
        rel_err = abs_err / max_nll

        if rel_err > max_rel_err:
            max_rel_err = rel_err

        if rel_err >= 1e-4:
            failures.append((pid, nll_seq, nll_pscan, rel_err))

    print(
        f"\n[{model}] N=154 max relative error: {max_rel_err:.2e} "
        f"({'PASS' if not failures else 'FAIL'})"
    )

    assert not failures, (
        f"[{model}] {len(failures)}/{len(participants)} participants exceeded 1e-4 "
        f"relative error threshold. Max rel_err={max_rel_err:.2e}. "
        f"First failure: participant={failures[0][0]}, seq={failures[0][1]:.4f}, "
        f"pscan={failures[0][2]:.4f}, rel_err={failures[0][3]:.2e}"
    )


# =============================================================================
# PRECOMPUTATION FUNCTIONS (Phase 20)
# =============================================================================

from scripts.fitting.jax_likelihoods import (
    precompute_last_action_global,
    precompute_last_actions_per_stimulus,
)


class TestPrecomputeLastActionGlobal:
    """Tests for precompute_last_action_global."""

    def test_precompute_last_action_global_basic(self):
        """5-trial all-valid: result[0]=-1, result[t]=actions[t-1]."""
        actions = jnp.array([2, 0, 1, 2, 0], dtype=jnp.int32)
        mask = jnp.ones(5)

        result = precompute_last_action_global(actions, mask)

        expected = jnp.array([-1, 2, 0, 1, 2], dtype=jnp.int32)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))

    def test_precompute_last_action_global_masked(self):
        """Masked trials propagate last valid action forward."""
        actions = jnp.array([2, 0, 1, 2, 0], dtype=jnp.int32)
        # Trial 2 (action=1) is masked -> should not update last_action
        mask = jnp.array([1.0, 1.0, 0.0, 1.0, 1.0])

        result = precompute_last_action_global(actions, mask)

        # t=0: -1 (no previous)
        # t=1: 2 (action at t=0, valid)
        # t=2: 0 (action at t=1, valid)
        # t=3: 0 (action at t=1 propagated, since t=2 was masked)
        # t=4: 2 (action at t=3, valid)
        expected = jnp.array([-1, 2, 0, 0, 2], dtype=jnp.int32)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))

    def test_precompute_last_action_global_all_masked(self):
        """All trials masked: result should be -1 everywhere."""
        actions = jnp.array([2, 0, 1], dtype=jnp.int32)
        mask = jnp.zeros(3)

        result = precompute_last_action_global(actions, mask)

        expected = jnp.array([-1, -1, -1], dtype=jnp.int32)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))


class TestPrecomputeLastActionsPerStimulus:
    """Tests for precompute_last_actions_per_stimulus."""

    def test_precompute_last_actions_per_stimulus_basic(self):
        """6-trial, 2 stimuli: verify per-stimulus tracking."""
        # stimulus: 0  1  0  1  0  1
        # action:   2  1  0  2  1  0
        stimuli = jnp.array([0, 1, 0, 1, 0, 1], dtype=jnp.int32)
        actions = jnp.array([2, 1, 0, 2, 1, 0], dtype=jnp.int32)
        mask = jnp.ones(6)

        result = precompute_last_actions_per_stimulus(
            stimuli, actions, mask, num_stimuli=2
        )

        # t=0: stim=0, never seen before -> -1
        # t=1: stim=1, never seen before -> -1
        # t=2: stim=0, last action for stim 0 was 2 (t=0) -> 2
        # t=3: stim=1, last action for stim 1 was 1 (t=1) -> 1
        # t=4: stim=0, last action for stim 0 was 0 (t=2) -> 0
        # t=5: stim=1, last action for stim 1 was 2 (t=3) -> 2
        expected = jnp.array([-1, -1, 2, 1, 0, 2], dtype=jnp.int32)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))

    def test_precompute_last_actions_per_stimulus_masked(self):
        """Masked trials do not update per-stimulus last_action."""
        stimuli = jnp.array([0, 1, 0, 1, 0], dtype=jnp.int32)
        actions = jnp.array([2, 1, 0, 2, 1], dtype=jnp.int32)
        # Trial 2 (stim=0, action=0) is masked
        mask = jnp.array([1.0, 1.0, 0.0, 1.0, 1.0])

        result = precompute_last_actions_per_stimulus(
            stimuli, actions, mask, num_stimuli=2
        )

        # t=0: stim=0, never seen -> -1
        # t=1: stim=1, never seen -> -1
        # t=2: stim=0, last action for stim 0 was 2 (t=0) -> 2
        # t=3: stim=1, last action for stim 1 was 1 (t=1) -> 1
        # t=4: stim=0, last action for stim 0 is STILL 2 (t=2 was masked, so
        #       the action=0 at t=2 did not update stim 0) -> 2
        expected = jnp.array([-1, -1, 2, 1, 2], dtype=jnp.int32)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))


class TestPrecomputeAgreesWithScan:
    """Verify precomputed arrays match the sequential lax.scan carry."""

    def test_precompute_agrees_with_m3_scan(self):
        """Precomputed global last_action matches M3 Phase 2 scan carry."""
        # Simulate a block with known parameters
        rng = np.random.RandomState(42)
        T = 50
        n_stim, n_act = 6, 3
        stimuli = rng.randint(0, n_stim, T).astype(np.int32)
        actions = rng.randint(0, n_act, T).astype(np.int32)
        rewards = rng.randint(0, 2, T).astype(np.float32)
        set_sizes = np.full(T, n_stim, dtype=np.float32)
        # Add some masked trials at the end (padding)
        mask = np.ones(T, dtype=np.float32)
        mask[45:] = 0.0

        stimuli_j = jnp.array(stimuli)
        actions_j = jnp.array(actions)
        rewards_j = jnp.array(rewards)
        set_sizes_j = jnp.array(set_sizes)
        mask_j = jnp.array(mask)

        # Precompute using the new function
        precomputed = precompute_last_action_global(actions_j, mask_j)

        # Extract the sequential carry from the M3 Phase 2 scan
        # by running a scan that only tracks last_action
        def _extract_last_action(carry, inputs):
            last_action = carry
            action, valid = inputs
            out = last_action
            new_last_action = jnp.where(valid, action, last_action).astype(
                jnp.int32
            )
            return new_last_action, out

        _, sequential_last_action = jax.lax.scan(
            _extract_last_action,
            jnp.array(-1, dtype=jnp.int32),
            (actions_j, mask_j),
        )

        np.testing.assert_array_equal(
            np.asarray(precomputed),
            np.asarray(sequential_last_action),
        )

    def test_precompute_agrees_with_m6a_scan(self):
        """Precomputed per-stimulus last_action matches M6a Phase 2 scan carry."""
        rng = np.random.RandomState(123)
        T = 60
        n_stim, n_act = 6, 3
        stimuli = rng.randint(0, n_stim, T).astype(np.int32)
        actions = rng.randint(0, n_act, T).astype(np.int32)
        mask = np.ones(T, dtype=np.float32)
        # Add scattered masked trials
        mask[10] = 0.0
        mask[25] = 0.0
        mask[50:] = 0.0

        stimuli_j = jnp.array(stimuli)
        actions_j = jnp.array(actions)
        mask_j = jnp.array(mask)

        # Precompute using the new function
        precomputed = precompute_last_actions_per_stimulus(
            stimuli_j, actions_j, mask_j, num_stimuli=n_stim
        )

        # Extract the sequential carry from the M6a Phase 2 scan
        def _extract_last_actions(carry, inputs):
            last_actions = carry
            stimulus, action, valid = inputs
            last_action_s = last_actions[stimulus]
            new_last_actions = last_actions.at[stimulus].set(
                jnp.where(valid, action, last_action_s).astype(jnp.int32)
            )
            return new_last_actions, last_action_s

        init = jnp.full((n_stim,), -1, dtype=jnp.int32)
        _, sequential_last_action = jax.lax.scan(
            _extract_last_actions,
            init,
            (stimuli_j, actions_j, mask_j),
        )

        np.testing.assert_array_equal(
            np.asarray(precomputed),
            np.asarray(sequential_last_action),
        )
