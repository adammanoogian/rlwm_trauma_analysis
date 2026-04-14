"""
Unit tests for parallel scan (pscan) likelihood primitives.

Tests:
- affine_scan: basic correctness, identity, reset handling
- associative_scan_q_update: agreement with sequential Q-learning
- associative_scan_wm_update: decay-only and decay+overwrite against sequential

All comparisons use element-wise relative error thresholds documented in
``docs/PARALLEL_SCAN_LIKELIHOOD.md``:
- Typical parameters (alpha <= 0.5): < 1e-5 relative error
- Extreme alpha (~0.95): < 1e-3 relative error (alpha approximation degrades)
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from scripts.fitting.jax_likelihoods import (
    affine_scan,
    associative_scan_q_update,
    associative_scan_wm_update,
)


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
