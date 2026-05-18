"""Unit tests for associative_scan_q_update with single learning rate.

Phase 33 W1 gate: associative_scan_q_update must accept a single `alpha`
parameter (not `alpha` + `alpha`).  This file contains:

1. A regression test that the single-alpha pscan agrees with a reference
   sequential lax.scan implementation to relative tolerance < 1e-6.
2. A smoke test verifying finite outputs for typical parameters.
3. A backwards-compatibility test that the old two-argument signature
   raises TypeError (enforces the migration is complete).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import lax

from rlwm.fitting.core import associative_scan_q_update

# ---------------------------------------------------------------------------
# Reference sequential implementation (used only in tests)
# ---------------------------------------------------------------------------


def _sequential_q_update_reference(
    stimuli: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    masks: jnp.ndarray,
    alpha: float,
    q_init: float = 0.5,
    num_stimuli: int = 6,
    num_actions: int = 3,
) -> jnp.ndarray:
    """Sequential reference using lax.scan.

    Single learning rate for positive AND negative outcomes (Phase 33).
    Returns Q_for_policy[t] = Q BEFORE update at trial t, shape (T, S, A).
    """
    S, A = num_stimuli, num_actions
    q_init_table = jnp.ones((S, A)) * q_init

    def step(Q_table, inputs):
        stim, act, rew, valid = inputs
        # Q-value update for the active (s, a)
        q_cur = Q_table[stim, act]
        q_new = q_cur + alpha * (rew - q_cur)
        Q_new = Q_table.at[stim, act].set(jnp.where(valid, q_new, q_cur))
        return Q_new, Q_table  # return old table as output (Q before update)

    _, Q_for_policy = lax.scan(step, q_init_table, (stimuli, actions, rewards, masks))
    return Q_for_policy


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAssociativeScanQUpdateSingleAlpha:
    """Test suite for the single-alpha associative_scan_q_update."""

    def _make_block(self, n_trials: int = 30, seed: int = 42):
        """Generate a deterministic synthetic block for testing."""
        key = jax.random.PRNGKey(seed)
        k1, k2, k3 = jax.random.split(key, 3)
        stimuli = jax.random.randint(k1, (n_trials,), 0, 6)
        actions = jax.random.randint(k2, (n_trials,), 0, 3)
        rewards = jax.random.bernoulli(k3, 0.7, (n_trials,)).astype(jnp.float32)
        masks = jnp.ones(n_trials)
        return stimuli, actions, rewards, masks

    def test_single_alpha_signature_accepted(self):
        """associative_scan_q_update must accept keyword `alpha` (not alpha/alpha)."""
        stimuli, actions, rewards, masks = self._make_block(n_trials=10)
        # This should not raise
        result = associative_scan_q_update(
            stimuli, actions, rewards, masks,
            alpha=0.3,
            q_init=0.5,
            num_stimuli=6,
            num_actions=3,
        )
        assert result.shape == (10, 6, 3), (
            f"Expected shape (10, 6, 3), got {result.shape}"
        )

    def test_output_is_finite(self):
        """All Q-values in Q_for_policy must be finite for typical parameters."""
        stimuli, actions, rewards, masks = self._make_block(n_trials=30)
        Q_for_policy = associative_scan_q_update(
            stimuli, actions, rewards, masks,
            alpha=0.3,
            q_init=0.5,
            num_stimuli=6,
            num_actions=3,
        )
        assert jnp.all(jnp.isfinite(Q_for_policy)), (
            "Q_for_policy contains non-finite values"
        )

    def test_q_values_bounded(self):
        """Q-values must remain in [0, 1] for rewards in {0, 1} and q_init=0.5."""
        stimuli, actions, rewards, masks = self._make_block(n_trials=30)
        Q_for_policy = associative_scan_q_update(
            stimuli, actions, rewards, masks,
            alpha=0.3,
            q_init=0.5,
            num_stimuli=6,
            num_actions=3,
        )
        assert jnp.all(Q_for_policy >= 0.0), "Q-values below 0"
        assert jnp.all(Q_for_policy <= 1.0), "Q-values above 1"

    def test_agrees_with_sequential_reference(self):
        """Pscan must agree with sequential lax.scan reference to rtol < 1e-6.

        This is the core regression guard: parallel scan and sequential scan
        must produce identical Q_for_policy trajectories when using a single
        learning rate (the reward-conditional alpha_t = alpha for all outcomes).
        """
        stimuli, actions, rewards, masks = self._make_block(n_trials=30)
        alpha = 0.3

        Q_pscan = associative_scan_q_update(
            stimuli, actions, rewards, masks,
            alpha=alpha,
            q_init=0.5,
            num_stimuli=6,
            num_actions=3,
        )
        Q_ref = _sequential_q_update_reference(
            stimuli, actions, rewards, masks,
            alpha=alpha,
            q_init=0.5,
            num_stimuli=6,
            num_actions=3,
        )

        max_rel_err = jnp.max(
            jnp.abs(Q_pscan - Q_ref) / (jnp.abs(Q_ref) + 1e-8)
        )
        assert float(max_rel_err) < 1e-6, (
            f"Pscan vs sequential reference max relative error = {float(max_rel_err):.2e}"
            f" (expected < 1e-6). Single-alpha scan may have a logic error."
        )

    def test_agrees_with_sequential_high_alpha(self):
        """Agreement must hold even for high alpha (0.9, extreme parameters)."""
        stimuli, actions, rewards, masks = self._make_block(n_trials=30, seed=99)
        alpha = 0.9  # extreme

        Q_pscan = associative_scan_q_update(
            stimuli, actions, rewards, masks,
            alpha=alpha,
            q_init=0.5,
            num_stimuli=6,
            num_actions=3,
        )
        Q_ref = _sequential_q_update_reference(
            stimuli, actions, rewards, masks,
            alpha=alpha,
            q_init=0.5,
            num_stimuli=6,
            num_actions=3,
        )

        max_rel_err = jnp.max(
            jnp.abs(Q_pscan - Q_ref) / (jnp.abs(Q_ref) + 1e-8)
        )
        assert float(max_rel_err) < 1e-5, (
            f"High-alpha case: max relative error = {float(max_rel_err):.2e}"
            f" (expected < 1e-5)"
        )

    def test_old_two_alpha_signature_rejected(self):
        """Calling with 9 positional args (old alpha+/alpha- API) must raise TypeError.

        This enforces that the Phase 33 migration is complete. The old API
        accepted alpha_pos + alpha_neg as two separate floats (positions 5 and 6).
        After Phase 33, the function only accepts a single alpha (position 5),
        so passing a 9th positional argument raises TypeError.
        """
        stimuli, actions, rewards, masks = self._make_block(n_trials=10)
        with pytest.raises(TypeError):
            # 9 positional args: old signature was (stim, act, rew, mask, alpha_pos, alpha_neg,
            # q_init, S, A) — now raises because max positional args is 8
            associative_scan_q_update(
                stimuli, actions, rewards, masks,
                0.3, 0.2, 0.5, 6, 3,  # 9 args total: alpha_pos, alpha_neg, q_init, S, A
            )

    def test_q_init_respected(self):
        """Q_for_policy[0] must equal q_init_table before any updates."""
        n_trials = 5
        stimuli = jnp.zeros(n_trials, dtype=jnp.int32)
        actions = jnp.zeros(n_trials, dtype=jnp.int32)
        rewards = jnp.ones(n_trials)
        masks = jnp.ones(n_trials)
        q_init = 0.333

        Q_for_policy = associative_scan_q_update(
            stimuli, actions, rewards, masks,
            alpha=0.5,
            q_init=q_init,
            num_stimuli=6,
            num_actions=3,
        )
        # First row should be the initial Q-table (all q_init)
        assert jnp.allclose(Q_for_policy[0], q_init, atol=1e-6), (
            f"Q_for_policy[0] should be all {q_init}, got {Q_for_policy[0]}"
        )

    def test_masked_padding_no_effect(self):
        """Padding trials (mask=0) must not change Q-values.

        Append zero-masked trials; Q-values after real trials must match
        those from an unpadded run on the real trials alone.
        """
        n_real = 15
        n_pad = 10
        stimuli_r, actions_r, rewards_r, masks_r = self._make_block(n_real)

        # Padded version
        stimuli_pad = jnp.concatenate([stimuli_r, jnp.zeros(n_pad, dtype=jnp.int32)])
        actions_pad = jnp.concatenate([actions_r, jnp.zeros(n_pad, dtype=jnp.int32)])
        rewards_pad = jnp.concatenate([rewards_r, jnp.zeros(n_pad)])
        masks_pad = jnp.concatenate([masks_r, jnp.zeros(n_pad)])

        Q_real = associative_scan_q_update(
            stimuli_r, actions_r, rewards_r, masks_r,
            alpha=0.3, q_init=0.5, num_stimuli=6, num_actions=3,
        )
        Q_pad = associative_scan_q_update(
            stimuli_pad, actions_pad, rewards_pad, masks_pad,
            alpha=0.3, q_init=0.5, num_stimuli=6, num_actions=3,
        )

        assert jnp.allclose(Q_real, Q_pad[:n_real], atol=1e-6), (
            "Padding changed Q_for_policy values for real trials"
        )
