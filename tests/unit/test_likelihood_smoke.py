"""W2 smoke tests: all 7 models compile and return finite log-likelihoods with single alpha.

Phase 33 gate: each model's block_likelihood function must:
1. Accept single `alpha` (not alpha + alpha).
2. Return a finite scalar log-likelihood for synthetic data.
3. Import cleanly (no broken references to alpha/alpha).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_block(n_trials: int = 20, seed: int = 7):
    """Generate deterministic synthetic block data."""
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    stimuli = jax.random.randint(k1, (n_trials,), 0, 6)
    actions = jax.random.randint(k2, (n_trials,), 0, 3)
    rewards = jax.random.bernoulli(k3, 0.7, (n_trials,)).astype(jnp.float32)
    mask = jnp.ones(n_trials)
    return stimuli, actions, rewards, mask


def _make_set_sizes(n_trials: int = 20, set_size: int = 3):
    """Set-size array for WM-RL models."""
    return jnp.full(n_trials, set_size, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# M1: Q-Learning
# ---------------------------------------------------------------------------

class TestQLearningSmoke:
    """Smoke tests for M1 Q-learning with single alpha."""

    def test_block_likelihood_finite(self):
        """q_learning_block_likelihood returns finite scalar with single alpha."""
        from rlwm.fitting.models.qlearning import q_learning_block_likelihood
        stimuli, actions, rewards, mask = _make_block()
        ll = q_learning_block_likelihood(
            stimuli, actions, rewards,
            alpha=0.3,
            epsilon=0.05,
            mask=mask,
        )
        assert jnp.isfinite(ll), f"M1 block likelihood not finite: {ll}"
        assert float(ll) < 0, f"Log-likelihood should be negative, got {ll}"

    def test_no_alpha_pos_kwarg(self):
        """Passing alpha_pos= must raise TypeError (old API rejected, Phase 33)."""
        from rlwm.fitting.models.qlearning import q_learning_block_likelihood
        stimuli, actions, rewards, mask = _make_block()
        with pytest.raises(TypeError):
            q_learning_block_likelihood(
                stimuli, actions, rewards,
                alpha_pos=0.3,
                epsilon=0.05, mask=mask,
            )

    def test_pscan_finite(self):
        """q_learning_block_likelihood_pscan returns finite scalar."""
        from rlwm.fitting.models.qlearning import q_learning_block_likelihood_pscan
        stimuli, actions, rewards, mask = _make_block()
        ll = q_learning_block_likelihood_pscan(
            stimuli, actions, rewards,
            alpha=0.3, epsilon=0.05, mask=mask,
        )
        assert jnp.isfinite(ll), f"M1 pscan likelihood not finite: {ll}"


# ---------------------------------------------------------------------------
# M2: WM-RL
# ---------------------------------------------------------------------------

class TestWMRLSmoke:
    """Smoke tests for M2 WM-RL with single alpha."""

    def test_block_likelihood_finite(self):
        """wmrl_block_likelihood returns finite scalar with single alpha."""
        from rlwm.fitting.models.wmrl import wmrl_block_likelihood
        stimuli, actions, rewards, mask = _make_block()
        set_sizes = _make_set_sizes()
        ll = wmrl_block_likelihood(
            stimuli, actions, rewards, set_sizes,
            alpha=0.3, phi=0.8, rho=0.7,
            capacity=3.0, epsilon=0.05, mask=mask,
        )
        assert jnp.isfinite(ll), f"M2 block likelihood not finite: {ll}"

    def test_no_alpha_pos_kwarg(self):
        """Passing alpha_pos= must raise TypeError (old API rejected, Phase 33)."""
        from rlwm.fitting.models.wmrl import wmrl_block_likelihood
        stimuli, actions, rewards, mask = _make_block()
        set_sizes = _make_set_sizes()
        with pytest.raises(TypeError):
            wmrl_block_likelihood(
                stimuli, actions, rewards, set_sizes,
                alpha_pos=0.3,
                phi=0.8, rho=0.7, capacity=3.0, epsilon=0.05, mask=mask,
            )


# ---------------------------------------------------------------------------
# M3: WM-RL+kappa
# ---------------------------------------------------------------------------

class TestWMRLM3Smoke:
    """Smoke tests for M3 with single alpha."""

    def test_block_likelihood_finite(self):
        """wmrl_m3_block_likelihood returns finite scalar."""
        from rlwm.fitting.models.wmrl_m3 import wmrl_m3_block_likelihood
        stimuli, actions, rewards, mask = _make_block()
        set_sizes = _make_set_sizes()
        ll = wmrl_m3_block_likelihood(
            stimuli, actions, rewards, set_sizes,
            alpha=0.3, phi=0.8, rho=0.7,
            capacity=3.0, kappa=0.2, epsilon=0.05, mask=mask,
        )
        assert jnp.isfinite(ll), f"M3 block likelihood not finite: {ll}"

    def test_no_alpha_pos_kwarg(self):
        """Passing alpha_pos= must raise TypeError (old API rejected, Phase 33)."""
        from rlwm.fitting.models.wmrl_m3 import wmrl_m3_block_likelihood
        stimuli, actions, rewards, mask = _make_block()
        set_sizes = _make_set_sizes()
        with pytest.raises(TypeError):
            wmrl_m3_block_likelihood(
                stimuli, actions, rewards, set_sizes,
                alpha_pos=0.3,
                phi=0.8, rho=0.7, capacity=3.0, kappa=0.2, epsilon=0.05, mask=mask,
            )


# ---------------------------------------------------------------------------
# M5: WM-RL+phi_rl
# ---------------------------------------------------------------------------

class TestWMRLM5Smoke:
    """Smoke tests for M5 with single alpha."""

    def test_block_likelihood_finite(self):
        """wmrl_m5_block_likelihood returns finite scalar."""
        from rlwm.fitting.models.wmrl_m5 import wmrl_m5_block_likelihood
        stimuli, actions, rewards, mask = _make_block()
        set_sizes = _make_set_sizes()
        ll = wmrl_m5_block_likelihood(
            stimuli, actions, rewards, set_sizes,
            alpha=0.3, phi=0.8, rho=0.7,
            capacity=3.0, kappa=0.2, phi_rl=0.3, epsilon=0.05, mask=mask,
        )
        assert jnp.isfinite(ll), f"M5 block likelihood not finite: {ll}"

    def test_no_alpha_pos_kwarg(self):
        """Passing alpha_pos= must raise TypeError (old API rejected, Phase 33)."""
        from rlwm.fitting.models.wmrl_m5 import wmrl_m5_block_likelihood
        stimuli, actions, rewards, mask = _make_block()
        set_sizes = _make_set_sizes()
        with pytest.raises(TypeError):
            wmrl_m5_block_likelihood(
                stimuli, actions, rewards, set_sizes,
                alpha_pos=0.3,
                phi=0.8, rho=0.7, capacity=3.0, kappa=0.2, phi_rl=0.3,
                epsilon=0.05, mask=mask,
            )


# ---------------------------------------------------------------------------
# M6a: WM-RL+kappa_s
# ---------------------------------------------------------------------------

class TestWMRLM6aSmoke:
    """Smoke tests for M6a with single alpha."""

    def test_block_likelihood_finite(self):
        """wmrl_m6a_block_likelihood returns finite scalar."""
        from rlwm.fitting.models.wmrl_m6a import wmrl_m6a_block_likelihood
        stimuli, actions, rewards, mask = _make_block()
        set_sizes = _make_set_sizes()
        ll = wmrl_m6a_block_likelihood(
            stimuli, actions, rewards, set_sizes,
            alpha=0.3, phi=0.8, rho=0.7,
            capacity=3.0, kappa_s=0.2, epsilon=0.05, mask=mask,
        )
        assert jnp.isfinite(ll), f"M6a block likelihood not finite: {ll}"

    def test_no_alpha_pos_kwarg(self):
        """Passing alpha_pos= must raise TypeError (old API rejected, Phase 33)."""
        from rlwm.fitting.models.wmrl_m6a import wmrl_m6a_block_likelihood
        stimuli, actions, rewards, mask = _make_block()
        set_sizes = _make_set_sizes()
        with pytest.raises(TypeError):
            wmrl_m6a_block_likelihood(
                stimuli, actions, rewards, set_sizes,
                alpha_pos=0.3,
                phi=0.8, rho=0.7, capacity=3.0, kappa_s=0.2,
                epsilon=0.05, mask=mask,
            )


# ---------------------------------------------------------------------------
# M6b: WM-RL+dual
# ---------------------------------------------------------------------------

class TestWMRLM6bSmoke:
    """Smoke tests for M6b with single alpha."""

    def test_block_likelihood_finite(self):
        """wmrl_m6b_block_likelihood returns finite scalar."""
        from rlwm.fitting.models.wmrl_m6b import wmrl_m6b_block_likelihood
        stimuli, actions, rewards, mask = _make_block()
        set_sizes = _make_set_sizes()
        # kappa = kappa_total * kappa_share; kappa_s = kappa_total * (1 - kappa_share)
        kappa_total, kappa_share = 0.4, 0.5
        ll = wmrl_m6b_block_likelihood(
            stimuli, actions, rewards, set_sizes,
            alpha=0.3, phi=0.8, rho=0.7,
            capacity=3.0,
            kappa=kappa_total * kappa_share,
            kappa_s=kappa_total * (1 - kappa_share),
            epsilon=0.05, mask=mask,
        )
        assert jnp.isfinite(ll), f"M6b block likelihood not finite: {ll}"

    def test_no_alpha_pos_kwarg(self):
        """Passing alpha_pos= must raise TypeError (old API rejected, Phase 33)."""
        from rlwm.fitting.models.wmrl_m6b import wmrl_m6b_block_likelihood
        stimuli, actions, rewards, mask = _make_block()
        set_sizes = _make_set_sizes()
        with pytest.raises(TypeError):
            wmrl_m6b_block_likelihood(
                stimuli, actions, rewards, set_sizes,
                alpha_pos=0.3,
                phi=0.8, rho=0.7, capacity=3.0,
                kappa=0.2, kappa_s=0.2, epsilon=0.05, mask=mask,
            )


# ---------------------------------------------------------------------------
# M4: RLWM-LBA
# ---------------------------------------------------------------------------

class TestWMRLM4Smoke:
    """Smoke tests for M4 (LBA) with single alpha.

    M4 is a NumPyro-only model — no standalone block_likelihood function exists.
    Its likelihood is computed inside wmrl_m4_hierarchical_model.
    We verify: (a) module imports cleanly, (b) no alpha/alpha present,
    (c) the hierarchical model function is importable.
    """

    def test_module_imports_cleanly(self):
        """wmrl_m4 module must import without errors."""
        import importlib
        import sys
        if "rlwm.fitting.models.wmrl_m4" in sys.modules:
            del sys.modules["rlwm.fitting.models.wmrl_m4"]
        mod = importlib.import_module("rlwm.fitting.models.wmrl_m4")
        assert hasattr(mod, "wmrl_m4_hierarchical_model"), (
            "wmrl_m4 must expose wmrl_m4_hierarchical_model"
        )

    def test_no_alpha_in_module(self):
        """wmrl_m4 module must not contain alpha_pos or alpha_neg strings (Phase 33)."""
        import inspect
        import sys
        if "rlwm.fitting.models.wmrl_m4" in sys.modules:
            del sys.modules["rlwm.fitting.models.wmrl_m4"]
        from rlwm.fitting.models import wmrl_m4
        src = inspect.getsource(wmrl_m4)
        assert "alpha_pos" not in src, "wmrl_m4 still contains alpha_pos"
        assert "alpha_neg" not in src, "wmrl_m4 still contains alpha_neg"

    def test_hierarchical_model_callable(self):
        """wmrl_m4_hierarchical_model must be callable (import + introspection)."""
        import inspect

        from rlwm.fitting.models.wmrl_m4 import wmrl_m4_hierarchical_model
        sig = inspect.signature(wmrl_m4_hierarchical_model)
        params = list(sig.parameters.keys())
        assert "participant_data_stacked" in params, (
            f"Expected 'participant_data_stacked' in M4 model signature, got {params}"
        )
