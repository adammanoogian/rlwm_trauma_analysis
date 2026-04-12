"""Tests for pointwise log-likelihood return path (INFRA-03 prerequisite)."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from scripts.fitting.jax_likelihoods import (
    MAX_TRIALS_PER_BLOCK,
    pad_block_to_max,
    q_learning_block_likelihood,
    q_learning_multiblock_likelihood_stacked,
    wmrl_block_likelihood,
    wmrl_m3_block_likelihood,
    wmrl_m5_block_likelihood,
    wmrl_m6a_block_likelihood,
    wmrl_m6b_block_likelihood,
    wmrl_m3_multiblock_likelihood_stacked,
    wmrl_m5_multiblock_likelihood_stacked,
    wmrl_m6a_multiblock_likelihood_stacked,
    wmrl_m6b_multiblock_likelihood_stacked,
    wmrl_multiblock_likelihood_stacked,
)

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

N_STIM = 6
N_ACT = 3
N_REAL = 20  # real trials per block (remainder will be padding)
N_BLOCKS = 3
SET_SIZE = 6


def _make_qlearning_block() -> tuple[jnp.ndarray, ...]:
    """Return (stimuli, actions, rewards, mask) padded to MAX_TRIALS_PER_BLOCK."""
    rng = np.random.default_rng(42)
    s = rng.integers(0, N_STIM, size=N_REAL).astype(np.int32)
    a = rng.integers(0, N_ACT, size=N_REAL).astype(np.int32)
    r = rng.binomial(1, 0.6, size=N_REAL).astype(np.float32)
    padded = pad_block_to_max(
        jnp.array(s), jnp.array(a), jnp.array(r)
    )
    stimuli_p, actions_p, rewards_p, mask_p = padded
    return stimuli_p, actions_p, rewards_p, mask_p


def _make_wmrl_block() -> tuple[jnp.ndarray, ...]:
    """Return (stimuli, actions, rewards, set_sizes, mask) padded to MAX_TRIALS_PER_BLOCK.

    pad_block_to_max with set_sizes returns:
        (stimuli_padded, actions_padded, rewards_padded, set_sizes_padded, mask)
    """
    rng = np.random.default_rng(99)
    s = rng.integers(0, N_STIM, size=N_REAL).astype(np.int32)
    a = rng.integers(0, N_ACT, size=N_REAL).astype(np.int32)
    r = rng.binomial(1, 0.6, size=N_REAL).astype(np.float32)
    ss = np.full(N_REAL, SET_SIZE, dtype=np.int32)
    stimuli_p, actions_p, rewards_p, set_sizes_p, mask_p = pad_block_to_max(
        jnp.array(s), jnp.array(a), jnp.array(r),
        set_sizes=jnp.array(ss),
    )
    return stimuli_p, actions_p, rewards_p, set_sizes_p, mask_p


def _make_stacked_qlearning(n_blocks: int = N_BLOCKS) -> dict:
    """Create stacked arrays for q_learning_multiblock_likelihood_stacked."""
    blocks = [_make_qlearning_block() for _ in range(n_blocks)]
    stimuli_stacked = jnp.stack([b[0] for b in blocks])
    actions_stacked = jnp.stack([b[1] for b in blocks])
    rewards_stacked = jnp.stack([b[2] for b in blocks])
    masks_stacked = jnp.stack([b[3] for b in blocks])
    return dict(
        stimuli_stacked=stimuli_stacked,
        actions_stacked=actions_stacked,
        rewards_stacked=rewards_stacked,
        masks_stacked=masks_stacked,
        alpha_pos=0.3,
        alpha_neg=0.15,
    )


def _make_stacked_wmrl(n_blocks: int = N_BLOCKS) -> dict:
    """Create stacked arrays for wmrl_multiblock_likelihood_stacked."""
    blocks = [_make_wmrl_block() for _ in range(n_blocks)]
    stimuli_stacked = jnp.stack([b[0] for b in blocks])
    actions_stacked = jnp.stack([b[1] for b in blocks])
    rewards_stacked = jnp.stack([b[2] for b in blocks])
    set_sizes_stacked = jnp.stack([b[3] for b in blocks])
    masks_stacked = jnp.stack([b[4] for b in blocks])
    return dict(
        stimuli_stacked=stimuli_stacked,
        actions_stacked=actions_stacked,
        rewards_stacked=rewards_stacked,
        set_sizes_stacked=set_sizes_stacked,
        masks_stacked=masks_stacked,
        alpha_pos=0.3,
        alpha_neg=0.15,
        phi=0.1,
        rho=0.7,
        capacity=4.0,
    )


# ---------------------------------------------------------------------------
# Block likelihood tests (q_learning)
# ---------------------------------------------------------------------------


def test_qlearning_block_default_returns_scalar():
    """Default call (no return_pointwise) must return a scalar float."""
    stimuli, actions, rewards, mask = _make_qlearning_block()
    result = q_learning_block_likelihood(stimuli, actions, rewards, 0.3, 0.15, mask=mask)
    # Must be a scalar (0-d JAX array), NOT a tuple
    assert not isinstance(result, tuple), "Default should return scalar, not tuple"
    assert result.ndim == 0, f"Expected 0-d array, got shape {result.shape}"


def test_qlearning_block_pointwise_returns_tuple():
    """return_pointwise=True must return (scalar, 1-d array) with correct shape."""
    stimuli, actions, rewards, mask = _make_qlearning_block()
    result = q_learning_block_likelihood(
        stimuli, actions, rewards, 0.3, 0.15, mask=mask, return_pointwise=True
    )
    assert isinstance(result, tuple), "return_pointwise=True should return tuple"
    total, per_trial = result
    assert total.ndim == 0, f"total_log_lik should be 0-d, got shape {total.shape}"
    assert per_trial.shape == (MAX_TRIALS_PER_BLOCK,), (
        f"Expected shape ({MAX_TRIALS_PER_BLOCK},), got {per_trial.shape}"
    )


def test_qlearning_block_pointwise_sum_equals_total():
    """Sum of per-trial log-probs must equal total log-lik (within float tolerance)."""
    stimuli, actions, rewards, mask = _make_qlearning_block()
    total, per_trial = q_learning_block_likelihood(
        stimuli, actions, rewards, 0.3, 0.15, mask=mask, return_pointwise=True
    )
    np.testing.assert_allclose(
        float(total), float(per_trial.sum()),
        atol=1e-5,
        err_msg=f"sum mismatch: total={float(total)}, sum={float(per_trial.sum())}",
    )


def test_qlearning_block_padding_zeros():
    """Padding trials (mask=0) must have log_prob = 0.0."""
    stimuli, actions, rewards, mask = _make_qlearning_block()
    _, per_trial = q_learning_block_likelihood(
        stimuli, actions, rewards, 0.3, 0.15, mask=mask, return_pointwise=True
    )
    # Padding positions are where mask == 0
    padding_mask = mask == 0.0
    padding_probs = per_trial[padding_mask]
    np.testing.assert_allclose(
        np.array(padding_probs), 0.0,
        atol=1e-7,
        err_msg="Padding trials must have log_prob = 0.0",
    )


# ---------------------------------------------------------------------------
# Stacked wrapper test (q_learning)
# ---------------------------------------------------------------------------


def test_qlearning_stacked_pointwise_shape():
    """Stacked wrapper with return_pointwise=True must return flat (n_blocks * max_trials,)."""
    kwargs = _make_stacked_qlearning(n_blocks=3)
    result = q_learning_multiblock_likelihood_stacked(
        **kwargs, return_pointwise=True
    )
    assert isinstance(result, tuple), "return_pointwise=True should return tuple"
    total, per_trial = result
    expected_len = N_BLOCKS * MAX_TRIALS_PER_BLOCK
    assert per_trial.shape == (expected_len,), (
        f"Expected shape ({expected_len},), got {per_trial.shape}"
    )


def test_qlearning_stacked_scalar_default():
    """Stacked wrapper default must return scalar (backward compat)."""
    kwargs = _make_stacked_qlearning(n_blocks=3)
    result = q_learning_multiblock_likelihood_stacked(**kwargs)
    assert not isinstance(result, tuple), "Default should return scalar, not tuple"
    assert result.ndim == 0


# ---------------------------------------------------------------------------
# Parametric: all 6 block likelihood functions have pointwise
# ---------------------------------------------------------------------------


def _call_block_fn(fn, padded_block):
    """Call a block likelihood function with appropriate args."""
    if fn is q_learning_block_likelihood:
        stimuli, actions, rewards, mask = padded_block[:4]
        return fn(stimuli, actions, rewards, 0.3, 0.15, mask=mask, return_pointwise=True)
    else:
        # All WMRL variants need set_sizes
        stimuli, actions, rewards, set_sizes, mask = padded_block
        base_kwargs = dict(
            stimuli=stimuli, actions=actions, rewards=rewards,
            set_sizes=set_sizes,
            alpha_pos=0.3, alpha_neg=0.15,
            phi=0.1, rho=0.7, capacity=4.0,
            mask=mask, return_pointwise=True,
        )
        if fn is wmrl_block_likelihood:
            return fn(**base_kwargs)
        elif fn is wmrl_m3_block_likelihood:
            return fn(**base_kwargs, kappa=0.1)
        elif fn is wmrl_m5_block_likelihood:
            return fn(**base_kwargs, kappa=0.1, phi_rl=0.05)
        elif fn is wmrl_m6a_block_likelihood:
            # M6a uses kappa_s (stimulus-specific), not kappa (global)
            return fn(**base_kwargs, kappa_s=0.1)
        elif fn is wmrl_m6b_block_likelihood:
            # M6b uses both kappa (global) and kappa_s (stimulus-specific)
            return fn(**base_kwargs, kappa=0.05, kappa_s=0.05)
        else:
            raise ValueError(f"Unknown function: {fn}")


BLOCK_FNS = [
    q_learning_block_likelihood,
    wmrl_block_likelihood,
    wmrl_m3_block_likelihood,
    wmrl_m5_block_likelihood,
    wmrl_m6a_block_likelihood,
    wmrl_m6b_block_likelihood,
]


@pytest.mark.parametrize("fn", BLOCK_FNS, ids=[f.__name__ for f in BLOCK_FNS])
def test_all_block_functions_have_pointwise(fn):
    """All 6 block likelihood functions return (scalar, array) with return_pointwise=True."""
    if fn is q_learning_block_likelihood:
        padded_block = _make_qlearning_block()
    else:
        padded_block = _make_wmrl_block()

    result = _call_block_fn(fn, padded_block)
    assert isinstance(result, tuple), f"{fn.__name__}: return_pointwise=True must return tuple"
    total, per_trial = result
    assert total.ndim == 0, f"{fn.__name__}: total must be scalar"
    assert per_trial.shape == (MAX_TRIALS_PER_BLOCK,), (
        f"{fn.__name__}: expected shape ({MAX_TRIALS_PER_BLOCK},), got {per_trial.shape}"
    )


@pytest.mark.parametrize("fn", BLOCK_FNS, ids=[f.__name__ for f in BLOCK_FNS])
def test_pointwise_sum_equals_total(fn):
    """For each block function, per-trial log-prob sum equals total log-lik."""
    if fn is q_learning_block_likelihood:
        padded_block = _make_qlearning_block()
    else:
        padded_block = _make_wmrl_block()

    result = _call_block_fn(fn, padded_block)
    total, per_trial = result
    np.testing.assert_allclose(
        float(total), float(per_trial.sum()),
        atol=1e-5,
        err_msg=f"{fn.__name__}: sum mismatch total={float(total):.6f} vs sum={float(per_trial.sum()):.6f}",
    )


# ---------------------------------------------------------------------------
# Parametric: all 6 stacked functions have pointwise
# ---------------------------------------------------------------------------


STACKED_FNS_AND_BUILDERS = [
    (q_learning_multiblock_likelihood_stacked, _make_stacked_qlearning),
    (wmrl_multiblock_likelihood_stacked, _make_stacked_wmrl),
    (wmrl_m3_multiblock_likelihood_stacked, _make_stacked_wmrl),
    (wmrl_m5_multiblock_likelihood_stacked, _make_stacked_wmrl),
    (wmrl_m6a_multiblock_likelihood_stacked, _make_stacked_wmrl),
    (wmrl_m6b_multiblock_likelihood_stacked, _make_stacked_wmrl),
]

STACKED_EXTRA_KWARGS: dict = {
    wmrl_m3_multiblock_likelihood_stacked: dict(kappa=0.1),
    wmrl_m5_multiblock_likelihood_stacked: dict(kappa=0.1, phi_rl=0.05),
    wmrl_m6a_multiblock_likelihood_stacked: dict(kappa_s=0.1),
    wmrl_m6b_multiblock_likelihood_stacked: dict(kappa=0.05, kappa_s=0.05),
}


@pytest.mark.parametrize(
    "fn, builder",
    STACKED_FNS_AND_BUILDERS,
    ids=[fn.__name__ for fn, _ in STACKED_FNS_AND_BUILDERS],
)
def test_all_stacked_functions_have_pointwise(fn, builder):
    """All 6 stacked wrappers return tuple with flat (n_blocks * max_trials,) shape."""
    kwargs = builder(n_blocks=N_BLOCKS)
    extra = STACKED_EXTRA_KWARGS.get(fn, {})
    result = fn(**kwargs, **extra, return_pointwise=True)
    assert isinstance(result, tuple), f"{fn.__name__}: return_pointwise=True must return tuple"
    total, per_trial = result
    expected_len = N_BLOCKS * MAX_TRIALS_PER_BLOCK
    assert per_trial.shape == (expected_len,), (
        f"{fn.__name__}: expected flat shape ({expected_len},), got {per_trial.shape}"
    )


@pytest.mark.parametrize(
    "fn, builder",
    STACKED_FNS_AND_BUILDERS,
    ids=[fn.__name__ for fn, _ in STACKED_FNS_AND_BUILDERS],
)
def test_stacked_pointwise_sum_equals_total(fn, builder):
    """For each stacked wrapper, per-trial sum equals total log-lik."""
    kwargs = builder(n_blocks=N_BLOCKS)
    extra = STACKED_EXTRA_KWARGS.get(fn, {})
    total, per_trial = fn(**kwargs, **extra, return_pointwise=True)
    np.testing.assert_allclose(
        float(total), float(per_trial.sum()),
        atol=1e-4,
        err_msg=f"{fn.__name__}: sum mismatch total={float(total):.6f} vs sum={float(per_trial.sum()):.6f}",
    )
