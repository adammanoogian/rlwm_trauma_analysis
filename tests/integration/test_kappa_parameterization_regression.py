"""SC-3 bit-equivalence regression test.

Asserts that ``--kappa-parameterization=convex`` reproduces v5.0 pre-Phase-32
log-likelihoods exactly (within 1e-10) on a fixed 10-trial toy input for
the four perseveration-bearing choice-only models (M3 / M5 / M6a / M6b).

Reference values captured: 2026-05-03T14:20:39Z from SHA
9adc858811a62fc056833d5fc726de727533a65b via Task 0 of plan 32-04
(see .planning/phases/32-mcmc-methodology-update/32-04-references.json).

Fail mode: if convex-mode LL diverges from REF_* by more than 1e-10, the
convex revert path is broken and Phase 32 must NOT ship. Fix the broken
parameterization branch before merging.

Fixture rationale (10-trial set_size=3, 3 stimuli, 3 actions): chosen so
the JAX scan exercises both the first-trial last_action=-1 sentinel branch
AND repeated stimulus encounters (per-stimulus carry update). Set sizes
is a length-N_trials array (NOT a Python scalar) because the JAX
likelihood functions index it per-trial inside the scan body.

Phase 32-04 Task 6.5; see .planning/phases/32-mcmc-methodology-update/
32-04-PLAN.md.
"""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from src.rlwm.fitting.models.wmrl_m3 import wmrl_m3_block_likelihood
from src.rlwm.fitting.models.wmrl_m5 import wmrl_m5_block_likelihood
from src.rlwm.fitting.models.wmrl_m6a import wmrl_m6a_block_likelihood
from src.rlwm.fitting.models.wmrl_m6b import wmrl_m6b_block_likelihood

# Pre-Phase-32 reference values — DO NOT MODIFY without re-deriving from a
# clean v5.0 checkout. See module docstring + Task 0 JSON
# (.planning/phases/32-mcmc-methodology-update/32-04-references.json).
REF_M3: float = -17.76891326904297
REF_M5: float = -17.65542221069336
REF_M6A: float = -15.698081016540527
REF_M6B: float = -17.142436981201172

TOL: float = 1e-10  # bit-equivalence tolerance (float32 same-bits)


@pytest.fixture
def toy_block() -> dict:
    """Fixed 10-trial toy block. IDENTICAL to Task 0 capture-script fixture."""
    return dict(
        stimuli=jnp.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]),
        actions=jnp.array([0, 1, 2, 1, 0, 2, 0, 2, 1, 0]),
        rewards=jnp.array(
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        ),
        # set_sizes is a length-N_trials array, not a scalar. The JAX
        # likelihood indexes it per-trial inside the scan body.
        set_sizes=jnp.full(10, 3),
    )


@pytest.fixture
def shared_params() -> dict:
    """Per-trial RLWM kwargs. IDENTICAL to Task 0 capture-script fixture."""
    return dict(
        alpha_pos=0.3,
        alpha_neg=0.2,
        phi=0.05,
        rho=0.7,
        capacity=4.0,
        epsilon=0.05,
        num_stimuli=3,
        num_actions=3,
    )


@pytest.mark.scientific
def test_m3_convex_matches_v5_reference(toy_block, shared_params) -> None:
    """M3: convex-mode log-likelihood must match v5.0 pre-Phase-32 reference."""
    ll = float(
        wmrl_m3_block_likelihood(
            **toy_block,
            kappa=0.4,
            **shared_params,
            parameterization="convex",
        )
    )
    assert abs(ll - REF_M3) < TOL, (
        f"M3 convex regression broken: expected {REF_M3} got {ll} "
        f"(diff {abs(ll - REF_M3):.2e}, tol {TOL:.0e}). The convex revert "
        f"path must reproduce v5.0 pre-Phase-32 fits bit-for-bit."
    )


@pytest.mark.scientific
def test_m5_convex_matches_v5_reference(toy_block, shared_params) -> None:
    """M5: convex-mode log-likelihood must match v5.0 pre-Phase-32 reference."""
    ll = float(
        wmrl_m5_block_likelihood(
            **toy_block,
            kappa=0.4,
            phi_rl=0.1,
            **shared_params,
            parameterization="convex",
        )
    )
    assert abs(ll - REF_M5) < TOL, (
        f"M5 convex regression broken: expected {REF_M5} got {ll} "
        f"(diff {abs(ll - REF_M5):.2e}, tol {TOL:.0e})."
    )


@pytest.mark.scientific
def test_m6a_convex_matches_v5_reference(toy_block, shared_params) -> None:
    """M6a: convex-mode log-likelihood must match v5.0 pre-Phase-32 reference."""
    ll = float(
        wmrl_m6a_block_likelihood(
            **toy_block,
            kappa_s=0.4,
            **shared_params,
            parameterization="convex",
        )
    )
    assert abs(ll - REF_M6A) < TOL, (
        f"M6a convex regression broken: expected {REF_M6A} got {ll} "
        f"(diff {abs(ll - REF_M6A):.2e}, tol {TOL:.0e})."
    )


@pytest.mark.scientific
def test_m6b_convex_matches_v5_reference(toy_block, shared_params) -> None:
    """M6b: convex-mode log-likelihood must match v5.0 pre-Phase-32 reference."""
    ll = float(
        wmrl_m6b_block_likelihood(
            **toy_block,
            kappa=0.3,
            kappa_s=0.1,
            **shared_params,
            parameterization="convex",
        )
    )
    assert abs(ll - REF_M6B) < TOL, (
        f"M6b convex regression broken: expected {REF_M6B} got {ll} "
        f"(diff {abs(ll - REF_M6B):.2e}, tol {TOL:.0e})."
    )
