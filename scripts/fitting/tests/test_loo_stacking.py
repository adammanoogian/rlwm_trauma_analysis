"""Smoke tests for ``scripts/21_compute_loo_stacking.py``.

Phase 21 Wave 5 orchestrator unit tests. Exercises the pure-function core
:func:`compute_loo_stacking_bms` with hand-crafted stub ``InferenceData``
objects (built via :func:`arviz.from_dict`) so there is no dependency on
the full baseline fitting pipeline.

Covered scenarios
-----------------
- ``test_loo_stacking_dominant_winner_synthetic``: three stub idatas, one
  with much higher per-observation log-likelihood than the others. The
  top model should receive stacking weight ~1 and the winner_type should
  be ``DOMINANT_SINGLE``.
- ``test_rfx_bms_integration``: verifies the RFX-BMS result dict returned
  from the orchestrator has the expected keys (``alpha``, ``r``, ``xp``,
  ``bor``, ``pxp``) with the expected shapes.

The script ``scripts/21_compute_loo_stacking.py`` has a leading digit in
its filename, which makes direct ``import`` illegal; we load it via
``importlib.util.spec_from_file_location`` following the same pattern as
``test_prior_predictive.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import arviz as az
import numpy as np
import pytest


def _load_orchestrator_module() -> ModuleType:
    """Load ``scripts/21_compute_loo_stacking.py`` by filepath.

    The leading ``21_`` in the filename forbids a normal ``import`` — we
    use :func:`importlib.util.spec_from_file_location` to get the module
    object anyway. This mirrors the pattern in
    :mod:`scripts.fitting.tests.test_prior_predictive`.

    Returns
    -------
    types.ModuleType
        The loaded orchestrator module exposing
        :func:`compute_loo_stacking_bms` and :func:`main`.
    """
    mod_path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "21_compute_loo_stacking.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_loo_stacking_test_mod", mod_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_stub_idata(
    log_lik_mean: float,
    seed: int,
    n_chain: int = 2,
    n_draw: int = 50,
    n_ppt: int = 8,
    n_trial: int = 30,
) -> az.InferenceData:
    """Build a stub ``InferenceData`` with known log-likelihood magnitude.

    The log-likelihood values are drawn i.i.d. ``Normal(log_lik_mean, 0.05)``
    with a small jitter to avoid az.compare's degenerate-matrix path. The
    participant coordinate is an integer range ``0..n_ppt-1`` (matching
    the convention in the real baseline NetCDFs).

    Parameters
    ----------
    log_lik_mean : float
        Mean of the per-observation log-likelihood. More negative means a
        worse-fitting model.
    seed : int
        RNG seed for reproducibility.
    n_chain, n_draw, n_ppt, n_trial : int
        Dimensions of the synthetic log-likelihood tensor.

    Returns
    -------
    arviz.InferenceData
        Stub idata with posterior + log_likelihood groups and a
        ``participant`` coordinate on the ``obs`` variable.
    """
    rng = np.random.default_rng(seed)
    log_lik = rng.normal(log_lik_mean, 0.05, (n_chain, n_draw, n_ppt, n_trial))
    posterior = rng.standard_normal((n_chain, n_draw, 1))
    idata = az.from_dict(
        posterior={"x": posterior},
        log_likelihood={"obs": log_lik},
        coords={"participant": np.arange(n_ppt, dtype=int)},
        dims={"obs": ["participant", "trial_padded"]},
    )
    return idata


def test_loo_stacking_dominant_winner_synthetic() -> None:
    """Dominant model wins — winner_type == 'DOMINANT_SINGLE', winner is top.

    Constructs three stub idatas with very different per-observation
    log-likelihood means (``-0.2``, ``-5.0``, ``-10.0``). Model A is the
    runaway best fit; stacking weight should concentrate on A and the
    verdict should be ``DOMINANT_SINGLE`` with A as the sole winner.
    """
    mod = _load_orchestrator_module()

    idata_A = _make_stub_idata(log_lik_mean=-0.2, seed=0)
    idata_B = _make_stub_idata(log_lik_mean=-5.0, seed=1)
    idata_C = _make_stub_idata(log_lik_mean=-10.0, seed=2)

    compare_dict = {"A": idata_A, "B": idata_B, "C": idata_C}
    result = mod.compute_loo_stacking_bms(compare_dict)

    assert result["winner_type"] == "DOMINANT_SINGLE", (
        f"Expected DOMINANT_SINGLE; got {result['winner_type']}. "
        f"Winners: {result['winners']}, "
        f"weights: {result['comparison']['weight'].to_dict()}"
    )
    assert result["winners"] == ["A"], (
        f"Expected A to be sole winner; got {result['winners']}"
    )

    # Stacking weights should concentrate on A.
    comparison = result["comparison"]
    a_weight = float(comparison.loc["A", "weight"])
    assert a_weight >= 0.95, (
        f"Expected stacking weight on A >= 0.95; got {a_weight:.4f}"
    )

    # Pareto-k percentages exposed per model.
    assert set(result["pct_high_per_model"].keys()) == {"A", "B", "C"}

    # Participant IDs should match the stub coord exactly.
    np.testing.assert_array_equal(
        result["participant_ids"], np.arange(8, dtype=int)
    )


def test_loo_stacking_requires_two_models() -> None:
    """Singleton compare_dict raises ValueError (<2 models)."""
    mod = _load_orchestrator_module()
    idata_A = _make_stub_idata(log_lik_mean=-0.2, seed=0)

    with pytest.raises(ValueError, match="at least"):
        mod.compute_loo_stacking_bms({"A": idata_A})


def test_loo_stacking_force_winners_override() -> None:
    """--force-winners bypasses automatic determination and returns given set."""
    mod = _load_orchestrator_module()

    idata_A = _make_stub_idata(log_lik_mean=-0.2, seed=0)
    idata_B = _make_stub_idata(log_lik_mean=-5.0, seed=1)

    compare_dict = {"A": idata_A, "B": idata_B}
    # A is the auto-winner, but force B.
    result = mod.compute_loo_stacking_bms(
        compare_dict, force_winners=["B"]
    )

    assert result["winner_type"] == "FORCED"
    assert result["winners"] == ["B"]


def test_loo_stacking_force_winners_unknown_name() -> None:
    """Unknown display name in --force-winners raises ValueError."""
    mod = _load_orchestrator_module()

    idata_A = _make_stub_idata(log_lik_mean=-0.2, seed=0)
    idata_B = _make_stub_idata(log_lik_mean=-5.0, seed=1)

    with pytest.raises(ValueError, match="not in compare_dict"):
        mod.compute_loo_stacking_bms(
            {"A": idata_A, "B": idata_B},
            force_winners=["NotAModel"],
        )


@pytest.mark.slow
def test_rfx_bms_integration() -> None:
    """RFX-BMS result dict has PXP + expected keys with correct shapes.

    Marked slow because :func:`rfx_bms` runs 1M Dirichlet MC samples by
    default. The orchestrator call path wraps :func:`rfx_bms`, so this
    test confirms the integration (shape/key contract) rather than the
    numerical correctness of RFX-BMS itself (that is covered in
    :mod:`scripts.fitting.tests.test_bms`).
    """
    mod = _load_orchestrator_module()

    idata_A = _make_stub_idata(log_lik_mean=-0.2, seed=0)
    idata_B = _make_stub_idata(log_lik_mean=-5.0, seed=1)

    compare_dict = {"A": idata_A, "B": idata_B}
    result = mod.compute_loo_stacking_bms(compare_dict)

    bms_result = result["bms_result"]
    assert set(bms_result.keys()) == {"alpha", "r", "xp", "bor", "pxp"}

    alpha = np.asarray(bms_result["alpha"])
    r = np.asarray(bms_result["r"])
    xp = np.asarray(bms_result["xp"])
    pxp = np.asarray(bms_result["pxp"])
    bor = float(bms_result["bor"])

    # Shapes: one value per model.
    assert alpha.shape == (2,)
    assert r.shape == (2,)
    assert xp.shape == (2,)
    assert pxp.shape == (2,)

    # Frequencies + exceedances sum to 1.
    np.testing.assert_allclose(r.sum(), 1.0, atol=1e-10)
    np.testing.assert_allclose(xp.sum(), 1.0, atol=1e-10)
    np.testing.assert_allclose(pxp.sum(), 1.0, atol=1e-10)

    # BOR in [0, 1].
    assert 0.0 <= bor <= 1.0

    # PXP formula: pxp = (1 - bor) * xp + bor / K (Rigoux 2014).
    k = pxp.shape[0]
    np.testing.assert_allclose(pxp, (1.0 - bor) * xp + bor / k, atol=1e-12)


def test_loo_stacking_participant_mismatch_raises() -> None:
    """Different participant sets across models raises ValueError.

    RFX-BMS requires an identical cohort across all models — the
    per-participant log-evidence matrix must have consistent row
    ordering. Two paths reject a mismatched cohort:

    1. ArviZ's ``az.compare`` short-circuits with
       "The number of observations should be the same across all models"
       because its LOO matrix assembly refuses unequal-row-count inputs.
    2. If a cohort mismatch somehow slipped past ArviZ (e.g., same
       observation count but different participant labels), the
       orchestrator's explicit post-loop participant-ID check would
       raise ``Participant mismatch ...``.

    Either path is an acceptable reject; we match with a regex union.
    """
    mod = _load_orchestrator_module()

    idata_A = _make_stub_idata(log_lik_mean=-0.2, seed=0, n_ppt=8)
    idata_B = _make_stub_idata(log_lik_mean=-5.0, seed=1, n_ppt=6)

    with pytest.raises(
        ValueError,
        match=r"(Participant mismatch|number of observations should be the same)",
    ):
        mod.compute_loo_stacking_bms({"A": idata_A, "B": idata_B})
