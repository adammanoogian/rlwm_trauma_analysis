"""Unit tests for scripts/fitting/bms.py.

Covers the four canonical regimes of random-effects Bayesian Model Selection:

* Uniform log-evidence -> BOR near 1, XP ~ 1/K, PXP ~ 1/K (null regime).
* Single dominant model -> XP_dominant ~ 1, BOR near 0, PXP_dominant ~ 1.
* Two-winner heterogeneous mixture -> XP concentrates on both winners.
* Input validation -> ValueError on 1-D arrays and non-finite entries.

Plus a numerical identity test that the PXP = (1 - BOR) * XP + BOR / K formula
holds on a live output.

References
----------
Stephan et al. 2009 (DOI 10.1016/j.neuroimage.2009.03.025); Rigoux et al. 2014
(DOI 10.1016/j.neuroimage.2013.08.065).
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.fitting.bms import rfx_bms


def test_rfx_bms_uniform() -> None:
    """Uniform log-evidence -> null regime: XP ~ 1/K, BOR high, PXP ~ 1/K.

    When every participant assigns identical log evidence to every model,
    there is no information to prefer any model and the null hypothesis of
    equal frequencies is strongly supported. Stephan (2009) / Rigoux (2014)
    predict XP ~ 1/K (equal by symmetry) and BOR ~ 1 (null plausible), so
    PXP = (1 - BOR) * XP + BOR / K collapses to 1/K.
    """
    n_subjects, n_models = 40, 4
    log_evidence = np.zeros((n_subjects, n_models))

    result = rfx_bms(log_evidence, seed=0)

    np.testing.assert_allclose(
        result["xp"], np.full(n_models, 0.25), atol=0.02
    )
    assert result["bor"] > 0.7, (
        f"Expected BOR > 0.7 for uniform log-evidence; got {result['bor']:.4f}"
    )
    np.testing.assert_allclose(
        result["pxp"], np.full(n_models, 0.25), atol=0.02
    )
    np.testing.assert_allclose(result["xp"].sum(), 1.0, atol=1e-10)
    np.testing.assert_allclose(result["pxp"].sum(), 1.0, atol=1e-10)


def test_rfx_bms_dominant() -> None:
    """One dominant model -> XP_0 ~ 1, BOR low, PXP_0 high.

    When every participant prefers model 0 by a large margin, the alternative
    (heterogeneous) free energy vastly exceeds the null and BOR collapses to
    near zero. XP and PXP should both concentrate mass on model 0.
    """
    n_subjects, n_models = 40, 4
    log_evidence = np.tile(np.array([10.0, 0.0, 0.0, 0.0]), (n_subjects, 1))

    result = rfx_bms(log_evidence, seed=0)

    assert result["xp"][0] > 0.99, (
        f"Expected XP[0] > 0.99 for dominant model; got {result['xp'][0]:.4f}"
    )
    assert result["bor"] < 0.05, (
        f"Expected BOR < 0.05 for dominant model; got {result['bor']:.4f}"
    )
    assert result["pxp"][0] > 0.95, (
        f"Expected PXP[0] > 0.95 for dominant model; got {result['pxp'][0]:.4f}"
    )
    np.testing.assert_allclose(result["xp"].sum(), 1.0, atol=1e-10)
    np.testing.assert_allclose(result["pxp"].sum(), 1.0, atol=1e-10)


def test_rfx_bms_mixed_heterogeneous() -> None:
    """Two-winner mixture -> XP concentrates on both winners, null rejected.

    Twenty participants prefer model 0 by +5, twenty prefer model 1 by +5,
    and models 2 and 3 are never preferred. Under the RFX-BMS model this is
    clearly heterogeneous (null implausible) but not dominant: XP should be
    split roughly 50/50 between models 0 and 1.
    """
    n_subjects_per_group = 20
    n_models = 4
    log_evidence = np.zeros((2 * n_subjects_per_group, n_models))
    log_evidence[:n_subjects_per_group, 0] = 5.0
    log_evidence[n_subjects_per_group:, 1] = 5.0

    result = rfx_bms(log_evidence, seed=0)

    # Models 0 and 1 should dominate XP.
    assert result["xp"][0] + result["xp"][1] > 0.98, (
        f"XP for models 0+1 should dominate; got {result['xp']}"
    )
    np.testing.assert_allclose(
        result["xp"][0], result["xp"][1], atol=0.1
    )
    assert result["xp"][2] < 0.02 and result["xp"][3] < 0.02, (
        f"XP for non-winning models should be ~0; got {result['xp']}"
    )
    assert result["bor"] < 0.2, (
        f"Expected BOR < 0.2 for heterogeneous data; got {result['bor']:.4f}"
    )
    np.testing.assert_allclose(result["xp"].sum(), 1.0, atol=1e-10)
    np.testing.assert_allclose(result["pxp"].sum(), 1.0, atol=1e-10)


def test_rfx_bms_input_validation() -> None:
    """Shape and finiteness validation raise ValueError with informative text."""
    # 1-D input rejected.
    with pytest.raises(ValueError, match="2-D"):
        rfx_bms(np.zeros(10))

    # 3-D input rejected.
    with pytest.raises(ValueError, match="2-D"):
        rfx_bms(np.zeros((5, 3, 2)))

    # NaN entries rejected.
    bad = np.zeros((5, 3))
    bad[2, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        rfx_bms(bad)

    # +inf entries rejected.
    bad_inf = np.zeros((5, 3))
    bad_inf[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        rfx_bms(bad_inf)

    # Non-positive alpha0 rejected.
    with pytest.raises(ValueError, match="alpha0"):
        rfx_bms(np.zeros((5, 3)), alpha0=0.0)


def test_pxp_formula() -> None:
    """Numerically verify PXP = (1 - BOR) * XP + BOR / K on a live output.

    Reconstructs the protected exceedance probability from the returned xp
    and bor using the Rigoux (2014) formula and asserts it matches the
    returned pxp to machine precision.
    """
    log_evidence = np.tile(np.array([3.0, 1.0, 0.0, -1.0]), (30, 1))

    result = rfx_bms(log_evidence, seed=0)

    xp = result["xp"]
    bor = result["bor"]
    pxp = result["pxp"]
    k = xp.shape[0]

    pxp_reconstructed = (1.0 - bor) * xp + bor / k

    np.testing.assert_allclose(pxp, pxp_reconstructed, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(pxp.sum(), 1.0, atol=1e-10)
