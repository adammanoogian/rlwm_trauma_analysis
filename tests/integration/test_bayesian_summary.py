"""Tests for bayesian_summary_writer schema parity with MLE CSV output.

Verifies that:
1. Output column names match the schema-parity reference.
2. No MLE-only columns (grad_norm, hessian_*, _se, _ci_*, high_correlations) appear.
3. Bayesian-specific columns (_hdi_low, _hdi_high, _sd, convergence diagnostics) are present.
4. The converged flag logic is correct.
5. parameterization_version is written to every row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Columns present in MLE fits but NOT in Bayesian fits
_MLE_ONLY_COLUMNS = {
    "grad_norm",
    "hessian_condition",
    "hessian_invertible",
    "high_correlations",
    "n_successful_starts",
    "n_near_best",
}
# Bayesian-specific column suffixes
_BAYESIAN_SUFFIXES = ("_hdi_low", "_hdi_high", "_sd")
# Convergence diagnostics required in Bayesian output
_CONVERGENCE_COLS = {"max_rhat", "min_ess_bulk", "num_divergences"}
# Columns that must appear verbatim (not as patterns)
_REQUIRED_COLS = {
    "participant_id",
    "nll",
    "aic",
    "bic",
    "aicc",
    "pseudo_r2",
    "n_trials",
    "converged",
    "at_bounds",
    "parameterization_version",
}

_QLEARNING_PARAMS = ["alpha_pos", "alpha_neg", "epsilon"]


def _load_reference_csv() -> pd.DataFrame:
    """Load the canonical reference CSV for qlearning Bayesian fits."""
    ref_path = _FIXTURES_DIR / "qlearning_bayesian_reference.csv"
    assert ref_path.exists(), (
        f"Reference CSV not found at {ref_path}. "
        "Run the fixture generation step first."
    )
    return pd.read_csv(ref_path)


def _build_expected_columns(params: list[str]) -> list[str]:
    """Build the full expected column list from a parameter list."""
    cols = ["participant_id"]
    cols += params
    cols += ["nll", "aic", "bic", "aicc", "pseudo_r2"]
    for p in params:
        cols += [f"{p}_hdi_low", f"{p}_hdi_high", f"{p}_sd"]
    cols += ["max_rhat", "min_ess_bulk", "num_divergences"]
    cols += ["n_trials", "converged", "at_bounds", "parameterization_version"]
    return cols


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reference_csv_exists():
    """Reference CSV fixture must exist."""
    ref_path = _FIXTURES_DIR / "qlearning_bayesian_reference.csv"
    assert ref_path.exists(), (
        f"Expected fixture at {ref_path}. "
        "Create it at tests/integration/fixtures/qlearning_bayesian_reference.csv."
    )


def test_schema_parity_column_names():
    """Reference CSV column names must match the expected schema-parity layout exactly."""
    ref = _load_reference_csv()
    expected = _build_expected_columns(_QLEARNING_PARAMS)

    ref_cols = list(ref.columns)
    assert ref_cols == expected, (
        f"Column mismatch.\n"
        f"  Expected: {expected}\n"
        f"  Got:      {ref_cols}\n"
        f"  Missing from ref:   {[c for c in expected if c not in ref_cols]}\n"
        f"  Extra in ref:       {[c for c in ref_cols if c not in expected]}"
    )


def test_schema_parity_no_mle_only_columns():
    """Reference CSV must NOT contain MLE-only Hessian/gradient columns."""
    ref = _load_reference_csv()
    present_mle_only = _MLE_ONLY_COLUMNS & set(ref.columns)
    assert not present_mle_only, (
        f"Found MLE-only columns in Bayesian reference CSV: {present_mle_only}. "
        "Remove them from the schema."
    )


def test_bayesian_extra_columns_present():
    """Reference CSV must contain HDI and convergence diagnostic columns."""
    ref = _load_reference_csv()
    cols = set(ref.columns)

    # Check HDI suffixes for every parameter
    for param in _QLEARNING_PARAMS:
        for suffix in _BAYESIAN_SUFFIXES:
            col = f"{param}{suffix}"
            assert col in cols, (
                f"Bayesian column '{col}' missing from reference CSV. "
                f"Available columns: {sorted(cols)}"
            )

    # Check convergence diagnostics
    for col in _CONVERGENCE_COLS:
        assert col in cols, (
            f"Convergence diagnostic column '{col}' missing from reference CSV. "
            f"Available columns: {sorted(cols)}"
        )


def test_parameterization_version_present():
    """Every row must have a parameterization_version value."""
    ref = _load_reference_csv()
    assert "parameterization_version" in ref.columns, (
        "parameterization_version column missing from reference CSV."
    )
    assert ref["parameterization_version"].notna().all(), (
        "parameterization_version must be non-null in every row; "
        f"found NaN in rows: {ref[ref['parameterization_version'].isna()].index.tolist()}"
    )


class TestConvergedLogic:
    """Unit tests for the converged flag definition."""

    @staticmethod
    def _converged(max_rhat: float, min_ess: float, num_div: int) -> bool:
        """Mirror the converged logic from write_bayesian_summary."""
        return (
            (not np.isnan(max_rhat) and max_rhat < 1.01)
            and (not np.isnan(min_ess) and min_ess > 400)
            and (num_div == 0)
        )

    def test_converged_all_good(self):
        assert self._converged(1.005, 600.0, 0) is True

    def test_not_converged_high_rhat(self):
        assert self._converged(1.05, 600.0, 0) is False

    def test_not_converged_low_ess(self):
        assert self._converged(1.005, 200.0, 0) is False

    def test_not_converged_has_divergences(self):
        assert self._converged(1.005, 600.0, 3) is False

    def test_not_converged_nan_rhat(self):
        assert self._converged(float("nan"), 600.0, 0) is False

    def test_not_converged_nan_ess(self):
        assert self._converged(1.005, float("nan"), 0) is False

    def test_boundary_rhat_exact(self):
        """Boundary: max_rhat == 1.01 is NOT converged (strict <)."""
        assert self._converged(1.01, 600.0, 0) is False

    def test_boundary_ess_exact(self):
        """Boundary: min_ess == 400 is NOT converged (strict >)."""
        assert self._converged(1.005, 400.0, 0) is False


def test_reference_csv_row_count():
    """Reference CSV should have at least one data row."""
    ref = _load_reference_csv()
    assert len(ref) >= 1, "Reference CSV has no data rows."


def test_reference_csv_participant_id_column():
    """Reference CSV must have participant_id as first column."""
    ref = _load_reference_csv()
    assert ref.columns[0] == "participant_id", (
        f"First column must be 'participant_id'; got '{ref.columns[0]}'."
    )


# ---------------------------------------------------------------------------
# Phase 32-01: Tier-1 BFMI gate + per-chain ESS schema additions
# ---------------------------------------------------------------------------


def _build_minimal_idata(
    n_chains: int = 4,
    n_draws: int = 100,
    n_participants: int = 2,
    *,
    seed: int = 0,
):
    """Construct a minimal ArviZ InferenceData with energy diagnostics.

    Parameters
    ----------
    n_chains : int
        Number of MCMC chains to simulate.
    n_draws : int
        Draws per chain.
    n_participants : int
        Plate (participant) dimension size.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    az.InferenceData
        InferenceData with one plate-indexed parameter (``alpha_pos``)
        and a ``sample_stats.energy`` array sufficient for ``az.bfmi``.
    """
    import arviz as az

    rng = np.random.default_rng(seed)
    posterior = {
        "alpha_pos": rng.uniform(
            0.1, 0.9, size=(n_chains, n_draws, n_participants)
        ).astype(float),
    }
    # Healthy energy: high within-chain variance => BFMI close to 1.
    sample_stats = {
        "energy": rng.normal(
            loc=0.0, scale=1.0, size=(n_chains, n_draws)
        ).astype(float),
        "diverging": np.zeros((n_chains, n_draws), dtype=bool),
    }
    return az.from_dict(posterior=posterior, sample_stats=sample_stats)


def test_summary_includes_bfmi_and_per_chain_ess_columns(tmp_path):
    """Phase 32-01: CSV must gain min_bfmi and per_chain_ess_bulk columns."""
    from scripts.fitting.bayesian_summary_writer import write_bayesian_summary

    idata = _build_minimal_idata(n_chains=4, n_draws=100, n_participants=2)
    out_path = write_bayesian_summary(
        idata,
        model_name="qlearning",
        output_dir=tmp_path,
        param_names=["alpha_pos"],
        participant_ids=["S001", "S002"],
        parameterization_version="test-32-01",
        n_trials_per_participant=[420, 420],
    )

    df = pd.read_csv(out_path)
    assert "min_bfmi" in df.columns, (
        f"Expected min_bfmi column in {out_path}; got {list(df.columns)}"
    )
    assert "per_chain_ess_bulk" in df.columns, (
        f"Expected per_chain_ess_bulk column in {out_path}; "
        f"got {list(df.columns)}"
    )

    # min_bfmi should be a finite positive float on a healthy synthetic fit.
    assert df["min_bfmi"].notna().all(), (
        f"min_bfmi must be populated for every row; got {df['min_bfmi'].tolist()}"
    )
    assert (df["min_bfmi"] > 0).all(), (
        "Expected positive BFMI values on synthetic fit; "
        f"got {df['min_bfmi'].tolist()}"
    )

    # per_chain_ess_bulk should contain n_chains - 1 semicolons (4 chains -> 3 ;).
    per_chain_strs = df["per_chain_ess_bulk"].astype(str).tolist()
    for s in per_chain_strs:
        assert s.count(";") == 3, (
            f"Expected 3 semicolons for 4 chains; got '{s}'"
        )


def test_converged_flag_fails_under_low_bfmi(tmp_path, monkeypatch):
    """Phase 32-01: converged must be False when min BFMI < 0.2.

    Constructs a synthetic idata with healthy R-hat / ESS / divergences,
    then monkey-patches ``az.bfmi`` to return ``[0.05, 0.5, 0.5, 0.5]`` so
    the minimum across chains is 0.05 < 0.2. The converged flag must
    reflect the failed BFMI gate even though all other metrics pass.
    """
    import arviz as az

    from scripts.fitting import bayesian_summary_writer

    idata = _build_minimal_idata(n_chains=4, n_draws=200, n_participants=2)

    def _fake_bfmi(_idata):
        return np.array([0.05, 0.5, 0.5, 0.5])

    # The writer rebinds ``az`` via a function-local ``import arviz as az``,
    # so patching the attribute on the arviz module itself takes effect.
    monkeypatch.setattr(az, "bfmi", _fake_bfmi)

    out_path = bayesian_summary_writer.write_bayesian_summary(
        idata,
        model_name="qlearning",
        output_dir=tmp_path,
        param_names=["alpha_pos"],
        participant_ids=["S001", "S002"],
        parameterization_version="test-32-01-low-bfmi",
        n_trials_per_participant=[420, 420],
    )

    df = pd.read_csv(out_path)
    assert "min_bfmi" in df.columns
    np.testing.assert_allclose(df["min_bfmi"].to_numpy(), 0.05, rtol=1e-6)
    assert not df["converged"].any(), (
        "converged must be False for every row when min_bfmi < 0.2; "
        f"got max_rhat={df['max_rhat'].tolist()}, "
        f"min_ess_bulk={df['min_ess_bulk'].tolist()}, "
        f"min_bfmi={df['min_bfmi'].tolist()}, "
        f"converged={df['converged'].tolist()}"
    )
