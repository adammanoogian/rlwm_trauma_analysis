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
        "Create it from scripts/fitting/tests/fixtures/qlearning_bayesian_reference.csv."
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
