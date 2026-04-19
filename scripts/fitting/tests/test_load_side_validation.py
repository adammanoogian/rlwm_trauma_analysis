"""CLEAN-04 enforcement: no bare NetCDF loads in enumerated downstream consumer scripts.

Every NetCDF load must go through config.load_netcdf_with_validation.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Enumerated consumer files that must NOT contain bare NetCDF calls
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_ENUMERATED_FILES: list[Path] = [
    _PROJECT_ROOT / "scripts" / "14_compare_models.py",
    _PROJECT_ROOT / "scripts" / "18_bayesian_level2_effects.py",
    _PROJECT_ROOT / "scripts" / "21_fit_with_l2.py",
    _PROJECT_ROOT / "scripts" / "21_compute_loo_stacking.py",
    _PROJECT_ROOT / "scripts" / "21_baseline_audit.py",
    _PROJECT_ROOT / "scripts" / "21_model_averaging.py",
    _PROJECT_ROOT / "scripts" / "21_scale_audit.py",
    _PROJECT_ROOT / "scripts" / "visualization" / "plot_posterior_diagnostics.py",
    _PROJECT_ROOT / "scripts" / "visualization" / "plot_group_parameters.py",
    _PROJECT_ROOT / "scripts" / "visualization" / "plot_model_comparison.py",
    _PROJECT_ROOT / "scripts" / "visualization" / "quick_arviz_plots.py",
    _PROJECT_ROOT / "scripts" / "simulations" / "generate_data.py",
    _PROJECT_ROOT / "validation" / "compare_posterior_to_mle.py",
]

# Patterns that must NOT appear in the enumerated files (as active code,
# not within the wrapper itself or in comments).
_FORBIDDEN_PATTERNS: list[str] = [
    r"az\.from_netcdf\(",
    r"xr\.open_dataset\(",
]


def _is_code_line(line: str) -> bool:
    """Return True if the line is not a pure comment or blank line.

    Parameters
    ----------
    line : str
        Raw source line (with leading whitespace preserved).

    Returns
    -------
    bool
        True when the stripped line is non-empty and does not start with ``#``.
    """
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def test_no_bare_az_from_netcdf_in_consumer_scripts() -> None:
    """Every enumerated consumer file must route NetCDF loads through the wrapper.

    Walks each file in the hardcoded enumerated list and asserts that no
    active (non-comment) line contains a bare ``az.from_netcdf(`` or
    ``xr.open_dataset(`` call.

    Raises
    ------
    AssertionError
        If any enumerated file contains a forbidden pattern on a code line.
        The failure message includes the file path, line number, and the
        offending line text.
    """
    violations: list[str] = []

    for fpath in _ENUMERATED_FILES:
        if not fpath.exists():
            # File missing is not a test failure — it may not yet exist on a
            # fresh clone that hasn't run fits.  Flag it as a warning instead.
            violations.append(
                f"[MISSING] {fpath.relative_to(_PROJECT_ROOT)} — "
                f"file not found; add to repo or update enumeration list"
            )
            continue

        source = fpath.read_text(encoding="utf-8")
        lines = source.splitlines()
        for lineno, raw_line in enumerate(lines, start=1):
            if not _is_code_line(raw_line):
                continue
            for pattern in _FORBIDDEN_PATTERNS:
                if re.search(pattern, raw_line):
                    violations.append(
                        f"{fpath.relative_to(_PROJECT_ROOT)}:{lineno}: "
                        f"bare call '{pattern}' found — "
                        f"use config.load_netcdf_with_validation instead.\n"
                        f"  Line: {raw_line.rstrip()}"
                    )

    assert not violations, (
        "CLEAN-04 violation: bare NetCDF load(s) found in enumerated consumer scripts.\n"
        + "\n".join(violations)
    )


def test_no_bare_xr_open_dataset_anywhere() -> None:
    """No .py file in scripts/ or validation/ may contain xr.open_dataset(.

    Walks the entire ``scripts/`` and ``validation/`` trees (excluding
    ``__pycache__/`` and ``tests/fixtures/`` subdirectories) and asserts
    that no Python file contains a bare ``xr.open_dataset(`` call on a
    code line.

    Zero live uses exist today (2026-04-19 baseline); this test prevents
    future reintroduction.

    Raises
    ------
    AssertionError
        If any Python file outside excluded trees contains ``xr.open_dataset(``.
    """
    search_roots: list[Path] = [
        _PROJECT_ROOT / "scripts",
        _PROJECT_ROOT / "validation",
    ]
    # Exclude test files themselves (which legitimately document the pattern in
    # docstrings/comments), __pycache__, and fixture caches.
    excluded_subtrees: tuple[str, ...] = ("__pycache__", "tests/fixtures", "tests/")

    violations: list[str] = []
    pattern = r"xr\.open_dataset\("

    for root in search_roots:
        for fpath in root.rglob("*.py"):
            # Skip excluded subtrees
            rel = str(fpath.relative_to(_PROJECT_ROOT)).replace("\\", "/")
            if any(excl in rel for excl in excluded_subtrees):
                continue

            source = fpath.read_text(encoding="utf-8")
            lines = source.splitlines()
            for lineno, raw_line in enumerate(lines, start=1):
                if not _is_code_line(raw_line):
                    continue
                if re.search(pattern, raw_line):
                    violations.append(
                        f"{fpath.relative_to(_PROJECT_ROOT)}:{lineno}: "
                        f"bare xr.open_dataset( found — "
                        f"load NetCDF via config.load_netcdf_with_validation.\n"
                        f"  Line: {raw_line.rstrip()}"
                    )

    assert not violations, (
        "CLEAN-04 violation: xr.open_dataset( call(s) found in scripts/ or validation/.\n"
        + "\n".join(violations)
    )
