"""K-bounds invariant guard for scripts/fitting/mle_utils.py.

Enforcement mechanism for CLEAN-02 (SC#2): every MLE bounds dict must use
Collins K ∈ [2.0, 6.0] and the source file must contain no legacy [1, 7]
substrings.  These tests are expected to pass immediately because mle_utils.py
was already cleaned before Phase 23; their purpose is to make future regressions
fail CI loudly rather than silently contaminating v5.0 cold-start fits.

See: .planning/phases/23-tech-debt-sweep-pre-flight-cleanup/23-02-PLAN.md
"""

from __future__ import annotations

import pathlib

from scripts.fitting import mle_utils

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BOUNDS_DICTS: list[tuple[str, dict]] = [
    ("WMRL_BOUNDS", mle_utils.WMRL_BOUNDS),
    ("WMRL_M3_BOUNDS", mle_utils.WMRL_M3_BOUNDS),
    ("WMRL_M5_BOUNDS", mle_utils.WMRL_M5_BOUNDS),
    ("WMRL_M6A_BOUNDS", mle_utils.WMRL_M6A_BOUNDS),
    ("WMRL_M6B_BOUNDS", mle_utils.WMRL_M6B_BOUNDS),
    ("WMRL_M4_BOUNDS", mle_utils.WMRL_M4_BOUNDS),
]

_EXPECTED_CAPACITY = (2.0, 6.0)

_LEGACY_SUBSTRINGS: list[str] = [
    "1, 7",
    "[1,7]",
    "[1, 7]",
    "K_BOUNDS_LEGACY",
    "(1.0, 7.0)",
    "(1, 7)",
]

_MLEUTILS_PATH = (
    # tests/integration/<file>.py -> repo root -> scripts/fitting/mle_utils.py
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "fitting"
    / "mle_utils.py"
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mle_capacity_bounds_are_collins() -> None:
    """Assert every MLE bounds dict with a 'capacity' key uses Collins K ∈ [2.0, 6.0].

    Rationale
    ---------
    The K-parameterization was narrowed from [1, 7] to [2, 6] in v4.0 per the
    K-01 identifiability analysis.  This test prevents any future commit from
    widening or changing the capacity bounds in the MLE layer without a
    deliberate, visible CI failure.
    """
    failures: list[str] = []
    for name, bounds_dict in _BOUNDS_DICTS:
        if "capacity" not in bounds_dict:
            continue
        actual = bounds_dict["capacity"]
        if actual != _EXPECTED_CAPACITY:
            failures.append(
                f"dict={name}: expected={_EXPECTED_CAPACITY}, actual={actual}"
            )

    assert not failures, (
        "One or more MLE bounds dicts have non-Collins capacity bounds:\n"
        + "\n".join(f"  {f}" for f in failures)
    )


def test_mle_utils_source_has_no_legacy_k_bounds() -> None:
    """Assert mle_utils.py source text contains no legacy [1, 7] K-bound substrings.

    Rationale
    ---------
    This is a source-level grep invariant matching ROADMAP SC#2 exactly.
    It catches comments, dead code, or accidentally restored branches that
    would not be caught by the runtime dict inspection in
    test_mle_capacity_bounds_are_collins.
    """
    source = _MLEUTILS_PATH.read_text(encoding="utf-8")
    lines = source.splitlines()

    failures: list[str] = []
    for substring in _LEGACY_SUBSTRINGS:
        for lineno, line in enumerate(lines, start=1):
            if substring in line:
                failures.append(
                    f"line {lineno}: found forbidden substring {substring!r}: {line.rstrip()}"
                )

    assert not failures, (
        f"mle_utils.py ({_MLEUTILS_PATH}) contains legacy K-bound substrings:\n"
        + "\n".join(f"  {f}" for f in failures)
    )
