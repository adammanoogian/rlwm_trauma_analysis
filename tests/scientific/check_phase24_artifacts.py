"""Phase 24 cold-start artifact audit — CCDS-canonical path constants (stub).

This module is the verification surface for Phase 24's EXEC-02 + EXEC-04
requirements. Wave 0 (plan 24-00) pins the CCDS-canonical path expectations
as module-level constants so the Wave 2 implementation (plan 24-02)
cannot drift back to the legacy `output/bayesian/...` layout.

CCDS layout source: docs/PROJECT_STRUCTURE.md (Phase 31 closure, 2026-04-24).
Any artifact appearing outside these canonical paths is a FAIL in the
Wave 2 audit — there is NO legacy-fallback path.

Wave 2 (plan 24-02) replaces the NotImplementedError below with the
actual check functions (see 24-02-PLAN.md Task 1 for the full list).

Notes
-----
The CheckResult dataclass + main() printer pattern mirrors
tests/scientific/check_v4_closure.py (Phase 22 closure guard).
File location: tests/scientific/ per Phase 31 CCDS consolidation
(validation/ -> tests/scientific/ via plan 31-04; plan 24-00 deviation
from original path spec of validation/check_phase24_artifacts.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- CCDS-canonical path constants (pinned by Wave 0 / plan 24-00) ------

MODELS_DIR: Path = Path("models/bayesian")
PRIOR_PREDICTIVE_DIR: Path = MODELS_DIR / "21_prior_predictive"
RECOVERY_DIR: Path = MODELS_DIR / "21_recovery"
BASELINE_DIR: Path = MODELS_DIR / "21_baseline"
L2_DIR: Path = MODELS_DIR / "21_l2"

TABLES_DIR: Path = Path("reports/tables/model_comparison")
FIGURES_DIR: Path = Path("reports/figures/bayesian/21_bayesian")

EXEC_LOG: Path = MODELS_DIR / "21_execution_log.md"
WINNERS_TXT: Path = BASELINE_DIR / "winners.txt"
CONVERGENCE_TABLE: Path = BASELINE_DIR / "convergence_table.csv"
WINNER_REPORT: Path = BASELINE_DIR / "winner_report.md"

BASELINE_MODELS: tuple[str, ...] = (
    "qlearning",
    "wmrl",
    "wmrl_m3",
    "wmrl_m5",
    "wmrl_m6a",
    "wmrl_m6b",
)

# Reporting labels for winner-determination audit (CONTEXT.md Area 3 §Convergence).
WINNER_TYPES_REPORTABLE: frozenset[str] = frozenset(
    {"DOMINANT_SINGLE", "TOP_TWO", "TOP_K", "FORCED", "INCONCLUSIVE_MULTIPLE"}
)


@dataclass(frozen=True)
class CheckResult:
    """Single audit-check outcome (mirrors tests/scientific/check_v4_closure.py)."""

    name: str
    ok: bool
    detail: str


def main() -> int:
    """Run all Phase 24 audit checks; implemented in Wave 2 (plan 24-02)."""
    raise NotImplementedError(
        "tests/scientific/check_phase24_artifacts.py is a stub pinned by Wave 0 "
        "(plan 24-00). The full audit implementation lands in Wave 2 "
        "(plan 24-02) after the cold-start pipeline terminates. "
        "Do NOT invoke before Wave 2 is shipped."
    )


if __name__ == "__main__":
    import sys

    sys.exit(main())
