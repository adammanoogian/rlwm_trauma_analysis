"""Phase 22 closure-state regression test.

Invokes ``validation.check_v4_closure.check_all`` directly (no subprocess)
and asserts exit code 0. Runs on every ``pytest`` invocation so CI catches
closure drift the moment a plan-status doc becomes inconsistent with on-disk
phase artifacts.

The test does NOT exercise the CLI entry point — that is verified by
``validation/check_v4_closure.py --help`` contract tests at release time.
What this test cares about is: ``check_all()`` returns (0, all-passed-list)
on the current commit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add repo root to sys.path so ``from validation.check_v4_closure import ...``
# works regardless of pytest invocation directory.  Mirrors the conftest pattern
# already in use for scripts/fitting/tests/.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.check_v4_closure import CheckResult, check_all  # noqa: E402


def test_v4_closure_passes() -> None:
    """Milestone v4.0 closure invariants all hold on current HEAD.

    If this test fails, read the ``details`` list of the failing CheckResult
    for the exact invariant that broke.  Re-running
    ``python validation/check_v4_closure.py --milestone v4.0 --verbose``
    from the command line produces the same diagnostic.
    """
    exit_code, results = check_all()
    failed = [r for r in results if not r.passed]
    assert exit_code == 0, (
        f"Phase 22 closure check returned exit {exit_code}. "
        f"Expected 0. {len(failed)} of {len(results)} invariants failed:\n"
        + "\n".join(f"  - {r.name}: {r.message}" for r in failed)
    )


def test_v4_closure_deterministic() -> None:
    """Two successive invocations produce byte-identical result objects.

    Regression guard for the SC#7 determinism requirement: if this test
    starts flaking, the checker has picked up a nondeterministic source
    (datetime, random, unsorted glob) and must be fixed before the next
    audit-milestone run.
    """
    exit1, r1 = check_all()
    exit2, r2 = check_all()
    assert exit1 == exit2, f"exit codes drifted: {exit1} vs {exit2}"
    names1 = [r.name for r in r1]
    names2 = [r.name for r in r2]
    assert names1 == names2, f"check order drifted: {names1} vs {names2}"
    messages1 = [r.message for r in r1]
    messages2 = [r.message for r in r2]
    assert messages1 == messages2, "check messages drifted between invocations"


def test_v4_closure_rejects_wrong_milestone() -> None:
    """``main(['--milestone', 'v3.0'])`` raises or exits non-zero.

    Guards against copy-paste drift when this script is reused for a future
    milestone without being updated for that milestone's invariants.
    """
    from validation.check_v4_closure import main

    with pytest.raises((ValueError, SystemExit)):
        main(["--milestone", "v3.0"])
