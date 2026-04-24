"""CLEAN-01 enforcement guard: no live imports of scripts.fitting.legacy.

This module is the pytest enforcement mechanism behind CLEAN-01 of the v5.0
tech-debt sweep.  Without this guard, a future refactor could silently
reintroduce a ``from scripts.fitting.legacy.numpyro_models import ...`` line
and the error would only surface at runtime on the cluster.

The guard checks two independent invariants:

1. **No live import lines**: every ``.py`` file under ``scripts/`` (excluding
   ``__pycache__/`` and the legacy directory itself, which is about to be
   deleted) must not contain ``from scripts.fitting.legacy`` or
   ``import scripts.fitting.legacy``.

2. **Directory non-existence**: ``scripts/fitting/legacy/`` must not exist on
   disk after the CLEAN-01 deletion commit lands.

Paired commit: ``chore(tech-debt): delete scripts/fitting/legacy/ directory``.
"""

from __future__ import annotations

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Repo root is 2 levels above this file:
#   tests/integration/test_no_legacy_imports.py  (Phase 31 consolidated layout)
#   ^2    ^1
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
SCRIPTS_DIR: Path = REPO_ROOT / "scripts"
LEGACY_DIR: Path = SCRIPTS_DIR / "fitting" / "legacy"

# Patterns to detect (uncommented import lines are caught by a simple
# substring match; the regex variant is available for more precise checking).
_LEGACY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"from\s+scripts\.fitting\.legacy"),
    re.compile(r"import\s+scripts\.fitting\.legacy"),
]

# Directories to skip entirely during the walk.
_SKIP_DIR_NAMES: frozenset[str] = frozenset({"__pycache__", "legacy"})

# This file itself is exempt — its docstrings deliberately contain the
# forbidden pattern strings as documentation.  Exclude by resolved path so
# the skip is unconditional regardless of invocation directory.
_SELF_PATH: Path = Path(__file__).resolve()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _collect_py_files(root: Path) -> list[Path]:
    """Return all ``.py`` files under *root*, skipping excluded dirs.

    Parameters
    ----------
    root:
        Directory to walk.

    Returns
    -------
    list[Path]
        Sorted list of ``Path`` objects for every ``.py`` file found.
    """
    py_files: list[Path] = []
    for path in root.rglob("*.py"):
        # Skip if any component of the path is a blacklisted directory name.
        if any(part in _SKIP_DIR_NAMES for part in path.parts):
            continue
        py_files.append(path)
    return sorted(py_files)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_no_legacy_imports() -> None:
    """Assert no live scripts/ file imports from scripts.fitting.legacy.

    Also asserts that the ``scripts/fitting/legacy/`` directory does not exist
    on disk (CLEAN-01 deletion postcondition).

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If any ``.py`` file under ``scripts/`` (outside ``legacy/`` and
        ``__pycache__/``) contains a ``scripts.fitting.legacy`` import, or if
        the ``scripts/fitting/legacy/`` directory still exists on disk.
    """
    # ------------------------------------------------------------------
    # Invariant 1: directory must not exist
    # ------------------------------------------------------------------
    legacy_exists = LEGACY_DIR.exists()
    assert not legacy_exists, (
        f"Expected: scripts/fitting/legacy/ does NOT exist on disk.\n"
        f"Actual:   {LEGACY_DIR} EXISTS — "
        "CLEAN-01 deletion has not been applied yet (or the directory was "
        "recreated)."
    )

    # ------------------------------------------------------------------
    # Invariant 2: no live import lines
    # ------------------------------------------------------------------
    py_files = _collect_py_files(SCRIPTS_DIR)
    violations: list[tuple[Path, int, str]] = []

    for py_file in py_files:
        # Skip the guard file itself — its docstrings legitimately contain
        # the pattern strings as documentation text.
        if py_file.resolve() == _SELF_PATH:
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
        except OSError:
            # Unreadable file — skip (not a false negative for our purpose).
            continue

        for lineno, line in enumerate(source.splitlines(), start=1):
            for pattern in _LEGACY_PATTERNS:
                if pattern.search(line):
                    violations.append((py_file, lineno, line.strip()))

    if violations:
        lines = "\n".join(
            f"  {path.relative_to(REPO_ROOT)}:{lineno}  {text}"
            for path, lineno, text in violations
        )
        raise AssertionError(
            f"Expected: zero live imports of scripts.fitting.legacy in scripts/.\n"
            f"Actual:   {len(violations)} violation(s) found:\n{lines}\n"
            "Remove or rewrite these lines before merging."
        )
