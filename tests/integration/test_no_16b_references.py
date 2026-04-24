"""CLEAN-03 enforcement: no live source reference to 16b_bayesian_regression in scripts/,
cluster/, or docs/03_methods_reference/; and no `16b*` file anywhere outside `.planning/`
(archival tree deliberately excluded per v4.0 closure protocol). The source file was deleted
in a prior cleanup; this guard prevents silent reintroduction via `git revert` or copy-paste.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]  # tests/integration/<file>.py -> repo root

# Directories to walk for text-reference checks
_TEXT_SEARCH_ROOTS = [
    _REPO_ROOT / "scripts",
    _REPO_ROOT / "cluster",
    _REPO_ROOT / "docs" / "03_methods_reference",
]

# Directory name components to skip when iterating source files
_SKIP_DIR_NAMES = {".planning", ".git", "__pycache__", "fixtures"}

# File extensions to inspect for text references
_TEXT_EXTENSIONS = {".py", ".slurm", ".sh", ".md"}

# Patterns that must not appear in live source
_FORBIDDEN_PATTERNS = ["16b_bayesian_regression", "scripts/16b"]

# Trees to exclude entirely from the file-existence walk
_EXCLUDED_ROOTS = [
    _REPO_ROOT / ".planning",
    _REPO_ROOT / ".git",
]


# This guard file is itself excluded from the text-reference scan to avoid
# a trivial self-reference (the file necessarily names the forbidden patterns
# as string literals).
_THIS_FILE = Path(__file__).resolve()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_under_excluded_root(path: Path) -> bool:
    """Return True if *path* is inside any directory in ``_EXCLUDED_ROOTS``.

    Parameters
    ----------
    path : Path
        Absolute file path to test.

    Returns
    -------
    bool
        ``True`` if *path* is a descendant of an excluded root.
    """
    for excl in _EXCLUDED_ROOTS:
        try:
            path.relative_to(excl)
            return True  # path is inside this excluded root
        except ValueError:
            continue  # not under this root; try next
    return False


def _iter_source_files(root: Path) -> list[Path]:
    """Return all source files under *root* with a checked extension.

    Parameters
    ----------
    root : Path
        Directory tree to walk.

    Returns
    -------
    list[Path]
        Sorted list of files whose suffix is in ``_TEXT_EXTENSIONS``,
        excluding any path whose component names overlap with
        ``_SKIP_DIR_NAMES``.
    """
    found: list[Path] = []
    if not root.exists():
        return found
    for path in root.rglob("*"):
        # Skip if any component is a forbidden directory name
        if any(part in _SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.is_file() and path.suffix in _TEXT_EXTENSIONS:
            found.append(path)
    return sorted(found)


# ---------------------------------------------------------------------------
# Guard tests
# ---------------------------------------------------------------------------


def test_no_16b_text_references_in_live_source() -> None:
    """No live source file may reference 16b_bayesian_regression or scripts/16b.

    Walks scripts/, cluster/, and docs/03_methods_reference/ (the three live
    source trees). Skips .planning/ (archival — intentionally preserved per
    v4.0 closure protocol), .git/, and __pycache__/.

    On failure, reports every offending file + line number + offending text
    so the developer knows exactly where to look.
    """
    violations: list[str] = []

    for root in _TEXT_SEARCH_ROOTS:
        for filepath in _iter_source_files(root):
            # Skip this guard file itself (it necessarily names the forbidden patterns
            # as string literals to check against them)
            if filepath == _THIS_FILE:
                continue
            try:
                content = filepath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for lineno, line in enumerate(content.splitlines(), start=1):
                for pattern in _FORBIDDEN_PATTERNS:
                    if pattern in line:
                        violations.append(
                            f"{filepath}:{lineno}: found '{pattern}' → {line.strip()!r}"
                        )

    assert not violations, (
        f"Found {len(violations)} live reference(s) to 16b_bayesian_regression "
        f"in source files:\n" + "\n".join(violations)
    )


def test_no_16b_files_outside_planning_tree() -> None:
    """No file named `16b*` may exist outside .planning/ and .git/.

    This is the file-existence analog of:

        find . -path ./.planning -prune -o -path ./.git -prune -o -name '16b*' -print

    returning empty output. Enforces SC#3 from ROADMAP.md Phase 23 with the
    archival-policy .planning/ exclusion encoded deterministically. Pycache
    files (.pyc) under __pycache__/ also count as violations — they indicate
    a stale bytecode cache for a deleted source file.
    """
    violations: list[str] = []

    for path in _REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if _is_under_excluded_root(path):
            continue
        if path.name.startswith("16b"):
            violations.append(str(path))

    assert not violations, (
        f"Found {len(violations)} file(s) named '16b*' outside .planning/ "
        f"(expected zero after CLEAN-03 cleanup):\n"
        + "\n".join(violations)
        + "\n\nExpected: no files. "
        "Actual: files listed above. "
        "Delete stale pycache entries or source files to resolve."
    )
