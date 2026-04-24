"""Phase 29 closure-state checker: canonical 01-06 scripts layout.

Every invariant below is deterministic — runs against the filesystem state and
committed hash manifests. Any future phase that regresses these structures
causes this test to fail, giving immediate feedback.

Referenced success criteria from ``.planning/phases/29-pipeline-canonical-reorg/
29-CONTEXT.md``:

- SC#1  — canonical 6 stage dirs at scripts/ top level
- SC#2  — 04_model_fitting/{a_mle,b_bayesian,c_level2} sub-letters present
- SC#3  — dead folders gone from top level (under scripts/legacy/ if retained)
- SC#4  — simulator single-source in scripts/utils/ppc.py
- SC#5  — docs spare files merged; originals live under docs/legacy/
- SC#6  — docs/CLUSTER_GPU_LESSONS.md byte-identical to committed snapshot
- SC#10 — zero ``from scripts.<old_grouping>`` imports in active tree
- SC#12 — utils/ canonical short names present; verbose names absent (29-03)

NOT expressible as pytest assertions (documented here for traceability):

- SC#7  — cluster ``sbatch --dry-run`` / ``bash -n`` (covered by 29-05 plan)
- SC#8  — ``quarto render manuscript/paper.qmd`` (covered by 29-06 plan)
- SC#9  — v4 closure guard (covered by ``tests/integration/test_v4_closure.py``
          and ``tests/scientific/check_v4_closure.py``)
- SC#11 — full pytest suite green (covered by ``pytest`` top-level)
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # tests/integration/<file>.py -> repo root
SCRIPTS = REPO_ROOT / "scripts"
DOCS = REPO_ROOT / "docs"


# -------------------- SC#1: canonical stage folders exist --------------------

STAGE_FOLDERS = [
    "01_data_preprocessing",
    "02_behav_analyses",
    "03_model_prefitting",
    "04_model_fitting",
    "05_post_fitting_checks",
    "06_fit_analyses",
]


@pytest.mark.parametrize("stage", STAGE_FOLDERS)
def test_stage_folder_exists(stage: str) -> None:
    """SC#1: scripts/NN_*/ exists for NN in 01..06."""
    path = SCRIPTS / stage
    assert path.is_dir(), (
        f"Missing canonical stage folder: expected {path}, got absent"
    )


# -------------------- SC#2: 04_model_fitting sub-letters -------------------

def test_04_model_fitting_subletters_exist() -> None:
    """SC#2: 04_model_fitting has a_mle/, b_bayesian/, c_level2/ sub-letters."""
    for sub in ("a_mle", "b_bayesian", "c_level2"):
        path = SCRIPTS / "04_model_fitting" / sub
        assert path.is_dir(), (
            f"Missing 04_model_fitting/{sub}: expected {path}, got absent"
        )


# -------------------- SC#3: dead folders absent from top level --------------------

DEAD_FOLDERS = [
    "analysis",
    "results",
    "simulations",
    "statistical_analyses",
    "visualization",
    "data_processing",
    "behavioral",
    "simulations_recovery",
    "post_mle",
    "bayesian_pipeline",
]


@pytest.mark.parametrize("folder", DEAD_FOLDERS)
def test_dead_folder_absent_from_top_level(folder: str) -> None:
    """SC#3, SC#1: scripts/<folder>/ must not exist at top level.

    Phase 28 grouping folders and pre-Phase-28 sibling folders both belong under
    ``scripts/legacy/<folder>/`` if any content is retained — never at the
    ``scripts/`` top level.
    """
    path = SCRIPTS / folder
    assert not path.exists(), (
        f"Dead folder re-appeared at top level: expected absent, got {path}. "
        f"If this content is needed, it belongs under scripts/legacy/."
    )


# -------------------- SC#4: canonical simulator single-source --------------------

def test_utils_ppc_exists_and_nontrivial() -> None:
    """SC#4: scripts/utils/ppc.py is the canonical simulator home."""
    path = SCRIPTS / "utils" / "ppc.py"
    assert path.is_file(), (
        f"Missing canonical simulator: expected {path}, got absent"
    )
    content = path.read_text(encoding="utf-8")
    function_count = content.count("def ")
    assert function_count >= 2, (
        f"scripts/utils/ppc.py is too thin: expected >=2 function definitions, "
        f"got {function_count}"
    )


def test_simulator_not_duplicated_outside_utils() -> None:
    """SC#4: simulator functions defined ONLY in scripts/utils/ppc.py.

    ``scripts/legacy/`` archived content is ignored — only active-tree duplicates
    are a regression signal.
    """
    simulator_fns = ("run_prior_ppc", "run_posterior_ppc", "simulate_from_samples")
    try:
        result = subprocess.run(
            [
                "rg",
                "-l",
                r"^def (run_prior_ppc|run_posterior_ppc|simulate_from_samples)\b",
                "scripts/",
                "--glob",
                "*.py",
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
        )
        files = [
            line.strip() for line in result.stdout.splitlines() if line.strip()
        ]
    except FileNotFoundError:
        # Fallback: Python-native grep
        files = []
        for p in SCRIPTS.rglob("*.py"):
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if any(f"def {fn}(" in txt for fn in simulator_fns):
                files.append(str(p.relative_to(REPO_ROOT)).replace("\\", "/"))
    expected = "scripts/utils/ppc.py"
    # Filter out scripts/legacy/ (archival content is allowed duplicates)
    extras = [
        f
        for f in files
        if f != expected and not f.startswith("scripts/legacy/")
    ]
    assert not extras, (
        f"Simulator functions defined outside utils/: expected only {expected}, "
        f"got duplicates in {extras}"
    )


# -------------------- SC#5: docs spare files merged --------------------

DOCS_SPARE_FILES = [
    "HIERARCHICAL_BAYESIAN.md",
    "K_PARAMETERIZATION.md",
    "SCALES_AND_FITTING_AUDIT.md",
]


@pytest.mark.parametrize("name", DOCS_SPARE_FILES)
def test_docs_spare_files_moved_to_legacy(name: str) -> None:
    """SC#5: docs/<name> no longer at top level; lives at docs/legacy/<name>."""
    top = DOCS / name
    legacy = DOCS / "legacy" / name
    assert not top.exists(), (
        f"Orphan doc still at top level: expected absent, got {top}"
    )
    assert legacy.is_file(), (
        f"Missing legacy archive: expected {legacy}, got absent"
    )


# -------------------- SC#6: CLUSTER_GPU_LESSONS.md byte-identical --------------------

def test_cluster_gpu_lessons_untouched() -> None:
    """SC#6: docs/CLUSTER_GPU_LESSONS.md sha256 matches pre-phase snapshot.

    The manifest lives at the repo root as ``pre_phase29_cluster_gpu_lessons.sha256``
    and captures the expected sha256 digest. Any Phase-29 or later phase
    that modifies this file will flip the hash and fail this test.
    """
    manifest = REPO_ROOT / "pre_phase29_cluster_gpu_lessons.sha256"
    target = DOCS / "CLUSTER_GPU_LESSONS.md"
    assert manifest.is_file(), (
        f"Missing hash manifest: expected {manifest} (29-07 should have committed it)"
    )
    assert target.is_file(), f"Missing target: expected {target}"
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    expected = manifest.read_text(encoding="utf-8").strip().split()[0]
    assert actual == expected, (
        f"CLUSTER_GPU_LESSONS.md was modified after the Phase 29 snapshot: "
        f"expected sha256={expected}, got sha256={actual}"
    )


# -------------------- SC#10: zero importers of old grouping paths --------------------

OLD_IMPORT_PATTERNS = [
    "from scripts.data_processing",
    "from scripts.behavioral",
    "from scripts.simulations_recovery",
    "from scripts.post_mle",
    "from scripts.bayesian_pipeline",
]


@pytest.mark.parametrize("pattern", OLD_IMPORT_PATTERNS)
def test_no_old_grouping_imports(pattern: str) -> None:
    """SC#10: zero live ``from scripts.<old_grouping>`` imports outside .planning/."""
    hits: list[str] = []
    search_dirs = ["scripts", "tests", "src"]
    self_rel = str(Path(__file__).resolve().relative_to(REPO_ROOT)).replace(
        "\\", "/"
    )
    for d in search_dirs:
        root = REPO_ROOT / d
        if not root.is_dir():
            continue
        for p in root.rglob("*.py"):
            rel = str(p.relative_to(REPO_ROOT)).replace("\\", "/")
            # Skip self (this test file contains the patterns as string literals)
            if rel == self_rel:
                continue
            # Skip legacy archives (historical imports may remain there)
            if "/legacy/" in rel or rel.startswith(
                ("scripts/legacy/", "tests/legacy/")
            ):
                continue
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if pattern in txt:
                hits.append(rel)
    assert not hits, (
        f"Stale import pattern `{pattern}` found in: expected zero, got {hits}"
    )


# -------------------- SC#12 bonus: utils canonical short names (29-03) --------------------

UTILS_RENAME_PAIRS = [
    ("plotting", "plotting_utils"),
    ("stats", "statistical_tests"),
    ("scoring", "scoring_functions"),
]


@pytest.mark.parametrize("short,long_", UTILS_RENAME_PAIRS)
def test_utils_canonical_short_names(short: str, long_: str) -> None:
    """29-03 invariant: canonical short names exist; verbose names do not."""
    assert (SCRIPTS / "utils" / f"{short}.py").is_file(), (
        f"Missing canonical utils file: expected scripts/utils/{short}.py, "
        f"got absent"
    )
    assert not (SCRIPTS / "utils" / f"{long_}.py").exists(), (
        f"Verbose name still present: expected absent "
        f"scripts/utils/{long_}.py, got file"
    )
