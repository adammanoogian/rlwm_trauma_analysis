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


# =========================================================================
# Phase 31 — Final-Package Restructure (CCDS layout + test/log consolidation)
# =========================================================================
#
# Phase 31 asserts a CCDS-aligned top-level layout for final-package readiness.
# References:
#   - Phase 31 RESEARCH.md (.planning/phases/31-final-package-restructure/)
#   - CCDS docs: https://cookiecutter-data-science.drivendata.org/
#   - Success criteria in ROADMAP.md Phase 31 section

# ------ Phase 31 SC#1: data/ CCDS tier structure ------

PHASE31_DATA_TIERS = ["raw", "interim", "processed", "external"]


@pytest.mark.parametrize("tier", PHASE31_DATA_TIERS)
def test_phase31_data_tier_exists(tier: str) -> None:
    """Phase 31 SC#1: data/<tier>/ exists (CCDS data tiers)."""
    path = REPO_ROOT / "data" / tier
    assert path.is_dir(), (
        f"Missing Phase 31 data tier: expected {path}, got absent"
    )


# ------ Phase 31 SC#2: models/ subdirectories ------

PHASE31_MODELS_SUBDIRS = [
    "bayesian",
    "mle",
    "ppc",
    "recovery",
]


@pytest.mark.parametrize("sub", PHASE31_MODELS_SUBDIRS)
def test_phase31_models_subdir_exists(sub: str) -> None:
    """Phase 31 SC#2: models/<sub>/ exists (CCDS models tier)."""
    path = REPO_ROOT / "models" / sub
    assert path.is_dir(), (
        f"Missing Phase 31 models subdirectory: expected {path}, got absent"
    )


# ------ Phase 31 SC#3: reports/ structure ------

PHASE31_REPORTS_SUBDIRS = ["figures", "tables"]


@pytest.mark.parametrize("sub", PHASE31_REPORTS_SUBDIRS)
def test_phase31_reports_subdir_exists(sub: str) -> None:
    """Phase 31 SC#3: reports/<sub>/ exists (CCDS reports tier)."""
    path = REPO_ROOT / "reports" / sub
    assert path.is_dir(), (
        f"Missing Phase 31 reports subdirectory: expected {path}, got absent"
    )


# ------ Phase 31 SC#4: legacy output/, figures/, validation/ removed ------

PHASE31_REMOVED_DIRS = ["output", "figures", "validation"]


@pytest.mark.parametrize("removed", PHASE31_REMOVED_DIRS)
def test_phase31_legacy_dir_removed(removed: str) -> None:
    """Phase 31 SC#4: legacy output/, figures/, validation/ gone from top level."""
    path = REPO_ROOT / removed
    assert not path.exists(), (
        f"Legacy top-level directory still present: expected absent, got {path}. "
        f"Phase 31 consolidated these into CCDS tiers — content belongs under "
        f"data/, models/, reports/, or tests/scientific/."
    )


# ------ Phase 31 SC#5: tests/ tier structure ------

PHASE31_TESTS_TIERS = ["unit", "integration", "scientific"]


@pytest.mark.parametrize("tier", PHASE31_TESTS_TIERS)
def test_phase31_tests_tier_exists(tier: str) -> None:
    """Phase 31 SC#5: tests/<tier>/ exists (unit/integration/scientific)."""
    path = REPO_ROOT / "tests" / tier
    assert path.is_dir(), (
        f"Missing Phase 31 test tier: expected {path}, got absent"
    )


def test_phase31_scripts_fitting_tests_removed() -> None:
    """Phase 31 SC#5: scripts/fitting/tests/ no longer exists at that location."""
    path = REPO_ROOT / "scripts" / "fitting" / "tests"
    assert not path.exists(), (
        f"scripts/fitting/tests/ still exists: expected absent, got {path}. "
        f"Phase 31 consolidated into tests/integration/."
    )


# ------ Phase 31 SC#6: logs/ at root, cluster/logs/ gone ------

def test_phase31_logs_at_root_only() -> None:
    """Phase 31 SC#6: logs/ exists at root; cluster/logs/ gone (unified)."""
    logs_root = REPO_ROOT / "logs"
    cluster_logs = REPO_ROOT / "cluster" / "logs"
    assert logs_root.is_dir(), (
        f"Missing top-level logs/: expected {logs_root}, got absent"
    )
    assert not cluster_logs.exists(), (
        f"cluster/logs/ still exists: expected absent, got {cluster_logs}. "
        f"Phase 31 merged into {logs_root}."
    )


# ------ Phase 31 SC#7: config.py no longer exposes legacy aliases ------

def test_phase31_config_has_ccds_constants() -> None:
    """Phase 31 SC#7: config.py exposes DATA_RAW_DIR, INTERIM_DIR, etc."""
    import importlib
    import sys as _sys
    if "config" in _sys.modules:
        importlib.reload(_sys.modules["config"])
    import config as _config
    required = [
        "DATA_RAW_DIR", "INTERIM_DIR", "PROCESSED_DIR", "DATA_EXTERNAL_DIR",
        "MODELS_DIR", "MODELS_BAYESIAN_DIR", "MODELS_MLE_DIR",
        "MODELS_PARAMETER_EXPLORATION_DIR",
        "REPORTS_DIR", "REPORTS_FIGURES_DIR", "REPORTS_TABLES_DIR",
        "LOGS_DIR",
    ]
    missing = [name for name in required if not hasattr(_config, name)]
    assert not missing, (
        f"config.py missing Phase 31 CCDS constants: expected all of "
        f"{required}, got missing {missing}"
    )


def test_phase31_config_no_legacy_aliases() -> None:
    """Phase 31 SC#7: config.py removed OUTPUT_DIR/FIGURES_DIR legacy aliases."""
    import importlib
    import sys as _sys
    if "config" in _sys.modules:
        importlib.reload(_sys.modules["config"])
    import config as _config
    forbidden = ["OUTPUT_DIR", "FIGURES_DIR", "OUTPUT_VERSION_DIR", "FIGURES_VERSION_DIR"]
    present = [name for name in forbidden if hasattr(_config, name)]
    assert not present, (
        f"config.py still exposes Phase 31 legacy aliases: expected all gone, "
        f"got {present}. Waves B/C should have migrated consumers; plan 31-05 "
        f"removes these constants."
    )


# ------ Phase 31 SC#8: no live legacy path strings ------

PHASE31_LEGACY_PATH_PATTERNS = [
    "output/bayesian",
    "output/mle",
    "output/descriptives",
    "output/model_comparison",
    "output/regressions",
]


@pytest.mark.parametrize("pattern", PHASE31_LEGACY_PATH_PATTERNS)
def test_phase31_no_legacy_output_paths(pattern: str) -> None:
    """Phase 31 SC#8: zero live references to legacy output/* paths.

    Excludes /legacy/ archives and .planning/ (which may reference historic
    paths as documentation). Only active code tree is checked.
    """
    hits: list[str] = []
    search_dirs = ["scripts", "tests", "src", "cluster", "manuscript"]
    self_rel = str(Path(__file__).resolve().relative_to(REPO_ROOT)).replace(
        "\\", "/"
    )
    for d in search_dirs:
        root = REPO_ROOT / d
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            # Only check text files we care about
            if p.suffix not in {".py", ".qmd", ".slurm", ".sh", ".md", ".toml", ".ini"}:
                continue
            rel = str(p.relative_to(REPO_ROOT)).replace("\\", "/")
            if rel == self_rel:
                continue
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
        f"Legacy path pattern `{pattern}` found in: expected zero, got {hits}. "
        f"Phase 31 Waves B/C/E should have migrated these."
    )
