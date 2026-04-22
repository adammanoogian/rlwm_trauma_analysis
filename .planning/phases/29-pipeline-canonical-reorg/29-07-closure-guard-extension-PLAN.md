---
phase: 29-pipeline-canonical-reorg
plan: 07
type: execute
wave: 5
depends_on: [29-01, 29-02, 29-03, 29-04, 29-04b, 29-05, 29-06]
files_modified:
  - tests/test_v5_phase29_structure.py              (new — pytest closure guard for canonical shape)
  - validation/check_v4_closure.py                  (updated if new path invariants supersede v4 paths)
  - .planning/REQUIREMENTS.md                       (append REFAC-14..REFAC-20 rows + requirement-to-phase ledger rows)
  - .planning/phases/29-pipeline-canonical-reorg/29-VERIFICATION.md  (new — end-of-phase verification report)
autonomous: true

must_haves:
  truths:
    - "pytest tests/test_v5_phase29_structure.py passes — asserts the 6 stage folders exist, dead folders are gone from top level, utils/ canonical names exist, docs spare files are merged, CLUSTER_GPU_LESSONS.md hash unchanged"  # SC#12
    - "validation/check_v4_closure.py --milestone v4.0 still exits 0 (no v4 invariants regressed)"  # SC#9
    - "REQUIREMENTS.md contains REFAC-14..REFAC-20 rows and the requirement-to-phase ledger entries"
    - "SC#1, SC#2, SC#3, SC#4, SC#5, SC#6, SC#9, SC#10, SC#12 and utils canonical-name invariants are verifiable from this pytest file. SC#7 (cluster SLURM dry-run) and SC#8 (quarto render) are verified by Plan 29-05 and Plan 29-06 verify steps respectively, which run external processes (sbatch --dry-run, quarto render) not expressible as pytest assertions. SC#11 (full-suite pytest green) is verified by running the full pytest suite as a separate step within this plan."
  artifacts:
    - path: "tests/test_v5_phase29_structure.py"
      provides: "pytest closure guard for Phase 29 canonical structure"
      min_lines: 60
      contains: "def test_"
    - path: ".planning/REQUIREMENTS.md"
      provides: "REFAC-14..REFAC-20 requirement rows appended + ledger"
      contains: "REFAC-14"
    - path: ".planning/phases/29-pipeline-canonical-reorg/29-VERIFICATION.md"
      provides: "end-of-phase verification report (status pass/gaps_found + evidence per SC)"
  key_links:
    - from: "tests/test_v5_phase29_structure.py"
      to: "scripts/0{1..6}_*"
      via: "Path().is_dir() asserts"
      pattern: "scripts/0[1-6]_"
    - from: "tests/test_v5_phase29_structure.py"
      to: "pre_phase29_cluster_gpu_lessons.sha256"
      via: "hash comparison"
      pattern: "pre_phase29_cluster_gpu_lessons"
---

<objective>
Lock in the Phase 29 canonical structure with a pytest invariant that Phase 30+ will run automatically. Write the 29-VERIFICATION.md end-of-phase report. Append REFAC-14..REFAC-20 rows to `.planning/REQUIREMENTS.md`. Confirm v4 closure guard remains green.

Purpose: Make the reorganization permanent — if a future phase accidentally re-introduces a `scripts/bayesian_pipeline/` folder or deletes `scripts/utils/ppc.py`, the closure test will catch it.

Output: tests/test_v5_phase29_structure.py + REQUIREMENTS.md updates + 29-VERIFICATION.md.
</objective>

<execution_context>
@C:\Users\aman0087\.claude\get-shit-done\workflows\execute-plan.md
@C:\Users\aman0087\.claude\get-shit-done\templates\summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/REQUIREMENTS.md
@.planning/phases/29-pipeline-canonical-reorg/29-CONTEXT.md
@.planning/phases/29-pipeline-canonical-reorg/29-01-SUMMARY.md
@.planning/phases/29-pipeline-canonical-reorg/29-02-SUMMARY.md
@.planning/phases/29-pipeline-canonical-reorg/29-03-SUMMARY.md
@.planning/phases/29-pipeline-canonical-reorg/29-04-SUMMARY.md
@.planning/phases/29-pipeline-canonical-reorg/29-05-SUMMARY.md
@.planning/phases/29-pipeline-canonical-reorg/29-06-SUMMARY.md
@scripts/fitting/tests/test_v4_closure.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Write tests/test_v5_phase29_structure.py closure guard</name>
  <files>
    - tests/test_v5_phase29_structure.py (new)
  </files>
  <action>
    1. Create `tests/test_v5_phase29_structure.py` with the following invariants (one pytest function per invariant for clear failure isolation):
    
    ```python
    """Phase 29 closure-state checker: canonical 01–06 scripts layout.
    
    Every invariant below is deterministic — runs against the filesystem state and
    committed hash manifests. Any future phase that regresses these structures
    causes this test to fail, giving immediate feedback.
    
    Referenced success criteria: SC#1, SC#3, SC#4, SC#5, SC#6, SC#9, SC#10, SC#12
    from .planning/phases/29-pipeline-canonical-reorg/29-CONTEXT.md.
    """
    
    from __future__ import annotations
    
    import hashlib
    import subprocess
    from pathlib import Path
    
    import pytest
    
    REPO_ROOT = Path(__file__).resolve().parents[1]
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
        assert path.is_dir(), f"Missing canonical stage folder: expected {path}, got absent"
    
    
    def test_04_model_fitting_subletters_exist() -> None:
        """SC#2: 04_model_fitting has a_mle/, b_bayesian/, c_level2/ sub-letters."""
        for sub in ("a_mle", "b_bayesian", "c_level2"):
            path = SCRIPTS / "04_model_fitting" / sub
            assert path.is_dir(), f"Missing 04_model_fitting/{sub}: expected {path}, got absent"
    
    
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
        """SC#3, SC#1: scripts/<folder>/ must not exist at top level (may exist under scripts/legacy/)."""
        path = SCRIPTS / folder
        assert not path.exists(), (
            f"Dead folder re-appeared at top level: expected absent, got {path}. "
            f"If this content is needed, it belongs under scripts/legacy/."
        )
    
    
    # -------------------- SC#4: canonical simulator single-source --------------------
    
    def test_utils_ppc_exists_and_nontrivial() -> None:
        """SC#4: scripts/utils/ppc.py is the canonical simulator home."""
        path = SCRIPTS / "utils" / "ppc.py"
        assert path.is_file(), f"Missing canonical simulator: expected {path}, got absent"
        content = path.read_text(encoding="utf-8")
        function_count = content.count("def ")
        assert function_count >= 2, (
            f"scripts/utils/ppc.py is too thin: expected >=2 function definitions, got {function_count}"
        )
    
    
    def test_simulator_not_duplicated_outside_utils() -> None:
        """SC#4: simulator functions defined ONLY in scripts/utils/ppc.py (no duplicates)."""
        # Use ripgrep if available, else Python grep
        try:
            result = subprocess.run(
                ["rg", "-l", r"^def (run_prior_ppc|run_posterior_ppc|simulate_from_samples)\b",
                 "scripts/", "--glob", "*.py"],
                capture_output=True, text=True, cwd=REPO_ROOT, check=False,
            )
            files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        except FileNotFoundError:
            # Fallback: Python-native grep
            files = []
            for p in SCRIPTS.rglob("*.py"):
                txt = p.read_text(encoding="utf-8", errors="ignore")
                if any(f"def {fn}(" in txt for fn in ("run_prior_ppc", "run_posterior_ppc", "simulate_from_samples")):
                    files.append(str(p.relative_to(REPO_ROOT)).replace("\\", "/"))
        expected = "scripts/utils/ppc.py"
        extras = [f for f in files if f != expected]
        assert not extras, (
            f"Simulator functions defined outside utils/: expected only {expected}, got duplicates in {extras}"
        )
    
    
    # -------------------- SC#5: docs spare files merged --------------------
    
    @pytest.mark.parametrize("name", ["HIERARCHICAL_BAYESIAN.md", "K_PARAMETERIZATION.md", "SCALES_AND_FITTING_AUDIT.md"])
    def test_docs_spare_files_moved_to_legacy(name: str) -> None:
        """SC#5: docs/<name> no longer at top level; lives at docs/legacy/<name>."""
        top = DOCS / name
        legacy = DOCS / "legacy" / name
        assert not top.exists(), f"Orphan doc still at top level: expected absent, got {top}"
        assert legacy.is_file(), f"Missing legacy archive: expected {legacy}, got absent"
    
    
    # -------------------- SC#6: CLUSTER_GPU_LESSONS.md byte-identical (hash invariant) --------------------
    
    def test_cluster_gpu_lessons_untouched() -> None:
        """SC#6: docs/CLUSTER_GPU_LESSONS.md sha256 matches pre-phase snapshot."""
        manifest = REPO_ROOT / "pre_phase29_cluster_gpu_lessons.sha256"
        target = DOCS / "CLUSTER_GPU_LESSONS.md"
        assert manifest.is_file(), f"Missing hash manifest: expected {manifest} (29-02 should have committed it)"
        assert target.is_file(), f"Missing target: expected {target}"
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        expected = manifest.read_text(encoding="utf-8").strip().split()[0]
        assert actual == expected, (
            f"CLUSTER_GPU_LESSONS.md was modified during Phase 29: expected sha256={expected}, got sha256={actual}"
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
        """SC#10: zero live `from scripts.<old_grouping>` imports outside .planning/."""
        hits: list[str] = []
        search_dirs = ["scripts", "tests", "validation", "src"]
        for d in search_dirs:
            for p in (REPO_ROOT / d).rglob("*.py"):
                try:
                    txt = p.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if pattern in txt:
                    hits.append(str(p.relative_to(REPO_ROOT)).replace("\\", "/"))
        assert not hits, f"Stale import pattern `{pattern}` found in: expected zero, got {hits}"
    
    
    # -------------------- SC#12 bonus: utils canonical short names --------------------
    
    @pytest.mark.parametrize("short,long_",
                             [("plotting", "plotting_utils"),
                              ("stats", "statistical_tests"),
                              ("scoring", "scoring_functions")])
    def test_utils_canonical_short_names(short: str, long_: str) -> None:
        """29-03 invariant: canonical short names exist; verbose names do not."""
        assert (SCRIPTS / "utils" / f"{short}.py").is_file(), (
            f"Missing canonical utils file: expected scripts/utils/{short}.py, got absent"
        )
        assert not (SCRIPTS / "utils" / f"{long_}.py").exists(), (
            f"Verbose name still present: expected absent scripts/utils/{long_}.py, got file"
        )
    ```
    
    2. Run the test locally: `pytest tests/test_v5_phase29_structure.py -v` — every test must PASS.
    3. If any test fails, the corresponding plan's implementation has a gap — loop back to that plan's verification step and fix before proceeding.
    4. Re-run v4 closure guard: `python validation/check_v4_closure.py --milestone v4.0` — expect exit 0.
    5. Re-run full-repo pytest to confirm no regressions: `pytest scripts/fitting/tests/ tests/ validation/ -v` — expect no NEW failures vs. the pre-phase baseline.
  </action>
  <verify>
    - `pytest tests/test_v5_phase29_structure.py -v` PASSES 100% (all parametrize cases)
    - `python validation/check_v4_closure.py --milestone v4.0` exits 0
    - `pytest scripts/fitting/tests/test_v4_closure.py -v` PASSES 3/3
  </verify>
  <done>Closure guard green; no v4 regressions; phase 29 invariants pinned.</done>
</task>

<task type="auto">
  <name>Task 2: Append REFAC-14..REFAC-20 to REQUIREMENTS.md + write 29-VERIFICATION.md</name>
  <files>
    - .planning/REQUIREMENTS.md (appended rows)
    - .planning/phases/29-pipeline-canonical-reorg/29-VERIFICATION.md (new)
  </files>
  <action>
    1. Read `.planning/REQUIREMENTS.md` current state.
    2. Append the following requirement rows UNDER the existing CLEAN/EXEC/REPRO/MANU/CLOSE/REFAC sections (the REFAC block currently ends at REFAC-13). Use the existing row format — bullet with `**REFAC-NN**: <description>`:
    
       ```
       - [ ] **REFAC-14**: Scripts canonical reorganization — move grouped Phase-28 folders (data_processing/, behavioral/, simulations_recovery/, post_mle/, bayesian_pipeline/) and top-level entry scripts (12_fit_mle.py, 13_fit_bayesian.py, 14_compare_models.py) into canonical paper-directional 01–06 stage folders via `git mv` (history preserved); update all importers across scripts/, tests/, validation/, cluster/, manuscript/, docs/, src/ in one atomic commit (or stage-boundary atomic commits); `scripts/04_model_fitting/{a_mle,b_bayesian,c_level2}/` sub-letters capture parallel-alternative fitting paths (MLE vs Bayesian vs Level-2); `grep -rn "from scripts\.(data_processing|behavioral|simulations_recovery|post_mle|bayesian_pipeline)\." scripts/ tests/ validation/ src/` returns zero matches; `pytest scripts/fitting/tests/test_v4_closure.py` still passes 3/3

       - [ ] **REFAC-15**: Docs spare-file integration — merge `docs/HIERARCHICAL_BAYESIAN.md` (→ `docs/04_methods/README.md#hierarchical-bayesian-architecture`), `docs/K_PARAMETERIZATION.md` (→ `docs/03_methods_reference/MODEL_REFERENCE.md#k-parameterization`), and `docs/SCALES_AND_FITTING_AUDIT.md` (→ `docs/04_methods/README.md#scales-orthogonalization-and-audit`); `git mv` originals to `docs/legacy/`; update `manuscript/paper.qmd` line 166 caption reference to the merged SCALES location; `docs/CLUSTER_GPU_LESSONS.md` byte-identical to pre-phase content (enforced via committed `pre_phase29_cluster_gpu_lessons.sha256` hash manifest and sha256 invariant test); `docs/PARALLEL_SCAN_LIKELIHOOD.md` left in place (user directive)

       - [ ] **REFAC-16**: Utils consolidation — extract PPC simulator logic into canonical single-source `scripts/utils/ppc.py` (used by 03 prior-PPC, 03 synthetic-data-generation, and 05 posterior-PPC); rename `scripts/utils/plotting_utils.py`→`plotting.py`, `statistical_tests.py`→`stats.py`, `scoring_functions.py`→`scoring.py` via `git mv`; create empty `scripts/utils/__init__.py` package marker; create `scripts/05_post_fitting_checks/run_posterior_ppc.py` thin orchestrator (mirror of stage 03 prior-PPC); evaluate `remap_mle_ids.py`/`sync_experiment_data.py`/`update_participant_mapping.py` per import-use-count → retain in utils/ or move to scripts/_maintenance/; `grep -rn "def run_prior_ppc\|def run_posterior_ppc\|def simulate_from_samples" scripts/ --include="*.py"` shows definitions ONLY in `scripts/utils/ppc.py`

       - [ ] **REFAC-17**: Dead-folder audit and cleanup — per-folder grep audit (excluding `.planning/` historical docs) documented in `scripts/legacy/README.md` with live-reference counts; `scripts/analysis/` (9 files), `scripts/results/` (5 files), `scripts/simulations/` (5 files), `scripts/statistical_analyses/` (1 file), `scripts/visualization/` (11 files) archived to `scripts/legacy/<folder>/` via `git mv` (history preserved) or deleted if zero live refs; importers for live references rewritten (`validation/test_unified_simulator.py`, `tests/test_wmrl_exploration.py`, `scripts/03_model_prefitting/09_generate_synthetic_data.py`, `scripts/03_model_prefitting/10_run_parameter_sweep.py`, `manuscript/paper.tex` line 244); `scripts/` top level contains only 01–06 stage dirs + utils/ + fitting/ library remnant + legacy/ (+ optional _maintenance/)

       - [ ] **REFAC-18**: Cluster SLURM consolidation — update every `cluster/*.slurm` internal `python scripts/...` path to canonical 01–06 locations; create stage-numbered entry points (`cluster/01_data_processing.slurm`, `02_behav_analyses.slurm`, `03_prefitting_{cpu,gpu}.slurm`, `04a_mle_{cpu,gpu}.slurm`, `04b_bayesian_{cpu,gpu}.slurm`, `04c_level2.slurm`, `05_post_checks.slurm`, `06_fit_analyses.slurm`); consolidate per-model templates (12_mle_single, 13_bayesian_m4_gpu, 13_bayesian_m6b_subscale) into parameterized SLURMs via `--export=MODEL=<name>,TIME=<HH:MM:SS>` following the Phase 28 `13_bayesian_choice_only.slurm` pattern; create `cluster/submit_all.sh` master orchestrator chaining all six stages via `--afterok` with `--dry-run` path-validation mode; `cluster/21_submit_pipeline.sh` either deleted or delegates to `submit_all.sh`; `bash -n cluster/*.slurm` exits 0 for every SLURM; `bash cluster/submit_all.sh --dry-run` exits 0 with all python paths resolving

       - [ ] **REFAC-19**: Paper.qmd + paper.tex script-path updates — rewrite every `scripts/`-prefixed path reference in `manuscript/paper.qmd` (known hits: lines 171, 630, 650) and `manuscript/paper.tex` (known hit: line 244) to the new canonical 01–06 layout; `grep -n "scripts/(data_processing|behavioral|simulations_recovery|post_mle|bayesian_pipeline)" manuscript/paper.qmd manuscript/paper.tex` returns zero matches; `quarto render manuscript/paper.qmd` exits 0 without path-not-found errors (graceful-fallback `{python}` cells from Phase 28 still absorb missing data artifacts — those are not blocking)

       - [ ] **REFAC-20**: Phase 29 closure-guard extension — new pytest `tests/test_v5_phase29_structure.py` asserts the canonical structure shape (6 stage folders present; 4b sub-letters present; 10 dead folders absent from top level; `scripts/utils/ppc.py` has ≥ 2 function definitions; simulator defined ONLY in utils; 3 docs spare files merged and originals live under `docs/legacy/`; `docs/CLUSTER_GPU_LESSONS.md` sha256 matches `pre_phase29_cluster_gpu_lessons.sha256` manifest; zero `from scripts.{data_processing,behavioral,simulations_recovery,post_mle,bayesian_pipeline}.` imports in active tree; utils canonical short names `plotting.py`/`stats.py`/`scoring.py` present, verbose names absent); `python validation/check_v4_closure.py --milestone v4.0` still exits 0
       ```
    
    3. Find the requirement-to-phase ledger table (lines ~143-157 in the existing REQUIREMENTS.md). Append 7 new rows:
       ```
       | REFAC-14 | Phase 29 | Planned |
       | REFAC-15 | Phase 29 | Planned |
       | REFAC-16 | Phase 29 | Planned |
       | REFAC-17 | Phase 29 | Planned |
       | REFAC-18 | Phase 29 | Planned |
       | REFAC-19 | Phase 29 | Planned |
       | REFAC-20 | Phase 29 | Planned |
       ```
       (Status flips to "Complete" when executor updates after each plan lands.)
    
    4. Update the Phase 29 section of the requirement-totals paragraph if present (e.g., if the file says "v1 Requirements: 21 requirements across 5 categories" — since Phase 28 added 13 more and Phase 29 adds 7, the total rises to 41+; the planner should locate and update this paragraph or leave it unchanged if the paragraph is scope-specific).
    
    5. Create `.planning/phases/29-pipeline-canonical-reorg/29-VERIFICATION.md`:
       ```markdown
       # Phase 29 End-of-Phase Verification Report

       **Status:** {pass | gaps_found}
       **Date:** {YYYY-MM-DD}
       **Verifier:** {executor identity}

       ## Success Criteria Coverage

       | SC# | Criterion | Status | Evidence |
       |-----|-----------|--------|----------|
       | 1 | Canonical 6 stage dirs at scripts/ top level | pass | `ls -d scripts/*/` output |
       | 2 | 04_model_fitting/{a,b,c} sub-letters present | pass | `ls -d scripts/04_model_fitting/*/` |
       | 3 | Dead folders gone or under legacy/ | pass | scripts/legacy/README.md audit record |
       | 4 | Simulator single-source in utils/ppc.py | pass | grep result |
       | 5 | Docs spare files merged; originals in legacy/ | pass | file existence checks |
       | 6 | CLUSTER_GPU_LESSONS.md byte-identical | pass | sha256 hash invariant |
       | 7 | cluster/*.slurm canonical paths + submit_all.sh | pass | dry-run output |
       | 8 | paper.qmd renders via quarto | pass | quarto render exit 0 |
       | 9 | v4 closure still green | pass | check_v4_closure.py exit 0 |
       | 10 | Zero old-grouping imports in active tree | pass | grep result |
       | 11 | pytest full suite passes | pass | pytest output |
       | 12 | test_v5_phase29_structure.py pass | pass | pytest output |

       ## Plan-Level Evidence

       | Plan | Status | Commit SHA | Blockers |
       |------|--------|------------|----------|
       | 29-01 scripts reorg | Complete | {sha} | none |
       | 29-02 docs merges | Complete | {sha} | none |
       | 29-03 utils consolidation | Complete | {sha} | none |
       | 29-04 dead-folder audit | Complete | {sha} | none |
       | 29-05 cluster SLURM | Complete | {sha} | none |
       | 29-06 paper.qmd | Complete | {sha} | none |
       | 29-07 closure guard | Complete | {sha} | this plan |
       | 29-08 src/rlwm/fitting/ refactor | {deferred/complete} | {sha or n/a} | user decision |

       ## Deferred Items (v6.0 candidates)

       - (List anything the execution flagged as future work)

       ## Sign-Off

       Phase 29 closure: {pass | gaps_found}
       ```
    
    6. Commit the closure guard and requirements/verification artifacts together:
       ```
       test(29-07): add tests/test_v5_phase29_structure.py closure guard + REQUIREMENTS.md REFAC-14..20
       
       - tests/test_v5_phase29_structure.py — pytest asserts all 12 Phase 29 SC invariants
       - .planning/REQUIREMENTS.md appends REFAC-14..REFAC-20 rows + ledger rows
       - .planning/phases/29-pipeline-canonical-reorg/29-VERIFICATION.md written (status=pass)
       - pytest tests/test_v5_phase29_structure.py: PASSES 100%
       - validation/check_v4_closure.py --milestone v4.0: exits 0
       ```
  </action>
  <verify>
    - `grep -c "REFAC-14\|REFAC-15\|REFAC-16\|REFAC-17\|REFAC-18\|REFAC-19\|REFAC-20" .planning/REQUIREMENTS.md` returns at least 14 (7 bullet rows + 7 ledger rows)
    - `test -f .planning/phases/29-pipeline-canonical-reorg/29-VERIFICATION.md`
    - `grep -c "^| SC#" .planning/phases/29-pipeline-canonical-reorg/29-VERIFICATION.md` returns 1 (header row of SC table)
    - `pytest tests/test_v5_phase29_structure.py -v` PASSES all cases
    - `validation/check_v4_closure.py --milestone v4.0` exits 0
  </verify>
  <done>REFAC-14..REFAC-20 committed to requirements; 29-VERIFICATION.md filled in with evidence; closure guard green.</done>
</task>

</tasks>

<verification>
```bash
# Closure guard passes
pytest tests/test_v5_phase29_structure.py -v

# v4 still green
python validation/check_v4_closure.py --milestone v4.0
pytest scripts/fitting/tests/test_v4_closure.py -v

# Requirements updated
grep -c "^- \[ \] \*\*REFAC-1[4-9]\|^- \[ \] \*\*REFAC-20" .planning/REQUIREMENTS.md
grep -c "^| REFAC-1[4-9] \| Phase 29\|^| REFAC-20 \| Phase 29" .planning/REQUIREMENTS.md

# Verification report
test -f .planning/phases/29-pipeline-canonical-reorg/29-VERIFICATION.md
grep -c "| SC#" .planning/phases/29-pipeline-canonical-reorg/29-VERIFICATION.md

# Full test suite
pytest scripts/fitting/tests/ tests/ validation/ -v
```
</verification>

<success_criteria>
1. `tests/test_v5_phase29_structure.py` passes all parametrize cases (SC#12).
2. `validation/check_v4_closure.py --milestone v4.0` exits 0 (SC#9).
3. `.planning/REQUIREMENTS.md` contains REFAC-14..REFAC-20 rows + ledger entries.
4. `.planning/phases/29-pipeline-canonical-reorg/29-VERIFICATION.md` exists with the SC-evidence table and status=pass.
5. Full pytest suite passes with no new failures vs. pre-phase baseline (SC#11).
</success_criteria>

<output>
After completion, create `.planning/phases/29-pipeline-canonical-reorg/29-07-SUMMARY.md` with:
- Test file content summary (number of test functions, parametrize cases)
- pytest output (last ~30 lines)
- REQUIREMENTS.md diff (7 new bullet rows + 7 ledger rows)
- 29-VERIFICATION.md filled in with evidence per SC
- Commit SHA
</output>
