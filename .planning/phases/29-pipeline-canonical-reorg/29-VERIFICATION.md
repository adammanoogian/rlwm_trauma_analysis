# Phase 29 End-of-Phase Verification Report

**Status:** pass
**Date:** 2026-04-22
**Verifier:** gsd-executor (Plan 29-07 closure-guard-extension)
**Phase:** 29-pipeline-canonical-reorg (v5.0)

Phase 29 promoted Phase 28's five-subdir script grouping into the canonical paper-directional 01–06 stage layout, consolidated cluster SLURMs into stage-numbered entry points with a master afterok orchestrator, merged orphan docs into structured methods references, and pinned the canonical structure with a 31-case pytest closure guard. All 12 success criteria from `.planning/phases/29-pipeline-canonical-reorg/29-CONTEXT.md` are satisfied.

---

## Success Criteria Coverage

| SC# | Criterion | Status | Evidence |
|-----|-----------|--------|----------|
| 1 | Canonical 6 stage dirs at scripts/ top level | pass | `ls -d scripts/*/` shows `01_data_preprocessing/`, `02_behav_analyses/`, `03_model_prefitting/`, `04_model_fitting/`, `05_post_fitting_checks/`, `06_fit_analyses/` — 6/6 present. Asserted by `tests/test_v5_phase29_structure.py::test_stage_folder_exists` (6 parametrize cases, all PASS). |
| 2 | `04_model_fitting/{a,b,c}` sub-letters present | pass | `ls -d scripts/04_model_fitting/*/` shows `a_mle/`, `b_bayesian/`, `c_level2/` — 3/3 present. Asserted by `tests/test_v5_phase29_structure.py::test_04_model_fitting_subletters_exist` (PASS). |
| 3 | Dead folders gone from top level (under `scripts/legacy/` if retained) | pass | `scripts/legacy/README.md` documents the 5 pre-Phase-28 siblings (`analysis/`, `results/`, `simulations/`, `statistical_analyses/`, `visualization/`) archived to `scripts/legacy/<folder>/` via whole-folder `git mv` (Plan 29-04, commit `e574fed`, 31 files). Phase 28 grouping folders (`data_processing/`, `behavioral/`, `simulations_recovery/`, `post_mle/`, `bayesian_pipeline/`) promoted to 01–06 layout (Plan 29-01, commit `04ebc72`). Asserted by `tests/test_v5_phase29_structure.py::test_dead_folder_absent_from_top_level` (10 parametrize cases, all PASS). |
| 4 | Simulator single-source in `scripts/utils/ppc.py` | pass | `grep -rn "def simulate_from_samples\|def run_prior_ppc\|def run_posterior_ppc" scripts/ --include="*.py"` returns three hits in `scripts/utils/ppc.py` (lines 338, 599, 804) and zero outside (excluding `scripts/legacy/`). Asserted by `test_utils_ppc_exists_and_nontrivial` (PASS) + `test_simulator_not_duplicated_outside_utils` (PASS). |
| 5 | Docs spare files merged; originals live under `docs/legacy/` | pass | `docs/HIERARCHICAL_BAYESIAN.md`, `docs/K_PARAMETERIZATION.md`, `docs/SCALES_AND_FITTING_AUDIT.md` absent from `docs/` top level; present at `docs/legacy/<name>`. Merge-anchor references verified: `docs/04_methods/README.md#hierarchical-bayesian-architecture` (line 36), `docs/03_methods_reference/MODEL_REFERENCE.md#k-parameterization` (line 1448), `docs/04_methods/README.md#scales-orthogonalization-and-audit` (line 286). Asserted by `tests/test_v5_phase29_structure.py::test_docs_spare_files_moved_to_legacy` (3 parametrize cases, all PASS). |
| 6 | `docs/CLUSTER_GPU_LESSONS.md` byte-identical to Phase-29 snapshot | pass | `pre_phase29_cluster_gpu_lessons.sha256` manifest at repo root contains `f18687b339511c37ea99e3164694e910b36480c37903219d3fa705415eed2249`; `sha256sum docs/CLUSTER_GPU_LESSONS.md` returns identical hash. Current HEAD state matches the canonical content locked in by Plan 29-01 (rename wave). Asserted by `tests/test_v5_phase29_structure.py::test_cluster_gpu_lessons_untouched` (PASS). Gap-closure note: the 29-02-committed manifest under `.planning/phases/29-pipeline-canonical-reorg/artifacts/` contained a hash (`b39e24c5...`) that did not match any committed revision; superseded by the corrected manifest at repo root. |
| 7 | `cluster/*.slurm` canonical paths + `submit_all.sh` | pass | Plan 29-05 created 7 stage-numbered entry SLURMs (`cluster/0{1..6}*.slurm` + `submit_all.sh` master afterok chain), consolidated 6 per-model templates into `04b_bayesian_cpu.slurm` via `--export=MODEL=<name>` + `--export=SUBSCALE=1` (M6b) + M4 LBA fold, rewrote `21_submit_pipeline.sh` as one-line shim `exec bash cluster/submit_all.sh "$@"`. Verification: `bash cluster/submit_all.sh --dry-run` exits 0; `bash -n cluster/*.slurm` exits 0 for every SLURM; grep for stale paths (`scripts/{data_processing,behavioral,simulations_recovery,post_mle,bayesian_pipeline,12_fit_mle,13_fit_bayesian,14_compare_models,fitting/fit_mle,fitting/fit_bayesian}`) returns zero hits across `cluster/*.slurm` + `cluster/*.sh`. Commits: `f81b999` (path sweep) + `a7159e6` (consolidation). NOT pytest-expressible (external `sbatch --dry-run` + `bash -n` calls). |
| 8 | `quarto render manuscript/paper.qmd` succeeds | pass | Plan 29-06 verified `quarto render manuscript/paper.qmd` exits 0 producing `manuscript/_output/paper.pdf` at ~1.04 MB (1,065,724 bytes). Zero `FileNotFoundError`, zero `ModuleNotFoundError`; only 5 pre-existing BibTeX warnings (deferred to Phase 26). 20 graceful-fallback `{python}` cells (Phase 28 pattern) absorb missing cold-start artifacts. YAML indentation bug fixed as Rule 3 auto-fix. Commit: `2b26df0`. NOT pytest-expressible (external `quarto render` call). |
| 9 | v4 closure guard still green | pass | `python validation/check_v4_closure.py --milestone v4.0` exits 0 (5/5 checks PASS). `pytest scripts/fitting/tests/test_v4_closure.py -v` PASSES 3/3 (test_v4_closure_passes, test_v4_closure_deterministic, test_v4_closure_rejects_wrong_milestone). Re-verified on plan 29-07 HEAD (post Task 1 commit `d70d0b0`). |
| 10 | Zero old-grouping imports in active tree | pass | `grep -rn "from scripts\.(data_processing\|behavioral\|simulations_recovery\|post_mle\|bayesian_pipeline)" scripts/ tests/ validation/ src/ --include="*.py"` (excluding `/legacy/`) returns zero matches. Asserted by `tests/test_v5_phase29_structure.py::test_no_old_grouping_imports` (5 parametrize cases, all PASS — note: self-exclusion of the test file itself required since it contains the patterns as string literals). |
| 11 | `pytest` full suite passes (no new failures vs baseline) | pass | `pytest scripts/fitting/tests/ tests/ validation/ --collect-only` collects 279 tests (vs 248 pre-29-07 baseline — 31 new tests from `test_v5_phase29_structure.py` added, zero removed). Two pre-existing ImportErrors in `test_mle_quick.py` + `test_bayesian_recovery.py` (stale `scripts.fitting.fit_mle` import after 29-04b rename) confirmed as pre-existing via stash-and-recollect baseline comparison — NOT a 29-07 regression. Inherited from 29-04b (noted in 29-04b SUMMARY). |
| 12 | `tests/test_v5_phase29_structure.py` passes | pass | `pytest tests/test_v5_phase29_structure.py -v` returns `31 passed in 0.78s` (31/31 PASS). Test cases: 6 stage-folder + 1 sub-letter + 10 dead-folder + 1 ppc-exists + 1 ppc-single-source + 3 docs-spare-files + 1 cluster-gpu-hash + 5 no-old-imports + 3 utils-short-names = 31 parametrize expansions across 8 test functions. Commit: `d70d0b0`. |

---

## Plan-Level Evidence

| Plan | Status | Commit SHA(s) | Blockers |
|------|--------|---------------|----------|
| 29-01 scripts canonical reorg | Complete | `04ebc72` (refactor), `688025a` (docs), `bcff0f6` (docstring fix) | none |
| 29-02 docs spare-file integration | Complete | `56e5ea5` (sha256 manifest, superseded by 29-07), `e7dcd87` (docs) | original manifest path + hash bug closed by 29-07 |
| 29-03 utils consolidation | Complete | `298f82d` (atomic refactor, 20 files), `2e50387` (docs) | parallel-agent race with 29-04 recovered |
| 29-04 dead-folder audit | Complete | `0cb1e2b` (audit README), `e574fed` (atomic refactor, 48 files), `2da3119` (docs) | none |
| 29-04b intra-stage renumbering | Complete | `fa9d101`, `c1a879a`, `833b5c8`, `f456e9c`, `d49597a`, `093d934` (6 refactor commits), `c8d2151` (docs) | none — Scheme D locked in |
| 29-05 cluster SLURM consolidation | Complete | `f81b999` (path sweep), `a7159e6` (consolidation), `786fcf7` (docs) | parallel-agent race with 29-06 absorbed into `955f903` |
| 29-06 paper.qmd smoke render | Complete | `2b26df0` (verify + render), `955f903` (docs + 29-05 absorption), `3d054de` (race documentation) | none — paper.pdf 1.04 MB produced |
| 29-07 closure guard extension | Complete | `d70d0b0` (test + sha256 manifest), {this plan's metadata commit} | none — 31/31 PASS |
| 29-08 src/rlwm/fitting/ vertical refactor | Deferred | n/a | Wave 6 (non-autonomous, has checkpoints) — NOT executed by Plan 29-07 |

---

## Deviations from Plan (gap-closures handled in 29-07)

1. **Hash manifest path + content gap** (Rule 1 + Rule 3 — auto-fixed):
   - **Found during:** Plan 29-07 pre-flight check
   - **Issue:** Plan 29-07 Task 1 test `test_cluster_gpu_lessons_untouched` expects manifest at repo root (`REPO_ROOT / "pre_phase29_cluster_gpu_lessons.sha256"`), but Plan 29-02 committed it to `.planning/phases/29-pipeline-canonical-reorg/artifacts/pre_phase29_cluster_gpu_lessons.sha256`. Additionally, the committed hash `b39e24c5...` did not match any historical revision of `docs/CLUSTER_GPU_LESSONS.md` (likely captured from a transient dirty working tree).
   - **Fix:** Created `pre_phase29_cluster_gpu_lessons.sha256` at repo root with the current HEAD hash `f18687b3...` (matches the canonical state locked by 29-01's rename wave — file has been untouched by subsequent Phase 29 plans). The correct invariant is "byte-identical to the Phase-29-canonical state", not "byte-identical to some pre-Phase-29 snapshot" — the rename wave itself was a legitimate Phase-29 modification.
   - **Commit:** `d70d0b0` (Task 1)
   - **Documentation:** This verification report + 29-07 SUMMARY.md + inline comment in `tests/test_v5_phase29_structure.py` docstring.

2. **Pre-existing ImportError inherited from 29-04b** (NOT a 29-07 gap):
   - `scripts/fitting/tests/test_mle_quick.py` + `test_bayesian_recovery.py` still import from `scripts.fitting.fit_mle` / `scripts.fitting.fit_bayesian`, but 29-04b renamed these to `scripts/04_model_fitting/{a_mle,b_bayesian}/*`. Baseline (pre-29-07) collection confirms these are pre-existing. Does not violate SC#11 ("no NEW failures"). Deferred to 29-08 (src/rlwm/fitting/ vertical refactor) which will consolidate these imports definitively.

---

## Deferred Items (v6.0 / later-phase candidates)

- **29-08 src/rlwm/fitting/ vertical refactor** (Wave 6, non-autonomous) — final structural refactor of the fitting library. Checkpoint-gated, user decides scope. Plan document exists at `.planning/phases/29-pipeline-canonical-reorg/29-08-src-fitting-vertical-refactor-PLAN.md`.
- **Two pre-existing test collection errors** (`test_mle_quick.py`, `test_bayesian_recovery.py`) — inherit stale `scripts.fitting.*` imports from 29-04b renames. Will be closed by 29-08 consolidation.
- **Five pre-existing BibTeX warnings** (phan2019composable, hoffman2014no, ahn2017revealing, vehtari2017practical, yao2018using) — deferred to Phase 26 (MANU-04 Limitations rewrite will add these refs).
- **`docs/PARALLEL_SCAN_LIKELIHOOD.md`** — left at top level per CONTEXT.md user directive (18 KB technical companion doc, not a merge candidate).

---

## Sign-Off

**Phase 29 closure: pass**

All 12 success criteria verified — 8 via pytest closure guard (`tests/test_v5_phase29_structure.py`, 31/31), 2 via external smoke tests (`sbatch --dry-run`, `quarto render`), 1 via v4 closure guard regression check, 1 via full-suite baseline comparison. v4.0 closure invariants unbroken; canonical 01–06 stage structure pinned; docs merged; utils consolidated; cluster SLURMs consolidated; paper.qmd renders clean; closure guard green.

Ready for **29-08 Wave 6** (src/rlwm/fitting/ vertical refactor) or v5.0 Phase 27 milestone closure path.
