---
phase: 31-final-package-restructure
phase_status: COMPLETE
milestone: v5.0
waves_executed: 5  # W1=31-01, W2=31-02 ‖ 31-03, W3=31-04, W4=31-05, W5=31-06
plans_executed: 6
total_commits: 22
commit_range: "80b3823..a1aeba7"
phase_started: 2026-04-24
phase_completed: 2026-04-24
phase_duration_days: 1
plans_closed: [31-01, 31-02, 31-03, 31-04, 31-05, 31-06]
phase_type: repo-restructure-final-package
tags: [ccds, cookiecutter-data-science-v2, final-package, phase-closure, milestone-v5.0-phase-31]
---

# Phase 31: Final-Package Restructure — Phase-Closure Rollup

**Transitioned the repository from a development layout to a final-package layout suitable for journal submission + long-term reuse. Six coordinated plans across five waves adopted Cookiecutter Data Science v2 conventions, consolidated the test tree into tiered `tests/{unit,integration,scientific}/`, unified `logs/`, removed legacy config aliases, added CITATION.cff, and extended the structure guard from 31 to 56 executable invariants. All 7 phase-level ROADMAP success criteria verified green. Milestone v5.0 Phase 31 of 31 COMPLETE.**

## At a glance

| Metric | Value |
| --- | --- |
| Plans executed | 6 / 6 |
| Waves | 5 (W1 foundation, W2 data+models parallel, W3 tests, W4 cleanup, W5 docs) |
| Total commits | 22 (including plan metadata commits) |
| Task commits | 18 (atomic per-task) |
| Plan metadata commits | 4 (SUMMARY commits for plans 31-01, 31-02, 31-03, 31-05) |
| Commit range | `80b3823..a1aeba7` |
| Phase start | 2026-04-24 (80b3823 add CCDS Path constants) |
| Phase complete | 2026-04-24 (a1aeba7 add CITATION.cff) |
| Physical file moves (git mv) | 104 logs + ~20 data CSVs + ~20 model artifacts |
| Files modified | 60+ across scripts/, src/rlwm/, cluster/, docs/, manuscript/, tests/ |
| Files created | docs/PROJECT_STRUCTURE.md, CITATION.cff, 6 plan SUMMARYs, 1 phase SUMMARY |
| Files deleted | output/, figures/, cluster/logs/, validation/, scripts/fitting/tests/, empty v1/ scaffolds |
| Structure guard tests | 31 (pre-phase) -> 56 (post-phase; +20 new + 5 re-parametrized) |
| ROADMAP success criteria | 7 / 7 verified green |

## Plan-by-plan summary

### Plan 31-01 (Wave W1 — foundation)

**Commits:** 80b3823, 080e206, d17afb4, ed5898f (4 commits)
**SUMMARY:** [.planning/phases/31-final-package-restructure/31-01-SUMMARY.md](./31-01-SUMMARY.md)
**Delivered:**

- `config.py` CCDS Path constants added (13 convenience constants); sole writer for config.py during Waves 1-2 eliminates parallel-execution write race.
- 15 `.gitkeep` files + CCDS .gitignore block scaffolded for `data/{raw,interim,processed,external}`, `models/{bayesian,mle,ppc,recovery,parameter_exploration}`, `reports/{figures,tables}`, `tests/{unit,integration,scientific}`, `logs/`.
- `pytest.ini` + `pyproject.toml` `testpaths` collapsed to `tests/` + scientific marker registered.

### Plan 31-02 (Wave W2a — data tier — parallel with 31-03)

**Commits:** 222cf0a, a64c490, d0c83a5 (3 commits)
**SUMMARY:** [.planning/phases/31-final-package-restructure/31-02-SUMMARY.md](./31-02-SUMMARY.md)
**Delivered:**

- ~20 canonical data CSVs moved via git mv to `data/{raw,interim,processed}/` tiers.
- Stage-01 preprocessing scripts (`01_parse_raw_data.py` .. `04_create_summary_csv.py`) rewritten to use CCDS Path constants via `from config import PROCESSED_DIR, INTERIM_DIR, DATA_RAW_DIR`.
- `data/raw/` gitignored (sensitive), `data/processed/` tracked (reproducible-but-expensive), `data/interim/` gitignored.

### Plan 31-03 (Wave W2b — models+reports tier — parallel with 31-02)

**Commits:** 4a14ab0, b9952ef, 34f15c1, fe4905f (4 commits)
**SUMMARY:** [.planning/phases/31-final-package-restructure/31-03-SUMMARY.md](./31-03-SUMMARY.md)
**Delivered:**

- `output/` subtree split into `models/{bayesian,mle,ppc,recovery}/` + `reports/{figures,tables}/`. Bayesian posteriors (.nc), MLE fits (CSV), and PPC results under `models/`; descriptive tables, model-comparison tables, regression tables, all figures under `reports/`.
- Stage 02-06 scripts (behavioral, prefitting, fitting sub-engines, post-fit checks, fit analyses) migrated off `output/` to CCDS constants.
- 13 SLURM files (cluster/*.slurm) rewritten for CCDS paths; 1 `paper.qmd` rewritten.
- `src/rlwm/fitting/` library retained as single source of truth (jax_likelihoods.py, numpyro_models.py, numpyro_helpers.py); imports via dotted `rlwm.fitting.*`.

### Plan 31-04 (Wave W3 — test consolidation)

**Commits:** 90e0a3d, 3c5f1bf, 99ee03f, 8f36a82 (4 commits)
**SUMMARY:** [.planning/phases/31-final-package-restructure/31-04-SUMMARY.md](./31-04-SUMMARY.md)
**Delivered:**

- `validation/` (scientific-validation tier) + `scripts/fitting/tests/` (fitting-pipeline integration tests) merged into unified `tests/{unit,integration,scientific}/` tree.
- `tests/integration/test_v5_phase29_structure.py` moved from `tests/` to `tests/integration/`; REPO_ROOT recalculated to `parents[2]`.
- Closure-guard invariants updated for post-move paths (2 import fixes + 2 REPO_ROOT depth fixes).
- Fast-tier pytest recovers to green (227 passed pre-31-06 smoke).

### Plan 31-05 (Wave W4 — legacy cleanup + log consolidation)

**Commits:** 716e1e5, e673dc8, 08b57f0, dd554e0 (4 commits)
**SUMMARY:** [.planning/phases/31-final-package-restructure/31-05-SUMMARY.md](./31-05-SUMMARY.md)
**Delivered:**

- 104 log files moved via `git mv cluster/logs/*.{err,out} logs/`; `cluster/logs/` directory deleted.
- 18 active SLURMs rewritten to `#SBATCH --output=logs/<jobname>_%j.out` (single authoritative log location).
- `config.py` legacy aliases removed: `OUTPUT_DIR`, `FIGURES_DIR`, `VERSION`, `OUTPUT_VERSION_DIR`, `FIGURES_VERSION_DIR` deleted entirely (ImportError is intended migration signal).
- 3 DataParams fields removed (SIMULATED_DATA, FITTED_POSTERIORS, MODEL_COMPARISON — zero live consumers).
- `output/` directory removed entirely; `.gitignore` 40+ legacy patterns pruned.
- 5-way final audit: 0 legacy-output / 0 legacy-figures / 0 legacy-config-import / 0 cluster-logs / 0 from-validation.
- `data/sync_log.txt` moved to `logs/sync_log.txt` (reclassified from data to operational log).

### Plan 31-06 (Wave W5 — docs + CITATION + structure-guard extension) — THIS PLAN

**Commits:** b35aba8, ab45884, a1aeba7 (3 commits)
**SUMMARY:** [.planning/phases/31-final-package-restructure/31-06-SUMMARY.md](./31-06-SUMMARY.md)
**Delivered:**

- `tests/integration/test_v5_phase29_structure.py` extended with 20 Phase 31 parametrized invariants across 8 success-criterion headers (SC#1-SC#8). Phase 29 section byte-identical. Total: 56/56 PASSED.
- `docs/PROJECT_STRUCTURE.md` created (129 lines, 10 CCDS refs) — canonical reader-facing layout doc.
- `CITATION.cff` v1.2.0 at repo root (cffconvert --validate GREEN) — Zenodo + GitHub citation-widget + JOSS ready.
- `CLAUDE.md` + `README.md` + `docs/04_methods/README.md` path refs fully migrated to CCDS layout.
- 7-step fresh-clone-equivalent smoke test: all green (bash -n submit_all.sh, CCDS constants import, pip install -e, pytest fast-tier 227 PASS, v4 closure pytest 3/3, v4 closure CLI 5/5, submit_all.sh --dry-run exit 0).
- Quarto render: paper.pdf 16 pages, rendered cleanly.

## Phase-level ROADMAP success criteria (all 7 green)

| # | Criterion | Verification | Status |
| --- | --- | --- | --- |
| 1 | Top-level layout matches CCDS conventions (documented in docs/PROJECT_STRUCTURE.md) | `[ -f docs/PROJECT_STRUCTURE.md ]` TRUE; 129 lines; 10 CCDS refs | ✓ |
| 2 | config.py single Path source; zero hardcoded `output/...` in scripts | 5-way audit (plan 31-05) + SC#8 audit (plan 31-06) both 0 legacy hits | ✓ |
| 3 | tests/ is single top-level tree; validation/ gone; pytest discovers consolidated suite | SC#4 `test_phase31_legacy_dir_removed[validation]` PASS + SC#5 `test_phase31_tests_tier_exists` PASS + fast pytest 227 passed | ✓ |
| 4 | cluster/submit_all.sh --dry-run exits 0 | `bash cluster/submit_all.sh --dry-run` -> exit 0 with "every stage SLURM passed bash -n and every python target resolved on disk." | ✓ |
| 5 | quarto render manuscript/paper.qmd produces paper.pdf | `quarto render paper.qmd --to pdf` -> "Output created: _output\paper.pdf" (16 pages, 1.0 MB) | ✓ |
| 6 | Phase 29 closure guard still passes 31/31 + Phase 31 additions pass | `pytest tests/integration/test_v5_phase29_structure.py` -> 56 passed (36 phase-29 + 20 phase-31) | ✓ |
| 7 | Fresh clone + pip install -e + pytest green — final package readiness | 7-step smoke test in plan 31-06 Task 3 all green (pre-existing Windows JAX flake documented as non-regression) | ✓ |

## Aggregated file metrics

| Category | Count | Notes |
| --- | --- | --- |
| **Top-level layout changes** | | |
| New top-level directories | 6 | data/, models/, reports/, tests/ (tier), logs/, scripts/_maintenance/ (already existed; tracked here for accounting) |
| Deleted top-level directories | 4 | output/, figures/, validation/, cluster/logs/ |
| New top-level files | 2 | CITATION.cff, docs/PROJECT_STRUCTURE.md |
| **File moves (git mv)** | | |
| Log files (cluster/logs/ -> logs/) | 104 | all .err + .out pairs |
| Canonical data CSVs | ~20 | to data/{raw,interim,processed}/ |
| Model artifacts | ~20 | to models/{bayesian,mle,ppc,recovery}/ |
| Report artifacts (figures + tables) | ~30 | to reports/{figures,tables}/ |
| Test files | ~15 | validation/ + scripts/fitting/tests/ -> tests/{scientific,integration}/ |
| **Structural file changes** | | |
| config.py edits | 4 | across 31-01 (CCDS constants add) + 31-02 (data tier flips) + 31-03 (13 convenience constants) + 31-05 (legacy aliases removed) |
| SLURM files rewritten | 18 | all active .slurm: 01-06 stage dispatch + 13_bayesian + cluster helpers |
| Scripts migrated from output/* | 30+ | across stages 01-06 + 04c_level2 + fitting library |
| **Documentation changes** | | |
| .gitignore lines pruned | ~50 | legacy output/, figures/, data/* patterns |
| CLAUDE.md edits | 6 | across Phase 31 (most in 31-06) |
| README.md edits | 1 | 31-06 Project Structure section added |
| New plan SUMMARYs | 6 | 31-01 through 31-06 |
| New phase SUMMARY | 1 | this file |
| **Tests** | | |
| Structure guard tests pre-phase | 31 | Phase 29 SC coverage |
| Structure guard tests post-phase | 56 | +20 Phase 31 + re-parametrize |
| Fast-tier pytest pre-phase | ~200 | baseline v4.0 closure |
| Fast-tier pytest post-phase | 227 | +27 from test consolidation (31-04 moves) |
| v4.0 closure guard (pytest + CLI) | 3 + 5 = 8 | unchanged, still green |

## Complete commit log (22 commits)

| # | Commit | Message | Plan |
| --- | --- | --- | --- |
| 1 | 80b3823 | feat(31-01): add CCDS Path constants + convenience constants to config.py | 31-01 |
| 2 | 080e206 | chore(31-01): scaffold CCDS directories with .gitkeep + update .gitignore | 31-01 |
| 3 | d17afb4 | chore(31-01): collapse pytest testpaths to tests/ + register scientific marker | 31-01 |
| 4 | ed5898f | docs(31-01): complete CCDS foundation plan | 31-01 |
| 5 | 222cf0a | refactor(31-02): move canonical data CSVs to CCDS data/ tiers | 31-02 |
| 6 | a64c490 | refactor(31-02): route stage-01 preprocessing scripts through CCDS constants | 31-02 |
| 7 | 4a14ab0 | refactor(31-03): move output/ and figures/ to CCDS models/ and reports/ | 31-03 |
| 8 | d0c83a5 | docs(31-02): complete data-tier CCDS migration plan | 31-02 |
| 9 | b9952ef | refactor(31-03): adopt CCDS config constants across stage 02-06 + fitting library | 31-03 |
| 10 | 34f15c1 | refactor(31-03): rewrite cluster SLURMs + manuscript paths to CCDS tree | 31-03 |
| 11 | fe4905f | docs(31-03): complete CCDS models/ + reports/ migration plan | 31-03 |
| 12 | 90e0a3d | refactor(31-04): consolidate test tree into tests/{unit,integration,scientific}/ | 31-04 |
| 13 | 3c5f1bf | fix(31-04): update REPO_ROOT depths + closure-guard invariants for post-move paths | 31-04 |
| 14 | 99ee03f | fix(31-04): fix depth + import regressions in moved scientific/integration tests | 31-04 |
| 15 | 8f36a82 | docs(31-04): complete test-tree consolidation plan | 31-04 |
| 16 | 716e1e5 | refactor(31-05): merge cluster/logs/ into logs/ + rewrite SLURM --output directives | 31-05 |
| 17 | e673dc8 | refactor(31-05): remove legacy OUTPUT_DIR/FIGURES_DIR aliases from config.py | 31-05 |
| 18 | 08b57f0 | chore(31-05): delete legacy output/ directory and prune stale .gitignore patterns | 31-05 |
| 19 | dd554e0 | docs(31-05): complete legacy cleanup + log consolidation plan | 31-05 |
| 20 | b35aba8 | test(31-06): extend structure guard with Phase 31 CCDS invariants | 31-06 |
| 21 | ab45884 | docs(31-06): add PROJECT_STRUCTURE.md + sync CLAUDE/README to CCDS layout | 31-06 |
| 22 | a1aeba7 | chore(31-06): add CITATION.cff for JOSS/Zenodo compatibility | 31-06 |

(+ the plan-31-06 plan-metadata commit will follow this phase summary.)

## Decisions Made (phase-level synthesis)

- **Scheme D scripts layout from Phase 29 was orthogonal and kept intact.** Phase 31 touched top-level ONLY (data/, models/, reports/, tests/, logs/). scripts/ internal layout (01..06 stage folders, intra-stage reset numbering, a_mle/b_bayesian/c_level2 subfolders) was designed in Phase 29 and NEVER renamed here.

- **CCDS v2 (not v1) cited.** cookiecutter-data-science.drivendata.org docs are v2-updated; the v1 structure mostly survived but v2 formalized `{raw, interim, processed, external}` tiers and clarified `models/` vs `reports/`. Linked from CITATION.cff + PROJECT_STRUCTURE.md.

- **Legacy aliases removed outright, not deprecated.** Plan 31-05 chose hard `del` over `DeprecationWarning` wrapper. ImportError at import time is the intended migration signal — immediate + noisy + uncoercible.

- **sole-writer pattern for config.py in Waves 1-2.** All config.py edits across Waves 1-2 happen in plan 31-01 (Wave 1). This eliminates parallel-execution write races in Wave 2 (plans 31-02 + 31-03 running concurrently). Wave 4 (plan 31-05) is the only other config.py writer (alias removal after consumers migrated).

- **Unified logs/ + cluster/logs/ retired.** Some projects keep per-subsystem logs (dev/, cluster/, webserver/). This project unified because (a) single SLURM orchestrator dispatches all cluster jobs, (b) dev vs cluster log separation wasn't load-bearing, (c) simpler for reviewer.

- **Structure guard append-only.** Phase 29 SC#1-SC#12 never renumbered or refactored; Phase 31 SC#1-SC#8 appended after. Clean diff for reviewers; zero risk of breaking Phase 29 assertions.

- **CITATION.cff gated on cffconvert validator.** YAML parse alone would miss schema violations. Installing cffconvert 2.0.0 (dev-only) added a strict schema check — "Citation metadata are valid according to schema version 1.2.0."

- **Quarto render is the empirical smoke test for paper.qmd.** Full pipeline dependencies (posterior NetCDFs from Phase 24 cold-start) aren't present locally; the test confirms that (a) paper.qmd syntax is valid, (b) graceful-fallback cells (Phase 28 pattern) render, (c) paper.tex + paper.pdf build clean, (d) no legacy path refs cause render failures. 16 pages of real output delivered.

## Patterns Established

- **Wave-5 closure plan structure.** Plan 31-06 is the template for future "final plan in a multi-wave phase" plans: (a) structure-guard extension, (b) reader-facing docs refresh, (c) citation metadata, (d) fresh-clone-equivalent smoke test spanning all waves' outputs, (e) two SUMMARYs (plan-level + phase-level).

- **Fresh-clone-equivalent smoke test.** 7-step sequence that exercises (i) bash -n orchestrator, (ii) CCDS constant imports from clean Python subprocess, (iii) pip install -e . idempotent, (iv) fast pytest, (v) v4 closure guards pytest + CLI, (vi) submit_all.sh --dry-run, (vii) Quarto render. Surfaces phase regressions without network access.

- **CCDS-first documentation layering.** CITATION.cff (machine) -> README.md (quick orientation) -> docs/PROJECT_STRUCTURE.md (canonical reference) -> CLAUDE.md (AI-assistant alignment) -> tests/integration/test_v5_phase29_structure.py (executable invariants). Each layer is one click (or one grep) away from the next.

- **5-way audit sentinel pattern.** Five grep-counts that should all equal 0 in the live tree. Plan 31-05 established; reviewer can re-verify any time by running the 5 commands from SUMMARY. Phase 31-06 extends with 5 additional parametrized audit cases (SC#8 `output/bayesian`, `output/mle`, etc).

- **Sole-writer rule for shared state files.** config.py in Waves 1-2, .gitignore in 31-05 Task 3 — when multiple plans could touch the same file, assign one plan as sole writer and make downstream plans consume the result. Prevents parallel-execution write races.

## Milestone v5.0 status after Phase 31

| Phase | Milestone | Status |
| --- | --- | --- |
| 23. Tech-Debt Sweep | v5.0 | COMPLETE |
| 24. Cold-Start Pipeline Execution | v5.0 | Not started (empirical prerequisite for final paper) |
| 25. Reproducibility Regression | v5.0 | Not started (unblocked by Phase 31 fresh-clone template) |
| 26. Manuscript Finalization | v5.0 | Not started (unblocked by Phase 31 path stability) |
| 27. Milestone v5.0 Closure | v5.0 | Not started (aggregator for 24/25/26) |
| 28. Bayesian-First Manuscript Restructure | v5.0 | COMPLETE |
| 29. Pipeline Canonical Reorganization | v5.0 | COMPLETE |
| **31. Final-Package Restructure** | **v5.0** | **COMPLETE (this phase, 2026-04-24)** |
| 30. JAX Simulator Consolidation | v5.0/v5.1 | Deferred |

**v5.0 milestone state:** 15/20 plans complete (Phases 23, 28, 29, 31 closed; Phase 30 deferred; Phases 24/25/26/27 pending). Phase 31 closure de-risks remaining v5.0 phases by locking the layout — no more path churn will affect Phase 24/25/26/27 deliverables.

## Next steps

### Immediate (Phase 31 closure)

- [x] Plan 31-06 SUMMARY written ([.planning/phases/31-final-package-restructure/31-06-SUMMARY.md](./31-06-SUMMARY.md))
- [x] Phase 31 PHASE-SUMMARY written (this file)
- [ ] STATE.md updated with Phase 31 COMPLETE entry (handled by execute-plan workflow)
- [ ] Plan-metadata commit for the two SUMMARYs (handled by execute-plan workflow)

### Downstream phases (unblocked)

- **Phase 24 cold-start pipeline execution** — independent empirical prerequisite; Phase 31 path changes are complete so Phase 24 scripts will resolve CCDS paths cleanly.
- **Phase 25 reproducibility regression** — can now extend the Plan 31-06 Task 3 fresh-clone-equivalent smoke test pattern with full cold-start reproduction from data/raw/.
- **Phase 26 manuscript finalization** — Bayesian-first paper.qmd from Phase 28 renders cleanly (Plan 31-06 Task 3 Step 8 proved this); Phase 26 adds the real Phase 24 empirical artifacts + final Pareto-k limitations.
- **Phase 27 milestone v5.0 closure** — aggregator for Phases 24/25/26 + optional Phase 30 decision.

### Residual items (not blockers)

- **2 pre-existing Windows local JAX flakes** (test_affine_scan_ar1, test_affine_scan_reset). Pass in isolation; cluster pytest is authoritative. Documented across plans 31-04, 31-05, 31-06 SUMMARYs.
- **Phase 30 (JAX simulator consolidation)** remains deferred. Orthogonal to Phase 31's top-level restructure.
- **manuscript/paper.tex** auto-regenerated by Quarto render in 31-06 Task 3; left uncommitted (build artifact).

---

## Acceptance

**Phase 31 is COMPLETE for user acceptance.** All 7 ROADMAP success criteria verifiable by the dual closure guards (pytest wrapper + CLI) plus the Phase 31 structure guard (56 tests). The fresh-clone-equivalent smoke test proves that a user cloning this repo today can run `pip install -e . && pytest -m "not slow and not scientific"` and get a green test suite against the final-package layout.

User acceptance checkpoint (for the orchestrator):

- Structure guard 56/56 PASSED
- ROADMAP SC#4 submit_all.sh --dry-run exit 0
- ROADMAP SC#5 Quarto render 16pp paper.pdf
- CITATION.cff cffconvert-valid
- docs/PROJECT_STRUCTURE.md + CLAUDE.md + README.md all CCDS-aligned
- Zero legacy top-level dirs (output/, figures/, validation/, cluster/logs/)
- Zero legacy config imports (OUTPUT_DIR, FIGURES_DIR, OUTPUT_VERSION_DIR)

---

*Phase: 31-final-package-restructure*
*Milestone: v5.0*
*Phase plans: 6 / 6 complete*
*Phase commits: 22 (commit range `80b3823..a1aeba7`)*
*Phase started: 2026-04-24*
*Phase completed: 2026-04-24*
