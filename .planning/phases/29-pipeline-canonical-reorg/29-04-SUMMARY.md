---
phase: 29-pipeline-canonical-reorg
plan: 04
subsystem: infra
tags: [refactor, repo-hygiene, legacy-archive, git-mv, importer-rewrite]

# Dependency graph
requires:
  - phase: 29-pipeline-canonical-reorg
    plan: 01
    provides: canonical 01–06 stage layout; test_load_side_validation.py enumeration with 5 TODO(29-04) stubs
provides:
  - scripts/legacy/ archive with per-folder audit README
  - 5 sibling folders purged from scripts/ top level
  - 7 live importers rewritten to scripts.legacy.simulations paths
  - test_load_side_validation.py enumeration restored to scripts/legacy/ paths
  - manuscript/paper.{tex,qmd} captions updated to reflect legacy location
affects:
  - 29-05-cluster-slurm-consolidation (no SLURM refs to 5 folders expected; confirmed zero matches in cluster/)
  - 29-06-paper-qmd-smoke-render (caption paths now point at legacy/, quarto render should still succeed with updated prose)
  - 29-07-closure-guard-extension (closure guard will now be able to enforce "no scripts/{analysis,results,simulations,statistical_analyses,visualization}/ at top level" invariant)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "scripts/legacy/ archive convention with per-folder audit trail in README.md"
    - "Git-mv-based archival (preserves history, avoids destructive delete)"

key-files:
  created:
    - scripts/legacy/README.md
    - scripts/legacy/analysis/ (9 files archived)
    - scripts/legacy/results/ (5 files archived)
    - scripts/legacy/simulations/ (5 files + README archived)
    - scripts/legacy/statistical_analyses/ (1 file archived)
    - scripts/legacy/visualization/ (11 files archived)
  modified:
    - scripts/03_model_prefitting/09_generate_synthetic_data.py
    - scripts/03_model_prefitting/10_run_parameter_sweep.py
    - scripts/fitting/tests/test_load_side_validation.py
    - validation/test_unified_simulator.py
    - tests/test_wmrl_exploration.py
    - tests/legacy/examples/explore_prior_parameter_space.py
    - tests/legacy/examples/example_parameter_sweep.py
    - tests/legacy/examples/example_visualize_sweeps.py
    - manuscript/paper.qmd
    - manuscript/paper.tex
    - run_data_pipeline.sh
    - docs/README.md
    - docs/04_results/README.md
    - docs/02_pipeline_guide/PLOTTING_REFERENCE.md
    - docs/01_project_protocol/plotting_config_guide.md
    - docs/01_project_protocol/PARTICIPANT_EXCLUSIONS.md

key-decisions:
  - "All 5 folders archived as LEGACY-ARCHIVE (whole-folder git mv) rather than DELETE — preserves git-log-visible content for future audit without cost"
  - "scripts/simulations/ kept archived in legacy/ rather than semantically relocated to scripts/utils/ or src/rlwm/simulators/ — cheapest churn (7 importer rewrites) matched the dead-folder-audit scope; a future milestone can promote the simulator if warranted"
  - "Manuscript caption paths rewritten to scripts/legacy/ rather than salvaging trauma_scale_distributions.py into 02_behav_analyses/ — the caption is textual attribution not runtime, and the file is not a pipeline producer, so archival is the natural fate"
  - "run_data_pipeline.sh pre-existing stale references commented out rather than deleted — preserves history of what the pipeline used to do and allows a future tech-debt plan to decide on restoration vs. removal"
  - "Documentation references in docs/ updated to scripts/legacy/ paths + historical banner added to PLOTTING_REFERENCE.md — chose path-correctness over banner-only because the grep invariant requires consistent paths, and leaving stale docs paths would fail the closure check when 29-07 lands"

patterns-established:
  - "LEGACY-ARCHIVE protocol: per-folder audit README + git mv (not git rm) + importer rewrite"
  - "Five-folder purge commit: atomic refactor with rename detection preserving history"
  - "Parallel plan coordination: staged-file hygiene when 29-03 runs in parallel (git diff --cached --stat check before commit, reset --mixed if 29-03 work leaks in)"

# Metrics
duration: 47min
completed: 2026-04-22
---

# Phase 29 Plan 04: Dead-Folder Audit Summary

**Archived 5 sibling folders (31 files) from scripts/ top level to scripts/legacy/ via whole-folder git mv + audit README; 7 live importers rewritten to scripts.legacy.simulations paths; test_load_side_validation.py enumeration restored from TODO(29-04) stubs.**

## Performance

- **Duration:** 47 min (includes recovery from a staging-mix mishap caused by parallel 29-03 work)
- **Started:** 2026-04-22 (immediately after 29-02 close; 29-03 running in parallel)
- **Completed:** 2026-04-22
- **Tasks:** 2
- **Commits:** 2 atomic (Task 1 + Task 2)
- **Files touched:** 48 (staged in Task 2) + 1 new (scripts/legacy/README.md in Task 1) = 49 total

## Accomplishments

- scripts/ top level reduced to the canonical 9-directory shape (plus `_maintenance/` which 29-03 is authoring in the working tree)
- Every file in the 5 audited folders has a documented decision in `scripts/legacy/README.md` with live-reference grep evidence
- All live importers rewritten; `grep` for `from scripts.{analysis,results,simulations,statistical_analyses,visualization}.*` returns zero outside `scripts/legacy/` and `.planning/`
- v4 closure (3/3) and load-side validation (2/2) still PASS post-move
- Plan-internal coordination with 29-01 resolved: 5 `_ENUMERATED_FILES` entries restored with `scripts/legacy/` paths

## Task Commits

Each task was committed atomically (per CLAUDE.md no Co-Authored-By convention):

1. **Task 1: Per-folder live-reference audit + write scripts/legacy/README.md** — `0cb1e2b` (docs)
2. **Task 2: Execute decisions — salvage LIVE files + move LEGACY-ARCHIVE folders + delete DELETE files** — `e574fed` (refactor)

## Per-Folder Decisions

### scripts/analysis/ → scripts/legacy/analysis/ (LEGACY-ARCHIVE, 9 files)

**Live-reference scan:**
- `tests/legacy/examples/explore_prior_parameter_space.py:63` imports `scripts.analysis.plotting_utils` (itself a legacy test example)
- `scripts/simulations/visualize_parameter_sweeps.py:24` imports same (itself moving to legacy too)
- `manuscript/paper.tex:244` + `paper.qmd:171` textual caption attribution for `trauma_scale_distributions.py` (NOT runtime imports)
- `run_data_pipeline.sh` references 4 files that **do not exist on disk** (stale refs from an earlier cleanup wave, pre-Phase-28)

**Action:** `git mv scripts/analysis scripts/legacy/analysis`. Manuscript captions rewritten to point at legacy path. Stale shell-script refs commented out with a block comment explaining the history.

### scripts/results/ → scripts/legacy/results/ (LEGACY-ARCHIVE, 5 files)

**Live-reference scan:** Zero.

**Action:** `git mv scripts/results scripts/legacy/results`. No importer rewrites.

### scripts/simulations/ → scripts/legacy/simulations/ (LEGACY-ARCHIVE, 5 files + README)

**Live-reference scan (the ONLY folder with active-pipeline imports):**
- `validation/test_unified_simulator.py:24`
- `tests/test_wmrl_exploration.py:16`
- `scripts/03_model_prefitting/09_generate_synthetic_data.py:48`
- `scripts/03_model_prefitting/10_run_parameter_sweep.py:57`
- `tests/legacy/examples/{explore_prior_parameter_space, example_parameter_sweep, example_visualize_sweeps}.py`

**Action:** `git mv scripts/simulations scripts/legacy/simulations`. All 7 importers rewritten to `from scripts.legacy.simulations.<module>`. Intra-legacy cross-imports (simulations → analysis plotting_utils; parameter_sweep/generate_data → unified_simulator; README.md code example) also rewritten to keep the archived folder self-consistent.

### scripts/statistical_analyses/ → scripts/legacy/statistical_analyses/ (LEGACY-ARCHIVE, 1 file)

**Live-reference scan:** Zero.

**Action:** `git mv scripts/statistical_analyses scripts/legacy/statistical_analyses`. No importer rewrites.

### scripts/visualization/ → scripts/legacy/visualization/ (LEGACY-ARCHIVE, 11 files)

**Live-reference scan:**
- Zero live Python imports (confirmed via grep)
- Five docs files with prose/shell-command references (PLOTTING_REFERENCE.md, docs/README.md, docs/04_results/README.md, docs/01_project_protocol/{plotting_config_guide, PARTICIPANT_EXCLUSIONS}.md)
- `manuscript/paper.qmd` does NOT invoke any visualization script at render time

**Action:** `git mv scripts/visualization scripts/legacy/visualization`. Docs references updated to `scripts/legacy/visualization/` paths; PLOTTING_REFERENCE.md got a "Historical reference" banner at the top. Test_load_side_validation.py's 4 commented-out TODO(29-04) entries restored to `scripts/legacy/visualization/` paths.

## Importer Rewrite Count

**7 live importers rewritten** (all to `from scripts.legacy.simulations.*`):

1. `scripts/03_model_prefitting/09_generate_synthetic_data.py:48`
2. `scripts/03_model_prefitting/10_run_parameter_sweep.py:57`
3. `validation/test_unified_simulator.py:24`
4. `tests/test_wmrl_exploration.py:16`
5. `tests/legacy/examples/explore_prior_parameter_space.py:62+63` (2 imports: simulations + analysis)
6. `tests/legacy/examples/example_parameter_sweep.py:22`
7. `tests/legacy/examples/example_visualize_sweeps.py:19`

**4 intra-legacy cross-imports rewritten** (legacy/* self-consistency):

1. `scripts/legacy/simulations/generate_data.py:42`
2. `scripts/legacy/simulations/parameter_sweep.py:44`
3. `scripts/legacy/simulations/visualize_parameter_sweeps.py:24` (to `scripts.legacy.analysis.plotting_utils`)
4. `scripts/legacy/simulations/README.md:16` (code example)

**Plan-internal coordination with 29-01:** 5 commented-out `_ENUMERATED_FILES` entries in `scripts/fitting/tests/test_load_side_validation.py` restored with `scripts/legacy/` paths; TODO(29-04) marker removed; test passes with zero `[MISSING]` violations.

## Files Created/Modified

**Created:**
- `scripts/legacy/README.md` (231 lines; per-folder audit record)

**Moved (via git mv, history preserved):**
- `scripts/analysis/*` (9 files) → `scripts/legacy/analysis/`
- `scripts/results/*` (5 files) → `scripts/legacy/results/`
- `scripts/simulations/*` (6 files incl. README) → `scripts/legacy/simulations/`
- `scripts/statistical_analyses/*` (1 file) → `scripts/legacy/statistical_analyses/`
- `scripts/visualization/*` (11 files) → `scripts/legacy/visualization/`

**Content-modified inside legacy/** (import rewrites):
- `scripts/legacy/simulations/generate_data.py`
- `scripts/legacy/simulations/parameter_sweep.py`
- `scripts/legacy/simulations/visualize_parameter_sweeps.py`
- `scripts/legacy/simulations/README.md`

**Modified outside legacy/:**
- `scripts/03_model_prefitting/09_generate_synthetic_data.py` (import)
- `scripts/03_model_prefitting/10_run_parameter_sweep.py` (import)
- `scripts/fitting/tests/test_load_side_validation.py` (restored 5 enumeration entries; removed TODO marker)
- `validation/test_unified_simulator.py` (import)
- `tests/test_wmrl_exploration.py` (import)
- `tests/legacy/examples/{explore_prior_parameter_space,example_parameter_sweep,example_visualize_sweeps}.py` (imports)
- `manuscript/paper.qmd` (caption path)
- `manuscript/paper.tex` (caption path)
- `run_data_pipeline.sh` (commented out 4 stale-ref lines)
- `docs/{README.md, 04_results/README.md, 01_project_protocol/plotting_config_guide.md, 01_project_protocol/PARTICIPANT_EXCLUSIONS.md}` (docs prose updated)
- `docs/02_pipeline_guide/PLOTTING_REFERENCE.md` (added historical banner; all 8 `scripts/visualization/` refs → `scripts/legacy/visualization/`)

## Final `scripts/` Top-Level Structure

```
scripts/
├── _maintenance/           ← authored by parallel plan 29-03 (present in working tree; not in 29-04 commits)
├── 01_data_preprocessing/
├── 02_behav_analyses/
├── 03_model_prefitting/
├── 04_model_fitting/
├── 05_post_fitting_checks/
├── 06_fit_analyses/
├── fitting/                ← library remnant from 29-01
├── legacy/                 ← new archive (29-04)
│   ├── README.md
│   ├── analysis/
│   ├── results/
│   ├── simulations/
│   ├── statistical_analyses/
│   └── visualization/
└── utils/
```

## Decisions Made

1. **Archive-whole-folder vs. partial-salvage for `scripts/analysis/`:** Chose whole-folder archive because the only in-active-tree reference (`trauma_scale_distributions.py` in manuscript caption) is textual attribution, not a runtime producer. Moving one file into `02_behav_analyses/` as a standalone "salvage" would create a 1-file stub that doesn't fit the folder's theme; the manuscript caption is equally happy pointing at the legacy path.

2. **`scripts/simulations/` stays archived (Option A) rather than relocated to `scripts/utils/simulators.py` (Option B) or `src/rlwm/simulators/` (Option C):** The live callers are all tests + 2 stage-03 wrapper scripts — closer to "test fixture" than "shared library helper." Cheapest churn (7 import-site rewrites vs. reshaping `src/rlwm/`) matches the dead-folder-audit scope. If the closure guard (29-07) or a future phase flags the legacy/ import as "active pipeline in disguise," a follow-up plan can promote the simulator then.

3. **`docs/02_pipeline_guide/PLOTTING_REFERENCE.md` fully rewritten to `scripts/legacy/` paths + historical banner added** rather than leaving prose with old paths and only a banner. Rationale: the plan's success-criterion grep sweeps `docs/` too, so leaving stale paths would fail the closure check; updating paths makes the doc correct-as-written and the banner explains the provenance.

4. **`run_data_pipeline.sh` stale refs commented out, not deleted.** The 4 referenced files (`visualize_human_performance.py` etc.) were deleted from `scripts/analysis/` in a pre-Phase-28 cleanup wave — the pipeline script has been silently broken for those 4 steps for a while. Commenting out with a block comment preserves the history of what the pipeline used to do and signals a future tech-debt plan can decide whether to (a) restore+modernize or (b) prune permanently.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `run_data_pipeline.sh` references 4 non-existent files**
- **Found during:** Task 1 audit (live-reference grep)
- **Issue:** Lines 35/38/41/44 invoke `scripts/analysis/visualize_human_performance.py`, `visualize_scale_distributions.py`, `visualize_scale_correlations.py`, `summarize_behavioral_data.py` — none of which exist on disk (deleted in an earlier cleanup wave). The script silently fails at these steps even without Phase 29-04's work.
- **Fix:** Block-commented the 4 lines with a header explaining the history and suggesting a future tech-debt plan decide the fate. Not deleted outright because that would erase provenance.
- **Files modified:** `run_data_pipeline.sh`
- **Verification:** `bash -n run_data_pipeline.sh` parses without error; the commented block is grep-invisible for `from scripts.{dead-folder}.*` patterns.
- **Committed in:** `e574fed` (Task 2)

**2. [Rule 3 - Blocking] Parallel-plan staging contamination (29-03 work absorbed into first Task 2 commit attempt)**
- **Found during:** First Task 2 commit (`f0f02bd`, now reset)
- **Issue:** During a stash-pop recovery dance, 29-03's working-tree changes migrated into my staging area. My first Task 2 commit absorbed ALL of 29-03's work (scripts/utils/plotting.py renames, scripts/_maintenance/, scripts/utils/ppc.py, scripts/05_post_fitting_checks/run_posterior_ppc.py, modifications to 10+ files 29-03 was editing) and NONE of my 29-04 moves.
- **Fix:** `git reset --mixed HEAD~1` to unwind the bad commit (keeping both my changes and 29-03's as unstaged). Then staged EXPLICITLY only my 29-04 files with targeted `git add` + `git diff --cached --name-only` verification that the staging area contains nothing outside 29-04 scope.
- **Files modified:** N/A (staging hygiene; reset dropped the bad commit)
- **Verification:** Final commit `e574fed` contains exactly 48 files, all verifiably in 29-04 scope (grep --cached --stat shows only legacy/ renames + targeted importer rewrites + docs/manuscript/shell edits).
- **Committed in:** `e574fed` (Task 2 after recovery)

**3. [Rule 1 - Bug] Docs references to `scripts/analysis/regress_parameters_on_scales.py` at non-existent path**
- **Found during:** Task 2 verification grep
- **Issue:** `docs/01_project_protocol/PARTICIPANT_EXCLUSIONS.md:56` referenced `scripts/analysis/regress_parameters_on_scales.py` — but this file is actually at `scripts/06_fit_analyses/regress_parameters_on_scales.py` (moved by plan 29-01). The doc reference was already stale before 29-04; 29-04 surfaces it during its grep sweep.
- **Fix:** Updated doc reference to the canonical `scripts/06_fit_analyses/` path.
- **Files modified:** `docs/01_project_protocol/PARTICIPANT_EXCLUSIONS.md`
- **Verification:** `regress_parameters_on_scales.py` exists at the new path; grep for `scripts/analysis/regress_parameters_on_scales` returns zero in `docs/`.
- **Committed in:** `e574fed` (Task 2)

---

**Total deviations:** 3 auto-fixed (2 × Rule 1 bugs, 1 × Rule 3 blocking)
**Impact on plan:** All auto-fixes necessary for correctness. No scope creep — all three relate to stale references surfacing during the audit's own grep sweeps, which is exactly what the audit is designed to catch.

## Issues Encountered

1. **Parallel-plan coordination with 29-03 caused staging-area contamination once.** Fully recovered (documented above as Rule 3 blocking deviation). Lesson: when two plans touch adjacent parts of `scripts/`, always run `git diff --cached --name-only` before committing, and be wary of `git stash`-based recovery because it merges unstaged changes back without distinguishing source.

2. **`scripts/utils/ppc.py:24`** has a docstring prose reference to `scripts/simulations/unified_simulator.py` (old path). This is in 29-03's new file. Coordination-wise, 29-03's plan is the owner of `scripts/utils/ppc.py`; the instruction was to NOT touch 29-03's files. The reference is prose, not an import, and will resolve when 29-03 commits + notices the stale path, OR when 29-05/29-06 scans docs later. Documented here so downstream plans know where it came from.

## User Setup Required

None — this is a refactor/archive plan with no external service involvement.

## Next Phase Readiness

**Ready for:**
- **29-05 (cluster SLURM consolidation):** zero `scripts/{analysis,results,simulations,statistical_analyses,visualization}/` references in `cluster/*.slurm` confirmed via grep; 29-05 can proceed without coordination with 29-04.
- **29-06 (paper.qmd + paper.tex path smoke render):** caption paths rewritten to `scripts/legacy/`; quarto render should still succeed (textual citation only, not runtime). 29-06 should verify the figure still exists at its output path (`figures/scale_distributions.png`) regardless of where the producer script now lives.
- **29-07 (closure guard extension):** can now safely assert "scripts/analysis/, scripts/results/, scripts/simulations/, scripts/statistical_analyses/, scripts/visualization/ MUST NOT exist at top level" as a closure invariant.

**Blockers:** None for 29-04 itself.

**Outstanding coordination items:**
- `scripts/utils/ppc.py:24` prose reference to old `scripts/simulations/unified_simulator.py` path — 29-03's file; expect 29-03 to resolve when their plan commits.
- `run_data_pipeline.sh` block-commented stale references — suggests a future tech-debt plan (v6.0?) should decide restore-vs-prune.

---
*Phase: 29-pipeline-canonical-reorg*
*Completed: 2026-04-22*
