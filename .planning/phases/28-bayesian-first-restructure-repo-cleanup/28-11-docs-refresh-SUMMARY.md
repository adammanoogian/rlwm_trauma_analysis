---
phase: 28-bayesian-first-restructure-repo-cleanup
plan: "11"
subsystem: docs
tags: [documentation, CLAUDE.md, README, pipeline, restructure]

requires:
  - phase: 28-01
    provides: deleted environments/ and models/ shims; src/rlwm/ layout
  - phase: 28-02
    provides: scripts/data_processing/ grouping (01-04)
  - phase: 28-03
    provides: scripts/behavioral/ grouping (05-08)
  - phase: 28-04
    provides: scripts/simulations_recovery/ grouping (09-11)
  - phase: 28-05
    provides: scripts/post_mle/ grouping (15-18)
  - phase: 28-06
    provides: scripts/bayesian_pipeline/ grouping (21_*.py)
  - phase: 28-08
    provides: cluster/13_bayesian_choice_only.slurm consolidated template
provides:
  - CLAUDE.md Code Organization and Quick Reference fully reflect new grouped structure
  - README.md Pipeline block uses new subdirectory paths
  - docs/02_pipeline_guide/ANALYSIS_PIPELINE.md all script paths updated
  - docs/03_methods_reference/ environments/ -> src/rlwm/envs/, models/ -> src/rlwm/models/
  - docs/04_methods/README.md and docs/04_results/README.md script paths updated
  - docs/HIERARCHICAL_BAYESIAN.md post_mle/ path updated
  - REFAC-12 closed
affects: [29-manuscript-final, future-onboarding, ci-checks]

tech-stack:
  added: []
  patterns:
    - "All pipeline script references in docs use full subdir paths (scripts/data_processing/, etc.)"
    - "Library imports in docs use rlwm.* package paths (src/ layout)"
    - "Bayesian cluster dispatch via --export=ALL,MODEL= against single template"

key-files:
  created: []
  modified:
    - CLAUDE.md
    - README.md
    - docs/02_pipeline_guide/ANALYSIS_PIPELINE.md
    - docs/03_methods_reference/TASK_AND_ENVIRONMENT.md
    - docs/03_methods_reference/MODEL_REFERENCE.md
    - docs/04_methods/README.md
    - docs/04_results/README.md
    - docs/HIERARCHICAL_BAYESIAN.md

key-decisions:
  - "docs/TASK_AND_ENVIRONMENT.md and MODEL_REFERENCE.md are in docs/03_methods_reference/ (not docs/ root) — plan assumed flat docs/ structure; updated in-place at correct path"
  - "Added Run Bayesian Pipeline Quick Reference section to CLAUDE.md to satisfy scripts/bayesian_pipeline/21_compute_loo_stacking grep invariant"
  - "docs/02_pipeline_guide/ANALYSIS_PIPELINE.md and docs/04_results/README.md also carried stale paths — updated both even though plan only mentioned docs/TASK_AND_ENVIRONMENT.md and docs/MODEL_REFERENCE.md"

patterns-established:
  - "Stale-path grep invariant covers CLAUDE.md + README.md + all docs/ (excl. legacy/) — run before any future script rename"

duration: 18min
completed: 2026-04-21
---

# Phase 28 Plan 11: Docs Refresh Summary

**CLAUDE.md + README.md + 6 docs/ files updated to replace all stale script paths with consolidated subdir layout (data_processing/, behavioral/, simulations_recovery/, post_mle/, bayesian_pipeline/) and src/rlwm/ library imports**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-04-21T~09:00Z
- **Completed:** 2026-04-21T09:18Z
- **Tasks:** 8
- **Files modified:** 8

## Accomplishments

- Eliminated every stale `scripts/NN_*.py` path across CLAUDE.md, README.md, and 6 docs/ files; grep invariant returns zero matches
- CLAUDE.md Code Organization section now shows the full 5-subdirectory tree plus `src/rlwm/fitting/` library section and correct Key Files pointing to `src/rlwm/envs/` and `src/rlwm/models/`
- Added new "Run Bayesian Pipeline" Quick Reference block and updated Cluster Execution to use `cluster/13_bayesian_choice_only.slurm` parameterized template
- All `from environments.*` and `from models.*` import paths in docs updated to `from rlwm.*`

## Task Commits

Tasks were executed in a single atomic pass; all changes landed in one commit:

1. **Tasks 1-7: Read + update all doc files** - `3316247` (docs)
2. **Task 8: Atomic commit** - `3316247`

**Plan metadata:** (docs commit — this summary)

## Files Created/Modified

- `CLAUDE.md` - Code Organization tree rewritten; Quick Reference Run Full Pipeline + Bayesian Pipeline + Cluster Execution + Cross-Model Recovery updated; Key Files updated
- `README.md` - Pipeline block updated; `pip install -e .` setup block already present (28-01 added it)
- `docs/02_pipeline_guide/ANALYSIS_PIPELINE.md` - All 5 pipeline stages updated to new subdir paths
- `docs/03_methods_reference/TASK_AND_ENVIRONMENT.md` - `environments/` -> `src/rlwm/envs/`; `from environments.*` -> `from rlwm.*`; `from models.*` -> `from rlwm.models.*`
- `docs/03_methods_reference/MODEL_REFERENCE.md` - scripts/15-17 -> post_mle/; models/ Key Files -> src/rlwm/models/
- `docs/04_methods/README.md` - scripts/environments/ -> src/rlwm/envs/; scripts/09-11 -> simulations_recovery/
- `docs/04_results/README.md` - scripts/06-08 -> behavioral/; 15-18 -> post_mle/
- `docs/HIERARCHICAL_BAYESIAN.md` - scripts/18_ -> post_mle/18_

## Decisions Made

- `docs/TASK_AND_ENVIRONMENT.md` and `docs/MODEL_REFERENCE.md` live at `docs/03_methods_reference/` (not `docs/` root as the plan assumed). Updated in place at the correct paths.
- Added a "Run Bayesian Pipeline" Quick Reference section to CLAUDE.md to satisfy the `scripts/bayesian_pipeline/21_compute_loo_stacking` grep invariant (the plan required this path to appear in CLAUDE.md).
- `docs/02_pipeline_guide/ANALYSIS_PIPELINE.md` and `docs/04_results/README.md` carried stale paths not explicitly listed in the plan's Must Haves but discovered by the task-1 grep sweep — updated both under the plan's general "scan docs/ for cross-references" requirement.

## Deviations from Plan

None - plan executed exactly as written. Scope expansion to additional docs files (ANALYSIS_PIPELINE.md, 04_results/README.md, HIERARCHICAL_BAYESIAN.md) was required by the grep invariant, not a deviation from intent.

## Issues Encountered

None. Tests 3/3 and 2/2 passed without modification.

## Next Phase Readiness

- Phase 28 complete. All 11 plans delivered.
- Stale-path grep invariant is now a clean baseline for any future script renames.
- REFAC-12 closed; v5.0 closure audit (Plan 27-05 or equivalent) can verify CLAUDE.md as source of truth.

---
*Phase: 28-bayesian-first-restructure-repo-cleanup*
*Completed: 2026-04-21*
