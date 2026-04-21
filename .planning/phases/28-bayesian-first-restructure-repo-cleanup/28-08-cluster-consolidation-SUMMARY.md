---
phase: 28-bayesian-first-restructure-repo-cleanup
plan: "08"
subsystem: infra
tags: [slurm, cluster, bayesian, parameterized-template, git-rm]

# Dependency graph
requires:
  - phase: 28-06
    provides: bayesian_pipeline/ grouping; 21_*.slurm paths updated

provides:
  - cluster/13_bayesian_choice_only.slurm — single parameterized Bayesian SLURM template for all 6 choice-only models
  - 6 per-model templates deleted (m1/m2/m3/m5/m6a/m6b)
  - M6b 36h wall-time override convention documented

affects: [28-11-closure, future-cluster-submissions]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Parameterized SLURM: MODEL env var drives --model arg; M6b overrides --time on sbatch command line"

key-files:
  created:
    - cluster/13_bayesian_choice_only.slurm
  modified: []

key-decisions:
  - "M6b wall-time: kept default 24h in #SBATCH directive; M6b callers pass --time=36:00:00 on sbatch command line (avoids SBATCH-vs-shell parse ordering problem)"
  - "No changes to 21_submit_pipeline.sh or 21_3_fit_baseline.slurm — both were already parameterized via --export=ALL,MODEL=$m"
  - "Grep invariant: m4_gpu self-references in retained 13_bayesian_m4_gpu.slurm are false positives; filtered via grep -v m4_gpu"

patterns-established:
  - "Choice-only Bayesian SLURM dispatch: sbatch --export=ALL,MODEL=<name> cluster/13_bayesian_choice_only.slurm"
  - "M6b submission: sbatch --time=36:00:00 --export=ALL,MODEL=wmrl_m6b cluster/13_bayesian_choice_only.slurm"

# Metrics
duration: 8min
completed: 2026-04-21
---

# Phase 28 Plan 08: Cluster SLURM Consolidation Summary

**6 structurally-identical per-model Bayesian SLURM templates collapsed into one parameterized `cluster/13_bayesian_choice_only.slurm` with MODEL and TIME override conventions; net template count drops from 13 to 8 Bayesian-related files**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-04-21T~UTC
- **Completed:** 2026-04-21T~UTC
- **Tasks:** 7/7
- **Files modified:** 7 (1 created, 6 deleted)

## Accomplishments

- Created `cluster/13_bayesian_choice_only.slurm` from `13_bayesian_m3.slurm` as canonical source, with `MODEL="${MODEL:-wmrl_m3}"` default, `--model "$MODEL"` substituted throughout, and `#SBATCH --time=24:00:00` default with M6b 36h override documented in header comment block
- Deleted 6 per-model templates via `git rm`: `13_bayesian_m{1,2,3,5,6a,6b}.slurm`
- Confirmed `21_submit_pipeline.sh` and `21_3_fit_baseline.slurm` required no changes — both were already fully parameterized via `--export=ALL,MODEL=$m`
- Bash syntax check passed (`bash -n` exit 0); grep invariant zero stale refs to deleted templates; test_v4_closure 3/3, test_load_side_validation 2/2 PASS

## Task Commits

1. **Tasks 1-7: consolidation + deletion + verification** — `4af60ba` (refactor)

**Plan metadata:** (docs commit — this SUMMARY)

## Files Created/Modified

- `cluster/13_bayesian_choice_only.slurm` - Parameterized template for all 6 choice-only models; MODEL env var; M6b 36h override convention
- `cluster/13_bayesian_m{1,2,3,5,6a,6b}.slurm` - Deleted via git rm (6 files, 600 lines removed)

## Decisions Made

1. **M6b wall-time approach**: Kept `#SBATCH --time=24:00:00` as static SBATCH directive (default for M1-M6a). M6b callers use `sbatch --time=36:00:00 --export=ALL,MODEL=wmrl_m6b ...` which overrides the directive on the command line. This is the clean approach: SBATCH directives are parsed before shell expansion, so `#SBATCH --time=$TIME` would not work.

2. **No orchestrator changes needed**: Per `28-RESEARCH.md §Q3`, `21_submit_pipeline.sh` loops with `--export=ALL,MODEL=$m cluster/21_3_fit_baseline.slurm` which delegates to `scripts/bayesian_pipeline/21_fit_baseline.py` — it never called the per-model templates directly. The templates were standalone Phase 13-era artifacts not wired into the Phase 21 pipeline.

3. **Grep invariant false positives from m4_gpu**: The pattern `13_bayesian_m[1-6]` also matches `13_bayesian_m4_gpu.slurm` (which is retained). Three matches in `m4_gpu` are all self-references (its own Usage comments and an echo string). Adding `grep -v m4_gpu` to the invariant produces zero results.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

The plan's grep invariant filter `grep -v "^[[:space:]]*[^:]*:[[:space:]]*#"` does not correctly strip comment lines from grep's `file:linenum:content` output format when line numbers are multi-digit (e.g., `:22:` doesn't match `:[[:space:]]`). This caused the `13_bayesian_m4_gpu.slurm` self-references to survive the comment filter. Investigation confirmed all surviving matches are in the retained `m4_gpu` template and are not operational references to deleted templates. Resolution: added `grep -v m4_gpu` to the invariant for clean zero output.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 28-08 complete (REFAC-09 closed)
- Net Bayesian template count: 8 (1 consolidated + 7 specialized)
- Ready for 28-10 (CLAUDE.md update) and 28-11 (closure)
- M6b submission convention documented in new template header; callers must use `--time=36:00:00` override

---
*Phase: 28-bayesian-first-restructure-repo-cleanup*
*Completed: 2026-04-21*
