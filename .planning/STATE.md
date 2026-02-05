# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-05)

**Core value:** Correctly dissociate perseverative responding from learning-rate effects (α₋) to accurately identify whether post-reversal failures reflect motor perseveration or outcome insensitivity
**Current focus:** Phase 4 - Regression Visualization (v2)

## Current Position

Phase: 4 of 7 (Regression Visualization)
Plan: 1 of 2 complete
Status: In progress
Last activity: 2026-02-05 — Completed 04-01-PLAN.md (plotting utility + Script 15)

Progress: [█░░░░░░░░░] 10% (v2: 0/4 phases complete, 1/10 total plans complete)

## Performance Metrics

**v1 Milestone:**
- Total plans completed: 6
- Average duration: 25 min
- Total execution time: 2.5 hours
- Timeline: 2 days (2026-01-29 → 2026-01-30)

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 2 | 59 min | 30 min |
| 2 | 2 | 35 min | 18 min |
| 3 | 2 | 52 min | 26 min |

**v2 Milestone:**
- Total plans completed: 1
- Average duration: 19 min
- Total execution time: 19 min
- Timeline: Started 2026-02-05

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 4 | 1/2 | 19 min | 19 min |

## Accumulated Context

### Model Naming

- M1: Q-learning (α₊, α₋, ε) — existing
- M2: WM-RL hybrid (α₊, α₋, φ, ρ, K, ε) — existing
- M3: WM-RL + κ perseveration (α₊, α₋, φ, ρ, K, ε, κ) — v1 shipped

### v2 Roadmap Structure

- Phase 4: Regression Visualization (REGR-01, REGR-02, REGR-03) — user priority
- Phase 5: Parameter Recovery (RECV-01 through RECV-06) — substantial new code
- Phase 6: Cluster Monitoring (MNTR-01, MNTR-02) — small improvements
- Phase 7: Publication Polish (PUBL-01, PUBL-02) — depends on Phase 4

### Key Decisions

See PROJECT.md Key Decisions table for full history.

**Phase 4 decisions:**
- TRAUMA_GROUP_COLORS constant in plotting_utils matches Script 15 colors
- color-by is visual overlay only (does not change statistical analyses)
- Model-specific figure filenames prevent overwrites (e.g., correlation_heatmap_wmrl_m3.png)
- WM-RL+K display name for M3 model in plots

### Pending Todos

None.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-02-05 (completed 04-01-PLAN.md)
Stopped at: Completed Plan 04-01, ready for Plan 04-02 (Script 16)
Resume file: None
