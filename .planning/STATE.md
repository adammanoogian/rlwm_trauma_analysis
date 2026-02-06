# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-05)

**Core value:** Correctly dissociate perseverative responding from learning-rate effects (α₋) to accurately identify whether post-reversal failures reflect motor perseveration or outcome insensitivity
**Current focus:** Phase 4 - Regression Visualization (v2)

## Current Position

Phase: 4 of 7 (Regression Visualization)
Plan: 2 of 2 complete
Status: Phase 4 complete — verified ✓
Last activity: 2026-02-06 — Phase 4 verification passed (5/5 must-haves)

Progress: [███░░░░░░░] 25% (v2: 1/4 phases complete, 2/10 total plans complete)

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
- Total plans completed: 2
- Average duration: 19 min
- Total execution time: 37 min
- Timeline: Started 2026-02-05

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 4 | 2/2 | 37 min | 19 min | ✓

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
- Script 16 auto-detects params path from model name (removed --params argument)
- Model-specific subdirectories for regression outputs (output/regressions/{model}/)
- Section column in CSV groups each scale x parameter regression

### Pending Todos

None.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-02-06 (Phase 4 verified and completed)
Stopped at: Ready to start Phase 5 (Parameter Recovery)
Resume file: None
