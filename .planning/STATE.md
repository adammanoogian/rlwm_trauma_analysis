# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-05)

**Core value:** Correctly dissociate perseverative responding from learning-rate effects (α₋) to accurately identify whether post-reversal failures reflect motor perseveration or outcome insensitivity
**Current focus:** Phase 5 - Parameter Recovery & PPC (v2)

## Current Position

Phase: 5 of 7 (Parameter Recovery & PPC)
Plan: 5 of 5 complete
Status: Phase 5 complete - Parameter Recovery + PPC validation operational
Last activity: 2026-02-06 — Completed 05-05-PLAN.md (Model recovery check + Script 09 orchestrator)

Progress: [██████░░░░] 58% (v2: 2/4 phases complete, 7/12 total plans complete)

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
- Total plans completed: 7
- Average duration: 20 min
- Total execution time: 140 min
- Timeline: Started 2026-02-05

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 4 | 2/2 | 37 min | 19 min | ✓
| 5 | 5/5 | 103 min | 21 min | ✓

## Accumulated Context

### Model Naming

- M1: Q-learning (α₊, α₋, ε) — existing
- M2: WM-RL hybrid (α₊, α₋, φ, ρ, K, ε) — existing
- M3: WM-RL + κ perseveration (α₊, α₋, φ, ρ, K, ε, κ) — v1 shipped

### v2 Roadmap Structure

- Phase 4: Regression Visualization (REGR-01, REGR-02, REGR-03) — ✓ complete
- Phase 5: Parameter Recovery & PPC (RECV-01-06, PPC-01-05) — ✓ complete
- Phase 6: Cluster Monitoring (MNTR-01, MNTR-02) — ready to start
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

**Phase 5 decisions:**
- Use JAX for synthetic agent simulation (faster than agent classes)
- Fixed beta=50 in synthetic data (matches real fitting for identifiability)
- Synthetic sona_id starts at 90000 (avoids collision with real data)
- Pass participant_id as int to prepare_participant_data (matches int64 column type)
- Generic plotting utilities follow ax-based pattern for composability
- PASS/FAIL badge based on r >= 0.80 threshold (Senta et al., 2025)
- Distribution comparison plots for sanity checking synthetic data realism
- Exit code 0 if all params pass, 1 if any fail (for automation)
- Script 11 as thin wrapper calling model_recovery functions (no code duplication)
- Multi-model support with --model all for batch validation
- PPC mode outputs to output/ppc/ and figures/ppc/ (separate from recovery artifacts)
- Auto-detect fitted params path from model name if not specified in PPC mode
- Use participant sona_id as seed for reproducible PPC synthetic data
- Behavioral metrics: overall accuracy, set-size accuracy, learning curve, post-reversal
- Model recovery via subprocess calls to Script 12 (maintains pipeline separation)
- Winner determined by summed AIC across participants (lowest wins)
- Auto-detect ID column (sona_id or participant_id) for MLE results compatibility
- Script 09 exit code 0 if pass, 1 if fail (enables automated validation)

### Pending Todos

None.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-02-06 18:51:06
Stopped at: Completed 05-05-PLAN.md (Model recovery check + Script 09 orchestrator)
Resume file: None
