# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-29)

**Core value:** Correctly dissociate perseverative responding from learning-rate effects (α₋) to accurately identify whether post-reversal failures reflect motor perseveration or outcome insensitivity
**Current focus:** Phase 1 - Core Implementation

## Current Position

Phase: 1 of 3 (Core Implementation)
Plan: 1 of 3 in current phase
Status: In progress
Last activity: 2026-01-29 — Completed 01-01-PLAN.md (M3 likelihood functions)

Progress: [███░░░░░░░] 33% (Phase 1: 1/3 plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 1
- Average duration: 23 min
- Total execution time: 0.4 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-core-implementation | 1 | 23 min | 23 min |

**Recent Trend:**
- Last 5 plans: 01-01 (23min)
- Trend: First plan complete

*Updated after each plan completion*

## Accumulated Context

### Model Naming

- M1: Q-learning (α₊, α₋, ε) — existing
- M2: WM-RL hybrid (α₊, α₋, φ, ρ, K, ε) — existing
- M3: WM-RL + κ perseveration (α₊, α₋, φ, ρ, K, ε, κ) ← **this project**

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Name: M3 (not wmrl_kappa) — Follows M1/M2 naming convention
- Extend WMRLHybridAgent with optional kappa parameter — Avoid code duplication
- Global (not stimulus-specific) perseveration — Captures motor-level response stickiness
- κ ∈ [0, 1] bounds — Matches Senta et al. parameter constraint convention
- Reset last_action at block boundaries — Matches existing Q/WM reset pattern
- Infrastructure only — User runs fits on cluster

**From 01-01 (M3 Likelihoods):**
- last_action=-1 sentinel at block start — Indicates no previous action
- Rep(a) via one-hot encoding — jnp.eye(num_actions)[last_action]
- Perseveration in value space — Added before softmax, not to probabilities
- Carry extends M2 pattern — Minimal modification (4-tuple → 5-tuple)

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-01-29 (plan execution)
Stopped at: Completed 01-01-PLAN.md (M3 likelihood functions)
Resume file: None

**Next:** 01-02-PLAN.md (WMRLHybridAgent kappa parameter)
