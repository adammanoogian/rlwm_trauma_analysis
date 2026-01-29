# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-29)

**Core value:** Correctly dissociate perseverative responding from learning-rate effects (α₋) to accurately identify whether post-reversal failures reflect motor perseveration or outcome insensitivity
**Current focus:** Phase 1 - Core Implementation

## Current Position

Phase: 1 of 3 (Core Implementation)
Plan: 2 of 3 in current phase
Status: In progress
Last activity: 2026-01-29 — Completed 01-02-PLAN.md (M3 agent integration)

Progress: [██████░░░░] 67% (Phase 1: 2/3 plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 2
- Average duration: 30 min
- Total execution time: 1.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-core-implementation | 2 | 59 min | 30 min |

**Recent Trend:**
- Last 5 plans: 01-01 (23min), 01-02 (36min)
- Trend: Stable velocity (23-36min range)

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

**From 01-02 (M3 Agent):**
- Conditional execution path for backward compatibility — M2 mode (kappa=0) vs M3 mode (kappa>0)
- last_action=None in agent (not -1) — Python convention, reset() clears to None
- Perseveration in value space confirmed — V_hybrid + kappa*Rep(a) → softmax
- Block boundary reset pattern — reset() clears last_action like Q/WM matrices

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-01-29 (plan execution)
Stopped at: Completed 01-02-PLAN.md (M3 agent integration)
Resume file: None

**Next:** 01-03-PLAN.md (M3 fitting infrastructure)
