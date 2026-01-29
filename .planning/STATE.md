# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-29)

**Core value:** Correctly dissociate perseverative responding from learning-rate effects (α₋) to accurately identify whether post-reversal failures reflect motor perseveration or outcome insensitivity
**Current focus:** Phase 2 - MLE Infrastructure

## Current Position

Phase: 2 of 3 (MLE Infrastructure)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-01-29 — Phase 1 complete (Core Implementation)

Progress: [███░░░░░░░] 33%

## Performance Metrics

**Velocity:**
- Total plans completed: 2
- Average duration: 30 min
- Total execution time: 1.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 2 | 59 min | 30 min |

**Recent Trend:**
- Last 5 plans: 01-01 (23 min), 01-02 (36 min)
- Trend: First phase complete

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

### Phase 1 Summary (Complete)

**Completed 2026-01-29**

Key implementations:
- `wmrl_m3_block_likelihood()` at scripts/fitting/jax_likelihoods.py:666
- `wmrl_m3_multiblock_likelihood()` at scripts/fitting/jax_likelihoods.py:947
- WMRLHybridAgent extended with kappa parameter at models/wm_rl_hybrid.py:81
- Backward compatibility verified: kappa=0 produces identical M2 results

Technical decisions:
- last_action=-1 sentinel in JAX (block start indicator)
- last_action=None in agent class (Python convention)
- Perseveration added in value space before softmax
- Rep(a) via one-hot encoding: jnp.eye(num_actions)[last_action]

### Pending Todos

None yet.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-01-29 (Phase 1 execution)
Stopped at: Phase 1 complete, ready for Phase 2 planning
Resume file: None
