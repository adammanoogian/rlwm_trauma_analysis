# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-29)

**Core value:** Correctly dissociate perseverative responding from learning-rate effects (α₋) to accurately identify whether post-reversal failures reflect motor perseveration or outcome insensitivity
**Current focus:** Phase 2 - MLE Infrastructure

## Current Position

Phase: 2 of 3 (MLE Infrastructure)
Plan: 2 of TBD in current phase
Status: In progress
Last activity: 2026-01-29 — Completed 02-02-PLAN.md

Progress: [████░░░░░░] 44%

## Performance Metrics

**Velocity:**
- Total plans completed: 4
- Average duration: 23 min
- Total execution time: 1.5 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 2 | 59 min | 30 min |
| 2 | 2 | 35 min | 18 min |

**Recent Trend:**
- Last 5 plans: 01-01 (23 min), 01-02 (36 min), 02-01 (15 min), 02-02 (20 min)
- Trend: Phase 2 maintaining high velocity

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
- kappa bounds (0.0, 1.0) allow M2 equivalence — 02-01: Enable testing that kappa=0 produces M2 results
- Parameter ordering matches likelihood signature — 02-01: Prevent parameter misalignment bugs
- Default kappa=0.0 provides M2 baseline — 02-01: M2 behavior by default
- Extended if/else to if/elif/else pattern — 02-02: Proper handling of three models with explicit error for unknown models
- n_params=7 for wmrl_m3 model — 02-02: Reflects addition of kappa parameter
- CLI help text shows M1/M2/M3 naming — 02-02: Consistency with project convention

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

### Phase 2 Progress

**Plan 02-01: MLE Parameter Infrastructure (Complete 2026-01-29)**

Key additions:
- WMRL_M3_BOUNDS constant with kappa: (0.0, 1.0) at scripts/fitting/mle_utils.py:35
- WMRL_M3_PARAMS list with 7-parameter ordering at scripts/fitting/mle_utils.py:50
- Extended 7 utility functions to support wmrl_m3 model type
- Verified parameter ordering matches wmrl_m3_multiblock_likelihood signature

**Plan 02-02: MLE CLI Integration (Complete 2026-01-29)**

Key additions:
- fit_mle.py accepts --model wmrl_m3 with CLI integration complete
- _objective_wmrl_m3() function for 7-parameter negative log-likelihood
- Model dispatch extended across fit_participant_mle(), prepare_participant_data(), fit_all_participants()
- CLI help text shows M1/M2/M3 naming convention

### Pending Todos

None yet.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-01-29 (Phase 2 execution)
Stopped at: Completed 02-02-PLAN.md (MLE CLI Integration)
Resume file: None
