# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-29)

**Core value:** Correctly dissociate perseverative responding from learning-rate effects (α₋) to accurately identify whether post-reversal failures reflect motor perseveration or outcome insensitivity
**Current focus:** Phase 3 - Validation & Comparison

## Current Position

Phase: 3 of 3 (Validation & Comparison)
Plan: 2 of TBD in current phase
Status: In progress
Last activity: 2026-01-30 — Completed 03-01-PLAN.md

Progress: [████████░░] 83%

## Performance Metrics

**Velocity:**
- Total plans completed: 6
- Average duration: 24 min
- Total execution time: 2.4 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 2 | 59 min | 30 min |
| 2 | 2 | 35 min | 18 min |
| 3 | 2 | 52 min | 26 min |

**Recent Trend:**
- Last 5 plans: 02-01 (15 min), 02-02 (20 min), 03-01 (35 min), 03-02 (17 min)
- Trend: 03-01 took longer due to critical bug discovery and fix

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
- M3 likelihood branches on kappa=0 for backward compatibility — 03-01: Use M2 probability mixing when kappa=0, M3 value mixing+perseveration when kappa>0
- Tests use strict rtol=1e-5 for numerical equivalence — 03-01: Not just similarity, but true backward compatibility
- Dict-based model naming for comparison — 03-02: Enables flexible N-model comparison without hardcoding
- Keep legacy CLI args in compare_mle_models.py — 03-02: Backward compatibility, don't break existing workflows

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

### Phase 2 Summary (Complete)

**Completed 2026-01-29**

Key implementations:
- WMRL_M3_BOUNDS constant with kappa: (0.0, 1.0) at scripts/fitting/mle_utils.py:41
- WMRL_M3_PARAMS list with 7-parameter ordering at scripts/fitting/mle_utils.py:50
- Extended 7 utility functions in mle_utils.py for wmrl_m3 model type
- _objective_wmrl_m3() function at scripts/fitting/fit_mle.py:157
- CLI accepts --model wmrl_m3 at scripts/fitting/fit_mle.py:601
- Model dispatch extended across all fitting functions

Technical decisions:
- kappa bounds (0.0, 1.0) allow M2 equivalence at kappa=0
- Parameter ordering matches likelihood signature exactly
- Default kappa=0.0 provides M2 baseline behavior
- Extended if/else to if/elif/else pattern for three-model dispatch

### Phase 3 Progress

**Plan 03-01 Complete (2026-01-30)**

Key implementations:
- `validation/test_m3_backward_compat.py` - 24 comprehensive backward compatibility tests
- Bug fix in `wmrl_m3_block_likelihood()` - branching logic for M2 compatibility when kappa=0

Technical decisions:
- M3 must branch on kappa=0: Use M2 probability mixing (backward compat), not M3 value mixing
- Agent class already had this branching, but JAX likelihood didn't - bug fixed
- Tests validated M3(kappa=0) matches M2 to rtol=1e-5 across multiple seeds, parameters, block counts

Critical bug discovered and fixed:
- M3 was always using value mixing: v = ω·WM + (1-ω)·Q, then softmax
- M2 uses probability mixing: p = ω·softmax(WM) + (1-ω)·softmax(Q)
- These are fundamentally different - M3(kappa=0) was NOT matching M2
- Fix: Added branching to match agent class implementation
- All 24 tests now pass with strict tolerance

**Plan 03-02 Complete (2026-01-30)**

Key implementations:
- `compute_akaike_weights_n()` at scripts/fitting/compare_mle_models.py:144
- `compare_models()` at scripts/fitting/compare_mle_models.py:168
- `count_participant_wins_n()` at scripts/fitting/compare_mle_models.py:208
- CLI accepts --m1, --m2, --m3 for 3-way comparison
- Legacy --qlearning and --wmrl preserved for backward compatibility
- M3 kappa parameter summary in reporting

Technical decisions:
- Dict-based model naming for flexible N-model comparison
- Generalized comparison functions handle any number of models
- M3 kappa parameter reported separately to highlight perseveration extension

### Pending Todos

None yet.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-01-30 (Phase 3 execution)
Stopped at: Completed 03-01-PLAN.md with critical bug fix
Resume file: None
