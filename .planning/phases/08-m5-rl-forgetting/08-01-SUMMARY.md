---
phase: 08-m5-rl-forgetting
plan: 01
subsystem: modelling
tags: [jax, mle, fitting, wmrl, rl-forgetting, q-learning, parameter-estimation]

# Dependency graph
requires:
  - phase: 04-mle-fitting
    provides: jax_likelihoods.py, mle_utils.py, fit_mle.py with M3 pattern to extend
provides:
  - wmrl_m5_block_likelihood with phi_rl Q-decay BEFORE delta-rule
  - wmrl_m5_multiblock_likelihood, wmrl_m5_multiblock_likelihood_stacked
  - WMRL_M5_BOUNDS, WMRL_M5_PARAMS, M5 parameter transforms
  - CLI: python scripts/12_fit_mle.py --model wmrl_m5
  - output/wmrl_m5_individual_fits.csv with 8 parameter columns
affects: [09-m6a-stim-specific, 11-m4-lba, compare_mle_models, 14-compare-models]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "M5 extends M3 by adding phi_rl decay step BEFORE delta-rule (not after)"
    - "Q decay target is Q0=1/nA=0.333 (not q_init=0.5) — matches WM baseline convention"
    - "phi_rl=0 algebraic identity reduces M5 exactly to M3 (no conditional branch needed)"
    - "M5 parameter order: alpha_pos, alpha_neg, phi, rho, capacity, kappa, phi_rl, epsilon (index 6, 7)"

key-files:
  created: []
  modified:
    - scripts/fitting/jax_likelihoods.py
    - scripts/fitting/mle_utils.py
    - scripts/fitting/fit_mle.py

key-decisions:
  - "phi_rl decay target is Q0=1/nA=0.333, not q_init=0.5 — consistent with WM baseline"
  - "phi_rl=0 backward compatibility with M3 achieved via algebraic identity (no jnp.where branch)"
  - "phi_rl at index 6 (before epsilon at index 7) per WMRL_M5_PARAMS order"
  - "phi_rl default starting value 0.1 matching phi default per plan"

patterns-established:
  - "Model extension pattern: copy M3 dispatch blocks, add new parameter as penultimate arg before epsilon"
  - "Output path for quick validation: --output output (not default output/mle/)"

# Metrics
duration: 22min
completed: 2026-04-02
---

# Phase 8 Plan 1: M5 RL Forgetting Model Summary

**WM-RL M5 model with per-trial Q-value decay (phi_rl) toward Q0=1/nA before delta-rule, extending M3 with backward-compatible 8-parameter fitting pipeline**

## Performance

- **Duration:** 22 min
- **Started:** 2026-04-02T18:22:25Z
- **Completed:** 2026-04-02T18:44:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Implemented wmrl_m5_block_likelihood with phi_rl Q-decay step inserted between WM decay and policy computation; Q_decayed = (1-phi_rl)*Q_table + phi_rl*Q0 where Q0=1/nA
- Verified phi_rl=0 reduces exactly to M3: backward compatibility difference = 0.00e+00
- Full MLE fitting pipeline: 46 participants fit, 45/46 converged (98%), phi_rl mean=0.076 +/- 0.007

## Task Commits

1. **Task 1: M5 JAX likelihood + parameter registration** - `b26280c` (feat)
2. **Task 2: M5 fitting pipeline (fit_mle.py CLI integration)** - `181ca56` (feat)

## Files Created/Modified

- `scripts/fitting/jax_likelihoods.py` — Added wmrl_m5_block_likelihood, wmrl_m5_multiblock_likelihood, wmrl_m5_multiblock_likelihood_stacked; inline tests (smoke, backward compat, padding)
- `scripts/fitting/mle_utils.py` — Added WMRL_M5_BOUNDS, WMRL_M5_PARAMS, jax_unconstrained_to_params_wmrl_m5, jax_bounded_to_unconstrained_wmrl_m5; wmrl_m5 elif in all dispatch functions
- `scripts/fitting/fit_mle.py` — Added M5 imports, objective functions (bounded/jax/gpu), wmrl_m5 in all dispatch blocks, argparse choice

## Decisions Made

- phi_rl decay target Q0=1/nA=0.333, not q_init=0.5 — consistent with WM baseline (wm_init=1/nA)
- Used algebraic identity (no conditional branch) for phi_rl=0 backward compatibility — cleaner JAX trace
- phi_rl placed at parameter index 6 (penultimate before epsilon) to minimize signature disruption from M3

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- M5 fitting pipeline complete and validated end-to-end
- phi_rl=0 backward compatibility with M3 verified (exact match)
- Output files ready: output/wmrl_m5_individual_fits.csv (8 param cols + nll/aic/bic)
- Phase 9 (M6a stimulus-specific perseveration) can follow the same extension pattern

---
*Phase: 08-m5-rl-forgetting*
*Completed: 2026-04-02*
