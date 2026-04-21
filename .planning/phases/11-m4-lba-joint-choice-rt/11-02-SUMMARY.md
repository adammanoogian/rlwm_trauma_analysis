---
phase: 11-m4-lba-joint-choice-rt
plan: "02"
subsystem: fitting
tags: [jax, lba, float64, mle, joint-choice-rt, lax-scan, mle-utils, fit-mle, b-A-delta-reparameterization]

# Dependency graph
requires:
  - phase: 11-m4-lba-joint-choice-rt
    plan: "01"
    provides: lba_pdf, lba_cdf, lba_sf, lba_joint_log_lik, preprocess_rt_block -- the building blocks for M4 likelihood
provides:
  - "wmrl_m4_block_likelihood: M3 learning dynamics + LBA decision step, no epsilon, in lax.scan"
  - "wmrl_m4_multiblock_likelihood: loop-based multiblock version"
  - "wmrl_m4_multiblock_likelihood_stacked: fori_loop stacked version for JIT/GPU"
  - "WMRL_M4_BOUNDS (10 params), WMRL_M4_PARAMS, jax_unconstrained_to_params_wmrl_m4, jax_bounded_to_unconstrained_wmrl_m4"
  - "All mle_utils dispatch functions extended with wmrl_m4 elif branches"
  - "Three fit_mle.py objective functions: bounded, jax, gpu (with 7-arg data signature for M4)"
  - "prepare_participant_data RT extraction: rt column -> ms to seconds, outlier filter, combined mask"
  - "fit_all_gpu: all 4 dispatch points handle wmrl_m4 with separate vmap (7 data dims)"
  - "Full end-to-end: python scripts/12_fit_mle.py --model wmrl_m4 writes valid CSV"
affects:
  - 11-03 (mle_utils downstream utils -- already complete in this plan)
  - 11-04 (model_recovery RT simulation for wmrl_m4)
  - scripts/14_compare_models.py (M4 separate comparison track per STATE.md decision)
  - scripts/15_analyze_mle_by_trauma.py (param-trauma relationships for M4)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "M4 lazy float64 import: jax_enable_x64 only when model==wmrl_m4, never at module top-level"
    - "b > A reparameterization decode in objective functions (not in transform): b = A + delta"
    - "7-argument GPU vmap for M4 (separate _run_one branch in fit_all_gpu Stage 3)"
    - "RT preprocessing in prepare_participant_data: pad valid_rt to max_trials before combining with padding mask"
    - "M4 carry identical to M3: (Q, WM, WM_baseline, log_lik, last_action) -- only decision step changes"
    - "No epsilon in M4: pi_hybrid feeds directly into v_all = v_scale * pi_hybrid"

key-files:
  created: []
  modified:
    - scripts/fitting/lba_likelihood.py
    - scripts/fitting/mle_utils.py
    - scripts/fitting/fit_mle.py

key-decisions:
  - "M4 NLL returned directly from wmrl_m4_block_likelihood (sign convention: returns -log_lik_total so positive NLL for minimization, opposite to choice-only models that return log_lik)"
  - "valid_rt padding: pad valid_rt array to max_trials (not block size) before multiplying with padding mask to avoid shape mismatch"
  - "fit_participant_mle gets rts_blocks parameter (not **kwargs): cleaner, explicit API"
  - "wmrl_m4_multiblock_likelihood_stacked takes NLL directly (not negated log-lik); objectives call it without negation"

patterns-established:
  - "GPU vmap separation: models with extra data dimensions (M4+rts) need their own _run_one branch in fit_all_gpu"
  - "RT preprocessing placed in prepare_participant_data (not fit_participant_mle): consistent with how set_sizes are extracted"

# Metrics
duration: 12min
completed: 2026-04-03
---

# Phase 11 Plan 02: M4 Likelihood Functions + MLE Pipeline Summary

**WM-RL M4 (LBA joint choice+RT) end-to-end: lax.scan block likelihood with M3 learning dynamics and LBA decision, b=A+delta reparameterization, RT preprocessing pipeline, and full fit_mle.py CLI integration producing valid 10-parameter output CSV**

## Performance

- **Duration:** 12 min
- **Started:** 2026-04-03T13:47:12Z
- **Completed:** 2026-04-03T14:00:10Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Implemented `wmrl_m4_block_likelihood` in `lba_likelihood.py` using `lax.scan` with M3-identical carry (Q, WM, WM_baseline, log_lik, last_action) and LBA decision step replacing softmax
- Added `wmrl_m4_multiblock_likelihood` and `wmrl_m4_multiblock_likelihood_stacked` (fori_loop) for full pipeline use
- Extended all 9 `mle_utils.py` dispatch functions with `wmrl_m4` elif branches (bounds, params, transforms, get_n_params, sample_lhs_starts, check_at_bounds, summarize_all_parameters, Hessian diagnostics)
- Integrated M4 into `fit_mle.py` at all required dispatch points: 3 objective functions (bounded/jax/gpu), 4 fit_all_gpu stages, prepare_participant_data RT extraction, argparse choices, param_cols dispatch
- Verified: `python scripts/12_fit_mle.py --model wmrl_m4 --limit 2` completes, writes CSV with 10 parameter columns (no epsilon), wmrl_m3 and wmrl_m6b regressions unaffected

## Task Commits

Each task was committed atomically:

1. **Task 1: M4 likelihood functions + parameter registration** - `cf74c6f` (feat)
2. **Task 2: M4 fitting pipeline (fit_mle.py CLI integration)** - `c1d4242` (feat)

**Plan metadata:** (to be added in final commit)

## Files Created/Modified

- `scripts/fitting/lba_likelihood.py` - Added wmrl_m4_block_likelihood, wmrl_m4_multiblock_likelihood, wmrl_m4_multiblock_likelihood_stacked; 3 M4 smoke tests; updated __main__ block
- `scripts/fitting/mle_utils.py` - Added WMRL_M4_BOUNDS, WMRL_M4_PARAMS, jax_unconstrained_to_params_wmrl_m4, jax_bounded_to_unconstrained_wmrl_m4; extended all dispatch functions with wmrl_m4 elif
- `scripts/fitting/fit_mle.py` - Added lazy float64 import pattern, 3 M4 objective functions, RT extraction in prepare_participant_data, M4 support in all fit_all_gpu stages, argparse choices, param_cols dispatch

## Decisions Made

- **M4 NLL sign convention:** `wmrl_m4_block_likelihood` returns `-log_lik_total` (positive NLL for minimization). This differs from choice-only models (which return `log_lik_total` and objectives negate). The objective functions do NOT negate for M4. This is consistent because the LBA accumulates log-probabilities (which sum to negative values for reasonable data), so `-log_lik_total` is a positive NLL.
- **valid_rt padding shape mismatch fixed (Rule 1 - Bug):** Plan said `combined_mask = mask * valid_rt.astype(jnp.float32)` but `mask` is shape `(max_trials,)` and `valid_rt` is shape `(n_real_trials,)`. Fixed by padding `valid_rt` to `max_trials` with zeros before multiplying.
- **rts_blocks as explicit parameter:** Added `rts_blocks: list | None = None` to `fit_participant_mle` signature rather than using `**kwargs`. Cleaner API, consistent with `set_sizes_blocks` pattern.
- **wmrl_m4_multiblock_likelihood_stacked returns NLL directly:** The stacked likelihood function returns the accumulated NLL (positive), so objective functions call it without negation. Bounded and JAX objectives have `return nll` not `return -log_lik`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed valid_rt shape mismatch in prepare_participant_data**

- **Found during:** Task 2 (running `python scripts/12_fit_mle.py --model wmrl_m4 --limit 2`)
- **Issue:** Plan specified `combined_mask = mask * valid_rt.astype(jnp.float32)`. The `mask` from `pad_block_to_max` has shape `(max_trials=100,)`, but `valid_rt` from `preprocess_rt_block` has shape `(n_real_trials,)`. JAX broadcast multiply raised `TypeError: mul got incompatible shapes for broadcasting: (100,), (30,)`.
- **Fix:** Added padding step before combining: `valid_rt_padded = jnp.zeros(max_trials, dtype=jnp.float32); valid_rt_padded = valid_rt_padded.at[:n_real].set(valid_rt.astype(jnp.float32))` then `combined_mask = mask * valid_rt_padded`.
- **Files modified:** scripts/fitting/fit_mle.py
- **Verification:** `python scripts/12_fit_mle.py --model wmrl_m4 --limit 2` runs successfully
- **Committed in:** c1d4242 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — shape mismatch bug in plan specification)
**Impact on plan:** Critical fix for correctness. The fix adds 3 lines; zero scope change.

## Issues Encountered

None beyond the shape mismatch above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `wmrl_m4_block_likelihood`, `wmrl_m4_multiblock_likelihood_stacked` are ready for Plan 04 (model_recovery RT simulation)
- Plan 04 will need to add RT simulation (LBA inverse CDF: `t_i = (b - U*A) / v_i, winner = argmin`) to `model_recovery.py`
- `prepare_participant_data` RT extraction is complete; Plan 04 only needs to wire synthetic RTs through the recovery pipeline
- Full fit pipeline works end-to-end; Plan 03 (per STATE.md) was folded into this plan -- all mle_utils and fit_mle dispatch is done

---
*Phase: 11-m4-lba-joint-choice-rt*
*Completed: 2026-04-03*
