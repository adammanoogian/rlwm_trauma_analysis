---
phase: 19-associative-scan-likelihood
plan: 01
subsystem: fitting
tags: [jax, associative-scan, parallel, q-learning, wm-rl, ar1, likelihood]

# Dependency graph
requires:
  - phase: 16-choice-only-family-extension
    provides: complete sequential lax.scan likelihood implementations for all 6 models
  - phase: 18-integration-comparison-manuscript
    provides: model comparison results establishing which models are scientifically credible
provides:
  - affine_scan: generic O(log T) AR(1) parallel prefix scan primitive
  - associative_scan_q_update: (T,S,A) Q-value trajectory via parallel scan
  - associative_scan_wm_update: (T,S,A) WM decay+overwrite trajectory via parallel scan
  - docs/PARALLEL_SCAN_LIKELIHOOD.md: implementation guide covering AR(1) proof, WM encoding, linear/non-linear decomposition, related work, alpha approximation, numerical stability
  - 8 unit tests covering all three primitives against sequential references
affects:
  - 19-02: future plan implementing per-model pscan likelihood variants (will build on these primitives)
  - 20-deer: DEER non-linear scan (Phase 20 for softmax/policy components)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - AR(1) affine operator (a_r*a_l, a_r*b_l+b_r) for parallel prefix scan of any linear recurrence
    - Multiplicative reset (a=0, b=r) to encode hard overwrite within associative scan
    - Post-scan vectorized wm_for_policy = (1-phi)*carry_in + phi*wm_init without second scan pass
    - Reward-based alpha approximation for data-dependent learning rate in parallel scan

key-files:
  created:
    - docs/PARALLEL_SCAN_LIKELIHOOD.md
    - scripts/fitting/tests/test_pscan_likelihoods.py
  modified:
    - scripts/fitting/jax_likelihoods.py

key-decisions:
  - "Single-pass WM scan: decay+overwrite combined in one scan (not two passes). Post-scan vectorized recovery of wm_for_policy."
  - "Padding trials use decay coefficients (not identity), matching sequential model where decay happens for all trials."
  - "Alpha approximation: reward-based (r==1 -> alpha_pos) not delta-sign rule. Enables data-independent coefficient arrays."
  - "wm_after_update[t] = WM after overwrite at trial t (= WM_all[t] directly, no prepend/drop)."

patterns-established:
  - "Parallel scan functions named associative_scan_{component}: placed after apply_epsilon_noise, before q_learning_step"
  - "Sequential references for tests implemented inline in test file (not using full block_likelihood)"
  - "Tolerance thresholds: < 1e-5 typical, < 1e-3 extreme alpha"

# Metrics
duration: 38min
completed: 2026-04-14
---

# Phase 19 Plan 01: Parallel Scan Primitives Summary

**AR(1) associative scan primitives for RLWM likelihood parallelization: affine_scan, Q-update, and WM decay+overwrite scans verified to < 1e-5 relative error against sequential reference**

## Performance

- **Duration:** 38 min
- **Started:** 2026-04-14T06:20:44Z
- **Completed:** 2026-04-14T06:58:00Z
- **Tasks:** 2
- **Files modified:** 3 (1 created, 2 new)

## Accomplishments
- Implementation guide covers AR(1) linear recurrence proof, WM hard-overwrite as multiplicative reset, linear vs. non-linear RLWM decomposition table, four related-work paragraphs (PaMoRL/S4/Kalman/DEER), alpha approximation accuracy analysis, and float32 numerical stability
- Three primitives added to jax_likelihoods.py: `affine_scan` (generic O(log T)), `associative_scan_q_update` (reward-based alpha approximation), `associative_scan_wm_update` (single-pass combined decay+overwrite)
- 8 unit tests: AR(1) basic/identity/reset, Q-update typical and extreme-alpha, WM decay-only, overwrite spot-check, and 1000-trial full-agreement test — all pass

## Task Commits

1. **Task 1: Create docs/PARALLEL_SCAN_LIKELIHOOD.md** - `f03fd67` (docs)
2. **Task 2: Implement pscan primitives and tests** - `8cc3157` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `docs/PARALLEL_SCAN_LIKELIHOOD.md` - Implementation guide: AR(1) formulation, WM overwrite encoding, linearity table, related work, alpha approximation, numerical stability
- `scripts/fitting/jax_likelihoods.py` - Added affine_scan, associative_scan_q_update, associative_scan_wm_update after apply_epsilon_noise (~line 282)
- `scripts/fitting/tests/test_pscan_likelihoods.py` - 8 unit tests for all three primitives

## Decisions Made

**Single-pass WM scan (instead of two passes):** Initially planned as two separate scans (Pass 1: decay only for policy; Pass 2: decay+overwrite for state). Revised to a single scan that combines decay+overwrite, with `wm_for_policy` recovered via post-scan vectorized computation `(1-phi)*carry_in + phi*wm_init`. This is correct because padding trials use decay (not identity), matching the sequential model.

**Padding trials decay (not identity):** The plan spec suggested identity coefficients for padding trials. Confirmed via code inspection that `wmrl_m3_block_likelihood` applies decay on all trials ("Decay happens for all trials (valid or not)"); only the overwrite is gated by the mask. Using identity would be incorrect.

**wm_after_update indexing:** The plan spec says `wm_after_update[t]` = WM after overwrite at trial t. This is `WM_all[t]` directly (no prepend/drop), not the carry-entering-t indexing used for Q_for_policy.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected wm_for_policy implementation from two-pass to single-pass**
- **Found during:** Task 2 (implementing associative_scan_wm_update, test_wm_update_with_overwrite)
- **Issue:** Initial two-pass design computed wm_for_policy via pure-decay scan assuming no overwrites, which gave wrong values at trials following an overwrite. Tests failed with rel_error ~2.33 (3.33e7 for wm_after).
- **Fix:** Single-pass scan (decay+overwrite), wm_for_policy = (1-phi)*carry_in + phi*wm_init post-scan.
- **Files modified:** scripts/fitting/jax_likelihoods.py
- **Verification:** All 8 tests pass, wm trace matches sequential to < 1e-5 on 1000-trial block.
- **Committed in:** 8cc3157

**2. [Rule 1 - Bug] Corrected padding trial coefficients from identity to decay**
- **Found during:** Task 2 (debug of wm_for_policy test failure)
- **Issue:** Plan spec suggested identity (a=1, b=0) for padding trials. Sequential code applies decay on all trials — padding means no overwrite, not no decay.
- **Fix:** Removed padding identity override; base decay coefficients already correct for padding.
- **Files modified:** scripts/fitting/jax_likelihoods.py
- **Verification:** wm_after_update[t=3] = 0.8 (decay from 1.0) confirmed correct after fix.
- **Committed in:** 8cc3157

---

**Total deviations:** 2 auto-fixed (both Rule 1 - Bug)
**Impact on plan:** Both fixes required for mathematical correctness. No scope creep.

## Issues Encountered
- `_sequential_wm_decay_only` helper initially returned pre-decay WM (appended before decay). Fixed to return post-decay WM to match actual model semantics (decay first, then use for policy).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `affine_scan`, `associative_scan_q_update`, `associative_scan_wm_update` ready for use in per-model pscan likelihood variants (Plan 19-02)
- Sequential implementations untouched — no risk to Phase 13-18 fits
- Tolerance thresholds documented: < 1e-5 typical, < 1e-3 extreme alpha
- One open question for 19-02: the alpha approximation accuracy for the reward-based rule affects only the Q-update scan; WM overwrite does not have an approximation (it is exact)

---
*Phase: 19-associative-scan-likelihood*
*Completed: 2026-04-14*
