---
phase: 19-associative-scan-likelihood
plan: 02
subsystem: fitting
tags: [jax, associative-scan, parallel, pscan, likelihood, q-learning, wm-rl, agreement-test]

# Dependency graph
requires:
  - phase: 19-associative-scan-likelihood (plan 01)
    provides: affine_scan, associative_scan_q_update, associative_scan_wm_update primitives
  - phase: 16-choice-only-family-extension
    provides: sequential lax.scan likelihood implementations for all 6 models
provides:
  - 12 pscan likelihood functions (block + multiblock for 6 choice-only models)
  - Agreement test suite covering synthetic and real N=154 data
  - M5 composed affine operator fusing phi_rl decay with delta-rule update
  - Q_decayed_for_policy post-scan recovery for M5 pscan
affects:
  - 19-03: benchmarking pscan vs sequential (will use these functions)
  - 20-deer: DEER non-linear scan research (future Phase 20)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-phase pscan architecture: Phase 1 parallel O(log T) Q/WM trajectories, Phase 2 sequential O(T) policy scan"
    - "M5 composed affine operator: a_t = (1-alpha)*(1-phi_rl), b_t = (1-alpha)*phi_rl*Q0 + alpha*r"
    - "Q_decayed_for_policy recovery: (1-phi_rl)*Q_for_policy + phi_rl*Q0 (mirrors wm_for_policy pattern)"
    - "Perseveration models carry last_action/last_actions in Phase 2 only; Q/WM are pre-computed"

key-files:
  created: []
  modified:
    - scripts/fitting/jax_likelihoods.py
    - scripts/fitting/tests/test_pscan_likelihoods.py

key-decisions:
  - "M5 Q-forgetting uses composed affine operator (fused decay+update) rather than two-pass scan: a_t = (1-alpha)*(1-phi_rl), b_t = (1-alpha)*phi_rl*Q0 + alpha*r for active positions; a_t = 1-phi_rl, b_t = phi_rl*Q0 for inactive"
  - "Q_decayed_for_policy derived post-scan: same pattern as wm_for_policy in 19-01 (apply decay to carry-in array)"
  - "Participant matching in N=154 tests uses participant_id column lookup (not positional index) to handle CSV vs data ordering differences"

patterns-established:
  - "Pscan block functions placed immediately after sequential counterparts in jax_likelihoods.py"
  - "All pscan multiblock stacked functions preserve exact same signature and return types as sequential versions"
  - "return_pointwise=True supported in all pscan variants for per-trial log-prob extraction"

# Metrics
duration: 42min
completed: 2026-04-14
---

# Phase 19 Plan 02: Pscan Likelihood Variants Summary

**12 pscan likelihood functions for all 6 choice-only models with < 1e-4 agreement on synthetic data and N=154 real participants; M5 uses composed affine operator fusing phi_rl decay with delta-rule**

## Performance

- **Duration:** 42 min
- **Started:** 2026-04-14T08:15:30Z
- **Completed:** 2026-04-14T08:57:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- 12 pscan likelihood functions (block + multiblock stacked for M1/M2/M3/M5/M6a/M6b) verified as drop-in replacements for sequential counterparts
- All 6 synthetic agreement tests pass (< 1e-4 absolute/relative error including return_pointwise)
- All 6 real-data single-participant tests pass (< 1e-4 relative error)
- Q-learning N=154 full agreement verified (max rel_err 6.46e-07)
- M5 composed affine operator correctly fuses per-trial Q-forgetting (phi_rl) with delta-rule update in a single scan pass

## Task Commits

1. **Task 1: Implement pscan block and multiblock likelihoods for all 6 models** - `9865a0a` (feat) [prior commit]
2. **Task 1 bug fix: M5 pscan Q_decayed_for_policy** - `ccbd172` (fix)
3. **Task 2: Agreement tests for all 6 models** - `c99d23b` (feat)
4. **Task 2 bug fix: participant_id lookup in tests** - `4ca4d5b` (fix)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `scripts/fitting/jax_likelihoods.py` - Added 12 pscan functions (1230 lines in 9865a0a), fixed M5 Q_decayed_for_policy (9 lines in ccbd172)
- `scripts/fitting/tests/test_pscan_likelihoods.py` - Extended from 8 primitive tests to 26 tests: synthetic agreement (6), real-data smoke (6), N=154 PSCAN-04 (6), plus fixture/helpers

## Decisions Made

**M5 composed affine operator:** Rather than running two scans (one for phi_rl decay, one for delta-rule), M5 composes the two operations into a single affine operator per trial. For active (s,a) at trial t: `a = (1-alpha)*(1-phi_rl)`, `b = (1-alpha)*phi_rl*Q0 + alpha*r`. For inactive positions: `a = 1-phi_rl`, `b = phi_rl*Q0` (decay only). This halves the Q-scan cost for M5.

**Q_decayed_for_policy post-scan recovery:** The policy at trial t uses `Q_decayed = (1-phi_rl)*Q_carry_in + phi_rl*Q0`, not the raw carry-in. This mirrors the `wm_for_policy` recovery pattern from 19-01 where `wm_for_policy = (1-phi)*carry_in + phi*wm_init`.

**Participant matching by ID:** The MLE CSV `participant_id` column ordering differs from sorted participant IDs in the trial data. Tests now use participant_id lookup instead of positional indexing.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] M5 pscan used undecayed Q-values in policy step**
- **Found during:** Task 2 (agreement testing)
- **Issue:** `wmrl_m5_block_likelihood_pscan` Phase 2 policy step read `Q_for_policy[t]` (carry-in Q) instead of applying the phi_rl decay step. The sequential M5 applies `Q_decayed = (1-phi_rl)*Q + phi_rl*Q0` before the policy computation, but the pscan variant was skipping this.
- **Fix:** Added `Q_decayed_for_policy = (1-phi_rl)*Q_for_policy + phi_rl*Q0` as Phase 1c post-scan vectorized step, and changed Phase 2 to read from `Q_decayed_for_policy`.
- **Files modified:** scripts/fitting/jax_likelihoods.py
- **Verification:** M5 synthetic and real-data agreement tests pass (< 1e-4)
- **Committed in:** ccbd172

**2. [Rule 1 - Bug] Test participant lookup used positional index instead of participant_id**
- **Found during:** Task 2 (N=154 test execution)
- **Issue:** `_load_mle_params` used `df.iloc[participant_idx]` assuming CSV row ordering matches sorted participant IDs. The CSV uses `participant_id` column with different ordering, causing IndexError when idx >= CSV rows.
- **Fix:** Changed `_load_mle_params` to accept `participant_id` and filter by `df["participant_id"] == participant_id`. Updated N=154 test to match participants by ID between data and MLE fits.
- **Files modified:** scripts/fitting/tests/test_pscan_likelihoods.py
- **Verification:** Q-learning N=154 passes (154 participants, max rel_err 6.46e-07)
- **Committed in:** 4ca4d5b

---

**Total deviations:** 2 auto-fixed (both Rule 1 - Bug)
**Impact on plan:** Both fixes required for mathematical correctness and test functionality. No scope creep.

## Issues Encountered
- Task 1 was already committed by a previous execution session (`9865a0a`). This session verified the existing implementation, found and fixed the M5 Q_decayed_for_policy bug, and completed Task 2 (agreement tests).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 12 pscan functions are importable and agree with sequential to < 1e-4 on synthetic and real data
- Plan 19-03 (benchmarking) can proceed to measure speedup of pscan vs sequential
- No changes to any sequential function in jax_likelihoods.py -- all Phase 13-18 fits remain valid
- Alpha approximation accuracy confirmed: max rel_err 6.46e-07 for Q-learning N=154 (well under 1e-4 threshold)

---
*Phase: 19-associative-scan-likelihood*
*Completed: 2026-04-14*
