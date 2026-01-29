---
phase: 01-core-implementation
plan: 02
subsystem: models
tags: [python, numpy, wm-rl, perseveration, agent, m3]

# Dependency graph
requires:
  - phase: 01-core-implementation
    plan: 01
    provides: M3 likelihood functions with kappa parameter
provides:
  - WMRLHybridAgent with optional kappa perseveration parameter
  - M3 agent behavior (kappa > 0) compatible with M2 (kappa = 0)
  - Action tracking across trials within blocks
  - Test suite verifying perseveration effects
affects: [01-03-fitting, simulation-scripts]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Optional kappa parameter with default=0 for backward compatibility"
    - "Conditional execution path: M2 mode (kappa=0) vs M3 mode (kappa>0)"
    - "Perseveration bonus in value space before softmax"

key-files:
  created: []
  modified:
    - models/wm_rl_hybrid.py

key-decisions:
  - "Use conditional path in get_hybrid_probs() for exact M2 backward compatibility"
  - "Apply perseveration bonus in value space (not probability space)"
  - "Track last_action globally (not per-stimulus)"

patterns-established:
  - "Block boundary: reset() clears last_action to None"
  - "Within-block: update() sets last_action after each trial"
  - "Perseveration formula: kappa * Rep(a) where Rep(a) = 1 if a==last_action else 0"

# Metrics
duration: 36min
completed: 2026-01-29
---

# Phase 01 Plan 02: Agent Integration Summary

**WMRLHybridAgent extended with optional kappa perseveration parameter (M3 model) while maintaining exact M2 backward compatibility (kappa=0)**

## Performance

- **Duration:** 36 min
- **Started:** 2026-01-29T12:16:05Z
- **Completed:** 2026-01-29T12:52:16Z
- **Tasks:** 6
- **Files modified:** 1

## Accomplishments
- Added kappa parameter to WMRLHybridAgent with default=0 (M2 behavior)
- Implemented last_action tracking across trials with block boundary reset
- Extended get_hybrid_probs() to include kappa*Rep(a) perseveration bonus
- Created M3 agent test suite verifying perseveration effects
- Fixed pre-existing bugs (BETA_WM_DEFAULT missing, Unicode characters)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add kappa parameter and last_action tracking** - `f8ed7ec` (feat)
2. **Task 2: Update reset() to clear last_action** - `263843e` (feat)
3. **Task 3: Update get_hybrid_probs() to include perseveration** - `1850999` (feat)
4. **Task 4: Update update() to track last_action** - `8f932b1` (feat)
5. **Task 5: Update parameter getter/setter and factory** - `fc81239` (feat)
6. **Task 6: Add M3 test to verify perseveration** - `d144c24` (feat)

_Note: No plan metadata commit - autonomous execution_

## Files Created/Modified
- `models/wm_rl_hybrid.py` - Extended WMRLHybridAgent class with M3 perseveration support

## Decisions Made

**Conditional execution path for backward compatibility**
- Rationale: Ensure kappa=0 produces EXACTLY the same numerical results as original M2
- Implementation: M2 path (weighted probability average) vs M3 path (value space + softmax)
- Benefit: Can directly compare M2/M3 fits without numerical drift

**Perseveration in value space**
- Rationale: Perseveration operates in log-probability space (additive bonus)
- Implementation: V_hybrid + kappa*Rep(a) → softmax
- Avoids: Incorrect probability space addition which would violate softmax properties

**Global last_action tracking**
- Rationale: Captures motor-level response stickiness (not stimulus-specific memory)
- Implementation: Single self.last_action attribute reset at block boundaries
- Matches: Senta et al. 2025 M3 specification

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed missing BETA_WM_DEFAULT constant**
- **Found during:** Task 1 (parameter initialization)
- **Issue:** ModelParams.BETA_WM_DEFAULT does not exist in config.py, causing AttributeError
- **Fix:** Changed to ModelParams.BETA_DEFAULT (which exists) in both __init__ and create_wm_rl_agent()
- **Files modified:** models/wm_rl_hybrid.py (2 occurrences)
- **Verification:** File imports successfully, tests run
- **Committed in:** f8ed7ec (Task 1 commit)

**2. [Rule 1 - Bug] Fixed Unicode encoding errors in Windows console**
- **Found during:** Task 6 (running tests)
- **Issue:** Greek characters (ω, ρ) cause UnicodeEncodeError on Windows console (cp1252 codec)
- **Fix:** Replaced Greek letters with ASCII equivalents (omega, rho) in print statements
- **Files modified:** models/wm_rl_hybrid.py (2 print statements)
- **Verification:** Tests run to completion without encoding errors
- **Committed in:** d144c24 (Task 6 commit)

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both bugs blocked execution. First bug was pre-existing (file never ran). Second bug is Windows-specific console limitation. No scope creep.

## Issues Encountered

**Pre-existing broken test**
- The original file had BETA_WM_DEFAULT bug preventing it from running
- Fixed as part of Task 1 (auto-fix Rule 1)
- No delays, handled inline

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for 01-03 (M3 Fitting)**
- Agent can be instantiated with kappa parameter
- get_parameters() includes kappa for fitting extraction
- set_parameters() accepts kappa for optimizer updates
- Test suite confirms M3 behavior correct

**Ready for simulation scripts**
- create_wm_rl_agent() factory accepts kappa argument
- Backward compatible: kappa=0 produces identical M2 results
- Perseveration tracked automatically via update()

**Verification passed:**
- M3 with kappa=0.3 and last_action=1 gives 47.7% to action 1 (vs 33.3% baseline)
- M3 with no last_action matches M2 exactly (0.000000 difference)
- reset() correctly clears last_action (block boundary)

---
*Phase: 01-core-implementation*
*Completed: 2026-01-29*
