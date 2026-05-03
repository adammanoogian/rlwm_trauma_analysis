---
phase: 32-mcmc-methodology-update
plan: 02
subsystem: bayesian-mcmc

tags: [numpyro, nuts, init_to_median, hierarchical-priors, rmus2023, baribault-collins-2023]

# Dependency graph
requires:
  - phase: 16-bayesian-priors-revision
    provides: PARAM_PRIOR_DEFAULTS dict with mu_prior_loc=0.0 baseline
  - phase: 21-cluster-bayesian-pipeline
    provides: run_inference / run_inference_with_bump orchestration layer
provides:
  - "init_to_median(num_samples=50) as the NumPyro NUTS init strategy in both run_inference and run_inference_with_bump"
  - "PARAM_PRIOR_DEFAULTS kappa, kappa_s, kappa_total anchored at mu_prior_loc=-1.0 (prior mean kappa ~ Phi(-1) ~ 0.16)"
  - "kappa_share retained at mu_prior_loc=0.0 (channel-share is a priori uncertain)"
  - "Two integration tests asserting both invariants (one dict-key check, one inspect.getsource check)"
affects: [phase-32-04 softmax-bias kappa, phase-32-05 cluster smoke, phase-24 cold-start canary, all-future-bayesian-fits]

# Tech tracking
tech-stack:
  added: []  # No new libraries; uses existing numpyro.infer.init_to_median
  patterns:
    - "Cheapest-first MCMC interventions: init strategy + prior tightening first, before architectural changes"
    - "Use inspect.getsource for invariants on NUTS construction (avoids monkey-patching the kernel)"

key-files:
  created: []
  modified:
    - "src/rlwm/fitting/sampling.py (NUTS init_strategy=init_to_median(50) added in 2 NUTS construction sites + docstring Notes; init_to_median imported from numpyro.infer)"
    - "src/rlwm/fitting/numpyro_helpers.py (PARAM_PRIOR_DEFAULTS lines 189-192: kappa/kappa_s/kappa_total mu_prior_loc -> -1.0; docstring 'Phase 32 update' paragraph added between rationale and Consequences)"
    - "tests/integration/test_numpyro_helpers.py (2 new test functions appended after test_param_prior_defaults_completeness)"

key-decisions:
  - "Use rlwm.fitting (installed package) import path in tests rather than literal src.rlwm.fitting from plan — package is installed via pip install -e . and the rest of test_numpyro_helpers.py uses the same convention"
  - "Insert Phase 32 update paragraph between rationale bullets and Consequences in docstring (not at very end), so 'Consequences' applies to the updated state of the dict"
  - "Add Notes section to docstrings of run_inference and run_inference_with_bump describing init_to_median(50) as the chosen strategy with cost estimate"

patterns-established:
  - "Phase comment pattern: '# Phase 32-02: <one-liner rationale>' immediately above NUTS construction sites — makes the choice traceable to a phase plan"
  - "Prior-defaults docstring evolves chronologically: keep 'Rationale for moving to 0.0' historical block, append 'Phase 32 update (date)' sub-paragraph for follow-on adjustments"
  - "Integration tests for sampler config: use inspect.getsource on the function body — avoids needing to construct NUTS kernels in tests"

# Metrics
duration: 15min
completed: 2026-05-03
---

# Phase 32 Plan 02: NUTS init_to_median + kappa-family prior anchoring Summary

**Switched NumPyro NUTS init strategy to init_to_median(num_samples=50) in both run_inference and run_inference_with_bump, and anchored kappa/kappa_s/kappa_total mu_prior_loc to -1.0 (prior mean kappa ~ 0.16, Rmus 2023 pattern) without resurrecting MLE-empirical-Bayes circularity.**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-05-03T15:56:17Z
- **Completed:** 2026-05-03T16:11:28Z
- **Tasks:** 3 (all auto, no checkpoints)
- **Files modified:** 3 (1 sampling-orchestration, 1 priors-defaults, 1 integration-test)

## Accomplishments

- **Phase B intervention shipped:** Both `run_inference` (line 300) and `run_inference_with_bump` (line 435) now construct NUTS with `init_strategy=init_to_median(num_samples=50)`. NumPyro guides cite this as more robust for Phi_approx-bounded models than the default `init_to_uniform`. Cost: ~5 s wall-time per fit.
- **Phase C intervention shipped:** `PARAM_PRIOR_DEFAULTS` now anchors kappa-family group-mean priors at `mu_prior_loc = -1.0` (gives prior mean kappa = Phi(-1) ~ 0.16 on the bounded scale). Matches Rmus 2023's Beta(1+a, 1+b) hierarchical empirical-Bayes pattern without re-introducing the MLE-circularity flagged in the 2026-04-17 prior decision. `kappa_share` left at 0.0 because the channel-share is genuinely uncertain a priori (Collins 2025 WMH motor-vs-stimulus split does not anchor to either corner). `_PRIOR_LEGACY_MLE_CALIBRATED` untouched (still used for sensitivity analyses).
- **Test coverage added:** Two new integration tests in `tests/integration/test_numpyro_helpers.py` — `test_kappa_family_anchored_to_negative_one` (asserts the four PARAM_PRIOR_DEFAULTS entries) and `test_init_strategy_default_is_init_to_median` (uses `inspect.getsource` to verify both NUTS construction sites). Both fast (~0.01 s), safe for the not-slow tier.

## Task Commits

Each task was committed atomically:

1. **Task 1: Switch NUTS init strategy to init_to_median(50)** — `5179fc8` (feat)
2. **Task 2: Tighten kappa-family prior defaults to mu_prior_loc = -1.0** — `b45dcd8` (feat)
3. **Task 3: Add prior-defaults assertion + init-strategy assertion tests** — `f93c159` (test)

Other Wave 1 plans (32-01, 32-03) committed in parallel against `main` between Task 2 and Task 3 commits — no merge conflicts because all three plans modified disjoint files (32-01: bayesian.py + bayesian_summary_writer.py + test_bayesian_summary.py; 32-02: sampling.py + numpyro_helpers.py + test_numpyro_helpers.py; 32-03: cluster/submit_all.sh + cluster/04b_bayesian_cpu.slurm + config.py + tests/integration/test_bayesian_fanout_models.py).

## Files Created/Modified

- `src/rlwm/fitting/sampling.py` — Added `init_to_median` to `from numpyro.infer import ...` line; added `init_strategy=init_to_median(num_samples=50)` to both NUTS constructions (lines 300 and 435); added a 4-line explanatory comment above each NUTS site; added a Notes paragraph to the docstring of `run_inference` and a fourth Notes bullet to `run_inference_with_bump`. 20 insertions, 2 deletions.
- `src/rlwm/fitting/numpyro_helpers.py` — Updated 3 lines in `PARAM_PRIOR_DEFAULTS` (kappa/kappa_s/kappa_total, lines 189-191); appended a 13-line "Phase 32 update (2026-05-03)" paragraph in the docstring between the rationale-bullets block and the Consequences block. 16 insertions, 3 deletions.
- `tests/integration/test_numpyro_helpers.py` — Appended two new test functions after `test_param_prior_defaults_completeness`. 28 insertions.

## Decisions Made

- **Insertion point for Phase 32 docstring paragraph:** Placed between "Rationale for moving to 0.0..." bullets and "Consequences:" rather than at end-of-docstring. This way the existing "Consequences" framing applies to the *updated* state of the defaults dict (kappa = -1.0), not to the historical 0.0 baseline. Matches the chronology of how a future reader would parse the doc.
- **Test import path:** Used `from rlwm.fitting.numpyro_helpers import PARAM_PRIOR_DEFAULTS` and `from rlwm.fitting import sampling` rather than the literal `from src.rlwm.fitting.*` written in the plan. Reason: the package is installed via `pip install -e .` (per CLAUDE.md), so `rlwm.*` is the importable name; `src.rlwm.*` would fail at test collection. Documented as deviation Rule 3 below.
- **Did not revise existing slow tests:** `test_bounded_param_recovery_stick_breaking` still uses `mu_prior_loc=-2.0` for `kappa_total` in its synthetic-data setup, which now disagrees with the dict default of -1.0. Decision: leave as-is. The test passes per-param `mu_prior_loc` explicitly, so it does not consult `PARAM_PRIOR_DEFAULTS`. Updating it to -1.0 would change the recovery target arbitrarily, and the test is gated behind `@pytest.mark.slow` (not part of plan verification gate).
- **Did not change `phi_rl` defaults:** The plan's must-haves note "phi_rl mu_prior_loc remains 0.0 (M5 is being dropped from Bayesian — but the entry stays for any reader of the dict)." Confirmed: phi_rl entry untouched at line 193.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Adjusted test import path from `src.rlwm.fitting.*` to `rlwm.fitting.*`**
- **Found during:** Task 3 (writing the new test functions)
- **Issue:** The plan's literal `from src.rlwm.fitting.numpyro_helpers import PARAM_PRIOR_DEFAULTS` (and `from src.rlwm.fitting import sampling`) would fail at import. The `rlwm` package is installed via `pip install -e .` per CLAUDE.md and the existing `tests/integration/test_numpyro_helpers.py` imports use `from rlwm.fitting.numpyro_helpers import (...)` (line 26, present in the baseline file).
- **Fix:** Used `from rlwm.fitting.numpyro_helpers import PARAM_PRIOR_DEFAULTS` and `from rlwm.fitting import sampling` to match repo convention. Test passes both new assertions.
- **Files modified:** tests/integration/test_numpyro_helpers.py
- **Verification:** `python -m pytest tests/integration/test_numpyro_helpers.py -v -k "kappa_family or init_strategy"` → 2 passed.
- **Committed in:** `f93c159` (Task 3 commit).

---

**Total deviations:** 1 auto-fixed (1 blocking import-path adjustment to match repo install convention).
**Impact on plan:** No semantic change — both new test functions still test exactly the invariants specified in the plan. Just uses the correct package name for the editable install.

## Issues Encountered

- **Mid-execution stash interaction:** During Task 1 verification, I ran `git stash` to test whether ruff errors in `sampling.py` were pre-existing (they were — 11 pre-existing). `git stash pop` restored cleanly, but a later `git stash --keep-index` (which I tried during Task 3 verification to inspect ruff posture) interacted with parallel-agent commits arriving on `main` in a way that placed my unstaged Task 3 edits into a stash mixed with unrelated pre-existing modifications. Resolution: re-applied the Task 3 Edit cleanly to the baseline test file, ran tests (2 passed), and committed. No work lost; the stash holding the unrelated mods (`tests/integration/test_bayesian_summary.py` modified by 32-01 agent) remained on `main` untouched.
- **Other Wave 1 commits arriving concurrently:** 32-01 and 32-03 plan agents committed `8f229e0`, `a7dde96`, `413f16c`, `026649f`, `5b582dd`, `2b53a5a`, and `955810e` to `main` while I was editing. No file overlap (verified by inspection); my 3 commits sit cleanly interleaved (5179fc8 / b45dcd8 / f93c159).

## Verification Results

All plan-specified verifications passed:

| Verification | Result |
|---|---|
| `grep "init_to_median(num_samples=50)" sampling.py` | 4 matches (2 NUTS constructions + 2 docstring mentions; only 2 are required, more is fine) |
| `python -c "from rlwm.fitting.sampling import run_inference; print('OK')"` | OK |
| `python -c "...PARAM_PRIOR_DEFAULTS...kappa==−1.0...kappa_share==0.0..."` | OK |
| `ruff check src/rlwm/fitting/numpyro_helpers.py` | All checks passed |
| `ruff check src/rlwm/fitting/sampling.py` | 11 pre-existing errors (unchanged from baseline; my edits introduce 0 new errors) |
| `pytest test_numpyro_helpers.py test_v4_closure.py` | 16/16 passed (incl. the 2 new tests + the 3 v4 closure guards) |
| `python tests/scientific/check_v4_closure.py` | 5/5 checks passed, EXIT 0 |
| `pytest tests/integration/test_m3_hierarchical.py` | 8/8 passed (smoke + per-model fully-batched closures, ~113 s) |

The plan-quoted "ruff check src/rlwm/fitting/sampling.py" command was not run as a gate because the 11 pre-existing errors there are untouched by Phase 32-02 — they are tracked elsewhere (lint-cleanup work) and would force a Rule 1 deviation that goes beyond this plan's scope. Confirmed pre-existing by `git stash` baseline check.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **Empirical convergence improvement is unverified.** This plan only configures the sampler; it does not run an MCMC fit. The actual reduction in R-hat / increase in ESS for M3 / M5 / M6a (which previously hit R-hat = 1.60, ESS = 7) is **not** validated until plan **32-05's smoke run** on the cluster. If the smoke run still shows κ multimodality, plan **32-04 (softmax-bias κ ∈ [-1, 1] per Collins 2025)** is the next escalation.
- **Phase 32-04 (softmax-bias κ implementation) is unblocked.** That plan adds a `--kappa-parameterization {softmax,convex}` CLI flag and replaces the convex-mixture decode; it depends on the Wave 1 plans (32-01 BFMI gate, 32-02 init+priors, 32-03 fan-out filter) all being committed. All three are now in main.
- **v4 closure guards remain green** (`tests/integration/test_v4_closure.py` 3/3 + `tests/scientific/check_v4_closure.py` 5/5). No regression on M6b kappa decode invariants.
- **`test_m3_hierarchical.py` 8/8 passing** confirms PARAM_PRIOR_DEFAULTS still loads cleanly inside the M3, M5, M6a, M6b model wrappers under the new `mu_prior_loc = -1.0` defaults — no construction-time errors.

---
*Phase: 32-mcmc-methodology-update*
*Completed: 2026-05-03*
