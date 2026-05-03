---
phase: 32-mcmc-methodology-update
plan: 03
subsystem: infra

tags: [bayesian, numpyro, mcmc, slurm, cluster, fan-out, model-comparison, kappa-share, m6b, collins-2025]

# Dependency graph
requires:
  - phase: 22-bayesian-model-suite
    provides: M6b kappa_share simplex parameterization that subsumes M3 (kappa_share=1) and M6a (kappa_share=0)
  - phase: 32-01-bfmi-gate
    provides: BFMI gate exposing the misspecification artefacts (R-hat=1.60, ESS=7) on M3/M5/M6a that this plan removes from default fan-out
  - phase: 32-02-prior-tightening
    provides: Tightened kappa-family mu_prior_loc=-1.0 and init_to_median NUTS init that make M6b a robust standalone Bayesian fit

provides:
  - "config.BAYESIAN_FANOUT_MODELS = ['qlearning', 'wmrl', 'wmrl_m6b'] — single source of truth for narrowed default Bayesian fan-out"
  - "cluster/submit_all.sh --bayes-models CLI flag with default 'qlearning wmrl wmrl_m6b'"
  - "cluster/04b_bayesian_{cpu,gpu}.slurm standalone defaults flipped to MODEL=wmrl_m6b (canonical winner)"
  - "Two new closure-guard tests in tests/integration/test_v4_closure.py locking the contract"
  - "~3h cluster compute saved per pipeline run (M3+M5+M6a Bayesian fits dropped from default chain)"

affects:
  - phase: 24-cold-start-pipeline-execution  # Wave 1 canary now runs only 3 Bayesian fits, not 6
  - phase: 25-reproducibility                # Reproducibility regression baseline now narrower fan-out
  - phase: 26-manuscript-finalization        # Tables/figures will read 3-model Bayesian set (M1, M2, M6b); MLE table still 7-model
  - phase: 27-closure                        # v5.0 closure invariants need updating for narrowed Bayesian set

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single-source-of-truth pattern: cluster/submit_all.sh reads default from config.BAYESIAN_FANOUT_MODELS-equivalent literal; closure-guard test asserts the literal stays in sync"
    - "Override-preserving narrowing: production default narrowed but --bayes-models flag preserved for sensitivity analyses (M3/M5/M6a remain submittable manually)"
    - "Decoupled MLE/Bayesian fan-out: MLE keeps all 7 models (AIC table needs full set), Bayesian narrowed to 3 unrestricted models (constrained corner cases handled by M6b posterior on kappa_share)"

key-files:
  created:
    - .planning/phases/32-mcmc-methodology-update/32-03-SUMMARY.md
  modified:
    - config.py (BAYESIAN_FANOUT_MODELS constant added below CHOICE_ONLY_MODELS, line 297-307)
    - cluster/submit_all.sh (BAYES_MODELS variable + --bayes-models flag + dual banner + narrowed 04b loop)
    - cluster/04b_bayesian_gpu.slurm (Phase 32-03 header + MODEL default wmrl_m3 -> wmrl_m6b)
    - cluster/04b_bayesian_cpu.slurm (Phase 32-03 header + MODEL default wmrl_m3 -> wmrl_m6b)
    - tests/integration/test_v4_closure.py (test_phase32_bayesian_fanout_narrowed + test_phase32_submit_all_narrowed_bayes_default)

key-decisions:
  - "Drop M3/M5/M6a from DEFAULT Bayesian fan-out, NOT from MLE. Keep all 7 in ALL_MODELS so AIC table stays unchanged."
  - "Preserve --bayes-models override path so users can still manually fit M3/M5/M6a as sensitivity analyses (no information loss)."
  - "Standalone SLURM default flipped from wmrl_m3 to wmrl_m6b — interactive sbatch without --export now defaults to canonical winner."
  - "Closure-guard literal-string assertion on submit_all.sh prevents config.BAYESIAN_FANOUT_MODELS and submit_all.sh from drifting apart."

patterns-established:
  - "Pattern: Production-default narrowing with override preserved — default trimmed for compute economy + scientific defensibility, but the dropped configurations stay one CLI flag away for sensitivity analyses."
  - "Pattern: Test-locked literal-string contract between Python config and shell orchestrator — closure-guard greps the literal default out of submit_all.sh, so any future drift fails the test."

# Metrics
duration: 13min
completed: 2026-05-03
---

# Phase 32-03: Narrow Default Bayesian Fan-out Summary

**Default hierarchical Bayesian fan-out narrowed from 6 to 3 models ({qlearning, wmrl, wmrl_m6b}); M3/M5/M6a dropped because they are M6b corner cases (kappa_share=1.0, kappa_share=0.0) or Collins 2025 ruled out (phi_rl unnecessary), saving ~3h cluster compute per pipeline run.**

## Performance

- **Duration:** 13 min
- **Started:** 2026-05-03T13:56:34Z
- **Completed:** 2026-05-03T14:09:10Z
- **Tasks:** 4/4
- **Files modified:** 5 (1 config + 1 orchestrator + 2 SLURMs + 1 test)
- **Commits:** 4 atomic + 1 metadata

## Accomplishments

- Added `BAYESIAN_FANOUT_MODELS: list[str] = ["qlearning", "wmrl", "wmrl_m6b"]` as single source of truth in `config.py` (just below `CHOICE_ONLY_MODELS`).
- Narrowed the stage-04b Bayesian fan-out loop in `cluster/submit_all.sh` to iterate over a separate `BAYES_MODELS` variable (default `"qlearning wmrl wmrl_m6b"`); MLE fan-out and stage-03 prior-predictive / Bayesian-recovery loops still iterate over `MODELS` (all 6 choice-only).
- Added `--bayes-models` CLI flag to `cluster/submit_all.sh` with override semantics (`--bayes-models "qlearning wmrl wmrl_m6b wmrl_m3"` re-enables M3 as a sensitivity analysis without code changes).
- Updated banner echo to print both lists side-by-side: `models (MLE):` and `models (Bayes):` so operators see the asymmetry at submission time.
- Flipped standalone defaults of `cluster/04b_bayesian_{cpu,gpu}.slurm` from `MODEL=wmrl_m3` to `MODEL=wmrl_m6b` so a bare `sbatch cluster/04b_bayesian_gpu.slurm` now fits the canonical winning model rather than a now-dropped one.
- Added Phase 32-03 header comment to both SLURM templates documenting the narrowing rationale and the standalone-sbatch path for sensitivity-only M3/M5/M6a fits.
- Added two regression tests in `tests/integration/test_v4_closure.py`:
  - `test_phase32_bayesian_fanout_narrowed`: asserts `BAYESIAN_FANOUT_MODELS == ['qlearning', 'wmrl', 'wmrl_m6b']` AND all 7 MLE models still in `ALL_MODELS`.
  - `test_phase32_submit_all_narrowed_bayes_default`: asserts `cluster/submit_all.sh` contains the literal default line `BAYES_MODELS="${BAYES_MODELS:-qlearning wmrl wmrl_m6b}"`.
- Verified all 4 plan-level verification gates pass:
  - `bash -n cluster/submit_all.sh` exit 0
  - `python -c "from config import BAYESIAN_FANOUT_MODELS; ..."` returns `['qlearning', 'wmrl', 'wmrl_m6b']`
  - `pytest tests/integration/test_v4_closure.py` 5/5 PASS
  - `python tests/scientific/check_v4_closure.py --milestone v4.0` exit 0 (5/5 PASS)
  - Smoke: `bash cluster/submit_all.sh --dry-run --bayes-models "qlearning"` shows `models (Bayes): qlearning` and `[04b] qlearning -> 1001` only.

## Task Commits

Each task committed atomically (4 atomic commits + 1 metadata commit pending below):

1. **Task 1: Add BAYESIAN_FANOUT_MODELS constant to config.py** — `e144b35` (feat)
2. **Task 2: Narrow submit_all.sh Bayesian fan-out** — `a7dde96` (feat)
3. **Task 3: Audit cluster SLURM templates** — `026649f` (feat)
4. **Task 4: Add closure-guard test** — `2b53a5a` (test)

**Plan metadata commit:** _pending_ (docs(32-03): close plan)

## Files Created/Modified

- `config.py` — Added `BAYESIAN_FANOUT_MODELS` 3-element list with 6-line comment block citing M3/M6a/M5 reduction logic
- `cluster/submit_all.sh` — Added `BAYES_MODELS` default, `--bayes-models` parser arm, dual MLE/Bayes banner echo, narrowed `for m in $BAYES_MODELS` loop in the [04b] block (stage 03 still uses `$MODELS`); MLE fan-out untouched
- `cluster/04b_bayesian_gpu.slurm` — Phase 32-03 header comment block + `MODEL="${MODEL:-wmrl_m6b}"` default flip + Usage line update
- `cluster/04b_bayesian_cpu.slurm` — Phase 32-03 header comment block + `MODEL="${MODEL:-wmrl_m6b}"` default flip + Usage line annotations marking M3/M5/M6a sensitivity-only post 32-03
- `tests/integration/test_v4_closure.py` — Two new test functions appended; total now 5 tests (3 v4 + 2 phase 32). Existing 3 tests still pass byte-identically.

## Decisions Made

- **MLE fan-out kept at 6 choice-only models, not narrowed.** AIC table in the manuscript needs all 7 models (M1/M2/M3/M5/M6a/M6b/M4) for the formal comparison; only the hierarchical Bayesian fan-out is narrowed because the M6b posterior on kappa_share gives the Bayesian evidence about which sub-model the data prefers.
- **Stage 03 prior-predictive / Bayesian-recovery loops kept on `$MODELS`, not `$BAYES_MODELS`.** Plan explicitly says these sweeps "should keep running on all models for consistency with prior runs"; they are diagnostic tools, not production fits, and skipping them on M3/M5/M6a would create a gap in the recovery audit history.
- **Standalone SLURM default flipped to `wmrl_m6b`, not removed entirely.** Plan asked for default flip; the alternative (require `--export=ALL,MODEL=...` always) would be a breaking change for any cluster-side runbook still typing `sbatch cluster/04b_bayesian_gpu.slurm` interactively.
- **Closure-guard test asserts literal string match in `submit_all.sh`, not just the count.** Asserting `'BAYES_MODELS="${BAYES_MODELS:-qlearning wmrl wmrl_m6b}"' in submit_all` (rather than e.g. `len(matches) == 3`) catches the most likely future drift mode: someone reorders the list or replaces `wmrl_m6b` with `wmrl_m3` while keeping the count constant.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Re-applied Task 3 SLURM edits after a silent revert during dry-run smoke test**

- **Found during:** Task 3 verification (between initial Edit calls and `git status` check)
- **Issue:** After applying Task 3 edits to both `cluster/04b_bayesian_{cpu,gpu}.slurm` (MODEL default flip + Phase 32-03 header), running `bash cluster/submit_all.sh --dry-run` triggered the orchestrator's CRLF-strip step (`sed -i 's/\r$//' cluster/*.slurm cluster/*.sh`), which on a Windows working tree appears to have silently reverted my freshly-edited files to their pre-edit content. `git diff` after that showed zero diff for the SLURMs even though the Edit tool had reported success.
- **Fix:** Confirmed pre-edit content via `grep -n "MODEL.*:-wmrl"` (returned `wmrl_m3` defaults), then re-applied all three Task 3 edits (gpu header, gpu MODEL default, cpu header, cpu MODEL default) and committed IMMEDIATELY without running any intervening shell commands that could re-trigger the CRLF strip.
- **Files modified:** `cluster/04b_bayesian_gpu.slurm`, `cluster/04b_bayesian_cpu.slurm`
- **Verification:** Post-commit `grep -n "MODEL.*:-wmrl_m6b\|Phase 32-03"` returns 6 matches across both files; `git log --oneline -5` shows `026649f feat(32-03): flip standalone Bayesian SLURM defaults to wmrl_m6b` cleanly committed; subsequent `bash -n` syntax checks pass on all 3 cluster scripts.
- **Committed in:** `026649f` (Task 3 commit, second attempt)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Single transient revert recovered cleanly; no scope creep, no scientific change. The CRLF-strip step in `submit_all.sh` is a known Windows-on-Linux-cluster compatibility step (line 108) that should not normally affect a freshly-edited file's content, only its line endings — root-cause of the revert is unclear (possibly a Windows-line-ending interaction with `sed -i` plus the editor's auto-save) but the second-attempt commit is byte-correct.

## Issues Encountered

- **Concurrent Phase 32 commits during execution.** While I was working, parallel agent(s) shipped commits `8f229e0` (32-01 BFMI gate), `5179fc8` (32-02 init_to_median), `b45dcd8` (32-02 kappa prior anchor), `413f16c` (32-01 BFMI logging), `f93c159` and `5b582dd` (32-01/32-02 tests). These commits did NOT touch any file in my plan's scope (config.py BAYESIAN section, submit_all.sh, the two SLURMs, test_v4_closure.py), so my atomic commits applied cleanly with no merge conflicts. Final `git log --oneline -10` shows my 4 commits interleaved correctly with the parallel work.

## User Setup Required

None — no external service configuration required. Cluster runbook (`bash cluster/submit_all.sh`) continues to work; users who want the old wide fan-out can run `bash cluster/submit_all.sh --bayes-models "qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b"` to reproduce pre-32-03 behavior.

## Authentication Gates

None — execution was fully local; no cluster sbatch, no remote auth, no API keys.

## Next Phase Readiness

**Ready for downstream phases:**
- **Phase 24 (cold-start pipeline execution):** Next `bash cluster/submit_all.sh` cold-start submission will fan out 3 Bayesian fits (qlearning + wmrl + wmrl_m6b) instead of 6. Wall-clock budget for the Bayesian leg drops from ~6h to ~3-4h on the GPU path; the M6b 36h time override stays in place.
- **Phase 26 (manuscript finalization):** `quarto render manuscript/paper.qmd` graceful-fallback cells need to handle the narrowed Bayesian set. Manuscript text should now describe model comparison with three Bayesian fits and a 7-model MLE AIC table. M3/M5/M6a sub-model evidence comes from the M6b kappa_share posterior, not standalone fits.
- **Phase 27 (v5.0 closure):** Closure invariants in `tests/scientific/check_v4_closure.py` are unaffected (5/5 still PASS). New Phase 32 invariants live in `tests/integration/test_v4_closure.py` (5/5 PASS) — both gates are green simultaneously.

**Blockers/concerns:**
- None for this plan.
- **Watch-item for Phase 26:** any manuscript-table builder that currently iterates over `CHOICE_ONLY_MODELS` for Bayesian artifacts will now find 3 of the 6 entries missing on disk. Builders should switch to `BAYESIAN_FANOUT_MODELS` for Bayesian-source tables and keep `CHOICE_ONLY_MODELS` (or `ALL_MODELS`) for MLE-source tables.

---
*Phase: 32-mcmc-methodology-update*
*Completed: 2026-05-03*
