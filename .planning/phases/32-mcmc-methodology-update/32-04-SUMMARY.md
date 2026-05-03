---
phase: 32-mcmc-methodology-update
plan: 04
subsystem: infra
tags: [jax, numpyro, mcmc, kappa-parameterization, collins-2025, senta-2025]

requires:
  - phase: 32-mcmc-methodology-update
    provides: Wave 1 BFMI gate, init_to_median, kappa-prior anchor, narrow Bayesian fan-out (32-01..32-03)
provides:
  - Dual-mode kappa parameterization (Senta 2025 convex-mixture vs Collins 2025 softmax-bias)
  - Single CLI flag --kappa-parameterization {softmax,convex} threading from CLI -> engine -> JAX -> NumPyro -> SLURM
  - sample_unbounded_normal_param helper for kappa on the real line
  - get_param_bounds(model, kappa_parameterization) factory in config.py
  - kappa_parameterization metadata column in Bayesian summary CSV
  - SC-3 bit-equivalence regression test (4/4 scientific tests, 1e-10 tolerance)
  - KAPPA_MODE env var threaded through cluster orchestrator + 4 SLURM templates
affects: [32-05, 32-06, manuscript paper.qmd Methods §Perseveration]

tech-stack:
  added: []
  patterns:
    - "Phase 32-04 dual-mode kappa: 'softmax' branch (Collins 2025, default) + 'convex' branch (Senta 2025, legacy revert) gated by parameterization kwarg in JAX likelihood + NumPyro wrapper + MLE engine + Bayesian engine + run-metadata column + SLURM env var."
    - "kappa_share is intentionally EXCLUDED from KAPPA_FAMILY_PARAMS — channel-share weight, not a softmax bias; always stays [0, 1] under both modes."
    - "WM/RL channel mix asymmetry: softmax mode mixes in LOGIT space (Collins 2025 WMH); convex mode mixes in PROBABILITY space (Senta 2025). Documented explicitly in MODEL_REFERENCE.md §3.6.4."
    - "Bit-equivalence regression: hard-coded float reference constants (NOT runtime JSON loads) in scientific-tier pytest gate. Reference floats captured by Task 0 against the unmodified pre-edit codebase, then substituted into Task 6.5 post-edit."

key-files:
  created:
    - .planning/phases/32-mcmc-methodology-update/32-04-references.json
    - .planning/phases/32-mcmc-methodology-update/32-04-mle-refit-jobs.txt
    - tests/integration/test_kappa_parameterization_regression.py
  modified:
    - src/rlwm/fitting/numpyro_helpers.py
    - src/rlwm/fitting/models/wmrl_m3.py
    - src/rlwm/fitting/models/wmrl_m5.py
    - src/rlwm/fitting/models/wmrl_m6a.py
    - src/rlwm/fitting/models/wmrl_m6b.py
    - src/rlwm/fitting/models/wmrl_m4.py
    - src/rlwm/fitting/mle.py
    - src/rlwm/fitting/bayesian.py
    - config.py
    - scripts/04_model_fitting/a_mle/fit_mle.py
    - scripts/04_model_fitting/b_bayesian/fit_baseline.py
    - scripts/04_model_fitting/b_bayesian/fit_bayesian.py
    - scripts/fitting/bayesian_summary_writer.py
    - scripts/fitting/lba_likelihood.py
    - scripts/fitting/mle_utils.py
    - cluster/submit_all.sh
    - cluster/04a_mle_gpu.slurm
    - cluster/04a_mle_cpu.slurm
    - cluster/04b_bayesian_gpu.slurm
    - cluster/04b_bayesian_cpu.slurm
    - tests/integration/test_bayesian_summary.py
    - docs/03_methods_reference/MODEL_REFERENCE.md

key-decisions:
  - "Default project policy: --kappa-parameterization softmax (Phase 32-04 user decision 2026-05-03 to ship Collins 2025 alignment as v5.0 default; convex revert path preserved for sensitivity analysis and v5.0 reproducibility)."
  - "Logit-space WM/RL channel mix in softmax branch (matches Collins 2025 WMH formulation: mixed_logits = omega*beta*W + (1-omega)*beta*Q, then softmax). Probability-space mix preserved in convex branch (matches Senta 2025). This asymmetry is intentional and documented; convex-mode at kappa=0 is bit-equivalent to v5.0 pre-Phase-32, but softmax-mode at kappa=0 differs from convex-mode at kappa=0 by ~0.17 nats on the toy fixture (logit-vs-prob mix)."
  - "M6a/M6b per-stimulus last_action_s lookup unchanged across modes — softmax branch uses the same per-stimulus carry (M6a) or dual-carry (M6b) as the convex branch; only the kernel application changes (one_hot bias vs probability mixture)."
  - "M6b stick-breaking decode unchanged: kappa = kappa_total * kappa_share, kappa_s = kappa_total * (1 - kappa_share). Under softmax mode, kappa_total ∈ [-1, 1] but kappa_share always [0, 1] (KAPPA_FAMILY_PARAMS excludes kappa_share). When kappa_total < 0, both kappa and kappa_s share the (non-positive) sign — partitioned action-avoidance budget."
  - "Test parameterization: Option A (loop over both parameterizations inside the existing test functions) chosen for the existing M6b corner-case tests (lines 869, 920 of wmrl_m6b.py) — minimal LOC, same function signatures, both modes gated by single assertion. Both corner-case tests (kappa_share=1 -> M3, kappa_share=0 -> M6a) PASS under BOTH parameterizations at diff=0.00e+00."
  - "Task 6 backward-compat tests file (tests/integration/test_wmrl_model.py) NOT modified for Phase 32-04. Existing tests there exercise only model compilation + prior ranges, not kappa-specific behavior; the new kappa-aware tests live in test_kappa_parameterization_regression.py and the in-file M6b corner-case tests."
  - "MLE re-fit submission DEFERRED to Plan 32-05 cluster execution. Plan 32-04 completes when CODE/CONFIG/SLURM changes are merged; submission command + manifest documented in 32-04-mle-refit-jobs.txt for human to run on M3."

patterns-established:
  - "Kappa-aware bounds resolution: get_param_bounds(model_name, kappa_parameterization) factory in config.py is the single source of truth for kappa-aware (lower, upper) tuples going forward. MLE engine's _kappa_aware_bounds_dict mirrors the factory locally for transform-time bounds resolution."
  - "Run-metadata schema additive evolution: kappa_parameterization column appended AFTER parameterization_version (legacy readers tolerate trailing columns; CSV remains forward-compatible)."
  - "Cluster orchestrator KAPPA_MODE env var convention: --kappa-parameterization on submit_all.sh CLI -> KAPPA_MODE in --export=ALL,...,KAPPA_MODE=... -> KAPPA_MODE env-var read in SLURM body -> --kappa-parameterization $KAPPA_MODE on python CLI."

duration: ~150min
completed: 2026-05-03
---

# Phase 32 Plan 04: Softmax-Bias Kappa Parameterization Summary

**Dual-mode kappa parameterization (Senta 2025 convex-mixture vs Collins 2025 softmax-bias) gated by single --kappa-parameterization {softmax,convex} CLI flag threading through CLI -> JAX -> NumPyro -> Bayesian summary CSV -> SLURM env var; convex-mode bit-equivalent to v5.0 pre-Phase-32 fits at 1e-10 tolerance for M3/M5/M6a/M6b.**

## Performance

- **Duration:** ~150 min
- **Started:** 2026-05-03T14:20:39Z (Task 0 reference capture)
- **Completed:** 2026-05-03 (Task 8 docs commit)
- **Tasks:** 9 of 9 (Task 9 cluster submission deferred to Plan 32-05 per orchestrator constraint)
- **Files modified:** 22 (3 created + 19 modified)
- **Commits:** 9 atomic feature commits + this metadata commit

## Accomplishments

- **Single CLI flag threads through 6 layers:** fit_mle.py / fit_baseline.py / fit_bayesian.py CLIs -> rlwm.fitting.mle.fit_model + rlwm.fitting.bayesian.fit_model engines -> get_param_bounds factory in config.py -> 5 JAX block likelihoods (M3/M5/M6a/M6b/M4) + their multiblock/stacked/pscan variants -> 5 NumPyro hierarchical wrappers (M3/M5/M6a/M6b/M4) -> bayesian_summary_writer CSV column -> 4 SLURM templates (04a/04b GPU+CPU) + submit_all.sh master orchestrator.
- **Convex revert path bit-equivalent to v5.0 pre-Phase-32:** all 4 perseveration models (M3/M5/M6a/M6b) reproduce reference log-likelihoods at diff=0.00e+00 (much tighter than the 1e-10 tolerance requirement) on the 10-trial toy fixture. Captured against SHA 9adc858 BEFORE any Phase 32-04 edits.
- **Boundary-pathology motivation actionable:** softmax-bias supports negative kappa (κ ∈ [-1, 1], action-avoidance) and has informative gradient everywhere on the support. Documented in MODEL_REFERENCE.md §3.6.4 with comparison table and Collins 2025 + Senta 2025 citations.
- **M6b corner-case containment preserved under both parameterizations:** kappa_share=1.0 -> M3 and kappa_share=0.0 -> M6a both hold bit-perfectly (diff=0.00e+00) under softmax AND convex modes.
- **v4 closure remained green throughout:** integration 5/5 PASS + scientific 5/5 PASS at every commit boundary.
- **Bayesian summary CSV gains kappa_parameterization metadata column** as the 35th/36th field (after parameterization_version). Existing 17/17 test_bayesian_summary tests updated and passing.

## Task Commits

Each task was committed atomically (8 feature commits + 1 task-0 capture commit):

1. **Task 0: Capture v5.0 pre-Phase-32 reference log-likelihoods** — `42da7c4` (feat)
   - `.planning/phases/32-mcmc-methodology-update/32-04-references.json` with 4 finite reference floats + git SHA + UTC timestamp metadata.
2. **Task 1: Add sample_unbounded_normal_param + get_param_bounds factory** — `9c8cafc` (feat)
   - New helper for unbounded kappa on R (clipped to [-1, 1]) in numpyro_helpers.py.
   - New get_param_bounds factory + KAPPA_FAMILY_PARAMS constant.
3. **Task 2: Dual-mode kappa in M3/M5/M6a/M6b JAX block likelihoods** — `5d6f5a6` (feat)
   - Every block / multiblock / stacked / fully_batched / pscan / multiblock_stacked_pscan signature now accepts `parameterization='softmax'` kwarg.
   - M6b in-file corner-case tests refactored to Option A (loop over both parameterizations).
4. **Task 3: Hierarchical NumPyro wrappers thread kappa_parameterization** — `a48b0a8` (feat)
   - All 5 hierarchical wrappers (M3/M5/M6a/M6b + M6b subscale) gain kwarg with default 'softmax'. M4 wrapper handled in Task 4 commit (combined with engine).
5. **Task 4: MLE engine + LBA + CLI threading** — `94efcc6` (feat)
   - 5 _make_jax_objective_* + 5 _make_bounded_objective_* + 5 jax_unconstrained_to_params_* + lba_likelihood (M4) + fit_all_participants/fit_participant_mle + main() argparse all threaded.
   - New _kappa_aware_bounds_dict helper resolves kappa-family bounds at L-BFGS-B-bound-construction time.
   - M4 hierarchical wrapper in src/rlwm/fitting/models/wmrl_m4.py also threaded (kappa-family params dispatch through sample_unbounded_normal_param under softmax).
6. **Task 5: Bayesian engine + CLI + run-metadata schema** — `6b7a010` (feat)
   - bayesian.py main()/fit_model/_fit_stacked_model/save_results all threaded.
   - fit_baseline.py gains --kappa-parameterization flag, threaded through sys.argv injection.
   - bayesian_summary_writer.py: required kwarg, validates membership, populates new column. Two existing tests updated.
7. **Task 6.5: SC-3 bit-equivalence regression test (4 scientific tests)** — `76bcb76` (test)
   - tests/integration/test_kappa_parameterization_regression.py: 4 hard-coded REF_M{3,5,6A,6B} floats, @pytest.mark.scientific, all 4/4 PASS at diff=0.00e+00.
8. **Task 7: Cluster orchestrator + 4 SLURM templates** — `1a994d2` (feat)
   - submit_all.sh: --kappa-parameterization flag, KAPPA_MODE env var, validation, banner, threaded into stage 04b sbatch --export.
   - 04a_mle_{gpu,cpu}.slurm: KAPPA_MODE env var, banner, threaded into fit_mle.py CLI. GPU SLURM dispatch path also forwards KAPPA_MODE to sub-jobs.
   - 04b_bayesian_{gpu,cpu}.slurm: KAPPA_MODE env var, banner, threaded into fit_baseline.py CLI.
9. **Task 8: MODEL_REFERENCE.md §3.6.4 documentation** — `75ad079` (docs)
   - ~70 lines of new Markdown: side-by-side mathematical definitions, boundary-behavior comparison table, Collins+Senta 2025 citations, M6b stick-breaking under softmax mode, cross-reference to .planning/research/mcmc-collins-bayesian.md.
10. **Task 9: MLE re-fit submission** — `pending-cluster` (deferred)
    - 32-04-mle-refit-jobs.txt manifest documents the exact `bash cluster/submit_all.sh --kappa-parameterization softmax --bayes-models 'qlearning wmrl wmrl_m6b'` command for human to run on M3 at Plan 32-05.

**Plan metadata commit:** _will be created after this SUMMARY.md is written._

## Files Created/Modified

### Created (3)

- `.planning/phases/32-mcmc-methodology-update/32-04-references.json` — v5.0 pre-Phase-32 reference floats + provenance metadata.
- `.planning/phases/32-mcmc-methodology-update/32-04-mle-refit-jobs.txt` — submission manifest for Plan 32-05.
- `tests/integration/test_kappa_parameterization_regression.py` — 4 scientific-tier bit-equivalence tests.

### Modified (19)

**Library code (8):**
- `src/rlwm/fitting/numpyro_helpers.py` — sample_unbounded_normal_param, KAPPA_FAMILY_PARAMS, sample_model_params dispatch.
- `src/rlwm/fitting/models/wmrl_m3.py` — block/multiblock/stacked/fully_batched/pscan/multiblock_stacked_pscan + hierarchical wrapper.
- `src/rlwm/fitting/models/wmrl_m5.py` — same six function variants + hierarchical wrapper.
- `src/rlwm/fitting/models/wmrl_m6a.py` — same six function variants + hierarchical wrapper.
- `src/rlwm/fitting/models/wmrl_m6b.py` — same six function variants + 2 hierarchical wrappers (main + subscale) + 2 in-file corner-case tests refactored.
- `src/rlwm/fitting/models/wmrl_m4.py` — hierarchical wrapper.
- `src/rlwm/fitting/mle.py` — _kappa_aware_bounds_dict helper, 5 _make_jax_objective + 5 _make_bounded_objective + fit_participant_mle + fit_all_participants + main() argparse + sequential/parallel paths.
- `src/rlwm/fitting/bayesian.py` — main() argparse + fit_model + _fit_stacked_model + save_results.

**Configuration / module-level (2):**
- `config.py` — _KAPPA_FAMILY_BOUNDS, _STATIC_PARAM_BOUNDS, get_param_bounds factory.
- `scripts/fitting/mle_utils.py` — _kappa_bounds helper, 5 jax_unconstrained_to_params_wmrl_m{3,5,6a,6b,4} updates.

**LBA / M4 likelihood (1):**
- `scripts/fitting/lba_likelihood.py` — wmrl_m4_block_likelihood + wmrl_m4_multiblock_likelihood + wmrl_m4_multiblock_likelihood_stacked.

**CLI entries (2):**
- `scripts/04_model_fitting/a_mle/fit_mle.py` — no source change (delegates to mle.main()).
- `scripts/04_model_fitting/b_bayesian/fit_baseline.py` — argparse + sys.argv injection.

**Run-metadata writer (1):**
- `scripts/fitting/bayesian_summary_writer.py` — _build_column_order + write_bayesian_summary signature + per-row population.

**Cluster orchestrator + SLURMs (5):**
- `cluster/submit_all.sh` — --kappa-parameterization flag, KAPPA_MODE env var, validation, banner, stage-04b sbatch threading.
- `cluster/04a_mle_gpu.slurm` — KAPPA_MODE read + threaded into fit_mle.py CLI + dispatch-loop EXPORTS.
- `cluster/04a_mle_cpu.slurm` — KAPPA_MODE read + threaded into fit_mle.py CLI.
- `cluster/04b_bayesian_gpu.slurm` — KAPPA_MODE read + threaded into fit_baseline.py CLI.
- `cluster/04b_bayesian_cpu.slurm` — KAPPA_MODE read + threaded into fit_baseline.py CLI.

**Tests (2):**
- `tests/integration/test_bayesian_summary.py` — 2 existing test calls now pass `kappa_parameterization='convex'` (newly required kwarg).
- (test_kappa_parameterization_regression.py listed under Created above)

**Docs (1):**
- `docs/03_methods_reference/MODEL_REFERENCE.md` — new §3.6.4 (~70 lines).

## Decisions Made

1. **Softmax is project default.** Phase 32-04 user decision (2026-05-03) promoted "Senta convex -> Collins softmax" from a v5.1 candidate to a v5.0 ship. Boundary-pathology argument (∂L/∂κ → 0 at κ=0; degenerate at κ=1) plausibly explains M3's multimodal Bayesian fit (R-hat=1.60, ESS=7) where Collins's MLE fits converge.

2. **Logit-vs-probability channel-mix asymmetry is intentional.** Softmax branch mixes WM and RL channels in **logit** space (matches Collins 2025 WMH formulation: `mixed_logits = ω·β·W + (1-ω)·β·Q`); convex branch mixes in **probability** space (preserves Senta 2025 bit-equivalently). Consequence: at κ=0 the two modes do NOT produce identical likelihoods (they differ by the non-linearity of softmax-after-mix vs softmax-then-mix). On the toy fixture this is ~0.17 nats. This is documented explicitly in MODEL_REFERENCE.md §3.6.4 and in code comments.

3. **kappa_share stays bounded [0, 1] under both modes.** Channel-share weight in M6b stick-breaking is NOT a softmax bias — it's a partitioning ratio for the kappa_total budget across the global vs stimulus-specific channels. Under softmax mode, kappa_total ∈ [-1, 1] but kappa_share ∈ [0, 1]; when kappa_total < 0, both kappa and kappa_s share the (non-positive) sign, representing a partitioned action-avoidance budget. Encoded in `KAPPA_FAMILY_PARAMS = {kappa, kappa_s, kappa_total}` (kappa_share excluded) in numpyro_helpers.py.

4. **Test parameterization Option A (loop in-place) for M6b corner cases.** The two existing in-file tests (test_wmrl_m6b_kappa_share_one_matches_m3 at line 869, ..._zero_matches_m6a at line 920 of wmrl_m6b.py) were refactored to loop over `('softmax', 'convex')` inside the function body rather than splitting into 4 functions (Option B). Justification: minimal LOC, same function-name registration in `__all__`, single assertion gates both modes, both modes pass at diff=0.00e+00.

5. **Reference fixture set_sizes is an array, not a scalar.** The plan's Task 0 fixture said `set_size = jnp.array(3)` (scalar). The JAX likelihood functions index `set_sizes` per-trial inside the scan body via `set_sizes[t]`, which fails on a 0-D scalar with `IndexError: tuple index out of range`. Replaced with `jnp.full(10, 3)` (length-10 array of constant value 3). Documented in 32-04-references.json's `_fixture_note` field. Both the Task 0 capture and Task 6.5 regression test use the array form, so reproducibility is maintained.

6. **Task 9 cluster submission deferred to Plan 32-05.** Per orchestrator constraint: this plan completes when CODE/CONFIG/SLURM changes are merged; submission command + manifest documented in `.planning/phases/32-mcmc-methodology-update/32-04-mle-refit-jobs.txt` for human to run on M3. No `sbatch` invocations made from the local Windows dev box.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Task 0 fixture scalar set_size doesn't index correctly**
- **Found during:** Task 0 (reference log-likelihood capture)
- **Issue:** Plan said `set_size = jnp.array(3)` (0-D scalar). JAX likelihood scans index `set_sizes[t]` per-trial; 0-D scalars raise `IndexError: tuple index out of range`.
- **Fix:** Used `set_sizes = jnp.full(10, 3)` (length-10 array of constant value 3) in both Task 0 capture script and Task 6.5 regression test fixture. Reproducibility maintained because both use identical fixture.
- **Files modified:** `.planning/phases/32-mcmc-methodology-update/32-04-references.json` (`_fixture_note` field), `tests/integration/test_kappa_parameterization_regression.py` (toy_block fixture comment).
- **Verification:** Task 0 capture produces 4 finite floats; Task 6.5 regression test 4/4 PASS at diff=0.00e+00.
- **Committed in:** `42da7c4` (Task 0) + `76bcb76` (Task 6.5).

**2. [Rule 1 - Bug] sub-job EXPORTS string in 04a_mle_gpu.slurm dispatch loop missing KAPPA_MODE forwarding**
- **Found during:** Task 7 (SLURM editing)
- **Issue:** `04a_mle_gpu.slurm` has a multi-model dispatch loop (lines ~170-194) that re-submits per-model sub-jobs via `sbatch --export="$EXPORTS" ...`. The original EXPORTS string built `MODEL=$model` only. If KAPPA_MODE were set on the parent job but not threaded into the sub-job EXPORTS, the sub-job would fall back to the default 'softmax' silently — defeating the whole point of `--kappa-parameterization convex` for sensitivity analyses.
- **Fix:** Updated EXPORTS to `MODEL=$model,KAPPA_MODE=$KAPPA_MODE`; per-model echo line tags `(kappa=$KAPPA_MODE)`.
- **Files modified:** `cluster/04a_mle_gpu.slurm` (lines ~178-185).
- **Verification:** `bash -n` exit 0; `grep KAPPA_MODE cluster/04a_mle_gpu.slurm` returns 5 hits (env-var read + banner + dispatch EXPORTS + dispatch echo + CLI thread).
- **Committed in:** `1a994d2` (Task 7).

**3. [Rule 3 - Blocking] M4 hierarchical wrapper wasn't in plan's Task 3 scope but had to be threaded**
- **Found during:** Task 4 (MLE engine threading) — discovered M4 hierarchical wrapper still references the unmodified `kappa` directly via `sample_bounded_param`, breaking under softmax mode.
- **Issue:** Plan Task 3 said "M3/M5/M6a/M6b hierarchical wrappers"; Task 4 said "M4 LBA likelihood threading". But M4's hierarchical wrapper at `src/rlwm/fitting/models/wmrl_m4.py:173` ALSO needed `kappa_parameterization` kwarg + sample_unbounded_normal_param dispatch under softmax mode. Without this, M4 Bayesian (already a separate pipeline) would silently use convex bounds even when user passes `--kappa-parameterization softmax`.
- **Fix:** Added `kappa_parameterization='softmax'` kwarg to `wmrl_m4_hierarchical_model`. Kappa-family params dispatch through `sample_unbounded_normal_param` (clip [-1, 1]) under softmax mode; the per-participant likelihood call now passes `parameterization=kappa_parameterization` to `wmrl_m4_multiblock_likelihood_stacked`.
- **Files modified:** `src/rlwm/fitting/models/wmrl_m4.py` (lines 173, 232-263, 316-339).
- **Verification:** M4 hierarchical wrapper signature inspection shows `kappa_parameterization='softmax'` default. Combined with Task 4 lba_likelihood + mle.py changes, the full M4 pipeline (MLE + Bayesian) now respects the parameterization mode.
- **Committed in:** `94efcc6` (combined with Task 4 MLE engine commit).

---

**Total deviations:** 3 auto-fixed (1 plan-fixture bug, 1 SLURM dispatch bug, 1 plan-scope gap)
**Impact on plan:** All auto-fixes were necessary for correctness. No scope creep — every change directly serves the dual-mode kappa goal.

## Issues Encountered

- **None.** All 9 tasks executed in sequence. Pre-existing ruff warnings (UP037 quoted annotations in config.py, F401/F841/B007/B905 in model files, E712 in mle.py) were verified as baseline-equal both before and after Phase 32-04 edits — no new violations introduced.
- **set_sizes scalar fixture issue** caught at Task 0 (first task) via direct JAX run; resolved before any model-file edits. No downstream impact.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Plan 32-05 (cluster smoke test) is unblocked.** Plan 32-05 owners must:

1. SSH to Monash M3 and pull the v5.0 main branch (commits `42da7c4..1a994d2` + this metadata commit).
2. Trigger the 7-model MLE re-fit + 3-model Bayesian re-fit under softmax mode:
   ```bash
   bash cluster/submit_all.sh --kappa-parameterization softmax \
                              --bayes-models "qlearning wmrl wmrl_m6b"
   ```
3. Validate the smoke gate at the documented endpoints (32-05 PLAN's success criteria — convergence + 1e-3 agreement on synthetic recovery data).
4. The empirical convergence improvement under softmax mode (R-hat reduction on M3/M5/M6a vs the v5.0 multimodal R-hat=1.60 baseline) is **NOT validated locally** — only Plan 32-05's M3-cluster smoke can confirm or refute this hypothesis.

**Concerns / pending verifications:**

- Logit-vs-probability channel-mix asymmetry means softmax-mode log-likelihoods are ~0.17 nats different from convex-mode at kappa=0 on the toy fixture. This is the documented M2-base difference between the two parameterizations. Empirically this difference may be larger or smaller on real data; only Plan 32-05 cluster smoke will reveal whether the softmax-mode LL is competitive with v5.0 convex-mode LL on the N=178 cohort.

- Pre-existing CRLF line-ending warnings on commit (`warning: LF will be replaced by CRLF`) are Windows-local-checkout artefacts and do NOT affect the bytes pushed to GitHub. The Phase 31-04 + 32-03 workflows already documented this; submit_all.sh's `sed -i 's/\r$//' cluster/*.slurm cluster/*.sh` step strips them on the cluster side at runtime.

- v4 closure remained green throughout — both `tests/integration/test_v4_closure.py` (5/5 PASS) and `tests/scientific/check_v4_closure.py` (5/5 PASS) verified at the end of Task 8.

**To trigger MLE re-fit on M3 (Plan 32-05 task):**
```bash
ssh m3
cd ~/scratch/rlwm_trauma_analysis
git pull origin main
bash cluster/submit_all.sh --kappa-parameterization softmax \
                            --bayes-models "qlearning wmrl wmrl_m6b"
```

See `.planning/phases/32-mcmc-methodology-update/32-04-mle-refit-jobs.txt` for the full submission manifest with alternate Option-B per-model commands.

---
*Phase: 32-mcmc-methodology-update*
*Completed: 2026-05-03*
