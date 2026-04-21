---
phase: 21-principled-bayesian-model-selection-pipeline
plan: 07
subsystem: bayesian-fitting
tags: [numpyro, jax, hierarchical, level2-regression, subscale, slurm, phase21]

# Dependency graph
requires:
  - phase: 21-principled-bayesian-model-selection-pipeline/21-06
    provides: winners.txt (comma-separated display names) consumed by step 21.6 to validate --model before refit
  - phase: 21-principled-bayesian-model-selection-pipeline/21-11
    provides: covariate_iesr kwarg on M3/M5/M6a hierarchical models + build_level2_design_matrix_2cov helper
  - phase: 21-principled-bayesian-model-selection-pipeline/21-04
    provides: output_subdir plumbing through save_results, write_bayesian_summary, run_posterior_predictive_check
  - phase: 16-l2-subscale
    provides: wmrl_m6b_hierarchical_model_subscale + fit_bayesian.main --subscale CLI flag
provides:
  - scripts/21_fit_with_l2.py winner-refit orchestrator (~661 lines) with 3-branch L2 dispatch (copy / 2-cov / subscale)
  - cluster/21_6_fit_with_l2.slurm 12h/64G/4-CPU/comp submission template (per-MODEL JAX cache, winners.txt validation)
  - Load-bearing expected.exists() convergence-gate-surface shim for --dependency=afterok chain propagation
affects:
  - 21-08 model averaging (consumes {winner}_posterior.nc from output/bayesian/21_l2/)
  - 21-10 master pipeline orchestrator (submits this SLURM per-winner with --dependency=afterok:$LOO_JOBID)
  - v5.0 sensitivity analysis (4-covariate subscale extension for M3/M5/M6a — deferred)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Three-branch L2 dispatch pattern (copy / 2-cov / subscale) driven by frozenset-membership checks on internal model id"
    - "Local model_args shim — assemble dict with covariate_iesr in the orchestrator rather than patching fit_bayesian._fit_stacked_model which hard-codes covariate_lec only"
    - "Post-fit expected.exists() check as convergence-gate-surface shim (mirrors 21_fit_baseline.py from plan 21-04)"
    - "winners.txt display-name <-> internal-id bidirectional validation (reject unknown + reject not-in-winner-set)"

key-files:
  created:
    - scripts/21_fit_with_l2.py
    - cluster/21_6_fit_with_l2.slurm
    - .planning/phases/21-principled-bayesian-model-selection-pipeline/21-07-SUMMARY.md
  modified:
    - .planning/STATE.md

key-decisions:
  - "Option (a) local model_args shim over Option (b) _fit_stacked_model patch — keeps blast radius inside scripts/21_fit_with_l2.py; fit_bayesian.py is exercised by every downstream fit"
  - "M1/M2 winners use shutil.copy2 pass-through over a cheap re-fit — avoids wasted cluster cycles producing an identical posterior"
  - "12h SLURM cap fits all three tiers (M6b subscale worst ~8-12h, M3/M5/M6a 2-cov ~45-75min, M1/M2 copy ~1s)"
  - "Verify both beta_lec_{target} AND beta_iesr_{target} land in saved NetCDF for M3/M5/M6a — abort with exit 1 if either missing (target=kappa for M3/M5, kappa_s for M6a)"
  - "Per-MODEL JAX compilation cache (.jax_cache_21_l2_${MODEL}) prevents cross-winner collisions when multiple refit jobs run in parallel"

patterns-established:
  - "L2 dispatch by frozenset membership: _COPY_BASELINE_MODELS {qlearning, wmrl}, _TWO_COV_MODELS {wmrl_m3, wmrl_m5, wmrl_m6a}, _SUBSCALE_MODELS {wmrl_m6b}"
  - "Winners.txt consumption: parse comma-separated display names, map via DISPLAY_TO_INTERNAL, require --model to be in the winner set before any fitting starts"
  - "Deferred imports (jax/numpyro inside _fit_two_covariate_l2) keep --help path free of slow startup"

# Metrics
duration: 18min
completed: 2026-04-18
---

# Phase 21 Plan 07: Winner L2 Refit Orchestrator Summary

**Per-winner L2 dispatcher (M1/M2 copy, M3/M5/M6a 2-cov via covariate_iesr, M6b subscale) writing to output/bayesian/21_l2/ with 12h SLURM submission template.**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-04-18
- **Completed:** 2026-04-18
- **Tasks:** 2
- **Files created:** 2 (script + SLURM)

## Accomplishments

- `scripts/21_fit_with_l2.py` (661 lines) — reads `output/bayesian/21_baseline/winners.txt`, validates `--model` against the winner set in both directions, dispatches to one of three L2 variants. Copy path uses `shutil.copy2`. 2-cov path assembles `model_args` locally with both `covariate_lec` and `covariate_iesr` and calls `run_inference_with_bump` + `save_results(output_subdir="21_l2")` directly — avoids patching `fit_bayesian._fit_stacked_model` which hard-codes single-covariate. Subscale path delegates to `fit_bayesian.main()` via `sys.argv` rewrite with `--subscale --output-subdir 21_l2`.
- `cluster/21_6_fit_with_l2.slurm` (196 lines) — 12h/64G/4-CPU/comp partition with `MODEL`/`WARMUP`/`SAMPLES`/`CHAINS`/`SEED`/`MAX_TREE_DEPTH`/`WINNERS_FILE` env-var overrides, ds_env conda ladder, per-MODEL JAX cache at `.jax_cache_21_l2_${MODEL}`, expected-artifacts audit loop, `source cluster/autopush.sh`, `exit $EXIT_CODE` (load-bearing for master orchestrator's afterok chain).
- Load-bearing convergence-gate-surface shim: post-fit `expected.exists()` check converts `save_results`'s silent-None-return on HIER-07 gate failure into a non-zero exit so SLURM `--dependency=afterok` chains do not silently skip to step 21.7 (mirrors plan 21-04 `21_fit_baseline.py` pattern).

## Task Commits

1. **Task 1: Winner L2 refit orchestrator** — `118c06f` (feat)
2. **Task 2: SLURM for step 21.6** — `c5a8433` (feat)

## Files Created/Modified

- `scripts/21_fit_with_l2.py` — 3-branch L2 dispatch orchestrator; reads winners.txt, validates --model, routes to copy / 2-cov / subscale; post-fit expected.exists() gate-surface shim.
- `cluster/21_6_fit_with_l2.slurm` — 12h submission template; parameterized MODEL env var; writes exclusively under output/bayesian/21_l2/; exit-code propagation for afterok chain.

## Decisions Made

- **Option (a) local shim over Option (b) _fit_stacked_model patch for 2-cov path.** `fit_bayesian._fit_stacked_model` currently builds `model_args` with only `covariate_lec` (hard-coded). Patching it would force every Phase 21 fit through the new path; building the dict locally in `_fit_two_covariate_l2` keeps blast radius to this orchestrator only and leaves every other downstream fit untouched.
- **Copy-through for M1/M2 preferred over cheap re-fit.** `shutil.copy2` of the already-converged 21_baseline posterior is `O(seconds)` and produces an identical posterior. A re-fit would waste cluster cycles for zero scientific content (M1/M2 have no L2-compatible parameter target — no kappa, no kappa_s, no stick-breaking).
- **Verify both beta_lec_{target} AND beta_iesr_{target} in saved NetCDF for M3/M5/M6a.** Target is `kappa` for M3/M5 and `kappa_s` for M6a. If either site is missing after save, exit 1 — this guards against a scenario where the model function silently dropped one covariate due to a JAX shape mismatch or None-guard falling through.
- **12h SLURM cap absorbs all three tiers.** M6b subscale worst case ~8-12h dictates the upper bound. M3/M5/M6a 2-cov ~45-75min and M1/M2 copy ~1s sit comfortably below. Using a single cap avoids branching SLURM templates by model family.
- **Per-MODEL JAX cache path.** `.jax_cache_21_l2_${MODEL}` prevents compilation-cache corruption when two winners (e.g., M3 and M6b) run in parallel with different model functions — mirrors the plan 21-04 pattern at `.jax_cache_21_baseline_${MODEL}`.

## Deviations from Plan

None — plan executed exactly as written. The plan pre-emptively warned to inspect actual signatures of `run_inference_with_bump` and `save_results`:

- Confirmed `run_inference_with_bump(model, model_args, num_warmup, num_samples, num_chains, seed, target_accept_probs, max_tree_depth)` — argument names match plan snippet.
- Confirmed `save_results(mcmc, data, model, output_dir, save_plots, participant_data_stacked, use_pscan, *, output_subdir)` — requires `data` (pd.DataFrame) as second positional arg for `n_trials_per_ppt` computation. Added local `load_and_prepare_data` call in the 2-cov path to produce `data` before calling `save_results`. This was not an auto-fix (Rule 1/2/3) — it was a straightforward signature match, documented in the `_fit_two_covariate_l2` body.

## Issues Encountered

- Initial dry-run attempt via `exec(open(...).read())` failed with `NameError: __file__ is not defined` because `_THIS_FILE = Path(__file__).resolve()` triggers at module load. Switched to `subprocess.run([sys.executable, 'scripts/21_fit_with_l2.py', ...])` — canonical invocation; copy-through dry-run exited 0 and produced identical bytes at `output/bayesian/21_l2/qlearning_posterior.nc`.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Step 21.7 (plan 21-08, model averaging / posterior summarization) can consume `output/bayesian/21_l2/{winner}_posterior.nc` for each winner in `winners.txt`.
- The plan 21-10 master pipeline orchestrator is now unblocked for its Wave 6 step: loop over `winners.txt`, `sbatch --dependency=afterok:$LOO_JOBID --export=MODEL=$m cluster/21_6_fit_with_l2.slurm` per entry.
- 4-covariate subscale extension for M3/M5/M6a (adding `iesr_intr_resid` + `iesr_avd_resid`) documented as v5.0 deferral — current Phase 21 pipeline executes Option C middle ground only.

---
*Phase: 21-principled-bayesian-model-selection-pipeline*
*Completed: 2026-04-18*
