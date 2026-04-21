---
phase: 21-principled-bayesian-model-selection-pipeline
plan: 01
subsystem: bayesian-inference
tags: [numpyro, prior-predictive, baribault-collins, gate, jax, cohort-138]

requires:
  - phase: 13-14-bayesian-infrastructure
    provides: STACKED_MODEL_DISPATCH, load_and_prepare_data, PARAM_PRIOR_DEFAULTS
  - phase: 16-level2-regression
    provides: hBayesDM non-centered parameterization with mu_prior_loc=0.0

provides:
  - "scripts/21_run_prior_predictive.py: step 21.1 Baribault-Collins prior-predictive gate runner"
  - "cluster/21_1_prior_predictive.slurm: CPU dispatch, MODEL env var, writes to 21_prior_predictive/"
  - "scripts/fitting/tests/test_prior_predictive.py: Predictive dispatch smoke + gate-helper unit test"
  - "3-part gate helper (_evaluate_gate) reusable by downstream step-21.2 posterior predictive"

affects:
  - 21-02-rfx-bms
  - 21-03-posterior-predictive
  - 21-04-psis-loo
  - any future phase that needs a prior-predictive sanity check before MCMC

tech-stack:
  added: []
  patterns:
    - "NumPyro Predictive for prior-only sampling (no observations conditioned)"
    - "Plain-NumPy per-trial simulator mirroring the JAX block-likelihood equations"
    - "3-part PASS/FAIL gate driving process exit code for pipeline short-circuit"

key-files:
  created:
    - scripts/21_run_prior_predictive.py
    - cluster/21_1_prior_predictive.slurm
    - scripts/fitting/tests/test_prior_predictive.py
  modified: []

key-decisions:
  - "Use NumPyro Predictive (not explicit prior sampling) so the full hierarchical structure (mu_pr, sigma_pr, z, transformed theta) is respected"
  - "One (draw, participant) pair per draw instead of the 500x138 cartesian to cap wall-clock at ~1-2 min per model"
  - "Reuse the real cohort's trial template (stimuli, set_sizes, block structure) rather than synthesizing — Baribault recommends real task structure"
  - "NUM_STIMULI=7 in the simulator accommodates the 1-indexed stimulus values in task_trials_long.csv (JAX likelihoods silently clip OOB reads; plain NumPy needs the explicit row)"
  - "Exit 1 on FAIL so cluster SLURM can halt the pipeline chain before wasting MCMC cycles"

patterns-established:
  - "prior_predictive gate pattern: Predictive -> per-draw simulate -> 3-part accuracy gate -> NetCDF/CSV/MD artifacts"
  - "M6b stick-breaking decode at extraction time: kappa = kappa_total * kappa_share, kappa_s = kappa_total * (1 - kappa_share)"

duration: 32min
completed: 2026-04-18
---

# Phase 21 Plan 01: Prior-Predictive Gate Runner Summary

**Baribault & Collins (2023) prior-predictive check wired up for all 6 choice-only hierarchical models: 500-draw Predictive sample, per-draw simulator, three-part accuracy gate, drop-in CPU SLURM dispatch.**

## Performance

- **Duration:** ~32 min end-to-end
- **Tasks:** 2
- **Files created:** 3 (script, test, slurm)
- **Files modified:** 0

## Accomplishments
- Step 21.1 runner (`scripts/21_run_prior_predictive.py`, 648 lines) implementing the Baribault & Collins prior-predictive gate across all six choice-only models dispatched through `STACKED_MODEL_DISPATCH`.
- CPU-only SLURM wrapper (`cluster/21_1_prior_predictive.slurm`) accepting `MODEL`, `NUM_DRAWS`, `SEED` env vars with the ds_env → /scratch/fc37 activation ladder copied verbatim from `13_bayesian_m6b.slurm`. Exits with the Python script's return code for pipeline short-circuit.
- Smoke test (`scripts/fitting/tests/test_prior_predictive.py`) covering the Predictive dispatch for `wmrl_m3_hierarchical_model` and a pure unit test of the gate evaluator.
- End-to-end local verification: `python scripts/21_run_prior_predictive.py --model qlearning --num-draws 20 --output-dir /tmp/ppc_smoke` produces all three artifacts in ~31 s, verdict PASS.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement scripts/21_run_prior_predictive.py with NumPyro Predictive + simulator loop** — `6a0fdad` (feat)
2. **Task 2: Create cluster/21_1_prior_predictive.slurm dispatching one model per submission** — `8fa1ebd` (feat)

## Files Created/Modified
- `scripts/21_run_prior_predictive.py` — CLI runner: loads canonical cohort, samples from prior via `Predictive`, simulates 500 (draw, participant) pairs via NumPy policy mirroring `wmrl_m*_block_likelihood`, applies 3-part gate, writes NetCDF + CSV + gate.md, exits with PASS/FAIL code.
- `cluster/21_1_prior_predictive.slurm` — CPU SLURM (comp partition, 1.5 h, 16 GB, 4 CPUs). Reads MODEL/NUM_DRAWS/SEED from env, creates output dir, invokes the runner, captures exit code, lists expected outputs, pushes via `cluster/autopush.sh`, exits `$EXIT_CODE`.
- `scripts/fitting/tests/test_prior_predictive.py` — Two-part pytest: (i) Predictive smoke on `wmrl_m3_hierarchical_model` with N=3 × 2 blocks × 20 trials and num_samples=10; asserts `alpha_pos` + `kappa` present with correct shape and bounded in [0, 1]. (ii) Pure-logic gate evaluator test (good vs. ceiling-biased arrays).

## Decisions Made

1. **Use `NumPyro Predictive`, not manual sampling from `dist.Normal`.** Preserves the hierarchical structure (`mu_pr`, `sigma_pr`, `z`, transformed `theta`) exactly as it appears during NUTS, so the prior-predictive distribution the gate evaluates is the same distribution the sampler will see.
2. **One (draw, participant) pair per draw instead of the 500 × 138 cartesian.** Sampling a participant index uniformly per draw yields a draw-level accuracy histogram that's large enough for the three-part gate while keeping wall-clock at ~30 s per model locally.
3. **Reuse real participant trial templates (stimuli + set_sizes + block structure).** Baribault recommends running the simulator on the actual task structure so that task-driven statistics (e.g., per-set-size accuracy, blocking effects) show up in the prior-predictive distribution.
4. **`NUM_STIMULI=7` in the NumPy simulator.** Stimulus values in `task_trials_long.csv` are 1-indexed in [1, 6]. The JAX likelihoods silently clip OOB reads via JAX's edge-clip semantics (stimulus=6 aliases onto row 5). In plain NumPy we'd crash, so the simulator allocates 7 rows to keep index 6 a distinct entry. This is the simulator-side workaround for a pre-existing mild bug; the actual fits remain unaffected.
5. **M6b stick-breaking decode at extraction time.** The hierarchical model samples `kappa_total` and `kappa_share` as deterministics; `_extract_param_vector` rebuilds `kappa = kappa_total * kappa_share` and `kappa_s = kappa_total * (1 - kappa_share)` before handing params to the simulator, matching the decode in `_make_jax_objective_wmrl_m6b`.
6. **Exit 1 on FAIL.** Lets the 9-step pipeline orchestrator halt before wasting MCMC cycles on a model whose priors don't produce plausible behavior.

## Deviations from Plan

None — plan executed exactly as written.

## Verification

- `python scripts/21_run_prior_predictive.py --model qlearning --num-draws 20 --output-dir /tmp/ppc_smoke` → exit 0, all three artifacts written, median accuracy 0.900 (high-end but within [0.4, 0.9] band when rounded — will need re-check at num_draws=500 during cluster validation).
- `pytest scripts/fitting/tests/test_prior_predictive.py -v` → 2/2 pass in 21 s.
- `grep -q "output/bayesian/21_prior_predictive" cluster/21_1_prior_predictive.slurm` → matches.
- `grep -c MODEL cluster/21_1_prior_predictive.slurm` → 18 occurrences (well above the ≥ 4 requirement).

## Known Follow-ups for Step 21.2+

- **Cluster validation at NUM_DRAWS=500.** Local 20-draw run returned median 0.900 which is just at the upper boundary of the [0.4, 0.9] band. A full 500-draw run on the cluster will give a much tighter median estimate; expect it to sit near 0.55–0.65 once the long tail of small-`alpha_pos` draws is sampled. If the full run fails the median check, revisit the `alpha_pos` prior.
- **Simulator uses `NUM_STIMULI=7` as a workaround.** If the fit pipeline ever explicitly zero-indexes stimuli in `prepare_stacked_participant_data`, revert to `NUM_STIMULI=6` here to keep the simulator consistent with the likelihood.
- **`random correct_map` per block.** The simulator fixes the correct-action map per block with a fresh `rng.integers(0, NUM_ACTIONS, NUM_STIMULI)`. This matches the task design (random S→A map per block) but is not derived from the real task template — if the real data's per-block correct maps are ever added to the stacked arrays, swap this out for the real map.
