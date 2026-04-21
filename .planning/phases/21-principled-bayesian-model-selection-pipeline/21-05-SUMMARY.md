---
phase: 21-principled-bayesian-model-selection-pipeline
plan: 05
subsystem: infra
tags: [bayesian, arviz, convergence, rhat, ess, bfmi, ppc, baribault-collins-2023, slurm, pipeline-gate]

# Dependency graph
requires:
  - phase: 21-04
    provides: "output/bayesian/21_baseline/{model}_posterior.nc + {model}_ppc_results.csv written to the same subdir via Approach A ppc_output_dir plumbing"
  - phase: 21-01
    provides: "Baribault & Collins (2023) convergence-gate pattern + SLURM conda ladder + autopush template"
  - phase: 16
    provides: "MODEL_REGISTRY[model]['params'] list consumed by az.summary var_names"
provides:
  - "scripts/21_baseline_audit.py — convergence + PPC audit over the 6 baseline posteriors"
  - "cluster/21_4_baseline_audit.slurm — 30 min / 8 G / 2 CPU SLURM submission with exit-code propagation"
  - "Baribault & Collins (2023) gate as HARD PIPELINE BLOCK: R-hat <= 1.05 AND ESS_bulk >= 400 AND divergences == 0 AND BFMI >= 0.2"
  - "Pipeline action schema: PROCEED_TO_LOO / EXCLUDED_RHAT / EXCLUDED_ESS / EXCLUDED_DIVERGENCES / EXCLUDED_BFMI / EXCLUDED_MISSING_FILE"
  - "Machine-readable convergence_table.csv + human-readable convergence_report.md"
  - "WARNING_FILE_MISSING surfacing for missing PPC CSVs (upstream bug signal, not silent swallow)"
affects: [21-06-loo-stacking, 21-07-winner-l2-refit, 21-10-master-orchestrator]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Convergence gate as SLURM --dependency=afterok chain-breaker (non-zero exit blocks step 21.5)"
    - "Pipeline-action enum per model (PROCEED_TO_LOO / EXCLUDED_*) as the interface between 21.4 and 21.5/21.6"
    - "Loud surfacing of upstream bugs: missing PPC CSV after 21-04 fix -> WARNING_FILE_MISSING, never silent 'not_available'"
    - "Precedence-ordered exclusion reason (R-hat > ESS > divergences > BFMI) to match NUTS diagnostic workflow"

key-files:
  created:
    - "scripts/21_baseline_audit.py"
    - "cluster/21_4_baseline_audit.slurm"
  modified: []

key-decisions:
  - "Use az.summary(idata, var_names=MODEL_REGISTRY[model]['params']) rather than scanning all posterior vars — keeps the gate focused on scientifically-meaningful individual-level params, not arviz auxiliaries like 'diverging'"
  - "BFMI precedence LOWEST among the 4 criteria (R-hat > ESS > divergences > BFMI) — matches standard NUTS diagnosis workflow where BFMI is the most specific signal and usually downstream of the others"
  - "MIN_MODELS_FOR_STACKING=2 (not 1, not 3) — stacking over a singleton is not meaningful (reduces to posterior averaging of one model); RFX-BMS PXP also requires >= 2 to have a non-trivial ranking; 3 would over-constrain Phase 21 SC #2"
  - "WARNING_FILE_MISSING is an in-band table value (not silent 'not_available') + dedicated WARNINGS section in the report — 21-04's Approach A PPC plumbing guarantees the file exists when the NetCDF exists, so absence signals an upstream bug that must surface loudly"
  - "EXCLUDED_MISSING_FILE is NOT counted in n_excluded for gate logic (logged as warning only) — matches plan spec rationale: missing files usually mean the upstream step.3 SLURM job failed, not a convergence issue per se, so they don't block via the <2-passing rule on their own (but they still appear as FAIL in the table for transparency)"
  - "Exit-code propagation via `exit $EXIT_CODE` at the end of the SLURM script — load-bearing for the plan 21-10 master orchestrator's --dependency=afterok chain to step 21.5"

patterns-established:
  - "Pattern: per-model convergence audit via az.from_netcdf + az.summary + az.bfmi + pandas PPC read, assembled into a ModelAudit dataclass, written as both CSV (machine) and MD (human)"
  - "Pattern: pipeline-gate SLURM scripts use 2-CPU/8G budgets + ds_env ladder (no JAX needed for pure-ArviZ work)"
  - "Pattern: surfacing upstream bugs loudly — when an artefact absence is guaranteed impossible by the upstream fix, treat its absence as a WARNING with a dedicated report section, not a silent NA"

# Metrics
duration: ~20min
completed: 2026-04-18
---

# Phase 21 Plan 05: Baseline Convergence Audit Summary

**Hard pipeline gate between step 21.3 baseline fits and step 21.5 PSIS-LOO + stacking: `scripts/21_baseline_audit.py` applies the Baribault & Collins (2023) gate (R-hat <= 1.05 AND ESS_bulk >= 400 AND divergences == 0 AND BFMI >= 0.2) over all 6 baseline posteriors and exits 1 when < 2 models pass — SLURM `--dependency=afterok` chains block naturally.**

## Performance

- **Duration:** ~20 min (both tasks on local Windows dev)
- **Started:** 2026-04-18 (late afternoon)
- **Completed:** 2026-04-18
- **Tasks:** 2/2
- **Files created:** 2 (`scripts/21_baseline_audit.py`, `cluster/21_4_baseline_audit.slurm`)
- **Files modified:** 0

## Accomplishments

- `scripts/21_baseline_audit.py` (~500 lines, well above the 120-line floor) loads each of the 6 baseline NetCDFs via `arviz.from_netcdf`, runs `az.summary(var_names=MODEL_REGISTRY[model]['params'])` for max R-hat / min ESS_bulk, reads `idata.sample_stats.diverging.values.sum()` (with a 0 fallback for non-MCMC InferenceData), calls `az.bfmi()` for per-chain values and takes the minimum, and reads `{model}_ppc_results.csv` to compute block-level 95% envelope coverage.
- Pipeline-action enum encoded as 6 values: `PROCEED_TO_LOO` (gate PASSES), `EXCLUDED_MISSING_FILE`, `EXCLUDED_RHAT`, `EXCLUDED_ESS`, `EXCLUDED_DIVERGENCES`, `EXCLUDED_BFMI`. Precedence for the 4 failure reasons: R-hat > ESS > divergences > BFMI (matches standard NUTS diagnostic workflow).
- Missing PPC CSV despite the plan-21-04 Approach-A fix surfaces as `ppc_coverage="WARNING_FILE_MISSING"` in the CSV **AND** a dedicated `## WARNINGS` section in the MD report — the 21-04 fix makes absence impossible for converged models, so any missing PPC signals an upstream bug that must not be silently swallowed as `"not_available"`.
- Outputs: `output/bayesian/21_baseline/convergence_table.csv` (8 columns: `model, max_rhat, min_ess_bulk, n_divergences, min_bfmi, ppc_coverage, gate_status, pipeline_action`) + `output/bayesian/21_baseline/convergence_report.md` (top Summary block with `n_passing` / `n_excluded` / `models_proceeding_to_loo` / `models_excluded` lists, optional WARNINGS section, per-model sections with metrics + pipeline action + free-text notes).
- Exit logic: `n_passing >= 2` -> exit 0 (pipeline continues); `n_passing < 2` -> exit 1 with a `[PIPELINE BLOCK] Only <n> models passed convergence gate; step 21.5 requires >= 2` message on stderr. Missing NetCDFs are logged as warnings but not counted for the gate (consistent with plan spec: step 21.3 SLURM failures don't double-block via convergence gate).
- `cluster/21_4_baseline_audit.slurm` (137 lines): 30 min / 8 G / 2 CPU / comp partition; ds_env -> `/scratch/fc37` conda ladder; no JAX required (pure ArviZ / pandas); `mkdir -p output/bayesian/21_baseline logs`; overridable `RHAT` / `ESS` / `BFMI` thresholds via `--export`; captures `$?` into `EXIT_CODE`; prints `[PIPELINE BLOCK] Convergence gate failed` on non-zero; `source cluster/autopush.sh` to push logs + reports; `exit $EXIT_CODE` is load-bearing for the plan-21-10 master orchestrator's `--dependency=afterok` chain to step 21.5.

## Task Commits

1. **Task 1: Implement scripts/21_baseline_audit.py** — `83de94f` (feat)
2. **Task 2: Create cluster/21_4_baseline_audit.slurm** — `c64e622` (feat)

## Files Created/Modified

**Created:**
- `scripts/21_baseline_audit.py` — convergence + PPC audit with 4-criterion gate, 6 pipeline actions, MD + CSV outputs
- `cluster/21_4_baseline_audit.slurm` — SLURM submission with exit-code propagation

## Decisions Made

- **az.summary restricted to `MODEL_REGISTRY[model]['params']`:** Scanning all posterior variables would include bookkeeping like `mu_pr`, `sigma_pr`, `z`, and nondiagnostic transforms, which inflate the max-R-hat calculation and can mask real convergence issues in the scientifically-meaningful per-participant parameters. Using the registered params list aligns the gate with the parameters downstream LOO / stacking actually use.
- **BFMI last in precedence (R-hat > ESS > divergences > BFMI):** R-hat is the most universal non-convergence signal (the first thing a user looks at); ESS is second because it determines the reliability of quantile estimates; divergences are third because auto-bump usually resolves them but residuals indicate geometry issues; BFMI is last because it's the most specific signal (heavy-tailed energy distribution) and usually appears downstream of the other three.
- **`MIN_MODELS_FOR_STACKING = 2` (not 1, not 3):** Stacking over a singleton reduces to posterior averaging of one model — meaningless for Phase 21's model-space inference. RFX-BMS PXP requires ≥2 models for a non-trivial ranking. Requiring ≥3 would over-constrain Phase 21 SC #2 (which only requires "models converge or are explicitly dropped", not a minimum count).
- **`WARNING_FILE_MISSING` is in-band, not NA:** Plan 21-04's Approach A fix plumbs `ppc_output_dir` through `save_results` -> `run_posterior_predictive_check`, which means any model passing step 21.3's gate MUST have produced the PPC CSV. Absence post-21-04 is therefore an upstream bug, not expected-missing. Reporting it as silent `"not_available"` would hide the bug; treating it as a distinct WARNING with a dedicated report section makes it impossible to miss.
- **EXCLUDED_MISSING_FILE doesn't double-count against gate:** Missing NetCDFs usually mean the upstream step-21.3 SLURM job crashed, not that the model failed convergence. Counting them in `n_excluded` would conflate two different failure modes. Plan spec was explicit: "Missing models logged as warnings but don't fail immediately — allow the pipeline to proceed if >= 2 passes". The table still shows them as FAIL with action `EXCLUDED_MISSING_FILE` for transparency.
- **Exit-code propagation via `exit $EXIT_CODE` at end of SLURM:** Plan 21-10 master orchestrator chains step 21.5 via `--dependency=afterok:$AUDIT_JOBID`. Without explicit propagation the `source cluster/autopush.sh` in between would mask the Python script's non-zero exit (autopush exits 0 on success). This is the same pattern used in `21_1_prior_predictive.slurm` and `21_3_fit_baseline.slurm`.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## Verification

- [x] `python scripts/21_baseline_audit.py --help` prints all 5 expected args (`--baseline-dir`, `--output-dir`, `--rhat-threshold`, `--ess-threshold`, `--bfmi-threshold`)
- [x] Smoke test (stub NetCDF via `arviz.from_dict` with 4 chains × 500 draws, 3 params, `qlearning`) produces `convergence_table.csv` + `convergence_report.md`, and the `qlearning` row shows `gate_status=PASS` / `pipeline_action=PROCEED_TO_LOO` / PPC coverage `10/12 (83.3%)`
- [x] Smoke test with 5 missing models + 1 passing -> exit code 1 (`[PIPELINE BLOCK] Only 1 models passed...`)
- [x] Smoke test with 2 passing models (qlearning + wmrl stubs) -> exit code 0 (`[STEP 21.4 COMPLETE] 2 models proceeding to LOO`)
- [x] Smoke test with missing PPC CSV despite passing NetCDF -> `ppc_coverage=WARNING_FILE_MISSING` in table + `## WARNINGS` section in MD + exit 0 preserved (PPC absence doesn't affect convergence gate)
- [x] `grep -c "21_baseline" cluster/21_4_baseline_audit.slurm` returns **11** (well above the ≥2 requirement)
- [x] SLURM script has `exit $EXIT_CODE` as the final line

## Next Phase Readiness

- **Unblocks 21-06 (PSIS-LOO + stacking + RFX-BMS):** Reads `convergence_table.csv` (or filters on `pipeline_action == "PROCEED_TO_LOO"`) to determine which models go into the stacking ranking. Non-converged models are excluded by construction — no circular MLE-winner bias.
- **Unblocks 21-10 (master orchestrator):** The `21_4_baseline_audit.slurm` script is now submittable with `--dependency=afterok:$FIT_ARRAY_JOBID` after the 6 parallel step-21.3 fits complete. A non-zero exit here automatically blocks step 21.5 via the master orchestrator's `--dependency=afterok:$AUDIT_JOBID` chain.
- **Gate thresholds ready for sensitivity sweeps:** `RHAT` / `ESS` / `BFMI` exposed as SLURM env vars (`--export=RHAT=1.01,ESS=500`) if we later need to test more conservative settings (e.g., Vehtari et al. 2021's 1.01 recommendation instead of the 1.05 operational threshold).

---
*Phase: 21-principled-bayesian-model-selection-pipeline*
*Completed: 2026-04-18*
