# Phase 5: Parameter Recovery - Context

**Gathered:** 2026-02-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Complete parameter recovery pipeline validating MLE fitting quality per Senta et al. (2025). Generate synthetic data from known parameters, fit via MLE, compare recovered vs. true parameters. Report pass/fail against r >= 0.80 criterion.

</domain>

<decisions>
## Implementation Decisions

### GPU/Fitting Integration
- Import fit_single_participant() from fit_mle.py directly (not subprocess)
- Sequential fitting only (no parallelization due to LLVM issues on CPU parallel)
- Add --use-gpu flag to model_recovery.py mirroring fit_mle.py
- Create dedicated run_recovery_gpu.slurm script for cluster execution

### Synthetic Data Format
- Match exact structure of task_trials_long.csv (same columns)
- Keep synthetic data in-memory (DataFrame passed to fitting functions)
- Sample true parameters uniformly from MLE bounds
- Match real data trial structure exactly (21 blocks, 3 stimuli, reversals)

### Recovery Workflow Design
- CLI: `python scripts/fitting/model_recovery.py --model wmrl_m3 --n-subjects N --n-datasets M --use-gpu`
- Default --n-subjects to actual sample size from real data (match study power)
- Script 11 is thin wrapper: imports model_recovery, runs it, evaluates r >= 0.80, prints pass/fail
- tqdm progress bar for dataset fitting progress
- Post-fitting sanity checks:
  - AIC/BIC distribution comparison between real and synthetic (visual sanity check)
  - Distribution overlap plots (KDE) comparing real fitted params vs recovered params

### Output Organization
- Data outputs: `output/recovery/{model}/`
  - recovery_results.csv (wide format: true_param, recovered_param columns per parameter)
  - recovery_metrics.csv (param, pearson_r, rmse, bias, pass_fail columns)
- Figure outputs: `figures/recovery/{model}/`
  - True vs recovered scatter plot per parameter with r, RMSE, bias annotated
  - Distribution overlap plots for real vs recovered comparison

### Claude's Discretion
- Exact implementation of synthetic data generator
- How to handle edge cases in parameter sampling
- Specific tqdm formatting and verbosity levels
- KDE bandwidth selection for distribution plots

</decisions>

<specifics>
## Specific Ideas

- "Do as many real human subjects were run" — match actual sample size for --n-subjects default
- Compare AIC/BIC between real and synthetic fits as sanity check that simulation is realistic
- Distribution overlap plots (KDE) to visually compare real fitted parameters vs recovered parameters
- Sequential GPU fitting preferred over CPU parallel due to LLVM compilation issues

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-parameter-recovery*
*Context gathered: 2026-02-06*
