---
phase: 05-parameter-recovery
plan: 02
subsystem: validation
tags: [parameter-recovery, visualization, cli, slurm, gpu, jax, matplotlib, seaborn]

# Dependency graph
requires:
  - phase: 05-01
    provides: Core recovery functions (sample_parameters, generate_synthetic_participant, run_parameter_recovery, compute_recovery_metrics)
provides:
  - CLI interface for parameter recovery with argparse
  - CSV output for recovery results and metrics
  - Generic plotting utilities (plot_scatter_with_annotations, plot_kde_comparison)
  - Recovery visualization (scatter plots, KDE overlays)
  - GPU SLURM script for cluster execution
affects: [05-03, phase-6-clinical-params]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Generic plotting utilities with ax-based pattern for composability"
    - "CLI with argparse for user-runnable pipelines"
    - "Exit code convention (0=pass, 1=fail) for automation"

key-files:
  created:
    - cluster/run_recovery_gpu.slurm
  modified:
    - scripts/fitting/model_recovery.py
    - scripts/utils/plotting_utils.py

key-decisions:
  - "Generic plotting utilities follow existing ax-based pattern for composability"
  - "PASS/FAIL badge based on r >= 0.80 threshold (Senta et al., 2025)"
  - "Distribution comparison plots for sanity checking synthetic data realism"
  - "Exit code 0 if all params pass, 1 if any fail (for automation)"

patterns-established:
  - "plot_scatter_with_annotations(): Generic scatter with identity/regression lines, annotations, optional PASS/FAIL badge"
  - "plot_kde_comparison(): Overlapping KDE distributions with auto-color generation"
  - "CLI pipelines return exit codes for automation (0=success, 1=failure)"

# Metrics
duration: 13min
completed: 2026-02-06
---

# Phase 5 Plan 2: Parameter Recovery CLI and Visualization

**User-runnable CLI with argparse, CSV output, and publication-ready scatter/KDE plots using generic plotting utilities**

## Performance

- **Duration:** 13 min
- **Started:** 2026-02-06T12:38:56Z
- **Completed:** 2026-02-06T12:52:16Z
- **Tasks:** 3
- **Files modified:** 3 (1 created, 2 modified)

## Accomplishments

- Complete CLI interface with argparse (--model, --n-subjects, --n-datasets, --use-gpu, --seed)
- CSV outputs (recovery_results.csv, recovery_metrics.csv) in structured format
- Generic plotting utilities added to plotting_utils.py for reuse across analysis scripts
- Recovery visualization with scatter plots (true vs recovered + annotations) and KDE overlays (recovered vs real fitted params)
- GPU SLURM script for cluster execution with configurable parameters

## Task Commits

Each task was committed atomically:

1. **Task 1: Add main() CLI with argparse** - `7c493f0` (feat)
2. **Task 2: Add visualization functions** - `b693523` (feat)
3. **Task 3: Create GPU SLURM script** - `e8a9bdc` (feat)

## Files Created/Modified

- `scripts/fitting/model_recovery.py` - Added main() CLI with argparse, plot_recovery_scatter(), plot_distribution_comparison() functions
- `scripts/utils/plotting_utils.py` - Added plot_scatter_with_annotations() and plot_kde_comparison() generic utilities
- `cluster/run_recovery_gpu.slurm` - Created GPU SLURM script for cluster execution

## Decisions Made

1. **Generic plotting utilities follow ax-based pattern**: Consistent with existing add_colored_scatter() function for composability
2. **PASS/FAIL badge on scatter plots**: Visual indicator based on r >= 0.80 threshold (Senta et al., 2025 methodology)
3. **Distribution comparison plots**: KDE overlays comparing recovered vs real fitted parameters for sanity checking synthetic data realism
4. **Exit code convention**: 0 if all parameters pass recovery (r >= 0.80), 1 if any fail (enables automation in cluster scripts)
5. **Real params path fallback**: Defaults to output/mle_results/{model}_individual_fits.csv, skips distribution plots if not found

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. Pipeline executed smoothly with small test dataset (10 subjects, 1 dataset).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for 05-03:** Wrapper script (11_run_parameter_recovery.py) can now call model_recovery.py CLI for integration with numbered pipeline.

**CLI usage:**
```bash
# Q-learning recovery
python scripts/fitting/model_recovery.py --model qlearning --n-subjects 50 --n-datasets 10

# WM-RL M3 with GPU
python scripts/fitting/model_recovery.py --model wmrl_m3 --n-subjects 100 --n-datasets 10 --use-gpu

# Cluster execution
sbatch cluster/run_recovery_gpu.slurm
sbatch --export=MODEL=qlearning,NSUBJ=100,NDATASETS=10 cluster/run_recovery_gpu.slurm
```

**Outputs:**
- `output/recovery/{model}/recovery_results.csv` - True/recovered params for all subjects
- `output/recovery/{model}/recovery_metrics.csv` - Pearson r, RMSE, bias, pass/fail per parameter
- `figures/recovery/{model}/*_recovery.png` - Scatter plots with annotations
- `figures/recovery/{model}/parameter_distributions.png` - KDE overlays (if real params available)

---
*Phase: 05-parameter-recovery*
*Completed: 2026-02-06*
