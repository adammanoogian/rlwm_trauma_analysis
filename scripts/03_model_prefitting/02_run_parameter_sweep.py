#!/usr/bin/env python
"""
10: Run Parameter Sweep
=======================

Systematically explores model parameter space to understand how different
parameter combinations affect task performance.

This is a pipeline script that wraps the simulations library module.

Key Features:
    - Grid or random sampling of parameter space
    - Parallelized execution
    - Automatic visualization of results
    - Support for both Q-learning and WM-RL models

Inputs:
    - Task configuration (from config.py)
    - Parameter ranges (specified via args or defaults)

Outputs:
    - models/parameter_exploration/<model>_sweep_<timestamp>.csv
    - reports/figures/parameter_sweeps/<model>_heatmaps.png
    - reports/figures/parameter_sweeps/<model>_marginal_effects.png

Usage:
    # Quick sweep of Q-learning parameters
    python scripts/03_model_prefitting/02_run_parameter_sweep.py --model qlearning --n-samples 100

    # Full sweep with parallel execution
    python scripts/03_model_prefitting/02_run_parameter_sweep.py --model both --n-samples 500 --n-jobs -1

    # WM-RL focused on capacity effects
    python scripts/03_model_prefitting/02_run_parameter_sweep.py --model wmrl --n-samples 200 --set-sizes 3 5 6

    # Test with fewer trials for speed
    python scripts/03_model_prefitting/02_run_parameter_sweep.py --model qlearning --n-samples 50 --num-trials 30

Next Steps:
    - Review heatmaps to understand parameter-performance relationships
    - Identify optimal parameter ranges for model fitting
    - Run 03_run_model_recovery.py to validate fitting procedure
"""

import sys
from pathlib import Path

# Add project root to path (parents[2] = project root; parents[1] = scripts/)
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
# Ensure `src/` is importable (so `import rlwm` works without installation).
src_root = project_root / "src"
if src_root.exists():
    sys.path.insert(0, str(src_root))

# Import the main function from the library module
from scripts.legacy.simulations.parameter_sweep import main

if __name__ == '__main__':
    main()
