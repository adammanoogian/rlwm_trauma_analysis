#!/usr/bin/env python
"""
11: Run Model Recovery
======================

Validates the model fitting procedure by generating synthetic data
with known parameters and attempting to recover them via MLE.

This is a pipeline script that wraps the fitting library module.

Purpose:
    - Verify that fitting procedure can recover true parameters
    - Identify parameter identifiability issues
    - Estimate expected parameter uncertainty

Procedure:
    1. Sample parameters from prior distributions
    2. Generate synthetic data with those parameters
    3. Fit model to synthetic data using MLE
    4. Compare recovered vs true parameters
    5. Visualize recovery quality

Inputs:
    - Task configuration (from config.py)
    - Model specification (via --model)

Outputs:
    - output/model_recovery/<model>_recovery_<timestamp>.csv
    - figures/model_recovery/<model>_recovery_scatter.png
    - figures/model_recovery/<model>_recovery_histogram.png

Usage:
    # Basic Q-learning recovery test
    python scripts/11_run_model_recovery.py --model qlearning --n-subjects 50

    # Full WM-RL recovery (slower)
    python scripts/11_run_model_recovery.py --model wmrl --n-subjects 100 --n-jobs 4

    # Quick test
    python scripts/11_run_model_recovery.py --model qlearning --n-subjects 20 --num-trials 50

Interpretation:
    - High correlation (r > 0.8): Good parameter recovery
    - Moderate correlation (0.5 < r < 0.8): Acceptable, but consider more data
    - Low correlation (r < 0.5): Parameter identifiability issues

Next Steps:
    - If recovery is good: proceed to 12_fit_mle.py with real data
    - If recovery is poor: consider model simplification or more trials
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Import the main function from the library module
from scripts.fitting.model_recovery import main

if __name__ == '__main__':
    main()
