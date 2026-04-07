#!/usr/bin/env python
"""
12: Fit MLE
===========

Maximum Likelihood Estimation fitting for computational models.

This is a pipeline script that wraps the fitting library module.

Models Available:
    - qlearning: Q-learning (M1) - 3 parameters (α₊, α₋, ε)
    - wmrl: WM-RL (M2) - 6 parameters (α₊, α₋, φ, ρ, K, ε)
    - wmrl_m3: WM-RL with perseveration (M3) - 7 parameters

Key Features:
    - JAX-accelerated likelihood computation
    - Multi-start optimization via jaxopt.LBFGS
    - Parallel fitting across participants (--n-jobs)
    - Optional GPU acceleration (--use-gpu)
    - AIC/BIC computation for model comparison

Inputs:
    - output/task_trials_long.csv (behavioral data)

Outputs:
    - output/mle/<model>_mle_results.csv (fitted parameters)
    - output/mle/<model>_model_fit.json (fit metadata)

Usage:
    # Fit Q-learning model
    python scripts/12_fit_mle.py --model qlearning --data output/task_trials_long.csv

    # Fit WM-RL model
    python scripts/12_fit_mle.py --model wmrl --data output/task_trials_long.csv

    # Fit WM-RL with perseveration
    python scripts/12_fit_mle.py --model wmrl_m3 --data output/task_trials_long.csv

    # Parallel fitting (faster)
    python scripts/12_fit_mle.py --model wmrl_m3 --data output/task_trials_long.csv --n-jobs 16

    # GPU-accelerated (requires rlwm_gpu environment)
    python scripts/12_fit_mle.py --model wmrl_m3 --data output/task_trials_long.csv --use-gpu

    # Include practice blocks
    python scripts/12_fit_mle.py --model qlearning --data output/task_trials_long_all.csv --include-practice

Cluster Execution:
    # Sequential
    sbatch cluster/run_mle.slurm

    # Parallel (16 cores)
    sbatch cluster/run_mle_parallel.slurm

    # GPU-accelerated
    sbatch cluster/run_mle_gpu.slurm

Next Steps:
    - Run 14_compare_models.py for model selection
    - Run 15_analyze_mle_by_trauma.py for parameter-trauma relationships
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Import the main function from the library module
from scripts.fitting.fit_mle import main

if __name__ == '__main__':
    main()
