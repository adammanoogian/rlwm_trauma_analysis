#!/usr/bin/env python
"""
13: Fit Bayesian
================

Bayesian hierarchical model fitting using NumPyro NUTS.

This is a pipeline script that wraps the fitting library module.

Models Available:
    - qlearning: Q-learning (M1) [full Bayesian]
    - wmrl: WM-RL (M2) [full Bayesian]
    - wmrl_m3: WM-RL+kappa (M3) [MLE only -- Bayesian not yet implemented]
    - wmrl_m5: WM-RL+phi_rl (M5) [MLE only]
    - wmrl_m6a: WM-RL+kappa_s (M6a) [MLE only]
    - wmrl_m6b: WM-RL+dual (M6b) [MLE only]
    - wmrl_m4: RLWM-LBA (M4) [MLE only]

Key Features:
    - Full posterior distributions (not just point estimates)
    - Hierarchical structure: individual + group-level parameters
    - NUTS sampler for efficient posterior exploration
    - ArviZ integration for diagnostics and visualization

Advantages over MLE:
    - Uncertainty quantification for all parameters
    - Better handling of individual differences
    - WAIC/LOO for principled model comparison
    - Posterior predictive checks

Inputs:
    - output/task_trials_long.csv (behavioral data)

Outputs:
    - output/bayesian_fits/<model>_trace.nc (ArviZ InferenceData)
    - output/bayesian_fits/<model>_summary.csv (parameter summary)
    - figures/bayesian_fits/<model>_posterior.png

Usage:
    # Fit Q-learning (recommended first - faster)
    python scripts/13_fit_bayesian.py --model qlearning --data output/task_trials_long.csv

    # Fit WM-RL
    python scripts/13_fit_bayesian.py --model wmrl --data output/task_trials_long.csv

    # Custom MCMC settings
    python scripts/13_fit_bayesian.py --model qlearning --chains 4 --warmup 1000 --samples 2000

    # Include practice blocks
    python scripts/13_fit_bayesian.py --model qlearning --data output/task_trials_long_all.csv --include-practice

Note:
    Bayesian fitting is slower than MLE but provides richer output.
    For initial exploration, use MLE (12_fit_mle.py).
    For final analysis and publication, consider Bayesian.

Next Steps:
    - Run 14_compare_models.py with --use-waic for Bayesian model comparison
    - Examine posterior distributions for parameter uncertainty
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Import the main function from the library module
from scripts.fitting.fit_bayesian import main

if __name__ == '__main__':
    main()
