#!/usr/bin/env python
"""
09: Generate Synthetic Data
===========================

Generates synthetic behavioral data from computational models for:
- Posterior predictive checks
- Model validation
- Qualitative model-data comparisons

This is a pipeline script that wraps the simulations library module.

Inputs:
    - Model parameters (from MLE fits or specified manually)
    - Task configuration (from config.py)

Outputs:
    - output/synthetic_data/synthetic_<model>_<timestamp>.csv
    - figures/synthetic_data/ (optional visualizations)

Usage:
    # Generate data from Q-learning with default parameters
    python scripts/03_model_prefitting/01_generate_synthetic_data.py --model qlearning

    # Generate data from WM-RL with fitted parameters
    python scripts/03_model_prefitting/01_generate_synthetic_data.py --model wmrl --params-file output/mle/wmrl_params.csv

    # Generate for specific set sizes
    python scripts/03_model_prefitting/01_generate_synthetic_data.py --model qlearning --set-sizes 3 5

    # Multiple participants (for averaging)
    python scripts/03_model_prefitting/01_generate_synthetic_data.py --model wmrl --n-subjects 50

Next Steps:
    - Compare synthetic vs human data visually
    - Use for posterior predictive checks (scripts/05_post_fitting_checks/03_run_posterior_ppc.py)
    - Run 02_run_parameter_sweep.py for systematic exploration
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Import the main function from the library module
from scripts.legacy.simulations.generate_data import main

if __name__ == '__main__':
    main()
