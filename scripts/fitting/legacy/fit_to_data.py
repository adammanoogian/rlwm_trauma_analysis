"""
Fit RL Models to Human Data using Bayesian Inference

This script loads human behavioral data and fits Q-learning and WM-RL models
using hierarchical Bayesian estimation with PyMC.

Workflow:
1. Load task_trials_long.csv with human data
2. Prepare data (filter practice blocks, ensure 0-indexing)
3. Fit Q-learning model
4. Fit WM-RL hybrid model
5. Compare models (WAIC, LOO)
6. Save posteriors and diagnostics
7. Generate posterior predictive checks

Usage:
    python fitting/fit_to_data.py --model qlearning --data output/task_trials_long.csv
    python fitting/fit_to_data.py --model wmrl --chains 4 --samples 2000
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime
from typing import Dict

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import PyMCParams, DataParams, OUTPUT_VERSION_DIR

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    print("ERROR: PyMC not installed. Install with: pip install pymc arviz")
    sys.exit(1)

from scripts.fitting.pymc_models import (
    build_qlearning_model,
    build_wmrl_model,
    prepare_data_for_fitting,
    compute_model_comparison
)

# Import functional models (PyTensor-compatible)
from scripts.fitting.pymc_models_functional import (
    build_qlearning_model_functional
)


def load_and_prepare_data(
    data_path: Path,
    min_block: int = 3
) -> pd.DataFrame:
    """
    Load and prepare human behavioral data for fitting.

    Parameters
    ----------
    data_path : Path
        Path to task_trials_long.csv
    min_block : int
        Minimum block to include

    Returns
    -------
    pd.DataFrame
        Prepared data
    """
    print(f"Loading data from: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load data
    df = pd.read_csv(data_path)

    print(f"  Loaded {len(df)} trials from {df['sona_id'].nunique()} participants")

    # Prepare for fitting
    df_clean = prepare_data_for_fitting(
        df,
        participant_col='sona_id',
        block_col='block',
        trial_col='trial_in_block',
        stimulus_col='stimulus',
        action_col='key_press',
        correct_col='correct',
        min_block=min_block
    )

    print(f"  After filtering: {len(df_clean)} trials from {df_clean['sona_id'].nunique()} participants")

    return df_clean


def fit_qlearning_model(
    data: pd.DataFrame,
    num_chains: int = PyMCParams.NUM_CHAINS,
    num_samples: int = PyMCParams.NUM_SAMPLES,
    num_tune: int = PyMCParams.NUM_TUNE,
    target_accept: float = PyMCParams.TARGET_ACCEPT,
) -> az.InferenceData:
    """
    Fit Q-learning model to data.

    Parameters
    ----------
    data : pd.DataFrame
        Prepared trial data
    num_chains, num_samples, num_tune : int
        MCMC parameters
    target_accept : float
        Target acceptance rate

    Returns
    -------
    az.InferenceData
        Posterior samples and diagnostics
    """
    print("\n" + "=" * 80)
    print("FITTING Q-LEARNING MODEL")
    print("=" * 80)

    # Build model
    print("\nBuilding model...")
    with build_qlearning_model(
        data,
        participant_col='sona_id',
        stimulus_col='stimulus',
        action_col='key_press',
        reward_col='reward'
    ) as model:
        print(f"  Model variables: {list(model.named_vars.keys())}")

        # Sample
        # NOTE: Using Metropolis sampler because agent classes use pure Python
        # (not PyTensor operations), so no gradients available for NUTS
        print(f"\nSampling: {num_chains} chains, {num_samples} samples, {num_tune} tune")
        print("  Using Metropolis sampler (agent classes are pure Python)")
        trace = pm.sample(
            draws=num_samples,
            tune=num_tune,
            chains=num_chains,
            step=pm.Metropolis(),  # Use Metropolis for pure Python functions
            return_inferencedata=True,
            random_seed=42
        )

    print("\n" + "Sampling complete!")

    # Diagnostics
    print("\nDiagnostics:")
    print(f"  Divergences: {trace.sample_stats.diverging.sum().values}")

    summary = az.summary(trace, var_names=['mu_alpha', 'sigma_alpha', 'mu_beta', 'sigma_beta'])
    print("\nGroup-level parameters:")
    print(summary)

    return trace


def fit_qlearning_model_functional(
    data: pd.DataFrame,
    num_chains: int = PyMCParams.NUM_CHAINS,
    num_samples: int = PyMCParams.NUM_SAMPLES,
    num_tune: int = PyMCParams.NUM_TUNE,
    target_accept: float = PyMCParams.TARGET_ACCEPT,
) -> az.InferenceData:
    """
    Fit Q-learning model to data using functional PyTensor approach.

    This uses pure functional operations (no mutation) which allows
    full PyTensor compatibility and gradient-based sampling.

    Parameters
    ----------
    data : pd.DataFrame
        Prepared trial data
    num_chains, num_samples, num_tune : int
        MCMC parameters
    target_accept : float
        Target acceptance rate

    Returns
    -------
    az.InferenceData
        Posterior samples and diagnostics
    """
    print("\n" + "=" * 80)
    print("FITTING Q-LEARNING MODEL (FUNCTIONAL)")
    print("=" * 80)

    # Build model using functional approach
    print("\nBuilding model (functional PyTensor implementation)...")
    with build_qlearning_model_functional(
        data,
        participant_col='sona_id',
        stimulus_col='stimulus',
        action_col='key_press',
        reward_col='reward'
    ) as model:
        print(f"  Model variables: {list(model.named_vars.keys())}")

        # Sample
        print(f"\nSampling: {num_chains} chains, {num_samples} samples, {num_tune} tune")
        print("  Using Metropolis sampler (functional approach)")

        trace = pm.sample(
            draws=num_samples,
            tune=num_tune,
            chains=num_chains,
            step=pm.Metropolis(),  # Still use Metropolis for now
            return_inferencedata=True,
            random_seed=42
        )

    print("\nSampling complete!")

    # Diagnostics
    print("\nDiagnostics:")
    print(f"  Divergences: {trace.sample_stats.diverging.sum().values if hasattr(trace, 'sample_stats') else 'N/A'}")

    summary = az.summary(trace, var_names=['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta'])
    print("\nGroup-level parameters:")
    print(summary)

    return trace


def fit_wmrl_model(
    data: pd.DataFrame,
    num_chains: int = PyMCParams.NUM_CHAINS,
    num_samples: int = PyMCParams.NUM_SAMPLES,
    num_tune: int = PyMCParams.NUM_TUNE,
    target_accept: float = PyMCParams.TARGET_ACCEPT,
) -> az.InferenceData:
    """
    Fit WM-RL hybrid model to data.

    Parameters
    ----------
    data : pd.DataFrame
        Prepared trial data
    num_chains, num_samples, num_tune : int
        MCMC parameters
    target_accept : float
        Target acceptance rate

    Returns
    -------
    az.InferenceData
        Posterior samples
    """
    print("\n" + "=" * 80)
    print("FITTING WM-RL HYBRID MODEL")
    print("=" * 80)

    print("\nBuilding model...")
    with build_wmrl_model(
        data,
        participant_col='sona_id',
        stimulus_col='stimulus',
        action_col='key_press',
        reward_col='reward',
        set_size_col='set_size'
    ) as model:
        print(f"  Model variables: {list(model.named_vars.keys())}")

        print(f"\nSampling: {num_chains} chains, {num_samples} samples, {num_tune} tune")
        print("  Using Metropolis sampler (agent classes are pure Python)")
        trace = pm.sample(
            draws=num_samples,
            tune=num_tune,
            chains=num_chains,
            step=pm.Metropolis(),  # Use Metropolis for pure Python functions
            return_inferencedata=True,
            random_seed=42
        )

    print("\nSampling complete!")

    print("\nDiagnostics:")
    print(f"  Divergences: {trace.sample_stats.diverging.sum().values}")

    summary = az.summary(trace, var_names=['mu_alpha', 'mu_beta', 'mu_capacity', 'mu_w_wm'])
    print("\nGroup-level parameters:")
    print(summary)

    return trace


def save_results(
    trace_dict: Dict[str, az.InferenceData],
    output_dir: Path = OUTPUT_VERSION_DIR
):
    """
    Save fitting results.

    Parameters
    ----------
    trace_dict : dict
        Dictionary of model name -> InferenceData
    output_dir : Path
        Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_name, trace in trace_dict.items():
        # Save NetCDF
        output_file = output_dir / f'{model_name}_posterior_{timestamp}.nc'
        trace.to_netcdf(output_file)
        print(f"\nSaved {model_name} posterior to: {output_file}")

        # Save summary
        summary_file = output_dir / f'{model_name}_summary_{timestamp}.csv'
        summary = az.summary(trace)
        summary.to_csv(summary_file)
        print(f"Saved {model_name} summary to: {summary_file}")


def compare_models(
    trace_dict: Dict[str, az.InferenceData],
    output_dir: Path = OUTPUT_VERSION_DIR
):
    """
    Compare models using WAIC and LOO.

    Parameters
    ----------
    trace_dict : dict
        Dictionary of traces
    output_dir : Path
        Output directory
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    if len(trace_dict) < 2:
        print("Need at least 2 models for comparison")
        return

    comparison = compute_model_comparison(trace_dict)
    print("\nComparison results:")
    print(comparison)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = output_dir / f'model_comparison_{timestamp}.csv'
    comparison.to_csv(comparison_file)
    print(f"\nSaved comparison to: {comparison_file}")


def main():
    """Main fitting workflow."""
    parser = argparse.ArgumentParser(description='Fit RL models to human data')
    parser.add_argument('--model', type=str, default='both',
                        choices=['qlearning', 'wmrl', 'both'],
                        help='Which model to fit')
    parser.add_argument('--method', type=str, default='functional',
                        choices=['functional', 'agent'],
                        help='Fitting method: functional (PyTensor scan) or agent (OOP classes)')
    parser.add_argument('--data', type=str, default=str(DataParams.TASK_TRIALS_LONG),
                        help='Path to task_trials_long.csv')
    parser.add_argument('--chains', type=int, default=PyMCParams.NUM_CHAINS,
                        help='Number of MCMC chains')
    parser.add_argument('--samples', type=int, default=PyMCParams.NUM_SAMPLES,
                        help='Number of samples per chain')
    parser.add_argument('--tune', type=int, default=PyMCParams.NUM_TUNE,
                        help='Number of tuning samples')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_VERSION_DIR),
                        help='Output directory')

    args = parser.parse_args()

    print("=" * 80)
    print("BAYESIAN MODEL FITTING FOR RLWM TASK")
    print("=" * 80)
    print(f"\nSettings:")
    print(f"  Model: {args.model}")
    print(f"  Method: {args.method}")
    print(f"  Data: {args.data}")
    print(f"  Chains: {args.chains}")
    print(f"  Samples: {args.samples}")
    print(f"  Tune: {args.tune}")
    print(f"  Output: {args.output_dir}")

    # Load data
    data = load_and_prepare_data(Path(args.data))

    # Fit models
    traces = {}

    if args.model in ['qlearning', 'both']:
        # Choose fitting method
        if args.method == 'functional':
            trace_qlearning = fit_qlearning_model_functional(
                data,
                num_chains=args.chains,
                num_samples=args.samples,
                num_tune=args.tune
            )
        else:
            trace_qlearning = fit_qlearning_model(
                data,
                num_chains=args.chains,
                num_samples=args.samples,
                num_tune=args.tune
            )
        traces['qlearning'] = trace_qlearning

    if args.model in ['wmrl', 'both']:
        trace_wmrl = fit_wmrl_model(
            data,
            num_chains=args.chains,
            num_samples=args.samples,
            num_tune=args.tune
        )
        traces['wmrl'] = trace_wmrl

    # Save results
    save_results(traces, Path(args.output_dir))

    # Compare models if both fitted
    if len(traces) >= 2:
        compare_models(traces, Path(args.output_dir))

    print("\n" + "=" * 80)
    print("FITTING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
