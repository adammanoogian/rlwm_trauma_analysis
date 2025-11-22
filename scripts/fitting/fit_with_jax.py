"""
Fit RL Models to Human Data using JAX/NumPyro

This is the main entry point for fitting Q-learning and WM-RL models to
human behavioral data using JAX/NumPyro.

Advantages over PyMC/PyTensor:
- 10-100x faster compilation (XLA)
- NUTS sampler works efficiently (gradient-based)
- Clean functional API
- Native JAX operations

Usage:
------
# Fit single block (validation)
python scripts/fitting/fit_with_jax.py \
    --data output/task_trials_single_block.csv \
    --chains 2 --warmup 500 --samples 1000

# Fit full 2-participant dataset
python scripts/fitting/fit_with_jax.py \
    --data output/task_trials_long.csv \
    --chains 4 --warmup 1000 --samples 2000

# Save results
python scripts/fitting/fit_with_jax.py \
    --data output/task_trials_long.csv \
    --output output/posteriors/ \
    --save-plots

Author: Generated for RLWM trauma analysis project
Date: 2025-11-22
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import numpyro
import arviz as az

from scripts.fitting.numpyro_models import (
    qlearning_hierarchical_model,
    prepare_data_for_numpyro,
    run_inference,
    samples_to_arviz
)

from config import OUTPUT_VERSION_DIR


def load_and_prepare_data(
    data_path: Path,
    min_block: int = 3
) -> pd.DataFrame:
    """
    Load and prepare human behavioral data.

    Parameters
    ----------
    data_path : Path
        Path to CSV file with trial data
    min_block : int
        Minimum block number to include (default: 3, excludes practice)

    Returns
    -------
    pd.DataFrame
        Cleaned data ready for fitting
    """
    print(f"\n>> Loading data from: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load data
    df = pd.read_csv(data_path)

    # Remove rows with NaN participant IDs
    df = df.dropna(subset=['sona_id']).copy()

    print(f"  ✓ Loaded {len(df)} trials from {df['sona_id'].nunique()} participants")

    # Filter blocks
    if min_block is not None:
        df = df[df['block'] >= min_block].copy()
        print(f"  ✓ Filtered to blocks >= {min_block}: {len(df)} trials remain")

    # Ensure correct data types
    df['stimulus'] = df['stimulus'].astype(int)
    df['key_press'] = df['key_press'].astype(int)
    df['reward'] = df['correct'].astype(float)  # Reward = 1 if correct, 0 otherwise

    # Print summary
    print(f"\n>> Data summary:")
    print(f"  Participants: {df['sona_id'].nunique()}")
    print(f"  Blocks: {df['block'].nunique()}")
    print(f"  Total trials: {len(df)}")
    print(f"  Trials per participant: {len(df) / df['sona_id'].nunique():.0f}")

    # Per-participant summary
    for pid in df['sona_id'].unique():
        pdata = df[df['sona_id'] == pid]
        print(f"    Participant {int(pid)}: {len(pdata)} trials, {pdata['block'].nunique()} blocks")

    return df


def fit_qlearning_jax(
    data: pd.DataFrame,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
    seed: int = 42
):
    """
    Fit hierarchical Q-learning model using JAX/NumPyro.

    Parameters
    ----------
    data : pd.DataFrame
        Prepared trial data
    num_warmup : int
        Number of warmup samples
    num_samples : int
        Number of posterior samples per chain
    num_chains : int
        Number of MCMC chains
    seed : int
        Random seed

    Returns
    -------
    mcmc : numpyro.infer.MCMC
        MCMC object with samples
    """
    print("\n" + "=" * 80)
    print("FITTING Q-LEARNING MODEL WITH JAX/NUMPYRO")
    print("=" * 80)

    # Prepare data in block-structured format
    print("\n>> Preparing data...")
    participant_data = prepare_data_for_numpyro(
        data,
        participant_col='sona_id',
        block_col='block',
        stimulus_col='stimulus',
        action_col='key_press',
        reward_col='reward'
    )

    num_participants = len(participant_data)
    total_blocks = sum([len(pdata['stimuli_blocks']) for pdata in participant_data.values()])
    total_trials = sum([
        sum([len(block) for block in pdata['stimuli_blocks']])
        for pdata in participant_data.values()
    ])

    print(f"  ✓ Prepared {num_participants} participants")
    print(f"  ✓ Total blocks: {total_blocks}")
    print(f"  ✓ Total trials: {total_trials}")

    # Run inference
    print(f"\n>> Running MCMC inference...")
    print(f"  Chains: {num_chains}")
    print(f"  Warmup: {num_warmup}")
    print(f"  Samples: {num_samples}")
    print(f"  Sampler: NUTS (gradient-based)")

    mcmc = run_inference(
        model=qlearning_hierarchical_model,
        model_args={'participant_data': participant_data},
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed
    )

    print("\n>> Sampling complete!")

    # Get samples
    samples = mcmc.get_samples()

    # Print posterior summaries
    print("\n" + "=" * 80)
    print("POSTERIOR ESTIMATES")
    print("=" * 80)

    print("\nGroup-level parameters:")
    print(f"  μ_α+ : {samples['mu_alpha_pos'].mean():.3f} ± {samples['mu_alpha_pos'].std():.3f}")
    print(f"  μ_α- : {samples['mu_alpha_neg'].mean():.3f} ± {samples['mu_alpha_neg'].std():.3f}")
    print(f"  μ_β  : {samples['mu_beta'].mean():.3f} ± {samples['mu_beta'].std():.3f}")

    print("\nIndividual parameters:")
    for i in range(num_participants):
        print(f"  Participant {i}:")
        print(f"    α+ : {samples['alpha_pos'][:, i].mean():.3f} ± {samples['alpha_pos'][:, i].std():.3f}")
        print(f"    α- : {samples['alpha_neg'][:, i].mean():.3f} ± {samples['alpha_neg'][:, i].std():.3f}")
        print(f"    β  : {samples['beta'][:, i].mean():.3f} ± {samples['beta'][:, i].std():.3f}")

    return mcmc


def save_results(
    mcmc,
    data: pd.DataFrame,
    output_dir: Path = OUTPUT_VERSION_DIR,
    save_plots: bool = True
):
    """
    Save fitting results to disk.

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        MCMC object with samples
    data : pd.DataFrame
        Original data
    output_dir : Path
        Output directory
    save_plots : bool
        Whether to save diagnostic plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Convert to ArviZ InferenceData
    print("\n>> Converting to ArviZ format...")
    idata = samples_to_arviz(mcmc, data)

    # Save NetCDF
    netcdf_file = output_dir / f'qlearning_jax_posterior_{timestamp}.nc'
    idata.to_netcdf(netcdf_file)
    print(f"  ✓ Saved posterior: {netcdf_file}")

    # Save summary CSV
    summary_file = output_dir / f'qlearning_jax_summary_{timestamp}.csv'
    summary = az.summary(idata)
    summary.to_csv(summary_file)
    print(f"  ✓ Saved summary: {summary_file}")

    # Save diagnostic plots
    if save_plots:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        # Trace plot
        trace_file = output_dir / f'qlearning_jax_trace_{timestamp}.png'
        az.plot_trace(idata, var_names=['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta'])
        plt.tight_layout()
        plt.savefig(trace_file, dpi=150, bbox_inches='tight')
        plt.close('all')
        print(f"  ✓ Saved trace plot: {trace_file}")

        # Posterior plot
        posterior_file = output_dir / f'qlearning_jax_posterior_{timestamp}.png'
        az.plot_posterior(idata, var_names=['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta'])
        plt.tight_layout()
        plt.savefig(posterior_file, dpi=150, bbox_inches='tight')
        plt.close('all')
        print(f"  ✓ Saved posterior plot: {posterior_file}")

    print("\n>> All results saved successfully!")


def main():
    """Main fitting workflow."""
    parser = argparse.ArgumentParser(
        description='Fit Q-learning model to human data using JAX/NumPyro'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='output/task_trials_long.csv',
        help='Path to trial data CSV'
    )
    parser.add_argument(
        '--min-block',
        type=int,
        default=3,
        help='Minimum block to include (default: 3, excludes practice)'
    )
    parser.add_argument(
        '--chains',
        type=int,
        default=4,
        help='Number of MCMC chains (default: 4)'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=1000,
        help='Number of warmup samples (default: 1000)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=2000,
        help='Number of posterior samples per chain (default: 2000)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=str(OUTPUT_VERSION_DIR),
        help='Output directory for results'
    )
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save diagnostic plots'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("BAYESIAN Q-LEARNING FIT WITH JAX/NUMPYRO")
    print("=" * 80)
    print(f"\nSettings:")
    print(f"  Data: {args.data}")
    print(f"  Min block: {args.min_block}")
    print(f"  Chains: {args.chains}")
    print(f"  Warmup: {args.warmup}")
    print(f"  Samples: {args.samples}")
    print(f"  Output: {args.output}")
    print(f"  Seed: {args.seed}")

    # Set NumPyro to use all available CPU cores
    num_devices = min(args.chains, jax.local_device_count())
    numpyro.set_host_device_count(num_devices)
    print(f"  JAX devices: {num_devices}")

    # Load data
    data = load_and_prepare_data(Path(args.data), min_block=args.min_block)

    # Fit model
    mcmc = fit_qlearning_jax(
        data,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        seed=args.seed
    )

    # Save results
    save_results(mcmc, data, Path(args.output), save_plots=args.save_plots)

    print("\n" + "=" * 80)
    print("FITTING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
