"""
Fit Both Q-Learning and WM-RL Models with Model Comparison

This script:
1. Fits both Q-learning and WM-RL models to the full dataset
2. Runs model comparison (WAIC, LOO)
3. Saves all results with comprehensive diagnostics

Usage:
------
python scripts/fitting/fit_both_models.py \
    --data output/task_trials_long_all_participants.csv \
    --chains 4 --warmup 1000 --samples 2000

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
    wmrl_hierarchical_model,
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

    Handles both 'response' and 'key_press' column names.
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

    # Drop any NaN values in critical columns first
    df = df.dropna(subset=['stimulus', 'key_press', 'correct']).copy()
    print(f"  ✓ Dropped NaN values in critical columns: {len(df)} trials remain")

    # Ensure correct data types
    df['stimulus'] = df['stimulus'].astype(int)
    df['key_press'] = df['key_press'].astype(int)
    df['reward'] = df['correct'].astype(float)  # Reward = 1 if correct, 0 otherwise

    # Check for set_size column (needed for WM-RL)
    if 'set_size' not in df.columns:
        print(f"  ⚠ Warning: 'set_size' column not found, using default value of 6")
        df['set_size'] = 6

    # Print summary
    print(f"\n>> Data summary:")
    print(f"  Participants: {df['sona_id'].nunique()}")
    print(f"  Blocks: {df['block'].nunique()}")
    print(f"  Total trials: {len(df)}")
    print(f"  Trials per participant: {len(df) / df['sona_id'].nunique():.0f}")

    # Per-participant summary
    for pid in sorted(df['sona_id'].unique()):
        pdata = df[df['sona_id'] == pid]
        print(f"    Participant {int(pid)}: {len(pdata)} trials, {pdata['block'].nunique()} blocks")

    return df


def fit_qlearning(
    data: pd.DataFrame,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    seed: int
):
    """Fit Q-learning model."""
    print("\n" + "=" * 80)
    print("FITTING Q-LEARNING MODEL")
    print("=" * 80)

    # Prepare data
    print("\n>> Preparing data...")
    participant_data = prepare_data_for_numpyro(
        data,
        participant_col='sona_id',
        block_col='block',
        stimulus_col='stimulus',
        action_col='key_press',
        reward_col='reward',
        set_size_col='set_size'
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
    mcmc = run_inference(
        model=qlearning_hierarchical_model,
        model_args={'participant_data': participant_data},
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed
    )

    # Print summary
    samples = mcmc.get_samples()
    print("\n" + "=" * 80)
    print("Q-LEARNING POSTERIOR ESTIMATES")
    print("=" * 80)
    print("\nGroup-level parameters:")
    print(f"  μ_α+ : {samples['mu_alpha_pos'].mean():.3f} ± {samples['mu_alpha_pos'].std():.3f}")
    print(f"  μ_α- : {samples['mu_alpha_neg'].mean():.3f} ± {samples['mu_alpha_neg'].std():.3f}")
    print(f"  μ_β  : {samples['mu_beta'].mean():.3f} ± {samples['mu_beta'].std():.3f}")

    return mcmc, participant_data


def fit_wmrl(
    data: pd.DataFrame,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    seed: int
):
    """Fit WM-RL model."""
    print("\n" + "=" * 80)
    print("FITTING WM-RL MODEL")
    print("=" * 80)

    # Prepare data
    print("\n>> Preparing data...")
    participant_data = prepare_data_for_numpyro(
        data,
        participant_col='sona_id',
        block_col='block',
        stimulus_col='stimulus',
        action_col='key_press',
        reward_col='reward',
        set_size_col='set_size'
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
    mcmc = run_inference(
        model=wmrl_hierarchical_model,
        model_args={'participant_data': participant_data},
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed
    )

    # Print summary
    samples = mcmc.get_samples()
    print("\n" + "=" * 80)
    print("WM-RL POSTERIOR ESTIMATES")
    print("=" * 80)
    print("\nGroup-level parameters:")
    print(f"  μ_α+    : {samples['mu_alpha_pos'].mean():.3f} ± {samples['mu_alpha_pos'].std():.3f}")
    print(f"  μ_α-    : {samples['mu_alpha_neg'].mean():.3f} ± {samples['mu_alpha_neg'].std():.3f}")
    print(f"  μ_β     : {samples['mu_beta'].mean():.3f} ± {samples['mu_beta'].std():.3f}")
    print(f"  μ_β_WM  : {samples['mu_beta_wm'].mean():.3f} ± {samples['mu_beta_wm'].std():.3f}")
    print(f"  μ_φ     : {samples['mu_phi'].mean():.3f} ± {samples['mu_phi'].std():.3f}")
    print(f"  μ_ρ     : {samples['mu_rho'].mean():.3f} ± {samples['mu_rho'].std():.3f}")
    print(f"  μ_K     : {samples['mu_capacity'].mean():.3f} ± {samples['mu_capacity'].std():.3f}")

    return mcmc, participant_data


def compare_models(
    mcmc_qlearning,
    mcmc_wmrl,
    data: pd.DataFrame
):
    """
    Compare models using WAIC and LOO.
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    # Convert to ArviZ
    print("\n>> Converting to ArviZ format...")
    idata_ql = samples_to_arviz(mcmc_qlearning, data)
    idata_wmrl = samples_to_arviz(mcmc_wmrl, data)

    # Compute WAIC
    print("\n>> Computing WAIC...")
    waic_ql = az.waic(idata_ql)
    waic_wmrl = az.waic(idata_wmrl)

    print(f"\nWAIC Comparison:")
    print(f"  Q-Learning: {waic_ql.elpd_waic:.2f} (SE: {waic_ql.se:.2f})")
    print(f"  WM-RL:      {waic_wmrl.elpd_waic:.2f} (SE: {waic_wmrl.se:.2f})")
    print(f"  Difference: {waic_wmrl.elpd_waic - waic_ql.elpd_waic:.2f}")

    if waic_wmrl.elpd_waic > waic_ql.elpd_waic:
        print(f"  ✓ WM-RL wins by {waic_wmrl.elpd_waic - waic_ql.elpd_waic:.2f} ELPD points")
    else:
        print(f"  ✓ Q-Learning wins by {waic_ql.elpd_waic - waic_wmrl.elpd_waic:.2f} ELPD points")

    # Compute LOO
    print("\n>> Computing LOO-CV...")
    try:
        loo_ql = az.loo(idata_ql)
        loo_wmrl = az.loo(idata_wmrl)

        print(f"\nLOO-CV Comparison:")
        print(f"  Q-Learning: {loo_ql.elpd_loo:.2f} (SE: {loo_ql.se:.2f})")
        print(f"  WM-RL:      {loo_wmrl.elpd_loo:.2f} (SE: {loo_wmrl.se:.2f})")
        print(f"  Difference: {loo_wmrl.elpd_loo - loo_ql.elpd_loo:.2f}")

        if loo_wmrl.elpd_loo > loo_ql.elpd_loo:
            print(f"  ✓ WM-RL wins by {loo_wmrl.elpd_loo - loo_ql.elpd_loo:.2f} ELPD points")
        else:
            print(f"  ✓ Q-Learning wins by {loo_ql.elpd_loo - loo_wmrl.elpd_loo:.2f} ELPD points")
    except Exception as e:
        print(f"  ⚠ LOO computation failed: {e}")
        loo_ql, loo_wmrl = None, None

    return {
        'waic_ql': waic_ql,
        'waic_wmrl': waic_wmrl,
        'loo_ql': loo_ql,
        'loo_wmrl': loo_wmrl,
        'idata_ql': idata_ql,
        'idata_wmrl': idata_wmrl
    }


def save_results(
    mcmc_qlearning,
    mcmc_wmrl,
    comparison_results,
    data: pd.DataFrame,
    output_dir: Path = OUTPUT_VERSION_DIR,
    save_plots: bool = True
):
    """Save all results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Save Q-learning results
    print("\n>> Saving Q-learning results...")
    idata_ql = comparison_results['idata_ql']

    netcdf_file_ql = output_dir / f'qlearning_posterior_{timestamp}.nc'
    idata_ql.to_netcdf(netcdf_file_ql)
    print(f"  ✓ Q-learning posterior: {netcdf_file_ql}")

    summary_file_ql = output_dir / f'qlearning_summary_{timestamp}.csv'
    az.summary(idata_ql).to_csv(summary_file_ql)
    print(f"  ✓ Q-learning summary: {summary_file_ql}")

    # Save WM-RL results
    print("\n>> Saving WM-RL results...")
    idata_wmrl = comparison_results['idata_wmrl']

    netcdf_file_wmrl = output_dir / f'wmrl_posterior_{timestamp}.nc'
    idata_wmrl.to_netcdf(netcdf_file_wmrl)
    print(f"  ✓ WM-RL posterior: {netcdf_file_wmrl}")

    summary_file_wmrl = output_dir / f'wmrl_summary_{timestamp}.csv'
    az.summary(idata_wmrl).to_csv(summary_file_wmrl)
    print(f"  ✓ WM-RL summary: {summary_file_wmrl}")

    # Save model comparison
    print("\n>> Saving model comparison...")
    comparison_file = output_dir / f'model_comparison_{timestamp}.txt'
    with open(comparison_file, 'w') as f:
        f.write("MODEL COMPARISON RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write("WAIC:\n")
        f.write(f"  Q-Learning: {comparison_results['waic_ql'].elpd_waic:.2f} (SE: {comparison_results['waic_ql'].se:.2f})\n")
        f.write(f"  WM-RL:      {comparison_results['waic_wmrl'].elpd_waic:.2f} (SE: {comparison_results['waic_wmrl'].se:.2f})\n")
        f.write(f"  Difference: {comparison_results['waic_wmrl'].elpd_waic - comparison_results['waic_ql'].elpd_waic:.2f}\n\n")

        if comparison_results['loo_ql'] is not None:
            f.write("LOO-CV:\n")
            f.write(f"  Q-Learning: {comparison_results['loo_ql'].elpd_loo:.2f} (SE: {comparison_results['loo_ql'].se:.2f})\n")
            f.write(f"  WM-RL:      {comparison_results['loo_wmrl'].elpd_loo:.2f} (SE: {comparison_results['loo_wmrl'].se:.2f})\n")
            f.write(f"  Difference: {comparison_results['loo_wmrl'].elpd_loo - comparison_results['loo_ql'].elpd_loo:.2f}\n")

    print(f"  ✓ Model comparison: {comparison_file}")

    # Save plots
    if save_plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Q-learning plots
        print("\n>> Generating Q-learning plots...")
        trace_file_ql = output_dir / f'qlearning_trace_{timestamp}.png'
        az.plot_trace(idata_ql, var_names=['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta'])
        plt.tight_layout()
        plt.savefig(trace_file_ql, dpi=150, bbox_inches='tight')
        plt.close('all')
        print(f"  ✓ Q-learning trace: {trace_file_ql}")

        posterior_file_ql = output_dir / f'qlearning_posterior_{timestamp}.png'
        az.plot_posterior(idata_ql, var_names=['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta'])
        plt.tight_layout()
        plt.savefig(posterior_file_ql, dpi=150, bbox_inches='tight')
        plt.close('all')
        print(f"  ✓ Q-learning posterior: {posterior_file_ql}")

        # WM-RL plots
        print("\n>> Generating WM-RL plots...")
        trace_file_wmrl = output_dir / f'wmrl_trace_{timestamp}.png'
        az.plot_trace(idata_wmrl, var_names=['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta', 'mu_beta_wm', 'mu_phi', 'mu_rho', 'mu_capacity'])
        plt.tight_layout()
        plt.savefig(trace_file_wmrl, dpi=300, bbox_inches='tight')
        plt.close('all')
        print(f"  ✓ WM-RL trace: {trace_file_wmrl}")

        posterior_file_wmrl = output_dir / f'wmrl_posterior_{timestamp}.png'
        az.plot_posterior(idata_wmrl, var_names=['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta', 'mu_beta_wm', 'mu_phi', 'mu_rho', 'mu_capacity'])
        plt.tight_layout()
        plt.savefig(posterior_file_wmrl, dpi=300, bbox_inches='tight')
        plt.close('all')
        print(f"  ✓ WM-RL posterior: {posterior_file_wmrl}")

    print("\n>> All results saved successfully!")


def main():
    """Main workflow."""
    parser = argparse.ArgumentParser(
        description='Fit both Q-learning and WM-RL models with model comparison'
    )
    parser.add_argument('--data', type=str, required=True, help='Path to trial data CSV')
    parser.add_argument('--min-block', type=int, default=3, help='Minimum block to include')
    parser.add_argument('--chains', type=int, default=4, help='Number of MCMC chains')
    parser.add_argument('--warmup', type=int, default=1000, help='Number of warmup samples')
    parser.add_argument('--samples', type=int, default=2000, help='Number of posterior samples')
    parser.add_argument('--output', type=str, default=str(OUTPUT_VERSION_DIR), help='Output directory')
    parser.add_argument('--save-plots', action='store_true', help='Save diagnostic plots')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--skip-qlearning', action='store_true', help='Skip Q-learning fit')
    parser.add_argument('--skip-wmrl', action='store_true', help='Skip WM-RL fit')

    args = parser.parse_args()

    print("=" * 80)
    print("FIT BOTH MODELS WITH MODEL COMPARISON")
    print("=" * 80)
    print(f"\nSettings:")
    print(f"  Data: {args.data}")
    print(f"  Chains: {args.chains}")
    print(f"  Warmup: {args.warmup}")
    print(f"  Samples: {args.samples}")

    # Set NumPyro devices
    num_devices = min(args.chains, jax.local_device_count())
    numpyro.set_host_device_count(num_devices)
    print(f"  JAX devices: {num_devices}")

    # Load data
    data = load_and_prepare_data(Path(args.data), min_block=args.min_block)

    # Fit Q-learning
    if not args.skip_qlearning:
        mcmc_ql, _ = fit_qlearning(data, args.warmup, args.samples, args.chains, args.seed)
    else:
        print("\n>> Skipping Q-learning fit")
        mcmc_ql = None

    # Fit WM-RL
    if not args.skip_wmrl:
        mcmc_wmrl, _ = fit_wmrl(data, args.warmup, args.samples, args.chains, args.seed)
    else:
        print("\n>> Skipping WM-RL fit")
        mcmc_wmrl = None

    # Model comparison
    if mcmc_ql is not None and mcmc_wmrl is not None:
        comparison_results = compare_models(mcmc_ql, mcmc_wmrl, data)
        save_results(mcmc_ql, mcmc_wmrl, comparison_results, data, Path(args.output), args.save_plots)
    else:
        print("\n>> Skipping model comparison (one or both models not fitted)")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
