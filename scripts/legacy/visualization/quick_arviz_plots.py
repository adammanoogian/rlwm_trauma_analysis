#!/usr/bin/env python3
"""
Quick ArviZ visualization script for posterior analysis.

Usage:
    python scripts/visualization/quick_arviz_plots.py \
        --posterior output/v1/qlearning_jax_posterior_20251122_200043.nc \
        --output-dir output/v1/figures
"""

from __future__ import annotations

import sys
from pathlib import Path

import argparse
import arviz as az
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import load_netcdf_with_validation  # noqa: E402


def create_all_diagnostic_plots(posterior_path: Path, output_dir: Path):
    """Create comprehensive diagnostic plots using ArviZ."""

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>> Loading posterior from: {posterior_path}")
    # Get model name from filename (used for wrapper + plot labels)
    model_name = posterior_path.stem.replace('_posterior', '').replace('_jax', '')
    idata = load_netcdf_with_validation(posterior_path, model_name)

    print(f"\n>> Creating diagnostic plots for: {model_name}")
    print("="*80)

    # 1. Trace plots (chains, mixing)
    print("\n1. Trace plots (chains and mixing)...")
    fig = az.plot_trace(
        idata,
        var_names=['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta'],
        figsize=(12, 8)
    )
    plt.suptitle(f'{model_name}: Trace Plots', fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_trace.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"   ✓ Saved: {model_name}_trace.png")

    # 2. Posterior distributions
    print("\n2. Posterior distributions with HDI...")
    fig = az.plot_posterior(
        idata,
        var_names=['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta'],
        hdi_prob=0.94,
        figsize=(12, 4)
    )
    plt.suptitle(f'{model_name}: Posterior Distributions (94% HDI)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_posterior.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"   ✓ Saved: {model_name}_posterior.png")

    # 3. Forest plot (all parameters)
    print("\n3. Forest plot (all group-level parameters)...")
    fig = az.plot_forest(
        idata,
        var_names=['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta'],
        combined=True,
        hdi_prob=0.94,
        figsize=(10, 6)
    )
    plt.suptitle(f'{model_name}: Forest Plot (94% HDI)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_forest.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"   ✓ Saved: {model_name}_forest.png")

    # 4. Rank plots (for diagnosing convergence issues)
    print("\n4. Rank plots (convergence diagnostics)...")
    fig = az.plot_rank(
        idata,
        var_names=['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta'],
        figsize=(12, 8)
    )
    plt.suptitle(f'{model_name}: Rank Plots', fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_rank.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"   ✓ Saved: {model_name}_rank.png")

    # 5. Autocorrelation plots
    print("\n5. Autocorrelation plots...")
    fig = az.plot_autocorr(
        idata,
        var_names=['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta'],
        figsize=(12, 8)
    )
    plt.suptitle(f'{model_name}: Autocorrelation', fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_autocorr.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"   ✓ Saved: {model_name}_autocorr.png")

    # 6. Energy plot (HMC diagnostic)
    print("\n6. Energy plot (HMC/NUTS diagnostic)...")
    try:
        fig = az.plot_energy(idata, figsize=(10, 6))
        plt.suptitle(f'{model_name}: Energy Plot', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_energy.png', dpi=300, bbox_inches='tight')
        plt.close('all')
        print(f"   ✓ Saved: {model_name}_energy.png")
    except Exception as e:
        print(f"   ⚠ Could not create energy plot: {e}")

    # 7. Pair plot (parameter correlations)
    print("\n7. Pair plot (parameter correlations)...")
    fig = az.plot_pair(
        idata,
        var_names=['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta'],
        kind='kde',
        marginals=True,
        figsize=(10, 10)
    )
    plt.suptitle(f'{model_name}: Parameter Correlations', fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_pair.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"   ✓ Saved: {model_name}_pair.png")

    # 8. Individual parameter plots (if available)
    print("\n8. Individual-level parameters (subset)...")
    if 'alpha_pos' in idata.posterior:
        n_participants = idata.posterior['alpha_pos'].shape[-1]
        if n_participants <= 10:  # Only plot if reasonable number
            fig = az.plot_forest(
                idata,
                var_names=['alpha_pos', 'alpha_neg', 'beta'],
                combined=True,
                hdi_prob=0.94,
                figsize=(10, max(8, n_participants))
            )
            plt.suptitle(f'{model_name}: Individual Parameters', fontsize=14)
            plt.tight_layout()
            plt.savefig(output_dir / f'{model_name}_individual_forest.png', dpi=300, bbox_inches='tight')
            plt.close('all')
            print(f"   ✓ Saved: {model_name}_individual_forest.png")
        else:
            print(f"   ⚠ Skipping individual plot (too many participants: {n_participants})")

    # Print summary statistics
    print("\n" + "="*80)
    print("POSTERIOR SUMMARY STATISTICS")
    print("="*80)
    summary = az.summary(idata, var_names=['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta'])
    print(summary)

    # Save summary to CSV
    summary_path = output_dir / f'{model_name}_summary.csv'
    summary.to_csv(summary_path)
    print(f"\n✓ Summary saved to: {summary_path}")

    print("\n" + "="*80)
    print("✓✓ ALL DIAGNOSTIC PLOTS CREATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nPlots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Create comprehensive ArviZ diagnostic plots'
    )
    parser.add_argument(
        '--posterior',
        type=str,
        required=True,
        help='Path to posterior NetCDF file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='figures',
        help='Directory to save plots'
    )

    args = parser.parse_args()

    posterior_path = Path(args.posterior)
    output_dir = Path(args.output_dir)

    if not posterior_path.exists():
        print(f"ERROR: Posterior file not found: {posterior_path}")
        return

    create_all_diagnostic_plots(posterior_path, output_dir)


if __name__ == '__main__':
    main()
