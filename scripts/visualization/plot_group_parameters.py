#!/usr/bin/env python3
"""
Plot group-level parameter estimates from hierarchical Bayesian models.

Creates forest plots showing posterior distributions of group-level parameters
(mu parameters) with credible intervals and parameter bounds.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from pathlib import Path
import argparse
from typing import Dict, List, Tuple


def load_posterior_samples(netcdf_path: Path) -> az.InferenceData:
    """Load posterior samples from NetCDF file."""
    print(f"Loading posterior from: {netcdf_path}")
    idata = az.from_netcdf(netcdf_path)
    return idata


def extract_group_parameters(
    idata: az.InferenceData,
    model_name: str
) -> pd.DataFrame:
    """
    Extract group-level (mu) parameters from posterior.

    Returns DataFrame with columns: parameter, mean, hdi_low, hdi_high
    """
    posterior = idata.posterior

    # Get all mu_ parameters
    mu_params = [var for var in posterior.data_vars if var.startswith('mu_')]

    results = []
    for param in mu_params:
        # Get posterior samples (flatten chains)
        samples = posterior[param].values.flatten()

        # Compute summary statistics
        mean_val = np.mean(samples)
        hdi = az.hdi(samples, hdi_prob=0.94)

        # Handle scalar vs array HDI
        if isinstance(hdi, np.ndarray) and hdi.ndim == 0:
            hdi_low = float(hdi)
            hdi_high = float(hdi)
        elif isinstance(hdi, np.ndarray) and len(hdi) == 2:
            hdi_low = float(hdi[0])
            hdi_high = float(hdi[1])
        else:
            hdi_low = float(np.min(hdi))
            hdi_high = float(np.max(hdi))

        # Clean parameter name
        param_clean = param.replace('mu_', '')

        results.append({
            'parameter': param_clean,
            'mean': mean_val,
            'hdi_low': hdi_low,
            'hdi_high': hdi_high,
            'model': model_name
        })

    return pd.DataFrame(results)


def get_parameter_bounds() -> Dict[str, Tuple[float, float]]:
    """Get theoretical bounds for each parameter."""
    bounds = {
        'alpha_pos': (0, 1),
        'alpha_neg': (0, 1),
        'beta': (0, 10),  # Practical upper bound
        'beta_wm': (0, 10),
        'phi': (0, 1),
        'rho': (0, 1),
        'capacity': (1, 7)
    }
    return bounds


def get_parameter_labels() -> Dict[str, str]:
    """Get formatted labels for parameters."""
    labels = {
        'alpha_pos': 'α+ (Learning Rate, Positive)',
        'alpha_neg': 'α- (Learning Rate, Negative)',
        'beta': 'β (Inverse Temperature, RL)',
        'beta_wm': 'β_WM (Inverse Temperature, WM)',
        'phi': 'φ (WM Decay Rate)',
        'rho': 'ρ (WM Reliance)',
        'capacity': 'K (WM Capacity)'
    }
    return labels


def plot_group_parameters_forest(
    df_params: pd.DataFrame,
    output_path: Path,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Create forest plot of group-level parameters.

    Y-axis: Parameter names
    X-axis: Parameter values
    Error bars: 94% HDI
    Background shading: Theoretical bounds
    """
    bounds = get_parameter_bounds()
    labels = get_parameter_labels()

    # Set up plot
    fig, ax = plt.subplots(figsize=figsize)

    # Sort parameters by model then by name
    df_sorted = df_params.sort_values(['model', 'parameter'])

    # Create y-positions
    y_positions = np.arange(len(df_sorted))

    # Color by model
    colors = {'Q-Learning': '#1f77b4', 'WM-RL': '#ff7f0e'}

    # Plot each parameter
    for idx, row in df_sorted.iterrows():
        y_pos = np.where(df_sorted.index == idx)[0][0]
        param = row['parameter']
        model = row['model']

        # Get bounds for this parameter
        if param in bounds:
            x_min, x_max = bounds[param]
            # Draw shaded background showing valid range
            ax.axhspan(y_pos - 0.4, y_pos + 0.4,
                      xmin=(x_min - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]),
                      xmax=(x_max - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]),
                      alpha=0.1, color='gray', zorder=0)

        # Plot HDI as error bar
        ax.errorbar(
            x=row['mean'],
            y=y_pos,
            xerr=[[row['mean'] - row['hdi_low']], [row['hdi_high'] - row['mean']]],
            fmt='o',
            markersize=8,
            linewidth=2,
            capsize=5,
            capthick=2,
            color=colors.get(model, 'black'),
            label=model if idx == df_sorted[df_sorted['model'] == model].index[0] else None,
            zorder=2
        )

        # Add mean value text
        ax.text(row['mean'], y_pos + 0.15, f"{row['mean']:.3f}",
               ha='center', va='bottom', fontsize=8)

    # Set y-tick labels
    y_labels = [labels.get(row['parameter'], row['parameter']) + f"\n({row['model']})"
                for _, row in df_sorted.iterrows()]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=10)

    # Labels and title
    ax.set_xlabel('Parameter Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Parameter', fontsize=12, fontweight='bold')
    ax.set_title('Group-Level Parameter Estimates\n(Posterior Mean ± 94% HDI)',
                fontsize=14, fontweight='bold', pad=20)

    # Grid and legend
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, frameon=True, shadow=True)

    # Adjust layout
    plt.tight_layout()

    # Save
    print(f"Saving figure to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Forest plot saved successfully!")


def plot_group_parameters_comparison(
    df_ql: pd.DataFrame,
    df_wmrl: pd.DataFrame,
    output_path: Path,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Create side-by-side comparison of shared parameters between models.
    """
    # Find shared parameters (present in both models)
    shared_params = set(df_ql['parameter']) & set(df_wmrl['parameter'])

    if not shared_params:
        print("No shared parameters between models, skipping comparison plot")
        return

    labels = get_parameter_labels()

    # Filter to shared parameters
    df_ql_shared = df_ql[df_ql['parameter'].isin(shared_params)].copy()
    df_wmrl_shared = df_wmrl[df_wmrl['parameter'].isin(shared_params)].copy()

    # Set up plot
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by parameter name
    shared_params_sorted = sorted(shared_params)
    x_positions = np.arange(len(shared_params_sorted))
    width = 0.35

    # Plot Q-learning
    ql_means = []
    ql_errors_low = []
    ql_errors_high = []
    for param in shared_params_sorted:
        row = df_ql_shared[df_ql_shared['parameter'] == param].iloc[0]
        ql_means.append(row['mean'])
        ql_errors_low.append(row['mean'] - row['hdi_low'])
        ql_errors_high.append(row['hdi_high'] - row['mean'])

    ax.bar(x_positions - width/2, ql_means, width,
           label='Q-Learning', color='#1f77b4', alpha=0.8)
    ax.errorbar(x_positions - width/2, ql_means,
               yerr=[ql_errors_low, ql_errors_high],
               fmt='none', ecolor='black', capsize=5, linewidth=2)

    # Plot WM-RL
    wmrl_means = []
    wmrl_errors_low = []
    wmrl_errors_high = []
    for param in shared_params_sorted:
        row = df_wmrl_shared[df_wmrl_shared['parameter'] == param].iloc[0]
        wmrl_means.append(row['mean'])
        wmrl_errors_low.append(row['mean'] - row['hdi_low'])
        wmrl_errors_high.append(row['hdi_high'] - row['mean'])

    ax.bar(x_positions + width/2, wmrl_means, width,
           label='WM-RL', color='#ff7f0e', alpha=0.8)
    ax.errorbar(x_positions + width/2, wmrl_means,
               yerr=[wmrl_errors_low, wmrl_errors_high],
               fmt='none', ecolor='black', capsize=5, linewidth=2)

    # Labels
    x_labels = [labels.get(p, p) for p in shared_params_sorted]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel('Parameter Value', fontsize=12, fontweight='bold')
    ax.set_title('Comparison of Shared Parameters\n(Posterior Mean ± 94% HDI)',
                fontsize=14, fontweight='bold', pad=20)

    ax.legend(loc='best', fontsize=11, frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    print(f"Saving comparison figure to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Comparison plot saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Plot group-level parameter estimates from fitted models'
    )
    parser.add_argument(
        '--qlearning',
        type=str,
        help='Path to Q-learning posterior NetCDF file'
    )
    parser.add_argument(
        '--wmrl',
        type=str,
        help='Path to WM-RL posterior NetCDF file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='figures',
        help='Directory to save plots'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='',
        help='Prefix for output filenames'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("PLOT GROUP-LEVEL PARAMETERS")
    print("="*80)

    # Collect parameter data from both models
    all_params = []

    if args.qlearning:
        print("\n>> Loading Q-Learning model...")
        idata_ql = load_posterior_samples(Path(args.qlearning))
        df_ql = extract_group_parameters(idata_ql, 'Q-Learning')
        print(f"  ✓ Extracted {len(df_ql)} group-level parameters")
        print(df_ql)
        all_params.append(df_ql)

    if args.wmrl:
        print("\n>> Loading WM-RL model...")
        idata_wmrl = load_posterior_samples(Path(args.wmrl))
        df_wmrl = extract_group_parameters(idata_wmrl, 'WM-RL')
        print(f"  ✓ Extracted {len(df_wmrl)} group-level parameters")
        print(df_wmrl)
        all_params.append(df_wmrl)

    if not all_params:
        print("ERROR: No model files provided. Use --qlearning and/or --wmrl")
        return

    # Combine all parameters
    df_all = pd.concat(all_params, ignore_index=True)

    # Create forest plot
    print("\n>> Creating forest plot...")
    prefix = f"{args.prefix}_" if args.prefix else ""
    forest_path = output_dir / f"{prefix}group_parameters_forest.png"
    plot_group_parameters_forest(df_all, forest_path)

    # Create comparison plot if both models present
    if args.qlearning and args.wmrl:
        print("\n>> Creating comparison plot...")
        comparison_path = output_dir / f"{prefix}group_parameters_comparison.png"
        plot_group_parameters_comparison(df_ql, df_wmrl, comparison_path)

    print("\n" + "="*80)
    print("✓✓ All plots created successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
