#!/usr/bin/env python3
"""
Plot model comparison metrics (WAIC, LOO, AIC, BIC) and goodness of fit.

Creates visualizations comparing Q-learning and WM-RL models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from typing import Dict, List, Tuple
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import load_netcdf_with_validation  # noqa: E402


def load_posterior_samples(netcdf_path: Path, model: str) -> az.InferenceData:
    """Load posterior samples from NetCDF file.

    Parameters
    ----------
    netcdf_path : Path
        Path to the NetCDF posterior file.
    model : str
        Internal model key (e.g. ``"qlearning"``, ``"wmrl"``).

    Returns
    -------
    az.InferenceData
        The loaded posterior, validated.
    """
    print(f"Loading posterior from: {netcdf_path}")
    idata = load_netcdf_with_validation(netcdf_path, model)
    return idata


def compute_model_comparison_metrics(
    idata_ql: az.InferenceData,
    idata_wmrl: az.InferenceData
) -> pd.DataFrame:
    """
    Compute WAIC, LOO, and other comparison metrics.

    Returns DataFrame with columns: model, metric, value, se
    """
    results = []

    # Compute WAIC
    print("\n>> Computing WAIC...")
    try:
        waic_ql = az.waic(idata_ql)
        results.append({
            'model': 'Q-Learning',
            'metric': 'WAIC',
            'value': waic_ql.elpd_waic,
            'se': waic_ql.se
        })
        print(f"  Q-Learning WAIC: {waic_ql.elpd_waic:.2f} ± {waic_ql.se:.2f}")
    except Exception as e:
        print(f"  Warning: Could not compute WAIC for Q-Learning: {e}")

    try:
        waic_wmrl = az.waic(idata_wmrl)
        results.append({
            'model': 'WM-RL',
            'metric': 'WAIC',
            'value': waic_wmrl.elpd_waic,
            'se': waic_wmrl.se
        })
        print(f"  WM-RL WAIC: {waic_wmrl.elpd_waic:.2f} ± {waic_wmrl.se:.2f}")
    except Exception as e:
        print(f"  Warning: Could not compute WAIC for WM-RL: {e}")

    # Compute LOO
    print("\n>> Computing LOO...")
    try:
        loo_ql = az.loo(idata_ql)
        results.append({
            'model': 'Q-Learning',
            'metric': 'LOO',
            'value': loo_ql.elpd_loo,
            'se': loo_ql.se
        })
        print(f"  Q-Learning LOO: {loo_ql.elpd_loo:.2f} ± {loo_ql.se:.2f}")
    except Exception as e:
        print(f"  Warning: Could not compute LOO for Q-Learning: {e}")

    try:
        loo_wmrl = az.loo(idata_wmrl)
        results.append({
            'model': 'WM-RL',
            'metric': 'LOO',
            'value': loo_wmrl.elpd_loo,
            'se': loo_wmrl.se
        })
        print(f"  WM-RL LOO: {loo_wmrl.elpd_loo:.2f} ± {loo_wmrl.se:.2f}")
    except Exception as e:
        print(f"  Warning: Could not compute LOO for WM-RL: {e}")

    return pd.DataFrame(results)


def plot_model_comparison_bar(
    df_metrics: pd.DataFrame,
    output_path: Path,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Create bar chart comparing model metrics (WAIC, LOO).

    Higher values are better for ELPD metrics.
    """
    # Set up plot
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique metrics
    metrics = df_metrics['metric'].unique()
    models = df_metrics['model'].unique()

    # Set up positions
    x = np.arange(len(metrics))
    width = 0.35

    # Colors
    colors = {'Q-Learning': '#1f77b4', 'WM-RL': '#ff7f0e'}

    # Plot bars for each model
    for i, model in enumerate(models):
        df_model = df_metrics[df_metrics['model'] == model]

        # Get values and errors for each metric
        values = []
        errors = []
        for metric in metrics:
            row = df_model[df_model['metric'] == metric]
            if len(row) > 0:
                values.append(row['value'].iloc[0])
                errors.append(row['se'].iloc[0])
            else:
                values.append(0)
                errors.append(0)

        offset = width * (i - len(models)/2 + 0.5)
        ax.bar(x + offset, values, width,
               label=model, color=colors.get(model, 'gray'),
               alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.errorbar(x + offset, values, yerr=errors,
                   fmt='none', ecolor='black', capsize=5, linewidth=2)

        # Add value labels on bars
        for j, (val, err) in enumerate(zip(values, errors)):
            ax.text(x[j] + offset, val + err + 5, f'{val:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Labels and title
    ax.set_xlabel('Information Criterion', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expected Log Pointwise Predictive Density (ELPD)',
                  fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison: Information Criteria\n(Higher is Better)',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)

    ax.legend(loc='best', fontsize=11, frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    print(f"Saving model comparison bar chart to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Model comparison chart saved successfully!")


def simulate_model_predictions(
    idata: az.InferenceData,
    data: pd.DataFrame,
    model_type: str
) -> np.ndarray:
    """
    Generate posterior predictive samples for goodness of fit.

    Returns array of shape (n_samples, n_trials) with predicted choices.
    """
    # This is a placeholder - would need actual posterior predictive sampling
    # For now, return dummy data
    print(f"  Note: Posterior predictive sampling not yet implemented for {model_type}")
    n_trials = len(data)
    return np.random.choice([0, 1, 2], size=(100, n_trials))


def plot_goodness_of_fit_matrix(
    idata_ql: az.InferenceData,
    idata_wmrl: az.InferenceData,
    data: pd.DataFrame,
    output_path: Path,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Create goodness of fit visualization comparing observed vs predicted choices.

    Shows confusion matrix style plot for each model.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    models = [
        ('Q-Learning', idata_ql, '#1f77b4'),
        ('WM-RL', idata_wmrl, '#ff7f0e')
    ]

    for ax, (model_name, idata, color) in zip(axes, models):
        # Extract observed choices (placeholder - would need actual data)
        # For now, create dummy confusion matrix
        confusion = np.array([
            [0.8, 0.15, 0.05],
            [0.1, 0.75, 0.15],
            [0.05, 0.1, 0.85]
        ])

        # Plot heatmap
        im = ax.imshow(confusion, cmap='Blues', aspect='auto', vmin=0, vmax=1)

        # Add text annotations
        for i in range(3):
            for j in range(3):
                text = ax.text(j, i, f'{confusion[i, j]:.2f}',
                             ha="center", va="center", color="black",
                             fontsize=12, fontweight='bold')

        # Labels
        ax.set_xlabel('Predicted Action', fontsize=11, fontweight='bold')
        ax.set_ylabel('Observed Action', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name}\nPredictive Accuracy',
                    fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['Action 0', 'Action 1', 'Action 2'])
        ax.set_yticklabels(['Action 0', 'Action 1', 'Action 2'])

    # Add colorbar
    fig.colorbar(im, ax=axes, orientation='horizontal',
                pad=0.1, fraction=0.05, label='Proportion')

    plt.suptitle('Goodness of Fit: Observed vs Predicted Actions\n(Placeholder - Needs Posterior Predictive Sampling)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    print(f"Saving goodness of fit matrix to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Goodness of fit plot saved successfully!")
    print(f"  Note: This is currently a placeholder visualization.")


def plot_model_weights(
    df_metrics: pd.DataFrame,
    output_path: Path,
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Compute and plot model weights based on WAIC.

    Uses Akaike weights: w_i = exp(-0.5 * Δ_i) / Σ exp(-0.5 * Δ_j)
    """
    # Get WAIC values
    df_waic = df_metrics[df_metrics['metric'] == 'WAIC'].copy()

    if len(df_waic) < 2:
        print("Not enough models with WAIC to compute weights")
        return

    # Compute relative WAIC (lower is better for raw WAIC, but we have ELPD which is higher is better)
    # Convert ELPD to WAIC: WAIC = -2 * ELPD
    df_waic['waic_raw'] = -2 * df_waic['value']
    min_waic = df_waic['waic_raw'].min()
    df_waic['delta_waic'] = df_waic['waic_raw'] - min_waic

    # Compute Akaike weights
    df_waic['weight'] = np.exp(-0.5 * df_waic['delta_waic'])
    df_waic['weight'] = df_waic['weight'] / df_waic['weight'].sum()

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    colors = {'Q-Learning': '#1f77b4', 'WM-RL': '#ff7f0e'}
    bar_colors = [colors.get(model, 'gray') for model in df_waic['model']]

    bars = ax.bar(df_waic['model'], df_waic['weight'],
                 color=bar_colors, alpha=0.8,
                 edgecolor='black', linewidth=2)

    # Add value labels
    for bar, weight in zip(bars, df_waic['weight']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{weight:.3f}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Model Weight', fontsize=12, fontweight='bold')
    ax.set_title('Akaike Model Weights (Based on WAIC)\nHigher weight = Better model',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    print(f"Saving model weights plot to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Model weights plot saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Plot model comparison metrics and goodness of fit'
    )
    parser.add_argument(
        '--qlearning',
        type=str,
        required=True,
        help='Path to Q-learning posterior NetCDF file'
    )
    parser.add_argument(
        '--wmrl',
        type=str,
        required=True,
        help='Path to WM-RL posterior NetCDF file'
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Path to behavioral data CSV (for goodness of fit)'
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
    print("MODEL COMPARISON VISUALIZATION")
    print("="*80)

    # Load both models
    print("\n>> Loading models...")
    idata_ql = load_posterior_samples(Path(args.qlearning), "qlearning")
    idata_wmrl = load_posterior_samples(Path(args.wmrl), "wmrl")

    # Compute comparison metrics
    print("\n>> Computing comparison metrics...")
    df_metrics = compute_model_comparison_metrics(idata_ql, idata_wmrl)
    print("\nMetrics summary:")
    print(df_metrics)

    # Create comparison bar chart
    print("\n>> Creating comparison bar chart...")
    prefix = f"{args.prefix}_" if args.prefix else ""
    bar_path = output_dir / f"{prefix}model_comparison_bar.png"
    plot_model_comparison_bar(df_metrics, bar_path)

    # Create model weights plot
    print("\n>> Creating model weights plot...")
    weights_path = output_dir / f"{prefix}model_weights.png"
    plot_model_weights(df_metrics, weights_path)

    # Create goodness of fit matrix
    if args.data:
        print("\n>> Creating goodness of fit matrix...")
        data = pd.read_csv(args.data)
        gof_path = output_dir / f"{prefix}goodness_of_fit.png"
        plot_goodness_of_fit_matrix(idata_ql, idata_wmrl, data, gof_path)
    else:
        print("\n>> Skipping goodness of fit (no data file provided)")

    print("\n" + "="*80)
    print("✓✓ All model comparison plots created successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
