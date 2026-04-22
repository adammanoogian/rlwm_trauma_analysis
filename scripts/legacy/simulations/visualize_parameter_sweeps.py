"""
Parameter Sweep Visualization

Functions for visualizing parameter sweep results, including individual model
plots and comparative visualizations across Q-learning and WM-RL models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.legacy.analysis.plotting_utils import (
    get_color_palette,
    save_figure,
    setup_plot_style,
)


def plot_qlearning_sweep(
    df: pd.DataFrame,
    save_dir: Path | None = None,
    show: bool = False
) -> plt.Figure:
    """
    Create comprehensive visualization of Q-learning parameter sweep.

    Creates 4 subplots showing effects of alpha_pos, alpha_neg, beta, and heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Q-learning sweep results with columns: alpha_pos, alpha_neg, beta, set_size, accuracy_mean
    save_dir : Path, optional
        Directory to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    plt.Figure
    """
    setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = get_color_palette('set_size')
    set_sizes = sorted(df['set_size'].unique())

    # Plot 1: Alpha_pos effect (average across alpha_neg)
    ax = axes[0, 0]
    for ss in set_sizes:
        ss_data = df[df['set_size'] == ss]
        grouped = ss_data.groupby('alpha_pos')['accuracy_mean'].mean()
        ax.plot(grouped.index, grouped.values, 'o-', color=colors[ss], label=f'Set Size {ss}', linewidth=2)

    ax.set_xlabel('Alpha+ (Positive PE Learning Rate)', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Effect of Positive PE Learning Rate', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5, label='Chance')

    # Plot 2: Alpha_neg effect (average across alpha_pos)
    ax = axes[0, 1]
    for ss in set_sizes:
        ss_data = df[df['set_size'] == ss]
        grouped = ss_data.groupby('alpha_neg')['accuracy_mean'].mean()
        ax.plot(grouped.index, grouped.values, 's-', color=colors[ss], label=f'Set Size {ss}', linewidth=2)

    ax.set_xlabel('Alpha- (Negative PE Learning Rate)', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Effect of Negative PE Learning Rate', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5)

    # Plot 3: Beta effect
    ax = axes[1, 0]
    for ss in set_sizes:
        ss_data = df[df['set_size'] == ss]
        grouped = ss_data.groupby('beta')['accuracy_mean'].mean()
        ax.plot(grouped.index, grouped.values, '^-', color=colors[ss], label=f'Set Size {ss}', linewidth=2)

    ax.set_xlabel('Beta (Inverse Temperature)', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Effect of Inverse Temperature', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5)

    # Plot 4: Heatmap (alpha_pos x beta, averaged across alpha_neg and set_sizes)
    ax = axes[1, 1]
    pivot_data = df.groupby(['alpha_pos', 'beta'])['accuracy_mean'].mean().reset_index()
    pivot_table = pivot_data.pivot(index='alpha_pos', columns='beta', values='accuracy_mean')

    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                ax=ax, cbar_kws={'label': 'Accuracy'})
    ax.set_xlabel('Beta (Inverse Temperature)', fontweight='bold')
    ax.set_ylabel('Alpha+ (Positive PE Learning Rate)', fontweight='bold')
    ax.set_title('Accuracy Heatmap: Alpha+ × Beta', fontweight='bold')

    fig.suptitle('Q-Learning Parameter Sweep Results', fontsize=16, fontweight='bold', y=0.995)
    fig.tight_layout()

    if save_dir:
        save_figure(fig, 'qlearning_individual', subdir='parameter_sweeps')

    if show:
        pass  # plt.show() removed for headless compatibility
    else:
        plt.close(fig)

    return fig

def plot_wmrl_sweep(
    df: pd.DataFrame,
    save_dir: Path | None = None,
    show: bool = False
) -> plt.Figure:
    """
    Create comprehensive visualization of WM-RL parameter sweep.

    Creates 4 subplots showing effects of capacity, rho, phi, and heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        WM-RL sweep results
    save_dir : Path, optional
        Directory to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    plt.Figure
    """
    setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = get_color_palette('set_size')
    set_sizes = sorted(df['set_size'].unique())

    # Plot 1: Capacity effect
    ax = axes[0, 0]
    for ss in set_sizes:
        ss_data = df[df['set_size'] == ss]
        grouped = ss_data.groupby('capacity')['accuracy_mean'].mean()
        ax.plot(grouped.index, grouped.values, 'o-', color=colors[ss], label=f'Set Size {ss}', linewidth=2)

    ax.set_xlabel('WM Capacity (K)', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Effect of WM Capacity', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Rho (base WM reliance) effect
    ax = axes[0, 1]
    for ss in set_sizes:
        ss_data = df[df['set_size'] == ss]
        grouped = ss_data.groupby('rho')['accuracy_mean'].mean()
        ax.plot(grouped.index, grouped.values, 's-', color=colors[ss], label=f'Set Size {ss}', linewidth=2)

    ax.set_xlabel('Rho (Base WM Reliance)', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Effect of WM Reliance Parameter', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5)

    # Plot 3: Phi (decay) effect
    ax = axes[1, 0]
    for ss in set_sizes:
        ss_data = df[df['set_size'] == ss]
        grouped = ss_data.groupby('phi')['accuracy_mean'].mean()
        ax.plot(grouped.index, grouped.values, '^-', color=colors[ss], label=f'Set Size {ss}', linewidth=2)

    ax.set_xlabel('Phi (WM Decay Rate)', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Effect of WM Decay', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5)

    # Plot 4: Heatmap (capacity x rho)
    ax = axes[1, 1]
    pivot_data = df.groupby(['capacity', 'rho'])['accuracy_mean'].mean().reset_index()
    pivot_table = pivot_data.pivot(index='capacity', columns='rho', values='accuracy_mean')

    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                ax=ax, cbar_kws={'label': 'Accuracy'})
    ax.set_xlabel('Rho (Base WM Reliance)', fontweight='bold')
    ax.set_ylabel('Capacity (K)', fontweight='bold')
    ax.set_title('Accuracy Heatmap: Capacity × Rho', fontweight='bold')

    fig.suptitle('WM-RL Parameter Sweep Results', fontsize=16, fontweight='bold', y=0.995)
    fig.tight_layout()

    if save_dir:
        save_figure(fig, 'wmrl_individual', subdir='parameter_sweeps')

    if show:
        pass  # plt.show() removed for headless compatibility
    else:
        plt.close(fig)

    return fig

def plot_comparative_accuracy_by_setsize(
    qlearning_df: pd.DataFrame,
    wmrl_df: pd.DataFrame,
    save_dir: Path | None = None,
    show: bool = False
) -> plt.Figure:
    """
    Compare Q-learning vs WM-RL accuracy across set sizes (best parameters).

    Parameters
    ----------
    qlearning_df : pd.DataFrame
        Q-learning results
    wmrl_df : pd.DataFrame
        WM-RL results
    save_dir : Path, optional
        Save directory
    show : bool
        Display figure

    Returns
    -------
    plt.Figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = get_color_palette('set_size')

    # Get best parameters for each model and set size
    set_sizes = sorted(qlearning_df['set_size'].unique())

    q_best_accs = []
    wmrl_best_accs = []

    for ss in set_sizes:
        # Q-learning best
        q_ss = qlearning_df[qlearning_df['set_size'] == ss]
        best_q = q_ss.loc[q_ss['accuracy_mean'].idxmax()]
        q_best_accs.append(best_q['accuracy_mean'])

        # WM-RL best
        wmrl_ss = wmrl_df[wmrl_df['set_size'] == ss]
        best_wmrl = wmrl_ss.loc[wmrl_ss['accuracy_mean'].idxmax()]
        wmrl_best_accs.append(best_wmrl['accuracy_mean'])

    x = np.arange(len(set_sizes))
    width = 0.35

    bars1 = ax.bar(x - width/2, q_best_accs, width, label='Q-Learning (Best)',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, wmrl_best_accs, width, label='WM-RL (Best)',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Set Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (Best Parameters)', fontsize=14, fontweight='bold')
    ax.set_title('Model Comparison: Best Accuracy by Set Size', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticks(set_sizes)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1/3, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Chance')
    ax.set_ylim(0, 1.05)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    if save_dir:
        save_figure(fig, 'comparative_accuracy', subdir='parameter_sweeps')

    if show:
        pass  # plt.show() removed for headless compatibility
    else:
        plt.close(fig)

    return fig

def plot_comparative_heatmaps(
    qlearning_df: pd.DataFrame,
    wmrl_df: pd.DataFrame,
    save_dir: Path | None = None,
    show: bool = False
) -> plt.Figure:
    """
    Side-by-side heatmaps comparing Q-learning and WM-RL parameter spaces.

    Parameters
    ----------
    qlearning_df : pd.DataFrame
        Q-learning results
    wmrl_df : pd.DataFrame
        WM-RL results
    save_dir : Path, optional
        Save directory
    show : bool
        Display figure

    Returns
    -------
    plt.Figure
    """
    setup_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Q-learning heatmap (alpha_pos x beta)
    ax = axes[0]
    q_pivot = qlearning_df.groupby(['alpha_pos', 'beta'])['accuracy_mean'].mean().reset_index()
    q_table = q_pivot.pivot(index='alpha_pos', columns='beta', values='accuracy_mean')

    sns.heatmap(q_table, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.3, vmax=0.9,
                ax=ax, cbar_kws={'label': 'Accuracy'})
    ax.set_xlabel('Beta (Inverse Temperature)', fontweight='bold')
    ax.set_ylabel('Alpha+ (Positive PE LR)', fontweight='bold')
    ax.set_title('Q-Learning: Alpha+ × Beta', fontweight='bold', fontsize=14)

    # WM-RL heatmap (capacity x rho)
    ax = axes[1]
    wmrl_pivot = wmrl_df.groupby(['capacity', 'rho'])['accuracy_mean'].mean().reset_index()
    wmrl_table = wmrl_pivot.pivot(index='capacity', columns='rho', values='accuracy_mean')

    sns.heatmap(wmrl_table, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.3, vmax=0.9,
                ax=ax, cbar_kws={'label': 'Accuracy'})
    ax.set_xlabel('Rho (Base WM Reliance)', fontweight='bold')
    ax.set_ylabel('Capacity (K)', fontweight='bold')
    ax.set_title('WM-RL: Capacity × Rho', fontweight='bold', fontsize=14)

    fig.suptitle('Parameter Space Comparison', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()

    if save_dir:
        save_figure(fig, 'comparative_heatmaps', subdir='parameter_sweeps')

    if show:
        pass  # plt.show() removed for headless compatibility
    else:
        plt.close(fig)

    return fig

def main():
    """
    Load and visualize parameter sweep results.
    """
    print("=" * 80)
    print("PARAMETER SWEEP VISUALIZATION")
    print("=" * 80)
    print()

    # Load data
    sweep_dir = project_root / 'output' / 'parameter_sweeps'

    # Find most recent sweep files
    q_files = list(sweep_dir.glob('*qlearning*.csv'))
    wmrl_files = list(sweep_dir.glob('*wmrl*.csv'))

    if not q_files or not wmrl_files:
        print("ERROR: No sweep results found in output/parameter_sweeps/")
        print("Run a parameter sweep first!")
        return

    q_file = sorted(q_files)[-1]  # Most recent
    wmrl_file = sorted(wmrl_files)[-1]

    print(f"Loading Q-learning results: {q_file.name}")
    qlearning_df = pd.read_csv(q_file)

    print(f"Loading WM-RL results: {wmrl_file.name}")
    wmrl_df = pd.read_csv(wmrl_file)

    print()
    print("-" * 80)
    print("Generating visualizations...")
    print("-" * 80)
    print()

    # Create visualizations
    print("1. Q-learning individual plots...")
    plot_qlearning_sweep(qlearning_df)

    print("2. WM-RL individual plots...")
    plot_wmrl_sweep(wmrl_df)

    print("3. Comparative accuracy plot...")
    plot_comparative_accuracy_by_setsize(qlearning_df, wmrl_df)

    print("4. Comparative heatmaps...")
    plot_comparative_heatmaps(qlearning_df, wmrl_df)

    print()
    print("=" * 80)
    print("COMPLETE! All visualizations saved to figures/parameter_sweeps/")
    print("=" * 80)

if __name__ == "__main__":
    main()
