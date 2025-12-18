"""
Analyze Fitted Parameters by Trauma Group

This script integrates:
1. Trauma group assignments (hypothesis-driven A/B/C classification)
2. Fitted model parameters (from JAX/NumPyro Bayesian fitting)
3. Behavioral performance metrics

Creates visualizations and statistical summaries showing how parameters
differ across trauma groups.

Usage:
    python scripts/analysis/analyze_parameters_by_trauma_group.py \
        --params output/v1/qlearning_jax_summary_*.csv \
        --posterior output/v1/qlearning_jax_posterior_*.nc \
        --output-dir figures/trauma_groups

Author: Generated for RLWM trauma analysis
Date: 2025-11-24
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import arviz for posterior analysis
try:
    import arviz as az
    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False
    print("Warning: arviz not installed. Posterior analysis will be limited.")

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_trauma_groups(path: Path = None) -> pd.DataFrame:
    """Load trauma group assignments."""
    if path is None:
        path = Path('output/trauma_groups/group_assignments.csv')

    if not path.exists():
        raise FileNotFoundError(
            f"Trauma group assignments not found at {path}. "
            "Run: python scripts/analysis/trauma_grouping_analysis.py"
        )

    df = pd.read_csv(path)
    print(f"✓ Loaded trauma groups for {len(df)} participants")

    # Filter to hypothesis groups only (exclude Low-High)
    df_filtered = df[df['hypothesis_group'] != 'Excluded_Low_High'].copy()
    print(f"  Using {len(df_filtered)} participants in hypothesis groups A/B/C")
    print(f"  (Excluded {len(df) - len(df_filtered)} participants)")

    return df_filtered


def get_participant_order(data_path: Path = None) -> list:
    """
    Get the order of participants as they appear in the data.

    This is critical for mapping parameter indices [0], [1], ...
    to actual sona_ids.
    """
    if data_path is None:
        data_path = Path('output/task_trials_long_all_participants.csv')

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Read just the participant column efficiently
    df = pd.read_csv(data_path, usecols=['sona_id'])

    # Get unique participants in order of first appearance
    participant_order = df['sona_id'].unique().tolist()

    print(f"✓ Detected participant order from data:")
    print(f"  {participant_order[:5]}..." if len(participant_order) > 5 else f"  {participant_order}")

    return participant_order


def load_fitted_parameters(
    summary_path: Path,
    participant_order: list,
    trauma_groups: pd.DataFrame
) -> pd.DataFrame:
    """
    Load fitted parameters and merge with trauma groups.

    Parameters
    ----------
    summary_path : Path
        Path to fitted parameter summary CSV
    participant_order : list
        Ordered list of sona_ids matching parameter indices
    trauma_groups : pd.DataFrame
        Trauma group assignments

    Returns
    -------
    pd.DataFrame
        Integrated dataframe with parameters and trauma groups
    """
    if not summary_path.exists():
        raise FileNotFoundError(f"Fitted parameters not found: {summary_path}")

    # Load parameter summary
    params_df = pd.read_csv(summary_path, index_col=0)

    print(f"\n✓ Loaded fitted parameters from: {summary_path.name}")

    # Extract individual-level parameters (exclude group-level mu/sigma and z-scores)
    individual_params = {}

    for param_name in ['alpha_pos', 'alpha_neg', 'beta']:
        # Find rows matching pattern: param_name[i]
        matching_rows = [idx for idx in params_df.index if idx.startswith(f'{param_name}[')]

        if not matching_rows:
            print(f"  Warning: No individual parameters found for {param_name}")
            continue

        # Extract index from param_name[i] and get posterior mean
        indices = []
        values = []
        for row in matching_rows:
            # Parse index: "alpha_pos[0]" -> 0
            idx = int(row.split('[')[1].split(']')[0])
            indices.append(idx)
            values.append(params_df.loc[row, 'mean'])

        individual_params[param_name] = dict(zip(indices, values))

    n_fitted = len(individual_params.get('alpha_pos', {}))
    print(f"  Found parameters for {n_fitted} participants")

    # Create dataframe mapping indices to sona_ids
    records = []
    for idx in range(n_fitted):
        if idx >= len(participant_order):
            print(f"  Warning: Parameter index {idx} exceeds participant list length")
            break

        sona_id = participant_order[idx]
        record = {'sona_id': sona_id, 'param_index': idx}

        for param_name, param_dict in individual_params.items():
            record[f'{param_name}_mean'] = param_dict.get(idx, np.nan)

        records.append(record)

    params_by_id = pd.DataFrame(records)

    # Merge with trauma groups
    integrated = trauma_groups.merge(params_by_id, on='sona_id', how='left')

    # Report merge success
    n_with_params = integrated['param_index'].notna().sum()
    n_total = len(integrated)
    print(f"\n✓ Merged with trauma groups: {n_with_params}/{n_total} have fitted parameters")

    return integrated


def print_summary_stats(df: pd.DataFrame, output_dir: Path):
    """Print and save summary statistics by trauma group."""
    print("\n" + "=" * 80)
    print("PARAMETER ESTIMATES BY TRAUMA GROUP")
    print("=" * 80)

    param_cols = [col for col in df.columns if col.endswith('_mean')]

    summary_data = []

    for group in ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']:
        group_data = df[df['hypothesis_group'] == group]
        n_total = len(group_data)
        n_fitted = group_data['param_index'].notna().sum()

        print(f"\n{group} (n={n_total}, fitted={n_fitted}):")

        for param_col in param_cols:
            param_name = param_col.replace('_mean', '')
            values = group_data[param_col].dropna()

            if len(values) == 0:
                print(f"  {param_name}: No data")
                continue

            mean = values.mean()
            std = values.std()
            sem = values.sem()

            print(f"  {param_name}: {mean:.3f} ± {std:.3f} (SEM: {sem:.3f})")

            summary_data.append({
                'group': group,
                'parameter': param_name,
                'n': len(values),
                'mean': mean,
                'std': std,
                'sem': sem,
                'min': values.min(),
                'max': values.max()
            })

    # Save to CSV
    summary_df = pd.DataFrame(summary_data)
    output_path = output_dir / 'parameter_summary_by_group.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved summary statistics: {output_path}")

    return summary_df


def plot_parameters_by_group(df: pd.DataFrame, output_dir: Path):
    """
    Create violin plots showing parameter distributions by trauma group.
    """
    print("\n>> Creating parameter distribution plots...")

    param_cols = [col for col in df.columns if col.endswith('_mean')]

    if not param_cols:
        print("  No parameters to plot")
        return

    # Filter to participants with fitted parameters
    df_fitted = df[df['param_index'].notna()].copy()

    if len(df_fitted) == 0:
        print("  No fitted parameters available for plotting")
        return

    if len(df_fitted) < 2:
        print(f"  Only {len(df_fitted)} participant(s) with fitted parameters")
        print("  Skipping violin plot (need 2+ for meaningful visualization)")
        print("  Re-run after fitting completes for better visualizations")
        return

    # Create figure
    n_params = len(param_cols)
    fig, axes = plt.subplots(1, n_params, figsize=(6 * n_params, 6))

    if n_params == 1:
        axes = [axes]

    # Group colors (matching trauma grouping script)
    group_colors = {
        'No Trauma': '#2ecc71',                    # Green
        'Trauma - No Ongoing Impact': '#f39c12',  # Orange
        'Trauma - Ongoing Impact': '#e74c3c'      # Red
    }

    group_labels = {
        'No Trauma': 'No Trauma',
        'Trauma - No Ongoing Impact': 'Trauma\nNo Ongoing\nImpact',
        'Trauma - Ongoing Impact': 'Trauma\nOngoing\nImpact'
    }

    for idx, param_col in enumerate(param_cols):
        ax = axes[idx]
        param_name = param_col.replace('_mean', '')

        # Prepare data for plotting
        plot_data = df_fitted[['hypothesis_group', param_col]].copy()
        plot_data = plot_data[plot_data['hypothesis_group'].isin(group_colors.keys())]

        # Collect data per group (filter out empty groups)
        group_data = {}
        for g in ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']:
            values = plot_data[plot_data['hypothesis_group'] == g][param_col].values
            if len(values) > 0:
                group_data[g] = values

        # Only create violin plot if we have at least one group with 2+ points
        violin_groups = {g: v for g, v in group_data.items() if len(v) >= 2}

        if violin_groups:
            # Create violin plot for groups with enough data
            positions = [['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact'].index(g) + 1
                        for g in violin_groups.keys()]
            parts = ax.violinplot(
                list(violin_groups.values()),
                positions=positions,
                showmeans=True,
                showmedians=False,
                widths=0.7
            )

            # Color the violins
            for pc, group in zip(parts['bodies'], violin_groups.keys()):
                pc.set_facecolor(group_colors[group])
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)

        # Overlay individual points with jitter (for all groups with data)
        for pos, group in enumerate(['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact'], 1):
            if group in group_data:
                group_values = group_data[group]
                jitter = np.random.normal(0, 0.05, len(group_values))
                ax.scatter(
                    pos + jitter,
                    group_values,
                    c=group_colors[group],
                    s=100,
                    alpha=0.8,
                    edgecolors='black',
                    linewidth=1,
                    zorder=3
                )
                # Add sample size
                y_pos = group_values.min() if len(group_values) > 0 else 0
                ax.text(pos, y_pos - 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                       f'n={len(group_values)}',
                       ha='center', va='top', fontsize=9)

        # Formatting
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([group_labels[g] for g in ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']])
        ax.set_ylabel(f'{param_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_title(f'{param_name.replace("_", " ").title()} Distribution',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim(0.5, 3.5)

    plt.suptitle('Model Parameters by Trauma Group',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    output_path = output_dir / 'parameters_by_trauma_group_violin.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")

    plt.close()


def plot_parameter_scatter_matrix(df: pd.DataFrame, output_dir: Path):
    """
    Create scatter plot matrix showing parameter relationships colored by trauma group.
    """
    print("\n>> Creating parameter scatter matrix...")

    param_cols = [col for col in df.columns if col.endswith('_mean')]

    if len(param_cols) < 2:
        print("  Need at least 2 parameters for scatter matrix")
        return

    # Filter to fitted participants
    df_fitted = df[df['param_index'].notna()].copy()

    if len(df_fitted) == 0:
        print("  No fitted parameters available")
        return

    if len(df_fitted) < 2:
        print(f"  Only {len(df_fitted)} participant(s) with fitted parameters")
        print("  Skipping scatter matrix (need 2+ for meaningful visualization)")
        return

    # Group colors
    group_colors = {
        'No Trauma': '#2ecc71',
        'Trauma - No Ongoing Impact': '#f39c12',
        'Trauma - Ongoing Impact': '#e74c3c'
    }

    # Create pairplot
    plot_data = df_fitted[param_cols + ['hypothesis_group']].copy()
    plot_data = plot_data[plot_data['hypothesis_group'].isin(group_colors.keys())]

    # Rename columns for cleaner labels
    rename_dict = {col: col.replace('_mean', '').replace('_', ' ').title()
                   for col in param_cols}
    plot_data = plot_data.rename(columns=rename_dict)

    g = sns.pairplot(
        plot_data,
        hue='hypothesis_group',
        palette=group_colors,
        diag_kind='kde',
        plot_kws={'alpha': 0.7, 's': 100, 'edgecolor': 'black', 'linewidth': 1},
        diag_kws={'alpha': 0.7, 'linewidth': 2}
    )

    g.fig.suptitle('Parameter Relationships by Trauma Group',
                   fontsize=16, fontweight='bold', y=1.01)

    # Update legend
    g._legend.set_title('Trauma Group')
    new_labels = ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']
    for t, label in zip(g._legend.texts, new_labels):
        t.set_text(label)

    # Save
    output_path = output_dir / 'parameter_scatter_matrix_by_group.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")

    plt.close()


def plot_trauma_parameter_correlations(df: pd.DataFrame, output_dir: Path):
    """
    Create heatmap showing correlations between trauma measures and parameters.
    """
    print("\n>> Creating trauma-parameter correlation heatmap...")

    # Select relevant columns
    trauma_cols = ['lec_total_events', 'ies_total']
    param_cols = [col for col in df.columns if col.endswith('_mean')]

    if not param_cols:
        print("  No parameters available")
        return

    df_fitted = df[df['param_index'].notna()].copy()

    if len(df_fitted) < 3:
        print("  Not enough participants for correlation analysis")
        return

    # Compute correlations
    corr_data = df_fitted[trauma_cols + param_cols]
    corr_matrix = corr_data.corr()

    # Extract trauma-parameter correlations only
    trauma_param_corr = corr_matrix.loc[trauma_cols, param_cols]

    # Clean up labels
    trauma_param_corr.index = ['LEC Total Events', 'IES-R Total Score']
    trauma_param_corr.columns = [col.replace('_mean', '').replace('_', ' ').title()
                                  for col in param_cols]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        trauma_param_corr,
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={'label': 'Spearman Correlation'},
        linewidths=1,
        linecolor='black',
        ax=ax
    )

    ax.set_title('Correlations: Trauma Measures × Model Parameters',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Model Parameters', fontsize=12, fontweight='bold')
    ax.set_ylabel('Trauma Measures', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = output_dir / 'trauma_parameter_correlations.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")

    plt.close()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Analyze fitted parameters by trauma group'
    )
    parser.add_argument(
        '--params',
        type=str,
        default='output/v1/qlearning_jax_summary_*.csv',
        help='Path to fitted parameter summary CSV (supports wildcards)'
    )
    parser.add_argument(
        '--posterior',
        type=str,
        default=None,
        help='Path to posterior NetCDF file (optional, for full posterior analysis)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='output/task_trials_long_all_participants.csv',
        help='Path to trial data (for participant ordering)'
    )
    parser.add_argument(
        '--trauma-groups',
        type=str,
        default='output/trauma_groups/group_assignments.csv',
        help='Path to trauma group assignments'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='figures/trauma_groups',
        help='Output directory for figures'
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ANALYZING PARAMETERS BY TRAUMA GROUP")
    print("=" * 80)

    # Load data
    print("\n>> Loading data...")
    trauma_groups = load_trauma_groups(Path(args.trauma_groups))
    participant_order = get_participant_order(Path(args.data))

    # Find parameter summary file (handle wildcards)
    from glob import glob
    param_files = glob(args.params)

    if not param_files:
        raise FileNotFoundError(f"No parameter files found matching: {args.params}")

    # Use most recent file
    param_file = Path(sorted(param_files)[-1])
    print(f"\n>> Using parameter file: {param_file.name}")

    # Load and merge parameters
    integrated_df = load_fitted_parameters(param_file, participant_order, trauma_groups)

    # Save integrated dataset
    integrated_path = output_dir.parent / 'trauma_groups' / 'integrated_parameters.csv'
    integrated_path.parent.mkdir(parents=True, exist_ok=True)
    integrated_df.to_csv(integrated_path, index=False)
    print(f"\n✓ Saved integrated dataset: {integrated_path}")

    # Generate summaries and visualizations
    print("\n" + "=" * 80)
    print("GENERATING SUMMARIES AND VISUALIZATIONS")
    print("=" * 80)

    summary_stats = print_summary_stats(integrated_df, output_dir.parent / 'trauma_groups')

    plot_parameters_by_group(integrated_df, output_dir)
    plot_parameter_scatter_matrix(integrated_df, output_dir)
    plot_trauma_parameter_correlations(integrated_df, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}/")
    print("\nNext steps:")
    print("  1. Review parameter distributions by trauma group")
    print("  2. Consider running statistical tests (e.g., Kruskal-Wallis)")
    print("  3. Examine correlations between trauma severity and parameters")
    print()


if __name__ == '__main__':
    main()
