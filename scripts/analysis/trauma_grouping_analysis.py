"""
Trauma-Based Participant Grouping Analysis

Implements two approaches to grouping participants based on trauma measures:
1. Hypothesis-driven: 3-group classification using median splits
2. Unsupervised: Hierarchical clustering to discover natural groupings

Usage:
    python scripts/analysis/trauma_grouping_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Import plotting config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from plotting_config import PlotConfig, ScatterPlotConfig

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# Paths
OUTPUT_DIR = Path('output/trauma_groups')
FIGURES_DIR = Path('figures/trauma_groups')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_participant_data():
    """Load participant summary data with trauma scores."""
    data_path = Path('output/summary_participant_metrics.csv')

    if not data_path.exists():
        raise FileNotFoundError(
            f"Participant data not found at {data_path}. "
            "Run the data pipeline first: python run_data_pipeline.py"
        )

    df = pd.read_csv(data_path)

    # Check required columns
    required_cols = ['sona_id', 'lec_total_events', 'ies_total',
                     'accuracy_overall', 'mean_rt_overall']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Remove any rows with missing trauma scores
    df_clean = df.dropna(subset=['lec_total_events', 'ies_total'])

    print(f"Loaded {len(df_clean)} participants with complete trauma data")
    print(f"(excluded {len(df) - len(df_clean)} participants with missing data)")

    return df_clean

def create_hypothesis_groups(df):
    """
    Create 3-group classification based on median splits.

    Groups:
    - Group A (Low-Low): Low LEC, Low IES
    - Group B (High-Low): High LEC, Low IES
    - Group C (High-High): High LEC, High IES

    Note: Low LEC + High IES is theoretically inconsistent, so excluded if present
    """
    # Calculate medians
    lec_median = df['lec_total_events'].median()
    ies_median = df['ies_total'].median()

    print("\n" + "=" * 80)
    print("HYPOTHESIS-DRIVEN GROUPING (MEDIAN SPLITS)")
    print("=" * 80)
    print(f"\nLEC-5 Total Events Median: {lec_median}")
    print(f"IES-R Total Score Median: {ies_median}")

    # Create binary splits
    df['lec_high'] = (df['lec_total_events'] >= lec_median).astype(int)
    df['ies_high'] = (df['ies_total'] >= ies_median).astype(int)

    # Create group labels
    def assign_group(row):
        if row['lec_high'] == 0 and row['ies_high'] == 0:
            return 'No Trauma'
        elif row['lec_high'] == 1 and row['ies_high'] == 0:
            return 'Trauma - No Ongoing Impact'
        elif row['lec_high'] == 1 and row['ies_high'] == 1:
            return 'Trauma - Ongoing Impact'
        else:  # Low LEC, High IES - theoretically inconsistent
            return 'Excluded_Low_High'

    df['hypothesis_group'] = df.apply(assign_group, axis=1)

    # Print group sizes
    print("\nGroup Sizes:")
    group_counts = df['hypothesis_group'].value_counts().sort_index()
    for group, count in group_counts.items():
        print(f"  {group}: n={count}")

    # Warn if any Low-High participants
    n_excluded = (df['hypothesis_group'] == 'Excluded_Low_High').sum()
    if n_excluded > 0:
        print(f"\nWARNING: {n_excluded} participant(s) with Low LEC + High IES")
        print("This pattern is theoretically inconsistent (PTSD without trauma exposure)")

    # Print group characteristics
    print("\nGroup Characteristics (Mean ± SD):")
    for group in ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']:
        if group in df['hypothesis_group'].values:
            group_data = df[df['hypothesis_group'] == group]
            lec_mean = group_data['lec_total_events'].mean()
            lec_std = group_data['lec_total_events'].std()
            ies_mean = group_data['ies_total'].mean()
            ies_std = group_data['ies_total'].std()

            print(f"\n  {group}:")
            print(f"    LEC: {lec_mean:.2f} ± {lec_std:.2f}")
            print(f"    IES-R: {ies_mean:.2f} ± {ies_std:.2f}")

    return df, lec_median, ies_median

def perform_hierarchical_clustering(df, k_range=[2, 3, 4]):
    """
    Perform hierarchical clustering with Ward linkage.

    Tests multiple cluster solutions (k=2, 3, 4) and evaluates quality.
    """
    print("\n" + "=" * 80)
    print("HIERARCHICAL CLUSTERING ANALYSIS")
    print("=" * 80)

    # Prepare features
    features = df[['lec_total_events', 'ies_total']].values

    print(f"\nFeatures: LEC Total Events, IES-R Total Score")
    print(f"N participants: {len(features)}")

    # Standardize features (z-scores)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    print("\nStandardizing features (z-scores) to give equal weight...")
    print(f"  LEC mean={scaler.mean_[0]:.2f}, std={scaler.scale_[0]:.2f}")
    print(f"  IES-R mean={scaler.mean_[1]:.2f}, std={scaler.scale_[1]:.2f}")

    # Compute linkage matrix
    print("\nComputing hierarchical clustering with Ward linkage...")
    linkage_matrix = linkage(features_scaled, method='ward')

    # Test different numbers of clusters
    print("\n" + "-" * 80)
    print("Testing cluster solutions:")
    print("-" * 80)

    silhouette_scores = {}
    cluster_assignments = {}

    for k in k_range:
        # Cut dendrogram to get k clusters
        clusters = fcluster(linkage_matrix, t=k, criterion='maxclust')
        cluster_assignments[k] = clusters

        # Calculate silhouette score
        if k > 1 and k < len(features):
            sil_score = silhouette_score(features_scaled, clusters)
            silhouette_scores[k] = sil_score

            # Get cluster sizes
            unique, counts = np.unique(clusters, return_counts=True)
            cluster_sizes = dict(zip(unique, counts))

            print(f"\nk={k} clusters:")
            print(f"  Silhouette score: {sil_score:.3f}")
            print(f"  Cluster sizes: {cluster_sizes}")

    # Identify optimal k
    if silhouette_scores:
        optimal_k = max(silhouette_scores, key=silhouette_scores.get)
        print(f"\n>>> Optimal k={optimal_k} (highest silhouette: {silhouette_scores[optimal_k]:.3f})")
    else:
        optimal_k = 3  # Default to 3 to match hypothesis

    # Add cluster assignments to dataframe
    for k in k_range:
        df[f'cluster_k{k}'] = cluster_assignments[k]

    return df, linkage_matrix, features_scaled, silhouette_scores, optimal_k

def plot_hypothesis_groups(df, lec_median, ies_median):
    """Visualize the 3-group hypothesis-driven classification."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define colors and markers for each group
    group_colors = {
        'No Trauma': '#2ecc71',                    # Green - no trauma exposure, no symptoms
        'Trauma - No Ongoing Impact': '#f39c12',  # Orange - trauma exposure, no ongoing symptoms
        'Trauma - Ongoing Impact': '#e74c3c',     # Red - trauma exposure, ongoing symptoms
        'Excluded_Low_High': '#95a5a6'            # Gray - excluded (theoretically inconsistent)
    }

    group_labels = {
        'No Trauma': 'No Trauma',
        'Trauma - No Ongoing Impact': 'Trauma - No Ongoing Impact',
        'Trauma - Ongoing Impact': 'Trauma - Ongoing Impact',
        'Excluded_Low_High': 'Excluded: Low Trauma, High Symptoms'
    }

    # Plot each group - use ScatterPlotConfig for marker properties
    for group in df['hypothesis_group'].unique():
        group_data = df[df['hypothesis_group'] == group]
        ax.scatter(
            group_data['lec_total_events'],
            group_data['ies_total'],
            c=group_colors.get(group, 'gray'),
            label=group_labels.get(group, group),
            s=ScatterPlotConfig.SIZE,
            alpha=ScatterPlotConfig.ALPHA,
            edgecolors=ScatterPlotConfig.EDGE_COLOR,
            linewidth=ScatterPlotConfig.EDGE_WIDTH
        )

    # Add median lines
    ax.axvline(lec_median, color='black', linestyle='--', alpha=0.5, linewidth=1, label=f'LEC Median ({lec_median})')
    ax.axhline(ies_median, color='black', linestyle='--', alpha=0.5, linewidth=1, label=f'IES-R Median ({ies_median})')

    # Formatting - use PlotConfig for font sizes
    ax.set_xlabel('LEC-5 Total Events', fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('IES-R Total Score', fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
    ax.set_title('Hypothesis-Driven Trauma Grouping\n(Median Split Classification)',
                 fontsize=PlotConfig.TITLE_SIZE, fontweight='bold', pad=PlotConfig.PAD)
    ax.legend(loc='upper left', fontsize=PlotConfig.LEGEND_SIZE, framealpha=0.9)
    ax.grid(True, alpha=PlotConfig.GRID_ALPHA)
    ax.tick_params(axis='both', which='major', labelsize=PlotConfig.TICK_LABEL_SIZE)

    # Add quadrant labels
    ax.text(0.02, 0.98, 'Low Exposure\nLow Symptoms',
            transform=ax.transAxes, fontsize=PlotConfig.ANNOTATION_SIZE, alpha=0.5,
            verticalalignment='top', horizontalalignment='left')
    ax.text(0.98, 0.98, 'High Exposure\nLow Symptoms',
            transform=ax.transAxes, fontsize=PlotConfig.ANNOTATION_SIZE, alpha=0.5,
            verticalalignment='top', horizontalalignment='right')
    ax.text(0.98, 0.02, 'High Exposure\nHigh Symptoms',
            transform=ax.transAxes, fontsize=PlotConfig.ANNOTATION_SIZE, alpha=0.5,
            verticalalignment='bottom', horizontalalignment='right')

    plt.tight_layout()

    # Save
    output_path = FIGURES_DIR / 'hypothesis_groups_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[SAVED] {output_path}")

    plt.close()

def plot_dendrogram(linkage_matrix, df):
    """Create dendrogram visualization."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create dendrogram
    dendrogram(
        linkage_matrix,
        ax=ax,
        labels=df['sona_id'].astype(str).values,
        leaf_font_size=10,
        color_threshold=0,
        above_threshold_color='black'
    )

    ax.set_xlabel('Participant ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ward Linkage Distance', fontsize=12, fontweight='bold')
    ax.set_title('Hierarchical Clustering Dendrogram\n(Ward Linkage on Standardized LEC + IES-R)',
                 fontsize=14, fontweight='bold', pad=20)

    # Add horizontal lines at common cut points for k=2,3,4
    # These are approximate - adjust based on actual dendrogram
    ax.axhline(y=ax.get_ylim()[1]*0.6, color='red', linestyle='--',
               alpha=0.5, linewidth=1, label='Potential cut for k=2')
    ax.axhline(y=ax.get_ylim()[1]*0.4, color='orange', linestyle='--',
               alpha=0.5, linewidth=1, label='Potential cut for k=3')

    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    # Save
    output_path = FIGURES_DIR / 'hierarchical_dendrogram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {output_path}")

    plt.close()

def plot_cluster_comparison(df, features_scaled, optimal_k):
    """Compare different cluster solutions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    k_values = [2, 3, 4]

    for idx, k in enumerate(k_values):
        ax = axes[idx]

        if f'cluster_k{k}' in df.columns:
            clusters = df[f'cluster_k{k}']

            # Plot
            scatter = ax.scatter(
                df['lec_total_events'],
                df['ies_total'],
                c=clusters,
                cmap='viridis',
                s=150,
                alpha=0.7,
                edgecolors='black',
                linewidth=1.5
            )

            # Formatting
            ax.set_xlabel('LEC-5 Total Events', fontsize=12, fontweight='bold')
            ax.set_ylabel('IES-R Total Score', fontsize=12, fontweight='bold')

            title_suffix = " (Optimal)" if k == optimal_k else ""
            ax.set_title(f'k={k} Clusters{title_suffix}', fontsize=13, fontweight='bold')

            ax.grid(True, alpha=0.3)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Cluster', fontsize=10)

    plt.suptitle('Hierarchical Clustering: Comparing Different k Solutions',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    output_path = FIGURES_DIR / 'cluster_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {output_path}")

    plt.close()

def plot_silhouette_analysis(df, features_scaled, silhouette_scores):
    """Create silhouette analysis plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    k_values = sorted(silhouette_scores.keys())
    scores = [silhouette_scores[k] for k in k_values]

    # Bar plot
    bars = ax.bar(k_values, scores, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)

    # Highlight best
    max_idx = scores.index(max(scores))
    bars[max_idx].set_color('darkgreen')
    bars[max_idx].set_alpha(0.9)

    # Formatting
    ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax.set_title('Cluster Quality: Silhouette Analysis\n(Higher = Better Separation)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1,
               label='Good threshold (>0.5)')
    ax.legend()

    # Add value labels on bars
    for i, (k, score) in enumerate(zip(k_values, scores)):
        ax.text(k, score + 0.02, f'{score:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = FIGURES_DIR / 'cluster_silhouette.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {output_path}")

    plt.close()

def compare_grouping_methods(df, optimal_k):
    """Compare hypothesis-driven vs unsupervised clustering."""
    print("\n" + "=" * 80)
    print("COMPARING GROUPING METHODS")
    print("=" * 80)

    # Create cross-tabulation
    crosstab = pd.crosstab(
        df['hypothesis_group'],
        df[f'cluster_k{optimal_k}'],
        rownames=['Hypothesis Group'],
        colnames=[f'Cluster (k={optimal_k})']
    )

    print(f"\nCross-Tabulation (Hypothesis Groups vs k={optimal_k} Clustering):")
    print(crosstab)

    # Chi-square test of independence
    if crosstab.shape[0] > 1 and crosstab.shape[1] > 1:
        chi2, p_value, dof, expected = chi2_contingency(crosstab)
        print(f"\nChi-square test of independence:")
        print(f"  χ² = {chi2:.3f}, p = {p_value:.4f}, df = {dof}")

        if p_value < 0.05:
            print("  >>> Significant association between methods (p < 0.05)")
        else:
            print("  >>> No significant association (p ≥ 0.05)")

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        crosstab,
        annot=True,
        fmt='d',
        cmap='YlOrRd',
        cbar_kws={'label': 'Count'},
        linewidths=1,
        linecolor='black',
        ax=ax
    )

    ax.set_title(f'Concordance: Hypothesis Groups vs Hierarchical Clustering (k={optimal_k})',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(f'Hierarchical Cluster (k={optimal_k})', fontsize=12, fontweight='bold')
    ax.set_ylabel('Hypothesis Group', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = FIGURES_DIR / 'group_comparison_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[SAVED] {output_path}")

    plt.close()

    return crosstab

def save_results(df, silhouette_scores, optimal_k, lec_median, ies_median):
    """Save all grouping results to CSV files."""
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Save group assignments
    output_cols = ['sona_id', 'lec_total_events', 'ies_total',
                   'hypothesis_group', 'cluster_k2', 'cluster_k3', 'cluster_k4']
    output_cols = [col for col in output_cols if col in df.columns]

    assignments_path = OUTPUT_DIR / 'group_assignments.csv'
    df[output_cols].to_csv(assignments_path, index=False)
    print(f"\n[SAVED] {assignments_path}")

    # Save group summary statistics
    summary_data = []

    # Hypothesis groups (exclude theoretically inconsistent group)
    for group in ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']:
        if group in df['hypothesis_group'].values:
            group_data = df[df['hypothesis_group'] == group]
            summary_data.append({
                'method': 'hypothesis',
                'group': group,
                'n': len(group_data),
                'lec_mean': group_data['lec_total_events'].mean(),
                'lec_std': group_data['lec_total_events'].std(),
                'ies_mean': group_data['ies_total'].mean(),
                'ies_std': group_data['ies_total'].std()
            })

    # Clustering groups
    for k in [2, 3, 4]:
        if f'cluster_k{k}' in df.columns:
            for cluster in df[f'cluster_k{k}'].unique():
                cluster_data = df[df[f'cluster_k{k}'] == cluster]
                summary_data.append({
                    'method': f'hierarchical_k{k}',
                    'group': f'cluster_{cluster}',
                    'n': len(cluster_data),
                    'lec_mean': cluster_data['lec_total_events'].mean(),
                    'lec_std': cluster_data['lec_total_events'].std(),
                    'ies_mean': cluster_data['ies_total'].mean(),
                    'ies_std': cluster_data['ies_total'].std()
                })

    summary_df = pd.DataFrame(summary_data)
    summary_path = OUTPUT_DIR / 'group_summary_stats.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"[SAVED] {summary_path}")

    # Save clustering metrics
    metrics_data = {
        'k': list(silhouette_scores.keys()),
        'silhouette_score': list(silhouette_scores.values())
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df['optimal'] = metrics_df['k'] == optimal_k

    metrics_path = OUTPUT_DIR / 'clustering_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[SAVED] {metrics_path}")

    # Save cutoff values
    cutoffs_data = {
        'measure': ['LEC-5 Total Events', 'IES-R Total Score'],
        'cutoff_type': ['median', 'median'],
        'cutoff_value': [lec_median, ies_median]
    }
    cutoffs_df = pd.DataFrame(cutoffs_data)
    cutoffs_path = OUTPUT_DIR / 'cutoff_values.csv'
    cutoffs_df.to_csv(cutoffs_path, index=False)
    print(f"[SAVED] {cutoffs_path}")

def main():
    """Main execution function."""
    print("=" * 80)
    print("TRAUMA-BASED PARTICIPANT GROUPING ANALYSIS")
    print("=" * 80)
    print("\nThis script implements two grouping approaches:")
    print("1. Hypothesis-driven: Median split on LEC + IES-R")
    print("2. Unsupervised: Hierarchical clustering")
    print()

    # Load data
    df = load_participant_data()

    # Hypothesis-driven grouping
    df, lec_median, ies_median = create_hypothesis_groups(df)

    # Hierarchical clustering
    df, linkage_matrix, features_scaled, silhouette_scores, optimal_k = \
        perform_hierarchical_clustering(df)

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_hypothesis_groups(df, lec_median, ies_median)
    plot_dendrogram(linkage_matrix, df)
    plot_cluster_comparison(df, features_scaled, optimal_k)
    plot_silhouette_analysis(df, features_scaled, silhouette_scores)

    # Compare methods
    crosstab = compare_grouping_methods(df, optimal_k)

    # Save results
    save_results(df, silhouette_scores, optimal_k, lec_median, ies_median)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review visualizations in figures/trauma_groups/")
    print("  2. Run validation: python scripts/analysis/validate_trauma_groups.py")
    print("  3. Examine group assignments in output/trauma_groups/group_assignments.csv")
    print()

if __name__ == '__main__':
    main()
