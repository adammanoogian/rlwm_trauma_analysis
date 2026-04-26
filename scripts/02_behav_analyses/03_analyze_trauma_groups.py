#!/usr/bin/env python
"""
07: Analyze Trauma Groups
=========================

Implements trauma-based participant grouping and validation.

This MERGED script combines:
1. trauma_grouping_analysis.py - Creates 3-group classification + hierarchical clustering
2. validate_trauma_groups.py - Validates groups against behavioral outcomes

Grouping Approaches:
    1. Hypothesis-driven: LESS endorsement (≥1) + IES-R cutoff (24)
       - No Trauma: LESS = 0
       - Trauma - No Ongoing Impact: LESS ≥ 1 AND IES-R < 24
       - Trauma - Ongoing Impact: LESS ≥ 1 AND IES-R ≥ 24

    2. Unsupervised: Hierarchical clustering (Ward linkage)
       - Tests k=2, 3, 4 solutions
       - Identifies optimal k via silhouette analysis

Inputs:
    - data/processed/summary_participant_metrics.csv

Outputs:
    - reports/tables/trauma_groups/group_assignments.csv
    - reports/tables/trauma_groups/group_summary_stats.csv
    - reports/tables/trauma_groups/clustering_metrics.csv
    - reports/tables/trauma_groups/cutoff_values.csv
    - reports/tables/trauma_groups/validation_report.txt
    - figures/trauma_groups/hypothesis_groups_scatter.png
    - figures/trauma_groups/hierarchical_dendrogram.png
    - figures/trauma_groups/cluster_comparison.png
    - figures/trauma_groups/cluster_silhouette.png
    - figures/trauma_groups/group_comparison_heatmap.png
    - figures/trauma_groups/behavioral_by_group.png

Usage:
    # Run full analysis (grouping + validation)
    python scripts/02_behav_analyses/03_analyze_trauma_groups.py

    # Run only grouping (skip validation)
    python scripts/02_behav_analyses/03_analyze_trauma_groups.py --grouping-only

    # Run only validation (requires existing group assignments)
    python scripts/02_behav_analyses/03_analyze_trauma_groups.py --validation-only

Next Steps:
    - Run 04_run_statistical_analyses.py for ANOVAs and regressions
    - Run 04_analyze_mle_by_trauma.py (in 06_fit_analyses/) after model fitting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import chi2_contingency, stats
import argparse
import warnings
warnings.filterwarnings('ignore')

import sys
# parents[2] = project root; parents[1] = scripts/
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from config import (
    PROCESSED_DIR,
    REPORTS_FIGURES_DIR,
    REPORTS_TABLES_TRAUMA_GROUPS,
)

# Paths
OUTPUT_DIR = REPORTS_TABLES_TRAUMA_GROUPS
FIGURES_DIR = REPORTS_FIGURES_DIR / 'trauma_groups'

# ============================================================================
# PART 1: TRAUMA GROUPING ANALYSIS
# ============================================================================

def load_participant_data():
    """Load participant summary data with trauma scores."""
    data_path = PROCESSED_DIR / 'summary_participant_metrics.csv'

    if not data_path.exists():
        raise FileNotFoundError(
            f"Participant data not found at {data_path}. "
            "Run the data pipeline first: python run_data_pipeline.py"
        )

    df = pd.read_csv(data_path)

    # Filter to analysis cohort (excluded participants carry no valid task metrics)
    if 'included_in_analysis' in df.columns:
        df = df[df['included_in_analysis'] == True].copy()

    # Normalize column names (LESS → LEC naming convention used downstream)
    rename_map = {
        'less_total_events': 'lec_total_events',
        'less_personal_events': 'lec_personal_events',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

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
    Create trauma groups based on LESS endorsement and IES-R cutoff.

    GROUPING CRITERIA:
    - No Trauma Exposure: LESS = 0 (no endorsed events)
    - Trauma Exposure - No Ongoing Impact: LESS ≥ 1 AND IES-R < 24
    - Trauma Exposure - Ongoing Impact: LESS ≥ 1 AND IES-R ≥ 24
    """
    IES_CUTOFF = 24

    print("\n" + "=" * 80)
    print("HYPOTHESIS-DRIVEN GROUPING (LESS ENDORSEMENT + IES-R CUTOFF)")
    print("=" * 80)
    print(f"\nCriteria:")
    print(f"  - No Trauma Exposure: LESS = 0")
    print(f"  - Trauma - No Ongoing Impact: LESS ≥ 1 AND IES-R < {IES_CUTOFF}")
    print(f"  - Trauma - Ongoing Impact: LESS ≥ 1 AND IES-R ≥ {IES_CUTOFF}")

    def assign_group(row):
        if row['lec_total_events'] == 0:
            return 'No Trauma'
        elif row['lec_total_events'] >= 1 and row['ies_total'] < IES_CUTOFF:
            return 'Trauma - No Ongoing Impact'
        else:
            return 'Trauma - Ongoing Impact'

    df['hypothesis_group'] = df.apply(assign_group, axis=1)

    # Print group sizes
    print("\nGroup Sizes:")
    group_counts = df['hypothesis_group'].value_counts().sort_index()
    for group, count in group_counts.items():
        print(f"  {group}: n={count}")

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

    return df


def perform_hierarchical_clustering(df, k_range=[2, 3, 4]):
    """Perform hierarchical clustering with Ward linkage."""
    print("\n" + "=" * 80)
    print("HIERARCHICAL CLUSTERING ANALYSIS")
    print("=" * 80)

    features = df[['lec_total_events', 'ies_total']].values

    print(f"\nFeatures: LEC Total Events, IES-R Total Score")
    print(f"N participants: {len(features)}")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    print("\nStandardizing features (z-scores)...")
    print(f"  LEC mean={scaler.mean_[0]:.2f}, std={scaler.scale_[0]:.2f}")
    print(f"  IES-R mean={scaler.mean_[1]:.2f}, std={scaler.scale_[1]:.2f}")

    print("\nComputing hierarchical clustering with Ward linkage...")
    linkage_matrix = linkage(features_scaled, method='ward')

    print("\n" + "-" * 80)
    print("Testing cluster solutions:")
    print("-" * 80)

    silhouette_scores = {}
    cluster_assignments = {}

    for k in k_range:
        clusters = fcluster(linkage_matrix, t=k, criterion='maxclust')
        cluster_assignments[k] = clusters

        if k > 1 and k < len(features):
            sil_score = silhouette_score(features_scaled, clusters)
            silhouette_scores[k] = sil_score

            unique, counts = np.unique(clusters, return_counts=True)
            cluster_sizes = dict(zip(unique, counts))

            print(f"\nk={k} clusters:")
            print(f"  Silhouette score: {sil_score:.3f}")
            print(f"  Cluster sizes: {cluster_sizes}")

    if silhouette_scores:
        optimal_k = max(silhouette_scores, key=silhouette_scores.get)
        print(f"\n>>> Optimal k={optimal_k} (highest silhouette: {silhouette_scores[optimal_k]:.3f})")
    else:
        optimal_k = 3

    for k in k_range:
        df[f'cluster_k{k}'] = cluster_assignments[k]

    return df, linkage_matrix, features_scaled, silhouette_scores, optimal_k


def plot_hypothesis_groups(df, ies_median=24):
    """Visualize the 3-group hypothesis-driven classification."""
    fig, ax = plt.subplots(figsize=(10, 8))

    group_colors = {
        'No Trauma': '#2ecc71',
        'Trauma - No Ongoing Impact': '#f39c12',
        'Trauma - Ongoing Impact': '#e74c3c',
        'Excluded_Low_High': '#95a5a6'
    }

    for group in df['hypothesis_group'].unique():
        group_data = df[df['hypothesis_group'] == group]
        ax.scatter(
            group_data['lec_total_events'],
            group_data['ies_total'],
            c=group_colors.get(group, 'gray'),
            label=group,
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=1
        )

    if ies_median is not None:
        ax.axhline(ies_median, color='red', linestyle='--', alpha=0.7, linewidth=2,
                   label=f'IES-R Cutoff = {ies_median}')

    ax.set_xlabel('LEC-5 Total Events (LESS)', fontsize=12, fontweight='bold')
    ax.set_ylabel('IES-R Total Score', fontsize=12, fontweight='bold')
    ax.set_title('Hypothesis-Driven Trauma Grouping\n(LESS Endorsement + IES-R ≥ 24)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = FIGURES_DIR / 'hypothesis_groups_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[SAVED] {output_path}")
    plt.close()


def plot_dendrogram(linkage_matrix, df):
    """Create dendrogram visualization."""
    fig, ax = plt.subplots(figsize=(14, 8))

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

    ax.axhline(y=ax.get_ylim()[1]*0.6, color='red', linestyle='--',
               alpha=0.5, linewidth=1, label='Potential cut for k=2')
    ax.axhline(y=ax.get_ylim()[1]*0.4, color='orange', linestyle='--',
               alpha=0.5, linewidth=1, label='Potential cut for k=3')

    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    output_path = FIGURES_DIR / 'hierarchical_dendrogram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {output_path}")
    plt.close()


def save_grouping_results(df, silhouette_scores, optimal_k):
    """Save grouping results to CSV files."""
    print("\n" + "=" * 80)
    print("SAVING GROUPING RESULTS")
    print("=" * 80)

    # Save group assignments
    output_cols = ['sona_id', 'lec_total_events', 'ies_total',
                   'hypothesis_group', 'cluster_k2', 'cluster_k3', 'cluster_k4']
    output_cols = [col for col in output_cols if col in df.columns]

    assignments_path = OUTPUT_DIR / 'group_assignments.csv'
    df[output_cols].to_csv(assignments_path, index=False)
    print(f"\n[SAVED] {assignments_path}")

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


# ============================================================================
# PART 2: VALIDATION
# ============================================================================

def test_hypothesis_groups(df):
    """Test behavioral differences across hypothesis-driven groups."""
    print("\n" + "=" * 80)
    print("HYPOTHESIS-DRIVEN GROUPS: BEHAVIORAL VALIDATION")
    print("=" * 80)

    main_groups = ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']
    df_test = df[df['hypothesis_group'].isin(main_groups)].copy()

    if len(df_test) < 3:
        print("\nINSUFFICIENT DATA: Need at least 3 participants for analysis")
        return None

    outcomes = {
        'accuracy_overall': 'Overall Accuracy',
        'mean_rt_overall': 'Mean RT (ms)'
    }

    results = []

    for outcome, label in outcomes.items():
        if outcome not in df_test.columns:
            continue

        df_outcome = df_test.dropna(subset=[outcome])

        if len(df_outcome) < 3:
            continue

        print(f"\n{'-' * 80}")
        print(f"Outcome: {label}")
        print(f"{'-' * 80}")

        print("\nDescriptive Statistics:")
        for group in main_groups:
            if group in df_outcome['hypothesis_group'].values:
                group_data = df_outcome[df_outcome['hypothesis_group'] == group][outcome]
                mean_val = group_data.mean()
                std_val = group_data.std()
                n_val = len(group_data)
                print(f"  {group}: M={mean_val:.3f}, SD={std_val:.3f}, n={n_val}")

        # Kruskal-Wallis test
        groups_data = [
            df_outcome[df_outcome['hypothesis_group'] == g][outcome].values
            for g in main_groups
            if g in df_outcome['hypothesis_group'].values and len(df_outcome[df_outcome['hypothesis_group'] == g]) > 0
        ]

        if len(groups_data) >= 2:
            from scipy.stats import kruskal
            h_stat, p_value = kruskal(*groups_data)

            print(f"\nKruskal-Wallis H Test:")
            print(f"  H={h_stat:.3f}, p={p_value:.4f}")

            if p_value < 0.05:
                print(f"  >>> SIGNIFICANT group difference (p < 0.05)")
            else:
                print(f"  >>> No significant group difference (p ≥ 0.05)")

            results.append({
                'outcome': label,
                'test': 'Kruskal-Wallis H',
                'statistic': h_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })

    return pd.DataFrame(results)


def save_validation_report(hyp_results):
    """Save validation results to text report."""
    report_path = OUTPUT_DIR / 'validation_report.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAUMA GROUP VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("HYPOTHESIS-DRIVEN GROUPS\n")
        f.write("-" * 80 + "\n")
        if hyp_results is not None and len(hyp_results) > 0:
            for _, row in hyp_results.iterrows():
                f.write(f"\nOutcome: {row['outcome']}\n")
                f.write(f"  Test: {row['test']}\n")
                f.write(f"  Statistic: {row['statistic']:.3f}\n")
                f.write(f"  p-value: {row['p_value']:.4f}\n")
                f.write(f"  Significant: {'Yes' if row['significant'] else 'No'}\n")
        else:
            f.write("\nNo results available\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("INTERPRETATION NOTES\n")
        f.write("-" * 80 + "\n")
        f.write("\nWith small N, statistical power is limited. Non-significant results\n")
        f.write("do not rule out true group differences.\n")

    print(f"\n[SAVED] {report_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze trauma groups (grouping + validation)'
    )
    parser.add_argument('--grouping-only', action='store_true',
                        help='Run only grouping analysis (skip validation)')
    parser.add_argument('--validation-only', action='store_true',
                        help='Run only validation (requires existing group assignments)')

    args = parser.parse_args()

    # Create directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TRAUMA GROUP ANALYSIS")
    print("=" * 80)
    print("\nThis script performs:")
    print("  1. Hypothesis-driven grouping (LESS + IES-R cutoff)")
    print("  2. Hierarchical clustering analysis")
    print("  3. Behavioral validation of groups")

    if args.validation_only:
        # Load existing assignments
        assignments_path = OUTPUT_DIR / 'group_assignments.csv'
        if not assignments_path.exists():
            print(f"\nERROR: No group assignments found. Run without --validation-only first.")
            return

        df = pd.read_csv(assignments_path)
        # Load full data for behavioral outcomes
        full_df = pd.read_csv(PROCESSED_DIR / 'summary_participant_metrics.csv')
        df = df.merge(full_df, on='sona_id', suffixes=('', '_full'))
    else:
        # Run grouping
        df = load_participant_data()
        df = create_hypothesis_groups(df)
        df, linkage_matrix, features_scaled, silhouette_scores, optimal_k = \
            perform_hierarchical_clustering(df)

        # Generate visualizations
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)

        plot_hypothesis_groups(df, ies_median=24)
        plot_dendrogram(linkage_matrix, df)

        # Save results
        save_grouping_results(df, silhouette_scores, optimal_k)

    if not args.grouping_only:
        # Run validation
        hyp_results = test_hypothesis_groups(df)
        save_validation_report(hyp_results)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nOutputs:")
    print(f"  - {OUTPUT_DIR}/")
    print(f"  - {FIGURES_DIR}/")
    print("\nNext steps:")
    print("  - Run 04_run_statistical_analyses.py for ANOVAs and regressions")
    print("  - Use group_assignments.csv for downstream analysis")


if __name__ == '__main__':
    main()
