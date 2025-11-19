"""
Validate Trauma-Based Groups Against Behavioral Outcomes

Tests whether hypothesis-driven and clustering-based groups show
significant differences in task performance metrics.

Usage:
    python scripts/analysis/validate_trauma_groups.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# Paths
OUTPUT_DIR = Path('output/trauma_groups')
FIGURES_DIR = Path('figures/trauma_groups')

def load_group_assignments():
    """Load group assignments from trauma grouping analysis."""
    assignments_path = OUTPUT_DIR / 'group_assignments.csv'

    if not assignments_path.exists():
        raise FileNotFoundError(
            f"Group assignments not found at {assignments_path}. "
            "Run trauma_grouping_analysis.py first."
        )

    df = pd.read_csv(assignments_path)

    # Load full participant data to get behavioral outcomes
    full_data_path = Path('output/summary_participant_metrics.csv')
    df_full = pd.read_csv(full_data_path)

    # Merge
    df_merged = df.merge(df_full, on='sona_id', suffixes=('', '_full'))

    print(f"Loaded group assignments for {len(df_merged)} participants")

    return df_merged

def test_hypothesis_groups(df):
    """Test behavioral differences across hypothesis-driven groups."""
    print("\n" + "=" * 80)
    print("HYPOTHESIS-DRIVEN GROUPS: BEHAVIORAL VALIDATION")
    print("=" * 80)

    # Filter to main groups (exclude Low-High if present)
    main_groups = ['A_Low_Low', 'B_High_Low', 'C_High_High']
    df_test = df[df['hypothesis_group'].isin(main_groups)].copy()

    if len(df_test) < 3:
        print("\nINSUFFICIENT DATA: Need at least 3 participants for analysis")
        return None

    # Behavioral outcomes to test
    outcomes = {
        'accuracy_overall': 'Overall Accuracy',
        'mean_rt_overall': 'Mean RT (ms)'
    }

    results = []

    for outcome, label in outcomes.items():
        if outcome not in df_test.columns:
            print(f"\nWARNING: {outcome} not found in data, skipping...")
            continue

        # Remove missing values
        df_outcome = df_test.dropna(subset=[outcome])

        if len(df_outcome) < 3:
            continue

        print(f"\n{'-' * 80}")
        print(f"Outcome: {label}")
        print(f"{'-' * 80}")

        # Descriptive statistics by group
        print("\nDescriptive Statistics:")
        for group in main_groups:
            if group in df_outcome['hypothesis_group'].values:
                group_data = df_outcome[df_outcome['hypothesis_group'] == group][outcome]
                mean_val = group_data.mean()
                std_val = group_data.std()
                n_val = len(group_data)
                print(f"  {group}: M={mean_val:.3f}, SD={std_val:.3f}, n={n_val}")

        # Test for normality (Shapiro-Wilk)
        print("\nNormality Tests (Shapiro-Wilk):")
        all_normal = True
        for group in main_groups:
            if group in df_outcome['hypothesis_group'].values:
                group_data = df_outcome[df_outcome['hypothesis_group'] == group][outcome]
                if len(group_data) >= 3:
                    stat, p = stats.shapiro(group_data)
                    normal = "Yes" if p > 0.05 else "No"
                    all_normal = all_normal and (p > 0.05)
                    print(f"  {group}: W={stat:.3f}, p={p:.4f} ({normal})")

        # Choose appropriate test
        groups_data = [
            df_outcome[df_outcome['hypothesis_group'] == g][outcome].values
            for g in main_groups
            if g in df_outcome['hypothesis_group'].values
        ]

        if len(groups_data) < 2:
            print("\nINSUFFICIENT GROUPS for comparison")
            continue

        # ANOVA (parametric) or Kruskal-Wallis (non-parametric)
        if all_normal and all(len(g) >= 3 for g in groups_data):
            # One-way ANOVA
            f_stat, p_value = stats.f_oneway(*groups_data)
            test_name = "One-way ANOVA"
            stat_name = "F"
            stat_value = f_stat

            # Calculate effect size (eta-squared)
            # SS_between / SS_total
            grand_mean = df_outcome[outcome].mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups_data)
            ss_total = sum((df_outcome[outcome] - grand_mean)**2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0

            effect_size_name = "η²"
            effect_size = eta_squared

        else:
            # Kruskal-Wallis (non-parametric)
            h_stat, p_value = stats.kruskal(*groups_data)
            test_name = "Kruskal-Wallis H"
            stat_name = "H"
            stat_value = h_stat

            # Effect size: epsilon-squared
            n = len(df_outcome)
            k = len(groups_data)
            epsilon_squared = (h_stat - k + 1) / (n - k) if (n - k) > 0 else 0

            effect_size_name = "ε²"
            effect_size = epsilon_squared

        print(f"\n{test_name}:")
        print(f"  {stat_name}={stat_value:.3f}, p={p_value:.4f}")
        print(f"  Effect size ({effect_size_name})={effect_size:.3f}")

        if p_value < 0.05:
            print(f"  >>> SIGNIFICANT group difference (p < 0.05)")

            # Post-hoc pairwise comparisons
            print("\n  Post-hoc Pairwise Comparisons (Mann-Whitney U with Bonferroni):")
            n_comparisons = len(groups_data) * (len(groups_data) - 1) // 2
            alpha_corrected = 0.05 / n_comparisons

            group_names = [g for g in main_groups if g in df_outcome['hypothesis_group'].values]
            for i in range(len(group_names)):
                for j in range(i+1, len(group_names)):
                    g1_data = df_outcome[df_outcome['hypothesis_group'] == group_names[i]][outcome]
                    g2_data = df_outcome[df_outcome['hypothesis_group'] == group_names[j]][outcome]

                    u_stat, p_pair = stats.mannwhitneyu(g1_data, g2_data, alternative='two-sided')

                    # Effect size: rank-biserial correlation
                    n1, n2 = len(g1_data), len(g2_data)
                    r_rb = 1 - (2*u_stat) / (n1 * n2)

                    sig_marker = "***" if p_pair < alpha_corrected else ""
                    print(f"    {group_names[i]} vs {group_names[j]}: "
                          f"U={u_stat:.1f}, p={p_pair:.4f}, r={r_rb:.3f} {sig_marker}")

        else:
            print(f"  >>> No significant group difference (p ≥ 0.05)")

        # Store results
        results.append({
            'outcome': label,
            'test': test_name,
            'statistic': stat_value,
            'p_value': p_value,
            'effect_size': effect_size,
            'effect_size_type': effect_size_name,
            'significant': p_value < 0.05
        })

    return pd.DataFrame(results)

def test_cluster_groups(df, optimal_k=3):
    """Test behavioral differences across clustering-based groups."""
    print("\n" + "=" * 80)
    print(f"HIERARCHICAL CLUSTERING (k={optimal_k}): BEHAVIORAL VALIDATION")
    print("=" * 80)

    cluster_col = f'cluster_k{optimal_k}'

    if cluster_col not in df.columns:
        print(f"\nERROR: {cluster_col} not found in data")
        return None

    # Behavioral outcomes
    outcomes = {
        'accuracy_overall': 'Overall Accuracy',
        'mean_rt_overall': 'Mean RT (ms)'
    }

    results = []

    for outcome, label in outcomes.items():
        if outcome not in df.columns:
            continue

        df_outcome = df.dropna(subset=[outcome, cluster_col])

        if len(df_outcome) < 3:
            continue

        print(f"\n{'-' * 80}")
        print(f"Outcome: {label}")
        print(f"{'-' * 80}")

        # Descriptive statistics
        print("\nDescriptive Statistics by Cluster:")
        for cluster in sorted(df_outcome[cluster_col].unique()):
            cluster_data = df_outcome[df_outcome[cluster_col] == cluster][outcome]
            mean_val = cluster_data.mean()
            std_val = cluster_data.std()
            n_val = len(cluster_data)
            print(f"  Cluster {cluster}: M={mean_val:.3f}, SD={std_val:.3f}, n={n_val}")

        # Kruskal-Wallis test (non-parametric, safer for small N)
        groups_data = [
            df_outcome[df_outcome[cluster_col] == c][outcome].values
            for c in sorted(df_outcome[cluster_col].unique())
        ]

        if len(groups_data) >= 2:
            h_stat, p_value = stats.kruskal(*groups_data)

            print(f"\nKruskal-Wallis H Test:")
            print(f"  H={h_stat:.3f}, p={p_value:.4f}")

            if p_value < 0.05:
                print(f"  >>> SIGNIFICANT cluster difference (p < 0.05)")
            else:
                print(f"  >>> No significant cluster difference (p ≥ 0.05)")

            results.append({
                'outcome': label,
                'test': 'Kruskal-Wallis H',
                'statistic': h_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })

    return pd.DataFrame(results)

def plot_behavioral_by_group(df):
    """Create visualizations of behavioral outcomes by group."""
    print("\n" + "=" * 80)
    print("GENERATING BEHAVIORAL COMPARISON PLOTS")
    print("=" * 80)

    # Filter to main hypothesis groups
    main_groups = ['A_Low_Low', 'B_High_Low', 'C_High_High']
    df_plot = df[df['hypothesis_group'].isin(main_groups)].copy()

    if len(df_plot) < 3:
        print("\nINSUFFICIENT DATA for plotting")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    outcomes = [
        ('accuracy_overall', 'Overall Accuracy', axes[0]),
        ('mean_rt_overall', 'Mean RT (ms)', axes[1])
    ]

    group_colors = {
        'A_Low_Low': '#2ecc71',
        'B_High_Low': '#f39c12',
        'C_High_High': '#e74c3c'
    }

    for outcome, label, ax in outcomes:
        if outcome not in df_plot.columns:
            continue

        # Box plot with individual points
        df_outcome = df_plot.dropna(subset=[outcome])

        # Violin plot
        parts = ax.violinplot(
            [df_outcome[df_outcome['hypothesis_group'] == g][outcome].values
             for g in main_groups if g in df_outcome['hypothesis_group'].values],
            positions=range(len([g for g in main_groups if g in df_outcome['hypothesis_group'].values])),
            showmeans=True,
            showmedians=True
        )

        # Color violin plots
        for i, pc in enumerate(parts['bodies']):
            group = [g for g in main_groups if g in df_outcome['hypothesis_group'].values][i]
            pc.set_facecolor(group_colors[group])
            pc.set_alpha(0.3)

        # Overlay scatter plot
        for i, group in enumerate([g for g in main_groups if g in df_outcome['hypothesis_group'].values]):
            group_data = df_outcome[df_outcome['hypothesis_group'] == group]
            x_vals = np.random.normal(i, 0.04, len(group_data))
            ax.scatter(x_vals, group_data[outcome],
                      color=group_colors[group], s=80, alpha=0.7,
                      edgecolors='black', linewidth=1, zorder=3)

        # Formatting
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Trauma Group', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len([g for g in main_groups if g in df_outcome['hypothesis_group'].values])))
        ax.set_xticklabels([g.replace('_', '\n') for g in main_groups if g in df_outcome['hypothesis_group'].values],
                          fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Behavioral Performance by Trauma Group',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    output_path = FIGURES_DIR / 'behavioral_by_group.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[SAVED] {output_path}")

    plt.close()

def save_validation_report(hyp_results, cluster_results):
    """Save validation results to text report."""
    report_path = OUTPUT_DIR / 'validation_report.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAUMA GROUP VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("HYPOTHESIS-DRIVEN GROUPS (Median Split)\n")
        f.write("-" * 80 + "\n")
        if hyp_results is not None and len(hyp_results) > 0:
            for _, row in hyp_results.iterrows():
                f.write(f"\nOutcome: {row['outcome']}\n")
                f.write(f"  Test: {row['test']}\n")
                f.write(f"  Statistic: {row['statistic']:.3f}\n")
                f.write(f"  p-value: {row['p_value']:.4f}\n")
                f.write(f"  Effect size ({row['effect_size_type']}): {row['effect_size']:.3f}\n")
                f.write(f"  Significant: {'Yes' if row['significant'] else 'No'}\n")
        else:
            f.write("\nNo results available\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("HIERARCHICAL CLUSTERING GROUPS\n")
        f.write("-" * 80 + "\n")
        if cluster_results is not None and len(cluster_results) > 0:
            for _, row in cluster_results.iterrows():
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
        f.write("\nWith N=17, statistical power is limited. Non-significant results\n")
        f.write("do not rule out true group differences - they may reflect\n")
        f.write("insufficient sample size rather than absence of effects.\n\n")
        f.write("Effect sizes provide magnitude estimates independent of p-values:\n")
        f.write("  - Small: η²=0.01, ε²=0.01\n")
        f.write("  - Medium: η²=0.06, ε²=0.06\n")
        f.write("  - Large: η²=0.14, ε²=0.14\n\n")
        f.write("This analysis should be considered exploratory/pilot.\n")
        f.write("Larger samples (N>60) needed for confirmatory analysis.\n")

    print(f"\n[SAVED] {report_path}")

def main():
    """Main execution function."""
    print("=" * 80)
    print("TRAUMA GROUP VALIDATION: BEHAVIORAL OUTCOMES")
    print("=" * 80)
    print("\nTesting whether trauma-based groups differ on task performance...")
    print()

    # Load data
    df = load_group_assignments()

    # Test hypothesis groups
    hyp_results = test_hypothesis_groups(df)

    # Test clustering groups
    cluster_results = test_cluster_groups(df, optimal_k=3)

    # Generate plots
    plot_behavioral_by_group(df)

    # Save report
    save_validation_report(hyp_results, cluster_results)

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print("\nOutputs:")
    print("  - figures/trauma_groups/behavioral_by_group.png")
    print("  - output/trauma_groups/validation_report.txt")
    print("\nNext steps:")
    print("  - Review validation_report.txt for statistical results")
    print("  - Examine effect sizes (more informative than p-values with small N)")
    print("  - Consider this exploratory - plan larger sample for confirmation")
    print()

if __name__ == '__main__':
    main()
