#!/usr/bin/env python
"""
08: Run Statistical Analyses
============================

Performs statistical analyses and generates descriptive tables.

This MERGED script combines:
1. run_statistical_analyses.py - Mixed ANOVAs, regressions
2. generate_descriptive_tables.py - Formatted descriptive statistics tables

Analyses Performed:
    1. Descriptive Statistics:
       - Demographics by trauma group
       - Trauma scale scores by group
       - Task performance by group and load
       - Additional behavioral metrics

    2. Assumption Checks:
       - Shapiro-Wilk normality tests
       - Levene's test for homogeneity of variance

    3. Inferential Statistics:
       - Mixed ANOVAs (Load × Trauma Group)
       - Linear regressions with continuous trauma predictors

Inputs:
    - data/processed/summary_participant_metrics.csv
    - data/processed/task_trials_long.csv

Outputs:
    - reports/tables/descriptives/table1_demographics.csv
    - reports/tables/descriptives/table2_trauma_scores.csv
    - reports/tables/descriptives/table3_task_performance.csv
    - reports/tables/descriptives/table4_additional_metrics.csv
    - reports/tables/statistical_analyses/data_long_format.csv
    - reports/tables/statistical_analyses/anova_*.csv
    - reports/tables/statistical_analyses/regression_*.txt

Usage:
    # Run all analyses (tables + statistics)
    python scripts/02_behav_analyses/04_run_statistical_analyses.py

    # Generate only descriptive tables
    python scripts/02_behav_analyses/04_run_statistical_analyses.py --tables-only

    # Run only inferential statistics
    python scripts/02_behav_analyses/04_run_statistical_analyses.py --stats-only

Next Steps:
    - Review tables for thesis/publication
    - Examine ANOVA results for main effects and interactions
    - Check regression outputs for trauma-performance associations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

# Add project root to path (parents[2] = project root; parents[1] = scripts/)
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import from library modules
from config import (
    EXCLUDED_PARTICIPANTS,
    PROCESSED_DIR,
    REPORTS_TABLES_DESCRIPTIVES,
    REPORTS_TABLES_DIR,
)
from scripts.utils.stats import (
    check_normality,
    check_homogeneity_of_variance,
    run_mixed_anova,
    run_multiple_regressions,
    create_anova_summary_table
)

# ============================================================================
# PART 1: DESCRIPTIVE TABLES
# ============================================================================

def create_trauma_groups_simple(df):
    """Create trauma groups based on LESS endorsement and IES-R cutoff."""
    IES_CUTOFF = 24

    def assign_group(row):
        if pd.isna(row.get('lec_total_events')) or pd.isna(row.get('ies_total')):
            return 'Missing'
        elif row['lec_total_events'] == 0:
            return 'No Trauma'
        elif row['lec_total_events'] >= 1 and row['ies_total'] < IES_CUTOFF:
            return 'Trauma - No Ongoing Impact'
        else:
            return 'Trauma - Ongoing Impact'

    df['trauma_group'] = df.apply(assign_group, axis=1)
    return df


def calculate_demographics_by_group(summary_df):
    """Calculate demographic statistics by trauma group."""
    df_clean = create_trauma_groups_simple(summary_df.copy())
    df_clean = df_clean[
        (~df_clean['sona_id'].isin(EXCLUDED_PARTICIPANTS)) &
        (df_clean['trauma_group'] != 'Missing')
    ].copy()

    groups = ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']

    results = []

    # Overall N
    for group in groups:
        n = (df_clean['trauma_group'] == group).sum()
        results.append({
            'Variable': 'N',
            'Group': group,
            'Value': str(n)
        })

    # Continuous variables
    continuous_vars = ['age', 'screen_time']
    for var in continuous_vars:
        if var in df_clean.columns:
            for group in groups:
                group_data = df_clean[df_clean['trauma_group'] == group][var].dropna()
                if len(group_data) > 0:
                    mean_val = group_data.mean()
                    sd_val = group_data.std()
                    results.append({
                        'Variable': var.replace('_', ' ').title(),
                        'Group': group,
                        'Value': f"{mean_val:.2f} ± {sd_val:.2f}"
                    })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        pivot_df = results_df.pivot(index='Variable', columns='Group', values='Value')
        existing_groups = [g for g in groups if g in pivot_df.columns]
        return pivot_df[existing_groups]

    return pd.DataFrame()


def calculate_trauma_scores_by_group(summary_df):
    """Calculate trauma scale scores by group."""
    df_clean = create_trauma_groups_simple(summary_df.copy())
    df_clean = df_clean[
        (~df_clean['sona_id'].isin(EXCLUDED_PARTICIPANTS)) &
        (df_clean['trauma_group'] != 'Missing')
    ].copy()

    groups = ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']

    trauma_vars = {
        'lec_total_events': 'LEC-5 Total Events',
        'ies_total': 'IES-R Total',
        'ies_intrusion': 'IES-R Intrusion',
        'ies_avoidance': 'IES-R Avoidance',
        'ies_hyperarousal': 'IES-R Hyperarousal'
    }

    results = []

    for var, label in trauma_vars.items():
        if var in df_clean.columns:
            for group in groups:
                group_data = df_clean[df_clean['trauma_group'] == group][var].dropna()
                if len(group_data) > 0:
                    mean_val = group_data.mean()
                    sd_val = group_data.std()
                    results.append({
                        'Variable': label,
                        'Group': group,
                        'Value': f"{mean_val:.2f} ± {sd_val:.2f}"
                    })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        pivot_df = results_df.pivot(index='Variable', columns='Group', values='Value')
        existing_groups = [g for g in groups if g in pivot_df.columns]
        return pivot_df[existing_groups]

    return pd.DataFrame()


def calculate_task_performance_by_group_and_load(summary_df, trials_df):
    """Calculate task performance metrics by group and load."""
    # Filter to experimental blocks
    exp_trials = trials_df[trials_df['block'] >= 3].copy()

    df_grouped = create_trauma_groups_simple(summary_df.copy())
    df_clean = df_grouped[
        (~df_grouped['sona_id'].isin(EXCLUDED_PARTICIPANTS)) &
        (df_grouped['trauma_group'] != 'Missing')
    ].copy()

    # Merge trauma group info with trials
    exp_trials = exp_trials.merge(
        df_clean[['sona_id', 'trauma_group']],
        on='sona_id',
        how='inner'
    )

    groups = ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']
    loads = ['low', 'high']

    results = []

    for load in loads:
        if 'load_condition' not in exp_trials.columns:
            break

        load_trials = exp_trials[exp_trials['load_condition'] == load]

        for group in groups:
            group_trials = load_trials[load_trials['trauma_group'] == group]

            # Accuracy
            if 'correct' in group_trials.columns:
                by_participant = group_trials.groupby('sona_id')['correct'].mean()
                if len(by_participant) > 0:
                    mean_val = by_participant.mean() * 100
                    sd_val = by_participant.std() * 100
                    results.append({
                        'Metric': 'Accuracy (%)',
                        'Load': 'Low Load' if load == 'low' else 'High Load',
                        'Group': group,
                        'Value': f"{mean_val:.2f} ± {sd_val:.2f}"
                    })

            # RT
            if 'rt' in group_trials.columns:
                by_participant = group_trials.groupby('sona_id')['rt'].median()
                if len(by_participant) > 0:
                    mean_val = by_participant.mean()
                    sd_val = by_participant.std()
                    results.append({
                        'Metric': 'Median RT (ms)',
                        'Load': 'Low Load' if load == 'low' else 'High Load',
                        'Group': group,
                        'Value': f"{mean_val:.0f} ± {sd_val:.0f}"
                    })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        pivot_df = results_df.pivot_table(
            index=['Metric', 'Load'],
            columns='Group',
            values='Value',
            aggfunc='first'
        )
        existing_groups = [g for g in groups if g in pivot_df.columns]
        return pivot_df[existing_groups]

    return pd.DataFrame()


def generate_descriptive_tables(output_dir):
    """Generate all descriptive statistics tables."""
    print("\n" + "=" * 80)
    print("GENERATING DESCRIPTIVE STATISTICS TABLES")
    print("=" * 80)

    # Load data
    summary_path = PROCESSED_DIR / 'summary_participant_metrics.csv'
    trials_path = PROCESSED_DIR / 'task_trials_long.csv'

    if not summary_path.exists():
        print(f"\nERROR: Summary data not found at {summary_path}")
        return

    summary_df = pd.read_csv(summary_path)
    if 'included_in_analysis' in summary_df.columns:
        summary_df = summary_df[summary_df['included_in_analysis'] == True].copy()
    # Normalize column names (LESS → LEC naming convention used downstream)
    rename_map = {
        'less_total_events': 'lec_total_events',
        'less_personal_events': 'lec_personal_events',
    }
    summary_df = summary_df.rename(
        columns={k: v for k, v in rename_map.items() if k in summary_df.columns}
    )
    trials_df = pd.read_csv(trials_path) if trials_path.exists() else None

    print(f"\nLoaded {len(summary_df)} participants")
    if trials_df is not None:
        print(f"Loaded {len(trials_df)} trials")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Table 1: Demographics
    print("\n" + "-" * 80)
    print("Table 1: Demographics by Trauma Group")
    print("-" * 80)

    demographics_table = calculate_demographics_by_group(summary_df)
    if len(demographics_table) > 0:
        print(demographics_table)
        save_path = output_path / 'table1_demographics.csv'
        demographics_table.to_csv(save_path)
        print(f"\n[SAVED] {save_path}")

    # Table 2: Trauma Scores
    print("\n" + "-" * 80)
    print("Table 2: Trauma Scores by Group")
    print("-" * 80)

    trauma_table = calculate_trauma_scores_by_group(summary_df)
    if len(trauma_table) > 0:
        print(trauma_table)
        save_path = output_path / 'table2_trauma_scores.csv'
        trauma_table.to_csv(save_path)
        print(f"\n[SAVED] {save_path}")

    # Table 3: Task Performance
    if trials_df is not None:
        print("\n" + "-" * 80)
        print("Table 3: Task Performance by Group and Load")
        print("-" * 80)

        performance_table = calculate_task_performance_by_group_and_load(summary_df, trials_df)
        if len(performance_table) > 0:
            print(performance_table)
            save_path = output_path / 'table3_task_performance.csv'
            performance_table.to_csv(save_path)
            print(f"\n[SAVED] {save_path}")


# ============================================================================
# PART 2: INFERENTIAL STATISTICS
# ============================================================================

def prepare_long_format_data(summary_df, trials_df):
    """Prepare data in long format for mixed ANOVA."""
    df_grouped = create_trauma_groups_simple(summary_df.copy())

    df_clean = df_grouped[
        (~df_grouped['sona_id'].isin(EXCLUDED_PARTICIPANTS)) &
        (df_grouped['trauma_group'] != 'Missing')
    ].copy()

    print(f"\nParticipant filtering:")
    print(f"  Total in summary: {len(summary_df)}")
    print(f"  After trauma grouping: {len(df_grouped)}")
    print(f"  Final sample: {len(df_clean)}")

    # Filter to experimental blocks
    exp_trials = trials_df[trials_df['block'] >= 3].copy()

    if 'load_condition' not in exp_trials.columns:
        print("\nWARNING: load_condition column not found. Skipping long format preparation.")
        return None, df_clean

    # Calculate load-specific accuracy for each participant
    load_accuracy = []
    for sona_id in exp_trials['sona_id'].unique():
        p_trials = exp_trials[exp_trials['sona_id'] == sona_id]

        low_load_trials = p_trials[p_trials['load_condition'] == 'low']
        high_load_trials = p_trials[p_trials['load_condition'] == 'high']

        load_accuracy.append({
            'sona_id': sona_id,
            'accuracy_low': low_load_trials['correct'].mean() if len(low_load_trials) > 0 else np.nan,
            'accuracy_high': high_load_trials['correct'].mean() if len(high_load_trials) > 0 else np.nan,
            'rt_low': low_load_trials['rt'].median() if len(low_load_trials) > 0 and 'rt' in low_load_trials.columns else np.nan,
            'rt_high': high_load_trials['rt'].median() if len(high_load_trials) > 0 and 'rt' in high_load_trials.columns else np.nan
        })

    load_df = pd.DataFrame(load_accuracy)

    # Merge with trauma groups
    df_merged = df_clean.merge(load_df, on='sona_id', how='inner')

    # Create long format
    long_data = []
    for _, row in df_merged.iterrows():
        # Low load
        long_data.append({
            'sona_id': row['sona_id'],
            'trauma_group': row['trauma_group'],
            'load': 'Low',
            'accuracy': row['accuracy_low'] * 100 if pd.notna(row['accuracy_low']) else np.nan,
            'rt': row['rt_low']
        })

        # High load
        long_data.append({
            'sona_id': row['sona_id'],
            'trauma_group': row['trauma_group'],
            'load': 'High',
            'accuracy': row['accuracy_high'] * 100 if pd.notna(row['accuracy_high']) else np.nan,
            'rt': row['rt_high']
        })

    long_df = pd.DataFrame(long_data)

    return long_df, df_merged


def run_anova_analyses(long_df, output_dir):
    """Run mixed ANOVAs for DVs."""
    print("\n" + "=" * 80)
    print("MIXED ANOVAs: Load × Trauma Group")
    print("=" * 80)

    anova_dvs = ['accuracy', 'rt']
    output_path = Path(output_dir)

    for dv in anova_dvs:
        if dv not in long_df.columns:
            continue

        print(f"\n{'-' * 80}")
        print(f"DV: {dv}")
        print(f"{'-' * 80}")

        try:
            # Check assumptions
            print("\nAssumption Checks:")
            normality_results = check_normality(long_df, group_col='trauma_group', dv_col=dv)
            print(normality_results)

            variance_results = check_homogeneity_of_variance(long_df, dv, 'trauma_group')
            print(f"\nLevene's test: F={variance_results['F']:.3f}, p={variance_results['p']:.4f}")

            # Run ANOVA
            aov_results = run_mixed_anova(
                data=long_df,
                dv=dv,
                within_factor='load',
                between_factor='trauma_group',
                subject_id='sona_id'
            )

            summary_table = create_anova_summary_table(aov_results)
            print("\nANOVA Results:")
            print(summary_table)

            # Save
            aov_results.to_csv(output_path / f'anova_{dv}_full.csv', index=False)
            summary_table.to_csv(output_path / f'anova_{dv}_summary.csv')

        except Exception as e:
            print(f"Error running ANOVA for {dv}: {e}")


def run_statistical_analyses(output_dir):
    """Run all statistical analyses."""
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSES")
    print("=" * 80)

    # Load data
    summary_path = PROCESSED_DIR / 'summary_participant_metrics.csv'
    trials_path = PROCESSED_DIR / 'task_trials_long.csv'

    if not summary_path.exists():
        print(f"\nERROR: Summary data not found at {summary_path}")
        return

    summary_df = pd.read_csv(summary_path)
    if 'included_in_analysis' in summary_df.columns:
        summary_df = summary_df[summary_df['included_in_analysis'] == True].copy()
    # Normalize column names (LESS → LEC naming convention used downstream)
    rename_map = {
        'less_total_events': 'lec_total_events',
        'less_personal_events': 'lec_personal_events',
    }
    summary_df = summary_df.rename(
        columns={k: v for k, v in rename_map.items() if k in summary_df.columns}
    )
    trials_df = pd.read_csv(trials_path) if trials_path.exists() else None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if trials_df is None:
        print("\nWARNING: No trials data found. Skipping ANOVA analyses.")
        return

    # Prepare long-format data
    long_df, summary_with_groups = prepare_long_format_data(summary_df, trials_df)

    if long_df is not None:
        long_df.to_csv(output_path / 'data_long_format.csv', index=False)
        run_anova_analyses(long_df, output_dir)

    summary_with_groups.to_csv(output_path / 'data_summary_with_groups.csv', index=False)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run statistical analyses and generate descriptive tables'
    )
    parser.add_argument('--tables-only', action='store_true',
                        help='Generate only descriptive tables')
    parser.add_argument('--stats-only', action='store_true',
                        help='Run only inferential statistics')
    parser.add_argument('--output-descriptives', type=str,
                        default=str(REPORTS_TABLES_DESCRIPTIVES),
                        help='Output directory for descriptive tables')
    parser.add_argument('--output-stats', type=str,
                        default=str(REPORTS_TABLES_DIR / 'statistical_analyses'),
                        help='Output directory for statistical analyses')

    args = parser.parse_args()

    print("=" * 80)
    print("STATISTICAL ANALYSES AND DESCRIPTIVE TABLES")
    print("=" * 80)

    if not args.stats_only:
        generate_descriptive_tables(args.output_descriptives)

    if not args.tables_only:
        run_statistical_analyses(args.output_stats)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nDescriptive tables saved to: {args.output_descriptives}/")
    print(f"Statistical results saved to: {args.output_stats}/")
    print("\nNext steps:")
    print("  - Review assumption check results")
    print("  - Examine ANOVA tables for main effects and interactions")
    print("  - Use tables for thesis/publication")


if __name__ == '__main__':
    main()
