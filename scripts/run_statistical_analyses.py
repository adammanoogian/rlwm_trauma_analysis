"""
Run statistical analyses for thesis.

Performs:
1. Assumption checks (normality, homogeneity of variance)
2. Mixed ANOVAs for each dependent variable
3. Linear regressions with continuous trauma predictors
4. Saves formatted output tables
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from config import EXCLUDED_PARTICIPANTS
from utils.statistical_tests import (
    check_normality,
    check_homogeneity_of_variance,
    run_mixed_anova,
    run_between_anova,
    run_welch_anova,
    post_hoc_tests,
    run_linear_regression,
    run_multiple_regressions,
    create_anova_summary_table,
    save_statistical_results
)
from analysis.visualize_load_by_trauma_group import create_trauma_groups


def prepare_long_format_data(summary_df, trials_df):
    """
    Prepare data in long format for mixed ANOVA.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary participant data
    trials_df : pd.DataFrame
        Trial-level data
        
    Returns
    -------
    tuple
        (long_format_df, summary_with_groups_df)
    """
    # Add trauma groups
    df_grouped, _, _ = create_trauma_groups(summary_df)
    
    # Exclude participants based on data quality and duplicates
    df_clean = df_grouped[
        (~df_grouped['sona_id'].isin(EXCLUDED_PARTICIPANTS)) &
        (df_grouped['trauma_group'] != 'Excluded')
    ].copy()
    
    print(f"\nParticipant filtering:")
    print(f"  Total in summary: {len(summary_df)}")
    print(f"  After trauma grouping: {len(df_grouped)}")
    print(f"  Excluded for data quality: {len([p for p in EXCLUDED_PARTICIPANTS if p in df_grouped['sona_id'].values])}")
    print(f"  Final sample: {len(df_clean)}")
    
    # Filter to experimental blocks
    exp_trials = trials_df[trials_df['block'] >= 3].copy()
    
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
            'rt_low': low_load_trials['rt'].median() if len(low_load_trials) > 0 else np.nan,
            'rt_high': high_load_trials['rt'].median() if len(high_load_trials) > 0 else np.nan
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
            'accuracy': row['accuracy_low'] * 100,  # Convert to percentage
            'rt': row['rt_low'],
            'less_total_events': row['less_total_events'],
            'ies_total': row['ies_total'],
            'ies_intrusion': row['ies_intrusion'],
            'ies_avoidance': row['ies_avoidance'],
            'ies_hyperarousal': row['ies_hyperarousal']
        })
        
        # High load
        long_data.append({
            'sona_id': row['sona_id'],
            'trauma_group': row['trauma_group'],
            'load': 'High',
            'accuracy': row['accuracy_high'] * 100,
            'rt': row['rt_high'],
            'less_total_events': row['less_total_events'],
            'ies_total': row['ies_total'],
            'ies_intrusion': row['ies_intrusion'],
            'ies_avoidance': row['ies_avoidance'],
            'ies_hyperarousal': row['ies_hyperarousal']
        })
    
    long_df = pd.DataFrame(long_data)
    
    return long_df, df_merged


def check_anova_assumptions(long_df, dv, output_dir):
    """
    Check ANOVA assumptions and save results.
    
    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format data
    dv : str
        Dependent variable name
    output_dir : Path
        Directory to save results
    """
    print(f"\n{'='*80}")
    print(f"ASSUMPTION CHECKS: {dv}")
    print('='*80)
    
    # Test normality within each group
    print("\nShapiro-Wilk Test of Normality:")
    normality_results = check_normality(long_df, group_col='trauma_group', dv_col=dv)
    print(normality_results)
    
    # Test homogeneity of variance
    print("\nLevene's Test of Homogeneity of Variance:")
    variance_results = check_homogeneity_of_variance(long_df, dv, 'trauma_group')
    print(pd.Series(variance_results))
    
    # Save results
    normality_results.to_csv(output_dir / f'assumptions_{dv}_normality.csv', index=False)
    pd.Series(variance_results).to_csv(output_dir / f'assumptions_{dv}_variance.csv')
    
    # Recommendations
    all_normal = normality_results['normal'].all()
    variances_equal = variance_results['homogeneous']
    
    print("\nRecommendations:")
    if all_normal and variances_equal:
        print("  ✓ All assumptions met. Proceed with standard ANOVA.")
    elif not all_normal and variances_equal:
        print("  ⚠ Normality violated but variances equal. ANOVA is robust to this with n>30 per group.")
    elif all_normal and not variances_equal:
        print("  ⚠ Variances unequal. Consider Welch's ANOVA for between-subjects effects.")
    else:
        print("  ⚠ Multiple assumptions violated. Consider non-parametric alternatives or Welch's correction.")
    
    return normality_results, variance_results


def run_anova_for_dv(long_df, dv, output_dir):
    """
    Run mixed ANOVA for a single dependent variable.
    
    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format data
    dv : str
        Dependent variable name
    output_dir : Path
        Directory to save results
    """
    print(f"\n{'='*80}")
    print(f"MIXED ANOVA: {dv}")
    print('='*80)
    
    try:
        # Run mixed ANOVA
        aov_results = run_mixed_anova(
            data=long_df,
            dv=dv,
            within_factor='load',
            between_factor='trauma_group',
            subject_id='sona_id'
        )
        
        # Format and display
        summary_table = create_anova_summary_table(aov_results)
        print("\nANOVA Results:")
        print(summary_table)
        
        # Save
        aov_results.to_csv(output_dir / f'anova_{dv}_full.csv', index=False)
        summary_table.to_csv(output_dir / f'anova_{dv}_summary.csv')
        
        # Interpret key effects
        print("\nKey Effects:")
        for _, row in aov_results.iterrows():
            source = row['Source']
            p_val = row['p-unc']
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            
            if 'load' in source.lower():
                print(f"  Load Effect: p = {p_val:.4f} {sig}")
            elif 'trauma' in source.lower() or 'group' in source.lower():
                print(f"  Group Effect: p = {p_val:.4f} {sig}")
            elif 'interaction' in source.lower() or '*' in source or ':' in source:
                print(f"  Load × Group Interaction: p = {p_val:.4f} {sig}")
        
        return aov_results
        
    except Exception as e:
        print(f"Error running ANOVA for {dv}: {e}")
        return None


def run_regressions_continuous_trauma(summary_df, dvs, output_dir):
    """
    Run linear regressions with continuous trauma predictors.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary data with DVs and trauma scores
    dvs : list of str
        Dependent variable names
    output_dir : Path
        Directory to save results
    """
    print(f"\n{'='*80}")
    print("LINEAR REGRESSIONS: Continuous Trauma Scores")
    print('='*80)
    
    # Predictors
    predictor_sets = {
        'LESS_Total': ['less_total_events'],
        'IES_Total': ['ies_total'],
        'IES_Subscales': ['ies_intrusion', 'ies_avoidance', 'ies_hyperarousal'],
        'LESS_and_IES': ['less_total_events', 'ies_total']
    }
    
    all_results = {}
    
    for pred_name, predictors in predictor_sets.items():
        print(f"\n{'='*80}")
        print(f"Predictor Set: {pred_name}")
        print('='*80)
        
        results = run_multiple_regressions(summary_df, dvs, predictors)
        
        # Save each regression
        for dv, result in results.items():
            if result is not None:
                filename = f'regression_{dv}_{pred_name}.txt'
                filepath = output_dir / filename
                with open(filepath, 'w') as f:
                    f.write(str(result['summary']))
                print(f"Saved: {filepath}")
                
                all_results[f'{dv}_{pred_name}'] = result
    
    return all_results


def run_all_statistical_analyses(output_dir='output/statistical_analyses'):
    """
    Run all statistical analyses.
    
    Parameters
    ----------
    output_dir : str or Path
        Directory to save results
    """
    print("\n" + "="*80)
    print("STATISTICAL ANALYSES FOR THESIS")
    print("="*80)
    
    # Load data
    summary_path = Path('output/summary_participant_metrics.csv')
    trials_path = Path('output/task_trials_long_all_participants.csv')
    
    if not summary_path.exists():
        print(f"\nERROR: Summary data not found at {summary_path}")
        print("Run the data pipeline first: python run_data_pipeline.py")
        return
    
    if not trials_path.exists():
        print(f"\nERROR: Task trials not found at {trials_path}")
        print("Run the data pipeline first: python run_data_pipeline.py")
        return
    
    summary_df = pd.read_csv(summary_path)
    trials_df = pd.read_csv(trials_path)
    
    print(f"\nLoaded {len(summary_df)} participants, {len(trials_df)} trials")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare long-format data
    print("\n" + "="*80)
    print("PREPARING DATA")
    print("="*80)
    
    long_df, summary_with_groups = prepare_long_format_data(summary_df, trials_df)
    print(f"Long-format data: {len(long_df)} observations")
    print(f"Participants with complete data: {long_df['sona_id'].nunique()}")
    
    # Save prepared data
    long_df.to_csv(output_path / 'data_long_format.csv', index=False)
    summary_with_groups.to_csv(output_path / 'data_summary_with_groups.csv', index=False)
    
    # Define dependent variables for ANOVAs
    anova_dvs = ['accuracy', 'rt']
    
    # Run assumption checks and ANOVAs
    print("\n" + "="*80)
    print("MIXED ANOVAs: Load × Trauma Group")
    print("="*80)
    
    anova_results = {}
    for dv in anova_dvs:
        # Check assumptions
        check_anova_assumptions(long_df, dv, output_path)
        
        # Run ANOVA
        result = run_anova_for_dv(long_df, dv, output_path)
        if result is not None:
            anova_results[dv] = result
    
    # Define dependent variables for regressions
    regression_dvs = [
        'accuracy_overall',
        'median_rt',
        'feedback_sensitivity',
        'learning_slope',
        'performance_drop_post_reversal',
        'adaptation_rate_post_reversal'
    ]
    
    # Filter to DVs that exist in data
    available_dvs = [dv for dv in regression_dvs if dv in summary_with_groups.columns]
    
    if len(available_dvs) > 0:
        # Run regressions
        print("\n" + "="*80)
        print("LINEAR REGRESSIONS")
        print("="*80)
        
        regression_results = run_regressions_continuous_trauma(
            summary_with_groups,
            available_dvs,
            output_path
        )
    else:
        print("\nWarning: No regression DVs found in data")
        regression_results = {}
    
    # Summary
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_path}/")
    print(f"  - {len(anova_results)} ANOVA tables")
    print(f"  - {len(regression_results)} regression outputs")
    print("\nNext steps:")
    print("  1. Review assumption check results")
    print("  2. Examine ANOVA tables for main effects and interactions")
    print("  3. Check regression outputs for trauma score associations")
    print("  4. Report significant findings in thesis")
    print()


if __name__ == '__main__':
    run_all_statistical_analyses()
