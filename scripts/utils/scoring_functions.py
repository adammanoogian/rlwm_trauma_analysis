"""
Scoring functions for surveys and task performance metrics.
"""

import pandas as pd
import numpy as np


# IES-R subscale definitions (Weiss & Marmar, 1997)
INTRUSION_ITEMS = [1, 3, 6, 9, 14, 16, 20]
AVOIDANCE_ITEMS = [5, 7, 8, 11, 12, 13, 17, 22]
HYPERAROUSAL_ITEMS = [2, 4, 10, 15, 18, 19, 21]


def score_less(survey1_df):
    """
    Calculate LESS (Life Events Scale - Short) summary scores.

    Parameters:
    -----------
    survey1_df : pandas.DataFrame
        Survey 1 data with parsed binary columns

    Returns:
    --------
    pandas.DataFrame : DataFrame with summary scores added
    """
    result_df = survey1_df.copy()

    # Find all any_exposure columns
    any_exposure_cols = [col for col in result_df.columns if col.endswith('_any_exposure')]
    personal_cols = [col for col in result_df.columns if col.endswith('_personal')]

    # Calculate summary scores
    result_df['less_total_events'] = result_df[any_exposure_cols].sum(axis=1)
    result_df['less_personal_events'] = result_df[personal_cols].sum(axis=1)

    return result_df


def score_ies_r(survey2_df):
    """
    Calculate IES-R (Impact of Event Scale - Revised) total and subscale scores.

    Parameters:
    -----------
    survey2_df : pandas.DataFrame
        Survey 2 data with s2_item01 through s2_item22 columns

    Returns:
    --------
    pandas.DataFrame : DataFrame with subscale and total scores added
    """
    result_df = survey2_df.copy()

    # Get item columns
    item_cols = [f's2_item{i:02d}' for i in range(1, 23)]

    # Ensure all items exist
    missing_items = [col for col in item_cols if col not in result_df.columns]
    if missing_items:
        print(f"Warning: Missing IES-R items: {missing_items}")
        for col in missing_items:
            result_df[col] = np.nan

    # Calculate subscale scores
    intrusion_cols = [f's2_item{i:02d}' for i in INTRUSION_ITEMS]
    avoidance_cols = [f's2_item{i:02d}' for i in AVOIDANCE_ITEMS]
    hyperarousal_cols = [f's2_item{i:02d}' for i in HYPERAROUSAL_ITEMS]

    result_df['ies_intrusion'] = result_df[intrusion_cols].sum(axis=1)
    result_df['ies_avoidance'] = result_df[avoidance_cols].sum(axis=1)
    result_df['ies_hyperarousal'] = result_df[hyperarousal_cols].sum(axis=1)
    result_df['ies_total'] = result_df['ies_intrusion'] + result_df['ies_avoidance'] + result_df['ies_hyperarousal']

    return result_df


def calculate_overall_task_metrics(task_df):
    """
    Calculate overall task performance metrics.

    Parameters:
    -----------
    task_df : pandas.DataFrame
        Task trial data for a single participant

    Returns:
    --------
    dict : Overall performance metrics
    """
    if len(task_df) == 0:
        return {
            'n_trials_total': 0,
            'n_trials_completed': 0,
            'accuracy_overall': np.nan,
            'mean_rt_overall': np.nan,
            'median_rt_overall': np.nan
        }

    completed_trials = task_df[task_df['key_press'].notna()]

    metrics = {
        'n_trials_total': len(task_df),
        'n_trials_completed': len(completed_trials),
        'accuracy_overall': task_df['correct'].mean() if 'correct' in task_df.columns else np.nan,
        'mean_rt_overall': completed_trials['rt'].mean() if 'rt' in completed_trials.columns else np.nan,
        'median_rt_overall': completed_trials['rt'].median() if 'rt' in completed_trials.columns else np.nan
    }

    return metrics


def calculate_load_condition_metrics(task_df):
    """
    Calculate performance metrics by load condition (low vs high).

    Parameters:
    -----------
    task_df : pandas.DataFrame
        Task trial data for a single participant

    Returns:
    --------
    dict : Load-specific performance metrics
    """
    metrics = {}

    if 'load_condition' not in task_df.columns or len(task_df) == 0:
        for load in ['low', 'high']:
            metrics[f'accuracy_{load}_load'] = np.nan
            metrics[f'mean_rt_{load}_load'] = np.nan
            metrics[f'n_trials_{load}_load'] = 0
        return metrics

    for load in ['low', 'high']:
        load_trials = task_df[task_df['load_condition'] == load]
        completed_load = load_trials[load_trials['key_press'].notna()]

        metrics[f'accuracy_{load}_load'] = load_trials['correct'].mean() if len(load_trials) > 0 else np.nan
        metrics[f'mean_rt_{load}_load'] = completed_load['rt'].mean() if len(completed_load) > 0 else np.nan
        metrics[f'n_trials_{load}_load'] = len(load_trials)

    return metrics


def calculate_set_size_metrics(task_df):
    """
    Calculate performance metrics by set size (2, 3, 5, 6).

    Parameters:
    -----------
    task_df : pandas.DataFrame
        Task trial data for a single participant

    Returns:
    --------
    dict : Set-size specific performance metrics
    """
    metrics = {}

    if 'set_size' not in task_df.columns or len(task_df) == 0:
        for set_size in [2, 3, 5, 6]:
            metrics[f'accuracy_setsize_{set_size}'] = np.nan
            metrics[f'mean_rt_setsize_{set_size}'] = np.nan
            metrics[f'n_trials_setsize_{set_size}'] = 0
        return metrics

    for set_size in [2, 3, 5, 6]:
        ss_trials = task_df[task_df['set_size'] == set_size]
        completed_ss = ss_trials[ss_trials['key_press'].notna()]

        metrics[f'accuracy_setsize_{set_size}'] = ss_trials['correct'].mean() if len(ss_trials) > 0 else np.nan
        metrics[f'mean_rt_setsize_{set_size}'] = completed_ss['rt'].mean() if len(completed_ss) > 0 else np.nan
        metrics[f'n_trials_setsize_{set_size}'] = len(ss_trials)

    return metrics


def calculate_block_tercile_metrics(task_df):
    """
    Calculate performance metrics by block terciles (early/middle/late).

    Parameters:
    -----------
    task_df : pandas.DataFrame
        Task trial data for a single participant

    Returns:
    --------
    dict : Block-specific performance metrics
    """
    metrics = {}

    if 'block' not in task_df.columns or len(task_df) == 0:
        for period in ['early', 'middle', 'late']:
            metrics[f'accuracy_{period}_blocks'] = np.nan
            metrics[f'mean_rt_{period}_blocks'] = np.nan
        return metrics

    # Block ranges (main task blocks: 3-23)
    early_blocks = range(3, 9)    # blocks 3-8
    middle_blocks = range(9, 16)  # blocks 9-15
    late_blocks = range(16, 24)   # blocks 16-23

    for period, block_range in [('early', early_blocks), ('middle', middle_blocks), ('late', late_blocks)]:
        period_trials = task_df[task_df['block'].isin(block_range)]
        completed_period = period_trials[period_trials['key_press'].notna()]

        metrics[f'accuracy_{period}_blocks'] = period_trials['correct'].mean() if len(period_trials) > 0 else np.nan
        metrics[f'mean_rt_{period}_blocks'] = completed_period['rt'].mean() if len(completed_period) > 0 else np.nan

    return metrics


def calculate_learning_curve(task_df):
    """
    Calculate block-by-block learning metrics.

    Parameters:
    -----------
    task_df : pandas.DataFrame
        Task trial data for a single participant

    Returns:
    --------
    dict : Learning curve metrics
    """
    if 'block' not in task_df.columns or len(task_df) == 0:
        return {
            'learning_slope': np.nan,
            'learning_improvement_early_to_late': np.nan
        }

    # Calculate accuracy per block
    block_accuracy = task_df.groupby('block')['correct'].mean()

    if len(block_accuracy) < 2:
        return {
            'learning_slope': np.nan,
            'learning_improvement_early_to_late': np.nan
        }

    # Calculate linear slope across blocks
    blocks = block_accuracy.index.values
    accuracy = block_accuracy.values

    # Fit linear regression
    slope = np.polyfit(blocks, accuracy, 1)[0] if len(blocks) > 1 else np.nan

    # Calculate improvement from first 3 blocks to last 3 blocks
    early_acc = block_accuracy.iloc[:3].mean() if len(block_accuracy) >= 3 else block_accuracy.iloc[0]
    late_acc = block_accuracy.iloc[-3:].mean() if len(block_accuracy) >= 3 else block_accuracy.iloc[-1]
    improvement = late_acc - early_acc

    return {
        'learning_slope': slope,
        'learning_improvement_early_to_late': improvement
    }


def identify_reversals(task_df):
    """
    Identify reversal trials based on counter reset or explicit reversal markers.

    Parameters:
    -----------
    task_df : pandas.DataFrame
        Task trial data for a single participant

    Returns:
    --------
    pandas.Series : Boolean series indicating reversal trials
    """
    if 'counter' not in task_df.columns or len(task_df) == 0:
        return pd.Series([False] * len(task_df), index=task_df.index)

    # Sort by trial order
    sorted_df = task_df.sort_values(['block', 'trial']).copy()

    # Reversal occurs when counter resets (goes from high to 0 or low value)
    counter_diff = sorted_df['counter'].diff()

    # A reset is indicated by a large negative difference
    is_reversal = counter_diff < -3

    return is_reversal.reindex(task_df.index, fill_value=False)


def calculate_reversal_metrics(task_df):
    """
    Calculate reversal-specific performance metrics.

    Parameters:
    -----------
    task_df : pandas.DataFrame
        Task trial data for a single participant

    Returns:
    --------
    dict : Reversal performance metrics
    """
    if len(task_df) == 0:
        return {
            'n_reversals': 0,
            'performance_drop_post_reversal': np.nan,
            'adaptation_rate_post_reversal': np.nan
        }

    # Identify reversals
    is_reversal = identify_reversals(task_df)
    reversal_indices = task_df[is_reversal].index

    if len(reversal_indices) == 0:
        return {
            'n_reversals': 0,
            'performance_drop_post_reversal': np.nan,
            'adaptation_rate_post_reversal': np.nan
        }

    # Calculate performance before and after reversals
    pre_reversal_acc = []
    post_reversal_acc_immediate = []
    post_reversal_acc_adapted = []

    sorted_df = task_df.sort_values(['block', 'trial']).reset_index(drop=True)
    reversal_positions = sorted_df.index[sorted_df.index.isin(reversal_indices)].tolist()

    for rev_pos in reversal_positions:
        # Get 5 trials before reversal
        pre_start = max(0, rev_pos - 5)
        pre_trials = sorted_df.iloc[pre_start:rev_pos]

        # Get 3 trials immediately after reversal
        post_immediate = sorted_df.iloc[rev_pos:rev_pos + 3]

        # Get trials 10-15 after reversal (adaptation period)
        post_adapted = sorted_df.iloc[rev_pos + 10:rev_pos + 15]

        if len(pre_trials) > 0:
            pre_reversal_acc.append(pre_trials['correct'].mean())
        if len(post_immediate) > 0:
            post_reversal_acc_immediate.append(post_immediate['correct'].mean())
        if len(post_adapted) > 0:
            post_reversal_acc_adapted.append(post_adapted['correct'].mean())

    # Calculate metrics
    mean_pre = np.mean(pre_reversal_acc) if len(pre_reversal_acc) > 0 else np.nan
    mean_post_immediate = np.mean(post_reversal_acc_immediate) if len(post_reversal_acc_immediate) > 0 else np.nan
    mean_post_adapted = np.mean(post_reversal_acc_adapted) if len(post_reversal_acc_adapted) > 0 else np.nan

    performance_drop = mean_pre - mean_post_immediate if not np.isnan(mean_pre) and not np.isnan(mean_post_immediate) else np.nan
    adaptation_rate = mean_post_adapted - mean_post_immediate if not np.isnan(mean_post_adapted) and not np.isnan(mean_post_immediate) else np.nan

    return {
        'n_reversals': len(reversal_indices),
        'performance_drop_post_reversal': performance_drop,
        'adaptation_rate_post_reversal': adaptation_rate
    }


def calculate_all_task_metrics(task_df):
    """
    Calculate all task performance metrics for a single participant.

    Parameters:
    -----------
    task_df : pandas.DataFrame
        Task trial data for a single participant

    Returns:
    --------
    dict : All task performance metrics
    """
    metrics = {}

    # Overall metrics
    metrics.update(calculate_overall_task_metrics(task_df))

    # By load condition
    metrics.update(calculate_load_condition_metrics(task_df))

    # By set size
    metrics.update(calculate_set_size_metrics(task_df))

    # By block terciles
    metrics.update(calculate_block_tercile_metrics(task_df))

    # Learning curves
    metrics.update(calculate_learning_curve(task_df))

    # Reversal metrics
    metrics.update(calculate_reversal_metrics(task_df))

    return metrics
