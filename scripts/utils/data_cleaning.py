"""
Data cleaning and parsing utilities for jsPsych RLWM experiment data.
"""

import json
import pandas as pd
import numpy as np


def parse_survey1_response(response_string):
    """
    Parse Survey 1 (LEC-5) multi-select response into binary indicators.

    Parameters:
    -----------
    response_string : str
        Semicolon-separated string of selected options
        Example: "(a) it happened to me personally; (c) I learned about it..."

    Returns:
    --------
    dict : Binary indicators for each response option
        Keys: 'any_exposure', 'personal', 'witnessed', 'learned', 'job', 'unsure', 'na'
    """
    if pd.isna(response_string) or response_string == '' or response_string == '{}':
        return {
            'any_exposure': 0,
            'personal': 0,
            'witnessed': 0,
            'learned': 0,
            'job': 0,
            'unsure': 0,
            'na': 0
        }

    response_str = str(response_string)

    indicators = {
        'personal': int('(a)' in response_str),
        'witnessed': int('(b)' in response_str),
        'learned': int('(c)' in response_str),
        'job': int('(d)' in response_str),
        'unsure': int('(e)' in response_str),
        'na': int('(f)' in response_str)
    }

    # Any exposure = selected any option except "doesn't apply" (f)
    indicators['any_exposure'] = int(any([
        indicators['personal'],
        indicators['witnessed'],
        indicators['learned'],
        indicators['job'],
        indicators['unsure']
    ]))

    return indicators


def extract_ies_scores(scored_responses_json):
    """
    Extract IES-R item scores from JSON string.

    Parameters:
    -----------
    scored_responses_json : str
        JSON string containing item scores
        Example: '{"s2_item01": 2, "s2_item02": 3, ...}'

    Returns:
    --------
    dict : Item scores for s2_item01 through s2_item22
    """
    if pd.isna(scored_responses_json) or scored_responses_json == '{}' or scored_responses_json == '':
        return {f's2_item{i:02d}': np.nan for i in range(1, 23)}

    try:
        scores = json.loads(scored_responses_json)
        # Ensure all 22 items are present
        item_scores = {f's2_item{i:02d}': scores.get(f's2_item{i:02d}', np.nan)
                       for i in range(1, 23)}
        return item_scores
    except json.JSONDecodeError:
        return {f's2_item{i:02d}': np.nan for i in range(1, 23)}


def filter_trial_type(df, section_name):
    """
    Filter dataframe to specific trial types based on section column.

    Parameters:
    -----------
    df : pandas.DataFrame
        Full jsPsych data
    section_name : str
        Section identifier (e.g., 'demographics', 'survey1', 'survey2', 'task')

    Returns:
    --------
    pandas.DataFrame : Filtered data
    """
    return df[df['section'] == section_name].copy()


def extract_demographics(df):
    """
    Extract demographic data for each participant.

    Parameters:
    -----------
    df : pandas.DataFrame
        Full jsPsych data

    Returns:
    --------
    pandas.DataFrame : One row per participant with demographic columns
    """
    demographic_cols = [
        'sona_id', 'age_years', 'country', 'primary_language',
        'gender', 'education', 'relationship_status',
        'living_arrangement', 'screen_time'
    ]

    # Get rows with demographics data (usually section == 'demographics')
    demo_data = df[df['section'] == 'demographics'].copy()

    if len(demo_data) == 0:
        # Try to get from any row that has demographic columns filled
        demo_data = df.dropna(subset=['age_years'], how='all')

    # Get one row per participant with demographic info
    available_cols = [col for col in demographic_cols if col in demo_data.columns]
    # Exclude sona_id from aggregation columns since it's the grouping key
    agg_cols = [col for col in available_cols if col != 'sona_id']
    demographics = demo_data.groupby('sona_id')[agg_cols].first().reset_index()

    return demographics


def extract_survey1_data(df):
    """
    Extract and parse Survey 1 (LEC-5) responses.

    Parameters:
    -----------
    df : pandas.DataFrame
        Full jsPsych data

    Returns:
    --------
    pandas.DataFrame : One row per participant with parsed survey1 responses
    """
    survey1_cols = [f's1_item{i:02d}' for i in range(1, 16)]

    # Get survey1 section
    survey1_data = df[df['section'] == 'survey1'].copy()

    if len(survey1_data) == 0:
        print("Warning: No survey1 data found")
        return pd.DataFrame()

    # Get one row per participant
    available_cols = [col for col in survey1_cols if col in survey1_data.columns]
    survey1 = survey1_data.groupby('sona_id')[available_cols].first().reset_index()

    # Parse each item's multi-select responses
    for item_col in [col for col in survey1_cols if col in survey1.columns]:
        parsed = survey1[item_col].apply(parse_survey1_response)

        # Create columns for any_exposure and personal only (30 columns total)
        survey1[f'{item_col}_any_exposure'] = parsed.apply(lambda x: x['any_exposure'])
        survey1[f'{item_col}_personal'] = parsed.apply(lambda x: x['personal'])

    # Drop original columns (keep parsed versions)
    survey1 = survey1.drop(columns=[col for col in survey1_cols if col in survey1.columns])

    return survey1


def extract_survey2_data(df):
    """
    Extract and parse Survey 2 (IES-R) responses from JSON.

    Parameters:
    -----------
    df : pandas.DataFrame
        Full jsPsych data

    Returns:
    --------
    pandas.DataFrame : One row per participant with 22 IES-R item scores
    """
    # Get survey2 section
    survey2_data = df[df['section'] == 'survey2'].copy()

    if len(survey2_data) == 0:
        print("Warning: No survey2 data found")
        return pd.DataFrame()

    # Get one row per participant
    survey2 = survey2_data.groupby('sona_id')['scored_responses'].first().reset_index()

    # Parse JSON responses
    parsed_scores = survey2['scored_responses'].apply(extract_ies_scores)

    # Convert to DataFrame
    scores_df = pd.DataFrame(parsed_scores.tolist())
    survey2 = pd.concat([survey2[['sona_id']], scores_df], axis=1)

    return survey2


def extract_task_trials(df):
    """
    Extract main task trial data.

    Parameters:
    -----------
    df : pandas.DataFrame
        Full jsPsych data

    Returns:
    --------
    pandas.DataFrame : Trial-by-trial task data for main task (block >= 3)
    """
    # Filter to task trials - identified by having 'block' column filled
    # Task trials typically have section=NaN in jsPsych output
    task_data = df[df['block'].notna()].copy()

    if len(task_data) == 0:
        print("Warning: No task data found")
        return pd.DataFrame()

    # Filter to main task only (block >= 3, excluding practice)
    task_data = task_data[task_data['block'] >= 3].copy()

    # Select relevant columns
    task_cols = [
        'sona_id', 'trial_index', 'time_elapsed', 'rt',
        'stimulus', 'key_press', 'correct',
        'set', 'block', 'trial', 'set_size', 'load_condition',
        'phase_type', 'key_answer', 'reversal_crit', 'counter'
    ]

    available_cols = [col for col in task_cols if col in task_data.columns]
    task_trials = task_data[available_cols].copy()

    # Convert data types
    if 'correct' in task_trials.columns:
        task_trials['correct'] = task_trials['correct'].astype(float)
    if 'rt' in task_trials.columns:
        task_trials['rt'] = pd.to_numeric(task_trials['rt'], errors='coerce')
    if 'key_press' in task_trials.columns:
        task_trials['key_press'] = pd.to_numeric(task_trials['key_press'], errors='coerce')

    return task_trials


def validate_data(df):
    """
    Perform data validation checks.

    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned data

    Returns:
    --------
    dict : Validation results and warnings
    """
    validation = {
        'n_participants': df['sona_id'].nunique() if 'sona_id' in df.columns else 0,
        'warnings': []
    }

    # Check for missing sona_id
    if 'sona_id' in df.columns:
        missing_id = df['sona_id'].isna().sum()
        if missing_id > 0:
            validation['warnings'].append(f"{missing_id} rows with missing sona_id")

    # Check for duplicate participants
    if 'sona_id' in df.columns and len(df) > 0:
        if df['sona_id'].duplicated().any():
            validation['warnings'].append("Duplicate participant IDs found")

    return validation


def clean_participant_id(sona_id):
    """
    Clean and standardize participant IDs.

    Parameters:
    -----------
    sona_id : str or numeric
        Raw participant ID

    Returns:
    --------
    str : Cleaned participant ID
    """
    if pd.isna(sona_id):
        return None
    return str(sona_id).strip()
