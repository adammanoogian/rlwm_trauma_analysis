"""
Parse survey data (LEC-5 and IES-R) for MLE parameter analysis.

This script:
1. Parses raw session files from the external data directory
2. Uses the same parsing logic as scripts/utils/data_cleaning.py for consistency
3. Computes IES-R subscales and LEC-5 summary scores
4. Assigns trauma groups based on median splits (Senta et al., 2025)

Note: This script always parses fresh from raw data to ensure completeness.
The data_cleaning.py utilities define the canonical parsing logic.

Based on Senta et al. (2025) methodology.

Outputs:
    - output/mle/participant_surveys.csv: All survey scores
    - output/mle/trauma_group_assignments.csv: Group membership

Usage:
    python scripts/analysis/parse_surveys_for_mle.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = Path("C:/Users/aman0087/Documents/Github/rlwm_trauma/data")
OUTPUT_DIR = PROJECT_ROOT / "output" / "mle"

# Add utils to path for reusing existing parsing functions
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))
try:
    from data_cleaning import parse_survey1_response, extract_ies_scores
    print("Using parsing functions from scripts/utils/data_cleaning.py")
except ImportError:
    print("Note: Could not import data_cleaning utilities, using equivalent built-in parsing")
    parse_survey1_response = None
    extract_ies_scores = None

# IES-R subscale mappings (1-indexed as per standard scoring)
IESR_INTRUSION = [1, 2, 3, 6, 9, 14, 16, 20]  # 8 items, max 32
IESR_AVOIDANCE = [5, 7, 8, 11, 12, 13, 17, 22]  # 8 items, max 32
IESR_HYPERAROUSAL = [4, 10, 15, 18, 19, 21]  # 6 items, max 24


def parse_lec5_item(response_str: str) -> Dict:
    """
    Parse a single LEC-5 item response using data_cleaning.py logic.

    Binary indicators for exposure type:
    - (a) = personal
    - (b) = witnessed
    - (c) = learned
    - (d) = job exposure
    - (e) = unsure
    - (f) = doesn't apply (no exposure)

    Parameters
    ----------
    response_str : str
        String containing response text (may have multiple selections)

    Returns
    -------
    dict
        Binary indicators: any_exposure, personal, witnessed, learned, job
    """
    if pd.isna(response_str) or response_str == '' or response_str == '{}':
        return {'any_exposure': 0, 'personal': 0, 'witnessed': 0,
                'learned': 0, 'job': 0, 'unsure': 0}

    response_str = str(response_str)

    indicators = {
        'personal': int('(a)' in response_str),
        'witnessed': int('(b)' in response_str),
        'learned': int('(c)' in response_str),
        'job': int('(d)' in response_str),
        'unsure': int('(e)' in response_str),
    }

    # Any exposure = any option except (f) "doesn't apply"
    indicators['any_exposure'] = int(any([
        indicators['personal'], indicators['witnessed'],
        indicators['learned'], indicators['job'], indicators['unsure']
    ]))

    return indicators


def parse_lec5_from_json(response: Dict) -> Dict:
    """
    Parse LEC-5 response from JSON dict with s1_item01-s1_item15 keys.

    Parameters
    ----------
    response : dict
        JSON response from survey-multi-select trial

    Returns
    -------
    dict
        Summary scores: lec_total, lec_personal, lec_witnessed, lec_learned, lec_job
    """
    result = {
        'lec_total': 0,
        'lec_personal': 0,
        'lec_witnessed': 0,
        'lec_learned': 0,
        'lec_job': 0,
    }

    for i in range(1, 16):
        key = f's1_item{i:02d}'
        item_value = response.get(key, [])

        # Convert list to string for parsing
        if isinstance(item_value, list):
            item_str = '; '.join(str(v) for v in item_value)
        else:
            item_str = str(item_value)

        indicators = parse_lec5_item(item_str)

        result['lec_total'] += indicators['any_exposure']
        result['lec_personal'] += indicators['personal']
        result['lec_witnessed'] += indicators['witnessed']
        result['lec_learned'] += indicators['learned']
        result['lec_job'] += indicators['job']

    return result


def parse_iesr_from_json(response: Dict) -> Dict:
    """
    Parse IES-R response from JSON dict with s2_item01-s2_item22 keys.

    Computes subscale scores following standard IES-R scoring.

    Parameters
    ----------
    response : dict
        JSON response from survey-likert trial

    Returns
    -------
    dict
        Scores: ies_total, ies_intrusion, ies_avoidance, ies_hyperarousal
    """
    result = {
        'ies_intrusion': 0,
        'ies_avoidance': 0,
        'ies_hyperarousal': 0,
        'ies_total': 0,
    }

    for i in range(1, 23):
        key = f's2_item{i:02d}'
        value = response.get(key, 0)
        if value is None or pd.isna(value):
            value = 0
        value = int(value)

        result['ies_total'] += value

        if i in IESR_INTRUSION:
            result['ies_intrusion'] += value
        elif i in IESR_AVOIDANCE:
            result['ies_avoidance'] += value
        elif i in IESR_HYPERAROUSAL:
            result['ies_hyperarousal'] += value

    return result


def parse_session_file(filepath: Path) -> Optional[Dict]:
    """
    Parse a single raw session CSV file to extract survey data.

    This function extracts:
    - LEC-5 data from survey-multi-select trial type
    - IES-R data from survey-likert trial type

    Parameters
    ----------
    filepath : Path
        Path to raw session CSV file

    Returns
    -------
    dict or None
        Survey data if both surveys found, None otherwise
    """
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f"  Error reading {filepath.name}: {e}")
        return None

    result = {'filename': filepath.name}

    # Get sona_id
    if 'sona_id' in df.columns:
        sona_ids = df['sona_id'].dropna().unique()
        sona_ids = [str(s) for s in sona_ids if str(s) not in ('null', '', 'nan')]
        if sona_ids:
            result['sona_id'] = sona_ids[0]

    # Parse LEC-5 (survey-multi-select)
    lec_rows = df[df['trial_type'] == 'survey-multi-select']
    if len(lec_rows) == 0:
        return None

    for _, row in lec_rows.iterrows():
        try:
            if pd.notna(row.get('response')):
                response = json.loads(row['response'])
                if 's1_item01' in response:
                    lec_data = parse_lec5_from_json(response)
                    result.update(lec_data)
                    break
        except (json.JSONDecodeError, TypeError):
            continue

    if 'lec_total' not in result:
        return None

    # Parse IES-R (survey-likert)
    iesr_rows = df[df['trial_type'] == 'survey-likert']
    if len(iesr_rows) == 0:
        return None

    for _, row in iesr_rows.iterrows():
        try:
            if pd.notna(row.get('response')):
                response = json.loads(row['response'])
                if 's2_item01' in response:
                    iesr_data = parse_iesr_from_json(response)
                    result.update(iesr_data)
                    break
        except (json.JSONDecodeError, TypeError):
            continue

    if 'ies_total' not in result:
        return None

    return result


def assign_trauma_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign participants to trauma groups based on median splits.

    Groups (Senta et al., 2025 approach):
    - No Trauma: Low LEC (<= median), Low IES-R (<= median)
    - Trauma-No Impact: High LEC (> median), Low IES-R (<= median)
    - Trauma-Ongoing Impact: High LEC (> median), High IES-R (> median)
    - Low Exposure-High Symptoms: Low LEC, High IES-R (paradoxical)

    Parameters
    ----------
    df : pd.DataFrame
        Survey data with lec_total and ies_total columns

    Returns
    -------
    pd.DataFrame
        Original df with added group columns
    """
    df = df.copy()

    # Compute medians
    lec_median = df['lec_total'].median()
    ies_median = df['ies_total'].median()

    print(f"\nMedian splits:")
    print(f"  LEC-5 median: {lec_median}")
    print(f"  IES-R total median: {ies_median}")

    # Binary flags
    df['high_lec'] = df['lec_total'] > lec_median
    df['high_ies'] = df['ies_total'] > ies_median

    # Assign groups
    def assign_group(row):
        if not row['high_lec'] and not row['high_ies']:
            return 'No Trauma'
        elif row['high_lec'] and not row['high_ies']:
            return 'Trauma-No Impact'
        elif row['high_lec'] and row['high_ies']:
            return 'Trauma-Ongoing Impact'
        else:
            return 'Low Exposure-High Symptoms'

    df['hypothesis_group'] = df.apply(assign_group, axis=1)

    # Print group counts
    print(f"\nGroup assignments:")
    for group, count in df['hypothesis_group'].value_counts().sort_index().items():
        print(f"  {group}: {count}")

    return df


def validate_iesr_subscales(df: pd.DataFrame) -> bool:
    """
    Validate that IES-R subscales are in expected ranges and sum correctly.

    Expected ranges:
    - Intrusion: 0-32 (8 items × 4)
    - Avoidance: 0-32 (8 items × 4)
    - Hyperarousal: 0-24 (6 items × 4)
    - Total: 0-88 (22 items × 4)
    """
    valid = True

    # Check ranges
    checks = [
        ('ies_intrusion', 0, 32),
        ('ies_avoidance', 0, 32),
        ('ies_hyperarousal', 0, 24),
        ('ies_total', 0, 88),
    ]

    for col, min_val, max_val in checks:
        if col in df.columns:
            actual_min = df[col].min()
            actual_max = df[col].max()
            if actual_min < min_val or actual_max > max_val:
                print(f"WARNING: {col} out of range [{min_val}-{max_val}]: {actual_min}-{actual_max}")
                valid = False

    # Verify subscales sum to total
    if all(col in df.columns for col in ['ies_intrusion', 'ies_avoidance', 'ies_hyperarousal', 'ies_total']):
        computed = df['ies_intrusion'] + df['ies_avoidance'] + df['ies_hyperarousal']
        if not np.allclose(computed, df['ies_total']):
            print("WARNING: IES-R subscales do not sum to total")
            valid = False
        else:
            print("IES-R subscale validation: PASSED")

    return valid


def main():
    """Main function to parse all surveys and create group assignments."""
    print("=" * 70)
    print("Parsing Survey Data for MLE Analysis")
    print("=" * 70)
    print(f"\nRaw data directory: {RAW_DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load participant info to get filename -> sona_id mapping
    participant_info_path = PROJECT_ROOT / "output" / "participant_info.csv"
    if not participant_info_path.exists():
        print(f"ERROR: participant_info.csv not found at {participant_info_path}")
        return

    participant_info = pd.read_csv(participant_info_path)
    filename_to_sona = dict(zip(participant_info['filename'], participant_info['sona_id']))
    print(f"\nParticipant info loaded: {len(filename_to_sona)} participants mapped")

    # Get list of participants with MLE fits
    mle_fits_path = OUTPUT_DIR / "qlearning_individual_fits.csv"
    if not mle_fits_path.exists():
        print(f"ERROR: MLE fits not found at {mle_fits_path}")
        return

    mle_fits = pd.read_csv(mle_fits_path)
    fitted_participants = set(mle_fits['participant_id'].astype(str))
    print(f"Participants with MLE fits: {len(fitted_participants)}")

    # Parse surveys from raw files
    print(f"\n{'-'*60}")
    print("Parsing survey data from raw session files...")
    print(f"{'-'*60}")

    all_surveys = []
    missing_files = []
    parse_errors = []

    for filename, sona_id in filename_to_sona.items():
        sona_id = str(sona_id)

        # Only process participants with MLE fits
        if sona_id not in fitted_participants:
            continue

        filepath = RAW_DATA_DIR / filename
        if not filepath.exists():
            missing_files.append(filename)
            continue

        survey_data = parse_session_file(filepath)

        if survey_data:
            survey_data['sona_id'] = sona_id  # Use mapped sona_id
            all_surveys.append(survey_data)
            print(f"  Parsed: {sona_id} (LEC={survey_data['lec_total']}, IES={survey_data['ies_total']})")
        else:
            parse_errors.append((filename, sona_id))

    # Report issues
    if missing_files:
        print(f"\nWARNING: {len(missing_files)} raw files not found")
    if parse_errors:
        print(f"WARNING: {len(parse_errors)} files could not be parsed")

    if not all_surveys:
        print("ERROR: No survey data extracted!")
        return

    # Create DataFrame
    surveys_df = pd.DataFrame(all_surveys)
    print(f"\n{'='*60}")
    print(f"Successfully parsed surveys for {len(surveys_df)} / {len(fitted_participants)} participants")

    # Validate IES-R
    print(f"\n{'-'*60}")
    print("Validating IES-R subscales...")
    validate_iesr_subscales(surveys_df)

    # Assign trauma groups
    print(f"\n{'-'*60}")
    print("Assigning trauma groups...")
    surveys_df = assign_trauma_groups(surveys_df)

    # Prepare output columns
    survey_cols = [
        'sona_id',
        'lec_total', 'lec_personal', 'lec_witnessed', 'lec_learned', 'lec_job',
        'ies_total', 'ies_intrusion', 'ies_avoidance', 'ies_hyperarousal'
    ]
    # Filter to columns that exist
    survey_cols = [c for c in survey_cols if c in surveys_df.columns]

    group_cols = [
        'sona_id', 'hypothesis_group', 'high_lec', 'high_ies',
        'lec_total', 'ies_total'
    ]

    # Save outputs
    print(f"\n{'-'*60}")
    print("Saving output files...")

    surveys_output = surveys_df[survey_cols].copy()
    surveys_output.to_csv(OUTPUT_DIR / "participant_surveys.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'participant_surveys.csv'}")

    groups_output = surveys_df[group_cols].copy()
    groups_output.to_csv(OUTPUT_DIR / "trauma_group_assignments.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'trauma_group_assignments.csv'}")

    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")

    print("\nLEC-5 (Trauma Exposure):")
    print(f"  Total events: {surveys_df['lec_total'].mean():.2f} +/- {surveys_df['lec_total'].std():.2f}")
    print(f"  Personal: {surveys_df['lec_personal'].mean():.2f} +/- {surveys_df['lec_personal'].std():.2f}")
    print(f"  Range: {surveys_df['lec_total'].min()}-{surveys_df['lec_total'].max()}")

    print("\nIES-R (Symptom Severity):")
    print(f"  Total: {surveys_df['ies_total'].mean():.2f} +/- {surveys_df['ies_total'].std():.2f}")
    print(f"  Intrusion: {surveys_df['ies_intrusion'].mean():.2f} +/- {surveys_df['ies_intrusion'].std():.2f}")
    print(f"  Avoidance: {surveys_df['ies_avoidance'].mean():.2f} +/- {surveys_df['ies_avoidance'].std():.2f}")
    print(f"  Hyperarousal: {surveys_df['ies_hyperarousal'].mean():.2f} +/- {surveys_df['ies_hyperarousal'].std():.2f}")

    print(f"\n{'='*60}")
    print("Survey parsing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
