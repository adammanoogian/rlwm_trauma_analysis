"""
Parse raw jsPsych experiment data: extract task trials, surveys, and demographics.

This consolidated script combines best practices from multiple parsing approaches:
- Robust data cleaning (practice trial removal, invalid response filtering)
- Survey extraction and scoring (LEC-5 → LESS, IES-R → subscales)
- Consistent participant ID mapping
- Detailed summary statistics

Usage:
    python scripts/01_data_preprocessing/01_parse_raw_data.py

Outputs (CCDS tiered layout — populated by plan 31-02):
    data/processed/task_trials_long.csv                   - Main task only (canonical input to fitting)
    data/processed/task_trials_long_all.csv               - All blocks incl. practice (is_practice flag)
    data/processed/summary_participant_metrics.csv        - Combined participant metrics (tracked)
    data/interim/participant_info.csv                     - Participant summary info (gitignored PII)
    data/interim/parsed_survey1.csv                       - LEC-5 with LESS scores (gitignored PII)
    data/interim/parsed_survey2.csv                       - IES-R with subscale scores (gitignored PII)
    data/interim/parsed_demographics.csv                  - Demographics data (gitignored PII)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Add utils to path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'utils'))
from data_cleaning import (
    extract_demographics,
    extract_survey1_data,
    extract_survey2_data,
)
from scoring import score_ies_r, score_less

# Import config CCDS constants + excluded participants
# (CCDS constants landed in plan 31-01; physical files moved in plan 31-02)
from config import (
    DATA_RAW_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    DataParams,
)

try:
    from config import EXCLUDED_PARTICIPANTS
except ImportError:
    EXCLUDED_PARTICIPANTS = []

# ============================================================================
# Configuration (CCDS tiered: raw → interim → processed)
# ============================================================================
MAPPING_FILE = DATA_RAW_DIR / 'participant_id_mapping.json'

# Minimum trials to include a participant (exclude incomplete sessions)
MIN_TRIALS = 100


def load_participant_mapping():
    """Load participant ID mapping from JSON file."""
    if not MAPPING_FILE.exists():
        print(f"Warning: No ID mapping file found at {MAPPING_FILE}")
        print("Will use sona_id from data files directly.")
        return None

    with open(MAPPING_FILE) as f:
        return json.load(f)


def parse_single_file(filepath: Path, assigned_id: int = None) -> dict:
    """
    Parse all data from a single participant CSV file.

    Returns dict with task_trials, demographics, survey1, survey2 DataFrames.
    """
    result = {
        'task_trials': None,
        'demographics': None,
        'survey1': None,
        'survey2': None,
        'n_trials': 0,
        'sona_id': assigned_id
    }

    try:
        df = pd.read_csv(filepath, low_memory=False)

        # Determine participant ID
        if assigned_id is not None:
            sona_id = assigned_id
        elif 'sona_id' in df.columns:
            sona_values = df['sona_id'].dropna().unique()
            valid_sona = [s for s in sona_values if s and str(s).strip() and str(s) != 'nan']
            sona_id = int(valid_sona[0]) if valid_sona else None
        else:
            sona_id = None

        if sona_id is None:
            # Deterministic fallback: use filename stem (stable across runs)
            sona_id = f"anon_{filepath.stem}"

        result['sona_id'] = sona_id
        df['sona_id'] = sona_id

        # ---- Extract Task Trials ----
        if 'trial_type' in df.columns:
            task_data = df[df['trial_type'] == 'categorize-html'].copy()

            if len(task_data) > 0:
                task_data['sona_id'] = sona_id
                task_data['source_file'] = filepath.name
                task_data['trial_in_experiment'] = range(1, len(task_data) + 1)

                # Handle block column
                if 'block' not in task_data.columns or task_data['block'].isna().all():
                    task_data['block'] = (task_data['trial_in_experiment'] // 50) + 3

                # Map key_answer to key_press if needed
                if 'key_answer' in task_data.columns and 'key_press' not in task_data.columns:
                    task_data['key_press'] = task_data['key_answer']

                # Select columns (including phase_type for practice/main classification)
                task_cols = ['sona_id', 'trial_in_experiment', 'block', 'stimulus',
                            'key_press', 'correct', 'rt', 'time_elapsed', 'set_size',
                            'load_condition', 'phase_type', 'source_file']
                cols_available = [c for c in task_cols if c in task_data.columns]

                result['task_trials'] = task_data[cols_available].copy()
                result['n_trials'] = len(task_data)

        # ---- Extract Demographics ----
        demographics = extract_demographics(df)
        if len(demographics) > 0:
            demographics['sona_id'] = sona_id
            result['demographics'] = demographics

        # ---- Extract Survey 1 (LEC-5) ----
        survey1 = extract_survey1_data(df)
        if len(survey1) > 0:
            survey1['sona_id'] = sona_id
            result['survey1'] = survey1

        # ---- Extract Survey 2 (IES-R) ----
        survey2 = extract_survey2_data(df)
        if len(survey2) > 0:
            survey2['sona_id'] = sona_id
            result['survey2'] = survey2

    except Exception as e:
        print(f"  Error parsing {filepath.name}: {e}")

    return result


def clean_task_data(task_df: pd.DataFrame, filter_practice: bool = False) -> pd.DataFrame:
    """
    Apply data cleaning filters to task data.

    Args:
        task_df: Raw task trial DataFrame
        filter_practice: If True, remove practice trials (blocks 1-2). Default False.

    Returns:
        Cleaned DataFrame with is_practice column added
    """
    if len(task_df) == 0:
        return task_df

    df = task_df.copy()
    n_original = len(df)

    # Ensure correct data types
    if 'stimulus' in df.columns:
        df['stimulus'] = pd.to_numeric(df['stimulus'], errors='coerce').fillna(-1).astype(int)
        # Convert to 0-indexed if needed (stimulus 1-6 becomes 0-5)
        if df['stimulus'].min() >= 1:
            df['stimulus'] = df['stimulus'] - 1
        df.loc[df['stimulus'] < 0, 'stimulus'] = -1

    if 'correct' in df.columns:
        df['correct'] = pd.to_numeric(df['correct'], errors='coerce')

    if 'block' in df.columns:
        df['block'] = pd.to_numeric(df['block'], errors='coerce')
        df = df.dropna(subset=['block'])
        df['block'] = df['block'].astype(int)

    if 'key_press' in df.columns:
        df['key_press'] = pd.to_numeric(df['key_press'], errors='coerce').fillna(-1).astype(int)

    # Add is_practice column (based on phase_type if available, else block number)
    if 'phase_type' in df.columns:
        df['is_practice'] = df['phase_type'].isin(['practice_static', 'practice_dynamic'])
    else:
        df['is_practice'] = df['block'] < 3

    # Count practice trials before any filtering
    n_practice_trials = df['is_practice'].sum()

    # 1. Optionally remove practice trials (blocks 1-2)
    if filter_practice:
        df = df[~df['is_practice']].copy()
        n_after_practice = len(df)
    else:
        n_after_practice = n_original

    # 2. Filter out invalid trials (key_press == -1 or stimulus < 0)
    df = df[(df['key_press'] >= 0) & (df['stimulus'] >= 0)]
    n_after_invalid = len(df)

    # Add derived columns
    df['trial_in_block'] = df.groupby(['sona_id', 'block']).cumcount() + 1

    # Add reward column
    if 'correct' in df.columns:
        df['reward'] = df['correct'].astype(float)

    # Infer set_size if not present
    if 'set_size' not in df.columns or df['set_size'].isna().all():
        df['set_size'] = df.groupby(['sona_id', 'block'])['stimulus'].transform('max') + 1

    # Add load_condition if not present
    if 'load_condition' not in df.columns or df['load_condition'].isna().all():
        df['load_condition'] = df['set_size'].apply(
            lambda x: 'low' if x <= 3 else 'high' if x >= 4 else 'unknown'
        )

    # Report cleaning stats
    n_removed_practice = n_original - n_after_practice if filter_practice else 0
    n_invalid = n_after_practice - n_after_invalid
    if n_removed_practice > 0 or n_invalid > 0:
        print(f"  Cleaned: removed {n_removed_practice} practice + {n_invalid} invalid trials")
    elif n_practice_trials > 0 and not filter_practice:
        print(f"  Cleaned: {n_practice_trials} practice trials preserved (is_practice=True)")

    return df


def main():
    print("=" * 80)
    print("STEP 1: Parsing Raw jsPsych Data (Consolidated)")
    print("=" * 80)
    print()

    # Create output directories (CCDS tiered)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load ID mapping
    id_mapping = load_participant_mapping()

    # Find CSV files to process (data/raw/ — sensitive jsPsych drops)
    if id_mapping:
        csv_files = [(DATA_RAW_DIR / filename, info['assigned_id'])
                     for filename, info in id_mapping.items()]
        print(f"Using ID mapping: {len(csv_files)} participants")
    else:
        csv_files = [(f, None) for f in sorted(DATA_RAW_DIR.glob('rlwm_trauma_*.csv'))]
        print(f"Found {len(csv_files)} CSV files in {DATA_RAW_DIR}")

    if len(csv_files) == 0:
        # Fail-fast with non-zero exit so SLURM afterok chains halt instead of
        # cascading into 02-04 (which then error-and-exit-1 on missing interim
        # files, masking the real cause). Raw data is PII / gitignored and must
        # be synced manually to the cluster.
        print(
            f"ERROR: No participant CSV files found in {DATA_RAW_DIR}",
            file=sys.stderr,
        )
        print(
            "       Expected pattern: rlwm_trauma_*.csv (+ participant_id_mapping.json)\n"
            "       Raw data is PII and gitignored — sync from local via:\n"
            "         rsync -av --include='rlwm_trauma_*.csv' \\\n"
            "               --include='participant_id_mapping.json' --exclude='*' \\\n"
            "               <local>/data/raw/ <cluster>:<project>/data/raw/",
            file=sys.stderr,
        )
        sys.exit(1)

    # Parse each file
    all_task_trials = []
    all_demographics = []
    all_survey1 = []
    all_survey2 = []
    participant_info = []
    # Corrupted-CSV participants: surfaced to 04_create_summary_csv.py so they
    # appear in summary_participant_metrics.csv with exclusion_reason='corrupted_csv'.
    corrupted_participants: list[dict] = []

    # NOTE: task_trials_long.csv stays filtered to included participants
    # (analysis-ready). Per-participant inclusion flag with exclusion_reason
    # lives in summary_participant_metrics.csv (see docs/CODEBOOK.md).
    print("\nParsing files...")
    for filepath, assigned_id in tqdm(csv_files, desc='Parsing', unit='file'):
        if not filepath.exists():
            continue

        # Skip excluded participants
        if assigned_id in EXCLUDED_PARTICIPANTS:
            continue

        try:
            result = parse_single_file(filepath, assigned_id)
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
            print(f"  Corrupted CSV (skipped): {filepath.name}: {exc}")
            corrupted_participants.append({'sona_id': assigned_id, 'filename': filepath.name})
            continue

        # Only include participants with sufficient trials
        if result['task_trials'] is not None and result['n_trials'] >= MIN_TRIALS:
            all_task_trials.append(result['task_trials'])

            participant_info.append({
                'sona_id': result['sona_id'],
                'filename': filepath.name,
                'n_trials_raw': result['n_trials'],
                'has_survey1': result['survey1'] is not None,
                'has_survey2': result['survey2'] is not None,
                'has_demographics': result['demographics'] is not None
            })

        if result['demographics'] is not None:
            all_demographics.append(result['demographics'])

        if result['survey1'] is not None:
            all_survey1.append(result['survey1'])

        if result['survey2'] is not None:
            all_survey2.append(result['survey2'])

    if corrupted_participants:
        corrupted_df = pd.DataFrame(corrupted_participants)
        corrupted_path = INTERIM_DIR / 'corrupted_participants.csv'
        corrupted_df.to_csv(corrupted_path, index=False)
        print(f"\n  Corrupted CSVs logged: {corrupted_path} ({len(corrupted_participants)} files)")

    print(f"\nSuccessfully parsed: {len(all_task_trials)} participants (>={MIN_TRIALS} trials)")

    # ========================================================================
    # Combine and save task trials
    # ========================================================================
    if all_task_trials:
        print("\n" + "-" * 80)
        print("Processing task data...")

        task_df_raw = pd.concat(all_task_trials, ignore_index=True)

        # Clean data but keep all trials (including practice) for the "all" file
        task_df_all = clean_task_data(task_df_raw, filter_practice=False)

        # Create main-task-only version (exclude practice blocks)
        task_df_main = task_df_all[~task_df_all['is_practice']].copy()

        # Update participant info with cleaned trial counts (main task only)
        for info in participant_info:
            pid = info['sona_id']
            info['n_trials_clean'] = len(task_df_main[task_df_main['sona_id'] == pid])
            info['n_trials_all'] = len(task_df_all[task_df_all['sona_id'] == pid])
            info['status'] = 'complete' if info['n_trials_clean'] >= 700 else 'partial'

        # Save ALL task trials (including practice, with is_practice flag)
        output_path_all = DataParams.TASK_TRIALS_ALL
        task_df_all.to_csv(output_path_all, index=False)
        print(f"[SAVED] {output_path_all}")
        print(f"  {len(task_df_all):,} trials from {task_df_all['sona_id'].nunique()} participants")
        print(f"  (includes {task_df_all['is_practice'].sum():,} practice trials)")

        # Save MAIN TASK ONLY (backwards compatible with existing pipelines)
        output_path_main = DataParams.TASK_TRIALS_LONG
        task_df_main.to_csv(output_path_main, index=False)
        print(f"[SAVED] {output_path_main}")
        print(f"  {len(task_df_main):,} trials from {task_df_main['sona_id'].nunique()} participants")
        print("  (main task only, practice excluded)")

        # Use main task data for summary statistics
        task_df = task_df_main

    # ========================================================================
    # Save participant info
    # ========================================================================
    if participant_info:
        info_df = pd.DataFrame(participant_info)
        info_path = INTERIM_DIR / 'participant_info.csv'
        info_df.to_csv(info_path, index=False)
        print(f"[SAVED] {info_path}")

    # ========================================================================
    # Combine and save Survey 1 (LEC-5) with LESS scores
    # ========================================================================
    if all_survey1:
        print("\n" + "-" * 80)
        print("Processing Survey 1 (LEC-5 → LESS scores)...")

        survey1_df = pd.concat(all_survey1, ignore_index=True)
        survey1_df = score_less(survey1_df)

        output_path = DataParams.PARSED_SURVEY1
        survey1_df.to_csv(output_path, index=False)
        print(f"[SAVED] {output_path}")
        print(f"  {len(survey1_df)} participants with LEC-5 data")

    # ========================================================================
    # Combine and save Survey 2 (IES-R) with subscale scores
    # ========================================================================
    if all_survey2:
        print("\n" + "-" * 80)
        print("Processing Survey 2 (IES-R → subscale scores)...")

        survey2_df = pd.concat(all_survey2, ignore_index=True)
        survey2_df = score_ies_r(survey2_df)

        output_path = DataParams.PARSED_SURVEY2
        survey2_df.to_csv(output_path, index=False)
        print(f"[SAVED] {output_path}")
        print(f"  {len(survey2_df)} participants with IES-R data")

    # ========================================================================
    # Combine and save demographics
    # ========================================================================
    if all_demographics:
        print("\n" + "-" * 80)
        print("Processing demographics...")

        demo_df = pd.concat(all_demographics, ignore_index=True)
        demo_df = demo_df.drop_duplicates(subset=['sona_id'], keep='first')

        output_path = DataParams.PARSED_DEMOGRAPHICS
        demo_df.to_csv(output_path, index=False)
        print(f"[SAVED] {output_path}")
        print(f"  {len(demo_df)} participants with demographics")

    # ========================================================================
    # Create summary metrics file
    # ========================================================================
    if all_survey1 and all_survey2:
        print("\n" + "-" * 80)
        print("Creating summary participant metrics...")

        # Get LESS summary scores
        less_cols = ['sona_id', 'less_total_events', 'less_personal_events']
        less_summary = survey1_df[[c for c in less_cols if c in survey1_df.columns]]

        # Get IES-R summary scores
        ies_cols = ['sona_id', 'ies_total', 'ies_intrusion', 'ies_avoidance', 'ies_hyperarousal']
        ies_summary = survey2_df[[c for c in ies_cols if c in survey2_df.columns]]

        # Merge survey data
        summary_df = less_summary.merge(ies_summary, on='sona_id', how='outer')

        # Add task metrics if available
        if all_task_trials and len(task_df) > 0:
            task_metrics = task_df.groupby('sona_id').agg({
                'correct': ['mean', 'count'],
                'rt': 'mean'
            }).reset_index()
            task_metrics.columns = ['sona_id', 'accuracy_overall', 'n_trials_total', 'mean_rt_overall']
            summary_df = summary_df.merge(task_metrics, on='sona_id', how='left')

        # Save
        output_path = DataParams.SUMMARY_METRICS
        summary_df.to_csv(output_path, index=False)
        print(f"[SAVED] {output_path}")
        print(f"  {len(summary_df)} participants with combined metrics")

    # ========================================================================
    # Print summary statistics
    # ========================================================================
    if all_task_trials:
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)

        n_participants = task_df['sona_id'].nunique()
        trials_per_participant = task_df.groupby('sona_id').size()

        print(f"\nTotal participants: {n_participants}")
        print(f"Total trials: {len(task_df):,}")
        print("\nTrials per participant:")
        print(f"  Mean: {trials_per_participant.mean():.1f}")
        print(f"  Min:  {trials_per_participant.min()}")
        print(f"  Max:  {trials_per_participant.max()}")

        # Complete vs partial
        complete = (trials_per_participant >= 700).sum()
        partial = ((trials_per_participant >= MIN_TRIALS) & (trials_per_participant < 700)).sum()
        print(f"\nComplete participants (>=700 trials): {complete}")
        print(f"Partial participants ({MIN_TRIALS}-699 trials): {partial}")

        # Accuracy summary
        if 'correct' in task_df.columns:
            acc = task_df['correct'].mean()
            print(f"\nOverall accuracy: {acc:.2%}")

            participant_acc = task_df.groupby('sona_id')['correct'].mean()
            print("\nParticipant-level accuracy:")
            print(f"  Mean: {participant_acc.mean():.2%}")
            print(f"  Std:  {participant_acc.std():.2%}")
            print(f"  Min:  {participant_acc.min():.2%}")
            print(f"  Max:  {participant_acc.max():.2%}")

    print("\n" + "=" * 80)
    print("STEP 1 COMPLETE: Raw data parsing finished successfully")
    print("=" * 80)
    print("\nOutputs created (CCDS tiered layout):")
    print(f"  - {DataParams.TASK_TRIALS_ALL}   (ALL trials including practice)")
    print(f"  - {DataParams.TASK_TRIALS_LONG}  (Main task only, for fitting)")
    print(f"  - {INTERIM_DIR / 'participant_info.csv'}")
    print(f"  - {DataParams.PARSED_SURVEY1}  (LEC-5 + LESS scores)")
    print(f"  - {DataParams.PARSED_SURVEY2}  (IES-R + subscale scores)")
    print(f"  - {DataParams.PARSED_DEMOGRAPHICS}")
    print(f"  - {DataParams.SUMMARY_METRICS}")
    print("\nNote: Use task_trials_long_all.csv with --include-practice for")
    print("      fitting models on practice data.")
    print("\nNext step: Run 02_create_collated_csv.py")
    print()


if __name__ == '__main__':
    main()
