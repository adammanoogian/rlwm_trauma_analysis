#!/usr/bin/env python
"""
04_1: Explore Survey Data
=========================

Parses survey data and generates exploratory visualizations.

This CONSOLIDATED script combines:
1. parse_surveys_for_mle.py - Parse raw survey data (LEC-5, IES-R)
2. visualize_scale_distributions.py - Plot trauma scale distributions
3. visualize_scale_correlations.py - Correlation heatmap of scales

Note: This script parses data from raw session files in the external data
directory. Use --parse to re-parse from raw data, or --plots to generate
visualizations from already-parsed data.

Inputs:
    - Raw session CSV files (for parsing)
    - output/summary_participant_metrics.csv (for plots)
    - output/mle/qlearning_individual_fits.csv (optional, for MLE-based parsing)

Outputs:
    - output/mle/participant_surveys.csv
    - output/mle/trauma_group_assignments.csv
    - figures/behavioral_summary/scale_distributions.png
    - figures/behavioral_summary/scale_correlations.png
    - figures/behavioral_summary/performance_distributions.png

Usage:
    # Run all (parse + plots)
    python scripts/04_1_explore_survey_data.py

    # Parse survey data only
    python scripts/04_1_explore_survey_data.py --parse

    # Generate plots only (requires parsed data)
    python scripts/04_1_explore_survey_data.py --plots
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from config import OUTPUT_DIR, FIGURES_DIR, AnalysisParams

# ============================================================================
# PART 1: PARSE SURVEY DATA
# ============================================================================

# Project paths
RAW_DATA_DIR = Path("C:/Users/aman0087/Documents/Github/rlwm_trauma/data")
MLE_OUTPUT_DIR = project_root / "output" / "mle"

# IES-R subscale mappings (1-indexed as per standard scoring)
IESR_INTRUSION = [1, 2, 3, 6, 9, 14, 16, 20]  # 8 items, max 32
IESR_AVOIDANCE = [5, 7, 8, 11, 12, 13, 17, 22]  # 8 items, max 32
IESR_HYPERAROUSAL = [4, 10, 15, 18, 19, 21]  # 6 items, max 24


def parse_lec5_item(response_str: str) -> Dict:
    """
    Parse a single LEC-5 item response.

    Binary indicators for exposure type:
    - (a) = personal
    - (b) = witnessed
    - (c) = learned
    - (d) = job exposure
    - (e) = unsure
    - (f) = doesn't apply (no exposure)
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

    indicators['any_exposure'] = int(any([
        indicators['personal'], indicators['witnessed'],
        indicators['learned'], indicators['job'], indicators['unsure']
    ]))

    return indicators


def parse_lec5_from_json(response: Dict) -> Dict:
    """Parse LEC-5 response from JSON dict with s1_item01-s1_item15 keys."""
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
    """Parse IES-R response from JSON dict with s2_item01-s2_item22 keys."""
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
    """Parse a single raw session CSV file to extract survey data."""
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
    """
    df = df.copy()

    lec_median = df['lec_total'].median()
    ies_median = df['ies_total'].median()

    print(f"\nMedian splits:")
    print(f"  LEC-5 median: {lec_median}")
    print(f"  IES-R total median: {ies_median}")

    df['high_lec'] = df['lec_total'] > lec_median
    df['high_ies'] = df['ies_total'] > ies_median

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

    print(f"\nGroup assignments:")
    for group, count in df['hypothesis_group'].value_counts().sort_index().items():
        print(f"  {group}: {count}")

    return df


def run_parsing():
    """Main function to parse all surveys and create group assignments."""
    print("=" * 70)
    print("Parsing Survey Data for MLE Analysis")
    print("=" * 70)
    print(f"\nRaw data directory: {RAW_DATA_DIR}")
    print(f"Output directory: {MLE_OUTPUT_DIR}")

    MLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load participant info
    participant_info_path = project_root / "output" / "participant_info.csv"
    if not participant_info_path.exists():
        print(f"ERROR: participant_info.csv not found at {participant_info_path}")
        return

    participant_info = pd.read_csv(participant_info_path)
    filename_to_sona = dict(zip(participant_info['filename'], participant_info['sona_id']))
    print(f"\nParticipant info loaded: {len(filename_to_sona)} participants mapped")

    # Get list of participants with MLE fits
    mle_fits_path = MLE_OUTPUT_DIR / "qlearning_individual_fits.csv"
    if not mle_fits_path.exists():
        print(f"WARNING: MLE fits not found at {mle_fits_path}")
        print("Will parse all participants instead")
        fitted_participants = set(str(s) for s in filename_to_sona.values())
    else:
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

        if sona_id not in fitted_participants:
            continue

        filepath = RAW_DATA_DIR / filename
        if not filepath.exists():
            missing_files.append(filename)
            continue

        survey_data = parse_session_file(filepath)

        if survey_data:
            survey_data['sona_id'] = sona_id
            all_surveys.append(survey_data)
            print(f"  Parsed: {sona_id} (LEC={survey_data['lec_total']}, IES={survey_data['ies_total']})")
        else:
            parse_errors.append((filename, sona_id))

    if missing_files:
        print(f"\nWARNING: {len(missing_files)} raw files not found")
    if parse_errors:
        print(f"WARNING: {len(parse_errors)} files could not be parsed")

    if not all_surveys:
        print("ERROR: No survey data extracted!")
        return

    surveys_df = pd.DataFrame(all_surveys)
    print(f"\n{'='*60}")
    print(f"Successfully parsed surveys for {len(surveys_df)} / {len(fitted_participants)} participants")

    # Assign trauma groups
    print(f"\n{'-'*60}")
    print("Assigning trauma groups...")
    surveys_df = assign_trauma_groups(surveys_df)

    # Save outputs
    print(f"\n{'-'*60}")
    print("Saving output files...")

    survey_cols = [
        'sona_id',
        'lec_total', 'lec_personal', 'lec_witnessed', 'lec_learned', 'lec_job',
        'ies_total', 'ies_intrusion', 'ies_avoidance', 'ies_hyperarousal'
    ]
    survey_cols = [c for c in survey_cols if c in surveys_df.columns]

    group_cols = [
        'sona_id', 'hypothesis_group', 'high_lec', 'high_ies',
        'lec_total', 'ies_total'
    ]

    surveys_output = surveys_df[survey_cols].copy()
    surveys_output.to_csv(MLE_OUTPUT_DIR / "participant_surveys.csv", index=False)
    print(f"  Saved: {MLE_OUTPUT_DIR / 'participant_surveys.csv'}")

    groups_output = surveys_df[group_cols].copy()
    groups_output.to_csv(MLE_OUTPUT_DIR / "trauma_group_assignments.csv", index=False)
    print(f"  Saved: {MLE_OUTPUT_DIR / 'trauma_group_assignments.csv'}")

    print(f"\n{'='*60}")
    print("Survey parsing complete!")


# ============================================================================
# PART 2: VISUALIZE SCALE DISTRIBUTIONS
# ============================================================================

def plot_scale_distributions(summary_df, save_dir):
    """Plot distributions of scale metrics."""
    n_participants = len(summary_df)

    lec_cols = ['lec_total_events', 'lec_personal_events', 'lec_sum_exposures']
    ies_cols = ['ies_total', 'ies_intrusion', 'ies_avoidance', 'ies_hyperarousal']

    available_lec = [c for c in lec_cols if c in summary_df.columns]
    available_ies = [c for c in ies_cols if c in summary_df.columns]

    if not available_lec and not available_ies:
        print("  [WARN] No scale metrics found in data")
        return

    n_plots = len(available_lec) + len(available_ies)
    n_cols = min(3, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    plot_idx = 0

    # Plot LEC-5 metrics
    for col in available_lec:
        ax = axes[plot_idx]
        data = summary_df[col].dropna()

        if n_participants == 1:
            ax.bar([col.replace('lec_', '').replace('_', ' ').title()],
                   [data.values[0]], color='steelblue', alpha=0.7)
            ax.set_ylabel('Count/Score', fontweight='bold')
            ax.set_title(f'LEC-5: {col.replace("lec_", "").replace("_", " ").title()}',
                        fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.text(0, data.values[0] + 0.5, f'{data.values[0]:.1f}',
                   ha='center', fontweight='bold')
        else:
            ax.hist(data, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Score', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title(f'LEC-5: {col.replace("lec_", "").replace("_", " ").title()}',
                        fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            mean_val = data.mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.2f}')
            ax.legend()

        plot_idx += 1

    # Plot IES-R metrics
    for col in available_ies:
        ax = axes[plot_idx]
        data = summary_df[col].dropna()

        if n_participants == 1:
            ax.bar([col.replace('ies_', '').title()],
                   [data.values[0]], color='coral', alpha=0.7)
            ax.set_ylabel('Score', fontweight='bold')
            ax.set_title(f'IES-R: {col.replace("ies_", "").title()}', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.text(0, data.values[0] + 0.5, f'{data.values[0]:.1f}',
                   ha='center', fontweight='bold')
        else:
            ax.hist(data, bins=15, color='coral', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Score', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title(f'IES-R: {col.replace("ies_", "").title()}', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            mean_val = data.mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.2f}')
            ax.legend()

        plot_idx += 1

    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Scale Metric Distributions (N={n_participants})', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    save_path = save_dir / 'scale_distributions.png'
    plt.savefig(save_path, dpi=AnalysisParams.FIG_DPI, format=AnalysisParams.FIG_FORMAT, bbox_inches='tight')
    print(f"  [OK] Saved: {save_path}")
    plt.close()


def plot_task_performance_distributions(summary_df, save_dir):
    """Plot distributions of task performance metrics."""
    n_participants = len(summary_df)

    perf_metrics = {
        'Accuracy Overall': 'accuracy_overall',
        'Accuracy Low Load': 'accuracy_low_load',
        'Accuracy High Load': 'accuracy_high_load',
        'Mean RT (ms)': 'mean_rt_overall',
        'Learning Slope': 'learning_slope',
        'Learning Improvement': 'learning_improvement_early_to_late'
    }

    available_metrics = {k: v for k, v in perf_metrics.items() if v in summary_df.columns}

    if not available_metrics:
        print("  [WARN] No performance metrics found")
        return

    n_plots = len(available_metrics)
    n_cols = 3
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (label, col) in enumerate(available_metrics.items()):
        ax = axes[idx]
        data = summary_df[col].dropna()

        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label, fontweight='bold')
            continue

        if n_participants == 1:
            ax.bar([label], [data.values[0]], color='mediumseagreen', alpha=0.7)
            ax.set_ylabel('Value', fontweight='bold')
            ax.set_title(label, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.text(0, data.values[0] + abs(data.values[0])*0.05, f'{data.values[0]:.3f}',
                   ha='center', fontweight='bold')
        else:
            ax.hist(data, bins=15, color='mediumseagreen', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Value', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title(label, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            mean_val = data.mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.3f}')
            ax.legend()

    for idx in range(len(available_metrics), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Task Performance Distributions (N={n_participants})', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    save_path = save_dir / 'performance_distributions.png'
    plt.savefig(save_path, dpi=AnalysisParams.FIG_DPI, format=AnalysisParams.FIG_FORMAT, bbox_inches='tight')
    print(f"  [OK] Saved: {save_path}")
    plt.close()


# ============================================================================
# PART 3: VISUALIZE SCALE CORRELATIONS
# ============================================================================

def plot_scale_correlations(summary_df, save_dir):
    """Plot correlation matrix of scale metrics."""
    n_participants = len(summary_df)

    if n_participants < 2:
        print("  [WARN] Need at least 2 participants to compute correlations")
        print("          Creating template visualization...")

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.text(0.5, 0.5,
               f'Correlation Matrix\n\n'
               f'Current: N={n_participants} participant\n'
               f'Required: N>=2 participants\n\n'
               f'This visualization will show correlations between:\n'
               f'- LEC-5 trauma exposure metrics\n'
               f'- IES-R PTSD symptom scores\n'
               f'- Task performance measures\n\n'
               f'Add more participants to enable correlation analysis',
               ha='center', va='center', fontsize=12, transform=ax.transAxes)

        ax.axis('off')
        plt.title('Scale & Performance Correlation Matrix (Template)', fontsize=14, fontweight='bold')

        save_path = save_dir / 'scale_correlations.png'
        plt.savefig(save_path, dpi=AnalysisParams.FIG_DPI, format=AnalysisParams.FIG_FORMAT, bbox_inches='tight')
        print(f"  [OK] Saved template: {save_path}")
        plt.close()
        return

    scale_cols = []

    lec_cols = ['lec_personal_events', 'lec_sum_exposures']
    scale_cols.extend([c for c in lec_cols if c in summary_df.columns])

    ies_cols = ['ies_intrusion', 'ies_avoidance', 'ies_hyperarousal']
    scale_cols.extend([c for c in ies_cols if c in summary_df.columns])

    available_cols = [c for c in scale_cols if c in summary_df.columns and summary_df[c].notna().sum() >= 2]

    if len(available_cols) < 2:
        print("  [WARN] Insufficient data for correlation matrix")
        return

    corr_data = summary_df[available_cols].dropna()
    corr_matrix = corr_data.corr()

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
        vmin=-1,
        vmax=1,
        ax=ax,
        annot_kws={'size': 16}
    )

    labels = [c.replace('_', ' ').replace('lec ', 'LEC-5 ').replace('ies ', 'IES-R ').title()
             for c in available_cols]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=14)
    ax.set_yticklabels(labels, rotation=0, fontsize=14)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Correlation Coefficient', fontsize=14)

    plt.title(f'Subscale Correlation Matrix (N={n_participants})', fontsize=20, fontweight='bold', pad=20)
    plt.tight_layout()

    save_path = save_dir / 'scale_correlations.png'
    plt.savefig(save_path, dpi=AnalysisParams.FIG_DPI, format=AnalysisParams.FIG_FORMAT, bbox_inches='tight')
    print(f"  [OK] Saved: {save_path}")
    plt.close()


def run_visualizations(data_path=None):
    """Generate all visualizations."""
    print("=" * 80)
    print("SCALE DISTRIBUTION AND CORRELATION VISUALIZATION")
    print("=" * 80)

    if data_path is None:
        data_path = project_root / 'output' / 'summary_participant_metrics.csv'

    figure_dir = FIGURES_DIR / 'behavioral_summary'
    figure_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading summary metrics: {data_path}")
    summary_df = pd.read_csv(data_path)
    print(f"  Loaded {len(summary_df)} participants")

    if len(summary_df) == 1:
        print("\n  [NOTE] Only 1 participant - visualizations will show individual values")
        print("          Distributions will be more informative with additional participants")

    print(f"\nCreating visualizations...")

    plot_scale_distributions(summary_df, figure_dir)
    plot_task_performance_distributions(summary_df, figure_dir)
    plot_scale_correlations(summary_df, figure_dir)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nFigures saved to: {figure_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Explore survey data (parse + visualize)'
    )
    parser.add_argument('--parse', action='store_true',
                        help='Parse survey data from raw files')
    parser.add_argument('--plots', action='store_true',
                        help='Generate visualizations only')
    parser.add_argument('--data', type=str,
                        default='output/summary_participant_metrics.csv',
                        help='Path to summary metrics data (for plots)')

    args = parser.parse_args()

    # If neither flag specified, run both
    run_parse = args.parse or (not args.parse and not args.plots)
    run_plots = args.plots or (not args.parse and not args.plots)

    print("=" * 80)
    print("SURVEY DATA EXPLORATION")
    print("=" * 80)

    if run_parse:
        run_parsing()

    if run_plots:
        run_visualizations(project_root / args.data)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  - Review scale distributions for outliers")
    print("  - Check correlations for expected patterns")
    print("  - Run 05_summarize_behavioral_data.py for behavioral summary")


if __name__ == '__main__':
    main()
