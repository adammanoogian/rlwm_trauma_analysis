"""
Visualize learning from feedback: Accuracy at trial n+1 after feedback at trial n.

This script analyzes how accuracy at trial n+1 differs based on whether
trial n was correct (positive feedback) or incorrect (negative feedback),
separated by set size.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from config import TaskParams, OUTPUT_DIR, FIGURES_DIR, AnalysisParams


def process_feedback_learning_data(trials_df):
    """
    Process trial data to track accuracy after positive vs negative feedback.
    
    Parameters
    ----------
    trials_df : pd.DataFrame
        Task trials data with columns: sona_id, block, trial_in_block, 
        stimulus, set_size, correct, etc.
    
    Returns
    -------
    pd.DataFrame
        Processed data with feedback tracking and n+1 accuracy
    """
    processed = []
    
    # Filter out practice blocks (block < 3)
    trials_df = trials_df[trials_df['block'] >= 3].copy()
    
    # Process each participant and block
    for (subject_id, block_id), block_data in trials_df.groupby(['sona_id', 'block']):
        block_data = block_data.sort_values('trial_in_block').reset_index(drop=True)
        
        # Track encounters per stimulus
        stimulus_encounter_count = defaultdict(int)
        
        # Need to look ahead one trial
        for idx in range(len(block_data) - 1):
            current_row = block_data.iloc[idx]
            next_row = block_data.iloc[idx + 1]
            
            # Get current trial info
            current_stimulus = int(float(current_row['stimulus'])) - 1
            current_set_size = int(float(current_row['set_size']))
            current_correct = int(float(current_row['correct']))
            
            # Get next trial info
            next_stimulus = int(float(next_row['stimulus'])) - 1
            next_set_size = int(float(next_row['set_size']))
            next_correct = int(float(next_row['correct']))
            
            # Increment encounter count for current stimulus
            stimulus_encounter_count[current_stimulus] += 1
            encounter_num = stimulus_encounter_count[current_stimulus]
            
            # Determine feedback type
            feedback_type = 'positive' if current_correct == 1 else 'negative'
            
            # Record data for analysis
            processed.append({
                'subject_id': subject_id,
                'block': block_id,
                'trial_n': current_row['trial_in_block'],
                'trial_n_plus_1': next_row['trial_in_block'],
                'stimulus_n': current_stimulus,
                'stimulus_n_plus_1': next_stimulus,
                'set_size_n': current_set_size,
                'set_size_n_plus_1': next_set_size,
                'encounter_num': encounter_num,
                'correct_n': current_correct,
                'correct_n_plus_1': next_correct,
                'feedback_type': feedback_type,
                'same_stimulus': current_stimulus == next_stimulus,
                'same_set_size': current_set_size == next_set_size
            })
    
    return pd.DataFrame(processed)


def plot_feedback_learning_by_setsize(
    data_df: pd.DataFrame,
    save_dir: Path,
    filter_same_stimulus: bool = False
):
    """
    Plot accuracy at n+1 after negative feedback, by set size and encounter.
    
    Parameters
    ----------
    data_df : pd.DataFrame
        Processed feedback learning data
    save_dir : Path
        Directory to save figures
    filter_same_stimulus : bool
        If True, only include trials where stimulus_n == stimulus_n_plus_1
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Filter for negative feedback only (incorrect at trial n)
    negative_feedback_df = data_df[data_df['feedback_type'] == 'negative'].copy()
    
    if filter_same_stimulus:
        negative_feedback_df = negative_feedback_df[negative_feedback_df['same_stimulus'] == True]
    
    # Set sizes to analyze
    set_sizes = sorted(negative_feedback_df['set_size_n'].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(set_sizes)))
    
    # LEFT PANEL: Accuracy at n+1 after incorrect n, by encounter
    ax = axes[0]
    
    for set_size, color in zip(set_sizes, colors):
        subset = negative_feedback_df[negative_feedback_df['set_size_n'] == set_size]
        
        # Group by encounter number
        grouped = subset.groupby('encounter_num').agg({
            'correct_n_plus_1': ['mean', 'sem', 'count']
        }).reset_index()
        
        grouped.columns = ['encounter_num', 'mean', 'sem', 'count']
        
        # Filter out encounters with too few observations
        grouped = grouped[grouped['count'] >= 5]
        
        # Plot
        ax.errorbar(
            grouped['encounter_num'],
            grouped['mean'] * 100,
            yerr=grouped['sem'] * 100,
            label=f'Set Size {set_size}',
            marker='o',
            linewidth=2,
            markersize=6,
            capsize=4,
            color=color,
            alpha=0.8
        )
    
    ax.set_xlabel('Encounter Number with Stimulus', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy at Trial n+1 (%)', fontsize=12, fontweight='bold')
    ax.set_title('Recovery After Incorrect Response\n(Accuracy at n+1 | Incorrect at n)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 100])
    ax.axhline(50, color='gray', linestyle=':', alpha=0.5, label='Chance')
    
    # RIGHT PANEL: Comparison of accuracy after positive vs negative feedback
    ax = axes[1]
    
    # Prepare data for comparison
    comparison_data = []
    
    for set_size in set_sizes:
        # After negative feedback
        neg_subset = data_df[(data_df['feedback_type'] == 'negative') & 
                             (data_df['set_size_n'] == set_size)]
        neg_acc = neg_subset['correct_n_plus_1'].mean() * 100
        neg_sem = neg_subset['correct_n_plus_1'].sem() * 100
        
        # After positive feedback
        pos_subset = data_df[(data_df['feedback_type'] == 'positive') & 
                             (data_df['set_size_n'] == set_size)]
        pos_acc = pos_subset['correct_n_plus_1'].mean() * 100
        pos_sem = pos_subset['correct_n_plus_1'].sem() * 100
        
        comparison_data.append({
            'set_size': set_size,
            'negative_acc': neg_acc,
            'negative_sem': neg_sem,
            'positive_acc': pos_acc,
            'positive_sem': pos_sem
        })
    
    comp_df = pd.DataFrame(comparison_data)
    
    x = np.arange(len(set_sizes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, comp_df['negative_acc'], width,
                   yerr=comp_df['negative_sem'], label='After Incorrect (Negative FB)',
                   color='#e74c3c', alpha=0.8, capsize=5)
    bars2 = ax.bar(x + width/2, comp_df['positive_acc'], width,
                   yerr=comp_df['positive_sem'], label='After Correct (Positive FB)',
                   color='#2ecc71', alpha=0.8, capsize=5)
    
    ax.set_xlabel('Set Size at Trial n', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy at Trial n+1 (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy After Feedback Type\n(Overall Comparison)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'SS {ss}' for ss in set_sizes])
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim([0, 100])
    ax.axhline(50, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure
    suffix = '_same_stim' if filter_same_stimulus else '_all_trials'
    save_path = save_dir / f'feedback_learning_by_setsize{suffix}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_feedback_learning_detailed(
    data_df: pd.DataFrame,
    save_dir: Path
):
    """
    Create detailed multi-panel visualization of feedback learning.
    
    Parameters
    ----------
    data_df : pd.DataFrame
        Processed feedback learning data
    save_dir : Path
        Directory to save figures
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    set_sizes = sorted(data_df['set_size_n'].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(set_sizes)))
    
    # PANEL 1: After incorrect, same stimulus
    ax = axes[0, 0]
    subset_incorrect_same = data_df[
        (data_df['feedback_type'] == 'negative') &
        (data_df['same_stimulus'] == True)
    ]
    
    for set_size, color in zip(set_sizes, colors):
        ss_data = subset_incorrect_same[subset_incorrect_same['set_size_n'] == set_size]
        grouped = ss_data.groupby('encounter_num').agg({
            'correct_n_plus_1': ['mean', 'sem', 'count']
        }).reset_index()
        grouped.columns = ['encounter_num', 'mean', 'sem', 'count']
        grouped = grouped[grouped['count'] >= 5]
        
        ax.errorbar(grouped['encounter_num'], grouped['mean'] * 100,
                   yerr=grouped['sem'] * 100, label=f'SS {set_size}',
                   marker='o', linewidth=2, color=color, alpha=0.8, capsize=4)
    
    ax.set_xlabel('Encounter Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy at n+1 (%)', fontsize=11, fontweight='bold')
    ax.set_title('After Incorrect: Same Stimulus Repeated', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 100])
    
    # PANEL 2: After incorrect, different stimulus
    ax = axes[0, 1]
    subset_incorrect_diff = data_df[
        (data_df['feedback_type'] == 'negative') &
        (data_df['same_stimulus'] == False)
    ]
    
    for set_size, color in zip(set_sizes, colors):
        ss_data = subset_incorrect_diff[subset_incorrect_diff['set_size_n'] == set_size]
        grouped = ss_data.groupby('encounter_num').agg({
            'correct_n_plus_1': ['mean', 'sem', 'count']
        }).reset_index()
        grouped.columns = ['encounter_num', 'mean', 'sem', 'count']
        grouped = grouped[grouped['count'] >= 5]
        
        ax.errorbar(grouped['encounter_num'], grouped['mean'] * 100,
                   yerr=grouped['sem'] * 100, label=f'SS {set_size}',
                   marker='s', linewidth=2, color=color, alpha=0.8, capsize=4)
    
    ax.set_xlabel('Encounter Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy at n+1 (%)', fontsize=11, fontweight='bold')
    ax.set_title('After Incorrect: Different Stimulus', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 100])
    
    # PANEL 3: After correct, same stimulus
    ax = axes[1, 0]
    subset_correct_same = data_df[
        (data_df['feedback_type'] == 'positive') &
        (data_df['same_stimulus'] == True)
    ]
    
    for set_size, color in zip(set_sizes, colors):
        ss_data = subset_correct_same[subset_correct_same['set_size_n'] == set_size]
        grouped = ss_data.groupby('encounter_num').agg({
            'correct_n_plus_1': ['mean', 'sem', 'count']
        }).reset_index()
        grouped.columns = ['encounter_num', 'mean', 'sem', 'count']
        grouped = grouped[grouped['count'] >= 5]
        
        ax.errorbar(grouped['encounter_num'], grouped['mean'] * 100,
                   yerr=grouped['sem'] * 100, label=f'SS {set_size}',
                   marker='o', linewidth=2, color=color, alpha=0.8, capsize=4)
    
    ax.set_xlabel('Encounter Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy at n+1 (%)', fontsize=11, fontweight='bold')
    ax.set_title('After Correct: Same Stimulus Repeated', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 100])
    
    # PANEL 4: Summary statistics table-like visualization
    ax = axes[1, 1]
    
    summary_data = []
    for set_size in set_sizes:
        neg_same = data_df[
            (data_df['feedback_type'] == 'negative') &
            (data_df['same_stimulus'] == True) &
            (data_df['set_size_n'] == set_size)
        ]['correct_n_plus_1'].mean() * 100
        
        neg_diff = data_df[
            (data_df['feedback_type'] == 'negative') &
            (data_df['same_stimulus'] == False) &
            (data_df['set_size_n'] == set_size)
        ]['correct_n_plus_1'].mean() * 100
        
        pos_same = data_df[
            (data_df['feedback_type'] == 'positive') &
            (data_df['same_stimulus'] == True) &
            (data_df['set_size_n'] == set_size)
        ]['correct_n_plus_1'].mean() * 100
        
        summary_data.append([set_size, neg_same, neg_diff, pos_same])
    
    summary_df = pd.DataFrame(summary_data, 
                              columns=['Set Size', 'Inc→Same', 'Inc→Diff', 'Corr→Same'])
    
    x = np.arange(len(set_sizes))
    width = 0.25
    
    ax.bar(x - width, summary_df['Inc→Same'], width, label='After Inc: Same Stim',
           color='#e74c3c', alpha=0.8)
    ax.bar(x, summary_df['Inc→Diff'], width, label='After Inc: Diff Stim',
           color='#f39c12', alpha=0.8)
    ax.bar(x + width, summary_df['Corr→Same'], width, label='After Corr: Same Stim',
           color='#2ecc71', alpha=0.8)
    
    ax.set_xlabel('Set Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Accuracy at n+1 (%)', fontsize=11, fontweight='bold')
    ax.set_title('Summary: Feedback Learning by Condition', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{ss}' for ss in set_sizes])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    
    save_path = save_dir / 'feedback_learning_detailed.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualize feedback learning effects in human data'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=str(OUTPUT_DIR / 'task_trials_long_all_participants.csv'),
        help='Path to task trials CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(FIGURES_DIR / 'feedback_learning'),
        help='Directory to save output figures'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.data}")
    task_trials_df = pd.read_csv(args.data)
    print(f"Loaded {len(task_trials_df)} trials from {task_trials_df['sona_id'].nunique()} participants")
    
    # Process data
    print("\nProcessing feedback learning data...")
    feedback_df = process_feedback_learning_data(task_trials_df)
    print(f"Processed {len(feedback_df)} trial pairs (n → n+1)")
    
    # Print summary statistics
    print("\n=== Data Summary ===")
    print(f"Total trial pairs: {len(feedback_df)}")
    print(f"After negative feedback: {len(feedback_df[feedback_df['feedback_type'] == 'negative'])}")
    print(f"After positive feedback: {len(feedback_df[feedback_df['feedback_type'] == 'positive'])}")
    print(f"\nSame stimulus repeated: {feedback_df['same_stimulus'].sum()}")
    print(f"Different stimulus: {(~feedback_df['same_stimulus']).sum()}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\n=== Generating Plots ===")
    
    print("\n1. Main feedback learning by set size (all trials)...")
    plot_feedback_learning_by_setsize(feedback_df, output_dir, filter_same_stimulus=False)
    
    print("\n2. Feedback learning by set size (same stimulus only)...")
    plot_feedback_learning_by_setsize(feedback_df, output_dir, filter_same_stimulus=True)
    
    print("\n3. Detailed multi-panel analysis...")
    plot_feedback_learning_detailed(feedback_df, output_dir)
    
    print(f"\n✓ All plots saved to: {output_dir}")
    print("\nDone!")


if __name__ == '__main__':
    main()
