"""
Compute Derived Behavioral Metrics for N=47 Matched Dataset

Computes from trial-level data:
1. Set-size effects (accuracy & RT)
2. Learning slope (accuracy improvement across blocks)
3. Feedback sensitivity (win-stay / lose-shift)
4. Perseveration index (choice repetition regardless of feedback)

Adds these to behavioral_summary_matched.csv for parameter-behavior analyses.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import linregress

# Configuration
TRIAL_DATA = "output/task_trials_long_all_participants.csv"
BEHAVIORAL_MATCHED = "output/mle/behavioral_summary_matched.csv"
OUTPUT_FILE = "output/mle/behavioral_summary_matched_with_metrics.csv"

print("="*70)
print("COMPUTING DERIVED BEHAVIORAL METRICS")
print("="*70)

# Load data
print("\nLoading trial-level data...")
trials_df = pd.read_csv(TRIAL_DATA)

# Rename columns to match
if 'sona_id' in trials_df.columns:
    trials_df['participant_id'] = trials_df['sona_id']
if 'key_press' in trials_df.columns:
    trials_df['choice'] = trials_df['key_press']
if 'trial_in_experiment' in trials_df.columns:
    trials_df['trial'] = trials_df['trial_in_experiment']

print(f"  Trial data: {len(trials_df)} trials, {trials_df['participant_id'].nunique()} participants")

behavioral_df = pd.read_csv(BEHAVIORAL_MATCHED)
print(f"  Behavioral matched: {len(behavioral_df)} participants")

# Get matched participant IDs
matched_ids = behavioral_df['participant_id'].unique()
print(f"\nFiltering trials to N={len(matched_ids)} matched participants")

# Filter trials to matched participants only
trials_matched = trials_df[trials_df['participant_id'].isin(matched_ids)].copy()
print(f"  Filtered trials: {len(trials_matched)} trials")

# ============================================================================
# 1. SET-SIZE EFFECTS (simple subtraction)
# ============================================================================
print("\n" + "="*70)
print("1. COMPUTING SET-SIZE EFFECTS")
print("="*70)

behavioral_df['set_size_effect_accuracy'] = behavioral_df['accuracy_low'] - behavioral_df['accuracy_high']
behavioral_df['set_size_effect_rt'] = behavioral_df['rt_high'] - behavioral_df['rt_low']

print(f"\nSet-size effect accuracy: M = {behavioral_df['set_size_effect_accuracy'].mean():.3f}")
print(f"Set-size effect RT: M = {behavioral_df['set_size_effect_rt'].mean():.1f} ms")

# ============================================================================
# 2. LEARNING SLOPE (accuracy improvement across blocks)
# ============================================================================
print("\n" + "="*70)
print("2. COMPUTING LEARNING SLOPE")
print("="*70)

def compute_learning_slope(participant_trials):
    """Compute slope of accuracy improvement across blocks."""
    # Group by block and compute accuracy
    block_accuracy = participant_trials.groupby('block')['correct'].mean().reset_index()
    block_accuracy.columns = ['block', 'accuracy']
    
    # Fit linear regression
    if len(block_accuracy) < 3:  # Need at least 3 blocks
        return np.nan
    
    slope, intercept, r_value, p_value, std_err = linregress(
        block_accuracy['block'], 
        block_accuracy['accuracy']
    )
    
    return slope

learning_slopes = []
for participant_id in matched_ids:
    p_trials = trials_matched[trials_matched['participant_id'] == participant_id]
    slope = compute_learning_slope(p_trials)
    learning_slopes.append({'participant_id': participant_id, 'learning_slope': slope})

learning_df = pd.DataFrame(learning_slopes)
behavioral_df = behavioral_df.merge(learning_df, on='participant_id', how='left')

print(f"\nLearning slope: M = {behavioral_df['learning_slope'].mean():.4f}")
print(f"  (Positive = improvement, Negative = decline)")

# ============================================================================
# 3. FEEDBACK SENSITIVITY (win-stay / lose-shift)
# ============================================================================
print("\n" + "="*70)
print("3. COMPUTING FEEDBACK SENSITIVITY")
print("="*70)

def compute_feedback_sensitivity(participant_trials):
    """Compute win-stay and lose-shift rates."""
    # Sort by trial number to ensure temporal order
    p_trials = participant_trials.sort_values('trial').copy()
    
    # Get current and next trial info
    p_trials['next_choice'] = p_trials['choice'].shift(-1)
    p_trials['next_stimulus'] = p_trials['stimulus'].shift(-1)
    
    # Only look at trials where stimulus repeats (can compare choice)
    same_stimulus = p_trials['stimulus'] == p_trials['next_stimulus']
    p_trials_repeat = p_trials[same_stimulus].copy()
    
    if len(p_trials_repeat) < 10:  # Need sufficient data
        return np.nan
    
    # Did they repeat the same choice?
    p_trials_repeat['repeated_choice'] = (
        p_trials_repeat['choice'] == p_trials_repeat['next_choice']
    )
    
    # Win-stay: after correct, did they stay?
    win_trials = p_trials_repeat[p_trials_repeat['correct'] == 1]
    win_stay = win_trials['repeated_choice'].mean() if len(win_trials) > 0 else np.nan
    
    # Lose-shift: after incorrect, did they shift?
    lose_trials = p_trials_repeat[p_trials_repeat['correct'] == 0]
    lose_shift = (1 - lose_trials['repeated_choice'].mean()) if len(lose_trials) > 0 else np.nan
    
    # Feedback sensitivity = (win-stay + lose-shift) / 2
    if pd.notna(win_stay) and pd.notna(lose_shift):
        return (win_stay + lose_shift) / 2
    else:
        return np.nan

feedback_sensitivity_list = []
for participant_id in matched_ids:
    p_trials = trials_matched[trials_matched['participant_id'] == participant_id]
    sensitivity = compute_feedback_sensitivity(p_trials)
    feedback_sensitivity_list.append({
        'participant_id': participant_id, 
        'feedback_sensitivity': sensitivity
    })

feedback_df = pd.DataFrame(feedback_sensitivity_list)
behavioral_df = behavioral_df.merge(feedback_df, on='participant_id', how='left')

print(f"\nFeedback sensitivity: M = {behavioral_df['feedback_sensitivity'].mean():.3f}")
print(f"  (Range: 0-1, higher = more sensitive to feedback)")

# ============================================================================
# 4. PERSEVERATION INDEX (choice repetition regardless of feedback)
# ============================================================================
print("\n" + "="*70)
print("4. COMPUTING PERSEVERATION INDEX")
print("="*70)

def compute_perseveration(participant_trials):
    """Compute tendency to repeat choices regardless of feedback."""
    # Sort by trial number
    p_trials = participant_trials.sort_values('trial').copy()
    
    # Get previous choice
    p_trials['prev_choice'] = p_trials['choice'].shift(1)
    p_trials['prev_stimulus'] = p_trials['stimulus'].shift(1)
    p_trials['prev_correct'] = p_trials['correct'].shift(1)
    
    # Only look at trials where stimulus repeats
    same_stimulus = p_trials['stimulus'] == p_trials['prev_stimulus']
    p_trials_repeat = p_trials[same_stimulus].copy()
    
    if len(p_trials_repeat) < 10:
        return np.nan
    
    # Did they repeat choice?
    p_trials_repeat['repeated'] = (
        p_trials_repeat['choice'] == p_trials_repeat['prev_choice']
    )
    
    # Perseveration = repetition rate AFTER INCORRECT feedback
    # (High perseveration = keep doing same thing even when wrong)
    incorrect_trials = p_trials_repeat[p_trials_repeat['prev_correct'] == 0]
    
    if len(incorrect_trials) < 5:
        return np.nan
    
    perseveration_rate = incorrect_trials['repeated'].mean()
    
    return perseveration_rate

perseveration_list = []
for participant_id in matched_ids:
    p_trials = trials_matched[trials_matched['participant_id'] == participant_id]
    perseveration = compute_perseveration(p_trials)
    perseveration_list.append({
        'participant_id': participant_id,
        'perseveration_index': perseveration
    })

perseveration_df = pd.DataFrame(perseveration_list)
behavioral_df = behavioral_df.merge(perseveration_df, on='participant_id', how='left')

print(f"\nPerseveration index: M = {behavioral_df['perseveration_index'].mean():.3f}")
print(f"  (Higher = more repetition after errors)")

# ============================================================================
# SAVE UPDATED FILE
# ============================================================================
print("\n" + "="*70)
print("SAVING UPDATED BEHAVIORAL FILE")
print("="*70)

behavioral_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✓ Saved: {OUTPUT_FILE}")

# Summary
print("\n" + "="*70)
print("SUMMARY OF DERIVED METRICS")
print("="*70)

metrics = [
    'set_size_effect_accuracy',
    'set_size_effect_rt', 
    'learning_slope',
    'feedback_sensitivity',
    'perseveration_index'
]

print(f"\n{'Metric':<30} {'Mean':<10} {'SD':<10} {'N valid':<10}")
print("-"*70)
for metric in metrics:
    values = behavioral_df[metric].dropna()
    print(f"{metric:<30} {values.mean():<10.3f} {values.std():<10.3f} {len(values):<10}")

print("\n" + "="*70)
print("NEXT STEP")
print("="*70)
print("\nUpdate analysis pipeline to use:")
print(f"  BEHAVIORAL_SUMMARY = '{OUTPUT_FILE}'")
print("\nThen rerun: python scripts/analysis/analysis_modelling_base_models.py")
print("="*70 + "\n")
