"""
Run full MLE fitting on all participants with progress logging.

This script runs Q-learning and WM-RL model fitting on all participants
with progress updates saved to a log file.
"""
import sys
import os
from datetime import datetime
from pathlib import Path

# Create log file
log_path = Path('output/mle_full_fitting_log.txt')
log_file = open(log_path, 'w')

def log(msg):
    """Log to both stdout and file."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    log_file.write(line + '\n')
    log_file.flush()

log("=" * 60)
log("FULL MLE FITTING - Q-Learning Model")
log("=" * 60)

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

log("Loading data...")
data = pd.read_csv('output/task_trials_long_all_participants.csv')

# Create reward column if needed
if 'reward' not in data.columns and 'correct' in data.columns:
    data['reward'] = data['correct'].astype(float)

# Filter to blocks >= 3 (exclude practice)
data = data[data['block'] >= 3].copy()

n_participants = data['sona_id'].nunique()
n_trials = len(data)
log(f"Participants: {n_participants}")
log(f"Total trials: {n_trials:,}")
log(f"Trials per participant: {n_trials // n_participants}")

# Import fitting functions
log("Importing fitting modules...")
from scripts.fitting.fit_mle import fit_all_participants, compute_group_summary

# Run Q-learning fitting
log("")
log("Starting Q-learning MLE fitting...")
log("Parameters: n_starts=10, seed=42")
log("")

start_time = datetime.now()

results_ql = fit_all_participants(
    data=data,
    model='qlearning',
    n_starts=10,  # 10 random starts per participant
    seed=42,
    verbose=True
)

end_time = datetime.now()
duration = (end_time - start_time).total_seconds()

log("")
log(f"Fitting complete in {duration/60:.1f} minutes ({duration:.0f} seconds)")
log(f"Converged: {results_ql['converged'].sum()}/{len(results_ql)}")

# Save results
output_dir = Path('output/mle')
output_dir.mkdir(exist_ok=True)

results_ql.to_csv(output_dir / 'qlearning_individual_fits.csv', index=False)
log(f"Saved: {output_dir / 'qlearning_individual_fits.csv'}")

# Compute and save group summary
summary_ql = compute_group_summary(results_ql, 'qlearning')
summary_ql.to_csv(output_dir / 'qlearning_group_summary.csv', index=False)
log(f"Saved: {output_dir / 'qlearning_group_summary.csv'}")

log("")
log("=== Q-Learning Group Summary ===")
log(summary_ql.to_string())

log("")
log("=" * 60)
log("Q-LEARNING FITTING COMPLETE")
log("=" * 60)

log_file.close()
print(f"\nLog saved to: {log_path}")
