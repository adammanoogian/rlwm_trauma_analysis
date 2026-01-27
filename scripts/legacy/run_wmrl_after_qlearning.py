"""
Monitor Q-learning fitting and run WM-RL when complete.

This script:
1. Monitors the Q-learning log file for completion
2. When Q-learning finishes, starts WM-RL fitting
3. Saves all results to output/mle/
"""
import sys
import os
import time
from datetime import datetime
from pathlib import Path

# Create log file
log_path = Path('output/mle_wmrl_fitting_log.txt')
log_file = open(log_path, 'w')

def log(msg):
    """Log to both stdout and file."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    log_file.write(line + '\n')
    log_file.flush()

log("=" * 60)
log("WM-RL FITTING MONITOR")
log("=" * 60)
log("Waiting for Q-learning fitting to complete...")
log("")

# Monitor Q-learning completion
ql_log_path = Path('output/mle_full_fitting_log.txt')
ql_results_path = Path('output/mle/qlearning_individual_fits.csv')

check_interval = 300  # Check every 5 minutes
max_wait_hours = 8    # Maximum wait time

start_wait = datetime.now()
check_count = 0

while True:
    check_count += 1

    # Check if Q-learning results file has been updated recently
    if ql_results_path.exists():
        # Check if file was modified in the last 10 minutes (indicating completion)
        mtime = datetime.fromtimestamp(ql_results_path.stat().st_mtime)
        age_minutes = (datetime.now() - mtime).total_seconds() / 60

        # Also check the log file for "COMPLETE" message
        if ql_log_path.exists():
            with open(ql_log_path, 'r') as f:
                log_content = f.read()
                if 'FITTING COMPLETE' in log_content or 'Q-LEARNING FITTING COMPLETE' in log_content:
                    log(f"Q-learning fitting completed!")
                    log(f"Results file age: {age_minutes:.1f} minutes")
                    break

    # Check timeout
    wait_hours = (datetime.now() - start_wait).total_seconds() / 3600
    if wait_hours > max_wait_hours:
        log(f"WARNING: Max wait time ({max_wait_hours} hours) exceeded")
        log("Starting WM-RL anyway...")
        break

    log(f"Check #{check_count}: Q-learning still running... (waited {wait_hours:.1f} hours)")
    time.sleep(check_interval)

log("")
log("=" * 60)
log("STARTING WM-RL FITTING")
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

# Import fitting functions
log("Importing fitting modules...")
from scripts.fitting.fit_mle import fit_all_participants, compute_group_summary

# Run WM-RL fitting
log("")
log("Starting WM-RL MLE fitting...")
log("Parameters: n_starts=10, seed=42")
log("Note: WM-RL has 6 parameters (vs 3 for Q-learning), so takes longer")
log("")

start_time = datetime.now()

results_wmrl = fit_all_participants(
    data=data,
    model='wmrl',
    n_starts=10,
    seed=42,
    verbose=True
)

end_time = datetime.now()
duration = (end_time - start_time).total_seconds()

log("")
log(f"WM-RL fitting complete in {duration/60:.1f} minutes ({duration/3600:.1f} hours)")
log(f"Converged: {results_wmrl['converged'].sum()}/{len(results_wmrl)}")

# Save results
output_dir = Path('output/mle')
output_dir.mkdir(exist_ok=True)

results_wmrl.to_csv(output_dir / 'wmrl_individual_fits.csv', index=False)
log(f"Saved: {output_dir / 'wmrl_individual_fits.csv'}")

# Compute and save group summary
summary_wmrl = compute_group_summary(results_wmrl, 'wmrl')
summary_wmrl.to_csv(output_dir / 'wmrl_group_summary.csv', index=False)
log(f"Saved: {output_dir / 'wmrl_group_summary.csv'}")

log("")
log("=== WM-RL Group Summary ===")
log(summary_wmrl.to_string())

log("")
log("=" * 60)
log("WM-RL FITTING COMPLETE")
log("=" * 60)
log("")
log("Both models have been fitted!")
log("Results saved to output/mle/")
log("  - qlearning_individual_fits.csv")
log("  - qlearning_group_summary.csv")
log("  - wmrl_individual_fits.csv")
log("  - wmrl_group_summary.csv")
log("")
log("Next step: Run model comparison with:")
log("  python scripts/fitting/compare_mle_models.py --qlearning output/mle/qlearning_individual_fits.csv --wmrl output/mle/wmrl_individual_fits.csv")

log_file.close()
print(f"\nLog saved to: {log_path}")
