"""
Run WM-RL MLE fitting on all participants.
"""
import sys
import os
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
log("WM-RL MLE FITTING")
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

# Run WM-RL fitting
log("")
log("Starting WM-RL MLE fitting...")
log("Parameters: n_starts=10, seed=42")
log("WM-RL has 6 parameters: alpha_pos, alpha_neg, phi, rho, capacity, epsilon")
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
log(f"Fitting complete in {duration/60:.1f} minutes ({duration/3600:.2f} hours)")
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
log("Next step: Run model comparison with:")
log("  python scripts/fitting/compare_mle_models.py \\")
log("    --qlearning output/mle/qlearning_individual_fits.csv \\")
log("    --wmrl output/mle/wmrl_individual_fits.csv")

log_file.close()
print(f"\nLog saved to: {log_path}")
