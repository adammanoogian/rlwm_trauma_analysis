"""Quick MLE fitting test with logging to file."""
import sys
import os
from datetime import datetime

# Create log file
log_file = open('output/mle_test_log.txt', 'w')

def log(msg):
    """Log to both stdout and file."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    log_file.write(line + '\n')
    log_file.flush()

log("Starting small MLE test...")

import pandas as pd
import numpy as np
from pathlib import Path

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

# Get just 3 participants for testing
participants = data['sona_id'].unique()[:3]
log(f"Testing with {len(participants)} participants: {list(participants)}")
data_subset = data[data['sona_id'].isin(participants)]

log(f"Data shape: {data_subset.shape}")
log(f"Trials per participant: {data_subset.groupby('sona_id').size().to_dict()}")

# Now import the fitting functions
log("Importing fitting modules...")
from scripts.fitting.fit_mle import fit_all_participants

log("Starting fitting (3 participants, 10 starts each)...")
results = fit_all_participants(
    data=data_subset,
    model='qlearning',
    n_starts=10,  # Fewer starts for quick test
    seed=42,
    verbose=True
)

log("=== Results ===")
log(results.to_string())

log("=== Summary ===")
log(f"Converged: {results['converged'].sum()}/{len(results)}")
log(f"Mean alpha_pos: {results['alpha_pos'].mean():.3f}")
log(f"Mean alpha_neg: {results['alpha_neg'].mean():.3f}")
log(f"Mean epsilon: {results['epsilon'].mean():.3f}")

log("Test complete!")
log_file.close()
