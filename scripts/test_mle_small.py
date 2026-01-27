"""Quick MLE fitting test on a few participants."""
import sys
import os

# Ensure output is unbuffered
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

print("Starting small MLE test...", flush=True)

import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("Loading data...", flush=True)
data = pd.read_csv('output/task_trials_long_all_participants.csv')

# Create reward column if needed
if 'reward' not in data.columns and 'correct' in data.columns:
    data['reward'] = data['correct'].astype(float)

# Filter to blocks >= 3 (exclude practice)
data = data[data['block'] >= 3].copy()

# Get just 3 participants for testing
participants = data['sona_id'].unique()[:3]
print(f"Testing with {len(participants)} participants: {list(participants)}", flush=True)
data_subset = data[data['sona_id'].isin(participants)]

print(f"Data shape: {data_subset.shape}", flush=True)
print(f"Trials per participant: {data_subset.groupby('sona_id').size().to_dict()}", flush=True)

# Now import the fitting functions
print("\nImporting fitting modules...", flush=True)
from scripts.fitting.fit_mle import fit_all_participants

print("\nStarting fitting (3 participants, 10 starts each)...", flush=True)
results = fit_all_participants(
    data=data_subset,
    model='qlearning',
    n_starts=10,  # Fewer starts for quick test
    seed=42,
    verbose=True
)

print("\n=== Results ===", flush=True)
print(results.to_string(), flush=True)

print("\n=== Summary ===", flush=True)
print(f"Converged: {results['converged'].sum()}/{len(results)}", flush=True)
print(f"Mean alpha_pos: {results['alpha_pos'].mean():.3f}", flush=True)
print(f"Mean alpha_neg: {results['alpha_neg'].mean():.3f}", flush=True)
print(f"Mean epsilon: {results['epsilon'].mean():.3f}", flush=True)

print("\nTest complete!", flush=True)
