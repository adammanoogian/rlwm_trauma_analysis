#!/usr/bin/env python
"""Wrapper to run MLE fitting with proper output handling."""
import sys
import subprocess
import os

# Ensure output is unbuffered
os.environ['PYTHONUNBUFFERED'] = '1'

# Print immediately
print("Starting MLE fitting wrapper...", flush=True)
print(f"Working directory: {os.getcwd()}", flush=True)

# Run the actual fitting
cmd = [
    sys.executable,
    'scripts/fitting/fit_mle.py',
    '--model', 'qlearning',
    '--data', 'output/task_trials_long_all_participants.csv',
    '--output', 'output/mle/',
    '--n-starts', '20',
    '--seed', '42'
]

print(f"Running command: {' '.join(cmd)}", flush=True)
print("=" * 60, flush=True)

result = subprocess.run(cmd, capture_output=False, text=True)

print("=" * 60, flush=True)
print(f"Exit code: {result.returncode}", flush=True)
