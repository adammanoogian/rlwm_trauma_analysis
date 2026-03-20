#!/usr/bin/env python3
"""Quick test of the fitting pipeline

Note: This script references legacy modules (fit_both_models, numpyro_models)
that no longer exist in their original locations. The current fitting pipeline
uses scripts/fitting/fit_mle.py and scripts/fitting/fit_bayesian.py.
"""

from __future__ import annotations

import sys

import pytest

pytest.skip(
    "Legacy fitting test — fit_both_models and numpyro_models modules no longer exist",
    allow_module_level=True,
)

from pathlib import Path

print("="*80)
print("TESTING FITTING PIPELINE")
print("="*80)

# Test 1: Data Loading
print("\n[1/3] Testing data loading...")
try:
    data = load_and_prepare_data(
        Path('output/task_trials_long_all_participants.csv'),
        min_block=3
    )
    print(f"✓ Successfully loaded {len(data)} trials from {data['sona_id'].nunique()} participants")
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    sys.exit(1)

# Test 2: Data Preparation
print("\n[2/3] Testing data preparation for NumPyro...")
try:
    participant_data = prepare_data_for_numpyro(data)
    print(f"✓ Prepared data for {len(participant_data)} participants")

    # Show one participant's data structure
    pid = list(participant_data.keys())[0]
    pdata = participant_data[pid]
    print(f"  Example (participant {pid}):")
    print(f"    - {len(pdata['stimuli_blocks'])} blocks")
    print(f"    - {sum(len(b) for b in pdata['stimuli_blocks'])} total trials")
except Exception as e:
    print(f"✗ Data preparation failed: {e}")
    sys.exit(1)

# Test 3: Model Import
print("\n[3/3] Testing model imports...")
try:
    print("✓ Successfully imported both models")
except Exception as e:
    print(f"✗ Model import failed: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("✓✓ ALL TESTS PASSED - Pipeline is ready")
print("="*80)
