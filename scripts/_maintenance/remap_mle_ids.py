"""
One-time remapping of MLE participant IDs to assigned_ids.

Problem:
  The MLE fitting was run when 01_parse_raw_data.py used abs(hash(filename)) % 100000
  for anonymous participants. Python's hash() is non-deterministic across sessions,
  so those IDs can't be reproduced. Additionally, 5 SONA IDs (8932, 8944, 8959, 8968,
  9175) appeared directly in the MLE but were reassigned to 10000+ IDs in the current
  pipeline. Only 9187 is a direct match (preserved in the mapping).

Solution:
  Match each old MLE participant to a new assigned_id by re-evaluating the
  stored NLL against each candidate participant's trial data. The correct
  participant reproduces the stored NLL exactly (within floating-point tolerance).

Usage:
    python scripts/utils/remap_mle_ids.py
"""

import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from rlwm.fitting.models.qlearning import (
    prepare_block_data,
    q_learning_multiblock_likelihood,
)
from rlwm.fitting.models.wmrl import wmrl_multiblock_likelihood
from config import (
    EXCLUDED_PARTICIPANTS,
    MODELS_MLE_DIR,
    PROCESSED_DIR,
)

# CCDS-aligned paths (Phase 31: formerly output/, output/mle/)
MLE_DIR = MODELS_MLE_DIR

# Only 9187 appears in both the MLE files AND the trial data with the same ID
DIRECT_MATCH_IDS = {9187}

# NLL tolerance for matching (should be nearly exact)
NLL_TOLERANCE = 0.01


def load_trial_data():
    """Load trial data (with assigned_ids), applying same cleaning as fit_mle.py."""
    path = PROCESSED_DIR / 'task_trials_long_all_participants.csv'
    df = pd.read_csv(path)

    # Exclude participants
    df = df[~df['sona_id'].isin(EXCLUDED_PARTICIPANTS)]

    # Same cleaning as fit_mle.py load_and_prepare_data()
    # 1. Create reward column from correct
    if 'reward' not in df.columns and 'correct' in df.columns:
        df['reward'] = df['correct'].astype(float)

    # 2. Exclude practice blocks (blocks 1-2)
    df = df[df['block'] >= 3].copy()

    # 3. Convert 1-indexed stimuli to 0-indexed (if needed)
    if df['stimulus'].min() >= 1:
        df['stimulus'] = df['stimulus'] - 1

    # 4. Remove invalid trials (key_press < 0)
    df = df[df['key_press'] >= 0].copy()

    return df


def evaluate_qlearning_nll(block_data_pid, alpha_pos, alpha_neg, epsilon):
    """Evaluate Q-learning NLL for one participant's block data."""
    stimuli_blocks = [block_data_pid[b]['stimuli'] for b in sorted(block_data_pid.keys())]
    actions_blocks = [block_data_pid[b]['actions'] for b in sorted(block_data_pid.keys())]
    rewards_blocks = [block_data_pid[b]['rewards'] for b in sorted(block_data_pid.keys())]

    log_lik = q_learning_multiblock_likelihood(
        stimuli_blocks, actions_blocks, rewards_blocks,
        alpha_pos=alpha_pos, alpha_neg=alpha_neg, epsilon=epsilon,
    )
    return -float(log_lik)


def evaluate_wmrl_nll(block_data_pid, alpha_pos, alpha_neg, phi, rho, capacity, epsilon, set_sizes_blocks):
    """Evaluate WM-RL NLL for one participant's block data."""
    stimuli_blocks = [block_data_pid[b]['stimuli'] for b in sorted(block_data_pid.keys())]
    actions_blocks = [block_data_pid[b]['actions'] for b in sorted(block_data_pid.keys())]
    rewards_blocks = [block_data_pid[b]['rewards'] for b in sorted(block_data_pid.keys())]

    log_lik = wmrl_multiblock_likelihood(
        stimuli_blocks, actions_blocks, rewards_blocks, set_sizes_blocks,
        alpha_pos=alpha_pos, alpha_neg=alpha_neg,
        phi=phi, rho=rho, capacity=capacity, epsilon=epsilon,
    )
    return -float(log_lik)


def prepare_set_sizes(trial_df, pid, block_data_pid):
    """Prepare set_sizes_blocks for WM-RL evaluation."""
    import jax.numpy as jnp
    pid_trials = trial_df[trial_df['sona_id'] == pid]
    set_sizes_blocks = []
    for block_num in sorted(block_data_pid.keys()):
        block_trials = pid_trials[pid_trials['block'] == block_num]
        if 'set_size' in block_trials.columns and len(block_trials) > 0:
            set_sizes_blocks.append(jnp.array(block_trials['set_size'].values, dtype=jnp.float32))
        else:
            n_trials = len(block_data_pid[block_num]['stimuli'])
            set_sizes_blocks.append(jnp.ones(n_trials, dtype=jnp.float32) * 3.0)
    return set_sizes_blocks


def remap_model(mle_csv_path, trial_df, block_data, model_type):
    """
    Remap participant IDs in one MLE fits CSV.
    """
    mle_df = pd.read_csv(mle_csv_path)
    print(f"\n{'='*60}")
    print(f"Remapping: {mle_csv_path.name} ({model_type})")
    print(f"{'='*60}")
    print(f"  MLE participants: {len(mle_df)}")

    mle_df['participant_id'] = mle_df['participant_id'].astype(str)

    # Only 9187 is a direct match
    direct_mask = mle_df['participant_id'].apply(
        lambda x: x.isdigit() and int(x) in DIRECT_MATCH_IDS
    )
    rows_to_remap = mle_df[~direct_mask].copy()
    direct_rows = mle_df[direct_mask].copy()

    print(f"  Direct matches: {len(direct_rows)} -> {list(direct_rows['participant_id'])}")
    print(f"  IDs to remap: {len(rows_to_remap)}")

    # Build candidate pools by trial count for faster matching
    trial_counts = trial_df.groupby('sona_id').size().to_dict()
    all_new_pids = set(trial_df['sona_id'].unique())
    already_matched = set(int(x) for x in direct_rows['participant_id'])
    candidate_pids = sorted(all_new_pids - already_matched)
    print(f"  Candidate IDs: {len(candidate_pids)}")

    # Pre-compute set_sizes for WM-RL
    set_sizes_cache = {}
    if model_type == 'wmrl':
        print("  Pre-computing set_sizes for all candidates...")
        for pid in candidate_pids:
            if pid in block_data:
                set_sizes_cache[pid] = prepare_set_sizes(trial_df, pid, block_data[pid])

    id_mapping = {}
    unmatched = []

    print("\n  Matching participants:")
    for idx, row in tqdm(rows_to_remap.iterrows(), total=len(rows_to_remap), desc="  Matching"):
        old_id = row['participant_id']
        stored_nll = row['nll']
        stored_n_trials = int(row['n_trials'])

        best_match = None
        best_diff = float('inf')

        for new_pid in candidate_pids:
            if new_pid in id_mapping.values():
                continue

            # Quick filter: trial counts should be within 5% (allow some variance)
            pid_n_trials = trial_counts.get(new_pid, 0)
            if abs(pid_n_trials - stored_n_trials) > max(10, stored_n_trials * 0.1):
                continue

            pid_block_data = block_data.get(new_pid)
            if pid_block_data is None:
                continue

            try:
                if model_type == 'qlearning':
                    computed_nll = evaluate_qlearning_nll(
                        pid_block_data,
                        alpha_pos=row['alpha_pos'],
                        alpha_neg=row['alpha_neg'],
                        epsilon=row['epsilon'],
                    )
                else:
                    computed_nll = evaluate_wmrl_nll(
                        pid_block_data,
                        alpha_pos=row['alpha_pos'],
                        alpha_neg=row['alpha_neg'],
                        phi=row['phi'],
                        rho=row['rho'],
                        capacity=row['capacity'],
                        epsilon=row['epsilon'],
                        set_sizes_blocks=set_sizes_cache.get(new_pid, []),
                    )

                diff = abs(computed_nll - stored_nll)
                if diff < best_diff:
                    best_diff = diff
                    best_match = new_pid

                # Early exit if we found an exact match
                if diff < NLL_TOLERANCE:
                    break

            except Exception as e:
                continue

        if best_match is not None and best_diff < NLL_TOLERANCE:
            id_mapping[old_id] = best_match
            tqdm.write(f"    + {old_id} -> {best_match} (dNLL={best_diff:.6f})")
        else:
            unmatched.append(old_id)
            diff_str = f"{best_diff:.4f}" if best_match else "N/A"
            tqdm.write(f"    x {old_id} UNMATCHED (best diff: {diff_str})")

    # Apply mapping
    def remap_id(pid_str):
        if pid_str in id_mapping:
            return id_mapping[pid_str]
        if pid_str.isdigit():
            return int(pid_str)
        return pid_str

    mle_df['participant_id'] = mle_df['participant_id'].apply(remap_id)

    mle_df.to_csv(mle_csv_path, index=False)
    print(f"\n  Saved: {mle_csv_path}")
    print(f"  Matched: {len(id_mapping)}/{len(rows_to_remap)}")
    print(f"  Unmatched: {len(unmatched)}")

    return id_mapping, unmatched


def main():
    print("MLE Participant ID Remapping")
    print("=" * 60)

    trial_df = load_trial_data()
    print(f"Trial data: {trial_df['sona_id'].nunique()} participants (after exclusions)")

    print("Preparing block data...")
    block_data = prepare_block_data(trial_df)
    print(f"Block data prepared for {len(block_data)} participants")

    # Remap Q-learning fits
    ql_path = MLE_DIR / 'qlearning_individual_fits.csv'
    ql_mapping, ql_unmatched = {}, []
    if ql_path.exists():
        ql_mapping, ql_unmatched = remap_model(ql_path, trial_df, block_data, 'qlearning')

    # Remap WM-RL fits
    wmrl_path = MLE_DIR / 'wmrl_individual_fits.csv'
    wmrl_mapping, wmrl_unmatched = {}, []
    if wmrl_path.exists():
        wmrl_mapping, wmrl_unmatched = remap_model(wmrl_path, trial_df, block_data, 'wmrl')

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_direct = len(DIRECT_MATCH_IDS)
    print(f"Q-learning: {len(ql_mapping)} remapped + {n_direct} direct = {len(ql_mapping)+n_direct}/49")
    print(f"WM-RL:      {len(wmrl_mapping)} remapped + {n_direct} direct = {len(wmrl_mapping)+n_direct}/49")

    if ql_unmatched or wmrl_unmatched:
        print("\nWARNING: Some participants could not be matched!")
        if ql_unmatched:
            print(f"  Q-learning unmatched: {ql_unmatched}")
        if wmrl_unmatched:
            print(f"  WM-RL unmatched: {wmrl_unmatched}")
    else:
        print("\nAll participants successfully remapped!")


if __name__ == '__main__':
    main()
