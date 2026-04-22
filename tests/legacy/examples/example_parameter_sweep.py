"""
Example: Tiny Parameter Sweep

Demonstrates how to run a small parameter sweep to test model behavior
across different parameter values and task conditions.

This is a minimal, easy-to-understand example that you can modify.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from rlwm.envs.rlwm_env import create_rlwm_env
from rlwm.models.q_learning import QLearningAgent
from rlwm.models.wm_rl_hybrid import WMRLHybridAgent
from scripts.legacy.simulations.unified_simulator import simulate_agent_fixed

print("=" * 80)
print("TINY PARAMETER SWEEP EXAMPLE")
print("=" * 80)
print()

# ============================================================================
# DEFINE PARAMETER GRID (tiny version)
# ============================================================================

print("Setting up parameter grid...")
print()

# Q-Learning parameters to test
qlearning_grid = {
    'alpha_pos': [0.1, 0.3, 0.5],  # 3 values
    'alpha_neg': [0.05, 0.1],      # 2 values
    'beta': [1.0, 3.0],            # 2 values
}

# WM-RL parameters to test
wmrl_grid = {
    'alpha_pos': [0.3],            # 1 value
    'alpha_neg': [0.1],            # 1 value
    'beta': [2.0],                 # 1 value
    'beta_wm': [3.0],              # 1 value
    'capacity': [3, 4, 5],         # 3 values
    'phi': [0.1],                  # 1 value
    'rho': [0.5, 0.7],             # 2 values
}

# Task conditions
set_sizes = [3, 5]  # Test 2 set sizes
num_trials = 50     # Short trials for speed
num_reps = 3        # Few reps for speed

print(f"Q-Learning grid size: {len(qlearning_grid['alpha_pos']) * len(qlearning_grid['alpha_neg']) * len(qlearning_grid['beta'])} parameter combinations")
print(f"WM-RL grid size: {len(wmrl_grid['capacity']) * len(wmrl_grid['rho'])} parameter combinations")
print(f"Set sizes: {set_sizes}")
print(f"Trials per condition: {num_trials}")
print(f"Reps per condition: {num_reps}")
print()

# ============================================================================
# RUN Q-LEARNING SWEEP
# ============================================================================

print("-" * 80)
print("RUNNING Q-LEARNING SWEEP")
print("-" * 80)
print()

qlearning_results = []

for alpha_pos in qlearning_grid['alpha_pos']:
    for alpha_neg in qlearning_grid['alpha_neg']:
        for beta in qlearning_grid['beta']:
            for set_size in set_sizes:

                # Print current condition
                print(f"Q-Learning: α+={alpha_pos:.2f}, α-={alpha_neg:.2f}, β={beta:.1f}, set_size={set_size}")

                # Run multiple repetitions
                accuracies = []
                for rep in range(num_reps):
                    # Create environment
                    env = create_rlwm_env(
                        set_size=set_size,
                        phase_type='main_task',
                        max_trials_per_block=num_trials,
                        seed=42 + rep
                    )

                    # Define parameters
                    params = {
                        'num_stimuli': 6,
                        'num_actions': 3,
                        'alpha_pos': alpha_pos,
                        'alpha_neg': alpha_neg,
                        'beta': beta,
                        'gamma': 0.0,
                        'q_init': 0.5
                    }

                    # Run simulation
                    result = simulate_agent_fixed(
                        agent_class=QLearningAgent,
                        params=params,
                        env=env,
                        num_trials=num_trials,
                        seed=42 + rep
                    )

                    accuracies.append(result.accuracy)

                # Store results
                qlearning_results.append({
                    'model': 'qlearning',
                    'alpha_pos': alpha_pos,
                    'alpha_neg': alpha_neg,
                    'beta': beta,
                    'set_size': set_size,
                    'accuracy_mean': np.mean(accuracies),
                    'accuracy_std': np.std(accuracies),
                    'num_trials': num_trials,
                    'num_reps': num_reps
                })

                print(f"  → Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")

print()

# ============================================================================
# RUN WM-RL SWEEP
# ============================================================================

print("-" * 80)
print("RUNNING WM-RL SWEEP")
print("-" * 80)
print()

wmrl_results = []

for capacity in wmrl_grid['capacity']:
    for rho in wmrl_grid['rho']:
        for set_size in set_sizes:

            print(f"WM-RL: K={capacity}, ρ={rho:.1f}, set_size={set_size}")

            accuracies = []
            for rep in range(num_reps):
                env = create_rlwm_env(
                    set_size=set_size,
                    phase_type='main_task',
                    max_trials_per_block=num_trials,
                    seed=42 + rep
                )

                params = {
                    'num_stimuli': 6,
                    'num_actions': 3,
                    'alpha_pos': wmrl_grid['alpha_pos'][0],
                    'alpha_neg': wmrl_grid['alpha_neg'][0],
                    'beta': wmrl_grid['beta'][0],
                    'beta_wm': wmrl_grid['beta_wm'][0],
                    'capacity': capacity,
                    'phi': wmrl_grid['phi'][0],
                    'rho': rho,
                    'gamma': 0.0,
                    'q_init': 0.5,
                    'wm_init': 0.0
                }

                result = simulate_agent_fixed(
                    agent_class=WMRLHybridAgent,
                    params=params,
                    env=env,
                    num_trials=num_trials,
                    seed=42 + rep
                )

                accuracies.append(result.accuracy)

            wmrl_results.append({
                'model': 'wmrl',
                'capacity': capacity,
                'rho': rho,
                'set_size': set_size,
                'accuracy_mean': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'num_trials': num_trials,
                'num_reps': num_reps
            })

            print(f"  → Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")

print()

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("-" * 80)
print("SAVING RESULTS")
print("-" * 80)
print()

# Create output directory
output_dir = project_root / 'output' / 'parameter_sweeps'
output_dir.mkdir(parents=True, exist_ok=True)

# Save Q-learning results
qlearning_df = pd.DataFrame(qlearning_results)
qlearning_file = output_dir / 'tiny_sweep_qlearning.csv'
qlearning_df.to_csv(qlearning_file, index=False)
print(f"Saved Q-learning results: {qlearning_file}")

# Save WM-RL results
wmrl_df = pd.DataFrame(wmrl_results)
wmrl_file = output_dir / 'tiny_sweep_wmrl.csv'
wmrl_df.to_csv(wmrl_file, index=False)
print(f"Saved WM-RL results: {wmrl_file}")

print()

# ============================================================================
# DISPLAY SUMMARY
# ============================================================================

print("-" * 80)
print("SUMMARY")
print("-" * 80)
print()

print("Q-LEARNING RESULTS:")
print(qlearning_df.to_string(index=False))
print()

print("WM-RL RESULTS:")
print(wmrl_df.to_string(index=False))
print()

# Find best parameters
print("BEST PARAMETERS:")
print()

# Q-Learning: Best for each set size
for ss in set_sizes:
    ss_data = qlearning_df[qlearning_df['set_size'] == ss]
    best_idx = ss_data['accuracy_mean'].idxmax()
    best_row = ss_data.loc[best_idx]
    print(f"  Q-Learning (set_size={ss}): α+={best_row['alpha_pos']:.2f}, α-={best_row['alpha_neg']:.2f}, β={best_row['beta']:.1f} → acc={best_row['accuracy_mean']:.3f}")

print()

# WM-RL: Best for each set size
for ss in set_sizes:
    ss_data = wmrl_df[wmrl_df['set_size'] == ss]
    best_idx = ss_data['accuracy_mean'].idxmax()
    best_row = ss_data.loc[best_idx]
    print(f"  WM-RL (set_size={ss}): K={best_row['capacity']}, ρ={best_row['rho']:.1f} → acc={best_row['accuracy_mean']:.3f}")

print()
print("=" * 80)
print("COMPLETE!")
print("=" * 80)
