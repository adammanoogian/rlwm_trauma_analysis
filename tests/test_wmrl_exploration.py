"""
Quick test script to debug WM-RL parameter exploration issues.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rlwm.envs.rlwm_env import create_rlwm_env
from rlwm.models.wm_rl_hybrid import WMRLHybridAgent
from scripts.simulations.unified_simulator import simulate_agent_fixed

print("=" * 80)
print("TESTING WM-RL PARAMETER EXPLORATION")
print("=" * 80)
print()

# Sample WM-RL parameters (similar to prior sampling)
np.random.seed(42)
params_sample = {
    'alpha_pos': 0.5,
    'alpha_neg': 0.2,
    'beta': 2.0,
    'beta_wm': 3.0,
    'capacity': 4,
    'phi': 0.1,
    'rho': 0.7,
}

print("Test parameters:")
for k, v in params_sample.items():
    print(f"  {k} = {v}")
print()

# Create environment
print("Creating environment...")
env = create_rlwm_env(
    set_size=3,
    phase_type='main_task',
    max_trials_per_block=20,
    seed=42
)
print("  ✓ Environment created")
print()

# Prepare full parameter dict
print("Preparing parameters for agent...")
params = {
    'num_stimuli': 6,
    'num_actions': 3,
    'gamma': 0.0,
    'q_init': 0.5,
    'wm_init': 0.0,
}
params.update(params_sample)

print("Full parameter dict:")
for k, v in params.items():
    print(f"  {k} = {v}")
print()

# Try creating agent directly
print("Test 1: Creating WM-RL agent directly...")
try:
    agent = WMRLHybridAgent(**params, seed=42)
    print("  ✓ Agent created successfully")
    print()
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    print(f"  Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    print()
    sys.exit(1)

# Try running simulation
print("Test 2: Running simulation...")
try:
    result = simulate_agent_fixed(
        agent_class=WMRLHybridAgent,
        params=params,
        env=env,
        num_trials=20,
        seed=42
    )
    print("  ✓ Simulation completed successfully")
    print(f"  Accuracy: {result.accuracy:.3f}")
    print()
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    print(f"  Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    print()
    sys.exit(1)

# Test with multiple parameter sets
print("Test 3: Multiple parameter sets (like in exploration)...")
print()

n_samples = 5
np.random.seed(42)

param_samples = pd.DataFrame({
    'alpha_pos': np.random.beta(2, 2, n_samples),
    'alpha_neg': np.random.beta(2, 2, n_samples),
    'beta': np.random.gamma(2, 1, n_samples),
    'beta_wm': np.random.gamma(2, 1, n_samples),
    'capacity': np.random.randint(2, 7, n_samples),
    'phi': np.random.beta(2, 2, n_samples),
    'rho': np.random.beta(2, 2, n_samples),
})

print(f"Testing {n_samples} parameter sets:")
print(param_samples)
print()

for idx, row in param_samples.iterrows():
    print(f"Testing parameter set {idx + 1}/{n_samples}...")

    # Prepare params
    test_params = {
        'num_stimuli': 6,
        'num_actions': 3,
        'gamma': 0.0,
        'q_init': 0.5,
        'wm_init': 0.0,
        'alpha_pos': row['alpha_pos'],
        'alpha_neg': row['alpha_neg'],
        'beta': row['beta'],
        'beta_wm': row['beta_wm'],
        'capacity': int(row['capacity']),
        'phi': row['phi'],
        'rho': row['rho'],
    }

    # Create fresh environment
    test_env = create_rlwm_env(
        set_size=3,
        phase_type='main_task',
        max_trials_per_block=20,
        seed=42 + idx
    )

    try:
        result = simulate_agent_fixed(
            agent_class=WMRLHybridAgent,
            params=test_params,
            env=test_env,
            num_trials=20,
            seed=42 + idx
        )
        print(f"  ✓ Accuracy: {result.accuracy:.3f}")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        print(f"  Parameters that caused error:")
        for k, v in test_params.items():
            print(f"    {k} = {v} (type: {type(v).__name__})")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print()
print("=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
print()
print("WM-RL model is working correctly.")
print("If the full exploration script fails, the issue is elsewhere.")
