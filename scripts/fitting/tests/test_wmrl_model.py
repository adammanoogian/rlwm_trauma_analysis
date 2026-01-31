"""
Quick test of WM-RL hierarchical model compilation.

This tests that:
1. WM-RL likelihood functions compile correctly
2. WM-RL hierarchical model structure is valid
3. Model can sample from prior (no inference, just structure check)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
import jax
import jax.numpy as jnp
import numpyro

from scripts.fitting.numpyro_models import wmrl_hierarchical_model


def test_wmrl_model_compilation(wmrl_participant_data):
    """Test WM-RL model compilation with synthetic data."""
    participant_data = wmrl_participant_data

    # Sample from prior (no inference, just check model structure)
    rng_key = jax.random.PRNGKey(42)

    prior_samples = numpyro.infer.Predictive(
        wmrl_hierarchical_model,
        num_samples=10
    )(rng_key, participant_data=participant_data)

    # Check that key parameters are present
    assert 'mu_alpha_pos' in prior_samples
    assert 'mu_beta' in prior_samples
    assert 'mu_beta_wm' in prior_samples
    assert 'mu_phi' in prior_samples
    assert 'mu_rho' in prior_samples
    assert 'mu_capacity' in prior_samples

    # Check shapes
    assert prior_samples['mu_alpha_pos'].shape == (10,)
    assert prior_samples['alpha_pos'].shape[0] == 10


def test_wmrl_prior_ranges(wmrl_participant_data):
    """Test that prior samples are in valid ranges."""
    participant_data = wmrl_participant_data

    rng_key = jax.random.PRNGKey(123)

    prior_samples = numpyro.infer.Predictive(
        wmrl_hierarchical_model,
        num_samples=100
    )(rng_key, participant_data=participant_data)

    # Check all samples are in valid ranges
    assert jnp.all(prior_samples['mu_alpha_pos'] >= 0)
    assert jnp.all(prior_samples['mu_alpha_pos'] <= 1)
    assert jnp.all(prior_samples['mu_phi'] >= 0)
    assert jnp.all(prior_samples['mu_phi'] <= 1)
    assert jnp.all(prior_samples['mu_rho'] >= 0)
    assert jnp.all(prior_samples['mu_rho'] <= 1)
    assert jnp.all(prior_samples['mu_capacity'] >= 1)


# Allow running as standalone script
if __name__ == "__main__":
    # Set NumPyro to use multiple CPU cores
    numpyro.set_host_device_count(4)

    print("=" * 80)
    print("TESTING WM-RL HIERARCHICAL MODEL COMPILATION")
    print("=" * 80)

    # Create synthetic data for 2 participants, 2 blocks each
    key = jax.random.PRNGKey(42)

    participant_data = {}
    for i in range(2):
        stimuli_blocks = []
        actions_blocks = []
        rewards_blocks = []
        set_sizes_blocks = []

        # 2 blocks of 30 trials each
        for block in range(2):
            key, subkey = jax.random.split(key)
            stimuli = jax.random.randint(subkey, (30,), 0, 6)

            key, subkey = jax.random.split(key)
            actions = jax.random.randint(subkey, (30,), 0, 3)

            key, subkey = jax.random.split(key)
            rewards = jax.random.bernoulli(subkey, 0.7, (30,)).astype(jnp.float32)

            set_sizes = jnp.ones((30,)) * 5  # Set size of 5

            stimuli_blocks.append(stimuli)
            actions_blocks.append(actions)
            rewards_blocks.append(rewards)
            set_sizes_blocks.append(set_sizes)

        participant_data[i] = {
            'stimuli_blocks': stimuli_blocks,
            'actions_blocks': actions_blocks,
            'rewards_blocks': rewards_blocks,
            'set_sizes_blocks': set_sizes_blocks
        }

    print(f"\nGenerated synthetic data:")
    print(f"  Participants: {len(participant_data)}")
    print(f"  Blocks per participant: {len(participant_data[0]['stimuli_blocks'])}")
    print(f"  Trials per block: {len(participant_data[0]['stimuli_blocks'][0])}")

    # Test model compilation by sampling from prior
    print("\nTesting model structure by sampling from prior...")

    rng_key = jax.random.PRNGKey(42)

    # Sample from prior (no inference, just check model structure)
    prior_samples = numpyro.infer.Predictive(
        wmrl_hierarchical_model,
        num_samples=10
    )(rng_key, participant_data=participant_data)

    print("\n✓ Model compilation successful!")
    print(f"\nPrior sample keys: {list(prior_samples.keys())}")

    # Group-level parameters
    print(f"\nGroup-level parameter shapes:")
    print(f"  mu_alpha_pos: {prior_samples['mu_alpha_pos'].shape}")
    print(f"  mu_beta: {prior_samples['mu_beta'].shape}")
    print(f"  mu_beta_wm: {prior_samples['mu_beta_wm'].shape}")
    print(f"  mu_phi: {prior_samples['mu_phi'].shape}")
    print(f"  mu_rho: {prior_samples['mu_rho'].shape}")
    print(f"  mu_capacity: {prior_samples['mu_capacity'].shape}")

    # Individual-level parameters
    print(f"\nIndividual-level parameter shapes:")
    print(f"  alpha_pos: {prior_samples['alpha_pos'].shape}")
    print(f"  beta: {prior_samples['beta'].shape}")
    print(f"  beta_wm: {prior_samples['beta_wm'].shape}")
    print(f"  phi: {prior_samples['phi'].shape}")
    print(f"  rho: {prior_samples['rho'].shape}")
    print(f"  capacity: {prior_samples['capacity'].shape}")

    # Print some prior values to check they're in valid ranges
    print(f"\nPrior sample statistics (mean across samples):")
    print(f"  mu_alpha_pos: {prior_samples['mu_alpha_pos'].mean():.3f}")
    print(f"  mu_beta: {prior_samples['mu_beta'].mean():.3f}")
    print(f"  mu_beta_wm: {prior_samples['mu_beta_wm'].mean():.3f}")
    print(f"  mu_phi: {prior_samples['mu_phi'].mean():.3f}")
    print(f"  mu_rho: {prior_samples['mu_rho'].mean():.3f}")
    print(f"  mu_capacity: {prior_samples['mu_capacity'].mean():.3f}")

    print("\n" + "=" * 80)
    print("WM-RL MODEL TEST COMPLETE!")
    print("=" * 80)
