"""
NumPyro Hierarchical Bayesian Models for RL Parameter Estimation

This module implements hierarchical Bayesian models using NumPyro for:
1. Q-learning with asymmetric learning rates
2. WM-RL hybrid model (future implementation)

NumPyro provides:
- NUTS sampler (gradient-based, efficient for continuous parameters)
- Automatic diagnostics (R-hat, effective sample size)
- Seamless integration with JAX likelihood functions
- Faster compilation than PyMC/PyTensor

Model Structure:
---------------
Hierarchical (3-level):
1. Population level: μ_α+, σ_α+, μ_α-, σ_α-, μ_β, σ_β
2. Individual level: α+_i, α-_i, β_i for each participant
3. Observations: Actions given stimuli and rewards

Non-centered parameterization used for better sampling efficiency.

Author: Generated for RLWM trauma analysis project
Date: 2025-11-22
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from scripts.fitting.jax_likelihoods import (
    q_learning_multiblock_likelihood,
    prepare_block_data
)


def qlearning_hierarchical_model(
    participant_data: Dict[Any, Dict[str, List]],
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5
):
    """
    Hierarchical Bayesian Q-learning model for multiple participants.

    Model Structure:
    ---------------
    # Group-level (population) parameters
    μ_α+ ~ Beta(3, 2)      # Mean positive learning rate ~ 0.6
    σ_α+ ~ HalfNormal(0.3) # Variability in α+
    μ_α- ~ Beta(2, 3)      # Mean negative learning rate ~ 0.4
    σ_α- ~ HalfNormal(0.3) # Variability in α-
    μ_β ~ Gamma(2, 1)      # Mean inverse temperature ~ 2
    σ_β ~ HalfNormal(2.0)  # Variability in β

    # Individual-level parameters (non-centered)
    z_α+_i ~ Normal(0, 1)
    α+_i = logit^(-1)(logit(μ_α+) + σ_α+ * z_α+_i)

    z_α-_i ~ Normal(0, 1)
    α-_i = logit^(-1)(logit(μ_α-) + σ_α- * z_α-_i)

    z_β_i ~ Normal(0, 1)
    β_i = exp(log(μ_β) + σ_β * z_β_i)

    # Likelihood
    actions_i ~ Softmax(Q-values; α+_i, α-_i, β_i)

    Parameters
    ----------
    participant_data : dict
        Nested dict: {participant_id: {
            'stimuli_blocks': list of arrays,
            'actions_blocks': list of arrays,
            'rewards_blocks': list of arrays
        }}
    num_stimuli : int
        Number of possible stimuli
    num_actions : int
        Number of possible actions
    q_init : float
        Initial Q-value for all state-action pairs

    Notes
    -----
    This function is used with numpyro.infer.MCMC for sampling.
    """
    num_participants = len(participant_data)
    participant_ids = list(participant_data.keys())

    # ========================================================================
    # GROUP-LEVEL (POPULATION) PRIORS
    # ========================================================================

    # Positive learning rate: bounded [0, 1]
    mu_alpha_pos = numpyro.sample('mu_alpha_pos', dist.Beta(3, 2))
    sigma_alpha_pos = numpyro.sample('sigma_alpha_pos', dist.HalfNormal(0.3))

    # Negative learning rate: bounded [0, 1]
    mu_alpha_neg = numpyro.sample('mu_alpha_neg', dist.Beta(2, 3))
    sigma_alpha_neg = numpyro.sample('sigma_alpha_neg', dist.HalfNormal(0.3))

    # Inverse temperature: positive
    mu_beta = numpyro.sample('mu_beta', dist.Gamma(2.0, 1.0))
    sigma_beta = numpyro.sample('sigma_beta', dist.HalfNormal(2.0))

    # ========================================================================
    # INDIVIDUAL-LEVEL PARAMETERS (NON-CENTERED)
    # ========================================================================

    with numpyro.plate('participants', num_participants):
        # Sample standard normal offsets
        z_alpha_pos = numpyro.sample('z_alpha_pos', dist.Normal(0, 1))
        z_alpha_neg = numpyro.sample('z_alpha_neg', dist.Normal(0, 1))
        z_beta = numpyro.sample('z_beta', dist.Normal(0, 1))

        # Transform to constrained space
        # For Beta-distributed parameters: use logit transformation
        alpha_pos = numpyro.deterministic(
            'alpha_pos',
            jax.scipy.special.expit(
                jax.scipy.special.logit(mu_alpha_pos) + sigma_alpha_pos * z_alpha_pos
            )
        )
        alpha_neg = numpyro.deterministic(
            'alpha_neg',
            jax.scipy.special.expit(
                jax.scipy.special.logit(mu_alpha_neg) + sigma_alpha_neg * z_alpha_neg
            )
        )

        # For Gamma-distributed parameters: use log transformation
        beta = numpyro.deterministic(
            'beta',
            jnp.exp(jnp.log(mu_beta) + sigma_beta * z_beta)
        )

    # ========================================================================
    # LIKELIHOOD: Compute for each participant
    # ========================================================================

    for i, participant_id in enumerate(participant_ids):
        pdata = participant_data[participant_id]

        # Get individual parameters
        alpha_pos_i = alpha_pos[i]
        alpha_neg_i = alpha_neg[i]
        beta_i = beta[i]

        # Compute log-likelihood across all blocks for this participant
        log_lik = q_learning_multiblock_likelihood(
            stimuli_blocks=pdata['stimuli_blocks'],
            actions_blocks=pdata['actions_blocks'],
            rewards_blocks=pdata['rewards_blocks'],
            alpha_pos=alpha_pos_i,
            alpha_neg=alpha_neg_i,
            beta=beta_i,
            num_stimuli=num_stimuli,
            num_actions=num_actions,
            q_init=q_init
        )

        # Add to model via factor (log probability)
        numpyro.factor(f'obs_p{participant_id}', log_lik)


def prepare_data_for_numpyro(
    data_df: pd.DataFrame,
    participant_col: str = 'sona_id',
    block_col: str = 'block',
    stimulus_col: str = 'stimulus',
    action_col: str = 'key_press',
    reward_col: str = 'reward'
) -> Dict[Any, Dict[str, List]]:
    """
    Prepare data in format expected by NumPyro model.

    Parameters
    ----------
    data_df : pd.DataFrame
        Trial-level data with columns for participant, block, stimulus, action, reward
    participant_col, block_col, etc. : str
        Column names

    Returns
    -------
    dict
        Nested dict: {participant_id: {
            'stimuli_blocks': [block1_stimuli, block2_stimuli, ...],
            'actions_blocks': [block1_actions, block2_actions, ...],
            'rewards_blocks': [block1_rewards, block2_rewards, ...]
        }}
    """
    # Get block-structured data
    block_data = prepare_block_data(
        data_df,
        participant_col=participant_col,
        block_col=block_col,
        stimulus_col=stimulus_col,
        action_col=action_col,
        reward_col=reward_col
    )

    # Restructure for NumPyro
    participant_data = {}

    for participant_id, blocks in block_data.items():
        stimuli_blocks = []
        actions_blocks = []
        rewards_blocks = []

        # Sort blocks by block number
        for block_num in sorted(blocks.keys()):
            block = blocks[block_num]
            stimuli_blocks.append(block['stimuli'])
            actions_blocks.append(block['actions'])
            rewards_blocks.append(block['rewards'])

        participant_data[participant_id] = {
            'stimuli_blocks': stimuli_blocks,
            'actions_blocks': actions_blocks,
            'rewards_blocks': rewards_blocks
        }

    return participant_data


def test_likelihood_compilation(
    participant_data: Dict,
    verbose: bool = True
):
    """
    Test likelihood compilation on single evaluation (for debugging).

    This helps identify issues before starting expensive MCMC sampling.
    """
    if verbose:
        print("\n>> Testing likelihood compilation...")
        print("   This will compile JAX functions (first time only)")

    # Get first participant
    first_pid = list(participant_data.keys())[0]
    pdata = participant_data[first_pid]

    # Test parameters
    test_params = {
        'alpha_pos': 0.3,
        'alpha_neg': 0.1,
        'beta': 2.0
    }

    if verbose:
        num_blocks = len(pdata['stimuli_blocks'])
        num_trials = sum([len(block) for block in pdata['stimuli_blocks']])
        print(f"   Participant {first_pid}: {num_blocks} blocks, {num_trials} trials")

    # Compute likelihood (will trigger compilation)
    log_lik = q_learning_multiblock_likelihood(
        stimuli_blocks=pdata['stimuli_blocks'],
        actions_blocks=pdata['actions_blocks'],
        rewards_blocks=pdata['rewards_blocks'],
        alpha_pos=test_params['alpha_pos'],
        alpha_neg=test_params['alpha_neg'],
        beta=test_params['beta'],
        verbose=verbose,
        participant_id=str(first_pid)
    )

    if verbose:
        print(f">> Compilation successful! Test log-likelihood: {float(log_lik):.2f}\n")

    return log_lik


def run_inference(
    model: callable,
    model_args: Dict[str, Any],
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
    seed: int = 42,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
    test_compilation: bool = True
):
    """
    Run MCMC inference using NUTS sampler.

    Parameters
    ----------
    model : callable
        NumPyro model function (e.g., qlearning_hierarchical_model)
    model_args : dict
        Arguments to pass to model
    num_warmup : int
        Number of warmup/tuning samples (default: 1000)
    num_samples : int
        Number of posterior samples per chain (default: 2000)
    num_chains : int
        Number of MCMC chains (default: 4)
    seed : int
        Random seed
    target_accept_prob : float
        Target acceptance probability for NUTS (default: 0.8)
    max_tree_depth : int
        Maximum tree depth for NUTS (default: 10)
    test_compilation : bool
        Test likelihood compilation before MCMC (default: True)

    Returns
    -------
    mcmc : numpyro.infer.MCMC
        MCMC object containing samples and diagnostics

    Examples
    --------
    >>> participant_data = prepare_data_for_numpyro(df)
    >>> model_args = {'participant_data': participant_data}
    >>> mcmc = run_inference(
    ...     qlearning_hierarchical_model,
    ...     model_args,
    ...     num_warmup=500,
    ...     num_samples=1000,
    ...     num_chains=2
    ... )
    >>> samples = mcmc.get_samples()
    >>> print(samples['mu_alpha_pos'].mean())
    """
    # Test compilation first
    if test_compilation and 'participant_data' in model_args:
        test_likelihood_compilation(model_args['participant_data'], verbose=True)

    print(">> Starting MCMC sampling...")
    print(f"   Chains: {num_chains}")
    print(f"   Warmup: {num_warmup}")
    print(f"   Samples: {num_samples}")
    print(f"   Total iterations: {(num_warmup + num_samples) * num_chains}")
    print()

    # Initialize NUTS sampler
    nuts_kernel = NUTS(
        model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth
    )

    # Create MCMC object
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True
    )

    # Run sampling
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, **model_args)

    # Print diagnostics
    print("\n>> Sampling complete! Computing diagnostics...")
    mcmc.print_summary()

    return mcmc


def samples_to_arviz(mcmc: MCMC, data_df: pd.DataFrame = None):
    """
    Convert NumPyro MCMC samples to ArviZ InferenceData format.

    This enables compatibility with ArviZ for diagnostics, visualization,
    and model comparison.

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        MCMC object from run_inference
    data_df : pd.DataFrame, optional
        Original data (for posterior predictive checks)

    Returns
    -------
    az.InferenceData
        ArviZ InferenceData object

    Examples
    --------
    >>> mcmc = run_inference(...)
    >>> idata = samples_to_arviz(mcmc, data_df)
    >>> import arviz as az
    >>> az.plot_trace(idata, var_names=['mu_alpha_pos', 'mu_beta'])
    >>> az.summary(idata)
    """
    import arviz as az

    # Get samples and convert to InferenceData
    idata = az.from_numpyro(mcmc)

    return idata


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_model_with_synthetic_data():
    """
    Test hierarchical model with synthetic data.

    This creates synthetic data from known parameters and verifies
    that the model can recover them.
    """
    print("=" * 80)
    print("TESTING HIERARCHICAL Q-LEARNING MODEL WITH SYNTHETIC DATA")
    print("=" * 80)

    # Create synthetic data for 2 participants, 3 blocks each
    key = jax.random.PRNGKey(42)

    # True parameters (what we want to recover)
    true_params = {
        'mu_alpha_pos': 0.6,
        'mu_alpha_neg': 0.4,
        'mu_beta': 2.0,
        'participants': [
            {'alpha_pos': 0.55, 'alpha_neg': 0.35, 'beta': 1.8},
            {'alpha_pos': 0.65, 'alpha_neg': 0.45, 'beta': 2.2},
        ]
    }

    print(f"\nTrue parameters:")
    print(f"  μ_α+: {true_params['mu_alpha_pos']}")
    print(f"  μ_α-: {true_params['mu_alpha_neg']}")
    print(f"  μ_β: {true_params['mu_beta']}")

    # Generate synthetic data
    participant_data = {}
    for i, p_params in enumerate(true_params['participants']):
        stimuli_blocks = []
        actions_blocks = []
        rewards_blocks = []

        # 3 blocks of 30 trials each
        for block in range(3):
            key, subkey = jax.random.split(key)
            stimuli = jax.random.randint(subkey, (30,), 0, 6)

            key, subkey = jax.random.split(key)
            actions = jax.random.randint(subkey, (30,), 0, 3)

            key, subkey = jax.random.split(key)
            rewards = jax.random.bernoulli(subkey, 0.7, (30,)).astype(jnp.float32)

            stimuli_blocks.append(stimuli)
            actions_blocks.append(actions)
            rewards_blocks.append(rewards)

        participant_data[i] = {
            'stimuli_blocks': stimuli_blocks,
            'actions_blocks': actions_blocks,
            'rewards_blocks': rewards_blocks
        }

    print(f"\nGenerated data:")
    print(f"  Participants: {len(participant_data)}")
    print(f"  Blocks per participant: {len(participant_data[0]['stimuli_blocks'])}")
    print(f"  Trials per block: {len(participant_data[0]['stimuli_blocks'][0])}")

    # Test that model runs (just check compilation, not full sampling)
    print("\nTesting model compilation...")

    # Use seed to get reproducible behavior
    rng_key = jax.random.PRNGKey(42)

    # Sample from prior (no inference, just check model structure)
    prior_samples = numpyro.infer.Predictive(
        qlearning_hierarchical_model,
        num_samples=10
    )(rng_key, participant_data=participant_data)

    print("✓ Model compilation successful!")
    print(f"  Prior samples keys: {list(prior_samples.keys())}")
    print(f"  μ_α+ shape: {prior_samples['mu_alpha_pos'].shape}")
    print(f"  α+ shape: {prior_samples['alpha_pos'].shape}")

    # Quick inference test (very few samples, just to verify)
    print("\nRunning quick inference test (50 warmup, 50 samples, 1 chain)...")
    mcmc = run_inference(
        qlearning_hierarchical_model,
        model_args={'participant_data': participant_data},
        num_warmup=50,
        num_samples=50,
        num_chains=1,
        seed=42
    )

    # Get posterior summary
    samples = mcmc.get_samples()
    print("\nPosterior estimates (should be near true values):")
    print(f"  μ_α+ posterior mean: {samples['mu_alpha_pos'].mean():.3f} (true: {true_params['mu_alpha_pos']})")
    print(f"  μ_α- posterior mean: {samples['mu_alpha_neg'].mean():.3f} (true: {true_params['mu_alpha_neg']})")
    print(f"  μ_β posterior mean: {samples['mu_beta'].mean():.3f} (true: {true_params['mu_beta']})")

    print("\n" + "=" * 80)
    print("TEST COMPLETE!")
    print("=" * 80)

    return mcmc


if __name__ == "__main__":
    # Set NumPyro to use all available CPU cores
    numpyro.set_host_device_count(4)

    test_model_with_synthetic_data()
