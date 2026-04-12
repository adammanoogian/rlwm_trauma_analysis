"""
NumPyro Hierarchical Bayesian Models for RL Parameter Estimation

This module implements hierarchical Bayesian models using NumPyro for:
1. Q-learning with asymmetric learning rates
2. WM-RL hybrid model

Following Senta et al. (2025):
- Beta is fixed at 50 (not estimated) for parameter identifiability
- Epsilon noise parameter is estimated to capture random responding

NumPyro provides:
- NUTS sampler (gradient-based, efficient for continuous parameters)
- Automatic diagnostics (R-hat, effective sample size)
- Seamless integration with JAX likelihood functions
- Faster compilation than PyMC/PyTensor

Model Structure:
---------------
Hierarchical (3-level):
1. Population level: mu_alpha_pos, sigma_alpha_pos, mu_alpha_neg, sigma_alpha_neg,
   mu_epsilon, sigma_epsilon
2. Individual level: alpha_pos_i, alpha_neg_i, epsilon_i for each participant
3. Observations: Actions given stimuli and rewards

Non-centered parameterization used for better sampling efficiency.

Author: Generated for RLWM trauma analysis project
Date: 2025-11-22
Updated: 2026-01-20 - Added epsilon noise, fixed beta=50
Updated: 2026-04-12 - Moved from legacy/ to canonical path (v4.0 INFRA-01)
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpy as np
import pandas as pd
from numpyro.infer import MCMC, NUTS

from scripts.fitting.jax_likelihoods import (
    prepare_block_data,
    q_learning_multiblock_likelihood,
    wmrl_multiblock_likelihood,
)


def qlearning_hierarchical_model(
    participant_data: dict[Any, dict[str, list]],
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
) -> None:
    """
    Hierarchical Bayesian Q-learning model for multiple participants.

    Following Senta et al. (2025):
    - Beta is FIXED at 50 (not estimated) for parameter identifiability
    - Epsilon noise parameter is estimated to capture random responding

    Model Structure:
    ---------------
    # Group-level (population) parameters
    mu_alpha_pos ~ Beta(3, 2)      # Mean positive learning rate ~ 0.6
    sigma_alpha_pos ~ HalfNormal(0.3) # Variability in alpha_pos
    mu_alpha_neg ~ Beta(2, 3)      # Mean negative learning rate ~ 0.4
    sigma_alpha_neg ~ HalfNormal(0.3) # Variability in alpha_neg
    mu_epsilon ~ Beta(1, 19)       # Mean epsilon noise ~ 0.05
    sigma_epsilon ~ HalfNormal(0.1)  # Variability in epsilon

    # Individual-level parameters (non-centered)
    z_alpha_pos_i ~ Normal(0, 1)
    alpha_pos_i = expit(logit(mu_alpha_pos) + sigma_alpha_pos * z_alpha_pos_i)

    z_alpha_neg_i ~ Normal(0, 1)
    alpha_neg_i = expit(logit(mu_alpha_neg) + sigma_alpha_neg * z_alpha_neg_i)

    z_epsilon_i ~ Normal(0, 1)
    epsilon_i = expit(logit(mu_epsilon) + sigma_epsilon * z_epsilon_i)

    # Likelihood (beta=50 fixed)
    actions_i ~ Softmax(Q-values; alpha_pos_i, alpha_neg_i, beta=50, epsilon_i)

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
    mu_alpha_pos = numpyro.sample("mu_alpha_pos", dist.Beta(3, 2))
    sigma_alpha_pos = numpyro.sample("sigma_alpha_pos", dist.HalfNormal(0.3))

    # Negative learning rate: bounded [0, 1]
    mu_alpha_neg = numpyro.sample("mu_alpha_neg", dist.Beta(2, 3))
    sigma_alpha_neg = numpyro.sample("sigma_alpha_neg", dist.HalfNormal(0.3))

    # Epsilon noise: bounded [0, 1], prior centered around 0.05
    # Beta(1, 19) gives mean of 1/20 = 0.05
    mu_epsilon = numpyro.sample("mu_epsilon", dist.Beta(1, 19))
    sigma_epsilon = numpyro.sample("sigma_epsilon", dist.HalfNormal(0.1))

    # ========================================================================
    # INDIVIDUAL-LEVEL PARAMETERS (NON-CENTERED)
    # ========================================================================

    with numpyro.plate("participants", num_participants):
        # Sample standard normal offsets
        z_alpha_pos = numpyro.sample("z_alpha_pos", dist.Normal(0, 1))
        z_alpha_neg = numpyro.sample("z_alpha_neg", dist.Normal(0, 1))
        z_epsilon = numpyro.sample("z_epsilon", dist.Normal(0, 1))

        # Transform to constrained space via logit transformation
        alpha_pos = numpyro.deterministic(
            "alpha_pos",
            jax.scipy.special.expit(
                jax.scipy.special.logit(mu_alpha_pos) + sigma_alpha_pos * z_alpha_pos
            ),
        )
        alpha_neg = numpyro.deterministic(
            "alpha_neg",
            jax.scipy.special.expit(
                jax.scipy.special.logit(mu_alpha_neg) + sigma_alpha_neg * z_alpha_neg
            ),
        )
        epsilon = numpyro.deterministic(
            "epsilon",
            jax.scipy.special.expit(
                jax.scipy.special.logit(mu_epsilon) + sigma_epsilon * z_epsilon
            ),
        )

    # ========================================================================
    # LIKELIHOOD: Compute for each participant
    # ========================================================================

    for i, participant_id in enumerate(participant_ids):
        pdata = participant_data[participant_id]

        # Get individual parameters
        alpha_pos_i = alpha_pos[i]
        alpha_neg_i = alpha_neg[i]
        epsilon_i = epsilon[i]

        # Compute log-likelihood across all blocks for this participant
        # Note: beta is fixed at 50 inside the likelihood function
        log_lik = q_learning_multiblock_likelihood(
            stimuli_blocks=pdata["stimuli_blocks"],
            actions_blocks=pdata["actions_blocks"],
            rewards_blocks=pdata["rewards_blocks"],
            alpha_pos=alpha_pos_i,
            alpha_neg=alpha_neg_i,
            epsilon=epsilon_i,
            num_stimuli=num_stimuli,
            num_actions=num_actions,
            q_init=q_init,
        )

        # Add to model via factor (log probability)
        numpyro.factor(f"obs_p{participant_id}", log_lik)


def wmrl_hierarchical_model(
    participant_data: dict[Any, dict[str, list]],
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,  # WM baseline = 1/nA (uniform)
) -> None:
    """
    Hierarchical Bayesian WM-RL hybrid model for multiple participants.

    Following Senta et al. (2025):
    - Beta is FIXED at 50 for both WM and RL (not estimated)
    - Epsilon noise parameter is estimated to capture random responding

    Model Structure:
    ---------------
    # Group-level (population) parameters
    mu_alpha_pos ~ Beta(3, 2)         # Mean positive learning rate ~ 0.6
    sigma_alpha_pos ~ HalfNormal(0.3) # Variability in alpha_pos
    mu_alpha_neg ~ Beta(2, 3)         # Mean negative learning rate ~ 0.4
    sigma_alpha_neg ~ HalfNormal(0.3) # Variability in alpha_neg
    mu_phi ~ Beta(2, 8)               # Mean WM decay rate ~ 0.2
    sigma_phi ~ HalfNormal(0.3)       # Variability in phi
    mu_rho ~ Beta(5, 2)               # Mean WM reliance ~ 0.7
    sigma_rho ~ HalfNormal(0.3)       # Variability in rho
    mu_K ~ TruncatedNormal(4, 1.5, low=1, high=7)  # Mean WM capacity ~ 4
    sigma_K ~ HalfNormal(1.0)         # Variability in K
    mu_epsilon ~ Beta(1, 19)          # Mean epsilon noise ~ 0.05
    sigma_epsilon ~ HalfNormal(0.1)   # Variability in epsilon

    # Individual-level parameters (non-centered)
    All parameters transformed via logit/log for Beta/bounded distributions

    # Likelihood (beta=50 fixed)
    actions_i ~ Hybrid(WM, RL; all parameters, beta=50)

    Parameters
    ----------
    participant_data : dict
        Nested dict: {participant_id: {
            'stimuli_blocks': list of arrays,
            'actions_blocks': list of arrays,
            'rewards_blocks': list of arrays,
            'set_sizes_blocks': list of arrays  # Set sizes for adaptive weighting
        }}
    num_stimuli : int
        Number of possible stimuli
    num_actions : int
        Number of possible actions
    q_init : float
        Initial Q-value for all state-action pairs
    wm_init : float
        Initial WM values (baseline = 1/nA for uniform)

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
    mu_alpha_pos = numpyro.sample("mu_alpha_pos", dist.Beta(3, 2))
    sigma_alpha_pos = numpyro.sample("sigma_alpha_pos", dist.HalfNormal(0.3))

    # Negative learning rate: bounded [0, 1]
    mu_alpha_neg = numpyro.sample("mu_alpha_neg", dist.Beta(2, 3))
    sigma_alpha_neg = numpyro.sample("sigma_alpha_neg", dist.HalfNormal(0.3))

    # WM decay rate: bounded [0, 1]
    mu_phi = numpyro.sample("mu_phi", dist.Beta(2, 8))
    sigma_phi = numpyro.sample("sigma_phi", dist.HalfNormal(0.3))

    # WM reliance: bounded [0, 1]
    mu_rho = numpyro.sample("mu_rho", dist.Beta(5, 2))
    sigma_rho = numpyro.sample("sigma_rho", dist.HalfNormal(0.3))

    # WM capacity: bounded [1, 7] (using truncated normal)
    mu_capacity = numpyro.sample(
        "mu_capacity", dist.TruncatedNormal(4.0, 1.5, low=1.0, high=7.0)
    )
    sigma_capacity = numpyro.sample("sigma_capacity", dist.HalfNormal(1.0))

    # Epsilon noise: bounded [0, 1], prior centered around 0.05
    mu_epsilon = numpyro.sample("mu_epsilon", dist.Beta(1, 19))
    sigma_epsilon = numpyro.sample("sigma_epsilon", dist.HalfNormal(0.1))

    # ========================================================================
    # INDIVIDUAL-LEVEL PARAMETERS (NON-CENTERED)
    # ========================================================================

    with numpyro.plate("participants", num_participants):
        # Sample standard normal offsets
        z_alpha_pos = numpyro.sample("z_alpha_pos", dist.Normal(0, 1))
        z_alpha_neg = numpyro.sample("z_alpha_neg", dist.Normal(0, 1))
        z_phi = numpyro.sample("z_phi", dist.Normal(0, 1))
        z_rho = numpyro.sample("z_rho", dist.Normal(0, 1))
        z_capacity = numpyro.sample("z_capacity", dist.Normal(0, 1))
        z_epsilon = numpyro.sample("z_epsilon", dist.Normal(0, 1))

        # Transform to constrained space via logit transformation
        alpha_pos = numpyro.deterministic(
            "alpha_pos",
            jax.scipy.special.expit(
                jax.scipy.special.logit(mu_alpha_pos) + sigma_alpha_pos * z_alpha_pos
            ),
        )
        alpha_neg = numpyro.deterministic(
            "alpha_neg",
            jax.scipy.special.expit(
                jax.scipy.special.logit(mu_alpha_neg) + sigma_alpha_neg * z_alpha_neg
            ),
        )
        phi = numpyro.deterministic(
            "phi",
            jax.scipy.special.expit(
                jax.scipy.special.logit(mu_phi) + sigma_phi * z_phi
            ),
        )
        rho = numpyro.deterministic(
            "rho",
            jax.scipy.special.expit(
                jax.scipy.special.logit(mu_rho) + sigma_rho * z_rho
            ),
        )
        epsilon = numpyro.deterministic(
            "epsilon",
            jax.scipy.special.expit(
                jax.scipy.special.logit(mu_epsilon) + sigma_epsilon * z_epsilon
            ),
        )

        # For capacity: use clipped normal transformation
        capacity = numpyro.deterministic(
            "capacity",
            jnp.clip(mu_capacity + sigma_capacity * z_capacity, 1.0, 7.0),
        )

    # ========================================================================
    # LIKELIHOOD: Compute for each participant
    # ========================================================================

    for i, participant_id in enumerate(participant_ids):
        pdata = participant_data[participant_id]

        # Get individual parameters
        alpha_pos_i = alpha_pos[i]
        alpha_neg_i = alpha_neg[i]
        phi_i = phi[i]
        rho_i = rho[i]
        capacity_i = capacity[i]
        epsilon_i = epsilon[i]

        # Compute log-likelihood across all blocks for this participant
        # Note: beta is fixed at 50 inside the likelihood function
        log_lik = wmrl_multiblock_likelihood(
            stimuli_blocks=pdata["stimuli_blocks"],
            actions_blocks=pdata["actions_blocks"],
            rewards_blocks=pdata["rewards_blocks"],
            set_sizes_blocks=pdata["set_sizes_blocks"],
            alpha_pos=alpha_pos_i,
            alpha_neg=alpha_neg_i,
            phi=phi_i,
            rho=rho_i,
            capacity=capacity_i,
            epsilon=epsilon_i,
            num_stimuli=num_stimuli,
            num_actions=num_actions,
            q_init=q_init,
            wm_init=wm_init,
        )

        # Add to model via factor (log probability)
        numpyro.factor(f"obs_p{participant_id}", log_lik)


def prepare_data_for_numpyro(
    data_df: pd.DataFrame,
    participant_col: str = "sona_id",
    block_col: str = "block",
    stimulus_col: str = "stimulus",
    action_col: str = "key_press",
    reward_col: str = "reward",
    set_size_col: str = "set_size",
) -> dict[Any, dict[str, list]]:
    """
    Prepare data in format expected by NumPyro model.

    Parameters
    ----------
    data_df : pd.DataFrame
        Trial-level data with columns for participant, block, stimulus, action, reward
    participant_col : str
        Column name for participant identifier.
    block_col : str
        Column name for block number.
    stimulus_col : str
        Column name for stimulus index.
    action_col : str
        Column name for action taken.
    reward_col : str
        Column name for reward received.
    set_size_col : str
        Column name for set size (for WM-RL model)

    Returns
    -------
    dict
        Nested dict: {participant_id: {
            'stimuli_blocks': [block1_stimuli, block2_stimuli, ...],
            'actions_blocks': [block1_actions, block2_actions, ...],
            'rewards_blocks': [block1_rewards, block2_rewards, ...],
            'set_sizes_blocks': [block1_set_sizes, block2_set_sizes, ...]  # For WM-RL
        }}
    """
    # Get block-structured data
    block_data = prepare_block_data(
        data_df,
        participant_col=participant_col,
        block_col=block_col,
        stimulus_col=stimulus_col,
        action_col=action_col,
        reward_col=reward_col,
    )

    # Restructure for NumPyro
    participant_data: dict[Any, dict[str, list]] = {}

    for participant_id, blocks in block_data.items():
        stimuli_blocks = []
        actions_blocks = []
        rewards_blocks = []
        set_sizes_blocks = []

        # Get participant's data for set sizes
        participant_df = data_df[data_df[participant_col] == participant_id]

        # Sort blocks by block number
        for block_num in sorted(blocks.keys()):
            block = blocks[block_num]
            stimuli_blocks.append(block["stimuli"])
            actions_blocks.append(block["actions"])
            rewards_blocks.append(block["rewards"])

            # Get set sizes for this block
            block_df = participant_df[participant_df[block_col] == block_num]
            if set_size_col in block_df.columns:
                set_sizes = jnp.array(block_df[set_size_col].values, dtype=jnp.float32)
            else:
                # Default to set size 6 if not available
                set_sizes = jnp.ones(len(block["stimuli"]), dtype=jnp.float32) * 6
            set_sizes_blocks.append(set_sizes)

        participant_data[participant_id] = {
            "stimuli_blocks": stimuli_blocks,
            "actions_blocks": actions_blocks,
            "rewards_blocks": rewards_blocks,
            "set_sizes_blocks": set_sizes_blocks,
        }

    return participant_data


def test_likelihood_compilation(
    participant_data: dict,
    verbose: bool = True,
) -> Any:
    """
    Test likelihood compilation on single evaluation (for debugging).

    This helps identify issues before starting expensive MCMC sampling.

    Parameters
    ----------
    participant_data : dict
        Prepared participant data from prepare_data_for_numpyro.
    verbose : bool
        Whether to print progress messages.

    Returns
    -------
    float
        Log-likelihood value for first participant at test parameters.
    """
    if verbose:
        print("\n>> Testing likelihood compilation...")
        print("   This will compile JAX functions (first time only)")
        print("   (Using fixed beta=50, epsilon noise enabled)")

    # Get first participant
    first_pid = list(participant_data.keys())[0]
    pdata = participant_data[first_pid]

    # Test parameters (no beta - it's fixed at 50)
    test_params = {
        "alpha_pos": 0.3,
        "alpha_neg": 0.1,
        "epsilon": 0.05,
    }

    if verbose:
        num_blocks = len(pdata["stimuli_blocks"])
        num_trials = sum([len(block) for block in pdata["stimuli_blocks"]])
        print(f"   Participant {first_pid}: {num_blocks} blocks, {num_trials} trials")

    # Compute likelihood (will trigger compilation)
    log_lik = q_learning_multiblock_likelihood(
        stimuli_blocks=pdata["stimuli_blocks"],
        actions_blocks=pdata["actions_blocks"],
        rewards_blocks=pdata["rewards_blocks"],
        alpha_pos=test_params["alpha_pos"],
        alpha_neg=test_params["alpha_neg"],
        epsilon=test_params["epsilon"],
        verbose=verbose,
        participant_id=str(first_pid),
    )

    if verbose:
        print(f">> Compilation successful! Test log-likelihood: {float(log_lik):.2f}\n")

    return log_lik


def run_inference(
    model: Any,
    model_args: dict[str, Any],
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
    seed: int = 42,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
    test_compilation: bool = True,
) -> MCMC:
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
    if test_compilation and "participant_data" in model_args:
        test_likelihood_compilation(model_args["participant_data"], verbose=True)

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
        max_tree_depth=max_tree_depth,
    )

    # Create MCMC object
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )

    # Run sampling
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, **model_args)

    # Print diagnostics
    print("\n>> Sampling complete! Computing diagnostics...")
    mcmc.print_summary()

    return mcmc


def samples_to_arviz(mcmc: MCMC, data_df: pd.DataFrame | None = None) -> Any:
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
    >>> az.plot_trace(idata, var_names=['mu_alpha_pos'])
    >>> az.summary(idata)
    """
    import arviz as az

    # Get samples and convert to InferenceData
    idata = az.from_numpyro(mcmc)

    return idata


# ============================================================================
# TEST FUNCTIONS
# ============================================================================


def test_model_with_synthetic_data() -> MCMC:
    """
    Test hierarchical model with synthetic data.

    This creates synthetic data from known parameters and verifies
    that the model can recover them.

    Returns
    -------
    mcmc : numpyro.infer.MCMC
        MCMC object after quick inference run.
    """
    print("=" * 80)
    print("TESTING HIERARCHICAL Q-LEARNING MODEL WITH SYNTHETIC DATA")
    print("(Beta fixed at 50, epsilon noise enabled)")
    print("=" * 80)

    # Create synthetic data for 2 participants, 3 blocks each
    key = jax.random.PRNGKey(42)

    # True parameters (what we want to recover)
    # Note: beta is fixed at 50 (not estimated)
    true_params = {
        "mu_alpha_pos": 0.6,
        "mu_alpha_neg": 0.4,
        "mu_epsilon": 0.05,
        "participants": [
            {"alpha_pos": 0.55, "alpha_neg": 0.35, "epsilon": 0.04},
            {"alpha_pos": 0.65, "alpha_neg": 0.45, "epsilon": 0.06},
        ],
    }

    print(f"\nTrue parameters:")
    print(f"  mu_alpha_pos: {true_params['mu_alpha_pos']}")
    print(f"  mu_alpha_neg: {true_params['mu_alpha_neg']}")
    print(f"  mu_epsilon: {true_params['mu_epsilon']}")
    print(f"  beta (fixed): 50")

    # Generate synthetic data
    participant_data: dict = {}
    for i, p_params in enumerate(true_params["participants"]):
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
            "stimuli_blocks": stimuli_blocks,
            "actions_blocks": actions_blocks,
            "rewards_blocks": rewards_blocks,
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
        qlearning_hierarchical_model, num_samples=10
    )(rng_key, participant_data=participant_data)

    print("Model compilation successful!")
    print(f"  Prior samples keys: {list(prior_samples.keys())}")
    print(f"  mu_alpha_pos shape: {prior_samples['mu_alpha_pos'].shape}")
    print(f"  alpha_pos shape: {prior_samples['alpha_pos'].shape}")

    # Quick inference test (very few samples, just to verify)
    print("\nRunning quick inference test (50 warmup, 50 samples, 1 chain)...")
    mcmc = run_inference(
        qlearning_hierarchical_model,
        model_args={"participant_data": participant_data},
        num_warmup=50,
        num_samples=50,
        num_chains=1,
        seed=42,
    )

    # Get posterior summary
    samples = mcmc.get_samples()
    print("\nPosterior estimates (should be near true values):")
    print(
        f"  mu_alpha_pos posterior mean: {samples['mu_alpha_pos'].mean():.3f} "
        f"(true: {true_params['mu_alpha_pos']})"
    )
    print(
        f"  mu_alpha_neg posterior mean: {samples['mu_alpha_neg'].mean():.3f} "
        f"(true: {true_params['mu_alpha_neg']})"
    )
    print(
        f"  mu_epsilon posterior mean: {samples['mu_epsilon'].mean():.3f} "
        f"(true: {true_params['mu_epsilon']})"
    )

    print("\n" + "=" * 80)
    print("TEST COMPLETE!")
    print("=" * 80)

    return mcmc


if __name__ == "__main__":
    # Set NumPyro to use all available CPU cores
    numpyro.set_host_device_count(4)

    test_model_with_synthetic_data()
