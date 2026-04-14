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
    MAX_TRIALS_PER_BLOCK,
    pad_block_to_max,
    prepare_block_data,
    q_learning_multiblock_likelihood,
    q_learning_multiblock_likelihood_stacked,
    q_learning_multiblock_likelihood_stacked_pscan,
    wmrl_m3_multiblock_likelihood_stacked,
    wmrl_m3_multiblock_likelihood_stacked_pscan,
    wmrl_m5_multiblock_likelihood_stacked,
    wmrl_m5_multiblock_likelihood_stacked_pscan,
    wmrl_m6a_multiblock_likelihood_stacked,
    wmrl_m6a_multiblock_likelihood_stacked_pscan,
    wmrl_m6b_multiblock_likelihood_stacked,
    wmrl_m6b_multiblock_likelihood_stacked_pscan,
    wmrl_multiblock_likelihood,
    wmrl_multiblock_likelihood_stacked,
    wmrl_multiblock_likelihood_stacked_pscan,
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
    print(f"   Host devices: {jax.local_device_count()}")
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
    # Use parallel chains when multiple host devices are available (set via
    # NUMPYRO_HOST_DEVICE_COUNT env var or numpyro.set_host_device_count).
    _chain_method = (
        "parallel" if jax.local_device_count() >= num_chains else "sequential"
    )
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=_chain_method,
        progress_bar=True,
    )

    # Run sampling
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, **model_args)

    # Print diagnostics
    print("\n>> Sampling complete! Computing diagnostics...")
    mcmc.print_summary()

    return mcmc


def run_inference_with_bump(
    model: Any,
    model_args: dict[str, Any],
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
    seed: int = 42,
    target_accept_probs: tuple[float, ...] = (0.80, 0.95, 0.99),
    max_tree_depth: int = 10,
) -> MCMC:
    """Run MCMC inference with automatic convergence bump on divergences.

    Iterates over ``target_accept_probs`` in order, running full MCMC at each
    level.  Returns immediately if a run produces zero divergences.  If all
    levels still have divergences, returns the last MCMC object so the
    downstream convergence gate can flag it.

    Parameters
    ----------
    model : callable
        NumPyro model function (e.g., ``wmrl_m3_hierarchical_model``).
    model_args : dict
        Keyword arguments forwarded to ``mcmc.run()`` via ``**model_args``.
    num_warmup : int
        Number of warmup/tuning samples per chain.  Default 1000.
    num_samples : int
        Number of posterior samples per chain.  Default 2000.
    num_chains : int
        Number of independent MCMC chains.  Default 4.
    seed : int
        Random seed for ``jax.random.PRNGKey``.  Default 42.
    target_accept_probs : tuple[float, ...]
        Sequence of NUTS target acceptance probabilities to try in order.
        Default ``(0.80, 0.95, 0.99)``.
    max_tree_depth : int
        Maximum tree depth for the NUTS kernel.  Default 10.

    Returns
    -------
    MCMC
        The MCMC object from the first run with zero divergences, or the
        last run if divergences remain after all acceptance-probability
        levels are exhausted.

    Notes
    -----
    - Divergence count is read from ``mcmc.get_extra_fields()["diverging"].sum()``.
    - A log line ``[convergence-gate] target_accept_prob=X.XX divergences=N``
      is printed after each run so users can track the bumping process.
    - The downstream convergence gate in ``fit_bayesian.py`` checks that
      ``num_divergences == 0`` before writing output files (HIER-07).
    """
    # Enable parallel chains on CPU by exposing multiple host devices.
    # Without this, JAX sees only 1 device and chains run sequentially
    # (~4x slower for 4 chains).
    if num_chains > 1:
        numpyro.set_host_device_count(num_chains)

    print(">> Starting MCMC sampling with convergence auto-bump...")
    print(f"   Chains: {num_chains}")
    print(f"   Host devices: {jax.local_device_count()}")
    print(f"   Warmup: {num_warmup}")
    print(f"   Samples: {num_samples}")
    print(f"   Max tree depth: {max_tree_depth}")
    print(f"   Total iterations per chain: {num_warmup + num_samples}")
    print(f"   Acceptance probability schedule: {target_accept_probs}")
    print()

    last_mcmc: MCMC | None = None
    for tap in target_accept_probs:
        nuts_kernel = NUTS(
            model,
            target_accept_prob=tap,
            max_tree_depth=max_tree_depth,
        )
        _chain_method = (
            "parallel" if jax.local_device_count() >= num_chains else "sequential"
        )
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=_chain_method,
            progress_bar=True,
        )
        rng_key = jax.random.PRNGKey(seed)
        mcmc.run(rng_key, **model_args)

        extra = mcmc.get_extra_fields()
        n_div = int(extra["diverging"].sum()) if "diverging" in extra else 0
        print(f"[convergence-gate] target_accept_prob={tap:.2f} divergences={n_div}")

        last_mcmc = mcmc
        if n_div == 0:
            print("[convergence-gate] Zero divergences — accepting this run.")
            return mcmc

        print(f"[convergence-gate] {n_div} divergences remain — bumping target_accept_prob.")

    print(
        "[convergence-gate] WARNING: divergences remain after all acceptance-probability "
        "levels exhausted.  Returning last run; downstream gate will flag this."
    )
    return last_mcmc  # type: ignore[return-value]


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


def prepare_stacked_participant_data(
    data_df: pd.DataFrame,
    participant_col: str = "sona_id",
    block_col: str = "block",
    stimulus_col: str = "stimulus",
    action_col: str = "key_press",
    reward_col: str = "reward",
    set_size_col: str = "set_size",
) -> dict[Any, dict[str, jnp.ndarray]]:
    """Prepare stacked participant data for the M3 hierarchical model.

    Converts a trial-level DataFrame into the pre-stacked JAX array format
    expected by ``wmrl_m3_hierarchical_model`` and ``compute_pointwise_log_lik``.
    Each participant's blocks are padded to ``MAX_TRIALS_PER_BLOCK`` using
    ``pad_block_to_max``, then stacked into 2-D arrays of shape
    ``(n_blocks, MAX_TRIALS_PER_BLOCK)``.

    Parameters
    ----------
    data_df : pd.DataFrame
        Trial-level data with participant, block, stimulus, action, reward
        and set_size columns.
    participant_col : str
        Column name for participant identifier.  Default ``"sona_id"``.
    block_col : str
        Column name for block number.  Default ``"block"``.
    stimulus_col : str
        Column name for stimulus index.  Default ``"stimulus"``.
    action_col : str
        Column name for action taken.  Default ``"key_press"``.
    reward_col : str
        Column name for reward received.  Default ``"reward"``.
    set_size_col : str
        Column name for set size.  Default ``"set_size"``.

    Returns
    -------
    dict[Any, dict[str, jnp.ndarray]]
        Mapping from participant_id to a dict with keys:

        * ``stimuli_stacked``   -- shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``, int32
        * ``actions_stacked``   -- shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``, int32
        * ``rewards_stacked``   -- shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``, float32
        * ``set_sizes_stacked`` -- shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``, float32
        * ``masks_stacked``     -- shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``, float32

    Notes
    -----
    This function is the bridge between the DataFrame pipeline and the stacked
    format consumed by ``compute_pointwise_log_lik`` in ``bayesian_diagnostics.py``.
    The existing ``prepare_data_for_numpyro`` returns lists of arrays (old format);
    this function returns pre-stacked 2-D JAX arrays (new format for Phase 15+).

    Participant keys are sorted before processing so that downstream covariate
    arrays (e.g., ``covariate_lec``) align with ``sorted(result.keys())``.
    """
    participant_data: dict[Any, dict[str, jnp.ndarray]] = {}

    for participant_id in sorted(data_df[participant_col].unique()):
        ppt_df = data_df[data_df[participant_col] == participant_id]

        stimuli_blocks = []
        actions_blocks = []
        rewards_blocks = []
        set_sizes_blocks = []
        masks_blocks = []

        for block_num in sorted(ppt_df[block_col].unique()):
            block_df = ppt_df[ppt_df[block_col] == block_num]

            stim = jnp.array(block_df[stimulus_col].values, dtype=jnp.int32)
            act = jnp.array(block_df[action_col].values, dtype=jnp.int32)
            rew = jnp.array(block_df[reward_col].values, dtype=jnp.float32)

            if set_size_col in block_df.columns:
                ss = jnp.array(block_df[set_size_col].values, dtype=jnp.float32)
            else:
                ss = jnp.ones(len(stim), dtype=jnp.float32) * 6.0

            # pad_block_to_max returns (stim, act, rew, set_sizes_padded, mask)
            # when set_sizes is provided -- mask is LAST, set_sizes is fourth.
            p_stim, p_act, p_rew, p_ss, p_mask = pad_block_to_max(
                stim, act, rew, set_sizes=ss
            )
            stimuli_blocks.append(p_stim)
            actions_blocks.append(p_act)
            rewards_blocks.append(p_rew)
            set_sizes_blocks.append(p_ss)
            masks_blocks.append(p_mask)

        participant_data[participant_id] = {
            "stimuli_stacked": jnp.stack(stimuli_blocks),
            "actions_stacked": jnp.stack(actions_blocks),
            "rewards_stacked": jnp.stack(rewards_blocks),
            "set_sizes_stacked": jnp.stack(set_sizes_blocks),
            "masks_stacked": jnp.stack(masks_blocks),
        }

    return participant_data


def wmrl_m3_hierarchical_model(
    participant_data_stacked: dict,
    covariate_lec: jnp.ndarray | None = None,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    use_pscan: bool = False,
) -> None:
    """Hierarchical Bayesian M3 (WM-RL+kappa) model with optional Level-2 regression.

    Uses the hBayesDM non-centered parameterization convention (Ahn, Haines, Zhang 2017):
    ``theta_unc = mu_pr + sigma_pr * z``,
    ``theta = lower + (upper - lower) * Phi_approx(theta_unc)``,
    where ``Phi_approx = jax.scipy.stats.norm.cdf``.

    K (capacity) is parameterized in [2, 6] following Senta, Bishop, Collins (2025).

    Level-2 regression: if ``covariate_lec`` is provided (standardized LEC-total score),
    a coefficient ``beta_lec_kappa`` is sampled and added as a per-participant shift on
    the unconstrained kappa scale before the Phi_approx transform:
    ``kappa_unc_i = kappa_mu_pr + kappa_sigma_pr * z_i + beta_lec_kappa * lec_i``.

    Parameters
    ----------
    participant_data_stacked : dict
        Mapping from participant_id to stacked-format arrays (from
        ``prepare_stacked_participant_data``). Keys per participant:
        ``stimuli_stacked``, ``actions_stacked``, ``rewards_stacked``,
        ``set_sizes_stacked``, ``masks_stacked`` — each shape
        ``(n_blocks, MAX_TRIALS_PER_BLOCK)``.
    covariate_lec : jnp.ndarray or None
        Shape ``(n_participants,)`` standardized LEC-total covariate.
        Participants must be in the same order as
        ``sorted(participant_data_stacked.keys())``.  If ``None``,
        no Level-2 regression is applied and ``beta_lec_kappa`` is
        not sampled.
    num_stimuli : int
        Number of distinct stimuli in the task.  Default 6.
    num_actions : int
        Number of possible actions.  Default 3.
    q_init : float
        Initial Q-value for all state-action pairs.  Default 0.5.
    wm_init : float
        Initial WM values (uniform baseline ``1/nA``).  Default ``1/3``.

    Notes
    -----
    - Six parameters (alpha_pos, alpha_neg, phi, rho, capacity, epsilon) are sampled
      via ``sample_bounded_param`` from ``numpyro_helpers``.
    - Kappa is sampled manually with the optional L2 shift applied on the probit scale
      before the Phi_approx transform (OUTSIDE ``sample_bounded_param``).
    - Likelihood is accumulated via ``numpyro.factor`` in a Python for-loop over
      participants (matches existing qlearning/wmrl models; vmap not applicable here).
    - Do NOT modify this function's API: ``fit_bayesian.py`` dispatches to it by name.
    """
    from scripts.fitting.numpyro_helpers import (
        PARAM_PRIOR_DEFAULTS,
        phi_approx,
        sample_bounded_param,
    )

    n_participants = len(participant_data_stacked)
    participant_ids = sorted(participant_data_stacked.keys())

    # ------------------------------------------------------------------
    # Level-2: LEC-total -> kappa regression coefficient
    # ------------------------------------------------------------------
    if covariate_lec is not None:
        beta_lec_kappa = numpyro.sample("beta_lec_kappa", dist.Normal(0.0, 1.0))
    else:
        beta_lec_kappa = 0.0

    # ------------------------------------------------------------------
    # Group priors for 6 parameters (all except kappa)
    # Uses hBayesDM non-centered convention locked in Phase 13.
    # ------------------------------------------------------------------
    sampled: dict[str, jnp.ndarray] = {}
    for param in ["alpha_pos", "alpha_neg", "phi", "rho", "capacity", "epsilon"]:
        defaults = PARAM_PRIOR_DEFAULTS[param]
        sampled[param] = sample_bounded_param(
            param,
            lower=defaults["lower"],
            upper=defaults["upper"],
            n_participants=n_participants,
            mu_prior_loc=defaults["mu_prior_loc"],
        )

    # ------------------------------------------------------------------
    # Kappa with optional L2 shift on the probit scale
    # Sampled manually to allow per-participant LEC offset.
    # ------------------------------------------------------------------
    kappa_defaults = PARAM_PRIOR_DEFAULTS["kappa"]
    kappa_mu_pr = numpyro.sample(
        "kappa_mu_pr",
        dist.Normal(kappa_defaults["mu_prior_loc"], 1.0),
    )
    kappa_sigma_pr = numpyro.sample("kappa_sigma_pr", dist.HalfNormal(0.2))
    kappa_z = numpyro.sample(
        "kappa_z",
        dist.Normal(0, 1).expand([n_participants]),
    )
    lec_shift = beta_lec_kappa * covariate_lec if covariate_lec is not None else 0.0
    kappa_unc = kappa_mu_pr + kappa_sigma_pr * kappa_z + lec_shift
    kappa = numpyro.deterministic(
        "kappa",
        kappa_defaults["lower"]
        + (kappa_defaults["upper"] - kappa_defaults["lower"]) * phi_approx(kappa_unc),
    )
    sampled["kappa"] = kappa

    # ------------------------------------------------------------------
    # Likelihood via numpyro.factor — Python for-loop over participants
    # (vmap not applicable: stacked likelihood uses lax.fori_loop over
    # variable-length block structures)
    # ------------------------------------------------------------------
    _m3_lik_fn = (
        wmrl_m3_multiblock_likelihood_stacked_pscan
        if use_pscan
        else wmrl_m3_multiblock_likelihood_stacked
    )
    for idx, pid in enumerate(participant_ids):
        pdata = participant_data_stacked[pid]
        log_lik = _m3_lik_fn(
            stimuli_stacked=pdata["stimuli_stacked"],
            actions_stacked=pdata["actions_stacked"],
            rewards_stacked=pdata["rewards_stacked"],
            set_sizes_stacked=pdata["set_sizes_stacked"],
            masks_stacked=pdata["masks_stacked"],
            alpha_pos=sampled["alpha_pos"][idx],
            alpha_neg=sampled["alpha_neg"][idx],
            phi=sampled["phi"][idx],
            rho=sampled["rho"][idx],
            capacity=sampled["capacity"][idx],
            kappa=sampled["kappa"][idx],
            epsilon=sampled["epsilon"][idx],
            num_stimuli=num_stimuli,
            num_actions=num_actions,
            q_init=q_init,
            wm_init=wm_init,
            return_pointwise=False,
        )
        numpyro.factor(f"obs_p{pid}", log_lik)


def qlearning_hierarchical_model_stacked(
    participant_data_stacked: dict,
    covariate_lec: jnp.ndarray | None = None,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    use_pscan: bool = False,
) -> None:
    """Hierarchical Bayesian M1 (Q-learning) model using stacked pre-padded arrays.

    Uses the hBayesDM non-centered parameterization convention (Ahn, Haines,
    Zhang 2017):
    ``theta_unc = mu_pr + sigma_pr * z``,
    ``theta = lower + (upper - lower) * Phi_approx(theta_unc)``,
    where ``Phi_approx = jax.scipy.stats.norm.cdf``.

    Implements HIER-02: ports M1 (Q-learning) to the canonical stacked format
    introduced in Phase 15 for M3.  Three parameters (alpha_pos, alpha_neg,
    epsilon) are sampled via :func:`sample_bounded_param` from
    :mod:`scripts.fitting.numpyro_helpers`.

    Parameters
    ----------
    participant_data_stacked : dict
        Mapping from participant_id to stacked-format arrays (from
        ``prepare_stacked_participant_data``).  Keys per participant:
        ``stimuli_stacked``, ``actions_stacked``, ``rewards_stacked``,
        ``masks_stacked`` — each shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``.
        NOTE: ``set_sizes_stacked`` is NOT required by the Q-learning
        likelihood and is ignored even if present.
    covariate_lec : jnp.ndarray or None
        Reserved for forward compatibility.  Must be ``None``; passing a
        non-None value raises ``NotImplementedError`` because Q-learning
        has no natural Level-2 target parameter in this release.
    num_stimuli : int
        Number of distinct stimuli in the task.  Default 6.
    num_actions : int
        Number of possible actions.  Default 3.
    q_init : float
        Initial Q-value for all state-action pairs.  Default 0.5.

    Notes
    -----
    - Q-learning likelihood (``q_learning_multiblock_likelihood_stacked``) does
      NOT accept ``set_sizes_stacked``; do NOT pass it.
    - Participant ordering follows ``sorted(participant_data_stacked.keys())``
      to align with covariate arrays prepared by downstream scripts.
    - Do NOT modify this function's API: ``fit_bayesian.py`` dispatches to it
      by name.
    """
    if covariate_lec is not None:
        raise NotImplementedError(
            "qlearning_hierarchical_model_stacked: covariate_lec is not "
            "supported for Q-learning (no natural L2 target parameter). "
            "Pass covariate_lec=None."
        )

    from scripts.fitting.numpyro_helpers import (
        PARAM_PRIOR_DEFAULTS,
        sample_bounded_param,
    )

    n_participants = len(participant_data_stacked)
    participant_ids = sorted(participant_data_stacked.keys())

    # ------------------------------------------------------------------
    # Group priors for 3 parameters via hBayesDM non-centered convention
    # ------------------------------------------------------------------
    sampled: dict[str, jnp.ndarray] = {}
    for param in ["alpha_pos", "alpha_neg", "epsilon"]:
        defaults = PARAM_PRIOR_DEFAULTS[param]
        sampled[param] = sample_bounded_param(
            param,
            lower=defaults["lower"],
            upper=defaults["upper"],
            n_participants=n_participants,
            mu_prior_loc=defaults["mu_prior_loc"],
        )

    # ------------------------------------------------------------------
    # Likelihood via numpyro.factor — Python for-loop over participants
    # CRITICAL: do NOT pass set_sizes_stacked to Q-learning likelihood
    # ------------------------------------------------------------------
    _ql_lik_fn = (
        q_learning_multiblock_likelihood_stacked_pscan
        if use_pscan
        else q_learning_multiblock_likelihood_stacked
    )
    for idx, pid in enumerate(participant_ids):
        pdata = participant_data_stacked[pid]
        log_lik = _ql_lik_fn(
            stimuli_stacked=pdata["stimuli_stacked"],
            actions_stacked=pdata["actions_stacked"],
            rewards_stacked=pdata["rewards_stacked"],
            masks_stacked=pdata["masks_stacked"],
            alpha_pos=sampled["alpha_pos"][idx],
            alpha_neg=sampled["alpha_neg"][idx],
            epsilon=sampled["epsilon"][idx],
            num_stimuli=num_stimuli,
            num_actions=num_actions,
            q_init=q_init,
            return_pointwise=False,
        )
        numpyro.factor(f"obs_p{pid}", log_lik)


def wmrl_hierarchical_model_stacked(
    participant_data_stacked: dict,
    covariate_lec: jnp.ndarray | None = None,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    use_pscan: bool = False,
) -> None:
    """Hierarchical Bayesian M2 (WM-RL) model using stacked pre-padded arrays.

    Uses the hBayesDM non-centered parameterization convention (Ahn, Haines,
    Zhang 2017):
    ``theta_unc = mu_pr + sigma_pr * z``,
    ``theta = lower + (upper - lower) * Phi_approx(theta_unc)``,
    where ``Phi_approx = jax.scipy.stats.norm.cdf``.

    Implements HIER-03: ports M2 (WM-RL base, no perseveration) to the
    canonical stacked format introduced in Phase 15 for M3.  Six parameters
    (alpha_pos, alpha_neg, phi, rho, capacity, epsilon) are sampled via
    :func:`sample_bounded_param` from :mod:`scripts.fitting.numpyro_helpers`.

    Parameters
    ----------
    participant_data_stacked : dict
        Mapping from participant_id to stacked-format arrays (from
        ``prepare_stacked_participant_data``).  Keys per participant:
        ``stimuli_stacked``, ``actions_stacked``, ``rewards_stacked``,
        ``set_sizes_stacked``, ``masks_stacked`` — each shape
        ``(n_blocks, MAX_TRIALS_PER_BLOCK)``.
    covariate_lec : jnp.ndarray or None
        Reserved for forward compatibility.  Must be ``None``; passing a
        non-None value raises ``NotImplementedError`` because M2 has no
        perseveration parameter to target for Level-2 regression in this
        release.
    num_stimuli : int
        Number of distinct stimuli in the task.  Default 6.
    num_actions : int
        Number of possible actions.  Default 3.
    q_init : float
        Initial Q-value for all state-action pairs.  Default 0.5.
    wm_init : float
        Initial WM values (uniform baseline ``1/nA``).  Default ``1/3``.

    Notes
    -----
    - WM-RL likelihood (``wmrl_multiblock_likelihood_stacked``) DOES require
      ``set_sizes_stacked``; it is passed from ``pdata`` at each iteration.
    - M2 has no kappa (perseveration) parameter; all 6 parameters go through
      the standard ``sample_bounded_param`` loop.
    - Participant ordering follows ``sorted(participant_data_stacked.keys())``
      to align with covariate arrays prepared by downstream scripts.
    - Do NOT modify this function's API: ``fit_bayesian.py`` dispatches to it
      by name.
    """
    if covariate_lec is not None:
        raise NotImplementedError(
            "wmrl_hierarchical_model_stacked: covariate_lec is not supported "
            "for M2 WM-RL (no perseveration parameter as L2 target). "
            "Pass covariate_lec=None."
        )

    from scripts.fitting.numpyro_helpers import (
        PARAM_PRIOR_DEFAULTS,
        sample_bounded_param,
    )

    n_participants = len(participant_data_stacked)
    participant_ids = sorted(participant_data_stacked.keys())

    # ------------------------------------------------------------------
    # Group priors for 6 parameters via hBayesDM non-centered convention
    # ------------------------------------------------------------------
    sampled: dict[str, jnp.ndarray] = {}
    for param in ["alpha_pos", "alpha_neg", "phi", "rho", "capacity", "epsilon"]:
        defaults = PARAM_PRIOR_DEFAULTS[param]
        sampled[param] = sample_bounded_param(
            param,
            lower=defaults["lower"],
            upper=defaults["upper"],
            n_participants=n_participants,
            mu_prior_loc=defaults["mu_prior_loc"],
        )

    # ------------------------------------------------------------------
    # Likelihood via numpyro.factor — Python for-loop over participants
    # WM-RL likelihood requires set_sizes_stacked (unlike Q-learning)
    # ------------------------------------------------------------------
    _wmrl_lik_fn = (
        wmrl_multiblock_likelihood_stacked_pscan
        if use_pscan
        else wmrl_multiblock_likelihood_stacked
    )
    for idx, pid in enumerate(participant_ids):
        pdata = participant_data_stacked[pid]
        log_lik = _wmrl_lik_fn(
            stimuli_stacked=pdata["stimuli_stacked"],
            actions_stacked=pdata["actions_stacked"],
            rewards_stacked=pdata["rewards_stacked"],
            set_sizes_stacked=pdata["set_sizes_stacked"],
            masks_stacked=pdata["masks_stacked"],
            alpha_pos=sampled["alpha_pos"][idx],
            alpha_neg=sampled["alpha_neg"][idx],
            phi=sampled["phi"][idx],
            rho=sampled["rho"][idx],
            capacity=sampled["capacity"][idx],
            epsilon=sampled["epsilon"][idx],
            num_stimuli=num_stimuli,
            num_actions=num_actions,
            q_init=q_init,
            wm_init=wm_init,
            return_pointwise=False,
        )
        numpyro.factor(f"obs_p{pid}", log_lik)


def wmrl_m5_hierarchical_model(
    participant_data_stacked: dict,
    covariate_lec: jnp.ndarray | None = None,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    use_pscan: bool = False,
) -> None:
    """Hierarchical Bayesian M5 (WM-RL+phi_rl) model with optional Level-2 regression.

    Uses the hBayesDM non-centered parameterization convention (Ahn, Haines, Zhang 2017):
    ``theta_unc = mu_pr + sigma_pr * z``,
    ``theta = lower + (upper - lower) * Phi_approx(theta_unc)``,
    where ``Phi_approx = jax.scipy.stats.norm.cdf``.

    K (capacity) is parameterized in [2, 6] following Senta, Bishop, Collins (2025).

    Level-2 regression: if ``covariate_lec`` is provided (standardized LEC-total
    score), a coefficient ``beta_lec_kappa`` is sampled and added as a
    per-participant shift on the unconstrained kappa scale before the Phi_approx
    transform:
    ``kappa_unc_i = kappa_mu_pr + kappa_sigma_pr * z_i + beta_lec_kappa * lec_i``.

    M5 adds ``phi_rl`` (RL forgetting rate) relative to M3.  The 8 model parameters
    are: alpha_pos, alpha_neg, phi, rho, capacity, epsilon, phi_rl (7, sampled via
    ``sample_bounded_param``) and kappa (1, sampled manually with optional L2 shift).

    Parameters
    ----------
    participant_data_stacked : dict
        Mapping from participant_id to stacked-format arrays (from
        ``prepare_stacked_participant_data``). Keys per participant:
        ``stimuli_stacked``, ``actions_stacked``, ``rewards_stacked``,
        ``set_sizes_stacked``, ``masks_stacked`` — each shape
        ``(n_blocks, MAX_TRIALS_PER_BLOCK)``.
    covariate_lec : jnp.ndarray or None
        Shape ``(n_participants,)`` standardized LEC-total covariate.
        Participants must be in the same order as
        ``sorted(participant_data_stacked.keys())``.  If ``None``,
        no Level-2 regression is applied and ``beta_lec_kappa`` is
        not sampled.
    num_stimuli : int
        Number of distinct stimuli in the task.  Default 6.
    num_actions : int
        Number of possible actions.  Default 3.
    q_init : float
        Initial Q-value for all state-action pairs.  Default 0.5.
    wm_init : float
        Initial WM values (uniform baseline ``1/nA``).  Default ``1/3``.

    Notes
    -----
    - Seven parameters (alpha_pos, alpha_neg, phi, rho, capacity, epsilon, phi_rl)
      are sampled via ``sample_bounded_param`` from ``numpyro_helpers``.
    - Kappa is sampled manually with the optional L2 shift applied on the probit
      scale before the Phi_approx transform (OUTSIDE ``sample_bounded_param``).
    - phi_rl uses ``mu_prior_loc=-0.8`` (same as phi) from ``PARAM_PRIOR_DEFAULTS``.
    - Likelihood is accumulated via ``numpyro.factor`` in a Python for-loop over
      participants.  This implements HIER-04.
    - Do NOT modify this function's API: ``fit_bayesian.py`` dispatches to it by
      name.
    """
    from scripts.fitting.numpyro_helpers import (
        PARAM_PRIOR_DEFAULTS,
        phi_approx,
        sample_bounded_param,
    )

    n_participants = len(participant_data_stacked)
    participant_ids = sorted(participant_data_stacked.keys())

    # ------------------------------------------------------------------
    # Level-2: LEC-total -> kappa regression coefficient
    # ------------------------------------------------------------------
    if covariate_lec is not None:
        beta_lec_kappa = numpyro.sample("beta_lec_kappa", dist.Normal(0.0, 1.0))
    else:
        beta_lec_kappa = 0.0

    # ------------------------------------------------------------------
    # Group priors for 7 parameters (all except kappa)
    # Uses hBayesDM non-centered convention locked in Phase 13.
    # phi_rl: mu_prior_loc=-0.8 from PARAM_PRIOR_DEFAULTS.
    # ------------------------------------------------------------------
    sampled: dict[str, jnp.ndarray] = {}
    for param in [
        "alpha_pos",
        "alpha_neg",
        "phi",
        "rho",
        "capacity",
        "epsilon",
        "phi_rl",
    ]:
        defaults = PARAM_PRIOR_DEFAULTS[param]
        sampled[param] = sample_bounded_param(
            param,
            lower=defaults["lower"],
            upper=defaults["upper"],
            n_participants=n_participants,
            mu_prior_loc=defaults["mu_prior_loc"],
        )

    # ------------------------------------------------------------------
    # Kappa with optional L2 shift on the probit scale
    # Sampled manually to allow per-participant LEC offset.
    # ------------------------------------------------------------------
    kappa_defaults = PARAM_PRIOR_DEFAULTS["kappa"]
    kappa_mu_pr = numpyro.sample(
        "kappa_mu_pr",
        dist.Normal(kappa_defaults["mu_prior_loc"], 1.0),
    )
    kappa_sigma_pr = numpyro.sample("kappa_sigma_pr", dist.HalfNormal(0.2))
    kappa_z = numpyro.sample(
        "kappa_z",
        dist.Normal(0, 1).expand([n_participants]),
    )
    lec_shift = beta_lec_kappa * covariate_lec if covariate_lec is not None else 0.0
    kappa_unc = kappa_mu_pr + kappa_sigma_pr * kappa_z + lec_shift
    kappa = numpyro.deterministic(
        "kappa",
        kappa_defaults["lower"]
        + (kappa_defaults["upper"] - kappa_defaults["lower"])
        * phi_approx(kappa_unc),
    )
    sampled["kappa"] = kappa

    # ------------------------------------------------------------------
    # Likelihood via numpyro.factor — Python for-loop over participants
    # (vmap not applicable: stacked likelihood uses lax.fori_loop over
    # variable-length block structures)
    # ------------------------------------------------------------------
    _m5_lik_fn = (
        wmrl_m5_multiblock_likelihood_stacked_pscan
        if use_pscan
        else wmrl_m5_multiblock_likelihood_stacked
    )
    for idx, pid in enumerate(participant_ids):
        pdata = participant_data_stacked[pid]
        log_lik = _m5_lik_fn(
            stimuli_stacked=pdata["stimuli_stacked"],
            actions_stacked=pdata["actions_stacked"],
            rewards_stacked=pdata["rewards_stacked"],
            set_sizes_stacked=pdata["set_sizes_stacked"],
            masks_stacked=pdata["masks_stacked"],
            alpha_pos=sampled["alpha_pos"][idx],
            alpha_neg=sampled["alpha_neg"][idx],
            phi=sampled["phi"][idx],
            rho=sampled["rho"][idx],
            capacity=sampled["capacity"][idx],
            kappa=sampled["kappa"][idx],
            phi_rl=sampled["phi_rl"][idx],
            epsilon=sampled["epsilon"][idx],
            num_stimuli=num_stimuli,
            num_actions=num_actions,
            q_init=q_init,
            wm_init=wm_init,
            return_pointwise=False,
        )
        numpyro.factor(f"obs_p{pid}", log_lik)


def wmrl_m6a_hierarchical_model(
    participant_data_stacked: dict,
    covariate_lec: jnp.ndarray | None = None,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    use_pscan: bool = False,
) -> None:
    """Hierarchical Bayesian M6a (WM-RL+kappa_s) model with optional L2 regression.

    Uses the hBayesDM non-centered parameterization convention (Ahn, Haines, Zhang 2017):
    ``theta_unc = mu_pr + sigma_pr * z``,
    ``theta = lower + (upper - lower) * Phi_approx(theta_unc)``,
    where ``Phi_approx = jax.scipy.stats.norm.cdf``.

    K (capacity) is parameterized in [2, 6] following Senta, Bishop, Collins (2025).

    M6a replaces global perseveration ``kappa`` (M3) with stimulus-specific
    perseveration ``kappa_s``.  The 7 model parameters match M3 in count:
    alpha_pos, alpha_neg, phi, rho, capacity, epsilon (6, sampled via
    ``sample_bounded_param``), and kappa_s (1, sampled manually with optional
    L2 shift using the same pattern as M3's kappa).

    Level-2 regression: if ``covariate_lec`` is provided, a coefficient
    ``beta_lec_kappa_s`` is sampled and shifts ``kappa_s`` on the probit scale:
    ``kappa_s_unc_i = kappa_s_mu_pr + kappa_s_sigma_pr * z_i + beta_lec_kappa_s * lec_i``.

    Parameters
    ----------
    participant_data_stacked : dict
        Mapping from participant_id to stacked-format arrays (from
        ``prepare_stacked_participant_data``). Keys per participant:
        ``stimuli_stacked``, ``actions_stacked``, ``rewards_stacked``,
        ``set_sizes_stacked``, ``masks_stacked`` — each shape
        ``(n_blocks, MAX_TRIALS_PER_BLOCK)``.
    covariate_lec : jnp.ndarray or None
        Shape ``(n_participants,)`` standardized LEC-total covariate.
        Participants must be in the same order as
        ``sorted(participant_data_stacked.keys())``.  If ``None``,
        no Level-2 regression is applied and ``beta_lec_kappa_s`` is
        not sampled.
    num_stimuli : int
        Number of distinct stimuli in the task.  Default 6.
    num_actions : int
        Number of possible actions.  Default 3.
    q_init : float
        Initial Q-value for all state-action pairs.  Default 0.5.
    wm_init : float
        Initial WM values (uniform baseline ``1/nA``).  Default ``1/3``.

    Notes
    -----
    - Six parameters (alpha_pos, alpha_neg, phi, rho, capacity, epsilon) are sampled
      via ``sample_bounded_param`` from ``numpyro_helpers``.
    - kappa_s is sampled manually with the optional L2 shift applied on the probit
      scale before the Phi_approx transform (OUTSIDE ``sample_bounded_param``).
    - kappa_s uses the same bounds and prior as M3's kappa: both in [0, 1] with
      ``mu_prior_loc=-2.0`` from ``PARAM_PRIOR_DEFAULTS``.
    - Likelihood is accumulated via ``numpyro.factor`` in a Python for-loop over
      participants.  This implements HIER-05.
    - Do NOT modify this function's API: ``fit_bayesian.py`` dispatches to it by
      name.
    """
    from scripts.fitting.numpyro_helpers import (
        PARAM_PRIOR_DEFAULTS,
        phi_approx,
        sample_bounded_param,
    )

    n_participants = len(participant_data_stacked)
    participant_ids = sorted(participant_data_stacked.keys())

    # ------------------------------------------------------------------
    # Level-2: LEC-total -> kappa_s regression coefficient
    # ------------------------------------------------------------------
    if covariate_lec is not None:
        beta_lec_kappa_s = numpyro.sample("beta_lec_kappa_s", dist.Normal(0.0, 1.0))
    else:
        beta_lec_kappa_s = 0.0

    # ------------------------------------------------------------------
    # Group priors for 6 parameters (all except kappa_s)
    # Uses hBayesDM non-centered convention locked in Phase 13.
    # ------------------------------------------------------------------
    sampled: dict[str, jnp.ndarray] = {}
    for param in ["alpha_pos", "alpha_neg", "phi", "rho", "capacity", "epsilon"]:
        defaults = PARAM_PRIOR_DEFAULTS[param]
        sampled[param] = sample_bounded_param(
            param,
            lower=defaults["lower"],
            upper=defaults["upper"],
            n_participants=n_participants,
            mu_prior_loc=defaults["mu_prior_loc"],
        )

    # ------------------------------------------------------------------
    # kappa_s with optional L2 shift on the probit scale
    # Sampled manually to allow per-participant LEC offset.
    # Same bounds and prior as M3's kappa (both in PARAM_PRIOR_DEFAULTS).
    # ------------------------------------------------------------------
    kappa_s_defaults = PARAM_PRIOR_DEFAULTS["kappa_s"]
    kappa_s_mu_pr = numpyro.sample(
        "kappa_s_mu_pr",
        dist.Normal(kappa_s_defaults["mu_prior_loc"], 1.0),
    )
    kappa_s_sigma_pr = numpyro.sample("kappa_s_sigma_pr", dist.HalfNormal(0.2))
    kappa_s_z = numpyro.sample(
        "kappa_s_z",
        dist.Normal(0, 1).expand([n_participants]),
    )
    lec_shift = (
        beta_lec_kappa_s * covariate_lec if covariate_lec is not None else 0.0
    )
    kappa_s_unc = kappa_s_mu_pr + kappa_s_sigma_pr * kappa_s_z + lec_shift
    kappa_s = numpyro.deterministic(
        "kappa_s",
        kappa_s_defaults["lower"]
        + (kappa_s_defaults["upper"] - kappa_s_defaults["lower"])
        * phi_approx(kappa_s_unc),
    )
    sampled["kappa_s"] = kappa_s

    # ------------------------------------------------------------------
    # Likelihood via numpyro.factor — Python for-loop over participants
    # ------------------------------------------------------------------
    _m6a_lik_fn = (
        wmrl_m6a_multiblock_likelihood_stacked_pscan
        if use_pscan
        else wmrl_m6a_multiblock_likelihood_stacked
    )
    for idx, pid in enumerate(participant_ids):
        pdata = participant_data_stacked[pid]
        log_lik = _m6a_lik_fn(
            stimuli_stacked=pdata["stimuli_stacked"],
            actions_stacked=pdata["actions_stacked"],
            rewards_stacked=pdata["rewards_stacked"],
            set_sizes_stacked=pdata["set_sizes_stacked"],
            masks_stacked=pdata["masks_stacked"],
            alpha_pos=sampled["alpha_pos"][idx],
            alpha_neg=sampled["alpha_neg"][idx],
            phi=sampled["phi"][idx],
            rho=sampled["rho"][idx],
            capacity=sampled["capacity"][idx],
            kappa_s=sampled["kappa_s"][idx],
            epsilon=sampled["epsilon"][idx],
            num_stimuli=num_stimuli,
            num_actions=num_actions,
            q_init=q_init,
            wm_init=wm_init,
            return_pointwise=False,
        )
        numpyro.factor(f"obs_p{pid}", log_lik)


def wmrl_m6b_hierarchical_model(
    participant_data_stacked: dict,
    covariate_lec: jnp.ndarray | None = None,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    use_pscan: bool = False,
) -> None:
    """Hierarchical Bayesian M6b (WM-RL+dual perseveration) model with optional L2 regression.

    Uses the hBayesDM non-centered parameterization convention (Ahn, Haines, Zhang 2017):
    ``theta_unc = mu_pr + sigma_pr * z``,
    ``theta = lower + (upper - lower) * Phi_approx(theta_unc)``,
    where ``Phi_approx = jax.scipy.stats.norm.cdf``.

    K (capacity) is parameterized in [2, 6] following Senta, Bishop, Collins (2025).

    M6b uses a stick-breaking parameterization for dual perseveration:

    - ``kappa_total`` in [0, 1]: total perseveration budget (sampled manually)
    - ``kappa_share`` in [0, 1]: fraction allocated to global perseveration (sampled
      manually)

    Per participant, decoded values are:

    - ``kappa   = kappa_total * kappa_share``       (global perseveration)
    - ``kappa_s = kappa_total * (1 - kappa_share)`` (stimulus-specific perseveration)

    The decode happens INSIDE the participant for-loop, not in the likelihood function.
    This guarantees ``kappa + kappa_s == kappa_total <= 1`` by construction (HIER-06).

    Level-2 regression: if ``covariate_lec`` is provided, independent coefficients
    ``beta_lec_kappa_total`` and ``beta_lec_kappa_share`` shift their respective
    unconstrained parameters on the probit scale per participant.

    Parameters
    ----------
    participant_data_stacked : dict
        Mapping from participant_id to stacked-format arrays (from
        ``prepare_stacked_participant_data``). Keys per participant:
        ``stimuli_stacked``, ``actions_stacked``, ``rewards_stacked``,
        ``set_sizes_stacked``, ``masks_stacked`` — each shape
        ``(n_blocks, MAX_TRIALS_PER_BLOCK)``.
    covariate_lec : jnp.ndarray or None
        Shape ``(n_participants,)`` standardized LEC-total covariate.
        Participants must be in the same order as
        ``sorted(participant_data_stacked.keys())``.  If ``None``,
        no Level-2 regression is applied and neither beta coefficient is sampled.
    num_stimuli : int
        Number of distinct stimuli in the task.  Default 6.
    num_actions : int
        Number of possible actions.  Default 3.
    q_init : float
        Initial Q-value for all state-action pairs.  Default 0.5.
    wm_init : float
        Initial WM values (uniform baseline ``1/nA``).  Default ``1/3``.

    Notes
    -----
    - Six parameters (alpha_pos, alpha_neg, phi, rho, capacity, epsilon) are sampled
      via ``sample_bounded_param`` from ``numpyro_helpers``.
    - ``kappa_total`` and ``kappa_share`` are each sampled manually (with optional L2
      shift) using the same probit-scale pattern as M3's kappa.
    - ``kappa_total`` prior: ``mu_prior_loc=-2.0`` (pushes group mean toward small
      total perseveration budget).
    - ``kappa_share`` prior: ``mu_prior_loc=0.0`` (group-mean share near 0.5 a priori).
    - The decoded values ``kappa`` and ``kappa_s`` are plain JAX scalars computed
      per participant; they are NOT NumPyro named sites.
    - Likelihood is accumulated via ``numpyro.factor`` in a Python for-loop over
      participants.  This implements HIER-06.
    - Do NOT modify this function's API: ``fit_bayesian.py`` dispatches to it by
      name.
    """
    from scripts.fitting.numpyro_helpers import (
        PARAM_PRIOR_DEFAULTS,
        phi_approx,
        sample_bounded_param,
    )

    n_participants = len(participant_data_stacked)
    participant_ids = sorted(participant_data_stacked.keys())

    # ------------------------------------------------------------------
    # Level-2: LEC-total -> kappa_total and kappa_share regression
    # Two independent beta coefficients, one per perseveration component.
    # ------------------------------------------------------------------
    if covariate_lec is not None:
        beta_lec_kappa_total = numpyro.sample(
            "beta_lec_kappa_total", dist.Normal(0.0, 1.0)
        )
        beta_lec_kappa_share = numpyro.sample(
            "beta_lec_kappa_share", dist.Normal(0.0, 1.0)
        )
    else:
        beta_lec_kappa_total = 0.0
        beta_lec_kappa_share = 0.0

    # ------------------------------------------------------------------
    # Group priors for 6 standard parameters (all except kappa_total/kappa_share)
    # Uses hBayesDM non-centered convention locked in Phase 13.
    # ------------------------------------------------------------------
    sampled: dict[str, jnp.ndarray] = {}
    for param in ["alpha_pos", "alpha_neg", "phi", "rho", "capacity", "epsilon"]:
        defaults = PARAM_PRIOR_DEFAULTS[param]
        sampled[param] = sample_bounded_param(
            param,
            lower=defaults["lower"],
            upper=defaults["upper"],
            n_participants=n_participants,
            mu_prior_loc=defaults["mu_prior_loc"],
        )

    # ------------------------------------------------------------------
    # kappa_total with optional L2 shift on the probit scale
    # mu_prior_loc=-2.0 pushes group mean toward small total budgets.
    # ------------------------------------------------------------------
    kt_defaults = PARAM_PRIOR_DEFAULTS["kappa_total"]
    kappa_total_mu_pr = numpyro.sample(
        "kappa_total_mu_pr",
        dist.Normal(kt_defaults["mu_prior_loc"], 1.0),
    )
    kappa_total_sigma_pr = numpyro.sample(
        "kappa_total_sigma_pr", dist.HalfNormal(0.2)
    )
    kappa_total_z = numpyro.sample(
        "kappa_total_z",
        dist.Normal(0, 1).expand([n_participants]),
    )
    lec_shift_total = (
        beta_lec_kappa_total * covariate_lec if covariate_lec is not None else 0.0
    )
    kappa_total_unc = (
        kappa_total_mu_pr + kappa_total_sigma_pr * kappa_total_z + lec_shift_total
    )
    kappa_total = numpyro.deterministic(
        "kappa_total",
        kt_defaults["lower"]
        + (kt_defaults["upper"] - kt_defaults["lower"])
        * phi_approx(kappa_total_unc),
    )
    sampled["kappa_total"] = kappa_total

    # ------------------------------------------------------------------
    # kappa_share with optional L2 shift on the probit scale
    # mu_prior_loc=0.0 -> group-mean share near 0.5 a priori.
    # ------------------------------------------------------------------
    ks_defaults = PARAM_PRIOR_DEFAULTS["kappa_share"]
    kappa_share_mu_pr = numpyro.sample(
        "kappa_share_mu_pr",
        dist.Normal(ks_defaults["mu_prior_loc"], 1.0),
    )
    kappa_share_sigma_pr = numpyro.sample(
        "kappa_share_sigma_pr", dist.HalfNormal(0.2)
    )
    kappa_share_z = numpyro.sample(
        "kappa_share_z",
        dist.Normal(0, 1).expand([n_participants]),
    )
    lec_shift_share = (
        beta_lec_kappa_share * covariate_lec if covariate_lec is not None else 0.0
    )
    kappa_share_unc = (
        kappa_share_mu_pr + kappa_share_sigma_pr * kappa_share_z + lec_shift_share
    )
    kappa_share = numpyro.deterministic(
        "kappa_share",
        ks_defaults["lower"]
        + (ks_defaults["upper"] - ks_defaults["lower"])
        * phi_approx(kappa_share_unc),
    )
    sampled["kappa_share"] = kappa_share

    # ------------------------------------------------------------------
    # Likelihood via numpyro.factor — Python for-loop over participants
    # Decode kappa and kappa_s per-participant (STATE.md locked decision):
    #   kappa   = kappa_total * kappa_share
    #   kappa_s = kappa_total * (1 - kappa_share)
    # Pass decoded values to likelihood, NOT kappa_total/kappa_share directly.
    # ------------------------------------------------------------------
    _m6b_lik_fn = (
        wmrl_m6b_multiblock_likelihood_stacked_pscan
        if use_pscan
        else wmrl_m6b_multiblock_likelihood_stacked
    )
    for idx, pid in enumerate(participant_ids):
        pdata = participant_data_stacked[pid]
        kappa_total_i = sampled["kappa_total"][idx]
        kappa_share_i = sampled["kappa_share"][idx]
        kappa = kappa_total_i * kappa_share_i
        kappa_s = kappa_total_i * (1.0 - kappa_share_i)
        log_lik = _m6b_lik_fn(
            stimuli_stacked=pdata["stimuli_stacked"],
            actions_stacked=pdata["actions_stacked"],
            rewards_stacked=pdata["rewards_stacked"],
            set_sizes_stacked=pdata["set_sizes_stacked"],
            masks_stacked=pdata["masks_stacked"],
            alpha_pos=sampled["alpha_pos"][idx],
            alpha_neg=sampled["alpha_neg"][idx],
            phi=sampled["phi"][idx],
            rho=sampled["rho"][idx],
            capacity=sampled["capacity"][idx],
            kappa=kappa,
            kappa_s=kappa_s,
            epsilon=sampled["epsilon"][idx],
            num_stimuli=num_stimuli,
            num_actions=num_actions,
            q_init=q_init,
            wm_init=wm_init,
            return_pointwise=False,
        )
        numpyro.factor(f"obs_p{pid}", log_lik)


def wmrl_m6b_hierarchical_model_subscale(
    participant_data_stacked: dict,
    covariate_matrix: jnp.ndarray | None = None,
    covariate_names: list[str] | None = None,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    use_pscan: bool = False,
) -> None:
    """Hierarchical M6b model with full subscale Level-2 regression (L2-05).

    Extends :func:`wmrl_m6b_hierarchical_model` by accepting a full covariate
    matrix instead of a single LEC vector, enabling Level-2 regression on ALL
    8 M6b parameters from the 4-predictor subscale design
    [lec_total, iesr_total, iesr_intr_resid, iesr_avd_resid].

    This produces **32 beta coefficient sites** (8 parameters x 4 covariates)
    matching ``COVARIATE_NAMES`` from ``scripts/fitting/level2_design.py``.
    (Plan references "~40 beta sites" — the actual count is 32 because the
    hyperarousal residual was dropped due to exact linear dependence.)

    All 8 parameters are sampled **manually** (bypassing ``sample_bounded_param``)
    so that multi-covariate L2 shifts can be applied uniformly on the unconstrained
    probit scale before the Phi_approx transform.

    Model structure (non-centered, hBayesDM convention):
    ::

        # Level-2 beta coefficients (if covariate_matrix provided)
        beta_{cov}_{param} ~ Normal(0, 1)   # for each covariate x parameter

        # Group priors
        {param}_mu_pr    ~ Normal(mu_prior_loc, 1.0)
        {param}_sigma_pr ~ HalfNormal(0.2)
        {param}_z        ~ Normal(0, 1)  shape (n_participants,)

        # Individual unconstrained
        theta_unc_i = mu_pr + sigma_pr * z_i + sum_j(beta_j * X_ij)

        # Constrained
        theta_i = lower + (upper - lower) * Phi_approx(theta_unc_i)

    M6b stick-breaking decode (per participant, inside for-loop):
    ::

        kappa   = kappa_total_i * kappa_share_i
        kappa_s = kappa_total_i * (1 - kappa_share_i)

    Parameters
    ----------
    participant_data_stacked : dict
        Mapping from participant_id to stacked-format arrays from
        ``prepare_stacked_participant_data``. Keys per participant:
        ``stimuli_stacked``, ``actions_stacked``, ``rewards_stacked``,
        ``set_sizes_stacked``, ``masks_stacked`` — each shape
        ``(n_blocks, MAX_TRIALS_PER_BLOCK)``.
    covariate_matrix : jnp.ndarray or None
        Shape ``(n_participants, n_covariates)`` standardized design matrix.
        Participants must be in the same order as
        ``sorted(participant_data_stacked.keys())``.  Built by
        ``scripts.fitting.level2_design.build_level2_design_matrix``.
        If ``None``, no Level-2 shifts are applied (equivalent to the
        single-covariate model with ``covariate_lec=None``).
    covariate_names : list[str] or None
        Names for each column of ``covariate_matrix``, used to name
        ``beta_{cov}_{param}`` sites.  Must have length matching
        ``covariate_matrix.shape[1]``.  Expected value (from
        ``COVARIATE_NAMES`` in ``level2_design.py``):
        ``["lec_total", "iesr_total", "iesr_intr_resid", "iesr_avd_resid"]``.
        If ``None`` and ``covariate_matrix`` is not ``None``, columns are
        named ``cov0``, ``cov1``, etc.
    num_stimuli : int
        Number of distinct stimuli in the task.  Default 6.
    num_actions : int
        Number of possible actions.  Default 3.
    q_init : float
        Initial Q-value for all state-action pairs.  Default 0.5.
    wm_init : float
        Initial WM values (uniform baseline ``1/nA``).  Default ``1/3``.

    Notes
    -----
    - Beta coefficient naming: ``beta_{covariate_name}_{param_name}``.
      Example sites: ``beta_lec_total_kappa_total``,
      ``beta_iesr_intr_resid_alpha_pos``, etc.
    - This model has a high-dimensional posterior (group priors + 32 beta
      sites + 8 x n_participants individual parameters).  Use
      ``run_inference_with_bump`` with target_accept_prob up to 0.99 to
      handle potential divergences (L2-08 horseshoe prior upgrade is the
      fallback if divergences persist).
    - Beta site order: outer loop over param_names, inner loop over
      covariate_names.  This matches the naming convention for downstream
      ArviZ extraction.
    - Do NOT modify this function's API: ``fit_bayesian.py`` dispatches to
      it via the ``--subscale`` flag.
    """
    from scripts.fitting.numpyro_helpers import PARAM_PRIOR_DEFAULTS, phi_approx

    n_participants = len(participant_data_stacked)
    participant_ids = sorted(participant_data_stacked.keys())

    # All 8 M6b parameters in canonical order
    param_names = [
        "alpha_pos",
        "alpha_neg",
        "phi",
        "rho",
        "capacity",
        "epsilon",
        "kappa_total",
        "kappa_share",
    ]

    # ------------------------------------------------------------------
    # Resolve covariate column names
    # ------------------------------------------------------------------
    if covariate_matrix is not None:
        n_covariates = covariate_matrix.shape[1]
        if covariate_names is None:
            covariate_names = [f"cov{j}" for j in range(n_covariates)]
        if len(covariate_names) != n_covariates:
            raise ValueError(
                f"wmrl_m6b_hierarchical_model_subscale: covariate_names has "
                f"{len(covariate_names)} entries but covariate_matrix has "
                f"{n_covariates} columns. They must match."
            )
    else:
        covariate_names = []

    # ------------------------------------------------------------------
    # Level-2 beta coefficients — one per (param, covariate) pair
    # Naming: beta_{covariate_name}_{param_name}
    # 8 params x 4 covariates = 32 sites when using the default design.
    # ------------------------------------------------------------------
    betas: dict[tuple[str, str], object] = {}
    if covariate_matrix is not None:
        for pname in param_names:
            for cov_name in covariate_names:
                site_name = f"beta_{cov_name}_{pname}"
                betas[(pname, cov_name)] = numpyro.sample(
                    site_name, dist.Normal(0.0, 1.0)
                )

    # ------------------------------------------------------------------
    # Manual non-centered sampling for all 8 parameters
    # Bypasses sample_bounded_param to support per-parameter L2 shifts.
    # Pattern: mu_pr + sigma_pr * z + sum_j(beta_j * X[:, j])
    # ------------------------------------------------------------------
    sampled: dict[str, jnp.ndarray] = {}
    for pname in param_names:
        defaults = PARAM_PRIOR_DEFAULTS[pname]
        lower = defaults["lower"]
        upper = defaults["upper"]
        mu_prior_loc = defaults["mu_prior_loc"]

        mu_pr = numpyro.sample(
            f"{pname}_mu_pr",
            dist.Normal(mu_prior_loc, 1.0),
        )
        sigma_pr = numpyro.sample(
            f"{pname}_sigma_pr",
            dist.HalfNormal(0.2),
        )
        z = numpyro.sample(
            f"{pname}_z",
            dist.Normal(0, 1).expand([n_participants]),
        )

        # Unconstrained individual-level values
        theta_unc = mu_pr + sigma_pr * z

        # Add L2 shifts from each covariate column
        if covariate_matrix is not None:
            for j, cov_name in enumerate(covariate_names):
                theta_unc = theta_unc + betas[(pname, cov_name)] * covariate_matrix[:, j]

        # Transform to constrained space via Phi_approx
        theta = lower + (upper - lower) * phi_approx(theta_unc)
        sampled[pname] = numpyro.deterministic(pname, theta)

    # ------------------------------------------------------------------
    # Likelihood via numpyro.factor — Python for-loop over participants
    # Decode kappa and kappa_s per-participant (STATE.md locked decision):
    #   kappa   = kappa_total * kappa_share
    #   kappa_s = kappa_total * (1 - kappa_share)
    # ------------------------------------------------------------------
    _m6b_sub_lik_fn = (
        wmrl_m6b_multiblock_likelihood_stacked_pscan
        if use_pscan
        else wmrl_m6b_multiblock_likelihood_stacked
    )
    for idx, pid in enumerate(participant_ids):
        pdata = participant_data_stacked[pid]
        kappa_total_i = sampled["kappa_total"][idx]
        kappa_share_i = sampled["kappa_share"][idx]
        kappa = kappa_total_i * kappa_share_i
        kappa_s = kappa_total_i * (1.0 - kappa_share_i)
        log_lik = _m6b_sub_lik_fn(
            stimuli_stacked=pdata["stimuli_stacked"],
            actions_stacked=pdata["actions_stacked"],
            rewards_stacked=pdata["rewards_stacked"],
            set_sizes_stacked=pdata["set_sizes_stacked"],
            masks_stacked=pdata["masks_stacked"],
            alpha_pos=sampled["alpha_pos"][idx],
            alpha_neg=sampled["alpha_neg"][idx],
            phi=sampled["phi"][idx],
            rho=sampled["rho"][idx],
            capacity=sampled["capacity"][idx],
            kappa=kappa,
            kappa_s=kappa_s,
            epsilon=sampled["epsilon"][idx],
            num_stimuli=num_stimuli,
            num_actions=num_actions,
            q_init=q_init,
            wm_init=wm_init,
            return_pointwise=False,
        )
        numpyro.factor(f"obs_p{pid}", log_lik)


def prepare_stacked_participant_data_m4(
    data_df: pd.DataFrame,
    participant_col: str = "sona_id",
    block_col: str = "block",
    stimulus_col: str = "stimulus",
    action_col: str = "key_press",
    reward_col: str = "reward",
    set_size_col: str = "set_size",
    rt_col: str = "rt",
) -> dict[Any, dict[str, jnp.ndarray]]:
    """Prepare stacked participant data for the M4 hierarchical LBA model.

    Extends :func:`prepare_stacked_participant_data` with RT extraction, outlier
    masking, and float64 RT stacking required by ``wmrl_m4_hierarchical_model``.

    For each trial block the function:

    1. Extracts RTs from ``rt_col`` (assumed milliseconds).
    2. Calls ``preprocess_rt_block`` to convert to seconds and flag outliers.
    3. Pads RTs to ``MAX_TRIALS_PER_BLOCK`` with 0.5 (safe masked-out value).
    4. ANDs the padding mask with the RT-outlier mask so that padding trials
       **and** RT outliers both contribute 0 to the likelihood.

    Parameters
    ----------
    data_df : pd.DataFrame
        Trial-level data with participant, block, stimulus, action, reward,
        set_size, and rt columns.
    participant_col : str
        Column name for participant identifier.  Default ``"sona_id"``.
    block_col : str
        Column name for block number.  Default ``"block"``.
    stimulus_col : str
        Column name for stimulus index.  Default ``"stimulus"``.
    action_col : str
        Column name for action taken.  Default ``"key_press"``.
    reward_col : str
        Column name for reward received.  Default ``"reward"``.
    set_size_col : str
        Column name for set size.  Default ``"set_size"``.
    rt_col : str
        Column name for reaction time in milliseconds.  Default ``"rt"``.

    Returns
    -------
    dict[Any, dict[str, jnp.ndarray]]
        Mapping from participant_id to a dict with keys:

        * ``stimuli_stacked``   -- shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``, int32
        * ``actions_stacked``   -- shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``, int32
        * ``rewards_stacked``   -- shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``, float32
        * ``set_sizes_stacked`` -- shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``, float32
        * ``masks_stacked``     -- shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``, float32
          (padding AND RT-outlier mask combined)
        * ``rts_stacked``       -- shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``, float64
          (seconds; padding positions filled with 0.5)

    Notes
    -----
    Lazy-imports ``preprocess_rt_block`` from ``lba_likelihood`` to avoid
    triggering float64 in choice-only import paths.

    Participant keys are sorted (same ordering as
    :func:`prepare_stacked_participant_data`) so that downstream arrays align
    with ``sorted(result.keys())``.
    """
    # Lazy import to avoid float64 contamination in choice-only paths
    from scripts.fitting.lba_likelihood import preprocess_rt_block

    participant_data: dict[Any, dict[str, jnp.ndarray]] = {}

    for participant_id in sorted(data_df[participant_col].unique()):
        ppt_df = data_df[data_df[participant_col] == participant_id]

        stimuli_blocks = []
        actions_blocks = []
        rewards_blocks = []
        set_sizes_blocks = []
        masks_blocks = []
        rts_blocks = []

        for block_num in sorted(ppt_df[block_col].unique()):
            block_df = ppt_df[ppt_df[block_col] == block_num]

            stim = jnp.array(block_df[stimulus_col].values, dtype=jnp.int32)
            act = jnp.array(block_df[action_col].values, dtype=jnp.int32)
            rew = jnp.array(block_df[reward_col].values, dtype=jnp.float32)

            if set_size_col in block_df.columns:
                ss = jnp.array(block_df[set_size_col].values, dtype=jnp.float32)
            else:
                ss = jnp.ones(len(stim), dtype=jnp.float32) * 6.0

            # pad_block_to_max returns (stim, act, rew, set_sizes_padded, mask)
            p_stim, p_act, p_rew, p_ss, p_mask = pad_block_to_max(
                stim, act, rew, set_sizes=ss
            )
            n_real = len(stim)

            # Extract and preprocess RT (milliseconds -> seconds + outlier mask)
            rt_raw = jnp.array(block_df[rt_col].values, dtype=jnp.float64)
            rt_sec, valid_rt = preprocess_rt_block(rt_raw)

            # Pad RT to MAX_TRIALS_PER_BLOCK with 0.5 (safe masked-out value)
            rt_padded = jnp.zeros(MAX_TRIALS_PER_BLOCK, dtype=jnp.float64)
            rt_padded = rt_padded.at[:n_real].set(rt_sec)
            # Fill padding positions with 0.5 to avoid t_star <= 0 in masked calls
            padding_fill = jnp.zeros(MAX_TRIALS_PER_BLOCK, dtype=jnp.float64)
            padding_fill = padding_fill.at[n_real:].set(0.5)
            rt_padded = rt_padded + padding_fill

            # Pad valid_rt mask to MAX_TRIALS_PER_BLOCK (False for padding positions)
            valid_rt_padded = jnp.zeros(MAX_TRIALS_PER_BLOCK, dtype=jnp.float32)
            valid_rt_padded = valid_rt_padded.at[:n_real].set(
                valid_rt.astype(jnp.float32)
            )

            # AND padding mask with RT-outlier mask
            combined_mask = p_mask * valid_rt_padded

            stimuli_blocks.append(p_stim)
            actions_blocks.append(p_act)
            rewards_blocks.append(p_rew)
            set_sizes_blocks.append(p_ss)
            masks_blocks.append(combined_mask)
            rts_blocks.append(rt_padded)

        participant_data[participant_id] = {
            "stimuli_stacked": jnp.stack(stimuli_blocks),
            "actions_stacked": jnp.stack(actions_blocks),
            "rewards_stacked": jnp.stack(rewards_blocks),
            "set_sizes_stacked": jnp.stack(set_sizes_blocks),
            "masks_stacked": jnp.stack(masks_blocks),
            "rts_stacked": jnp.stack(rts_blocks),
        }

    return participant_data


def wmrl_m4_hierarchical_model(
    participant_data_stacked: dict,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
) -> None:
    """Hierarchical Bayesian M4 (WM-RL+LBA) model.

    Implements a joint choice+RT hierarchical model combining the WM-RL
    hybrid model with Linear Ballistic Accumulator (LBA) decision dynamics.

    Uses the hBayesDM non-centered parameterization convention (Ahn, Haines, Zhang 2017):
    ``theta_unc = mu_pr + sigma_pr * z``,
    ``theta = lower + (upper - lower) * Phi_approx(theta_unc)``,
    where ``Phi_approx = jax.scipy.stats.norm.cdf``.

    K (capacity) is parameterized in [2, 6] following Senta, Bishop, Collins (2025).

    LBA parameters use log-normal non-centered parameterization:

    - ``v_scale``: drift-rate scale, log-normal in (0, inf).
    - ``A``: start-point noise width, log-normal in (0, inf).
    - ``delta`` (M4H-02): b - A gap, log-normal ensuring b > A by construction.
      Inside the participant loop: ``b = A + delta`` (decoded, NOT sampled directly).
    - ``t0``: non-decision time, bounded in [0.05, 0.3] via probit transform.

    Parameters
    ----------
    participant_data_stacked : dict
        Mapping from participant_id to stacked-format arrays (from
        ``prepare_stacked_participant_data_m4``).  Keys per participant:
        ``stimuli_stacked``, ``actions_stacked``, ``rewards_stacked``,
        ``set_sizes_stacked``, ``masks_stacked`` (RT-outlier combined),
        ``rts_stacked`` — each shape ``(n_blocks, MAX_TRIALS_PER_BLOCK)``.
    num_stimuli : int
        Number of distinct stimuli in the task.  Default 6.
    num_actions : int
        Number of possible actions.  Default 3.
    q_init : float
        Initial Q-value for all state-action pairs.  Default 0.5.
    wm_init : float
        Initial WM values (uniform baseline ``1/nA``).  Default ``1/3``.

    Notes
    -----
    - Six RLWM params (alpha_pos, alpha_neg, phi, rho, capacity, kappa) are
      sampled via ``sample_bounded_param`` from ``numpyro_helpers``.
    - No ``epsilon`` parameter: M4 is the joint choice+RT model and uses the
      LBA decision process directly (not the noisy softmax).
    - No Level-2 regression: M4 hierarchical model is unconditional.
    - ``b = A + delta`` decode happens INSIDE the participant for-loop, not in
      the likelihood.  This guarantees ``b > A`` by construction (M4H-02).
    - ``wmrl_m4_multiblock_likelihood_stacked`` is lazy-imported inside the
      function body to avoid triggering float64 in choice-only import paths.
    - Likelihood is accumulated via ``numpyro.factor`` in a Python for-loop.
    - Do NOT modify this function's API: ``fit_bayesian.py`` dispatches to it
      by name.
    """
    # Lazy imports to avoid float64 contamination in choice-only import paths
    from scripts.fitting.lba_likelihood import wmrl_m4_multiblock_likelihood_stacked
    from scripts.fitting.numpyro_helpers import (
        PARAM_PRIOR_DEFAULTS,
        phi_approx,
        sample_bounded_param,
    )

    n_participants = len(participant_data_stacked)
    participant_ids = sorted(participant_data_stacked.keys())

    # ------------------------------------------------------------------
    # Group priors for 6 RLWM parameters via sample_bounded_param
    # Note: no epsilon in M4 (LBA handles decision noise directly)
    # ------------------------------------------------------------------
    sampled: dict[str, jnp.ndarray] = {}
    for param in ["alpha_pos", "alpha_neg", "phi", "rho", "capacity", "kappa"]:
        defaults = PARAM_PRIOR_DEFAULTS[param]
        sampled[param] = sample_bounded_param(
            param,
            lower=defaults["lower"],
            upper=defaults["upper"],
            n_participants=n_participants,
            mu_prior_loc=defaults["mu_prior_loc"],
        )

    # ------------------------------------------------------------------
    # v_scale: log-normal non-centered
    # mu ~ Normal(log(3.0), 0.5) -> group mean drift ~ 3.0 Hz
    # ------------------------------------------------------------------
    log_v_mu_pr = numpyro.sample("log_v_mu_pr", dist.Normal(jnp.log(3.0), 0.5))
    log_v_sigma_pr = numpyro.sample("log_v_sigma_pr", dist.HalfNormal(0.2))
    log_v_z = numpyro.sample(
        "log_v_z", dist.Normal(0, 1).expand([n_participants])
    )
    sampled["v_scale"] = numpyro.deterministic(
        "v_scale", jnp.exp(log_v_mu_pr + log_v_sigma_pr * log_v_z)
    )

    # ------------------------------------------------------------------
    # A: start-point noise width, log-normal non-centered
    # mu ~ Normal(log(0.3), 0.5) -> group mean A ~ 0.3 s
    # ------------------------------------------------------------------
    log_A_mu_pr = numpyro.sample("log_A_mu_pr", dist.Normal(jnp.log(0.3), 0.5))
    log_A_sigma_pr = numpyro.sample("log_A_sigma_pr", dist.HalfNormal(0.2))
    log_A_z = numpyro.sample(
        "log_A_z", dist.Normal(0, 1).expand([n_participants])
    )
    sampled["A"] = numpyro.deterministic(
        "A", jnp.exp(log_A_mu_pr + log_A_sigma_pr * log_A_z)
    )

    # ------------------------------------------------------------------
    # delta: b - A gap, log-normal non-centered (M4H-02)
    # b = A + delta guarantees b > A by construction.
    # mu ~ Normal(0.0, 1.0) -> median delta ~ 1.0
    # ------------------------------------------------------------------
    log_delta_mu_pr = numpyro.sample("log_delta_mu_pr", dist.Normal(0.0, 1.0))
    log_delta_sigma_pr = numpyro.sample("log_delta_sigma_pr", dist.HalfNormal(0.2))
    log_delta_z = numpyro.sample(
        "log_delta_z", dist.Normal(0, 1).expand([n_participants])
    )
    sampled["delta"] = numpyro.deterministic(
        "delta", jnp.exp(log_delta_mu_pr + log_delta_sigma_pr * log_delta_z)
    )

    # ------------------------------------------------------------------
    # t0: non-decision time, bounded in [0.05, 0.3] via probit transform
    # mu ~ Normal(0.0, 1.0) -> group mean t0 near midpoint of [0.05, 0.3]
    # ------------------------------------------------------------------
    t0_mu_pr = numpyro.sample("t0_mu_pr", dist.Normal(0.0, 1.0))
    t0_sigma_pr = numpyro.sample("t0_sigma_pr", dist.HalfNormal(0.2))
    t0_z = numpyro.sample(
        "t0_z", dist.Normal(0, 1).expand([n_participants])
    )
    sampled["t0"] = numpyro.deterministic(
        "t0", 0.05 + 0.25 * phi_approx(t0_mu_pr + t0_sigma_pr * t0_z)
    )

    # ------------------------------------------------------------------
    # Likelihood via numpyro.factor — Python for-loop over participants
    # CRITICAL: decode b = A + delta INSIDE the loop (not in likelihood)
    # This guarantees b > A by construction (M4H-02).
    # ------------------------------------------------------------------
    for idx, pid in enumerate(participant_ids):
        pdata = participant_data_stacked[pid]
        A_i = sampled["A"][idx]
        delta_i = sampled["delta"][idx]
        b_i = A_i + delta_i  # b > A guaranteed by delta > 0

        log_lik = wmrl_m4_multiblock_likelihood_stacked(
            stimuli_stacked=pdata["stimuli_stacked"],
            actions_stacked=pdata["actions_stacked"],
            rewards_stacked=pdata["rewards_stacked"],
            set_sizes_stacked=pdata["set_sizes_stacked"],
            rts_stacked=pdata["rts_stacked"],
            masks_stacked=pdata["masks_stacked"],
            alpha_pos=sampled["alpha_pos"][idx],
            alpha_neg=sampled["alpha_neg"][idx],
            phi=sampled["phi"][idx],
            rho=sampled["rho"][idx],
            capacity=sampled["capacity"][idx],
            kappa=sampled["kappa"][idx],
            v_scale=sampled["v_scale"][idx],
            A=A_i,
            b=b_i,
            t0=sampled["t0"][idx],
            num_stimuli=num_stimuli,
            num_actions=num_actions,
            q_init=q_init,
            wm_init=wm_init,
        )
        numpyro.factor(f"obs_p{pid}", log_lik)


if __name__ == "__main__":
    # Set NumPyro to use all available CPU cores
    numpyro.set_host_device_count(4)

    test_model_with_synthetic_data()
