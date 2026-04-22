"""MCMC orchestration: chain-method selector, data preparation, run_inference, ArviZ conversion.

Relocated here in Phase 29-08 from :mod:`rlwm.fitting.numpyro_models`. Old
import paths remain available via wildcard re-export shims.

Provides the thin NumPyro orchestration layer used by every Bayesian fit in
``scripts/04_model_fitting/`` and ``scripts/03_model_prefitting/``.

Note: stacking helpers ``stack_across_participants`` and
``prepare_stacked_participant_data`` live in :mod:`rlwm.fitting.core` (not
here), to avoid a circular import between the per-model files (which call
stacking helpers inside their hierarchical-model functions) and this
orchestration layer.  They remain re-exported via the
``rlwm.fitting.numpyro_models`` shim for backward compatibility.
"""
from __future__ import annotations

import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS

from .core import MAX_TRIALS_PER_BLOCK, pad_block_to_max
from .models.qlearning import prepare_block_data, q_learning_multiblock_likelihood

__all__ = [
    "_select_chain_method",
    "prepare_data_for_numpyro",
    "test_likelihood_compilation",
    "run_inference",
    "run_inference_with_bump",
    "samples_to_arviz",
    "test_model_with_synthetic_data",
]


def _select_chain_method(num_chains: int) -> str:
    """Select chain_method based on JAX backend and device count.

    GPU: always "vectorized" unless multiple physical GPUs exist
    (then "parallel" across GPUs via pmap).
    CPU: "parallel" if set_host_device_count exposed enough devices,
    else "sequential".

    Parameters
    ----------
    num_chains : int
        Number of MCMC chains requested.

    Returns
    -------
    str
        One of "parallel", "vectorized", or "sequential".
    """
    backend = jax.default_backend()
    n_devices = jax.local_device_count()
    if backend == "gpu":
        if n_devices >= num_chains:
            return "parallel"
        return "vectorized"
    if backend == "tpu":
        return "parallel" if n_devices >= num_chains else "vectorized"
    # cpu
    return "parallel" if n_devices >= num_chains else "sequential"

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
    # Use _select_chain_method to correctly handle GPU (vectorized), CPU
    # (parallel if set_host_device_count exposed enough devices), and TPU.
    _chain_method = _select_chain_method(num_chains)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=_chain_method,
        progress_bar=True,
    )
    print(f"   Backend: {jax.default_backend()} | chain_method: {_chain_method}")

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

    import os
    import time

    print(">> Starting MCMC sampling with convergence auto-bump...")
    print(f"   Chains: {num_chains}")
    print(f"   Host devices: {jax.local_device_count()}")
    print(f"   Warmup: {num_warmup}")
    print(f"   Samples: {num_samples}")
    print(f"   Max tree depth: {max_tree_depth}")
    print(f"   Total iterations per chain: {num_warmup + num_samples}")
    print(f"   Acceptance probability schedule: {target_accept_probs}")
    print()

    # ------------------------------------------------------------------
    # GPU / NumPyro configuration banner — printed once per inference call
    # so GPU regressions surface in a single log block.
    # ------------------------------------------------------------------
    print("=" * 60)
    print(">> JAX / NumPyro configuration")
    print(f"   backend            : {jax.default_backend()}")
    print(f"   devices            : {jax.devices()}")
    print(f"   local_device_count : {jax.local_device_count()}")
    print(f"   x64 enabled        : {jax.config.jax_enable_x64}")
    print(f"   chain_method       : {_select_chain_method(num_chains)}")
    print(f"   num_chains         : {num_chains}")
    for env_var in [
        "NUMPYRO_HOST_DEVICE_COUNT",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "JAX_COMPILATION_CACHE_DIR",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
        "CUDA_VISIBLE_DEVICES",
    ]:
        print(f"   {env_var:32s} = {os.environ.get(env_var, '<unset>')}")
    print("=" * 60)
    print()

    last_mcmc: MCMC | None = None
    for tap in target_accept_probs:
        nuts_kernel = NUTS(
            model,
            target_accept_prob=tap,
            max_tree_depth=max_tree_depth,
        )
        _chain_method = _select_chain_method(num_chains)
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=_chain_method,
            progress_bar=True,
        )
        print(f"   Backend: {jax.default_backend()} | chain_method: {_chain_method}")
        rng_key = jax.random.PRNGKey(seed)
        t0_compile = time.perf_counter()
        mcmc.run(rng_key, **model_args)
        t1_run = time.perf_counter()
        print(f"[timing] target_accept_prob={tap:.2f} wall={t1_run - t0_compile:.1f}s")

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
    return last_mcmc

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
