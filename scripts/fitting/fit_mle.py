"""
Maximum Likelihood Estimation for RLWM Models

Following Senta et al. (2025) methodology:
- Individual fits with 50 starting points (Latin Hypercube Sampling)
- ScipyBoundedMinimize (L-BFGS-B) for bounded optimization with JAX gradients
- AIC/BIC for model comparison
- Group statistics: mean +/- SEM across participants

Usage:
    python scripts/fitting/fit_mle.py --model qlearning --data output/task_trials_long.csv
    python scripts/fitting/fit_mle.py --model wmrl --data output/task_trials_long.csv
    python scripts/fitting/fit_mle.py --model wmrl_m3 --data output/task_trials_long.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Optional memory tracking
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Cross-platform memory tracking fallback (Unix)
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

def timestamp() -> str:
    """Return current time as HH:MM:SS string for logging."""
    return datetime.now().strftime('%H:%M:%S')

def log_memory_usage(label: str, verbose: bool = True) -> float:
    """
    Log current memory usage for debugging OOM issues.

    Uses psutil if available (cross-platform), falls back to resource module (Unix).

    Args:
        label: Description of checkpoint (e.g., "START", "PRE-JIT", "POST-JIT")
        verbose: If True, print the memory info

    Returns:
        Memory usage in MB, or 0 if unavailable
    """
    mem_mb = 0.0
    mem_source = "unavailable"

    if HAS_PSUTIL:
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024**2
        mem_source = "psutil"
    elif HAS_RESOURCE:
        # resource.getrusage returns KB on Linux, bytes on macOS
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in KB on Linux
        mem_mb = usage.ru_maxrss / 1024
        mem_source = "resource"

    if verbose and mem_mb > 0:
        print(f"[MEMORY] {label}: {mem_mb:.1f}MB ({mem_source})")

    return mem_mb

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxopt import ScipyBoundedMinimize

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import config for exclusions
from config import EXCLUDED_PARTICIPANTS

# Import JAX likelihood functions
from scripts.fitting.jax_likelihoods import (
    MAX_BLOCKS,
    MAX_TRIALS_PER_BLOCK,
    pad_block_to_max,
    pad_blocks_to_max,
    q_learning_multiblock_likelihood,
    q_learning_multiblock_likelihood_stacked,  # Fast stacked version
    wmrl_m3_multiblock_likelihood,
    wmrl_m3_multiblock_likelihood_stacked,  # Fast stacked version
    wmrl_m5_multiblock_likelihood,
    wmrl_m5_multiblock_likelihood_stacked,  # Fast stacked version (M5)
    wmrl_multiblock_likelihood,
    wmrl_multiblock_likelihood_stacked,  # Fast stacked version
)

# Import MLE utilities
from scripts.fitting.mle_utils import (
    QLEARNING_BOUNDS,
    QLEARNING_PARAMS,
    WMRL_BOUNDS,
    WMRL_M3_BOUNDS,
    WMRL_M3_PARAMS,
    WMRL_M5_BOUNDS,
    WMRL_M5_PARAMS,
    WMRL_PARAMS,
    check_at_bounds,
    check_convergence,
    check_gradient_norm,
    compute_aic,
    compute_aicc,
    compute_bic,
    compute_confidence_intervals,
    compute_hessian_diagnostics,
    # Diagnostic functions
    compute_pseudo_r2,
    get_high_correlations,
    get_n_params,
    jax_bounded_to_unconstrained_qlearning,
    jax_bounded_to_unconstrained_wmrl,
    jax_bounded_to_unconstrained_wmrl_m3,
    jax_bounded_to_unconstrained_wmrl_m5,
    jax_unconstrained_to_params_qlearning,
    jax_unconstrained_to_params_wmrl,
    jax_unconstrained_to_params_wmrl_m3,
    jax_unconstrained_to_params_wmrl_m5,
    params_to_unconstrained,
    sample_lhs_starts,
    summarize_all_parameters,
)

# Suppress JAX warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# JAX Compilation Warmup (for parallel execution)
# =============================================================================

def warmup_jax_compilation(model: str, verbose: bool = True):
    """
    Pre-compile JAX functions before spawning parallel workers.

    This populates the JAX persistent cache (JAX_COMPILATION_CACHE_DIR)
    so worker processes can read cached compilations instead of
    each compiling independently. This dramatically reduces overhead
    when using joblib for parallel fitting.

    Args:
        model: 'qlearning', 'wmrl', or 'wmrl_m3'
        verbose: Print warmup status
    """
    if verbose:
        print(f"Warming up JAX compilation for {model}...")

    # Create dummy data matching typical participant shapes
    key = jax.random.PRNGKey(0)
    n_blocks = 7
    n_trials = 50

    stimuli_blocks = []
    actions_blocks = []
    rewards_blocks = []
    set_sizes_blocks = []

    for _ in range(n_blocks):
        key, k1, k2, k3 = jax.random.split(key, 4)
        stimuli_blocks.append(jax.random.randint(k1, (n_trials,), 0, 6))
        actions_blocks.append(jax.random.randint(k2, (n_trials,), 0, 3))
        rewards_blocks.append(jax.random.bernoulli(k3, 0.7, (n_trials,)).astype(jnp.float32))
        set_sizes_blocks.append(jnp.full((n_trials,), 4, dtype=jnp.int32))

    # Call likelihood function to trigger JIT compilation
    if model == 'qlearning':
        q_learning_multiblock_likelihood(
            stimuli_blocks, actions_blocks, rewards_blocks,
            alpha_pos=0.3, alpha_neg=0.1, epsilon=0.05
        )
    elif model == 'wmrl':
        wmrl_multiblock_likelihood(
            stimuli_blocks, actions_blocks, rewards_blocks, set_sizes_blocks,
            alpha_pos=0.3, alpha_neg=0.1, phi=0.1, rho=0.7, capacity=4.0, epsilon=0.05
        )
    elif model == 'wmrl_m3':
        wmrl_m3_multiblock_likelihood(
            stimuli_blocks, actions_blocks, rewards_blocks, set_sizes_blocks,
            alpha_pos=0.3, alpha_neg=0.1, phi=0.1, rho=0.7, capacity=4.0,
            kappa=0.3, epsilon=0.05
        )
    elif model == 'wmrl_m5':
        wmrl_m5_multiblock_likelihood(
            stimuli_blocks, actions_blocks, rewards_blocks, set_sizes_blocks,
            alpha_pos=0.3, alpha_neg=0.1, phi=0.1, rho=0.7, capacity=4.0,
            kappa=0.3, phi_rl=0.1, epsilon=0.05
        )

    if verbose:
        print(f"  JAX compilation cached for {model}\n")

# =============================================================================
# JAX-Compatible Objective Functions (for jaxopt with automatic differentiation)
# =============================================================================

def _make_jax_objective_qlearning(
    stimuli_blocks: list[jnp.ndarray],
    actions_blocks: list[jnp.ndarray],
    rewards_blocks: list[jnp.ndarray],
    masks_blocks: list[jnp.ndarray] = None,
):
    """
    Create a JAX-compatible objective function for Q-learning.

    Returns a JIT-compiled pure function that takes unconstrained parameters
    and returns NLL. The JIT compilation is CRITICAL for performance:
    - Without JIT: Every call goes through Python dispatch (~10-100x slower)
    - With JIT: Function is compiled to XLA and runs entirely on GPU

    Args:
        stimuli_blocks: list of stimulus arrays per block
        actions_blocks: list of action arrays per block
        rewards_blocks: list of reward arrays per block
        masks_blocks: list of mask arrays per block (1.0 for real trials, 0.0 for padding).
                     When provided, enables fixed-size compilation for efficiency.
    """
    # Pre-stack blocks ONCE - use stacked version to avoid list/restack overhead
    stimuli_stacked = jnp.stack(stimuli_blocks)
    actions_stacked = jnp.stack(actions_blocks)
    rewards_stacked = jnp.stack(rewards_blocks)
    if masks_blocks is not None:
        masks_stacked = jnp.stack(masks_blocks)
    else:
        masks_stacked = jnp.ones((len(stimuli_blocks), stimuli_stacked.shape[1]))

    def objective(x: jnp.ndarray) -> float:
        alpha_pos, alpha_neg, epsilon = jax_unconstrained_to_params_qlearning(x)
        # Use stacked version directly - avoids list conversion and restacking
        log_lik = q_learning_multiblock_likelihood_stacked(
            stimuli_stacked=stimuli_stacked,
            actions_stacked=actions_stacked,
            rewards_stacked=rewards_stacked,
            masks_stacked=masks_stacked,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            epsilon=epsilon,
        )
        return -log_lik  # Negative for minimization

    # JIT-compile for massive speedup (eliminates Python dispatch overhead)
    return jax.jit(objective)

def _make_jax_objective_wmrl(
    stimuli_blocks: list[jnp.ndarray],
    actions_blocks: list[jnp.ndarray],
    rewards_blocks: list[jnp.ndarray],
    set_sizes_blocks: list[jnp.ndarray],
    masks_blocks: list[jnp.ndarray] = None,
):
    """
    Create a JAX-compatible objective function for WM-RL.

    Returns a JIT-compiled pure function that takes unconstrained parameters
    and returns NLL. The JIT compilation is CRITICAL for performance.

    Args:
        stimuli_blocks: list of stimulus arrays per block
        actions_blocks: list of action arrays per block
        rewards_blocks: list of reward arrays per block
        set_sizes_blocks: list of set size arrays per block
        masks_blocks: list of mask arrays per block (1.0 for real trials, 0.0 for padding).
                     When provided, enables fixed-size compilation for efficiency.
    """
    # Pre-stack blocks ONCE - use stacked version to avoid list/restack overhead
    stimuli_stacked = jnp.stack(stimuli_blocks)
    actions_stacked = jnp.stack(actions_blocks)
    rewards_stacked = jnp.stack(rewards_blocks)
    set_sizes_stacked = jnp.stack(set_sizes_blocks)
    if masks_blocks is not None:
        masks_stacked = jnp.stack(masks_blocks)
    else:
        masks_stacked = jnp.ones((len(stimuli_blocks), stimuli_stacked.shape[1]))

    def objective(x: jnp.ndarray) -> float:
        alpha_pos, alpha_neg, phi, rho, capacity, epsilon = jax_unconstrained_to_params_wmrl(x)
        # Use stacked version directly - avoids list conversion and restacking
        log_lik = wmrl_multiblock_likelihood_stacked(
            stimuli_stacked=stimuli_stacked,
            actions_stacked=actions_stacked,
            rewards_stacked=rewards_stacked,
            set_sizes_stacked=set_sizes_stacked,
            masks_stacked=masks_stacked,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            phi=phi,
            rho=rho,
            capacity=capacity,
            epsilon=epsilon,
        )
        return -log_lik

    # JIT-compile for massive speedup (eliminates Python dispatch overhead)
    return jax.jit(objective)

def _make_jax_objective_wmrl_m3(
    stimuli_blocks: list[jnp.ndarray],
    actions_blocks: list[jnp.ndarray],
    rewards_blocks: list[jnp.ndarray],
    set_sizes_blocks: list[jnp.ndarray],
    masks_blocks: list[jnp.ndarray] = None,
):
    """
    Create a JAX-compatible objective function for WM-RL M3.

    Returns a JIT-compiled pure function that takes unconstrained parameters
    and returns NLL. The JIT compilation is CRITICAL for performance.

    Args:
        stimuli_blocks: list of stimulus arrays per block
        actions_blocks: list of action arrays per block
        rewards_blocks: list of reward arrays per block
        set_sizes_blocks: list of set size arrays per block
        masks_blocks: list of mask arrays per block (1.0 for real trials, 0.0 for padding).
                     When provided, enables fixed-size compilation for efficiency.
    """
    # Pre-stack blocks ONCE - use stacked version to avoid list/restack overhead
    stimuli_stacked = jnp.stack(stimuli_blocks)
    actions_stacked = jnp.stack(actions_blocks)
    rewards_stacked = jnp.stack(rewards_blocks)
    set_sizes_stacked = jnp.stack(set_sizes_blocks)
    if masks_blocks is not None:
        masks_stacked = jnp.stack(masks_blocks)
    else:
        masks_stacked = jnp.ones((len(stimuli_blocks), stimuli_stacked.shape[1]))

    def objective(x: jnp.ndarray) -> float:
        alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon = jax_unconstrained_to_params_wmrl_m3(x)
        # Use stacked version directly - avoids list conversion and restacking
        log_lik = wmrl_m3_multiblock_likelihood_stacked(
            stimuli_stacked=stimuli_stacked,
            actions_stacked=actions_stacked,
            rewards_stacked=rewards_stacked,
            set_sizes_stacked=set_sizes_stacked,
            masks_stacked=masks_stacked,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            phi=phi,
            rho=rho,
            capacity=capacity,
            kappa=kappa,
            epsilon=epsilon,
        )
        return -log_lik

    # JIT-compile for massive speedup (eliminates Python dispatch overhead)
    return jax.jit(objective)

def _make_jax_objective_wmrl_m5(
    stimuli_blocks: list[jnp.ndarray],
    actions_blocks: list[jnp.ndarray],
    rewards_blocks: list[jnp.ndarray],
    set_sizes_blocks: list[jnp.ndarray],
    masks_blocks: list[jnp.ndarray] = None,
):
    """
    Create a JAX-compatible objective function for WM-RL M5 (RL forgetting).

    Returns a JIT-compiled pure function that takes unconstrained parameters
    and returns NLL. The JIT compilation is CRITICAL for performance.

    Args:
        stimuli_blocks: list of stimulus arrays per block
        actions_blocks: list of action arrays per block
        rewards_blocks: list of reward arrays per block
        set_sizes_blocks: list of set size arrays per block
        masks_blocks: list of mask arrays per block (1.0 for real trials, 0.0 for padding).
    """
    # Pre-stack blocks ONCE - use stacked version to avoid list/restack overhead
    stimuli_stacked = jnp.stack(stimuli_blocks)
    actions_stacked = jnp.stack(actions_blocks)
    rewards_stacked = jnp.stack(rewards_blocks)
    set_sizes_stacked = jnp.stack(set_sizes_blocks)
    if masks_blocks is not None:
        masks_stacked = jnp.stack(masks_blocks)
    else:
        masks_stacked = jnp.ones((len(stimuli_blocks), stimuli_stacked.shape[1]))

    def objective(x: jnp.ndarray) -> float:
        alpha_pos, alpha_neg, phi, rho, capacity, kappa, phi_rl, epsilon = jax_unconstrained_to_params_wmrl_m5(x)
        # Use stacked version directly - avoids list conversion and restacking
        log_lik = wmrl_m5_multiblock_likelihood_stacked(
            stimuli_stacked=stimuli_stacked,
            actions_stacked=actions_stacked,
            rewards_stacked=rewards_stacked,
            set_sizes_stacked=set_sizes_stacked,
            masks_stacked=masks_stacked,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            phi=phi,
            rho=rho,
            capacity=capacity,
            kappa=kappa,
            phi_rl=phi_rl,
            epsilon=epsilon,
        )
        return -log_lik

    # JIT-compile for massive speedup (eliminates Python dispatch overhead)
    return jax.jit(objective)

# =============================================================================
# Bounded Objective Functions (for ScipyBoundedMinimize / L-BFGS-B)
# =============================================================================
# These take parameters DIRECTLY in bounded space (no transforms needed).
# L-BFGS-B handles bounds natively, so we skip the logit/inv_logit layer.

def _make_bounded_objective_qlearning(
    stimuli_blocks: list[jnp.ndarray],
    actions_blocks: list[jnp.ndarray],
    rewards_blocks: list[jnp.ndarray],
    masks_blocks: list[jnp.ndarray] = None,
):
    """
    Create bounded-space objective for Q-learning (no parameter transforms).

    Used with ScipyBoundedMinimize which handles bounds natively via L-BFGS-B.
    """
    stimuli_stacked = jnp.stack(stimuli_blocks)
    actions_stacked = jnp.stack(actions_blocks)
    rewards_stacked = jnp.stack(rewards_blocks)
    if masks_blocks is not None:
        masks_stacked = jnp.stack(masks_blocks)
    else:
        masks_stacked = jnp.ones((len(stimuli_blocks), stimuli_stacked.shape[1]))

    @jax.jit
    def objective(params: jnp.ndarray) -> float:
        alpha_pos, alpha_neg, epsilon = params[0], params[1], params[2]
        log_lik = q_learning_multiblock_likelihood_stacked(
            stimuli_stacked=stimuli_stacked,
            actions_stacked=actions_stacked,
            rewards_stacked=rewards_stacked,
            masks_stacked=masks_stacked,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            epsilon=epsilon,
        )
        return -log_lik

    return objective

def _make_bounded_objective_wmrl(
    stimuli_blocks: list[jnp.ndarray],
    actions_blocks: list[jnp.ndarray],
    rewards_blocks: list[jnp.ndarray],
    set_sizes_blocks: list[jnp.ndarray],
    masks_blocks: list[jnp.ndarray] = None,
):
    """
    Create bounded-space objective for WM-RL (no parameter transforms).

    Used with ScipyBoundedMinimize which handles bounds natively via L-BFGS-B.
    """
    stimuli_stacked = jnp.stack(stimuli_blocks)
    actions_stacked = jnp.stack(actions_blocks)
    rewards_stacked = jnp.stack(rewards_blocks)
    set_sizes_stacked = jnp.stack(set_sizes_blocks)
    if masks_blocks is not None:
        masks_stacked = jnp.stack(masks_blocks)
    else:
        masks_stacked = jnp.ones((len(stimuli_blocks), stimuli_stacked.shape[1]))

    @jax.jit
    def objective(params: jnp.ndarray) -> float:
        alpha_pos = params[0]
        alpha_neg = params[1]
        phi = params[2]
        rho = params[3]
        capacity = params[4]
        epsilon = params[5]
        log_lik = wmrl_multiblock_likelihood_stacked(
            stimuli_stacked=stimuli_stacked,
            actions_stacked=actions_stacked,
            rewards_stacked=rewards_stacked,
            set_sizes_stacked=set_sizes_stacked,
            masks_stacked=masks_stacked,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            phi=phi,
            rho=rho,
            capacity=capacity,
            epsilon=epsilon,
        )
        return -log_lik

    return objective

def _make_bounded_objective_wmrl_m3(
    stimuli_blocks: list[jnp.ndarray],
    actions_blocks: list[jnp.ndarray],
    rewards_blocks: list[jnp.ndarray],
    set_sizes_blocks: list[jnp.ndarray],
    masks_blocks: list[jnp.ndarray] = None,
):
    """
    Create bounded-space objective for WM-RL M3 (no parameter transforms).

    Used with ScipyBoundedMinimize which handles bounds natively via L-BFGS-B.
    """
    stimuli_stacked = jnp.stack(stimuli_blocks)
    actions_stacked = jnp.stack(actions_blocks)
    rewards_stacked = jnp.stack(rewards_blocks)
    set_sizes_stacked = jnp.stack(set_sizes_blocks)
    if masks_blocks is not None:
        masks_stacked = jnp.stack(masks_blocks)
    else:
        masks_stacked = jnp.ones((len(stimuli_blocks), stimuli_stacked.shape[1]))

    @jax.jit
    def objective(params: jnp.ndarray) -> float:
        alpha_pos = params[0]
        alpha_neg = params[1]
        phi = params[2]
        rho = params[3]
        capacity = params[4]
        kappa = params[5]
        epsilon = params[6]
        log_lik = wmrl_m3_multiblock_likelihood_stacked(
            stimuli_stacked=stimuli_stacked,
            actions_stacked=actions_stacked,
            rewards_stacked=rewards_stacked,
            set_sizes_stacked=set_sizes_stacked,
            masks_stacked=masks_stacked,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            phi=phi,
            rho=rho,
            capacity=capacity,
            kappa=kappa,
            epsilon=epsilon,
        )
        return -log_lik

    return objective

def _make_bounded_objective_wmrl_m5(
    stimuli_blocks: list[jnp.ndarray],
    actions_blocks: list[jnp.ndarray],
    rewards_blocks: list[jnp.ndarray],
    set_sizes_blocks: list[jnp.ndarray],
    masks_blocks: list[jnp.ndarray] = None,
):
    """
    Create bounded-space objective for WM-RL M5 (no parameter transforms).

    Used with ScipyBoundedMinimize which handles bounds natively via L-BFGS-B.
    8 parameters: alpha_pos, alpha_neg, phi, rho, capacity, kappa, phi_rl, epsilon
    """
    stimuli_stacked = jnp.stack(stimuli_blocks)
    actions_stacked = jnp.stack(actions_blocks)
    rewards_stacked = jnp.stack(rewards_blocks)
    set_sizes_stacked = jnp.stack(set_sizes_blocks)
    if masks_blocks is not None:
        masks_stacked = jnp.stack(masks_blocks)
    else:
        masks_stacked = jnp.ones((len(stimuli_blocks), stimuli_stacked.shape[1]))

    @jax.jit
    def objective(params: jnp.ndarray) -> float:
        alpha_pos = params[0]
        alpha_neg = params[1]
        phi = params[2]
        rho = params[3]
        capacity = params[4]
        kappa = params[5]
        phi_rl = params[6]
        epsilon = params[7]
        log_lik = wmrl_m5_multiblock_likelihood_stacked(
            stimuli_stacked=stimuli_stacked,
            actions_stacked=actions_stacked,
            rewards_stacked=rewards_stacked,
            set_sizes_stacked=set_sizes_stacked,
            masks_stacked=masks_stacked,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            phi=phi,
            rho=rho,
            capacity=capacity,
            kappa=kappa,
            phi_rl=phi_rl,
            epsilon=epsilon,
        )
        return -log_lik

    return objective

# =============================================================================
# GPU Objective Functions (data-as-args for vmap compatibility)
# =============================================================================
# Unlike the closure-based objectives above, these take data arrays as explicit
# positional arguments so jax.vmap can vary over them (participants dimension)
# and over starting points (parameter dimension).

def _gpu_objective_qlearning(x, stimuli, actions, rewards, masks):
    """
    Unconstrained objective for Q-learning with data as explicit args.

    Args:
        x: unconstrained parameter vector, shape (3,)
        stimuli: shape (n_blocks, max_trials)
        actions: shape (n_blocks, max_trials)
        rewards: shape (n_blocks, max_trials)
        masks: shape (n_blocks, max_trials)

    Returns:
        Scalar negative log-likelihood
    """
    alpha_pos, alpha_neg, epsilon = jax_unconstrained_to_params_qlearning(x)
    log_lik = q_learning_multiblock_likelihood_stacked(
        stimuli_stacked=stimuli,
        actions_stacked=actions,
        rewards_stacked=rewards,
        masks_stacked=masks,
        alpha_pos=alpha_pos,
        alpha_neg=alpha_neg,
        epsilon=epsilon,
    )
    return -log_lik

def _gpu_objective_wmrl(x, stimuli, actions, rewards, masks, set_sizes):
    """
    Unconstrained objective for WM-RL with data as explicit args.

    Args:
        x: unconstrained parameter vector, shape (6,)
        stimuli, actions, rewards, masks: shape (n_blocks, max_trials)
        set_sizes: shape (n_blocks, max_trials)

    Returns:
        Scalar negative log-likelihood
    """
    alpha_pos, alpha_neg, phi, rho, capacity, epsilon = jax_unconstrained_to_params_wmrl(x)
    log_lik = wmrl_multiblock_likelihood_stacked(
        stimuli_stacked=stimuli,
        actions_stacked=actions,
        rewards_stacked=rewards,
        set_sizes_stacked=set_sizes,
        masks_stacked=masks,
        alpha_pos=alpha_pos,
        alpha_neg=alpha_neg,
        phi=phi,
        rho=rho,
        capacity=capacity,
        epsilon=epsilon,
    )
    return -log_lik

def _gpu_objective_wmrl_m3(x, stimuli, actions, rewards, masks, set_sizes):
    """
    Unconstrained objective for WM-RL M3 with data as explicit args.

    Args:
        x: unconstrained parameter vector, shape (7,)
        stimuli, actions, rewards, masks: shape (n_blocks, max_trials)
        set_sizes: shape (n_blocks, max_trials)

    Returns:
        Scalar negative log-likelihood
    """
    alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon = jax_unconstrained_to_params_wmrl_m3(x)
    log_lik = wmrl_m3_multiblock_likelihood_stacked(
        stimuli_stacked=stimuli,
        actions_stacked=actions,
        rewards_stacked=rewards,
        set_sizes_stacked=set_sizes,
        masks_stacked=masks,
        alpha_pos=alpha_pos,
        alpha_neg=alpha_neg,
        phi=phi,
        rho=rho,
        capacity=capacity,
        kappa=kappa,
        epsilon=epsilon,
    )
    return -log_lik

def _gpu_objective_wmrl_m5(x, stimuli, actions, rewards, masks, set_sizes):
    """
    Unconstrained objective for WM-RL M5 (RL forgetting) with data as explicit args.

    Args:
        x: unconstrained parameter vector, shape (8,)
        stimuli, actions, rewards, masks: shape (n_blocks, max_trials)
        set_sizes: shape (n_blocks, max_trials)

    Returns:
        Scalar negative log-likelihood
    """
    alpha_pos, alpha_neg, phi, rho, capacity, kappa, phi_rl, epsilon = jax_unconstrained_to_params_wmrl_m5(x)
    log_lik = wmrl_m5_multiblock_likelihood_stacked(
        stimuli_stacked=stimuli,
        actions_stacked=actions,
        rewards_stacked=rewards,
        set_sizes_stacked=set_sizes,
        masks_stacked=masks,
        alpha_pos=alpha_pos,
        alpha_neg=alpha_neg,
        phi=phi,
        rho=rho,
        capacity=capacity,
        kappa=kappa,
        phi_rl=phi_rl,
        epsilon=epsilon,
    )
    return -log_lik

# =============================================================================
# GPU-Vectorized Fitting (jaxopt.LBFGS + vmap over starts and participants)
# =============================================================================

def fit_all_gpu(
    data: pd.DataFrame,
    model: str = 'qlearning',
    n_starts: int = 50,
    seed: int = 42,
    verbose: bool = True,
    compute_diagnostics: bool = False,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    """
    GPU-accelerated MLE fitting using vectorized jaxopt.LBFGS.

    Fits ALL participants x ALL starting points in a single JIT-compiled call
    using nested vmap over participants (outer) and starts (inner). This replaces
    the sequential loop in fit_all_participants() when GPU is available.

    Args:
        data: DataFrame with columns [sona_id, block, stimulus, key_press, reward, set_size]
        model: 'qlearning', 'wmrl', or 'wmrl_m3'
        n_starts: Number of random starting points per participant
        seed: Random seed
        verbose: Show progress output
        compute_diagnostics: Whether to compute Hessian diagnostics post-fit

    Returns:
        Same tuple format as fit_all_participants():
        (fits_df, timing_info, timing_records)
    """
    import time
    import jaxopt

    start_time = time.time()
    participants = data['sona_id'].unique()
    n_participants = len(participants)

    if verbose:
        start_datetime = datetime.now()
        print(f"\n{'='*60}")
        print(f"GPU-Vectorized MLE Fitting: {model.upper()}")
        print(f"{'='*60}")
        print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Participants: {n_participants}")
        print(f"Random starts per participant: {n_starts}")
        print(f"Method: jaxopt.LBFGS + jax.vmap (fully vectorized)")
        print(f"{'='*60}\n")

    # =========================================================================
    # Stage 1: Prepare and stack all participant data
    # =========================================================================
    if verbose:
        print("Stage 1: Preparing and stacking participant data...", flush=True)
        t0 = time.time()

    all_stimuli = []
    all_actions = []
    all_rewards = []
    all_masks = []
    all_set_sizes = []
    trial_counts = []

    for pid in participants:
        pdata = prepare_participant_data(data, pid, model, pad_blocks=True)
        all_stimuli.append(jnp.stack(pdata['stimuli_blocks']))
        all_actions.append(jnp.stack(pdata['actions_blocks']))
        all_rewards.append(jnp.stack(pdata['rewards_blocks']))
        all_masks.append(jnp.stack(pdata['masks_blocks']))

        if model in ('wmrl', 'wmrl_m3', 'wmrl_m5'):
            all_set_sizes.append(jnp.stack(pdata['set_sizes_blocks']))

        trial_counts.append(int(sum(jnp.sum(m) for m in pdata['masks_blocks'])))

    # Stack into batched arrays: (n_participants, MAX_BLOCKS, MAX_TRIALS_PER_BLOCK)
    stimuli_batch = jnp.stack(all_stimuli)
    actions_batch = jnp.stack(all_actions)
    rewards_batch = jnp.stack(all_rewards)
    masks_batch = jnp.stack(all_masks)
    set_sizes_batch = jnp.stack(all_set_sizes) if model in ('wmrl', 'wmrl_m3', 'wmrl_m5') else None

    if verbose:
        print(f"  Stacked data shape: {stimuli_batch.shape} ({time.time() - t0:.1f}s)")

    # =========================================================================
    # Stage 2: Generate LHS starting points in unconstrained space
    # =========================================================================
    if verbose:
        print("Stage 2: Generating unconstrained starting points...", flush=True)

    x0_bounded = jnp.array(sample_lhs_starts(model, n_starts, seed=seed))

    if model == 'qlearning':
        to_unc = jax_bounded_to_unconstrained_qlearning
    elif model == 'wmrl':
        to_unc = jax_bounded_to_unconstrained_wmrl
    elif model == 'wmrl_m3':
        to_unc = jax_bounded_to_unconstrained_wmrl_m3
    else:
        to_unc = jax_bounded_to_unconstrained_wmrl_m5

    x0_unc = jax.vmap(to_unc)(x0_bounded)  # (n_starts, n_params)

    if verbose:
        print(f"  Starting points: {x0_unc.shape}")

    # =========================================================================
    # Stage 3: Create solver and vmapped fitting function
    # =========================================================================
    if verbose:
        print("Stage 3: Building vmapped optimizer...", flush=True)

    # Select model-specific objective
    if model == 'qlearning':
        objective = _gpu_objective_qlearning
    elif model == 'wmrl':
        objective = _gpu_objective_wmrl
    elif model == 'wmrl_m3':
        objective = _gpu_objective_wmrl_m3
    else:
        objective = _gpu_objective_wmrl_m5

    solver = jaxopt.LBFGS(
        fun=objective,
        maxiter=1000,
        tol=1e-5,
        history_size=10,
        jit=True,            # Must be True for lax.while_loop (needed by vmap)
        implicit_diff=False,  # Not doing bilevel optimization
    )

    # Build nested vmap: starts (inner) → participants (outer)
    if model == 'qlearning':
        def _run_one(x0, stimuli, actions, rewards, masks):
            params, state = solver.run(x0, stimuli, actions, rewards, masks)
            nll = objective(params, stimuli, actions, rewards, masks)
            return params, nll, state.error

        # vmap over starts: x0 varies (axis 0), data fixed (None)
        _run_all_starts = jax.vmap(_run_one, in_axes=(0, None, None, None, None))
        # vmap over participants: x0 fixed (None), data varies (axis 0)
        _run_all = jax.jit(jax.vmap(_run_all_starts, in_axes=(None, 0, 0, 0, 0)))
    else:
        # wmrl, wmrl_m3, and wmrl_m5 share the same data signature (includes set_sizes)
        def _run_one(x0, stimuli, actions, rewards, masks, set_sizes):
            params, state = solver.run(x0, stimuli, actions, rewards, masks, set_sizes)
            nll = objective(params, stimuli, actions, rewards, masks, set_sizes)
            return params, nll, state.error

        _run_all_starts = jax.vmap(_run_one, in_axes=(0, None, None, None, None, None))
        _run_all = jax.jit(jax.vmap(_run_all_starts, in_axes=(None, 0, 0, 0, 0, 0)))

    # =========================================================================
    # Stage 4: Execute single JIT-compiled fitting call
    # =========================================================================
    if verbose:
        print("Stage 4: Running GPU-vectorized fitting (JIT compiling + executing)...", flush=True)
        t0 = time.time()

    initial_mem = log_memory_usage("PRE-GPU-FIT", verbose=verbose)

    if model == 'qlearning':
        all_params, all_nlls, all_errors = _run_all(
            x0_unc, stimuli_batch, actions_batch, rewards_batch, masks_batch
        )
    else:
        all_params, all_nlls, all_errors = _run_all(
            x0_unc, stimuli_batch, actions_batch, rewards_batch, masks_batch, set_sizes_batch
        )

    # Block until GPU computation is complete
    all_params.block_until_ready()
    gpu_time = time.time() - t0

    post_mem = log_memory_usage("POST-GPU-FIT", verbose=verbose)

    if verbose:
        print(f"  GPU fitting complete: {gpu_time:.1f}s")
        print(f"  Output shapes: params={all_params.shape}, nlls={all_nlls.shape}")

    # all_params: (n_participants, n_starts, n_params) in unconstrained space
    # all_nlls:   (n_participants, n_starts)
    # all_errors: (n_participants, n_starts) gradient norms at convergence

    # =========================================================================
    # Stage 5: Select best starts and transform back to bounded space
    # =========================================================================
    if verbose:
        print("Stage 5: Selecting best results and transforming parameters...", flush=True)

    # Find best start per participant (lowest NLL)
    best_idx = jnp.argmin(all_nlls, axis=1)  # (n_participants,)
    best_params_unc = all_params[jnp.arange(n_participants), best_idx]
    best_nlls = all_nlls[jnp.arange(n_participants), best_idx]
    best_errors = all_errors[jnp.arange(n_participants), best_idx]

    # Transform best params back to bounded space (vectorized)
    if model == 'qlearning':
        bounded_tuple = jax.vmap(jax_unconstrained_to_params_qlearning)(best_params_unc)
        param_names = QLEARNING_PARAMS
        bounds_dict = QLEARNING_BOUNDS
    elif model == 'wmrl':
        bounded_tuple = jax.vmap(jax_unconstrained_to_params_wmrl)(best_params_unc)
        param_names = WMRL_PARAMS
        bounds_dict = WMRL_BOUNDS
    elif model == 'wmrl_m3':
        bounded_tuple = jax.vmap(jax_unconstrained_to_params_wmrl_m3)(best_params_unc)
        param_names = WMRL_M3_PARAMS
        bounds_dict = WMRL_M3_BOUNDS
    else:
        bounded_tuple = jax.vmap(jax_unconstrained_to_params_wmrl_m5)(best_params_unc)
        param_names = WMRL_M5_PARAMS
        bounds_dict = WMRL_M5_BOUNDS

    # =========================================================================
    # Stage 6: Build result dictionaries
    # =========================================================================
    if verbose:
        print("Stage 6: Building result dictionaries...", flush=True)

    n_params_model = get_n_params(model)
    results = []

    for i in range(n_participants):
        pid = participants[i]
        nll = float(best_nlls[i])
        n_trials = trial_counts[i]

        # Extract bounded parameters for this participant
        best_params = {}
        for j, name in enumerate(param_names):
            best_params[name] = float(bounded_tuple[j][i])

        # Information criteria
        k = n_params_model
        aic = compute_aic(nll, k)
        bic = compute_bic(nll, k, n_trials)
        aicc = compute_aicc(nll, k, n_trials)
        pseudo_r2 = compute_pseudo_r2(nll, n_trials)

        # Convergence: gradient norm at best solution
        grad_norm = float(best_errors[i])
        converged = grad_norm < 1e-4

        # Count starts near best
        participant_nlls = np.array(all_nlls[i])
        finite_nlls = participant_nlls[np.isfinite(participant_nlls)]
        n_successful = len(finite_nlls)
        n_near_best = int(np.sum(np.abs(finite_nlls - nll) < 1.0)) if len(finite_nlls) > 0 else 0

        # Check at bounds
        at_bounds = check_at_bounds(best_params, model)

        result = {
            'participant_id': pid,
            **best_params,
            'nll': nll,
            'aic': aic,
            'bic': bic,
            'aicc': aicc,
            'pseudo_r2': pseudo_r2,
            'n_trials': n_trials,
            'converged': converged,
            'n_successful_starts': n_successful,
            'n_near_best': n_near_best,
            'at_bounds': at_bounds,
            'grad_norm': grad_norm,
        }

        results.append(result)

    # =========================================================================
    # Stage 7: Optional Hessian diagnostics (sequential, post-vmap)
    # =========================================================================
    if compute_diagnostics:
        if verbose:
            print("Stage 7: Computing Hessian diagnostics (sequential)...", flush=True)
            t0 = time.time()

        for i, result in enumerate(results):
            pid = result['participant_id']
            best_params = {name: result[name] for name in param_names}

            # Create closure-based unbounded objective for Hessian computation
            pdata = prepare_participant_data(data, pid, model)
            if model == 'qlearning':
                objective_fn = _make_jax_objective_qlearning(
                    pdata['stimuli_blocks'], pdata['actions_blocks'],
                    pdata['rewards_blocks'], masks_blocks=pdata.get('masks_blocks')
                )
            elif model == 'wmrl':
                objective_fn = _make_jax_objective_wmrl(
                    pdata['stimuli_blocks'], pdata['actions_blocks'],
                    pdata['rewards_blocks'], pdata['set_sizes_blocks'],
                    masks_blocks=pdata.get('masks_blocks')
                )
            elif model == 'wmrl_m3':
                objective_fn = _make_jax_objective_wmrl_m3(
                    pdata['stimuli_blocks'], pdata['actions_blocks'],
                    pdata['rewards_blocks'], pdata['set_sizes_blocks'],
                    masks_blocks=pdata.get('masks_blocks')
                )
            else:
                objective_fn = _make_jax_objective_wmrl_m5(
                    pdata['stimuli_blocks'], pdata['actions_blocks'],
                    pdata['rewards_blocks'], pdata['set_sizes_blocks'],
                    masks_blocks=pdata.get('masks_blocks')
                )

            x_unconstrained = params_to_unconstrained(best_params, model)

            # Check gradient norm in unconstrained space
            grad_norm_unc, _ = check_gradient_norm(objective_fn, x_unconstrained)
            result['grad_norm'] = grad_norm_unc

            # Hessian diagnostics
            hess_diag = compute_hessian_diagnostics(objective_fn, x_unconstrained, model)
            if hess_diag['success']:
                result['hessian_condition'] = hess_diag['condition_number']
                result['hessian_invertible'] = hess_diag['hessian_invertible']
                se_bounded = hess_diag.get('se_bounded', {})
                for param, se in se_bounded.items():
                    result[f'{param}_se'] = se
                ci = compute_confidence_intervals(best_params, se_bounded)
                for param, (lower, upper) in ci.items():
                    result[f'{param}_ci_lower'] = lower
                    result[f'{param}_ci_upper'] = upper
                correlations = hess_diag.get('correlations', {})
                high_corr = get_high_correlations(correlations, threshold=0.9)
                result['high_correlations'] = high_corr
            else:
                result['hessian_condition'] = np.nan
                result['hessian_invertible'] = False

            if verbose and (i + 1) % 10 == 0:
                print(f"    Hessian: {i + 1}/{n_participants} done", flush=True)

        if verbose:
            hess_success = sum(1 for r in results if r.get('hessian_invertible', False))
            print(f"  Hessian diagnostics complete: {hess_success}/{n_participants} successful ({time.time() - t0:.1f}s)")

    # =========================================================================
    # Summary and return
    # =========================================================================
    total_time = time.time() - start_time

    if verbose:
        n_converged = sum(r['converged'] for r in results)
        conv_rate = 100.0 * n_converged / n_participants if n_participants > 0 else 0

        print(f"\n{'='*60}")
        print("GPU FITTING COMPLETE")
        print(f"{'='*60}")
        print(f"Total duration: {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"  GPU optimization: {gpu_time:.1f}s")
        print(f"Participants: {n_participants} | Converged: {n_converged} ({conv_rate:.0f}%)")
        if post_mem > 0:
            print(f"Peak memory: {max(initial_mem, post_mem)/1024:.2f}GB")
        print(f"{'='*60}\n")

    # Build DataFrame with same column order as fit_all_participants()
    df = pd.DataFrame(results)

    # Reorder columns
    cols = ['participant_id'] + param_names
    se_cols = [f'{p}_se' for p in param_names]
    ci_lower_cols = [f'{p}_ci_lower' for p in param_names]
    ci_upper_cols = [f'{p}_ci_upper' for p in param_names]
    cols += ['nll', 'aic', 'bic', 'aicc', 'pseudo_r2']
    cols += ['grad_norm', 'hessian_condition', 'hessian_invertible']
    cols += se_cols + ci_lower_cols + ci_upper_cols
    cols += ['n_trials', 'converged', 'n_successful_starts', 'n_near_best', 'at_bounds', 'high_correlations']
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    # Build timing info
    timing_info = {
        'total_time': total_time,
        'gpu_time': gpu_time,
        'fit_times': [],
        'jit_time': None,
        'steady_state_times': [],
        'n_jobs': 1,
        'n_participants': n_participants,
        'initial_mem_mb': initial_mem,
        'peak_mem_mb': max(initial_mem, post_mem),
        'model': model,
        'method': 'gpu_vmap',
    }

    # Build timing records (one per participant, times not individually tracked in GPU mode)
    timing_records = []
    for i, pid in enumerate(participants):
        timing_records.append({
            'participant_id': pid,
            'n_trials': trial_counts[i],
            'n_blocks': int(stimuli_batch.shape[1]),
            'fit_time_s': gpu_time / n_participants,  # Average estimate
            'is_first_fit': False,
            'memory_before_mb': initial_mem,
            'memory_after_mb': post_mem,
            'nll': results[i].get('nll', np.nan),
            'converged': results[i].get('converged', False),
        })

    return df, timing_info, timing_records

# =============================================================================
# Single Participant Fitting (using L-BFGS-B with analytical gradients)
# =============================================================================

class _JaxoptResult:
    """Simple wrapper to make jaxopt results compatible with convergence checking."""
    def __init__(self, x, fun, success):
        self.x = x
        self.fun = fun
        self.success = success

def fit_participant_mle(
    stimuli_blocks: list[np.ndarray],
    actions_blocks: list[np.ndarray],
    rewards_blocks: list[np.ndarray],
    set_sizes_blocks: list[np.ndarray] | None = None,
    masks_blocks: list[np.ndarray] | None = None,
    model: str = 'qlearning',
    n_starts: int = 50,
    seed: int | None = None,
    compute_diagnostics: bool = True,
    verbose: bool = False,
    participant_index: int = 0,
) -> dict:
    """
    Fit a single participant using MLE with multiple starting points.

    Uses ScipyBoundedMinimize (L-BFGS-B) for bounded optimization with analytical
    gradients via JAX autodiff. Starting points are generated via Latin Hypercube
    Sampling for even coverage of the parameter space.
    Following Senta et al. (2025): 50 starting points, keep best result.
    Note: Beta is fixed at 50 inside the likelihood functions.

    Args:
        stimuli_blocks: list of stimulus arrays per block
        actions_blocks: list of action arrays per block
        rewards_blocks: list of reward arrays per block
        set_sizes_blocks: list of set size arrays per block (WM-RL/M3 only)
        masks_blocks: list of mask arrays per block (1.0 for real trials, 0.0 for padding).
                     When provided, enables fixed-size compilation for JAX efficiency.
        model: 'qlearning', 'wmrl', or 'wmrl_m3'
        n_starts: Number of random starting points
        seed: Random seed for reproducibility
        compute_diagnostics: Whether to compute Hessian-based diagnostics (default: True)
        verbose: Print progress (default: False)
        participant_index: Index in the outer loop (0-based). Steps [1/5]-[4/5]
                          only print for participant_index==0 to reduce noise.

    Returns:
        Dictionary with:
        - Fitted parameters (alpha_pos, alpha_neg, etc.)
        - Model fit metrics (nll, aic, bic, aicc, pseudo_r2)
        - Convergence info (converged, n_successful_starts, n_near_best, at_bounds)
        - Diagnostics (grad_norm, hessian_condition, parameter SEs and CIs)
    """
    import time as _time  # Local import to avoid namespace collision
    rng = np.random.default_rng(seed)

    # Only show setup steps [1/5]-[4/5] for first participant (they're instantaneous after JIT)
    show_setup_steps = verbose and participant_index == 0

    # Count total REAL trials (accounting for masks if provided)
    if show_setup_steps:
        print("      [1/5] Counting trials...", end=" ", flush=True)
        _t0 = _time.time()
    if masks_blocks is not None:
        # Sum of mask values = number of real trials (mask=1 for real, 0 for padding)
        n_trials = sum(int(jnp.sum(m)) for m in masks_blocks)
    else:
        n_trials = sum(len(s) for s in stimuli_blocks)
    if show_setup_steps:
        print(f"done ({_time.time() - _t0:.2f}s)", flush=True)

    # Convert data to JAX arrays for efficient computation
    # Note: If data is already JAX arrays (from prepare_participant_data with padding),
    # jnp.array() is a no-op
    if show_setup_steps:
        print("      [2/5] Preparing JAX arrays...", end=" ", flush=True)
        _t0 = _time.time()
    stimuli_jax = [jnp.array(s, dtype=jnp.int32) for s in stimuli_blocks]
    actions_jax = [jnp.array(a, dtype=jnp.int32) for a in actions_blocks]
    rewards_jax = [jnp.array(r, dtype=jnp.float32) for r in rewards_blocks]
    masks_jax = masks_blocks  # Already JAX arrays from prepare_participant_data, or None
    if show_setup_steps:
        print(f"done ({_time.time() - _t0:.2f}s)", flush=True)

    # Create objective functions
    # Note: Beta is fixed at 50 inside the likelihood functions
    # We create TWO objectives:
    #   1. bounded_objective: for L-BFGS-B (takes bounded params directly)
    #   2. objective: unbounded version (kept for Hessian diagnostics which need transforms)
    if show_setup_steps:
        print("      [3/5] Creating objective functions...", end=" ", flush=True)
        _t0 = _time.time()
    if model == 'qlearning':
        bounded_objective = _make_bounded_objective_qlearning(
            stimuli_jax, actions_jax, rewards_jax, masks_blocks=masks_jax
        )
        # Unbounded objective for Hessian diagnostics
        objective = _make_jax_objective_qlearning(
            stimuli_jax, actions_jax, rewards_jax, masks_blocks=masks_jax
        )
        n_params = 3
        param_names = QLEARNING_PARAMS
        bounds_dict = QLEARNING_BOUNDS
    elif model == 'wmrl':
        if set_sizes_blocks is None:
            raise ValueError("set_sizes_blocks required for WM-RL model")
        set_sizes_jax = [jnp.array(s, dtype=jnp.int32) for s in set_sizes_blocks]
        bounded_objective = _make_bounded_objective_wmrl(
            stimuli_jax, actions_jax, rewards_jax, set_sizes_jax, masks_blocks=masks_jax
        )
        objective = _make_jax_objective_wmrl(
            stimuli_jax, actions_jax, rewards_jax, set_sizes_jax, masks_blocks=masks_jax
        )
        n_params = 6
        param_names = WMRL_PARAMS
        bounds_dict = WMRL_BOUNDS
    elif model == 'wmrl_m3':
        if set_sizes_blocks is None:
            raise ValueError("set_sizes_blocks required for WM-RL M3 model")
        set_sizes_jax = [jnp.array(s, dtype=jnp.int32) for s in set_sizes_blocks]
        bounded_objective = _make_bounded_objective_wmrl_m3(
            stimuli_jax, actions_jax, rewards_jax, set_sizes_jax, masks_blocks=masks_jax
        )
        objective = _make_jax_objective_wmrl_m3(
            stimuli_jax, actions_jax, rewards_jax, set_sizes_jax, masks_blocks=masks_jax
        )
        n_params = 7
        param_names = WMRL_M3_PARAMS
        bounds_dict = WMRL_M3_BOUNDS
    elif model == 'wmrl_m5':
        if set_sizes_blocks is None:
            raise ValueError("set_sizes_blocks required for WM-RL M5 model")
        set_sizes_jax = [jnp.array(s, dtype=jnp.int32) for s in set_sizes_blocks]
        bounded_objective = _make_bounded_objective_wmrl_m5(
            stimuli_jax, actions_jax, rewards_jax, set_sizes_jax, masks_blocks=masks_jax
        )
        objective = _make_jax_objective_wmrl_m5(
            stimuli_jax, actions_jax, rewards_jax, set_sizes_jax, masks_blocks=masks_jax
        )
        n_params = 8
        param_names = WMRL_M5_PARAMS
        bounds_dict = WMRL_M5_BOUNDS
    else:
        raise ValueError(f"Unknown model: {model}")
    if show_setup_steps:
        print(f"done ({_time.time() - _t0:.2f}s)", flush=True)

    # Create ScipyBoundedMinimize solver (L-BFGS-B with native bounds)
    if show_setup_steps:
        print("      [4/5] Creating L-BFGS-B solver...", end=" ", flush=True)
        _t0 = _time.time()

    lower_bounds = jnp.array([bounds_dict[p][0] for p in param_names])
    upper_bounds = jnp.array([bounds_dict[p][1] for p in param_names])
    bounds = (lower_bounds, upper_bounds)

    solver = ScipyBoundedMinimize(
        fun=bounded_objective,
        method='L-BFGS-B',
        maxiter=1000,
        options={'ftol': 1e-6, 'gtol': 1e-5}
    )
    if show_setup_steps:
        print(f"done ({_time.time() - _t0:.2f}s)", flush=True)

    # Generate starting points using Latin Hypercube Sampling
    # LHS ensures even coverage of parameter space (better than random Normal)
    x0_batch = jnp.array(sample_lhs_starts(model, n_starts, seed=seed))

    # Sequential optimization with iteration logging
    if verbose:
        print(f"      [5/5] Running {n_starts} optimizations (L-BFGS-B)...", flush=True)
        _opt_start = _time.time()

    results = []
    iteration_stats = []

    for i in range(n_starts):
        x0 = x0_batch[i]
        try:
            if verbose and i == 0:
                print(f"            Start 1/{n_starts} (JIT compiling)...", end=" ", flush=True)
                _t0 = _time.time()

            result_params, state = solver.run(x0, bounds=bounds)
            final_nll = float(bounded_objective(result_params))

            # Use both finite NLL AND scipy's own convergence flag
            nll_finite = bool(jnp.isfinite(final_nll))
            scipy_success = bool(state.success) if hasattr(state, 'success') else nll_finite
            success = nll_finite  # For _JaxoptResult.success: finite NLL means usable result

            result = _JaxoptResult(
                x=np.array(result_params),
                fun=final_nll,
                success=success
            )
            results.append(result)

            # Track full optimizer state
            opt_info = {
                'final_nll': final_nll,
                'scipy_converged': scipy_success,
                'scipy_status': int(state.status) if hasattr(state, 'status') else -1,
                'iterations': int(state.iter_num) if hasattr(state, 'iter_num') else -1,
                'fun_evals': int(state.num_fun_eval) if hasattr(state, 'num_fun_eval') else -1,
            }
            iteration_stats.append(opt_info)

            if verbose and i == 0:
                print(f"done ({_time.time() - _t0:.2f}s, NLL={final_nll:.1f})", flush=True)
                if n_starts > 1:
                    print(f"            Starts 2-{n_starts}...", end=" ", flush=True)

        except Exception as e:
            if verbose and i == 0:
                print(f"failed: {e}", flush=True)
            # Record failed start in iteration_stats too
            iteration_stats.append({
                'final_nll': float('inf'),
                'scipy_converged': False,
                'scipy_status': -1,
                'iterations': -1,
                'fun_evals': -1,
            })
            continue  # Skip failed optimizations

    if verbose:
        _opt_elapsed = _time.time() - _opt_start
        # Rich summary: optimizer convergence + NLL distribution
        if iteration_stats:
            nlls = [s['final_nll'] for s in iteration_stats if np.isfinite(s['final_nll'])]
            n_scipy_conv = sum(1 for s in iteration_stats if s['scipy_converged'])
            n_hit_limit = sum(1 for s in iteration_stats
                              if not s['scipy_converged'] and np.isfinite(s['final_nll']))
            n_successful = len(nlls)
            best_nll = min(nlls) if nlls else float('inf')
            nll_range = max(nlls) - min(nlls) if len(nlls) > 1 else 0.0
            n_within_1 = sum(1 for nll in nlls if abs(nll - best_nll) < 1.0)
            n_within_5 = sum(1 for nll in nlls if abs(nll - best_nll) < 5.0)

            print(f"done (total: {_opt_elapsed:.1f}s)", flush=True)
            print(f"            Optimizer: {n_scipy_conv}/{n_starts} scipy-converged, {n_hit_limit} hit maxiter", flush=True)
            print(f"            NLLs: best={best_nll:.1f} | {n_within_1} within 1.0 | {n_within_5} within 5.0 | range={nll_range:.1f}", flush=True)
        else:
            print(f"done (total: {_opt_elapsed:.1f}s)", flush=True)

    # Check convergence (pass iteration_stats for scipy success flag)
    convergence_info = check_convergence(results, iteration_stats=iteration_stats)

    if not results or convergence_info['best_nll'] == np.inf:
        # All optimizations failed
        return {
            **{p: np.nan for p in param_names},
            'nll': np.nan,
            'aic': np.nan,
            'bic': np.nan,
            'aicc': np.nan,
            'n_trials': n_trials,
            'converged': False,
            'n_successful_starts': 0,
            'at_bounds': []
        }

    # Get best result (parameters are already in bounded space from L-BFGS-B)
    best_result = min(results, key=lambda r: r.fun if r.success else np.inf)
    best_params = {name: float(best_result.x[i]) for i, name in enumerate(param_names)}
    best_nll = best_result.fun

    # Compute information criteria
    k = get_n_params(model)
    aic = compute_aic(best_nll, k)
    bic = compute_bic(best_nll, k, n_trials)
    aicc = compute_aicc(best_nll, k, n_trials)

    # Compute pseudo-R² (variance explained)
    pseudo_r2 = compute_pseudo_r2(best_nll, n_trials)

    # Check if parameters hit bounds
    at_bounds = check_at_bounds(best_params, model)

    # Build result dictionary
    result = {
        **best_params,
        'nll': best_nll,
        'aic': aic,
        'bic': bic,
        'aicc': aicc,
        'pseudo_r2': pseudo_r2,
        'n_trials': n_trials,
        'converged': convergence_info['converged'],
        'n_successful_starts': convergence_info['n_successful'],
        'n_near_best': convergence_info['n_near_best'],
        'at_bounds': at_bounds
    }

    # Compute Hessian-based diagnostics if requested
    if compute_diagnostics:
        if verbose:
            print("    Computing Hessian diagnostics...", flush=True)
        # Convert bounded params to unconstrained space for Hessian computation
        # (the unbounded objective + Hessian SEs use the transform-based parameterization)
        x_unconstrained = params_to_unconstrained(best_params, model)

        # Check gradient norm at optimum (in unconstrained space)
        grad_norm, grad_converged = check_gradient_norm(objective, x_unconstrained)
        result['grad_norm'] = grad_norm

        # Compute Hessian diagnostics (SEs, correlations, condition number)
        hess_diag = compute_hessian_diagnostics(objective, x_unconstrained, model)

        if hess_diag['success']:
            result['hessian_condition'] = hess_diag['condition_number']
            result['hessian_invertible'] = hess_diag['hessian_invertible']

            # Add standard errors for each parameter
            se_bounded = hess_diag.get('se_bounded', {})
            for param, se in se_bounded.items():
                result[f'{param}_se'] = se

            # Compute 95% confidence intervals
            ci = compute_confidence_intervals(best_params, se_bounded)
            for param, (lower, upper) in ci.items():
                result[f'{param}_ci_lower'] = lower
                result[f'{param}_ci_upper'] = upper

            # Check for high correlations (identifiability issues)
            correlations = hess_diag.get('correlations', {})
            high_corr = get_high_correlations(correlations, threshold=0.9)
            result['high_correlations'] = high_corr
        else:
            # Hessian computation failed
            result['hessian_condition'] = np.nan
            result['hessian_invertible'] = False
            result['hessian_error'] = hess_diag.get('error', 'Unknown error')

    return result

# =============================================================================
# Checkpoint Functions for Incremental Saving
# =============================================================================

def _get_checkpoint_path(output_dir: Path, model: str) -> Path:
    """Get path to checkpoint file for incremental saves."""
    return output_dir / f'{model}_checkpoint.csv'

def _load_checkpoint(checkpoint_path: Path) -> tuple[pd.DataFrame, set]:
    """Load existing checkpoint if it exists.

    Returns:
        tuple of (results_df, completed_participant_ids)
    """
    if checkpoint_path.exists():
        df = pd.read_csv(checkpoint_path)
        # Convert boolean columns that CSV reads as strings
        bool_columns = ['converged', 'hessian_invertible']
        for col in bool_columns:
            if col in df.columns:
                # Handle string "True"/"False" from CSV
                df[col] = df[col].apply(lambda x: str(x).lower() == 'true' if pd.notna(x) else False)
        completed = set(df['participant_id'].unique())
        return df, completed
    return pd.DataFrame(), set()

def _save_checkpoint(result: dict, checkpoint_path: Path, is_first: bool = False):
    """Append a single result to checkpoint file.

    Args:
        result: Fit result dictionary for one participant
        checkpoint_path: Path to checkpoint CSV
        is_first: If True, write header; if False, append without header
    """
    df = pd.DataFrame([result])
    if is_first or not checkpoint_path.exists():
        df.to_csv(checkpoint_path, index=False, mode='w')
    else:
        df.to_csv(checkpoint_path, index=False, mode='a', header=False)

# =============================================================================
# Multi-Participant Fitting
# =============================================================================

def prepare_participant_data(
    data: pd.DataFrame,
    participant_id: str,
    model: str = 'qlearning',
    pad_blocks: bool = True,
    max_trials: int = MAX_TRIALS_PER_BLOCK
) -> dict:
    """
    Prepare data for a single participant with optional block padding.

    Block padding standardizes all blocks to the same size, which dramatically
    reduces JAX compilation overhead. Without padding, JAX compiles a separate
    kernel for each unique block size (5 sizes = 5 compilations per participant).
    With padding, JAX compiles ONE kernel that gets reused for all blocks.

    Args:
        data: Full DataFrame with all participants
        participant_id: ID of participant to extract
        model: Model type (determines whether to include set_sizes)
        pad_blocks: Whether to pad blocks to fixed size (default: True)
        max_trials: Target size for padded blocks (default: MAX_TRIALS_PER_BLOCK=100)

    Returns:
        Dictionary with block-organized arrays. If pad_blocks=True, includes
        'masks_blocks' with 1.0 for real trials and 0.0 for padding.
    """
    pdata = data[data['sona_id'] == participant_id].copy()

    stimuli_blocks = []
    actions_blocks = []
    rewards_blocks = []
    set_sizes_blocks = []
    masks_blocks = []

    for block_num in sorted(pdata['block'].unique()):
        block_data = pdata[pdata['block'] == block_num]

        # Extract raw arrays
        stimuli = jnp.array(block_data['stimulus'].values, dtype=jnp.int32)
        actions = jnp.array(block_data['key_press'].values, dtype=jnp.int32)
        rewards = jnp.array(block_data['reward'].values, dtype=jnp.float32)

        if model in ('wmrl', 'wmrl_m3', 'wmrl_m5') and 'set_size' in block_data.columns:
            set_sizes = jnp.array(block_data['set_size'].values, dtype=jnp.int32)
        else:
            set_sizes = None

        if pad_blocks:
            # Pad to fixed size for JAX compilation efficiency
            if set_sizes is not None:
                stimuli_pad, actions_pad, rewards_pad, set_sizes_pad, mask = pad_block_to_max(
                    stimuli, actions, rewards, max_trials=max_trials, set_sizes=set_sizes
                )
                set_sizes_blocks.append(set_sizes_pad)
            else:
                stimuli_pad, actions_pad, rewards_pad, mask = pad_block_to_max(
                    stimuli, actions, rewards, max_trials=max_trials
                )

            stimuli_blocks.append(stimuli_pad)
            actions_blocks.append(actions_pad)
            rewards_blocks.append(rewards_pad)
            masks_blocks.append(mask)
        else:
            # No padding (original behavior)
            stimuli_blocks.append(stimuli)
            actions_blocks.append(actions)
            rewards_blocks.append(rewards)
            if set_sizes is not None:
                set_sizes_blocks.append(set_sizes)

    # Pad to fixed number of blocks for consistent JAX compilation
    # This eliminates recompilation when participants have different block counts
    if pad_blocks and masks_blocks:
        (stimuli_blocks, actions_blocks, rewards_blocks,
         masks_blocks, set_sizes_blocks) = pad_blocks_to_max(
            stimuli_blocks, actions_blocks, rewards_blocks,
            masks_blocks,
            max_blocks=MAX_BLOCKS,
            set_sizes_blocks=set_sizes_blocks if model in ('wmrl', 'wmrl_m3', 'wmrl_m5') else None
        )

    result = {
        'stimuli_blocks': stimuli_blocks,
        'actions_blocks': actions_blocks,
        'rewards_blocks': rewards_blocks,
    }

    if pad_blocks:
        result['masks_blocks'] = masks_blocks

    if model in ('wmrl', 'wmrl_m3', 'wmrl_m5'):
        result['set_sizes_blocks'] = set_sizes_blocks

    return result

def _fit_single_participant_worker(args: tuple) -> dict:
    """
    Picklable worker function for parallel fitting.

    This wrapper is needed because joblib requires picklable functions.
    The function receives all arguments as a tuple and unpacks them.

    Args:
        args: tuple of (pid, data_dict, model, n_starts, seed,
               [compute_diagnostics, [verbose, [participant_index]]])

    Returns:
        Fit result dictionary with participant_id added
    """
    # Handle variable number of arguments for backward compatibility
    if len(args) == 8:
        pid, data_dict, model, n_starts, seed, compute_diagnostics, verbose, participant_index = args
    elif len(args) == 7:
        pid, data_dict, model, n_starts, seed, compute_diagnostics, verbose = args
        participant_index = 0
    elif len(args) == 6:
        pid, data_dict, model, n_starts, seed, compute_diagnostics = args
        verbose = False
        participant_index = 0
    else:
        pid, data_dict, model, n_starts, seed = args
        compute_diagnostics = True  # Default
        verbose = False
        participant_index = 0

    result = fit_participant_mle(
        stimuli_blocks=data_dict['stimuli_blocks'],
        actions_blocks=data_dict['actions_blocks'],
        rewards_blocks=data_dict['rewards_blocks'],
        set_sizes_blocks=data_dict.get('set_sizes_blocks'),
        masks_blocks=data_dict.get('masks_blocks'),  # Pass masks for padded blocks
        model=model,
        n_starts=n_starts,
        seed=seed,
        compute_diagnostics=compute_diagnostics,
        verbose=verbose,
        participant_index=participant_index,
    )
    result['participant_id'] = pid
    return result

def fit_all_participants(
    data: pd.DataFrame,
    model: str = 'qlearning',
    n_starts: int = 50,
    seed: int = 42,
    verbose: bool = True,
    n_jobs: int = 1,
    compute_diagnostics: bool = False,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    """
    Fit all participants using MLE.

    Args:
        data: DataFrame with columns [sona_id, block, stimulus, key_press, reward, set_size]
        model: 'qlearning', 'wmrl', or 'wmrl_m3'
        n_starts: Number of random starting points per participant
        seed: Random seed
        verbose: Show progress bar
        n_jobs: Number of parallel workers (1 = sequential, -1 = all cores)
        compute_diagnostics: Whether to compute Hessian-based diagnostics (default: True)
        output_dir: Output directory for checkpoint files (enables incremental saving)

    Returns:
        tuple of (fits_df, timing_info, timing_records):
        - fits_df: DataFrame with fitted parameters and diagnostics
        - timing_info: Summary timing dict for extrapolation
        - timing_records: list of per-participant timing dicts for logging
    """
    import time

    from joblib import Parallel, delayed

    participants = data['sona_id'].unique()
    n_participants = len(participants)

    # Track timing for ETA estimation
    start_time = time.time()

    # Track memory usage throughout
    initial_mem_mb = log_memory_usage("START", verbose=verbose)
    peak_mem_mb = initial_mem_mb
    if HAS_PSUTIL:
        process = psutil.Process()
    else:
        process = None

    if verbose:
        start_datetime = datetime.now()
        print(f"\n{'='*60}")
        print(f"MLE Fitting: {model.upper()}")
        print(f"{'='*60}")
        print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Participants: {n_participants}")
        print(f"Random starts per participant: {n_starts}")
        print(f"Parallel workers: {n_jobs if n_jobs > 0 else 'all available cores'}")
        print(f"{'='*60}\n")

    # Prepare all participant data upfront (needed for parallel execution)
    if verbose:
        print("Preparing participant data...", flush=True)
        prep_start = time.time()
    participant_args = []
    for i, pid in enumerate(participants):
        pdata = prepare_participant_data(data, pid, model)
        participant_args.append((pid, pdata, model, n_starts, seed + i))
        if verbose and (i + 1) % 10 == 0:
            print(f"  Prepared {i + 1}/{n_participants} participants...", flush=True)
    if verbose:
        print(f"  Data preparation complete ({time.time() - prep_start:.1f}s)\n", flush=True)

    # Execute fitting
    timing_records = []  # Per-participant timing for log file

    if n_jobs == 1:
        # Check for GPU — use vectorized GPU path if available
        gpu_devices = [d for d in jax.devices() if d.platform == 'gpu']
        if gpu_devices:
            if verbose:
                print(f"GPU detected ({gpu_devices[0]}). Using vectorized GPU fitting path.\n")
            return fit_all_gpu(
                data=data,
                model=model,
                n_starts=n_starts,
                seed=seed,
                verbose=verbose,
                compute_diagnostics=compute_diagnostics,
            )

        # Sequential CPU execution with per-participant progress and incremental saving
        results = []
        fit_times = []

        # Load checkpoint if resuming (only for sequential execution with output_dir)
        checkpoint_path = _get_checkpoint_path(output_dir, model) if output_dir else None
        completed_pids = set()
        if checkpoint_path and checkpoint_path.exists():
            checkpoint_df, completed_pids = _load_checkpoint(checkpoint_path)
            results = checkpoint_df.to_dict('records')
            if verbose and completed_pids:
                print(f"Resuming from checkpoint: {len(completed_pids)} participants already completed")

        for i, args in enumerate(participant_args):
            pid = args[0]
            data_dict = args[1]

            # Skip if already completed (resume mode)
            if pid in completed_pids:
                if verbose:
                    print(f"[{i+1:3d}/{n_participants}] Skipping participant {pid} (already in checkpoint)")
                continue

            # Count real trials (using masks if available, otherwise array lengths)
            if 'masks_blocks' in data_dict and data_dict['masks_blocks'] is not None:
                n_trials = sum(int(jnp.sum(m)) for m in data_dict['masks_blocks'])
            else:
                n_trials = sum(len(s) for s in data_dict['stimuli_blocks'])
            n_blocks = len(data_dict['stimuli_blocks'])

            # Track memory before fit
            mem_before_mb = log_memory_usage(f"PRE-FIT participant {i+1}", verbose=False)

            # Calculate percentage and ETA
            pct_complete = 100.0 * (i) / n_participants
            if fit_times and len(fit_times) >= 1:
                # Use steady-state average (skip first which includes JIT)
                steady_times = fit_times[1:] if len(fit_times) > 1 else fit_times
                avg_time_for_eta = np.mean(steady_times)
                remaining_participants = n_participants - i
                eta_seconds = remaining_participants * avg_time_for_eta
                if eta_seconds >= 60:
                    eta_str = f" | ETA: {eta_seconds/60:.1f}min"
                else:
                    eta_str = f" | ETA: {eta_seconds:.0f}s"
            else:
                eta_str = ""

            if verbose:
                # Print progress header for this participant with timestamp, percentage, ETA
                print(f"\n[{timestamp()}] [{i+1:3d}/{n_participants}] ({pct_complete:5.1f}%) Fitting {pid} ({n_trials} trials){eta_str}", flush=True)

            fit_start = time.time()
            result = _fit_single_participant_worker(args + (compute_diagnostics, verbose, i))
            fit_time = time.time() - fit_start
            fit_times.append(fit_time)
            results.append(result)

            # INCREMENTAL SAVE after each participant (crash protection)
            if checkpoint_path:
                is_first = (len(results) == 1 and not completed_pids)
                _save_checkpoint(result, checkpoint_path, is_first=is_first)

            # Track memory after fit
            mem_after_mb = log_memory_usage(f"POST-FIT participant {i+1}", verbose=False)
            peak_mem_mb = max(peak_mem_mb, mem_after_mb)

            # Log JIT compilation memory spike (first participant)
            if i == 0 and verbose:
                log_memory_usage("POST-JIT (compilation complete)", verbose=True)

            # Record timing for this participant
            timing_records.append({
                'participant_id': pid,
                'n_trials': n_trials,
                'n_blocks': n_blocks,
                'fit_time_s': fit_time,
                'is_first_fit': (i == 0),  # First fit includes JIT compilation
                'memory_before_mb': mem_before_mb,
                'memory_after_mb': mem_after_mb,
                'nll': result.get('nll', np.nan),
                'converged': result.get('converged', False)
            })

            if verbose:
                # Print result summary with status indicator
                status_indicator = "[OK]" if result['converged'] else "[!!]"
                nll_str = f"NLL={result['nll']:.1f}" if not np.isnan(result['nll']) else "NLL=NaN"
                r2_str = f"R²={result.get('pseudo_r2', 0):.2f}" if 'pseudo_r2' in result and not np.isnan(result.get('pseudo_r2', np.nan)) else ""
                near_str = f"near_best={result.get('n_near_best', '?')}" if 'n_near_best' in result else ""
                print(f"      Result: {status_indicator} {nll_str}, {r2_str}, {near_str}, total={fit_time:.1f}s", flush=True)

                # Every 10 participants, print a formatted progress box
                if (i + 1) % 10 == 0:
                    elapsed_time = time.time() - start_time
                    avg_time = np.mean(fit_times)
                    # Use steady-state average for remaining estimate
                    steady_times = fit_times[1:] if len(fit_times) > 1 else fit_times
                    steady_avg = np.mean(steady_times) if steady_times else avg_time
                    n_converged = sum(r['converged'] for r in results)
                    remaining = (n_participants - i - 1) * steady_avg
                    current_mem_mb = log_memory_usage(f"checkpoint {i+1}", verbose=False)
                    mem_str = f"{current_mem_mb/1024:.2f}GB" if current_mem_mb > 0 else "N/A"
                    conv_rate = 100.0 * n_converged / len(results)

                    print()
                    print("    +----------------------------------------------------------")
                    print(f"    | PROGRESS: {i+1}/{n_participants} ({100.0*(i+1)/n_participants:.0f}%)")
                    print(f"    | Elapsed: {elapsed_time/60:.1f}min | Remaining: ~{remaining/60:.1f}min")
                    print(f"    | Converged: {n_converged}/{len(results)} ({conv_rate:.0f}%) | Avg: {steady_avg:.1f}s/participant")
                    print(f"    | Memory: {mem_str}")
                    print("    +----------------------------------------------------------")
                    print()
    else:
        # Parallel execution using joblib

        # Warmup JAX compilation in main process (populates disk cache)
        # This ensures workers can read cached compilations instead of
        # each independently compiling, reducing overhead from ~30s to ~5s
        jit_start = time.time()
        warmup_jax_compilation(model, verbose=verbose)
        jit_elapsed = time.time() - jit_start
        if verbose:
            print(f"[{timestamp()}] JAX compilation complete: {jit_elapsed:.1f}s\n")

        if verbose:
            print(f"Running parallel fitting with {n_jobs} workers...")

        # Add compute_diagnostics to args for parallel workers
        parallel_args = [args + (compute_diagnostics,) for args in participant_args]

        # joblib handles the parallel execution
        # verbose=10 shows progress; verbose=0 is silent
        try:
            results = Parallel(
                n_jobs=n_jobs,
                verbose=10 if verbose else 0,
                backend='loky'  # Process-based backend for true parallelism
            )(
                delayed(_fit_single_participant_worker)(args)
                for args in parallel_args
            )
        except Exception as e:
            error_msg = str(e)
            if 'SIGKILL' in error_msg or 'TerminatedWorkerError' in str(type(e).__name__):
                print(f"\n{'='*60}")
                print("ERROR: Worker processes killed (likely out of memory)")
                print(f"{'='*60}")
                print("The parallel workers were terminated by the OS (SIGKILL).")
                print("This typically means JAX + joblib exceeded available memory.")
                print("\nSuggested fixes:")
                print("  1. Reduce --n-jobs (try 8 or 4 instead of 16)")
                print("  2. Request more memory in SLURM (--mem=64G)")
                print("  3. Use sequential fitting (--n-jobs 1)")
                print(f"{'='*60}\n")
            raise

        # For parallel execution, we can't track individual times
        # Create timing records from results (without detailed per-participant timing)
        fit_times = []
        for i, (args, result) in enumerate(zip(participant_args, results)):
            pid = args[0]
            data_dict = args[1]
            if 'masks_blocks' in data_dict and data_dict['masks_blocks'] is not None:
                n_trials = sum(int(jnp.sum(m)) for m in data_dict['masks_blocks'])
            else:
                n_trials = sum(len(s) for s in data_dict['stimuli_blocks'])
            n_blocks = len(data_dict['stimuli_blocks'])

            timing_records.append({
                'participant_id': pid,
                'n_trials': n_trials,
                'n_blocks': n_blocks,
                'fit_time_s': np.nan,  # Not available in parallel mode
                'is_first_fit': False,
                'memory_before_mb': np.nan,
                'memory_after_mb': np.nan,
                'nll': result.get('nll', np.nan),
                'converged': result.get('converged', False)
            })

    # Calculate timing (always, for timing_info return)
    total_time = time.time() - start_time

    # Update peak memory
    final_mem_mb = log_memory_usage("FINAL", verbose=verbose)
    peak_mem_mb = max(peak_mem_mb, final_mem_mb)

    # Print enhanced summary with completion details
    if verbose:
        end_datetime = datetime.now()
        n_converged = sum(r['converged'] for r in results)
        conv_rate = 100.0 * n_converged / n_participants if n_participants > 0 else 0

        print(f"\n{'='*60}")
        print("FITTING COMPLETE")
        print(f"{'='*60}")
        print(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {total_time/60:.1f} min ({total_time:.0f}s)")
        print(f"Participants: {n_participants} | Converged: {n_converged} ({conv_rate:.0f}%)")

        # Timing breakdown
        if fit_times:
            jit_time = fit_times[0] if fit_times else 0
            steady_times = fit_times[1:] if len(fit_times) > 1 else []
            steady_avg = np.mean(steady_times) if steady_times else jit_time
            print(f"JIT compilation: {jit_time:.1f}s | Avg steady-state: {steady_avg:.1f}s/participant")
        else:
            # Parallel mode
            avg_per_participant = total_time / n_participants if n_participants > 0 else 0
            print(f"Avg time per participant: {avg_per_participant:.1f}s (parallel, {n_jobs} workers)")

        # Memory summary
        if peak_mem_mb > 0:
            print(f"Peak memory: {peak_mem_mb/1024:.2f}GB")

        # Report diagnostic summary if computed
        if compute_diagnostics:
            hess_success = sum(1 for r in results if r.get('hessian_invertible', False))
            print(f"Hessian diagnostics: {hess_success}/{n_participants} successful")

        print(f"{'='*60}\n")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Build column order: params → fit metrics → diagnostics → convergence info
    if model == 'qlearning':
        param_cols = QLEARNING_PARAMS
    elif model == 'wmrl':
        param_cols = WMRL_PARAMS
    elif model == 'wmrl_m3':
        param_cols = WMRL_M3_PARAMS
    elif model == 'wmrl_m5':
        param_cols = WMRL_M5_PARAMS
    else:
        param_cols = []

    # Build comprehensive column order
    cols = ['participant_id'] + param_cols

    # SE columns (if diagnostics computed)
    se_cols = [f'{p}_se' for p in param_cols]
    ci_lower_cols = [f'{p}_ci_lower' for p in param_cols]
    ci_upper_cols = [f'{p}_ci_upper' for p in param_cols]

    # Fit quality metrics
    cols += ['nll', 'aic', 'bic', 'aicc', 'pseudo_r2']

    # Diagnostic columns
    cols += ['grad_norm', 'hessian_condition', 'hessian_invertible']

    # Standard errors and CIs (interleaved: param, se, ci_lower, ci_upper)
    cols += se_cols + ci_lower_cols + ci_upper_cols

    # Convergence info
    cols += ['n_trials', 'converged', 'n_successful_starts', 'n_near_best', 'at_bounds', 'high_correlations']

    # Filter to only columns that exist
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    # Build timing info dict for extrapolation
    timing_info = {
        'total_time': total_time,
        'fit_times': fit_times,
        'jit_time': fit_times[0] if fit_times else None,
        'steady_state_times': fit_times[1:] if len(fit_times) > 1 else [],
        'n_jobs': n_jobs,
        'n_participants': n_participants,
        'initial_mem_mb': initial_mem_mb,
        'peak_mem_mb': peak_mem_mb,
        'model': model,
    }

    return df, timing_info, timing_records

# =============================================================================
# Group Statistics
# =============================================================================

def compute_group_summary(
    fits_df: pd.DataFrame,
    model: str
) -> pd.DataFrame:
    """
    Compute group-level summary statistics.

    Following Senta et al.: mean +/- SEM for each parameter.

    Args:
        fits_df: DataFrame with individual fit results
        model: 'qlearning' or 'wmrl'

    Returns:
        DataFrame with group statistics
    """
    # Only include converged fits
    converged_df = fits_df[fits_df['converged'] == True]

    if len(converged_df) == 0:
        print("Warning: No converged fits!")
        converged_df = fits_df  # Use all if none converged

    summary = summarize_all_parameters(converged_df, model)

    # Convert to DataFrame format
    rows = []
    for param, stats in summary.items():
        rows.append({
            'parameter': param,
            'mean': stats['mean'],
            'sd': stats['sd'],
            'se': stats['se'],
            'ci_lower': stats['ci_lower'],
            'ci_upper': stats['ci_upper'],
            'n': stats['n']
        })

    # Add fit quality metrics
    rows.append({
        'parameter': 'nll',
        'mean': converged_df['nll'].mean(),
        'sd': converged_df['nll'].std(),
        'se': converged_df['nll'].std() / np.sqrt(len(converged_df)),
        'ci_lower': np.nan,
        'ci_upper': np.nan,
        'n': len(converged_df)
    })

    for metric in ['aic', 'bic']:
        rows.append({
            'parameter': metric,
            'mean': converged_df[metric].mean(),
            'sd': converged_df[metric].std(),
            'se': converged_df[metric].std() / np.sqrt(len(converged_df)),
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n': len(converged_df)
        })

    return pd.DataFrame(rows)

# =============================================================================
# Main CLI
# =============================================================================

def load_and_prepare_data(filepath: str, exclude_practice: bool = True) -> pd.DataFrame:
    """Load and prepare data for fitting."""
    data = pd.read_csv(filepath)

    # Required columns
    required = ['sona_id', 'block', 'stimulus', 'key_press']
    for col in required:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    # Create reward column if needed
    if 'reward' not in data.columns and 'correct' in data.columns:
        data['reward'] = data['correct'].astype(float)

    # Exclude practice blocks (blocks 1-2)
    if exclude_practice:
        data = data[data['block'] >= 3].copy()

    # Map key_press if needed (J=0, K=1, L=2)
    if data['key_press'].dtype == object or data['key_press'].max() > 2:
        key_map = {'j': 0, 'k': 1, 'l': 2, 'J': 0, 'K': 1, 'L': 2}
        if data['key_press'].dtype == object:
            data['key_press'] = data['key_press'].map(key_map)

    return data

def main():
    parser = argparse.ArgumentParser(
        description='MLE fitting for RLWM models (Senta et al. methodology)'
    )
    parser.add_argument('--model', type=str, required=True,
                        choices=['qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5'],
                        help='Model to fit (qlearning=M1, wmrl=M2, wmrl_m3=M3, wmrl_m5=M5)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to trial data CSV')
    parser.add_argument('--output', type=str, default='output/mle/',
                        help='Output directory')
    parser.add_argument('--n-starts', type=int, default=50,
                        help='Number of random starting points (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    parser.add_argument('--include-practice', action='store_true',
                        help='Include practice blocks (1-2) in fitting. Default excludes practice.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit to first N participants (for timing/testing)')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel workers (default: 1 = sequential)')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU acceleration if available (requires JAX CUDA)')
    parser.add_argument('--debug-timing', action='store_true',
                        help='Enable detailed timing instrumentation for debugging performance')
    parser.add_argument('--compute-diagnostics', action='store_true',
                        help='Compute Hessian diagnostics (adds 5-30s per participant)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # GPU detection
    gpu_available = False
    gpu_devices = []
    if args.use_gpu:
        try:
            devices = jax.devices()
            gpu_devices = [d for d in devices if d.platform == 'gpu']
            gpu_available = len(gpu_devices) > 0
            if gpu_available:
                # JAX will automatically use GPU when available
                print(f"GPU detected: {gpu_devices}")
            else:
                print("WARNING: --use-gpu specified but no GPU found. Using CPU.")
                print(f"Available devices: {devices}")
        except Exception as e:
            print(f"WARNING: GPU detection failed: {e}")
            print("Falling back to CPU.")

    print(f"\n{'='*60}")
    print(f"MLE Fitting: {args.model.upper()}")
    print(f"{'='*60}")
    print(f"Data: {args.data}")
    print(f"Random starts: {args.n_starts}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output}")
    print(f"Include practice: {args.include_practice}")
    print(f"Parallel workers: {args.n_jobs if args.n_jobs > 0 else 'all available cores'}")
    if args.use_gpu:
        print(f"GPU acceleration: {'enabled' if gpu_available else 'requested but unavailable'}")
    print(f"Hessian diagnostics: {'enabled (adds 5-30s per participant)' if args.compute_diagnostics else 'disabled'}")
    if args.debug_timing:
        print("Debug timing: ENABLED (detailed instrumentation)")
        # Enable JAX compile logging for debugging
        import os
        os.environ['JAX_LOG_COMPILES'] = '1'
        print("  JAX_LOG_COMPILES=1 (will log every JIT compilation)")
    if args.limit:
        print(f"Limit: {args.limit} participants (testing mode)")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    exclude_practice = not args.include_practice
    data = load_and_prepare_data(args.data, exclude_practice=exclude_practice)

    # Exclude participants based on data quality
    initial_n = data['sona_id'].nunique()
    data = data[~data['sona_id'].isin(EXCLUDED_PARTICIPANTS)].copy()
    n_participants = data['sona_id'].nunique()
    n_excluded = initial_n - n_participants

    n_trials = len(data)
    print(f"  Total participants in data: {initial_n}")
    if n_excluded > 0:
        print(f"  Excluded participants: {n_excluded} (insufficient data/duplicates)")
    print(f"  Final sample: {n_participants}")
    print(f"  Total trials: {n_trials}")

    # Apply limit filter if specified
    full_participant_count = n_participants
    if args.limit is not None:
        unique_pids = data['sona_id'].unique()[:args.limit]
        data = data[data['sona_id'].isin(unique_pids)].copy()
        n_participants = len(unique_pids)
        print(f"  Limited to first {args.limit} participants (testing mode)")
    print()

    # Fit all participants
    fits_df, timing_info, timing_records = fit_all_participants(
        data,
        model=args.model,
        n_starts=args.n_starts,
        seed=args.seed,
        verbose=not args.quiet,
        n_jobs=args.n_jobs,
        compute_diagnostics=args.compute_diagnostics,
        output_dir=output_dir,  # For incremental checkpoint saving
    )

    # Compute group summary
    summary_df = compute_group_summary(fits_df, args.model)

    # Timing analysis and extrapolation (when using --limit)
    if timing_info and args.limit:
        print(f"\n{'='*60}")
        print("TIMING ANALYSIS & EXTRAPOLATION")
        print(f"{'='*60}")

        jit_time = timing_info['jit_time']
        steady_times = timing_info['steady_state_times']

        if jit_time is not None:
            print(f"JIT compilation (1st participant): {jit_time:.1f}s")

        if steady_times:
            steady_mean = np.mean(steady_times)
            steady_std = np.std(steady_times)
            print(f"Steady-state per participant: {steady_mean:.1f}s ± {steady_std:.1f}s")

            # Extrapolate to full dataset
            total_n = full_participant_count
            estimated = jit_time + (total_n - 1) * steady_mean
            print(f"\nExtrapolated time for {total_n} participants: {estimated/60:.1f} min")
        elif jit_time is not None:
            # Only 1 participant tested
            total_n = full_participant_count
            # Conservative estimate: assume JIT time includes ~50% overhead
            steady_estimate = jit_time * 0.6
            estimated = jit_time + (total_n - 1) * steady_estimate
            print("\nNote: Only 1 participant tested (includes JIT overhead)")
            print(f"Rough estimate for {total_n} participants: {estimated/60:.1f} min")
            print("(Run with --limit 3 for more accurate extrapolation)")

        # Memory usage if psutil available
        if HAS_PSUTIL:
            process = psutil.Process()
            mem_mb = process.memory_info().rss / 1024 / 1024
            print(f"\nPeak memory usage: {mem_mb:.0f} MB")

        print(f"{'='*60}")

    # Save results
    import json
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fits_path = output_dir / f'{args.model}_individual_fits.csv'
    summary_path = output_dir / f'{args.model}_group_summary.csv'
    timing_log_path = output_dir / f'{args.model}_timing_log.csv'
    performance_path = output_dir / f'{args.model}_performance_summary.json'

    fits_df.to_csv(fits_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # Save timing log (per-participant timing details)
    if timing_records:
        timing_df = pd.DataFrame(timing_records)
        timing_df.to_csv(timing_log_path, index=False)

    # Save performance summary JSON
    performance_summary = {
        'model': args.model,
        'n_participants': n_participants,
        'n_starts': args.n_starts,
        'total_time_s': timing_info.get('total_time', 0),
        'total_time_min': timing_info.get('total_time', 0) / 60,
        'avg_time_per_participant_s': timing_info.get('total_time', 0) / n_participants if n_participants > 0 else 0,
        'jit_compilation_time_s': timing_info.get('jit_time'),
        'steady_state_avg_s': np.mean(timing_info.get('steady_state_times', [])) if timing_info.get('steady_state_times') else None,
        'n_jobs': args.n_jobs,
        'peak_memory_mb': timing_info.get('peak_mem_mb'),
        'initial_memory_mb': timing_info.get('initial_mem_mb'),
        'memory_increase_mb': (timing_info.get('peak_mem_mb', 0) - timing_info.get('initial_mem_mb', 0)) if timing_info.get('peak_mem_mb') else None,
        'timestamp': datetime.now().isoformat(),
        'seed': args.seed,
        'data_file': args.data,
        'n_converged': int(fits_df['converged'].sum()),
        'convergence_rate': float(fits_df['converged'].sum() / n_participants) if n_participants > 0 else 0,
    }

    # Add diagnostic success rate if available
    if 'hessian_invertible' in fits_df.columns:
        performance_summary['hessian_success_count'] = int(fits_df['hessian_invertible'].sum())
        performance_summary['hessian_success_rate'] = float(fits_df['hessian_invertible'].mean())

    with open(performance_path, 'w') as f:
        json.dump(performance_summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    # Print summary
    n_converged = fits_df['converged'].sum()
    print(f"\nConvergence: {n_converged}/{n_participants} ({100*n_converged/n_participants:.1f}%)")

    print(f"\nGroup Statistics (n={n_converged}):")
    print("-" * 50)
    if args.model == 'qlearning':
        param_names = QLEARNING_PARAMS
    elif args.model == 'wmrl':
        param_names = WMRL_PARAMS
    elif args.model == 'wmrl_m3':
        param_names = WMRL_M3_PARAMS
    elif args.model == 'wmrl_m5':
        param_names = WMRL_M5_PARAMS
    else:
        param_names = []
    for param in param_names:
        row = summary_df[summary_df['parameter'] == param].iloc[0]
        print(f"  {param:12s}: {row['mean']:.3f} +/- {row['se']:.3f}  (SD={row['sd']:.3f})")

    print("\nModel Fit:")
    print("-" * 50)
    for metric in ['nll', 'aic', 'bic']:
        row = summary_df[summary_df['parameter'] == metric].iloc[0]
        print(f"  {metric.upper():12s}: {row['mean']:.2f} +/- {row['se']:.2f}")

    # Display pseudo-R² if available
    if 'pseudo_r2' in fits_df.columns:
        avg_r2 = fits_df['pseudo_r2'].mean()
        std_r2 = fits_df['pseudo_r2'].std()
        print(f"  {'PSEUDO-R²':12s}: {avg_r2:.3f} +/- {std_r2:.3f}")

    # Display Hessian diagnostic success rate
    if 'hessian_invertible' in fits_df.columns:
        hess_success = fits_df['hessian_invertible'].sum()
        hess_total = len(fits_df)
        print("\nHessian Diagnostics:")
        print("-" * 50)
        print(f"  Successfully computed: {hess_success}/{hess_total}")
        if 'hessian_condition' in fits_df.columns:
            valid_cond = fits_df[fits_df['hessian_condition'].notna()]['hessian_condition']
            if len(valid_cond) > 0:
                print(f"  Median condition number: {valid_cond.median():.1f}")
                high_cond = (valid_cond > 1000).sum()
                if high_cond > 0:
                    print(f"  Warning: {high_cond} participants with condition > 1000 (identifiability issues)")

    # Check for parameters at bounds
    all_at_bounds = []
    for bounds_list in fits_df['at_bounds']:
        if isinstance(bounds_list, list):
            all_at_bounds.extend(bounds_list)

    if all_at_bounds:
        from collections import Counter
        bounds_counts = Counter(all_at_bounds)
        print("\nWarning: Parameters hitting bounds:")
        for param, count in bounds_counts.items():
            print(f"  {param}: {count} participants")

    print(f"\n{'='*60}")
    print("Results saved to:")
    print(f"  Individual fits:     {fits_path}")
    print(f"  Group summary:       {summary_path}")
    if timing_records:
        print(f"  Timing log:          {timing_log_path}")
    print(f"  Performance summary: {performance_path}")
    print(f"{'='*60}")

    # Final status summary - clear indication of success/failure
    n_failed = fits_df['nll'].isna().sum()
    n_total = len(fits_df)

    print(f"\n{'='*60}")
    if n_failed == 0 and n_converged == n_total:
        print(f"STATUS: SUCCESS - All {n_total} participants fit successfully")
    elif n_failed == 0:
        print(f"STATUS: COMPLETE - {n_total} participants fit ({n_converged} converged, {n_total - n_converged} did not converge)")
    else:
        print(f"STATUS: PARTIAL - {n_total - n_failed}/{n_total} participants fit, {n_failed} FAILED")
        failed_pids = fits_df[fits_df['nll'].isna()]['participant_id'].tolist()
        print(f"  Failed participants: {failed_pids[:10]}{'...' if len(failed_pids) > 10 else ''}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
