"""
PyMC Model Specifications for Bayesian Fitting of RLWM Models

Implements hierarchical Bayesian models for:
1. Q-learning model
2. WM-RL hybrid model

Hierarchical structure:
- Group level: μ_param, σ_param (population mean and std)
- Individual level: param_i ~ Normal(μ_param, σ_param) for each participant
- Bounded transforms to ensure parameters stay in valid ranges

Likelihood:
- Categorical(choice | softmax(Q-values)) for each trial
- Models compute Q-values/action probs trial-by-trial

**UNIFIED ARCHITECTURE:**
This module uses the agent classes directly from models/ via the unified_simulator
module. This eliminates code duplication between parameter sweeps and PyMC fitting.

Usage:
    import pymc as pm
    from fitting.pymc_models import build_qlearning_model

    with build_qlearning_model(data, participant_ids) as model:
        # Use Metropolis sampler since agent classes are pure Python
        trace = pm.sample(2000, tune=1000, chains=4, step=pm.Metropolis())
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

try:
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    print("Warning: PyMC not installed. Install with: pip install pymc")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import PyMCParams, ModelParams, TaskParams
from scripts.simulations.unified_simulator import (
    simulate_qlearning_for_likelihood,
    simulate_wmrl_for_likelihood
)


# ============================================================================
# Q-LEARNING MODEL
# ============================================================================
# NOTE: simulate_qlearning_for_likelihood is imported from unified_simulator
# This uses the QLearningAgent class directly - NO CODE DUPLICATION


def build_qlearning_model(
    data: pd.DataFrame,
    participant_col: str = 'sona_id',
    stimulus_col: str = 'stimulus',
    action_col: str = 'key_press',
    reward_col: str = 'correct',
) -> pm.Model:
    """
    Build hierarchical Bayesian Q-learning model.

    Parameters
    ----------
    data : pd.DataFrame
        Trial-level data with columns: participant, stimulus, action, reward
    participant_col : str
        Column name for participant IDs
    stimulus_col : str
        Column name for stimulus IDs (0-indexed)
    action_col : str
        Column name for actions (0-indexed)
    reward_col : str
        Column name for rewards (0 or 1)

    Returns
    -------
    pm.Model
        PyMC model ready for sampling
    """
    if not PYMC_AVAILABLE:
        raise ImportError("PyMC is required. Install with: pip install pymc")

    # Get unique participants
    participants = data[participant_col].unique()
    n_participants = len(participants)
    participant_map = {p: i for i, p in enumerate(participants)}

    # Prepare data arrays
    participant_idx = data[participant_col].map(participant_map).values
    stimuli = data[stimulus_col].values.astype(int)
    actions = data[action_col].values.astype(int)
    rewards = data[reward_col].values.astype(float)

    with pm.Model() as model:
        # ====================================================================
        # PRIORS: Group-level (population)
        # ====================================================================

        # Learning rate for positive PE: bounded [0, 1]
        mu_alpha_pos = pm.Beta('mu_alpha_pos', alpha=3, beta=2)  # Mean ~0.6 (faster learning from correct)
        sigma_alpha_pos = pm.HalfNormal('sigma_alpha_pos', sigma=0.3)

        # Learning rate for negative PE: bounded [0, 1]
        mu_alpha_neg = pm.Beta('mu_alpha_neg', alpha=2, beta=3)  # Mean ~0.4 (slower learning from incorrect)
        sigma_alpha_neg = pm.HalfNormal('sigma_alpha_neg', sigma=0.3)

        # Inverse temperature (beta): positive
        mu_beta = pm.Gamma('mu_beta', alpha=2, beta=1)  # Mean ~2
        sigma_beta = pm.HalfNormal('sigma_beta', sigma=2)

        # ====================================================================
        # PRIORS: Individual-level
        # ====================================================================

        # Learning rate (positive PE) per participant (transformed to [0,1])
        alpha_pos_raw = pm.Normal('alpha_pos_raw', mu=0, sigma=1, shape=n_participants)
        alpha_pos = pm.Deterministic(
            'alpha_pos',
            pm.math.invlogit(pm.math.log(mu_alpha_pos / (1 - mu_alpha_pos)) + sigma_alpha_pos * alpha_pos_raw)
        )

        # Learning rate (negative PE) per participant (transformed to [0,1])
        alpha_neg_raw = pm.Normal('alpha_neg_raw', mu=0, sigma=1, shape=n_participants)
        alpha_neg = pm.Deterministic(
            'alpha_neg',
            pm.math.invlogit(pm.math.log(mu_alpha_neg / (1 - mu_alpha_neg)) + sigma_alpha_neg * alpha_neg_raw)
        )

        # Inverse temperature per participant (positive)
        beta_raw = pm.Normal('beta_raw', mu=0, sigma=1, shape=n_participants)
        beta = pm.Deterministic(
            'beta',
            pm.math.exp(pm.math.log(mu_beta) + sigma_beta * beta_raw)
        )

        # ====================================================================
        # LIKELIHOOD: Trial-by-trial choices
        # ====================================================================

        # For each participant, simulate Q-learning and get action probs
        # This is done using a custom likelihood function

        def logp_func(actions, participant_idx, stimuli, rewards, alpha_pos, alpha_neg, beta):
            """Compute log-likelihood for all trials with asymmetric learning rates."""
            logp_total = 0.0

            for p in range(n_participants):
                # Get trials for this participant
                mask = participant_idx == p
                if not np.any(mask):
                    continue

                p_stimuli = stimuli[mask]
                p_actions = actions[mask]
                p_rewards = rewards[mask]

                # Simulate Q-learning for this participant
                # Uses QLearningAgent class via unified_simulator
                action_probs = simulate_qlearning_for_likelihood(
                    stimuli=p_stimuli,
                    rewards=p_rewards,
                    alpha_pos=alpha_pos[p],
                    alpha_neg=alpha_neg[p],
                    beta=beta[p],
                    gamma=ModelParams.GAMMA_DEFAULT,
                    q_init=ModelParams.Q_INIT_VALUE
                )

                # Log-likelihood: sum of log P(action_t | model)
                trial_logp = np.log(action_probs[np.arange(len(p_actions)), p_actions] + 1e-10)
                logp_total += np.sum(trial_logp)

            return logp_total

        # Create custom likelihood
        pm.Potential(
            'likelihood',
            logp_func(actions, participant_idx, stimuli, rewards, alpha_pos, alpha_neg, beta)
        )

    return model


# ============================================================================
# WM-RL HYBRID MODEL
# ============================================================================
# NOTE: simulate_wmrl_for_likelihood is imported from unified_simulator
# This uses the WMRLHybridAgent class directly - NO CODE DUPLICATION


def build_wmrl_model(
    data: pd.DataFrame,
    participant_col: str = 'sona_id',
    stimulus_col: str = 'stimulus',
    action_col: str = 'key_press',
    reward_col: str = 'correct',
    set_size_col: str = 'set_size',
) -> pm.Model:
    """
    Build hierarchical Bayesian WM-RL hybrid model (matrix-based architecture).

    Model uses:
    - Asymmetric learning rates (alpha_pos, alpha_neg) for RL component
    - Global WM decay (phi) toward baseline
    - Adaptive weighting (rho) based on capacity and set size
    - Separate inverse temperatures for WM (beta_wm) and RL (beta)

    Parameters
    ----------
    data : pd.DataFrame
        Trial-level data with columns: participant, stimulus, action, reward, set_size
    participant_col, stimulus_col, action_col, reward_col, set_size_col : str
        Column names

    Returns
    -------
    pm.Model
        PyMC model ready for sampling
    """
    if not PYMC_AVAILABLE:
        raise ImportError("PyMC is required")

    # Get participants
    participants = data[participant_col].unique()
    n_participants = len(participants)
    participant_map = {p: i for i, p in enumerate(participants)}

    # Data arrays
    participant_idx = data[participant_col].map(participant_map).values
    stimuli = data[stimulus_col].values.astype(int)
    actions = data[action_col].values.astype(int)
    rewards = data[reward_col].values.astype(float)
    set_sizes = data[set_size_col].values.astype(int)

    with pm.Model() as model:
        # ====================================================================
        # PRIORS: Group-level (population)
        # ====================================================================

        # Learning rates for RL component (asymmetric)
        mu_alpha_pos = pm.Beta('mu_alpha_pos', alpha=3, beta=2)  # Mean ~0.6
        sigma_alpha_pos = pm.HalfNormal('sigma_alpha_pos', sigma=0.3)

        mu_alpha_neg = pm.Beta('mu_alpha_neg', alpha=2, beta=3)  # Mean ~0.4
        sigma_alpha_neg = pm.HalfNormal('sigma_alpha_neg', sigma=0.3)

        # Inverse temperature for RL component
        mu_beta = pm.Gamma('mu_beta', alpha=2, beta=1)  # Mean ~2
        sigma_beta = pm.HalfNormal('sigma_beta', sigma=2)

        # Inverse temperature for WM component
        mu_beta_wm = pm.Gamma('mu_beta_wm', alpha=3, beta=1)  # Mean ~3 (slightly higher)
        sigma_beta_wm = pm.HalfNormal('sigma_beta_wm', sigma=2)

        # WM capacity (K)
        mu_capacity = pm.TruncatedNormal('mu_capacity', mu=4, sigma=1.5, lower=1, upper=7)
        sigma_capacity = pm.HalfNormal('sigma_capacity', sigma=1)

        # WM global decay parameter (phi) - bounded [0, 1]
        mu_phi = pm.Beta('mu_phi', alpha=1, beta=9)  # Mean ~0.1 (slow decay)
        sigma_phi = pm.HalfNormal('sigma_phi', sigma=0.2)

        # Base WM reliance (rho) - bounded [0, 1]
        mu_rho = pm.Beta('mu_rho', alpha=7, beta=3)  # Mean ~0.7 (high WM reliance)
        sigma_rho = pm.HalfNormal('sigma_rho', sigma=0.3)

        # ====================================================================
        # PRIORS: Individual-level
        # ====================================================================

        # RL learning rates (asymmetric)
        alpha_pos_raw = pm.Normal('alpha_pos_raw', mu=0, sigma=1, shape=n_participants)
        alpha_pos = pm.Deterministic(
            'alpha_pos',
            pm.math.invlogit(pm.math.log(mu_alpha_pos / (1 - mu_alpha_pos)) + sigma_alpha_pos * alpha_pos_raw)
        )

        alpha_neg_raw = pm.Normal('alpha_neg_raw', mu=0, sigma=1, shape=n_participants)
        alpha_neg = pm.Deterministic(
            'alpha_neg',
            pm.math.invlogit(pm.math.log(mu_alpha_neg / (1 - mu_alpha_neg)) + sigma_alpha_neg * alpha_neg_raw)
        )

        # Inverse temperatures
        beta_raw = pm.Normal('beta_raw', mu=0, sigma=1, shape=n_participants)
        beta = pm.Deterministic(
            'beta',
            pm.math.exp(pm.math.log(mu_beta) + sigma_beta * beta_raw)
        )

        beta_wm_raw = pm.Normal('beta_wm_raw', mu=0, sigma=1, shape=n_participants)
        beta_wm = pm.Deterministic(
            'beta_wm',
            pm.math.exp(pm.math.log(mu_beta_wm) + sigma_beta_wm * beta_wm_raw)
        )

        # WM capacity (clipped to [1, 7])
        capacity_raw = pm.Normal('capacity_raw', mu=0, sigma=1, shape=n_participants)
        capacity = pm.Deterministic(
            'capacity',
            pm.math.clip(mu_capacity + sigma_capacity * capacity_raw, 1, 7)
        )

        # WM decay parameter (phi) bounded [0, 1]
        phi_raw = pm.Normal('phi_raw', mu=0, sigma=1, shape=n_participants)
        phi = pm.Deterministic(
            'phi',
            pm.math.invlogit(pm.math.log(mu_phi / (1 - mu_phi)) + sigma_phi * phi_raw)
        )

        # Base WM reliance (rho) bounded [0, 1]
        rho_raw = pm.Normal('rho_raw', mu=0, sigma=1, shape=n_participants)
        rho = pm.Deterministic(
            'rho',
            pm.math.invlogit(pm.math.log(mu_rho / (1 - mu_rho)) + sigma_rho * rho_raw)
        )

        # ====================================================================
        # LIKELIHOOD: Trial-by-trial choices
        # ====================================================================

        def logp_func(actions, participant_idx, stimuli, rewards, set_sizes,
                     alpha_pos, alpha_neg, beta, beta_wm, capacity, phi, rho):
            """Compute log-likelihood for all trials with matrix-based WM-RL."""
            logp_total = 0.0

            for p in range(n_participants):
                mask = participant_idx == p
                if not np.any(mask):
                    continue

                p_stimuli = stimuli[mask]
                p_actions = actions[mask]
                p_rewards = rewards[mask]
                p_set_sizes = set_sizes[mask]

                # Uses WMRLHybridAgent class via unified_simulator
                action_probs = simulate_wmrl_for_likelihood(
                    stimuli=p_stimuli,
                    rewards=p_rewards,
                    set_sizes=p_set_sizes,
                    alpha_pos=alpha_pos[p],
                    alpha_neg=alpha_neg[p],
                    beta=beta[p],
                    beta_wm=beta_wm[p],
                    capacity=int(capacity[p]),
                    phi=phi[p],
                    rho=rho[p],
                    gamma=ModelParams.GAMMA_DEFAULT,
                    q_init=ModelParams.Q_INIT_VALUE,
                    wm_init=ModelParams.WM_INIT_VALUE
                )

                trial_logp = np.log(action_probs[np.arange(len(p_actions)), p_actions] + 1e-10)
                logp_total += np.sum(trial_logp)

            return logp_total

        pm.Potential(
            'likelihood',
            logp_func(actions, participant_idx, stimuli, rewards, set_sizes,
                     alpha_pos, alpha_neg, beta, beta_wm, capacity, phi, rho)
        )

    return model


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def prepare_data_for_fitting(
    data: pd.DataFrame,
    participant_col: str = 'sona_id',
    block_col: str = 'block',
    trial_col: str = 'trial',
    stimulus_col: str = 'stimulus',
    action_col: str = 'key_press',
    correct_col: str = 'correct',
    min_block: int = 3
) -> pd.DataFrame:
    """
    Prepare trial data for model fitting.

    Parameters
    ----------
    data : pd.DataFrame
        Raw trial data
    participant_col : str
        Participant ID column
    block_col, trial_col, stimulus_col, action_col, correct_col : str
        Column names
    min_block : int
        Minimum block to include (exclude practice)

    Returns
    -------
    pd.DataFrame
        Cleaned data ready for fitting
    """
    # Filter to main task blocks
    df = data[data[block_col] >= min_block].copy()

    # Ensure 0-indexed stimulus and action
    if stimulus_col in df.columns:
        df[stimulus_col] = df[stimulus_col] - 1  # Convert 1-indexed to 0-indexed

    # Ensure correct column is 0/1
    df['reward'] = df[correct_col].astype(float)

    # Sort by participant and trial
    df = df.sort_values([participant_col, block_col, trial_col])

    return df


def compute_model_comparison(
    trace_dict: Dict[str, any],
    model_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute model comparison statistics (WAIC, LOO).

    Parameters
    ----------
    trace_dict : dict
        Dictionary mapping model names to InferenceData objects
    model_names : list, optional
        Model names to include

    Returns
    -------
    pd.DataFrame
        Comparison table with WAIC, LOO, etc.
    """
    if not PYMC_AVAILABLE:
        raise ImportError("PyMC required")

    import arviz as az

    if model_names is None:
        model_names = list(trace_dict.keys())

    comparison = az.compare({name: trace_dict[name] for name in model_names})

    return comparison


# ============================================================================
# TESTING
# ============================================================================

def test_pymc_models():
    """Test PyMC model building with synthetic data."""
    print("Testing PyMC Model Building")
    print("=" * 80)

    if not PYMC_AVAILABLE:
        print("PyMC not installed. Skipping test.")
        return

    # Create synthetic data
    np.random.seed(42)
    n_participants = 3
    n_trials_per_participant = 50

    data_list = []
    for p in range(n_participants):
        for t in range(n_trials_per_participant):
            data_list.append({
                'sona_id': f'P{p:03d}',
                'block': 3,
                'trial': t,
                'stimulus': np.random.randint(0, 3),  # 0-indexed
                'key_press': np.random.randint(0, 3),  # 0-indexed
                'correct': np.random.randint(0, 2),
            })

    data = pd.DataFrame(data_list)

    print(f"\nSynthetic data: {len(data)} trials, {n_participants} participants")

    # Build Q-learning model
    print("\nBuilding Q-learning model...")
    try:
        with build_qlearning_model(data) as qlearning_model:
            print(f"  Variables: {list(qlearning_model.named_vars.keys())}")
            print("  Model built successfully!")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "=" * 80)
    print("Test complete!")


if __name__ == "__main__":
    test_pymc_models()
