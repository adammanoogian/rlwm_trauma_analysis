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

Usage:
    import pymc as pm
    from fitting.pymc_models import build_qlearning_model

    with build_qlearning_model(data, participant_ids) as model:
        trace = pm.sample(2000, tune=1000, chains=4)
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
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import PyMCParams, ModelParams, TaskParams


# ============================================================================
# Q-LEARNING MODEL
# ============================================================================

def simulate_qlearning_choices(
    stimuli: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    alpha: float,
    beta: float,
    q_init: float = 0.5,
    num_stimuli: int = TaskParams.MAX_STIMULI,
    num_actions: int = TaskParams.NUM_ACTIONS
) -> np.ndarray:
    """
    Simulate Q-learning model and return trial-by-trial action probabilities.

    Parameters
    ----------
    stimuli : np.ndarray
        Stimulus IDs for each trial
    actions : np.ndarray
        Actions taken (for updating Q-values)
    rewards : np.ndarray
        Rewards received (for updating Q-values)
    alpha : float
        Learning rate
    beta : float
        Inverse temperature
    q_init : float
        Initial Q-values
    num_stimuli : int
        Number of stimuli
    num_actions : int
        Number of actions

    Returns
    -------
    np.ndarray
        Action probabilities of shape (num_trials, num_actions)
    """
    num_trials = len(stimuli)
    Q = np.full((num_stimuli, num_actions), q_init, dtype=np.float64)
    action_probs = np.zeros((num_trials, num_actions))

    for t in range(num_trials):
        s = int(stimuli[t])
        a = int(actions[t])
        r = rewards[t]

        # Compute action probabilities (softmax)
        q_vals = Q[s, :]
        q_scaled = beta * (q_vals - np.max(q_vals))  # numerical stability
        exp_q = np.exp(q_scaled)
        probs = exp_q / np.sum(exp_q)
        action_probs[t, :] = probs

        # Update Q-value
        Q[s, a] += alpha * (r - Q[s, a])

    return action_probs


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

        # Learning rate (alpha): bounded [0, 1]
        mu_alpha = pm.Beta('mu_alpha', alpha=2, beta=2)  # Mean ~0.5
        sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=0.3)

        # Inverse temperature (beta): positive
        mu_beta = pm.Gamma('mu_beta', alpha=2, beta=1)  # Mean ~2
        sigma_beta = pm.HalfNormal('sigma_beta', sigma=2)

        # ====================================================================
        # PRIORS: Individual-level
        # ====================================================================

        # Learning rate per participant (transformed to [0,1])
        alpha_raw = pm.Normal('alpha_raw', mu=0, sigma=1, shape=n_participants)
        alpha = pm.Deterministic(
            'alpha',
            pm.math.invlogit(pm.math.log(mu_alpha / (1 - mu_alpha)) + sigma_alpha * alpha_raw)
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

        def logp_func(actions, participant_idx, stimuli, rewards, alpha, beta):
            """Compute log-likelihood for all trials."""
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
                action_probs = simulate_qlearning_choices(
                    p_stimuli, p_actions, p_rewards,
                    alpha[p], beta[p],
                    q_init=ModelParams.Q_INIT_VALUE
                )

                # Log-likelihood: sum of log P(action_t | model)
                trial_logp = np.log(action_probs[np.arange(len(p_actions)), p_actions] + 1e-10)
                logp_total += np.sum(trial_logp)

            return logp_total

        # Create custom likelihood
        pm.Potential(
            'likelihood',
            logp_func(actions, participant_idx, stimuli, rewards, alpha, beta)
        )

    return model


# ============================================================================
# WM-RL HYBRID MODEL (Placeholder)
# ============================================================================

def simulate_wmrl_choices(
    stimuli: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    alpha: float,
    beta: float,
    capacity: int,
    lambda_decay: float,
    w_wm: float,
    q_init: float = 0.5,
    num_stimuli: int = TaskParams.MAX_STIMULI,
    num_actions: int = TaskParams.NUM_ACTIONS
) -> np.ndarray:
    """
    Simulate WM-RL hybrid model and return action probabilities.

    NOTE: This is a simplified version. Full implementation would use
    the WMRLHybridAgent class, but that's complex for PyMC integration.

    Parameters
    ----------
    stimuli, actions, rewards : np.ndarray
        Trial data
    alpha, beta, capacity, lambda_decay, w_wm : float
        Model parameters
    q_init : float
        Initial Q-values
    num_stimuli, num_actions : int
        Task dimensions

    Returns
    -------
    np.ndarray
        Action probabilities
    """
    from collections import deque

    num_trials = len(stimuli)
    Q = np.full((num_stimuli, num_actions), q_init, dtype=np.float64)
    action_probs = np.zeros((num_trials, num_actions))

    # Working memory buffer
    wm_buffer = deque(maxlen=int(capacity))

    for t in range(num_trials):
        s = int(stimuli[t])
        a = int(actions[t])
        r = rewards[t]

        # RL probabilities
        q_vals = Q[s, :]
        q_scaled = beta * (q_vals - np.max(q_vals))
        exp_q = np.exp(q_scaled)
        rl_probs = exp_q / np.sum(exp_q)

        # WM probabilities
        wm_probs = np.ones(num_actions) / num_actions  # default uniform

        # Check if stimulus in WM
        for mem in wm_buffer:
            if mem['stimulus'] == s and mem['reward'] > 0:
                strength = np.exp(-lambda_decay * mem['age'])
                wm_probs = np.ones(num_actions) * (1 - strength) / num_actions
                wm_probs[mem['action']] += strength
                wm_probs /= np.sum(wm_probs)
                break

        # Hybrid
        probs = w_wm * wm_probs + (1 - w_wm) * rl_probs
        probs /= np.sum(probs)
        action_probs[t, :] = probs

        # Update
        Q[s, a] += alpha * (r - Q[s, a])
        wm_buffer.append({'stimulus': s, 'action': a, 'reward': r, 'age': 0})

        # Age memories
        for mem in wm_buffer:
            mem['age'] += 1

    return action_probs


def build_wmrl_model(
    data: pd.DataFrame,
    participant_col: str = 'sona_id',
    stimulus_col: str = 'stimulus',
    action_col: str = 'key_press',
    reward_col: str = 'correct',
) -> pm.Model:
    """
    Build hierarchical Bayesian WM-RL hybrid model.

    Parameters
    ----------
    data : pd.DataFrame
        Trial-level data
    participant_col, stimulus_col, action_col, reward_col : str
        Column names

    Returns
    -------
    pm.Model
        PyMC model
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

    with pm.Model() as model:
        # Group-level priors
        mu_alpha = pm.Beta('mu_alpha', alpha=2, beta=2)
        sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=0.3)

        mu_beta = pm.Gamma('mu_beta', alpha=2, beta=1)
        sigma_beta = pm.HalfNormal('sigma_beta', sigma=2)

        mu_capacity = pm.TruncatedNormal('mu_capacity', mu=4, sigma=1.5, lower=1, upper=7)
        sigma_capacity = pm.HalfNormal('sigma_capacity', sigma=1)

        mu_w_wm = pm.Beta('mu_w_wm', alpha=2, beta=2)
        sigma_w_wm = pm.HalfNormal('sigma_w_wm', sigma=0.3)

        # Individual-level priors
        alpha_raw = pm.Normal('alpha_raw', mu=0, sigma=1, shape=n_participants)
        alpha = pm.Deterministic(
            'alpha',
            pm.math.invlogit(pm.math.log(mu_alpha / (1 - mu_alpha)) + sigma_alpha * alpha_raw)
        )

        beta_raw = pm.Normal('beta_raw', mu=0, sigma=1, shape=n_participants)
        beta = pm.Deterministic(
            'beta',
            pm.math.exp(pm.math.log(mu_beta) + sigma_beta * beta_raw)
        )

        capacity_raw = pm.Normal('capacity_raw', mu=0, sigma=1, shape=n_participants)
        capacity = pm.Deterministic(
            'capacity',
            pm.math.clip(mu_capacity + sigma_capacity * capacity_raw, 1, 7)
        )

        w_wm_raw = pm.Normal('w_wm_raw', mu=0, sigma=1, shape=n_participants)
        w_wm = pm.Deterministic(
            'w_wm',
            pm.math.invlogit(pm.math.log(mu_w_wm / (1 - mu_w_wm)) + sigma_w_wm * w_wm_raw)
        )

        # Lambda decay (fixed for simplicity, or could be estimated)
        lambda_decay = pm.HalfNormal('lambda_decay', sigma=0.3, shape=n_participants)

        # Likelihood
        def logp_func(actions, participant_idx, stimuli, rewards, alpha, beta, capacity, lambda_decay, w_wm):
            logp_total = 0.0

            for p in range(n_participants):
                mask = participant_idx == p
                if not np.any(mask):
                    continue

                p_stimuli = stimuli[mask]
                p_actions = actions[mask]
                p_rewards = rewards[mask]

                action_probs = simulate_wmrl_choices(
                    p_stimuli, p_actions, p_rewards,
                    alpha[p], beta[p], capacity[p], lambda_decay[p], w_wm[p]
                )

                trial_logp = np.log(action_probs[np.arange(len(p_actions)), p_actions] + 1e-10)
                logp_total += np.sum(trial_logp)

            return logp_total

        pm.Potential(
            'likelihood',
            logp_func(actions, participant_idx, stimuli, rewards, alpha, beta, capacity, lambda_decay, w_wm)
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
