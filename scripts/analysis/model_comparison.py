"""
Model Comparison: BIC, AIC, WAIC, and LOO

Functions for computing and comparing information criteria across
Q-learning and WM-RL hybrid models fitted to behavioral data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import sys

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import arviz as az
    ARVIZ_AVAILABLE = True
except ImportError:
    ARVIZ_AVAILABLE = False
    print("Warning: ArviZ not available. WAIC/LOO will not work.")

from scripts.simulations.unified_simulator import (
    simulate_qlearning_for_likelihood,
    simulate_wmrl_for_likelihood
)
from scripts.analysis.plotting_utils import setup_plot_style, save_figure


# ============================================================================
# PARAMETER COUNTING
# ============================================================================

def count_parameters(
    model_type: str,
    n_participants: int,
    hierarchical: bool = True
) -> int:
    """
    Count the number of free parameters in a model.

    Parameters
    ----------
    model_type : str
        'qlearning' or 'wmrl'
    n_participants : int
        Number of participants
    hierarchical : bool
        Whether model has hierarchical structure (group-level parameters)

    Returns
    -------
    int
        Number of free parameters
    """
    if model_type == 'qlearning':
        # Per participant: alpha_pos, alpha_neg, beta
        per_participant = 3

        if hierarchical:
            # Group level: mu and sigma for each parameter
            group_level = 2 * per_participant  # 6 parameters
            return n_participants * per_participant + group_level
        else:
            return n_participants * per_participant

    elif model_type == 'wmrl':
        # Per participant: alpha_pos, alpha_neg, beta, beta_wm, capacity, phi, rho
        per_participant = 7

        if hierarchical:
            # Group level: mu and sigma for each parameter
            group_level = 2 * per_participant  # 14 parameters
            return n_participants * per_participant + group_level
        else:
            return n_participants * per_participant

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# MAP/MLE ESTIMATION
# ============================================================================

def get_map_estimates(trace: Any, model_type: str) -> Dict[str, np.ndarray]:
    """
    Extract MAP (Maximum A Posteriori) parameter estimates from posterior.

    Parameters
    ----------
    trace : InferenceData
        ArviZ InferenceData object from PyMC sampling
    model_type : str
        'qlearning' or 'wmrl'

    Returns
    -------
    dict
        Parameter arrays at MAP estimates
    """
    # Get posterior means as MAP approximation
    # (For true MAP, would need to find max posterior density)

    posterior = trace.posterior

    if model_type == 'qlearning':
        params = {
            'alpha_pos': posterior['alpha_pos'].mean(dim=['chain', 'draw']).values,
            'alpha_neg': posterior['alpha_neg'].mean(dim=['chain', 'draw']).values,
            'beta': posterior['beta'].mean(dim=['chain', 'draw']).values
        }
    elif model_type == 'wmrl':
        params = {
            'alpha_pos': posterior['alpha_pos'].mean(dim=['chain', 'draw']).values,
            'alpha_neg': posterior['alpha_neg'].mean(dim=['chain', 'draw']).values,
            'beta': posterior['beta'].mean(dim=['chain', 'draw']).values,
            'beta_wm': posterior['beta_wm'].mean(dim=['chain', 'draw']).values,
            'capacity': posterior['capacity'].mean(dim=['chain', 'draw']).values,
            'phi': posterior['phi'].mean(dim=['chain', 'draw']).values,
            'rho': posterior['rho'].mean(dim=['chain', 'draw']).values
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return params


# ============================================================================
# LOG-LIKELIHOOD COMPUTATION
# ============================================================================

def compute_log_likelihood_at_map(
    data: pd.DataFrame,
    params: Dict[str, np.ndarray],
    model_type: str,
    participant_col: str = 'sona_id'
) -> float:
    """
    Compute log-likelihood of data given MAP parameter estimates.

    Parameters
    ----------
    data : pd.DataFrame
        Trial-level behavioral data
    params : dict
        MAP parameter estimates (arrays indexed by participant)
    model_type : str
        'qlearning' or 'wmrl'
    participant_col : str
        Column name for participant IDs

    Returns
    -------
    float
        Total log-likelihood across all participants and trials
    """
    # Get unique participants
    participants = data[participant_col].unique()
    participant_map = {p: i for i, p in enumerate(participants)}

    total_log_likelihood = 0.0

    for p_id, p_idx in participant_map.items():
        # Get this participant's data
        p_data = data[data[participant_col] == p_id]

        stimuli = p_data['stimulus'].values.astype(int)
        actions = p_data['key_press'].values.astype(int)
        rewards = p_data['correct'].values.astype(float)

        # Get participant-specific parameters
        if model_type == 'qlearning':
            # Compute action probabilities using Q-learning model
            action_probs = simulate_qlearning_for_likelihood(
                stimuli=stimuli,
                rewards=rewards,
                alpha_pos=params['alpha_pos'][p_idx],
                alpha_neg=params['alpha_neg'][p_idx],
                beta=params['beta'][p_idx],
                gamma=0.0,
                q_init=0.5,
                num_stimuli=6,
                num_actions=3
            )

        elif model_type == 'wmrl':
            # Need set_sizes for WM-RL
            set_sizes = p_data['set_size'].values.astype(int)

            action_probs = simulate_wmrl_for_likelihood(
                stimuli=stimuli,
                rewards=rewards,
                set_sizes=set_sizes,
                alpha_pos=params['alpha_pos'][p_idx],
                alpha_neg=params['alpha_neg'][p_idx],
                beta=params['beta'][p_idx],
                beta_wm=params['beta_wm'][p_idx],
                capacity=int(params['capacity'][p_idx]),
                phi=params['phi'][p_idx],
                rho=params['rho'][p_idx],
                gamma=0.0,
                q_init=0.5,
                wm_init=0.0,
                num_stimuli=6,
                num_actions=3
            )

        # Compute log-likelihood for this participant
        # log P(action_t | model) for each trial
        trial_log_probs = np.log(action_probs[np.arange(len(actions)), actions] + 1e-10)
        total_log_likelihood += np.sum(trial_log_probs)

    return total_log_likelihood


# ============================================================================
# INFORMATION CRITERIA
# ============================================================================

def compute_aic(log_likelihood: float, n_params: int) -> float:
    """
    Compute Akaike Information Criterion (AIC).

    AIC = 2k - 2·log(L)

    Lower is better. Penalizes model complexity (k parameters).

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood at MAP/MLE
    n_params : int
        Number of free parameters

    Returns
    -------
    float
        AIC value
    """
    return 2 * n_params - 2 * log_likelihood


def compute_bic(
    log_likelihood: float,
    n_params: int,
    n_observations: int
) -> float:
    """
    Compute Bayesian Information Criterion (BIC).

    BIC = k·log(n) - 2·log(L)

    Lower is better. Penalizes model complexity more strongly than AIC
    for larger datasets.

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood at MAP/MLE
    n_params : int
        Number of free parameters
    n_observations : int
        Number of data points (trials)

    Returns
    -------
    float
        BIC value
    """
    return n_params * np.log(n_observations) - 2 * log_likelihood


def compute_waic(trace: Any) -> Dict[str, float]:
    """
    Compute Widely Applicable Information Criterion (WAIC).

    Uses ArviZ implementation. WAIC is fully Bayesian and accounts for
    posterior uncertainty.

    Parameters
    ----------
    trace : InferenceData
        PyMC trace with log-likelihood

    Returns
    -------
    dict
        WAIC results with 'waic', 'waic_se', 'p_waic'
    """
    if not ARVIZ_AVAILABLE:
        raise ImportError("ArviZ required for WAIC. Install with: pip install arviz")

    waic_result = az.waic(trace)

    return {
        'waic': waic_result.elpd_waic * -2,  # Convert to deviance scale
        'waic_se': waic_result.se * 2,
        'p_waic': waic_result.p_waic  # Effective number of parameters
    }


def compute_loo(trace: Any) -> Dict[str, float]:
    """
    Compute Leave-One-Out Cross-Validation (LOO).

    Uses ArviZ implementation with Pareto-Smoothed Importance Sampling (PSIS).
    LOO is preferred over WAIC for model comparison.

    Parameters
    ----------
    trace : InferenceData
        PyMC trace with log-likelihood

    Returns
    -------
    dict
        LOO results with 'loo', 'loo_se', 'p_loo'
    """
    if not ARVIZ_AVAILABLE:
        raise ImportError("ArviZ required for LOO. Install with: pip install arviz")

    loo_result = az.loo(trace)

    return {
        'loo': loo_result.elpd_loo * -2,  # Convert to deviance scale
        'loo_se': loo_result.se * 2,
        'p_loo': loo_result.p_loo  # Effective number of parameters
    }


# ============================================================================
# UNIFIED MODEL COMPARISON
# ============================================================================

def compute_all_criteria(
    trace: Any,
    data: pd.DataFrame,
    model_name: str,
    model_type: str,
    participant_col: str = 'sona_id'
) -> pd.Series:
    """
    Compute all information criteria (BIC, AIC, WAIC, LOO) for a model.

    Parameters
    ----------
    trace : InferenceData
        Fitted model posterior
    data : pd.DataFrame
        Behavioral data
    model_name : str
        Name for display (e.g., "Q-Learning")
    model_type : str
        Model type code ('qlearning' or 'wmrl')
    participant_col : str
        Participant ID column

    Returns
    -------
    pd.Series
        All criteria values
    """
    # Count parameters and observations
    n_participants = data[participant_col].nunique()
    n_observations = len(data)
    n_params = count_parameters(model_type, n_participants, hierarchical=True)

    # Get MAP estimates
    map_params = get_map_estimates(trace, model_type)

    # Compute log-likelihood at MAP
    log_likelihood = compute_log_likelihood_at_map(data, map_params, model_type, participant_col)

    # Compute AIC and BIC
    aic = compute_aic(log_likelihood, n_params)
    bic = compute_bic(log_likelihood, n_params, n_observations)

    # Compute WAIC and LOO
    waic_results = compute_waic(trace)
    loo_results = compute_loo(trace)

    # Compile results
    results = pd.Series({
        'model': model_name,
        'n_params': n_params,
        'n_observations': n_observations,
        'log_likelihood': log_likelihood,
        'AIC': aic,
        'BIC': bic,
        'WAIC': waic_results['waic'],
        'WAIC_SE': waic_results['waic_se'],
        'p_WAIC': waic_results['p_waic'],
        'LOO': loo_results['loo'],
        'LOO_SE': loo_results['loo_se'],
        'p_LOO': loo_results['p_loo']
    })

    return results


def compare_models(
    traces_dict: Dict[str, Any],
    data: pd.DataFrame,
    participant_col: str = 'sona_id'
) -> pd.DataFrame:
    """
    Compare multiple models using all information criteria.

    Parameters
    ----------
    traces_dict : dict
        Dictionary mapping model names to fitted traces
        e.g., {'qlearning': trace_q, 'wmrl': trace_wmrl}
    data : pd.DataFrame
        Behavioral data
    participant_col : str
        Participant ID column

    Returns
    -------
    pd.DataFrame
        Comparison table with all criteria
    """
    # Map display names to model types
    model_type_map = {
        'qlearning': 'qlearning',
        'Q-Learning': 'qlearning',
        'Q-learning': 'qlearning',
        'wmrl': 'wmrl',
        'WM-RL': 'wmrl',
        'WM-RL Hybrid': 'wmrl'
    }

    results_list = []

    for model_name, trace in traces_dict.items():
        # Determine model type
        model_type = model_type_map.get(model_name.lower(), model_name.lower())

        print(f"Computing criteria for {model_name}...")

        results = compute_all_criteria(
            trace=trace,
            data=data,
            model_name=model_name,
            model_type=model_type,
            participant_col=participant_col
        )

        results_list.append(results)

    # Combine into DataFrame
    comparison_df = pd.DataFrame(results_list)

    # Add rankings (lower is better for all criteria)
    for criterion in ['AIC', 'BIC', 'WAIC', 'LOO']:
        comparison_df[f'{criterion}_rank'] = comparison_df[criterion].rank()

    # Add model weights (Akaike weights for AIC)
    aic_min = comparison_df['AIC'].min()
    comparison_df['AIC_delta'] = comparison_df['AIC'] - aic_min
    comparison_df['AIC_weight'] = np.exp(-0.5 * comparison_df['AIC_delta'])
    comparison_df['AIC_weight'] /= comparison_df['AIC_weight'].sum()

    # Same for BIC
    bic_min = comparison_df['BIC'].min()
    comparison_df['BIC_delta'] = comparison_df['BIC'] - bic_min
    comparison_df['BIC_weight'] = np.exp(-0.5 * comparison_df['BIC_delta'])
    comparison_df['BIC_weight'] /= comparison_df['BIC_weight'].sum()

    return comparison_df


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_dir: Optional[Path] = None,
    show: bool = False
) -> plt.Figure:
    """
    Create bar plot comparing all information criteria across models.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Comparison results from compare_models()
    save_dir : Path, optional
        Save directory
    show : bool
        Display figure

    Returns
    -------
    plt.Figure
    """
    setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    criteria = ['AIC', 'BIC', 'WAIC', 'LOO']
    models = comparison_df['model'].values

    for idx, (ax, criterion) in enumerate(zip(axes.flat, criteria)):
        values = comparison_df[criterion].values

        # Normalize to delta (difference from best)
        min_val = values.min()
        deltas = values - min_val

        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(models)]

        bars = ax.bar(models, deltas, color=colors[:len(models)], alpha=0.8,
                     edgecolor='black', linewidth=1.5)

        ax.set_ylabel(f'Δ{criterion} (vs. best)', fontweight='bold')
        ax.set_title(f'{criterion} Comparison', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, delta, raw_val in zip(bars, deltas, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                   f'Δ={delta:.1f}\n({raw_val:.1f})',
                   ha='center', va='bottom', fontsize=9)

        # Add interpretation text
        if criterion in ['AIC', 'BIC']:
            ax.text(0.98, 0.98, 'Lower is better\nΔ<2: weak evidence\nΔ>10: strong evidence',
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                   fontsize=8)

    fig.suptitle('Model Comparison: Information Criteria', fontsize=16, fontweight='bold', y=0.995)
    fig.tight_layout()

    if save_dir:
        save_figure(fig, 'information_criteria_comparison', subdir='model_comparison')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_model_weights(
    comparison_df: pd.DataFrame,
    save_dir: Optional[Path] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot Akaike weights showing relative model support.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Comparison results
    save_dir : Path, optional
        Save directory
    show : bool
        Display figure

    Returns
    -------
    plt.Figure
    """
    setup_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = comparison_df['model'].values

    # AIC weights
    ax = axes[0]
    aic_weights = comparison_df['AIC_weight'].values
    colors = ['#3498db', '#e74c3c'][:len(models)]

    bars = ax.bar(models, aic_weights, color=colors, alpha=0.8,
                 edgecolor='black', linewidth=1.5)
    ax.set_ylabel('AIC Weight', fontweight='bold')
    ax.set_title('Model Support (AIC Weights)', fontweight='bold', fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, weight in zip(bars, aic_weights):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
               f'{weight:.3f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    # BIC weights
    ax = axes[1]
    bic_weights = comparison_df['BIC_weight'].values

    bars = ax.bar(models, bic_weights, color=colors, alpha=0.8,
                 edgecolor='black', linewidth=1.5)
    ax.set_ylabel('BIC Weight', fontweight='bold')
    ax.set_title('Model Support (BIC Weights)', fontweight='bold', fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, weight in zip(bars, bic_weights):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
               f'{weight:.3f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    fig.suptitle('Model Weights: Probability Best Model', fontsize=16, fontweight='bold')
    fig.tight_layout()

    if save_dir:
        save_figure(fig, 'model_weights', subdir='model_comparison')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
