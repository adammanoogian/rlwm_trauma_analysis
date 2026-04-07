"""
Central configuration for RLWM Trauma Analysis project.

This module contains all project paths, task parameters, model hyperparameters,
and analysis settings to ensure consistency across all scripts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'output'
FIGURES_DIR = PROJECT_ROOT / 'figures'
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
DOCS_DIR = PROJECT_ROOT / 'docs'

# ============================================================================
# PARTICIPANT EXCLUSIONS
# ============================================================================

# Hand-curated exclusions (duplicates, known bad data not caught by thresholds).
# These are always excluded regardless of trial count.
MANUAL_EXCLUSIONS: list[int] = []

# Threshold for automatic exclusion (participants below this are excluded).
# Must be defined here (not in DataParams) so get_excluded_participants() can
# reference it before the class is defined.
MIN_TRIALS_THRESHOLD = 400

# Version management
VERSION = 'v1'
OUTPUT_VERSION_DIR = OUTPUT_DIR / VERSION if VERSION else OUTPUT_DIR
FIGURES_VERSION_DIR = FIGURES_DIR / VERSION if VERSION else FIGURES_DIR

# Create directories if they don't exist
for directory in [OUTPUT_VERSION_DIR, FIGURES_VERSION_DIR, DOCS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# TASK PARAMETERS (from jsPsych implementation)
# ============================================================================

class TaskParams:
    """Parameters for the RLWM task environment."""

    # Response keys (J, K, L mapped to 0, 1, 2)
    NUM_ACTIONS = 3
    ACTIONS = [0, 1, 2]  # Discrete action space

    # Stimulus set configuration
    MAX_STIMULI = 6  # Maximum number of unique stimuli
    SET_SIZES = [2, 3, 5, 6]  # Possible set sizes per block
    EXCLUDE_SET_SIZES = [4]  # Set size 4 excluded from main task

    # Working memory load classification
    LOW_LOAD_THRESHOLD = 3  # Set sizes <= 3 are "low load"
    HIGH_LOAD_THRESHOLD = 4  # Set sizes >= 4 are "high load"

    # Reversal parameters
    RARE_REVERSALS = True  # Reversals are rare and late
    REVERSAL_MIN = 12  # Minimum consecutive correct before reversal
    REVERSAL_MAX = 18  # Maximum consecutive correct before reversal
    MAX_REVERSALS_PER_STIM = 1  # Max reversals per stimulus per block

    # Practice block parameters
    PRACTICE_DYNAMIC_REVERSAL_CRITERION = 5  # Consecutive correct for practice reversal
    PRACTICE_DYNAMIC_NUM_REVERSALS = 2  # Must detect 2 reversals in practice

    # Reward structure
    REWARD_CORRECT = 1.0  # Points for correct response
    REWARD_INCORRECT = 0.0  # Points for incorrect response

    # Timing (in milliseconds, for reference)
    FIXATION_DURATION = 500  # ms
    TRIAL_DURATION = 2000  # ms (maximum response time)
    FEEDBACK_DURATION = 500  # ms

    # Block structure (verified from raw jsPsych data)
    # Practice blocks
    PRACTICE_STATIC_BLOCK = 1     # Block 1: static practice (no reversals)
    PRACTICE_DYNAMIC_BLOCK = 2    # Block 2: dynamic practice (with reversals)
    NUM_PRACTICE_BLOCKS = 2       # Blocks 1-2: practice

    # Main task blocks
    MAIN_TASK_START_BLOCK = 3     # First main task block
    MAIN_TASK_END_BLOCK = 23      # Last main task block (maximum)
    NUM_MAIN_BLOCKS = 21          # Blocks 3-23: main task

    # Total
    TOTAL_BLOCKS = NUM_PRACTICE_BLOCKS + NUM_MAIN_BLOCKS  # 23 total

    # Trials per block (from actual experimental data)
    TRIALS_PER_BLOCK_MIN = 30  # Minimum trials observed
    TRIALS_PER_BLOCK_MAX = 90  # Maximum trials observed
    TRIALS_PER_BLOCK_MEAN = 58  # Mean trials per block
    TRIALS_PER_BLOCK_MEDIAN = 45  # Median trials per block
    TRIALS_PER_BLOCK_DEFAULT = 100  # Default for simulations (max envelope)

    # Phase types
    PHASE_PRACTICE_STATIC = 'practice_static'  # Block 1
    PHASE_PRACTICE_DYNAMIC = 'practice_dynamic'  # Block 2
    PHASE_MAIN_TASK = 'main_task'  # Blocks 3-23

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

class ModelParams:
    """
    Default hyperparameters for RL models.

    Following Senta et al. (2025):
    - Beta is FIXED at 50 during learning for parameter identifiability
    - Epsilon noise captures random responding (motor noise, lapses)
    """

    # Q-learning parameters (asymmetric learning rates)
    ALPHA_POS_DEFAULT = 0.3  # Learning rate for positive PE (correct trials)
    ALPHA_NEG_DEFAULT = 0.1  # Learning rate for negative PE (incorrect trials)
    ALPHA_MIN = 0.0
    ALPHA_MAX = 1.0

    # Inverse temperature - FIXED at 50 for identifiability (Senta et al., 2025)
    BETA_FIXED = 50.0  # Fixed inverse temperature during learning
    BETA_DEFAULT = 50.0  # Alias for backwards compatibility

    GAMMA_DEFAULT = 0.0  # Discount factor (fixed at 0 for this task)
    GAMMA_MIN = 0.0
    GAMMA_MAX = 1.0

    # Epsilon noise parameter (Senta et al., 2025)
    # Captures random responding: p_noisy = ε/nA + (1-ε)*p
    EPSILON_DEFAULT = 0.05  # Default epsilon noise (5% random)
    EPSILON_MIN = 0.0
    EPSILON_MAX = 1.0

    # Working Memory + RL Hybrid parameters
    WM_CAPACITY_DEFAULT = 4  # WM capacity for adaptive weighting
    WM_CAPACITY_MIN = 1
    WM_CAPACITY_MAX = 7

    PHI_DEFAULT = 0.1  # WM decay rate toward baseline (0-1)
    PHI_MIN = 0.0
    PHI_MAX = 1.0

    RHO_DEFAULT = 0.7  # Base WM reliance parameter (0-1)
    RHO_MIN = 0.0
    RHO_MAX = 1.0

    # Initialization
    Q_INIT_VALUE = 0.5  # Initial Q-values (optimistic initialization)
    WM_INIT_VALUE = 1.0 / 3.0  # Initial WM values (1/nA for uniform baseline)
    NUM_ACTIONS = 3  # Number of possible actions

# ============================================================================
# MODEL REGISTRY (single source of truth for all pipeline scripts)
# ============================================================================
# Use this for orchestration (model lists, file paths, display names).
# Do NOT replace the per-model PARAMS/BOUNDS in mle_utils.py — those are
# used inside the tight JAX optimization inner loop.

MODEL_REGISTRY: dict[str, dict] = {
    'qlearning': {
        'display_name': 'M1: Q-Learning',
        'short_name': 'M1',
        'params': ['alpha_pos', 'alpha_neg', 'epsilon'],
        'n_params': 3,
        'is_choice_only': True,
        'has_wm': False,
        'csv_filename': 'qlearning_individual_fits.csv',
    },
    'wmrl': {
        'display_name': 'M2: WM-RL',
        'short_name': 'M2',
        'params': ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'epsilon'],
        'n_params': 6,
        'is_choice_only': True,
        'has_wm': True,
        'csv_filename': 'wmrl_individual_fits.csv',
    },
    'wmrl_m3': {
        'display_name': 'M3: WM-RL+kappa',
        'short_name': 'M3',
        'params': ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa', 'epsilon'],
        'n_params': 7,
        'is_choice_only': True,
        'has_wm': True,
        'csv_filename': 'wmrl_m3_individual_fits.csv',
    },
    'wmrl_m5': {
        'display_name': 'M5: WM-RL+phi_rl',
        'short_name': 'M5',
        'params': ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa', 'phi_rl', 'epsilon'],
        'n_params': 8,
        'is_choice_only': True,
        'has_wm': True,
        'csv_filename': 'wmrl_m5_individual_fits.csv',
    },
    'wmrl_m6a': {
        'display_name': 'M6a: WM-RL+kappa_s',
        'short_name': 'M6a',
        'params': ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa_s', 'epsilon'],
        'n_params': 7,
        'is_choice_only': True,
        'has_wm': True,
        'csv_filename': 'wmrl_m6a_individual_fits.csv',
    },
    'wmrl_m6b': {
        'display_name': 'M6b: WM-RL+dual',
        'short_name': 'M6b',
        'params': [
            'alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity',
            'kappa_total', 'kappa_share', 'epsilon',
        ],
        'n_params': 8,
        'is_choice_only': True,
        'has_wm': True,
        'csv_filename': 'wmrl_m6b_individual_fits.csv',
    },
    'wmrl_m4': {
        'display_name': 'M4: RLWM-LBA',
        'short_name': 'M4',
        'params': [
            'alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa',
            'v_scale', 'A', 'delta', 't0',
        ],
        'n_params': 10,
        'is_choice_only': False,  # Joint choice + RT; AIC not comparable to choice-only
        'has_wm': True,
        'csv_filename': 'wmrl_m4_individual_fits.csv',
    },
}

ALL_MODELS: list[str] = list(MODEL_REGISTRY.keys())
CHOICE_ONLY_MODELS: list[str] = [k for k, v in MODEL_REGISTRY.items() if v['is_choice_only']]

# ============================================================================
# PYMC SAMPLING PARAMETERS
# ============================================================================

class PyMCParams:
    """Parameters for Bayesian model fitting with PyMC."""

    # MCMC sampling
    NUM_CHAINS = 4  # Number of parallel MCMC chains
    NUM_SAMPLES = 2000  # Samples per chain (after warmup)
    NUM_TUNE = 1000  # Warmup/tuning samples
    TARGET_ACCEPT = 0.95  # Target acceptance rate for NUTS

    # Hierarchical priors (group-level)
    # Learning rate (alpha) ~ Normal(mu_alpha, sigma_alpha)
    ALPHA_MU_PRIOR = 0.3  # Group mean learning rate
    ALPHA_SIGMA_PRIOR = 0.2  # Group std learning rate

    # Inverse temperature (beta) ~ Gamma(alpha, beta)
    BETA_ALPHA_PRIOR = 2.0  # Shape parameter
    BETA_BETA_PRIOR = 1.0  # Rate parameter

    # Working memory capacity ~ TruncatedNormal(mu, sigma, low=1, high=7)
    WM_CAPACITY_MU_PRIOR = 4.0
    WM_CAPACITY_SIGMA_PRIOR = 1.5

    # Model comparison
    USE_WAIC = True  # Compute WAIC (Watanabe-Akaike IC)
    USE_LOO = True  # Compute LOO (Leave-One-Out CV)

    # Posterior predictive checks
    NUM_POSTERIOR_SAMPLES = 100  # Samples for posterior predictive

# ============================================================================
# DATA PROCESSING
# ============================================================================

class DataParams:
    """Parameters for data processing and analysis."""

    # File paths
    RAW_DATA_DIR = DATA_DIR
    PARSED_DEMOGRAPHICS = OUTPUT_DIR / 'parsed_demographics.csv'
    PARSED_SURVEY1 = OUTPUT_DIR / 'parsed_survey1.csv'
    PARSED_SURVEY2 = OUTPUT_DIR / 'parsed_survey2.csv'
    PARSED_TASK_TRIALS = OUTPUT_DIR / 'parsed_task_trials.csv'
    COLLATED_DATA = OUTPUT_DIR / 'collated_participant_data.csv'

    # Task trial data files
    TASK_TRIALS_LONG = OUTPUT_DIR / 'task_trials_long.csv'       # Main task only (default for fitting)
    TASK_TRIALS_ALL = OUTPUT_DIR / 'task_trials_long_all.csv'    # All blocks including practice
    TASK_TRIALS_LEGACY = OUTPUT_DIR / 'task_trials_long_all_participants.csv'  # Legacy filename

    SUMMARY_METRICS = OUTPUT_DIR / 'summary_participant_metrics.csv'

    # Simulated data paths
    SIMULATED_DATA = OUTPUT_VERSION_DIR / 'simulated_data.csv'
    FITTED_POSTERIORS = OUTPUT_VERSION_DIR / 'fitted_posteriors.nc'  # NetCDF format
    MODEL_COMPARISON = OUTPUT_VERSION_DIR / 'model_comparison.csv'

    # Exclusion criteria (references module-level constant)
    MIN_TRIALS = MIN_TRIALS_THRESHOLD  # Minimum trials for inclusion (~50% of expected 807-1077)
    MIN_ACCURACY = 0.3  # Minimum accuracy for inclusion (below chance suggests invalid data)

    # Block filtering (for model fitting)
    MIN_BLOCKS = 8  # Minimum main task blocks for reliable parameter estimation
    MAIN_TASK_START_BLOCK = 3  # First main task block (exclude practice)

    # Trial filtering
    MIN_RT = 200  # Minimum RT (ms) - faster suggests anticipatory
    MAX_RT = 2000  # Maximum RT (ms) - task timeout

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

class AnalysisParams:
    """Parameters for statistical analyses."""

    # Significance threshold
    ALPHA = 0.05

    # Bootstrap parameters
    N_BOOTSTRAP = 10000
    BOOTSTRAP_CI = 95  # Confidence interval percentage

    # Figure settings
    FIG_DPI = 300
    FIG_FORMAT = 'png'
    FIGURE_SIZE_DEFAULT = (10, 6)

    # Color schemes
    COLORS_LOAD = {
        'low': '#3498db',  # Blue
        'high': '#e74c3c',  # Red
    }

    COLORS_PHASE = {
        'practice_static': '#95a5a6',  # Gray
        'practice_dynamic': '#9b59b6',  # Purple
        'main_task': '#2ecc71',  # Green
    }

    COLORS_SET_SIZE = {
        2: '#27ae60',  # Green
        3: '#3498db',  # Blue
        5: '#f39c12',  # Orange
        6: '#e74c3c',  # Red
    }

    # Random seed for reproducibility
    RANDOM_SEED = 42

    # Set numpy random seed
    np.random.seed(RANDOM_SEED)

# ============================================================================
# DYNAMIC EXCLUSION COMPUTATION
# ============================================================================

def get_excluded_participants(data_path: Path | None = None) -> list[int]:
    """
    Compute participant exclusion list from data quality thresholds.

    Automatically excludes participants with fewer than MIN_TRIALS_THRESHOLD
    trials, combined with any MANUAL_EXCLUSIONS. Runs once on import so
    exclusions stay current whenever new data is collected.

    Parameters
    ----------
    data_path : Path, optional
        Path to task_trials_long.csv. Defaults to OUTPUT_DIR / 'task_trials_long.csv'.

    Returns
    -------
    list[int]
        Sorted list of participant IDs to exclude.
    """
    if data_path is None:
        data_path = OUTPUT_DIR / 'task_trials_long.csv'

    if not data_path.exists():
        return sorted(MANUAL_EXCLUSIONS)

    import pandas as pd

    df = pd.read_csv(data_path, usecols=['sona_id'])
    trial_counts = df.groupby('sona_id').size()
    auto_excluded = trial_counts[trial_counts < MIN_TRIALS_THRESHOLD].index.tolist()

    return sorted(set(auto_excluded) | set(MANUAL_EXCLUSIONS))


# Computed on import — automatically updates when data changes
EXCLUDED_PARTICIPANTS = get_excluded_participants()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_set_size_load_condition(set_size):
    """
    Classify set size into load condition.

    Parameters
    ----------
    set_size : int
        Number of stimuli in the set

    Returns
    -------
    str
        'low' if set_size <= 3, 'high' otherwise
    """
    return 'low' if set_size <= TaskParams.LOW_LOAD_THRESHOLD else 'high'

def get_phase_type(block):
    """
    Determine phase type from block number.

    Parameters
    ----------
    block : int
        Block number (1-23)

    Returns
    -------
    str
        Phase type: 'practice_static', 'practice_dynamic', or 'main_task'
    """
    if block == 1:
        return TaskParams.PHASE_PRACTICE_STATIC
    elif block == 2:
        return TaskParams.PHASE_PRACTICE_DYNAMIC
    else:
        return TaskParams.PHASE_MAIN_TASK

def sample_reversal_point(rng=None):
    """
    Sample a reversal point uniformly from [REVERSAL_MIN, REVERSAL_MAX].

    Parameters
    ----------
    rng : np.random.RandomState, optional
        Random number generator. If None, uses numpy default.

    Returns
    -------
    int
        Number of consecutive correct responses before reversal
    """
    if rng is None:
        rng = np.random
    return rng.randint(TaskParams.REVERSAL_MIN, TaskParams.REVERSAL_MAX + 1)

# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================

def print_config_summary():
    """Print a summary of the current configuration."""
    print("=" * 80)
    print("RLWM TRAUMA ANALYSIS - CONFIGURATION SUMMARY")
    print("Following Senta et al. (2025) model specifications")
    print("=" * 80)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Version: {VERSION}")
    print("\nTask Parameters:")
    print(f"  - Actions: {TaskParams.NUM_ACTIONS}")
    print(f"  - Set Sizes: {TaskParams.SET_SIZES}")
    print(f"  - Reversal Range: [{TaskParams.REVERSAL_MIN}, {TaskParams.REVERSAL_MAX}]")
    print(f"  - Reward: Correct={TaskParams.REWARD_CORRECT}, Incorrect={TaskParams.REWARD_INCORRECT}")
    print("\nModel Defaults (Senta et al., 2025):")
    print(f"  - Learning Rate (α_pos): {ModelParams.ALPHA_POS_DEFAULT}")
    print(f"  - Learning Rate (α_neg): {ModelParams.ALPHA_NEG_DEFAULT}")
    print(f"  - Inverse Temperature (β): {ModelParams.BETA_FIXED} (FIXED)")
    print(f"  - Epsilon Noise (ε): {ModelParams.EPSILON_DEFAULT}")
    print(f"  - Discount Factor (γ): {ModelParams.GAMMA_DEFAULT}")
    print(f"  - WM Capacity (K): {ModelParams.WM_CAPACITY_DEFAULT}")
    print(f"  - WM Decay (φ): {ModelParams.PHI_DEFAULT}")
    print(f"  - WM Reliance (ρ): {ModelParams.RHO_DEFAULT}")
    print("\nPyMC Sampling:")
    print(f"  - Chains: {PyMCParams.NUM_CHAINS}")
    print(f"  - Samples: {PyMCParams.NUM_SAMPLES}")
    print(f"  - Tune: {PyMCParams.NUM_TUNE}")
    print("=" * 80)

if __name__ == "__main__":
    print_config_summary()
