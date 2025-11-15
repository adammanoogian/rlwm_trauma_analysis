"""
Central configuration for RLWM Trauma Analysis project.

This module contains all project paths, task parameters, model hyperparameters,
and analysis settings to ensure consistency across all scripts.
"""

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

    # Block structure
    NUM_PRACTICE_BLOCKS = 2  # Blocks 1-2: practice
    NUM_MAIN_BLOCKS = 21  # Blocks 3-23: main task
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
    """Default hyperparameters for RL models."""

    # Q-learning parameters (asymmetric learning rates)
    ALPHA_POS_DEFAULT = 0.3  # Learning rate for positive PE (correct trials)
    ALPHA_NEG_DEFAULT = 0.1  # Learning rate for negative PE (incorrect trials)
    ALPHA_MIN = 0.0
    ALPHA_MAX = 1.0

    BETA_DEFAULT = 2.0  # Inverse temperature (softmax) (>0)
    BETA_MIN = 0.01
    BETA_MAX = 20.0

    GAMMA_DEFAULT = 0.0  # Discount factor (fixed at 0 for this task)
    GAMMA_MIN = 0.0
    GAMMA_MAX = 1.0

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

    BETA_WM_DEFAULT = 3.0  # WM inverse temperature (>0)
    BETA_WM_MIN = 0.01
    BETA_WM_MAX = 20.0

    # Exploration parameters
    EPSILON_DEFAULT = 0.1  # Epsilon-greedy exploration (0-1)
    EPSILON_MIN = 0.0
    EPSILON_MAX = 1.0

    # Initialization
    Q_INIT_VALUE = 0.5  # Initial Q-values (optimistic initialization)
    WM_INIT_VALUE = 0.0  # Initial WM values (baseline)

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
    TASK_TRIALS_LONG = OUTPUT_DIR / 'task_trials_long.csv'
    SUMMARY_METRICS = OUTPUT_DIR / 'summary_participant_metrics.csv'

    # Simulated data paths
    SIMULATED_DATA = OUTPUT_VERSION_DIR / 'simulated_data.csv'
    FITTED_POSTERIORS = OUTPUT_VERSION_DIR / 'fitted_posteriors.nc'  # NetCDF format
    MODEL_COMPARISON = OUTPUT_VERSION_DIR / 'model_comparison.csv'

    # Exclusion criteria
    MIN_TRIALS = 50  # Minimum trials for inclusion
    MIN_ACCURACY = 0.3  # Minimum accuracy for inclusion (below chance suggests invalid data)

    # Trial filtering
    MIN_RT = 200  # Minimum RT (ms) - faster suggests anticipatory
    MAX_RT = 2000  # Maximum RT (ms) - task timeout

    # Block filtering
    MAIN_TASK_START_BLOCK = 3  # First main task block (exclude practice)

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
    print("=" * 80)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Version: {VERSION}")
    print(f"\nTask Parameters:")
    print(f"  - Actions: {TaskParams.NUM_ACTIONS}")
    print(f"  - Set Sizes: {TaskParams.SET_SIZES}")
    print(f"  - Reversal Range: [{TaskParams.REVERSAL_MIN}, {TaskParams.REVERSAL_MAX}]")
    print(f"  - Reward: Correct={TaskParams.REWARD_CORRECT}, Incorrect={TaskParams.REWARD_INCORRECT}")
    print(f"\nModel Defaults:")
    print(f"  - Learning Rate (α_pos): {ModelParams.ALPHA_POS_DEFAULT}")
    print(f"  - Learning Rate (α_neg): {ModelParams.ALPHA_NEG_DEFAULT}")
    print(f"  - Inverse Temperature (β): {ModelParams.BETA_DEFAULT}")
    print(f"  - Discount Factor (γ): {ModelParams.GAMMA_DEFAULT}")
    print(f"  - WM Capacity: {ModelParams.WM_CAPACITY_DEFAULT}")
    print(f"\nPyMC Sampling:")
    print(f"  - Chains: {PyMCParams.NUM_CHAINS}")
    print(f"  - Samples: {PyMCParams.NUM_SAMPLES}")
    print(f"  - Tune: {PyMCParams.NUM_TUNE}")
    print("=" * 80)

if __name__ == "__main__":
    print_config_summary()
