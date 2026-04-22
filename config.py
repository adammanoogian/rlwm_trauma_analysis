"""
Central configuration for RLWM Trauma Analysis project.

This module contains all project paths, task parameters, model hyperparameters,
and analysis settings to ensure consistency across all scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import arviz as az

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
    WM_CAPACITY_MIN = 2
    WM_CAPACITY_MAX = 6

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
# ANALYSIS COHORT (canonical inclusion criteria)
# ============================================================================

# Above-chance accuracy thresholds.  Chance = 1 / TaskParams.NUM_ACTIONS = 0.333.
# Overall accuracy >= MIN_OVERALL_ACCURACY AND late-block accuracy
# >= MIN_LATE_BLOCK_ACCURACY.  The late-block criterion confirms learning
# happened, not just chance performance across the whole session.
MIN_OVERALL_ACCURACY: float = 0.40      # ~20% above chance
MIN_LATE_BLOCK_ACCURACY: float = 0.50   # ~50% above chance in terminal blocks
LATE_BLOCK_N: int = 5                   # evaluate accuracy on last 5 blocks

# Columns treated as required for the trauma-scale regression.
REQUIRED_SURVEY_COLUMNS: tuple[str, ...] = (
    "less_total_events",   # LEC-5 exposure count
    "ies_total",           # IES-R symptom total
    "ies_intrusion",
    "ies_avoidance",
    "ies_hyperarousal",
)


def get_analysis_cohort(
    data_path: Path | None = None,
    surveys_path: Path | None = None,
    *,
    min_trials: int | None = None,
    min_blocks: int | None = None,
    min_overall_accuracy: float = MIN_OVERALL_ACCURACY,
    min_late_block_accuracy: float = MIN_LATE_BLOCK_ACCURACY,
    late_block_n: int = LATE_BLOCK_N,
    require_scales: bool = True,
    verbose: bool = False,
) -> list[int]:
    """Canonical analysis cohort for v4.0+.

    Returns the intersection of three inclusion criteria:

    1. **Task completeness** — at least ``min_trials`` trials
       (default ``MIN_TRIALS_THRESHOLD = 400``) and at least ``min_blocks``
       main-task blocks (default ``DataParams.MIN_BLOCKS = 8``).
    2. **Performance check** — overall accuracy >= ``min_overall_accuracy``
       AND mean accuracy over the last ``late_block_n`` main-task blocks
       >= ``min_late_block_accuracy``.  The late-block criterion confirms
       the participant learned rather than merely performed above chance
       by luck across the whole session.
    3. **Scale completeness** (if ``require_scales=True``) — non-NA values
       for every column in :data:`REQUIRED_SURVEY_COLUMNS`.

    Participants in ``MANUAL_EXCLUSIONS`` are always excluded.

    Parameters
    ----------
    data_path : Path, optional
        Path to ``task_trials_long.csv``.  Defaults to
        ``OUTPUT_DIR / 'task_trials_long.csv'``.
    surveys_path : Path, optional
        Path to ``summary_participant_metrics.csv``.  Defaults to
        ``DataParams.SUMMARY_METRICS``.
    min_trials, min_blocks : int, optional
        Override the task-completeness thresholds.
    min_overall_accuracy, min_late_block_accuracy, late_block_n : numeric
        Override the performance-check thresholds.
    require_scales : bool
        If True, require non-NA values for every column listed in
        :data:`REQUIRED_SURVEY_COLUMNS`.  Set False for diagnostic runs
        that don't need trauma regression.
    verbose : bool
        If True, print the exclusion breakdown (how many participants
        fail each criterion).

    Returns
    -------
    list[int]
        Sorted list of participant IDs that pass all enabled criteria.
        Returns an empty list if the required data files are missing.

    Notes
    -----
    This is the single source of truth for the v4.0 analysis cohort.
    All downstream scripts (fitting, regression, paper.qmd) should call
    this function rather than maintaining their own filter logic.
    """
    if data_path is None:
        data_path = OUTPUT_DIR / "task_trials_long.csv"
    if surveys_path is None:
        surveys_path = DataParams.SUMMARY_METRICS
    if min_trials is None:
        min_trials = MIN_TRIALS_THRESHOLD
    if min_blocks is None:
        min_blocks = DataParams.MIN_BLOCKS

    if not data_path.exists():
        if verbose:
            print(f"get_analysis_cohort: task data not found at {data_path}")
        return []

    import pandas as pd

    # ------------------------------------------------------------------
    # 1. Task-completeness: trial count + block count
    # ------------------------------------------------------------------
    df = pd.read_csv(data_path, usecols=["sona_id", "block", "correct"])
    df = df.dropna(subset=["sona_id"]).copy()
    df["sona_id"] = df["sona_id"].astype(int)

    trial_counts = df.groupby("sona_id").size()
    block_counts = df.groupby("sona_id")["block"].nunique()

    task_pass = set(trial_counts[trial_counts >= min_trials].index) & set(
        block_counts[block_counts >= min_blocks].index
    )

    # ------------------------------------------------------------------
    # 2. Performance check: overall + late-block accuracy above chance
    # ------------------------------------------------------------------
    overall_acc = df.groupby("sona_id")["correct"].mean()
    overall_pass = set(overall_acc[overall_acc >= min_overall_accuracy].index)

    # Late-block accuracy: per-participant max block, then select the last
    # `late_block_n` blocks for each participant.
    def _late_block_mean(group: "pd.DataFrame") -> float:
        max_b = int(group["block"].max())
        min_b = max_b - late_block_n + 1
        return float(group.loc[group["block"] >= min_b, "correct"].mean())

    late_acc = df.groupby("sona_id").apply(_late_block_mean, include_groups=False) \
        if "include_groups" in getattr(df.groupby("sona_id").apply, "__kwdefaults__", {}) or True \
        else df.groupby("sona_id").apply(_late_block_mean)
    # Robustness: pandas warns about include_groups; fall back cleanly.
    try:
        late_acc = df.groupby("sona_id").apply(_late_block_mean, include_groups=False)
    except TypeError:
        late_acc = df.groupby("sona_id").apply(_late_block_mean)
    late_pass = set(late_acc[late_acc >= min_late_block_accuracy].index)

    perf_pass = overall_pass & late_pass

    # ------------------------------------------------------------------
    # 3. Scale-completeness: non-NA for every required survey column
    # ------------------------------------------------------------------
    if require_scales and surveys_path.exists():
        surveys_df = pd.read_csv(surveys_path)
        surveys_df = surveys_df.dropna(subset=["sona_id"]).copy()
        surveys_df["sona_id"] = surveys_df["sona_id"].astype(int)
        complete_mask = surveys_df[list(REQUIRED_SURVEY_COLUMNS)].notna().all(
            axis=1
        )
        scales_pass = set(surveys_df.loc[complete_mask, "sona_id"].tolist())
    elif require_scales:
        if verbose:
            print(
                f"get_analysis_cohort: surveys file missing at {surveys_path}; "
                "treating no participants as scale-complete."
            )
        scales_pass = set()
    else:
        scales_pass = set(task_pass)  # effectively disables the gate

    cohort = task_pass & perf_pass & scales_pass
    cohort -= set(MANUAL_EXCLUSIONS)

    if verbose:
        all_ids = set(df["sona_id"].unique().tolist())
        print(f"Analysis cohort breakdown (total observed N={len(all_ids)}):")
        print(f"  Task-complete (>= {min_trials} trials, >= {min_blocks} blocks): {len(task_pass)}")
        print(f"  Overall accuracy >= {min_overall_accuracy}: {len(overall_pass)}")
        print(f"  Late-block accuracy >= {min_late_block_accuracy} (last {late_block_n}): {len(late_pass)}")
        print(f"  Both performance gates: {len(perf_pass)}")
        if require_scales:
            print(f"  Scale-complete ({len(REQUIRED_SURVEY_COLUMNS)} cols): {len(scales_pass)}")
        print(f"  Final cohort (intersection − manual exclusions): {len(cohort)}")

    return sorted(cohort)


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

# ============================================================================
# PARAMETERIZATION VERSION REGISTRY & VALIDATION
# ============================================================================

EXPECTED_PARAMETERIZATION: dict[str, str] = {
    "qlearning": "v4.0-phiapprox",
    "wmrl": "v4.0-K[2,6]-phiapprox",
    "wmrl_m3": "v4.0-K[2,6]-phiapprox",
    "wmrl_m5": "v4.0-K[2,6]-phiapprox",
    "wmrl_m6a": "v4.0-K[2,6]-phiapprox",
    "wmrl_m6b": "v4.0-K[2,6]-phiapprox-stickbreaking",
    "wmrl_m4": "v4.0-K[2,6]-phiapprox-lba",
}
"""Expected parameterization_version string for each model's fit CSV.

v4.0+ invariant: every fit CSV produced by this project MUST carry a
``parameterization_version`` column whose value matches the string above
for its model. :func:`load_fits_with_validation` enforces this on the
read side; :mod:`rlwm.fitting.jax_likelihoods` and :mod:`rlwm.fitting.numpyro_models`
enforce it on the write side (entry scripts: ``scripts/04_model_fitting/a_mle/fit_mle.py``
and ``scripts/04_model_fitting/b_bayesian/fit_bayesian.py``).

Mismatched or absent values are rejected with an informative error
(expected vs. actual) rather than silently producing wrong inferences.
The v3.0 K ∈ [1, 7] parameterization is no longer an accepted vocabulary
entry — only Collins K ∈ [2, 6] CSVs validate.
"""


def load_fits_with_validation(
    path: "Path",
    model: str,
) -> "pd.DataFrame":
    """Load a fit CSV and validate its parameterization_version.

    Raises loudly if the CSV lacks a ``parameterization_version`` column
    or if the version does not match the expected string for ``model``.
    This prevents stale v3.0 fits (K in [1, 7]) from contaminating v4.0
    hierarchical analyses.

    Parameters
    ----------
    path : Path
        Path to the CSV file (individual-level MLE or Bayesian fits).
    model : str
        Model name key into :data:`EXPECTED_PARAMETERIZATION`
        (e.g. ``"wmrl_m3"``).

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame, validated.

    Raises
    ------
    ValueError
        If ``parameterization_version`` column is absent.  Error message
        includes expected version so the user knows what to look for.
    ValueError
        If the column contains a value that does not match the expected
        string.  Error message includes expected vs. actual values.
    """
    import pandas as pd

    df = pd.read_csv(path)
    if "parameterization_version" not in df.columns:
        expected = EXPECTED_PARAMETERIZATION.get(model, "unknown")
        raise ValueError(
            f"{path} lacks 'parameterization_version' column — "
            f"(pre-v4.0 CSVs without this column are rejected). "
            f"Expected: {expected}"
        )
    expected = EXPECTED_PARAMETERIZATION[model]
    actual = df["parameterization_version"].unique()
    if len(actual) != 1 or actual[0] != expected:
        raise ValueError(
            f"{path} parameterization_version mismatch: "
            f"expected='{expected}', actual={list(actual)}"
        )
    return df


def load_netcdf_with_validation(
    path: "Path",
    model: str,
) -> "az.InferenceData":
    """Load a Bayesian posterior NetCDF and validate basic invariants.

    Companion to :func:`load_fits_with_validation` (which validates CSV
    fits). This wrapper handles the NetCDF load path and is the
    single entry point for every downstream Bayesian consumer script.

    Validation contract (v5.0 Phase 23 CLEAN-04):

    1. ``path.exists()`` is True and file size > 0.
    2. ``az.from_netcdf(path)`` returns a valid ``InferenceData`` with
       a ``posterior`` group.
    3. If ``idata.attrs`` contains a ``parameterization_version`` key,
       it must match ``EXPECTED_PARAMETERIZATION[model]``; mismatch
       raises :class:`ValueError` with expected vs. actual values.
    4. If the attr is absent (current v4.0 NetCDF write-side behaviour),
       emit a :class:`DeprecationWarning` — do NOT raise. Write-side
       attr retrofit is a v5.1 item tracked in ROADMAP.

    Parameters
    ----------
    path : Path
        Path to the NetCDF file (e.g. ``output/bayesian/wmrl_m3_posterior.nc``).
    model : str
        Model name key into :data:`EXPECTED_PARAMETERIZATION`
        (e.g. ``"wmrl_m3"``). Used for attr validation when present.

    Returns
    -------
    az.InferenceData
        The loaded InferenceData object, validated.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist or is empty.
    ValueError
        If the file loads but has no ``posterior`` group, or the
        embedded ``parameterization_version`` attr mismatches.
    """
    import warnings

    import arviz as az

    if not path.exists():
        raise FileNotFoundError(
            f"NetCDF posterior not found: {path} "
            f"(expected for model '{model}')"
        )
    if path.stat().st_size == 0:
        raise ValueError(
            f"NetCDF posterior is empty (0 bytes): {path} "
            f"(expected for model '{model}')"
        )

    idata = az.from_netcdf(str(path))

    if not hasattr(idata, "posterior"):
        raise ValueError(
            f"{path} loaded but has no 'posterior' group — "
            f"not a valid Bayesian posterior InferenceData"
        )

    attrs = getattr(idata, "attrs", {}) or {}
    actual_version = attrs.get("parameterization_version")

    if actual_version is not None:
        expected = EXPECTED_PARAMETERIZATION.get(model)
        if expected is None:
            raise ValueError(
                f"Unknown model '{model}' — not in "
                f"EXPECTED_PARAMETERIZATION registry"
            )
        if actual_version != expected:
            raise ValueError(
                f"{path} parameterization_version mismatch: "
                f"expected='{expected}', actual='{actual_version}'"
            )
    else:
        warnings.warn(
            f"{path} has no 'parameterization_version' attr — "
            f"cannot hard-validate against EXPECTED_PARAMETERIZATION. "
            f"v4.0 NetCDF write sites do not emit this attr; retrofit "
            f"tracked as a v5.1 item. Model: '{model}'.",
            DeprecationWarning,
            stacklevel=2,
        )

    return idata


if __name__ == "__main__":
    print_config_summary()
