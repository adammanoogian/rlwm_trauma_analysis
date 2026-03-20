"""
Pure-data configuration for the RLWM package.

Extracted from root config.py to be import-safe (no side effects,
no path-dependent logic, no numpy.random.seed calls).
"""

from __future__ import annotations


class TaskParams:
    """Parameters for the RLWM task environment."""

    # Response keys (J, K, L mapped to 0, 1, 2)
    NUM_ACTIONS = 3
    ACTIONS = [0, 1, 2]

    # Stimulus set configuration
    MAX_STIMULI = 6
    SET_SIZES = [2, 3, 5, 6]
    EXCLUDE_SET_SIZES = [4]

    # Working memory load classification
    LOW_LOAD_THRESHOLD = 3
    HIGH_LOAD_THRESHOLD = 4

    # Reversal parameters
    RARE_REVERSALS = True
    REVERSAL_MIN = 12
    REVERSAL_MAX = 18
    MAX_REVERSALS_PER_STIM = 1

    # Practice block parameters
    PRACTICE_DYNAMIC_REVERSAL_CRITERION = 5
    PRACTICE_DYNAMIC_NUM_REVERSALS = 2

    # Reward structure
    REWARD_CORRECT = 1.0
    REWARD_INCORRECT = 0.0

    # Timing (in milliseconds)
    FIXATION_DURATION = 500
    TRIAL_DURATION = 2000
    FEEDBACK_DURATION = 500

    # Block structure
    PRACTICE_STATIC_BLOCK = 1
    PRACTICE_DYNAMIC_BLOCK = 2
    NUM_PRACTICE_BLOCKS = 2
    MAIN_TASK_START_BLOCK = 3
    MAIN_TASK_END_BLOCK = 23
    NUM_MAIN_BLOCKS = 21
    TOTAL_BLOCKS = NUM_PRACTICE_BLOCKS + NUM_MAIN_BLOCKS

    # Trials per block
    TRIALS_PER_BLOCK_MIN = 30
    TRIALS_PER_BLOCK_MAX = 90
    TRIALS_PER_BLOCK_MEAN = 58
    TRIALS_PER_BLOCK_MEDIAN = 45
    TRIALS_PER_BLOCK_DEFAULT = 100

    # Phase types
    PHASE_PRACTICE_STATIC = "practice_static"
    PHASE_PRACTICE_DYNAMIC = "practice_dynamic"
    PHASE_MAIN_TASK = "main_task"


class ModelParams:
    """
    Default hyperparameters for RL models.

    Following Senta et al. (2025):
    - Beta is FIXED at 50 during learning for parameter identifiability
    - Epsilon noise captures random responding
    """

    ALPHA_POS_DEFAULT = 0.3
    ALPHA_NEG_DEFAULT = 0.1
    ALPHA_MIN = 0.0
    ALPHA_MAX = 1.0

    BETA_FIXED = 50.0
    BETA_DEFAULT = 50.0

    GAMMA_DEFAULT = 0.0
    GAMMA_MIN = 0.0
    GAMMA_MAX = 1.0

    EPSILON_DEFAULT = 0.05
    EPSILON_MIN = 0.0
    EPSILON_MAX = 1.0

    WM_CAPACITY_DEFAULT = 4
    WM_CAPACITY_MIN = 1
    WM_CAPACITY_MAX = 7

    PHI_DEFAULT = 0.1
    PHI_MIN = 0.0
    PHI_MAX = 1.0

    RHO_DEFAULT = 0.7
    RHO_MIN = 0.0
    RHO_MAX = 1.0

    Q_INIT_VALUE = 0.5
    WM_INIT_VALUE = 1.0 / 3.0
    NUM_ACTIONS = 3
