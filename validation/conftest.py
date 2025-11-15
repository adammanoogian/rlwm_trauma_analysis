"""
Shared pytest fixtures for RLWM trauma analysis tests.

These fixtures are automatically available to all test files.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_trial_data():
    """
    Generate simple trial sequence for testing.

    Returns
    -------
    dict
        Dictionary with 'stimuli', 'actions', 'rewards' arrays
    """
    return {
        'stimuli': np.array([0, 1, 2, 0, 1, 2]),
        'actions': np.array([0, 1, 2, 1, 0, 2]),
        'rewards': np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    }


@pytest.fixture
def sample_agent_params():
    """
    Standard agent parameters for testing.

    Returns
    -------
    dict
        Dictionary with parameter sets for each model type
    """
    return {
        'qlearning': {
            'alpha_pos': 0.3,
            'alpha_neg': 0.1,
            'beta': 3.0,
            'gamma': 0.0
        },
        'wmrl': {
            'alpha_pos': 0.3,
            'alpha_neg': 0.1,
            'beta': 2.0,
            'beta_wm': 3.0,
            'capacity': 4,
            'phi': 0.1,
            'rho': 0.7,
            'gamma': 0.0,
            'wm_init': 0.0
        }
    }


@pytest.fixture
def sample_participant_data():
    """
    Generate realistic participant data for testing.

    Returns
    -------
    pd.DataFrame
        DataFrame with trial-level data for one participant
    """
    np.random.seed(42)

    data = []
    for trial in range(50):
        data.append({
            'sona_id': 'TEST_P001',
            'block': 3,
            'trial': trial + 1,
            'stimulus': np.random.randint(0, 3),  # 0-indexed
            'key_press': np.random.randint(0, 3),
            'correct': np.random.randint(0, 2)
        })

    return pd.DataFrame(data)


@pytest.fixture
def sample_multiparticipant_data():
    """
    Generate data for multiple participants.

    Returns
    -------
    pd.DataFrame
        DataFrame with trial-level data for 3 participants
    """
    np.random.seed(42)

    data = []
    for p_idx in range(3):
        for trial in range(30):
            data.append({
                'sona_id': f'TEST_P{p_idx+1:03d}',
                'block': 3,
                'trial': trial + 1,
                'stimulus': np.random.randint(0, 3),
                'key_press': np.random.randint(0, 3),
                'correct': np.random.randint(0, 2)
            })

    return pd.DataFrame(data)


@pytest.fixture
def project_root():
    """
    Get project root directory.

    Returns
    -------
    Path
        Path to project root
    """
    return Path(__file__).parent.parent


@pytest.fixture
def output_dir(project_root, tmp_path):
    """
    Create temporary output directory for tests.

    Parameters
    ----------
    project_root : Path
        Project root fixture
    tmp_path : Path
        pytest temporary directory fixture

    Returns
    -------
    Path
        Temporary output directory
    """
    test_output = tmp_path / 'test_output'
    test_output.mkdir(exist_ok=True)
    return test_output
