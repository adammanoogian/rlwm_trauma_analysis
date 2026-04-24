"""Root conftest for the consolidated Phase 31 test tree.

Merges fixtures previously split across three conftests:

- ``scripts/fitting/tests/conftest.py`` (integration fixtures: Q-learning
  and WM-RL synthetic data generators; M4 participant-data fixtures).
- ``validation/conftest.py`` (scientific fixtures: sample trial data,
  sample agent params, sample participant data, project root, output dir).
- ``tests/conftest.py`` (previously absent at tests root).

Tier markers (auto-applied by ``pytest_collection_modifyitems``):

- ``unit``       — fast (< 1s), isolated  (``tests/unit/``)
- ``integration`` — medium (1-60s), cross-module  (``tests/integration/``)
- ``scientific`` — slow (> 60s), parameter recovery / v4 closure
  (``tests/scientific/``)

The ``unit`` and ``integration`` markers are declared under
``[pytest] markers`` in ``pytest.ini``; ``scientific`` and ``slow`` are
declared there too. ``slow`` is also added to scientific items so
``pytest -m "not slow"`` excludes them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Repo-root sys.path shim
# ---------------------------------------------------------------------------
#
# Make repo root importable so that
# ``from tests.scientific.check_v4_closure import ...`` resolves regardless
# of pytest invocation directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Tier auto-marking
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config, items):
    """Apply tier marker based on directory location.

    Parameters
    ----------
    config : pytest.Config
        Pytest configuration object (unused but required by hook).
    items : list[pytest.Item]
        Collected test items; markers applied in-place.
    """
    for item in items:
        path_str = str(item.fspath).replace("\\", "/")
        if "/tests/unit/" in path_str:
            item.add_marker(pytest.mark.unit)
        elif "/tests/integration/" in path_str:
            item.add_marker(pytest.mark.integration)
        elif "/tests/scientific/" in path_str:
            item.add_marker(pytest.mark.scientific)
            item.add_marker(pytest.mark.slow)


# ---------------------------------------------------------------------------
# Scientific-tier fixtures (migrated from validation/conftest.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_trial_data():
    """Generate simple trial sequence for testing.

    Returns
    -------
    dict
        Dictionary with ``stimuli``, ``actions``, ``rewards`` numpy arrays.
    """
    return {
        "stimuli": np.array([0, 1, 2, 0, 1, 2]),
        "actions": np.array([0, 1, 2, 1, 0, 2]),
        "rewards": np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
    }


@pytest.fixture
def sample_agent_params():
    """Standard agent parameters for testing.

    Returns
    -------
    dict
        Dictionary with parameter sets for each model type.
    """
    return {
        "qlearning": {
            "alpha_pos": 0.3,
            "alpha_neg": 0.1,
            "beta": 3.0,
            "gamma": 0.0,
        },
        "wmrl": {
            "alpha_pos": 0.3,
            "alpha_neg": 0.1,
            "beta": 2.0,
            "beta_wm": 3.0,
            "capacity": 4,
            "phi": 0.1,
            "rho": 0.7,
            "gamma": 0.0,
            "wm_init": 0.0,
        },
    }


@pytest.fixture
def sample_participant_data():
    """Generate realistic participant data for testing.

    Returns
    -------
    pd.DataFrame
        DataFrame with trial-level data for one participant.
    """
    np.random.seed(42)

    data = []
    for trial in range(50):
        data.append(
            {
                "sona_id": "TEST_P001",
                "block": 3,
                "trial": trial + 1,
                "stimulus": np.random.randint(0, 3),
                "key_press": np.random.randint(0, 3),
                "correct": np.random.randint(0, 2),
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def sample_multiparticipant_data():
    """Generate data for multiple participants.

    Returns
    -------
    pd.DataFrame
        DataFrame with trial-level data for 3 participants.
    """
    np.random.seed(42)

    data = []
    for p_idx in range(3):
        for trial in range(30):
            data.append(
                {
                    "sona_id": f"TEST_P{p_idx + 1:03d}",
                    "block": 3,
                    "trial": trial + 1,
                    "stimulus": np.random.randint(0, 3),
                    "key_press": np.random.randint(0, 3),
                    "correct": np.random.randint(0, 2),
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def project_root():
    """Return the project root directory.

    Returns
    -------
    Path
        Path to the repository root (``tests/`` parent).
    """
    return Path(__file__).parent.parent


@pytest.fixture
def output_dir(project_root, tmp_path):
    """Create a temporary output directory for tests.

    Parameters
    ----------
    project_root : Path
        Project root fixture.
    tmp_path : Path
        pytest temporary directory fixture.

    Returns
    -------
    Path
        Temporary output directory (under pytest ``tmp_path``).
    """
    test_output = tmp_path / "test_output"
    test_output.mkdir(exist_ok=True)
    return test_output


# ---------------------------------------------------------------------------
# Integration-tier fixtures
# (migrated from scripts/fitting/tests/conftest.py)
# ---------------------------------------------------------------------------


def simulate_qlearning_block(
    alpha_pos: float,
    alpha_neg: float,
    epsilon: float,
    beta: float,
    n_trials: int,
    n_stim: int,
    n_act: int,
    seed: int = 42,
):
    """Simulate a single block of Q-learning behavior.

    Parameters
    ----------
    alpha_pos : float
        Positive learning rate.
    alpha_neg : float
        Negative learning rate.
    epsilon : float
        Noise parameter.
    beta : float
        Inverse temperature.
    n_trials : int
        Number of trials to simulate.
    n_stim : int
        Number of unique stimuli.
    n_act : int
        Number of actions.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    tuple of np.ndarray
        ``(stimuli, actions, rewards)`` arrays.
    """
    Q = np.ones((n_stim, n_act)) * 0.5
    np.random.seed(seed)
    correct_actions = np.random.randint(0, n_act, n_stim)

    stimuli, actions, rewards = [], [], []

    for _ in range(n_trials):
        s = np.random.randint(0, n_stim)
        q_s = Q[s, :]
        exp_q = np.exp(beta * (q_s - q_s.max()))
        p = exp_q / exp_q.sum()
        p = epsilon / n_act + (1 - epsilon) * p

        a = np.random.choice(n_act, p=p)
        r = 1.0 if a == correct_actions[s] else 0.0

        stimuli.append(s)
        actions.append(a)
        rewards.append(r)

        delta = r - Q[s, a]
        alpha = alpha_pos if delta > 0 else alpha_neg
        Q[s, a] += alpha * delta

    return (
        np.array(stimuli, dtype=np.int32),
        np.array(actions, dtype=np.int32),
        np.array(rewards, dtype=np.float32),
    )


def simulate_wmrl_block(
    alpha_pos: float,
    alpha_neg: float,
    phi: float,
    rho: float,
    capacity: float,
    epsilon: float,
    beta: float,
    n_trials: int,
    n_stim: int,
    n_act: int,
    set_size: int = 6,
    seed: int = 42,
):
    """Simulate a single block of WM-RL behavior.

    Simplified WM-RL simulation that combines RL and WM components.

    Parameters
    ----------
    alpha_pos : float
        Positive learning rate (RL).
    alpha_neg : float
        Negative learning rate (RL).
    phi : float
        WM decay rate.
    rho : float
        Base WM reliance.
    capacity : float
        WM capacity (K).
    epsilon : float
        Noise parameter.
    beta : float
        Inverse temperature.
    n_trials : int
        Number of trials.
    n_stim : int
        Number of unique stimuli.
    n_act : int
        Number of actions.
    set_size : int, default=6
        Set size for capacity computation.
    seed : int, default=42
        Random seed.

    Returns
    -------
    tuple of np.ndarray
        ``(stimuli, actions, rewards, set_sizes)`` arrays.
    """
    np.random.seed(seed)

    Q = np.ones((n_stim, n_act)) * 0.5
    WM = np.ones((n_stim, n_act)) * (1.0 / n_act)

    correct_actions = np.random.randint(0, n_act, n_stim)

    stimuli, actions, rewards = [], [], []

    for _ in range(n_trials):
        s = np.random.randint(0, n_stim)

        w = rho * min(1.0, capacity / set_size)

        q_s = Q[s, :]
        wm_s = WM[s, :]

        exp_q = np.exp(beta * (q_s - q_s.max()))
        p_rl = exp_q / exp_q.sum()

        p = w * wm_s + (1 - w) * p_rl
        p = epsilon / n_act + (1 - epsilon) * p

        a = np.random.choice(n_act, p=p)
        r = 1.0 if a == correct_actions[s] else 0.0

        stimuli.append(s)
        actions.append(a)
        rewards.append(r)

        delta = r - Q[s, a]
        alpha = alpha_pos if delta > 0 else alpha_neg
        Q[s, a] += alpha * delta

        WM[s, a] = r
        WM = WM * (1 - phi) + phi * (1.0 / n_act)

    return (
        np.array(stimuli, dtype=np.int32),
        np.array(actions, dtype=np.int32),
        np.array(rewards, dtype=np.float32),
        np.full(n_trials, set_size, dtype=np.int32),
    )


@pytest.fixture
def qlearning_synthetic_data():
    """Generate synthetic Q-learning data for testing.

    Returns 3 blocks of data with known parameters.
    """
    true_params = {
        "alpha_pos": 0.4,
        "alpha_neg": 0.15,
        "epsilon": 0.05,
        "beta": 50.0,
    }

    stimuli_blocks, actions_blocks, rewards_blocks = [], [], []

    for i in range(3):
        s, a, r = simulate_qlearning_block(
            alpha_pos=true_params["alpha_pos"],
            alpha_neg=true_params["alpha_neg"],
            epsilon=true_params["epsilon"],
            beta=true_params["beta"],
            n_trials=30,
            n_stim=3,
            n_act=3,
            seed=42 + i,
        )
        stimuli_blocks.append(s)
        actions_blocks.append(a)
        rewards_blocks.append(r)

    return {
        "stimuli_blocks": stimuli_blocks,
        "actions_blocks": actions_blocks,
        "rewards_blocks": rewards_blocks,
        "true_params": true_params,
    }


@pytest.fixture
def wmrl_synthetic_data():
    """Generate synthetic WM-RL data for testing.

    Returns 2 blocks of data with known parameters.
    """
    true_params = {
        "alpha_pos": 0.3,
        "alpha_neg": 0.1,
        "phi": 0.1,
        "rho": 0.7,
        "capacity": 4.0,
        "epsilon": 0.05,
        "beta": 50.0,
    }

    stimuli_blocks, actions_blocks, rewards_blocks, set_sizes_blocks = (
        [],
        [],
        [],
        [],
    )

    for i in range(2):
        s, a, r, ss = simulate_wmrl_block(
            alpha_pos=true_params["alpha_pos"],
            alpha_neg=true_params["alpha_neg"],
            phi=true_params["phi"],
            rho=true_params["rho"],
            capacity=true_params["capacity"],
            epsilon=true_params["epsilon"],
            beta=true_params["beta"],
            n_trials=30,
            n_stim=6,
            n_act=3,
            set_size=6,
            seed=42 + i,
        )
        stimuli_blocks.append(s)
        actions_blocks.append(a)
        rewards_blocks.append(r)
        set_sizes_blocks.append(ss)

    return {
        "stimuli_blocks": stimuli_blocks,
        "actions_blocks": actions_blocks,
        "rewards_blocks": rewards_blocks,
        "set_sizes_blocks": set_sizes_blocks,
        "true_params": true_params,
    }


@pytest.fixture
def wmrl_participant_data():
    """Generate WM-RL data in the format expected by numpyro models.

    Returns participant_data dictionary for 2 participants.
    """
    key = jax.random.PRNGKey(42)
    participant_data = {}

    for i in range(2):
        stimuli_blocks = []
        actions_blocks = []
        rewards_blocks = []
        set_sizes_blocks = []

        for _ in range(2):
            key, subkey = jax.random.split(key)
            stimuli = jax.random.randint(subkey, (30,), 0, 6)

            key, subkey = jax.random.split(key)
            actions = jax.random.randint(subkey, (30,), 0, 3)

            key, subkey = jax.random.split(key)
            rewards = jax.random.bernoulli(subkey, 0.7, (30,)).astype(jnp.float32)

            set_sizes = jnp.ones((30,)) * 5

            stimuli_blocks.append(stimuli)
            actions_blocks.append(actions)
            rewards_blocks.append(rewards)
            set_sizes_blocks.append(set_sizes)

        participant_data[i] = {
            "stimuli_blocks": stimuli_blocks,
            "actions_blocks": actions_blocks,
            "rewards_blocks": rewards_blocks,
            "set_sizes_blocks": set_sizes_blocks,
        }

    return participant_data


@pytest.fixture
def m4_synthetic_data_small():
    """Generate 5 synthetic M4 participants for GPU smoke test.

    Returns a DataFrame matching the format expected by
    ``prepare_participant_data(data, pid, model='wmrl_m4')``.
    """
    rows = []
    for pid_idx in range(5):
        pid = f"SYNTH_{pid_idx:03d}"
        for block_idx in range(3):
            stim, act, rew, ss = simulate_wmrl_block(
                alpha_pos=0.5,
                alpha_neg=0.3,
                phi=0.2,
                rho=0.7,
                capacity=4.0,
                epsilon=0.1,
                beta=50.0,
                n_trials=30,
                n_stim=3,
                n_act=3,
                set_size=3,
                seed=42 + pid_idx * 100 + block_idx,
            )
            rts = np.random.default_rng(
                seed=42 + pid_idx * 100 + block_idx
            ).uniform(0.3, 0.8, size=len(stim))
            for t in range(len(stim)):
                rows.append(
                    {
                        "sona_id": pid,
                        "block": block_idx + 1,
                        "stimulus": int(stim[t]),
                        "key_press": int(act[t]),
                        "reward": float(rew[t]),
                        "set_size": int(ss[t]),
                        "rt": float(rts[t]),
                    }
                )
    return pd.DataFrame(rows)
