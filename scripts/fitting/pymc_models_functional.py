"""
Functional PyMC Models for RL Parameter Estimation

This module implements Q-learning and WM-RL models using pure functional
PyTensor operations (no mutation) to enable full Bayesian inference with PyMC.

The key difference from pymc_models.py:
- Uses pytensor.scan() for sequential dependencies
- All operations are pure functions (no state mutation)
- Compatible with PyTensor's symbolic computation graph
- Enables gradient-based sampling (NUTS, HMC)

Mathematical Background:
-----------------------
Q-learning has sequential dependencies: Q_t depends on all previous updates.
The standard agent classes use mutation (Q[s,a] += alpha * delta), which
doesn't work with PyTensor symbolic variables.

Solution: Use functional updates where each step returns a NEW Q-table.
PyTensor's scan() handles the sequential iteration over trials.

Author: Generated for RLWM trauma analysis project
Date: 2025-11-21
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from pathlib import Path

try:
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt
    from pytensor.tensor import TensorVariable
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    print("WARNING: PyMC/PyTensor not available")


def softmax(x: TensorVariable, beta: TensorVariable) -> TensorVariable:
    """
    Compute softmax with numerical stability.

    Parameters
    ----------
    x : TensorVariable
        Values to softmax (e.g., Q-values)
    beta : TensorVariable
        Inverse temperature

    Returns
    -------
    TensorVariable
        Softmax probabilities
    """
    # Numerical stability: subtract max before exp
    x_scaled = beta * (x - pt.max(x))
    exp_x = pt.exp(x_scaled)
    return exp_x / pt.sum(exp_x)


def q_learning_step(
    stimulus: TensorVariable,
    action: TensorVariable,
    reward: TensorVariable,
    Q_prev: TensorVariable,
    alpha_pos: TensorVariable,
    alpha_neg: TensorVariable,
    beta: TensorVariable,
    num_actions: int
) -> Tuple[TensorVariable, TensorVariable]:
    """
    Single Q-learning step (functional, no mutation).

    This function represents one trial of Q-learning:
    1. Compute action probabilities from current Q-table
    2. Get log probability of observed action (for likelihood)
    3. Compute Q-update based on prediction error
    4. Return NEW Q-table and log probability

    Parameters
    ----------
    stimulus : TensorVariable (scalar int)
        Current stimulus index
    action : TensorVariable (scalar int)
        Observed action index
    reward : TensorVariable (scalar float)
        Observed reward (0 or 1)
    Q_prev : TensorVariable, shape (num_stimuli, num_actions)
        Q-table from previous trial
    alpha_pos : TensorVariable (scalar)
        Learning rate for positive PEs
    alpha_neg : TensorVariable (scalar)
        Learning rate for negative PEs
    beta : TensorVariable (scalar)
        Inverse temperature
    num_actions : int
        Number of possible actions

    Returns
    -------
    Q_new : TensorVariable
        Updated Q-table
    log_prob : TensorVariable
        Log probability of observed action
    """
    # Get Q-values for current stimulus
    q_vals = Q_prev[stimulus, :]

    # Compute action probabilities (softmax policy)
    probs = softmax(q_vals, beta)

    # Log probability of observed action (for likelihood)
    log_prob = pt.log(probs[action] + 1e-8)  # Add epsilon for numerical stability

    # Compute prediction error
    q_current = Q_prev[stimulus, action]
    delta = reward - q_current

    # Asymmetric learning rate
    alpha = pt.switch(delta > 0, alpha_pos, alpha_neg)

    # Compute Q-value update (functional - creates new value)
    q_updated = q_current + alpha * delta

    # Create new Q-table with updated value
    # Use set_subtensor to create new tensor (no mutation)
    Q_new = pt.set_subtensor(Q_prev[stimulus, action], q_updated)

    return Q_new, log_prob


def compute_qlearning_likelihood(
    stimuli: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    alpha_pos: TensorVariable,
    alpha_neg: TensorVariable,
    beta: TensorVariable,
    num_stimuli: int,
    num_actions: int,
    q_init: float = 0.5
) -> TensorVariable:
    """
    Compute log-likelihood for Q-learning model using pytensor.scan.

    This is the core likelihood function for one participant.
    It uses scan() to iterate over trials, maintaining the Q-table state.

    Parameters
    ----------
    stimuli : array, shape (n_trials,)
        Observed stimulus sequence
    actions : array, shape (n_trials,)
        Observed action sequence
    rewards : array, shape (n_trials,)
        Observed reward sequence
    alpha_pos, alpha_neg, beta : TensorVariable
        Model parameters (can be symbolic)
    num_stimuli : int
        Number of possible stimuli
    num_actions : int
        Number of possible actions
    q_init : float
        Initial Q-value for all state-action pairs

    Returns
    -------
    TensorVariable (scalar)
        Total log-likelihood across all trials
    """
    # Convert numpy arrays to PyTensor tensors
    stimuli_t = pt.as_tensor_variable(stimuli, dtype='int32')
    actions_t = pt.as_tensor_variable(actions, dtype='int32')
    rewards_t = pt.as_tensor_variable(rewards, dtype='float64')

    # Initialize Q-table
    Q_init = pt.ones((num_stimuli, num_actions), dtype='float64') * q_init

    # Use scan to iterate over trials
    # scan returns (outputs, updates_dict)
    # outputs will be a list of [Q_sequence, log_probs]
    # Explicitly list non_sequences to avoid RNG capture
    (Q_sequence, log_probs), _ = pytensor.scan(
        fn=lambda s, a, r, Q_prev, alpha_p, alpha_n, b, n_act: q_learning_step(
            s, a, r, Q_prev, alpha_p, alpha_n, b, n_act
        ),
        sequences=[stimuli_t, actions_t, rewards_t],
        outputs_info=[Q_init, None],  # Q is carried forward, log_prob is output only
        non_sequences=[alpha_pos, alpha_neg, beta, num_actions],
        strict=True
    )

    # Sum log probabilities across trials
    total_log_likelihood = pt.sum(log_probs)

    return total_log_likelihood


def build_qlearning_model_functional(
    data: pd.DataFrame,
    participant_col: str = 'sona_id',
    stimulus_col: str = 'stimulus',
    action_col: str = 'key_press',
    reward_col: str = 'reward',
    num_stimuli: int = 6,
    num_actions: int = 3
) -> pm.Model:
    """
    Build hierarchical Bayesian Q-learning model (functional version).

    This uses the functional likelihood computation instead of agent classes,
    enabling full PyTensor compatibility.

    Parameters
    ----------
    data : pd.DataFrame
        Trial-level data with columns for participant, stimulus, action, reward
    participant_col : str
        Column name for participant IDs
    stimulus_col : str
        Column name for stimulus IDs (0-indexed)
    action_col : str
        Column name for actions (0-indexed)
    reward_col : str
        Column name for rewards (0 or 1)
    num_stimuli : int
        Number of possible stimuli
    num_actions : int
        Number of possible actions

    Returns
    -------
    pm.Model
        PyMC model ready for sampling
    """
    if not PYMC_AVAILABLE:
        raise ImportError("PyMC is required")

    # Get unique participants
    participants = data[participant_col].unique()
    n_participants = len(participants)
    participant_map = {p: i for i, p in enumerate(participants)}

    # Prepare data arrays
    participant_idx = data[participant_col].map(participant_map).values
    stimuli = data[stimulus_col].values.astype(int)
    actions = data[action_col].values.astype(int)
    rewards = data[reward_col].values.astype(float)

    # Group trials by participant
    participant_data = {}
    for pid, pidx in participant_map.items():
        mask = participant_idx == pidx
        participant_data[pidx] = {
            'stimuli': stimuli[mask],
            'actions': actions[mask],
            'rewards': rewards[mask]
        }

    with pm.Model() as model:
        # ====================================================================
        # PRIORS: Group-level (population)
        # ====================================================================

        # Learning rate for positive PE: bounded [0, 1]
        mu_alpha_pos = pm.Beta('mu_alpha_pos', alpha=3, beta=2)  # Mean ~0.6
        sigma_alpha_pos = pm.HalfNormal('sigma_alpha_pos', sigma=0.3)

        # Learning rate for negative PE: bounded [0, 1]
        mu_alpha_neg = pm.Beta('mu_alpha_neg', alpha=2, beta=3)  # Mean ~0.4
        sigma_alpha_neg = pm.HalfNormal('sigma_alpha_neg', sigma=0.3)

        # Inverse temperature (beta): positive
        mu_beta = pm.Gamma('mu_beta', alpha=2, beta=1)  # Mean ~2
        sigma_beta = pm.HalfNormal('sigma_beta', sigma=2.0)

        # ====================================================================
        # PRIORS: Individual-level
        # ====================================================================

        # Non-centered parameterization for better sampling
        alpha_pos_offset = pm.Normal('alpha_pos_offset', mu=0, sigma=1, shape=n_participants)
        alpha_neg_offset = pm.Normal('alpha_neg_offset', mu=0, sigma=1, shape=n_participants)
        beta_offset = pm.Normal('beta_offset', mu=0, sigma=1, shape=n_participants)

        # Transform to constrained space
        alpha_pos = pm.Deterministic(
            'alpha_pos',
            pm.math.invlogit(pm.math.logit(mu_alpha_pos) + sigma_alpha_pos * alpha_pos_offset)
        )
        alpha_neg = pm.Deterministic(
            'alpha_neg',
            pm.math.invlogit(pm.math.logit(mu_alpha_neg) + sigma_alpha_neg * alpha_neg_offset)
        )
        beta = pm.Deterministic(
            'beta',
            pm.math.exp(pm.math.log(mu_beta) + sigma_beta * beta_offset)
        )

        # ====================================================================
        # LIKELIHOOD: Compute for each participant
        # ====================================================================

        for pidx in range(n_participants):
            pdata = participant_data[pidx]

            # Compute log-likelihood for this participant using functional approach
            log_lik = compute_qlearning_likelihood(
                stimuli=pdata['stimuli'],
                actions=pdata['actions'],
                rewards=pdata['rewards'],
                alpha_pos=alpha_pos[pidx],
                alpha_neg=alpha_neg[pidx],
                beta=beta[pidx],
                num_stimuli=num_stimuli,
                num_actions=num_actions,
                q_init=0.5
            )

            # Add to model
            pm.Potential(f'likelihood_p{pidx}', log_lik)

    return model


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def test_functional_likelihood():
    """
    Test the functional likelihood computation on synthetic data.

    This creates a simple dataset and verifies that the likelihood
    can be computed without errors.
    """
    if not PYMC_AVAILABLE:
        print("PyMC not available, skipping test")
        return

    print("Testing functional Q-learning likelihood...")

    # Create synthetic data
    n_trials = 50
    stimuli = np.random.randint(0, 6, n_trials)
    actions = np.random.randint(0, 3, n_trials)
    rewards = np.random.binomial(1, 0.7, n_trials).astype(float)

    # Test parameters
    alpha_pos = 0.3
    alpha_neg = 0.1
    beta = 2.0

    # Compute likelihood
    log_lik = compute_qlearning_likelihood(
        stimuli=stimuli,
        actions=actions,
        rewards=rewards,
        alpha_pos=alpha_pos,
        alpha_neg=alpha_neg,
        beta=beta,
        num_stimuli=6,
        num_actions=3,
        q_init=0.5
    )

    # Evaluate (this should work with concrete values)
    log_lik_value = log_lik.eval()

    print(f"✓ Likelihood computed successfully: {log_lik_value:.2f}")
    print(f"  Average log probability per trial: {log_lik_value / n_trials:.3f}")

    return log_lik_value


if __name__ == "__main__":
    test_functional_likelihood()
