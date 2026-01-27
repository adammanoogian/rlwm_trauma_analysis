"""Quick test of MLE fitting on synthetic data."""

import numpy as np
import sys
sys.path.insert(0, '.')

# Debug: Test likelihood first
print("Testing JAX likelihood directly...")
import jax.numpy as jnp
from scripts.fitting.jax_likelihoods import q_learning_multiblock_likelihood

stim_test = [jnp.array([0, 1, 0, 1, 2], dtype=jnp.int32)]
act_test = [jnp.array([0, 1, 0, 1, 2], dtype=jnp.int32)]
rew_test = [jnp.array([1., 0., 1., 1., 0.], dtype=jnp.float32)]

# Note: beta is fixed at 50 inside the function, epsilon comes after alpha_neg
ll_test = q_learning_multiblock_likelihood(stim_test, act_test, rew_test, 0.3, 0.1, 0.05)
print(f"Direct likelihood test: {ll_test}")
print(f"Likelihood type: {type(ll_test)}")
print()

from scripts.fitting.fit_mle import fit_participant_mle

# Known parameters
true_alpha_pos = 0.4
true_alpha_neg = 0.15
true_epsilon = 0.05
true_beta = 50.0

# Simulate Q-learning block
def simulate_qlearning_block(alpha_pos, alpha_neg, epsilon, beta, n_trials, n_stim, n_act):
    Q = np.ones((n_stim, n_act)) * 0.5
    np.random.seed(42)
    correct_actions = np.random.randint(0, n_act, n_stim)

    stimuli, actions, rewards = [], [], []

    for t in range(n_trials):
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

    return np.array(stimuli, dtype=np.int32), np.array(actions, dtype=np.int32), np.array(rewards, dtype=np.float32)


if __name__ == '__main__':
    np.random.seed(42)

    # Simulate 3 blocks
    stimuli_blocks, actions_blocks, rewards_blocks = [], [], []
    for i in range(3):
        np.random.seed(42 + i)
        s, a, r = simulate_qlearning_block(true_alpha_pos, true_alpha_neg, true_epsilon, true_beta, 30, 3, 3)
        stimuli_blocks.append(s)
        actions_blocks.append(a)
        rewards_blocks.append(r)

    print('Testing MLE fitting on synthetic data...')
    print(f'True parameters: alpha_pos={true_alpha_pos}, alpha_neg={true_alpha_neg}, epsilon={true_epsilon}')
    print()

    # Fit with MLE
    result = fit_participant_mle(
        stimuli_blocks=stimuli_blocks,
        actions_blocks=actions_blocks,
        rewards_blocks=rewards_blocks,
        model='qlearning',
        n_starts=10,
        seed=123
    )

    print('MLE Results:')
    print(f'  alpha_pos: {result["alpha_pos"]:.3f} (true: {true_alpha_pos})')
    print(f'  alpha_neg: {result["alpha_neg"]:.3f} (true: {true_alpha_neg})')
    print(f'  epsilon:   {result["epsilon"]:.3f} (true: {true_epsilon})')
    print(f'  NLL:       {result["nll"]:.2f}')
    print(f'  Converged: {result["converged"]}')

    # Check recovery
    alpha_pos_error = abs(result['alpha_pos'] - true_alpha_pos)
    alpha_neg_error = abs(result['alpha_neg'] - true_alpha_neg)
    epsilon_error = abs(result['epsilon'] - true_epsilon)

    print()
    if result['converged'] and alpha_pos_error < 0.3:
        print('SUCCESS: MLE fitting is working!')
    else:
        print('Note: Recovery may vary with small data')
