# Simulations

Tools for simulating RLWM agent behavior, generating synthetic data, and exploring parameter space.

## Core Module: `unified_simulator.py`

The unified simulator provides a single implementation for all simulation needs. Both parameter sweeps and PyMC fitting use the same agent classes through this module.

### Two Simulation Modes

#### 1. Fixed Parameters

Simulate with specific parameter values (deterministic):

```python
from simulations.unified_simulator import simulate_agent_fixed
from models.q_learning import QLearningAgent
from environments.rlwm_env import create_rlwm_env

env = create_rlwm_env(set_size=3, seed=42)

result = simulate_agent_fixed(
    agent_class=QLearningAgent,
    params={'alpha': 0.3, 'beta': 2.0, 'gamma': 0.0, 'q_init': 0.5},
    env=env,
    num_trials=100,
    seed=42
)

print(f"Accuracy: {result.accuracy:.2f}")
print(f"Stimuli: {result.stimuli}")
print(f"Actions: {result.actions}")
```

**Use cases:**
- Parameter sweeps
- Testing specific parameter combinations
- Reproducible simulations

#### 2. Sampled Parameters

Simulate with parameters drawn from distributions:

```python
from simulations.unified_simulator import simulate_agent_sampled

def make_env(seed):
    return create_rlwm_env(set_size=3, seed=seed)

results = simulate_agent_sampled(
    agent_class=QLearningAgent,
    param_distributions={
        'alpha': lambda rng: rng.beta(2, 2),      # Sample from Beta(2,2)
        'beta': lambda rng: rng.gamma(2, 1)       # Sample from Gamma(2,1)
    },
    fixed_params={'gamma': 0.0, 'q_init': 0.5},  # Keep these constant
    env_factory=make_env,
    num_trials=100,
    num_samples=50,  # Generate 50 different parameter sets
    seed=42
)

# Results is a list of 50 SimulationResult objects, each with different α,β
accuracies = [r.accuracy for r in results]
print(f"Mean accuracy: {np.mean(accuracies):.2f} ± {np.std(accuracies):.2f}")
```

**Use cases:**
- Generating realistic synthetic data with inter-individual variability
- Prior predictive checks
- Posterior predictive checks
- Power analyses

### PyMC Likelihood Functions

For use inside PyMC models (both use agent classes internally):

```python
from simulations.unified_simulator import (
    simulate_qlearning_for_likelihood,
    simulate_wmrl_for_likelihood
)

# Inside PyMC model
def logp_func(actions, stimuli, rewards, alpha, beta):
    action_probs = simulate_qlearning_for_likelihood(
        stimuli=stimuli,
        rewards=rewards,
        alpha=alpha,
        beta=beta
    )
    return np.sum(np.log(action_probs[np.arange(len(actions)), actions]))
```

## Scripts

### `parameter_sweep.py`

Systematic exploration of parameter space:

```bash
# Q-learning sweep
python simulations/parameter_sweep.py --model qlearning --num-trials 100 --num-reps 5

# WM-RL sweep
python simulations/parameter_sweep.py --model wmrl --num-trials 50 --num-reps 3
```

**In Python:**
```python
from simulations.parameter_sweep import ParameterSweep

sweep = ParameterSweep(model_type='qlearning')
results_df = sweep.sweep_qlearning_parameters(
    alpha_range=[0.1, 0.3, 0.5],
    beta_range=[1, 3, 5],
    set_sizes=[2, 3, 5, 6],
    num_trials=100,
    num_reps=5
)

# Visualize
sweep.plot_accuracy_heatmap(results_df, set_size=3)
```

### `generate_data.py`

Generate synthetic behavioral data:

```bash
# From default parameters with noise
python simulations/generate_data.py --model qlearning --n-participants 50 --add-noise

# From fitted posteriors
python simulations/generate_data.py --model wmrl --posteriors output/v1/wmrl_posterior.nc --n-participants 100
```

**In Python:**
```python
from simulations.generate_data import generate_dataset

data = generate_dataset(
    n_participants=50,
    model_type='qlearning',
    use_posteriors=False,  # Use default distributions
    add_noise=True,
    seed=42
)

data.to_csv('output/v1/simulated_data.csv', index=False)
```

### `interactive_exploration.py`

Real-time parameter exploration with Jupyter widgets:

```python
from simulations.interactive_exploration import (
    explore_qlearning_interactive,
    explore_wmrl_interactive,
    compare_models_interactive
)

# In Jupyter notebook
explore_qlearning_interactive()
```

## Unified Architecture Benefits

1. **Consistency**: Parameter sweeps, PyMC fitting, and data generation all use the same agent code
2. **No duplication**: Agent behavior defined once in `models/`, reused everywhere
3. **Flexibility**: Easy to add new models or modify existing ones
4. **Testable**: Single implementation means comprehensive tests in `tests/test_unified_simulator.py`

## Examples

### Example 1: Compare fixed vs sampled simulation

```python
# Fixed: Everyone has α=0.3, β=2.0
fixed_results = []
for i in range(50):
    env = create_rlwm_env(set_size=3, seed=i)
    result = simulate_agent_fixed(
        QLearningAgent,
        {'alpha': 0.3, 'beta': 2.0, 'gamma': 0.0, 'q_init': 0.5},
        env, 100, seed=i
    )
    fixed_results.append(result.accuracy)

# Sampled: α ~ Beta(2,2), β ~ Gamma(2,1)
sampled_results = simulate_agent_sampled(
    QLearningAgent,
    {'alpha': lambda rng: rng.beta(2, 2), 'beta': lambda rng: rng.gamma(2, 1)},
    {'gamma': 0.0, 'q_init': 0.5},
    lambda s: create_rlwm_env(set_size=3, seed=s),
    100, 50, seed=42
)
sampled_accuracies = [r.accuracy for r in sampled_results]

print(f"Fixed:   {np.mean(fixed_results):.2f} ± {np.std(fixed_results):.2f}")
print(f"Sampled: {np.mean(sampled_accuracies):.2f} ± {np.std(sampled_accuracies):.2f}")
```

### Example 2: Prior predictive check

```python
# Generate data from prior distributions
prior_data = simulate_agent_sampled(
    WMRLHybridAgent,
    {
        'alpha': lambda rng: rng.beta(2, 2),
        'beta': lambda rng: rng.gamma(2, 1),
        'capacity': lambda rng: int(rng.integers(2, 7)),
        'w_wm': lambda rng: rng.beta(2, 2)
    },
    {'lambda_decay': 0.1, 'gamma': 0.0, 'q_init': 0.5},
    lambda s: create_rlwm_env(set_size=3, seed=s),
    num_trials=100,
    num_samples=100,
    seed=42
)

# Check if prior predictions are reasonable
accuracies = [r.accuracy for r in prior_data]
plt.hist(accuracies, bins=20)
plt.xlabel('Accuracy')
plt.ylabel('Count')
plt.title('Prior Predictive: Accuracy Distribution')
```

## See Also

- [ANALYSIS_PIPELINE.md](../docs/ANALYSIS_PIPELINE.md) - Full workflow
- [MODEL_REFERENCE.md](../docs/MODEL_REFERENCE.md) - Agent class details
- [tests/test_unified_simulator.py](../tests/test_unified_simulator.py) - Comprehensive tests
