# RL Model Reference

Complete documentation for the reinforcement learning models implemented for the RLWM task.

## Overview

Two models are implemented:
1. **Q-Learning**: Standard model-free RL baseline
2. **WM-RL Hybrid**: Combines working memory with Q-learning

Both models can be fitted to human data using hierarchical Bayesian inference with PyMC.

## Implementation Approaches

Two parallel implementations are available:

1. **Agent-Based (PyMC)**: Uses agent classes from `models/` for simulation and fitting
   - Pros: Clean separation, interpretable, consistent with simulations
   - Cons: Slower sampling (Metropolis), no gradients
   - Files: `models/q_learning.py`, `fitting/pymc_models.py`

2. **Functional (JAX/NumPyro)**: Pure functional likelihoods with JIT compilation
   - Pros: 10-100x faster, gradient-based NUTS sampler, native JAX operations
   - Cons: Separate implementation from agent classes
   - Files: `scripts/fitting/jax_likelihoods.py`, `scripts/fitting/numpyro_models.py`

**Recommendation**: Use JAX/NumPyro for production fitting (faster, better sampling). Use agent-based approach for parameter exploration and validation.

---

## Model 1: Q-Learning

**File**: `models/q_learning.py`

### Mathematical Formulation

#### Q-Value Update

Temporal difference learning rule with **asymmetric learning rates**:

```
Prediction Error (PE) = r - Q(s,a)

If PE > 0:  Q(s,a) ← Q(s,a) + α_pos · PE  (positive PE, correct trials)
If PE ≤ 0:  Q(s,a) ← Q(s,a) + α_neg · PE  (negative PE, incorrect trials)
```

For this task, γ=0 (no bootstrapping), so the update simplifies to learning from immediate rewards only.

#### Action Selection

Softmax policy:

```
P(a|s) = exp(β·Q(s,a)) / Σ_a' exp(β·Q(s,a'))
```

### Parameters

| Parameter | Symbol | Range | Default | Description |
|-----------|--------|-------|---------|-------------|
| Learning rate (positive PE) | α₊ | [0, 1] | 0.3 | How quickly Q-values increase from correct trials |
| Learning rate (negative PE) | α₋ | [0, 1] | 0.1 | How quickly Q-values decrease from incorrect trials |
| Inverse temperature | β | (0, ∞) | 2.0 | Exploration vs exploitation |
| Discount factor | γ | [0, 1] | 0.0 | Future reward weighting (fixed at 0) |
| Initial Q-value | Q₀ | ℝ | 0.5 | Optimistic initialization |

### Implementation

#### Class: QLearningAgent

```python
QLearningAgent(
    num_stimuli: int = 6,
    num_actions: int = 3,
    alpha_pos: float = 0.3,    # Learning rate for positive PE
    alpha_neg: float = 0.1,    # Learning rate for negative PE
    beta: float = 2.0,
    gamma: float = 0.0,
    q_init: float = 0.5,
    seed: Optional[int] = None
)
```

#### Key Methods

**choose_action(stimulus)**
```python
action = agent.choose_action(stimulus)
# Returns: int (0, 1, or 2)
```

**update(stimulus, action, reward, next_stimulus)**
```python
agent.update(stimulus, action, reward, next_stimulus)
# Updates Q-table based on observed reward
```

**get_action_probs(stimulus)**
```python
probs = agent.get_action_probs(stimulus)
# Returns: np.ndarray of shape (3,) with action probabilities
```

### Usage Example

```python
from environments.rlwm_env import create_rlwm_env
from models.q_learning import create_q_learning_agent

# Create environment and agent
env = create_rlwm_env(set_size=3, seed=42)
agent = create_q_learning_agent(alpha_pos=0.3, alpha_neg=0.1, beta=3.0, seed=42)

# Reset
obs, info = env.reset()
agent.reset()

# Run learning episode
for trial in range(100):
    stimulus = obs['stimulus']

    # Agent chooses action
    action = agent.choose_action(stimulus)

    # Environment responds
    obs, reward, terminated, truncated, info = env.step(action)

    # Agent learns
    next_stimulus = obs['stimulus'] if not truncated else None
    agent.update(stimulus, action, reward, next_stimulus)

    if truncated:
        break

# Check learned Q-values
print("Final Q-table:")
print(agent.get_q_table())
```

### Parameter Interpretation

**Asymmetric Learning Rates (α₊, α₋)**
- **α₊ (positive PE)**: Controls how quickly the agent learns from correct responses
  - High α₊: Fast learning from successes
  - Low α₊: Slower, more stable learning from positive outcomes
- **α₋ (negative PE)**: Controls how quickly the agent learns from incorrect responses
  - High α₋: Fast learning from mistakes
  - Low α₋: Less sensitive to negative feedback
- **Asymmetry**: α₊ > α₋ represents optimistic learning (faster from rewards than punishments)
- Trauma effects: May show altered asymmetry (e.g., heightened α₋, reflecting increased sensitivity to negative outcomes)

**Inverse Temperature (β)**
- **Low β (0.1-1)**: Exploratory. More random choices.
- **High β (5-20)**: Exploitative. Strong preference for best option.
- Related to decision confidence and certainty

---

## Model 2: WM-RL Hybrid

**File**: `models/wm_rl_hybrid.py`

### Mathematical Formulation

#### Components

1. **Working Memory**: Matrix-based distributed value storage (one-shot learning)
2. **Q-Learning**: Asymmetric learning rates (same as Model 1)
3. **Hybrid Decision**: Adaptively weighted combination based on capacity and set size

#### Working Memory (Matrix-Based)

**WM Matrix**: State-action value matrix `WM[s, a]` storing immediate reward outcomes.

**Update Rule** (one-shot learning, α=1):
```
WM_{t+1}(s,a) ← r_{t+1}  (immediate overwrite)
```

**Global Decay** (before update, every trial):
```
∀s,a: WM_{t+1}(s,a) ← (1 - φ)·WM_{t+1}(s,a) + φ·WM_0(s,a)
```
Where WM_0 is the baseline value (typically 0.0).

**WM Policy**:
```
P_WM(a|s) = softmax(β_WM · WM(s,:))
P_WM(a|s) = exp(β_WM·WM(s,a)) / Σ_a' exp(β_WM·WM(s,a'))
```

#### RL Component

Same asymmetric Q-learning as Model 1:
```
Q(s,a) ← Q(s,a) + α_pos/neg · [r - Q(s,a)]
```

**RL Policy**:
```
P_RL(a|s) = softmax(β · Q(s,:))
```

#### Adaptive Hybrid Decision

**Adaptive Weighting**:
```
ω = ρ · min(1, K/N_s)
```
Where:
- ρ: Base WM reliance parameter
- K: WM capacity
- N_s: Current set size

**Final Policy**:
```
P(a|s) = ω·P_WM(a|s) + (1 - ω)·P_RL(a|s)
```

The weight ω increases when set size is within capacity (K ≥ N_s), allowing WM to dominate. When set size exceeds capacity, ω decreases, shifting reliance to RL.

### Parameters

| Parameter | Symbol | Range | Default | Description |
|-----------|--------|-------|---------|-------------|
| Learning rate (positive PE) | α₊ | [0, 1] | 0.3 | RL update rate for correct trials |
| Learning rate (negative PE) | α₋ | [0, 1] | 0.1 | RL update rate for incorrect trials |
| RL inverse temperature | β | (0, ∞) | 2.0 | RL exploitation |
| WM inverse temperature | β_WM | (0, ∞) | 3.0 | WM exploitation (typically higher) |
| WM capacity | K | {1,...,7} | 4 | Capacity for adaptive weighting |
| WM decay rate | φ | [0, 1] | 0.1 | Global decay toward baseline |
| Base WM reliance | ρ | [0, 1] | 0.7 | Base WM weight in adaptive formula |
| Discount factor | γ | [0, 1] | 0.0 | Future rewards (fixed at 0) |
| WM baseline | WM_0 | ℝ | 0.0 | Decay target value |

### Implementation

#### Class: WMRLHybridAgent

```python
WMRLHybridAgent(
    num_stimuli: int = 6,
    num_actions: int = 3,
    alpha_pos: float = 0.3,        # RL learning rate (positive PE)
    alpha_neg: float = 0.1,        # RL learning rate (negative PE)
    beta: float = 2.0,             # RL inverse temperature
    beta_wm: float = 3.0,          # WM inverse temperature
    capacity: int = 4,             # WM capacity (K)
    phi: float = 0.1,              # WM global decay rate
    rho: float = 0.7,              # Base WM reliance
    gamma: float = 0.0,
    q_init: float = 0.5,
    wm_init: float = 0.0,          # WM baseline
    seed: Optional[int] = None
)
```

#### Key Methods

**choose_action(stimulus, set_size)**
```python
action = agent.choose_action(stimulus, set_size)
# Returns: int (0, 1, or 2)
# Note: set_size required for adaptive weighting
```

**choose_action(stimulus, set_size, return_info=True)**
```python
action, info = agent.choose_action(stimulus, set_size, return_info=True)
# Returns: (action, decision_info_dict)
```

Decision info includes:
- `probs`: Hybrid probabilities
- `wm_probs`: WM component probabilities
- `rl_probs`: RL component probabilities
- `omega`: Adaptive weight (ω = ρ · min(1, K/N_s))

**update(stimulus, action, reward, next_stimulus)**
```python
agent.update(stimulus, action, reward, next_stimulus)
# Updates both Q-table and WM matrix (with decay)
```

**get_wm_matrix()**
```python
wm_matrix = agent.get_wm_matrix()
# Returns: np.ndarray of shape (num_stimuli, num_actions)
```

**get_hybrid_probs(stimulus, set_size)**
```python
hybrid_info = agent.get_hybrid_probs(stimulus, set_size)
# Returns: Dict with 'probs', 'wm_probs', 'rl_probs', 'omega'
```

### Usage Example

```python
from environments.rlwm_env import create_rlwm_env
from models.wm_rl_hybrid import create_wm_rl_agent

# Create environment and agent
env = create_rlwm_env(set_size=5, seed=42)  # Larger set size
agent = create_wm_rl_agent(
    alpha_pos=0.3,
    alpha_neg=0.1,
    beta=2.0,
    beta_wm=3.0,
    capacity=4,
    phi=0.1,
    rho=0.7,
    seed=42
)

# Run episode
obs, info = env.reset()
agent.reset()

for trial in range(100):
    stimulus = obs['stimulus']
    set_size = int(obs['set_size'].item())  # Extract from observation

    # Choose action (get decision info)
    action, decision_info = agent.choose_action(stimulus, set_size, return_info=True)

    # Environment step
    obs, reward, terminated, truncated, info = env.step(action)

    # Update agent
    next_stimulus = obs['stimulus'] if not truncated else None
    agent.update(stimulus, action, reward, next_stimulus)

    # Track adaptive weighting
    omega = decision_info['omega']
    print(f"Trial {trial}: set_size={set_size}, omega={omega:.2f}, "
          f"WM_weight={omega:.2%}, RL_weight={(1-omega):.2%}")

    if truncated:
        break

# Inspect WM matrix
print("\nFinal WM matrix:")
wm_matrix = agent.get_wm_matrix()
print(wm_matrix)
```

### Parameter Interpretation

**WM Capacity (K)**
- Determines adaptive weighting via ω = ρ · min(1, K/N_s)
- When K ≥ set_size: Full WM reliance possible (ω = ρ)
- When K < set_size: Reduced WM reliance (ω = ρ·K/N_s)
- Individual differences: Lower capacity → stronger set-size effects
- Trauma effects: WM capacity deficits predicted

**WM Decay Rate (φ)**
- **Low φ (0.0-0.1)**: Slow global decay. WM values persist toward rewards.
- **High φ (0.3-1.0)**: Fast decay. WM values quickly return to baseline.
- Controls forgetting rate of all stimulus-action associations
- Different from buffer-based models: Decay is global, not age-based

**Base WM Reliance (ρ)**
- **ρ ≈ 0**: Minimal WM use (RL-dominant)
- **ρ ≈ 0.5**: Moderate WM reliance
- **ρ ≈ 1**: Maximal WM use (WM-dominant when K ≥ N_s)
- Combined with capacity: Final weight ω = ρ · min(1, K/N_s)
- Reflects individual preference for one-shot vs. incremental learning

**WM vs. RL Inverse Temperatures (β_WM, β)**
- **β_WM typically > β**: WM retrieval is more deterministic than RL
- Reflects certainty: WM provides immediate outcome info, RL accumulates noisy evidence
- Separate temperatures allow different exploration strategies per system

---

## Simulation Utilities

### For Q-Learning

```python
from models.q_learning import simulate_agent_on_env

results = simulate_agent_on_env(
    agent=agent,
    env=env,
    num_trials=100,
    log_history=True
)

# Results dict contains:
# - stimuli, actions, rewards, correct
# - accuracy, total_reward, num_trials
```

### For WM-RL Hybrid

```python
from models.wm_rl_hybrid import simulate_wm_rl_on_env

results = simulate_wm_rl_on_env(
    agent=agent,
    env=env,
    num_trials=100,
    log_history=True
)

# Additional in results:
# - wm_retrieved, wm_retrieval_rate
```

---

## Bayesian Model Fitting

### Two Fitting Approaches

#### Approach 1: PyMC (Agent-Based)

Uses agent classes from `models/` directory for simulation within PyMC likelihood.

**Hierarchical Structure**:

**Group Level** (population):
- μ_α, σ_α: Mean and SD of learning rate distribution
- μ_β, σ_β: Mean and SD of inverse temperature distribution
- (WM-RL only: μ_K, σ_K, μ_w, σ_w)

**Individual Level**:
- α_i ~ Normal(μ_α, σ_α) for each participant
- β_i ~ Gamma(shape, rate)
- Bounded transformations ensure valid ranges

**Likelihood**:
```
P(choice_t | model) = Categorical(softmax(Q-values))
```

Trial-by-trial simulation computes action probabilities, then log-likelihood summed over trials.

**Fitting Script**:

```bash
# Fit Q-learning model
python fitting/fit_to_data.py --model qlearning --chains 4 --samples 2000

# Fit WM-RL hybrid model
python fitting/fit_to_data.py --model wmrl --chains 4 --samples 2000

# Fit both and compare
python fitting/fit_to_data.py --model both
```

**PyMC Model Building**:

```python
from fitting.pymc_models import build_qlearning_model
import pymc as pm

# Load data
data = pd.read_csv('output/task_trials_long.csv')

# Build and fit
with build_qlearning_model(data) as model:
    trace = pm.sample(2000, tune=1000, chains=4)

# Analyze posteriors
import arviz as az
summary = az.summary(trace)
print(summary)
```

**Note**: PyMC approach uses Metropolis sampler (no gradients) since agent classes are not differentiable. This is slower but maintains consistency between simulation and fitting.

#### Approach 2: JAX/NumPyro (Functional)

**Files**:
- `scripts/fitting/jax_likelihoods.py` - Pure functional likelihoods
- `scripts/fitting/numpyro_models.py` - Hierarchical Bayesian models
- `scripts/fitting/fit_with_jax.py` - Main fitting script

**Key Advantages**:
- **10-100x faster compilation** via XLA
- **NUTS sampler** (gradient-based, efficient for continuous parameters)
- **JIT compilation** for trial-by-trial operations
- **Block-structured processing** with automatic reset

**Hierarchical Structure** (Q-Learning):

```python
# Group-level priors
μ_α+ ~ Beta(3, 2)           # Mean positive learning rate ~ 0.6
σ_α+ ~ HalfNormal(0.3)      # Variability in α+
μ_α- ~ Beta(2, 3)           # Mean negative learning rate ~ 0.4
σ_α- ~ HalfNormal(0.3)      # Variability in α-
μ_β ~ Gamma(2, 1)           # Mean inverse temperature ~ 2
σ_β ~ HalfNormal(2.0)       # Variability in β

# Individual-level (non-centered parameterization)
z_α+_i ~ Normal(0, 1)
α+_i = logit⁻¹(logit(μ_α+) + σ_α+ * z_α+_i)

z_α-_i ~ Normal(0, 1)
α-_i = logit⁻¹(logit(μ_α-) + σ_α- * z_α-_i)

z_β_i ~ Normal(0, 1)
β_i = exp(log(μ_β) + σ_β * z_β_i)

# Likelihood (computed via JAX scan)
actions_i ~ Softmax(Q-values; α+_i, α-_i, β_i)
```

**WM-RL Additional Parameters**:

```python
# WM-specific group-level priors
μ_β_WM ~ Gamma(3, 1)              # Mean WM inverse temperature ~ 3
σ_β_WM ~ HalfNormal(2.0)
μ_φ ~ Beta(2, 8)                  # Mean WM decay rate ~ 0.2
σ_φ ~ HalfNormal(0.3)
μ_ρ ~ Beta(5, 2)                  # Mean WM reliance ~ 0.7
σ_ρ ~ HalfNormal(0.3)
μ_K ~ TruncatedNormal(4, 1.5, 1, 7)  # Mean WM capacity ~ 4
σ_K ~ HalfNormal(1.0)
```

**JAX Likelihood Functions**:

The core of the JAX implementation uses `jax.lax.scan()` for efficient sequential operations:

```python
from scripts.fitting.jax_likelihoods import q_learning_multiblock_likelihood

# Compute log-likelihood for a participant across all blocks
log_lik = q_learning_multiblock_likelihood(
    stimuli_blocks=[block1_stim, block2_stim, ...],  # List of arrays
    actions_blocks=[block1_act, block2_act, ...],
    rewards_blocks=[block1_rew, block2_rew, ...],
    alpha_pos=0.3,
    alpha_neg=0.1,
    beta=2.0,
    num_stimuli=6,
    num_actions=3,
    q_init=0.5
)
```

**Key Implementation Details**:

1. **Block-aware processing**: Q-values reset at each block boundary (matches experimental design)
2. **Asymmetric learning**: Uses `jnp.where(delta > 0, alpha_pos, alpha_neg)` for conditional updates
3. **Numerical stability**: Softmax uses `exp(beta * (Q - max(Q)))` to prevent overflow
4. **Functional updates**: Uses `.at[].set()` for immutable array updates (JAX requirement)

**Fitting Script**:

```bash
# Basic usage (4 chains, NUTS sampler)
python scripts/fitting/fit_with_jax.py \
    --data output/task_trials_long_all_participants.csv \
    --chains 4 \
    --warmup 1000 \
    --samples 2000 \
    --save-plots

# Quick test (fewer samples)
python scripts/fitting/fit_with_jax.py \
    --data output/task_trials_long.csv \
    --chains 2 \
    --warmup 500 \
    --samples 1000

# Custom output location
python scripts/fitting/fit_with_jax.py \
    --data output/task_trials_long.csv \
    --output output/v1/ \
    --seed 42
```

**Programmatic Usage**:

```python
from scripts.fitting.numpyro_models import (
    qlearning_hierarchical_model,
    prepare_data_for_numpyro,
    run_inference
)
import pandas as pd

# Load and prepare data
data = pd.read_csv('output/task_trials_long.csv')
participant_data = prepare_data_for_numpyro(
    data,
    participant_col='sona_id',
    block_col='block',
    stimulus_col='stimulus',
    action_col='key_press',
    reward_col='reward'
)

# Run MCMC inference
mcmc = run_inference(
    model=qlearning_hierarchical_model,
    model_args={'participant_data': participant_data},
    num_warmup=1000,
    num_samples=2000,
    num_chains=4,
    seed=42
)

# Get samples
samples = mcmc.get_samples()
print(f"μ_α+: {samples['mu_alpha_pos'].mean():.3f}")
print(f"μ_α-: {samples['mu_alpha_neg'].mean():.3f}")
print(f"μ_β: {samples['mu_beta'].mean():.3f}")

# Save results
import arviz as az
idata = az.from_numpyro(mcmc)
idata.to_netcdf('output/v1/posterior.nc')
```

**Performance Comparison**:

| Metric | PyMC (Metropolis) | JAX/NumPyro (NUTS) |
|--------|-------------------|---------------------|
| Compilation time | ~5 min | ~30 sec |
| Sampling (2000 samples, 4 chains) | ~2-4 hours | ~20-40 min |
| Effective sample size | Lower (no gradients) | Higher (gradient-based) |
| Convergence | Slower (random walk) | Faster (Hamiltonian) |
| Divergences | N/A | Rare with good priors |

**When to Use Each**:

- **JAX/NumPyro**: Production fitting, large datasets, need speed
- **PyMC**: Validation, debugging, need agent class consistency

---

## Model Comparison

### Information Criteria

- **WAIC** (Watanabe-Akaike Information Criterion)
- **LOO** (Leave-One-Out Cross-Validation)

Lower values indicate better predictive performance.

### Running Comparison

```python
from fitting.pymc_models import compute_model_comparison

comparison = compute_model_comparison({
    'qlearning': trace_qlearning,
    'wmrl': trace_wmrl
})

print(comparison)
```

Output:
```
          rank    elpd_loo  ...       weight
qlearning    1   -5234.2   ...         0.85
wmrl         2   -5256.8   ...         0.15
```

---

## Parameter Recovery

Test whether fitting procedure can recover known parameters:

1. Generate synthetic data with known parameters
2. Fit model to synthetic data
3. Compare fitted vs. true parameters

```python
# Generate data with known params
from simulations.generate_data import generate_dataset

true_params = [{'alpha': 0.3, 'beta': 3.0, 'gamma': 0.0} for _ in range(50)]
data = generate_dataset(50, 'qlearning', true_params)

# Fit model
from fitting.pymc_models import build_qlearning_model
with build_qlearning_model(data) as model:
    trace = pm.sample(2000)

# Compare posteriors to true values
import arviz as az
az.plot_posterior(trace, var_names=['alpha'], ref_val=0.3)
```

---

## Extensions and Future Models

### Potential Extensions

1. **Decay RL**: Q-values decay over time
   - Q(s,a) ← (1-δ)·Q(s,a) + α[r - Q(s,a)]

2. **Asymmetric Learning**: Different α for positive vs negative outcomes
   - α_+ for rewards, α_- for no-rewards

3. **Perseveration**: Sticky choice (win-stay/lose-shift)
   - P(a|s) ∝ exp(β·Q(s,a) + ρ·I(a = prev_action))

4. **Attention-Weighted RL**: Modulate α by attention
   - α_effective = α · attention(stimulus)

5. **Bayesian RL**: Maintain distributions over Q-values
   - Account for uncertainty explicitly

### Trauma-Specific Hypotheses

- **Altered learning rates**: Trauma → increased α (heightened sensitivity)
- **WM capacity deficits**: Trauma → lower K (PTSD cognitive effects)
- **Exploration changes**: Trauma → altered β (avoidance/hypervigilance)
- **Decay effects**: Trauma → higher λ (memory consolidation deficits)

---

## Testing

Run built-in tests:

```bash
# Test Q-learning
python models/q_learning.py

# Test WM-RL hybrid
python models/wm_rl_hybrid.py
```

---

## See Also

- **Environment Reference**: `docs/ENVIRONMENT_REFERENCE.md`
- **Analysis Pipeline**: `docs/ANALYSIS_PIPELINE.md`
- **Configuration**: `config.py`
