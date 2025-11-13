# RL Model Reference

Complete documentation for the reinforcement learning models implemented for the RLWM task.

## Overview

Two models are implemented:
1. **Q-Learning**: Standard model-free RL baseline
2. **WM-RL Hybrid**: Combines working memory with Q-learning

Both models can be fitted to human data using hierarchical Bayesian inference with PyMC.

---

## Model 1: Q-Learning

**File**: `models/q_learning.py`

### Mathematical Formulation

#### Q-Value Update

Temporal difference learning rule:

```
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',·) - Q(s,a)]
```

For immediate rewards (γ=0), simplifies to:

```
Q(s,a) ← Q(s,a) + α[r - Q(s,a)]
```

#### Action Selection

Softmax policy:

```
P(a|s) = exp(β·Q(s,a)) / Σ_a' exp(β·Q(s,a'))
```

### Parameters

| Parameter | Symbol | Range | Default | Description |
|-----------|--------|-------|---------|-------------|
| Learning rate | α | [0, 1] | 0.1 | How quickly Q-values update |
| Inverse temperature | β | (0, ∞) | 2.0 | Exploration vs exploitation |
| Discount factor | γ | [0, 1] | 0.0 | Future reward weighting |
| Initial Q-value | Q₀ | ℝ | 0.5 | Optimistic initialization |

### Implementation

#### Class: QLearningAgent

```python
QLearningAgent(
    num_stimuli: int = 6,
    num_actions: int = 3,
    alpha: float = 0.1,
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
agent = create_q_learning_agent(alpha=0.3, beta=3.0, seed=42)

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

**Learning Rate (α)**
- **Low α (0.01-0.1)**: Slow, stable learning. Less sensitive to noise.
- **High α (0.5-0.9)**: Fast learning. More sensitive to recent outcomes.
- Trauma effects: Might alter α (e.g., increased volatility in learning)

**Inverse Temperature (β)**
- **Low β (0.1-1)**: Exploratory. More random choices.
- **High β (5-20)**: Exploitative. Strong preference for best option.
- Related to decision confidence and certainty

---

## Model 2: WM-RL Hybrid

**File**: `models/wm_rl_hybrid.py`

### Mathematical Formulation

#### Components

1. **Working Memory**: Episodic buffer storing recent (stimulus, action, reward) experiences
2. **Q-Learning**: Same as Model 1
3. **Hybrid Decision**: Weighted combination of WM and RL

#### Working Memory

Storage:
- Capacity: K items (typically 2-7)
- FIFO buffer: Oldest items evicted when capacity exceeded
- Memory strength: Decays with age

```
strength(mem) = exp(-λ · age)
```

Retrieval:
- If stimulus in WM: Retrieve associated action from most recent positive-reward memory
- Otherwise: Uniform probabilities

WM probabilities:
```
P_wm(a|s) = strength · I(a = retrieved_action) + (1 - strength) · uniform
```

#### Hybrid Decision

Combine WM and RL systems:

```
P(a|s) = w_wm · P_wm(a|s) + (1 - w_wm) · P_rl(a|s)
```

### Parameters

| Parameter | Symbol | Range | Default | Description |
|-----------|--------|-------|---------|-------------|
| Learning rate | α | [0, 1] | 0.1 | Q-learning update rate |
| Inverse temperature | β | (0, ∞) | 2.0 | RL exploitation |
| WM capacity | K | {1,...,7} | 4 | Max items in memory |
| Decay rate | λ | [0, 1] | 0.1 | Memory decay speed |
| WM weight | w_wm | [0, 1] | 0.5 | WM vs RL weighting |
| Discount factor | γ | [0, 1] | 0.0 | Future rewards |

### Implementation

#### Class: WMRLHybridAgent

```python
WMRLHybridAgent(
    num_stimuli: int = 6,
    num_actions: int = 3,
    alpha: float = 0.1,
    beta: float = 2.0,
    gamma: float = 0.0,
    capacity: int = 4,
    lambda_decay: float = 0.1,
    w_wm: float = 0.5,
    q_init: float = 0.5,
    seed: Optional[int] = None
)
```

#### Key Methods

**choose_action(stimulus)**
```python
action, info = agent.choose_action(stimulus)
# Returns: (action, decision_info_dict)
```

Decision info includes:
- `probs`: Hybrid probabilities
- `wm_probs`: WM component probabilities
- `rl_probs`: RL component probabilities
- `wm_retrieved`: Whether WM retrieval succeeded
- `wm_strength`: Strength of retrieved memory

**update(stimulus, action, reward, next_stimulus)**
```python
agent.update(stimulus, action, reward, next_stimulus)
# Updates both Q-table and WM buffer
```

**get_wm_contents()**
```python
contents = agent.get_wm_contents()
# Returns: List of memory items with stimulus, action, reward, age, strength
```

### Usage Example

```python
from environments.rlwm_env import create_rlwm_env
from models.wm_rl_hybrid import create_wm_rl_agent

# Create environment and agent
env = create_rlwm_env(set_size=5, seed=42)  # Larger set size
agent = create_wm_rl_agent(
    alpha=0.2,
    beta=3.0,
    capacity=4,
    lambda_decay=0.1,
    w_wm=0.6,
    seed=42
)

# Run episode
obs, info = env.reset()
agent.reset()

for trial in range(100):
    stimulus = obs['stimulus']

    # Choose action (get decision info)
    action, decision_info = agent.choose_action(stimulus)

    # Environment step
    obs, reward, terminated, truncated, info = env.step(action)

    # Update agent
    next_stimulus = obs['stimulus'] if not truncated else None
    agent.update(stimulus, action, reward, next_stimulus)

    # Track WM retrieval
    if decision_info['wm_retrieved']:
        print(f"Trial {trial}: WM retrieval (strength={decision_info['wm_strength']:.2f})")

    if truncated:
        break

# Inspect WM buffer
print("\nFinal WM contents:")
for mem in agent.get_wm_contents():
    print(f"  Stim {mem['stimulus']}: action={mem['action']}, "
          f"reward={mem['reward']}, age={mem['age']}, strength={mem['strength']:.3f}")
```

### Parameter Interpretation

**WM Capacity (K)**
- Determines how many stimulus-response pairs can be held
- Set-size effects: Performance degrades when set_size > capacity
- Individual differences: Lower capacity → stronger set-size effects
- Trauma effects: WM capacity deficits predicted

**Decay Rate (λ)**
- **Low λ (0.0-0.1)**: Slow forgetting. Memories persist.
- **High λ (0.3-1.0)**: Fast forgetting. Only recent memories useful.
- Affects how quickly old stimulus-response associations fade

**WM Weight (w_wm)**
- **w_wm ≈ 0**: Pure RL (ignores WM)
- **w_wm ≈ 0.5**: Balanced hybrid
- **w_wm ≈ 1**: Pure WM (ignores RL)
- Task-dependent: Low set size → higher w_wm useful

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

### Hierarchical Structure

Both models use hierarchical Bayesian estimation:

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

### Fitting Script

```bash
# Fit Q-learning model
python fitting/fit_to_data.py --model qlearning --chains 4 --samples 2000

# Fit WM-RL hybrid model
python fitting/fit_to_data.py --model wmrl --chains 4 --samples 2000

# Fit both and compare
python fitting/fit_to_data.py --model both
```

### PyMC Model Building

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
