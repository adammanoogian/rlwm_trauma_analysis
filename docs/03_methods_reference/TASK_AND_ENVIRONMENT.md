# Task and Environment Reference

Complete documentation for the RLWM (Reinforcement Learning + Working Memory) experimental task and gym environment.

---

## 1. Task Overview

The RLWM task is a probabilistic stimulus-response learning paradigm that probes both working memory and reinforcement learning systems. Participants learn stimulus-action mappings through trial-and-error feedback, with varying memory demands created by different set sizes.

### 1.1 Core Design Principles

- **Trial-based structure**: Each trial presents a stimulus; participant responds with one of three keys; feedback is given
- **Set size manipulation**: Number of stimuli varies across blocks (2, 3, 5, or 6) to modulate WM load
- **Rare, late reversals**: Stimulus-response mappings occasionally reverse to assess learning flexibility
- **Binary feedback**: Correct (+1) or incorrect (0) with no negative rewards

### 1.2 Working Memory Load Classification

| Load | Set Sizes | Description |
|------|-----------|-------------|
| **Low** | 2, 3 | Within typical WM capacity; WM-dominant learning expected |
| **High** | 5, 6 | Exceeds typical WM capacity; RL-dominant learning expected |

---

## 2. Task Structure (from Experimental Data)

### 2.1 Block Organization

| Block Type | Block Numbers | Count | Description |
|------------|---------------|-------|-------------|
| Practice Static | 1 | 1 | No reversals; learn basic stimulus-response mappings |
| Practice Dynamic | 2 | 1 | Reversals occur after 5 consecutive correct; teaches reversal detection |
| Main Task | 3-23 | 21 | Full task with rare, late reversals |
| **Total** | 1-23 | **23** | |

### 2.2 Trials per Block

**From actual experimental data:**

| Statistic | Value |
|-----------|-------|
| Mean | 58 trials |
| Median | 45 trials |
| Range | 30-90 trials |

**Distribution:**
- 30 trials: ~29% of blocks
- 45 trials: ~24% of blocks
- 75 trials: ~29% of blocks
- 90 trials: ~18% of blocks

**For simulations:**
- **Default**: 45 trials (median, good balance)
- **Quick tests**: 30 trials (minimum)
- **Full envelope**: 100 trials (covers max + buffer)

### 2.3 Set Sizes

| Set Size | WM Load | Stimuli per Block |
|----------|---------|-------------------|
| 2 | Low | 2 unique stimuli |
| 3 | Low | 3 unique stimuli |
| 4 | **Excluded** | Not used in this experiment |
| 5 | High | 5 unique stimuli |
| 6 | High | 6 unique stimuli |

### 2.4 Reversal Parameters

**Main Task (blocks 3-23):**
- **Reversal criterion**: 12-18 consecutive correct responses (uniformly sampled)
- **Type**: Rare, late reversals
- **Maximum**: 1 reversal per stimulus per block
- **Effect**: Correct response changes to a different action

**Practice Dynamic (block 2):**
- **Criterion**: 5 consecutive correct
- **Required**: Detect 2 reversals to proceed

**Practice Static (block 1):**
- **Reversals**: None
- **Purpose**: Basic learning without complexity

### 2.5 Stimuli and Responses

**Stimuli:**
- Maximum unique stimuli: 6
- Active per block: Depends on set size (2, 3, 5, or 6)

**Responses:**
- 3 action choices mapped to keyboard keys:
  - 0 → J key
  - 1 → K key
  - 2 → L key

### 2.6 Timing Parameters

*Note: Timing is recorded but not modeled in fitting*

| Phase | Duration |
|-------|----------|
| Fixation | 500 ms |
| Stimulus display | Up to 2000 ms (response timeout) |
| Feedback | 500 ms |
| Total trial | ~3000 ms |

### 2.7 Reward Structure

| Outcome | Reward |
|---------|--------|
| Correct response | +1 |
| Incorrect response | 0 |
| No response (timeout) | 0 |

---

## 3. Environment API

### 3.1 File Location

`environments/rlwm_env.py`

### 3.2 Class: RLWMEnv

#### Constructor

```python
RLWMEnv(
    set_size: Optional[int] = None,
    block_sequence: Optional[list] = None,
    max_trials_per_block: int = 100,
    phase_type: str = 'main_task',
    seed: Optional[int] = None
)
```

**Parameters:**
- `set_size`: Fixed set size for all blocks. If `None`, varies across blocks.
- `block_sequence`: List of set sizes per block. If `None`, uses default task structure.
- `max_trials_per_block`: Maximum trials per block (default: 100)
- `phase_type`: Task phase - `'practice_static'`, `'practice_dynamic'`, or `'main_task'`
- `seed`: Random seed for reproducibility

### 3.3 Observation Space

Dictionary space with the following keys:

| Key | Type | Range | Description |
|-----|------|-------|-------------|
| `stimulus` | Discrete(6) | 0-5 | Which stimulus is presented (0-indexed) |
| `set_size` | Box(1,) | 2,3,5,6 | Current set size (# of unique stimuli) |
| `block` | Discrete(24) | 0-23 | Current block number |
| `phase` | Discrete(3) | 0-2 | Phase type: 0=practice_static, 1=practice_dynamic, 2=main |

### 3.4 Action Space

`Discrete(3)` - Represents the three response keys

### 3.5 Core Methods

#### reset()

```python
reset(seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]
```

Resets the environment for a new block.

**Returns:**
- `observation`: Initial observation dict
- `info`: Dictionary with metadata

#### step()

```python
step(action: int) -> Tuple[Dict, float, bool, bool, Dict]
```

Execute one trial.

**Returns:**
- `observation`: Next observation
- `reward`: Reward received (+1 or 0)
- `terminated`: Always `False`
- `truncated`: `True` if block complete
- `info`: Trial information dict

**Info Dictionary:**
```python
{
    'block': int,              # Current block number
    'trial': int,              # Trial within block
    'set_size': int,           # Set size
    'phase_type': str,         # Phase type
    'stimulus': int,           # Stimulus shown this trial
    'action': int,             # Action taken
    'correct_response': int,   # Correct action
    'is_correct': bool,        # Whether response was correct
    'reversal_occurred': bool, # Whether reversal happened
    'correct_counter': int,    # Consecutive correct for this stimulus
    'accuracy': float,         # Running accuracy in block
    'total_reward': float,     # Cumulative reward in block
}
```

#### get_performance_metrics()

```python
get_performance_metrics() -> Dict[str, float]
```

Returns:
- `accuracy`: Proportion correct
- `total_reward`: Cumulative reward
- `num_trials`: Number of trials completed
- `num_reversals`: Number of reversals executed

### 3.6 Factory Function

```python
from environments.rlwm_env import create_rlwm_env

env = create_rlwm_env(
    set_size: Optional[int] = None,
    phase_type: str = 'main_task',
    seed: Optional[int] = None,
    **kwargs
) -> RLWMEnv
```

---

## 4. Usage Examples

### 4.1 Basic Usage

```python
from environments.rlwm_env import create_rlwm_env

# Create environment with set size 3
env = create_rlwm_env(set_size=3, phase_type='main_task', seed=42)

# Reset
obs, info = env.reset()
print(f"Initial stimulus: {obs['stimulus']}")

# Run 10 trials
for _ in range(10):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Trial {info['trial']}: "
          f"stim={info['stimulus']}, "
          f"action={action}, "
          f"correct={info['correct_response']}, "
          f"reward={reward}")

    if truncated:
        print("Block complete!")
        break
```

### 4.2 Multiple Blocks with Different Set Sizes

```python
set_sizes = [2, 3, 5, 6, 3]

for block_idx, set_size in enumerate(set_sizes):
    env = create_rlwm_env(set_size=set_size, seed=42)
    obs, info = env.reset(options={'block': block_idx + 3})

    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if truncated:
            metrics = env.get_performance_metrics()
            print(f"Block {block_idx+3}, Set size {set_size}: "
                  f"Accuracy = {metrics['accuracy']:.3f}")
            break
```

### 4.3 With RL Agent

```python
from environments.rlwm_env import create_rlwm_env
from models.q_learning import create_q_learning_agent

# Create environment and agent
env = create_rlwm_env(set_size=3, seed=42)
agent = create_q_learning_agent(alpha=0.3, beta=3.0, seed=42)

# Reset
obs, info = env.reset()
agent.reset()

# Run episode
for _ in range(100):
    stimulus = obs['stimulus']
    action = agent.choose_action(stimulus)

    obs, reward, terminated, truncated, info = env.step(action)

    # Update agent
    next_stimulus = obs['stimulus'] if not truncated else None
    agent.update(stimulus, action, reward, next_stimulus)

    if truncated:
        break

print(f"Final accuracy: {info['accuracy']:.3f}")
```

### 4.4 Using Actual Task Sequences

```python
from environments.task_config import TaskSequenceLoader

# Load sequence 0
loader = TaskSequenceLoader()
sequence = loader.load_sequence(0)

# Get block 3 configuration
block_config = loader.get_block_config(0, 3)
print(f"Block 3: set_size={block_config['set_size']}, "
      f"stimuli={block_config['stimuli']}")

# Create block sequence for simulation
set_sizes = loader.create_block_sequence(0, start_block=3, end_block=23)
```

---

## 5. Configuration Reference

All task parameters are centralized in `config.py`:

```python
from config import TaskParams

# Core task parameters
TaskParams.NUM_ACTIONS          # 3
TaskParams.SET_SIZES            # [2, 3, 5, 6]
TaskParams.EXCLUDE_SET_SIZES    # [4]

# Block structure
TaskParams.NUM_PRACTICE_BLOCKS  # 2
TaskParams.NUM_MAIN_BLOCKS      # 21
TaskParams.TOTAL_BLOCKS         # 23

# Reversal parameters
TaskParams.REVERSAL_MIN         # 12
TaskParams.REVERSAL_MAX         # 18
TaskParams.MAX_REVERSALS_PER_STIM  # 1

# Reward structure
TaskParams.REWARD_CORRECT       # 1.0
TaskParams.REWARD_INCORRECT     # 0.0

# Timing (ms, for reference)
TaskParams.FIXATION_DURATION    # 500
TaskParams.TRIAL_DURATION       # 2000
TaskParams.FEEDBACK_DURATION    # 500
```

---

## 6. Summary Table

| Parameter | Value | Source |
|-----------|-------|--------|
| **Block Organization** |||
| Total blocks | 23 | Experimental data |
| Practice blocks | 2 (blocks 1-2) | Experimental data |
| Main task blocks | 21 (blocks 3-23) | Experimental data |
| **Trials per Block** |||
| Mean | 58 trials | Experimental data |
| Median | 45 trials | Experimental data |
| Range | 30-90 trials | Experimental data |
| Simulation default | 45 trials | Median |
| **Set Sizes** |||
| Available | [2, 3, 5, 6] | Experimental data |
| Excluded | [4] | Experimental design |
| Low load | ≤ 3 (sizes 2, 3) | Configuration |
| High load | ≥ 4 (sizes 5, 6) | Configuration |
| **Reversals** |||
| Criterion range | 12-18 consecutive correct | Configuration |
| Type | Rare, late | Configuration |
| Max per stimulus | 1 per block | Configuration |
| **Responses** |||
| Number of actions | 3 (J, K, L) | Configuration |
| Reward correct | +1 | Configuration |
| Reward incorrect | 0 | Configuration |
| Response timeout | 2000ms | Configuration |

---

## 7. Testing

Run the built-in environment test:

```bash
python environments/rlwm_env.py
```

This will create an environment, run 20 trials with random actions, and display trial-by-trial output including reversals.

---

## See Also

- **Model Reference**: `docs/MODEL_REFERENCE.md` - Model mathematics and fitting
- **Configuration**: `config.py` - Central parameter configuration
