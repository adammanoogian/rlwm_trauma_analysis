# RLWM Environment Reference

Complete API documentation for the RLWM (Reinforcement Learning + Working Memory) gym environment.

## Overview

The RLWM environment is a **trial-based** reinforcement learning environment that simulates the experimental task from the jsPsych implementation. Each `env.step()` represents one complete trial: stimulus presentation → agent response → feedback.

**File**: `environments/rlwm_env.py`

## Environment Specification

### Observation Space

Dictionary space with the following keys:

| Key | Type | Range | Description |
|-----|------|-------|-------------|
| `stimulus` | Discrete(6) | 0-5 | Which stimulus is presented (0-indexed) |
| `set_size` | Box(1,) | 2,3,5,6 | Current set size (# of unique stimuli) |
| `block` | Discrete(24) | 0-23 | Current block number |
| `phase` | Discrete(3) | 0-2 | Phase type: 0=practice_static, 1=practice_dynamic, 2=main |

### Action Space

`Discrete(3)` - Represents the three response keys (J, K, L):
- 0 → J key
- 1 → K key
- 2 → L key

### Reward Structure

- **Correct response**: +1
- **Incorrect response**: 0

Binary rewards match the jsPsych implementation.

### Episode Termination

- **Terminated**: Always `False` (episodes don't naturally end)
- **Truncated**: `True` when `max_trials_per_block` is reached

## Class: RLWMEnv

### Constructor

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

### Methods

#### reset()

```python
reset(seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]
```

Resets the environment for a new block.

**Parameters:**
- `seed`: Random seed
- `options`: Optional dict with keys: `'block'`, `'set_size'`, `'phase_type'`

**Returns:**
- `observation`: Initial observation dict
- `info`: Dictionary with metadata

#### step()

```python
step(action: int) -> Tuple[Dict, float, bool, bool, Dict]
```

Execute one trial.

**Parameters:**
- `action`: Action to take (0, 1, or 2)

**Returns:**
- `observation`: Next observation
- `reward`: Reward received (+1 or 0)
- `terminated`: Always `False`
- `truncated`: `True` if block complete
- `info`: Trial information (see below)

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

Returns performance metrics for current episode:
- `accuracy`: Proportion correct
- `total_reward`: Cumulative reward
- `num_trials`: Number of trials completed
- `num_reversals`: Number of reversals executed

## Factory Function

```python
create_rlwm_env(
    set_size: Optional[int] = None,
    phase_type: str = 'main_task',
    seed: Optional[int] = None,
    **kwargs
) -> RLWMEnv
```

Convenient factory for creating environments with common configurations.

## Usage Examples

### Basic Usage

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

### Multiple Blocks

```python
# Simulate 5 blocks with different set sizes
set_sizes = [2, 3, 5, 6, 3]

for block_idx, set_size in enumerate(set_sizes):
    env = create_rlwm_env(set_size=set_size, seed=42)
    obs, info = env.reset(options={'block': block_idx + 3})

    # Run block
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if truncated:
            metrics = env.get_performance_metrics()
            print(f"Block {block_idx+3}, Set size {set_size}: "
                  f"Accuracy = {metrics['accuracy']:.3f}")
            break
```

### With RL Agent

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

# Check performance
print(f"Final accuracy: {info['accuracy']:.3f}")
```

## Task Configuration

### Using Sequence Files

Load actual experimental sequences from the jsPsych implementation:

```python
from environments.task_config import TaskSequenceLoader

# Load sequence 0
loader = TaskSequenceLoader()
sequence = loader.load_sequence(0)

# Get block 3 configuration
block_config = loader.get_block_config(0, 3)
print(f"Block 3: set_size={block_config['set_size']}, "
      f"stimuli={block_config['stimuli']}")

# Create environment with this configuration
set_sizes = loader.create_block_sequence(0, start_block=3, end_block=23)
```

### Generating Synthetic Configurations

```python
from environments.task_config import TaskConfigGenerator

generator = TaskConfigGenerator(seed=42)

# Generate block configuration
block_config = generator.generate_block_config(
    set_size=5,
    num_trials=100
)

# Generate block sequence
set_sizes = generator.generate_block_sequence(num_blocks=21, seed=42)
print(f"Set sizes: {set_sizes}")
```

## Reversal Logic

### Main Task (phase_type='main_task')

- Reversals are **rare and late**
- Occur after 12-18 consecutive correct responses (sampled uniformly)
- Maximum 1 reversal per stimulus per block
- After reversal, correct response changes to a different action

### Practice Dynamic (phase_type='practice_dynamic')

- Reversals occur after exactly 5 consecutive correct responses
- Used to teach participants about reversals
- Multiple reversals possible

### Practice Static (phase_type='practice_static')

- No reversals occur
- Fixed stimulus-response mappings throughout block

## Configuration Parameters

All task parameters are centralized in `config.py`:

```python
from config import TaskParams

TaskParams.NUM_ACTIONS          # 3
TaskParams.SET_SIZES            # [2, 3, 5, 6]
TaskParams.REVERSAL_MIN         # 12
TaskParams.REVERSAL_MAX         # 18
TaskParams.MAX_REVERSALS_PER_STIM  # 1
TaskParams.REWARD_CORRECT       # 1.0
TaskParams.REWARD_INCORRECT     # 0.0
```

## Testing

Run the built-in test:

```bash
python environments/rlwm_env.py
```

This will create an environment, run 20 trials with random actions, and display trial-by-trial output including reversals.

## Compatibility

The environment follows the Gymnasium (gym) API:
- Observation space: `gym.spaces.Dict`
- Action space: `gym.spaces.Discrete`
- Compatible with standard RL algorithms and libraries

## See Also

- **Model Reference**: `docs/MODEL_REFERENCE.md` - RL agent documentation
- **Analysis Pipeline**: `docs/ANALYSIS_PIPELINE.md` - Full workflow
- **Configuration**: `config.py` - Central parameter configuration
