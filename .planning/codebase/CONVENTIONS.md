# Coding Conventions

**Analysis Date:** 2026-01-28

## Naming Patterns

**Files:**
- snake_case for all Python files: `q_learning.py`, `rlwm_env.py`, `test_model_consistency.py`
- Numbered scripts for data pipeline: `01_parse_raw_data.py`, `02_create_collated_csv.py`, `03_create_task_trials_csv.py`
- Test files follow pattern: `test_*.py` (discovered by pytest)

**Functions:**
- snake_case consistently throughout: `get_action_probs()`, `choose_action()`, `simulate_agent_on_env()`
- Descriptive verbs: `test_deterministic_behavior()`, `parse_single_file()`, `load_participant_mapping()`
- Private methods use leading underscore: `_run_agent()`, `_generate_synthetic_data()`
- Factory functions named with pattern: `create_q_learning_agent()`, `create_wm_rl_agent()`, `create_rlwm_env()`

**Variables:**
- snake_case for all variables: `alpha_pos`, `alpha_neg`, `num_stimuli`, `set_size`
- Greek letters in comments (full names in code): `phi` (not φ), `rho` (not ρ), `beta` (not β)
- Constants in UPPERCASE: `MAX_STIMULI`, `NUM_ACTIONS`, `REWARD_CORRECT`, `EXCLUDED_PARTICIPANTS`
- Class attributes use snake_case: `self.Q`, `self.WM`, `self.rng`, `self.history`
- Loop/temporary variables are explicit: `stimulus`, `action`, `reward` (not `s`, `a`, `r`)

**Types:**
- Classes use PascalCase: `QLearningAgent`, `WMRLHybridAgent`, `RLWMEnv`, `TaskParams`, `ModelParams`
- Configuration classes (settings containers) use PascalCase: `TaskParams`, `ModelParams`, `PyMCParams`, `DataParams`

## Code Style

**Formatting:**
- No automated formatter detected (no `.flake8`, `.pylintrc`, `pyproject.toml`)
- Code follows consistent spacing and indentation throughout
- 79-80 character guideline observed in docstrings and comments
- Imports organized with proper spacing between standard library, third-party, and local imports

**Linting:**
- No linting configuration files present in repository
- Code follows PEP 8 conventions by convention (no automated enforcement)
- No type hints in function signatures despite Python 3.8+ support
- Type hints present only in docstrings (Parameters/Returns sections)

**Comments and Docstrings:**
- Module-level docstrings describe purpose, not implementation
- Use triple-quoted strings: `"""docstring"""`
- Models use docstring sections: Model Equations, Parameters, Notes, Key Differences

## Import Organization

**Order:**
1. Standard library imports: `os`, `sys`, `json`, `subprocess`
2. Third-party imports: `numpy`, `pandas`, `gymnasium`, `pytest`, `scipy`
3. Local imports: `from config import TaskParams`, `from models.q_learning import QLearningAgent`

**Path Aliases:**
- No aliases used (no imports like `import numpy as np` followed by alias usage)
- Imports from `config` module used consistently for configuration values

**Example patterns from codebase:**
```python
# Standard library
import sys
from pathlib import Path

# Third-party
import numpy as np
from typing import Dict, Tuple, Optional, List

# Project root path setup (common pattern)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Local imports
from config import TaskParams, ModelParams
```

## Error Handling

**Patterns:**
- Explicit exception handling for file operations: `except FileNotFoundError as e:`
- General exception handling in scripts: `except Exception as e:`
- Assertions in tests: `assert condition`, `np.testing.assert_allclose()`
- Graceful degradation: check for missing files and log warnings rather than crash
- Error messages are descriptive: `f"Alpha recovery error {alpha_error:.2%} > 20%"`

**Example from `task_config.py`:**
```python
try:
    sequence_file = DATA_DIR / f'sequence_block_{block}.json'
    with open(sequence_file, 'r') as f:
        return json.load(f)
except FileNotFoundError as e:
    logger.error(f"Sequence file not found: {sequence_file}")
    raise
```

## Logging

**Framework:** No centralized logging framework detected; uses `print()` statements for output

**Patterns:**
- Tests use print statements for diagnostics: `print(f"\nParameter Recovery Results:")`
- Quick validation scripts use print for step-by-step feedback
- Modules like `run_data_pipeline.py` print status messages
- No structured logging configuration present

**Example from validation tests:**
```python
print("\nParameter Recovery Results:")
print(f"  True α: {true_alpha:.3f}, Recovered: {recovered_alpha:.3f}")
print(f"  True β: {true_beta:.3f}, Recovered: {recovered_beta:.3f}")
```

## Comments

**When to Comment:**
- Module docstrings describe overall purpose and usage
- Complex algorithms are documented with inline comments (e.g., softmax implementation, WM decay)
- Model equations are shown in docstrings with mathematical notation
- Parameters section documents what each input means
- References to papers (Senta et al., 2025) included for model specifications

**JSDoc/Docstring Style:**
- NumPy docstring format used throughout
- Sections: Description, Model Equations, Parameters, Returns, Notes, Examples
- Parameters documented with type, range, and meaning
- Return values documented with type and description

**Example from `q_learning.py`:**
```python
def update(
    self,
    stimulus: int,
    action: int,
    reward: float,
    next_stimulus: Optional[int] = None,
):
    """
    Update Q-value using asymmetric learning rates.

    With γ=0 (no bootstrapping):
        δ = r - Q(s,a)                    [prediction error]
        α = α_pos if δ > 0 else α_neg    [select learning rate]
        Q(s,a) ← Q(s,a) + α·δ

    Parameters
    ----------
    stimulus : int
        Current stimulus
    action : int
        Action taken
    reward : float
        Reward received (typically 0 or 1)
    """
```

## Function Design

**Size:**
- Methods are focused and single-purpose (10-50 lines typical)
- Longer functions broken into logical sections with separator comments
- Example: `update()` method in both agent classes has clear sections for decay, update, and Q-table operations

**Parameters:**
- Optional parameters use `Optional[Type] = None` pattern
- Factory functions accept `**kwargs` for flexibility: `create_q_learning_agent(..., **kwargs)`
- Boolean flags are explicit: `return_probs: bool = False`, `log_history: bool = True`

**Return Values:**
- Single return preferred; tuples used when returning multiple values: `Tuple[int, Optional[Dict]]`
- Dictionaries returned for complex results: `Dict[str, Any]`
- Static methods clearly marked: `@staticmethod` decorator used

**Example from `wm_rl_hybrid.py`:**
```python
def get_hybrid_probs(self, stimulus: int, set_size: int) -> Dict[str, Any]:
    """
    Compute hybrid action probabilities combining WM and RL.

    Returns
    -------
    dict
        Dictionary with 'probs', 'wm_probs', 'rl_probs', 'omega'
    """
```

## Module Design

**Exports:**
- Public classes: `QLearningAgent`, `WMRLHybridAgent`, `RLWMEnv`
- Public functions: `create_*()` factory functions, `simulate_*_on_env()` utilities
- Test functions: `test_*()` for module-level testing
- Private helpers: `_run_agent()`, `_generate_synthetic_data()`

**Barrel Files:**
- No barrel file pattern observed
- `__init__.py` files are minimal (empty or minimal imports)
- Direct imports from submodules preferred: `from models.q_learning import QLearningAgent`

**Module Structure:**
- Each agent class in separate file: `q_learning.py`, `wm_rl_hybrid.py`
- Configuration centralized in `config.py`
- Environment in separate file: `environments/rlwm_env.py`
- Tests in `validation/` directory matching module names

---

*Convention analysis: 2026-01-28*
