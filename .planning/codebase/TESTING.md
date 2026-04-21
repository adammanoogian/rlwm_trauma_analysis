# Testing Patterns

**Analysis Date:** 2026-01-28

## Test Framework

**Runner:**
- pytest 6.x+ (specified in `pytest.ini`)
- Config: `pytest.ini` at project root

**Assertion Library:**
- `pytest` built-in assertions for basic checks
- `np.testing.assert_allclose()` for numerical comparisons (numpy)
- Custom assertions for scientific computing accuracy

**Run Commands:**
```bash
pytest validation/                       # Run all validation tests
pytest validation/ -m "not slow"         # Skip slow tests
pytest validation/test_model_consistency.py  # Run specific test file
pytest -v --tb=short --durations=10      # Verbose with timing
pytest --strict-markers                  # Enforce marker validation
```

## Test File Organization

**Location:**
- Tests in `validation/` directory (not `tests/`)
- Fixtures in `validation/conftest.py`
- Co-located with actual implementation (not separate `tests/` folder)

**Naming:**
- `test_*.py` pattern: `test_model_consistency.py`, `test_parameter_recovery.py`, `test_fitting_quick.py`
- Discovery configured in `pytest.ini`: `python_files = test_*.py`

**Structure:**
```
validation/
├── conftest.py                    # Shared fixtures
├── test_fitting_quick.py          # Quick smoke tests
├── test_model_consistency.py      # Unit tests for models
├── test_parameter_recovery.py     # Parameter fitting validation
├── test_pymc_integration.py       # Bayesian integration tests
└── test_unified_simulator.py      # End-to-end simulator tests
```

## Test Structure

**Suite Organization:**
```python
class TestQLearningConsistency:
    """Test Q-learning agent consistency and determinism."""

    def test_deterministic_behavior(self, sample_trial_data, sample_agent_params):
        """Agent with same params and seed produces identical results."""
        # Arrange
        params = sample_agent_params['qlearning']
        agent1 = QLearningAgent(**params, seed=42)

        # Act
        probs1, q_values1 = self._run_agent(agent1, sample_trial_data)

        # Assert
        np.testing.assert_allclose(probs1, probs2, rtol=1e-10)
```

**Patterns:**
- Test classes group related tests: `TestQLearningConsistency`, `TestWMRLConsistency`, `TestCrossModelComparison`
- Setup uses fixtures instead of setUp methods (pytest style)
- Helper methods prefixed with underscore: `_run_agent()`, `_generate_synthetic_data()`
- Tests are single-assertion focused (though some verify multiple properties)

**Teardown:**
- pytest fixtures handle cleanup automatically
- No explicit teardown needed due to fresh agent/env creation per test

## Fixtures and Factories

**Test Data:**
Located in `validation/conftest.py`:

```python
@pytest.fixture
def sample_trial_data():
    """Generate simple trial sequence for testing."""
    return {
        'stimuli': np.array([0, 1, 2, 0, 1, 2]),
        'actions': np.array([0, 1, 2, 1, 0, 2]),
        'rewards': np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    }

@pytest.fixture
def sample_agent_params():
    """Standard agent parameters for testing."""
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
    """Generate realistic participant data."""
    # Creates 50-trial DataFrame for one participant
    # With sona_id, block, trial, stimulus, key_press, correct columns
```

**Location:**
- Fixtures in `validation/conftest.py` (auto-discovered by pytest)
- Shared across all test files
- Provides data, parameters, and directory fixtures

## Mocking

**Framework:** Minimal mocking; tests use real objects

**Patterns:**
- Agents created with test parameters rather than mocked
- Environments created with controlled random seeds: `create_rlwm_env(set_size=3, seed=42)`
- Test data generated rather than mocked: `sample_trial_data` fixture

**What to Mock:**
- None observed in codebase; tests prefer real object behavior
- Rationale: RL model testing requires actual computation to verify learning

**What NOT to Mock:**
- Agent classes (real implementation tested)
- Environments (real trials run for validation)
- NumPy arrays (real computations needed)

## Coverage

**Requirements:** Not enforced (coverage config commented out in `pytest.ini`)

**View Coverage:**
```bash
# Commented out in pytest.ini:
# pytest --cov=models --cov=fitting --cov=environments \
#        --cov-report=html --cov-report=term-missing
```

**Current State:**
- Coverage measurement capability present but not active
- Tests focus on critical model behavior (consistency, parameter recovery)
- Validation tests marked as `@pytest.mark.slow` for parameterized recovery tests

## Test Types

**Unit Tests:**
- Scope: Individual agent methods and environment components
- Approach: Determinism checks, parameter effect validation
- Example: `test_deterministic_behavior()` - verifies same seed produces identical results
- Location: `test_model_consistency.py`, `test_parameter_recovery.py`

**Integration Tests:**
- Scope: Multi-step agent-environment interactions
- Approach: End-to-end trial sequences with real learning dynamics
- Example: `test_wm_capacity_effects()` - tests WM capacity affecting adaptive weighting across trials
- Location: Implicit in consistency tests where agents interact with data

**End-to-End Tests:**
- Scope: Full parameter recovery from synthetic data
- Approach: Generate synthetic data, fit with optimization, verify recovered parameters match truth
- Example: `test_recovery_with_scipy_mle()` - 100-trial synthetic, recovery within 20% error tolerance
- Location: `test_parameter_recovery.py`

**Note:** No separate E2E test framework (like Selenium); all E2E tests use pytest with real Python objects

## Common Patterns

**Async Testing:**
Not used (synchronous RL models, no async operations)

**Error Testing:**
Validation tests verify boundary conditions:

```python
def test_q_value_bounds(self, sample_trial_data, sample_agent_params):
    """Q-values stay within reasonable bounds."""
    agent = QLearningAgent(**params, q_init=0.5)
    _, q_values = self._run_agent(agent, sample_trial_data)

    # With rewards in [0, 1] and gamma=0, Q-values should be in [0, 1]
    assert np.all(q_values >= 0)
    assert np.all(q_values <= 1)

def test_action_prob_validity(self, sample_trial_data, sample_agent_params):
    """Action probabilities are valid probability distributions."""
    agent = QLearningAgent(**params, num_stimuli=6, num_actions=3)
    probs, _ = self._run_agent(agent, sample_trial_data)

    # All probabilities non-negative and sum to 1
    assert np.all(probs >= 0)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-6)
```

**Numerical Precision Testing:**
Strict tolerance for consistency checks, looser for recovery:

```python
# Consistency: exact match (floating point precision)
np.testing.assert_allclose(probs1, probs2, rtol=1e-10)

# Recovery: 20% error acceptable for parameters from 100-trial data
alpha_error = abs(recovered_alpha - true_alpha) / true_alpha
assert alpha_error < 0.2

# Correlation check when rho near zero
correlation = np.corrcoef(probs_q.flatten(), probs_wmrl.flatten())[0, 1]
assert correlation > 0.95
```

## Test Markers

**Configured markers** (from `pytest.ini`):
```
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    requires_pymc: tests that require PyMC installation
```

**Usage examples:**
```python
@pytest.mark.slow
def test_recovery_with_scipy_mle(self):
    """Recover Q-learning parameters using scipy MLE."""
    # Takes significant time

@pytest.mark.slow
def test_recovery_multiple_datasets(self):
    """Test recovery across multiple synthetic datasets."""
    # Outer loop over 5 seeds

@pytest.mark.requires_pymc
def test_pymc_qlearning_model(self):
    """Test PyMC model creation for Q-learning."""
```

## Quick Tests vs. Slow Tests

**Quick Tests** (in `test_model_consistency.py`):
- No external optimization
- Focus: behavior consistency, parameter effects, validity checks
- Runtime: <1 second per test class

**Slow Tests** (marked with `@pytest.mark.slow`):
- Parameter recovery using scipy MLE or PyMC
- Multiple runs (5 datasets in `test_recovery_multiple_datasets`)
- Runtime: 10-30 seconds per test

**Run quick only:**
```bash
pytest -m "not slow"
```

## Test Data Patterns

**Synthetic data generation:**
```python
@staticmethod
def _generate_synthetic_data(env, agent, n_trials):
    """Generate synthetic trial data from agent-environment interaction."""
    obs, _ = env.reset()
    data = []

    for trial in range(n_trials):
        stimulus = obs['stimulus']
        action = agent.choose_action(stimulus)
        obs, reward, terminated, truncated, info = env.step(action)

        data.append({
            'stimulus': stimulus,
            'key_press': action,
            'correct': info['is_correct']
        })

        agent.update(stimulus, action, reward)

        if terminated or truncated:
            break

    return pd.DataFrame(data)
```

**Fixture-based approach:**
- Reusable test data through fixtures
- Deterministic random seed (42) for reproducibility
- Multiple participant counts: 1 (`sample_participant_data`), 3 (`sample_multiparticipant_data`)

## Parameter Space Testing

**Example: Learning rate effects**
```python
def test_parameter_effects_alpha(self, sample_trial_data):
    """Different learning rates produce different behavior."""
    # High learning rates
    agent_high = QLearningAgent(
        alpha_pos=0.9, alpha_neg=0.9, beta=3.0,
        num_stimuli=6, num_actions=3, seed=42
    )
    probs_high, q_high = self._run_agent(agent_high, data)

    # Low learning rates
    agent_low = QLearningAgent(
        alpha_pos=0.1, alpha_neg=0.1, beta=3.0,
        num_stimuli=6, num_actions=3, seed=42
    )
    probs_low, q_low = self._run_agent(agent_low, data)

    # Q-values should differ significantly
    assert not np.allclose(q_high, q_low, rtol=0.1)
```

**Asymmetry testing:**
```python
def test_asymmetric_learning(self, sample_trial_data):
    """Asymmetric learning rates affect behavior differently."""
    # Optimistic (high α_pos, low α_neg)
    agent_optimistic = QLearningAgent(
        alpha_pos=0.9, alpha_neg=0.1, beta=3.0, seed=42
    )
    _, q_optimistic = self._run_agent(agent_optimistic, data)

    # Pessimistic (low α_pos, high α_neg)
    agent_pessimistic = QLearningAgent(
        alpha_pos=0.1, alpha_neg=0.9, beta=3.0, seed=42
    )
    _, q_pessimistic = self._run_agent(agent_pessimistic, data)

    assert not np.allclose(q_optimistic, q_pessimistic, rtol=0.1)
```

## CI/CD Configuration

**Test discovery:**
- pytest looks in `validation/` directory (specified in `pytest.ini`)
- Runs all `test_*.py` files
- Default options: `-v --strict-markers --tb=short --durations=10`

**No CI pipeline detected:**
- No `.github/workflows/`, `.travis.yml`, or similar
- Tests run locally with pytest

---

*Testing analysis: 2026-01-28*
