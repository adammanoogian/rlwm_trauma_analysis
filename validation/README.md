# RLWM Tests

Comprehensive test suite for the RLWM trauma analysis project using pytest.

## Test Structure

```
tests/
├── conftest.py                   # Shared fixtures
├── test_model_consistency.py     # Agent class determinism and consistency
├── test_parameter_recovery.py    # Can we recover known parameters?
├── test_pymc_integration.py      # PyMC + agent integration tests
└── README.md                     # This file
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run only fast tests (skip slow parameter recovery)
```bash
pytest -m "not slow"
```

### Run specific test file
```bash
pytest tests/test_model_consistency.py
```

### Run with coverage
```bash
pytest --cov=models --cov=fitting --cov=environments tests/
```

### Run tests in parallel
```bash
pytest -n auto
```

## Test Categories

### Model Consistency Tests (`test_model_consistency.py`)

Tests that agent classes behave deterministically and consistently:
- Agents with same seed produce identical results
- Reset works correctly
- Parameters affect behavior as expected
- Probability distributions are valid
- WM-RL with w_wm=0 matches Q-learning

**Run:** `pytest tests/test_model_consistency.py -v`

### Parameter Recovery Tests (`test_parameter_recovery.py`)

Tests that fitting procedures can recover known parameters from synthetic data:
- Q-learning parameter recovery (scipy MLE)
- WM-RL parameter recovery
- Recovery across multiple datasets
- Identifiability edge cases

**Run:** `pytest tests/test_parameter_recovery.py -v`

⚠️ **These are slow tests!** Use `-m "not slow"` to skip.

### PyMC Integration Tests (`test_pymc_integration.py`)

Tests PyMC models integrate correctly with agent classes:
- Models build without errors
- Models can sample (MCMC)
- Convergence diagnostics work
- Likelihood computation is correct

**Run:** `pytest tests/test_pymc_integration.py -v`

## Markers

Tests are marked with custom markers:

- `@pytest.mark.slow` - Tests that take >10 seconds
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.requires_pymc` - Requires PyMC installation

### Skip slow tests
```bash
pytest -m "not slow"
```

### Run only integration tests
```bash
pytest -m integration
```

## Fixtures

Shared fixtures in `conftest.py`:

- `sample_trial_data` - Simple trial sequence
- `sample_agent_params` - Standard Q-learning and WM-RL parameters
- `sample_participant_data` - One participant's data (50 trials)
- `sample_multiparticipant_data` - Three participants' data
- `project_root` - Project root directory
- `output_dir` - Temporary output directory

## Writing New Tests

### Template

```python
import pytest
import numpy as np

class TestMyFeature:
    """Test description."""

    def test_basic_functionality(self, sample_agent_params):
        """Test that basic functionality works."""
        # Arrange
        params = sample_agent_params['qlearning']

        # Act
        # ... your test code ...

        # Assert
        assert result == expected

    @pytest.mark.slow
    def test_comprehensive_check(self):
        """Slower, more comprehensive test."""
        # ... test code ...
```

### Best Practices

1. **Use fixtures** - Don't repeat data creation
2. **Mark slow tests** - Use `@pytest.mark.slow` for tests >10s
3. **Descriptive names** - Test names should explain what they test
4. **One concept per test** - Each test should test one thing
5. **Use classes** - Group related tests in classes
6. **Add docstrings** - Explain what the test does

## Continuous Integration

To set up CI (GitHub Actions example):

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run fast tests
        run: pytest -m "not slow" --cov=. --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Troubleshooting

### Tests fail with "ModuleNotFoundError"

Make sure project root is in Python path. This is handled automatically by `conftest.py`, but if running individual files:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### PyMC tests skipped

Install PyMC: `pip install pymc arviz`

### Tests are slow

Run only fast tests: `pytest -m "not slow"`

Or run tests in parallel: `pytest -n auto`

### Random test failures

Check that tests use proper seeding:
```python
agent = QLearningAgent(..., seed=42)
env = create_rlwm_env(..., seed=42)
```

## Coverage Goals

Target coverage:
- `models/`: >90%
- `fitting/`: >80%
- `environments/`: >90%
- `simulations/`: >70%

Check coverage:
```bash
pytest --cov=models --cov=fitting --cov=environments --cov=simulations --cov-report=html
open htmlcov/index.html
```

## See Also

- [pytest.ini](../pytest.ini) - Test configuration
- [requirements-dev.txt](../requirements-dev.txt) - Development dependencies
- [docs/ANALYSIS_PIPELINE.md](../docs/ANALYSIS_PIPELINE.md) - Full workflow including testing

## Legacy Files

Files moved to `validation/legacy/` (superseded, retained for git history only):

- `validation/legacy/check_phase_23_1_smoke.py` — Phase 23.1 smoke guard; Phase 23.1 is complete, no active caller.
- `validation/legacy/diagnose_gpu.py` — pre-Phase-21 GPU diagnostic; superseded by cluster-integrated diagnostics.

Deleted (was a self-skipping placeholder with zero test value):

- `validation/test_fitting_quick.py` — legacy fitting test, deleted in Phase 28 cleanup.
