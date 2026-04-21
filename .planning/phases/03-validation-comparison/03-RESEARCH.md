# Phase 3: Validation & Comparison - Research

**Researched:** 2026-01-30
**Domain:** Model validation, information criteria, backward compatibility testing
**Confidence:** HIGH

## Summary

Phase 3 focuses on validating the M3 (WM-RL + perseveration) model implementation and enabling rigorous comparison against the M2 (WM-RL) baseline. This phase ensures the M3 model is scientifically sound and produces trustworthy results for trauma analysis.

The standard approach involves three validation pillars:
1. **Backward compatibility testing**: Verify κ=0 produces numerically identical results to M2
2. **Information criteria computation**: Calculate AIC/BIC with correct parameter counts (7 for M3 vs 6 for M2)
3. **Model comparison utilities**: Enable head-to-head comparison using established statistical frameworks

The codebase already contains robust infrastructure for model comparison (AIC/BIC computation, Akaike weights, per-participant comparisons) used for M1 vs M2 comparison. Phase 3 extends this infrastructure to support M3, adds validation tests, and ensures backward compatibility.

**Primary recommendation:** Extend existing `compare_mle_models.py` to support 3-way comparison (M1/M2/M3), add pytest-based backward compatibility test, verify parameter counting is correct for information criteria.

## Standard Stack

### Core (Already Installed)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest | >=7.0.0 | Testing framework | Industry standard for Python testing, already used in validation/ |
| numpy | Latest | Numerical comparison | Required for allclose() tolerance testing |
| pandas | Latest | Model comparison tables | Used throughout codebase for fit results |
| JAX | Latest | Likelihood computation | Already used for M2/M3 likelihoods |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy.stats | (via scipy) | Statistical tests | Optional: likelihood ratio tests for nested models |
| matplotlib | >=3.7.0 | Visualization | Optional: plot AIC/BIC comparisons |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pytest | unittest | pytest has better fixtures and parametrization, already in requirements-dev.txt |
| AIC/BIC | WAIC/LOO | AIC/BIC simpler for MLE fits, WAIC/LOO require full posterior (Bayesian only) |
| Manual tests | Property-based testing (hypothesis) | Overkill for this phase; simple numerical comparison sufficient |

**Installation:**
No new dependencies required. All tools already in `requirements-dev.txt`.

## Architecture Patterns

### Recommended Project Structure
```
scripts/fitting/
├── jax_likelihoods.py      # Contains backward compatibility test at bottom
├── mle_utils.py            # AIC/BIC computation (already exists)
├── compare_mle_models.py   # Extend to support M3
└── test_m3_validation.py   # NEW: Dedicated pytest test file

validation/
├── test_m3_backward_compat.py  # NEW: Formal backward compatibility tests
└── conftest.py                  # Already has useful fixtures
```

### Pattern 1: Backward Compatibility Test

**What:** Verify M3(κ=0) produces numerically identical likelihood to M2

**When to use:** Critical validation - must pass before M3 is considered valid

**Example:**
```python
# Source: Existing pattern in scripts/fitting/jax_likelihoods.py lines 1213-1252
import jax.numpy as jnp
from scripts.fitting.jax_likelihoods import (
    wmrl_block_likelihood,
    wmrl_m3_block_likelihood
)

def test_wmrl_m3_backward_compatibility():
    """Verify M3 with kappa=0 matches M2 exactly."""
    # Shared test data
    stimuli = jax.random.randint(key, (50,), 0, 6)
    actions = jax.random.randint(key, (50,), 0, 3)
    rewards = jax.random.bernoulli(key, 0.7, (50,)).astype(jnp.float32)
    set_sizes = jnp.ones((50,)) * 5

    params_m2 = {
        'alpha_pos': 0.3, 'alpha_neg': 0.1,
        'phi': 0.1, 'rho': 0.7, 'capacity': 4.0,
        'epsilon': 0.05
    }

    # M2 likelihood
    log_lik_m2 = wmrl_block_likelihood(
        stimuli, actions, rewards, set_sizes, **params_m2
    )

    # M3 with kappa=0 (should match M2 EXACTLY)
    log_lik_m3 = wmrl_m3_block_likelihood(
        stimuli, actions, rewards, set_sizes, **params_m2, kappa=0.0
    )

    # Strict numerical equivalence
    assert jnp.allclose(log_lik_m2, log_lik_m3, rtol=1e-5), \
        f"M2: {log_lik_m2:.6f}, M3(κ=0): {log_lik_m3:.6f}"
```

### Pattern 2: Information Criteria with Correct Parameter Counts

**What:** Compute AIC/BIC ensuring parameter count matches model complexity

**When to use:** Every time M3 fits are compared to M2

**Example:**
```python
# Source: scripts/fitting/mle_utils.py lines 206-236
def compute_aic(nll: float, k: int) -> float:
    """AIC = 2k + 2*NLL"""
    return 2 * k + 2 * nll

def compute_bic(nll: float, k: int, n: int) -> float:
    """BIC = k*log(n) + 2*NLL"""
    return k * np.log(n) + 2 * nll

# Usage for M3
k_m3 = 7  # alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon
aic_m3 = compute_aic(best_nll, k_m3)
bic_m3 = compute_bic(best_nll, k_m3, n_trials)
```

### Pattern 3: Multi-Model Comparison Table

**What:** Generate comparison table showing all models side-by-side

**When to use:** After fitting M1, M2, M3 to same dataset

**Example:**
```python
# Source: Existing pattern in scripts/fitting/compare_mle_models.py
def compare_three_models(m1_fits, m2_fits, m3_fits, metric='aic'):
    """
    Compare M1, M2, M3 using AIC or BIC.

    Returns DataFrame with:
    - Aggregate sum(AIC) or sum(BIC) per model
    - Delta (difference from best model)
    - Akaike weights
    - Per-participant win counts
    """
    # Compute aggregate IC for each model
    agg_m1 = compute_aggregate_ic(m1_fits, metric)
    agg_m2 = compute_aggregate_ic(m2_fits, metric)
    agg_m3 = compute_aggregate_ic(m3_fits, metric)

    # Find best (minimum)
    min_ic = min(agg_m1, agg_m2, agg_m3)

    # Compute deltas and weights
    delta_m1 = agg_m1 - min_ic
    delta_m2 = agg_m2 - min_ic
    delta_m3 = agg_m3 - min_ic

    # Akaike weights
    w_total = (np.exp(-0.5 * delta_m1) +
               np.exp(-0.5 * delta_m2) +
               np.exp(-0.5 * delta_m3))

    return pd.DataFrame({
        'model': ['M1', 'M2', 'M3'],
        f'{metric}': [agg_m1, agg_m2, agg_m3],
        f'delta_{metric}': [delta_m1, delta_m2, delta_m3],
        'weight': [np.exp(-0.5*d)/w_total for d in [delta_m1, delta_m2, delta_m3]]
    })
```

### Anti-Patterns to Avoid

- **Using rtol=1e-2 for backward compatibility**: Too loose! M3(κ=0) should match M2 to machine precision. Use rtol=1e-5 or stricter.
- **Forgetting to test multiblock wrapper**: Testing single-block backward compatibility is insufficient. Test `wmrl_m3_multiblock_likelihood()` as well.
- **Comparing NLL directly without accounting for k**: Raw NLL favors complex models. Always use AIC/BIC which penalize parameters.
- **Computing AIC/BIC per-participant then averaging**: Wrong! Following Senta et al., sum AIC/BIC across participants first, then compare.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Numerical comparison with tolerance | Custom float comparator | `np.allclose(rtol=1e-5)` or `jnp.allclose()` | Handles floating-point edge cases correctly |
| AIC/BIC computation | Manual formula implementation | Existing `mle_utils.py` functions | Already tested, consistent with M1/M2 |
| Model comparison interpretation | Custom thresholds | Burnham & Anderson (2002) guidelines | Standard in literature: Δ<2 weak, Δ>10 strong |
| Test fixtures | Hardcoded data in each test | pytest fixtures in `conftest.py` | DRY principle, consistent test data |

**Key insight:** The codebase already has robust model comparison infrastructure from M1 vs M2. Don't rebuild it - extend it for M3.

## Common Pitfalls

### Pitfall 1: Backward Compatibility Test Runs But Doesn't Validate

**What goes wrong:** Test passes because it runs without errors, but doesn't actually verify numerical equivalence.

**Why it happens:** Using `assert log_lik_m3 is not None` instead of comparing values, or comparing with tolerance too loose.

**How to avoid:**
- Use explicit numerical comparison: `assert jnp.allclose(log_lik_m2, log_lik_m3, rtol=1e-5)`
- Print actual values on failure: Include error message showing both values
- Test on MULTIPLE random seeds to catch edge cases

**Warning signs:**
- Test completes in <1 second (likely not computing anything)
- No actual assertion comparing M2 vs M3 values
- Test passes even when you intentionally break M3 implementation

### Pitfall 2: Wrong Parameter Count in Information Criteria

**What goes wrong:** AIC/BIC computed with wrong k, leading to invalid model comparison.

**Why it happens:** Copy-paste from M2 code without updating k=6 to k=7, or hardcoding k instead of using `get_n_params('wmrl_m3')`.

**How to avoid:**
- Use `get_n_params(model)` function from `mle_utils.py` (line 261)
- Add assertion: `assert k_m3 == 7, "M3 has 7 parameters"`
- Verify in output: print parameter count alongside AIC/BIC

**Warning signs:**
- AIC/BIC for M3 unnaturally close to M2 despite extra parameter
- `k=6` appears in M3 fitting code
- Information criteria don't penalize M3's extra complexity

### Pitfall 3: Comparison Script Only Works for Two Models

**What goes wrong:** Existing `compare_mle_models.py` hardcoded for M1 vs M2, doesn't generalize to 3-way comparison.

**Why it happens:** Script uses `--qlearning` and `--wmrl` flags, assumes only two models.

**How to avoid:**
- Extend to accept `--m1`, `--m2`, `--m3` flags (optional)
- Allow subset comparisons: M2 vs M3, or M1 vs M2 vs M3
- Generalize delta computation and Akaike weight calculation to N models

**Warning signs:**
- Need to write separate comparison script for M2 vs M3
- Hardcoded model names in comparison functions
- Can't add M1 to M2 vs M3 comparison without code changes

### Pitfall 4: Ignoring Convergence Status in Comparisons

**What goes wrong:** Including non-converged fits in model comparison, biasing results.

**Why it happens:** Summing AIC/BIC across all participants without filtering `converged == True`.

**How to avoid:**
- Filter: `converged_df = fits_df[fits_df['converged'] == True]`
- Report convergence rates for each model
- Only compare participants where BOTH models converged

**Warning signs:**
- Different N for M2 and M3 in comparison
- Very high AIC/BIC values (from non-converged fits with NLL=inf)
- Comparison results change dramatically when one participant fails to converge

## Code Examples

Verified patterns from official sources:

### Backward Compatibility Test (pytest version)

```python
# Source: Adapted from scripts/fitting/jax_likelihoods.py:1213-1252
import pytest
import jax
import jax.numpy as jnp
from scripts.fitting.jax_likelihoods import (
    wmrl_block_likelihood,
    wmrl_m3_block_likelihood,
    wmrl_multiblock_likelihood,
    wmrl_m3_multiblock_likelihood,
)

def test_m3_backward_compatibility_single_block():
    """M3 with kappa=0 should match M2 exactly (single block)."""
    key = jax.random.PRNGKey(42)
    n_trials = 50

    # Generate test data
    stimuli = jax.random.randint(key, (n_trials,), 0, 6)
    key, subkey = jax.random.split(key)
    actions = jax.random.randint(subkey, (n_trials,), 0, 3)
    key, subkey = jax.random.split(key)
    rewards = jax.random.bernoulli(subkey, 0.7, (n_trials,)).astype(jnp.float32)
    set_sizes = jnp.ones((n_trials,)) * 5

    # Shared parameters
    params = {
        'alpha_pos': 0.3, 'alpha_neg': 0.1,
        'phi': 0.1, 'rho': 0.7, 'capacity': 4.0,
        'epsilon': 0.05
    }

    # Compute likelihoods
    log_lik_m2 = wmrl_block_likelihood(stimuli, actions, rewards, set_sizes, **params)
    log_lik_m3 = wmrl_m3_block_likelihood(stimuli, actions, rewards, set_sizes, **params, kappa=0.0)

    # Strict numerical equivalence (rtol=1e-5 accounts for floating-point arithmetic)
    assert jnp.allclose(log_lik_m2, log_lik_m3, rtol=1e-5, atol=1e-8), \
        f"Backward compatibility failed: M2={log_lik_m2:.8f}, M3(κ=0)={log_lik_m3:.8f}"


def test_m3_backward_compatibility_multiblock():
    """M3 with kappa=0 should match M2 exactly (multi-block)."""
    key = jax.random.PRNGKey(123)

    # Generate 3 blocks of data
    stimuli_blocks = []
    actions_blocks = []
    rewards_blocks = []
    set_sizes_blocks = []

    for i in range(3):
        key, subkey = jax.random.split(key)
        stimuli_blocks.append(jax.random.randint(subkey, (40,), 0, 6))
        key, subkey = jax.random.split(key)
        actions_blocks.append(jax.random.randint(subkey, (40,), 0, 3))
        key, subkey = jax.random.split(key)
        rewards_blocks.append(jax.random.bernoulli(subkey, 0.7, (40,)).astype(jnp.float32))
        set_sizes_blocks.append(jnp.ones((40,)) * (i + 3))  # Set sizes 3, 4, 5

    params = {
        'alpha_pos': 0.3, 'alpha_neg': 0.1,
        'phi': 0.1, 'rho': 0.7, 'capacity': 4.0,
        'epsilon': 0.05
    }

    # Compute multi-block likelihoods
    log_lik_m2 = wmrl_multiblock_likelihood(
        stimuli_blocks, actions_blocks, rewards_blocks, set_sizes_blocks, **params
    )
    log_lik_m3 = wmrl_m3_multiblock_likelihood(
        stimuli_blocks, actions_blocks, rewards_blocks, set_sizes_blocks, **params, kappa=0.0
    )

    assert jnp.allclose(log_lik_m2, log_lik_m3, rtol=1e-5, atol=1e-8), \
        f"Multi-block backward compatibility failed: M2={log_lik_m2:.8f}, M3(κ=0)={log_lik_m3:.8f}"


@pytest.mark.parametrize("seed", [42, 123, 456, 789, 1024])
def test_m3_backward_compatibility_multiple_seeds(seed):
    """Test backward compatibility across multiple random seeds."""
    key = jax.random.PRNGKey(seed)
    n_trials = 30

    stimuli = jax.random.randint(key, (n_trials,), 0, 6)
    key, subkey = jax.random.split(key)
    actions = jax.random.randint(subkey, (n_trials,), 0, 3)
    key, subkey = jax.random.split(key)
    rewards = jax.random.bernoulli(subkey, 0.6, (n_trials,)).astype(jnp.float32)
    set_sizes = jnp.ones((n_trials,)) * 4

    params = {
        'alpha_pos': 0.25, 'alpha_neg': 0.15,
        'phi': 0.2, 'rho': 0.6, 'capacity': 3.0,
        'epsilon': 0.08
    }

    log_lik_m2 = wmrl_block_likelihood(stimuli, actions, rewards, set_sizes, **params)
    log_lik_m3 = wmrl_m3_block_likelihood(stimuli, actions, rewards, set_sizes, **params, kappa=0.0)

    assert jnp.allclose(log_lik_m2, log_lik_m3, rtol=1e-5, atol=1e-8)
```

### Extended Model Comparison (3-way)

```python
# Source: Adapted from scripts/fitting/compare_mle_models.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

def compare_three_models(
    m1_fits: pd.DataFrame,
    m2_fits: pd.DataFrame,
    m3_fits: pd.DataFrame,
    metric: str = 'aic'
) -> pd.DataFrame:
    """
    Compare three models (M1, M2, M3) using AIC or BIC.

    Args:
        m1_fits: Q-learning individual fits
        m2_fits: WM-RL individual fits
        m3_fits: WM-RL+kappa individual fits
        metric: 'aic' or 'bic'

    Returns:
        DataFrame with comparison results
    """
    # Filter to converged fits only
    m1_conv = m1_fits[m1_fits['converged'] == True]
    m2_conv = m2_fits[m2_fits['converged'] == True]
    m3_conv = m3_fits[m3_fits['converged'] == True]

    # Aggregate IC (sum across participants)
    agg_m1 = m1_conv[metric].sum()
    agg_m2 = m2_conv[metric].sum()
    agg_m3 = m3_conv[metric].sum()

    # Delta from best model
    min_ic = min(agg_m1, agg_m2, agg_m3)
    delta_m1 = agg_m1 - min_ic
    delta_m2 = agg_m2 - min_ic
    delta_m3 = agg_m3 - min_ic

    # Akaike weights
    w_m1 = np.exp(-0.5 * delta_m1)
    w_m2 = np.exp(-0.5 * delta_m2)
    w_m3 = np.exp(-0.5 * delta_m3)
    w_total = w_m1 + w_m2 + w_m3

    # Build comparison table
    comparison = pd.DataFrame({
        'model': ['M1 (Q-learning)', 'M2 (WM-RL)', 'M3 (WM-RL+κ)'],
        'n_params': [3, 6, 7],
        'n_converged': [len(m1_conv), len(m2_conv), len(m3_conv)],
        f'{metric}': [agg_m1, agg_m2, agg_m3],
        f'delta_{metric}': [delta_m1, delta_m2, delta_m3],
        'weight': [w_m1/w_total, w_m2/w_total, w_m3/w_total]
    })

    # Add interpretation
    comparison['evidence'] = comparison[f'delta_{metric}'].apply(
        lambda d: 'Best' if d < 2 else
                  'Weak support' if d < 4 else
                  'Moderate support' if d < 7 else
                  'Strong support' if d < 10 else
                  'Very strong support'
    )

    return comparison


def count_participant_wins(
    fits_dict: Dict[str, pd.DataFrame],
    metric: str = 'aic'
) -> pd.DataFrame:
    """
    Count which model wins for each participant.

    Args:
        fits_dict: {'M1': df1, 'M2': df2, 'M3': df3}
        metric: 'aic' or 'bic'

    Returns:
        DataFrame with per-participant winner
    """
    # Merge on participant_id
    merged = fits_dict['M1'][['participant_id', metric, 'converged']].rename(
        columns={metric: 'aic_m1', 'converged': 'conv_m1'}
    )
    merged = merged.merge(
        fits_dict['M2'][['participant_id', metric, 'converged']].rename(
            columns={metric: 'aic_m2', 'converged': 'conv_m2'}
        ),
        on='participant_id', how='inner'
    )
    merged = merged.merge(
        fits_dict['M3'][['participant_id', metric, 'converged']].rename(
            columns={metric: 'aic_m3', 'converged': 'conv_m3'}
        ),
        on='participant_id', how='inner'
    )

    # Filter to all converged
    all_conv = merged[
        (merged['conv_m1'] == True) &
        (merged['conv_m2'] == True) &
        (merged['conv_m3'] == True)
    ].copy()

    # Determine winner per participant
    def find_winner(row):
        vals = [row['aic_m1'], row['aic_m2'], row['aic_m3']]
        min_idx = np.argmin(vals)
        return ['M1', 'M2', 'M3'][min_idx]

    all_conv['winner'] = all_conv.apply(find_winner, axis=1)

    # Count wins
    win_counts = all_conv['winner'].value_counts()

    return pd.DataFrame({
        'model': ['M1', 'M2', 'M3'],
        'wins': [win_counts.get('M1', 0), win_counts.get('M2', 0), win_counts.get('M3', 0)],
        'percent': [
            100 * win_counts.get('M1', 0) / len(all_conv),
            100 * win_counts.get('M2', 0) / len(all_conv),
            100 * win_counts.get('M3', 0) / len(all_conv)
        ]
    })
```

### Validate Perseveration Effect (Sanity Check)

```python
# Source: New - validates that kappa actually has an effect
import jax
import jax.numpy as jnp
from scripts.fitting.jax_likelihoods import wmrl_m3_block_likelihood

def test_kappa_has_effect():
    """Verify that kappa > 0 produces different likelihood than kappa = 0."""
    key = jax.random.PRNGKey(456)
    n_trials = 50

    # Data with some action repetition
    stimuli = jax.random.randint(key, (n_trials,), 0, 6)
    key, subkey = jax.random.split(key)

    # Create data with deliberate action repetition
    actions = jnp.array([0] * 10 + [1] * 10 + [2] * 10 + [0] * 10 + [1] * 10, dtype=jnp.int32)
    key, subkey = jax.random.split(key)
    rewards = jax.random.bernoulli(subkey, 0.7, (n_trials,)).astype(jnp.float32)
    set_sizes = jnp.ones((n_trials,)) * 5

    params = {
        'alpha_pos': 0.3, 'alpha_neg': 0.1,
        'phi': 0.1, 'rho': 0.7, 'capacity': 4.0,
        'epsilon': 0.05
    }

    # Likelihood with kappa=0 vs kappa=0.5
    log_lik_no_persev = wmrl_m3_block_likelihood(
        stimuli, actions, rewards, set_sizes, **params, kappa=0.0
    )
    log_lik_with_persev = wmrl_m3_block_likelihood(
        stimuli, actions, rewards, set_sizes, **params, kappa=0.5
    )

    # Likelihoods should be DIFFERENT (kappa should have an effect)
    assert not jnp.allclose(log_lik_no_persev, log_lik_with_persev, rtol=1e-3), \
        "kappa parameter has no effect on likelihood!"

    # With positive kappa and action repetition, likelihood should generally increase
    # (not guaranteed for all data, but expected with repetitive actions)
    print(f"kappa=0: {log_lik_no_persev:.4f}, kappa=0.5: {log_lik_with_persev:.4f}")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual AIC/BIC formulas | Centralized `mle_utils.py` functions | Phase 2 (2026-01-29) | Consistency, prevents formula errors |
| Testing in `if __name__ == "__main__"` blocks | pytest with fixtures | Existing codebase | Reproducible, automated testing |
| Two-model comparison only | N-model comparison framework | Phase 3 (needed) | Enables M1 vs M2 vs M3 analysis |
| WAIC/LOO for MLE fits | AIC/BIC only | N/A | WAIC/LOO require full posterior (Bayesian), not applicable to MLE |

**Deprecated/outdated:**
- **Nested model likelihood ratio tests**: While theoretically applicable (M2 nested in M3 when κ=0), AIC/BIC are simpler and already implemented. LRT requires chi-squared approximation which may not hold (see [A Note on Likelihood Ratio Tests for Models with Latent Variables](https://pmc.ncbi.nlm.nih.gov/articles/PMC7826319/)).
- **Cross-validation for model selection**: Not standard in Senta et al. methodology; AIC/BIC sufficient.

## Open Questions

1. **Should we compute likelihood ratio test for M2 vs M3?**
   - What we know: M2 is nested in M3 (when κ=0), so LRT theoretically valid
   - What's unclear: Whether added complexity justifies implementation vs. AIC/BIC
   - Recommendation: Start with AIC/BIC only. Add LRT later if reviewers request it.

2. **What tolerance for backward compatibility?**
   - What we know: rtol=1e-5 used in existing test (jax_likelihoods.py:1246)
   - What's unclear: Whether this is strict enough
   - Recommendation: Use rtol=1e-5, atol=1e-8. If fails, investigate numerical stability issues.

3. **Should validation tests be in `validation/` or `scripts/fitting/`?**
   - What we know: `validation/` has pytest infrastructure, `scripts/fitting/` has `if __name__` tests
   - What's unclear: Best location for M3-specific tests
   - Recommendation: Put backward compat tests in `validation/test_m3_backward_compat.py` (formal), keep quick test in `jax_likelihoods.py:if __name__` (developer convenience).

## Sources

### Primary (HIGH confidence)

- Existing codebase: `scripts/fitting/jax_likelihoods.py:1213-1252` - Backward compatibility test pattern
- Existing codebase: `scripts/fitting/mle_utils.py:206-271` - AIC/BIC computation and parameter counting
- Existing codebase: `scripts/fitting/compare_mle_models.py` - Two-model comparison implementation
- Existing codebase: `validation/conftest.py` - pytest fixture patterns
- Existing codebase: `pytest.ini` - Test configuration
- Existing codebase: `.planning/phases/01-core-implementation/01-VERIFICATION.md` - Verification criteria examples
- Existing codebase: `.planning/phases/02-mle-infrastructure/02-VERIFICATION.md` - Observable truths pattern

### Secondary (MEDIUM confidence)

- [Model selection and psychological theory: AIC vs BIC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3366160/) - AIC/BIC interpretation guidelines
- [Comparing Dynamic Causal Models using AIC, BIC and Free Energy](https://pmc.ncbi.nlm.nih.gov/articles/PMC3200437/) - Model comparison methodology
- [A Note on Likelihood Ratio Tests for Models with Latent Variables](https://pmc.ncbi.nlm.nih.gov/articles/PMC7826319/) - LRT limitations with latent variables
- [Computational Psychiatry Course 2026](https://www.translationalneuromodeling.org/cpcourse/) - Model validation practices

### Tertiary (LOW confidence)

- [Probabilistic Model Selection with AIC, BIC, and MDL](https://machinelearningmastery.com/probabilistic-model-selection-measures/) - General overview (not domain-specific)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All tools already in requirements-dev.txt, verified by reading files
- Architecture: HIGH - Existing codebase provides clear patterns (backward compat test, comparison scripts)
- Pitfalls: HIGH - Identified from actual codebase patterns and common modeling mistakes
- Information criteria: HIGH - Burnham & Anderson (2002) guidelines standard in field

**Research date:** 2026-01-30
**Valid until:** 2026-07-30 (180 days - stable methodological domain)

**Notes:**
- No new dependencies required
- All patterns verified against existing codebase
- Backward compatibility test already exists (jax_likelihoods.py:1213-1252) but needs pytest version
- Model comparison infrastructure already robust, just needs M3 extension
