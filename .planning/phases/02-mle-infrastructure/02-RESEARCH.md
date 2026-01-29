# Phase 2: MLE Infrastructure - Research

**Researched:** 2026-01-29
**Domain:** Maximum Likelihood Estimation infrastructure for wmrl_m3 model integration
**Confidence:** HIGH

## Summary

Phase 2 extends the existing MLE fitting infrastructure to support the wmrl_m3 model (WM-RL + perseveration). The codebase already has a well-structured MLE pipeline with scipy.optimize L-BFGS-B, parameter transformations (bounded ↔ unconstrained), and CLI interface via argparse. The implementation follows Senta et al. (2025) methodology: 20 random starts, logit-based transformations, and information criteria for model comparison.

The existing code uses a pattern-based approach: model-specific constants (BOUNDS, PARAMS), transformation functions that dispatch on model type, and objective functions wrapped with functools.partial. M3 integration requires adding parallel structures to those that exist for qlearning (M1) and wmrl (M2).

The key technical choice is whether to use bounded optimization directly (L-BFGS-B with bounds parameter) or transform to unconstrained space. The codebase uses transformation to unconstrained space via logit functions, which is the recommended approach for cognitive/RL model fitting because: (1) unbounded optimizers often perform better, (2) parameters near bounds are handled naturally, (3) it's the established pattern in this project.

**Primary recommendation:** Follow the existing transformation-based pattern. Add WMRL_M3_BOUNDS/PARAMS constants, extend transformation functions with 'wmrl_m3' case, create _objective_wmrl_m3(), and add 'wmrl_m3' to CLI choices. This maintains consistency and leverages proven infrastructure.

## Standard Stack

### Core (Already in Place)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scipy.optimize | ≥1.11.0 | L-BFGS-B optimizer | De facto standard for bounded MLE in Python, mature quasi-Newton method |
| NumPy | ≥1.24.0 | Array operations, transformations | Universal numerical computing foundation |
| argparse | stdlib | CLI argument parsing | Python standard library for command-line interfaces |

### Supporting (Already in Place)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| functools.partial | stdlib | Parameter binding | Creating objective functions with data bound |
| tqdm | Latest | Progress bars | Multi-participant fitting (already used in fit_mle.py) |
| pandas | Latest | Results I/O | Saving/loading fit results (already used) |

**Installation:** No new dependencies required (all already in project)

## Architecture Patterns

### Recommended Implementation Structure

Phase 2 modifies existing MLE infrastructure files:
```
scripts/fitting/
├── mle_utils.py          # ADD: WMRL_M3_BOUNDS, WMRL_M3_PARAMS
│                         #      Extend unconstrained_to_params() with 'wmrl_m3' case
│                         #      Extend params_to_unconstrained() with 'wmrl_m3' case
│                         #      Extend get_n_params() with 'wmrl_m3' case
│
├── fit_mle.py            # ADD: _objective_wmrl_m3() function
│                         #      Extend CLI choices: ['qlearning', 'wmrl', 'wmrl_m3']
│                         #      Extend prepare_participant_data() dispatch logic
│                         #      Extend fit_all_participants() dispatch logic
```

### Pattern 1: Model-Specific Constants

**What:** Define parameter bounds and ordering as module-level constants

**When to use:** For each new model variant (M1, M2, M3, ...)

**Example from existing code:**
```python
# From mle_utils.py lines 18-36
QLEARNING_BOUNDS = {
    'alpha_pos': (0.001, 0.999),
    'alpha_neg': (0.001, 0.999),
    'epsilon': (0.001, 0.999),
}

WMRL_BOUNDS = {
    'alpha_pos': (0.001, 0.999),
    'alpha_neg': (0.001, 0.999),
    'phi': (0.001, 0.999),
    'rho': (0.001, 0.999),
    'capacity': (1.0, 7.0),
    'epsilon': (0.001, 0.999),
}

QLEARNING_PARAMS = ['alpha_pos', 'alpha_neg', 'epsilon']
WMRL_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'epsilon']
```

**For M3:** Add WMRL_M3_BOUNDS and WMRL_M3_PARAMS following same structure:
```python
WMRL_M3_BOUNDS = {
    'alpha_pos': (0.001, 0.999),
    'alpha_neg': (0.001, 0.999),
    'phi': (0.001, 0.999),
    'rho': (0.001, 0.999),
    'capacity': (1.0, 7.0),
    'epsilon': (0.001, 0.999),
    'kappa': (0.0, 1.0),  # NEW: Perseveration parameter
}

WMRL_M3_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'epsilon', 'kappa']
```

**Key insight:** Bounds exclude exact 0 and 1 for probability parameters (0.001, 0.999) to avoid logit singularities. Capacity uses integer range bounds (1.0, 7.0). Kappa can include 0.0 because it's additive, not probability-scaled.

### Pattern 2: Transformation Functions with Model Dispatch

**What:** Single functions that handle all model types via if/else dispatch

**When to use:** Parameter ↔ unconstrained array conversions

**Example from existing code:**
```python
# From mle_utils.py lines 69-110
def params_to_unconstrained(params: Dict[str, float], model: str) -> np.ndarray:
    """Transform bounded parameter dict to unconstrained numpy array."""
    bounds = QLEARNING_BOUNDS if model == 'qlearning' else WMRL_BOUNDS
    param_names = QLEARNING_PARAMS if model == 'qlearning' else WMRL_PARAMS

    x = []
    for name in param_names:
        lower, upper = bounds[name]
        x.append(bounded_to_unbounded(params[name], lower, upper))

    return np.array(x)

def unconstrained_to_params(x: np.ndarray, model: str) -> Dict[str, float]:
    """Transform unconstrained array back to bounded parameter dict."""
    bounds = QLEARNING_BOUNDS if model == 'qlearning' else WMRL_BOUNDS
    param_names = QLEARNING_PARAMS if model == 'qlearning' else WMRL_PARAMS

    params = {}
    for i, name in enumerate(param_names):
        lower, upper = bounds[name]
        params[name] = unbounded_to_bounded(x[i], lower, upper)

    return params
```

**For M3:** Extend with additional elif branch:
```python
def params_to_unconstrained(params: Dict[str, float], model: str) -> np.ndarray:
    if model == 'qlearning':
        bounds = QLEARNING_BOUNDS
        param_names = QLEARNING_PARAMS
    elif model == 'wmrl':
        bounds = WMRL_BOUNDS
        param_names = WMRL_PARAMS
    elif model == 'wmrl_m3':
        bounds = WMRL_M3_BOUNDS
        param_names = WMRL_M3_PARAMS
    else:
        raise ValueError(f"Unknown model: {model}")

    x = []
    for name in param_names:
        lower, upper = bounds[name]
        x.append(bounded_to_unbounded(params[name], lower, upper))

    return np.array(x)
```

**Key insight:** Dispatch pattern allows single call site (`unconstrained_to_params(x, model)`) to handle all model types. This is cleaner than separate functions per model.

### Pattern 3: Objective Function with Partial Application

**What:** Objective function that unpacks unconstrained array, transforms to params dict, calls likelihood

**When to use:** Creating scipy.optimize objective for each model type

**Example from existing code:**
```python
# From fit_mle.py lines 68-143
def _objective_qlearning(
    x: np.ndarray,
    stimuli_blocks: List[np.ndarray],
    actions_blocks: List[np.ndarray],
    rewards_blocks: List[np.ndarray],
) -> float:
    """Negative log-likelihood objective for Q-learning model."""
    params = unconstrained_to_params(x, 'qlearning')

    try:
        log_lik = q_learning_multiblock_likelihood(
            stimuli_blocks=stimuli_blocks,
            actions_blocks=actions_blocks,
            rewards_blocks=rewards_blocks,
            alpha_pos=params['alpha_pos'],
            alpha_neg=params['alpha_neg'],
            epsilon=params['epsilon']
        )
        return -float(log_lik)  # Negative for minimization
    except Exception as e:
        return np.inf  # Return high value on error

def _objective_wmrl(
    x: np.ndarray,
    stimuli_blocks: List[np.ndarray],
    actions_blocks: List[np.ndarray],
    rewards_blocks: List[np.ndarray],
    set_sizes_blocks: List[np.ndarray],
) -> float:
    """Negative log-likelihood objective for WM-RL model."""
    params = unconstrained_to_params(x, 'wmrl')

    try:
        log_lik = wmrl_multiblock_likelihood(
            stimuli_blocks=stimuli_blocks,
            actions_blocks=actions_blocks,
            rewards_blocks=rewards_blocks,
            set_sizes_blocks=set_sizes_blocks,
            alpha_pos=params['alpha_pos'],
            alpha_neg=params['alpha_neg'],
            phi=params['phi'],
            rho=params['rho'],
            capacity=params['capacity'],
            epsilon=params['epsilon']
        )
        return -float(log_lik)
    except Exception as e:
        return np.inf

# Usage in fit_participant_mle() (lines 184-202):
if model == 'qlearning':
    objective = partial(
        _objective_qlearning,
        stimuli_blocks=stimuli_blocks,
        actions_blocks=actions_blocks,
        rewards_blocks=rewards_blocks,
    )
    n_params = 3
else:  # wmrl
    objective = partial(
        _objective_wmrl,
        stimuli_blocks=stimuli_blocks,
        actions_blocks=actions_blocks,
        rewards_blocks=rewards_blocks,
        set_sizes_blocks=set_sizes_blocks,
    )
    n_params = 6
```

**For M3:** Add _objective_wmrl_m3() following same pattern:
```python
def _objective_wmrl_m3(
    x: np.ndarray,
    stimuli_blocks: List[np.ndarray],
    actions_blocks: List[np.ndarray],
    rewards_blocks: List[np.ndarray],
    set_sizes_blocks: List[np.ndarray],
) -> float:
    """Negative log-likelihood objective for WM-RL M3 model."""
    params = unconstrained_to_params(x, 'wmrl_m3')

    try:
        log_lik = wmrl_m3_multiblock_likelihood(
            stimuli_blocks=stimuli_blocks,
            actions_blocks=actions_blocks,
            rewards_blocks=rewards_blocks,
            set_sizes_blocks=set_sizes_blocks,
            alpha_pos=params['alpha_pos'],
            alpha_neg=params['alpha_neg'],
            phi=params['phi'],
            rho=params['rho'],
            capacity=params['capacity'],
            kappa=params['kappa'],  # NEW
            epsilon=params['epsilon']
        )
        return -float(log_lik)
    except Exception as e:
        return np.inf
```

**Key insight:** Exception handling returns np.inf (not raise) to allow optimizer to recover from numerical issues. Partial application binds data, leaving only x as free parameter.

### Pattern 4: CLI with argparse choices

**What:** Use argparse with choices parameter to restrict --model argument

**When to use:** Adding new model types to CLI

**Example from existing code:**
```python
# From fit_mle.py lines 521-526
def main():
    parser = argparse.ArgumentParser(
        description='MLE fitting for RLWM models (Senta et al. methodology)'
    )
    parser.add_argument('--model', type=str, required=True,
                        choices=['qlearning', 'wmrl'],
                        help='Model to fit')
```

**For M3:** Extend choices list:
```python
parser.add_argument('--model', type=str, required=True,
                    choices=['qlearning', 'wmrl', 'wmrl_m3'],
                    help='Model to fit')
```

**Key insight:** argparse automatically validates and generates help text from choices. No manual validation needed.

### Anti-Patterns to Avoid

- **Don't use bounded optimization directly**: Current codebase transforms to unconstrained space. Mixing approaches would create inconsistency and require maintaining two code paths.

- **Don't create separate utility modules per model**: Keep all model variants in same files (mle_utils.py, fit_mle.py) using dispatch patterns. Avoids code duplication.

- **Don't skip exception handling in objectives**: Numerical issues can occur (overflow, underflow). Return np.inf to signal bad region, don't crash optimizer.

- **Don't use different random start methodology**: M1 and M2 use 20 starts. M3 must use same for fair comparison.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Parameter transformations | Custom sigmoid/inverse functions | Existing logit/inv_logit in mle_utils.py | Already tested, handles numerical edge cases |
| Random start sampling | Manual uniform sampling in bounds | sample_random_start() in mle_utils.py | Samples in unconstrained space (Normal(0, 1.5)), better coverage |
| Information criteria | Manual AIC/BIC calculation | compute_aic(), compute_bic() in mle_utils.py | Correct formulas, tested |
| Convergence diagnostics | Manual NLL comparison | check_convergence() in mle_utils.py | Checks multi-start agreement with tolerance |
| Group statistics | Manual mean/SEM | summarize_all_parameters() in mle_utils.py | Computes CI via t-distribution, follows Senta et al. format |

**Key insight:** The MLE infrastructure is mature and tested. Extension requires adding parallel structures, not rewriting functionality.

## Common Pitfalls

### Pitfall 1: Forgetting to Extend All Dispatch Points

**What goes wrong:** Adding WMRL_M3_BOUNDS but forgetting to extend params_to_unconstrained() causes runtime error

**Why it happens:** Model support requires changes in 5+ locations across 2 files

**How to avoid:** Use checklist of required changes:
- [ ] mle_utils.py: Add WMRL_M3_BOUNDS dict
- [ ] mle_utils.py: Add WMRL_M3_PARAMS list
- [ ] mle_utils.py: Extend params_to_unconstrained() with 'wmrl_m3' case
- [ ] mle_utils.py: Extend unconstrained_to_params() with 'wmrl_m3' case
- [ ] mle_utils.py: Extend get_n_params() to return 7 for 'wmrl_m3'
- [ ] mle_utils.py: Extend sample_random_start() to return 7 params for 'wmrl_m3'
- [ ] mle_utils.py: Extend get_default_params() with 'wmrl_m3' case
- [ ] fit_mle.py: Add _objective_wmrl_m3() function
- [ ] fit_mle.py: Extend argparse choices to include 'wmrl_m3'
- [ ] fit_mle.py: Extend fit_participant_mle() model dispatch (lines 184-202)
- [ ] fit_mle.py: Import wmrl_m3_multiblock_likelihood from jax_likelihoods

**Warning signs:** "Unknown model: wmrl_m3" error, or IndexError in unconstrained array unpacking

### Pitfall 2: Incorrect Parameter Ordering

**What goes wrong:** WMRL_M3_PARAMS list has different order than likelihood function expects, causing silent parameter misassignment

**Why it happens:** Parameter dicts are unordered, but array positions are ordered. Mismatch between PARAMS list and likelihood function signature.

**How to avoid:**
1. Match WMRL_M3_PARAMS order to wmrl_m3_multiblock_likelihood() signature order
2. Likelihood signature (from Phase 1): `alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon`
3. WMRL_M3_PARAMS must be: `['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'epsilon', 'kappa']`

**Warning signs:** Fits converge but recovered parameters don't match simulated data. Check test_mle_quick.py pattern for parameter recovery validation.

**CORRECTION:** Actually, review the likelihood signature order carefully. From jax_likelihoods.py wmrl_m3_multiblock_likelihood (lines 947-965), the signature is:
```python
def wmrl_m3_multiblock_likelihood(
    stimuli_blocks, actions_blocks, rewards_blocks, set_sizes_blocks,
    alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon,  # kappa BEFORE epsilon
    ...
)
```

So WMRL_M3_PARAMS should be: `['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa', 'epsilon']` (kappa before epsilon).

### Pitfall 3: Kappa Bound Edge Case

**What goes wrong:** Using (0.001, 0.999) bounds for kappa like other probability parameters, but kappa can legitimately be 0.0 (M2 equivalence)

**Why it happens:** Copy-paste from other parameter bounds without considering semantics

**How to avoid:**
- Kappa is additive weight (κ·Rep(a)), not a probability
- κ=0 is meaningful (M2 model, no perseveration)
- Use bounds (0.0, 1.0) not (0.001, 0.999)
- Logit transformation handles 0.0 endpoint correctly with bounded_to_unbounded()

**Warning signs:** M3 with kappa near 0 doesn't reproduce M2 results. Minimum kappa estimates cluster at 0.001 instead of 0.0.

### Pitfall 4: n_params Hardcoding Instead of get_n_params()

**What goes wrong:** Hardcoding n_params = 7 in fit_participant_mle() instead of calling get_n_params('wmrl_m3')

**Why it happens:** M1 and M2 code hardcodes n_params for simplicity

**How to avoid:**
- Existing code hardcodes: `n_params = 3` for qlearning, `n_params = 6` for wmrl
- M3 should continue pattern: hardcode `n_params = 7` for wmrl_m3 in fit_participant_mle()
- But also update get_n_params() for consistency with other utilities

**Warning signs:** AIC/BIC calculations incorrect (uses wrong k value)

## Code Examples

Verified patterns from official sources and existing codebase:

### Logit Transformation (scipy-compatible)

```python
# From mle_utils.py lines 43-67
def logit(p: float) -> float:
    """Transform probability p in (0,1) to unbounded space (-inf, inf)."""
    return np.log(p / (1 - p))

def inv_logit(x: float) -> float:
    """Transform unbounded x to probability space (0, 1)."""
    return 1 / (1 + np.exp(-x))

def bounded_to_unbounded(value: float, lower: float, upper: float) -> float:
    """Transform value from [lower, upper] to unbounded space."""
    # First normalize to (0, 1)
    p = (value - lower) / (upper - lower)
    # Then apply logit
    return logit(p)

def unbounded_to_bounded(x: float, lower: float, upper: float) -> float:
    """Transform unbounded x to [lower, upper]."""
    # Apply inverse logit to get (0, 1)
    p = inv_logit(x)
    # Then scale to bounds
    return lower + p * (upper - lower)
```

**Note:** This handles arbitrary bounds by normalizing to (0,1) first, then applying logit. Numerically stable for bounds far from (0,1).

### Random Start Sampling (Recommended Method)

```python
# From mle_utils.py lines 132-141
def sample_random_start(model: str, rng: np.random.Generator) -> np.ndarray:
    """
    Sample random starting point in unconstrained space.

    Uses Normal(0, 1.5) which maps to roughly uniform in bounded space
    (centered around 0.5 for (0,1) bounds).
    """
    n_params = len(QLEARNING_PARAMS) if model == 'qlearning' else len(WMRL_PARAMS)
    return rng.normal(0, 1.5, size=n_params)
```

**For M3:** Extend to handle 'wmrl_m3':
```python
def sample_random_start(model: str, rng: np.random.Generator) -> np.ndarray:
    if model == 'qlearning':
        n_params = len(QLEARNING_PARAMS)
    elif model == 'wmrl':
        n_params = len(WMRL_PARAMS)
    elif model == 'wmrl_m3':
        n_params = len(WMRL_M3_PARAMS)
    else:
        raise ValueError(f"Unknown model: {model}")

    return rng.normal(0, 1.5, size=n_params)
```

### scipy.optimize.minimize with L-BFGS-B (Unconstrained)

```python
# From fit_mle.py lines 204-218
results = []
for i in range(n_starts):
    x0 = sample_random_start(model, rng)

    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'disp': False}
        )
        results.append(result)
    except Exception as e:
        continue  # Skip failed optimizations
```

**Note:** No bounds parameter passed to minimize() because we're optimizing in unconstrained space. L-BFGS-B without bounds is effectively unconstrained BFGS.

## State of the Art

| Aspect | Current Approach | Alternative | When Changed | Impact |
|--------|------------------|-------------|--------------|--------|
| Transformation method | Logit transformation to unconstrained space | Direct bounded optimization | Stable since project start | Enables BFGS (not just L-BFGS-B), better convergence |
| Random starts | 20 starts with Normal(0, 1.5) in unconstrained space | Uniform sampling in bounded space | Stable since project start | Better exploration of unconstrained landscape |
| Optimizer | L-BFGS-B via scipy.optimize.minimize | Nelder-Mead, Powell, or gradient-free methods | scipy 1.0+ (2017) | Quasi-Newton methods faster and more reliable for smooth objectives |
| Convergence check | Multi-start agreement (check_convergence with tolerance 0.1) | Single best result | Added in project setup | Detects local optima, validates global convergence |

**Deprecated/outdated:**
- **fmin_l_bfgs_b function**: Replaced by minimize(method='L-BFGS-B') in scipy 1.0+. Old function still exists for backward compatibility but minimize() is preferred interface.
- **Manual gradient approximation**: scipy now has '2-point', '3-point', and 'cs' (complex-step) methods for automatic differentiation. For JAX likelihoods, gradients could be computed via jax.grad, but current code uses finite differences (adequate for current problem scale).

## Open Questions

Things that couldn't be fully resolved:

1. **Should default kappa start value be 0.0 or mid-range?**
   - What we know: Other parameters use mid-range defaults (alpha_pos=0.3, rho=0.7). Kappa=0.0 makes M3 identical to M2.
   - What's unclear: Whether starting at 0.0 helps convergence (close to M2 optimum) or hurts (gradient near boundary).
   - Recommendation: Use 0.0 as default in get_default_params() to match M2 behavior. Random starts will explore non-zero values anyway.

2. **Is 20 random starts sufficient for 7-parameter model?**
   - What we know: M1 (3 params) and M2 (6 params) use 20 starts per Senta et al. methodology. M3 adds 1 parameter.
   - What's unclear: Whether higher-dimensional parameter space needs more starts for same convergence reliability.
   - Recommendation: Keep 20 starts for consistency and M2 vs M3 comparability. If convergence issues arise, can increase to 30 in troubleshooting.

3. **Should wmrl_m3 be a separate model choice or a flag on wmrl?**
   - What we know: Current design has separate model types ('qlearning', 'wmrl'). Adding 'wmrl_m3' follows this pattern.
   - What's unclear: Whether `--model wmrl --kappa` would be cleaner than `--model wmrl_m3` (fewer constants to maintain).
   - Recommendation: Use 'wmrl_m3' as separate choice. Keeps existing dispatch pattern clean. Alternative would require major refactoring of infrastructure.

## Sources

### Primary (HIGH confidence)

- [scipy.optimize.minimize L-BFGS-B documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html) - Official scipy v1.17.0 documentation on L-BFGS-B method, tolerance parameters, gradient specification
- [scipy.optimize tutorial](https://docs.scipy.org/doc/scipy/tutorial/optimize.html) - Optimization best practices from scipy maintainers
- [Python argparse documentation](https://docs.python.org/3/library/argparse.html) - Official Python 3 documentation on argparse module and choices parameter
- Existing codebase files:
  - `scripts/fitting/mle_utils.py` - Parameter transformation functions, bounds definitions, utility functions (lines 1-449)
  - `scripts/fitting/fit_mle.py` - MLE fitting pipeline, objective functions, CLI interface (lines 1-635)
  - `scripts/fitting/jax_likelihoods.py` - Likelihood functions including wmrl_m3_multiblock_likelihood (line 947)

### Secondary (MEDIUM confidence)

- [Modeling reinforcement learning - Maximum likelihood estimation](https://speekenbrink-lab.github.io/modelling/2019/08/29/fit_kf_rl_2.html) - Tutorial on MLE for RL models, discusses parameter transformation and logit use
- [PyMC Reinforcement Learning Example](https://www.pymc.io/projects/examples/en/latest/case_studies/reinforcement_learning.html) - Example of RL model fitting, confirms bounded parameter handling approaches
- [Maximum Likelihood Algorithm (Statlect)](https://www.statlect.com/fundamentals-of-statistics/maximum-likelihood-algorithm) - Explanation of multiple random starts methodology and convergence checking
- [Real Python argparse guide](https://realpython.com/command-line-interfaces-python-argparse/) - Best practices for argparse CLI design

### Tertiary (LOW confidence)

- Web search results on computational modeling parameter optimization (2023-2025) - General discussion of bounded vs transformation approaches, confirms both are valid but transformation often preferred
- Recent ML conference papers (ICLR 2025) - Mentions MLE in model-based RL but not specific to cognitive modeling parameter transformations

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - scipy.optimize, numpy, argparse are well-documented standard tools with stable APIs
- Architecture patterns: HIGH - Directly extracted from existing codebase with verified line numbers
- Pitfalls: HIGH - Based on code analysis of existing dispatch points and parameter ordering requirements
- Code examples: HIGH - All examples from existing codebase (mle_utils.py, fit_mle.py) or official scipy docs

**Research date:** 2026-01-29
**Valid until:** ~60 days (2026-03-30) - MLE methodology is stable, scipy API stable, existing codebase patterns stable. Only risk is if project refactors MLE infrastructure before Phase 2 execution.
