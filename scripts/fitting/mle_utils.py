"""
MLE Utility Functions for RLWM Model Fitting

Parameter transformations, information criteria, and helper functions
for Maximum Likelihood Estimation following Senta et al. (2025) methodology.
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, List, Optional


# =============================================================================
# Parameter Bounds
# =============================================================================

# Q-Learning parameter bounds
QLEARNING_BOUNDS = {
    'alpha_pos': (0.001, 0.999),  # Positive learning rate
    'alpha_neg': (0.001, 0.999),  # Negative learning rate
    'epsilon': (0.001, 0.999),    # Noise parameter
}

# WM-RL parameter bounds
WMRL_BOUNDS = {
    'alpha_pos': (0.001, 0.999),
    'alpha_neg': (0.001, 0.999),
    'phi': (0.001, 0.999),        # WM decay rate
    'rho': (0.001, 0.999),        # Base WM reliance
    'capacity': (1.0, 7.0),       # WM capacity (K)
    'epsilon': (0.001, 0.999),
}

# WM-RL M3 parameter bounds (includes kappa perseveration)
WMRL_M3_BOUNDS = {
    'alpha_pos': (0.001, 0.999),
    'alpha_neg': (0.001, 0.999),
    'phi': (0.001, 0.999),
    'rho': (0.001, 0.999),
    'capacity': (1.0, 7.0),
    'kappa': (0.0, 1.0),      # Perseveration parameter - NOTE: 0.0 allowed (M2 equivalence)
    'epsilon': (0.001, 0.999),
}

# Parameter names in order (for array-dict conversion)
QLEARNING_PARAMS = ['alpha_pos', 'alpha_neg', 'epsilon']
WMRL_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'epsilon']
# CRITICAL: Order must match wmrl_m3_multiblock_likelihood() signature
# Signature: alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon
WMRL_M3_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa', 'epsilon']


# =============================================================================
# Parameter Transformations
# =============================================================================

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


def params_to_unconstrained(params: Dict[str, float], model: str) -> np.ndarray:
    """
    Transform bounded parameter dict to unconstrained numpy array.

    Args:
        params: Dictionary of parameter values
        model: 'qlearning', 'wmrl', or 'wmrl_m3'

    Returns:
        Unconstrained parameter array
    """
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


def unconstrained_to_params(x: np.ndarray, model: str) -> Dict[str, float]:
    """
    Transform unconstrained array back to bounded parameter dict.

    Args:
        x: Unconstrained parameter array
        model: 'qlearning', 'wmrl', or 'wmrl_m3'

    Returns:
        Dictionary of bounded parameter values
    """
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

    params = {}
    for i, name in enumerate(param_names):
        lower, upper = bounds[name]
        params[name] = unbounded_to_bounded(x[i], lower, upper)

    return params


def get_default_params(model: str) -> Dict[str, float]:
    """Get default starting parameters for a model."""
    if model == 'qlearning':
        return {
            'alpha_pos': 0.3,
            'alpha_neg': 0.1,
            'epsilon': 0.05
        }
    elif model == 'wmrl':
        return {
            'alpha_pos': 0.3,
            'alpha_neg': 0.1,
            'phi': 0.1,
            'rho': 0.7,
            'capacity': 4.0,
            'epsilon': 0.05
        }
    elif model == 'wmrl_m3':
        return {
            'alpha_pos': 0.3,
            'alpha_neg': 0.1,
            'phi': 0.1,
            'rho': 0.7,
            'capacity': 4.0,
            'kappa': 0.0,  # Default to M2 behavior (no perseveration)
            'epsilon': 0.05
        }
    else:
        raise ValueError(f"Unknown model: {model}")


def sample_random_start(model: str, rng: np.random.Generator) -> np.ndarray:
    """
    Sample random starting point in unconstrained space.

    Uses Normal(0, 1) which maps to roughly uniform in bounded space
    (centered around 0.5 for (0,1) bounds).

    Args:
        model: 'qlearning', 'wmrl', or 'wmrl_m3'
        rng: NumPy random generator

    Returns:
        Unconstrained parameter array
    """
    if model == 'qlearning':
        n_params = len(QLEARNING_PARAMS)
    elif model == 'wmrl':
        n_params = len(WMRL_PARAMS)
    elif model == 'wmrl_m3':
        n_params = len(WMRL_M3_PARAMS)
    else:
        raise ValueError(f"Unknown model: {model}")

    return rng.normal(0, 1.5, size=n_params)  # SD=1.5 gives reasonable spread


# =============================================================================
# Information Criteria
# =============================================================================

def compute_aic(nll: float, k: int) -> float:
    """
    Compute Akaike Information Criterion.

    AIC = 2k - 2*log(L) = 2k + 2*NLL

    Args:
        nll: Negative log-likelihood at MLE
        k: Number of free parameters

    Returns:
        AIC value (lower is better)
    """
    return 2 * k + 2 * nll


def compute_bic(nll: float, k: int, n: int) -> float:
    """
    Compute Bayesian Information Criterion.

    BIC = k*log(n) - 2*log(L) = k*log(n) + 2*NLL

    Args:
        nll: Negative log-likelihood at MLE
        k: Number of free parameters
        n: Number of observations (trials)

    Returns:
        BIC value (lower is better)
    """
    return k * np.log(n) + 2 * nll


def compute_aicc(nll: float, k: int, n: int) -> float:
    """
    Compute corrected AIC (for small sample sizes).

    AICc = AIC + (2k^2 + 2k) / (n - k - 1)

    Args:
        nll: Negative log-likelihood at MLE
        k: Number of free parameters
        n: Number of observations (trials)

    Returns:
        AICc value (lower is better)
    """
    aic = compute_aic(nll, k)
    if n - k - 1 > 0:
        correction = (2 * k * k + 2 * k) / (n - k - 1)
    else:
        correction = np.inf  # Not enough data
    return aic + correction


def get_n_params(model: str) -> int:
    """Get number of free parameters for a model."""
    if model == 'qlearning':
        return 3  # alpha_pos, alpha_neg, epsilon
    elif model == 'wmrl':
        return 6  # alpha_pos, alpha_neg, phi, rho, capacity, epsilon
    elif model == 'wmrl_m3':
        return 7  # alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon
    else:
        raise ValueError(f"Unknown model: {model}")


# =============================================================================
# Group Statistics (Senta et al. style)
# =============================================================================

def compute_group_statistics(
    param_values: np.ndarray,
    param_name: str
) -> Dict[str, float]:
    """
    Compute group-level statistics for a parameter.

    Following Senta et al. (2025): report mean +/- SEM across participants.

    Args:
        param_values: Array of parameter values across participants
        param_name: Name of the parameter

    Returns:
        Dictionary with mean, sd, se, ci_lower, ci_upper
    """
    n = len(param_values)
    mean = np.mean(param_values)
    sd = np.std(param_values, ddof=1)  # Sample SD
    se = sd / np.sqrt(n)  # Standard error of the mean

    # 95% CI using t-distribution
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_lower = mean - t_crit * se
    ci_upper = mean + t_crit * se

    return {
        'mean': mean,
        'sd': sd,
        'se': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n': n
    }


def summarize_all_parameters(
    fits_df,  # pd.DataFrame
    model: str
) -> Dict[str, Dict[str, float]]:
    """
    Compute group statistics for all parameters.

    Args:
        fits_df: DataFrame with individual fit results
        model: 'qlearning', 'wmrl', or 'wmrl_m3'

    Returns:
        Nested dict: {param_name: {mean, sd, se, ci_lower, ci_upper}}
    """
    if model == 'qlearning':
        param_names = QLEARNING_PARAMS
    elif model == 'wmrl':
        param_names = WMRL_PARAMS
    elif model == 'wmrl_m3':
        param_names = WMRL_M3_PARAMS
    else:
        raise ValueError(f"Unknown model: {model}")

    summary = {}
    for param in param_names:
        if param in fits_df.columns:
            values = fits_df[param].values
            summary[param] = compute_group_statistics(values, param)

    return summary


# =============================================================================
# Convergence Diagnostics
# =============================================================================

def check_convergence(
    results: List,  # List of scipy OptimizeResult
    tolerance: float = 0.1
) -> Dict[str, any]:
    """
    Check convergence across multiple random starts.

    Args:
        results: List of optimization results from different starts
        tolerance: Maximum NLL difference to consider "converged to same point"

    Returns:
        Dictionary with convergence diagnostics
    """
    if not results:
        return {
            'n_successful': 0,
            'n_converged_to_same': 0,
            'best_nll': np.inf,
            'nll_spread': np.inf,
            'converged': False
        }

    nlls = [r.fun for r in results if r.success]

    if not nlls:
        return {
            'n_successful': 0,
            'n_converged_to_same': 0,
            'best_nll': np.inf,
            'nll_spread': np.inf,
            'converged': False
        }

    best_nll = min(nlls)
    n_converged_to_same = sum(1 for nll in nlls if abs(nll - best_nll) < tolerance)

    return {
        'n_successful': len(nlls),
        'n_converged_to_same': n_converged_to_same,
        'best_nll': best_nll,
        'nll_spread': max(nlls) - min(nlls),
        'converged': n_converged_to_same >= len(nlls) * 0.5  # At least 50% agree
    }


def check_at_bounds(
    params: Dict[str, float],
    model: str,
    tolerance: float = 0.01
) -> List[str]:
    """
    Check if any parameters are at their bounds.

    Args:
        params: Parameter dictionary
        model: 'qlearning', 'wmrl', or 'wmrl_m3'
        tolerance: Distance from bound to consider "at bound"

    Returns:
        List of parameter names that hit bounds.
    """
    if model == 'qlearning':
        bounds = QLEARNING_BOUNDS
    elif model == 'wmrl':
        bounds = WMRL_BOUNDS
    elif model == 'wmrl_m3':
        bounds = WMRL_M3_BOUNDS
    else:
        raise ValueError(f"Unknown model: {model}")

    at_bounds = []
    for name, value in params.items():
        if name in bounds:
            lower, upper = bounds[name]
            if abs(value - lower) < tolerance or abs(value - upper) < tolerance:
                at_bounds.append(name)

    return at_bounds


# =============================================================================
# Model Comparison
# =============================================================================

def compare_models_aic(
    aic_model1: float,
    aic_model2: float,
    model1_name: str = 'model1',
    model2_name: str = 'model2'
) -> Dict[str, any]:
    """
    Compare two models using AIC.

    Returns interpretation following Burnham & Anderson guidelines.
    """
    delta_aic = aic_model1 - aic_model2  # Positive means model2 is better

    # Akaike weights
    min_aic = min(aic_model1, aic_model2)
    w1 = np.exp(-0.5 * (aic_model1 - min_aic))
    w2 = np.exp(-0.5 * (aic_model2 - min_aic))
    total_w = w1 + w2

    weight_model1 = w1 / total_w
    weight_model2 = w2 / total_w

    # Evidence strength
    abs_delta = abs(delta_aic)
    if abs_delta < 2:
        evidence = 'Weak (models essentially equivalent)'
    elif abs_delta < 4:
        evidence = 'Moderate'
    elif abs_delta < 10:
        evidence = 'Strong'
    else:
        evidence = 'Very strong'

    preferred = model2_name if delta_aic > 0 else model1_name

    return {
        'delta_aic': delta_aic,
        f'weight_{model1_name}': weight_model1,
        f'weight_{model2_name}': weight_model2,
        'evidence_strength': evidence,
        'preferred_model': preferred
    }


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("Testing MLE utilities...")

    # Test transformations
    print("\n1. Testing parameter transformations:")
    test_params_ql = {'alpha_pos': 0.3, 'alpha_neg': 0.1, 'epsilon': 0.05}
    x = params_to_unconstrained(test_params_ql, 'qlearning')
    recovered = unconstrained_to_params(x, 'qlearning')
    print(f"   Original: {test_params_ql}")
    print(f"   Unconstrained: {x}")
    print(f"   Recovered: {recovered}")

    # Check round-trip
    for key in test_params_ql:
        assert abs(test_params_ql[key] - recovered[key]) < 1e-10, f"Round-trip failed for {key}"
    print("   Round-trip: PASSED")

    # Test information criteria
    print("\n2. Testing information criteria:")
    nll = 100.0
    n_trials = 500
    k_ql = 3
    k_wmrl = 6

    aic_ql = compute_aic(nll, k_ql)
    bic_ql = compute_bic(nll, k_ql, n_trials)
    aic_wmrl = compute_aic(nll, k_wmrl)
    bic_wmrl = compute_bic(nll, k_wmrl, n_trials)

    print(f"   Q-Learning (k=3): AIC={aic_ql:.2f}, BIC={bic_ql:.2f}")
    print(f"   WM-RL (k=6): AIC={aic_wmrl:.2f}, BIC={bic_wmrl:.2f}")
    print(f"   Delta AIC (WMRL - QL): {aic_wmrl - aic_ql:.2f}")

    # Test group statistics
    print("\n3. Testing group statistics:")
    fake_alpha_pos = np.array([0.25, 0.30, 0.35, 0.28, 0.32, 0.40, 0.22])
    stats_result = compute_group_statistics(fake_alpha_pos, 'alpha_pos')
    print(f"   Mean: {stats_result['mean']:.3f}")
    print(f"   SD: {stats_result['sd']:.3f}")
    print(f"   SE: {stats_result['se']:.3f}")
    print(f"   95% CI: [{stats_result['ci_lower']:.3f}, {stats_result['ci_upper']:.3f}]")

    # Test model comparison
    print("\n4. Testing model comparison:")
    comparison = compare_models_aic(210, 200, 'qlearning', 'wmrl')
    print(f"   Delta AIC: {comparison['delta_aic']:.2f}")
    print(f"   Preferred: {comparison['preferred_model']}")
    print(f"   Evidence: {comparison['evidence_strength']}")

    print("\nAll tests passed!")
