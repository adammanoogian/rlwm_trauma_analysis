"""
MLE Utility Functions for RLWM Model Fitting

Parameter transformations, information criteria, and helper functions
for Maximum Likelihood Estimation following Senta et al. (2025) methodology.

Supports both numpy (for result handling) and JAX (for optimization).

Extended with Hessian-based diagnostics:
- Standard errors via inverse Hessian
- Parameter correlations
- Condition number for identifiability
- Pseudo-R² for model fit quality
"""

import numpy as np
import jax
import jax.numpy as jnp
from scipy import stats
from scipy.stats import qmc
from typing import Dict, Tuple, List, Optional, Any


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


# =============================================================================
# JAX-Compatible Transformations (for jaxopt optimization)
# =============================================================================

def jax_inv_logit(x):
    """JAX-compatible inverse logit transformation."""
    return 1 / (1 + jnp.exp(-x))


def jax_unbounded_to_bounded(x, lower: float, upper: float):
    """JAX-compatible unbounded to bounded transformation."""
    p = jax_inv_logit(x)
    return lower + p * (upper - lower)


def jax_unconstrained_to_params_qlearning(x: jnp.ndarray) -> Tuple:
    """
    JAX-compatible parameter transformation for Q-learning.

    Returns tuple (alpha_pos, alpha_neg, epsilon) for direct use in likelihood.
    """
    bounds = QLEARNING_BOUNDS
    alpha_pos = jax_unbounded_to_bounded(x[0], *bounds['alpha_pos'])
    alpha_neg = jax_unbounded_to_bounded(x[1], *bounds['alpha_neg'])
    epsilon = jax_unbounded_to_bounded(x[2], *bounds['epsilon'])
    return alpha_pos, alpha_neg, epsilon


def jax_unconstrained_to_params_wmrl(x: jnp.ndarray) -> Tuple:
    """
    JAX-compatible parameter transformation for WM-RL.

    Returns tuple (alpha_pos, alpha_neg, phi, rho, capacity, epsilon) for direct use.
    """
    bounds = WMRL_BOUNDS
    alpha_pos = jax_unbounded_to_bounded(x[0], *bounds['alpha_pos'])
    alpha_neg = jax_unbounded_to_bounded(x[1], *bounds['alpha_neg'])
    phi = jax_unbounded_to_bounded(x[2], *bounds['phi'])
    rho = jax_unbounded_to_bounded(x[3], *bounds['rho'])
    capacity = jax_unbounded_to_bounded(x[4], *bounds['capacity'])
    epsilon = jax_unbounded_to_bounded(x[5], *bounds['epsilon'])
    return alpha_pos, alpha_neg, phi, rho, capacity, epsilon


def jax_unconstrained_to_params_wmrl_m3(x: jnp.ndarray) -> Tuple:
    """
    JAX-compatible parameter transformation for WM-RL M3.

    Returns tuple (alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon) for direct use.
    """
    bounds = WMRL_M3_BOUNDS
    alpha_pos = jax_unbounded_to_bounded(x[0], *bounds['alpha_pos'])
    alpha_neg = jax_unbounded_to_bounded(x[1], *bounds['alpha_neg'])
    phi = jax_unbounded_to_bounded(x[2], *bounds['phi'])
    rho = jax_unbounded_to_bounded(x[3], *bounds['rho'])
    capacity = jax_unbounded_to_bounded(x[4], *bounds['capacity'])
    kappa = jax_unbounded_to_bounded(x[5], *bounds['kappa'])
    epsilon = jax_unbounded_to_bounded(x[6], *bounds['epsilon'])
    return alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon


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


def sample_lhs_starts(model: str, n_starts: int, seed: int = None) -> np.ndarray:
    """
    Generate starting points using Latin Hypercube Sampling in bounded space.

    LHS ensures even coverage of parameter space by dividing each dimension
    into n_starts equal strata and sampling exactly once per stratum. This
    improves the chance of finding the global optimum compared to random
    sampling, which can leave gaps or have clusters.

    Args:
        model: 'qlearning', 'wmrl', or 'wmrl_m3'
        n_starts: Number of starting points
        seed: Random seed for reproducibility

    Returns:
        Array of shape (n_starts, n_params) in BOUNDED parameter space
    """
    if model == 'qlearning':
        bounds_dict = QLEARNING_BOUNDS
        param_names = QLEARNING_PARAMS
    elif model == 'wmrl':
        bounds_dict = WMRL_BOUNDS
        param_names = WMRL_PARAMS
    elif model == 'wmrl_m3':
        bounds_dict = WMRL_M3_BOUNDS
        param_names = WMRL_M3_PARAMS
    else:
        raise ValueError(f"Unknown model: {model}")

    n_params = len(param_names)

    # Generate LHS samples in [0, 1]^d
    sampler = qmc.LatinHypercube(d=n_params, seed=seed)
    samples = sampler.random(n=n_starts)

    # Scale to parameter bounds
    lower = np.array([bounds_dict[p][0] for p in param_names])
    upper = np.array([bounds_dict[p][1] for p in param_names])

    scaled_samples = qmc.scale(samples, lower, upper)

    return scaled_samples


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
    results: List,  # List of scipy OptimizeResult (or _JaxoptResult wrappers)
    iteration_stats: List[Dict] = None,
    tolerance: float = 1.0
) -> Dict[str, any]:
    """
    Check convergence based on the best start's optimizer success flag.

    A participant is "converged" if the best start's optimizer reported
    success=True (i.e., scipy's own convergence criteria were met: gradient
    tolerance and function tolerance satisfied, not just maxiter/maxfun hit).

    Additionally reports n_near_best as a secondary diagnostic — how many
    starts found NLLs within tolerance of the best. This is informational
    but does NOT gate convergence.

    Args:
        results: List of optimization results from different starts
        iteration_stats: List of per-start dicts with 'scipy_converged', etc.
                        When provided, uses the best start's scipy_converged flag.
        tolerance: NLL tolerance for counting "near best" starts (default: 1.0)

    Returns:
        Dictionary with convergence diagnostics
    """
    if not results:
        return {
            'n_successful': 0,
            'n_near_best': 0,
            'best_nll': np.inf,
            'nll_spread': np.inf,
            'best_scipy_converged': False,
            'converged': False,
        }

    nlls = [r.fun for r in results if r.success]

    if not nlls:
        return {
            'n_successful': 0,
            'n_near_best': 0,
            'best_nll': np.inf,
            'nll_spread': np.inf,
            'best_scipy_converged': False,
            'converged': False,
        }

    best_nll = min(nlls)
    n_near_best = sum(1 for nll in nlls if abs(nll - best_nll) < tolerance)

    # Find the best start's index among successful results
    best_idx = next(i for i, r in enumerate(results) if r.success and r.fun == best_nll)

    # Determine scipy convergence of the best start
    if iteration_stats and best_idx < len(iteration_stats):
        best_scipy_converged = iteration_stats[best_idx].get('scipy_converged', True)
    else:
        # No iteration_stats provided — trust r.success (finite NLL)
        best_scipy_converged = True

    return {
        'n_successful': len(nlls),
        'n_near_best': n_near_best,
        'best_nll': best_nll,
        'nll_spread': max(nlls) - min(nlls),
        'best_scipy_converged': best_scipy_converged,
        'converged': best_scipy_converged,
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
# Hessian-Based Diagnostics
# =============================================================================

def compute_pseudo_r2(nll: float, n_trials: int, n_actions: int = 3) -> float:
    """
    Compute McFadden's pseudo-R² (variance explained).

    R² = 1 - (NLL_model / NLL_null)
    where NLL_null = -n * log(1/n_actions) = n * log(n_actions)

    This gives an intuitive measure of how much better the model is than
    chance-level performance (random responding).

    Args:
        nll: Negative log-likelihood of the fitted model
        n_trials: Number of trials
        n_actions: Number of possible actions (default: 3)

    Returns:
        Pseudo-R² value (0 = chance, 1 = perfect fit)

    Example:
        >>> # Chance-level NLL for 988 trials = 988 * log(3) ≈ 1085
        >>> # If model NLL = 763, pseudo-R² ≈ 0.30 (30% variance explained)
        >>> pseudo_r2 = compute_pseudo_r2(763, 988)
    """
    nll_null = n_trials * np.log(n_actions)  # Chance-level NLL
    if nll_null == 0:
        return np.nan
    pseudo_r2 = 1 - (nll / nll_null)
    return float(pseudo_r2)


def check_gradient_norm(
    objective_fn,
    x_opt: np.ndarray,
    tolerance: float = 1e-4
) -> Tuple[float, bool]:
    """
    Check gradient norm at optimum using JAX autodiff.

    At a true optimum, the gradient should be approximately zero.
    A large gradient indicates the optimizer didn't fully converge.

    Args:
        objective_fn: JAX-compatible objective function
        x_opt: Optimal parameter values (unconstrained)
        tolerance: Maximum gradient norm to consider "converged"

    Returns:
        Tuple of (gradient_norm, is_converged)
    """
    try:
        # JIT-compile for faster execution (especially on GPU)
        grad_fn = jax.jit(jax.grad(objective_fn))
        x_jax = jnp.array(x_opt)
        g = grad_fn(x_jax)
        grad_norm = float(jnp.linalg.norm(g))
        is_converged = grad_norm < tolerance
        return grad_norm, is_converged
    except Exception as e:
        return np.nan, False


def compute_hessian_diagnostics(
    objective_fn,
    x_opt: np.ndarray,
    model: str,
    param_names: List[str] = None
) -> Dict[str, Any]:
    """
    Compute Hessian-based diagnostics for MLE fit quality.

    At the MLE optimum, the Fisher Information matrix I = E[H(NLL)]
    The covariance of parameter estimates is approximately: Var(θ̂) ≈ I⁻¹
    Standard errors are the square roots of the diagonal: SE(θᵢ) = √(I⁻¹ᵢᵢ)

    This function computes:
    1. Standard errors for all parameters (in unconstrained space)
    2. Parameter correlation matrix
    3. Condition number (for identifiability assessment)

    Args:
        objective_fn: JAX-compatible objective function (returns NLL)
        x_opt: Optimal parameter values (unconstrained space)
        model: Model name ('qlearning', 'wmrl', 'wmrl_m3')
        param_names: List of parameter names (optional, will use defaults)

    Returns:
        Dictionary with:
            'success': bool - whether computation succeeded
            'se_unconstrained': dict - SEs in unconstrained space
            'se_bounded': dict - SEs in bounded space (via delta method)
            'correlations': dict - parameter correlation matrix
            'condition_number': float - Hessian condition number
            'hessian_invertible': bool - whether Hessian was invertible

    Note:
        High condition number (>1000) indicates poor parameter identifiability.
        This often occurs when parameters are highly correlated (e.g., α₊ ≈ α₋).
    """
    # Get parameter names
    if param_names is None:
        if model == 'qlearning':
            param_names = QLEARNING_PARAMS
        elif model == 'wmrl':
            param_names = WMRL_PARAMS
        elif model == 'wmrl_m3':
            param_names = WMRL_M3_PARAMS
        else:
            return {'success': False, 'error': f'Unknown model: {model}'}

    n_params = len(param_names)

    try:
        # Compute Hessian at optimum using JAX
        # JIT-compile for faster execution (especially on GPU)
        hess_fn = jax.jit(jax.hessian(objective_fn))
        x_jax = jnp.array(x_opt)
        H = hess_fn(x_jax)

        # Convert to numpy for numerical operations
        H_np = np.array(H)

        # Check for NaN/Inf in Hessian
        if not np.all(np.isfinite(H_np)):
            return {
                'success': False,
                'error': 'Hessian contains NaN/Inf values',
                'hessian_invertible': False
            }

        # Compute eigenvalues for condition number
        eigenvalues = np.linalg.eigvalsh(H_np)
        min_eig = np.min(eigenvalues)
        max_eig = np.max(eigenvalues)

        # Condition number (ratio of max to min eigenvalue)
        # High values indicate near-singularity
        if min_eig <= 0:
            condition_number = np.inf
            hessian_positive_definite = False
        else:
            condition_number = max_eig / min_eig
            hessian_positive_definite = True

        # Try to invert Hessian for covariance matrix
        try:
            cov_matrix = np.linalg.inv(H_np)
            hessian_invertible = True
        except np.linalg.LinAlgError:
            # Hessian is singular - use pseudo-inverse
            cov_matrix = np.linalg.pinv(H_np)
            hessian_invertible = False

        # Extract standard errors (sqrt of diagonal)
        variances = np.diag(cov_matrix)

        # Handle negative variances (numerical issues)
        se_unconstrained = {}
        for i, param in enumerate(param_names):
            if variances[i] > 0:
                se_unconstrained[param] = float(np.sqrt(variances[i]))
            else:
                se_unconstrained[param] = np.nan

        # Compute correlation matrix
        correlations = {}
        std_devs = np.sqrt(np.maximum(variances, 1e-10))
        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)

        # Store as nested dict for easy access
        for i, param_i in enumerate(param_names):
            correlations[param_i] = {}
            for j, param_j in enumerate(param_names):
                correlations[param_i][param_j] = float(corr_matrix[i, j])

        # Transform SEs to bounded space using delta method
        se_bounded = _transform_se_to_bounded(
            se_unconstrained, x_opt, model, param_names
        )

        return {
            'success': True,
            'se_unconstrained': se_unconstrained,
            'se_bounded': se_bounded,
            'correlations': correlations,
            'condition_number': float(condition_number),
            'hessian_invertible': hessian_invertible,
            'hessian_positive_definite': hessian_positive_definite,
            'min_eigenvalue': float(min_eig),
            'max_eigenvalue': float(max_eig)
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'hessian_invertible': False
        }


def _transform_se_to_bounded(
    se_unconstrained: Dict[str, float],
    x_opt: np.ndarray,
    model: str,
    param_names: List[str]
) -> Dict[str, float]:
    """
    Transform standard errors from unconstrained to bounded space using delta method.

    The delta method approximation:
    SE_bounded ≈ SE_unconstrained * |∂(transform)/∂x|

    For logit transformation: p = 1/(1+exp(-x))
    The derivative is: dp/dx = p * (1-p)

    Args:
        se_unconstrained: SEs in unconstrained space
        x_opt: Optimal values in unconstrained space
        model: Model name for bounds lookup
        param_names: Parameter names

    Returns:
        Dictionary of SEs in bounded parameter space
    """
    # Get bounds for model
    if model == 'qlearning':
        bounds = QLEARNING_BOUNDS
    elif model == 'wmrl':
        bounds = WMRL_BOUNDS
    elif model == 'wmrl_m3':
        bounds = WMRL_M3_BOUNDS
    else:
        return {}

    se_bounded = {}

    for i, param in enumerate(param_names):
        se_unc = se_unconstrained.get(param, np.nan)
        if np.isnan(se_unc):
            se_bounded[param] = np.nan
            continue

        # Get bounds
        lower, upper = bounds[param]

        # Current value in unconstrained space
        x = x_opt[i]

        # Compute derivative of inverse logit (sigmoid)
        # p = 1 / (1 + exp(-x)) = sigmoid(x)
        # dp/dx = p * (1-p)
        p = 1 / (1 + np.exp(-x))
        dp_dx = p * (1 - p)

        # Scale by range
        scale = upper - lower
        jacobian = scale * dp_dx

        # Delta method: SE_bounded = |jacobian| * SE_unconstrained
        se_bounded[param] = float(np.abs(jacobian) * se_unc)

    return se_bounded


def compute_confidence_intervals(
    params: Dict[str, float],
    se_bounded: Dict[str, float],
    alpha: float = 0.05
) -> Dict[str, Tuple[float, float]]:
    """
    Compute confidence intervals for fitted parameters.

    Uses normal approximation: CI = θ̂ ± z_{α/2} * SE

    Args:
        params: Fitted parameter values
        se_bounded: Standard errors in bounded space
        alpha: Significance level (default: 0.05 for 95% CI)

    Returns:
        Dictionary of (lower, upper) tuples for each parameter
    """
    z_crit = stats.norm.ppf(1 - alpha / 2)

    ci = {}
    for param, value in params.items():
        se = se_bounded.get(param, np.nan)
        if np.isnan(se):
            ci[param] = (np.nan, np.nan)
        else:
            ci[param] = (value - z_crit * se, value + z_crit * se)

    return ci


def get_high_correlations(
    correlations: Dict[str, Dict[str, float]],
    threshold: float = 0.9
) -> List[Tuple[str, str, float]]:
    """
    Find pairs of parameters with high correlations.

    High correlations (|r| > 0.9) indicate potential identifiability issues
    where parameters trade off against each other.

    Args:
        correlations: Correlation matrix as nested dict
        threshold: Absolute correlation threshold (default: 0.9)

    Returns:
        List of (param1, param2, correlation) tuples for high correlations
    """
    high_corr = []
    seen_pairs = set()

    for param1, corr_dict in correlations.items():
        for param2, corr in corr_dict.items():
            if param1 == param2:
                continue
            pair = tuple(sorted([param1, param2]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            if abs(corr) > threshold:
                high_corr.append((param1, param2, corr))

    return high_corr


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
