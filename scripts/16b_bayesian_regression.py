#!/usr/bin/env python
"""
16b: Bayesian Linear Regression (Parameters -> Scale Totals)
=============================================================

.. deprecated::
    This script is superseded by the Level-2 hierarchical pipeline
    implemented in Phases 15-16 (``scripts/13_fit_bayesian.py`` with
    ``--subscale``, and ``scripts/18_bayesian_level2_effects.py``).
    The Level-2 approach jointly estimates individual parameters and
    trauma associations in a single inference pass, providing proper
    uncertainty propagation from the participant level to the
    regression level.

    This script is retained as a fast-preview / supplementary path
    that runs a post-hoc NumPyro regression on MLE point estimates.
    It does NOT require MCMC convergence from Phase 15-17 fits and
    produces approximate directional evidence suitable for exploratory
    analysis.

    Use this script for: quick sanity-check before running the full pipeline.
    Use ``scripts/18_bayesian_level2_effects.py`` for: manuscript-quality results.

Bayesian multivariate regression of model parameters on trauma scale totals.
Supports any model via MODEL_REGISTRY. Uses NumPyro/JAX backend.

# v4.0: PyMC backend removed (INFRA-07). NumPyro-only.

Models:
    1. lec_total ~ beta_0 + beta_1*param_1 + ... + beta_k*param_k
    2. ies_total ~ beta_0 + beta_1*param_1 + ... + beta_k*param_k

Priors (weakly informative):
    - intercept ~ Normal(mean(y), 2*sd(y))
    - beta ~ Normal(0, sd(y))  [raw mode] or Normal(0, 1)  [z-scored mode]
    - sigma ~ HalfNormal(sd(y))

Reporting:
    - Posterior mean + 95% HDI for each coefficient
    - P(direction): probability of positive/negative effect
    - Bayesian R-squared (variance explained)
    - Forest plots with 50%/95% HDI bars

Inputs:
    - output/mle/<model>_individual_fits.csv (fitted parameters)
    - output/summary_participant_metrics.csv (trauma scales)

Outputs:
    - output/regressions/bayesian/<model>_regression_summary.csv
    - figures/regressions/bayesian/<model>_forest_plot_lec_total.png
    - figures/regressions/bayesian/<model>_forest_plot_ies_total.png

Usage:
    python scripts/16b_bayesian_regression.py --model wmrl_m6b
    python scripts/16b_bayesian_regression.py --model wmrl_m5 --standardize
    python scripts/16b_bayesian_regression.py --model wmrl_m6b --dry-run

Dependencies:
    pip install numpyro jax arviz  (NumPyro/JAX backend)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Project imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from config import EXCLUDED_PARTICIPANTS, MODEL_REGISTRY, ALL_MODELS

# Backend: NumPyro/JAX only (v4.0 INFRA-07: PyMC dropped)
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az
BACKEND = "numpyro"

# Plotting style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300

# Outcome scales
OUTCOME_SCALES = {
    'lec_total': 'LEC-5 Total Trauma Exposure',
    'ies_total': 'IES-R Total PTSD Symptoms',
}

# Display-friendly parameter labels
PARAM_LABELS = {
    'alpha_pos': r'$\alpha_+$',
    'alpha_neg': r'$\alpha_-$',
    'phi': r'$\phi$',
    'rho': r'$\rho$',
    'capacity': r'$K$',
    'kappa': r'$\kappa$',
    'kappa_total': r'$\kappa_{total}$',
    'kappa_share': r'$\kappa_{share}$',
    'kappa_s': r'$\kappa_s$',
    'phi_rl': r'$\phi_{RL}$',
    'epsilon': r'$\varepsilon$',
    'v_scale': r'$v_{scale}$',
    'A': r'$A$',
    'delta': r'$\delta$',
    't0': r'$t_0$',
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(params_path: Path, model: str) -> pd.DataFrame:
    """Load model parameters and trauma scales, merge on participant ID.

    Parameters
    ----------
    params_path : Path
        Path to fitted parameters CSV.
    model : str
        Model key (e.g., 'wmrl_m6b') — used to determine param columns.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with param columns + lec_total + ies_total.
    """
    param_cols = MODEL_REGISTRY[model]['params']

    params_df = pd.read_csv(params_path)
    if 'participant_id' in params_df.columns:
        params_df = params_df.rename(columns={'participant_id': 'sona_id'})
    params_df['sona_id'] = params_df['sona_id'].astype(str)

    # Load survey data from complete source
    surveys_df = pd.read_csv('output/summary_participant_metrics.csv')
    surveys_df['sona_id'] = surveys_df['sona_id'].astype(str)
    # Rename LEC columns to match expected names
    surveys_df = surveys_df.rename(columns={
        'less_total_events': 'lec_total',
        'less_personal_events': 'lec_personal',
    })

    # Exclude participants per config
    excluded_str = [str(x) for x in EXCLUDED_PARTICIPANTS]
    params_df = params_df[~params_df['sona_id'].isin(excluded_str)].copy()
    surveys_df = surveys_df[~surveys_df['sona_id'].isin(excluded_str)].copy()

    keep_cols = ['sona_id'] + [c for c in param_cols if c in params_df.columns]
    survey_cols = ['sona_id', 'lec_total', 'ies_total']
    survey_cols = [c for c in survey_cols if c in surveys_df.columns]

    df = params_df[keep_cols].merge(surveys_df[survey_cols], on='sona_id', how='inner')
    df = df.dropna(subset=[c for c in param_cols if c in df.columns] + list(OUTCOME_SCALES.keys()))

    print(f"Loaded {len(df)} participants with complete {model} params + trauma scales")
    return df, param_cols


def prepare_predictors(df: pd.DataFrame, param_cols: list[str],
                       standardize: bool = False) -> tuple:
    """Prepare predictor matrix for regression.

    Parameters
    ----------
    df : pd.DataFrame
        Data with parameter columns.
    param_cols : list[str]
        Parameter column names from MODEL_REGISTRY.
    standardize : bool
        If True, z-score all predictors.

    Returns
    -------
    tuple of (X, param_names, scale_label)
    """
    param_names = [p for p in param_cols if p in df.columns]
    X = df[param_names].values.astype(np.float64)

    if standardize:
        means = X.mean(axis=0)
        sds = X.std(axis=0)
        sds[sds < 1e-10] = 1.0
        X = (X - means) / sds
        scale_label = "z-scored"
    else:
        # Rescale capacity from [1,7] to [0,1], leave others raw
        if 'capacity' in param_names:
            cap_idx = param_names.index('capacity')
            X[:, cap_idx] = (X[:, cap_idx] - 1.0) / 6.0
        scale_label = "raw [0,1]"

    return X, param_names, scale_label


# ============================================================================
# INFERENCE (dual-backend: PyMC or NumPyro)
# ============================================================================

def run_mcmc(X, y, outcome_name, param_names, standardize=False,
             n_chains=4, n_warmup=1000, n_samples=2000, seed=42):
    """Run NUTS MCMC for one outcome model using NumPyro/JAX.

    Returns
    -------
    samples : dict
        Keys: 'intercept' (n_samples,), 'beta' (n_samples, n_params),
        'sigma' (n_samples,).
    """
    y_mean = float(np.mean(y))
    y_sd = max(float(np.std(y)), 0.1)
    n_pred = X.shape[1]

    print(f"\n  Running MCMC [{BACKEND}] for {outcome_name} ({n_chains} chains, "
          f"{n_warmup} warmup, {n_samples} samples)...")

    return _run_numpyro(X, y, y_mean, y_sd, n_pred,
                        standardize, n_chains, n_warmup, n_samples, seed)


def _run_numpyro(X, y, y_mean, y_sd, n_pred,
                 standardize, n_chains, n_warmup, n_samples, seed):
    """Run MCMC using NumPyro (JAX backend)."""
    def bayesian_linear_model(X, y_mean, y_sd, standardize, y=None):
        intercept = numpyro.sample('intercept', dist.Normal(y_mean, 2.0 * y_sd))
        beta_sd_val = 1.0 if standardize else y_sd
        beta = numpyro.sample('beta', dist.Normal(0.0, beta_sd_val).expand([X.shape[1]]))
        sigma = numpyro.sample('sigma', dist.HalfNormal(y_sd))
        mu = intercept + X @ beta
        numpyro.sample('y', dist.Normal(mu, sigma), obs=y)

    numpyro.set_host_device_count(n_chains)
    kernel = NUTS(bayesian_linear_model)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                num_chains=n_chains, progress_bar=True)
    rng_key = random.PRNGKey(seed)
    mcmc.run(rng_key, X=jnp.array(X), y_mean=y_mean, y_sd=y_sd,
             standardize=standardize, y=jnp.array(y))
    mcmc.print_summary(exclude_deterministic=False)
    samples = mcmc.get_samples()
    return {k: np.array(v) for k, v in samples.items()}


# ============================================================================
# POSTERIOR ANALYSIS
# ============================================================================

def compute_hdi(samples, prob=0.95):
    """Compute Highest Density Interval from samples.

    Sorts samples, finds the narrowest interval containing `prob` fraction.
    """
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    interval_size = int(np.ceil(prob * n))

    # Find narrowest interval
    widths = sorted_samples[interval_size:] - sorted_samples[:n - interval_size]
    best_idx = np.argmin(widths)

    return sorted_samples[best_idx], sorted_samples[best_idx + interval_size]


def summarize_posterior(samples, param_names, outcome_name):
    """Create summary table of posterior coefficients.

    Returns DataFrame with columns:
        outcome, parameter, mean, sd, hdi_2.5%, hdi_97.5%, hdi_25%, hdi_75%,
        p_direction, interpretation
    """
    beta_samples = np.array(samples['beta'])  # (n_samples, n_predictors)
    intercept_samples = np.array(samples['intercept'])
    sigma_samples = np.array(samples['sigma'])

    rows = []

    # Intercept
    hdi_95 = compute_hdi(intercept_samples, 0.95)
    hdi_50 = compute_hdi(intercept_samples, 0.50)
    rows.append({
        'outcome': outcome_name,
        'parameter': 'intercept',
        'mean': float(np.mean(intercept_samples)),
        'sd': float(np.std(intercept_samples)),
        'hdi_2.5%': hdi_95[0],
        'hdi_97.5%': hdi_95[1],
        'hdi_25%': hdi_50[0],
        'hdi_75%': hdi_50[1],
        'p_direction': np.nan,
        'interpretation': '',
    })

    # Regression coefficients
    for i, param in enumerate(param_names):
        b = beta_samples[:, i]
        hdi_95 = compute_hdi(b, 0.95)
        hdi_50 = compute_hdi(b, 0.50)
        p_pos = float(np.mean(b > 0))
        p_dir = max(p_pos, 1 - p_pos)

        if p_dir >= 0.95:
            interp = 'strong evidence'
        elif p_dir >= 0.90:
            interp = 'moderate evidence'
        else:
            interp = 'inconclusive'

        direction = '+' if p_pos > 0.5 else '-'

        rows.append({
            'outcome': outcome_name,
            'parameter': PARAM_LABELS.get(param, param),
            'mean': float(np.mean(b)),
            'sd': float(np.std(b)),
            'hdi_2.5%': hdi_95[0],
            'hdi_97.5%': hdi_95[1],
            'hdi_25%': hdi_50[0],
            'hdi_75%': hdi_50[1],
            'p_direction': p_dir,
            'interpretation': f'{interp} ({direction})',
        })

    # Sigma (residual SD)
    hdi_95 = compute_hdi(sigma_samples, 0.95)
    rows.append({
        'outcome': outcome_name,
        'parameter': 'sigma',
        'mean': float(np.mean(sigma_samples)),
        'sd': float(np.std(sigma_samples)),
        'hdi_2.5%': hdi_95[0],
        'hdi_97.5%': hdi_95[1],
        'hdi_25%': np.nan,
        'hdi_75%': np.nan,
        'p_direction': np.nan,
        'interpretation': '',
    })

    return pd.DataFrame(rows)


def compute_bayesian_r2(samples, X, y):
    """Compute Bayesian R-squared (Gelman et al., 2019).

    R2 = Var(predicted) / (Var(predicted) + Var(residual))

    Returns mean and HDI of R-squared distribution.
    """
    beta_samples = np.array(samples['beta'])
    intercept_samples = np.array(samples['intercept'])

    r2_samples = []
    for s in range(len(intercept_samples)):
        mu = intercept_samples[s] + X @ beta_samples[s]
        var_pred = np.var(mu)
        var_resid = float(np.array(samples['sigma'])[s]) ** 2
        r2 = var_pred / (var_pred + var_resid)
        r2_samples.append(r2)

    r2_samples = np.array(r2_samples)
    hdi = compute_hdi(r2_samples, 0.95)

    return float(np.mean(r2_samples)), hdi


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_forest(summary_df, outcome_name, outcome_label, output_path,
                scale_label="raw [0,1]", model_display="Model"):
    """Create horizontal forest plot of regression coefficients.

    Shows:
        - Point: posterior mean
        - Thick bar: 50% HDI
        - Thin bar: 95% HDI
        - Vertical line at 0
        - Color-coded by P(direction)
    """
    # Filter to regression coefficients only (exclude intercept, sigma)
    coef_df = summary_df[
        ~summary_df['parameter'].isin(['intercept', 'sigma'])
    ].copy()

    if len(coef_df) == 0:
        print(f"  No coefficients to plot for {outcome_name}")
        return

    fig, ax = plt.subplots(figsize=(8, max(4, 0.6 * len(coef_df) + 1.5)))

    y_positions = np.arange(len(coef_df))
    coef_df = coef_df.iloc[::-1]  # Reverse so first param is at top

    for i, (_, row) in enumerate(coef_df.iterrows()):
        p_dir = row['p_direction']
        if p_dir >= 0.95:
            color = '#e74c3c'  # Red: strong evidence
            alpha = 1.0
        elif p_dir >= 0.90:
            color = '#f39c12'  # Orange: moderate evidence
            alpha = 0.9
        else:
            color = '#95a5a6'  # Gray: inconclusive
            alpha = 0.7

        # 95% HDI (thin bar)
        ax.plot([row['hdi_2.5%'], row['hdi_97.5%']], [i, i],
                color=color, linewidth=1.5, alpha=alpha)

        # 50% HDI (thick bar)
        ax.plot([row['hdi_25%'], row['hdi_75%']], [i, i],
                color=color, linewidth=5, alpha=alpha, solid_capstyle='round')

        # Posterior mean (point)
        ax.scatter(row['mean'], i, color=color, s=60, zorder=5,
                   edgecolors='black', linewidths=0.5)

        # P(direction) annotation (placed just right of 95% HDI)
        ax.text(row['hdi_97.5%'] + 0.05, i, f"  P={p_dir:.2f}",
                va='center', fontsize=9, color=color, fontweight='bold')

    # Zero reference line
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(coef_df['parameter'].values, fontsize=11)
    if scale_label == "z-scored":
        xlabel = 'Standardized Coefficient (posterior mean +/- HDI)'
    else:
        xlabel = 'Coefficient (effect of 0 -> 1 on parameter, +/- HDI)'
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(f'Bayesian Regression: {model_display} -> {outcome_label}',
                 fontsize=13, fontweight='bold')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#e74c3c', linewidth=3, label='P(dir) >= 0.95 (strong)'),
        Line2D([0], [0], color='#f39c12', linewidth=3, label='P(dir) >= 0.90 (moderate)'),
        Line2D([0], [0], color='#95a5a6', linewidth=3, label='P(dir) < 0.90 (inconclusive)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [SAVED] {output_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Bayesian linear regression: model parameters -> trauma scales'
    )
    parser.add_argument('--model', type=str, default='wmrl_m6b',
                        choices=ALL_MODELS,
                        help='Model key (default: wmrl_m6b)')
    parser.add_argument('--chains', type=int, default=4,
                        help='Number of MCMC chains')
    parser.add_argument('--warmup', type=int, default=1000,
                        help='Number of warmup/burnin samples per chain')
    parser.add_argument('--samples', type=int, default=2000,
                        help='Number of posterior samples per chain')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for MCMC')
    parser.add_argument('--dry-run', action='store_true',
                        help='Load data and print summary only (no MCMC)')
    parser.add_argument('--standardize', action='store_true',
                        help='Z-score all predictors instead of using raw [0,1] scale')

    args = parser.parse_args()

    model_display = MODEL_REGISTRY[args.model]['display_name']
    print("=" * 80)
    print(f"BAYESIAN LINEAR REGRESSION: {model_display} PARAMETERS -> TRAUMA SCALES")
    print(f"  Backend: {BACKEND}")
    print("=" * 80)

    # Create output directories (per-model)
    output_dir = Path(f'output/regressions/bayesian/{args.model}')
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(f'figures/regressions/bayesian/{args.model}')
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    params_path = Path(f'output/mle/{MODEL_REGISTRY[args.model]["csv_filename"]}')
    if not params_path.exists():
        print(f"ERROR: {params_path} not found")
        return

    df, param_cols = load_data(params_path, args.model)

    # Prepare predictors
    X, param_names, scale_label = prepare_predictors(
        df, param_cols, standardize=args.standardize
    )
    print(f"\nPredictors ({len(param_names)} parameters, {scale_label} scale):")
    for i, p in enumerate(param_names):
        col = X[:, i]
        label = PARAM_LABELS.get(p, p)
        line = f"  {label:15s}: mean={col.mean():.4f}, range=[{col.min():.3f}, {col.max():.3f}]"
        if not args.standardize and p == 'capacity':
            line += "  (rescaled from [1,7])"
        print(line)

    print(f"\nOutcomes:")
    for outcome, label in OUTCOME_SCALES.items():
        y = df[outcome].values
        print(f"  {outcome:15s}: mean={np.mean(y):.2f}, sd={np.std(y):.2f}, "
              f"range=[{np.min(y):.0f}, {np.max(y):.0f}], N={len(y)}")

    if args.dry_run:
        print("\n[DRY-RUN] Skipping MCMC. Exiting.")
        return

    # Run models
    all_summaries = []

    for outcome, label in OUTCOME_SCALES.items():
        print("\n" + "=" * 80)
        print(f"MODEL: {label}")
        print("=" * 80)

        y = df[outcome].values.astype(np.float64)

        samples = run_mcmc(
            X, y, outcome, param_names,
            standardize=args.standardize,
            n_chains=args.chains,
            n_warmup=args.warmup,
            n_samples=args.samples,
            seed=args.seed,
        )

        # Summarize posterior
        summary_df = summarize_posterior(samples, param_names, outcome)
        all_summaries.append(summary_df)

        # Bayesian R-squared
        r2_mean, r2_hdi = compute_bayesian_r2(samples, X, y)
        print(f"\n  Bayesian R-squared: {r2_mean:.3f} [{r2_hdi[0]:.3f}, {r2_hdi[1]:.3f}]")

        r2_row = pd.DataFrame([{
            'outcome': outcome,
            'parameter': 'Bayesian_R2',
            'mean': r2_mean, 'sd': np.nan,
            'hdi_2.5%': r2_hdi[0], 'hdi_97.5%': r2_hdi[1],
            'hdi_25%': np.nan, 'hdi_75%': np.nan,
            'p_direction': np.nan, 'interpretation': '',
        }])
        all_summaries[-1] = pd.concat([all_summaries[-1], r2_row], ignore_index=True)

        # Print coefficient table
        coef_rows = summary_df[~summary_df['parameter'].isin(['intercept', 'sigma'])]
        print(f"\n  Coefficients ({scale_label}):")
        print(f"  {'Parameter':15s} {'Mean':>8s} {'SD':>8s} {'95% HDI':>22s} {'P(dir)':>8s} {'Interpretation'}")
        print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*22} {'-'*8} {'-'*20}")
        for _, row in coef_rows.iterrows():
            hdi_str = f"[{row['hdi_2.5%']:7.3f}, {row['hdi_97.5%']:7.3f}]"
            print(f"  {row['parameter']:15s} {row['mean']:8.3f} {row['sd']:8.3f} "
                  f"{hdi_str:>22s} {row['p_direction']:8.3f} {row['interpretation']}")

        # Forest plot
        plot_path = figures_dir / f'forest_plot_{outcome}.png'
        plot_forest(summary_df, outcome, label, plot_path,
                    scale_label=scale_label, model_display=model_display)

    # Save
    combined = pd.concat(all_summaries, ignore_index=True)
    summary_path = output_dir / 'bayesian_regression_summary.csv'
    combined.to_csv(summary_path, index=False)
    print(f"\n[SAVED] {summary_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results: {output_dir}/")
    print(f"Figures: {figures_dir}/")


if __name__ == '__main__':
    main()
