#!/usr/bin/env python
"""
16b: Bayesian Linear Regression (Parameters -> Scale Totals)
=============================================================

Bayesian multivariate regression of M3 WM-RL parameters on trauma scale totals.

Replaces 35+ underpowered univariate OLS regressions with 2 Bayesian models
that estimate the contribution of ALL M3 parameters to trauma outcomes jointly.

Models:
    1. lec_total ~ beta_0 + beta_1*alpha_pos + ... + beta_7*epsilon
    2. ies_total ~ beta_0 + beta_1*alpha_pos + ... + beta_7*epsilon

Predictor scaling (default: raw [0,1]):
    - 6 of 7 M3 parameters are already on [0,1]; capacity is rescaled from [1,7].
    - Coefficients represent the effect of a full min-to-max change in each parameter.
    - Optional --standardize flag z-scores all predictors instead.
    - Optional --drop-degenerate flag removes alpha_neg (96% at bound) and kappa
      (44% at zero), improving the predictor:N ratio from 6.4:1 to 9:1.

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
    - output/mle/wmrl_m3_individual_fits.csv (M3 fitted parameters)
    - output/mle/participant_surveys.csv (trauma scales)

Outputs:
    - output/regressions/bayesian/bayesian_regression_summary.csv
    - figures/regressions/bayesian/forest_plot_lec_total.png
    - figures/regressions/bayesian/forest_plot_ies_total.png

Usage:
    # Default: raw [0,1] scale, all 7 predictors
    python scripts/16b_bayesian_regression.py --model wmrl_m3

    # Drop degenerate parameters (5 predictors, better N:p ratio)
    python scripts/16b_bayesian_regression.py --model wmrl_m3 --drop-degenerate

    # Z-score all predictors (original behavior)
    python scripts/16b_bayesian_regression.py --model wmrl_m3 --standardize

    # With custom MCMC settings
    python scripts/16b_bayesian_regression.py --model wmrl_m3 --chains 4 --warmup 2000 --samples 4000

    # Dry-run: load data and print summary without running MCMC
    python scripts/16b_bayesian_regression.py --model wmrl_m3 --dry-run

Dependencies:
    pip install numpyro arviz  (arviz optional, used for Bayesian R-squared)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# JAX + NumPyro
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# Project imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from config import EXCLUDED_PARTICIPANTS

# Optional: arviz for Bayesian R-squared
try:
    import arviz as az
    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False

# Plotting style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300

# M3 parameter names (raw column names in MLE output)
M3_PARAM_COLS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa', 'epsilon']

# Outcome scales
OUTCOME_SCALES = {
    'lec_total': 'LEC-5 Total Trauma Exposure',
    'ies_total': 'IES-R Total PTSD Symptoms',
}

# Display-friendly parameter labels
PARAM_LABELS = {
    'alpha_pos': 'alpha+',
    'alpha_neg': 'alpha-',
    'phi': 'phi',
    'rho': 'rho',
    'capacity': 'K',
    'kappa': 'kappa',
    'epsilon': 'epsilon',
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(params_path: Path) -> pd.DataFrame:
    """Load M3 parameters and trauma scales, merge on participant ID.

    Returns DataFrame with columns: sona_id, [M3 params], lec_total, ies_total.
    """
    # Load M3 fitted parameters
    params_df = pd.read_csv(params_path)
    if 'participant_id' in params_df.columns:
        params_df = params_df.rename(columns={'participant_id': 'sona_id'})
    params_df['sona_id'] = params_df['sona_id'].astype(str)

    # Load trauma scales
    surveys_path = Path('output/mle/participant_surveys.csv')
    if not surveys_path.exists():
        raise FileNotFoundError(f"Surveys not found: {surveys_path}")
    surveys_df = pd.read_csv(surveys_path)
    surveys_df['sona_id'] = surveys_df['sona_id'].astype(str)

    # Exclude participants per config
    excluded_str = [str(x) for x in EXCLUDED_PARTICIPANTS]
    params_df = params_df[~params_df['sona_id'].isin(excluded_str)].copy()
    surveys_df = surveys_df[~surveys_df['sona_id'].isin(excluded_str)].copy()

    # Merge
    keep_cols = ['sona_id'] + M3_PARAM_COLS
    keep_cols = [c for c in keep_cols if c in params_df.columns]
    survey_cols = ['sona_id', 'lec_total', 'ies_total']
    survey_cols = [c for c in survey_cols if c in surveys_df.columns]

    df = params_df[keep_cols].merge(surveys_df[survey_cols], on='sona_id', how='inner')

    # Drop rows with missing values in any column
    df = df.dropna(subset=M3_PARAM_COLS + list(OUTCOME_SCALES.keys()))

    print(f"Loaded {len(df)} participants with complete M3 params + trauma scales")
    return df


def prepare_predictors(df: pd.DataFrame, drop_degenerate: bool = False,
                       standardize: bool = False) -> tuple:
    """Prepare predictor matrix for regression.

    Default: raw values with capacity rescaled from [1,7] to [0,1].
    With standardize=True: z-score all predictors instead.
    With drop_degenerate=True: remove alpha_neg and kappa before scaling.

    Returns:
        X: np.ndarray of shape (n, p), prepared predictors
        param_names: list of parameter names (in order of columns)
        scale_label: str describing the scaling mode
    """
    param_names = [p for p in M3_PARAM_COLS if p in df.columns]

    if drop_degenerate:
        param_names = [p for p in param_names if p not in ['alpha_neg', 'kappa']]

    X = df[param_names].values.astype(np.float64)

    if standardize:
        # Z-score all columns
        means = X.mean(axis=0)
        sds = X.std(axis=0)
        sds[sds < 1e-10] = 1.0  # avoid div-by-zero
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
# NUMPYRO MODEL
# ============================================================================

def bayesian_linear_model(X, y_mean, y_sd, standardize, likelihood='normal', y=None):
    """Bayesian linear regression with configurable likelihood.

    Priors (shared across likelihoods):
        intercept ~ Normal(mean(y), 2*sd(y))
        beta[k] ~ Normal(0, sd(y))  [raw mode] or Normal(0, 1) [z-scored mode]
        sigma ~ HalfNormal(sd(y))

    Likelihoods:
        normal:    y ~ Normal(mu, sigma)
        student-t: y ~ StudentT(nu, mu, sigma)  with nu ~ Gamma(2, 0.1)
        lognormal: y ~ Normal(mu, sigma)  [applied to log(y+1) pre-transformed data]

    Args:
        X: (n, p) array of predictors
        y_mean: mean of outcome (for centering intercept prior)
        y_sd: sd of outcome (for scaling priors)
        standardize: if True, use unit-scale beta prior (z-scored predictors)
        likelihood: 'normal', 'student-t', or 'lognormal'
        y: (n,) array of outcome values (None for prior predictive)
    """
    n_predictors = X.shape[1]

    intercept = numpyro.sample('intercept', dist.Normal(y_mean, 2.0 * y_sd))
    beta_sd = 1.0 if standardize else y_sd
    beta = numpyro.sample('beta', dist.Normal(0.0, beta_sd).expand([n_predictors]))
    sigma = numpyro.sample('sigma', dist.HalfNormal(y_sd))

    mu = intercept + X @ beta

    if likelihood == 'student-t':
        nu = numpyro.sample('nu', dist.Gamma(2.0, 0.1))
        numpyro.sample('y', dist.StudentT(nu, mu, sigma), obs=y)
    else:  # normal and lognormal (lognormal pre-transforms y before calling)
        numpyro.sample('y', dist.Normal(mu, sigma), obs=y)


# ============================================================================
# INFERENCE
# ============================================================================

def run_mcmc(X, y, outcome_name, standardize=False, likelihood='normal',
             n_chains=4, n_warmup=1000, n_samples=2000, seed=42):
    """Run NUTS MCMC for one outcome model.

    Returns:
        mcmc: fitted MCMC object
        samples: dict of posterior samples
    """
    y_mean = float(np.mean(y))
    y_sd = float(np.std(y))

    kernel = NUTS(bayesian_linear_model)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                num_chains=n_chains, progress_bar=True)

    rng_key = random.PRNGKey(seed)
    likelihood_label = f", likelihood={likelihood}" if likelihood != 'normal' else ''
    print(f"\n  Running MCMC for {outcome_name} ({n_chains} chains, "
          f"{n_warmup} warmup, {n_samples} samples{likelihood_label})...")

    mcmc.run(rng_key, X=jnp.array(X), y_mean=y_mean, y_sd=y_sd,
             standardize=standardize, likelihood=likelihood, y=jnp.array(y))

    mcmc.print_summary(exclude_deterministic=False)

    samples = mcmc.get_samples()
    return mcmc, samples


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

def plot_forest(summary_df, outcome_name, outcome_label, output_path, scale_label="raw [0,1]"):
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
    ax.set_title(f'Bayesian Regression: M3 Parameters -> {outcome_label}',
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
        description='Bayesian linear regression: M3 parameters -> trauma scales'
    )
    parser.add_argument('--model', type=str, default='wmrl_m3',
                        choices=['wmrl_m3'],
                        help='Model type (only wmrl_m3 supported)')
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
    parser.add_argument('--drop-degenerate', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Drop alpha_neg and kappa (default: True, use --no-drop-degenerate to keep)')
    parser.add_argument('--standardize', action='store_true',
                        help='Z-score all predictors instead of using raw [0,1] scale')
    parser.add_argument('--likelihood', type=str, default='normal',
                        choices=['normal', 'student-t', 'lognormal'],
                        help='Likelihood function (default: normal)')

    args = parser.parse_args()

    print("=" * 80)
    print("BAYESIAN LINEAR REGRESSION: M3 PARAMETERS -> TRAUMA SCALES")
    if args.likelihood != 'normal':
        print(f"  Likelihood: {args.likelihood}")
    print("=" * 80)

    # Create output directories
    output_dir = Path('output/regressions/bayesian')
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path('figures/regressions/bayesian')
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    params_path = Path(f'output/mle/{args.model}_individual_fits.csv')
    if not params_path.exists():
        print(f"ERROR: {params_path} not found")
        return

    df = load_data(params_path)

    # Prepare predictors
    X, param_names, scale_label = prepare_predictors(
        df, drop_degenerate=args.drop_degenerate, standardize=args.standardize
    )
    print(f"\nPredictors ({len(param_names)} M3 parameters, {scale_label} scale):")
    for i, p in enumerate(param_names):
        col = X[:, i]
        line = f"  {PARAM_LABELS.get(p, p):10s}: mean={col.mean():.4f}, range=[{col.min():.3f}, {col.max():.3f}]"
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

    # Set numpyro host device count for multi-chain
    numpyro.set_host_device_count(args.chains)

    # Run models
    all_summaries = []

    for outcome, label in OUTCOME_SCALES.items():
        print("\n" + "=" * 80)
        print(f"MODEL: {label}")
        print("=" * 80)

        y_raw = df[outcome].values.astype(np.float64)

        # Log-transform outcome for lognormal likelihood
        if args.likelihood == 'lognormal':
            y = np.log(y_raw + 1.0)
            print(f"  Transformed: log({outcome} + 1), range=[{y.min():.3f}, {y.max():.3f}]")
        else:
            y = y_raw

        mcmc, samples = run_mcmc(
            X, y, outcome,
            standardize=args.standardize,
            likelihood=args.likelihood,
            n_chains=args.chains,
            n_warmup=args.warmup,
            n_samples=args.samples,
            seed=args.seed
        )

        # Summarize posterior
        summary_df = summarize_posterior(samples, param_names, outcome)
        all_summaries.append(summary_df)

        # Bayesian R-squared (on the scale used for fitting)
        r2_mean, r2_hdi = compute_bayesian_r2(samples, X, y)
        print(f"\n  Bayesian R-squared: {r2_mean:.3f} [{r2_hdi[0]:.3f}, {r2_hdi[1]:.3f}]")

        # Add R-squared row to summary
        r2_row = pd.DataFrame([{
            'outcome': outcome,
            'parameter': 'Bayesian_R2',
            'mean': r2_mean,
            'sd': np.nan,
            'hdi_2.5%': r2_hdi[0],
            'hdi_97.5%': r2_hdi[1],
            'hdi_25%': np.nan,
            'hdi_75%': np.nan,
            'p_direction': np.nan,
            'interpretation': '',
        }])
        all_summaries[-1] = pd.concat([all_summaries[-1], r2_row], ignore_index=True)

        # Print Student-t degrees of freedom if applicable
        if args.likelihood == 'student-t' and 'nu' in samples:
            nu_samples = np.array(samples['nu'])
            nu_mean = float(np.mean(nu_samples))
            nu_hdi = compute_hdi(nu_samples, 0.95)
            print(f"\n  Student-t df (nu): {nu_mean:.1f} [{nu_hdi[0]:.1f}, {nu_hdi[1]:.1f}]")
            if nu_mean < 10:
                print(f"    -> Heavy tails detected (nu < 10 means outliers matter)")
            else:
                print(f"    -> Near-Normal tails (nu > 10, Normal is adequate)")

        # Print coefficient table
        coef_label = scale_label
        if args.likelihood == 'lognormal':
            coef_label += ", log(y+1) scale"
        coef_rows = summary_df[~summary_df['parameter'].isin(['intercept', 'sigma'])]
        print(f"\n  Coefficients ({coef_label}):")
        print(f"  {'Parameter':12s} {'Mean':>8s} {'SD':>8s} {'95% HDI':>22s} {'P(dir)':>8s} {'Interpretation'}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*22} {'-'*8} {'-'*20}")
        for _, row in coef_rows.iterrows():
            hdi_str = f"[{row['hdi_2.5%']:7.3f}, {row['hdi_97.5%']:7.3f}]"
            print(f"  {row['parameter']:12s} {row['mean']:8.3f} {row['sd']:8.3f} "
                  f"{hdi_str:>22s} {row['p_direction']:8.3f} {row['interpretation']}")

        # Forest plot
        plot_path = figures_dir / f'forest_plot_{outcome}.png'
        plot_forest(summary_df, outcome, label, plot_path, scale_label=scale_label)

    # Combine and save all summaries
    combined = pd.concat(all_summaries, ignore_index=True)
    summary_path = output_dir / 'bayesian_regression_summary.csv'
    combined.to_csv(summary_path, index=False)
    print(f"\n[SAVED] {summary_path}")

    # Final summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults: {output_dir}/")
    print(f"Figures: {figures_dir}/")
    print(f"\nNext steps:")
    print(f"  1. Check Rhat < 1.01 and n_eff in MCMC summary above")
    print(f"  2. Review bayesian_regression_summary.csv for all coefficients")
    print(f"  3. Examine forest plots for visual summary of effects")
    print(f"  4. Compare with frequentist results from script 16")
    print(f"\nInterpretation guide:")
    print(f"  - P(direction) > 0.95: strong directional evidence")
    print(f"  - P(direction) > 0.90: moderate directional evidence")
    print(f"  - 95% HDI excluding 0: 'significant' in Bayesian sense")
    if args.standardize:
        print(f"  - Coefficients are standardized (1 SD change in predictor)")
    else:
        print(f"  - Coefficients are on raw scale (min-to-max change in predictor)")
    if args.likelihood == 'lognormal':
        print(f"  - Outcome is log(y+1): coefficients are additive on log scale")
        print(f"    A beta of 0.5 means: 0->1 predictor change multiplies (y+1) by exp(0.5) = 1.65x")
    if args.likelihood == 'student-t':
        print(f"  - Student-t likelihood downweights outliers via estimated df (nu)")


if __name__ == '__main__':
    main()
