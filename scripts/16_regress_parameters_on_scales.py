#!/usr/bin/env python
"""
16: Regress Parameters on Scales
================================

Regression analysis of model parameters on continuous trauma scales.

Performs linear regression analyses testing associations between:
- Fitted RL model parameters (α+, α-, β)
- Trauma exposure measures (LEC-5)
- PTSD symptom measures (IES-R total and subscales)

Following computational psychiatry approach (e.g., Eckstein et al., 2022):
1. Individual regressions for each parameter
2. Multiple regression with subscales
3. Visualization of relationships
4. Reporting of effect sizes and confidence intervals

Key Analyses:
    1. Univariate regressions (each parameter ~ each scale)
    2. Multivariate regressions (parameter ~ multiple predictors)
    3. Hierarchical regressions (incremental variance explained)
    4. Visualization of regression results

Predictors:
    - LEC-5/LESS Total Events
    - IES-R Total Score
    - IES-R Subscales (Intrusion, Avoidance, Hyperarousal)
    - Demographics (optional: age, etc.)

Outcome Variables:
    Q-Learning: α₊, α₋, ε
    WM-RL: α₊, α₋, φ, ρ, K, ε, (κ for M3)

Statistical Approach:
    - OLS regression with robust standard errors
    - False Discovery Rate (FDR) correction for multiple comparisons
    - Effect size reporting (R², β coefficients)
    - Diagnostic plots for regression assumptions

Inputs:
    - output/mle_results/<model>_individual_fits.csv
    - output/summary_participant_metrics.csv

Outputs:
    - output/regressions/<model>_univariate_results.csv
    - output/regressions/<model>_multivariate_results.csv
    - figures/regressions/<model>_regression_coefficients.png
    - figures/regressions/<model>_scatter_matrix.png

Usage:
    # Analyze Q-learning parameters
    python scripts/16_regress_parameters_on_scales.py --model qlearning

    # Analyze WM-RL parameters
    python scripts/16_regress_parameters_on_scales.py --model wmrl

    # Run all models
    python scripts/16_regress_parameters_on_scales.py --model all

    # With specific predictors
    python scripts/16_regress_parameters_on_scales.py --model wmrl \
        --predictors ies_total less_total_events

    # Include covariates
    python scripts/16_regress_parameters_on_scales.py --model wmrl \
        --covariates age gender

Note:
    With small N, interpret regression results cautiously.
    Focus on effect sizes rather than p-values alone.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
import warnings

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EXCLUDED_PARTICIPANTS

# Import plotting utilities for color-by functionality
sys.path.insert(0, str(Path(__file__).parent))
from utils.plotting_utils import get_color_palette, add_colored_scatter, TRAUMA_GROUP_COLORS

# Try statsmodels for regression
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not installed. Using scipy for basic regressions only.")

warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300


def load_integrated_data(params_path: Path, model_type: str = 'qlearning',
                         min_accuracy: float = None, max_epsilon: float = None) -> pd.DataFrame:
    """
    Load integrated dataset with parameters and trauma scales.

    Loads parameters directly from params_path and merges with participant data.

    Args:
        params_path: Path to fitted parameters CSV
        model_type: 'qlearning' or 'wmrl'
        min_accuracy: Optional minimum accuracy threshold (0-1) for inclusion
    """
    # Load parameters directly from the provided path
    print(f"Loading fitted parameters from: {params_path}")
    params_df = pd.read_csv(params_path)

    # Standardize participant ID column name
    if 'participant_id' in params_df.columns:
        params_df = params_df.rename(columns={'participant_id': 'sona_id'})

    # Convert sona_id to string for consistent merging
    params_df['sona_id'] = params_df['sona_id'].astype(str)

    # Rename parameter columns to match expected format (add _mean suffix)
    param_rename = {}
    if model_type == 'qlearning':
        if 'alpha_pos' in params_df.columns:
            param_rename['alpha_pos'] = 'alpha_pos_mean'
        if 'alpha_neg' in params_df.columns:
            param_rename['alpha_neg'] = 'alpha_neg_mean'
        if 'beta' in params_df.columns:
            param_rename['beta'] = 'beta_mean'
        if 'epsilon' in params_df.columns:
            param_rename['epsilon'] = 'epsilon_mean'
    else:  # wmrl or wmrl_m3
        if 'alpha_pos' in params_df.columns:
            param_rename['alpha_pos'] = 'alpha_pos_mean'
        if 'alpha_neg' in params_df.columns:
            param_rename['alpha_neg'] = 'alpha_neg_mean'
        if 'phi' in params_df.columns:
            param_rename['phi'] = 'phi_mean'
        if 'rho' in params_df.columns:
            param_rename['rho'] = 'rho_mean'
        if 'capacity' in params_df.columns:
            param_rename['capacity'] = 'wm_capacity_mean'
        if 'kappa' in params_df.columns:
            param_rename['kappa'] = 'kappa_mean'
        if 'epsilon' in params_df.columns:
            param_rename['epsilon'] = 'epsilon_mean'

    params_df = params_df.rename(columns=param_rename)
    print(f"  Loaded {len(params_df)} participant fits")

    # Load trauma scales from participant_surveys.csv (uses same IDs as MLE fits)
    surveys_path = Path('output/mle/participant_surveys.csv')
    if surveys_path.exists():
        participant_data = pd.read_csv(surveys_path)
        # Rename columns to match expected names
        rename_map = {
            'lec_total': 'lec_total_events',
            'lec_personal': 'lec_personal_events'
        }
        participant_data = participant_data.rename(columns=rename_map)
    else:
        # Fallback to summary_participant_metrics_all.csv
        participant_data = pd.read_csv('output/summary_participant_metrics_all.csv')
    participant_data['sona_id'] = participant_data['sona_id'].astype(str)

    # Load trauma group assignments for --color-by hypothesis_group support
    groups_path = Path('output/trauma_groups/group_assignments.csv')
    if groups_path.exists():
        groups_df = pd.read_csv(groups_path)
        groups_df['sona_id'] = groups_df['sona_id'].astype(str)
        participant_data = participant_data.merge(
            groups_df[['sona_id', 'hypothesis_group']],
            on='sona_id',
            how='left'
        )

    # Load demographics for --color-by gender/age support
    demographics_path = Path('output/parsed_demographics.csv')
    if demographics_path.exists():
        demographics_df = pd.read_csv(demographics_path)
        demographics_df['sona_id'] = demographics_df['sona_id'].astype(str)
        participant_data = participant_data.merge(
            demographics_df[['sona_id', 'gender', 'age_years']],
            on='sona_id',
            how='left'
        )

    # Load accuracy data separately (from task_trials to compute per-participant accuracy)
    accuracy_data = None
    trials_path = Path('output/task_trials_long_all_participants.csv')
    if trials_path.exists():
        trials_df = pd.read_csv(trials_path)
        trials_df['sona_id'] = trials_df['sona_id'].astype(str)
        accuracy_data = trials_df.groupby('sona_id')['correct'].mean().reset_index()
        accuracy_data.columns = ['sona_id', 'accuracy_overall']
        # Merge accuracy into participant_data
        participant_data = participant_data.merge(accuracy_data, on='sona_id', how='left')

    # Exclude participants based on data quality (convert to string for comparison)
    excluded_str = [str(x) for x in EXCLUDED_PARTICIPANTS]
    participant_data = participant_data[~participant_data['sona_id'].isin(excluded_str)].copy()
    print(f"  {len(participant_data)} participants after data quality exclusions")

    # Optional accuracy-based exclusion
    if min_accuracy is not None:
        if 'accuracy_overall' not in participant_data.columns:
            print(f"  Warning: accuracy_overall column not found, cannot apply {min_accuracy:.0%} cutoff")
        else:
            low_accuracy_mask = participant_data['accuracy_overall'] < min_accuracy
            low_accuracy_ids = participant_data.loc[low_accuracy_mask, 'sona_id'].tolist()
            low_accuracy_values = participant_data.loc[low_accuracy_mask, ['sona_id', 'accuracy_overall']]
            print(f"  Excluding {len(low_accuracy_ids)} participants below {min_accuracy:.0%} accuracy:")
            for _, row in low_accuracy_values.iterrows():
                print(f"    - {row['sona_id']}: {row['accuracy_overall']:.2%}")
            # Exclude from both dataframes
            params_df = params_df[~params_df['sona_id'].isin(low_accuracy_ids)]
            participant_data = participant_data[~participant_data['sona_id'].isin(low_accuracy_ids)]

    # Optional epsilon-based exclusion (alternative performance filter using MLE epsilon parameter)
    if max_epsilon is not None:
        if 'epsilon' in params_df.columns or 'epsilon_mean' in params_df.columns:
            eps_col = 'epsilon' if 'epsilon' in params_df.columns else 'epsilon_mean'
            high_epsilon_mask = params_df[eps_col] > max_epsilon
            high_epsilon_ids = params_df.loc[high_epsilon_mask, 'sona_id'].tolist()
            high_epsilon_values = params_df.loc[high_epsilon_mask, ['sona_id', eps_col]]
            print(f"  Excluding {len(high_epsilon_ids)} participants with epsilon > {max_epsilon:.2f}:")
            for _, row in high_epsilon_values.iterrows():
                print(f"    - {row['sona_id']}: epsilon={row[eps_col]:.3f}")
            # Exclude from params_df (participant_data doesn't have epsilon)
            params_df = params_df[~params_df['sona_id'].isin(high_epsilon_ids)]
        else:
            print(f"  Warning: epsilon column not found, cannot apply max_epsilon={max_epsilon:.2f} filter")

    # Merge parameters with participant data
    merge_cols = ['sona_id', 'lec_total_events', 'lec_personal_events',
                  'ies_total', 'ies_intrusion', 'ies_avoidance', 'ies_hyperarousal',
                  'accuracy_overall', 'hypothesis_group', 'gender', 'age_years']
    merge_cols = [c for c in merge_cols if c in participant_data.columns]

    df = params_df.merge(
        participant_data[merge_cols],
        on='sona_id',
        how='inner'
    )

    print(f"\nLoaded data for {len(df)} participants")
    print(f"  {df['alpha_pos_mean'].notna().sum()} with fitted parameters")

    return df


def run_simple_regression(df: pd.DataFrame, param_name: str, predictor: str) -> dict:
    """
    Run simple linear regression: param ~ predictor

    Returns dictionary with regression results.
    """
    # Filter to complete cases
    data = df[[param_name, predictor]].dropna()

    if len(data) < 3:
        return {'error': 'Insufficient data', 'n': len(data)}

    X = data[predictor].values
    y = data[param_name].values

    if HAS_STATSMODELS:
        # Use statsmodels for full output
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()

        results = {
            'n': len(data),
            'beta': model.params[1],
            'se': model.bse[1],
            'ci_lower': model.conf_int()[1, 0],
            'ci_upper': model.conf_int()[1, 1],
            't_stat': model.tvalues[1],
            'p_value': model.pvalues[1],
            'r_squared': model.rsquared,
            'f_stat': model.fvalue,
            'residuals': model.resid,
            'fitted': model.fittedvalues,
            'model': model
        }

        # Residual diagnostics
        # 1. Shapiro-Wilk test for normality of residuals
        if len(model.resid) >= 3:
            sw_stat, sw_p = stats.shapiro(model.resid)
            results['shapiro_w'] = sw_stat
            results['shapiro_p'] = sw_p

        # 2. Breusch-Pagan test for homoscedasticity
        try:
            bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(model.resid, X_with_const)
            results['bp_stat'] = bp_lm
            results['bp_p'] = bp_lm_p
        except Exception:
            results['bp_stat'] = np.nan
            results['bp_p'] = np.nan
    else:
        # Use scipy for basic regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

        results = {
            'n': len(data),
            'beta': slope,
            'se': std_err,
            'p_value': p_value,
            'r_squared': r_value**2,
            't_stat': slope / std_err if std_err > 0 else np.nan
        }

    # Add Pearson correlation
    r, p = stats.pearsonr(X, y)
    results['r'] = r
    results['r_p'] = p

    return results


def run_multiple_regression(df: pd.DataFrame, param_name: str, predictors: list) -> dict:
    """
    Run multiple linear regression: param ~ predictor1 + predictor2 + ...

    Returns dictionary with regression results.
    """
    if not HAS_STATSMODELS:
        return {'error': 'statsmodels required for multiple regression'}

    # Filter to complete cases
    data = df[[param_name] + predictors].dropna()

    if len(data) < len(predictors) + 2:
        return {'error': 'Insufficient data', 'n': len(data)}

    X = data[predictors].values
    y = data[param_name].values

    # Add constant
    X_with_const = sm.add_constant(X)

    # Fit model
    model = sm.OLS(y, X_with_const).fit()

    # Extract results
    results = {
        'n': len(data),
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'f_stat': model.fvalue,
        'f_pvalue': model.f_pvalue,
        'aic': model.aic,
        'bic': model.bic,
        'coefficients': {},
        'model': model
    }

    # Extract coefficient info for each predictor
    for i, pred in enumerate(['Intercept'] + predictors):
        idx = i
        results['coefficients'][pred] = {
            'beta': model.params[idx],
            'se': model.bse[idx],
            'ci_lower': model.conf_int()[idx, 0],
            'ci_upper': model.conf_int()[idx, 1],
            't_stat': model.tvalues[idx],
            'p_value': model.pvalues[idx]
        }

    # Check multicollinearity (VIF)
    if len(predictors) > 1:
        vif_data = pd.DataFrame()
        vif_data['predictor'] = predictors
        vif_data['VIF'] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        results['vif'] = vif_data

    return results


def create_regression_table(results_dict: dict, output_path: Path):
    """Create a formatted regression results table with FDR correction and diagnostics."""

    rows = []

    for param_name, param_results in results_dict.items():
        for predictor, res in param_results.items():
            if 'error' in res:
                continue

            row = {
                'Section': f"{format_label(param_name)} ~ {format_label(predictor)}",
                'Parameter': param_name,
                'Predictor': predictor,
                'N': res['n'],
                'β': f"{res['beta']:.3f}",
                'SE': f"{res['se']:.3f}",
                't': f"{res['t_stat']:.2f}",
                'p': format_pvalue(res['p_value']),
                'R²': f"{res['r_squared']:.3f}"
            }

            if 'ci_lower' in res:
                row['95% CI'] = f"[{res['ci_lower']:.3f}, {res['ci_upper']:.3f}]"

            if 'r' in res:
                row['r'] = f"{res['r']:.3f}"

            # FDR-corrected p-value (added after all regressions computed)
            if 'p_fdr' in res:
                row['p_fdr'] = format_pvalue(res['p_fdr'])
                row['sig_fdr'] = res.get('sig_fdr', False)

            # Residual diagnostics
            if 'shapiro_w' in res:
                row['shapiro_w'] = f"{res['shapiro_w']:.3f}"
                row['shapiro_p'] = f"{res['shapiro_p']:.4f}"
                row['residuals_normal'] = res['shapiro_p'] > 0.05

            if 'bp_stat' in res and not np.isnan(res['bp_stat']):
                row['bp_stat'] = f"{res['bp_stat']:.3f}"
                row['bp_p'] = f"{res['bp_p']:.4f}"
                row['homoscedastic'] = res['bp_p'] > 0.05

            rows.append(row)

    df_table = pd.DataFrame(rows)

    # Sort by Section for structured output (groups each scale x parameter regression)
    if 'Section' in df_table.columns:
        df_table = df_table.sort_values('Section')

    # Save to CSV
    df_table.to_csv(output_path, index=False)
    print(f"Saved regression table: {output_path}")

    return df_table


def format_pvalue(p: float) -> str:
    """Format p-value with stars for significance."""
    if p < 0.001:
        return f"{p:.4f}***"
    elif p < 0.01:
        return f"{p:.3f}**"
    elif p < 0.05:
        return f"{p:.3f}*"
    else:
        return f"{p:.3f}"


def plot_regression_scatter(df: pd.DataFrame, param_name: str, predictor: str,
                           results: dict, output_path: Path,
                           color_by: str = None, color_palette: dict = None):
    """Create scatter plot with regression line."""

    # Filter to complete cases
    if color_by is not None:
        data = df[[param_name, predictor, color_by]].dropna()
    else:
        data = df[[param_name, predictor]].dropna()

    if len(data) < 3:
        print(f"Skipping plot for {param_name} ~ {predictor}: insufficient data")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    # Scatter plot (colored or single color)
    if color_by is not None and color_palette is not None:
        # Use colored scatter with legend
        add_colored_scatter(ax, predictor, param_name, data, color_by, color_palette,
                          alpha=0.6, s=80, edgecolors='black', linewidths=0.5)
    else:
        # Single color scatter (legacy behavior)
        ax.scatter(data[predictor], data[param_name],
                  alpha=0.6, s=80, edgecolors='black', linewidths=0.5)

    # Regression line
    X = data[predictor].values
    y = data[param_name].values

    if HAS_STATSMODELS and 'fitted' in results:
        # Use model-fitted values
        sorted_idx = np.argsort(X)
        ax.plot(X[sorted_idx], results['fitted'][sorted_idx],
               'r-', linewidth=2, label='Regression line')

        # Add confidence interval if available
        if 'model' in results:
            pred = results['model'].get_prediction()
            pred_summary = pred.summary_frame(alpha=0.05)
            ax.fill_between(X[sorted_idx],
                           pred_summary['obs_ci_lower'][sorted_idx],
                           pred_summary['obs_ci_upper'][sorted_idx],
                           alpha=0.2, color='red', label='95% CI')
    else:
        # Simple line fit
        slope = results['beta']
        intercept = np.mean(y) - slope * np.mean(X)
        x_line = np.array([X.min(), X.max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r-', linewidth=2, label='Regression line')

    # Labels
    ax.set_xlabel(format_label(predictor), fontsize=12)
    ax.set_ylabel(format_label(param_name), fontsize=12)

    # Stats annotation
    stats_text = f"r = {results['r']:.3f}, p = {results['p_value']:.3f}\n"
    stats_text += f"β = {results['beta']:.3f} ± {results['se']:.3f}\n"
    stats_text += f"R² = {results['r_squared']:.3f}, n = {results['n']}"

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plot: {output_path}")


def format_label(col_name: str) -> str:
    """Format column name for plotting."""
    label_map = {
        # Q-learning parameters
        'alpha_pos_mean': 'alpha+ (Positive Learning Rate)',
        'alpha_neg_mean': 'alpha- (Negative Learning Rate)',
        'beta_mean': 'beta (Inverse Temperature)',
        'epsilon_mean': 'epsilon (Random Responding)',
        # WM-RL parameters
        'phi_mean': 'phi (WM Decay Rate)',
        'rho_mean': 'rho (WM Weight)',
        'wm_capacity_mean': 'K (WM Capacity)',
        'kappa_mean': 'kappa (Perseveration)',
        # Trauma scales
        'lec_total_events': 'LEC-5 Total Events',
        'lec_personal_events': 'LEC-5 Personal Events',
        'ies_total': 'IES-R Total Score',
        'ies_intrusion': 'IES-R Intrusion',
        'ies_avoidance': 'IES-R Avoidance',
        'ies_hyperarousal': 'IES-R Hyperarousal'
    }
    return label_map.get(col_name, col_name)


def plot_regression_matrix(df: pd.DataFrame, results_dict: dict, output_dir: Path,
                           param_cols: list = None, color_by: str = None,
                           color_palette: dict = None):
    """Create matrix of all regression scatter plots."""

    # Default param_cols for backwards compatibility
    if param_cols is None:
        param_cols = ['alpha_pos_mean', 'alpha_neg_mean', 'beta_mean']

    # Filter to available params
    param_cols = [p for p in param_cols if p in df.columns]

    predictor_cols = ['lec_total_events', 'ies_total',
                     'ies_intrusion', 'ies_avoidance', 'ies_hyperarousal']

    # Filter to available predictors
    predictor_cols = [p for p in predictor_cols if p in df.columns]

    n_params = len(param_cols)
    n_preds = len(predictor_cols)

    fig, axes = plt.subplots(n_params, n_preds, figsize=(4*n_preds, 4*n_params))

    if n_params == 1:
        axes = axes.reshape(1, -1)
    if n_preds == 1:
        axes = axes.reshape(-1, 1)

    for i, param in enumerate(param_cols):
        for j, pred in enumerate(predictor_cols):
            ax = axes[i, j]

            # Get data
            if color_by is not None:
                data = df[[param, pred, color_by]].dropna()
            else:
                data = df[[param, pred]].dropna()

            if len(data) < 3:
                ax.text(0.5, 0.5, 'Insufficient\ndata',
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_xlabel(format_label(pred))
                if j == 0:
                    ax.set_ylabel(format_label(param))
                continue

            # Scatter (colored or single color)
            if color_by is not None and color_palette is not None:
                # Use colored scatter (no legend in matrix cells)
                add_colored_scatter(ax, pred, param, data, color_by, color_palette,
                                  alpha=0.6, s=50, show_legend=False)
            else:
                # Single color scatter (legacy behavior)
                ax.scatter(data[pred], data[param], alpha=0.6, s=50)

            # Get results
            results = results_dict.get(param, {}).get(pred, {})

            if 'beta' in results:
                # Regression line
                X = data[pred].values
                y = data[param].values
                slope = results['beta']
                intercept = np.mean(y) - slope * np.mean(X)
                x_line = np.array([X.min(), X.max()])
                y_line = slope * x_line + intercept

                # Color by significance
                color = 'red' if results['p_value'] < 0.05 else 'gray'
                linestyle = '-' if results['p_value'] < 0.05 else '--'
                ax.plot(x_line, y_line, color=color, linestyle=linestyle, linewidth=2)

                # Stats text
                stats_text = f"r={results['r']:.2f}\np={results['p_value']:.3f}"
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                       fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            ax.set_xlabel(format_label(pred) if i == n_params-1 else '')
            if j == 0:
                ax.set_ylabel(format_label(param))

    plt.tight_layout()
    output_path = output_dir / 'regression_matrix_all.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved regression matrix: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Regression analysis of model parameters on trauma scales'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='qlearning',
        choices=['qlearning', 'wmrl', 'wmrl_m3', 'all'],
        help='Model type (qlearning, wmrl, wmrl_m3, or all)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/regressions',
        help='Base output directory for CSV results (model subdirectories created automatically)'
    )
    parser.add_argument(
        '--figures-dir',
        type=str,
        default='figures/regressions',
        help='Base output directory for figures (model subdirectories created automatically)'
    )
    parser.add_argument(
        '--min-accuracy',
        type=float,
        default=None,
        help='Minimum accuracy threshold (0-1) for participant inclusion'
    )
    parser.add_argument(
        '--max-epsilon',
        type=float,
        default=None,
        help='Maximum epsilon threshold for participant inclusion (alternative to min-accuracy)'
    )
    parser.add_argument(
        '--color-by',
        type=str,
        default=None,
        help='Column name to color scatter plots by (e.g., hypothesis_group, gender)'
    )

    args = parser.parse_args()

    # Setup
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    base_figures_dir = Path(args.figures_dir)
    base_figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("REGRESSION ANALYSIS: PARAMETERS ON TRAUMA SCALES")
    print("=" * 80)
    if args.min_accuracy is not None:
        print(f"[!] Accuracy filter: excluding participants below {args.min_accuracy:.0%}")
    if args.max_epsilon is not None:
        print(f"[!] Epsilon filter: excluding participants with epsilon > {args.max_epsilon:.2f}")
    if args.color_by is not None:
        print(f"[!] Color-by: {args.color_by}")

    # Determine models to run
    models_to_run = ['qlearning', 'wmrl', 'wmrl_m3'] if args.model == 'all' else [args.model]

    # Loop over models
    for model in models_to_run:
        print("\n" + "=" * 80)
        print(f"MODEL: {model.upper()}")
        print("=" * 80)

        # Create model-specific output directories (CSV results and figures)
        model_output_dir = base_output_dir / model
        model_output_dir.mkdir(parents=True, exist_ok=True)

        model_figures_dir = base_figures_dir / model
        model_figures_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect params path from model name
        params_path = Path(f'output/mle/{model}_individual_fits.csv')
        if not params_path.exists():
            print(f"Warning: {params_path} not found, skipping {model}")
            continue

        # Load data
        df = load_integrated_data(params_path, model, args.min_accuracy, args.max_epsilon)

        # Validate color-by column if specified
        color_palette = None
        if args.color_by is not None:
            if args.color_by not in df.columns:
                print(f"Warning: Column '{args.color_by}' not found in data, skipping color-by")
                print(f"Available columns: {list(df.columns)}")
                color_by = None
            else:
                color_by = args.color_by
                # Generate palette (use custom colors for hypothesis_group)
                custom_colors = TRAUMA_GROUP_COLORS if color_by == 'hypothesis_group' else None
                color_palette = get_color_palette(df, color_by, custom_colors=custom_colors)
                print(f"  Color palette: {color_palette}")
        else:
            color_by = None

        # Define parameter columns based on model
        if model == 'qlearning':
            param_cols = ['alpha_pos_mean', 'alpha_neg_mean', 'epsilon_mean']
        elif model == 'wmrl':
            param_cols = ['alpha_pos_mean', 'alpha_neg_mean', 'phi_mean', 'rho_mean',
                          'wm_capacity_mean', 'epsilon_mean']
        elif model == 'wmrl_m3':
            param_cols = ['alpha_pos_mean', 'alpha_neg_mean', 'phi_mean', 'rho_mean',
                          'wm_capacity_mean', 'kappa_mean', 'epsilon_mean']
        else:
            print(f"Unknown model: {model}")
            continue

        # Only use parameters that exist in the data
        param_cols = [p for p in param_cols if p in df.columns]

        predictor_cols = ['lec_total_events', 'lec_personal_events',
                         'ies_total', 'ies_intrusion', 'ies_avoidance', 'ies_hyperarousal']

        # Only use predictors that exist
        predictor_cols = [p for p in predictor_cols if p in df.columns]

        print(f"\nParameters: {param_cols}")
        print(f"Predictors: {predictor_cols}")

        # Run simple regressions
        print("\n" + "=" * 80)
        print("SIMPLE REGRESSIONS (UNIVARIATE)")
        print("=" * 80)

        results_dict = {}

        for param in param_cols:
            results_dict[param] = {}
            print(f"\n{format_label(param)}:")
            print("-" * 60)

            for pred in predictor_cols:
                results = run_simple_regression(df, param, pred)
                results_dict[param][pred] = results

                if 'error' in results:
                    print(f"  {pred}: {results['error']} (n={results.get('n', 0)})")
                else:
                    sig = '***' if results['p_value'] < 0.001 else \
                          '**' if results['p_value'] < 0.01 else \
                          '*' if results['p_value'] < 0.05 else ''

                    # Build diagnostic flags
                    diag_flags = []
                    if 'shapiro_p' in results and results['shapiro_p'] < 0.05:
                        diag_flags.append('non-normal resid')
                    if 'bp_p' in results and not np.isnan(results['bp_p']) and results['bp_p'] < 0.05:
                        diag_flags.append('heteroscedastic')
                    diag_str = f"  [{', '.join(diag_flags)}]" if diag_flags else ''

                    print(f"  {format_label(pred):30s}: "
                          f"beta={results['beta']:7.3f} (SE={results['se']:.3f}), "
                          f"t={results['t_stat']:6.2f}, "
                          f"p={results['p_value']:.4f}{sig}, "
                          f"r={results['r']:6.3f}, "
                          f"R2={results['r_squared']:.3f}, "
                          f"n={results['n']}{diag_str}")

                    # Create individual plot (save to figures directory)
                    plot_path = model_figures_dir / f"{param}_{pred}.png"
                    plot_regression_scatter(df, param, pred, results, plot_path,
                                          color_by=color_by, color_palette=color_palette)

        # Apply FDR correction across all simple regressions
        all_pvalues = []
        all_keys = []  # (param, predictor) tuples to map back
        for param in param_cols:
            for pred in predictor_cols:
                res = results_dict.get(param, {}).get(pred, {})
                if 'error' not in res and 'p_value' in res:
                    all_pvalues.append(res['p_value'])
                    all_keys.append((param, pred))

        if HAS_STATSMODELS and len(all_pvalues) > 1:
            reject, p_fdr, _, _ = multipletests(all_pvalues, method='fdr_bh', alpha=0.05)
            for (param, pred), pf, sig in zip(all_keys, p_fdr, reject):
                results_dict[param][pred]['p_fdr'] = pf
                results_dict[param][pred]['sig_fdr'] = bool(sig)

            n_sig_uncorrected = sum(1 for p in all_pvalues if p < 0.05)
            n_sig_fdr = sum(reject)
            print(f"\n  FDR correction (Benjamini-Hochberg): {len(all_pvalues)} tests")
            print(f"  Significant (uncorrected p < 0.05): {n_sig_uncorrected}")
            print(f"  Significant (FDR q < 0.05): {n_sig_fdr}")

        # Save results table
        table_path = model_output_dir / 'regression_results_simple.csv'
        df_table = create_regression_table(results_dict, table_path)

        # Note: Skip console display due to Unicode encoding issues on Windows
        # Users can view the CSV file directly

        # Create matrix plot (save to figures directory)
        plot_regression_matrix(df, results_dict, model_figures_dir, param_cols,
                             color_by=color_by, color_palette=color_palette)

        # Multiple regressions with IES-R subscales
        if HAS_STATSMODELS and all(c in df.columns for c in ['ies_intrusion', 'ies_avoidance', 'ies_hyperarousal']):
            print("\n" + "=" * 80)
            print("MULTIPLE REGRESSION: IES-R SUBSCALES")
            print("=" * 80)

            subscales = ['ies_intrusion', 'ies_avoidance', 'ies_hyperarousal']

            multi_results = {}

            for param in param_cols:
                print(f"\n{format_label(param)}:")
                print("-" * 60)

                results = run_multiple_regression(df, param, subscales)
                multi_results[param] = results

                if 'error' in results:
                    print(f"  Error: {results['error']}")
                    continue

                print(f"  Model: R2 = {results['r_squared']:.3f}, "
                      f"Adj R2 = {results['adj_r_squared']:.3f}")
                print(f"  F({len(subscales)}, {results['n']-len(subscales)-1}) = {results['f_stat']:.2f}, "
                      f"p = {results['f_pvalue']:.4f}")
                print(f"  AIC = {results['aic']:.1f}, BIC = {results['bic']:.1f}")
                print(f"\n  Coefficients:")

                for pred, coef_info in results['coefficients'].items():
                    if pred == 'Intercept':
                        continue
                    sig = '***' if coef_info['p_value'] < 0.001 else \
                          '**' if coef_info['p_value'] < 0.01 else \
                          '*' if coef_info['p_value'] < 0.05 else ''

                    print(f"    {format_label(pred):30s}: "
                          f"beta={coef_info['beta']:7.3f} (SE={coef_info['se']:.3f}), "
                          f"t={coef_info['t_stat']:6.2f}, "
                          f"p={coef_info['p_value']:.4f}{sig}")

                if 'vif' in results:
                    print(f"\n  Multicollinearity (VIF):")
                    for _, row in results['vif'].iterrows():
                        warning = " (HIGH)" if row['VIF'] > 5 else ""
                        print(f"    {format_label(row['predictor']):30s}: {row['VIF']:.2f}{warning}")

            # Save multiple regression results
            multi_rows = []
            for param, results in multi_results.items():
                if 'error' in results:
                    continue

                for pred, coef_info in results['coefficients'].items():
                    if pred == 'Intercept':
                        continue

                    multi_rows.append({
                        'Parameter': param,
                        'Predictor': pred,
                        'β': f"{coef_info['beta']:.3f}",
                        'SE': f"{coef_info['se']:.3f}",
                        '95% CI': f"[{coef_info['ci_lower']:.3f}, {coef_info['ci_upper']:.3f}]",
                        't': f"{coef_info['t_stat']:.2f}",
                        'p': format_pvalue(coef_info['p_value'])
                    })

            df_multi = pd.DataFrame(multi_rows)
            multi_path = model_output_dir / 'regression_results_multiple.csv'
            df_multi.to_csv(multi_path, index=False)
            print(f"\nSaved multiple regression results: {multi_path}")

        print(f"\nCompleted analysis for {model}")
        print(f"   CSV results: {model_output_dir}/")
        print(f"   Figures: {model_figures_dir}/")

    # Final summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nCSV results: {base_output_dir}/")
    print(f"Figures: {base_figures_dir}/")
    if args.model == 'all':
        print("  Model-specific subdirectories:")
        for model in models_to_run:
            model_dir = base_output_dir / model
            fig_dir = base_figures_dir / model
            if model_dir.exists():
                print(f"    - {model}/ (CSV + figures)")
    print("\nNext steps:")
    print("1. Review regression_results_simple.csv for all univariate associations")
    print("2. Check p_fdr column for FDR-corrected significance")
    print("3. Check regression_matrix_all.png for visual overview")
    print("4. Review individual scatter plots for significant associations")
    if HAS_STATSMODELS:
        print("5. Review regression_results_multiple.csv for subscale-specific effects")
    print("\nInterpretation notes:")
    print("- Focus on effect sizes (beta, r) not just p-values")
    print("- p_fdr: Benjamini-Hochberg corrected across all param x predictor tests")
    print("- shapiro_p < 0.05: residuals deviate from normality (consider robust SE)")
    print("- bp_p < 0.05: heteroscedasticity detected (consider HC3 robust SE)")
    print("- With small N, consider p < 0.10 as marginally significant")
    print("- Check for outliers in individual scatter plots")
    print("- VIF > 5 indicates multicollinearity concerns")


if __name__ == '__main__':
    main()
