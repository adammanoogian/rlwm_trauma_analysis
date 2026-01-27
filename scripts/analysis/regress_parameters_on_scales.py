"""
Regression Analysis: Model Parameters on Trauma Scales

Performs linear regression analyses testing associations between:
- Fitted RL model parameters (α+, α-, β) 
- Trauma exposure measures (LEC-5)
- PTSD symptom measures (IES-R total and subscales)

Following computational psychiatry approach (e.g., Eckstein et al., 2022):
1. Individual regressions for each parameter
2. Multiple regression with subscales
3. Visualization of relationships
4. Reporting of effect sizes and confidence intervals

Usage:
    python scripts/analysis/regress_parameters_on_scales.py \
        --params output/v1/qlearning_jax_summary_20251122_200043.csv \
        --model qlearning

Author: RLWM Trauma Analysis
Date: 2026-01-22
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Try statsmodels for regression
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not installed. Using scipy for basic regressions only.")

warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300


def load_integrated_data(params_path: Path, model_type: str = 'qlearning') -> pd.DataFrame:
    """
    Load integrated dataset with parameters and trauma scales.
    
    Uses the integrated_parameters.csv if available, otherwise creates it.
    """
    # Check for pre-computed integrated file
    integrated_path = Path('figures/trauma_groups/integrated_parameters.csv')
    
    if integrated_path.exists():
        print(f"✓ Loading pre-integrated data from: {integrated_path}")
        df = pd.read_csv(integrated_path)
    else:
        print("Creating integrated dataset...")
        # Import the analysis script functions
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from analyze_parameters_by_trauma_group import (
            load_trauma_groups, 
            get_participant_order, 
            load_fitted_parameters
        )
        
        trauma_groups = load_trauma_groups()
        participant_order = get_participant_order()
        df = load_fitted_parameters(params_path, participant_order, trauma_groups)
    
    # Load full participant data to get IES-R subscales
    participant_data = pd.read_csv('output/summary_participant_metrics_all.csv')
    
    # Merge to get subscales
    merge_cols = ['sona_id', 'lec_total_events', 'lec_personal_events', 
                  'ies_total', 'ies_intrusion', 'ies_avoidance', 'ies_hyperarousal']
    merge_cols = [c for c in merge_cols if c in participant_data.columns]
    
    df = df.merge(
        participant_data[merge_cols],
        on='sona_id',
        how='left',
        suffixes=('', '_full')
    )
    
    print(f"\n✓ Loaded data for {len(df)} participants")
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
    """Create a formatted regression results table."""
    
    rows = []
    
    for param_name, param_results in results_dict.items():
        for predictor, res in param_results.items():
            if 'error' in res:
                continue
            
            row = {
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
            
            rows.append(row)
    
    df_table = pd.DataFrame(rows)
    
    # Save to CSV
    df_table.to_csv(output_path, index=False)
    print(f"✓ Saved regression table: {output_path}")
    
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
                           results: dict, output_path: Path):
    """Create scatter plot with regression line."""
    
    # Filter to complete cases
    data = df[[param_name, predictor]].dropna()
    
    if len(data) < 3:
        print(f"Skipping plot for {param_name} ~ {predictor}: insufficient data")
        return
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Scatter plot
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
    
    print(f"✓ Saved plot: {output_path}")


def format_label(col_name: str) -> str:
    """Format column name for plotting."""
    label_map = {
        'alpha_pos_mean': 'α+ (Positive Learning Rate)',
        'alpha_neg_mean': 'α- (Negative Learning Rate)',
        'beta_mean': 'β (Inverse Temperature)',
        'lec_total_events': 'LEC-5 Total Events',
        'lec_personal_events': 'LEC-5 Personal Events',
        'ies_total': 'IES-R Total Score',
        'ies_intrusion': 'IES-R Intrusion',
        'ies_avoidance': 'IES-R Avoidance',
        'ies_hyperarousal': 'IES-R Hyperarousal'
    }
    return label_map.get(col_name, col_name)


def plot_regression_matrix(df: pd.DataFrame, results_dict: dict, output_dir: Path):
    """Create matrix of all regression scatter plots."""
    
    param_cols = ['alpha_pos_mean', 'alpha_neg_mean', 'beta_mean']
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
            data = df[[param, pred]].dropna()
            
            if len(data) < 3:
                ax.text(0.5, 0.5, 'Insufficient\ndata', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_xlabel(format_label(pred))
                if j == 0:
                    ax.set_ylabel(format_label(param))
                continue
            
            # Scatter
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
    
    print(f"✓ Saved regression matrix: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Regression analysis of model parameters on trauma scales'
    )
    parser.add_argument(
        '--params',
        type=str,
        default='output/v1/qlearning_jax_summary_20251122_200043.csv',
        help='Path to fitted parameter summary CSV'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='qlearning',
        choices=['qlearning', 'wmrl'],
        help='Model type'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/regressions',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Setup
    params_path = Path(args.params)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("REGRESSION ANALYSIS: PARAMETERS ON TRAUMA SCALES")
    print("=" * 80)
    
    # Load data
    df = load_integrated_data(params_path, args.model)
    
    # Define parameter and predictor columns
    if args.model == 'qlearning':
        param_cols = ['alpha_pos_mean', 'alpha_neg_mean', 'beta_mean']
    else:  # wmrl
        param_cols = ['alpha_pos_mean', 'alpha_neg_mean', 'beta_mean', 
                     'wm_capacity_mean', 'wm_weight_mean']
    
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
                      
                print(f"  {format_label(pred):30s}: "
                      f"β={results['beta']:7.3f} (SE={results['se']:.3f}), "
                      f"t={results['t_stat']:6.2f}, "
                      f"p={results['p_value']:.4f}{sig}, "
                      f"r={results['r']:6.3f}, "
                      f"R²={results['r_squared']:.3f}, "
                      f"n={results['n']}")
                
                # Create individual plot
                plot_path = output_dir / f"{param}_{pred}.png"
                plot_regression_scatter(df, param, pred, results, plot_path)
    
    # Save results table
    table_path = output_dir / 'regression_results_simple.csv'
    df_table = create_regression_table(results_dict, table_path)
    
    print("\n" + df_table.to_string(index=False))
    
    # Create matrix plot
    plot_regression_matrix(df, results_dict, output_dir)
    
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
            
            print(f"  Model: R² = {results['r_squared']:.3f}, "
                  f"Adj R² = {results['adj_r_squared']:.3f}")
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
                      f"β={coef_info['beta']:7.3f} (SE={coef_info['se']:.3f}), "
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
        multi_path = output_dir / 'regression_results_multiple.csv'
        df_multi.to_csv(multi_path, index=False)
        print(f"\n✓ Saved multiple regression results: {multi_path}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print("\nNext steps:")
    print("1. Review regression_results_simple.csv for all univariate associations")
    print("2. Check regression_matrix_all.png for visual overview")
    print("3. Review individual scatter plots for significant associations")
    if HAS_STATSMODELS:
        print("4. Review regression_results_multiple.csv for subscale-specific effects")
    print("\nInterpretation notes:")
    print("- Focus on effect sizes (β, r) not just p-values")
    print("- With small N, consider p < 0.10 as marginally significant")
    print("- Check for outliers in individual scatter plots")
    print("- VIF > 5 indicates multicollinearity concerns")


if __name__ == '__main__':
    main()
