"""
Statistical tests for thesis analysis.

Includes:
- Assumption checks (normality, homogeneity of variance)
- Mixed ANOVAs
- Linear regressions
- Effect size calculations
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings

# Statistical packages
try:
    import pingouin as pg
except ImportError:
    warnings.warn("pingouin not installed. Run: pip install pingouin")
    pg = None

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
except ImportError:
    warnings.warn("statsmodels not installed. Run: pip install statsmodels")
    sm = None


def check_normality(data, group_col=None, dv_col=None):
    """
    Test normality assumption using Shapiro-Wilk test.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to test
    group_col : str, optional
        Column name for grouping variable (tests normality within each group)
    dv_col : str
        Column name for dependent variable
        
    Returns
    -------
    pd.DataFrame
        Results with W statistic and p-value for each group
    """
    results = []
    
    if group_col is None:
        # Test overall normality
        values = data[dv_col].dropna()
        if len(values) >= 3:
            w_stat, p_val = stats.shapiro(values)
            results.append({
                'group': 'All',
                'n': len(values),
                'W': w_stat,
                'p': p_val,
                'normal': p_val > 0.05
            })
    else:
        # Test normality within each group
        for group in data[group_col].unique():
            group_data = data[data[group_col] == group][dv_col].dropna()
            if len(group_data) >= 3:
                w_stat, p_val = stats.shapiro(group_data)
                results.append({
                    'group': group,
                    'n': len(group_data),
                    'W': w_stat,
                    'p': p_val,
                    'normal': p_val > 0.05
                })
    
    return pd.DataFrame(results)


def check_homogeneity_of_variance(data, dv_col, group_col):
    """
    Test homogeneity of variance using Levene's test.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to test
    dv_col : str
        Column name for dependent variable
    group_col : str
        Column name for grouping variable
        
    Returns
    -------
    dict
        Results with F statistic and p-value
    """
    groups = [group[dv_col].dropna().values for name, group in data.groupby(group_col)]
    
    # Filter out empty groups
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) < 2:
        return {
            'test': 'Levene',
            'F': np.nan,
            'p': np.nan,
            'homogeneous': np.nan,
            'error': 'Insufficient groups for variance test'
        }
    
    f_stat, p_val = stats.levene(*groups)
    
    return {
        'test': 'Levene',
        'F': f_stat,
        'p': p_val,
        'homogeneous': p_val > 0.05
    }


def run_mixed_anova(data, dv, within_factor, between_factor, subject_id):
    """
    Run mixed ANOVA (within-between design).
    
    Parameters
    ----------
    data : pd.DataFrame
        Long-format data
    dv : str
        Dependent variable column name
    within_factor : str
        Within-subjects factor column name (e.g., 'load')
    between_factor : str
        Between-subjects factor column name (e.g., 'trauma_group')
    subject_id : str
        Subject identifier column name
        
    Returns
    -------
    pd.DataFrame
        ANOVA results table
    """
    if pg is None:
        raise ImportError("pingouin is required for mixed ANOVA. Install with: pip install pingouin")
    
    # Run mixed ANOVA
    aov = pg.mixed_anova(
        data=data,
        dv=dv,
        within=within_factor,
        between=between_factor,
        subject=subject_id,
        correction=True  # Greenhouse-Geisser correction if sphericity violated
    )
    
    # Add interpretation
    aov['sig'] = aov['p-unc'] < 0.05
    
    return aov


def run_rm_anova(data, dv, within_factor, subject_id):
    """
    Run repeated measures ANOVA (within-subjects only).
    
    Parameters
    ----------
    data : pd.DataFrame
        Long-format data
    dv : str
        Dependent variable column name
    within_factor : str
        Within-subjects factor column name
    subject_id : str
        Subject identifier column name
        
    Returns
    -------
    pd.DataFrame
        ANOVA results table
    """
    if pg is None:
        raise ImportError("pingouin is required for RM-ANOVA. Install with: pip install pingouin")
    
    aov = pg.rm_anova(
        data=data,
        dv=dv,
        within=within_factor,
        subject=subject_id,
        correction=True
    )
    
    aov['sig'] = aov['p-unc'] < 0.05
    
    return aov


def run_between_anova(data, dv, between_factor):
    """
    Run between-subjects ANOVA.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data
    dv : str
        Dependent variable column name
    between_factor : str
        Between-subjects factor column name
        
    Returns
    -------
    pd.DataFrame
        ANOVA results table
    """
    if pg is None:
        raise ImportError("pingouin is required for ANOVA. Install with: pip install pingouin")
    
    aov = pg.anova(
        data=data,
        dv=dv,
        between=between_factor
    )
    
    aov['sig'] = aov['p-unc'] < 0.05
    
    return aov


def run_welch_anova(data, dv, between_factor):
    """
    Run Welch's ANOVA (does not assume equal variances).
    
    Parameters
    ----------
    data : pd.DataFrame
        Data
    dv : str
        Dependent variable column name
    between_factor : str
        Between-subjects factor column name
        
    Returns
    -------
    pd.DataFrame
        Welch ANOVA results
    """
    if pg is None:
        raise ImportError("pingouin is required for Welch ANOVA. Install with: pip install pingouin")
    
    aov = pg.welch_anova(
        data=data,
        dv=dv,
        between=between_factor
    )
    
    aov['sig'] = aov['p-unc'] < 0.05
    
    return aov


def post_hoc_tests(data, dv, between_factor, padjust='bonf'):
    """
    Run post-hoc pairwise comparisons.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data
    dv : str
        Dependent variable column name
    between_factor : str
        Between-subjects factor column name
    padjust : str
        Method for p-value adjustment ('bonf', 'holm', 'fdr_bh', etc.)
        
    Returns
    -------
    pd.DataFrame
        Pairwise comparison results
    """
    if pg is None:
        raise ImportError("pingouin is required for post-hoc tests. Install with: pip install pingouin")
    
    return pg.pairwise_tests(
        data=data,
        dv=dv,
        between=between_factor,
        padjust=padjust,
        parametric=True
    )


def calculate_cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size.
    
    Parameters
    ----------
    group1 : array-like
        First group values
    group2 : array-like
        Second group values
        
    Returns
    -------
    float
        Cohen's d
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def run_linear_regression(data, dv, predictors, include_intercept=True):
    """
    Run linear regression.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data containing DV and predictors
    dv : str
        Dependent variable column name
    predictors : list of str
        Predictor variable column names
    include_intercept : bool
        Whether to include intercept
        
    Returns
    -------
    dict
        Dictionary with model summary and results
    """
    if sm is None:
        raise ImportError("statsmodels is required for regression. Install with: pip install statsmodels")
    
    # Prepare data (drop missing)
    cols = [dv] + predictors
    clean_data = data[cols].dropna()
    
    # Build formula
    formula = f"{dv} ~ {' + '.join(predictors)}"
    
    # Fit model
    model = ols(formula, data=clean_data).fit()
    
    return {
        'summary': model.summary(),
        'params': model.params,
        'pvalues': model.pvalues,
        'rsquared': model.rsquared,
        'rsquared_adj': model.rsquared_adj,
        'fvalue': model.fvalue,
        'f_pvalue': model.f_pvalue,
        'aic': model.aic,
        'bic': model.bic,
        'model': model
    }


def run_multiple_regressions(data, dv_list, predictors):
    """
    Run multiple regressions (same predictors, different DVs).
    
    Parameters
    ----------
    data : pd.DataFrame
        Data
    dv_list : list of str
        List of dependent variable column names
    predictors : list of str
        Predictor variable column names
        
    Returns
    -------
    dict
        Dictionary mapping DV name to regression results
    """
    results = {}
    
    for dv in dv_list:
        print(f"\n{'='*80}")
        print(f"Regression: {dv} ~ {' + '.join(predictors)}")
        print('='*80)
        
        try:
            reg_result = run_linear_regression(data, dv, predictors)
            results[dv] = reg_result
            print(reg_result['summary'])
        except Exception as e:
            print(f"Error running regression for {dv}: {e}")
            results[dv] = None
    
    return results


def create_anova_summary_table(aov_results):
    """
    Format ANOVA results into a clean summary table.
    
    Parameters
    ----------
    aov_results : pd.DataFrame
        ANOVA results from pingouin
        
    Returns
    -------
    pd.DataFrame
        Formatted summary table
    """
    summary = aov_results.copy()
    
    # Round numeric columns
    if 'F' in summary.columns:
        summary['F'] = summary['F'].round(3)
    if 'p-unc' in summary.columns:
        summary['p'] = summary['p-unc'].round(4)
    if 'np2' in summary.columns:
        summary['η²p'] = summary['np2'].round(3)
    
    # Select key columns
    keep_cols = ['Source']
    if 'ddof1' in summary.columns and 'ddof2' in summary.columns:
        keep_cols.extend(['ddof1', 'ddof2'])
    keep_cols.extend(['F', 'p'])
    if 'η²p' in summary.columns:
        keep_cols.append('η²p')
    if 'sig' in summary.columns:
        keep_cols.append('sig')
    
    available_cols = [col for col in keep_cols if col in summary.columns]
    
    return summary[available_cols]


def save_statistical_results(results_dict, output_dir):
    """
    Save statistical results to CSV files.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary of results (ANOVA tables, regression outputs, etc.)
    output_dir : Path
        Directory to save results
    """
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, result in results_dict.items():
        if isinstance(result, pd.DataFrame):
            filepath = output_dir / f"{name}.csv"
            result.to_csv(filepath, index=False)
            print(f"Saved: {filepath}")
        elif isinstance(result, dict) and 'model' in result:
            # Regression result
            filepath = output_dir / f"{name}_regression.txt"
            with open(filepath, 'w') as f:
                f.write(str(result['summary']))
            print(f"Saved: {filepath}")
