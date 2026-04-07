#!/usr/bin/env python
"""
15: Analyze MLE by Trauma
=========================

Examines relationships between model parameters and trauma measures.

This script performs:
1. Group comparisons (Mann-Whitney U tests with Bonferroni correction)
2. Continuous correlations (Spearman with FWE correction)
3. OLS regressions (parameters ~ IES-R subscales)
4. Visualization (violin plots, heatmaps, forest plots)

Based on Senta et al. (2025) methodology.

Key Analyses:
    1. Parameter comparisons across trauma groups (ANOVA/Kruskal-Wallis)
    2. Correlations between parameters and continuous trauma scores
    3. Visualization of parameter distributions by group
    4. Hypothesis-specific tests (e.g., WM capacity × trauma)

Parameters Analyzed:
    Q-Learning (M1): α₊, α₋, ε
    WM-RL (M2): α₊, α₋, φ, ρ, K, ε
    WM-RL+κ (M3): α₊, α₋, φ, ρ, K, κ, ε

Trauma Measures:
    - LEC-5/LESS Total Events (exposure count)
    - IES-R Total Score (symptom severity)
    - IES-R Subscales: Intrusion, Avoidance, Hyperarousal

Inputs:
    - output/mle/<model>_individual_fits.csv
    - output/summary_participant_metrics.csv (trauma scores)
    - output/trauma_groups/group_assignments.csv (group labels)

Outputs:
    - output/mle_by_trauma/<model>_group_comparison.csv
    - output/mle_by_trauma/<model>_correlations.csv
    - figures/mle_by_trauma/<model>_parameters_by_group.png
    - figures/mle_by_trauma/<model>_parameter_correlations.png

Usage:
    # Analyze Q-learning parameters
    python scripts/15_analyze_mle_by_trauma.py --model qlearning

    # Analyze WM-RL parameters
    python scripts/15_analyze_mle_by_trauma.py --model wmrl

    # Analyze all models
    python scripts/15_analyze_mle_by_trauma.py --model all

    # Specify custom paths
    python scripts/15_analyze_mle_by_trauma.py --model wmrl \
        --mle-file output/mle/wmrl_individual_fits.csv \
        --trauma-file output/summary_participant_metrics.csv

Next Steps:
    - Interpret parameter-trauma relationships
    - Run 16_regress_parameters_on_scales.py for regression analysis
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.plotting_utils import (
    TRAUMA_GROUP_COLORS,
    add_colored_scatter,
    get_color_palette,
)

from plotting_config import PlotConfig

# Paths
OUTPUT_DIR = PROJECT_ROOT / "output" / "mle"
FIGURES_DIR = PROJECT_ROOT / "figures" / "mle_trauma_analysis"

# Group colors (matching plotting_config.py)
GROUP_COLORS = TRAUMA_GROUP_COLORS

# Model parameters
QLEARNING_PARAMS = ['alpha_pos', 'alpha_neg', 'epsilon']
WMRL_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'epsilon']
WMRL_M3_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa', 'epsilon']
WMRL_M4_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa',
                  'v_scale', 'A', 'delta', 't0']
WMRL_M5_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa', 'phi_rl', 'epsilon']
WMRL_M6A_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa_s', 'epsilon']
WMRL_M6B_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa_total', 'kappa_share', 'epsilon']

# Trauma scale predictors
TRAUMA_PREDICTORS = [
    'lec_total', 'lec_personal',
    'ies_intrusion', 'ies_avoidance', 'ies_hyperarousal'
]

# Parameter display names
PARAM_NAMES = {
    'alpha_pos': r'$\alpha_+$',
    'alpha_neg': r'$\alpha_-$',
    'epsilon': r'$\varepsilon$',
    'phi': r'$\phi$',
    'rho': r'$\rho$',
    'capacity': 'K',
    'kappa': r'$\kappa$',
    'phi_rl': r'$\phi_{RL}$',
    'kappa_s': r'$\kappa_s$',
    'kappa_total': r'$\kappa_{total}$',
    'kappa_share': r'$\kappa_{share}$',
    'v_scale': r'$v_{scale}$',
    'A': r'$A$',
    'delta': r'$\delta$',
    't0': r'$t_0$',
}

def load_data() -> tuple:
    """Load and merge MLE fits with survey/group data."""
    # Load survey data
    surveys = pd.read_csv(OUTPUT_DIR / "participant_surveys.csv")
    groups = pd.read_csv(PROJECT_ROOT / "output" / "trauma_groups" / "group_assignments.csv")

    # Convert sona_id to string for consistent merging (handles both numeric and anon IDs)
    surveys['sona_id'] = surveys['sona_id'].astype(str)
    groups['sona_id'] = groups['sona_id'].astype(str)

    # Load MLE fits
    qlearning = pd.read_csv(OUTPUT_DIR / "qlearning_individual_fits.csv")
    wmrl = pd.read_csv(OUTPUT_DIR / "wmrl_individual_fits.csv")
    wmrl_m3 = pd.read_csv(OUTPUT_DIR / "wmrl_m3_individual_fits.csv")

    # M5: load defensively (file may not exist)
    # Check output/mle/ first, then output/ (plan 01 used --output output)
    wmrl_m5_path = OUTPUT_DIR / "wmrl_m5_individual_fits.csv"
    if not wmrl_m5_path.exists():
        wmrl_m5_path = PROJECT_ROOT / "output" / "wmrl_m5_individual_fits.csv"
    wmrl_m5 = pd.read_csv(wmrl_m5_path) if wmrl_m5_path.exists() else None

    # M6a: load defensively (file may not exist)
    wmrl_m6a_path = OUTPUT_DIR / "wmrl_m6a_individual_fits.csv"
    if not wmrl_m6a_path.exists():
        wmrl_m6a_path = PROJECT_ROOT / "output" / "wmrl_m6a_individual_fits.csv"
    wmrl_m6a = pd.read_csv(wmrl_m6a_path) if wmrl_m6a_path.exists() else None

    # M6b: load defensively (file may not exist)
    wmrl_m6b_path = OUTPUT_DIR / "wmrl_m6b_individual_fits.csv"
    if not wmrl_m6b_path.exists():
        wmrl_m6b_path = PROJECT_ROOT / "output" / "wmrl_m6b_individual_fits.csv"
    wmrl_m6b = pd.read_csv(wmrl_m6b_path) if wmrl_m6b_path.exists() else None

    # M4: load defensively (file may not exist)
    wmrl_m4_path = OUTPUT_DIR / "wmrl_m4_individual_fits.csv"
    if not wmrl_m4_path.exists():
        wmrl_m4_path = PROJECT_ROOT / "output" / "wmrl_m4_individual_fits.csv"
    wmrl_m4 = pd.read_csv(wmrl_m4_path) if wmrl_m4_path.exists() else None

    # Convert participant_id to string for consistent merging
    qlearning['participant_id'] = qlearning['participant_id'].astype(str)
    wmrl['participant_id'] = wmrl['participant_id'].astype(str)
    wmrl_m3['participant_id'] = wmrl_m3['participant_id'].astype(str)
    if wmrl_m5 is not None:
        wmrl_m5['participant_id'] = wmrl_m5['participant_id'].astype(str)
    if wmrl_m6a is not None:
        wmrl_m6a['participant_id'] = wmrl_m6a['participant_id'].astype(str)
    if wmrl_m6b is not None:
        wmrl_m6b['participant_id'] = wmrl_m6b['participant_id'].astype(str)
    if wmrl_m4 is not None:
        wmrl_m4['participant_id'] = wmrl_m4['participant_id'].astype(str)

    # Merge with surveys
    qlearning = qlearning.merge(
        surveys, left_on='participant_id', right_on='sona_id', how='inner'
    )
    qlearning = qlearning.merge(
        groups[['sona_id', 'hypothesis_group']],
        on='sona_id', how='left'
    )

    wmrl = wmrl.merge(
        surveys, left_on='participant_id', right_on='sona_id', how='inner'
    )
    wmrl = wmrl.merge(
        groups[['sona_id', 'hypothesis_group']],
        on='sona_id', how='left'
    )

    wmrl_m3 = wmrl_m3.merge(
        surveys, left_on='participant_id', right_on='sona_id', how='inner'
    )
    wmrl_m3 = wmrl_m3.merge(
        groups[['sona_id', 'hypothesis_group']],
        on='sona_id', how='left'
    )

    if wmrl_m5 is not None:
        wmrl_m5 = wmrl_m5.merge(
            surveys, left_on='participant_id', right_on='sona_id', how='inner'
        )
        wmrl_m5 = wmrl_m5.merge(
            groups[['sona_id', 'hypothesis_group']],
            on='sona_id', how='left'
        )

    if wmrl_m6a is not None:
        wmrl_m6a = wmrl_m6a.merge(
            surveys, left_on='participant_id', right_on='sona_id', how='inner'
        )
        wmrl_m6a = wmrl_m6a.merge(
            groups[['sona_id', 'hypothesis_group']],
            on='sona_id', how='left'
        )

    if wmrl_m6b is not None:
        wmrl_m6b = wmrl_m6b.merge(
            surveys, left_on='participant_id', right_on='sona_id', how='inner'
        )
        wmrl_m6b = wmrl_m6b.merge(
            groups[['sona_id', 'hypothesis_group']],
            on='sona_id', how='left'
        )

    if wmrl_m4 is not None:
        wmrl_m4 = wmrl_m4.merge(
            surveys, left_on='participant_id', right_on='sona_id', how='inner'
        )
        wmrl_m4 = wmrl_m4.merge(
            groups[['sona_id', 'hypothesis_group']],
            on='sona_id', how='left'
        )

    print(f"Q-learning participants with surveys: {len(qlearning)}")
    print(f"WM-RL participants with surveys: {len(wmrl)}")
    print(f"WM-RL+K participants with surveys: {len(wmrl_m3)}")
    if wmrl_m5 is not None:
        print(f"WM-RL+M5 participants with surveys: {len(wmrl_m5)}")
    else:
        print("WM-RL+M5: not found (run 12_fit_mle.py --model wmrl_m5 first)")
    if wmrl_m6a is not None:
        print(f"WM-RL+M6a participants with surveys: {len(wmrl_m6a)}")
    else:
        print("WM-RL+M6a: not found (run 12_fit_mle.py --model wmrl_m6a first)")
    if wmrl_m6b is not None:
        print(f"WM-RL+M6b participants with surveys: {len(wmrl_m6b)}")
    else:
        print("WM-RL+M6b: not found (run 12_fit_mle.py --model wmrl_m6b first)")
    if wmrl_m4 is not None:
        print(f"WM-RL+M4 participants with surveys: {len(wmrl_m4)}")
    else:
        print("WM-RL+M4: not found (run 12_fit_mle.py --model wmrl_m4 first)")

    return qlearning, wmrl, wmrl_m3, surveys, groups, wmrl_m5, wmrl_m6a, wmrl_m6b, wmrl_m4

def mann_whitney_with_effect_size(group1: np.ndarray, group2: np.ndarray) -> dict:
    """
    Perform Mann-Whitney U test with rank-biserial correlation effect size.

    Parameters
    ----------
    group1, group2 : array-like
        Data for each group

    Returns
    -------
    dict
        Contains U statistic, p-value, and effect size (rank-biserial r)
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    # Filter NaN
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    if len(group1) < 3 or len(group2) < 3:
        return {'U': np.nan, 'p': np.nan, 'r_rb': np.nan, 'n1': len(group1), 'n2': len(group2)}

    stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')

    # Rank-biserial correlation: r = 1 - (2U)/(n1*n2)
    n1, n2 = len(group1), len(group2)
    r_rb = 1 - (2 * stat) / (n1 * n2)

    return {'U': stat, 'p': p, 'r_rb': r_rb, 'n1': n1, 'n2': n2}

def group_comparisons(df: pd.DataFrame, params: list, model_name: str) -> pd.DataFrame:
    """
    Perform pairwise group comparisons for all parameters.

    Uses Mann-Whitney U with Bonferroni correction.

    Parameters
    ----------
    df : pd.DataFrame
        Data with parameter values and hypothesis_group
    params : list
        List of parameter names to compare
    model_name : str
        Model name for output

    Returns
    -------
    pd.DataFrame
        Results with U stats, p-values, corrected p-values, effect sizes
    """
    # Define groups to compare (actual group names from group_assignments.csv)
    # Note: All participants have trauma exposure - no "No Trauma" group
    groups = ['Trauma Exposure - No Ongoing Impact', 'Trauma Exposure - Ongoing Impact']
    group_pairs = list(combinations(groups, 2))

    results = []
    n_tests = len(params) * len(group_pairs)
    alpha_corrected = 0.05 / n_tests

    print(f"\n{model_name} Group Comparisons")
    print(f"{'='*60}")
    print(f"Bonferroni-corrected alpha: {alpha_corrected:.5f} ({n_tests} tests)")

    for param in params:
        for g1, g2 in group_pairs:
            data1 = df[df['hypothesis_group'] == g1][param].values
            data2 = df[df['hypothesis_group'] == g2][param].values

            test_result = mann_whitney_with_effect_size(data1, data2)

            # Bonferroni correction
            p_corrected = min(test_result['p'] * n_tests, 1.0)

            results.append({
                'model': model_name,
                'parameter': param,
                'group1': g1,
                'group2': g2,
                'n1': test_result['n1'],
                'n2': test_result['n2'],
                'U': test_result['U'],
                'p_uncorrected': test_result['p'],
                'p_bonferroni': p_corrected,
                'significant': p_corrected < 0.05,
                'r_rank_biserial': test_result['r_rb'],
                'mean1': np.nanmean(data1),
                'mean2': np.nanmean(data2),
                'std1': np.nanstd(data1),
                'std2': np.nanstd(data2),
            })

            # Print significant results
            if p_corrected < 0.05:
                print(f"\n*** {param}: {g1} vs {g2}")
                print(f"    U = {test_result['U']:.1f}, p = {test_result['p']:.4f} (corrected: {p_corrected:.4f})")
                print(f"    Effect size (r_rb) = {test_result['r_rb']:.3f}")
                print(f"    {g1}: {np.nanmean(data1):.3f} +/- {np.nanstd(data1):.3f}")
                print(f"    {g2}: {np.nanmean(data2):.3f} +/- {np.nanstd(data2):.3f}")

    return pd.DataFrame(results)

def spearman_correlations(df: pd.DataFrame, params: list, predictors: list,
                          model_name: str) -> pd.DataFrame:
    """
    Compute Spearman correlations between trauma scales and parameters.

    Parameters
    ----------
    df : pd.DataFrame
        Data with parameters and predictors
    params : list
        Model parameters
    predictors : list
        Trauma scale predictors
    model_name : str
        Model name

    Returns
    -------
    pd.DataFrame
        Correlation results with FWE correction
    """
    results = []
    n_tests = len(params) * len(predictors)
    alpha_corrected = 0.05 / n_tests

    print(f"\n{model_name} Spearman Correlations")
    print(f"{'='*60}")
    print(f"FWE-corrected alpha: {alpha_corrected:.5f} ({n_tests} tests)")

    for param in params:
        for pred in predictors:
            # Drop NaN
            mask = ~(df[param].isna() | df[pred].isna())
            x = df.loc[mask, pred].values
            y = df.loc[mask, param].values

            if len(x) < 5:
                continue

            rho, p = stats.spearmanr(x, y)
            p_corrected = min(p * n_tests, 1.0)

            results.append({
                'model': model_name,
                'parameter': param,
                'predictor': pred,
                'n': len(x),
                'rho': rho,
                'p_uncorrected': p,
                'p_fwe': p_corrected,
                'significant': p_corrected < 0.05,
            })

            if p_corrected < 0.05:
                print(f"\n*** {param} ~ {pred}")
                print(f"    rho = {rho:.3f}, p = {p:.4f} (FWE: {p_corrected:.4f})")

    return pd.DataFrame(results)

def ols_regression(df: pd.DataFrame, params: list, model_name: str) -> pd.DataFrame:
    """
    Run OLS regressions: param ~ ies_intrusion + ies_avoidance + ies_hyperarousal

    Parameters
    ----------
    df : pd.DataFrame
        Data
    params : list
        Outcome parameters
    model_name : str
        Model name

    Returns
    -------
    pd.DataFrame
        Regression results
    """
    predictors = ['ies_intrusion', 'ies_avoidance', 'ies_hyperarousal']

    results = []

    print(f"\n{model_name} OLS Regressions")
    print(f"{'='*60}")
    print(f"Model: param ~ {' + '.join(predictors)}")

    for param in params:
        # Prepare data
        mask = ~df[param].isna() & ~df[predictors].isna().any(axis=1)
        y = df.loc[mask, param].values
        X = df.loc[mask, predictors].values
        X = sm.add_constant(X)

        if len(y) < 10:
            continue

        try:
            model = sm.OLS(y, X).fit()

            # Store results
            for i, pred in enumerate(['intercept'] + predictors):
                results.append({
                    'model': model_name,
                    'outcome': param,
                    'predictor': pred,
                    'beta': model.params[i],
                    'se': model.bse[i],
                    'ci_lower': model.conf_int()[i, 0],
                    'ci_upper': model.conf_int()[i, 1],
                    't': model.tvalues[i],
                    'p': model.pvalues[i],
                    'r2': model.rsquared,
                    'r2_adj': model.rsquared_adj,
                    'n': len(y),
                })

            # Print summary for each outcome
            print(f"\n{param}:")
            print(f"  R² = {model.rsquared:.3f}, R²_adj = {model.rsquared_adj:.3f}")
            for i, pred in enumerate(predictors):
                p = model.pvalues[i + 1]
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {pred}: β = {model.params[i+1]:.4f} [{model.conf_int()[i+1, 0]:.4f}, {model.conf_int()[i+1, 1]:.4f}], p = {p:.4f} {sig}")

        except Exception as e:
            print(f"  Error fitting {param}: {e}")

    return pd.DataFrame(results)

def ols_regression_extended(df: pd.DataFrame, params: list, model_name: str) -> pd.DataFrame:
    """
    Run extended OLS regressions with LESS and IES-R total scores.

    Models tested:
    1. param ~ lec_total_events (LESS exposure)
    2. param ~ ies_total (PTSD symptom severity)
    3. param ~ lec_total_events + ies_total (both - disentangle effects)

    Parameters
    ----------
    df : pd.DataFrame
        Data with fitted parameters and trauma scales
    params : list
        Outcome parameters to regress
    model_name : str
        Model name (Q-Learning or WM-RL)

    Returns
    -------
    pd.DataFrame
        Regression results for all models
    """
    # Define predictor specifications
    predictor_specs = [
        (['lec_total'], 'LESS'),
        (['ies_total'], 'IES-R'),
        (['lec_total', 'ies_total'], 'LESS + IES-R'),
    ]

    results = []

    print(f"\n{model_name} Extended OLS Regressions")
    print(f"{'='*60}")

    for param in params:
        print(f"\n{param}:")

        for predictors, model_spec in predictor_specs:
            # Prepare data
            mask = ~df[param].isna()
            for pred in predictors:
                mask &= ~df[pred].isna()

            y = df.loc[mask, param].values
            X = df.loc[mask, predictors].values
            X = sm.add_constant(X)

            if len(y) < 10:
                print(f"  {model_spec}: Insufficient data (n={len(y)})")
                continue

            try:
                model = sm.OLS(y, X).fit()

                # Store results for intercept
                results.append({
                    'model': model_name,
                    'outcome': param,
                    'predictor_set': model_spec,
                    'predictor': 'intercept',
                    'beta': model.params[0],
                    'se': model.bse[0],
                    'ci_lower': model.conf_int()[0, 0],
                    'ci_upper': model.conf_int()[0, 1],
                    't': model.tvalues[0],
                    'p': model.pvalues[0],
                    'r2': model.rsquared,
                    'r2_adj': model.rsquared_adj,
                    'n': len(y),
                })

                # Store results for each predictor
                for i, pred in enumerate(predictors):
                    results.append({
                        'model': model_name,
                        'outcome': param,
                        'predictor_set': model_spec,
                        'predictor': pred,
                        'beta': model.params[i + 1],
                        'se': model.bse[i + 1],
                        'ci_lower': model.conf_int()[i + 1, 0],
                        'ci_upper': model.conf_int()[i + 1, 1],
                        't': model.tvalues[i + 1],
                        'p': model.pvalues[i + 1],
                        'r2': model.rsquared,
                        'r2_adj': model.rsquared_adj,
                        'n': len(y),
                    })

                # Print summary
                print(f"  {model_spec}: R² = {model.rsquared:.3f}, R²_adj = {model.rsquared_adj:.3f}, n = {len(y)}")
                for i, pred in enumerate(predictors):
                    p = model.pvalues[i + 1]
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    print(f"    {pred}: β = {model.params[i+1]:.4f} [{model.conf_int()[i+1, 0]:.4f}, {model.conf_int()[i+1, 1]:.4f}], p = {p:.4f} {sig}")

            except Exception as e:
                print(f"  {model_spec}: Error - {e}")

    return pd.DataFrame(results)

def plot_parameters_by_group(df: pd.DataFrame, params: list, model_name: str,
                             figsize: tuple = None) -> plt.Figure:
    """
    Create violin + swarm plots of parameters by trauma group.
    """
    # Filter to main comparison groups (actual group names from data)
    plot_df = df[df['hypothesis_group'].isin(['Trauma Exposure - No Ongoing Impact', 'Trauma Exposure - Ongoing Impact'])].copy()

    n_params = len(params)
    ncols = 3
    nrows = int(np.ceil(n_params / ncols))

    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

    group_order = ['Trauma Exposure - No Ongoing Impact', 'Trauma Exposure - Ongoing Impact']
    palette = [GROUP_COLORS.get(g, '#808080') for g in group_order]

    for idx, param in enumerate(params):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        # Violin plot (using hue for seaborn v0.14+ compatibility)
        sns.violinplot(data=plot_df, x='hypothesis_group', y=param,
                       hue='hypothesis_group', order=group_order,
                       palette=GROUP_COLORS, inner=None, alpha=0.7,
                       legend=False, ax=ax)

        # Swarm plot overlay
        sns.swarmplot(data=plot_df, x='hypothesis_group', y=param,
                      order=group_order, color='black', alpha=0.5,
                      size=4, ax=ax)

        ax.set_xlabel('')
        ax.set_ylabel(PARAM_NAMES.get(param, param), fontsize=PlotConfig.AXIS_LABEL_SIZE)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No Ongoing\nImpact', 'Ongoing\nImpact'],
                          fontsize=PlotConfig.TICK_LABEL_SIZE - 2)

    # Remove empty subplots
    for idx in range(n_params, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)

    plt.suptitle(f'{model_name} Parameters by Trauma Group',
                 fontsize=PlotConfig.SUPTITLE_SIZE)
    plt.tight_layout()

    return fig

def plot_correlation_heatmap(corr_df: pd.DataFrame, params: list, model_name: str) -> plt.Figure:
    """
    Create heatmap of Spearman correlations.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Correlation results with columns: parameter, predictor, rho, significant
    params : list
        List of parameters for this model (determines ordering)
    model_name : str
        Model name for title
    """
    # Pivot to matrix form
    pivot = corr_df.pivot(index='parameter', columns='predictor', values='rho')

    # Reorder
    param_order = [p for p in params if p in pivot.index]
    pred_order = [p for p in TRAUMA_PREDICTORS if p in pivot.columns]
    pivot = pivot.loc[param_order, pred_order]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create mask for non-significant correlations (optional: for highlighting)
    sig_df = corr_df.pivot(index='parameter', columns='predictor', values='significant')

    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-0.5, vmax=0.5,
                xticklabels=['LEC Total', 'LEC Personal', 'Intrusion', 'Avoidance', 'Hyperarousal'],
                yticklabels=[PARAM_NAMES.get(p, p) for p in param_order],
                ax=ax, cbar_kws={'label': 'Spearman rho'})

    ax.set_xlabel('Trauma Scale', fontsize=PlotConfig.AXIS_LABEL_SIZE)
    ax.set_ylabel('Parameter', fontsize=PlotConfig.AXIS_LABEL_SIZE)
    ax.set_title(f'{model_name} Parameters × Trauma Scales Correlations',
                 fontsize=PlotConfig.TITLE_SIZE)

    plt.tight_layout()
    return fig

def plot_forest_group_means(df: pd.DataFrame, params: list, model_name: str) -> plt.Figure:
    """
    Create forest plot showing group means with error bars.
    """
    group_order = ['Trauma Exposure - No Ongoing Impact', 'Trauma Exposure - Ongoing Impact']
    plot_df = df[df['hypothesis_group'].isin(group_order)].copy()

    fig, axes = plt.subplots(1, len(params), figsize=(3 * len(params), 4))
    if len(params) == 1:
        axes = [axes]

    for ax, param in zip(axes, params):
        y_positions = np.arange(len(group_order))

        for i, group in enumerate(group_order):
            data = plot_df[plot_df['hypothesis_group'] == group][param].dropna()
            if len(data) == 0:
                continue
            mean = data.mean()
            sem = data.std() / np.sqrt(len(data))

            ax.errorbar(mean, i, xerr=sem, fmt='o',
                       color=GROUP_COLORS.get(group, '#808080'), markersize=10,
                       capsize=5, capthick=2, elinewidth=2)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(['No Ongoing Impact', 'Ongoing Impact'])
        ax.set_xlabel(PARAM_NAMES.get(param, param), fontsize=PlotConfig.AXIS_LABEL_SIZE)
        ax.axvline(x=plot_df[param].mean(), color='gray', linestyle='--', alpha=0.5)

    plt.suptitle(f'{model_name} Group Means (Mean ± SEM)',
                 fontsize=PlotConfig.SUPTITLE_SIZE)
    plt.tight_layout()

    return fig

def plot_key_scatter(df: pd.DataFrame, model_name: str, color_by: str | None = None) -> plt.Figure:
    """
    Scatter plots for key correlations with regression lines.

    Parameters
    ----------
    df : pd.DataFrame
        Data with parameters and predictors
    model_name : str
        Model name for title
    color_by : str, optional
        Column to color points by. If None, uses hypothesis_group with GROUP_COLORS.
    """
    # Key relationships to plot
    plots = [
        ('ies_intrusion', 'epsilon', 'Intrusion', r'$\varepsilon$ (Noise)'),
        ('ies_total', 'phi', 'IES-R Total', r'$\phi$ (WM Decay)'),
        ('lec_total', 'alpha_pos', 'LEC Total', r'$\alpha_+$ (Positive LR)'),
    ]

    # Filter to plots that exist in params
    if model_name == 'Q-Learning':
        plots = [p for p in plots if p[1] in QLEARNING_PARAMS]

    fig, axes = plt.subplots(1, len(plots), figsize=(5 * len(plots), 4))
    if len(plots) == 1:
        axes = [axes]

    for ax, (pred, param, xlabel, ylabel) in zip(axes, plots):
        if param not in df.columns or pred not in df.columns:
            ax.set_visible(False)
            continue

        # Color by specified column or default to hypothesis_group
        if color_by is not None:
            # Use plotting utility for flexible color-by
            custom_colors = TRAUMA_GROUP_COLORS if color_by == 'hypothesis_group' else None
            palette = get_color_palette(df, color_by, custom_colors=custom_colors)
            add_colored_scatter(ax, pred, param, df, color_by, palette,
                              alpha=0.7, s=50, show_legend=True)
        else:
            # Default: color by hypothesis_group (actual group names from data)
            for group in ['Trauma Exposure - No Ongoing Impact', 'Trauma Exposure - Ongoing Impact']:
                group_data = df[df['hypothesis_group'] == group]
                ax.scatter(group_data[pred], group_data[param],
                          c=GROUP_COLORS.get(group, 'gray'),
                          label=group.replace('Trauma Exposure - ', ''), alpha=0.7, s=50)
            ax.legend(fontsize=PlotConfig.SMALL_TEXT_SIZE)

        # Overall regression line
        mask = ~(df[pred].isna() | df[param].isna())
        x = df.loc[mask, pred].values
        y = df.loc[mask, param].values
        if len(x) > 5:
            slope, intercept, r, p, _ = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.5,
                   label=f'r={r:.2f}, p={p:.3f}')

        ax.set_xlabel(xlabel, fontsize=PlotConfig.AXIS_LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=PlotConfig.AXIS_LABEL_SIZE)

    plt.suptitle(f'{model_name}: Key Parameter-Trauma Relationships',
                 fontsize=PlotConfig.SUPTITLE_SIZE)
    plt.tight_layout()

    return fig

def main():
    """Main analysis pipeline."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Analyze MLE parameters by trauma group',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/15_analyze_mle_by_trauma.py --model qlearning
  python scripts/15_analyze_mle_by_trauma.py --model all
  python scripts/15_analyze_mle_by_trauma.py --model wmrl --color-by hypothesis_group
        """
    )
    parser.add_argument('--model', type=str, default='all',
                       choices=['qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b', 'wmrl_m4', 'all'],
                       help='Model to analyze (default: all)')
    parser.add_argument('--color-by', type=str, default=None,
                       help='Column to color scatter plots by (default: hypothesis_group)')
    args = parser.parse_args()

    print("=" * 70)
    print("MLE Parameter Analysis by Trauma Group")
    print("=" * 70)

    # Apply plotting defaults
    PlotConfig.apply_defaults()

    # Create output directories
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    qlearning, wmrl, wmrl_m3, surveys, groups, wmrl_m5, wmrl_m6a, wmrl_m6b, wmrl_m4 = load_data()

    # Model configuration
    MODEL_CONFIG = {
        'qlearning': {'name': 'Q-Learning', 'params': QLEARNING_PARAMS, 'data': qlearning},
        'wmrl': {'name': 'WM-RL', 'params': WMRL_PARAMS, 'data': wmrl},
        'wmrl_m3': {'name': 'WM-RL+K', 'params': WMRL_M3_PARAMS, 'data': wmrl_m3},
    }
    if wmrl_m5 is not None:
        MODEL_CONFIG['wmrl_m5'] = {'name': 'WM-RL+M5', 'params': WMRL_M5_PARAMS, 'data': wmrl_m5}
    if wmrl_m6a is not None:
        MODEL_CONFIG['wmrl_m6a'] = {'name': 'WM-RL+M6a', 'params': WMRL_M6A_PARAMS, 'data': wmrl_m6a}
    if wmrl_m6b is not None:
        MODEL_CONFIG['wmrl_m6b'] = {'name': 'WM-RL+M6b', 'params': WMRL_M6B_PARAMS, 'data': wmrl_m6b}
    if wmrl_m4 is not None:
        MODEL_CONFIG['wmrl_m4'] = {'name': 'WM-RL+M4 (LBA)', 'params': WMRL_M4_PARAMS, 'data': wmrl_m4}

    # Determine which models to analyze
    if args.model == 'all':
        models_to_analyze = [m for m in ['qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b', 'wmrl_m4']
                             if m in MODEL_CONFIG]
    else:
        if args.model not in MODEL_CONFIG:
            print(f"ERROR: Model '{args.model}' not available. "
                  f"Run 12_fit_mle.py --model {args.model} first.")
            return
        models_to_analyze = [args.model]

    all_group_results = []
    all_corr_results = []
    all_ols_results = []

    # ========================================
    # Run Analysis for Each Model
    # ========================================
    for model_key in models_to_analyze:
        config = MODEL_CONFIG[model_key]
        model_name = config['name']
        params = config['params']
        data = config['data']

        print("\n" + "=" * 70)
        print(f"{model_name.upper()} MODEL")
        print("=" * 70)

        # Check data
        print("\nGroup distribution:")
        print(data['hypothesis_group'].value_counts())

        # Group comparisons
        group_results = group_comparisons(data, params, model_name)
        all_group_results.append(group_results)

        # Spearman correlations
        corr_results = spearman_correlations(data, params, TRAUMA_PREDICTORS, model_name)
        all_corr_results.append(corr_results)

        # OLS regressions
        ols_results = ols_regression(data, params, model_name)
        all_ols_results.append(ols_results)

        # Extended regressions (LESS + IES-R total)
        ols_extended = ols_regression_extended(data, params, model_name)
        all_ols_results.append(ols_extended)

    # ========================================
    # Save Results
    # ========================================
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)

    group_df = pd.concat(all_group_results, ignore_index=True)
    group_df.to_csv(OUTPUT_DIR / "group_comparison_stats.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'group_comparison_stats.csv'}")

    corr_df = pd.concat(all_corr_results, ignore_index=True)
    corr_df.to_csv(OUTPUT_DIR / "spearman_correlations.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'spearman_correlations.csv'}")

    ols_df = pd.concat(all_ols_results, ignore_index=True)
    ols_df.to_csv(OUTPUT_DIR / "ols_regression_results.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'ols_regression_results.csv'}")

    # ========================================
    # Create Figures for Each Model
    # ========================================
    print("\n" + "=" * 70)
    print("Creating Figures")
    print("=" * 70)

    for model_key in models_to_analyze:
        config = MODEL_CONFIG[model_key]
        model_name = config['name']
        params = config['params']
        data = config['data']

        # Get correlation results for this model (corr_df may be empty if n < 5)
        if 'model' in corr_df.columns:
            model_corr_results = corr_df[corr_df['model'] == model_name]
        else:
            model_corr_results = pd.DataFrame()

        # Parameter violin plots
        n_params = len(params)
        ncols = 3
        nrows = int(np.ceil(n_params / ncols))
        figsize = (12, 4 * nrows)
        fig = plot_parameters_by_group(data, params, model_name, figsize=figsize)
        fig.savefig(FIGURES_DIR / f"parameters_by_group_{model_key}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: parameters_by_group_{model_key}.png")

        # Correlation heatmap
        if len(model_corr_results) > 0:
            fig = plot_correlation_heatmap(model_corr_results, params, model_name)
            fig.savefig(FIGURES_DIR / f"correlation_heatmap_{model_key}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: correlation_heatmap_{model_key}.png")

        # Forest plot
        fig = plot_forest_group_means(data, params, model_name)
        fig.savefig(FIGURES_DIR / f"forest_plot_group_means_{model_key}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: forest_plot_group_means_{model_key}.png")

        # Key scatter plots (with color-by support)
        fig = plot_key_scatter(data, model_name, color_by=args.color_by)
        fig.savefig(FIGURES_DIR / f"scatter_key_correlations_{model_key}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: scatter_key_correlations_{model_key}.png")

    # ========================================
    # Summary of Significant Findings
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY OF SIGNIFICANT FINDINGS")
    print("=" * 70)

    print("\nSignificant Group Comparisons (Bonferroni p < 0.05):")
    sig_groups = group_df[group_df['significant']]
    if len(sig_groups) > 0:
        for _, row in sig_groups.iterrows():
            print(f"  {row['model']} {row['parameter']}: {row['group1']} vs {row['group2']}")
            print(f"    r_rb = {row['r_rank_biserial']:.3f}, p_corrected = {row['p_bonferroni']:.4f}")
    else:
        print("  None")

    print("\nSignificant Correlations (FWE p < 0.05):")
    if 'significant' in corr_df.columns:
        sig_corrs = corr_df[corr_df['significant']]
        if len(sig_corrs) > 0:
            for _, row in sig_corrs.iterrows():
                print(f"  {row['model']} {row['parameter']} ~ {row['predictor']}: rho = {row['rho']:.3f}, p_FWE = {row['p_fwe']:.4f}")
        else:
            print("  None")
    else:
        print("  None (insufficient data for correlation analysis)")

    print("\nSignificant OLS Predictors (uncorrected p < 0.05):")
    if 'p' in ols_df.columns and 'predictor' in ols_df.columns:
        sig_ols = ols_df[(ols_df['p'] < 0.05) & (ols_df['predictor'] != 'intercept')]
        if len(sig_ols) > 0:
            for _, row in sig_ols.iterrows():
                print(f"  {row['model']} {row['outcome']} ~ {row['predictor']}: beta = {row['beta']:.4f}, p = {row['p']:.4f}")
        else:
            print("  None")
    else:
        print("  None (insufficient data for OLS analysis)")

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
