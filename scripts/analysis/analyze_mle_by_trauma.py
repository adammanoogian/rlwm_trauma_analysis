"""
Analyze MLE model parameters by trauma group.

This script performs:
1. Group comparisons (Mann-Whitney U tests with Bonferroni correction)
2. Continuous correlations (Spearman with FWE correction)
3. OLS regressions (parameters ~ IES-R subscales)
4. Visualization (violin plots, heatmaps, forest plots)

Based on Senta et al. (2025) methodology.

Usage:
    python scripts/analysis/analyze_mle_by_trauma.py
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from itertools import combinations

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from plotting_config import PlotConfig

# Paths
OUTPUT_DIR = PROJECT_ROOT / "output" / "mle"
FIGURES_DIR = PROJECT_ROOT / "figures" / "mle_trauma_analysis"

# Group colors (matching plotting_config.py)
GROUP_COLORS = {
    'No Trauma': '#06A77D',  # Green
    'Trauma-No Impact': '#F18F01',  # Orange
    'Trauma-Ongoing Impact': '#D62246',  # Red
    'Low Exposure-High Symptoms': '#6C757D',  # Gray (paradoxical)
}

# Model parameters
QLEARNING_PARAMS = ['alpha_pos', 'alpha_neg', 'epsilon']
WMRL_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'epsilon']

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
    'capacity': 'K'
}


def load_data() -> tuple:
    """Load and merge MLE fits with survey/group data."""
    # Load survey data
    surveys = pd.read_csv(OUTPUT_DIR / "participant_surveys.csv")
    groups = pd.read_csv(OUTPUT_DIR / "trauma_group_assignments.csv")

    # Load MLE fits
    qlearning = pd.read_csv(OUTPUT_DIR / "qlearning_individual_fits.csv")
    wmrl = pd.read_csv(OUTPUT_DIR / "wmrl_individual_fits.csv")

    # Merge with surveys
    qlearning = qlearning.merge(
        surveys, left_on='participant_id', right_on='sona_id', how='inner'
    )
    qlearning = qlearning.merge(
        groups[['sona_id', 'hypothesis_group']],
        left_on='participant_id', right_on='sona_id', how='left'
    )

    wmrl = wmrl.merge(
        surveys, left_on='participant_id', right_on='sona_id', how='inner'
    )
    wmrl = wmrl.merge(
        groups[['sona_id', 'hypothesis_group']],
        left_on='participant_id', right_on='sona_id', how='left'
    )

    print(f"Q-learning participants with surveys: {len(qlearning)}")
    print(f"WM-RL participants with surveys: {len(wmrl)}")

    return qlearning, wmrl, surveys, groups


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
    # Define groups to compare (exclude paradoxical group)
    groups = ['No Trauma', 'Trauma-No Impact', 'Trauma-Ongoing Impact']
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


def plot_parameters_by_group(df: pd.DataFrame, params: list, model_name: str,
                             figsize: tuple = None) -> plt.Figure:
    """
    Create violin + swarm plots of parameters by trauma group.
    """
    # Exclude paradoxical group for cleaner visualization
    plot_df = df[df['hypothesis_group'].isin(['No Trauma', 'Trauma-No Impact', 'Trauma-Ongoing Impact'])].copy()

    n_params = len(params)
    ncols = 3
    nrows = int(np.ceil(n_params / ncols))

    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

    group_order = ['No Trauma', 'Trauma-No Impact', 'Trauma-Ongoing Impact']
    palette = [GROUP_COLORS[g] for g in group_order]

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
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['No\nTrauma', 'Trauma\nNo Impact', 'Trauma\nOngoing'],
                          fontsize=PlotConfig.TICK_LABEL_SIZE - 2)

    # Remove empty subplots
    for idx in range(n_params, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)

    plt.suptitle(f'{model_name} Parameters by Trauma Group',
                 fontsize=PlotConfig.SUPTITLE_SIZE)
    plt.tight_layout()

    return fig


def plot_correlation_heatmap(corr_df: pd.DataFrame, model_name: str) -> plt.Figure:
    """
    Create heatmap of Spearman correlations.
    """
    # Pivot to matrix form
    pivot = corr_df.pivot(index='parameter', columns='predictor', values='rho')

    # Reorder
    param_order = [p for p in WMRL_PARAMS if p in pivot.index]
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
    group_order = ['No Trauma', 'Trauma-No Impact', 'Trauma-Ongoing Impact']
    plot_df = df[df['hypothesis_group'].isin(group_order)].copy()

    fig, axes = plt.subplots(1, len(params), figsize=(3 * len(params), 5))
    if len(params) == 1:
        axes = [axes]

    for ax, param in zip(axes, params):
        y_positions = np.arange(len(group_order))

        for i, group in enumerate(group_order):
            data = plot_df[plot_df['hypothesis_group'] == group][param].dropna()
            mean = data.mean()
            sem = data.std() / np.sqrt(len(data))

            ax.errorbar(mean, i, xerr=sem, fmt='o',
                       color=GROUP_COLORS[group], markersize=10,
                       capsize=5, capthick=2, elinewidth=2)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(['No Trauma', 'Trauma-No Impact', 'Trauma-Ongoing'])
        ax.set_xlabel(PARAM_NAMES.get(param, param), fontsize=PlotConfig.AXIS_LABEL_SIZE)
        ax.axvline(x=plot_df[param].mean(), color='gray', linestyle='--', alpha=0.5)

    plt.suptitle(f'{model_name} Group Means (Mean ± SEM)',
                 fontsize=PlotConfig.SUPTITLE_SIZE)
    plt.tight_layout()

    return fig


def plot_key_scatter(df: pd.DataFrame, model_name: str) -> plt.Figure:
    """
    Scatter plots for key correlations with regression lines.
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

        # Color by group
        for group in ['No Trauma', 'Trauma-No Impact', 'Trauma-Ongoing Impact']:
            group_data = df[df['hypothesis_group'] == group]
            ax.scatter(group_data[pred], group_data[param],
                      c=GROUP_COLORS.get(group, 'gray'),
                      label=group, alpha=0.7, s=50)

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
        ax.legend(fontsize=PlotConfig.SMALL_TEXT_SIZE)

    plt.suptitle(f'{model_name}: Key Parameter-Trauma Relationships',
                 fontsize=PlotConfig.SUPTITLE_SIZE)
    plt.tight_layout()

    return fig


def main():
    """Main analysis pipeline."""
    print("=" * 70)
    print("MLE Parameter Analysis by Trauma Group")
    print("=" * 70)

    # Apply plotting defaults
    PlotConfig.apply_defaults()

    # Create output directories
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    qlearning, wmrl, surveys, groups = load_data()

    # Check data
    print(f"\nGroup distribution in WM-RL data:")
    print(wmrl['hypothesis_group'].value_counts())

    all_group_results = []
    all_corr_results = []
    all_ols_results = []

    # ========================================
    # Q-Learning Analysis
    # ========================================
    print("\n" + "=" * 70)
    print("Q-LEARNING MODEL")
    print("=" * 70)

    group_results_ql = group_comparisons(qlearning, QLEARNING_PARAMS, 'Q-Learning')
    all_group_results.append(group_results_ql)

    corr_results_ql = spearman_correlations(qlearning, QLEARNING_PARAMS,
                                            TRAUMA_PREDICTORS, 'Q-Learning')
    all_corr_results.append(corr_results_ql)

    ols_results_ql = ols_regression(qlearning, QLEARNING_PARAMS, 'Q-Learning')
    all_ols_results.append(ols_results_ql)

    # ========================================
    # WM-RL Analysis
    # ========================================
    print("\n" + "=" * 70)
    print("WM-RL MODEL")
    print("=" * 70)

    group_results_wmrl = group_comparisons(wmrl, WMRL_PARAMS, 'WM-RL')
    all_group_results.append(group_results_wmrl)

    corr_results_wmrl = spearman_correlations(wmrl, WMRL_PARAMS,
                                              TRAUMA_PREDICTORS, 'WM-RL')
    all_corr_results.append(corr_results_wmrl)

    ols_results_wmrl = ols_regression(wmrl, WMRL_PARAMS, 'WM-RL')
    all_ols_results.append(ols_results_wmrl)

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
    # Create Figures
    # ========================================
    print("\n" + "=" * 70)
    print("Creating Figures")
    print("=" * 70)

    # Q-Learning parameter violin plots
    fig = plot_parameters_by_group(qlearning, QLEARNING_PARAMS, 'Q-Learning', figsize=(12, 4))
    fig.savefig(FIGURES_DIR / "parameters_by_group_qlearning.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: parameters_by_group_qlearning.png")

    # WM-RL parameter violin plots
    fig = plot_parameters_by_group(wmrl, WMRL_PARAMS, 'WM-RL', figsize=(12, 8))
    fig.savefig(FIGURES_DIR / "parameters_by_group_wmrl.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: parameters_by_group_wmrl.png")

    # Correlation heatmap (WM-RL)
    fig = plot_correlation_heatmap(corr_results_wmrl, 'WM-RL')
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: correlation_heatmap.png")

    # Forest plot (WM-RL)
    fig = plot_forest_group_means(wmrl, WMRL_PARAMS, 'WM-RL')
    fig.savefig(FIGURES_DIR / "forest_plot_group_means.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: forest_plot_group_means.png")

    # Key scatter plots (WM-RL)
    fig = plot_key_scatter(wmrl, 'WM-RL')
    fig.savefig(FIGURES_DIR / "scatter_key_correlations.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: scatter_key_correlations.png")

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
    sig_corrs = corr_df[corr_df['significant']]
    if len(sig_corrs) > 0:
        for _, row in sig_corrs.iterrows():
            print(f"  {row['model']} {row['parameter']} ~ {row['predictor']}: rho = {row['rho']:.3f}, p_FWE = {row['p_fwe']:.4f}")
    else:
        print("  None")

    print("\nSignificant OLS Predictors (uncorrected p < 0.05):")
    sig_ols = ols_df[(ols_df['p'] < 0.05) & (ols_df['predictor'] != 'intercept')]
    if len(sig_ols) > 0:
        for _, row in sig_ols.iterrows():
            print(f"  {row['model']} {row['outcome']} ~ {row['predictor']}: beta = {row['beta']:.4f}, p = {row['p']:.4f}")
    else:
        print("  None")

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
