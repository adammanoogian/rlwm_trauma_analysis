"""
Create Supplementary Tables and Figures for Manuscript

Generates:
- Supplementary Table S1: Regression analyses
- Supplementary Table S2: Distributional statistics
- Supplementary Figure S1: Distributional assumptions
- Supplementary Figure S2: IES-R by trauma group (validation)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import OUTPUT_DIR, FIGURES_DIR, EXCLUDED_PARTICIPANTS
from plotting_config import PlotConfig

# Apply publication settings
PlotConfig.apply_defaults()
DPI = PlotConfig.DPI_PRINT


def create_supplementary_table_s1():
    """
    Supplementary Table S1: Regression Models Predicting Overall Accuracy
    
    Four models:
    1. LESS Total
    2. IES-R Total  
    3. IES-R Subscales (Intrusion, Avoidance, Hyperarousal)
    4. LESS + IES-R Total
    """
    print("\n" + "="*80)
    print("SUPPLEMENTARY TABLE S1: Regression Analyses")
    print("="*80)
    
    # Load data
    summary = pd.read_csv(OUTPUT_DIR / 'statistical_analyses' / 'data_summary_with_groups.csv')
    summary = summary[~summary['sona_id'].isin(EXCLUDED_PARTICIPANTS)].copy()
    
    # Prepare data
    from sklearn.linear_model import LinearRegression
    from scipy.stats import f as f_dist
    
    results_list = []
    
    # Model 1: LESS Total
    X = summary[['less_total_events']].dropna()
    y = summary.loc[X.index, 'accuracy_overall']
    
    # Standardize
    X_std = (X - X.mean()) / X.std()
    
    model = LinearRegression()
    model.fit(X_std, y)
    
    y_pred = model.predict(X_std)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r_squared = 1 - (ss_res / ss_tot)
    n = len(y)
    p = 1
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    f_stat = (r_squared / p) / ((1 - r_squared) / (n - p - 1))
    p_value = 1 - f_dist.cdf(f_stat, p, n - p - 1)
    
    # t-test for coefficient
    se = np.sqrt(ss_res / (n - p - 1)) / np.sqrt(np.sum((X_std.values - X_std.values.mean())**2))
    t_stat = model.coef_[0] / se
    t_p = 2 * (1 - stats.t.cdf(abs(t_stat), n - p - 1))
    
    results_list.append({
        'Model': 'Model 1',
        'Predictor': 'LESS Total Events',
        'β': f"{model.coef_[0]:.4f}",
        't': f"{t_stat:.2f}",
        'p': f"{t_p:.3f}",
        'R²': f"{r_squared:.3f}",
        'Adj. R²': f"{adj_r_squared:.3f}",
        'F': f"F(1, {n-2}) = {f_stat:.2f}",
        'Model p': f"{p_value:.3f}"
    })
    
    # Model 2: IES-R Total
    X = summary[['ies_total']].dropna()
    y = summary.loc[X.index, 'accuracy_overall']
    X_std = (X - X.mean()) / X.std()
    
    model = LinearRegression()
    model.fit(X_std, y)
    
    y_pred = model.predict(X_std)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r_squared = 1 - (ss_res / ss_tot)
    n = len(y)
    p = 1
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    f_stat = (r_squared / p) / ((1 - r_squared) / (n - p - 1))
    p_value = 1 - f_dist.cdf(f_stat, p, n - p - 1)
    
    se = np.sqrt(ss_res / (n - p - 1)) / np.sqrt(np.sum((X_std.values - X_std.values.mean())**2))
    t_stat = model.coef_[0] / se
    t_p = 2 * (1 - stats.t.cdf(abs(t_stat), n - p - 1))
    
    results_list.append({
        'Model': 'Model 2',
        'Predictor': 'IES-R Total Score',
        'β': f"{model.coef_[0]:.4f}",
        't': f"{t_stat:.2f}",
        'p': f"{t_p:.3f}",
        'R²': f"{r_squared:.3f}",
        'Adj. R²': f"{adj_r_squared:.3f}",
        'F': f"F(1, {n-2}) = {f_stat:.2f}",
        'Model p': f"{p_value:.3f}"
    })
    
    # Model 3: IES-R Subscales
    X = summary[['ies_intrusion', 'ies_avoidance', 'ies_hyperarousal']].dropna()
    y = summary.loc[X.index, 'accuracy_overall']
    X_std = (X - X.mean()) / X.std()
    
    model = LinearRegression()
    model.fit(X_std, y)
    
    y_pred = model.predict(X_std)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r_squared = 1 - (ss_res / ss_tot)
    n = len(y)
    p = 3
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    f_stat = (r_squared / p) / ((1 - r_squared) / (n - p - 1))
    p_value = 1 - f_dist.cdf(f_stat, p, n - p - 1)
    
    # Compute t-statistics for each coefficient
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y, y_pred)
    var_coef = mse * np.linalg.inv(X_std.T @ X_std).diagonal()
    se_coef = np.sqrt(var_coef)
    t_stats = model.coef_ / se_coef
    t_ps = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
    
    for i, (pred_name, beta, t, p) in enumerate(zip(['IES-R Intrusion', 'IES-R Avoidance', 'IES-R Hyperarousal'],
                                                      model.coef_, t_stats, t_ps)):
        results_list.append({
            'Model': 'Model 3' if i == 0 else '',
            'Predictor': pred_name,
            'β': f"{beta:.4f}",
            't': f"{t:.2f}",
            'p': f"{p:.3f}",
            'R²': f"{r_squared:.3f}" if i == 0 else '',
            'Adj. R²': f"{adj_r_squared:.3f}" if i == 0 else '',
            'F': f"F(3, {n-4}) = {f_stat:.2f}" if i == 0 else '',
            'Model p': f"{p_value:.3f}" if i == 0 else ''
        })
    
    # Model 4: LESS + IES-R Total
    X = summary[['less_total_events', 'ies_total']].dropna()
    y = summary.loc[X.index, 'accuracy_overall']
    X_std = (X - X.mean()) / X.std()
    
    model = LinearRegression()
    model.fit(X_std, y)
    
    y_pred = model.predict(X_std)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r_squared = 1 - (ss_res / ss_tot)
    n = len(y)
    p = 2
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    f_stat = (r_squared / p) / ((1 - r_squared) / (n - p - 1))
    p_value = 1 - f_dist.cdf(f_stat, p, n - p - 1)
    
    mse = mean_squared_error(y, y_pred)
    var_coef = mse * np.linalg.inv(X_std.T @ X_std).diagonal()
    se_coef = np.sqrt(var_coef)
    t_stats = model.coef_ / se_coef
    t_ps = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
    
    for i, (pred_name, beta, t, p) in enumerate(zip(['LESS Total Events', 'IES-R Total Score'],
                                                      model.coef_, t_stats, t_ps)):
        results_list.append({
            'Model': 'Model 4' if i == 0 else '',
            'Predictor': pred_name,
            'β': f"{beta:.4f}",
            't': f"{t:.2f}",
            'p': f"{p:.3f}",
            'R²': f"{r_squared:.3f}" if i == 0 else '',
            'Adj. R²': f"{adj_r_squared:.3f}" if i == 0 else '',
            'F': f"F(2, {n-3}) = {f_stat:.2f}" if i == 0 else '',
            'Model p': f"{p_value:.3f}" if i == 0 else ''
        })
    
    table_s1 = pd.DataFrame(results_list)
    
    # Save
    output_dir = FIGURES_DIR / 'publication'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    table_s1.to_csv(output_dir / 'SupplementaryTable_S1.csv', index=False)
    
    with open(output_dir / 'SupplementaryTable_S1.txt', 'w') as f:
        f.write("Supplementary Table S1\n")
        f.write("Regression Analysis Predicting Overall Task Accuracy\n")
        f.write("="*100 + "\n\n")
        f.write(table_s1.to_string(index=False))
        f.write("\n\nNote. All models predict overall task accuracy (proportion correct).")
        f.write("\nβ values represent standardized regression coefficients.")
        f.write("\nNone of the regression models yielded statistically significant predictors.")
        f.write(f"\nAll models estimated with N = {len(summary)} participants.\n")
    
    print(f"\n✓ Supplementary Table S1 saved")
    print(f"  N = {len(summary)} participants")
    
    return table_s1


def create_supplementary_table_s2():
    """
    Supplementary Table S2: Distributional Statistics for Behavioral Variables
    
    Reports trial-level and participant-level statistics for:
    - Reaction time (overall and by group)
    - Accuracy (overall and by group)
    """
    print("\n" + "="*80)
    print("SUPPLEMENTARY TABLE S2: Distributional Statistics")
    print("="*80)
    
    # Load trial-level data
    trials = pd.read_csv(OUTPUT_DIR / 'task_trials_long_all_participants.csv')
    
    # Exclude participants
    trials = trials[~trials['sona_id'].isin(EXCLUDED_PARTICIPANTS)].copy()
    
    # Load summary data for trauma groups
    summary = pd.read_csv(OUTPUT_DIR / 'statistical_analyses' / 'data_summary_with_groups.csv')
    summary = summary[~summary['sona_id'].isin(EXCLUDED_PARTICIPANTS)].copy()
    
    # Merge trauma groups
    trials = trials.merge(summary[['sona_id', 'trauma_group']], on='sona_id', how='left')
    
    results = []
    
    # RT statistics (correct trials only)
    rt_data = trials[trials['correct'] == 1]['rt'].dropna()
    rt_overall = {
        'Variable': 'Reaction Time (ms)',
        'Group': 'Overall',
        'N (trials)': f"{len(rt_data):,}",
        'N (participants)': trials['sona_id'].nunique(),
        'Mean': f"{rt_data.mean():.2f}",
        'Median': f"{rt_data.median():.2f}",
        'SD': f"{rt_data.std():.2f}",
        'Min': f"{rt_data.min():.2f}",
        'Max': f"{rt_data.max():.2f}",
        'Skewness': f"{stats.skew(rt_data):.2f}",
        'Kurtosis': f"{stats.kurtosis(rt_data):.2f}"
    }
    results.append(rt_overall)
    
    # RT by group
    for group in ['Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']:
        group_trials = trials[(trials['trauma_group'] == group) & (trials['correct'] == 1)]
        rt_group = group_trials['rt'].dropna()
        
        results.append({
            'Variable': '',
            'Group': group.replace('Trauma - ', ''),
            'N (trials)': f"{len(rt_group):,}",
            'N (participants)': group_trials['sona_id'].nunique(),
            'Mean': f"{rt_group.mean():.2f}",
            'Median': f"{rt_group.median():.2f}",
            'SD': f"{rt_group.std():.2f}",
            'Min': f"{rt_group.min():.2f}",
            'Max': f"{rt_group.max():.2f}",
            'Skewness': f"{stats.skew(rt_group):.2f}",
            'Kurtosis': f"{stats.kurtosis(rt_group):.2f}"
        })
    
    # Accuracy statistics (participant-level)
    acc_overall = summary['accuracy_overall'].dropna()
    results.append({
        'Variable': 'Accuracy (proportion)',
        'Group': 'Overall',
        'N (trials)': f"{len(trials):,}",
        'N (participants)': len(acc_overall),
        'Mean': f"{acc_overall.mean():.3f}",
        'Median': f"{acc_overall.median():.3f}",
        'SD': f"{acc_overall.std():.3f}",
        'Min': f"{acc_overall.min():.3f}",
        'Max': f"{acc_overall.max():.3f}",
        'Skewness': f"{stats.skew(acc_overall):.2f}",
        'Kurtosis': f"{stats.kurtosis(acc_overall):.2f}"
    })
    
    # Accuracy by group
    for group in ['Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']:
        group_data = summary[summary['trauma_group'] == group]['accuracy_overall'].dropna()
        group_trials = trials[trials['trauma_group'] == group]
        
        results.append({
            'Variable': '',
            'Group': group.replace('Trauma - ', ''),
            'N (trials)': f"{len(group_trials):,}",
            'N (participants)': len(group_data),
            'Mean': f"{group_data.mean():.3f}",
            'Median': f"{group_data.median():.3f}",
            'SD': f"{group_data.std():.3f}",
            'Min': f"{group_data.min():.3f}",
            'Max': f"{group_data.max():.3f}",
            'Skewness': f"{stats.skew(group_data):.2f}",
            'Kurtosis': f"{stats.kurtosis(group_data):.2f}"
        })
    
    table_s2 = pd.DataFrame(results)
    
    # Save
    output_dir = FIGURES_DIR / 'publication'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    table_s2.to_csv(output_dir / 'SupplementaryTable_S2.csv', index=False)
    
    with open(output_dir / 'SupplementaryTable_S2.txt', 'w') as f:
        f.write("Supplementary Table S2\n")
        f.write("Distributional Statistics for Behavioural Variables\n")
        f.write("="*100 + "\n\n")
        f.write(table_s2.to_string(index=False))
        f.write("\n\nNote. Reaction time statistics based on correct trials only.")
        f.write("\nAccuracy statistics aggregated at participant level.")
        f.write("\nSkewness and kurtosis values characterize distributional properties;")
        f.write("\nvalues within |skew| < 2, |kurtosis| < 7 are acceptable for parametric analyses.\n")
    
    print(f"\n✓ Supplementary Table S2 saved")
    
    return table_s2


def create_supplementary_figure_s1():
    """
    Supplementary Figure S1: Distributional Assumptions
    
    Five panels:
    A: RT distribution (overlaid groups)
    B-C: Q-Q plots by trauma group
    D: Accuracy distributions (overlaid)
    E: RT violin plots
    """
    print("\n" + "="*80)
    print("SUPPLEMENTARY FIGURE S1: Distributional Assumptions")
    print("="*80)
    
    # Load data
    trials = pd.read_csv(OUTPUT_DIR / 'task_trials_long_all_participants.csv')
    trials = trials[~trials['sona_id'].isin(EXCLUDED_PARTICIPANTS)].copy()
    
    summary = pd.read_csv(OUTPUT_DIR / 'statistical_analyses' / 'data_summary_with_groups.csv')
    summary = summary[~summary['sona_id'].isin(EXCLUDED_PARTICIPANTS)].copy()
    
    # Merge trauma groups
    trials = trials.merge(summary[['sona_id', 'trauma_group']], on='sona_id', how='left')
    
    # Filter correct trials for RT
    rt_data = trials[trials['correct'] == 1].copy()
    
    # Create figure with specific layout
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.4, top=0.96, bottom=0.06, left=0.08, right=0.96)
    
    # Define groups
    groups = ['Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']
    colors = ['#06A77D', '#D62246']
    labels = ['Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']
    
    # Panel A: RT Distribution by Trauma Group (OVERLAID)
    ax_a = fig.add_subplot(gs[0, :])
    for group, color, label in zip(groups, colors, labels):
        group_rt = rt_data[rt_data['trauma_group'] == group]['rt'].dropna()
        ax_a.hist(group_rt, bins=60, color=color, alpha=0.6, edgecolor='black', 
                 linewidth=0.5, label=label)
    
    ax_a.set_xlabel('Reaction Time (ms)', fontsize=11)
    ax_a.set_ylabel('Frequency', fontsize=11)
    ax_a.set_title('A. Reaction Time Distribution by Trauma Group', fontsize=12, fontweight='bold', loc='left')
    ax_a.legend(loc='upper right')
    ax_a.set_xlim(0, 2000)
    
    # Panels B & C: Q-Q Plots by trauma group
    for idx, (group, color) in enumerate(zip(groups, colors)):
        group_rt = rt_data[rt_data['trauma_group'] == group]['rt'].dropna()
        
        ax_qq = fig.add_subplot(gs[1, idx])
        stats.probplot(group_rt, dist="norm", plot=ax_qq)
        
        # Remove default title
        ax_qq.set_title('')
        
        # Customize Q-Q plot
        ax_qq.get_lines()[0].set_markerfacecolor(color)
        ax_qq.get_lines()[0].set_markeredgecolor('none')
        ax_qq.get_lines()[0].set_markersize(3)
        ax_qq.get_lines()[0].set_alpha(1.0)
        ax_qq.get_lines()[1].set_color('black')
        ax_qq.get_lines()[1].set_linewidth(2)
        
        # Add skew and kurtosis
        skew_val = stats.skew(group_rt)
        kurt_val = stats.kurtosis(group_rt)
        ax_qq.text(0.05, 0.95, f'Skew: {skew_val:.2f}\nKurt: {kurt_val:.2f}',
                  transform=ax_qq.transAxes, fontsize=9,
                  verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set our custom title
        ax_qq.set_title(f'{"B" if idx==0 else "C"}. Q-Q Plot: {group.replace("Trauma - ", "")}',
                       fontsize=12, fontweight='bold', loc='left', pad=10)
        ax_qq.set_xlabel('Theoretical Quantiles', fontsize=10)
        ax_qq.set_ylabel('Sample Quantiles (ms)', fontsize=10)
    
    # Panel D: Accuracy Distribution by Trauma Group (OVERLAID)
    ax_d = fig.add_subplot(gs[2, 0])
    for group, color, label in zip(groups, colors, labels):
        group_acc = summary[summary['trauma_group'] == group]['accuracy_overall'].dropna()
        ax_d.hist(group_acc, bins=15, color=color, alpha=0.6, edgecolor='black',
                 linewidth=0.5, label=label)
    
    ax_d.set_xlabel('Accuracy (proportion correct)', fontsize=11)
    ax_d.set_ylabel('Number of Participants', fontsize=11)
    ax_d.set_title('D. Accuracy Distribution by Trauma Group', fontsize=12, fontweight='bold', loc='left')
    ax_d.legend(loc='upper left', fontsize=9)
    
    # Panel E: RT Distribution Comparison (Violin Plot)
    ax_e = fig.add_subplot(gs[2, 1:])
    rt_by_group = [rt_data[rt_data['trauma_group'] == group]['rt'].dropna() for group in groups]
    
    parts = ax_e.violinplot(rt_by_group, positions=[0, 1], showmeans=False, showmedians=True,
                            widths=0.7)
    
    # Color the violin plots
    for i, (pc, color) in enumerate(zip(parts['bodies'], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Customize median lines
    parts['cmedians'].set_edgecolor('black')
    parts['cmedians'].set_linewidth(2)
    
    # Add median values as text
    for i, (group_data, x_pos) in enumerate(zip(rt_by_group, [0, 1])):
        median_val = np.median(group_data)
        ax_e.text(x_pos, median_val + 50, f'Mdn: {median_val:.0f}ms',
                 ha='center', fontsize=9, fontweight='bold')
    
    ax_e.set_xticks([0, 1])
    ax_e.set_xticklabels(['No Ongoing\nImpact', 'Ongoing\nImpact'], fontsize=10)
    ax_e.set_ylabel('Reaction Time (ms)', fontsize=11)
    ax_e.set_title('E. RT Distribution Comparison (Violin Plot)', fontsize=12, fontweight='bold', loc='left')
    ax_e.set_ylim(0, 2000)
    
    # Save
    output_dir = FIGURES_DIR / 'publication'
    fig.savefig(output_dir / 'SupplementaryFigure_S1.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Supplementary Figure S1 saved")


def create_supplementary_figure_s2():
    """
    Supplementary Figure S2: IES-R by Trauma Group (Group Validation)
    
    Boxplots showing IES-R total scores with clinical cutoff line (24).
    """
    print("\n" + "="*80)
    print("SUPPLEMENTARY FIGURE S2: Group Validation")
    print("="*80)
    
    # Load data
    summary = pd.read_csv(OUTPUT_DIR / 'statistical_analyses' / 'data_summary_with_groups.csv')
    summary = summary[~summary['sona_id'].isin(EXCLUDED_PARTICIPANTS)].copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Prepare data
    groups = ['Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']
    colors = ['#06A77D', '#D62246']
    
    data_for_plot = [summary[summary['trauma_group'] == group]['ies_total'].dropna() for group in groups]
    
    # Create boxplot
    bp = ax.boxplot(data_for_plot, positions=[1, 2], widths=0.6,
                    patch_artist=True, showfliers=True,
                    medianprops=dict(color='black', linewidth=2),
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))
    
    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add clinical cutoff line
    ax.axhline(y=24, color='red', linestyle='--', linewidth=2, 
              label='Clinical Cutoff (IES-R = 24)', zorder=0)
    
    # Labels
    ax.set_xticks([1, 2])
    ax.set_xticklabels([g.replace('Trauma - ', '') for g in groups])
    ax.set_ylabel('IES-R Total Score', fontsize=12)
    ax.set_xlabel('Trauma Group', fontsize=12)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 70)
    ax.set_xlim(0.5, 2.5)
    
    # Add group statistics stacked on the left side
    y_positions = [54, 45]
    for i, (group_data, y_pos, color) in enumerate(zip(data_for_plot, y_positions, colors)):
        mean_val = group_data.mean()
        group_label = groups[i].replace('Trauma - ', '')
        ax.text(0.6, y_pos, f'{group_label}\nn = {len(group_data)}\nM = {mean_val:.1f}',
               fontsize=9, ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=color, linewidth=2))
    
    plt.tight_layout()
    
    # Save
    output_dir = FIGURES_DIR / 'publication'
    fig.savefig(output_dir / 'SupplementaryFigure_S2.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Supplementary Figure S2 saved")


def main():
    """Generate all supplementary materials."""
    print("\n" + "="*80)
    print("GENERATING SUPPLEMENTARY MATERIALS")
    print("="*80)
    
    # Create tables
    create_supplementary_table_s1()
    create_supplementary_table_s2()
    
    # Create figures
    create_supplementary_figure_s1()
    create_supplementary_figure_s2()
    
    print("\n" + "="*80)
    print("✓ ALL SUPPLEMENTARY MATERIALS GENERATED")
    print("="*80)
    print(f"\nLocation: {FIGURES_DIR / 'publication'}")
    print("\nFiles created:")
    print("  - SupplementaryTable_S1.csv/.txt")
    print("  - SupplementaryTable_S2.csv/.txt")
    print("  - SupplementaryFigure_S1.png")
    print("  - SupplementaryFigure_S2.png")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
