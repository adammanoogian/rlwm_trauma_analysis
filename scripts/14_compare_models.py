#!/usr/bin/env python
"""
14: Compare Models
==================

Model comparison and winning model analysis.

This MERGED script combines:
1. compare_mle_models.py - AIC/BIC comparison for MLE fits
2. run_model_comparison.py - Bayesian comparison with WAIC/LOO
3. plot_winning_model.py - Generate predictions from winning model

Comparison Workflow:
    1. Load fitted model results (MLE and/or Bayesian)
    2. Compare models using information criteria (AIC, BIC, WAIC, LOO)
    3. Identify winning model per participant and overall
    4. Generate predictions from winning model
    5. Create model comparison visualizations

Information Criteria:
    - AIC: Akaike Information Criterion (moderate complexity penalty)
    - BIC: Bayesian Information Criterion (stronger complexity penalty)
    - WAIC: Widely Applicable IC (fully Bayesian, requires posterior)
    - LOO: Leave-One-Out CV (gold standard, requires posterior)

Interpretation (Burnham & Anderson, 2002):
    - Δ < 2: Models essentially equivalent
    - Δ 2-4: Weak evidence for better model
    - Δ 4-7: Moderate evidence
    - Δ 7-10: Strong evidence
    - Δ > 10: Very strong evidence

Inputs:
    - output/mle_results/*_individual_fits.csv (MLE results)
    - output/bayesian_fits/*.nc (Bayesian posteriors, optional)

Outputs:
    - output/model_comparison/comparison_results.csv
    - output/model_comparison/participant_wins.csv
    - figures/model_comparison/information_criteria.png
    - figures/model_comparison/model_weights.png
    - figures/model_comparison/winning_model_predictions.png

Usage:
    # Compare MLE fits (default)
    python scripts/14_compare_models.py

    # Specify model files explicitly
    python scripts/14_compare_models.py \
        --m1 output/mle_results/qlearning_individual_fits.csv \
        --m2 output/mle_results/wmrl_individual_fits.csv

    # 3-way comparison (M1 vs M2 vs M3)
    python scripts/14_compare_models.py \
        --m1 output/mle_results/qlearning_individual_fits.csv \
        --m2 output/mle_results/wmrl_individual_fits.csv \
        --m3 output/mle_results/wmrl_m3_individual_fits.csv

    # Include Bayesian criteria (WAIC/LOO)
    python scripts/14_compare_models.py --use-waic --bayesian-dir output/bayesian_fits

    # Generate winning model predictions
    python scripts/14_compare_models.py --generate-predictions

Next Steps:
    - Run 15_analyze_mle_by_trauma.py for parameter-trauma relationships
    - Use comparison results for model selection in thesis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from config import EXCLUDED_PARTICIPANTS, FIGURES_DIR

# M4 parameter names (for per-param summary in separate track)
try:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.fitting.mle_utils import WMRL_M4_PARAMS
except ImportError:
    WMRL_M4_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa',
                      'v_scale', 'A', 'delta', 't0']

# ============================================================================
# MLE COMPARISON FUNCTIONS
# ============================================================================

def load_fits(filepath: str) -> pd.DataFrame:
    """Load individual fits CSV."""
    return pd.read_csv(filepath)

def compute_aggregate_ic(fits_df: pd.DataFrame, metric: str = 'aic') -> float:
    """Compute aggregate information criterion across participants."""
    converged = fits_df[fits_df['converged'] == True] if 'converged' in fits_df.columns else fits_df
    if len(converged) == 0:
        converged = fits_df
    return converged[metric].sum()

def compute_akaike_weights_n(aic_values: dict[str, float]) -> dict[str, float]:
    """Compute Akaike weights for N models."""
    min_aic = min(aic_values.values())
    deltas = {model: aic - min_aic for model, aic in aic_values.items()}
    exp_values = {model: np.exp(-0.5 * delta) for model, delta in deltas.items()}
    total = sum(exp_values.values())
    return {model: exp_val / total for model, exp_val in exp_values.items()}

def compare_models_mle(
    fits_dict: dict[str, pd.DataFrame],
    metric: str = 'aic'
) -> pd.DataFrame:
    """Compare N models using aggregate information criteria."""
    agg_ics = {model: compute_aggregate_ic(fits, metric) for model, fits in fits_dict.items()}
    min_ic = min(agg_ics.values())

    results = []
    for model_name, agg_ic in agg_ics.items():
        delta = agg_ic - min_ic
        rel_likelihood = np.exp(-0.5 * delta)
        results.append({
            'model': model_name,
            f'aggregate_{metric}': agg_ic,
            f'delta_{metric}': delta,
            'relative_likelihood': rel_likelihood
        })

    df = pd.DataFrame(results)
    return df.sort_values(f'aggregate_{metric}')

def count_participant_wins(
    fits_dict: dict[str, pd.DataFrame],
    metric: str = 'aic'
) -> pd.DataFrame:
    """Count per-participant model wins."""
    merged = None
    for model_name, fits_df in fits_dict.items():
        cols = ['participant_id', metric]
        if 'converged' in fits_df.columns:
            cols.append('converged')
        model_df = fits_df[cols].copy()
        model_df = model_df.rename(columns={
            metric: f'{metric}_{model_name}',
            'converged': f'converged_{model_name}'
        } if 'converged' in fits_df.columns else {
            metric: f'{metric}_{model_name}'
        })

        if merged is None:
            merged = model_df
        else:
            merged = pd.merge(merged, model_df, on='participant_id', how='inner')

    # Filter to converged fits if available
    converged_cols = [col for col in merged.columns if col.startswith('converged_')]
    if converged_cols:
        all_converged = merged[converged_cols].all(axis=1)
        merged = merged[all_converged]

    # Find winner for each participant
    ic_cols = [col for col in merged.columns if col.startswith(f'{metric}_')]
    merged['winner'] = merged[ic_cols].idxmin(axis=1).str.replace(f'{metric}_', '')

    # Count wins
    win_counts = merged['winner'].value_counts().to_dict()

    summary = []
    for model_name in fits_dict:
        summary.append({
            'model': model_name,
            'wins': win_counts.get(model_name, 0),
            'win_pct': 100 * win_counts.get(model_name, 0) / len(merged) if len(merged) > 0 else 0,
            'total': len(merged)
        })

    return pd.DataFrame(summary)

def interpret_delta(delta: float) -> str:
    """Interpret delta IC following Burnham & Anderson (2002)."""
    abs_delta = abs(delta)
    if abs_delta < 2:
        return 'Essentially equivalent'
    elif abs_delta < 4:
        return 'Weak evidence'
    elif abs_delta < 7:
        return 'Moderate evidence'
    elif abs_delta < 10:
        return 'Strong evidence'
    else:
        return 'Very strong evidence'

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_model_comparison(
    aic_comparison: pd.DataFrame,
    bic_comparison: pd.DataFrame,
    output_dir: Path
) -> None:
    """Create model comparison visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # AIC comparison
    ax = axes[0]
    models = aic_comparison['model'].tolist()
    aics = aic_comparison['aggregate_aic'].tolist()
    colors = sns.color_palette('husl', len(models))

    bars = ax.bar(models, aics, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Aggregate AIC', fontsize=12, fontweight='bold')
    ax.set_title('AIC Comparison (lower is better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, aics):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)

    # BIC comparison
    ax = axes[1]
    bics = bic_comparison['aggregate_bic'].tolist()

    bars = ax.bar(models, bics, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Aggregate BIC', fontsize=12, fontweight='bold')
    ax.set_title('BIC Comparison (lower is better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, bics):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    output_path = output_dir / 'information_criteria_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {output_path}")
    plt.close()

def plot_model_weights(
    weights: dict[str, float],
    output_dir: Path
) -> None:
    """Create model weights bar plot."""
    fig, ax = plt.subplots(figsize=(8, 5))

    models = list(weights.keys())
    weight_vals = [weights[m] for m in models]
    colors = sns.color_palette('husl', len(models))

    bars = ax.bar(models, weight_vals, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Akaike Weight', fontsize=12, fontweight='bold')
    ax.set_title('Model Weights (probability of being best model)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, weight_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    output_path = output_dir / 'model_weights.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {output_path}")
    plt.close()

def plot_participant_wins(
    wins_df: pd.DataFrame,
    metric: str,
    output_dir: Path
) -> None:
    """Create participant wins bar plot."""
    fig, ax = plt.subplots(figsize=(8, 5))

    models = wins_df['model'].tolist()
    wins = wins_df['wins'].tolist()
    colors = sns.color_palette('husl', len(models))

    bars = ax.bar(models, wins, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Number of Participants', fontsize=12, fontweight='bold')
    ax.set_title(f'Winning Model by {metric.upper()} (per participant)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, (w, pct) in zip(bars, zip(wins, wins_df['win_pct'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{w}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    output_path = output_dir / f'participant_wins_{metric}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {output_path}")
    plt.close()

# ============================================================================
# STRATIFIED COMPARISON FUNCTIONS
# ============================================================================

def get_per_participant_winners(
    fits_dict: dict[str, pd.DataFrame],
    metric: str = 'aic'
) -> pd.DataFrame:
    """Get per-participant winning model.

    Returns DataFrame with columns: participant_id, winner, and per-model IC values.
    """
    merged = None
    for model_name, fits_df in fits_dict.items():
        cols = ['participant_id', metric]
        model_df = fits_df[cols].copy()
        model_df = model_df.rename(columns={metric: f'{metric}_{model_name}'})
        if merged is None:
            merged = model_df
        else:
            merged = pd.merge(merged, model_df, on='participant_id', how='inner')

    ic_cols = [col for col in merged.columns if col.startswith(f'{metric}_')]
    merged['winner'] = merged[ic_cols].idxmin(axis=1).str.replace(f'{metric}_', '')
    return merged

def stratified_comparison(
    fits_dict: dict[str, pd.DataFrame],
    output_dir: Path,
    figures_dir: Path,
    metric: str = 'aic'
) -> None:
    """Compare winning models stratified by trauma group.

    Loads trauma group assignments, merges with per-participant model wins,
    and reports: crosstab, Fisher's exact test, per-group Akaike weights.
    """
    groups_path = Path('output/trauma_groups/group_assignments.csv')
    if not groups_path.exists():
        print("\n[SKIP] Stratified comparison: group_assignments.csv not found")
        return

    # Load group assignments
    groups_df = pd.read_csv(groups_path)
    groups_df['sona_id'] = groups_df['sona_id'].astype(str)

    # Exclude participants per config
    excluded_str = [str(x) for x in EXCLUDED_PARTICIPANTS]
    groups_df = groups_df[~groups_df['sona_id'].isin(excluded_str)].copy()

    # Shorten group labels for display
    label_map = {
        'Trauma Exposure - No Ongoing Impact': 'No Ongoing Impact',
        'Trauma Exposure - Ongoing Impact': 'Ongoing Impact',
    }
    groups_df['group'] = groups_df['hypothesis_group'].map(label_map).fillna(
        groups_df['hypothesis_group']
    )

    # Get per-participant winners
    winners = get_per_participant_winners(fits_dict, metric)
    winners['sona_id'] = winners['participant_id'].astype(str)

    # Merge with groups
    merged = winners.merge(groups_df[['sona_id', 'group']], on='sona_id', how='inner')
    print(f"\n  Matched {len(merged)} participants with group assignments")

    if len(merged) == 0:
        print("  [SKIP] No matched participants")
        return

    # ---- Crosstab: winner x group ----
    ct = pd.crosstab(merged['winner'], merged['group'], margins=True)
    print(f"\n  Crosstab (winner x group, {metric.upper()}):")
    print(ct.to_string())

    # Save crosstab
    ct_path = output_dir / 'model_group_crosstab.csv'
    ct.to_csv(ct_path)
    print(f"  [SAVED] {ct_path}")

    # ---- Fisher's exact test (2x2: focus on M2 vs M3) ----
    # Build 2x2 table for models that actually have wins
    unique_models = sorted(merged['winner'].unique())
    unique_groups = sorted(merged['group'].unique())

    if len(unique_models) >= 2 and len(unique_groups) == 2:
        # For 2x2 Fisher's: use the two most common models
        top_models = merged['winner'].value_counts().head(2).index.tolist()
        subset = merged[merged['winner'].isin(top_models)]
        ct_2x2 = pd.crosstab(subset['winner'], subset['group'])

        if ct_2x2.shape == (2, 2):
            odds_ratio, fisher_p = scipy_stats.fisher_exact(ct_2x2.values)
            print(f"\n  Fisher's exact test ({top_models[0]} vs {top_models[1]}):")
            print(f"    Odds ratio = {odds_ratio:.3f}")
            print(f"    p = {fisher_p:.4f}")
            if fisher_p < 0.05:
                print("    -> Significant: model preference differs by group")
            else:
                print("    -> Not significant: no evidence model preference differs by group")
        else:
            fisher_p = np.nan
            odds_ratio = np.nan
            print(f"\n  Fisher's exact test: cannot form 2x2 table (shape={ct_2x2.shape})")
    else:
        fisher_p = np.nan
        odds_ratio = np.nan
        print("\n  Fisher's exact test: need 2 groups and 2+ models")

    # ---- Per-group Akaike weights ----
    print(f"\n  Per-group aggregate {metric.upper()} and Akaike weights:")

    group_results = []
    for group_name in unique_groups:
        group_ids = merged.loc[merged['group'] == group_name, 'sona_id'].values

        # Filter each model's fits to this group
        group_fits = {}
        for model_name, fits_df in fits_dict.items():
            mask = fits_df['participant_id'].astype(str).isin(group_ids)
            group_fits[model_name] = fits_df[mask]

        # Aggregate IC per group
        group_ics = {m: compute_aggregate_ic(f, metric) for m, f in group_fits.items()}
        group_weights = compute_akaike_weights_n(group_ics)
        n_group = len(group_ids)

        # Per-group win counts
        group_winners = merged.loc[merged['group'] == group_name, 'winner']
        win_counts = group_winners.value_counts().to_dict()

        print(f"\n  {group_name} (n={n_group}):")
        for model_name in fits_dict:
            wins = win_counts.get(model_name, 0)
            pct = 100 * wins / n_group if n_group > 0 else 0
            print(f"    {model_name}: {metric.upper()}={group_ics[model_name]:.0f}, "
                  f"weight={group_weights[model_name]:.4f}, "
                  f"wins={wins} ({pct:.1f}%)")

        for model_name in fits_dict:
            group_results.append({
                'group': group_name,
                'n': n_group,
                'model': model_name,
                f'aggregate_{metric}': group_ics[model_name],
                'akaike_weight': group_weights[model_name],
                'wins': win_counts.get(model_name, 0),
                'win_pct': 100 * win_counts.get(model_name, 0) / n_group if n_group > 0 else 0,
            })

    # Save stratified results
    strat_df = pd.DataFrame(group_results)
    strat_path = output_dir / 'stratified_results.csv'
    strat_df.to_csv(strat_path, index=False)
    print(f"\n  [SAVED] {strat_path}")

    # ---- Visualization: stacked bar chart ----
    plot_stratified_wins(merged, metric, figures_dir)

    # ---- Summary row with Fisher's test ----
    if not np.isnan(fisher_p):
        summary_row = pd.DataFrame([{
            'group': 'Fisher_exact_test',
            'n': len(merged),
            'model': f"{top_models[0]}_vs_{top_models[1]}",
            f'aggregate_{metric}': np.nan,
            'akaike_weight': np.nan,
            'wins': np.nan,
            'win_pct': odds_ratio,
        }])
        strat_df = pd.concat([strat_df, summary_row], ignore_index=True)
        strat_df.to_csv(strat_path, index=False)

def plot_stratified_wins(
    merged: pd.DataFrame,
    metric: str,
    figures_dir: Path
) -> None:
    """Create stacked bar chart of model wins by trauma group."""
    fig, ax = plt.subplots(figsize=(8, 5))

    groups = sorted(merged['group'].unique())
    models = sorted(merged['winner'].unique())
    colors = sns.color_palette('husl', len(models))
    color_map = dict(zip(models, colors))

    x = np.arange(len(groups))
    bar_width = 0.5
    bottoms = np.zeros(len(groups))

    for model in models:
        heights = []
        for group in groups:
            n_group = (merged['group'] == group).sum()
            n_wins = ((merged['group'] == group) & (merged['winner'] == model)).sum()
            pct = 100 * n_wins / n_group if n_group > 0 else 0
            heights.append(pct)

        bars = ax.bar(x, heights, bar_width, bottom=bottoms,
                      label=model, color=color_map[model], alpha=0.85, edgecolor='black')

        # Add count labels inside bars
        for xi, h, b in zip(x, heights, bottoms):
            if h > 5:  # Only label if bar segment is large enough
                n_wins = int(round(h * (merged['group'] == groups[xi]).sum() / 100))
                ax.text(xi, b + h / 2, f'{n_wins}\n({h:.0f}%)',
                        ha='center', va='center', fontsize=9, fontweight='bold')

        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel('Percentage of Participants', fontsize=12, fontweight='bold')
    ax.set_title(f'Winning Model by Trauma Group ({metric.upper()})',
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 110])
    ax.legend(title='Winning Model', fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add group sizes
    for xi, group in enumerate(groups):
        n = (merged['group'] == group).sum()
        ax.text(xi, 103, f'n={n}', ha='center', va='bottom', fontsize=10, fontstyle='italic')

    plt.tight_layout()

    output_path = figures_dir / 'stratified_wins.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [SAVED] {output_path}")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def find_mle_files(mle_dir: Path) -> dict[str, Path]:
    """Auto-detect MLE result files.

    Searches mle_dir first, then falls back to output/ root
    (some models may have been fit with --output output instead of --output output/mle/).
    """
    files = {}

    # Look for standard naming patterns
    patterns = {
        'M1': ['qlearning_individual_fits.csv', 'qlearning_mle_results.csv'],
        'M2': ['wmrl_individual_fits.csv', 'wmrl_mle_results.csv'],
        'M3': ['wmrl_m3_individual_fits.csv', 'wmrl_m3_mle_results.csv'],
        'M4': ['wmrl_m4_individual_fits.csv', 'wmrl_m4_mle_results.csv'],
        'M5': ['wmrl_m5_individual_fits.csv', 'wmrl_m5_mle_results.csv'],
        'M6a': ['wmrl_m6a_individual_fits.csv', 'wmrl_m6a_mle_results.csv'],
        'M6b': ['wmrl_m6b_individual_fits.csv', 'wmrl_m6b_mle_results.csv'],
    }

    # Also search output/ root as fallback
    fallback_dir = Path('output')
    search_dirs = [mle_dir, fallback_dir] if mle_dir != fallback_dir else [mle_dir]

    for model, filenames in patterns.items():
        for search_dir in search_dirs:
            for filename in filenames:
                filepath = search_dir / filename
                if filepath.exists():
                    files[model] = filepath
                    break
            if model in files:
                break

    return files

def main():
    parser = argparse.ArgumentParser(
        description='Compare computational models using information criteria'
    )

    # Model file arguments
    parser.add_argument('--m1', type=str, default=None,
                        help='Path to M1 (Q-Learning) MLE results')
    parser.add_argument('--m2', type=str, default=None,
                        help='Path to M2 (WM-RL) MLE results')
    parser.add_argument('--m3', type=str, default=None,
                        help='Path to M3 (WM-RL+kappa) MLE results')
    parser.add_argument('--m5', type=str, default=None,
                        help='Path to M5 (WM-RL+phi_rl) individual fits CSV')
    parser.add_argument('--m6a', type=str, default=None,
                        help='Path to M6a (WM-RL+kappa_s) individual fits CSV')
    parser.add_argument('--m6b', type=str, default=None,
                        help='Path to M6b (WM-RL+kappa+kappa_s dual) individual fits CSV')
    parser.add_argument('--m4', type=str, default=None,
                        help='Path to M4 (WM-RL+LBA joint choice+RT) individual fits CSV')

    # Legacy arguments
    parser.add_argument('--qlearning', type=str, default=None,
                        help='Path to Q-Learning fits (legacy, same as --m1)')
    parser.add_argument('--wmrl', type=str, default=None,
                        help='Path to WM-RL fits (legacy, same as --m2)')

    # Options
    parser.add_argument('--mle-dir', type=str, default='output/mle',
                        help='Directory containing MLE results (for auto-detection)')
    parser.add_argument('--output-dir', type=str, default='output/model_comparison',
                        help='Output directory for results')
    parser.add_argument('--use-waic', action='store_true',
                        help='Include WAIC/LOO from Bayesian fits')
    parser.add_argument('--bayesian-dir', type=str, default='output/bayesian_fits',
                        help='Directory containing Bayesian fits')
    parser.add_argument('--generate-predictions', action='store_true',
                        help='Generate predictions from winning model')

    args = parser.parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = FIGURES_DIR / 'model_comparison'
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    # Build fits_dict from provided paths or auto-detect
    fits_dict = {}

    # Handle legacy arguments
    if args.qlearning:
        args.m1 = args.qlearning
    if args.wmrl:
        args.m2 = args.wmrl

    # Load explicitly provided models
    if args.m1:
        fits_dict['M1'] = load_fits(args.m1)
    if args.m2:
        fits_dict['M2'] = load_fits(args.m2)
    if args.m3:
        fits_dict['M3'] = load_fits(args.m3)
    if args.m5:
        fits_dict['M5'] = load_fits(args.m5)
    if args.m6a:
        fits_dict['M6a'] = load_fits(args.m6a)
    if args.m6b:
        fits_dict['M6b'] = load_fits(args.m6b)
    if args.m4:
        fits_dict['M4'] = load_fits(args.m4)

    # Auto-detect if no models provided
    if not fits_dict:
        print("\nNo model files specified, auto-detecting...")
        mle_dir = Path(args.mle_dir)
        if mle_dir.exists():
            detected = find_mle_files(mle_dir)
            for model, filepath in detected.items():
                print(f"  Found {model}: {filepath}")
                fits_dict[model] = load_fits(str(filepath))

    # ==============================
    # SEPARATE M4 FROM CHOICE-ONLY MODELS
    # M4 uses joint choice+RT likelihood; its AIC/BIC is NOT comparable
    # to choice-only models (M1, M2, M3, M5, M6a, M6b).
    # ==============================
    m4_fits = fits_dict.pop('M4', None)
    choice_only_dict = fits_dict  # All remaining models are choice-only

    if len(choice_only_dict) < 2 and m4_fits is None:
        print("\nERROR: At least 2 models required for comparison.")
        print("Provide paths via --m1/--m2/--m3 or ensure MLE results exist in output/mle_results/")
        return

    if len(choice_only_dict) < 2 and m4_fits is not None:
        print("\nNote: Only M4 found -- no choice-only comparison possible.")
        print("      M4 will be reported in its separate track below.")

    # Print loaded models
    if choice_only_dict:
        print(f"\nLoaded {len(choice_only_dict)} choice-only models (AIC/BIC comparable):")
        for model_name, fits_df in choice_only_dict.items():
            n_converged = fits_df['converged'].sum() if 'converged' in fits_df.columns else len(fits_df)
            print(f"  {model_name}: {n_converged}/{len(fits_df)} converged")
    if m4_fits is not None:
        n_converged = m4_fits['converged'].sum() if 'converged' in m4_fits.columns else len(m4_fits)
        print(f"\nLoaded M4 (joint choice+RT, SEPARATE track): {n_converged}/{len(m4_fits)} converged")

    # ==============================
    # AIC/BIC COMPARISON (choice-only models only)
    # ==============================
    print("\n" + "-" * 80)
    print("INFORMATION CRITERIA COMPARISON (Choice-Only Models: M1, M2, M3, M5, M6a, M6b)")
    print("-" * 80)

    if len(choice_only_dict) < 2:
        print("\n[SKIP] Fewer than 2 choice-only models available -- skipping AIC/BIC comparison.")
        aic_comparison = bic_comparison = pd.DataFrame()
        weights = {}
        best_aic = best_bic = None
    else:
        aic_comparison = compare_models_mle(choice_only_dict, 'aic')
        bic_comparison = compare_models_mle(choice_only_dict, 'bic')

    if len(choice_only_dict) >= 2:
        print("\nAIC Comparison:")
        print(aic_comparison.to_string(index=False))

        print("\nBIC Comparison:")
        print(bic_comparison.to_string(index=False))

        # Akaike weights
        aic_values = {row['model']: row['aggregate_aic'] for _, row in aic_comparison.iterrows()}
        weights = compute_akaike_weights_n(aic_values)

        print("\nAkaike Weights:")
        for model, weight in sorted(weights.items()):
            print(f"  {model}: {weight:.4f} ({100*weight:.2f}%)")

        # Interpretation
        best_aic = aic_comparison.iloc[0]['model']
        best_bic = bic_comparison.iloc[0]['model']

        print(f"\n>>> Best model (AIC): {best_aic}")
        print(f">>> Best model (BIC): {best_bic}")

        for _, row in aic_comparison.iloc[1:].iterrows():
            delta = row['delta_aic']
            print(f"    {row['model']} vs {best_aic}: dAIC = {delta:.2f} ({interpret_delta(delta)})")

    # ==============================
    # PER-PARTICIPANT COMPARISON (choice-only)
    # ==============================
    if len(choice_only_dict) >= 2:
        print("\n" + "-" * 80)
        print("PER-PARTICIPANT COMPARISON (Choice-Only)")
        print("-" * 80)

        for metric in ['aic', 'bic']:
            wins = count_participant_wins(choice_only_dict, metric)
            total = wins['total'].iloc[0]
            print(f"\n{metric.upper()} (n={total} participants):")
            for _, row in wins.iterrows():
                print(f"  {row['model']} wins: {row['wins']} ({row['win_pct']:.1f}%)")

            # Save participant wins plot
            plot_participant_wins(wins, metric, figures_dir)

    # ==============================
    # VISUALIZATIONS (choice-only)
    # ==============================
    if len(choice_only_dict) >= 2:
        print("\n" + "-" * 80)
        print("CREATING VISUALIZATIONS")
        print("-" * 80)

        plot_model_comparison(aic_comparison, bic_comparison, figures_dir)
        plot_model_weights(weights, figures_dir)

    # ==============================
    # SAVE RESULTS (choice-only)
    # ==============================
    print("\n" + "-" * 80)
    print("SAVING RESULTS")
    print("-" * 80)

    if len(choice_only_dict) >= 2:
        # Combine results
        results = aic_comparison.merge(
            bic_comparison[['model', 'aggregate_bic', 'delta_bic']],
            on='model'
        )
        results['akaike_weight'] = results['model'].map(weights)

        results_path = output_dir / 'comparison_results.csv'
        results.to_csv(results_path, index=False)
        print(f"[SAVED] {results_path}")

    # Save participant wins
    if len(choice_only_dict) >= 2:
        aic_wins = count_participant_wins(choice_only_dict, 'aic')
        bic_wins = count_participant_wins(choice_only_dict, 'bic')
        wins_combined = aic_wins.merge(
            bic_wins[['model', 'wins']].rename(columns={'wins': 'bic_wins'}),
            on='model'
        ).rename(columns={'wins': 'aic_wins'})

        wins_path = output_dir / 'participant_wins.csv'
        wins_combined.to_csv(wins_path, index=False)
        print(f"[SAVED] {wins_path}")

    # ==============================
    # STRATIFIED BY TRAUMA GROUP (choice-only)
    # ==============================
    if len(choice_only_dict) >= 2:
        print("\n" + "-" * 80)
        print("STRATIFIED COMPARISON (by trauma group)")
        print("-" * 80)

        stratified_comparison(choice_only_dict, output_dir, figures_dir, metric='aic')

    # ==============================
    # M4 SEPARATE TRACK (Joint Choice+RT)
    # ==============================
    if m4_fits is not None:
        print("\n" + "=" * 60)
        print("M4 (Joint Choice+RT) - SEPARATE TRACK")
        print("=" * 60)
        print("NOTE: M4 uses a joint choice+RT likelihood. Its AIC/BIC is NOT")
        print("comparable to choice-only models (M1, M2, M3, M5, M6a, M6b).")
        n_m4 = len(m4_fits)
        n_converged_m4 = m4_fits['converged'].sum() if 'converged' in m4_fits.columns else n_m4
        print(f"  M4 participants: {n_converged_m4}/{n_m4} converged")
        if n_converged_m4 > 0:
            m4_conv = m4_fits[m4_fits['converged'] == True] if 'converged' in m4_fits.columns else m4_fits
            print(f"  M4 total AIC:    {m4_conv['aic'].sum():.2f}")
            print(f"  M4 total BIC:    {m4_conv['bic'].sum():.2f}")
            print(f"  M4 mean NLL:     {m4_conv['nll'].mean():.2f} +/- {m4_conv['nll'].sem():.2f}")
            print("  Parameter summary (mean +/- SEM):")
            for param in WMRL_M4_PARAMS:
                if param in m4_conv.columns:
                    print(f"    {param:12s}: {m4_conv[param].mean():.3f} +/- {m4_conv[param].sem():.3f}")
        # Save M4 summary
        m4_summary_path = output_dir / 'm4_separate_track_summary.csv'
        m4_fits.to_csv(m4_summary_path, index=False)
        print(f"  [SAVED] {m4_summary_path}")
        print("=" * 60)

    # ==============================
    # SUMMARY
    # ==============================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if best_aic is not None:
        print(f"\nPreferred model (AIC): {best_aic}")
        print(f"Preferred model (BIC): {best_bic}")

        if best_aic == best_bic:
            print(f"\n[OK] Both criteria agree: {best_aic} is the preferred model")
        else:
            print(f"\n[NOTE] Criteria disagree - AIC favors {best_aic}, BIC favors {best_bic}")
            print("  (BIC applies stronger penalty for model complexity)")
    else:
        print("\n[NOTE] No choice-only models compared.")

    if m4_fits is not None:
        print("\n[NOTE] M4 (joint choice+RT) was loaded but NOT included in choice-only AIC table.")
        print(f"       See: {output_dir}/m4_separate_track_summary.csv")

    print(f"\nResults saved to: {output_dir}/")
    print(f"Figures saved to: {figures_dir}/")

if __name__ == '__main__':
    main()
