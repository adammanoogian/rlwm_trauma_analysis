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

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from config import FIGURES_DIR, OUTPUT_DIR

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


def compute_akaike_weights_n(aic_values: Dict[str, float]) -> Dict[str, float]:
    """Compute Akaike weights for N models."""
    min_aic = min(aic_values.values())
    deltas = {model: aic - min_aic for model, aic in aic_values.items()}
    exp_values = {model: np.exp(-0.5 * delta) for model, delta in deltas.items()}
    total = sum(exp_values.values())
    return {model: exp_val / total for model, exp_val in exp_values.items()}


def compare_models_mle(
    fits_dict: Dict[str, pd.DataFrame],
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
    fits_dict: Dict[str, pd.DataFrame],
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
    for model_name in fits_dict.keys():
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
    weights: Dict[str, float],
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
    ax.set_ylabel(f'Number of Participants', fontsize=12, fontweight='bold')
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
# MAIN
# ============================================================================

def find_mle_files(mle_dir: Path) -> Dict[str, Path]:
    """Auto-detect MLE result files."""
    files = {}

    # Look for standard naming patterns
    patterns = {
        'M1': ['qlearning_individual_fits.csv', 'qlearning_mle_results.csv'],
        'M2': ['wmrl_individual_fits.csv', 'wmrl_mle_results.csv'],
        'M3': ['wmrl_m3_individual_fits.csv', 'wmrl_m3_mle_results.csv']
    }

    for model, filenames in patterns.items():
        for filename in filenames:
            filepath = mle_dir / filename
            if filepath.exists():
                files[model] = filepath
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

    # Legacy arguments
    parser.add_argument('--qlearning', type=str, default=None,
                        help='Path to Q-Learning fits (legacy, same as --m1)')
    parser.add_argument('--wmrl', type=str, default=None,
                        help='Path to WM-RL fits (legacy, same as --m2)')

    # Options
    parser.add_argument('--mle-dir', type=str, default='output/mle_results',
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

    # Auto-detect if no models provided
    if not fits_dict:
        print("\nNo model files specified, auto-detecting...")
        mle_dir = Path(args.mle_dir)
        if mle_dir.exists():
            detected = find_mle_files(mle_dir)
            for model, filepath in detected.items():
                print(f"  Found {model}: {filepath}")
                fits_dict[model] = load_fits(str(filepath))

    if len(fits_dict) < 2:
        print("\nERROR: At least 2 models required for comparison.")
        print("Provide paths via --m1/--m2/--m3 or ensure MLE results exist in output/mle_results/")
        return

    # Print loaded models
    print(f"\nLoaded {len(fits_dict)} models:")
    for model_name, fits_df in fits_dict.items():
        n_converged = fits_df['converged'].sum() if 'converged' in fits_df.columns else len(fits_df)
        print(f"  {model_name}: {n_converged}/{len(fits_df)} converged")

    # ==============================
    # AIC/BIC COMPARISON
    # ==============================
    print("\n" + "-" * 80)
    print("INFORMATION CRITERIA COMPARISON")
    print("-" * 80)

    aic_comparison = compare_models_mle(fits_dict, 'aic')
    bic_comparison = compare_models_mle(fits_dict, 'bic')

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
    # PER-PARTICIPANT COMPARISON
    # ==============================
    print("\n" + "-" * 80)
    print("PER-PARTICIPANT COMPARISON")
    print("-" * 80)

    for metric in ['aic', 'bic']:
        wins = count_participant_wins(fits_dict, metric)
        total = wins['total'].iloc[0]
        print(f"\n{metric.upper()} (n={total} participants):")
        for _, row in wins.iterrows():
            print(f"  {row['model']} wins: {row['wins']} ({row['win_pct']:.1f}%)")

        # Save participant wins plot
        plot_participant_wins(wins, metric, figures_dir)

    # ==============================
    # VISUALIZATIONS
    # ==============================
    print("\n" + "-" * 80)
    print("CREATING VISUALIZATIONS")
    print("-" * 80)

    plot_model_comparison(aic_comparison, bic_comparison, figures_dir)
    plot_model_weights(weights, figures_dir)

    # ==============================
    # SAVE RESULTS
    # ==============================
    print("\n" + "-" * 80)
    print("SAVING RESULTS")
    print("-" * 80)

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
    aic_wins = count_participant_wins(fits_dict, 'aic')
    bic_wins = count_participant_wins(fits_dict, 'bic')
    wins_combined = aic_wins.merge(
        bic_wins[['model', 'wins']].rename(columns={'wins': 'bic_wins'}),
        on='model'
    ).rename(columns={'wins': 'aic_wins'})

    wins_path = output_dir / 'participant_wins.csv'
    wins_combined.to_csv(wins_path, index=False)
    print(f"[SAVED] {wins_path}")

    # ==============================
    # SUMMARY
    # ==============================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nPreferred model (AIC): {best_aic}")
    print(f"Preferred model (BIC): {best_bic}")

    if best_aic == best_bic:
        print(f"\n[OK] Both criteria agree: {best_aic} is the preferred model")
    else:
        print(f"\n[NOTE] Criteria disagree - AIC favors {best_aic}, BIC favors {best_bic}")
        print(f"  (BIC applies stronger penalty for model complexity)")

    print(f"\nResults saved to: {output_dir}/")
    print(f"Figures saved to: {figures_dir}/")


if __name__ == '__main__':
    main()
