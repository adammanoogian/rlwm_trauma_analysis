"""
Model Comparison for MLE Fits

Compare Q-Learning, WM-RL, and WM-RL+kappa models using AIC and BIC.
Following Senta et al. (2025) and Burnham & Anderson (2002) methodology.

Usage:
    # Legacy 2-way comparison (backward compatible)
    python scripts/fitting/compare_mle_models.py \
        --qlearning models/mle/qlearning_individual_fits.csv \
        --wmrl models/mle/wmrl_individual_fits.csv

    # 3-way comparison (M1 vs M2 vs M3)
    python scripts/fitting/compare_mle_models.py \
        --m1 models/mle/qlearning_individual_fits.csv \
        --m2 models/mle/wmrl_individual_fits.csv \
        --m3 models/mle/wmrl_m3_individual_fits.csv

    # Focused 2-way comparison (M2 vs M3 for kappa analysis)
    python scripts/fitting/compare_mle_models.py \
        --m2 models/mle/wmrl_individual_fits.csv \
        --m3 models/mle/wmrl_m3_individual_fits.csv
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def load_fits(filepath: str) -> pd.DataFrame:
    """Load individual fits CSV."""
    df = pd.read_csv(filepath)
    return df


def compute_aggregate_ic(fits_df: pd.DataFrame, metric: str = 'aic') -> float:
    """
    Compute aggregate information criterion across participants.

    Following Senta et al.: sum AIC/BIC across all participants.
    """
    converged = fits_df[fits_df['converged'] == True]
    if len(converged) == 0:
        converged = fits_df
    return converged[metric].sum()


def count_winning_model(
    ql_fits: pd.DataFrame,
    wmrl_fits: pd.DataFrame,
    metric: str = 'aic'
) -> Dict[str, int]:
    """
    Count how many participants favor each model.

    For each participant, compare their AIC/BIC for each model.
    Lower is better.
    """
    # Merge on participant_id
    merged = pd.merge(
        ql_fits[['participant_id', metric, 'converged']].rename(columns={metric: f'{metric}_ql', 'converged': 'converged_ql'}),
        wmrl_fits[['participant_id', metric, 'converged']].rename(columns={metric: f'{metric}_wmrl', 'converged': 'converged_wmrl'}),
        on='participant_id',
        how='inner'
    )

    # Only compare where both converged
    both_converged = merged[(merged['converged_ql'] == True) & (merged['converged_wmrl'] == True)]

    n_ql_wins = (both_converged[f'{metric}_ql'] < both_converged[f'{metric}_wmrl']).sum()
    n_wmrl_wins = (both_converged[f'{metric}_wmrl'] < both_converged[f'{metric}_ql']).sum()
    n_tie = len(both_converged) - n_ql_wins - n_wmrl_wins

    return {
        'qlearning_wins': n_ql_wins,
        'wmrl_wins': n_wmrl_wins,
        'ties': n_tie,
        'total': len(both_converged)
    }


def compute_delta_ic(
    ql_fits: pd.DataFrame,
    wmrl_fits: pd.DataFrame,
    metric: str = 'aic'
) -> pd.DataFrame:
    """
    Compute per-participant delta IC (WMRL - QL).

    Negative delta means WM-RL is better.
    Positive delta means Q-Learning is better.
    """
    merged = pd.merge(
        ql_fits[['participant_id', metric]].rename(columns={metric: f'{metric}_ql'}),
        wmrl_fits[['participant_id', metric]].rename(columns={metric: f'{metric}_wmrl'}),
        on='participant_id',
        how='inner'
    )

    merged[f'delta_{metric}'] = merged[f'{metric}_wmrl'] - merged[f'{metric}_ql']

    return merged


def interpret_delta_aic(delta: float) -> str:
    """
    Interpret delta AIC following Burnham & Anderson (2002).

    delta AIC = AIC_candidate - AIC_best
    """
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


def compute_akaike_weights(aic_ql: float, aic_wmrl: float) -> Tuple[float, float]:
    """
    Compute Akaike weights from aggregate AICs.

    w_i = exp(-0.5 * delta_i) / sum(exp(-0.5 * delta_j))
    """
    min_aic = min(aic_ql, aic_wmrl)
    delta_ql = aic_ql - min_aic
    delta_wmrl = aic_wmrl - min_aic

    exp_ql = np.exp(-0.5 * delta_ql)
    exp_wmrl = np.exp(-0.5 * delta_wmrl)
    total = exp_ql + exp_wmrl

    return exp_ql / total, exp_wmrl / total


def compute_akaike_weights_n(aic_values: Dict[str, float]) -> Dict[str, float]:
    """
    Compute Akaike weights for N models.

    Args:
        aic_values: Dictionary mapping model names to aggregate AIC values

    Returns:
        Dictionary mapping model names to Akaike weights (sum to 1.0)

    Following Burnham & Anderson (2002):
        w_i = exp(-0.5 * delta_i) / sum(exp(-0.5 * delta_j))
    """
    min_aic = min(aic_values.values())
    deltas = {model: aic - min_aic for model, aic in aic_values.items()}

    exp_values = {model: np.exp(-0.5 * delta) for model, delta in deltas.items()}
    total = sum(exp_values.values())

    weights = {model: exp_val / total for model, exp_val in exp_values.items()}

    return weights


def compare_models(
    fits_dict: Dict[str, pd.DataFrame],
    metric: str = 'aic'
) -> pd.DataFrame:
    """
    Compare N models using aggregate information criteria.

    Args:
        fits_dict: Dictionary mapping model names (e.g., 'M1', 'M2', 'M3') to fit DataFrames
        metric: Information criterion to use ('aic' or 'bic')

    Returns:
        DataFrame with columns: model, aggregate_ic, delta_ic, relative_likelihood
    """
    # Compute aggregate IC for each model
    agg_ics = {}
    for model_name, fits_df in fits_dict.items():
        agg_ics[model_name] = compute_aggregate_ic(fits_df, metric)

    # Find best model (minimum IC)
    min_ic = min(agg_ics.values())

    # Compute deltas and relative likelihoods
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
    df = df.sort_values(f'aggregate_{metric}')  # Best model first

    return df


def count_participant_wins_n(
    fits_dict: Dict[str, pd.DataFrame],
    metric: str = 'aic'
) -> pd.DataFrame:
    """
    Count per-participant model wins for N models.

    Args:
        fits_dict: Dictionary mapping model names to fit DataFrames
        metric: Information criterion to use ('aic' or 'bic')

    Returns:
        DataFrame with participant-level comparison and winner counts
    """
    # Merge all models on participant_id
    merged = None
    for model_name, fits_df in fits_dict.items():
        model_df = fits_df[['participant_id', metric, 'converged']].rename(
            columns={metric: f'{metric}_{model_name}', 'converged': f'converged_{model_name}'}
        )
        if merged is None:
            merged = model_df
        else:
            merged = pd.merge(merged, model_df, on='participant_id', how='inner')

    # Filter to participants where all models converged
    converged_cols = [col for col in merged.columns if col.startswith('converged_')]
    all_converged = merged[converged_cols].all(axis=1)
    merged = merged[all_converged]

    # For each participant, find winning model (lowest IC)
    ic_cols = [col for col in merged.columns if col.startswith(f'{metric}_')]
    merged['winner'] = merged[ic_cols].idxmin(axis=1).str.replace(f'{metric}_', '')

    # Count wins per model
    win_counts = merged['winner'].value_counts().to_dict()

    # Build summary DataFrame
    summary = []
    for model_name in fits_dict.keys():
        summary.append({
            'model': model_name,
            'wins': win_counts.get(model_name, 0),
            'win_pct': 100 * win_counts.get(model_name, 0) / len(merged) if len(merged) > 0 else 0
        })

    summary_df = pd.DataFrame(summary)
    summary_df['total'] = len(merged)

    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description='Compare MLE fits between Q-Learning, WM-RL, and WM-RL+kappa models'
    )

    # New-style arguments (explicit model naming)
    parser.add_argument('--m1', type=str, default=None,
                        help='Path to M1 (Q-Learning) individual fits CSV')
    parser.add_argument('--m2', type=str, default=None,
                        help='Path to M2 (WM-RL) individual fits CSV')
    parser.add_argument('--m3', type=str, default=None,
                        help='Path to M3 (WM-RL+kappa) individual fits CSV')

    # Legacy arguments (backward compatible)
    parser.add_argument('--qlearning', type=str, default=None,
                        help='Path to Q-Learning individual fits CSV (legacy, same as --m1)')
    parser.add_argument('--wmrl', type=str, default=None,
                        help='Path to WM-RL individual fits CSV (legacy, same as --m2)')

    parser.add_argument('--output', type=str, default=None,
                        help='Output path for comparison results (optional)')

    args = parser.parse_args()

    # Build fits_dict from provided paths
    fits_dict = {}

    # Handle legacy arguments (map to new-style)
    if args.qlearning:
        args.m1 = args.qlearning
    if args.wmrl:
        args.m2 = args.wmrl

    # Load models
    if args.m1:
        fits_dict['M1'] = load_fits(args.m1)
    if args.m2:
        fits_dict['M2'] = load_fits(args.m2)
    if args.m3:
        fits_dict['M3'] = load_fits(args.m3)

    # Require at least 2 models
    if len(fits_dict) < 2:
        parser.error("At least 2 models required for comparison. "
                     "Provide --m1/--m2/--m3 or --qlearning/--wmrl")

    # Load fits
    print("Loading fits...")
    for model_name, fits_df in fits_dict.items():
        print(f"  {model_name}: {len(fits_df)} participants")

    # Check convergence
    print(f"\nConvergence:")
    for model_name, fits_df in fits_dict.items():
        n_converged = fits_df['converged'].sum()
        print(f"  {model_name}: {n_converged}/{len(fits_df)} ({100*n_converged/len(fits_df):.1f}%)")

    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}")

    # 1. Aggregate AIC comparison
    print(f"\n1. AGGREGATE INFORMATION CRITERIA")
    print("-" * 70)

    aic_comparison = compare_models(fits_dict, metric='aic')
    print(f"\nAIC Comparison:")
    print(aic_comparison.to_string(index=False))

    # AIC interpretation
    best_aic_model = aic_comparison.iloc[0]['model']
    print(f"\n  Best model by AIC: {best_aic_model}")
    for _, row in aic_comparison.iloc[1:].iterrows():
        delta = row['delta_aic']
        print(f"  {row['model']} vs {best_aic_model}: delta AIC = {delta:.2f} ({interpret_delta_aic(delta)})")

    # Aggregate BIC comparison
    bic_comparison = compare_models(fits_dict, metric='bic')
    print(f"\nBIC Comparison:")
    print(bic_comparison.to_string(index=False))

    best_bic_model = bic_comparison.iloc[0]['model']
    print(f"\n  Best model by BIC: {best_bic_model}")

    # Akaike weights
    aic_values = {row['model']: row['aggregate_aic'] for _, row in aic_comparison.iterrows()}
    weights = compute_akaike_weights_n(aic_values)
    print(f"\nAkaike Weights:")
    for model_name in sorted(weights.keys()):
        print(f"  {model_name}: {weights[model_name]:.4f} ({100*weights[model_name]:.2f}%)")

    # 2. Per-participant comparison
    print(f"\n2. PER-PARTICIPANT COMPARISON")
    print("-" * 70)

    for metric in ['aic', 'bic']:
        wins = count_participant_wins_n(fits_dict, metric)
        print(f"\n{metric.upper()} (n={wins['total'].iloc[0]} with all models converged):")
        for _, row in wins.iterrows():
            print(f"  {row['model']} wins: {row['wins']} ({row['win_pct']:.1f}%)")

    # 3. Model-specific parameter summaries
    print(f"\n3. PARAMETER ESTIMATES (CONVERGED FITS)")
    print("-" * 70)

    # Shared parameters across all models
    shared_params = ['alpha_pos', 'alpha_neg', 'epsilon']
    print(f"\nShared parameters:")
    for param in shared_params:
        param_line = f"  {param:<12} "
        for model_name, fits_df in fits_dict.items():
            converged = fits_df[fits_df['converged'] == True]
            if param in converged.columns:
                mean = converged[param].mean()
                se = converged[param].std() / np.sqrt(len(converged))
                param_line += f"{model_name}: {mean:.3f}±{se:.3f}  "
        print(param_line)

    # Model-specific parameters
    if 'M2' in fits_dict or 'M3' in fits_dict:
        print(f"\nWM-RL parameters:")
        wmrl_params = ['phi', 'rho', 'capacity']
        for param in wmrl_params:
            param_line = f"  {param:<12} "
            for model_name in ['M2', 'M3']:
                if model_name in fits_dict:
                    fits_df = fits_dict[model_name]
                    converged = fits_df[fits_df['converged'] == True]
                    if param in converged.columns:
                        mean = converged[param].mean()
                        se = converged[param].std() / np.sqrt(len(converged))
                        param_line += f"{model_name}: {mean:.3f}±{se:.3f}  "
            print(param_line)

    # M3-specific: kappa perseveration parameter
    if 'M3' in fits_dict:
        print(f"\nM3 perseveration parameter:")
        m3_fits = fits_dict['M3']
        m3_converged = m3_fits[m3_fits['converged'] == True]
        if 'kappa' in m3_converged.columns:
            kappa_mean = m3_converged['kappa'].mean()
            kappa_se = m3_converged['kappa'].std() / np.sqrt(len(m3_converged))
            kappa_median = m3_converged['kappa'].median()
            print(f"  kappa:       Mean = {kappa_mean:.3f}±{kappa_se:.3f}, Median = {kappa_median:.3f}")
            print(f"               Range = [{m3_converged['kappa'].min():.3f}, {m3_converged['kappa'].max():.3f}]")

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Combine AIC and BIC comparisons
        comparison_results = pd.merge(
            aic_comparison.rename(columns=lambda x: x.replace('aic', 'metric') if 'aic' in x else x),
            bic_comparison[['model', 'aggregate_bic', 'delta_bic']],
            on='model'
        )

        # Add Akaike weights
        comparison_results['akaike_weight'] = comparison_results['model'].map(weights)

        # Add per-participant wins
        aic_wins = count_participant_wins_n(fits_dict, 'aic')
        bic_wins = count_participant_wins_n(fits_dict, 'bic')

        comparison_results = pd.merge(
            comparison_results,
            aic_wins[['model', 'wins']].rename(columns={'wins': 'aic_wins'}),
            on='model'
        )
        comparison_results = pd.merge(
            comparison_results,
            bic_wins[['model', 'wins']].rename(columns={'wins': 'bic_wins'}),
            on='model'
        )

        comparison_results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Preferred model (AIC): {best_aic_model}")
    print(f"Preferred model (BIC): {best_bic_model}")
    if best_aic_model == best_bic_model:
        print(f"\nBoth criteria agree: {best_aic_model} is preferred")
    else:
        print(f"\nCriteria disagree - AIC favors {best_aic_model}, BIC favors {best_bic_model}")
        print(f"(BIC applies stronger penalty for model complexity)")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
