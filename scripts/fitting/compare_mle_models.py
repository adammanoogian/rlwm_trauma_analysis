"""
Model Comparison for MLE Fits

Compare Q-Learning and WM-RL models using AIC and BIC.
Following Senta et al. (2025) methodology.

Usage:
    python scripts/fitting/compare_mle_models.py \
        --qlearning output/mle/qlearning_individual_fits.csv \
        --wmrl output/mle/wmrl_individual_fits.csv
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
        description='Compare MLE fits between Q-Learning and WM-RL models'
    )
    parser.add_argument('--qlearning', type=str, required=True,
                        help='Path to Q-Learning individual fits CSV')
    parser.add_argument('--wmrl', type=str, required=True,
                        help='Path to WM-RL individual fits CSV')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for comparison results (optional)')

    args = parser.parse_args()

    # Load fits
    print("Loading fits...")
    ql_fits = load_fits(args.qlearning)
    wmrl_fits = load_fits(args.wmrl)

    print(f"  Q-Learning: {len(ql_fits)} participants")
    print(f"  WM-RL: {len(wmrl_fits)} participants")

    # Check convergence
    ql_converged = ql_fits['converged'].sum()
    wmrl_converged = wmrl_fits['converged'].sum()
    print(f"\nConvergence:")
    print(f"  Q-Learning: {ql_converged}/{len(ql_fits)} ({100*ql_converged/len(ql_fits):.1f}%)")
    print(f"  WM-RL: {wmrl_converged}/{len(wmrl_fits)} ({100*wmrl_converged/len(wmrl_fits):.1f}%)")

    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")

    # Aggregate AIC
    agg_aic_ql = compute_aggregate_ic(ql_fits, 'aic')
    agg_aic_wmrl = compute_aggregate_ic(wmrl_fits, 'aic')
    delta_aic = agg_aic_wmrl - agg_aic_ql

    # Aggregate BIC
    agg_bic_ql = compute_aggregate_ic(ql_fits, 'bic')
    agg_bic_wmrl = compute_aggregate_ic(wmrl_fits, 'bic')
    delta_bic = agg_bic_wmrl - agg_bic_ql

    print(f"\n1. AGGREGATE INFORMATION CRITERIA")
    print("-" * 50)
    print(f"{'Metric':<15} {'Q-Learning':>15} {'WM-RL':>15} {'Delta':>12}")
    print("-" * 50)
    print(f"{'Sum AIC':<15} {agg_aic_ql:>15.2f} {agg_aic_wmrl:>15.2f} {delta_aic:>+12.2f}")
    print(f"{'Sum BIC':<15} {agg_bic_ql:>15.2f} {agg_bic_wmrl:>15.2f} {delta_bic:>+12.2f}")

    # Interpretation
    print(f"\nInterpretation:")
    if delta_aic < 0:
        print(f"  AIC favors WM-RL (delta = {delta_aic:.2f})")
        print(f"  Evidence strength: {interpret_delta_aic(delta_aic)}")
    else:
        print(f"  AIC favors Q-Learning (delta = {delta_aic:.2f})")
        print(f"  Evidence strength: {interpret_delta_aic(delta_aic)}")

    if delta_bic < 0:
        print(f"  BIC favors WM-RL (delta = {delta_bic:.2f})")
    else:
        print(f"  BIC favors Q-Learning (delta = {delta_bic:.2f})")

    # Akaike weights
    w_ql, w_wmrl = compute_akaike_weights(agg_aic_ql, agg_aic_wmrl)
    print(f"\nAkaike Weights:")
    print(f"  Q-Learning: {w_ql:.4f} ({100*w_ql:.2f}%)")
    print(f"  WM-RL:      {w_wmrl:.4f} ({100*w_wmrl:.2f}%)")

    # Per-participant comparison
    print(f"\n2. PER-PARTICIPANT COMPARISON")
    print("-" * 50)

    for metric in ['aic', 'bic']:
        counts = count_winning_model(ql_fits, wmrl_fits, metric)
        print(f"\n{metric.upper()} (n={counts['total']} with both converged):")
        print(f"  Q-Learning wins: {counts['qlearning_wins']} ({100*counts['qlearning_wins']/counts['total']:.1f}%)")
        print(f"  WM-RL wins:      {counts['wmrl_wins']} ({100*counts['wmrl_wins']/counts['total']:.1f}%)")
        if counts['ties'] > 0:
            print(f"  Ties:            {counts['ties']}")

    # Delta IC distribution
    delta_df = compute_delta_ic(ql_fits, wmrl_fits, 'aic')
    print(f"\nDelta AIC distribution (WMRL - QL):")
    print(f"  Mean: {delta_df['delta_aic'].mean():.2f}")
    print(f"  Median: {delta_df['delta_aic'].median():.2f}")
    print(f"  SD: {delta_df['delta_aic'].std():.2f}")
    print(f"  Range: [{delta_df['delta_aic'].min():.2f}, {delta_df['delta_aic'].max():.2f}]")

    # Parameter comparison
    print(f"\n3. PARAMETER ESTIMATES (CONVERGED FITS)")
    print("-" * 50)

    print(f"\n{'Parameter':<12} {'Q-Learning':>20} {'WM-RL':>20}")
    print("-" * 55)

    # Shared parameters
    for param in ['alpha_pos', 'alpha_neg', 'epsilon']:
        ql_vals = ql_fits[ql_fits['converged'] == True][param]
        wmrl_vals = wmrl_fits[wmrl_fits['converged'] == True][param]
        print(f"{param:<12} {ql_vals.mean():>8.3f} +/- {ql_vals.std()/np.sqrt(len(ql_vals)):>5.3f}"
              f"{wmrl_vals.mean():>10.3f} +/- {wmrl_vals.std()/np.sqrt(len(wmrl_vals)):>5.3f}")

    # WM-RL specific
    print(f"\nWM-RL specific parameters:")
    for param in ['phi', 'rho', 'capacity']:
        wmrl_vals = wmrl_fits[wmrl_fits['converged'] == True][param]
        print(f"  {param:<12} {wmrl_vals.mean():.3f} +/- {wmrl_vals.std()/np.sqrt(len(wmrl_vals)):.3f}")

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            'metric': ['sum_aic', 'sum_bic', 'delta_aic', 'delta_bic',
                       'weight_ql', 'weight_wmrl',
                       'n_ql_wins_aic', 'n_wmrl_wins_aic',
                       'n_ql_wins_bic', 'n_wmrl_wins_bic'],
            'qlearning': [agg_aic_ql, agg_bic_ql, None, None,
                          w_ql, None,
                          count_winning_model(ql_fits, wmrl_fits, 'aic')['qlearning_wins'], None,
                          count_winning_model(ql_fits, wmrl_fits, 'bic')['qlearning_wins'], None],
            'wmrl': [agg_aic_wmrl, agg_bic_wmrl, delta_aic, delta_bic,
                     None, w_wmrl,
                     None, count_winning_model(ql_fits, wmrl_fits, 'aic')['wmrl_wins'],
                     None, count_winning_model(ql_fits, wmrl_fits, 'bic')['wmrl_wins']]
        }
        pd.DataFrame(results).to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    winner_aic = 'WM-RL' if delta_aic < 0 else 'Q-Learning'
    winner_bic = 'WM-RL' if delta_bic < 0 else 'Q-Learning'
    print(f"Preferred model (AIC): {winner_aic}")
    print(f"Preferred model (BIC): {winner_bic}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
