"""
Generate predictions from fitted models and visualize performance.

This script:
1. Loads the best-fitting model (lowest BIC/AIC)
2. Generates trial-by-trial predictions on behavioral data
3. Creates performance visualizations:
   - Learning curves (accuracy over trials)
   - Performance by trial position (early/late, pre/post reversal)
"""

import numpy as np
import pandas as pd
import arviz as az
from pathlib import Path
import argparse
import sys
from typing import Tuple, Dict, Optional

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from models.q_learning import QLearningAgent
from models.wm_rl_hybrid import WMRLHybridAgent
from config import (
    TaskParams,
    OUTPUT_DIR, FIGURES_DIR
)
from scripts.analysis.visualize_model_performance import (
    plot_learning_curves,
    plot_performance_by_trial_position,
    plot_combined_performance_analysis
)


def load_fitted_model(model_name: str, posterior_dir: Path) -> Tuple[az.InferenceData, Dict]:
    """
    Load fitted model posterior.

    Parameters
    ----------
    model_name : str
        'qlearning' or 'wmrl'
    posterior_dir : Path
        Directory containing fitted posteriors

    Returns
    -------
    idata : az.InferenceData
        Posterior samples
    best_params : dict
        MAP (maximum a posteriori) parameter estimates
    """
    # Load posterior
    posterior_file = posterior_dir / f'{model_name}_posterior.nc'

    if not posterior_file.exists():
        raise FileNotFoundError(
            f"Fitted model not found: {posterior_file}\n"
            f"Please run fitting first: python scripts/fitting/fit_to_data.py --model {model_name}"
        )

    print(f"Loading fitted model: {posterior_file}")
    idata = az.from_netcdf(posterior_file)

    # Extract MAP estimates (median of posterior)
    posterior = idata.posterior
    best_params = {}

    for var in posterior.data_vars:
        if var.startswith('sigma') or var.startswith('mu'):
            continue  # Skip hyperparameters

        # Get median across chains and draws
        values = posterior[var].values
        if values.ndim == 2:  # (chain, draw)
            best_params[var] = float(np.median(values))
        elif values.ndim == 3:  # (chain, draw, subject) - use group mean
            best_params[var] = float(np.median(values[:, :, 0]))  # First subject as example

    print(f"  Loaded {len(best_params)} parameters")
    print(f"  Parameters: {list(best_params.keys())}")

    return idata, best_params


def generate_predictions(
    behavioral_data: pd.DataFrame,
    model_name: str,
    params: Dict,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate trial-by-trial predictions from fitted model.

    Parameters
    ----------
    behavioral_data : pd.DataFrame
        Behavioral data with columns: subject_id, block, trial, set_size,
        stimulus, response, correct
    model_name : str
        'qlearning' or 'wmrl'
    params : dict
        Model parameters
    seed : int
        Random seed

    Returns
    -------
    predictions_df : pd.DataFrame
        Predictions with columns:
        - All original behavioral data columns
        - model_choice: model's predicted action
        - model_correct: whether model was correct (0/1)
        - trial_num: trial number within block
        - trials_since_reversal: trials since last reversal
        - is_post_reversal: whether trial is after reversal
    """
    np.random.seed(seed)

    predictions = []

    # Group by subject and block
    for (subject_id, block_id), block_data in behavioral_data.groupby(['subject_id', 'block']):
        block_data = block_data.sort_values('trial').reset_index(drop=True)

        # Initialize agent
        if model_name == 'qlearning':
            agent = QLearningAgent(
                alpha_pos=params['alpha_pos'],
                alpha_neg=params['alpha_neg'],
                beta=params['beta']
            )
        elif model_name == 'wmrl':
            agent = WMRLHybridAgent(
                alpha_pos=params['alpha_pos'],
                alpha_neg=params['alpha_neg'],
                beta=params['beta'],
                beta_wm=params.get('beta_wm', params['beta']),
                capacity=int(params['capacity']),
                phi=params['phi'],
                rho=params['rho']
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Track reversals (using actual correct responses from data)
        correct_streak = 0
        reversal_trial = None
        trials_since_reversal = 0

        # Simulate block trial by trial
        for idx, row in block_data.iterrows():
            stimulus = int(row['stimulus'])
            set_size = int(row['set_size'])
            actual_response = int(row['response'])
            actual_correct = int(row['correct'])

            # Get model's choice
            if model_name == 'wmrl':
                model_choice, _ = agent.choose_action(stimulus, set_size)
            else:
                model_choice = agent.choose_action(stimulus, set_size)

            # Determine if model was correct (using actual task feedback)
            model_correct = int(model_choice == actual_response)

            # Update agent with actual feedback from task
            if model_name == 'wmrl':
                agent.update(stimulus, model_choice, actual_correct, set_size)
            else:
                agent.update(stimulus, model_choice, actual_correct)

            # Track reversals based on consecutive correct responses
            if actual_correct:
                correct_streak += 1
                if TaskParams.REVERSAL_MIN <= correct_streak <= TaskParams.REVERSAL_MAX:
                    if reversal_trial is None:  # First reversal
                        reversal_trial = idx
                        trials_since_reversal = 0
            else:
                correct_streak = 0

            # Record prediction
            predictions.append({
                'subject_id': subject_id,
                'block': block_id,
                'trial': row['trial'],
                'trial_num': idx + 1,  # 1-indexed
                'set_size': set_size,
                'stimulus': stimulus,
                'actual_response': actual_response,
                'actual_correct': actual_correct,
                'model_choice': model_choice,
                'model_correct': model_correct,
                'trials_since_reversal': trials_since_reversal,
                'is_post_reversal': reversal_trial is not None
            })

            # Increment trials since reversal
            if reversal_trial is not None:
                trials_since_reversal += 1

    predictions_df = pd.DataFrame(predictions)

    print(f"\nGenerated {len(predictions_df)} trial predictions")
    print(f"  Subjects: {predictions_df['subject_id'].nunique()}")
    print(f"  Blocks: {predictions_df['block'].nunique()}")
    print(f"  Overall accuracy: {predictions_df['model_correct'].mean():.3f}")

    return predictions_df


def identify_winning_model(posterior_dir: Path, criterion: str = 'BIC') -> str:
    """
    Identify the best-fitting model based on information criterion.

    Parameters
    ----------
    posterior_dir : Path
        Directory containing model comparison results
    criterion : str
        'BIC', 'AIC', 'WAIC', or 'LOO'

    Returns
    -------
    model_name : str
        Name of winning model
    """
    comparison_file = posterior_dir / 'model_comparison_summary.csv'

    if not comparison_file.exists():
        print(f"Model comparison not found: {comparison_file}")
        print("Defaulting to Q-learning model")
        return 'qlearning'

    # Load comparison
    comparison = pd.read_csv(comparison_file)

    # Sort by criterion (lower is better)
    comparison = comparison.sort_values(criterion)
    winning_model = comparison.iloc[0]['model']

    print(f"\nWinning model ({criterion}): {winning_model}")
    print(f"  {criterion} = {comparison.iloc[0][criterion]:.2f}")

    return winning_model


def main():
    parser = argparse.ArgumentParser(
        description='Generate predictions and visualize winning model performance'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/task_trials_long.csv',
        help='Path to behavioral data'
    )
    parser.add_argument(
        '--posterior-dir',
        type=str,
        default='output/fitting',
        help='Directory containing fitted models'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='auto',
        help='Model to plot: "auto" (use best-fitting), "qlearning", or "wmrl"'
    )
    parser.add_argument(
        '--criterion',
        type=str,
        default='BIC',
        choices=['BIC', 'AIC', 'WAIC', 'LOO'],
        help='Information criterion for selecting best model'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=4,
        help='Trial threshold for early/late classification'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Paths
    data_path = project_root / args.data
    posterior_dir = project_root / args.posterior_dir
    output_dir = OUTPUT_DIR / 'model_performance'
    figure_dir = FIGURES_DIR / 'model_performance'

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("WINNING MODEL PERFORMANCE VISUALIZATION")
    print("=" * 80)

    # Load behavioral data
    print(f"\nLoading behavioral data: {data_path}")
    behavioral_data = pd.read_csv(data_path)
    print(f"  Loaded {len(behavioral_data)} trials")
    print(f"  Subjects: {behavioral_data['subject_id'].nunique()}")
    print(f"  Blocks: {behavioral_data['block'].nunique()}")

    # Identify winning model
    if args.model == 'auto':
        model_name = identify_winning_model(posterior_dir, args.criterion)
    else:
        model_name = args.model
        print(f"\nUsing specified model: {model_name}")

    # Load fitted parameters
    idata, params = load_fitted_model(model_name, posterior_dir)
    print(f"\nModel parameters:")
    for param, value in params.items():
        print(f"  {param} = {value:.4f}")

    # Generate predictions
    print("\nGenerating model predictions...")
    predictions_df = generate_predictions(
        behavioral_data,
        model_name,
        params,
        seed=args.seed
    )

    # Save predictions
    predictions_file = output_dir / f'{model_name}_predictions.csv'
    predictions_df.to_csv(predictions_file, index=False)
    print(f"\n  ✓ Saved predictions: {predictions_file}")

    # Create visualizations
    print("\nCreating visualizations...")

    model_display_name = {
        'qlearning': 'Q-Learning',
        'wmrl': 'WM-RL Hybrid'
    }.get(model_name, model_name)

    # 1. Combined analysis
    print("  1. Combined performance analysis...")
    plot_combined_performance_analysis(
        predictions_df,
        n_trials_threshold=args.threshold,
        save_dir=figure_dir,
        model_name=model_display_name
    )

    # 2. Learning curve (since start)
    print("  2. Learning curve (trials since start)...")
    plot_learning_curves(
        predictions_df,
        trial_type='since_start',
        save_path=figure_dir / f'{model_name}_learning_curve_since_start.png',
        title=f'{model_display_name}: Learning Curve'
    )

    # 3. Learning curve (since reversal)
    print("  3. Learning curve (trials since reversal)...")
    plot_learning_curves(
        predictions_df,
        trial_type='since_reversal',
        save_path=figure_dir / f'{model_name}_learning_curve_since_reversal.png',
        title=f'{model_display_name}: Post-Reversal Learning'
    )

    # 4. Performance by trial position
    print("  4. Performance by trial position...")
    plot_performance_by_trial_position(
        predictions_df,
        n_trials_threshold=args.threshold,
        save_path=figure_dir / f'{model_name}_performance_by_position.png',
        title=f'{model_display_name}: Performance by Trial Position'
    )

    # Print summary statistics
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    print("\nOverall Performance:")
    print(f"  Accuracy: {predictions_df['model_correct'].mean():.3f}")

    print("\nPerformance by Set Size:")
    for ss in sorted(predictions_df['set_size'].unique()):
        ss_acc = predictions_df[predictions_df['set_size'] == ss]['model_correct'].mean()
        print(f"  Set Size {ss}: {ss_acc:.3f}")

    print("\nPerformance by Trial Position:")

    def categorize_trial(row):
        if not row['is_post_reversal']:
            if row['trial_num'] < args.threshold:
                return 'Early Block'
            else:
                return 'Late Block'
        else:
            if row['trials_since_reversal'] < args.threshold:
                return 'Early Post-Reversal'
            else:
                return 'Late Post-Reversal'

    predictions_df['position'] = predictions_df.apply(categorize_trial, axis=1)

    for pos in ['Early Block', 'Late Block', 'Early Post-Reversal', 'Late Post-Reversal']:
        pos_data = predictions_df[predictions_df['position'] == pos]
        if len(pos_data) > 0:
            print(f"  {pos}: {pos_data['model_correct'].mean():.3f} (n={len(pos_data)})")

    print("\n" + "=" * 80)
    print(f"Figures saved to: {figure_dir}")
    print(f"Predictions saved to: {predictions_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
