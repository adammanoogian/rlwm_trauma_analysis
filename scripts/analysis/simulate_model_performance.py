"""
Simulate model predictions on behavioral data using reasonable parameter values.

This script generates predictions without requiring fitted posteriors,
useful for initial model comparison and visualization.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys

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
    plot_combined_performance_analysis
)


def generate_model_predictions(
    behavioral_data: pd.DataFrame,
    model_name: str,
    params: dict,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate trial-by-trial predictions from model.

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
        Predictions with performance metrics
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

        # Track reversals
        correct_streak = 0
        reversal_trial = None
        trials_since_reversal = 0

        # Simulate block trial by trial
        for idx, row in block_data.iterrows():
            # Ensure values are proper Python ints, not numpy types
            # Note: stimulus in data is 1-indexed, but models expect 0-indexed
            stimulus = int(float(row['stimulus'])) - 1
            set_size = int(float(row['set_size']))
            actual_correct = int(float(row['correct']))

            # Get model's choice
            if model_name == 'wmrl':
                model_choice, _ = agent.choose_action(stimulus, set_size)
                model_choice = int(model_choice)  # Ensure proper int
            else:
                # Q-learning doesn't use set_size
                model_choice = int(agent.choose_action(stimulus))

            # Determine if model was correct (compare to ground truth reward)
            # We use actual_correct to determine the rewarded action
            if actual_correct == 1:
                rewarded_action = int(float(row['response']))
            else:
                # If participant was wrong, the other action was correct
                rewarded_action = 1 - int(float(row['response']))

            model_correct = int(model_choice == rewarded_action)

            # Update agent with actual feedback
            if model_name == 'wmrl':
                agent.update(stimulus, model_choice, model_correct, set_size)
            else:
                agent.update(stimulus, model_choice, model_correct)

            # Track reversals based on consecutive correct responses
            if actual_correct:
                correct_streak += 1
                if TaskParams.REVERSAL_MIN <= correct_streak <= TaskParams.REVERSAL_MAX:
                    if reversal_trial is None:
                        reversal_trial = idx
                        trials_since_reversal = 0
            else:
                correct_streak = 0

            # Record prediction
            predictions.append({
                'subject_id': subject_id,
                'block': block_id,
                'trial': row['trial'],
                'trial_num': idx + 1,
                'set_size': set_size,
                'stimulus': stimulus,
                'model_choice': model_choice,
                'correct': model_correct,
                'trials_since_reversal': trials_since_reversal,
                'is_post_reversal': reversal_trial is not None
            })

            # Increment trials since reversal
            if reversal_trial is not None:
                trials_since_reversal += 1

    return pd.DataFrame(predictions)


def main():
    parser = argparse.ArgumentParser(
        description='Simulate model predictions on behavioral data'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='output/task_trials_long.csv',
        help='Path to behavioral data'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['qlearning', 'wmrl'],
        help='Models to simulate: qlearning, wmrl, or both'
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
    output_dir = OUTPUT_DIR / 'model_performance'
    figure_dir = FIGURES_DIR / 'model_performance'

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MODEL PERFORMANCE SIMULATION")
    print("=" * 80)

    # Load behavioral data
    print(f"\nLoading behavioral data: {data_path}")
    behavioral_data = pd.read_csv(data_path)

    # Standardize column names
    if 'sona_id' in behavioral_data.columns:
        behavioral_data['subject_id'] = behavioral_data['sona_id']
    if 'key_answer' in behavioral_data.columns:
        behavioral_data['response'] = behavioral_data['key_answer']

    print(f"  Loaded {len(behavioral_data)} trials")
    print(f"  Subjects: {behavioral_data['subject_id'].nunique()}")
    print(f"  Blocks: {behavioral_data['block'].nunique()}")
    print(f"  Set sizes: {sorted(behavioral_data['set_size'].unique())}")

    # Define reasonable parameter values for each model
    # These are typical values from literature and prior exploration
    model_params = {
        'qlearning': {
            'alpha_pos': 0.6,   # Moderate positive learning rate
            'alpha_neg': 0.3,   # Lower negative learning rate (common asymmetry)
            'beta': 2.5         # Moderate exploration/exploitation
        },
        'wmrl': {
            'alpha_pos': 0.5,
            'alpha_neg': 0.3,
            'beta': 2.0,
            'beta_wm': 3.0,     # Higher inverse temp for WM (more confident)
            'capacity': 4,      # Moderate WM capacity
            'phi': 0.2,         # Some decay
            'rho': 0.7          # Moderate WM reliance
        }
    }

    # Process each model
    for model_name in args.models:
        if model_name not in model_params:
            print(f"\nSkipping unknown model: {model_name}")
            continue

        print("\n" + "=" * 80)
        print(f"SIMULATING {model_name.upper()} MODEL")
        print("=" * 80)

        params = model_params[model_name]
        print(f"\nUsing parameter values:")
        for param, value in params.items():
            print(f"  {param} = {value}")

        # Generate predictions
        print(f"\nGenerating predictions...")
        predictions_df = generate_model_predictions(
            behavioral_data,
            model_name,
            params,
            seed=args.seed
        )

        # Calculate performance
        overall_acc = predictions_df['correct'].mean()
        print(f"  Overall accuracy: {overall_acc:.3f}")

        print(f"\n  Performance by set size:")
        for ss in sorted(predictions_df['set_size'].unique()):
            ss_acc = predictions_df[predictions_df['set_size'] == ss]['correct'].mean()
            print(f"    Set Size {ss}: {ss_acc:.3f}")

        # Save predictions
        predictions_file = output_dir / f'{model_name}_predictions_simulated.csv'
        predictions_df.to_csv(predictions_file, index=False)
        print(f"\n  [OK] Saved predictions: {predictions_file}")

        # Create visualizations
        print(f"\nCreating visualizations...")

        model_display_name = {
            'qlearning': 'Q-Learning',
            'wmrl': 'WM-RL Hybrid'
        }.get(model_name, model_name)

        plot_combined_performance_analysis(
            predictions_df,
            n_trials_threshold=args.threshold,
            save_dir=figure_dir,
            model_name=model_display_name
        )

        # Performance summary by trial position
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

        print(f"\n  Performance by trial position:")
        for pos in ['Early Block', 'Late Block', 'Early Post-Reversal', 'Late Post-Reversal']:
            pos_data = predictions_df[predictions_df['position'] == pos]
            if len(pos_data) > 0:
                acc = pos_data['correct'].mean()
                print(f"    {pos:20s}: {acc:.3f} (n={len(pos_data)})")

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    print(f"\nFigures saved to: {figure_dir}")
    print(f"Predictions saved to: {output_dir}")
    print("\nGenerated plots:")
    print("  - Learning curves (trial-by-trial accuracy)")
    print("  - Performance by trial position (early/late × pre/post reversal)")
    print("=" * 80)


if __name__ == '__main__':
    main()
