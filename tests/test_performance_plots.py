"""
Test performance visualization functions with simulated data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pytest

from config import FIGURES_DIR

# Module was never created — skip entire file
pytest.skip(
    "scripts.analysis.visualize_model_performance does not exist",
    allow_module_level=True,
)


def generate_mock_predictions(
    n_blocks: int = 10,
    trials_per_block: int = 50,
    set_sizes: list = [2, 3, 5, 6],
    reversal_trial: int = 25,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate mock prediction data for testing visualizations.

    Simulates:
    - Learning over trials (accuracy improves)
    - Reversal effect (accuracy drops then recovers)
    - Set size effect (harder with larger sets)
    """
    np.random.seed(seed)

    predictions = []

    for block_id in range(n_blocks):
        # Random set size for this block
        set_size = np.random.choice(set_sizes)

        # Base accuracy depends on set size (harder with more items)
        base_acc = 0.8 - (set_size - 2) * 0.05

        for trial_num in range(1, trials_per_block + 1):
            # Learning effect: accuracy improves over trials
            learning_boost = min(0.15, 0.15 * (trial_num / 20))

            # Reversal effect
            is_post_reversal = trial_num > reversal_trial
            trials_since_reversal = trial_num - reversal_trial if is_post_reversal else 0

            if is_post_reversal:
                # Accuracy drops after reversal then recovers
                reversal_penalty = max(0, 0.25 - 0.02 * trials_since_reversal)
            else:
                reversal_penalty = 0

            # Compute accuracy for this trial
            trial_acc = base_acc + learning_boost - reversal_penalty

            # Add noise
            trial_acc += np.random.normal(0, 0.1)
            trial_acc = np.clip(trial_acc, 0.2, 0.95)

            # Bernoulli trial
            correct = int(np.random.random() < trial_acc)

            predictions.append({
                'subject_id': 1,
                'block': block_id,
                'trial': trial_num,
                'trial_num': trial_num,
                'set_size': set_size,
                'stimulus': np.random.randint(0, set_size),
                'model_choice': np.random.randint(0, 2),
                'correct': correct,
                'trials_since_reversal': trials_since_reversal,
                'is_post_reversal': is_post_reversal
            })

    return pd.DataFrame(predictions)


def main():
    print("=" * 80)
    print("TESTING PERFORMANCE VISUALIZATION FUNCTIONS")
    print("=" * 80)

    # Generate mock data
    print("\nGenerating mock predictions...")
    predictions_df = generate_mock_predictions(
        n_blocks=20,
        trials_per_block=50,
        set_sizes=[2, 3, 5, 6],
        reversal_trial=25,
        seed=42
    )

    print(f"  Generated {len(predictions_df)} trials")
    print(f"  Set sizes: {sorted(predictions_df['set_size'].unique())}")
    print(f"  Overall accuracy: {predictions_df['correct'].mean():.3f}")

    # Create test output directory
    test_dir = FIGURES_DIR / 'test_performance'
    test_dir.mkdir(parents=True, exist_ok=True)

    print("\nCreating visualizations...")

    # 1. Learning curve (since start)
    print("  1. Learning curve (since block start)...")
    plot_learning_curves(
        predictions_df,
        trial_type='since_start',
        save_path=test_dir / 'test_learning_curve_since_start.png',
        title='Test: Learning Curve (Since Block Start)'
    )

    # 2. Learning curve (since reversal)
    print("  2. Learning curve (since reversal)...")
    plot_learning_curves(
        predictions_df,
        trial_type='since_reversal',
        save_path=test_dir / 'test_learning_curve_since_reversal.png',
        title='Test: Learning Curve (Since Reversal)'
    )

    # 3. Performance by trial position
    print("  3. Performance by trial position...")
    plot_performance_by_trial_position(
        predictions_df,
        n_trials_threshold=4,
        save_path=test_dir / 'test_performance_by_position.png',
        title='Test: Performance by Trial Position'
    )

    # 4. Combined analysis
    print("  4. Combined performance analysis...")
    plot_combined_performance_analysis(
        predictions_df,
        n_trials_threshold=4,
        save_dir=test_dir,
        model_name='Test Model'
    )

    print("\n" + "=" * 80)
    print(f"Test figures saved to: {test_dir}")
    print("=" * 80)
    print("\nIf plots look good, you can use scripts/06_fit_analyses/01_compare_models.py with real data:")
    print("  python scripts/06_fit_analyses/01_compare_models.py --model qlearning")


if __name__ == '__main__':
    main()
