"""
Run Model Comparison

Load fitted model posteriors and compute all information criteria
(BIC, AIC, WAIC, LOO) to compare Q-learning vs WM-RL models.

Usage:
    python scripts/analysis/run_model_comparison.py
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import arviz as az
    ARVIZ_AVAILABLE = True
except ImportError:
    ARVIZ_AVAILABLE = False
    print("ERROR: ArviZ not installed. Install with: pip install arviz")
    sys.exit(1)

from scripts.analysis.model_comparison import (
    compare_models,
    plot_model_comparison,
    plot_model_weights
)


def load_fitted_models(fitting_dir: Path) -> dict:
    """
    Load fitted model posteriors from NetCDF files.

    Parameters
    ----------
    fitting_dir : Path
        Directory containing fitted model files

    Returns
    -------
    dict
        Dictionary mapping model names to InferenceData objects
    """
    traces = {}

    # Look for Q-learning trace
    q_files = list(fitting_dir.glob('*qlearning*.nc'))
    if q_files:
        q_file = sorted(q_files)[-1]  # Most recent
        print(f"Loading Q-learning model: {q_file.name}")
        traces['Q-Learning'] = az.from_netcdf(q_file)
    else:
        print("Warning: No Q-learning model found")

    # Look for WM-RL trace
    wmrl_files = list(fitting_dir.glob('*wmrl*.nc'))
    if wmrl_files:
        wmrl_file = sorted(wmrl_files)[-1]  # Most recent
        print(f"Loading WM-RL model: {wmrl_file.name}")
        traces['WM-RL'] = az.from_netcdf(wmrl_file)
    else:
        print("Warning: No WM-RL model found")

    if not traces:
        raise FileNotFoundError(
            f"No fitted models found in {fitting_dir}. "
            "Run model fitting first (scripts/fitting/fit_to_data.py)"
        )

    return traces


def load_behavioral_data(data_path: Path) -> pd.DataFrame:
    """
    Load behavioral data for model comparison.

    Parameters
    ----------
    data_path : Path
        Path to task_trials_long.csv

    Returns
    -------
    pd.DataFrame
        Behavioral data
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Run data processing scripts first (01-04)."
        )

    print(f"Loading behavioral data: {data_path.name}")
    data = pd.read_csv(data_path)

    # Filter to main task only
    data = data[data['phase_type'] == 'main_task'].copy()

    print(f"  {len(data)} trials from {data['sona_id'].nunique()} participants")

    return data


def main():
    """
    Main workflow for model comparison.
    """
    print("=" * 80)
    print("MODEL COMPARISON: BIC, AIC, WAIC, LOO")
    print("=" * 80)
    print()

    # Paths
    fitting_dir = project_root / 'output' / 'fitting'
    data_path = project_root / 'output' / 'task_trials_long.csv'
    output_dir = project_root / 'output' / 'model_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load fitted models
    print("-" * 80)
    print("LOADING FITTED MODELS")
    print("-" * 80)
    try:
        traces = load_fitted_models(fitting_dir)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nTo fit models first, run:")
        print("  python scripts/fitting/fit_to_data.py --model both")
        return

    print()

    # Load behavioral data
    print("-" * 80)
    print("LOADING BEHAVIORAL DATA")
    print("-" * 80)
    try:
        data = load_behavioral_data(data_path)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return

    print()

    # Compute all criteria
    print("-" * 80)
    print("COMPUTING INFORMATION CRITERIA")
    print("-" * 80)
    print()

    comparison_df = compare_models(traces, data)

    print()
    print("-" * 80)
    print("COMPARISON RESULTS")
    print("-" * 80)
    print()
    print(comparison_df[['model', 'n_params', 'AIC', 'BIC', 'WAIC', 'LOO']].to_string(index=False))
    print()

    print("Rankings (1 = best):")
    print(comparison_df[['model', 'AIC_rank', 'BIC_rank', 'WAIC_rank', 'LOO_rank']].to_string(index=False))
    print()

    print("Model Weights:")
    print(comparison_df[['model', 'AIC_weight', 'BIC_weight']].to_string(index=False))
    print()

    # Interpretation
    print("-" * 80)
    print("INTERPRETATION")
    print("-" * 80)
    print()

    # Find best model by each criterion
    for criterion in ['AIC', 'BIC', 'WAIC', 'LOO']:
        best_idx = comparison_df[criterion].idxmin()
        best_model = comparison_df.loc[best_idx, 'model']
        best_val = comparison_df.loc[best_idx, criterion]

        other_idx = 1 - best_idx
        other_model = comparison_df.loc[other_idx, 'model']
        other_val = comparison_df.loc[other_idx, criterion]
        delta = other_val - best_val

        print(f"{criterion}:")
        print(f"  Best model: {best_model} ({criterion}={best_val:.1f})")
        print(f"  Δ{criterion} = {delta:.1f}", end="")

        # Evidence strength (Kass & Raftery, 1995 for BIC)
        if criterion in ['BIC', 'AIC']:
            if delta < 2:
                evidence = "weak evidence"
            elif delta < 6:
                evidence = "positive evidence"
            elif delta < 10:
                evidence = "strong evidence"
            else:
                evidence = "very strong evidence"
            print(f" ({evidence} for {best_model})")
        else:
            print()

    print()

    # Save results
    print("-" * 80)
    print("SAVING RESULTS")
    print("-" * 80)
    print()

    # Save comparison table
    output_file = output_dir / 'comparison_all_criteria.csv'
    comparison_df.to_csv(output_file, index=False)
    print(f"Saved comparison table: {output_file}")

    # Create visualizations
    print("Creating visualizations...")

    plot_model_comparison(comparison_df)
    print(f"  ✓ Saved: figures/model_comparison/information_criteria_comparison.png")

    plot_model_weights(comparison_df)
    print(f"  ✓ Saved: figures/model_comparison/model_weights.png")

    print()
    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print()
    print(f"Results saved to: {output_dir}")
    print(f"Figures saved to: figures/model_comparison/")
    print()


if __name__ == "__main__":
    main()
