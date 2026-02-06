#!/usr/bin/env python
"""
09: Run Posterior Predictive Checks
====================================

Validates fitted models by simulating behavior with fitted parameters
and comparing to real behavioral patterns. Also runs model recovery
to verify the generative model wins.

This is a pipeline script that orchestrates the PPC workflow.

Purpose:
    - Validate that fitted model reproduces real behavioral patterns
    - Check if simulated accuracy, learning curves, set-size effects match real data
    - Run model recovery: fit all models to synthetic data, check if generative wins
    - Per Wilson & Collins (2019), Palminteri et al. (2017), Senta et al. (2025)

Usage:
    # Full PPC for a single model
    python scripts/09_run_ppc.py --model wmrl_m3

    # All models
    python scripts/09_run_ppc.py --model all

    # Skip model recovery (just behavioral comparison)
    python scripts/09_run_ppc.py --model wmrl_m3 --skip-model-recovery

    # With GPU acceleration
    python scripts/09_run_ppc.py --model wmrl_m3 --use-gpu

Outputs:
    - output/ppc/{model}/synthetic_trials.csv
    - output/ppc/{model}/behavioral_comparison.csv
    - output/ppc/{model}/mle_results/ (model recovery fits)
    - figures/ppc/{model}/*.png (comparison plots)
    - Console: Behavioral comparison + model recovery PASS/FAIL

Interpretation:
    - Behavioral match: synthetic metrics within ~5% of real data
    - Model recovery PASS: generative model wins on synthetic data
    - Model recovery FAIL: may indicate model misspecification
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from scripts.fitting.model_recovery import (
    run_posterior_predictive_check,
    run_model_recovery_check,
)


def run_ppc_for_model(
    model: str,
    fitted_params_path: str,
    real_data_path: str,
    output_dir: Path,
    figures_dir: Path,
    skip_model_recovery: bool = False,
    use_gpu: bool = False,
    n_jobs: int = 1,
    verbose: bool = True
) -> dict:
    """
    Run full PPC pipeline for a single model.

    Returns dict with behavioral comparison and model recovery results.
    """
    print(f"\n{'='*60}")
    print(f"POSTERIOR PREDICTIVE CHECK: {model.upper()}")
    print(f"{'='*60}")

    # 1. Generate synthetic data and compare behavior
    comparison_df = run_posterior_predictive_check(
        model=model,
        fitted_params_path=fitted_params_path,
        real_data_path=real_data_path,
        output_dir=output_dir,
        figures_dir=figures_dir,
        verbose=verbose
    )

    # 2. Model recovery (unless skipped)
    model_recovery_result = None
    if not skip_model_recovery:
        synthetic_data_path = output_dir / 'synthetic_trials.csv'
        model_recovery_result = run_model_recovery_check(
            synthetic_data_path=str(synthetic_data_path),
            generative_model=model,
            output_dir=output_dir,
            use_gpu=use_gpu,
            n_jobs=n_jobs,
            verbose=verbose
        )

    # 3. Summary
    print(f"\n{'='*60}")
    print(f"PPC SUMMARY: {model.upper()}")
    print(f"{'='*60}")
    print("\nBehavioral Comparison:")
    print(comparison_df.to_string(index=False))

    if model_recovery_result:
        status = "PASS" if model_recovery_result['generative_wins'] else "FAIL"
        print(f"\nModel Recovery: {status}")
        print(f"  Generative: {model_recovery_result['generative_model']}")
        print(f"  Winner:     {model_recovery_result['winning_model']}")

    print(f"{'='*60}\n")

    return {
        'model': model,
        'behavioral_comparison': comparison_df,
        'model_recovery': model_recovery_result
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run posterior predictive checks (Senta et al. / Wilson & Collins methodology)'
    )
    parser.add_argument('--model', type=str, required=True,
                        choices=['qlearning', 'wmrl', 'wmrl_m3', 'all'],
                        help='Model to validate (or "all" for all models)')
    parser.add_argument('--fitted-params-dir', type=str, default='output/mle_results',
                        help='Directory containing fitted params CSVs')
    parser.add_argument('--real-data', type=str, default='output/task_trials_long.csv',
                        help='Path to real trial data')
    parser.add_argument('--output-dir', type=str, default='output/ppc',
                        help='Output directory for PPC results')
    parser.add_argument('--figures-dir', type=str, default='figures/ppc',
                        help='Output directory for figures')
    parser.add_argument('--skip-model-recovery', action='store_true',
                        help='Skip model recovery (faster, just behavioral comparison)')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU acceleration for model recovery fits')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel jobs for CPU fitting (default: 1)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    args = parser.parse_args()

    # Determine models to run
    if args.model == 'all':
        models = ['qlearning', 'wmrl', 'wmrl_m3']
    else:
        models = [args.model]

    # Run PPC for each model
    all_results = {}
    for model in models:
        fitted_params_path = Path(args.fitted_params_dir) / f'{model}_individual_fits.csv'

        if not fitted_params_path.exists():
            print(f"Warning: Fitted params not found at {fitted_params_path}")
            print(f"  Run `python scripts/12_fit_mle.py --model {model}` first")
            continue

        result = run_ppc_for_model(
            model=model,
            fitted_params_path=str(fitted_params_path),
            real_data_path=args.real_data,
            output_dir=Path(args.output_dir) / model,
            figures_dir=Path(args.figures_dir) / model,
            skip_model_recovery=args.skip_model_recovery,
            use_gpu=args.use_gpu,
            n_jobs=args.n_jobs,
            verbose=not args.quiet
        )
        all_results[model] = result

    # Final summary for multi-model run
    if len(models) > 1:
        print(f"\n{'='*60}")
        print("FINAL PPC SUMMARY")
        print(f"{'='*60}")
        for model, result in all_results.items():
            if result.get('model_recovery'):
                status = "PASS" if result['model_recovery']['generative_wins'] else "FAIL"
            else:
                status = "SKIPPED"
            print(f"  {model}: Model Recovery = {status}")
        print(f"{'='*60}\n")

    # Exit code
    if args.skip_model_recovery:
        sys.exit(0)

    all_pass = all(
        r.get('model_recovery', {}).get('generative_wins', True)
        for r in all_results.values()
    )
    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
