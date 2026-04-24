#!/usr/bin/env python
"""
11: Run Model Recovery
======================

Validates the model fitting procedure by generating synthetic data
with known parameters and attempting to recover them via MLE.

This is a pipeline script that wraps the fitting library module.

Purpose:
    - Verify that fitting procedure can recover true parameters
    - Identify parameter identifiability issues
    - Report PASS/FAIL per Senta et al. (2025) criterion (r >= 0.80)
    - Cross-model recovery: verify each model wins AIC against all others

Usage:
    # Parameter recovery (default mode) - single model
    python scripts/03_model_prefitting/03_run_model_recovery.py --model wmrl_m3

    # Parameter recovery - all models
    python scripts/03_model_prefitting/03_run_model_recovery.py --model all

    # Quick test (fewer subjects/datasets)
    python scripts/03_model_prefitting/03_run_model_recovery.py --model qlearning --n-subjects 20 --n-datasets 2

    # With GPU acceleration
    python scripts/03_model_prefitting/03_run_model_recovery.py --model wmrl_m3 --use-gpu

    # Cross-model recovery (generates from each model, checks AIC winner)
    python scripts/03_model_prefitting/03_run_model_recovery.py --mode cross-model --model all --n-subjects 10 --n-datasets 3

    # Cross-model recovery for a single generating model
    python scripts/03_model_prefitting/03_run_model_recovery.py --mode cross-model --model wmrl_m5 --n-subjects 5 --n-datasets 1

Outputs:
    Parameter mode:
        - models/recovery/{model}/recovery_results.csv
        - models/recovery/{model}/recovery_metrics.csv
        - reports/figures/recovery/{model}/*.png
        - Console: PASS/FAIL summary per parameter

    Cross-model mode:
        - models/recovery/cross_model_confusion.csv
        - Console: confusion matrix (rows=generator, cols=AIC winner)

Interpretation:
    - PASS (r >= 0.80): Parameter recovery adequate per Senta et al.
    - FAIL (r < 0.80): Parameter identifiability issues, investigate
    - Cross-model PASS: generating model wins plurality of datasets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path (parents[2] = project root; parents[1] = scripts/)
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from config import MODELS_RECOVERY_DIR, REPORTS_FIGURES_DIR
from scripts.fitting.model_recovery import (
    CHOICE_ONLY_MODELS,
    compute_recovery_metrics,
    plot_distribution_comparison,
    plot_recovery_scatter,
    run_cross_model_recovery,
    run_parameter_recovery,
)


def run_recovery_for_model(model: str, n_subjects: int, n_datasets: int,
                           seed: int, use_gpu: bool, verbose: bool,
                           n_starts: int = 50, n_jobs: int = 1) -> bool:
    """
    Run parameter recovery for a single model and evaluate pass/fail.

    Returns True if all parameters pass (r >= 0.80), False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"PARAMETER RECOVERY: {model.upper()}")
    print(f"{'='*60}")
    print(f"N subjects: {n_subjects}")
    print(f"N datasets: {n_datasets}")
    print(f"N starts: {n_starts}")
    print(f"N jobs: {n_jobs}")
    print(f"Seed: {seed}")
    print(f"GPU: {use_gpu}")
    print(f"{'='*60}\n")

    # Run recovery
    results_df = run_parameter_recovery(
        model=model,
        n_subjects=n_subjects,
        n_datasets=n_datasets,
        seed=seed,
        use_gpu=use_gpu,
        verbose=verbose,
        n_starts=n_starts,
        n_jobs=n_jobs
    )

    # Compute metrics
    metrics_df = compute_recovery_metrics(results_df, model)

    # Save outputs
    output_dir = MODELS_RECOVERY_DIR / model
    figures_dir = REPORTS_FIGURES_DIR / 'recovery' / model
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / 'recovery_results.csv', index=False)
    metrics_df.to_csv(output_dir / 'recovery_metrics.csv', index=False)

    # Generate plots
    plot_recovery_scatter(results_df, metrics_df, model, figures_dir)
    plot_distribution_comparison(results_df, None, model, figures_dir)

    # Print pass/fail summary
    print(f"\n{'='*60}")
    print(f"RECOVERY RESULTS: {model.upper()}")
    print(f"{'='*60}")
    print(f"{'Parameter':<15} {'Pearson r':>10} {'RMSE':>10} {'Bias':>10} {'Status':>10}")
    print(f"{'-'*60}")

    all_pass = True
    for _, row in metrics_df.iterrows():
        status = "PASS" if row['pass_fail'] == 'PASS' else "FAIL"
        if row['pass_fail'] != 'PASS':
            all_pass = False
        print(f"{row['parameter']:<15} {row['pearson_r']:>10.3f} {row['rmse']:>10.3f} {row['bias']:>+10.3f} {status:>10}")

    print(f"{'-'*60}")
    overall = "PASS" if all_pass else "FAIL"
    n_pass = (metrics_df['pass_fail'] == 'PASS').sum()
    n_total = len(metrics_df)
    print(f"Overall: {overall} ({n_pass}/{n_total} parameters meet r >= 0.80)")
    print(f"{'='*60}\n")

    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description='Run parameter recovery analysis (Senta et al. methodology)'
    )
    parser.add_argument('--mode', type=str, default='parameter',
                        choices=['parameter', 'cross-model'],
                        help='Recovery mode: "parameter" (default) recovers params for one model, '
                             '"cross-model" generates from each model and checks AIC winner')
    parser.add_argument('--model', type=str, required=True,
                        choices=['qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b', 'wmrl_m4', 'all'],
                        help='Model to test (or "all" for all models)')
    parser.add_argument('--n-subjects', type=int, default=50,
                        help='Number of synthetic subjects per dataset')
    parser.add_argument('--n-datasets', type=int, default=10,
                        help='Number of independent datasets')
    parser.add_argument('--n-starts', type=int, default=50,
                        help='Number of random starts for MLE optimization (default: 50)')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel jobs (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU acceleration')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    args = parser.parse_args()

    if args.mode == 'cross-model':
        # ===== CROSS-MODEL RECOVERY MODE =====
        # Determine generating models (choice-only only; M4 excluded)
        if args.model == 'all':
            generating_models = list(CHOICE_ONLY_MODELS)
        elif args.model == 'wmrl_m4':
            print("Error: M4 is excluded from cross-model recovery (joint likelihood "
                  "incommensurable with choice-only AIC).")
            sys.exit(1)
        else:
            generating_models = [args.model]

        # Run cross-model recovery
        result = run_cross_model_recovery(
            generating_models=generating_models,
            n_subjects=args.n_subjects,
            n_datasets=args.n_datasets,
            seed=args.seed,
            use_gpu=args.use_gpu,
            n_jobs=args.n_jobs,
            n_starts=args.n_starts,
            verbose=not args.quiet
        )

        # Save confusion matrix
        output_path = MODELS_RECOVERY_DIR / 'cross_model_confusion.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result['confusion_matrix'].to_csv(output_path)
        print(f"\nConfusion matrix saved to: {output_path}")

        # Exit code
        sys.exit(0 if result['all_pass'] else 1)

    else:
        # ===== PARAMETER RECOVERY MODE (default) =====
        # Determine which models to run
        if args.model == 'all':
            models = ['qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b', 'wmrl_m4']
        else:
            models = [args.model]

        # Run recovery for each model
        all_results = {}
        for model in models:
            passed = run_recovery_for_model(
                model=model,
                n_subjects=args.n_subjects,
                n_datasets=args.n_datasets,
                seed=args.seed,
                use_gpu=args.use_gpu,
                verbose=not args.quiet,
                n_starts=args.n_starts,
                n_jobs=args.n_jobs
            )
            all_results[model] = passed

        # Print final summary if multiple models
        if len(models) > 1:
            print(f"\n{'='*60}")
            print("FINAL SUMMARY")
            print(f"{'='*60}")
            for model, passed in all_results.items():
                status = "PASS" if passed else "FAIL"
                print(f"  {model}: {status}")

            all_pass = all(all_results.values())
            print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
            print(f"{'='*60}\n")

        # Exit with appropriate code
        exit_code = 0 if all(all_results.values()) else 1
        sys.exit(exit_code)


if __name__ == '__main__':
    main()
