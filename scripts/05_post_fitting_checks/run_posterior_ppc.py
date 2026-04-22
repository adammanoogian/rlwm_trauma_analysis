#!/usr/bin/env python
"""Posterior-predictive check CLI — stage 05 thin orchestrator.

Mirror of ``scripts/03_model_prefitting/09_run_ppc.py``, intended to live
alongside the other stage-05 post-fitting diagnostics (``baseline_audit``,
``scale_audit``). Both orchestrators are thin wrappers around the canonical
simulator in :mod:`scripts.utils.ppc`; this one exists so the stage 05
post-fit workflow has a first-class PPC entry point without having to
reach into stage 03.

Usage
-----
python scripts/05_post_fitting_checks/run_posterior_ppc.py --model wmrl_m3
python scripts/05_post_fitting_checks/run_posterior_ppc.py --model all \\
    --skip-model-recovery
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so ``from scripts.utils.ppc import ...``
# resolves regardless of the caller's CWD.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from config import ALL_MODELS  # noqa: E402
from scripts.utils.ppc import run_posterior_ppc  # noqa: E402


def main() -> int:
    """Run posterior PPC from MLE point estimates for one or all models.

    Returns
    -------
    int
        Exit code 0 on pass, 1 on any model-recovery failure.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run posterior predictive checks (stage 05 entry point). "
            "Delegates to scripts.utils.ppc.run_posterior_ppc."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=ALL_MODELS + ["all"],
        help='Model to validate (or "all" for all 7 models).',
    )
    parser.add_argument(
        "--fitted-params-dir",
        type=str,
        default="output/mle",
        help="Directory containing fitted params CSVs.",
    )
    parser.add_argument(
        "--real-data",
        type=str,
        default="output/task_trials_long.csv",
        help="Path to real trial data.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/ppc",
        help="Output directory for PPC results.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="figures/ppc",
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--skip-model-recovery",
        action="store_true",
        help="Skip model recovery (faster, just behavioral comparison).",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU acceleration for model recovery fits.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for CPU fitting (default: 1).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    args = parser.parse_args()

    models = ALL_MODELS if args.model == "all" else [args.model]

    all_results: dict[str, dict] = {}
    for model in models:
        fitted_params_path = Path(args.fitted_params_dir) / f"{model}_individual_fits.csv"
        if not fitted_params_path.exists():
            print(f"Warning: Fitted params not found at {fitted_params_path}")
            continue

        result = run_posterior_ppc(
            model=model,
            fitted_params_path=str(fitted_params_path),
            real_data_path=args.real_data,
            output_dir=Path(args.output_dir) / model,
            figures_dir=Path(args.figures_dir) / model,
            skip_model_recovery=args.skip_model_recovery,
            use_gpu=args.use_gpu,
            n_jobs=args.n_jobs,
            verbose=not args.quiet,
        )
        all_results[model] = result

    if args.skip_model_recovery:
        return 0

    all_pass = all(
        r.get("model_recovery", {}).get("generative_wins", True)
        for r in all_results.values()
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
