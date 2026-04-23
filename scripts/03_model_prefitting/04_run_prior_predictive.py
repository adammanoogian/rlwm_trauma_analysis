"""Prior-predictive check CLI (Baribault gate) — thin orchestrator.

Implements the stage 03 prior-predictive check for the v4.0 principled
Bayesian model-selection pipeline. The simulator / gate logic lives in
:mod:`scripts.utils.ppc`; this script is a thin argparse wrapper.

Exits 1 on FAIL so a pipeline orchestrator can short-circuit.

References
----------
Baribault, B. & Collins, A. G. E. (2023). Troubleshooting Bayesian
cognitive models. *Psychological Methods*.
https://doi.org/10.1037/met0000554

Hess, B. et al. (2025). A robust Bayesian workflow for computational
psychiatry. *Computational Psychiatry*, 9(1):76-99.
https://doi.org/10.5334/cpsy.116

Usage
-----
python scripts/03_model_prefitting/04_run_prior_predictive.py \\
    --model wmrl_m3 --num-draws 500
python scripts/03_model_prefitting/04_run_prior_predictive.py \\
    --model qlearning --num-draws 20 --output-dir /tmp/ppc_smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so ``from scripts.utils.ppc import ...``
# resolves regardless of the caller's CWD.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpyro  # noqa: E402

from rlwm.fitting.bayesian import STACKED_MODEL_DISPATCH  # noqa: E402
from scripts.utils.ppc import run_prior_ppc  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with ``model``, ``data``, ``num_draws``, ``seed``,
        and ``output_dir`` fields.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Prior-predictive gate (Baribault & Collins 2023) for RLWM "
            "choice-only hierarchical models."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(STACKED_MODEL_DISPATCH.keys()),
        help="Choice-only hierarchical model name.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("output/task_trials_long.csv"),
        help="Path to trial-level CSV (default: output/task_trials_long.csv).",
    )
    parser.add_argument(
        "--num-draws",
        type=int,
        default=500,
        help="Number of prior draws to simulate (default: 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="PRNG seed (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/bayesian/21_prior_predictive"),
        help=(
            "Output directory (default: output/bayesian/21_prior_predictive)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Entry point — returns 0 on PASS, 1 on FAIL."""
    args = _parse_args()
    # Silence the NumPyro ``numpyro.set_host_device_count`` warning if any
    numpyro.set_host_device_count(1)
    return run_prior_ppc(
        model=args.model,
        data_path=args.data,
        num_draws=args.num_draws,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
