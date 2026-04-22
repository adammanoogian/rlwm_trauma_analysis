"""Step 21.3 baseline hierarchical fit runner (NO L2 scales, 21_baseline subdir).

Phase 21 Wave 3 orchestrator: fit one of the 6 choice-only models
hierarchically with ``covariate_lec=None`` (i.e. no Level-2 trauma design
matrix) on the canonical v4.0 N=138 cohort. This establishes the
anti-circular baseline for the Wave-5 PSIS-LOO + stacking + RFX-BMS/PXP
ranking in step 21.5. Every candidate model — not a pre-selected winner —
starts with equal footing.

All outputs are routed under ``output/bayesian/21_baseline/`` via the
``--output-subdir`` flag on the co-located Bayesian engine
(``scripts/04_model_fitting/b_bayesian/_engine.py``, plumbed through
``save_results``, ``write_bayesian_summary``, and
``run_posterior_predictive_check`` in plan 21-04 Task 1). This ensures the
Phase 16 posteriors at ``output/bayesian/{model}_posterior.nc`` are **never
overwritten**.

Convergence behaviour
---------------------
- NUTS auto-bumps ``target_accept_prob`` through (0.80, 0.95, 0.99) inside
  ``run_inference_with_bump`` on divergences.
- Convergence gate (R-hat < 1.01, ESS_bulk > 400, divergences == 0) lives
  inside ``save_results``; on failure, **no files are written** and the
  function returns ``None`` (silent exit 0 from the inner process).
- The post-fit ``expected.exists()`` check in :func:`main` below is
  **load-bearing**: it converts that silent failure into a non-zero exit so
  SLURM ``--dependency=afterok`` chains (plan 21-10 master orchestrator) do
  not silently skip to the next step. See plan 21-04 Task 2 action text.

Usage
-----
>>> python scripts/04_model_fitting/b_bayesian/fit_baseline.py --model wmrl_m6b
>>> python scripts/04_model_fitting/b_bayesian/fit_baseline.py --model qlearning \\
...     --warmup 500 --samples 1000 --chains 2 --seed 123

See also
--------
- ``cluster/21_3_fit_baseline.slurm`` — SLURM submission template, 10h wall.
- ``.planning/phases/21-principled-bayesian-model-selection-pipeline/``
  21-04-PLAN.md for the plan specification.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

# -- Path bootstrap so this script runs both interactively and under SLURM.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load the co-located Bayesian engine by absolute path.  The parent package
# `scripts.04_model_fitting.b_bayesian` cannot be imported via the standard
# dotted form because Python dotted names cannot start with a digit.  Plan
# 29-04b renamed the engine from `fit_bayesian.py` → `_engine.py` (Scheme
# D underscore-private convention) so the canonical name is free for the
# thin CLI entry script (`fit_bayesian.py`).
_ENGINE_PATH = _THIS_FILE.with_name("_engine.py")
_spec = importlib.util.spec_from_file_location(
    "_bayesian_engine", str(_ENGINE_PATH)
)
assert _spec is not None and _spec.loader is not None, (
    f"Could not create import spec for Bayesian engine at {_ENGINE_PATH}"
)
_engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_engine)
fit_main = _engine.main

# The 6 choice-only models that go through `STACKED_MODEL_DISPATCH` in
# fit_bayesian.py.  M4 (wmrl_m4) is the joint RT+choice LBA model and
# follows a separate pipeline (cluster/13_bayesian_m4_gpu.slurm); it is not
# part of the step 21.3 baseline ranking.
BASELINE_MODELS: tuple[str, ...] = (
    "qlearning",
    "wmrl",
    "wmrl_m3",
    "wmrl_m5",
    "wmrl_m6a",
    "wmrl_m6b",
)

# Subdirectory under `output/bayesian/` where all step 21.3 artefacts land.
# DO NOT CHANGE without also updating the master pipeline orchestrator
# (plan 21-10) and downstream convergence audit (plan 21-05).
BASELINE_SUBDIR: str = "21_baseline"


def main() -> None:
    """Thin orchestrator over the co-located Bayesian engine's ``main()``.

    Rewrites :data:`sys.argv` with the fixed baseline configuration
    (``--output-subdir 21_baseline`` always injected, no ``--subscale``, no
    ``--permutation-shuffle``) and delegates to the engine's ``main()``.
    After the fit returns, explicitly checks that the expected posterior
    NetCDF exists — if it does not, the inner convergence gate failed and
    this runner exits ``1`` so SLURM captures a real failure.

    Raises
    ------
    SystemExit
        With exit code 1 when the expected NetCDF is missing after
        ``fit_main()`` returns — indicates convergence gate failure inside
        ``save_results``. This is load-bearing for the downstream
        ``--dependency=afterok`` chain used by the Phase 21 master
        orchestrator.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Step 21.3 baseline hierarchical fit (NO L2 scales). Fits one "
            "of the 6 choice-only models with covariate_lec=None on the "
            "canonical v4.0 N=138 cohort and writes artefacts to "
            "output/bayesian/21_baseline/."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(BASELINE_MODELS),
        help=(
            "Choice-only model to fit. One of: "
            + ", ".join(BASELINE_MODELS)
            + "."
        ),
    )
    parser.add_argument(
        "--data",
        default="output/task_trials_long.csv",
        help="Path to trial-level CSV (default: output/task_trials_long.csv).",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of NUTS chains (default: 4).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1000,
        help="Number of NUTS warmup draws (default: 1000).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of posterior samples per chain (default: 2000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="JAX PRNG seed (default: 42).",
    )
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=10,
        help=(
            "Maximum NUTS tree depth (default: 10 → up to 1024 leapfrog "
            "steps). Matches the canonical v4.0 setting."
        ),
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Root output directory (default: output).",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("STEP 21.3 — BASELINE HIERARCHICAL FIT (NO L2)")
    print("=" * 80)
    print(f"  Model: {args.model}")
    print(f"  Data: {args.data}")
    print(f"  Chains: {args.chains}")
    print(f"  Warmup: {args.warmup}")
    print(f"  Samples: {args.samples}")
    print(f"  Seed: {args.seed}")
    print(f"  Max tree depth: {args.max_tree_depth}")
    print(f"  Output: {args.output}")
    print(f"  Output subdir: bayesian/{BASELINE_SUBDIR}/ (forced)")
    print("=" * 80)

    # Delegate to fit_bayesian.main() with the fixed baseline flags.
    # `covariate_lec` defaults to None inside `_fit_stacked_model`, so
    # omitting any L2 flag yields the no-scales baseline naturally. Do NOT
    # re-implement data loading or MCMC — reuse the Phase 16 pipeline.
    sys.argv = [
        "fit_bayesian.py",
        "--model", args.model,
        "--data", args.data,
        "--chains", str(args.chains),
        "--warmup", str(args.warmup),
        "--samples", str(args.samples),
        "--seed", str(args.seed),
        "--max-tree-depth", str(args.max_tree_depth),
        "--output", args.output,
        "--output-subdir", BASELINE_SUBDIR,
    ]
    fit_main()

    # ------------------------------------------------------------------
    # CRITICAL — convergence-gate failure surface (Issue #1, plan-checker).
    #
    # fit_main() returns silently (exit 0) when save_results returns None
    # on convergence-gate failure. Without this explicit check, SLURM sees
    # exit 0 and the downstream convergence audit (21.4) finds a missing
    # NetCDF and reports the model as "EXCLUDED_MISSING_FILE" instead of
    # a real SLURM failure. That hides the real blocker from Phase 21 SC #2
    # ("All 6 models converge OR are explicitly dropped with documented
    # reason").
    #
    # DO NOT REMOVE — this is the only thing that turns a silent
    # convergence-gate failure into a non-zero SLURM exit that blocks the
    # downstream `--dependency=afterok` chains used by the master
    # orchestrator (plan 21-10).
    # ------------------------------------------------------------------
    expected = (
        Path(args.output) / "bayesian" / BASELINE_SUBDIR
        / f"{args.model}_posterior.nc"
    )
    if not expected.exists():
        print(
            f"[CONVERGENCE GATE FAIL] Expected NetCDF missing: {expected}. "
            f"fit_bayesian either failed convergence (R-hat, ESS, divergences) "
            f"or aborted before save_results. Exiting 1 so SLURM captures the "
            f"failure (block pipeline SC #2).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\n[STEP 21.3 COMPLETE] Posterior NetCDF written: {expected}")


if __name__ == "__main__":
    main()
