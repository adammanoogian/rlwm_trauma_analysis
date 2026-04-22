"""Step 21.6 — winner refit WITH L2 scales (21_l2 subdir).

Phase 21 Wave 6 orchestrator: given the winner set produced by step 21.5
(``output/bayesian/21_baseline/winners.txt``), refit ONE winner at a time
hierarchically with the appropriate Level-2 design matrix. Dispatched in
three branches matching the user-approved Option C: "code all variants in
principle, pipeline uses one."

L2 dispatch
-----------
- **M1 (qlearning), M2 (wmrl)** — no L2-compatible parameter target.
  ``shutil.copy2`` the baseline posterior from ``output/bayesian/21_baseline/``
  to ``output/bayesian/21_l2/``; log a ``[SKIP L2]`` warning. Preferred over
  a cheap re-fit (wastes cluster cycles producing an identical posterior).

- **M3 (wmrl_m3), M5 (wmrl_m5), M6a (wmrl_m6a)** — 2-covariate L2 refit via
  the ``covariate_iesr`` kwarg from plan 21-11. Design matrix built with
  ``build_level2_design_matrix_2cov`` (LEC total + IES-R total, both
  z-scored). The model function is called directly with both covariates
  via a local ``model_args`` shim — ``fit_bayesian._fit_stacked_model``
  hard-codes only ``covariate_lec`` so it cannot be reused here. Samples
  both ``beta_lec_{target}`` and ``beta_iesr_{target}`` (target=kappa for
  M3/M5, kappa_s for M6a).

- **M6b (wmrl_m6b)** — full 4-covariate subscale L2 refit via the existing
  ``wmrl_m6b_hierarchical_model_subscale`` path. Delegate to
  ``fit_bayesian.main()`` with ``--subscale`` + ``--output-subdir 21_l2``
  via ``sys.argv`` rewrite. 32 beta sites (8 params x 4 covariates).

Per user-approved Option C (Phase 21 revision): M3/M5/M6a use 2-covariate
L2 (``lec_total`` + ``iesr_total``, both z-scored) via the ``covariate_iesr``
hook built in plan 21-11; M6b uses full 4-covariate subscale (32 betas).
Sensitivity analyses with full 4-covariate for M3/M5/M6a (adding
``iesr_intr_resid`` + ``iesr_avd_resid``) deferred to v5.0.

Convergence behaviour
---------------------
- For the M3/M5/M6a shim path, the convergence gate lives inside
  ``save_results`` (HIER-07: R-hat < 1.01 AND ESS > 400 AND divergences == 0).
  On failure, ``save_results`` returns ``None`` and writes no files; the
  post-fit ``expected.exists()`` check in :func:`main` below is
  **load-bearing** to convert that silent failure into a non-zero exit so
  SLURM ``--dependency=afterok`` chains (plan 21-10 master orchestrator)
  do not silently skip to step 21.7.
- For the M6b subscale delegate path, ``fit_bayesian.main()`` applies the
  same gate; the post-fit expected-file check still runs.
- For the M1/M2 copy path, ``shutil.copy2`` preserves the already-gated
  baseline NetCDF — no re-fit, no additional gate.

Usage
-----
>>> # Assuming winners.txt contains "M6b":
>>> python scripts/21_fit_with_l2.py --model wmrl_m6b
>>> # Assuming winners.txt contains "M3,M6b":
>>> python scripts/21_fit_with_l2.py --model wmrl_m3

See also
--------
- ``cluster/21_6_fit_with_l2.slurm`` — SLURM submission template, 12h cap.
- ``scripts/fitting/level2_design.py::build_level2_design_matrix_2cov`` —
  2-covariate design builder (plan 21-11).
- ``.planning/phases/21-principled-bayesian-model-selection-pipeline/``
  21-07-PLAN.md for the plan specification.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# -- Path bootstrap so this script runs both interactively and under SLURM.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import load_netcdf_with_validation  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Display-name <-> internal-id mappings (mirrors 14_compare_models.py and
# 21_compute_loo_stacking.py). winners.txt contains display names (M1, M3,
# M6b, ...); our --model CLI uses internal ids (qlearning, wmrl_m3,
# wmrl_m6b, ...). We validate --model against winners.txt in BOTH
# directions.
DISPLAY_TO_INTERNAL: dict[str, str] = {
    "M1": "qlearning",
    "M2": "wmrl",
    "M3": "wmrl_m3",
    "M5": "wmrl_m5",
    "M6a": "wmrl_m6a",
    "M6b": "wmrl_m6b",
}
INTERNAL_TO_DISPLAY: dict[str, str] = {v: k for k, v in DISPLAY_TO_INTERNAL.items()}

# Models that bypass L2 entirely (M1/M2 — no L2-compatible parameter target).
_COPY_BASELINE_MODELS: frozenset[str] = frozenset({"qlearning", "wmrl"})

# Models that use the 2-covariate L2 path (LEC + IES-R totals).
_TWO_COV_MODELS: frozenset[str] = frozenset({"wmrl_m3", "wmrl_m5", "wmrl_m6a"})

# M6b uses the 4-covariate subscale path.
_SUBSCALE_MODELS: frozenset[str] = frozenset({"wmrl_m6b"})

# Target parameter on which beta_lec_/beta_iesr_ sites are sampled per model.
# Used to verify both L2 sites land in the posterior NetCDF.
_BETA_TARGET: dict[str, str] = {
    "wmrl_m3": "kappa",
    "wmrl_m5": "kappa",
    "wmrl_m6a": "kappa_s",
}

# Subdirectory under ``output/bayesian/`` where all step 21.6 artefacts land.
# DO NOT CHANGE without also updating the master pipeline orchestrator
# (plan 21-10) and step 21.7.
L2_SUBDIR: str = "21_l2"
BASELINE_SUBDIR: str = "21_baseline"


def _parse_winners_file(winners_path: Path) -> list[str]:
    """Parse a winners.txt file into a list of internal model ids.

    Parameters
    ----------
    winners_path : Path
        Path to ``winners.txt`` from step 21.5 (comma-separated display names
        like ``"M3,M6b\\n"``).

    Returns
    -------
    list of str
        Internal model ids (e.g. ``["wmrl_m3", "wmrl_m6b"]``).

    Raises
    ------
    FileNotFoundError
        If ``winners_path`` does not exist.
    ValueError
        If a display name in the file is not a recognised key of
        ``DISPLAY_TO_INTERNAL``.
    """
    if not winners_path.exists():
        raise FileNotFoundError(
            f"winners file not found: {winners_path}. Expected step 21.5 "
            f"(scripts/21_compute_loo_stacking.py) to have written it."
        )

    raw = winners_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(
            f"winners file {winners_path} is empty. Expected comma-separated "
            f"display names like 'M3,M6b'."
        )

    display_names = [tok.strip() for tok in raw.split(",") if tok.strip()]
    internal_ids: list[str] = []
    for name in display_names:
        if name not in DISPLAY_TO_INTERNAL:
            raise ValueError(
                f"Unknown winner display name '{name}' in {winners_path}. "
                f"Expected one of: {sorted(DISPLAY_TO_INTERNAL.keys())}."
            )
        internal_ids.append(DISPLAY_TO_INTERNAL[name])
    return internal_ids


def _copy_baseline_posterior(
    model: str,
    output_root: Path,
) -> Path:
    """Copy baseline posterior NetCDF to the 21_l2 subdir (M1/M2 pass-through).

    Parameters
    ----------
    model : str
        Internal model id (``"qlearning"`` or ``"wmrl"``).
    output_root : Path
        Output root directory (typically ``output/``).

    Returns
    -------
    Path
        Destination NetCDF path (``output/bayesian/21_l2/{model}_posterior.nc``).

    Raises
    ------
    FileNotFoundError
        If the baseline NetCDF source file is missing (upstream 21.3 did not
        produce a converged posterior for this model).
    """
    src = output_root / "bayesian" / BASELINE_SUBDIR / f"{model}_posterior.nc"
    dst_dir = output_root / "bayesian" / L2_SUBDIR
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{model}_posterior.nc"

    if not src.exists():
        raise FileNotFoundError(
            f"Baseline posterior missing for {model}: expected {src}. "
            f"Step 21.3 (21_fit_baseline.py) must converge before 21.6 can "
            f"copy it forward. Check output/bayesian/21_baseline/ contents."
        )

    print(
        f"[SKIP L2] {model} has no L2-compatible parameter target "
        f"(M1/M2 never have a kappa/kappa_s site). "
        f"Copying baseline posterior instead of re-fitting."
    )
    shutil.copy2(src, dst)
    print(f"  Source: {src}")
    print(f"  Dest:   {dst}")
    return dst


def _fit_two_covariate_l2(
    model: str,
    args: argparse.Namespace,
) -> Path:
    """Fit M3/M5/M6a with 2-covariate L2 (LEC + IES-R totals, z-scored).

    Bypasses :func:`fit_bayesian._fit_stacked_model` because that function
    hard-codes only ``covariate_lec`` in its ``model_args`` dict. Instead we
    assemble ``model_args`` locally with both covariates and call
    :func:`run_inference_with_bump` + :func:`save_results` directly.

    Parameters
    ----------
    model : str
        Internal model id. Must be in :data:`_TWO_COV_MODELS`.
    args : argparse.Namespace
        Parsed CLI arguments (must include ``data``, ``chains``, ``warmup``,
        ``samples``, ``seed``, ``max_tree_depth``, ``output``).

    Returns
    -------
    Path
        Expected NetCDF path (``output/bayesian/21_l2/{model}_posterior.nc``).
        Existence is NOT verified here — caller runs the load-bearing
        ``expected.exists()`` check.

    Raises
    ------
    ValueError
        If ``model`` is not in :data:`_TWO_COV_MODELS`.
    FileNotFoundError
        If ``output/summary_participant_metrics.csv`` is missing (required
        for the 2-covariate L2 design matrix).
    """
    if model not in _TWO_COV_MODELS:
        raise ValueError(
            f"_fit_two_covariate_l2: model='{model}' not in 2-cov set "
            f"{sorted(_TWO_COV_MODELS)}."
        )

    # Local imports keep jax/numpyro startup cost out of --help path.
    import pandas as pd  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from scripts.fitting.fit_bayesian import (  # noqa: PLC0415
        STACKED_MODEL_DISPATCH,
        load_and_prepare_data,
        run_inference_with_bump,
        save_results,
    )
    from scripts.fitting.level2_design import (  # noqa: PLC0415
        build_level2_design_matrix_2cov,
    )
    from rlwm.fitting.numpyro_models import (  # noqa: PLC0415
        prepare_stacked_participant_data,
        stack_across_participants,
    )

    # ------------------------------------------------------------------
    # Step 1: Load trial data + build stacked participant dict.
    # ------------------------------------------------------------------
    print("\n>> Loading trial data (canonical v4.0 cohort)...")
    data = load_and_prepare_data(Path(args.data), use_cohort=True)
    participant_data_stacked = prepare_stacked_participant_data(
        data,
        participant_col="sona_id",
        block_col="block",
        stimulus_col="stimulus",
        action_col="key_press",
        reward_col="reward",
        set_size_col="set_size",
    )
    participant_ids = sorted(participant_data_stacked.keys())
    print(f"  {len(participant_ids)} participants (sorted by id).")

    # ------------------------------------------------------------------
    # Step 2: Load 2-covariate design matrix (LEC + IES-R totals, z-scored).
    # ------------------------------------------------------------------
    metrics_path = Path("output/summary_participant_metrics.csv")
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"summary_participant_metrics.csv not found at {metrics_path}. "
            f"This file is REQUIRED for the 2-covariate L2 design matrix "
            f"(contains less_total_events + ies_total columns)."
        )
    print(f"\n>> Loading 2-covariate L2 design matrix from {metrics_path}...")
    metrics = pd.read_csv(metrics_path)
    design, cov_names = build_level2_design_matrix_2cov(metrics, participant_ids)
    print(f"  Design shape: {design.shape}, covariates: {cov_names}")
    covariate_lec = jnp.array(design[:, 0], dtype=jnp.float32)
    covariate_iesr = jnp.array(design[:, 1], dtype=jnp.float32)

    # ------------------------------------------------------------------
    # Step 3: Pre-compute stacked arrays (fully-batched vmap path).
    # ------------------------------------------------------------------
    stacked_arrays = stack_across_participants(participant_data_stacked)
    model_fn = STACKED_MODEL_DISPATCH[model]
    model_args = {
        "participant_data_stacked": participant_data_stacked,
        "covariate_lec": covariate_lec,
        "covariate_iesr": covariate_iesr,
        "stacked_arrays": stacked_arrays,
        "use_pscan": False,
    }

    # ------------------------------------------------------------------
    # Step 4: NUTS with convergence auto-bump.
    # ------------------------------------------------------------------
    print("\n>> Running MCMC inference (NUTS + convergence auto-bump)...")
    mcmc = run_inference_with_bump(
        model_fn,
        model_args,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        seed=args.seed,
        max_tree_depth=args.max_tree_depth,
    )

    # ------------------------------------------------------------------
    # Step 5: Save results (HIER-07 convergence gate applied inside).
    # ------------------------------------------------------------------
    save_results(
        mcmc,
        data,
        model,
        Path(args.output),
        save_plots=False,
        participant_data_stacked=participant_data_stacked,
        use_pscan=False,
        output_subdir=L2_SUBDIR,
    )

    expected = Path(args.output) / "bayesian" / L2_SUBDIR / f"{model}_posterior.nc"
    return expected


def _verify_two_covariate_sites(model: str, expected: Path) -> int:
    """Verify that both beta_lec_* and beta_iesr_* sites exist in the NetCDF.

    Parameters
    ----------
    model : str
        Internal model id (must be in :data:`_TWO_COV_MODELS`).
    expected : Path
        Path to the saved posterior NetCDF.

    Returns
    -------
    int
        Number of beta_* sites found (expected: 2 for M3/M5/M6a 2-cov path).

    Raises
    ------
    RuntimeError
        If either ``beta_lec_{target}`` or ``beta_iesr_{target}`` is missing
        from the posterior group.
    """
    target = _BETA_TARGET[model]
    required_sites = {f"beta_lec_{target}", f"beta_iesr_{target}"}

    idata = load_netcdf_with_validation(expected, model)
    posterior_vars = set(idata.posterior.data_vars)
    missing = required_sites - posterior_vars
    if missing:
        raise RuntimeError(
            f"2-covariate L2 fit for {model} is missing required posterior "
            f"sites: {sorted(missing)}. Found beta_* vars: "
            f"{sorted(v for v in posterior_vars if v.startswith('beta_'))}. "
            f"Expected both beta_lec_{target} and beta_iesr_{target}."
        )

    found = sorted(v for v in posterior_vars if v.startswith("beta_"))
    print(f"  Verified beta_* sites in posterior: {found}")
    return len(found)


def _fit_subscale(model: str, args: argparse.Namespace) -> Path:
    """Fit M6b with the 4-covariate subscale design via fit_bayesian.main().

    Delegates to :func:`scripts.fitting.fit_bayesian.main` by rewriting
    ``sys.argv`` with ``--subscale`` + ``--output-subdir 21_l2``. Matches
    the Phase-16 M6b subscale pattern; requires no shim.

    Parameters
    ----------
    model : str
        Must be ``"wmrl_m6b"``.
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    Path
        Expected NetCDF path.
    """
    if model != "wmrl_m6b":
        raise ValueError(
            f"_fit_subscale: expected model='wmrl_m6b', got '{model}'."
        )

    from scripts.fitting.fit_bayesian import main as fit_main  # noqa: PLC0415

    sys.argv = [
        "fit_bayesian.py",
        "--model", model,
        "--data", args.data,
        "--chains", str(args.chains),
        "--warmup", str(args.warmup),
        "--samples", str(args.samples),
        "--seed", str(args.seed),
        "--max-tree-depth", str(args.max_tree_depth),
        "--output", args.output,
        "--output-subdir", L2_SUBDIR,
        "--subscale",
    ]
    fit_main()

    expected = Path(args.output) / "bayesian" / L2_SUBDIR / f"{model}_posterior.nc"
    return expected


def main() -> None:
    """Winner refit orchestrator entrypoint.

    Reads ``winners.txt``, validates the requested ``--model`` is in the
    winner set, and dispatches to one of three L2 variants (copy / 2-cov /
    subscale). Exits 0 on success, 1 on convergence-gate failure or any
    precondition check.

    Raises
    ------
    SystemExit
        Exit code 1 when any precondition fails or when the expected
        NetCDF is missing after fitting (load-bearing for the Phase 21
        master orchestrator's ``--dependency=afterok`` chain).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Step 21.6 winner refit WITH L2 scales. Reads winners.txt from "
            "step 21.5, dispatches ONE winner (passed via --model) to the "
            "appropriate L2 variant: M1/M2 copy baseline, M3/M5/M6a 2-cov "
            "(LEC + IES-R totals), M6b 4-cov subscale (32 betas)."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Internal model id to refit. One of: "
            + ", ".join(sorted(DISPLAY_TO_INTERNAL.values()))
            + ". Must appear in the winner set from --winners-file."
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
        help="Maximum NUTS tree depth (default: 10).",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Root output directory (default: output).",
    )
    parser.add_argument(
        "--winners-file",
        default="output/bayesian/21_baseline/winners.txt",
        help=(
            "Path to winners.txt from step 21.5 (default: "
            "output/bayesian/21_baseline/winners.txt). "
            "File must contain comma-separated display names "
            "(e.g. 'M3,M6b')."
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------
    print("=" * 80)
    print("STEP 21.6 — WINNER REFIT WITH L2 SCALES")
    print("=" * 80)
    print(f"  Model: {args.model}")
    print(f"  Data: {args.data}")
    print(f"  Winners file: {args.winners_file}")
    print(f"  Chains/Warmup/Samples: {args.chains}/{args.warmup}/{args.samples}")
    print(f"  Seed: {args.seed}")
    print(f"  Max tree depth: {args.max_tree_depth}")
    print(f"  Output: {args.output}/bayesian/{L2_SUBDIR}/")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Step A: Parse winners.txt and validate --model is in the winner set.
    # ------------------------------------------------------------------
    model_internal = args.model
    if model_internal not in INTERNAL_TO_DISPLAY:
        print(
            f"[FAIL] --model '{model_internal}' is not a recognised internal "
            f"id. Expected: {sorted(INTERNAL_TO_DISPLAY.keys())}.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        winners_internal = _parse_winners_file(Path(args.winners_file))
    except (FileNotFoundError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)

    winners_display = [INTERNAL_TO_DISPLAY[w] for w in winners_internal]
    print(
        f"\n>> Winners parsed from {args.winners_file}: "
        f"{winners_display} -> {winners_internal}"
    )

    if model_internal not in winners_internal:
        model_display = INTERNAL_TO_DISPLAY[model_internal]
        print(
            f"[FAIL] Model '{model_internal}' ({model_display}) is not in "
            f"the winner set {winners_internal} ({winners_display}). "
            f"Expected 21.5 to nominate this model before 21.6 refits it.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step B: Dispatch to the appropriate L2 variant.
    # ------------------------------------------------------------------
    output_root = Path(args.output)
    beta_count: int

    if model_internal in _COPY_BASELINE_MODELS:
        # M1/M2 pass-through: copy baseline posterior into 21_l2/.
        try:
            _copy_baseline_posterior(model_internal, output_root)
        except FileNotFoundError as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            sys.exit(1)
        beta_count = 0

    elif model_internal in _TWO_COV_MODELS:
        # M3/M5/M6a: 2-covariate L2 (LEC + IES-R totals) via covariate_iesr.
        try:
            expected = _fit_two_covariate_l2(model_internal, args)
        except FileNotFoundError as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            sys.exit(1)
        if not expected.exists():
            print(
                f"[CONVERGENCE GATE FAIL] Expected NetCDF missing: {expected}. "
                f"The 2-covariate L2 fit for {model_internal} either failed "
                f"convergence (R-hat, ESS, divergences) or aborted before "
                f"save_results. Exiting 1 so SLURM captures the failure.",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            beta_count = _verify_two_covariate_sites(model_internal, expected)
        except RuntimeError as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            sys.exit(1)

    elif model_internal in _SUBSCALE_MODELS:
        # M6b: full 4-covariate subscale L2 via fit_bayesian.main().
        expected = _fit_subscale(model_internal, args)
        if not expected.exists():
            print(
                f"[CONVERGENCE GATE FAIL] Expected NetCDF missing: {expected}. "
                f"The M6b subscale L2 fit either failed convergence "
                f"(R-hat, ESS, divergences) or aborted before save_results. "
                f"Exiting 1 so SLURM captures the failure.",
                file=sys.stderr,
            )
            sys.exit(1)
        beta_count = 32  # 8 params x 4 covariates (subscale design)

    else:
        print(
            f"[FAIL] Internal model id '{model_internal}' has no L2 "
            f"dispatch branch. Expected one of: "
            f"copy={sorted(_COPY_BASELINE_MODELS)}, "
            f"2-cov={sorted(_TWO_COV_MODELS)}, "
            f"subscale={sorted(_SUBSCALE_MODELS)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step C: Final expected-file check (load-bearing) + summary.
    # ------------------------------------------------------------------
    final_expected = (
        output_root / "bayesian" / L2_SUBDIR / f"{model_internal}_posterior.nc"
    )
    if not final_expected.exists():
        print(
            f"[FAIL] Final expected NetCDF missing: {final_expected}. "
            f"See prior messages for the underlying cause (convergence, copy, "
            f"or delegate failure).",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\n" + "=" * 80)
    print("[STEP 21.6 COMPLETE]")
    print("=" * 80)
    print(f"  Output NetCDF: {final_expected}")
    print(f"  beta_* site count: {beta_count} "
          f"({'copy (no L2)' if beta_count == 0 else 'L2 refit'})")
    print(f"  Model: {model_internal} ({INTERNAL_TO_DISPLAY[model_internal]})")
    print("  Proceed to step 21.7.")
    sys.exit(0)


if __name__ == "__main__":
    main()
