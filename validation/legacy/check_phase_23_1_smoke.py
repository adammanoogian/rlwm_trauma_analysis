"""Phase 23.1 multi-GPU smoke-validation auditor.

Asserts every observable success criterion from Phase 23.1 Plan 23.1-01 via
deterministic file-system + arviz invariants:

* INV-01 — All 5 unproven choice-only models produced a smoke posterior.
* INV-02 — Convergence sanity at the *relaxed* gate ``max_rhat <= 1.10``
  (production gate is 1.05; smoke uses N=10 / 100 warmup, too short for tight
  convergence — see Plan 23.1-01 RESEARCH §3 Decision B).
* INV-03 — Lazy LBA float64 isolation: every posterior variable is float32.
  If any variable is float64, ``jax.config.update("jax_enable_x64", True)``
  leaked from an LBA import upstream of a choice-only fit
  (CLUSTER_GPU_LESSONS.md §5).
* INV-04 — Smoke artifacts contained: nothing in ``output/v1/`` or
  ``output/bayesian/21_baseline/`` was modified during the smoke run.
* INV-05 — Smoke-data preflight produced exactly 10 unique participants in
  ``_smoke_data_10ppts.csv`` (canonical column ``sona_id``).

CLI::

    python validation/check_phase_23_1_smoke.py
    python validation/check_phase_23_1_smoke.py --strict-rhat 1.05

Exit 0 = all invariants hold.
Exit 1 = structured diff naming the failed invariant(s).

Dependencies: stdlib + arviz + numpy + pandas + xarray. **NO jax import at
module top** (the audit must run on Windows where JAX may not be installed
correctly, and the lazy-LBA invariant is checked via posterior dtype only).

Determinism guarantee: two successive invocations with no intervening edits
produce byte-identical stdout. No ``datetime.now()``, no random seeds, all
glob results sorted, all model iteration ordered.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

# 5 unproven choice-only models on the multi-GPU pmap (Template C) path.
# wmrl_m6b is intentionally excluded (proven via job 54894258).
EXPECTED_SMOKE_MODELS: tuple[str, ...] = (
    "qlearning",
    "wmrl",
    "wmrl_m3",
    "wmrl_m5",
    "wmrl_m6a",
)

# Forbidden paths (relative to REPO_ROOT) — smoke run MUST NOT modify any
# NetCDF posterior here. INV-04 walks these globs and compares mtimes.
FORBIDDEN_PATH_GLOBS: tuple[str, ...] = (
    "output/v1/**/*.nc",
    "output/bayesian/21_baseline/**/*.nc",
)

# Smoke-data preflight CSV (produced by 23.1_mgpu_smoke.slurm). Canonical
# participant ID column is `sona_id` (verified in
# scripts/04_model_fitting/b_bayesian/fit_bayesian.py lines 127, 129, 140, 141, 148, 152, 153,
# 157, 180, 183).
SMOKE_DATA_FILENAME = "_smoke_data_10ppts.csv"
SMOKE_DATA_ID_COL = "sona_id"
SMOKE_DATA_EXPECTED_N = 10


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Result of a single Phase 23.1 smoke invariant check.

    Parameters
    ----------
    name : str
        Machine-readable invariant name (e.g. ``INV-01``).
    passed : bool
        True if the invariant holds.
    message : str
        One-line human-readable summary.
    details : list[str]
        Ordered list of diagnostic strings printed when ``passed`` is False
        (or always when ``--verbose`` is set).
    """

    name: str
    passed: bool
    message: str
    details: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _smoke_posterior_path(smoke_dir: Path, model: str) -> Path:
    """Return the expected smoke posterior NetCDF path for a model.

    Parameters
    ----------
    smoke_dir : Path
        Phase 23.1 smoke output dir (typically
        ``output/bayesian/23.1_smoke``).
    model : str
        Model identifier (e.g. ``qlearning``, ``wmrl_m5``).

    Returns
    -------
    Path
        Absolute path to ``{smoke_dir}/{model}_smoke_posterior.nc``.
    """
    return smoke_dir / f"{model}_smoke_posterior.nc"


def _existing_smoke_posteriors(
    smoke_dir: Path, models: tuple[str, ...]
) -> list[tuple[str, Path]]:
    """Return ``(model, path)`` tuples for smoke posteriors that exist on disk.

    Iterates ``models`` in input order (caller is responsible for sorting if
    they want deterministic order).

    Parameters
    ----------
    smoke_dir : Path
        Phase 23.1 smoke output dir.
    models : tuple[str, ...]
        Model identifiers to probe.

    Returns
    -------
    list[tuple[str, Path]]
        Existing ``(model, posterior_path)`` pairs in input order.
    """
    out: list[tuple[str, Path]] = []
    for model in models:
        path = _smoke_posterior_path(smoke_dir, model)
        if path.exists() and path.stat().st_size > 0:
            out.append((model, path))
    return out


# ---------------------------------------------------------------------------
# Invariant checks
# ---------------------------------------------------------------------------


def check_inv_01_posteriors_exist(
    smoke_dir: Path, required_models: tuple[str, ...]
) -> CheckResult:
    """INV-01: All required smoke posteriors exist and are non-empty.

    Asserts that for every model in ``required_models``, the file
    ``{smoke_dir}/{model}_smoke_posterior.nc`` exists and is > 0 bytes.

    M6b (``wmrl_m6b``) is excluded from ``required_models`` by default — if
    its smoke posterior also exists, that is permitted but not required.

    Parameters
    ----------
    smoke_dir : Path
        Phase 23.1 smoke output dir.
    required_models : tuple[str, ...]
        Models that MUST have a smoke posterior.

    Returns
    -------
    CheckResult
        Passed if every required posterior exists and is non-empty.
    """
    name = "INV-01"
    details: list[str] = []
    found = 0
    for model in required_models:
        path = _smoke_posterior_path(smoke_dir, model)
        if not path.exists():
            details.append(f"missing: {path.relative_to(REPO_ROOT)}")
        elif path.stat().st_size == 0:
            details.append(f"empty (0 bytes): {path.relative_to(REPO_ROOT)}")
        else:
            found += 1

    n_required = len(required_models)
    passed = len(details) == 0
    message = (
        f"PASS — {found}/{n_required} smoke posteriors exist"
        if passed
        else f"FAIL — {found}/{n_required} smoke posteriors exist"
    )
    return CheckResult(name=name, passed=passed, message=message, details=details)


def check_inv_02_convergence(
    smoke_dir: Path,
    required_models: tuple[str, ...],
    rhat_gate: float,
) -> CheckResult:
    """INV-02: Relaxed-gate convergence ``max_rhat <= rhat_gate`` per model.

    For each existing smoke posterior, compute ``arviz.rhat(idata)`` over
    ``idata.posterior``, take the maximum across all variables, and assert
    it is <= ``rhat_gate``.

    The default gate is 1.10 (relaxed from production 1.05) because smoke
    runs use N=10 / 100 warmup, which is too short for tight convergence.
    See Plan 23.1-01 RESEARCH §3 Decision B.

    Parameters
    ----------
    smoke_dir : Path
        Phase 23.1 smoke output dir.
    required_models : tuple[str, ...]
        Models to gate. Missing posteriors are skipped (INV-01 catches them).
    rhat_gate : float
        Maximum acceptable r-hat (default 1.10).

    Returns
    -------
    CheckResult
        Passed if every existing posterior has max_rhat <= ``rhat_gate``.
    """
    name = "INV-02"
    details: list[str] = []
    per_model_max: dict[str, float] = {}

    for model, path in _existing_smoke_posteriors(smoke_dir, required_models):
        try:
            idata = az.from_netcdf(str(path))
        except Exception as exc:
            details.append(f"{model}: arviz.from_netcdf failed — {exc}")
            continue

        try:
            rhat_ds = az.rhat(idata)
        except Exception as exc:
            details.append(f"{model}: az.rhat failed — {exc}")
            continue

        # rhat returns an xr.Dataset; flatten across all variables and take max
        try:
            arr = rhat_ds.to_array()
            max_rhat = float(np.nanmax(arr.values))
        except Exception as exc:
            details.append(f"{model}: r-hat aggregation failed — {exc}")
            continue

        per_model_max[model] = max_rhat
        if not np.isfinite(max_rhat):
            details.append(
                f"{model}: max_rhat is non-finite ({max_rhat}) — sampling failed"
            )
        elif max_rhat > rhat_gate:
            details.append(
                f"{model}: max_rhat={max_rhat:.4f} exceeds gate {rhat_gate:.2f}"
            )

    if per_model_max:
        global_max = max(per_model_max.values())
        n = len(per_model_max)
        summary = (
            f"max_rhat across {n} model(s) = {global_max:.4f} "
            f"(gate <= {rhat_gate:.2f})"
        )
    else:
        summary = (
            f"no smoke posteriors found to gate (gate <= {rhat_gate:.2f})"
        )

    # Always emit per-model values for transparency.
    for model in sorted(per_model_max):
        details.append(f"{model}: max_rhat={per_model_max[model]:.4f}")

    # Failure when any per-model gate violation OR when no posteriors at all.
    passed = (
        len(per_model_max) > 0
        and all(np.isfinite(v) and v <= rhat_gate for v in per_model_max.values())
    )
    message = f"PASS — {summary}" if passed else f"FAIL — {summary}"
    return CheckResult(name=name, passed=passed, message=message, details=details)


def check_inv_03_no_float64_leak(
    smoke_dir: Path, required_models: tuple[str, ...]
) -> CheckResult:
    """INV-03: Lazy-LBA float64 isolation holds across every smoke posterior.

    Walks ``idata.posterior`` for each existing smoke posterior and asserts
    every variable is dtype ``float32`` (not ``float64``). Per
    CLUSTER_GPU_LESSONS.md §5, ``jax.config.update("jax_enable_x64", True)``
    is process-global — if anywhere upstream of the choice-only fit triggered
    an LBA import, every numpy/JAX result would be float64 and the
    choice-only models would be silently miscalibrated.

    Parameters
    ----------
    smoke_dir : Path
        Phase 23.1 smoke output dir.
    required_models : tuple[str, ...]
        Models to check. Missing posteriors are skipped (INV-01 catches them).

    Returns
    -------
    CheckResult
        Passed if every variable in every existing posterior is float32.
    """
    name = "INV-03"
    details: list[str] = []
    n_checked = 0
    n_clean = 0

    for model, path in _existing_smoke_posteriors(smoke_dir, required_models):
        try:
            idata = az.from_netcdf(str(path))
        except Exception as exc:
            details.append(f"{model}: arviz.from_netcdf failed — {exc}")
            continue

        try:
            posterior = idata.posterior
        except AttributeError:
            details.append(f"{model}: InferenceData has no posterior group")
            continue

        n_checked += 1
        # Sort variables for deterministic output
        bad_vars = []
        for var_name in sorted(posterior.data_vars):
            dtype = posterior[var_name].dtype
            # Anything wider than 4 bytes float fails the invariant.
            if dtype == np.float64:
                bad_vars.append(f"{var_name}={dtype}")

        if bad_vars:
            details.append(
                f"{model}: float64 variables detected — "
                + ", ".join(bad_vars)
            )
        else:
            n_clean += 1

    if n_checked == 0:
        message = "FAIL — no smoke posteriors available to verify dtypes"
        return CheckResult(name=name, passed=False, message=message, details=details)

    passed = n_clean == n_checked
    message = (
        f"PASS — all variables float32 across {n_clean}/{n_checked} model(s) "
        f"(no LBA float64 leak)"
        if passed
        else f"FAIL — {n_clean}/{n_checked} model(s) float32-clean; "
        f"{n_checked - n_clean} model(s) leaked float64"
    )
    return CheckResult(name=name, passed=passed, message=message, details=details)


def check_inv_04_path_contract(
    smoke_dir: Path, required_models: tuple[str, ...]
) -> CheckResult:
    """INV-04: No NetCDF in forbidden production paths was modified by the smoke run.

    Computes ``t0 = min(mtime)`` of all existing smoke posteriors as the
    smoke-job time anchor. Then globs each pattern in ``FORBIDDEN_PATH_GLOBS``
    and asserts no matching file has ``mtime > t0`` — which would imply the
    smoke run wrote into a production path.

    If no smoke posteriors exist, this check is skipped with a PASS (INV-01
    already failed; this check has no anchor to compare against).

    Parameters
    ----------
    smoke_dir : Path
        Phase 23.1 smoke output dir.
    required_models : tuple[str, ...]
        Models whose smoke posteriors anchor ``t0``.

    Returns
    -------
    CheckResult
        Passed if no forbidden-path file was touched after ``t0``.
    """
    name = "INV-04"
    details: list[str] = []

    existing = _existing_smoke_posteriors(smoke_dir, required_models)
    if not existing:
        return CheckResult(
            name=name,
            passed=True,
            message="SKIP — no smoke posteriors yet; nothing to anchor mtime against",
            details=["INV-01 already failed; INV-04 has no time anchor"],
        )

    mtimes = [p.stat().st_mtime for _, p in existing]
    t0 = min(mtimes)

    n_violations = 0
    for pattern in FORBIDDEN_PATH_GLOBS:
        # Sort glob results for deterministic output
        for path in sorted(REPO_ROOT.glob(pattern)):
            try:
                pmtime = path.stat().st_mtime
            except OSError:
                continue
            if pmtime > t0:
                rel = path.relative_to(REPO_ROOT)
                details.append(
                    f"forbidden-path file modified after smoke t0: {rel} "
                    f"(mtime={pmtime:.0f} > t0={t0:.0f})"
                )
                n_violations += 1

    passed = n_violations == 0
    message = (
        f"PASS — no files modified in {len(FORBIDDEN_PATH_GLOBS)} forbidden path(s) "
        f"after smoke t0"
        if passed
        else f"FAIL — {n_violations} file(s) in forbidden paths modified after smoke t0"
    )
    return CheckResult(name=name, passed=passed, message=message, details=details)


def check_inv_05_smoke_data_ppts(smoke_dir: Path) -> CheckResult:
    """INV-05: Smoke-data preflight CSV exists with exactly 10 unique participants.

    The SLURM script materializes ``{smoke_dir}/_smoke_data_10ppts.csv`` by
    subsetting the first 10 unique ``sona_id`` values from the canonical
    ``output/task_trials_long.csv``. This check asserts the artifact exists
    and has exactly 10 unique IDs in the canonical column ``sona_id``.

    Parameters
    ----------
    smoke_dir : Path
        Phase 23.1 smoke output dir.

    Returns
    -------
    CheckResult
        Passed if CSV exists with exactly 10 unique ``sona_id`` values.
    """
    name = "INV-05"
    details: list[str] = []
    csv_path = smoke_dir / SMOKE_DATA_FILENAME

    if not csv_path.exists():
        return CheckResult(
            name=name,
            passed=False,
            message=f"FAIL — smoke-data CSV not found: "
            f"{csv_path.relative_to(REPO_ROOT)}",
            details=[
                "Expected the SLURM preflight to write this file. Did the "
                "smoke job run on Monash?"
            ],
        )

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return CheckResult(
            name=name,
            passed=False,
            message=f"FAIL — pandas could not parse smoke-data CSV: {exc}",
            details=[f"path: {csv_path.relative_to(REPO_ROOT)}"],
        )

    if SMOKE_DATA_ID_COL not in df.columns:
        details.append(
            f"column '{SMOKE_DATA_ID_COL}' missing from smoke-data CSV; "
            f"present columns: {sorted(df.columns)}"
        )
        return CheckResult(
            name=name,
            passed=False,
            message=f"FAIL — canonical column '{SMOKE_DATA_ID_COL}' missing",
            details=details,
        )

    n_unique = int(df[SMOKE_DATA_ID_COL].nunique())
    if n_unique != SMOKE_DATA_EXPECTED_N:
        details.append(
            f"expected {SMOKE_DATA_EXPECTED_N} unique {SMOKE_DATA_ID_COL}, "
            f"found {n_unique}"
        )
        return CheckResult(
            name=name,
            passed=False,
            message=(
                f"FAIL — smoke-data CSV has {n_unique} unique participants; "
                f"expected {SMOKE_DATA_EXPECTED_N}"
            ),
            details=details,
        )

    return CheckResult(
        name=name,
        passed=True,
        message=(
            f"PASS — smoke-data CSV has exactly {SMOKE_DATA_EXPECTED_N} "
            f"unique {SMOKE_DATA_ID_COL} values"
        ),
        details=[f"path: {csv_path.relative_to(REPO_ROOT)}, rows={len(df)}"],
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def check_all(
    smoke_dir: Path,
    required_models: tuple[str, ...],
    rhat_gate: float,
) -> tuple[int, list[CheckResult]]:
    """Run every Phase 23.1 invariant in fixed order. Return (exit_code, results).

    Deterministic: invariants run in INV-01..INV-05 order. Two successive
    calls with no intervening edits produce byte-identical output.

    Parameters
    ----------
    smoke_dir : Path
        Phase 23.1 smoke output dir.
    required_models : tuple[str, ...]
        Models that MUST produce a smoke posterior (INV-01) and pass
        convergence + dtype gates (INV-02, INV-03).
    rhat_gate : float
        Relaxed convergence gate (default 1.10).

    Returns
    -------
    tuple[int, list[CheckResult]]
        ``(exit_code, results)`` — exit_code 0 if all invariants pass, else 1.
    """
    results: list[CheckResult] = [
        check_inv_01_posteriors_exist(smoke_dir, required_models),
        check_inv_02_convergence(smoke_dir, required_models, rhat_gate),
        check_inv_03_no_float64_leak(smoke_dir, required_models),
        check_inv_04_path_contract(smoke_dir, required_models),
        check_inv_05_smoke_data_ppts(smoke_dir),
    ]
    exit_code = 0 if all(r.passed for r in results) else 1
    return exit_code, results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the Phase 23.1 smoke-validation auditor.

    Parameters
    ----------
    argv : list[str] | None
        Argument list (defaults to ``sys.argv[1:]`` when None).

    Returns
    -------
    int
        Exit code: 0 if all invariants hold, 1 if any fail.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Phase 23.1 multi-GPU smoke-validation auditor. "
            "Exit 0 = all 5 invariants hold. Exit 1 = structured diff "
            "naming the failed invariant(s)."
        )
    )
    parser.add_argument(
        "--smoke-dir",
        default="output/bayesian/23.1_smoke",
        help=(
            "Phase 23.1 smoke output directory "
            "(default: output/bayesian/23.1_smoke)"
        ),
    )
    parser.add_argument(
        "--strict-rhat",
        type=float,
        default=1.10,
        help=(
            "Maximum acceptable r-hat for INV-02 (default: 1.10 — relaxed "
            "from production 1.05; smoke uses N=10/100-warmup which is too "
            "short for tight convergence)"
        ),
    )
    parser.add_argument(
        "--require-models",
        nargs="+",
        default=list(EXPECTED_SMOKE_MODELS),
        help=(
            "Models that MUST produce a smoke posterior. Default: "
            f"{list(EXPECTED_SMOKE_MODELS)} (M6b is intentionally excluded; "
            "already proven via job 54894258)"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-invariant details even for passing checks",
    )

    args = parser.parse_args(argv)

    smoke_dir_path = Path(args.smoke_dir)
    if not smoke_dir_path.is_absolute():
        smoke_dir_path = REPO_ROOT / smoke_dir_path

    # Sort and dedupe required models for determinism (input order is also
    # honored within the per-check helpers, but sorting at this boundary
    # makes the CLI byte-identical for any equivalent --require-models set).
    required_models = tuple(sorted(set(args.require_models)))

    exit_code, results = check_all(
        smoke_dir=smoke_dir_path,
        required_models=required_models,
        rhat_gate=args.strict_rhat,
    )

    # Structured report — fixed header, fixed invariant order.
    print("=" * 70)
    print("Phase 23.1 Multi-GPU Smoke Validation")
    print("=" * 70)
    for result in results:
        status_tag = "PASS" if result.passed else "FAIL"
        print(f"{result.name} [{status_tag}] {result.message}")
        if not result.passed or args.verbose:
            for detail in result.details:
                print(f"       {detail}")
    print("=" * 70)
    n_pass = sum(1 for r in results if r.passed)
    n_fail = len(results) - n_pass
    if n_fail == 0:
        print("ALL INVARIANTS PASS")
    else:
        print(f"FAILED: {n_fail}/{len(results)} invariants did not pass")
    print(f"EXIT {exit_code}")
    print("=" * 70)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
