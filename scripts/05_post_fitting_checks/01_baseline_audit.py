"""Step 21.4 — Convergence + PPC audit over the 6 baseline posteriors.

Phase 21 Wave 4 gatekeeper. Loads every NetCDF produced by step 21.3
(``scripts/04_model_fitting/b_bayesian/fit_baseline.py``) from
``output/bayesian/21_baseline/`` and
applies the Baribault & Collins (2023, DOI 10.1037/met0000554) convergence
gate before any model is allowed to participate in PSIS-LOO + stacking +
RFX-BMS/PXP in step 21.5.

Gate criteria (all must hold per model)
---------------------------------------
1. ``max R-hat <= 1.05`` — Gelman–Rubin shrink factor for all named
   parameters in the model's ``MODEL_REGISTRY[model]["params"]`` list.
2. ``min ESS_bulk >= 400`` — effective sample size for all named params.
3. ``n_divergences == 0`` — NUTS divergences from ``idata.sample_stats``
   (auto-bump in step 21.3 should already have resolved these; any
   residual indicates the posterior geometry defeated the sampler).
4. ``min BFMI >= 0.2`` — Bayesian Fraction of Missing Information per
   chain (:func:`arviz.bfmi` returns per-chain values; we take the worst).

If fewer than 2 models pass the gate, this script exits with code 1 so
SLURM ``--dependency=afterok`` chains (plan 21-10 master orchestrator)
block step 21.5 (stacking over a singleton model is not meaningful). This
is the direct implementation of Phase 21 ROADMAP success criterion #2:
"all 6 models converge OR are explicitly dropped with documented reason".

Posterior predictive audit
--------------------------
Reads ``{baseline_dir}/{model}_ppc_results.csv`` (written by the plan
21-04 Task 1 PPC-path fix) and summarises block-level 95% envelope
coverage. The CSV is **guaranteed** to exist for any model that passed
the step 21.3 ``save_results`` gate, because ``ppc_output_dir`` is now
threaded from ``save_results`` → ``run_posterior_predictive_check``. If
the CSV is missing despite the gate having passed, that is an upstream
bug: the audit surfaces it as ``ppc_coverage="WARNING_FILE_MISSING"`` and
adds the model to a dedicated WARNINGS section of the report rather than
silently swallowing the failure.

Outputs
-------
- ``{output_dir}/convergence_table.csv`` — machine-readable per-model
  metrics (max_rhat, min_ess_bulk, n_divergences, min_bfmi, ppc_coverage,
  gate_status, pipeline_action).
- ``{output_dir}/convergence_report.md`` — human-readable verdict with a
  top-level Summary section (n_passing, n_excluded, models_proceeding_to_loo)
  and per-model sections including metrics + pipeline action + WARNINGS.

Pipeline actions
----------------
- ``PROCEED_TO_LOO`` — model passes all 4 gate criteria.
- ``EXCLUDED_MISSING_FILE`` — ``{model}_posterior.nc`` not found.
- ``EXCLUDED_RHAT`` — max R-hat exceeds threshold.
- ``EXCLUDED_ESS`` — min ESS_bulk below threshold.
- ``EXCLUDED_DIVERGENCES`` — ``n_divergences > 0`` after auto-bump.
- ``EXCLUDED_BFMI`` — min BFMI below threshold.

Usage
-----
>>> python scripts/05_post_fitting_checks/01_baseline_audit.py
>>> python scripts/05_post_fitting_checks/01_baseline_audit.py \
...     --baseline-dir output/bayesian/21_baseline/ \
...     --output-dir output/bayesian/21_baseline/ \
...     --rhat-threshold 1.05 --ess-threshold 400 --bfmi-threshold 0.2

See also
--------
- ``cluster/21_4_baseline_audit.slurm`` — SLURM submission (30 min wall).
- ``.planning/phases/21-principled-bayesian-model-selection-pipeline/``
  21-05-PLAN.md for plan specification.

References
----------
Baribault, B., & Collins, A. G. E. (2023). Troubleshooting Bayesian
cognitive models. *Psychological Methods*. DOI 10.1037/met0000554.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

# -- Path bootstrap so this script runs both interactively and under SLURM.
# parents[2] = project root from 05_post_fitting_checks/
# (05_post_fitting_checks → scripts → <project root>).
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import MODEL_REGISTRY, load_netcdf_with_validation  # noqa: E402

# The 6 choice-only models produced by step 21.3. Must match
# ``scripts/04_model_fitting/b_bayesian/fit_baseline.py::BASELINE_MODELS``
# and ``MODEL_REGISTRY``.
MODELS: tuple[str, ...] = (
    "qlearning",
    "wmrl",
    "wmrl_m3",
    "wmrl_m5",
    "wmrl_m6a",
    "wmrl_m6b",
)

# Minimum number of models that must pass the convergence gate for step
# 21.5 (PSIS-LOO + stacking) to proceed. Stacking over a singleton is not
# meaningful; RFX-BMS requires >= 2 to compute protected exceedance.
MIN_MODELS_FOR_STACKING: int = 2


@dataclass
class ModelAudit:
    """Per-model audit record used for both CSV row and MD section.

    Attributes
    ----------
    model : str
        Model name (one of :data:`MODELS`).
    max_rhat : float
        Maximum Gelman-Rubin R-hat across named parameters. ``nan`` when
        the NetCDF could not be loaded.
    min_ess_bulk : float
        Minimum bulk ESS across named parameters. ``nan`` when not loaded.
    n_divergences : int
        Total NUTS divergences from ``idata.sample_stats.diverging``. 0 if
        the field is absent (conservative fallback).
    min_bfmi : float
        Minimum per-chain BFMI from :func:`arviz.bfmi`. ``nan`` when not
        computable.
    ppc_coverage : str
        Block-level 95% envelope coverage summary formatted as
        ``"{covered}/{total} ({pct:.1%})"``, or
        ``"WARNING_FILE_MISSING"`` if the PPC CSV is absent despite the
        plan 21-04 fix (indicates upstream bug — surfaced loudly).
    gate_status : str
        ``"PASS"`` or ``"FAIL"``.
    pipeline_action : str
        One of ``PROCEED_TO_LOO``, ``EXCLUDED_MISSING_FILE``,
        ``EXCLUDED_RHAT``, ``EXCLUDED_ESS``, ``EXCLUDED_DIVERGENCES``,
        ``EXCLUDED_BFMI``.
    notes : str
        Free-text explanation for the Markdown per-model section.
    ppc_file_missing : bool
        True iff ``{model}_ppc_results.csv`` is missing after the plan
        21-04 fix — used to flag the model in the WARNINGS section.
    """

    model: str
    max_rhat: float
    min_ess_bulk: float
    n_divergences: int
    min_bfmi: float
    ppc_coverage: str
    gate_status: str
    pipeline_action: str
    notes: str
    ppc_file_missing: bool


def _audit_one_model(
    model: str,
    baseline_dir: Path,
    rhat_threshold: float,
    ess_threshold: float,
    bfmi_threshold: float,
) -> ModelAudit:
    """Load one model's NetCDF and compute convergence + PPC summary.

    Parameters
    ----------
    model : str
        Model name (key into :data:`MODEL_REGISTRY`).
    baseline_dir : Path
        Directory containing ``{model}_posterior.nc`` and
        ``{model}_ppc_results.csv``.
    rhat_threshold, ess_threshold, bfmi_threshold : float
        Gate thresholds — see module docstring.

    Returns
    -------
    ModelAudit
        Per-model record ready for table/report writing.

    Notes
    -----
    Exclusion precedence when multiple criteria fail: R-hat > ESS >
    divergences > BFMI. This matches the order in which these problems
    typically surface during NUTS diagnosis (R-hat is the most universal
    non-convergence signal; BFMI is the most specific).
    """
    nc_path = baseline_dir / f"{model}_posterior.nc"
    ppc_path = baseline_dir / f"{model}_ppc_results.csv"

    if not nc_path.exists():
        return ModelAudit(
            model=model,
            max_rhat=float("nan"),
            min_ess_bulk=float("nan"),
            n_divergences=0,
            min_bfmi=float("nan"),
            ppc_coverage="not_available",
            gate_status="FAIL",
            pipeline_action="EXCLUDED_MISSING_FILE",
            notes=(
                f"Posterior NetCDF missing at {nc_path}. Step 21.3 fit "
                f"either hit the convergence gate inside save_results (which "
                f"returns None and writes nothing) or the SLURM job itself "
                f"failed before save_results was called. Check "
                f"logs/bayesian_21_3_*.{{out,err}} for the root cause."
            ),
            ppc_file_missing=False,  # expected-missing; not a bug
        )

    # ------------------------------------------------------------------
    # Load posterior + compute R-hat / ESS over named params only.
    # ------------------------------------------------------------------
    idata = load_netcdf_with_validation(nc_path, model)
    params = MODEL_REGISTRY[model]["params"]

    # Some samples may have theta_* wrappers; we use the raw parameter
    # names exactly as they appear in idata.posterior.  az.summary will
    # raise if any are missing — we catch that and annotate.
    try:
        summary = az.summary(idata, var_names=params)
    except KeyError as exc:
        return ModelAudit(
            model=model,
            max_rhat=float("nan"),
            min_ess_bulk=float("nan"),
            n_divergences=0,
            min_bfmi=float("nan"),
            ppc_coverage="not_available",
            gate_status="FAIL",
            pipeline_action="EXCLUDED_MISSING_FILE",
            notes=(
                f"NetCDF loaded but required parameter names missing: "
                f"{exc}. Expected params from MODEL_REGISTRY[{model!r}]: "
                f"{params}. Available in idata.posterior: "
                f"{list(idata.posterior.data_vars)}."
            ),
            ppc_file_missing=False,
        )

    max_rhat = float(summary["r_hat"].max())
    min_ess = float(summary["ess_bulk"].min())

    # ------------------------------------------------------------------
    # Divergences — fall back to 0 if the field is missing (non-MCMC
    # InferenceData). Auto-bump in step 21.3 should already have driven
    # divergences to zero; a non-zero count here means the sampler
    # exhausted target_accept=0.99 and still failed.
    # ------------------------------------------------------------------
    n_divergences: int = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        n_divergences = int(idata.sample_stats.diverging.values.sum())

    # ------------------------------------------------------------------
    # BFMI — per-chain energy vs. momentum check. az.bfmi returns an
    # array of per-chain values; use min as the worst chain.
    # ------------------------------------------------------------------
    try:
        bfmi_per_chain = np.asarray(az.bfmi(idata))
        min_bfmi = float(np.min(bfmi_per_chain)) if bfmi_per_chain.size else float("nan")
    except Exception as exc:  # noqa: BLE001 — ArviZ can raise misc errors
        min_bfmi = float("nan")
        print(
            f"[audit] {model}: az.bfmi() failed — {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # PPC summary. The 21-04 fix routes the CSV to the same subdir as the
    # NetCDF, so absence here is a bug signal.
    # ------------------------------------------------------------------
    ppc_file_missing = False
    if ppc_path.exists():
        ppc_df = pd.read_csv(ppc_path)
        if len(ppc_df) > 0 and "covered" in ppc_df.columns:
            covered = int(ppc_df["covered"].astype(bool).sum())
            total = int(len(ppc_df))
            pct = covered / total if total > 0 else 0.0
            ppc_coverage = f"{covered}/{total} ({pct:.1%})"
        else:
            ppc_coverage = "empty_or_malformed"
    else:
        ppc_coverage = "WARNING_FILE_MISSING"
        ppc_file_missing = True

    # ------------------------------------------------------------------
    # Apply the gate.
    # ------------------------------------------------------------------
    converged = (
        (max_rhat <= rhat_threshold)
        and (min_ess >= ess_threshold)
        and (n_divergences == 0)
        and (min_bfmi >= bfmi_threshold)
    )

    if converged:
        gate_status = "PASS"
        pipeline_action = "PROCEED_TO_LOO"
        notes = (
            f"All 4 gate criteria satisfied: max R-hat={max_rhat:.4f} <= "
            f"{rhat_threshold}, min ESS_bulk={min_ess:.0f} >= {ess_threshold}, "
            f"divergences={n_divergences}, min BFMI={min_bfmi:.3f} >= "
            f"{bfmi_threshold}. Eligible for PSIS-LOO + stacking in step 21.5."
        )
    else:
        gate_status = "FAIL"
        # Precedence: R-hat > ESS > divergences > BFMI.
        if not (max_rhat <= rhat_threshold):
            pipeline_action = "EXCLUDED_RHAT"
            notes = (
                f"R-hat gate FAILED: max R-hat={max_rhat:.4f} exceeds "
                f"threshold {rhat_threshold}. Chains have not mixed; the "
                f"posterior is not a reliable summary. (ESS_bulk={min_ess:.0f}, "
                f"divergences={n_divergences}, BFMI={min_bfmi:.3f})"
            )
        elif not (min_ess >= ess_threshold):
            pipeline_action = "EXCLUDED_ESS"
            notes = (
                f"ESS gate FAILED: min ESS_bulk={min_ess:.0f} below threshold "
                f"{ess_threshold}. Too few effective samples for reliable "
                f"quantile estimates. (R-hat={max_rhat:.4f}, "
                f"divergences={n_divergences}, BFMI={min_bfmi:.3f})"
            )
        elif n_divergences != 0:
            pipeline_action = "EXCLUDED_DIVERGENCES"
            notes = (
                f"Divergence gate FAILED: {n_divergences} divergent "
                f"transitions remain after auto-bump "
                f"(target_accept 0.80 -> 0.95 -> 0.99). Posterior geometry "
                f"defeated the sampler — reparameterise or tighten priors. "
                f"(R-hat={max_rhat:.4f}, ESS_bulk={min_ess:.0f}, "
                f"BFMI={min_bfmi:.3f})"
            )
        else:
            pipeline_action = "EXCLUDED_BFMI"
            notes = (
                f"BFMI gate FAILED: min BFMI={min_bfmi:.3f} below threshold "
                f"{bfmi_threshold}. Chain-level energy-momentum mismatch "
                f"indicates heavy-tailed posterior the NUTS step can't "
                f"traverse efficiently. (R-hat={max_rhat:.4f}, "
                f"ESS_bulk={min_ess:.0f}, divergences={n_divergences})"
            )

    return ModelAudit(
        model=model,
        max_rhat=max_rhat,
        min_ess_bulk=min_ess,
        n_divergences=n_divergences,
        min_bfmi=min_bfmi,
        ppc_coverage=ppc_coverage,
        gate_status=gate_status,
        pipeline_action=pipeline_action,
        notes=notes,
        ppc_file_missing=ppc_file_missing,
    )


def _write_convergence_table(
    audits: list[ModelAudit],
    out_csv: Path,
) -> None:
    """Write a machine-readable CSV of the audit (one row per model).

    Columns: ``model, max_rhat, min_ess_bulk, n_divergences, min_bfmi,
    ppc_coverage, gate_status, pipeline_action``.
    """
    rows = [
        {
            "model": a.model,
            "max_rhat": a.max_rhat,
            "min_ess_bulk": a.min_ess_bulk,
            "n_divergences": a.n_divergences,
            "min_bfmi": a.min_bfmi,
            "ppc_coverage": a.ppc_coverage,
            "gate_status": a.gate_status,
            "pipeline_action": a.pipeline_action,
        }
        for a in audits
    ]
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def _write_convergence_report(
    audits: list[ModelAudit],
    out_md: Path,
    rhat_threshold: float,
    ess_threshold: float,
    bfmi_threshold: float,
) -> None:
    """Write a human-readable Markdown report.

    Top of file contains a Summary block (n_passing, n_excluded,
    models_proceeding_to_loo) followed by per-model sections with the
    metrics and pipeline action. Any models with a missing PPC CSV are
    surfaced in a dedicated WARNINGS section so the upstream bug does not
    get buried.
    """
    passing = [a for a in audits if a.gate_status == "PASS"]
    excluded = [a for a in audits if a.gate_status == "FAIL"]
    ppc_warnings = [a for a in audits if a.ppc_file_missing]

    lines: list[str] = []
    lines.append("# Step 21.4 — Baseline Convergence Audit Report")
    lines.append("")
    lines.append(
        "Gate criteria (Baribault & Collins, 2023): "
        f"R-hat <= {rhat_threshold} AND ESS_bulk >= {ess_threshold} AND "
        f"divergences == 0 AND BFMI >= {bfmi_threshold}."
    )
    lines.append("")

    # Summary header.
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- n_passing: {len(passing)}")
    lines.append(f"- n_excluded: {len(excluded)}")
    lines.append(
        "- models_proceeding_to_loo: "
        + (", ".join(a.model for a in passing) if passing else "(none)")
    )
    lines.append(
        "- models_excluded: "
        + (", ".join(a.model for a in excluded) if excluded else "(none)")
    )
    lines.append("")

    # WARNINGS section — PPC file missing despite 21-04 fix.
    if ppc_warnings:
        lines.append("## WARNINGS")
        lines.append("")
        lines.append(
            "The following models converged but are missing their PPC results "
            "CSV. The plan 21-04 Task 1 fix routes `{model}_ppc_results.csv` "
            "to the same subdir as `{model}_posterior.nc`, so absence here "
            "indicates an upstream bug in `run_posterior_predictive_check` "
            "or `save_results` — investigate before proceeding:"
        )
        lines.append("")
        for a in ppc_warnings:
            lines.append(
                f"- `{a.model}`: expected "
                f"`{a.model}_ppc_results.csv` not found in baseline dir."
            )
        lines.append("")

    # Per-model sections.
    lines.append("## Per-model audit")
    lines.append("")
    for a in audits:
        lines.append(f"### {a.model}")
        lines.append("")
        lines.append(f"- **Gate status:** {a.gate_status}")
        lines.append(f"- **Pipeline action:** `{a.pipeline_action}`")
        lines.append(f"- max R-hat: {_fmt_float(a.max_rhat, 4)}")
        lines.append(f"- min ESS_bulk: {_fmt_float(a.min_ess_bulk, 0)}")
        lines.append(f"- divergences: {a.n_divergences}")
        lines.append(f"- min BFMI: {_fmt_float(a.min_bfmi, 3)}")
        lines.append(f"- PPC coverage: {a.ppc_coverage}")
        lines.append("")
        lines.append(f"**Notes:** {a.notes}")
        lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def _fmt_float(x: float, ndigits: int) -> str:
    """Format a float with ``ndigits`` decimals, or the literal ``'nan'``."""
    if np.isnan(x):
        return "nan"
    return f"{x:.{ndigits}f}"


def main() -> None:
    """CLI entry point.

    Parses arguments, runs the per-model audit, writes the CSV + MD
    artefacts, and exits with code 1 iff fewer than
    :data:`MIN_MODELS_FOR_STACKING` models pass the convergence gate.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Step 21.4 convergence + PPC audit over the 6 baseline "
            "posteriors. Enforces the Baribault & Collins (2023) "
            "convergence gate as a hard pipeline block between step 21.3 "
            "and step 21.5 (PSIS-LOO + stacking)."
        )
    )
    parser.add_argument(
        "--baseline-dir",
        default="output/bayesian/21_baseline/",
        help=(
            "Directory containing {model}_posterior.nc and "
            "{model}_ppc_results.csv from step 21.3 "
            "(default: output/bayesian/21_baseline/)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="output/bayesian/21_baseline/",
        help=(
            "Directory to write convergence_table.csv and "
            "convergence_report.md "
            "(default: output/bayesian/21_baseline/)."
        ),
    )
    parser.add_argument(
        "--rhat-threshold",
        type=float,
        default=1.05,
        help="Maximum allowed R-hat (default: 1.05).",
    )
    parser.add_argument(
        "--ess-threshold",
        type=float,
        default=400.0,
        help="Minimum allowed ESS_bulk (default: 400).",
    )
    parser.add_argument(
        "--bfmi-threshold",
        type=float,
        default=0.2,
        help="Minimum allowed BFMI (default: 0.2).",
    )
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output_dir)

    print("=" * 80)
    print("STEP 21.4 — BASELINE CONVERGENCE AUDIT")
    print("=" * 80)
    print(f"  Baseline dir: {baseline_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  R-hat threshold: <= {args.rhat_threshold}")
    print(f"  ESS_bulk threshold: >= {args.ess_threshold}")
    print(f"  BFMI threshold: >= {args.bfmi_threshold}")
    print(f"  Divergence threshold: == 0")
    print("=" * 80)

    audits: list[ModelAudit] = []
    for model in MODELS:
        print(f"\n[audit] {model} ...")
        audit = _audit_one_model(
            model,
            baseline_dir=baseline_dir,
            rhat_threshold=args.rhat_threshold,
            ess_threshold=args.ess_threshold,
            bfmi_threshold=args.bfmi_threshold,
        )
        audits.append(audit)
        print(
            f"[audit] {model}: {audit.gate_status} "
            f"(action={audit.pipeline_action}, R-hat="
            f"{_fmt_float(audit.max_rhat, 4)}, ESS="
            f"{_fmt_float(audit.min_ess_bulk, 0)}, div="
            f"{audit.n_divergences}, BFMI="
            f"{_fmt_float(audit.min_bfmi, 3)})"
        )

    # Write artefacts.
    out_csv = output_dir / "convergence_table.csv"
    out_md = output_dir / "convergence_report.md"
    _write_convergence_table(audits, out_csv)
    _write_convergence_report(
        audits,
        out_md,
        rhat_threshold=args.rhat_threshold,
        ess_threshold=args.ess_threshold,
        bfmi_threshold=args.bfmi_threshold,
    )
    print(f"\n[audit] Wrote {out_csv}")
    print(f"[audit] Wrote {out_md}")

    # Gate logic.
    n_passing = sum(1 for a in audits if a.gate_status == "PASS")
    n_excluded = sum(1 for a in audits if a.gate_status == "FAIL")
    missing = [a.model for a in audits if a.pipeline_action == "EXCLUDED_MISSING_FILE"]

    print("\n" + "=" * 80)
    print(f"  n_passing = {n_passing}")
    print(f"  n_excluded = {n_excluded}")
    if missing:
        print(
            f"  WARNING: missing NetCDFs (not counted as failures for gate "
            f"logic, but logged): {missing}"
        )
    print("=" * 80)

    if n_passing < MIN_MODELS_FOR_STACKING:
        print(
            f"\n[PIPELINE BLOCK] Only {n_passing} models passed convergence "
            f"gate; step 21.5 requires >= {MIN_MODELS_FOR_STACKING} "
            f"(stacking over a singleton is not meaningful). Fix the "
            f"excluded models and re-run step 21.3 before proceeding.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"\n[STEP 21.4 COMPLETE] {n_passing} models proceeding to LOO + "
        f"stacking: {[a.model for a in audits if a.gate_status == 'PASS']}"
    )


if __name__ == "__main__":
    main()
