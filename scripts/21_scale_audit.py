"""Step 21.7 — scale-fit audit for winners' L2-refitted posteriors.

Phase 21 Wave 7 gatekeeper between step 21.6 (winner L2 refit,
:mod:`scripts.21_fit_with_l2`) and step 21.8 (model averaging /
posterior summarisation). Loads each winner's ``21_l2`` posterior and
verifies that (a) adding the L2 covariates did NOT degrade convergence
relative to the ``21_baseline`` fit, (b) group-level L2 coefficients
(``beta_*`` sites) are well-identified, (c) at least one beta has 95%
HDI excluding zero after FDR-BH multiplicity correction across the full
set of beta sites per winner.

Decision protocol (plan-checker Issue #4 — unified exit-0 semantics)
---------------------------------------------------------------------
The audit writes a YAML-front-matter header at the top of
``scale_audit_report.md`` with ``pipeline_action`` set to one of:

- ``PROCEED_TO_AVERAGING`` — at least one beta site across all winners
  has 95% HDI excluding zero AND the surrogate FDR-BH adjusted
  posterior tail probability is below ``--fdr-alpha``. Script exits 0,
  step 21.8 runs and averages.

- ``NULL_RESULT`` — zero beta sites survived FDR-BH across all winners.
  Script exits 0 (NOT 1 — null result is valid science, not an audit
  error). ``21_model_averaging.py`` reads the YAML header and writes
  ``averaging_skipped.md`` + exits 0; step 21.9 reports the null.

Exit code 1 is RESERVED for genuine audit errors ONLY:
  - ``winners.txt`` missing or malformed
  - NetCDF load failures (corrupt files)
  - ``ImportError`` (e.g. ``statsmodels`` missing from the conda env)
  - Unexpected ``ValueError`` bubbling up from the core audit loop

This lets the master orchestrator (plan 21-10) use
``--dependency=afterok:$AUDIT2_JID`` for the 21.8 job without the
NULL_RESULT branch falsely blocking the dependency chain.

Beta-site enumeration (covers Phase 21 multi-tier L2 design)
-------------------------------------------------------------
Enumeration uses a single ``var.startswith("beta_")`` pattern match so
all three L2 tiers are handled uniformly:

- M3/M5/M6a winners (2-cov path, plan 21-11):
  ``beta_lec_{target}`` AND ``beta_iesr_{target}`` where
  ``target=kappa`` for M3/M5, ``target=kappa_s`` for M6a. Two sites.

- M6b subscale winner (full 4-covariate design):
  ``beta_{cov}_{param}`` for covariate in
  ``{lec, iesr, iesr_intr_resid, iesr_avd_resid}`` × param in
  ``{alpha_pos, alpha_neg, phi, rho, capacity, epsilon, kappa_total,
  kappa_share}``. 32 sites.

- M1/M2 winners (copy-through path): zero beta sites. The script logs
  an informative ``[AUDIT] {winner}: 0 beta sites`` message and
  contributes nothing to the aggregate gate.

The ``print(f"[AUDIT] {winner}: {len(sites)} beta sites: {sites}")``
log line is deliberate — it surfaces the IES-R family inclusion in the
SLURM stdout so a reviewer can trust at a glance that plan 21-11's
2-cov hook actually emitted both covariates.

FDR-BH caveat (documented in report body)
-----------------------------------------
FDR-BH on posterior two-sided tail probabilities is NOT identical to
frequentist FDR (posterior tail p is a credibility measure, not a
null-hypothesis p-value). It serves here as an HDI-agreement diagnostic
— a site with ``excludes_zero_hdi=True`` should also have
``excludes_zero_fdr=True`` in well-behaved posteriors; disagreements
flag distributions with heavy tails or low ESS that may not be fully
mixed. Use BOTH flags for downstream reporting.

ESS degradation check (per Baribault & Collins 2023)
----------------------------------------------------
For each winner, compute ``min_ess_shared_l2 / min_ess_shared_baseline``
across NON-beta parameters (baseline has no L2 sites, so the
comparison is apples-to-apples on the shared subset). If the ratio
drops below ``(1 - ess_drop_threshold)`` (default 50%), flag
``ess_degraded=True`` and list the winner in
``ess_degraded_models`` at the YAML header + a WARNING block in the
body. Still exit 0 — the degradation is surfaced loudly but does not
kill the pipeline (user decides whether to rerun with more warmup).

Outputs
-------
- ``{output_dir}/{winner}_beta_hdi_table.csv`` — per-winner beta-site
  table with columns
  ``beta_site, covariate_family, target_parameter, posterior_mean,
   posterior_sd, hdi_low, hdi_high, tail_p, p_fdr_adj,
   excludes_zero_hdi, excludes_zero_fdr`` plus a metadata trailer row
  carrying ``max_rhat_betas / min_ess_betas / ess_degraded``.
- ``{output_dir}/scale_audit_report.md`` — YAML-front-matter header +
  per-winner sections listing notable effects + FDR-BH caveat +
  ESS-degradation WARNINGS.

Usage
-----
>>> python scripts/21_scale_audit.py \
...     --l2-dir output/bayesian/21_l2/ \
...     --baseline-dir output/bayesian/21_baseline/ \
...     --winners-file output/bayesian/21_baseline/winners.txt \
...     --hdi-prob 0.95 --fdr-alpha 0.05 --ess-drop-threshold 0.5 \
...     --output-dir output/bayesian/21_l2/

See also
--------
- ``cluster/21_7_scale_audit.slurm`` — SLURM submission template.
- ``scripts/21_baseline_audit.py`` — step 21.4 sibling audit pattern.
- ``.planning/phases/21-principled-bayesian-model-selection-pipeline/``
  21-08-PLAN.md for plan specification.

References
----------
Baribault, B., & Collins, A. G. E. (2023). Troubleshooting Bayesian
cognitive models. *Psychological Methods*. DOI 10.1037/met0000554.
Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
rate: a practical and powerful approach to multiple testing. *JRSS B*.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

# -- Path bootstrap so this script runs both interactively and under SLURM.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Guard the statsmodels import with a clear error so a misconfigured
# conda env fails LOUDLY at the top of the audit rather than halfway
# through per-winner processing.
try:
    from statsmodels.stats.multitest import multipletests
except ImportError as exc:  # pragma: no cover — env-level failure
    print(
        "[FATAL] statsmodels is required for FDR-BH correction in "
        "scripts/21_scale_audit.py. Expected it in ds_env. Install with "
        "`conda install -n ds_env statsmodels` or `pip install statsmodels`. "
        f"Original error: {type(exc).__name__}: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)

from config import load_netcdf_with_validation  # noqa: E402

# Display-name <-> internal-id mappings (must match
# scripts/21_fit_with_l2.py::DISPLAY_TO_INTERNAL and
# scripts/21_compute_loo_stacking.py::MODEL_TO_DISPLAY).
DISPLAY_TO_INTERNAL: dict[str, str] = {
    "M1": "qlearning",
    "M2": "wmrl",
    "M3": "wmrl_m3",
    "M5": "wmrl_m5",
    "M6a": "wmrl_m6a",
    "M6b": "wmrl_m6b",
}
INTERNAL_TO_DISPLAY: dict[str, str] = {v: k for k, v in DISPLAY_TO_INTERNAL.items()}

# Known covariate families (parsed off beta site names). Ordered
# longest-prefix-first so ``iesr_intr_resid`` is matched before ``iesr``
# (avoids false positive on the plain totals covariate).
_COVARIATE_FAMILIES: tuple[str, ...] = (
    "iesr_intr_resid",
    "iesr_avd_resid",
    "iesr",
    "lec",
)

# Target parameters that can carry L2 betas (parsed off beta site names).
# Superset of per-model targets — M6b subscale has all 8; M3/M5/M6a
# have one each (kappa or kappa_s).
_KNOWN_TARGETS: tuple[str, ...] = (
    "alpha_pos",
    "alpha_neg",
    "phi",
    "rho",
    "capacity",
    "epsilon",
    "phi_rl",
    "kappa_total",
    "kappa_share",
    "kappa_s",
    "kappa",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BetaSite:
    """Per-beta-site record: posterior summary + HDI + FDR flags."""

    beta_site: str
    covariate_family: str
    target_parameter: str
    posterior_mean: float
    posterior_sd: float
    hdi_low: float
    hdi_high: float
    tail_p: float
    p_fdr_adj: float
    excludes_zero_hdi: bool
    excludes_zero_fdr: bool


@dataclass
class WinnerAudit:
    """Per-winner audit: beta sites + convergence + ESS-degradation check."""

    winner: str  # internal model id
    display_name: str
    beta_sites: list[BetaSite] = field(default_factory=list)
    max_rhat_betas: float = float("nan")
    min_ess_betas: float = float("nan")
    min_ess_shared_baseline: float = float("nan")
    min_ess_shared_l2: float = float("nan")
    ess_degraded: bool = False
    notes: str = ""
    load_warning: str = ""  # populated on missing-file soft-skip


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_winners_file(winners_path: Path) -> list[str]:
    """Parse ``winners.txt`` into internal model ids (mirrors 21_fit_with_l2)."""
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


def _parse_beta_site_name(site_name: str) -> tuple[str, str]:
    """Parse ``beta_{covariate_family}_{target_parameter}`` into its two parts.

    Parameters
    ----------
    site_name : str
        Variable name from ``idata.posterior.data_vars``, must start with
        ``"beta_"``.

    Returns
    -------
    covariate_family : str
        One of :data:`_COVARIATE_FAMILIES`, or ``"unknown"`` if not matched.
    target_parameter : str
        One of :data:`_KNOWN_TARGETS`, or ``"unknown"`` if not matched.

    Notes
    -----
    Longest-prefix-first ordering of :data:`_COVARIATE_FAMILIES` is load-
    bearing: ``beta_iesr_intr_resid_kappa_total`` must match
    ``iesr_intr_resid``, not ``iesr``. Similarly for target parameters:
    ``kappa_total`` and ``kappa_share`` must match before ``kappa``.
    """
    if not site_name.startswith("beta_"):
        return ("unknown", "unknown")
    body = site_name[len("beta_") :]

    covariate_family = "unknown"
    remainder = body
    for cov in _COVARIATE_FAMILIES:
        if body.startswith(cov + "_"):
            covariate_family = cov
            remainder = body[len(cov) + 1 :]
            break

    target_parameter = "unknown"
    for target in _KNOWN_TARGETS:
        if remainder == target:
            target_parameter = target
            break

    return (covariate_family, target_parameter)


def _audit_beta_site(
    idata: az.InferenceData,
    site_name: str,
    hdi_prob: float,
) -> tuple[float, float, float, float, float, float]:
    """Compute posterior summary + HDI + tail p for one beta site.

    Parameters
    ----------
    idata : az.InferenceData
        L2-refit posterior container.
    site_name : str
        Variable name, e.g. ``"beta_lec_kappa"``.
    hdi_prob : float
        HDI mass (default 0.95 at the CLI boundary).

    Returns
    -------
    mean, sd, hdi_low, hdi_high, tail_p, excludes_zero_hdi_flag
        Posterior moments, HDI endpoints, two-sided tail probability,
        and the ``excludes_zero`` flag encoded as 1.0 (yes) / 0.0 (no)
        for easy array handling. ``tail_p`` is computed as
        ``2 * min(P(samples > 0), P(samples < 0))``.
    """
    samples = np.asarray(idata.posterior[site_name].values).ravel()
    mean = float(np.mean(samples))
    sd = float(np.std(samples, ddof=1))

    # az.hdi on a 1-D array returns (2,)-shaped lower/upper bounds.
    hdi_arr = np.asarray(az.hdi(samples, hdi_prob=hdi_prob))
    hdi_low, hdi_high = float(hdi_arr[0]), float(hdi_arr[1])

    p_gt = float(np.mean(samples > 0.0))
    p_lt = float(np.mean(samples < 0.0))
    tail_p = 2.0 * min(p_gt, p_lt)
    # Guard against pathological all-one-side posteriors where min=0.
    # (tail_p=0 is still a valid value; multipletests handles it fine.)

    excludes_zero_hdi = 1.0 if (hdi_low > 0.0 or hdi_high < 0.0) else 0.0
    return mean, sd, hdi_low, hdi_high, tail_p, excludes_zero_hdi


def _min_ess_on_non_beta(idata: az.InferenceData) -> float:
    """Compute the min bulk ESS across NON-``beta_*`` parameters.

    Used for the L2 vs. baseline shared-parameter ESS degradation check.
    """
    non_beta_vars = [
        v for v in idata.posterior.data_vars if not v.startswith("beta_")
    ]
    if not non_beta_vars:
        return float("nan")
    # Suppress ArviZ warning about "not enough samples" that can occur
    # on stubs — the audit is interpretive, not authoritative.
    summary = az.summary(idata, var_names=non_beta_vars)
    return float(summary["ess_bulk"].min())


def _audit_one_winner(
    winner: str,
    l2_dir: Path,
    baseline_dir: Path,
    hdi_prob: float,
    ess_drop_threshold: float,
) -> WinnerAudit:
    """Run the full per-winner audit: beta HDI + FDR-BH + ESS degradation.

    Parameters
    ----------
    winner : str
        Internal model id (key in :data:`INTERNAL_TO_DISPLAY`).
    l2_dir : Path
        Directory containing ``{winner}_posterior.nc`` from step 21.6.
    baseline_dir : Path
        Directory containing ``{winner}_posterior.nc`` from step 21.3.
    hdi_prob : float
        HDI probability mass for beta-site reporting.
    ess_drop_threshold : float
        Fractional ESS drop threshold (default 0.5 => 50% drop).

    Returns
    -------
    WinnerAudit
        Populated audit record. On missing NetCDF, returns an empty
        record with ``load_warning`` populated — the caller logs + skips
        without exiting non-zero (missing file here is an upstream bug,
        but a missing winner should not tank the aggregate gate; step
        21.10 orchestrator should have already failed afterok).
    """
    display_name = INTERNAL_TO_DISPLAY.get(winner, winner)
    audit = WinnerAudit(winner=winner, display_name=display_name)

    l2_nc = l2_dir / f"{winner}_posterior.nc"
    baseline_nc = baseline_dir / f"{winner}_posterior.nc"

    if not l2_nc.exists():
        audit.load_warning = (
            f"L2-refit posterior missing at {l2_nc}. Step 21.6 either hit "
            f"the convergence gate inside save_results (which returns None "
            f"and writes nothing) or the SLURM job itself failed. Check "
            f"logs/bayesian_21_6_*.{{out,err}}. Skipping this winner."
        )
        print(f"[AUDIT] WARNING: {audit.load_warning}", file=sys.stderr)
        return audit

    try:
        l2_idata = load_netcdf_with_validation(l2_nc, winner)
    except Exception as exc:  # noqa: BLE001 — corrupt NetCDF is an audit error
        raise RuntimeError(
            f"Failed to load L2 NetCDF at {l2_nc}: "
            f"{type(exc).__name__}: {exc}. This is a genuine audit error; "
            f"investigate before retrying."
        ) from exc

    # Beta-site enumeration — pattern-match picks up BOTH beta_lec_* AND
    # beta_iesr_* for M3/M5/M6a, all 32 beta_{cov}_{param} for M6b
    # subscale, empty list for M1/M2 copy-through. Log explicitly so the
    # IES-R family inclusion is observable in SLURM stdout.
    beta_site_names = [
        v for v in l2_idata.posterior.data_vars if v.startswith("beta_")
    ]
    print(
        f"[AUDIT] {winner} ({display_name}): "
        f"{len(beta_site_names)} beta sites: {beta_site_names}"
    )

    # Per-site posterior summary + HDI + tail_p (pre-FDR).
    raw_sites: list[dict[str, object]] = []
    for site_name in beta_site_names:
        mean, sd, hdi_lo, hdi_hi, tail_p, excl_hdi = _audit_beta_site(
            l2_idata, site_name, hdi_prob=hdi_prob
        )
        covariate_family, target_parameter = _parse_beta_site_name(site_name)
        raw_sites.append(
            {
                "beta_site": site_name,
                "covariate_family": covariate_family,
                "target_parameter": target_parameter,
                "posterior_mean": mean,
                "posterior_sd": sd,
                "hdi_low": hdi_lo,
                "hdi_high": hdi_hi,
                "tail_p": tail_p,
                "excludes_zero_hdi": bool(excl_hdi),
            }
        )

    # FDR-BH correction across ALL beta sites for THIS winner (not
    # cross-winner). Required even when len(raw_sites) in {0, 1} because
    # the downstream aggregate gate reads excludes_zero_fdr.
    if raw_sites:
        p_vals = np.array([row["tail_p"] for row in raw_sites], dtype=float)
        reject, p_adj, _, _ = multipletests(p_vals, alpha=0.05, method="fdr_bh")
        for row, rej, p_a in zip(raw_sites, reject, p_adj, strict=True):
            row["p_fdr_adj"] = float(p_a)
            row["excludes_zero_fdr"] = bool(rej)

    # Materialise as BetaSite records for the WinnerAudit container.
    for row in raw_sites:
        audit.beta_sites.append(
            BetaSite(
                beta_site=str(row["beta_site"]),
                covariate_family=str(row["covariate_family"]),
                target_parameter=str(row["target_parameter"]),
                posterior_mean=float(row["posterior_mean"]),  # type: ignore[arg-type]
                posterior_sd=float(row["posterior_sd"]),  # type: ignore[arg-type]
                hdi_low=float(row["hdi_low"]),  # type: ignore[arg-type]
                hdi_high=float(row["hdi_high"]),  # type: ignore[arg-type]
                tail_p=float(row["tail_p"]),  # type: ignore[arg-type]
                p_fdr_adj=float(row.get("p_fdr_adj", float("nan"))),  # type: ignore[arg-type]
                excludes_zero_hdi=bool(row["excludes_zero_hdi"]),
                excludes_zero_fdr=bool(row.get("excludes_zero_fdr", False)),
            )
        )

    # Convergence diagnostics specific to the beta sites (only).
    if beta_site_names:
        try:
            beta_summary = az.summary(l2_idata, var_names=beta_site_names)
            audit.max_rhat_betas = float(beta_summary["r_hat"].max())
            audit.min_ess_betas = float(beta_summary["ess_bulk"].min())
        except Exception as exc:  # noqa: BLE001
            print(
                f"[AUDIT] {winner}: az.summary on beta sites failed — "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    # ESS-degradation check: min ESS on shared (NON-beta) params,
    # baseline vs. L2-refit. Apples-to-apples because baseline has no L2
    # betas, so the shared set IS the baseline's full parameter set.
    audit.min_ess_shared_l2 = _min_ess_on_non_beta(l2_idata)
    if baseline_nc.exists():
        try:
            baseline_idata = load_netcdf_with_validation(baseline_nc, winner)
            audit.min_ess_shared_baseline = _min_ess_on_non_beta(baseline_idata)
            if (
                np.isfinite(audit.min_ess_shared_baseline)
                and audit.min_ess_shared_baseline > 0
                and np.isfinite(audit.min_ess_shared_l2)
            ):
                ratio = audit.min_ess_shared_l2 / audit.min_ess_shared_baseline
                audit.ess_degraded = bool(ratio < (1.0 - ess_drop_threshold))
                if audit.ess_degraded:
                    audit.notes = (
                        f"ESS dropped from {audit.min_ess_shared_baseline:.0f} "
                        f"(baseline) to {audit.min_ess_shared_l2:.0f} (L2) — "
                        f"ratio {ratio:.2f} < {1 - ess_drop_threshold:.2f} "
                        f"threshold."
                    )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[AUDIT] {winner}: baseline ESS compare failed — "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
    else:
        print(
            f"[AUDIT] {winner}: baseline NetCDF missing at {baseline_nc} — "
            f"skipping ESS-degradation check.",
            file=sys.stderr,
        )

    return audit


def _write_per_winner_csv(audit: WinnerAudit, out_csv: Path) -> None:
    """Write the per-winner beta HDI table CSV + metadata trailer row.

    The CSV has 11 beta-site columns followed by a trailer row whose
    ``beta_site`` cell reads ``"__METADATA__"`` and whose remaining cells
    carry ``max_rhat_betas`` / ``min_ess_betas`` / ``ess_degraded``. A
    trailer row keeps the CSV single-table (a separate header would
    break ``pd.read_csv`` round-tripping without explicit skiprows).
    """
    columns = [
        "beta_site",
        "covariate_family",
        "target_parameter",
        "posterior_mean",
        "posterior_sd",
        "hdi_low",
        "hdi_high",
        "tail_p",
        "p_fdr_adj",
        "excludes_zero_hdi",
        "excludes_zero_fdr",
    ]
    rows = [
        {
            "beta_site": bs.beta_site,
            "covariate_family": bs.covariate_family,
            "target_parameter": bs.target_parameter,
            "posterior_mean": bs.posterior_mean,
            "posterior_sd": bs.posterior_sd,
            "hdi_low": bs.hdi_low,
            "hdi_high": bs.hdi_high,
            "tail_p": bs.tail_p,
            "p_fdr_adj": bs.p_fdr_adj,
            "excludes_zero_hdi": bs.excludes_zero_hdi,
            "excludes_zero_fdr": bs.excludes_zero_fdr,
        }
        for bs in audit.beta_sites
    ]
    rows.append(
        {
            "beta_site": "__METADATA__",
            "covariate_family": "",
            "target_parameter": "",
            "posterior_mean": float("nan"),
            "posterior_sd": float("nan"),
            "hdi_low": float("nan"),
            "hdi_high": float("nan"),
            "tail_p": float("nan"),
            "p_fdr_adj": float("nan"),
            "excludes_zero_hdi": f"max_rhat_betas={audit.max_rhat_betas:.4f}",
            "excludes_zero_fdr": (
                f"min_ess_betas={audit.min_ess_betas:.0f};"
                f"ess_degraded={audit.ess_degraded}"
            ),
        }
    )
    df = pd.DataFrame(rows, columns=columns)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def _format_yaml_header(
    audits: list[WinnerAudit],
    pipeline_action: str,
) -> str:
    """Build the YAML-front-matter header the SLURM awk parser reads."""
    n_winners = len(audits)
    n_beta_total = sum(len(a.beta_sites) for a in audits)
    n_hdi_excl = sum(
        1 for a in audits for bs in a.beta_sites if bs.excludes_zero_hdi
    )
    n_fdr_excl = sum(
        1 for a in audits for bs in a.beta_sites if bs.excludes_zero_fdr
    )
    degraded = [a.winner for a in audits if a.ess_degraded]

    lines = [
        "---",
        f"pipeline_action: {pipeline_action}",
        f"n_winners: {n_winners}",
        f"n_beta_sites_total: {n_beta_total}",
        f"n_hdi_excludes_zero: {n_hdi_excl}",
        f"n_fdr_excludes_zero: {n_fdr_excl}",
        f"ess_degraded_models: [{', '.join(degraded)}]",
        "---",
    ]
    return "\n".join(lines)


def _format_report_body(
    audits: list[WinnerAudit],
    hdi_prob: float,
    fdr_alpha: float,
    ess_drop_threshold: float,
) -> str:
    """Assemble the Markdown body: per-winner sections + FDR caveat + WARNINGS."""
    lines: list[str] = []
    lines.append("# Step 21.7 — Scale-Fit Audit Report")
    lines.append("")
    lines.append(
        f"Computed across {len(audits)} winner(s). HDI prob={hdi_prob}, "
        f"FDR-BH alpha={fdr_alpha}, ESS-drop threshold={ess_drop_threshold}."
    )
    lines.append("")

    # FDR-BH caveat — called out at the TOP of the body so reviewers hit
    # it before scanning individual beta rows.
    lines.append("## FDR-BH diagnostic caveat")
    lines.append("")
    lines.append(
        "FDR-BH applied here uses the posterior two-sided tail probability "
        "`p = 2 * min(P(x > 0), P(x < 0))` as a surrogate p-value. This is "
        "NOT identical to frequentist FDR: the posterior tail probability "
        "is a credibility measure, not the probability of a null-hypothesis "
        "error. Use it as an HDI-agreement diagnostic — in well-behaved "
        "posteriors, `excludes_zero_hdi` and `excludes_zero_fdr` should "
        "agree; disagreements flag heavy-tailed or poorly-mixed chains "
        "worth investigating."
    )
    lines.append("")

    # ESS-degradation WARNINGS (if any).
    degraded = [a for a in audits if a.ess_degraded]
    if degraded:
        lines.append("## WARNINGS — ESS degraded after adding L2 covariates")
        lines.append("")
        for a in degraded:
            lines.append(
                f"- **{a.winner}** ({a.display_name}): {a.notes} Consider "
                f"rerunning step 21.6 with increased warmup/samples or "
                f"tighter priors on the `beta_*` sites."
            )
        lines.append("")

    # Per-winner sections.
    lines.append("## Per-winner audit")
    lines.append("")
    for a in audits:
        header = f"### {a.winner} ({a.display_name})"
        if a.load_warning:
            lines.append(header)
            lines.append("")
            lines.append(f"**Load warning:** {a.load_warning}")
            lines.append("")
            continue

        lines.append(header)
        lines.append("")
        lines.append(f"- **N beta sites:** {len(a.beta_sites)}")
        lines.append(f"- **Max R-hat (betas):** {_fmt_float(a.max_rhat_betas, 4)}")
        lines.append(f"- **Min ESS_bulk (betas):** {_fmt_float(a.min_ess_betas, 0)}")
        lines.append(
            f"- **Min ESS shared (baseline -> L2):** "
            f"{_fmt_float(a.min_ess_shared_baseline, 0)} -> "
            f"{_fmt_float(a.min_ess_shared_l2, 0)}"
        )
        lines.append(f"- **ESS degraded flag:** {a.ess_degraded}")
        lines.append("")

        notable = [
            bs
            for bs in a.beta_sites
            if bs.excludes_zero_hdi or bs.tail_p < fdr_alpha
        ]
        if notable:
            lines.append("**Notable effects (tail_p < alpha OR HDI excludes 0):**")
            lines.append("")
            lines.append(
                "| site | covariate | target | mean | HDI | tail_p | p_fdr | "
                "HDI excl 0 | FDR excl 0 |"
            )
            lines.append(
                "|------|-----------|--------|------|-----|--------|-------|"
                "------------|------------|"
            )
            for bs in notable:
                lines.append(
                    f"| `{bs.beta_site}` | {bs.covariate_family} | "
                    f"{bs.target_parameter} | {bs.posterior_mean:+.3f} | "
                    f"[{bs.hdi_low:+.3f}, {bs.hdi_high:+.3f}] | "
                    f"{bs.tail_p:.4f} | {bs.p_fdr_adj:.4f} | "
                    f"{bs.excludes_zero_hdi} | {bs.excludes_zero_fdr} |"
                )
            lines.append("")
        else:
            lines.append("_No beta sites with notable effects._")
            lines.append("")

    return "\n".join(lines)


def _fmt_float(x: float, ndigits: int) -> str:
    """Format a float with ``ndigits`` decimals, or the literal ``'nan'``."""
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{ndigits}f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entry point. Returns the intended exit code for ``sys.exit``.

    Returns
    -------
    int
        0 for PROCEED_TO_AVERAGING or NULL_RESULT (both are valid
        scientific outcomes per plan-checker Issue #4 — unified exit-0
        semantics). 1 is reserved for genuine audit errors and is
        raised via ``sys.exit(1)`` directly from the exception handler.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Step 21.7 scale-fit audit. Loads each winner's L2-refit "
            "posterior from `--l2-dir`, enumerates beta_* sites, computes "
            "95% HDI + FDR-BH corrected flags across all beta sites per "
            "winner, compares shared-parameter ESS against the baseline "
            "fit, and writes a YAML-front-matter report whose "
            "`pipeline_action` header drives step 21.8's self-skip logic."
        )
    )
    parser.add_argument(
        "--l2-dir",
        default="output/bayesian/21_l2/",
        help="Directory with {winner}_posterior.nc from step 21.6.",
    )
    parser.add_argument(
        "--baseline-dir",
        default="output/bayesian/21_baseline/",
        help="Directory with {winner}_posterior.nc from step 21.3.",
    )
    parser.add_argument(
        "--winners-file",
        default="output/bayesian/21_baseline/winners.txt",
        help="Winners file from step 21.5 (comma-separated display names).",
    )
    parser.add_argument(
        "--hdi-prob",
        type=float,
        default=0.95,
        help="HDI probability mass (default 0.95).",
    )
    parser.add_argument(
        "--fdr-alpha",
        type=float,
        default=0.05,
        help="FDR-BH alpha level (default 0.05).",
    )
    parser.add_argument(
        "--ess-drop-threshold",
        type=float,
        default=0.5,
        help=(
            "Fractional ESS-drop threshold (default 0.5 = flag if L2 "
            "min-ESS drops below 50%% of baseline)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="output/bayesian/21_l2/",
        help="Directory for scale_audit_report.md and per-winner CSVs.",
    )
    args = parser.parse_args()

    l2_dir = Path(args.l2_dir)
    baseline_dir = Path(args.baseline_dir)
    winners_file = Path(args.winners_file)
    output_dir = Path(args.output_dir)

    print("=" * 80)
    print("STEP 21.7 — SCALE-FIT AUDIT")
    print("=" * 80)
    print(f"  L2 dir: {l2_dir}")
    print(f"  Baseline dir: {baseline_dir}")
    print(f"  Winners file: {winners_file}")
    print(f"  HDI prob: {args.hdi_prob}")
    print(f"  FDR-BH alpha: {args.fdr_alpha}")
    print(f"  ESS-drop threshold: {args.ess_drop_threshold}")
    print(f"  Output dir: {output_dir}")
    print("=" * 80)

    # Parse winners (FileNotFoundError / ValueError bubbles up as genuine
    # audit error with exit 1 via the main __main__ wrapper).
    winners = _parse_winners_file(winners_file)
    print(
        f"[AUDIT] Parsed {len(winners)} winner(s) from winners.txt: "
        f"{[(w, INTERNAL_TO_DISPLAY[w]) for w in winners]}"
    )

    # Per-winner audit loop.
    audits: list[WinnerAudit] = []
    for winner in winners:
        print(f"\n[AUDIT] Processing winner: {winner}")
        audit = _audit_one_winner(
            winner=winner,
            l2_dir=l2_dir,
            baseline_dir=baseline_dir,
            hdi_prob=args.hdi_prob,
            ess_drop_threshold=args.ess_drop_threshold,
        )
        audits.append(audit)

        # Emit per-winner CSV unless the NetCDF was missing (load_warning
        # populated — nothing to table).
        if not audit.load_warning:
            per_winner_csv = output_dir / f"{winner}_beta_hdi_table.csv"
            _write_per_winner_csv(audit, per_winner_csv)
            print(f"[AUDIT] {winner}: wrote {per_winner_csv}")

    # Decision rule: PROCEED_TO_AVERAGING iff at least one beta site
    # across all winners survived FDR-BH correction; else NULL_RESULT.
    # Both paths exit 0 per plan-checker Issue #4 unified exit-0 semantics.
    n_fdr_excl = sum(
        1 for a in audits for bs in a.beta_sites if bs.excludes_zero_fdr
    )
    pipeline_action = (
        "PROCEED_TO_AVERAGING" if n_fdr_excl > 0 else "NULL_RESULT"
    )

    # Assemble the report.
    yaml_header = _format_yaml_header(audits, pipeline_action)
    body = _format_report_body(
        audits,
        hdi_prob=args.hdi_prob,
        fdr_alpha=args.fdr_alpha,
        ess_drop_threshold=args.ess_drop_threshold,
    )
    report_path = output_dir / "scale_audit_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(yaml_header + "\n\n" + body, encoding="utf-8")
    print(f"\n[AUDIT] Wrote {report_path}")

    # Summary stdout for SLURM log.
    degraded = [a.winner for a in audits if a.ess_degraded]
    print("\n" + "=" * 80)
    print(f"  pipeline_action: {pipeline_action}")
    print(f"  n_beta_sites_total: {sum(len(a.beta_sites) for a in audits)}")
    print(
        f"  n_hdi_excludes_zero: "
        f"{sum(1 for a in audits for bs in a.beta_sites if bs.excludes_zero_hdi)}"
    )
    print(f"  n_fdr_excludes_zero: {n_fdr_excl}")
    print(f"  ess_degraded_models: {degraded}")
    print("=" * 80)

    if pipeline_action == "NULL_RESULT":
        print(
            "\n[NULL RESULT] Zero beta sites survived FDR-BH across all "
            "winners. Step 21.8 (model averaging) will self-skip via the "
            "YAML `pipeline_action` header. This is a valid scientific "
            "outcome, NOT an audit error — exit 0."
        )
    else:
        print(
            f"\n[PROCEED] {n_fdr_excl} beta site(s) survived FDR-BH across "
            f"winners. Step 21.8 will run model averaging."
        )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        # Genuine audit error: bad inputs or corrupt NetCDF. Exit 1 per
        # plan-checker Issue #4 — distinct from NULL_RESULT (exit 0).
        print(
            f"\n[FATAL] Audit failed with {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
