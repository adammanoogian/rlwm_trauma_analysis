"""Step 21.8 — stacking-weighted model averaging across winners.

Phase 21 Wave 8 averages the scale-effect posteriors (``beta_*`` sites)
across the winner set produced by step 21.5 (LOO + stacking weights).
This replaces "pick the single-winner HDI" with a Yao et al. (2018)
stacking-weighted posterior mixture so manuscript tables report
uncertainty propagated across the model set rather than conditional on
a single winner.

Mixed-cardinality handling (canonical-key matching)
---------------------------------------------------
Different winners can carry DIFFERENT numbers of ``beta_*`` sites:

- M3/M5: 2 sites each (``beta_lec_kappa``, ``beta_iesr_kappa``).
- M6a:   2 sites  (``beta_lec_kappa_s``, ``beta_iesr_kappa_s``).
- M6b (subscale path): 32 sites with names
  ``beta_{cov}_{param}`` for ``cov ∈ {lec_total, iesr_total,
  iesr_intr_resid, iesr_avd_resid}`` and ``param`` across 8 parameters.
- M1/M2: 0 sites (copy-through path) — skipped entirely from averaging.

To average across winners we parse each site name into a CANONICAL KEY
``(covariate_family, target_parameter)`` and normalise M6b's
``lec_total`` -> ``lec`` and ``iesr_total`` -> ``iesr``. This way,
``beta_lec_kappa`` from M3/M5 and ``beta_lec_total_kappa`` from M6b
BOTH collapse to key ``("lec", "kappa")`` and their posterior samples
can be mixed. Subscale-exclusive families (``iesr_intr_resid``,
``iesr_avd_resid``) keep their full names — keys sourced from M6b alone
are flagged ``single_source=True`` and reported verbatim with no
cross-winner averaging applied (user-approved Option C; no scientifically
defensible way to average what doesn't exist in the other winners).

Subsampling procedure
---------------------
For each winner with non-empty ``beta_*`` sites:
  1. Load ``{l2_dir}/{winner}_posterior.nc``.
  2. For each site, ``.stack(sample=("chain", "draw"))`` -> 1-D array.
  3. Draw ``n_sub = int(w * TARGET_TOTAL_SAMPLES)`` samples WITHOUT
     replacement, where ``w`` is the winner's normalised stacking
     weight. ``TARGET_TOTAL_SAMPLES = 8000`` is a fixed budget that
     keeps the mixture big enough for stable HDI computation but
     bounded enough to fit comfortably in RAM during the downstream
     ``az.hdi`` call.
  4. Group by canonical key across winners and concatenate.

For each canonical key with ``n_winners_contributing >= 2``, compute
``averaged_mean``, ``averaged_sd``, 95% HDI, ``averaged_excludes_zero``,
and a two-sided ``tail_p = 2 * min(P(>0), P(<0))``. For keys with
``n_winners_contributing == 1`` (M6b-exclusive subscale families),
report the single winner's posterior verbatim with
``single_source=True``.

Disagreement detection
----------------------
For each averaged key that exists in the single-winner HDI tables
(``{winner}_beta_hdi_table.csv`` from plan 21-08), compare
``single_winner_excludes_zero`` vs. ``averaged_excludes_zero`` and
set ``disagreement_flag=True`` when they differ. These are the rows
most likely to shift the manuscript narrative after averaging.

Three short-circuit paths
-------------------------
1. **NULL_RESULT from audit**: pipeline_action header in
   ``scale_audit_report.md`` is ``NULL_RESULT`` — no beta sites
   survived FDR-BH. Writes ``averaging_skipped.md`` + exits 0. This is
   valid science, not an error.

2. **Single winner**: ``len(winners) == 1`` — averaging would reduce to
   the single winner's posterior (already captured in plan 21-08 per-
   winner HDI tables). Writes ``single_winner_mode.md`` + exits 0.

3. **Multi-winner averaging**: the main path described above.

Optional M6b-subscale exploratory arm
-------------------------------------
If ``"wmrl_m6b"`` is in the winner set AND ``--launch-subscale-arm`` is
truthy (default True), writes ``launch_subscale.flag`` — a marker file
that the cluster wrapper ``cluster/21_8_model_averaging.slurm`` reads
to fire-and-forget ``sbatch cluster/13_bayesian_m6b_subscale.slurm``.
The subscale arm writes its posterior to the Phase-16 canonical path
``models/bayesian/wmrl_m6b_subscale_posterior.nc`` (not a 21_l2
subdirectory) — plan 21-10 reads from there. This avoids patching the
subscale SLURM script for a one-shot exploratory arm.

Outputs
-------
- ``{output_dir}/averaged_scale_effects.csv`` — primary artefact, per
  canonical key: ``covariate_family, target_parameter, canonical_key,
  averaged_mean, averaged_sd, hdi_low, hdi_high, averaged_excludes_zero,
  tail_p, n_winners_contributing, single_source, source_winner_if_single,
  disagreement_flag``.
- ``{output_dir}/averaging_summary.md`` — narrative + disagreement list.
- ``{output_dir}/averaging_skipped.md`` (only in NULL_RESULT branch).
- ``{output_dir}/single_winner_mode.md`` (only in single-winner branch).
- ``{output_dir}/launch_subscale.flag`` (only when M6b is a winner and
  ``--launch-subscale-arm`` is on).

Usage
-----
>>> python scripts/06_fit_analyses/03_model_averaging.py \\
...     --l2-dir models/bayesian/21_l2/ \\
...     --stacking-results models/bayesian/21_baseline/loo_stacking_results.csv \\
...     --winners-file models/bayesian/21_baseline/winners.txt \\
...     --audit-report models/bayesian/21_l2/scale_audit_report.md \\
...     --output-dir models/bayesian/21_l2/

References
----------
Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Using
stacking to average Bayesian predictive distributions.
*Bayesian Analysis*, 13(3), 917-1007. DOI 10.1214/17-BA1091.
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
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import load_netcdf_with_validation  # noqa: E402

# Display-name <-> internal-id mappings (must match
# scripts/05_post_fitting_checks/02_scale_audit.py and scripts/06_fit_analyses/02_compute_loo_stacking.py).
DISPLAY_TO_INTERNAL: dict[str, str] = {
    "M1": "qlearning",
    "M2": "wmrl",
    "M3": "wmrl_m3",
    "M5": "wmrl_m5",
    "M6a": "wmrl_m6a",
    "M6b": "wmrl_m6b",
}
INTERNAL_TO_DISPLAY: dict[str, str] = {v: k for k, v in DISPLAY_TO_INTERNAL.items()}

# Mixture sampling budget — keeps the concatenated posterior mixture
# bounded to a stable size independent of the raw posterior sample
# count. 8000 is large enough that `az.hdi` is well-conditioned and
# ``tail_p`` has enough resolution to distinguish 0.01 from 0.02.
TARGET_TOTAL_SAMPLES: int = 8000

# Known covariate families — ordered LONGEST-PREFIX-FIRST so
# ``iesr_intr_resid`` is matched before ``iesr``, ``iesr_total`` before
# ``iesr``, and ``lec_total`` before ``lec``. This ordering is
# load-bearing for canonical key parsing; do NOT alphabetise.
_COVARIATE_FAMILIES: tuple[str, ...] = (
    "iesr_intr_resid",
    "iesr_avd_resid",
    "iesr_total",
    "lec_total",
    "iesr",
    "lec",
)

# Covariate family -> canonical key covariate (M6b's ``_total`` suffix
# is stripped so subscale betas on ``lec_total`` / ``iesr_total`` match
# by canonical key with 2-cov betas on ``lec`` / ``iesr``).
_COVARIATE_CANONICAL: dict[str, str] = {
    "iesr_intr_resid": "iesr_intr_resid",
    "iesr_avd_resid": "iesr_avd_resid",
    "iesr_total": "iesr",
    "lec_total": "lec",
    "iesr": "iesr",
    "lec": "lec",
}

# Target parameters that can carry L2 betas. Longest-prefix-first
# matching means ``kappa_total``, ``kappa_share``, and ``kappa_s`` are
# all recognised distinctly from bare ``kappa``.
_KNOWN_TARGETS: tuple[str, ...] = (
    "alpha_pos",
    "alpha_neg",
    "phi_rl",
    "phi",
    "rho",
    "capacity",
    "epsilon",
    "kappa_total",
    "kappa_share",
    "kappa_s",
    "kappa",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WinnerContribution:
    """One winner's contribution to a canonical key's mixture."""

    winner: str  # internal id
    display_name: str
    site_name: str  # raw beta_* site name in that winner's posterior
    samples: np.ndarray  # subsampled 1-D array, length ~ w * TARGET


@dataclass
class AveragedBeta:
    """Per-canonical-key averaged beta record."""

    covariate_family: str
    target_parameter: str
    canonical_key: str  # "{cov}|{target}" human-readable join
    averaged_mean: float
    averaged_sd: float
    hdi_low: float
    hdi_high: float
    averaged_excludes_zero: bool
    tail_p: float
    n_winners_contributing: int
    single_source: bool
    source_winner_if_single: str  # internal id, empty when not single
    disagreement_flag: bool
    contributing_winners: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _coerce_bool(value: object) -> bool:
    """Coerce a CSV cell (possibly a string or numpy bool) to Python bool.

    Necessary because the plan-08 metadata trailer row stuffs string
    content into the ``excludes_zero_hdi`` column, forcing pandas to
    read the whole column as ``object`` dtype. A direct ``bool(val)`` on
    that column would return ``True`` for both ``"True"`` and
    ``"False"`` (non-empty string truthiness), silently inverting
    disagreement detection for every null site.

    Also handles ``numpy.bool_`` explicitly because NumPy 2.x does NOT
    subclass Python ``bool`` or ``int`` any more — a bare
    ``isinstance(np.True_, bool)`` returns ``False`` and the function
    would fall through to the ``return False`` default, silently
    inverting every True flag to False.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value) and not (
            isinstance(value, (float, np.floating)) and bool(np.isnan(value))
        )
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "t", "y")
    return False


def _parse_yaml_pipeline_action(audit_report_path: Path) -> str:
    """Extract ``pipeline_action`` from the audit report YAML front-matter.

    Parameters
    ----------
    audit_report_path : Path
        Path to ``scale_audit_report.md`` written by step 21.7.

    Returns
    -------
    str
        Value of ``pipeline_action`` — ``PROCEED_TO_AVERAGING`` or
        ``NULL_RESULT``. ``UNKNOWN`` if the report is missing or the
        header is malformed (caller treats as PROCEED since we can't
        safely infer NULL without evidence).

    Notes
    -----
    Hand-parse rather than depend on PyYAML — the expected header is a
    handful of lines between ``---`` markers and this avoids adding a
    dependency to ds_env beyond ArviZ / pandas / statsmodels already
    loaded by step 21.7.
    """
    if not audit_report_path.exists():
        print(
            f"[AVG] WARNING: audit report {audit_report_path} not found. "
            f"Proceeding as if pipeline_action=PROCEED_TO_AVERAGING.",
            file=sys.stderr,
        )
        return "PROCEED_TO_AVERAGING"

    text = audit_report_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    in_frontmatter = False
    for line in lines:
        stripped = line.strip()
        if stripped == "---":
            if not in_frontmatter:
                in_frontmatter = True
                continue
            # Closing marker — done.
            break
        if in_frontmatter and stripped.startswith("pipeline_action:"):
            return stripped.split(":", 1)[1].strip()
    return "UNKNOWN"


def _parse_winners_file(winners_path: Path) -> list[str]:
    """Parse ``winners.txt`` into internal model ids."""
    if not winners_path.exists():
        raise FileNotFoundError(
            f"winners file not found: {winners_path}. Expected step 21.5 "
            f"(scripts/06_fit_analyses/02_compute_loo_stacking.py) to have written it."
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


def _read_stacking_weights(
    stacking_csv: Path,
    winners_internal: list[str],
) -> dict[str, float]:
    """Read stacking weights and normalise to sum to 1 among the winner subset.

    Parameters
    ----------
    stacking_csv : Path
        ``loo_stacking_results.csv`` from step 21.5 (index is display
        name — "M1", "M3", etc. — with column ``weight`` per az.compare).
    winners_internal : list[str]
        Internal model ids in the winner subset.

    Returns
    -------
    dict[str, float]
        Mapping winner internal id -> normalised stacking weight.

    Raises
    ------
    FileNotFoundError
        Stacking CSV missing.
    KeyError
        Missing ``weight`` column or missing winner row.
    """
    if not stacking_csv.exists():
        raise FileNotFoundError(
            f"stacking results CSV not found: {stacking_csv}. Expected "
            f"step 21.5 (scripts/06_fit_analyses/02_compute_loo_stacking.py) output."
        )
    df = pd.read_csv(stacking_csv, index_col=0)
    if "weight" not in df.columns:
        raise KeyError(
            f"'weight' column missing from {stacking_csv}. Present columns: "
            f"{list(df.columns)}."
        )

    raw_weights: dict[str, float] = {}
    for winner in winners_internal:
        display = INTERNAL_TO_DISPLAY[winner]
        if display not in df.index:
            raise KeyError(
                f"Winner '{display}' (internal '{winner}') not in stacking "
                f"results index {list(df.index)}."
            )
        raw_weights[winner] = float(df.loc[display, "weight"])

    total = float(sum(raw_weights.values()))
    if total <= 0.0:
        # Degenerate — fall back to uniform so averaging still proceeds.
        print(
            f"[AVG] WARNING: winner stacking weights sum to {total} (<=0). "
            f"Falling back to uniform weights across {len(winners_internal)} "
            f"winners.",
            file=sys.stderr,
        )
        n = len(winners_internal)
        return {w: 1.0 / n for w in winners_internal}

    return {w: raw_weights[w] / total for w in winners_internal}


def _parse_beta_site_name(site_name: str) -> tuple[str, str, str]:
    """Parse ``beta_{cov}_{target}`` into (raw_cov, canonical_cov, target).

    Parameters
    ----------
    site_name : str
        Variable name from ``idata.posterior.data_vars``, must start
        with ``"beta_"``.

    Returns
    -------
    raw_covariate : str
        The covariate family as it appears in the raw site name
        (e.g. ``"lec_total"`` for an M6b subscale site).
    canonical_covariate : str
        Normalised covariate family (``"lec_total"`` -> ``"lec"``,
        ``"iesr_total"`` -> ``"iesr"``; subscale-exclusive families
        retain their full names).
    target_parameter : str
        One of :data:`_KNOWN_TARGETS`, or ``"unknown"`` if not matched.

    Notes
    -----
    Longest-prefix-first ordering of :data:`_COVARIATE_FAMILIES` is
    load-bearing.
    """
    if not site_name.startswith("beta_"):
        return ("unknown", "unknown", "unknown")
    body = site_name[len("beta_") :]

    raw_covariate = "unknown"
    remainder = body
    for cov in _COVARIATE_FAMILIES:
        if body.startswith(cov + "_"):
            raw_covariate = cov
            remainder = body[len(cov) + 1 :]
            break

    target_parameter = "unknown"
    for target in _KNOWN_TARGETS:
        if remainder == target:
            target_parameter = target
            break

    canonical_covariate = _COVARIATE_CANONICAL.get(raw_covariate, raw_covariate)
    return (raw_covariate, canonical_covariate, target_parameter)


# ---------------------------------------------------------------------------
# Core averaging logic
# ---------------------------------------------------------------------------


def _extract_weighted_samples(
    idata: az.InferenceData,
    site_name: str,
    weight: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Extract a weighted subsample from one winner's beta site.

    Parameters
    ----------
    idata : az.InferenceData
        Winner's L2-refit posterior.
    site_name : str
        Raw ``beta_*`` site name in ``idata.posterior.data_vars``.
    weight : float
        Normalised stacking weight (in [0, 1]); determines subsample size.
    rng : np.random.Generator
        Seeded RNG for reproducible subsampling.

    Returns
    -------
    np.ndarray
        1-D subsample, length ``min(n_sub, n_available)`` where
        ``n_sub = max(1, int(weight * TARGET_TOTAL_SAMPLES))``.
        Sampling is WITHOUT replacement — if ``n_sub`` exceeds the
        available sample count, returns the full posterior.
    """
    samples = (
        idata.posterior[site_name]
        .stack(sample=("chain", "draw"))
        .values
    )
    samples = np.asarray(samples).ravel()

    n_available = samples.size
    n_sub = max(1, int(weight * TARGET_TOTAL_SAMPLES))
    if n_sub >= n_available:
        # Degenerate case — return everything.
        return samples
    idx = rng.choice(n_available, size=n_sub, replace=False)
    return samples[idx]


def _build_key_contributions(
    winners_weights: dict[str, float],
    l2_dir: Path,
    rng: np.random.Generator,
) -> dict[tuple[str, str], list[WinnerContribution]]:
    """Group weighted subsamples by canonical (covariate, target) key.

    Parameters
    ----------
    winners_weights : dict[str, float]
        Normalised stacking weights, keyed by internal model id.
    l2_dir : Path
        Directory holding ``{winner}_posterior.nc`` files.
    rng : np.random.Generator
        Seeded RNG for reproducible subsampling.

    Returns
    -------
    dict[(cov_canonical, target), list[WinnerContribution]]
        Every canonical key that appeared in at least one winner's
        posterior, mapped to its contributions. Keys with length 1
        signal subscale-exclusive families (single_source path).
    """
    key_contributions: dict[tuple[str, str], list[WinnerContribution]] = {}

    for winner, weight in winners_weights.items():
        display = INTERNAL_TO_DISPLAY.get(winner, winner)
        nc_path = l2_dir / f"{winner}_posterior.nc"
        if not nc_path.exists():
            print(
                f"[AVG] WARNING: L2 posterior missing for {winner} at "
                f"{nc_path} — skipping. Check step 21.6 convergence gate.",
                file=sys.stderr,
            )
            continue

        try:
            idata = load_netcdf_with_validation(nc_path, winner)
        except Exception as exc:  # noqa: BLE001 — corrupt NetCDF is fatal
            raise RuntimeError(
                f"Failed to load {nc_path}: {type(exc).__name__}: {exc}"
            ) from exc

        beta_site_names = [
            v for v in idata.posterior.data_vars if v.startswith("beta_")
        ]
        print(
            f"[AVG] {winner} ({display}): weight={weight:.4f}, "
            f"{len(beta_site_names)} beta sites: {beta_site_names}"
        )

        if not beta_site_names:
            # M1/M2 copy-through path — no L2 sites, nothing to average.
            continue

        for site_name in beta_site_names:
            _, canonical_cov, target = _parse_beta_site_name(site_name)
            if canonical_cov == "unknown" or target == "unknown":
                print(
                    f"[AVG] WARNING: {winner}: site '{site_name}' did not "
                    f"parse cleanly (cov={canonical_cov}, target={target}). "
                    f"Skipping from averaging.",
                    file=sys.stderr,
                )
                continue

            samples = _extract_weighted_samples(idata, site_name, weight, rng)
            contribution = WinnerContribution(
                winner=winner,
                display_name=display,
                site_name=site_name,
                samples=samples,
            )
            key_contributions.setdefault(
                (canonical_cov, target), []
            ).append(contribution)

    return key_contributions


def _compute_averaged_beta(
    cov: str,
    target: str,
    contributions: list[WinnerContribution],
) -> AveragedBeta:
    """Compute the mixture posterior summary for one canonical key.

    Parameters
    ----------
    cov : str
        Canonical covariate family (already normalised upstream).
    target : str
        Canonical target parameter.
    contributions : list[WinnerContribution]
        All winners contributing to this key (length >= 1).

    Returns
    -------
    AveragedBeta
        Populated record. ``single_source=True`` when only one winner
        contributed (subscale-exclusive family or M6b-only path).
    """
    mixture = np.concatenate([c.samples for c in contributions])
    mean = float(np.mean(mixture))
    sd = float(np.std(mixture, ddof=1)) if mixture.size > 1 else 0.0

    hdi_arr = np.asarray(az.hdi(mixture, hdi_prob=0.95))
    hdi_low, hdi_high = float(hdi_arr[0]), float(hdi_arr[1])
    excludes_zero = bool(hdi_low > 0.0 or hdi_high < 0.0)

    p_gt = float(np.mean(mixture > 0.0))
    p_lt = float(np.mean(mixture < 0.0))
    tail_p = 2.0 * min(p_gt, p_lt)

    single_source = len(contributions) == 1
    source_winner = contributions[0].winner if single_source else ""
    contributing = [c.winner for c in contributions]

    return AveragedBeta(
        covariate_family=cov,
        target_parameter=target,
        canonical_key=f"{cov}|{target}",
        averaged_mean=mean,
        averaged_sd=sd,
        hdi_low=hdi_low,
        hdi_high=hdi_high,
        averaged_excludes_zero=excludes_zero,
        tail_p=tail_p,
        n_winners_contributing=len(contributions),
        single_source=single_source,
        source_winner_if_single=source_winner,
        disagreement_flag=False,  # populated in _flag_disagreements
        contributing_winners=contributing,
    )


def _flag_disagreements(
    averaged: list[AveragedBeta],
    l2_dir: Path,
    winners_internal: list[str],
) -> None:
    """Mutate ``averaged`` in place to set ``disagreement_flag``.

    Compares each canonical key's ``averaged_excludes_zero`` against the
    single-winner inference in the ``{winner}_beta_hdi_table.csv`` files
    produced by plan 21-08. A disagreement is when the two flags differ.

    Notes
    -----
    The plan-08 tables index rows by raw ``beta_site`` name, so we
    re-match each averaged key against every contributing winner's
    single-site row. If ANY contributing winner disagreed with the
    averaged flag, we set ``disagreement_flag=True`` — the most
    conservative rule (flag up, not down, for manuscript scrutiny).
    """
    # Pre-load each winner's plan-08 per-winner CSV.
    # NOTE: The plan-08 metadata trailer row puts string content
    # (``max_rhat_betas=...``) in the ``excludes_zero_hdi`` column, which
    # forces pandas to read the whole column as ``object`` dtype. A naive
    # ``bool(val)`` on that column would return True for BOTH ``"True"``
    # AND ``"False"`` (non-empty string truthiness). Coerce explicitly
    # via a string-aware parser.
    hdi_tables: dict[str, pd.DataFrame] = {}
    for winner in winners_internal:
        csv_path = l2_dir / f"{winner}_beta_hdi_table.csv"
        if not csv_path.exists():
            continue  # winner likely copy-through; no betas to check
        df = pd.read_csv(csv_path)
        # Drop the metadata trailer row so ``.set_index('beta_site')``
        # yields real sites only.
        df = df[df["beta_site"] != "__METADATA__"].copy()
        # Coerce excludes_zero_hdi to a clean bool column (Rule 1 fix:
        # avoids ``bool("False") == True`` on object-dtype columns).
        df["excludes_zero_hdi"] = df["excludes_zero_hdi"].map(_coerce_bool)
        hdi_tables[winner] = df.set_index("beta_site")

    for avg_rec in averaged:
        for contrib_winner in avg_rec.contributing_winners:
            df = hdi_tables.get(contrib_winner)
            if df is None:
                continue
            # Re-derive the raw site by re-parsing each row's beta_site
            # name and matching on canonical key. Contributions don't
            # carry the raw name back to this stage — the canonical key
            # is sufficient because at most one raw site per winner
            # collapses to any single canonical key.
            for site_name in df.index:
                _, canonical_cov, target = _parse_beta_site_name(str(site_name))
                key_here = f"{canonical_cov}|{target}"
                if key_here != avg_rec.canonical_key:
                    continue
                single_excl = _coerce_bool(df.loc[site_name, "excludes_zero_hdi"])
                if single_excl != avg_rec.averaged_excludes_zero:
                    avg_rec.disagreement_flag = True
                break  # only one site per winner per canonical key


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_averaged_csv(averaged: list[AveragedBeta], out_csv: Path) -> None:
    """Write the primary ``averaged_scale_effects.csv``."""
    rows = [
        {
            "covariate_family": a.covariate_family,
            "target_parameter": a.target_parameter,
            "canonical_key": a.canonical_key,
            "averaged_mean": a.averaged_mean,
            "averaged_sd": a.averaged_sd,
            "hdi_low": a.hdi_low,
            "hdi_high": a.hdi_high,
            "averaged_excludes_zero": a.averaged_excludes_zero,
            "tail_p": a.tail_p,
            "n_winners_contributing": a.n_winners_contributing,
            "single_source": a.single_source,
            "source_winner_if_single": a.source_winner_if_single,
            "disagreement_flag": a.disagreement_flag,
        }
        for a in averaged
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "covariate_family",
            "target_parameter",
            "canonical_key",
            "averaged_mean",
            "averaged_sd",
            "hdi_low",
            "hdi_high",
            "averaged_excludes_zero",
            "tail_p",
            "n_winners_contributing",
            "single_source",
            "source_winner_if_single",
            "disagreement_flag",
        ],
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def _write_averaging_summary(
    averaged: list[AveragedBeta],
    winners_weights: dict[str, float],
    out_md: Path,
) -> None:
    """Write the narrative ``averaging_summary.md``.

    Includes normalised weights, overlap vs. single-source key counts,
    and a table of disagreements (rows where single-winner and averaged
    HDI-excludes-zero flags differ).
    """
    n_winners = len(winners_weights)
    n_overlap = sum(1 for a in averaged if not a.single_source)
    n_single = sum(1 for a in averaged if a.single_source)
    disagreements = [a for a in averaged if a.disagreement_flag]

    lines: list[str] = []
    lines.append("# Step 21.8 — Model-Averaged Scale Effects Summary")
    lines.append("")
    lines.append(f"**Winner count:** {n_winners}")
    lines.append("")
    lines.append("## Normalised stacking weights")
    lines.append("")
    lines.append("| winner | display | weight |")
    lines.append("|--------|---------|--------|")
    for winner, w in sorted(
        winners_weights.items(), key=lambda kv: -kv[1]
    ):
        display = INTERNAL_TO_DISPLAY.get(winner, winner)
        lines.append(f"| `{winner}` | {display} | {w:.4f} |")
    lines.append("")

    lines.append("## Canonical key counts")
    lines.append("")
    lines.append(
        f"- **Overlapping keys** (averaged across >= 2 winners): "
        f"{n_overlap}"
    )
    lines.append(
        f"- **Subscale-exclusive keys** (reported from a single "
        f"winner alone): {n_single}"
    )
    lines.append("")

    if n_single > 0:
        lines.append(
            "Subscale-exclusive keys are flagged `single_source=True` in "
            "`averaged_scale_effects.csv`. They arise from M6b's "
            "`iesr_intr_resid` and `iesr_avd_resid` covariate families, "
            "which have no counterpart in the 2-cov winners (M3/M5/M6a). "
            "Reporting verbatim from M6b alone preserves their inference "
            "without introducing a spurious cross-winner mixture."
        )
        lines.append("")

    lines.append("## Disagreements (single-winner vs. averaged)")
    lines.append("")
    if disagreements:
        lines.append(
            "Rows where at least one contributing winner's single-model "
            "`excludes_zero_hdi` flag differs from the averaged "
            "`averaged_excludes_zero` flag. These are the keys most "
            "likely to shift the manuscript narrative after averaging."
        )
        lines.append("")
        lines.append(
            "| canonical_key | averaged_mean | HDI | averaged_excludes_zero | "
            "n_winners | contributing |"
        )
        lines.append(
            "|---------------|---------------|-----|------------------------|"
            "-----------|--------------|"
        )
        for a in disagreements:
            contrib = ", ".join(a.contributing_winners)
            lines.append(
                f"| `{a.canonical_key}` | {a.averaged_mean:+.3f} | "
                f"[{a.hdi_low:+.3f}, {a.hdi_high:+.3f}] | "
                f"{a.averaged_excludes_zero} | "
                f"{a.n_winners_contributing} | {contrib} |"
            )
        lines.append("")
    else:
        lines.append("_No disagreements detected._")
        lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def _write_null_result_marker(out_md: Path) -> None:
    """Write the NULL_RESULT short-circuit marker file."""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        "# Step 21.8 — Averaging Skipped (NULL_RESULT)\n\n"
        "The step 21.7 audit flagged `pipeline_action: NULL_RESULT` in "
        "`scale_audit_report.md`: no beta sites survived FDR-BH across "
        "any winner. Model averaging would be spurious (nothing to "
        "pool), so it has been skipped.\n\n"
        "This is a valid scientific outcome, not a pipeline error. "
        "Step 21.9 reports the null result from the per-winner HDI "
        "tables directly.\n",
        encoding="utf-8",
    )


def _write_single_winner_marker(out_md: Path, sole_winner: str) -> None:
    """Write the single-winner short-circuit marker file."""
    display = INTERNAL_TO_DISPLAY.get(sole_winner, sole_winner)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        f"# Step 21.8 — Single-Winner Mode\n\n"
        f"Only one winner in the winner set: `{sole_winner}` "
        f"({display}). Averaging would reduce to the single winner's "
        f"posterior, which is already captured in step 21.7's per-"
        f"winner HDI table `{sole_winner}_beta_hdi_table.csv`.\n\n"
        f"No cross-winner averaging was performed. Step 21.9 should "
        f"read from the plan-08 per-winner table directly.\n",
        encoding="utf-8",
    )


def _write_subscale_flag(out_flag: Path, sole_m6b_weight: float) -> None:
    """Write the M6b-subscale exploratory arm marker file."""
    out_flag.parent.mkdir(parents=True, exist_ok=True)
    out_flag.write_text(
        "# M6b subscale exploratory arm marker\n"
        "# The cluster wrapper cluster/21_8_model_averaging.slurm reads\n"
        "# this file and fires `sbatch cluster/13_bayesian_m6b_subscale.slurm`\n"
        "# fire-and-forget. Plan 21-10 reads the subscale posterior from\n"
        "# models/bayesian/wmrl_m6b_subscale_posterior.nc (Phase-16\n"
        "# canonical path) if it appeared by manuscript-build time.\n"
        f"m6b_stacking_weight={sole_m6b_weight:.6f}\n"
        "subscale_slurm=cluster/13_bayesian_m6b_subscale.slurm\n"
        "expected_output=models/bayesian/wmrl_m6b_subscale_posterior.nc\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entry point. Returns the intended exit code for ``sys.exit``.

    Returns
    -------
    int
        0 on success, NULL_RESULT short-circuit, or single-winner
        short-circuit (all valid scientific outcomes). 1 reserved for
        genuine errors propagated through the ``__main__`` wrapper.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Step 21.8 stacking-weighted model averaging. Averages "
            "scale-effect (beta_*) posteriors across winners using "
            "Yao et al. 2018 stacking weights. Handles mixed beta "
            "cardinality via canonical-key matching (strips _total "
            "suffix so M6b subscale beta_lec_total_kappa and M3/M5 "
            "beta_lec_kappa collapse to the same key)."
        )
    )
    parser.add_argument(
        "--l2-dir",
        default="models/bayesian/21_l2/",
        help="Directory with {winner}_posterior.nc from step 21.6.",
    )
    parser.add_argument(
        "--stacking-results",
        default="models/bayesian/21_baseline/loo_stacking_results.csv",
        help="Stacking-weights CSV from step 21.5.",
    )
    parser.add_argument(
        "--winners-file",
        default="models/bayesian/21_baseline/winners.txt",
        help="Winners file from step 21.5 (comma-separated display names).",
    )
    parser.add_argument(
        "--audit-report",
        default="models/bayesian/21_l2/scale_audit_report.md",
        help="Scale audit report from step 21.7 (YAML front-matter read).",
    )
    parser.add_argument(
        "--output-dir",
        default="models/bayesian/21_l2/",
        help=(
            "Directory for averaged_scale_effects.csv, averaging_summary.md, "
            "and the optional launch_subscale.flag marker."
        ),
    )
    parser.add_argument(
        "--launch-subscale-arm",
        action="store_true",
        default=True,
        help=(
            "Write launch_subscale.flag when M6b is a winner (default on). "
            "The cluster wrapper reads the flag and fires the subscale "
            "SLURM fire-and-forget."
        ),
    )
    parser.add_argument(
        "--no-launch-subscale-arm",
        dest="launch_subscale_arm",
        action="store_false",
        help="Disable the M6b-subscale exploratory arm (overrides default).",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="Seed for the subsampling RNG (default 42).",
    )
    args = parser.parse_args()

    l2_dir = Path(args.l2_dir)
    stacking_csv = Path(args.stacking_results)
    winners_file = Path(args.winners_file)
    audit_report = Path(args.audit_report)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 21.8 — MODEL-AVERAGED SCALE EFFECTS")
    print("=" * 80)
    print(f"  L2 dir: {l2_dir}")
    print(f"  Stacking results: {stacking_csv}")
    print(f"  Winners file: {winners_file}")
    print(f"  Audit report: {audit_report}")
    print(f"  Output dir: {output_dir}")
    print(f"  Launch subscale arm: {args.launch_subscale_arm}")
    print("=" * 80)

    # ----- Short-circuit path 1: NULL_RESULT from step 21.7 audit -----
    pipeline_action = _parse_yaml_pipeline_action(audit_report)
    print(f"[AVG] Parsed pipeline_action from audit: {pipeline_action}")
    if pipeline_action == "NULL_RESULT":
        print(
            "[AVG] NULL_RESULT short-circuit: no beta sites survived "
            "FDR-BH across any winner. Skipping averaging entirely."
        )
        _write_null_result_marker(output_dir / "averaging_skipped.md")
        print(f"[AVG] Wrote {output_dir / 'averaging_skipped.md'}")
        return 0

    # ----- Short-circuit path 2: single winner -----
    winners_internal = _parse_winners_file(winners_file)
    print(
        f"[AVG] Winners: "
        f"{[(w, INTERNAL_TO_DISPLAY[w]) for w in winners_internal]}"
    )
    if len(winners_internal) == 1:
        sole = winners_internal[0]
        print(
            f"[AVG] Single-winner short-circuit: only '{sole}' in the "
            f"winner set. Averaging reduces to the plan-08 per-winner table."
        )
        _write_single_winner_marker(
            output_dir / "single_winner_mode.md", sole_winner=sole
        )
        print(f"[AVG] Wrote {output_dir / 'single_winner_mode.md'}")
        return 0

    # ----- Main path: multi-winner averaging -----
    winners_weights = _read_stacking_weights(stacking_csv, winners_internal)
    print("[AVG] Normalised stacking weights:")
    for w, v in sorted(winners_weights.items(), key=lambda kv: -kv[1]):
        print(f"  {w} ({INTERNAL_TO_DISPLAY[w]}): {v:.4f}")

    rng = np.random.default_rng(args.rng_seed)
    key_contributions = _build_key_contributions(
        winners_weights=winners_weights,
        l2_dir=l2_dir,
        rng=rng,
    )
    print(f"[AVG] Canonical keys discovered: {len(key_contributions)}")

    if not key_contributions:
        print(
            "[AVG] WARNING: no canonical keys materialised across winners "
            "(all copy-through or parse failures). Writing empty "
            "averaged_scale_effects.csv + averaging_summary.md."
        )
        _write_averaged_csv([], output_dir / "averaged_scale_effects.csv")
        _write_averaging_summary(
            [], winners_weights, output_dir / "averaging_summary.md"
        )
        return 0

    averaged: list[AveragedBeta] = []
    for (cov, target), contributions in sorted(key_contributions.items()):
        rec = _compute_averaged_beta(cov, target, contributions)
        averaged.append(rec)
        note = (
            f"[NOTE] {rec.canonical_key} reported from "
            f"{rec.source_winner_if_single} posterior alone; no "
            f"cross-winner averaging applied (subscale-exclusive family)."
            if rec.single_source
            else (
                f"[AVG] {rec.canonical_key}: averaged across "
                f"{rec.n_winners_contributing} winners "
                f"({', '.join(rec.contributing_winners)})."
            )
        )
        print(note)

    # Compare averaged HDI-excludes-zero vs. single-winner HDI-excludes-
    # zero from the plan-08 per-winner tables, flag mismatches.
    _flag_disagreements(averaged, l2_dir, winners_internal)

    # Write primary outputs.
    avg_csv = output_dir / "averaged_scale_effects.csv"
    summary_md = output_dir / "averaging_summary.md"
    _write_averaged_csv(averaged, avg_csv)
    _write_averaging_summary(averaged, winners_weights, summary_md)
    print(f"[AVG] Wrote {avg_csv}")
    print(f"[AVG] Wrote {summary_md}")

    # ----- Optional M6b-subscale exploratory arm -----
    if args.launch_subscale_arm and "wmrl_m6b" in winners_internal:
        m6b_weight = winners_weights["wmrl_m6b"]
        flag_path = output_dir / "launch_subscale.flag"
        _write_subscale_flag(flag_path, m6b_weight)
        print(f"[AVG] Wrote {flag_path}")
        print(
            "[SUBSCALE ARM QUEUED] M6b-subscale exploratory arm "
            "(32 betas, 12h) queued by orchestrator."
        )
    else:
        if "wmrl_m6b" not in winners_internal:
            print(
                "[AVG] M6b not in winners — subscale exploratory arm "
                "NOT queued (no exploratory arm needed)."
            )
        else:
            print(
                "[AVG] --no-launch-subscale-arm active — subscale "
                "exploratory arm NOT queued (user override)."
            )

    # Summary stdout for SLURM log.
    n_overlap = sum(1 for a in averaged if not a.single_source)
    n_single = sum(1 for a in averaged if a.single_source)
    n_disagree = sum(1 for a in averaged if a.disagreement_flag)
    print("\n" + "=" * 80)
    print(f"  canonical_keys_total: {len(averaged)}")
    print(f"  overlapping_keys: {n_overlap}")
    print(f"  single_source_keys: {n_single}")
    print(f"  disagreements: {n_disagree}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        print(
            f"\n[FATAL] Model averaging failed with "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
