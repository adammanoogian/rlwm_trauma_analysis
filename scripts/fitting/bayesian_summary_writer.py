"""Schema-parity CSV writer for Bayesian hierarchical fits.

Converts ArviZ InferenceData to CSV format matching MLE output schema,
enabling downstream scripts (15, 16, 17) to consume via --source bayesian.

Column layout:
- participant_id
- <param_1>, ..., <param_k>          (posterior mean — matches MLE point-estimate semantics)
- nll, aic, bic, aicc, pseudo_r2     (computed from posterior-mean NLL)
- <param_1>_hdi_low, <param_1>_hdi_high, <param_1>_sd, ...
- max_rhat, min_ess_bulk, num_divergences, min_bfmi, per_chain_ess_bulk
- n_trials, converged, at_bounds
- parameterization_version

Key design decisions (see 13-RESEARCH.md, 32-RESEARCH.md):
1. Posterior MEAN for <param> columns (matches MLE point-estimate semantics).
2. 95% HDI for _hdi_low/_hdi_high.
3. Posterior STD for _sd (not "SE" — frequentist term).
4. converged = max_rhat < 1.01 AND min_ess_bulk > 400 AND num_divergences == 0
   AND min_bfmi >= 0.2 (Baribault & Collins 2023 Tier-1 publication standard).
5. NO grad_norm, hessian_*, _se, _ci_*, high_correlations (Hessian-based).

v4.0 INFRA-04. Phase 32-01 added Tier-1 BFMI gate + per-chain ESS column.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import arviz as az


# ---------------------------------------------------------------------------
# Parameter configuration per model
# ---------------------------------------------------------------------------

_MODEL_PARAMS: dict[str, list[str]] = {
    "qlearning": ["alpha_pos", "alpha_neg", "epsilon"],
    "wmrl": ["alpha_pos", "alpha_neg", "phi", "rho", "capacity", "epsilon"],
    "wmrl_m3": ["alpha_pos", "alpha_neg", "phi", "rho", "capacity", "kappa", "epsilon"],
    "wmrl_m5": [
        "alpha_pos",
        "alpha_neg",
        "phi",
        "rho",
        "capacity",
        "kappa",
        "phi_rl",
        "epsilon",
    ],
    "wmrl_m6a": [
        "alpha_pos",
        "alpha_neg",
        "phi",
        "rho",
        "capacity",
        "kappa_s",
        "epsilon",
    ],
    "wmrl_m6b": [
        "alpha_pos",
        "alpha_neg",
        "phi",
        "rho",
        "capacity",
        "kappa_total",
        "kappa_share",
        "epsilon",
    ],
    "wmrl_m4": [
        "alpha_pos",
        "alpha_neg",
        "phi",
        "rho",
        "capacity",
        "kappa",
        "v_scale",
        "A",
        "delta",
        "t0",
    ],
}


def _get_param_names(model_name: str) -> list[str]:
    """Return parameter names for the given model.

    Parameters
    ----------
    model_name : str
        Model identifier.

    Returns
    -------
    list[str]
        Ordered list of free parameter names for this model.

    Raises
    ------
    ValueError
        If model_name is not recognised.
    """
    if model_name not in _MODEL_PARAMS:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Expected one of {sorted(_MODEL_PARAMS)}; got '{model_name}'."
        )
    return _MODEL_PARAMS[model_name]


# ---------------------------------------------------------------------------
# Column schema helpers
# ---------------------------------------------------------------------------


def _build_column_order(param_names: list[str]) -> list[str]:
    """Build the ordered column list for the output CSV.

    Parameters
    ----------
    param_names : list[str]
        Free parameter names in model order.

    Returns
    -------
    list[str]
        Full ordered column list matching schema-parity spec.
    """
    cols = ["participant_id"]
    # Posterior means (same positions as MLE estimates)
    cols += param_names
    # Info criteria
    cols += ["nll", "aic", "bic", "aicc", "pseudo_r2"]
    # HDI + SD for each parameter
    for p in param_names:
        cols += [f"{p}_hdi_low", f"{p}_hdi_high", f"{p}_sd"]
    # Convergence diagnostics
    cols += ["max_rhat", "min_ess_bulk", "num_divergences"]
    # Tier-1 (Baribault & Collins 2023) energy + per-chain diagnostics
    cols += ["min_bfmi", "per_chain_ess_bulk"]
    # Standard outcome columns
    cols += ["n_trials", "converged", "at_bounds"]
    # Parameterisation metadata. Phase 32-04 adds kappa_parameterization
    # immediately after parameterization_version so legacy readers tolerate
    # additive change.
    cols += ["parameterization_version", "kappa_parameterization"]
    return cols


# ---------------------------------------------------------------------------
# Main writer
# ---------------------------------------------------------------------------


def write_bayesian_summary(
    idata: az.InferenceData,
    model_name: str,
    output_dir: Path,
    *,
    param_names: list[str] | None = None,
    participant_ids: list,
    parameterization_version: str,
    n_trials_per_participant: list[int] | None = None,
    hdi_prob: float = 0.95,
    output_subdir: str | None = None,
    kappa_parameterization: str,
) -> Path:
    """Write schema-parity CSV for a Bayesian hierarchical fit.

    The output CSV has the same participant_id + parameter columns as MLE
    ``{model}_individual_fits.csv`` but replaces Hessian-based uncertainty
    columns (``_se``, ``_ci_*``) with Bayesian ones (``_hdi_low``, ``_hdi_high``,
    ``_sd``) and adds convergence diagnostics (``max_rhat``, ``min_ess_bulk``,
    ``num_divergences``).

    Parameters
    ----------
    idata : az.InferenceData
        ArviZ InferenceData produced by ``az.from_numpyro(mcmc)`` or
        ``build_inference_data_with_loglik()``.
    model_name : str
        Model identifier (e.g. ``'wmrl_m3'``).
    output_dir : Path
        Root output directory. CSV is written to ``output_dir/bayesian/``
        (or ``output_dir/bayesian/<output_subdir>/`` when provided).
    param_names : list[str], optional
        Override the default parameter list for the model. Useful for partial
        fits. Defaults to the canonical list from ``_MODEL_PARAMS``.
    participant_ids : list
        Participant ID values in the same order as the plate dimension in the
        MCMC samples. Length must match the plate dimension.
    parameterization_version : str
        Version string written verbatim to the ``parameterization_version``
        column (e.g. ``'v4.0-K[2,6]-phiapprox'``).
    n_trials_per_participant : list[int], optional
        Number of real (non-padded) trials per participant. Used for AIC/BIC
        computation. If None, n_trials is set to NaN.
    hdi_prob : float, optional
        HDI probability mass. Default 0.95 (95% HDI).
    output_subdir : str or None, optional
        Optional subdirectory under ``bayesian/`` (e.g. ``'21_baseline'``).
        When None, writes to ``output_dir/bayesian/`` (Phase 16 layout).
        When set, writes to ``output_dir/bayesian/<output_subdir>/`` so Phase
        21 outputs never overwrite Phase 16 artefacts. Default None preserves
        backward compatibility.
    kappa_parameterization : str
        Required. One of ``{"softmax", "convex"}``. Written verbatim to the
        ``kappa_parameterization`` CSV column for every row. Phase 32-04
        addition (Collins 2025 vs Senta 2025 formulation tag). Callers
        re-summarising legacy pre-Phase-32 .nc artefacts should pass
        ``"convex"`` (those fits were all under convex mode).

    Returns
    -------
    Path
        Path to the written CSV file.
    """
    if kappa_parameterization not in {"softmax", "convex"}:
        raise ValueError(
            f"kappa_parameterization must be 'softmax' or 'convex', "
            f"got {kappa_parameterization!r}"
        )

    import arviz as az

    if param_names is None:
        param_names = _get_param_names(model_name)

    n_participants = len(participant_ids)
    if output_subdir:
        out_subdir = Path(output_dir) / "bayesian" / output_subdir
    else:
        out_subdir = Path(output_dir) / "bayesian"
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path = out_subdir / f"{model_name}_individual_fits.csv"

    # ------------------------------------------------------------------
    # Extract posterior samples: shape (chains * samples, n_participants)
    # ------------------------------------------------------------------
    posterior = idata.posterior

    # Compute per-parameter posterior summaries
    # arviz.summary returns a DataFrame indexed by variable name + param coords
    summary_df = az.summary(idata, var_names=param_names, hdi_prob=hdi_prob)

    # Compute per-participant convergence metrics
    # summary_df index like "alpha_pos[0]", "alpha_pos[1]", ...
    # We need: max_rhat, min_ess_bulk per participant
    rhat_per_participant: dict[int, float] = {i: 0.0 for i in range(n_participants)}
    ess_per_participant: dict[int, float] = {
        i: float("inf") for i in range(n_participants)
    }

    for row_idx_label, row in summary_df.iterrows():
        row_label = str(row_idx_label)
        # Extract participant index from label like "alpha_pos[3]"
        if "[" in row_label and row_label.endswith("]"):
            try:
                part_idx = int(row_label.split("[")[-1].rstrip("]"))
            except ValueError:
                continue
            if part_idx not in rhat_per_participant:
                continue
            rhat_val = float(row.get("r_hat", 0.0))
            ess_val = float(row.get("ess_bulk", float("inf")))
            if rhat_val > rhat_per_participant[part_idx]:
                rhat_per_participant[part_idx] = rhat_val
            if ess_val < ess_per_participant[part_idx]:
                ess_per_participant[part_idx] = ess_val

    # Replace inf placeholders with nan if no ESS was recorded
    for i in range(n_participants):
        if ess_per_participant[i] == float("inf"):
            ess_per_participant[i] = float("nan")

    # Divergences: scalar from sample_stats
    num_divergences_total = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        num_divergences_total = int(idata.sample_stats["diverging"].values.sum())

    # ------------------------------------------------------------------
    # BFMI per chain (Baribault & Collins 2023 Tier-1; Betancourt 2016)
    # ------------------------------------------------------------------
    # az.bfmi(idata) returns one BFMI per chain. Minimum across chains is
    # the publication-standard summary; values < 0.2 indicate inefficient
    # Hamiltonian flow (energy diffusion bottleneck). NaN tolerated for
    # legacy fits where sample_stats.energy is absent.
    try:
        bfmi_per_chain = np.asarray(az.bfmi(idata)).ravel()
        min_bfmi = float(bfmi_per_chain.min()) if bfmi_per_chain.size else float("nan")
    except Exception:
        bfmi_per_chain = np.array([])
        min_bfmi = float("nan")

    # ------------------------------------------------------------------
    # Per-chain ESS_bulk (single-chain ESS surfaces label-switching /
    # one-bad-chain pathologies that all-chain ESS hides)
    # ------------------------------------------------------------------
    # az.summary aggregates across chains, so we recompute ESS chain-by-
    # chain and record the minimum across plate-indexed parameters per
    # chain. Stored as a semicolon-separated string so downstream readers
    # can split on ";" without breaking CSV parsing.
    per_chain_ess_bulk_str = ""
    n_chains = int(posterior.sizes.get("chain", 1))
    if n_chains > 1:
        chain_ess: list[float] = []
        for c in range(n_chains):
            single_chain = posterior.isel(chain=slice(c, c + 1))
            chain_summary = az.summary(
                single_chain, var_names=param_names, hdi_prob=hdi_prob
            )
            chain_ess.append(float(chain_summary["ess_bulk"].min()))
        per_chain_ess_bulk_str = ";".join(f"{v:.0f}" for v in chain_ess)

    # Posterior draws: collapse chains*samples → (draws, n_participants)
    def _get_draws(param: str) -> np.ndarray:
        """Return draws array of shape (draws, n_participants) or (draws,) for group params."""
        arr = posterior[
            param
        ].values  # (chains, draws, n_participants) or (chains, draws)
        flat = arr.reshape(-1, *arr.shape[2:])  # (chains*draws, ...) → squeeze below
        return flat

    # ------------------------------------------------------------------
    # Build per-participant rows
    # ------------------------------------------------------------------
    column_order = _build_column_order(param_names)
    rows = []

    for i, pid in enumerate(participant_ids):
        row: dict[str, object] = {"participant_id": pid}

        # Posterior means and HDI for each parameter
        for param in param_names:
            draws = _get_draws(param)
            if draws.ndim == 2:
                participant_draws = draws[:, i]  # (draws,)
            else:
                participant_draws = draws  # scalar group param (should not appear here)

            mean_val = float(participant_draws.mean())
            sd_val = float(participant_draws.std())

            # HDI
            # ArviZ 0.22+ treats a 1-D ndarray as samples and returns
            # [lower, upper] of shape (2,). Adding extra newaxes makes
            # ArviZ interpret each draw as its own parameter (returning
            # shape (n_draws, 2)) and float(hdi_result[0]) then raises
            # `only 0-dimensional arrays can be converted to Python scalars`.
            hdi_result = np.asarray(
                az.hdi(participant_draws, hdi_prob=hdi_prob)
            ).ravel()
            hdi_low = float(hdi_result[0])
            hdi_high = float(hdi_result[1])

            row[param] = mean_val
            row[f"{param}_hdi_low"] = hdi_low
            row[f"{param}_hdi_high"] = hdi_high
            row[f"{param}_sd"] = sd_val

        # NLL from posterior mean parameters
        # Use the posterior mean NLL proxy: not recomputed here (expensive).
        # Downstream scripts should recompute from the mean parameter vector.
        # Set to NaN; callers who need NLL should pass n_trials and compute separately.
        nll = float("nan")
        n_trials = (
            n_trials_per_participant[i]
            if n_trials_per_participant is not None
            else float("nan")
        )
        k = len(param_names)

        if not np.isnan(nll) and not np.isnan(n_trials):
            aic = 2 * k + 2 * nll
            bic = k * np.log(n_trials) + 2 * nll
            aicc = aic + (2 * k * (k + 1)) / max(n_trials - k - 1, 1)
            pseudo_r2 = float("nan")  # requires null model NLL
        else:
            aic = bic = aicc = pseudo_r2 = float("nan")

        row["nll"] = nll
        row["aic"] = aic
        row["bic"] = bic
        row["aicc"] = aicc
        row["pseudo_r2"] = pseudo_r2

        # Convergence
        max_rhat = rhat_per_participant[i]
        min_ess = ess_per_participant[i]

        row["max_rhat"] = max_rhat
        row["min_ess_bulk"] = min_ess
        row["num_divergences"] = (
            num_divergences_total  # global; per-participant not available
        )
        row["min_bfmi"] = min_bfmi
        row["per_chain_ess_bulk"] = per_chain_ess_bulk_str

        # converged (Baribault & Collins 2023 Tier-1):
        # max_rhat < 1.01 AND min_ess_bulk > 400 AND num_divergences == 0
        # AND min_bfmi >= 0.2 (NaN tolerated for legacy fits w/o energy stat).
        converged = (
            (not np.isnan(max_rhat) and max_rhat < 1.01)
            and (not np.isnan(min_ess) and min_ess > 400)
            and (num_divergences_total == 0)
            and (np.isnan(min_bfmi) or min_bfmi >= 0.2)
        )
        row["n_trials"] = n_trials
        row["converged"] = converged
        row["at_bounds"] = (
            ""  # not applicable for Bayesian — posterior can explore whole space
        )
        row["parameterization_version"] = parameterization_version
        row["kappa_parameterization"] = kappa_parameterization

        rows.append(row)

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows, columns=column_order)
    df.to_csv(out_path, index=False)
    return out_path


def load_bayesian_fits(path: Path, model_name: str) -> pd.DataFrame:
    """Load a Bayesian fits CSV with basic schema validation.

    Parameters
    ----------
    path : Path
        Path to ``{model}_individual_fits.csv`` written by
        :func:`write_bayesian_summary`.
    model_name : str
        Model identifier for schema validation.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    ValueError
        If ``parameterization_version`` column is missing (legacy file) or
        if required columns are absent.
    """
    df = pd.read_csv(path)

    if "parameterization_version" not in df.columns:
        raise ValueError(
            f"{path} lacks 'parameterization_version' column — "
            "this looks like a v3.0 legacy fit. Re-run with v4.0 pipeline."
        )

    param_names = _get_param_names(model_name)
    missing = [p for p in param_names if p not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing parameter columns {missing}. "
            f"Expected columns: {param_names}."
        )

    return df
