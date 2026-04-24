"""Step 21.5 — PSIS-LOO + stacking (primary) and RFX-BMS/PXP (secondary).

Phase 21 Wave 5 orchestrator for principled Bayesian model comparison.

Pipeline position
-----------------
Runs AFTER plan 21-05's convergence gate
(``scripts/05_post_fitting_checks/01_baseline_audit.py``)
and BEFORE plan 21-07's winner-L2 refit. Consumes
``{baseline_dir}/convergence_table.csv`` (produced by 21.4), filters to rows
with ``pipeline_action == "PROCEED_TO_LOO"``, and ranks the survivors by
LOO-stacking weights. The winner set is then written as ``winners.txt`` for
programmatic consumption by ``cluster/21_6_winner_l2_refit.slurm`` (plan
21-07).

Primary ranking — LOO + stacking
--------------------------------
``arviz.compare(compare_dict, ic="loo", method="stacking")`` implements the
predictive-stacking criterion of Yao, Vehtari, Simpson & Gelman (2018). The
stacking weights are the minimisers of the leave-one-out-averaged KL
divergence between the mixture predictive and the held-out point mass; they
are NOT posterior model probabilities in the BMA sense. Stacking is robust to
M-open settings (the true generative process is outside the candidate set).

Secondary ranking — RFX-BMS with PXP
------------------------------------
Per-participant log evidence is assembled from ``idata.log_likelihood.obs``
(summing over trials, then marginalising chain+draw via ``logsumexp / log N``)
and passed to :func:`scripts.fitting.bms.rfx_bms` (from plan 21-02). This
treats the discrete model identity per participant as a latent categorical
variable, places a Dirichlet prior on population frequencies, and returns
the protected exceedance probability (Rigoux et al. 2014) plus Bayesian
Omnibus Risk. PXP is the group-level analogue of stacking weights.

Pareto-k diagnostic (SOFT GATE — plan-checker Issue #8 Option B)
----------------------------------------------------------------
For each model, ``pct_high = mean(pareto_k > 0.7) * 100`` is computed and
surfaced in the report. Models where ``pct_high > --pareto-k-pct-threshold``
(default 1%) are flagged with a WARNING marker but **not** auto-excluded from
the comparison. Exclusion is a scientific judgement call that belongs at the
Wave 4 human checkpoint at the end of this script, not a silent auto-gate.
ROADMAP success criterion #3 ("Pareto-k < 0.7 on > 99% of observations per
model") is DOCUMENTED, not ENFORCED-VIA-AUTOKILL.

Winner determination
--------------------
Three-tier decision over ranked stacking weights ``w``:

- ``w.iloc[0] >= --stacking-winner-threshold`` (default 0.5) ->
  ``DOMINANT_SINGLE``: exit 0, single winner advances to 21.7.
- ``w.iloc[0] + w.iloc[1] >= --combined-winner-threshold`` (default 0.8) ->
  ``TOP_TWO``: exit 0, top two advance to 21.7.
- Else: ``INCONCLUSIVE_MULTIPLE``: winners = all models with
  ``w >= --weak-winner-threshold`` (default 0.10); exit 2 so the pipeline
  SLURM ``--dependency=afterok`` chain blocks and the user reviews
  ``winner_report.md`` before rerunning with ``--force-winners``.

Outputs
-------
- ``{output_dir}/loo_stacking_results.csv`` — primary comparison table
  (rank, elpd_loo, p_loo, elpd_diff, weight, se, dse, warning, scale) plus
  a ``pct_high_pareto_k`` column.
- ``{output_dir}/rfx_bms_pxp.csv`` — secondary comparison (alpha, r, xp,
  bor, pxp, pxp_exceeds_95) per model.
- ``{output_dir}/winner_report.md`` — human-readable verdict with Summary,
  Primary (LOO+stacking), Secondary (RFX-BMS+PXP), Pareto-k diagnostic,
  Winner verdict, Pipeline action, and USER CHECKPOINT sections.
- ``{output_dir}/winners.txt`` — comma-separated display-name winners
  (e.g., ``M3,M6b``) consumed by plan 21-07.

Exit codes
----------
- 0 -> ``DOMINANT_SINGLE`` or ``TOP_TWO``: pipeline advances via
  ``--dependency=afterok:$THIS_JOBID`` to step 21.6.
- 1 -> fewer than 2 models passed convergence gate: hard abort.
- 2 -> ``INCONCLUSIVE_MULTIPLE``: pipeline pauses; user reviews
  ``winner_report.md`` and either accepts the multi-winner set (rerun with
  ``--force-winners M3,M6b,...``) or re-fits problematic models.

Usage
-----
>>> python scripts/06_fit_analyses/02_compute_loo_stacking.py \
...     --baseline-dir models/bayesian/21_baseline/ \
...     --output-dir models/bayesian/21_baseline/

>>> # Manual-resume after INCONCLUSIVE_MULTIPLE checkpoint:
>>> python scripts/06_fit_analyses/02_compute_loo_stacking.py \
...     --baseline-dir models/bayesian/21_baseline/ \
...     --force-winners M3,M6b

See also
--------
- ``cluster/21_5_loo_stacking_bms.slurm`` — 2h/32G/4-CPU SLURM submission.
- ``scripts/fitting/bms.py`` — :func:`rfx_bms` implementation (plan 21-02).
- ``scripts/05_post_fitting_checks/01_baseline_audit.py`` — upstream convergence gate (plan 21-05).

References
----------
Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Using stacking to
    average Bayesian predictive distributions. *Bayesian Analysis*, 13(3),
    917-1007. https://doi.org/10.1214/17-BA1091

Stephan, K. E., Penny, W. D., Daunizeau, J., Moran, R. J., & Friston, K. J.
    (2009). Bayesian model selection for group studies. *NeuroImage*, 46(4),
    1004-1017. https://doi.org/10.1016/j.neuroimage.2009.03.025

Rigoux, L., Stephan, K. E., Friston, K. J., & Daunizeau, J. (2014). Bayesian
    model selection for group studies - Revisited. *NeuroImage*, 84, 971-985.
    https://doi.org/10.1016/j.neuroimage.2013.08.065

Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model
    evaluation using leave-one-out cross-validation and WAIC. *Statistics
    and Computing*, 27(5), 1413-1432. https://doi.org/10.1007/s11222-016-9696-4
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import logsumexp

# -- Path bootstrap so this script runs both interactively and under SLURM.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import MODELS_BAYESIAN_BASELINE, load_netcdf_with_validation  # noqa: E402
from scripts.fitting.bms import rfx_bms  # noqa: E402

# ---------------------------------------------------------------------------
# Display-name map (mirrors BAYESIAN_NETCDF_MAP in scripts/06_fit_analyses/01_compare_models.py)
# ---------------------------------------------------------------------------
# Convergence-table model keys (from plan 21-05) use the underscore names
# from MODEL_REGISTRY; this maps them to the short display names used in
# az.compare output and the winner report.
MODEL_TO_DISPLAY: dict[str, str] = {
    "qlearning": "M1",
    "wmrl": "M2",
    "wmrl_m3": "M3",
    "wmrl_m5": "M5",
    "wmrl_m6a": "M6a",
    "wmrl_m6b": "M6b",
}

# Minimum number of convergence-eligible models needed for stacking to be
# meaningful. PXP over a singleton is not meaningful either.
MIN_MODELS_FOR_STACKING: int = 2


# ---------------------------------------------------------------------------
# Pure-function core — imported by smoke tests without subprocess overhead.
# ---------------------------------------------------------------------------
def compute_loo_stacking_bms(
    compare_dict: dict[str, az.InferenceData],
    pareto_k_threshold: float = 0.7,
    pareto_k_pct_threshold: float = 1.0,
    stacking_winner_threshold: float = 0.5,
    combined_winner_threshold: float = 0.8,
    weak_winner_threshold: float = 0.10,
    force_winners: list[str] | None = None,
) -> dict[str, object]:
    """Run the full LOO-stacking + RFX-BMS comparison on pre-loaded idatas.

    This is the pure-function core of the orchestrator; the CLI :func:`main`
    wraps it with argument parsing, NetCDF loading, and file I/O. Tests can
    call it directly with hand-crafted stub idatas and avoid subprocess.

    Parameters
    ----------
    compare_dict : dict[str, arviz.InferenceData]
        Display-name-keyed dict of InferenceData objects with
        ``log_likelihood.obs`` groups. Must have at least
        :data:`MIN_MODELS_FOR_STACKING` entries.
    pareto_k_threshold : float, default 0.7
        Threshold on the pointwise Pareto shape parameter k above which a
        PSIS-LOO estimate is considered unreliable (Vehtari et al. 2017).
    pareto_k_pct_threshold : float, default 1.0
        Soft-gate threshold on the percentage of observations exceeding
        ``pareto_k_threshold``. Models over this percentage are FLAGGED
        in the diagnostic column but not dropped from the comparison.
    stacking_winner_threshold : float, default 0.5
        Top-model stacking weight above which the winner is declared
        ``DOMINANT_SINGLE``.
    combined_winner_threshold : float, default 0.8
        Combined top-two stacking weight above which the winners are
        declared ``TOP_TWO``.
    weak_winner_threshold : float, default 0.10
        Minimum stacking weight for a model to be admitted into the
        inconclusive multi-winner set.
    force_winners : list[str] or None, optional
        Manual override. If provided, the automatic three-tier winner
        determination is bypassed and ``force_winners`` is returned
        verbatim as the winner set with ``winner_type = "FORCED"``.

    Returns
    -------
    dict
        Keys: ``comparison`` (pd.DataFrame with stacking results +
        ``pct_high_pareto_k`` column), ``bms_result`` (dict from
        :func:`rfx_bms`), ``pct_high_per_model`` (dict[display_name ->
        float]), ``winners`` (list[str] of display names in ranked
        order), ``winner_type`` (str), ``participant_ids`` (np.ndarray
        of integer participant IDs used for RFX-BMS), ``model_order``
        (list[str] of display names in the column order of the RFX-BMS
        log-evidence matrix).

    Raises
    ------
    ValueError
        If ``compare_dict`` has fewer than :data:`MIN_MODELS_FOR_STACKING`
        entries, if participant coordinates are inconsistent across
        idatas, or if ``force_winners`` names a model not in
        ``compare_dict``.
    """
    if len(compare_dict) < MIN_MODELS_FOR_STACKING:
        raise ValueError(
            f"compare_dict has {len(compare_dict)} models; need at least "
            f"{MIN_MODELS_FOR_STACKING} for meaningful stacking. Check the "
            f"upstream convergence table (plan 21-05)."
        )

    # -----------------------------------------------------------------
    # Pareto-k soft gate — compute pct_high per model, log WARNING, but
    # do NOT exclude from compare_dict. Plan-checker Issue #8 Option B.
    # -----------------------------------------------------------------
    pct_high_per_model: dict[str, float] = {}
    for display_name, idata in compare_dict.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # pareto-k warnings inline-handled
            loo_result = az.loo(idata, pointwise=True)
        pareto_k = np.asarray(loo_result.pareto_k.values)
        pct_high = float(np.mean(pareto_k > pareto_k_threshold) * 100.0)
        pct_high_per_model[display_name] = pct_high
        if pct_high > pareto_k_pct_threshold:
            print(
                f"[WARNING] {display_name}: {pct_high:.2f}% of observations "
                f"have pareto_k > {pareto_k_threshold} (threshold: "
                f"{pareto_k_pct_threshold}%). Flagged for human review at "
                f"Wave 4 checkpoint. NOT auto-excluded per plan-checker "
                f"Issue #8 Option B.",
                file=sys.stderr,
            )

    # -----------------------------------------------------------------
    # Primary — LOO + stacking weights.
    # -----------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pareto-k warnings already surfaced
        comparison = az.compare(compare_dict, ic="loo", method="stacking")

    weight_sum = float(comparison["weight"].sum())
    if abs(weight_sum - 1.0) >= 0.01:
        # ArviZ GitHub #2359 — stacking weight sum can drift outside [1-tol,
        # 1+tol] when the LBFGS optimiser bails early on a degenerate
        # problem. Log and proceed rather than assert-crash, since the
        # relative ranking is still the primary decision input.
        print(
            f"[WARNING] stacking weights sum to {weight_sum:.6f} (expected "
            f"1.0 +/- 0.01). This is a known ArviZ numerical tolerance "
            f"issue (GitHub #2359). Proceeding with comparison as-is.",
            file=sys.stderr,
        )

    # Attach per-model Pareto-k percentages as a new column for CSV export.
    # ArviZ indexes comparison rows by display name, so we can map directly.
    comparison = comparison.copy()
    comparison["pct_high_pareto_k"] = pd.Series(pct_high_per_model)

    # -----------------------------------------------------------------
    # Secondary — RFX-BMS on per-participant log-evidence matrix.
    # -----------------------------------------------------------------
    model_order = list(compare_dict.keys())
    participant_ids_list: list[np.ndarray] = []
    log_evidence_cols: list[np.ndarray] = []

    for display_name in model_order:
        idata = compare_dict[display_name]
        # Shape: (chain, draw, participant, trial_padded). Padded trials are
        # NaN (see scripts.fitting.bayesian_diagnostics.filter_padding_from_loglik),
        # so nansum drops them cleanly.
        ll = np.asarray(idata.log_likelihood.obs.values)
        if ll.ndim != 4:
            raise ValueError(
                f"{display_name}: idata.log_likelihood.obs has ndim "
                f"{ll.ndim}, expected 4 (chain, draw, participant, "
                f"trial_padded). Shape: {ll.shape}."
            )
        per_ppt_ll = np.nansum(ll, axis=-1)  # (chain, draw, participant)
        n_chain, n_draw, n_ppt = per_ppt_ll.shape
        log_evidence_per_ppt = logsumexp(
            per_ppt_ll, axis=(0, 1)
        ) - np.log(n_chain * n_draw)
        log_evidence_cols.append(log_evidence_per_ppt)

        # Participant coordinate for consistency check.
        ppt_coord = np.asarray(idata.log_likelihood.participant.values)
        participant_ids_list.append(np.sort(ppt_coord))

    # Verify participant lists agree across models (sorted). A diff implies
    # the models were fit on different cohort slices — bms over mismatched
    # participants would be meaningless.
    reference_ppts = participant_ids_list[0]
    for dn, ppts in zip(model_order[1:], participant_ids_list[1:], strict=True):
        if not np.array_equal(reference_ppts, ppts):
            raise ValueError(
                f"Participant mismatch between {model_order[0]} "
                f"({len(reference_ppts)} ppts) and {dn} "
                f"({len(ppts)} ppts). RFX-BMS requires identical cohorts."
            )

    log_evidence_matrix = np.column_stack(log_evidence_cols)
    bms_result = rfx_bms(log_evidence_matrix)

    # -----------------------------------------------------------------
    # Winner determination (three-tier) + optional --force-winners override.
    # -----------------------------------------------------------------
    if force_winners is not None:
        unknown = [w for w in force_winners if w not in compare_dict]
        if unknown:
            raise ValueError(
                f"--force-winners contains names not in compare_dict: "
                f"{unknown}. Available: {list(compare_dict.keys())}."
            )
        winners = list(force_winners)
        winner_type = "FORCED"
    else:
        # Sort by 'rank' column (0 = best). Stacking weights are in the
        # 'weight' column.
        comp_sorted = comparison.sort_values("rank")
        weights_sorted = comp_sorted["weight"]
        top_model = str(weights_sorted.index[0])
        top_weight = float(weights_sorted.iloc[0])

        if top_weight >= stacking_winner_threshold:
            winners = [top_model]
            winner_type = "DOMINANT_SINGLE"
        elif len(weights_sorted) >= 2:
            second_model = str(weights_sorted.index[1])
            second_weight = float(weights_sorted.iloc[1])
            if top_weight + second_weight >= combined_winner_threshold:
                winners = [top_model, second_model]
                winner_type = "TOP_TWO"
            else:
                winners = [
                    str(name)
                    for name, w in weights_sorted.items()
                    if float(w) >= weak_winner_threshold
                ]
                winner_type = "INCONCLUSIVE_MULTIPLE"
        else:
            # Only one model — already gated upstream, but defensive branch.
            winners = [top_model]
            winner_type = "DOMINANT_SINGLE"

    return {
        "comparison": comparison,
        "bms_result": bms_result,
        "pct_high_per_model": pct_high_per_model,
        "winners": winners,
        "winner_type": winner_type,
        "participant_ids": reference_ppts,
        "model_order": model_order,
    }


# ---------------------------------------------------------------------------
# Report writers.
# ---------------------------------------------------------------------------
def _write_loo_stacking_csv(comparison: pd.DataFrame, out_path: Path) -> None:
    """Write the LOO + stacking primary comparison table.

    Columns preserved from ``az.compare``: rank, elpd_loo, p_loo, elpd_diff,
    weight, se, dse, warning, scale. Appended by this pipeline:
    ``pct_high_pareto_k`` (percentage of observations with pareto_k > 0.7).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_path, index=True, index_label="model")


def _write_rfx_bms_csv(
    bms_result: dict[str, object],
    model_order: list[str],
    out_path: Path,
) -> None:
    """Write the RFX-BMS secondary comparison table.

    Columns: model (display name), alpha (Dirichlet posterior
    concentration), r (expected frequency), xp (exceedance probability),
    bor (Bayesian Omnibus Risk, constant across models), pxp (protected
    exceedance probability), pxp_exceeds_95 (bool flag for PXP > 0.95).
    """
    alpha = np.asarray(bms_result["alpha"])
    r = np.asarray(bms_result["r"])
    xp = np.asarray(bms_result["xp"])
    pxp = np.asarray(bms_result["pxp"])
    bor = float(bms_result["bor"])

    rows = []
    for i, name in enumerate(model_order):
        rows.append(
            {
                "model": name,
                "alpha": float(alpha[i]),
                "r": float(r[i]),
                "xp": float(xp[i]),
                "bor": bor,
                "pxp": float(pxp[i]),
                "pxp_exceeds_95": bool(pxp[i] > 0.95),
            }
        )

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def _write_winner_report(
    comparison: pd.DataFrame,
    bms_result: dict[str, object],
    pct_high_per_model: dict[str, float],
    winners: list[str],
    winner_type: str,
    model_order: list[str],
    participant_ids: np.ndarray,
    pareto_k_threshold: float,
    pareto_k_pct_threshold: float,
    out_path: Path,
) -> None:
    """Write the human-readable winner verdict with USER CHECKPOINT block.

    Sections: Summary, Primary (LOO + stacking), Secondary (RFX-BMS + PXP),
    Pareto-k diagnostic, Winner verdict, Pipeline action, USER CHECKPOINT.
    """
    lines: list[str] = []
    lines.append("# Step 21.5 — Model Comparison Winner Report")
    lines.append("")
    lines.append(
        "Primary ranking: LOO + stacking weights (Yao et al. 2018, "
        "DOI 10.1214/17-BA1091)."
    )
    lines.append(
        "Secondary ranking: RFX-BMS with protected exceedance probability "
        "(Stephan et al. 2009, DOI 10.1016/j.neuroimage.2009.03.025; "
        "Rigoux et al. 2014, DOI 10.1016/j.neuroimage.2013.08.065)."
    )
    lines.append("")

    # --- Summary ---
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Models compared: {len(model_order)} ({', '.join(model_order)})")
    lines.append(f"- Participants per model: {len(participant_ids)}")
    lines.append(f"- Winner type: **{winner_type}**")
    lines.append(f"- Winners: **{', '.join(winners)}**")
    lines.append("")

    # --- Primary: LOO + stacking ---
    lines.append("## Primary — LOO + stacking weights")
    lines.append("")
    lines.append("| Model | Rank | elpd_loo | p_loo | elpd_diff | weight | se | dse | pct_high_pareto_k |")
    lines.append("|-------|------|----------|-------|-----------|--------|----|----|--------------------|")
    for model_name, row in comparison.sort_values("rank").iterrows():
        lines.append(
            f"| {model_name} | {int(row['rank'])} | "
            f"{float(row['elpd_loo']):.2f} | {float(row['p_loo']):.2f} | "
            f"{float(row['elpd_diff']):.2f} | {float(row['weight']):.4f} | "
            f"{float(row['se']):.2f} | {float(row['dse']):.2f} | "
            f"{float(row['pct_high_pareto_k']):.2f}% |"
        )
    lines.append("")
    lines.append(
        f"Stacking weights sum to {float(comparison['weight'].sum()):.6f} "
        f"(expected 1.0; ArviZ GitHub #2359 may cause small deviations)."
    )
    lines.append("")

    # --- Secondary: RFX-BMS + PXP ---
    lines.append("## Secondary — RFX-BMS + PXP")
    lines.append("")
    alpha = np.asarray(bms_result["alpha"])
    r = np.asarray(bms_result["r"])
    xp = np.asarray(bms_result["xp"])
    pxp = np.asarray(bms_result["pxp"])
    bor = float(bms_result["bor"])
    lines.append(f"- **Bayesian Omnibus Risk (BOR):** {bor:.4f}")
    if bor > 0.25:
        lines.append(
            "  - BOR above 0.25 indicates the null hypothesis (uniform "
            "model frequencies across participants) is non-negligible — "
            "any PXP values below 0.95 should be interpreted as "
            "'no dominant model across participants'."
        )
    lines.append("")
    lines.append("| Model | alpha | r | xp | pxp | pxp > 0.95 |")
    lines.append("|-------|-------|---|----|----|------------|")
    for i, name in enumerate(model_order):
        exceeds = "**YES**" if pxp[i] > 0.95 else "no"
        lines.append(
            f"| {name} | {float(alpha[i]):.3f} | {float(r[i]):.4f} | "
            f"{float(xp[i]):.4f} | {float(pxp[i]):.4f} | {exceeds} |"
        )
    lines.append("")

    # --- Pareto-k diagnostic (SOFT GATE) ---
    lines.append("## Pareto-k diagnostic")
    lines.append("")
    lines.append(
        f"Threshold: pareto_k > {pareto_k_threshold}; flagged if "
        f">{pareto_k_pct_threshold}% of observations exceed (Vehtari et al. "
        f"2017, DOI 10.1007/s11222-016-9696-4)."
    )
    lines.append("")
    lines.append("| Model | pct_high_pareto_k | Status |")
    lines.append("|-------|-------------------|--------|")
    any_flagged = False
    for name in model_order:
        pct = pct_high_per_model[name]
        if pct > pareto_k_pct_threshold:
            status = "**WARNING — exceeds threshold**"
            any_flagged = True
        else:
            status = "OK"
        lines.append(f"| {name} | {pct:.2f}% | {status} |")
    lines.append("")
    if any_flagged:
        lines.append(
            "**Note:** Flagged models are retained in the comparison per "
            "plan-checker Issue #8 Option B (exclusion is a scientific "
            "judgement, not a silent auto-gate). Review each flagged model "
            "before promoting its winner status: consider k-fold CV as an "
            "alternative if influential observations dominate the LOO "
            "estimate."
        )
    else:
        lines.append(
            "All models satisfy ROADMAP success criterion #3 "
            "(Pareto-k < 0.7 on > 99% of observations)."
        )
    lines.append("")

    # --- Winner verdict + pipeline action ---
    lines.append("## Winner verdict")
    lines.append("")
    verdict_line = f"**{winner_type}** — winners: **{', '.join(winners)}**"
    lines.append(verdict_line)
    lines.append("")

    lines.append("## Pipeline action")
    lines.append("")
    if winner_type == "DOMINANT_SINGLE":
        lines.append(
            "Exit code 0. Single dominant winner — the SLURM "
            "`--dependency=afterok` chain advances to step 21.6 "
            "(winner-L2 refit) on this model."
        )
    elif winner_type == "TOP_TWO":
        lines.append(
            "Exit code 0. Two close winners — step 21.6 will refit L2 "
            "for both models and the manuscript reports the pair."
        )
    elif winner_type == "FORCED":
        lines.append(
            "Exit code 0. Manual override via `--force-winners` accepted. "
            "Step 21.6 proceeds on the forced set."
        )
    else:  # INCONCLUSIVE_MULTIPLE
        lines.append(
            "Exit code 2. **Pipeline paused — no dominant winner found.** "
            "The SLURM `--dependency=afterok` chain BLOCKS step 21.6 "
            "naturally until the user reruns this script with "
            "`--force-winners ...` accepting the multi-winner set, "
            "or re-fits models whose Pareto-k flags suggest LOO "
            "estimates are unreliable."
        )
    lines.append("")

    # --- USER CHECKPOINT ---
    lines.append("## USER CHECKPOINT")
    lines.append("")
    lines.append(
        "Review the tables above; if the automatic winner determination is "
        "appropriate, proceed to step 21.6 via the SLURM orchestrator. If "
        "you want to override the winner set (e.g., accept an "
        "INCONCLUSIVE_MULTIPLE verdict or drop a Pareto-k-flagged model), "
        "rerun this script with:"
    )
    lines.append("")
    lines.append("```bash")
    lines.append(
        "python scripts/06_fit_analyses/02_compute_loo_stacking.py \\"
    )
    lines.append(
        "    --baseline-dir models/bayesian/21_baseline/ \\"
    )
    lines.append(
        "    --output-dir models/bayesian/21_baseline/ \\"
    )
    lines.append(
        f"    --force-winners {','.join(winners)}"
    )
    lines.append("```")
    lines.append("")
    lines.append(
        "Then rerun the master pipeline orchestrator (plan 21-10) starting "
        "from step 21.6."
    )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the LOO + stacking + RFX-BMS runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Step 21.5 — PSIS-LOO + stacking (primary) and RFX-BMS/PXP "
            "(secondary) over convergence-gate-passing baseline models."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=MODELS_BAYESIAN_BASELINE,
        help=(
            "Directory containing convergence_table.csv (from 21-05) and "
            "{model}_posterior.nc files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_BAYESIAN_BASELINE,
        help=(
            "Directory to write loo_stacking_results.csv, rfx_bms_pxp.csv, "
            "winner_report.md, and winners.txt."
        ),
    )
    parser.add_argument(
        "--pareto-k-threshold",
        type=float,
        default=0.7,
        help="Pointwise Pareto shape k threshold (Vehtari et al. 2017).",
    )
    parser.add_argument(
        "--pareto-k-pct-threshold",
        type=float,
        default=1.0,
        help=(
            "Percentage of observations with pareto_k > pareto-k-threshold "
            "above which a WARNING is flagged (not auto-excluded, per "
            "plan-checker Issue #8 Option B)."
        ),
    )
    parser.add_argument(
        "--stacking-winner-threshold",
        type=float,
        default=0.5,
        help="Top-model stacking weight for DOMINANT_SINGLE verdict.",
    )
    parser.add_argument(
        "--combined-winner-threshold",
        type=float,
        default=0.8,
        help="Combined top-two stacking weight for TOP_TWO verdict.",
    )
    parser.add_argument(
        "--weak-winner-threshold",
        type=float,
        default=0.10,
        help=(
            "Minimum stacking weight for a model to enter the inconclusive "
            "multi-winner set."
        ),
    )
    parser.add_argument(
        "--force-winners",
        type=str,
        default=None,
        help=(
            "Comma-separated override winner set (e.g., 'M3,M6b') for "
            "manual resume after an INCONCLUSIVE_MULTIPLE checkpoint."
        ),
    )
    return parser.parse_args(argv)


def _load_convergence_eligible_models(
    baseline_dir: Path,
) -> list[str]:
    """Read convergence_table.csv and return models flagged PROCEED_TO_LOO.

    Parameters
    ----------
    baseline_dir : Path
        Directory containing ``convergence_table.csv`` from plan 21-05.

    Returns
    -------
    list[str]
        Model underscore names (e.g., ``'wmrl_m3'``) with
        ``pipeline_action == "PROCEED_TO_LOO"``.

    Raises
    ------
    FileNotFoundError
        If ``convergence_table.csv`` does not exist in ``baseline_dir``.
    KeyError
        If the CSV is missing the required ``pipeline_action`` column.
    """
    csv_path = baseline_dir / "convergence_table.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"convergence_table.csv not found at {csv_path}. Run plan 21-05 "
            f"(scripts/05_post_fitting_checks/01_baseline_audit.py) first."
        )
    df = pd.read_csv(csv_path)
    if "pipeline_action" not in df.columns:
        raise KeyError(
            f"{csv_path} is missing the 'pipeline_action' column. Expected "
            f"columns from plan 21-05: model, max_rhat, min_ess_bulk, "
            f"n_divergences, min_bfmi, ppc_coverage, gate_status, "
            f"pipeline_action. Got: {list(df.columns)}."
        )
    eligible_df = df[df["pipeline_action"] == "PROCEED_TO_LOO"]
    return list(eligible_df["model"].astype(str).values)


def _load_compare_dict(
    baseline_dir: Path,
    eligible_models: list[str],
) -> dict[str, az.InferenceData]:
    """Load {model}_posterior.nc for each eligible model into a display-name dict.

    Parameters
    ----------
    baseline_dir : Path
        Directory containing the NetCDFs.
    eligible_models : list[str]
        Model underscore names passed from
        :func:`_load_convergence_eligible_models`.

    Returns
    -------
    dict[str, arviz.InferenceData]
        Keyed by display name (M1, M2, M3, ...).
    """
    compare_dict: dict[str, az.InferenceData] = {}
    for model in eligible_models:
        nc_path = baseline_dir / f"{model}_posterior.nc"
        if not nc_path.exists():
            print(
                f"[WARNING] {model}: posterior NetCDF missing at {nc_path} "
                f"despite PROCEED_TO_LOO in convergence_table.csv. Skipping.",
                file=sys.stderr,
            )
            continue
        display_name = MODEL_TO_DISPLAY.get(model, model)
        compare_dict[display_name] = load_netcdf_with_validation(nc_path, model)
    return compare_dict


def main(argv: list[str] | None = None) -> int:
    """Entry point for the SLURM submission and direct CLI use.

    Returns
    -------
    int
        Exit code: 0 = DOMINANT_SINGLE/TOP_TWO/FORCED, 1 = abort (< 2
        eligible), 2 = INCONCLUSIVE_MULTIPLE.
    """
    args = _parse_args(argv)

    print("=" * 72)
    print("Phase 21 Step 21.5 — LOO + stacking + RFX-BMS")
    print("=" * 72)
    print(f"Baseline dir: {args.baseline_dir}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Pareto-k threshold: {args.pareto_k_threshold}")
    print(f"Pareto-k pct threshold: {args.pareto_k_pct_threshold}%")
    print(f"Stacking winner threshold: {args.stacking_winner_threshold}")
    print(f"Combined winner threshold: {args.combined_winner_threshold}")
    print(f"Weak winner threshold: {args.weak_winner_threshold}")
    if args.force_winners:
        print(f"Force winners: {args.force_winners}")
    print("=" * 72)

    # Load convergence-eligible models.
    try:
        eligible_models = _load_convergence_eligible_models(args.baseline_dir)
    except FileNotFoundError as exc:
        print(f"[ABORT] {exc}", file=sys.stderr)
        return 1

    print(
        f"\nFound {len(eligible_models)} model(s) flagged PROCEED_TO_LOO: "
        f"{eligible_models}"
    )

    if len(eligible_models) < MIN_MODELS_FOR_STACKING:
        print(
            f"[ABORT] fewer than 2 models passed convergence gate "
            f"(found {len(eligible_models)}). Stacking requires "
            f">= {MIN_MODELS_FOR_STACKING} models.",
            file=sys.stderr,
        )
        return 1

    # Load posteriors into display-name-keyed dict.
    compare_dict = _load_compare_dict(args.baseline_dir, eligible_models)

    if len(compare_dict) < MIN_MODELS_FOR_STACKING:
        print(
            f"[ABORT] only {len(compare_dict)} posterior NetCDFs loaded "
            f"(expected {len(eligible_models)}). Check warnings above.",
            file=sys.stderr,
        )
        return 1

    # Parse --force-winners.
    force_winners_list: list[str] | None = None
    if args.force_winners:
        force_winners_list = [
            s.strip() for s in args.force_winners.split(",") if s.strip()
        ]

    # Run the full pipeline.
    result = compute_loo_stacking_bms(
        compare_dict,
        pareto_k_threshold=args.pareto_k_threshold,
        pareto_k_pct_threshold=args.pareto_k_pct_threshold,
        stacking_winner_threshold=args.stacking_winner_threshold,
        combined_winner_threshold=args.combined_winner_threshold,
        weak_winner_threshold=args.weak_winner_threshold,
        force_winners=force_winners_list,
    )

    comparison = result["comparison"]
    bms_result = result["bms_result"]
    pct_high_per_model = result["pct_high_per_model"]
    winners = result["winners"]
    winner_type = result["winner_type"]
    participant_ids = result["participant_ids"]
    model_order = result["model_order"]

    # Print top-line summary to stdout.
    print("\nLOO + stacking weights (primary):")
    print(comparison.sort_values("rank").to_string())
    print(f"\nRFX-BMS BOR: {float(bms_result['bor']):.4f}")
    print("PXP per model:")
    for i, name in enumerate(model_order):
        print(f"  {name}: {float(np.asarray(bms_result['pxp'])[i]):.4f}")

    print(f"\nWinner type: {winner_type}")
    print(f"Winners: {winners}")

    # Write outputs.
    out_dir = Path(args.output_dir)
    _write_loo_stacking_csv(comparison, out_dir / "loo_stacking_results.csv")
    _write_rfx_bms_csv(bms_result, model_order, out_dir / "rfx_bms_pxp.csv")
    _write_winner_report(
        comparison,
        bms_result,
        pct_high_per_model,
        winners,
        winner_type,
        model_order,
        participant_ids,
        args.pareto_k_threshold,
        args.pareto_k_pct_threshold,
        out_dir / "winner_report.md",
    )
    (out_dir / "winners.txt").write_text(
        ",".join(winners) + "\n", encoding="utf-8"
    )

    print(f"\nWrote {out_dir / 'loo_stacking_results.csv'}")
    print(f"Wrote {out_dir / 'rfx_bms_pxp.csv'}")
    print(f"Wrote {out_dir / 'winner_report.md'}")
    print(f"Wrote {out_dir / 'winners.txt'} ({','.join(winners)})")

    # Exit-code mapping.
    if winner_type == "INCONCLUSIVE_MULTIPLE":
        print(
            "\n[CHECKPOINT] Inconclusive winner set — exit code 2. "
            "Review winner_report.md and rerun with --force-winners to "
            "resume the pipeline.",
            file=sys.stderr,
        )
        return 2
    # DOMINANT_SINGLE, TOP_TWO, FORCED all auto-advance.
    print("\n[OK] Winner verdict decisive — exit code 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
