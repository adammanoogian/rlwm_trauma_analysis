"""Level-2 design matrix builder for hierarchical regression.

Single source of truth for the standardized design matrix used in all
Level-2 (group-level) regression models in Phase 16 (L2-03).

ROADMAP DEVIATION — LEC-5 Subcategory Data Gap (L2-04)
-------------------------------------------------------
The Phase 16 ROADMAP specified a 6-predictor design matrix that included
LEC-5 physical/sexual/accident subcategories as a sixth predictor. After
auditing the raw data pipeline, these subcategories are NOT available:

  - ``scripts/utils/scoring_functions.py::score_less()`` only computes
    ``less_total_events`` and ``less_personal_events`` (total event counts).
  - The raw item-level columns (``s1_item01`` through ``s1_item15``) in
    ``output/parsed_survey1_all.csv`` are not mapped to a
    physical/sexual/accident taxonomy. No such mapping exists in the
    codebase.
  - The summary CSV ``output/summary_participant_metrics.csv`` contains only
    ``less_total_events`` and ``less_personal_events`` as LEC columns.

Decision: proceed with **5 predictors** instead of 6:
  1. ``lec_total``       — LEC-5 total events (``less_total_events``)
  2. ``iesr_total``      — IES-R total score (``ies_total``)
  3. ``iesr_intr_resid`` — IES-R intrusion, Gram-Schmidt residualized vs. total
  4. ``iesr_avd_resid``  — IES-R avoidance, Gram-Schmidt residualized vs. total
  5. ``iesr_hyp_resid``  — IES-R hyperarousal, Gram-Schmidt residualized vs. total

This design is documented as a deviation in:
  - ``16-01-SUMMARY.md`` (Phase 16, Plan 01 execution summary)
  - This module docstring and the ``build_level2_design_matrix`` docstring
  - The ``include_lec_subcategories`` parameter which raises ``ValueError``
    if set to True, explicitly blocking any caller that assumes subcategories
    are available.

Orthogonalization rationale (L2-02)
------------------------------------
IES-R total is highly correlated with each of its three subscales
(intrusion, avoidance, hyperarousal), creating near-multicollinearity.
Gram-Schmidt residualization projects each subscale against IES-R total,
ensuring the residualized subscales are orthogonal to the total by
construction. This reduces the condition number of the design sub-matrix
from a potentially large value to a value well below the target of 30.

The residualization is performed on mean-centred raw values before z-scoring:
    residual = subscale_centred - proj(subscale_centred, iesr_total_centred)
where proj(u, v) = (u·v / v·v) * v.

After residualization, each column (including lec_total and iesr_total) is
z-scored: (x - mean) / (std + 1e-8).

Usage
-----
>>> import pandas as pd
>>> from scripts.fitting.level2_design import build_level2_design_matrix
>>> metrics = pd.read_csv("output/summary_participant_metrics.csv")
>>> participant_ids = sorted(metrics["sona_id"].dropna().unique().tolist())
>>> X, names = build_level2_design_matrix(metrics, participant_ids)
>>> X.shape  # (n_participants, 5)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Expected covariate names for the 5-predictor design matrix.
COVARIATE_NAMES: list[str] = [
    "lec_total",
    "iesr_total",
    "iesr_intr_resid",
    "iesr_avd_resid",
    "iesr_hyp_resid",
]

#: Required columns in the metrics DataFrame.
_REQUIRED_COLS: list[str] = [
    "sona_id",
    "ies_total",
    "ies_intrusion",
    "ies_avoidance",
    "ies_hyperarousal",
    "less_total_events",
]


# ---------------------------------------------------------------------------
# Core design matrix builder
# ---------------------------------------------------------------------------


def build_level2_design_matrix(
    metrics_df: pd.DataFrame,
    participant_ids: list,
    include_lec_subcategories: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Build the standardized Level-2 design matrix for hierarchical regression.

    Constructs a 5-predictor design matrix (see module docstring for the
    ROADMAP deviation explaining why 5 predictors are used instead of 6).

    IES-R subscales are Gram-Schmidt residualized against the IES-R total
    before z-scoring, ensuring the subscale columns are orthogonal to the
    total score by construction.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Participant metrics with columns: ``sona_id``, ``ies_total``,
        ``ies_intrusion``, ``ies_avoidance``, ``ies_hyperarousal``,
        ``less_total_events``.
    participant_ids : list
        Sorted participant IDs. Must match the order produced by
        ``prepare_stacked_participant_data`` (i.e.,
        ``sorted(data_df[participant_col].unique())``).
    include_lec_subcategories : bool, optional
        If ``True``, raise ``ValueError`` — LEC-5 physical/sexual/accident
        subcategory columns are not available in the data. Kept as a
        parameter to make the data gap explicit at call sites.

    Returns
    -------
    X : np.ndarray
        Shape ``(n_participants, 5)``. All columns are z-scored. Column
        order matches ``COVARIATE_NAMES``.
    covariate_names : list[str]
        Names matching the columns of ``X``. Used for naming
        ``beta_*`` sites in the NumPyro hierarchical models.

    Raises
    ------
    ValueError
        If ``include_lec_subcategories=True`` (data unavailable) or if
        any required column is missing from ``metrics_df``.
    RuntimeError
        If the aligned metrics DataFrame contains NaN values after
        reindexing to ``participant_ids``.

    Notes
    -----
    ROADMAP DEVIATION (L2-04): The Phase 16 specification called for 6
    predictors including LEC-5 physical/sexual/accident subcategories.
    These are absent from the data — only ``less_total_events`` and
    ``less_personal_events`` exist. The design uses 5 predictors.

    Gram-Schmidt residualization:
        resid_i = subscale_i_centred
                  - (subscale_i_centred · total_centred) / (total_centred · total_centred)
                  * total_centred
    The dot products use mean-centred vectors, so the residual is orthogonal
    to the total score in the centred space.
    """
    # ------------------------------------------------------------------
    # Guard: subcategory data gap
    # ------------------------------------------------------------------
    if include_lec_subcategories:
        raise ValueError(
            "include_lec_subcategories=True requested, but LEC-5 physical/"
            "sexual/accident subcategory columns are not available in the "
            "current data pipeline. Only 'less_total_events' and "
            "'less_personal_events' exist in output/summary_participant_metrics.csv. "
            "See level2_design.py module docstring for details."
        )

    # ------------------------------------------------------------------
    # Guard: required columns
    # ------------------------------------------------------------------
    missing = [c for c in _REQUIRED_COLS if c not in metrics_df.columns]
    if missing:
        raise ValueError(
            f"build_level2_design_matrix: metrics_df is missing required "
            f"columns {missing}. Available columns: {list(metrics_df.columns)}"
        )

    # ------------------------------------------------------------------
    # Align to participant_ids
    # ------------------------------------------------------------------
    aligned = (
        metrics_df.set_index("sona_id")
        .reindex(participant_ids)
    )

    missing_pids = aligned.index[aligned["ies_total"].isna()].tolist()
    if missing_pids:
        raise RuntimeError(
            f"build_level2_design_matrix: {len(missing_pids)} participant_ids "
            f"have no data in metrics_df after reindex: {missing_pids[:5]}..."
        )

    # ------------------------------------------------------------------
    # Extract raw arrays (float64 for numerical stability)
    # ------------------------------------------------------------------
    lec_total: np.ndarray = aligned["less_total_events"].to_numpy(dtype=float)
    iesr_total: np.ndarray = aligned["ies_total"].to_numpy(dtype=float)
    iesr_intr: np.ndarray = aligned["ies_intrusion"].to_numpy(dtype=float)
    iesr_avd: np.ndarray = aligned["ies_avoidance"].to_numpy(dtype=float)
    iesr_hyp: np.ndarray = aligned["ies_hyperarousal"].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # Gram-Schmidt residualization of subscales against IES-R total
    # ------------------------------------------------------------------
    # Centre both vectors before projection (centre-first then residualize,
    # then z-score post-residualization).
    total_c = iesr_total - iesr_total.mean()

    def _residualize(subscale: np.ndarray, reference_c: np.ndarray) -> np.ndarray:
        """Project out the reference direction from the subscale."""
        sub_c = subscale - subscale.mean()
        proj_coeff = np.dot(sub_c, reference_c) / (np.dot(reference_c, reference_c) + 1e-12)
        return sub_c - proj_coeff * reference_c

    intr_resid = _residualize(iesr_intr, total_c)
    avd_resid = _residualize(iesr_avd, total_c)
    hyp_resid = _residualize(iesr_hyp, total_c)

    # ------------------------------------------------------------------
    # Z-score all columns
    # ------------------------------------------------------------------
    def _zscore(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / (x.std() + 1e-8)

    columns = [
        _zscore(lec_total),
        _zscore(iesr_total),
        _zscore(intr_resid),
        _zscore(avd_resid),
        _zscore(hyp_resid),
    ]

    X = np.column_stack(columns)  # shape (n_participants, 5)
    return X, COVARIATE_NAMES


# ---------------------------------------------------------------------------
# Collinearity audit
# ---------------------------------------------------------------------------


def run_collinearity_audit(metrics_df: pd.DataFrame) -> dict:
    """Compute IES-R subscale collinearity diagnostics before and after residualization.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Participant metrics with IES-R and LEC columns. Rows with any NaN
        in the IES-R or LEC columns are dropped before computation.

    Returns
    -------
    dict
        Keys:

        ``raw_corr_matrix`` : np.ndarray, shape (3, 3)
            Pearson correlation matrix of
            [ies_intrusion, ies_avoidance, ies_hyperarousal].
        ``raw_condition_number`` : float
            Condition number of the raw [intrusion, avoidance, hyperarousal]
            matrix (one column per subscale, one row per participant).
        ``resid_condition_number`` : float
            Condition number of the residualized subscale matrix (post Gram-Schmidt).
        ``lec_iesr_correlation`` : float
            Pearson r between ``less_total_events`` and ``ies_total``.
        ``n_participants`` : int
            Number of participants used (after dropping NaN rows).
        ``pass_verdict`` : bool
            True iff ``resid_condition_number < 30``.
        ``target_condition_number`` : float
            The threshold (30.0) used for PASS/FAIL.
    """
    required = ["ies_total", "ies_intrusion", "ies_avoidance", "ies_hyperarousal",
                "less_total_events"]
    missing = [c for c in required if c not in metrics_df.columns]
    if missing:
        raise ValueError(
            f"run_collinearity_audit: missing columns {missing}. "
            f"Available: {list(metrics_df.columns)}"
        )

    df = metrics_df[required].dropna()
    n = len(df)

    iesr_total = df["ies_total"].to_numpy(dtype=float)
    iesr_intr = df["ies_intrusion"].to_numpy(dtype=float)
    iesr_avd = df["ies_avoidance"].to_numpy(dtype=float)
    iesr_hyp = df["ies_hyperarousal"].to_numpy(dtype=float)
    lec_total = df["less_total_events"].to_numpy(dtype=float)

    # Raw correlation matrix
    X_raw = np.column_stack([iesr_intr, iesr_avd, iesr_hyp])
    corr_matrix = np.corrcoef(X_raw.T)
    cond_raw = float(np.linalg.cond(X_raw))

    # Gram-Schmidt residualization
    total_c = iesr_total - iesr_total.mean()

    def _residualize(subscale: np.ndarray, reference_c: np.ndarray) -> np.ndarray:
        sub_c = subscale - subscale.mean()
        proj = np.dot(sub_c, reference_c) / (np.dot(reference_c, reference_c) + 1e-12)
        return sub_c - proj * reference_c

    intr_r = _residualize(iesr_intr, total_c)
    avd_r = _residualize(iesr_avd, total_c)
    hyp_r = _residualize(iesr_hyp, total_c)

    X_resid = np.column_stack([intr_r, avd_r, hyp_r])
    cond_resid = float(np.linalg.cond(X_resid))

    # LEC–IES-R total correlation
    lec_iesr_r = float(np.corrcoef(lec_total, iesr_total)[0, 1])

    target = 30.0
    return {
        "raw_corr_matrix": corr_matrix,
        "raw_condition_number": cond_raw,
        "resid_condition_number": cond_resid,
        "lec_iesr_correlation": lec_iesr_r,
        "n_participants": n,
        "pass_verdict": cond_resid < target,
        "target_condition_number": target,
    }


# ---------------------------------------------------------------------------
# Audit report writer
# ---------------------------------------------------------------------------


def write_collinearity_report(
    audit: dict,
    output_path: str,
    metrics_df: pd.DataFrame | None = None,
) -> None:
    """Write the IES-R collinearity audit report to a Markdown file.

    Parameters
    ----------
    audit : dict
        Output from ``run_collinearity_audit``.
    output_path : str
        Path to the output ``.md`` file. Parent directories are created
        if they do not exist.
    metrics_df : pd.DataFrame or None, optional
        If provided, used to display descriptive statistics in the report.
    """
    import pathlib

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    corr = audit["raw_corr_matrix"]
    cond_raw = audit["raw_condition_number"]
    cond_resid = audit["resid_condition_number"]
    lec_r = audit["lec_iesr_correlation"]
    n = audit["n_participants"]
    verdict = "PASS" if audit["pass_verdict"] else "FAIL"
    target = audit["target_condition_number"]

    lines = [
        "# IES-R Subscale Collinearity Audit (L2-02)",
        "",
        f"**N participants (complete IES-R + LEC data):** {n}",
        "",
        "---",
        "",
        "## Raw Subscale Correlations",
        "",
        "Pearson *r* between IES-R subscales (before residualization):",
        "",
        "| | Intrusion | Avoidance | Hyperarousal |",
        "|---|---|---|---|",
        f"| **Intrusion** | {corr[0,0]:.3f} | {corr[0,1]:.3f} | {corr[0,2]:.3f} |",
        f"| **Avoidance** | {corr[1,0]:.3f} | {corr[1,1]:.3f} | {corr[1,2]:.3f} |",
        f"| **Hyperarousal** | {corr[2,0]:.3f} | {corr[2,1]:.3f} | {corr[2,2]:.3f} |",
        "",
        "---",
        "",
        "## Condition Numbers",
        "",
        f"- **Raw subscale matrix** [intrusion, avoidance, hyperarousal]: **{cond_raw:.1f}**",
        f"- **Residualized subscale matrix** [intr_resid, avd_resid, hyp_resid]: **{cond_resid:.1f}**",
        f"- **Target:** < {target:.0f}",
        "",
        "---",
        "",
        "## Contextual Correlations",
        "",
        f"- **LEC-5 total events vs. IES-R total:** *r* = {lec_r:.3f}",
        "",
        "---",
        "",
        "## ROADMAP Deviation: 5 Predictors Used (not 6)",
        "",
        "The Phase 16 ROADMAP specified 6 predictors including LEC-5 physical/sexual/"
        "accident subcategory columns. These subcategory columns are **not available** "
        "in the current data pipeline:",
        "",
        "- `output/summary_participant_metrics.csv` contains only `less_total_events` "
        "and `less_personal_events`.",
        "- `scripts/utils/scoring_functions.py::score_less()` does not define a "
        "physical/sexual/accident taxonomy.",
        "- Raw item-level columns (`s1_item01` through `s1_item15`) in "
        "`output/parsed_survey1_all.csv` are not mapped to trauma subcategories.",
        "",
        "**Decision:** Proceed with **5 predictors**:",
        "",
        "1. `lec_total` — LEC-5 total events (`less_total_events`)",
        "2. `iesr_total` — IES-R total score (`ies_total`)",
        "3. `iesr_intr_resid` — IES-R intrusion, residualized vs. IES-R total",
        "4. `iesr_avd_resid` — IES-R avoidance, residualized vs. IES-R total",
        "5. `iesr_hyp_resid` — IES-R hyperarousal, residualized vs. IES-R total",
        "",
        "---",
        "",
        "## Verdict",
        "",
        f"**{verdict}** — Residualized condition number {cond_resid:.1f} is "
        f"{'below' if audit['pass_verdict'] else 'AT OR ABOVE'} the target of "
        f"{target:.0f}.",
        "",
    ]

    if not audit["pass_verdict"]:
        lines += [
            "> **ACTION REQUIRED:** Condition number after residualization exceeds 30.",
            "> The Gram-Schmidt orthogonalization strategy must be revised before",
            "> any Level-2 fits proceed. Consider PCA-based orthogonalization or",
            "> dropping the highest-correlated subscale.",
            "",
        ]
    else:
        lines += [
            "The Gram-Schmidt residualization successfully reduces multicollinearity.",
            "The design matrix is well-conditioned for Level-2 regression.",
            "",
        ]

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main: run audit and verify design matrix on real data
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import pathlib

    DATA_PATH = "output/summary_participant_metrics.csv"
    REPORT_PATH = "output/bayesian/level2/ies_r_collinearity_audit.md"

    print(f"Loading metrics from: {DATA_PATH}")
    metrics = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(metrics)} rows, {len(metrics.columns)} columns")

    # ------------------------------------------------------------------
    # Run collinearity audit
    # ------------------------------------------------------------------
    print("\nRunning IES-R collinearity audit ...")
    audit = run_collinearity_audit(metrics)

    print(f"  N participants (complete data): {audit['n_participants']}")
    print(f"  Raw condition number:           {audit['raw_condition_number']:.2f}")
    print(f"  Residualized condition number:  {audit['resid_condition_number']:.2f}")
    print(f"  LEC-5 vs. IES-R total r:        {audit['lec_iesr_correlation']:.3f}")
    verdict_str = "PASS" if audit["pass_verdict"] else "FAIL"
    print(f"  Verdict:                        {verdict_str}")

    print(f"\nWriting audit report to: {REPORT_PATH}")
    write_collinearity_report(audit, REPORT_PATH, metrics_df=metrics)
    print("  Done.")

    # ------------------------------------------------------------------
    # Verify design matrix on real N=154 data
    # ------------------------------------------------------------------
    participant_ids = sorted(metrics["sona_id"].dropna().unique().tolist())
    print(f"\nBuilding design matrix for {len(participant_ids)} participants ...")

    # Use only participants with complete IES-R + LEC data
    complete_mask = metrics[
        ["ies_total", "ies_intrusion", "ies_avoidance", "ies_hyperarousal", "less_total_events"]
    ].notna().all(axis=1)
    complete_ids = sorted(
        metrics.loc[complete_mask, "sona_id"].dropna().unique().tolist()
    )
    print(f"  Participants with complete data: {len(complete_ids)}")

    X, names = build_level2_design_matrix(metrics, complete_ids)
    print(f"  Design matrix shape: {X.shape}")
    print(f"  Covariate names: {names}")
    print(f"  NaN count: {int(np.isnan(X).sum())}")
    print(f"  First 3 rows:\n{X[:3]}")
    print(f"  Column means (should be ~0): {X.mean(axis=0).round(4)}")
    print(f"  Column stds  (should be ~1): {X.std(axis=0).round(4)}")
