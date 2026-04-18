"""Level-2 design matrix builder for hierarchical regression.

Single source of truth for the standardized design matrix used in all
Level-2 (group-level) regression models in Phase 16 (L2-03).

ROADMAP DEVIATION 1 — LEC-5 Subcategory Data Gap (L2-04)
----------------------------------------------------------
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

ROADMAP DEVIATION 2 — IES-R Subscale Linear Dependence (4 predictors, not 5)
-----------------------------------------------------------------------------
An earlier draft of this module targeted 5 predictors, including all three
IES-R subscale residuals (intrusion, avoidance, hyperarousal). This is
mathematically infeasible because in the dataset:

    ies_total = ies_intrusion + ies_avoidance + ies_hyperarousal  (exact)

After Gram-Schmidt residualization of all three subscales against ies_total,
the three residuals satisfy:

    intr_resid + avd_resid + hyp_resid = 0  (exactly)

This makes the three-residual sub-matrix rank-2 with a near-infinite
condition number (~2.4e15). Only 2 of the 3 residualized subscales carry
independent information.

Decision: use **4 predictors** with intrusion and avoidance residuals:
  1. ``lec_total``       — LEC-5 total events (``less_total_events``)
  2. ``iesr_total``      — IES-R total score (``ies_total``)
  3. ``iesr_intr_resid`` — IES-R intrusion, Gram-Schmidt residualized vs. total
  4. ``iesr_avd_resid``  — IES-R avoidance, Gram-Schmidt residualized vs. total

Hyperarousal residual is omitted because:
    iesr_hyp_resid = -(iesr_intr_resid + iesr_avd_resid)  (by construction)
It carries no unique variance beyond intrusion and avoidance residuals.

Intrusion and avoidance are retained (rather than avoidance and hyperarousal,
or intrusion and hyperarousal) because they map to theoretically distinct
trauma symptom clusters: re-experiencing (intrusion) vs. effortful avoidance
(avoidance). Hyperarousal is physiological but its unique variance is
recovered via the combination of the other two.

Orthogonalization rationale (L2-02)
------------------------------------
IES-R total is highly correlated with each of its subscales (r > 0.90),
creating near-multicollinearity in any regression that includes both.
Gram-Schmidt residualization projects each subscale against the mean-centred
IES-R total, ensuring the residualized subscales are orthogonal to the total:

    residual_i = subscale_i_centred
                 - (subscale_i_centred · total_centred) / (total_centred · total_centred)
                 * total_centred

After residualization, each column is z-scored. The resulting 4-column full
design matrix has condition number ~11 (well below the 30 target).

Usage
-----
>>> import pandas as pd
>>> from scripts.fitting.level2_design import build_level2_design_matrix
>>> metrics = pd.read_csv("output/summary_participant_metrics.csv")
>>> complete = metrics.dropna(subset=["ies_total","ies_intrusion","ies_avoidance",
...                                   "ies_hyperarousal","less_total_events"])
>>> participant_ids = sorted(complete["sona_id"].unique().tolist())
>>> X, names = build_level2_design_matrix(metrics, participant_ids)
>>> X.shape  # (n_participants, 4)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Expected covariate names for the 4-predictor design matrix.
COVARIATE_NAMES: list[str] = [
    "lec_total",
    "iesr_total",
    "iesr_intr_resid",
    "iesr_avd_resid",
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
# Internal helpers
# ---------------------------------------------------------------------------


def _residualize(subscale_c: np.ndarray, reference_c: np.ndarray) -> np.ndarray:
    """Gram-Schmidt projection: remove reference direction from subscale.

    Both arrays must be mean-centred before calling this function.

    Parameters
    ----------
    subscale_c : np.ndarray
        Mean-centred subscale values, shape (n,).
    reference_c : np.ndarray
        Mean-centred reference vector, shape (n,).

    Returns
    -------
    np.ndarray
        Residualized subscale orthogonal to reference_c, shape (n,).
    """
    proj_coeff = np.dot(subscale_c, reference_c) / (
        np.dot(reference_c, reference_c) + 1e-12
    )
    return subscale_c - proj_coeff * reference_c


def _zscore(x: np.ndarray) -> np.ndarray:
    """Z-score an array: (x - mean) / (std + 1e-8)."""
    return (x - x.mean()) / (x.std() + 1e-8)


# ---------------------------------------------------------------------------
# Core design matrix builder
# ---------------------------------------------------------------------------


def build_level2_design_matrix(
    metrics_df: pd.DataFrame,
    participant_ids: list,
    include_lec_subcategories: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Build the standardized Level-2 design matrix for hierarchical regression.

    Constructs a **4-predictor** design matrix (see module docstring for the
    two ROADMAP deviations: LEC-5 subcategory data gap and IES-R subscale
    linear dependence).

    IES-R intrusion and avoidance subscales are Gram-Schmidt residualized
    against the IES-R total before z-scoring. The hyperarousal residual is
    omitted because it equals ``-(intr_resid + avd_resid)`` by construction
    (ies_total = ies_intrusion + ies_avoidance + ies_hyperarousal exactly),
    making it linearly dependent on the other two residuals.

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
        subcategory columns are not available in the data (ROADMAP Deviation 1).

    Returns
    -------
    X : np.ndarray
        Shape ``(n_participants, 4)``. All columns are z-scored. Column
        order matches ``COVARIATE_NAMES``:
        ``["lec_total", "iesr_total", "iesr_intr_resid", "iesr_avd_resid"]``.
    covariate_names : list[str]
        Names matching the columns of ``X``. Used for naming ``beta_*``
        sites in the NumPyro hierarchical models.

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
    ROADMAP Deviation 1 (L2-04): The Phase 16 specification called for 6+
    predictors including LEC-5 physical/sexual/accident subcategories.
    These are absent from the data — only ``less_total_events`` and
    ``less_personal_events`` exist.

    ROADMAP Deviation 2: The Phase 16 specification implied 5 predictors
    (lec_total + iesr_total + 3 subscale residuals). Because
    ``ies_total = ies_intrusion + ies_avoidance + ies_hyperarousal`` exactly
    in this dataset, all three subscale residuals are linearly dependent
    (their sum is zero by construction). The design uses 4 predictors,
    dropping the hyperarousal residual.
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
    aligned = metrics_df.set_index("sona_id").reindex(participant_ids)

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

    # ------------------------------------------------------------------
    # Gram-Schmidt residualization of intrusion and avoidance
    # (hyperarousal residual = -(intr_resid + avd_resid); omitted)
    # ------------------------------------------------------------------
    total_c = iesr_total - iesr_total.mean()
    intr_resid = _residualize(iesr_intr - iesr_intr.mean(), total_c)
    avd_resid = _residualize(iesr_avd - iesr_avd.mean(), total_c)

    # ------------------------------------------------------------------
    # Z-score all columns and stack
    # ------------------------------------------------------------------
    X = np.column_stack([
        _zscore(lec_total),
        _zscore(iesr_total),
        _zscore(intr_resid),
        _zscore(avd_resid),
    ])  # shape (n_participants, 4)

    return X, COVARIATE_NAMES


#: Expected covariate names for the Phase 21 2-predictor design matrix.
COVARIATE_NAMES_2COV: list[str] = ["lec_total", "iesr_total"]


def build_level2_design_matrix_2cov(
    metrics: pd.DataFrame,
    participant_ids: list[int | str],
) -> tuple[np.ndarray, list[str]]:
    """Build the 2-covariate L2 design matrix for M3/M5/M6a Phase 21 winners.

    Returns columns ``['lec_total', 'iesr_total']``, both z-scored across
    participants. Used by plan 21-07 when a Phase 21 winner is M3, M5, or
    M6a; M6b winners use ``build_level2_design_matrix`` (4 covariates)
    instead. M1/M2 bypass L2 entirely.

    Unlike the 4-covariate builder, this variant does NOT residualize the
    IES-R total against any subscale — it uses raw (z-scored) totals
    because only 2 covariates are present and collinearity between
    ``less_total_events`` and ``ies_total`` is moderate (r ~ 0.3 in the
    canonical N=138 cohort, well below the multicollinearity threshold).

    Parameters
    ----------
    metrics : pd.DataFrame
        Per-participant metric CSV (``output/summary_participant_metrics.csv``).
        Must contain columns ``sona_id``, ``less_total_events``, ``ies_total``.
    participant_ids : list of int or str
        Ordered participant ids; determines row order of the returned matrix.
        Must match the order produced by
        ``prepare_stacked_participant_data`` (i.e.,
        ``sorted(data_df[participant_col].unique())``).

    Returns
    -------
    design : np.ndarray, shape ``(n_participants, 2)``
        Column 0 = z-scored ``lec_total``; column 1 = z-scored ``iesr_total``.
        Both columns have mean ~0 and std ~1 across participants.
    names : list of str
        ``['lec_total', 'iesr_total']``.

    Raises
    ------
    ValueError
        If ``metrics`` is missing any of ``sona_id``, ``less_total_events``,
        ``ies_total``, or if any participant_id is absent from ``metrics``
        after reindexing.
    """
    required = ["sona_id", "less_total_events", "ies_total"]
    missing = [c for c in required if c not in metrics.columns]
    if missing:
        raise ValueError(
            f"build_level2_design_matrix_2cov: metrics is missing required "
            f"columns {missing}. Expected: {required}. Available columns: "
            f"{list(metrics.columns)}"
        )

    aligned = metrics.set_index("sona_id").reindex(participant_ids)

    missing_mask = (
        aligned["less_total_events"].isna() | aligned["ies_total"].isna()
    )
    missing_pids = aligned.index[missing_mask].tolist()
    if missing_pids:
        raise ValueError(
            f"build_level2_design_matrix_2cov: expected {len(participant_ids)} "
            f"participants with complete lec_total + iesr_total, got "
            f"{len(participant_ids) - len(missing_pids)} complete. Missing "
            f"data for participant ids (first 5): {missing_pids[:5]}"
        )

    lec_total_raw = aligned["less_total_events"].to_numpy(dtype=float)
    iesr_total_raw = aligned["ies_total"].to_numpy(dtype=float)

    design = np.column_stack([
        _zscore(lec_total_raw),
        _zscore(iesr_total_raw),
    ])  # shape (n_participants, 2)

    return design, list(COVARIATE_NAMES_2COV)


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
        ``full_design_condition_number`` : float
            Condition number of the full 4-column design matrix
            [lec_total, iesr_total, iesr_intr_resid, iesr_avd_resid].
            This is the operationally relevant condition number.
        ``lec_iesr_correlation`` : float
            Pearson r between ``less_total_events`` and ``ies_total``.
        ``n_participants`` : int
            Number of participants used (after dropping NaN rows).
        ``pass_verdict`` : bool
            True iff ``full_design_condition_number < 30``.
        ``target_condition_number`` : float
            The threshold (30.0) used for PASS/FAIL.
        ``subscales_sum_to_total`` : bool
            True if ies_intrusion + ies_avoidance + ies_hyperarousal
            equals ies_total exactly (explains rank deficiency).
    """
    required = [
        "ies_total", "ies_intrusion", "ies_avoidance", "ies_hyperarousal",
        "less_total_events",
    ]
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

    # Check if subscales sum exactly to total
    subscales_sum = iesr_intr + iesr_avd + iesr_hyp
    sums_to_total = bool(np.allclose(subscales_sum, iesr_total, atol=1e-6))

    # Raw correlation matrix of the three subscales
    X_raw = np.column_stack([iesr_intr, iesr_avd, iesr_hyp])
    corr_matrix = np.corrcoef(X_raw.T)
    cond_raw = float(np.linalg.cond(X_raw))

    # Gram-Schmidt residualization of intrusion and avoidance
    total_c = iesr_total - iesr_total.mean()
    intr_resid = _residualize(iesr_intr - iesr_intr.mean(), total_c)
    avd_resid = _residualize(iesr_avd - iesr_avd.mean(), total_c)

    # Full 4-column design matrix condition number
    X_full = np.column_stack([
        lec_total - lec_total.mean(),
        total_c,
        intr_resid,
        avd_resid,
    ])
    cond_full = float(np.linalg.cond(X_full))

    # LEC–IES-R total correlation
    lec_iesr_r = float(np.corrcoef(lec_total, iesr_total)[0, 1])

    target = 30.0
    return {
        "raw_corr_matrix": corr_matrix,
        "raw_condition_number": cond_raw,
        "full_design_condition_number": cond_full,
        "lec_iesr_correlation": lec_iesr_r,
        "n_participants": n,
        "pass_verdict": cond_full < target,
        "target_condition_number": target,
        "subscales_sum_to_total": sums_to_total,
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
    cond_full = audit["full_design_condition_number"]
    lec_r = audit["lec_iesr_correlation"]
    n = audit["n_participants"]
    verdict = "PASS" if audit["pass_verdict"] else "FAIL"
    target = audit["target_condition_number"]
    sums_flag = audit["subscales_sum_to_total"]

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
        "## IES-R Subscale Sum Structure",
        "",
        f"**Subscales sum exactly to IES-R total:** `{sums_flag}`",
        "",
        (
            "Because `ies_total = ies_intrusion + ies_avoidance + ies_hyperarousal` "
            "exactly in this dataset, after Gram-Schmidt residualization all three "
            "subscale residuals satisfy `intr_resid + avd_resid + hyp_resid = 0`. "
            "The three-residual sub-matrix is rank-2 with condition number ~2.4e15."
            if sums_flag
            else "Subscales do not sum exactly to IES-R total; all three residuals "
            "are linearly independent and can be included."
        ),
        "",
        "---",
        "",
        "## Condition Numbers",
        "",
        f"- **Raw subscale matrix** [intrusion, avoidance, hyperarousal]: **{cond_raw:.2f}**",
        "- **3-residual matrix** [intr_resid, avd_resid, hyp_resid]: "
        f"**~2.4e15** (rank-deficient; hyperarousal residual = -(intr_resid + avd_resid))",
        f"- **Full 4-column design** [lec_total, iesr_total, intr_resid, avd_resid]: **{cond_full:.2f}**",
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
        "## ROADMAP Deviation 1: 5 Predictors Reduced to 4 Due to Linear Dependence",
        "",
        "The Phase 16 specification implied 5 predictors: lec_total + iesr_total + "
        "3 subscale residuals. Because the three subscale residuals are linearly "
        "dependent (their sum is exactly zero), only 2 can enter the design. "
        "The hyperarousal residual is dropped; intrusion and avoidance are retained "
        "as they map to distinct theoretical symptom clusters (re-experiencing vs. "
        "effortful avoidance).",
        "",
        "**Final predictor set (4 predictors):**",
        "",
        "1. `lec_total` — LEC-5 total events (`less_total_events`)",
        "2. `iesr_total` — IES-R total score (`ies_total`)",
        "3. `iesr_intr_resid` — IES-R intrusion, residualized vs. IES-R total",
        "4. `iesr_avd_resid` — IES-R avoidance, residualized vs. IES-R total",
        "",
        "---",
        "",
        "## ROADMAP Deviation 2: LEC-5 Subcategory Columns Unavailable",
        "",
        "The Phase 16 ROADMAP specified 6+ predictors including LEC-5 "
        "physical/sexual/accident subcategories. These columns are **not available** "
        "in the current data pipeline:",
        "",
        "- `output/summary_participant_metrics.csv`: only `less_total_events` and "
        "`less_personal_events`.",
        "- `scripts/utils/scoring_functions.py::score_less()`: computes only totals, "
        "no subcategory taxonomy.",
        "",
        "---",
        "",
        "## Verdict",
        "",
        f"**{verdict}** — Full 4-column design matrix condition number {cond_full:.2f} is "
        f"{'below' if audit['pass_verdict'] else 'AT OR ABOVE'} the target of "
        f"{target:.0f}.",
        "",
    ]

    if not audit["pass_verdict"]:
        lines += [
            "> **ACTION REQUIRED:** Condition number after residualization exceeds 30.",
            "> The orthogonalization strategy must be revised before Level-2 fits proceed.",
            "",
        ]
    else:
        lines += [
            "The Gram-Schmidt residualization produces a well-conditioned design matrix.",
            "The 4-predictor design is approved for Level-2 regression.",
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
    print(f"  Subscales sum to total:          {audit['subscales_sum_to_total']}")
    print(f"  Raw condition number:             {audit['raw_condition_number']:.2f}")
    print(f"  Full design condition number:     {audit['full_design_condition_number']:.2f}")
    print(f"  LEC-5 vs. IES-R total r:          {audit['lec_iesr_correlation']:.3f}")
    verdict_str = "PASS" if audit["pass_verdict"] else "FAIL"
    print(f"  Verdict:                          {verdict_str}")

    print(f"\nWriting audit report to: {REPORT_PATH}")
    write_collinearity_report(audit, REPORT_PATH, metrics_df=metrics)
    print("  Done.")

    # ------------------------------------------------------------------
    # Verify design matrix on real data
    # ------------------------------------------------------------------
    complete_mask = metrics[
        ["ies_total", "ies_intrusion", "ies_avoidance", "ies_hyperarousal",
         "less_total_events"]
    ].notna().all(axis=1)
    complete_ids = sorted(
        metrics.loc[complete_mask, "sona_id"].dropna().unique().tolist()
    )
    print(f"\nBuilding design matrix for {len(complete_ids)} participants ...")

    X, names = build_level2_design_matrix(metrics, complete_ids)
    print(f"  Design matrix shape: {X.shape}")
    print(f"  Covariate names: {names}")
    print(f"  NaN count: {int(np.isnan(X).sum())}")
    print(f"  Condition number: {np.linalg.cond(X):.2f}")
    print(f"  First 3 rows:\n{X[:3]}")
    print(f"  Column means (should be ~0): {X.mean(axis=0).round(4)}")
    print(f"  Column stds  (should be ~1): {X.std(axis=0).round(4)}")
