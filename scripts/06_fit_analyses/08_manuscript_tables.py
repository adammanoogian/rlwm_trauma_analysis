"""Step 21.9 — manuscript tables and figures for the Phase 21 Bayesian pipeline.

Consolidates the outputs of ``models/bayesian/21_baseline/`` and
``models/bayesian/21_l2/`` into publication-ready Tables 1/2/3 (CSV + Markdown
+ LaTeX) and Figure 1 (forest plot of winners' Level-2 beta coefficients).

Citations woven into the methods paragraph and the table captions:
    - Baribault, B. & Collins, A.G.E. (2023). Troubleshooting Bayesian
      cognitive models. *Psychological Methods*. DOI ``10.1037/met0000554``.
    - Hess, A.S., Joseph, J.W., Cribb, J. et al. (2025). A staged Bayesian
      workflow for computational cognitive science. *Computational
      Psychiatry*. DOI ``10.5334/cpsy.116``.
    - Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Using stacking
      to average Bayesian predictive distributions.
      *Bayesian Analysis* 13(3), 917-1003. DOI ``10.1214/17-BA1091``.
    - Rigoux, L., Stephan, K.E., Friston, K.J., & Daunizeau, J. (2014).
      Bayesian model selection for group studies — revisited. *NeuroImage*
      84, 971-985. DOI ``10.1016/j.neuroimage.2013.08.065``.

Pipeline contract
-----------------
This script is the capstone of Phase 21 — it consumes ALL upstream artefacts
produced by steps 21.1 through 21.8 and emits the final manuscript Tables +
Figure. It is intentionally read-only with respect to upstream files so it can
be re-invoked safely for incremental edits.

Inputs
------
``--baseline-dir`` (default ``models/bayesian/21_baseline/``):
    - ``loo_stacking_results.csv`` (step 21.5) — primary LOO + stacking table.
    - ``rfx_bms_pxp.csv`` (step 21.5) — secondary RFX-BMS / PXP table.
    - ``winners.txt`` (step 21.5) — comma-separated winner display names.

``--l2-dir`` (default ``models/bayesian/21_l2/``):
    - ``scale_audit_report.md`` (step 21.7) — YAML front-matter pipeline_action
      header for null-result branch detection + body for narrative.
    - ``{winner}_beta_hdi_table.csv`` per winner (step 21.7).
    - ``averaged_scale_effects.csv`` (step 21.8, multi-winner only) — provides
      ``model_averaged_*`` columns for Table 3.
    - ``{winner}_posterior.nc`` per winner (step 21.6) — for Figure 1 forest
      plot via ``scripts/06_fit_analyses/07_bayesian_level2_effects.py``.

``models/bayesian/wmrl_m6b_subscale_posterior.nc`` — Phase-16 canonical path
    for the M6b subscale exploratory arm. **Guarded with ``Path.exists()``
    check** (plan-checker Issue #9): the subscale arm is fire-and-forget and
    may still be running when 21.9 fires; if missing, the subscale section is
    skipped with a NOTE log line instead of raising.

Outputs
-------
``--tables-dir`` (default ``models/bayesian/21_tables/``):
    - ``table1_loo_stacking.{csv,md,tex}`` — LOO + stacking weights.
    - ``table2_rfx_bms.{csv,md,tex}`` — RFX-BMS PXP/BOR.
    - ``table3_winner_betas.{csv,md,tex}`` — winner beta HDIs (with
      model-averaged columns when multi-winner).
    - ``null_result_summary.md`` (only when audit pipeline_action ==
      ``NULL_RESULT``) — clean null-result narrative.

``--figures-dir`` (default ``figures/21_bayesian/``):
    - ``forest_{winner}.png`` per winner (only when audit pipeline_action !=
      ``NULL_RESULT`` — a forest plot of an all-null result would be
      misleading).

``--paper`` (default ``paper.qmd``):
    - Insert a new ``### Bayesian Model Selection Pipeline`` Methods
      subsection before the locked ``{#sec-bayesian-regression}`` anchor (per
      Phase 18-05 decisions).
    - Adjust the ``M6b is the winning model`` Results sentence to reference
      the stacking-weight-based winner(s).
    - Add Quarto cross-refs ``@tbl-loo-stacking``, ``@tbl-rfx-bms``,
      ``@tbl-winner-betas``, ``@fig-forest-21`` at the relevant Results
      locations.

Use ``--no-paper-edit`` to skip the paper.qmd modification (cluster invocation
runs with this flag — paper.qmd lives in the repo root and edits are reviewed
locally as a Git diff, not on the cluster).
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Add project root to sys.path so config is importable.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (  # noqa: E402
    MODELS_BAYESIAN_BASELINE,
    MODELS_BAYESIAN_DIR,
    MODELS_BAYESIAN_L2,
    REPORTS_FIGURES_BAYESIAN,
)

# Module-level logger; reconfigured in ``main`` so library-style imports do
# not double-log when called from another script.
logger = logging.getLogger("scripts.06_fit_analyses.08_manuscript_tables")


# ---------------------------------------------------------------------------
# Constants and dataclasses
# ---------------------------------------------------------------------------

DISPLAY_TO_INTERNAL: dict[str, str] = {
    "M1": "qlearning",
    "M2": "wmrl",
    "M3": "wmrl_m3",
    "M5": "wmrl_m5",
    "M6a": "wmrl_m6a",
    "M6b": "wmrl_m6b",
}
INTERNAL_TO_DISPLAY: dict[str, str] = {v: k for k, v in DISPLAY_TO_INTERNAL.items()}

# Models with no L2 refit available (copy-through path in step 21.6).
COPY_THROUGH_MODELS: frozenset[str] = frozenset({"qlearning", "wmrl"})

# Subscale-exclusive covariate families that only appear in the M6b 4-cov
# subscale design. Used to footnote Table 3.
SUBSCALE_EXCLUSIVE_FAMILIES: tuple[str, ...] = ("iesr_intr_resid", "iesr_avd_resid")

# Path to the Phase-16 canonical M6b subscale posterior. Locked in plan 21-09
# as the read source for plan 21-10 (Option (a) — keep Phase-16 contract
# stable across downstream consumers).
SUBSCALE_NC_DEFAULT = MODELS_BAYESIAN_DIR / "wmrl_m6b_subscale_posterior.nc"


@dataclass
class TableArtefact:
    """One generated table written in three formats (CSV, Markdown, LaTeX).

    Parameters
    ----------
    name : str
        Short identifier used as the filename stem (e.g. ``table1_loo_stacking``).
    df : pd.DataFrame
        Source dataframe.
    caption : str
        Manuscript-ready caption (used as the LaTeX ``\\caption{}`` and the
        Markdown table's preceding bold line).
    label : str
        Quarto cross-reference label (e.g. ``tbl-loo-stacking``). Embedded as
        the LaTeX ``\\label{}`` and as a Markdown comment for traceability.
    bold_rows : list[int] | None
        Optional row indices to bold in the Markdown / LaTeX renderings (used
        to emphasise stacking-weight winners in Table 1).
    """

    name: str
    df: pd.DataFrame
    caption: str
    label: str
    bold_rows: list[int] | None = None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool = False) -> None:
    """Configure root logger for CLI invocation.

    Parameters
    ----------
    verbose : bool, optional
        If True, set DEBUG level. Otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _read_winners_txt(winners_file: Path) -> list[str]:
    """Read winners.txt and return display names (e.g. ``["M3", "M6b"]``).

    Parameters
    ----------
    winners_file : Path
        Path to the comma-separated winners file written by step 21.5.

    Returns
    -------
    list[str]
        Sorted display names. Empty list if file missing.
    """
    if not winners_file.exists():
        logger.warning(
            "winners.txt not found at %s; Table 3 will be empty.", winners_file
        )
        return []
    raw = winners_file.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    return [w.strip() for w in raw.split(",") if w.strip()]


def _read_audit_pipeline_action(audit_report: Path) -> str | None:
    """Parse the ``pipeline_action`` value from the audit report YAML header.

    Parameters
    ----------
    audit_report : Path
        Path to ``scale_audit_report.md`` produced by step 21.7.

    Returns
    -------
    str | None
        One of ``PROCEED_TO_AVERAGING`` / ``NULL_RESULT`` / None if missing.
    """
    if not audit_report.exists():
        logger.warning(
            "scale_audit_report.md not found at %s; assuming PROCEED.",
            audit_report,
        )
        return None
    in_yaml = False
    for line in audit_report.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "---":
            if in_yaml:
                break
            in_yaml = True
            continue
        if not in_yaml:
            continue
        m = re.match(r"^pipeline_action\s*:\s*(\S+)", stripped)
        if m:
            return m.group(1).strip()
    return None


def _round_numeric_cols(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Round every numeric column of ``df`` to ``decimals`` places.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    decimals : int, optional
        Decimal precision (default 2 — matches Table 1/2 caption text).

    Returns
    -------
    pd.DataFrame
        Copy with numeric columns rounded.
    """
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(decimals)
    return out


def _df_to_markdown(df: pd.DataFrame, bold_rows: list[int] | None = None) -> str:
    """Render a DataFrame to GitHub-flavoured Markdown, bolding rows if asked.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to render.
    bold_rows : list[int] | None
        Optional row indices (0-based) to wrap in ``**...**``.

    Returns
    -------
    str
        Markdown table string.
    """
    df_render = df.copy()
    if bold_rows:
        # Cast all columns to object dtype so the bold-wrapping does not
        # trigger pandas' incompatible-dtype FutureWarning on numeric cols.
        df_render = df_render.astype(object)
        for idx in bold_rows:
            if 0 <= idx < len(df_render):
                df_render.iloc[idx] = df_render.iloc[idx].apply(
                    lambda v: f"**{v}**"
                )
    try:
        return df_render.to_markdown(index=False)
    except ImportError:
        # tabulate not installed — fall back to a manual pipe table.
        return _manual_markdown_table(df_render)


def _manual_markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown pipe-table without ``tabulate``."""
    headers = [str(c) for c in df.columns]
    sep = ["---"] * len(headers)
    rows = [
        [str(v) if v is not None else "" for v in row]
        for row in df.itertuples(index=False, name=None)
    ]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(sep) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _df_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Render a DataFrame to a LaTeX ``table`` environment.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to render.
    caption : str
        Caption text (will be wrapped in ``\\caption{}``).
    label : str
        Quarto-style label (e.g. ``tbl-loo-stacking``); becomes ``\\label{tbl:...}``.

    Returns
    -------
    str
        Multi-line LaTeX source.
    """
    body = df.to_latex(index=False, escape=True, na_rep="--")
    return (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{body}"
        "\\end{table}\n"
    )


def _write_table_artefact(artefact: TableArtefact, tables_dir: Path) -> None:
    """Write a TableArtefact to CSV + Markdown + LaTeX in ``tables_dir``."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / f"{artefact.name}.csv"
    md_path = tables_dir / f"{artefact.name}.md"
    tex_path = tables_dir / f"{artefact.name}.tex"

    artefact.df.to_csv(csv_path, index=False)
    md = (
        f"<!-- {artefact.label} -->\n"
        f"**{artefact.caption}**\n\n"
        f"{_df_to_markdown(artefact.df, artefact.bold_rows)}\n"
    )
    md_path.write_text(md, encoding="utf-8")
    tex_path.write_text(
        _df_to_latex(artefact.df, artefact.caption, artefact.label),
        encoding="utf-8",
    )

    logger.info(
        "Wrote %s (csv=%d rows, md=%d bytes, tex=%d bytes)",
        artefact.name,
        len(artefact.df),
        md_path.stat().st_size,
        tex_path.stat().st_size,
    )


# ---------------------------------------------------------------------------
# Table 1 — LOO + stacking weights
# ---------------------------------------------------------------------------


def generate_table1_loo_stacking(
    baseline_dir: Path, tables_dir: Path
) -> TableArtefact | None:
    """Build Table 1 — PSIS-LOO + stacking weights for all six choice-only models.

    Reads ``loo_stacking_results.csv`` and renders columns
    ``Model, rank, elpd_loo, elpd_diff, SE, dse, weight, pct_pareto_k_gt_07,
    warning``. Numeric columns rounded to 2 decimals. Bolds the row(s) whose
    stacking weight is the maximum (the winner(s) reported in winners.txt).

    Parameters
    ----------
    baseline_dir : Path
        Directory containing ``loo_stacking_results.csv``.
    tables_dir : Path
        Output directory for ``table1_loo_stacking.{csv,md,tex}``.

    Returns
    -------
    TableArtefact | None
        Generated artefact or None if input missing.
    """
    src = baseline_dir / "loo_stacking_results.csv"
    if not src.exists():
        logger.error("Table 1 source missing: %s", src)
        return None

    df = pd.read_csv(src)

    # Map ArviZ-style column names to the canonical schema described in the
    # caption. ArviZ writes ``se`` and ``dse`` lowercase; the source step 21.5
    # already adds ``pct_high_pareto_k`` (per plan-checker Issue #8).
    rename_map = {
        "se": "SE",
        "dse": "dse",
        "pct_high_pareto_k": "pct_pareto_k_gt_07",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # Add display "Model" column from the first column (which holds either the
    # display name or the internal name depending on how step 21.5 wrote it).
    first_col = df.columns[0]
    if first_col != "Model":
        df.insert(0, "Model", df[first_col].astype(str))
        df = df.drop(columns=[first_col])

    # Promote ``Model`` to display names where possible (idempotent).
    df["Model"] = df["Model"].apply(
        lambda v: INTERNAL_TO_DISPLAY.get(str(v), str(v))
    )

    # Compute rank from stacking weight (descending). step 21.5's CSV is
    # already sorted but we recompute for safety.
    if "weight" in df.columns:
        df = df.sort_values("weight", ascending=False).reset_index(drop=True)
        if "rank" in df.columns:
            df = df.drop(columns=["rank"])
        df.insert(1, "rank", range(1, len(df) + 1))

    # Select & order final columns; tolerate missing columns gracefully.
    desired = [
        "Model",
        "rank",
        "elpd_loo",
        "elpd_diff",
        "SE",
        "dse",
        "weight",
        "pct_pareto_k_gt_07",
        "warning",
    ]
    final_cols = [c for c in desired if c in df.columns]
    df = df[final_cols]
    df = _round_numeric_cols(df, decimals=2)

    # Determine bold rows = winners (max weight; ties bolded together).
    bold_rows: list[int] = []
    if "weight" in df.columns and len(df) > 0:
        max_weight = df["weight"].max()
        bold_rows = df.index[df["weight"] == max_weight].tolist()

    artefact = TableArtefact(
        name="table1_loo_stacking",
        df=df,
        caption=(
            "Table 1. PSIS-LOO + stacking weights across six choice-only "
            "models (N=138). Winner(s) marked in bold."
        ),
        label="tbl-loo-stacking",
        bold_rows=bold_rows,
    )
    _write_table_artefact(artefact, tables_dir)
    return artefact


# ---------------------------------------------------------------------------
# Table 2 — RFX-BMS / PXP
# ---------------------------------------------------------------------------


def generate_table2_rfx_bms(
    baseline_dir: Path, tables_dir: Path
) -> TableArtefact | None:
    """Build Table 2 — random-effects BMS (Stephan 2009; Rigoux 2014).

    Parameters
    ----------
    baseline_dir : Path
        Directory containing ``rfx_bms_pxp.csv``.
    tables_dir : Path
        Output directory for ``table2_rfx_bms.{csv,md,tex}``.

    Returns
    -------
    TableArtefact | None
        Generated artefact, or None if input missing.
    """
    src = baseline_dir / "rfx_bms_pxp.csv"
    if not src.exists():
        logger.error("Table 2 source missing: %s", src)
        return None

    df = pd.read_csv(src)

    # Expected schema (plan 21-06):
    #   model, alpha, r, xp, bor, pxp, pxp_exceeds_95
    # Promote ``model`` -> ``Model`` and order columns.
    if "model" in df.columns and "Model" not in df.columns:
        df = df.rename(columns={"model": "Model"})
    if "Model" in df.columns:
        df["Model"] = df["Model"].apply(
            lambda v: INTERNAL_TO_DISPLAY.get(str(v), str(v))
        )

    desired = ["Model", "alpha", "r", "xp", "bor", "pxp"]
    final_cols = [c for c in desired if c in df.columns]
    df = df[final_cols]
    df = _round_numeric_cols(df, decimals=2)

    artefact = TableArtefact(
        name="table2_rfx_bms",
        df=df,
        caption=(
            "Table 2. Random-effects BMS (Stephan 2009; Rigoux 2014). "
            "PXP > 0.95 indicates strong protected exceedance probability."
        ),
        label="tbl-rfx-bms",
    )
    _write_table_artefact(artefact, tables_dir)
    return artefact


# ---------------------------------------------------------------------------
# Table 3 — winner beta HDIs (with model-averaged columns when multi-winner)
# ---------------------------------------------------------------------------


def _load_winner_beta_csv(l2_dir: Path, winner_display: str) -> pd.DataFrame:
    """Load a winner's per-row beta HDI table from step 21.7.

    Skips the metadata trailer row (``beta_site=='__METADATA__'``) per the
    plan 21-08 CSV schema.

    Parameters
    ----------
    l2_dir : Path
        Directory containing ``{winner}_beta_hdi_table.csv`` files.
    winner_display : str
        Display name (e.g. ``M3``); mapped to internal id for filename.

    Returns
    -------
    pd.DataFrame
        Beta-site rows with metadata trailer dropped. Empty DataFrame if
        the file is missing (e.g. M1/M2 copy-through).
    """
    internal = DISPLAY_TO_INTERNAL.get(winner_display, winner_display)
    src = l2_dir / f"{internal}_beta_hdi_table.csv"
    if not src.exists():
        logger.info(
            "No beta HDI table for %s (%s) at %s — likely copy-through model.",
            winner_display,
            internal,
            src,
        )
        return pd.DataFrame()
    df = pd.read_csv(src)
    if "beta_site" in df.columns:
        df = df[df["beta_site"] != "__METADATA__"].reset_index(drop=True)
    return df


def _load_averaged_effects(l2_dir: Path) -> pd.DataFrame:
    """Load ``averaged_scale_effects.csv`` if present (multi-winner only)."""
    src = l2_dir / "averaged_scale_effects.csv"
    if not src.exists():
        logger.info(
            "No averaged_scale_effects.csv at %s — single-winner or "
            "NULL_RESULT path; Table 3 will not include averaged columns.",
            src,
        )
        return pd.DataFrame()
    return pd.read_csv(src)


def generate_table3_winner_betas(
    l2_dir: Path,
    tables_dir: Path,
    winners: list[str],
    subscale_section: str | None,
) -> TableArtefact | None:
    """Build Table 3 — winner Level-2 beta HDIs (with averaged columns).

    For each winner with a beta HDI table (M3/M5/M6a → 2 rows; M6b → up to
    32 rows), produce one row per ``(winner, covariate_family,
    target_parameter)`` tuple with ``hdi_low``, ``hdi_high``,
    ``excludes_zero_hdi``, and the FDR-BH flag. If ``averaged_scale_effects.csv``
    exists, append ``model_averaged_mean``, ``model_averaged_hdi_low``,
    ``model_averaged_hdi_high``, ``single_source`` columns where the canonical
    key matches.

    Parameters
    ----------
    l2_dir : Path
        Step 21.6/21.7/21.8 output directory.
    tables_dir : Path
        Where to write ``table3_winner_betas.{csv,md,tex}``.
    winners : list[str]
        Display names from winners.txt.
    subscale_section : str | None
        Notes string (M6b subscale guard) — appears in the footer of the
        Markdown rendering when the subscale NetCDF is absent.

    Returns
    -------
    TableArtefact | None
        Generated artefact, or None if no winners have beta tables.
    """
    averaged = _load_averaged_effects(l2_dir)

    # Build canonical-key -> averaged-row map for fast lookup.
    averaged_lookup: dict[tuple[str, str], pd.Series] = {}
    if not averaged.empty:
        cov_col = (
            "covariate_family"
            if "covariate_family" in averaged.columns
            else "covariate"
        )
        tgt_col = (
            "target_parameter"
            if "target_parameter" in averaged.columns
            else "target"
        )
        for _, row in averaged.iterrows():
            key = (str(row[cov_col]), str(row[tgt_col]))
            averaged_lookup[key] = row

    rows: list[dict] = []
    copy_through_winners: list[str] = []

    for winner in winners:
        internal = DISPLAY_TO_INTERNAL.get(winner, winner)
        if internal in COPY_THROUGH_MODELS:
            copy_through_winners.append(winner)
            continue
        winner_df = _load_winner_beta_csv(l2_dir, winner)
        if winner_df.empty:
            copy_through_winners.append(winner)
            continue
        for _, brow in winner_df.iterrows():
            cov_family = str(brow.get("covariate_family", brow.get("covariate", "")))
            target_param = str(
                brow.get("target_parameter", brow.get("target", ""))
            )
            new_row: dict[str, object] = {
                "winner": winner,
                "covariate_family": cov_family,
                "target_parameter": target_param,
                "posterior_mean": brow.get("posterior_mean"),
                "hdi_low": brow.get("hdi_low"),
                "hdi_high": brow.get("hdi_high"),
                "excludes_zero_hdi": brow.get("excludes_zero_hdi"),
                "p_fdr_bh": brow.get("p_fdr_bh"),
                "fdr_bh_significant": brow.get("fdr_bh_significant"),
            }
            # Attach averaged columns if a canonical-key match exists.
            key = (cov_family, target_param)
            if key in averaged_lookup:
                arow = averaged_lookup[key]
                new_row["model_averaged_mean"] = arow.get("averaged_mean")
                new_row["model_averaged_hdi_low"] = arow.get("hdi_low")
                new_row["model_averaged_hdi_high"] = arow.get("hdi_high")
                new_row["single_source"] = arow.get("single_source")
            else:
                new_row["model_averaged_mean"] = None
                new_row["model_averaged_hdi_low"] = None
                new_row["model_averaged_hdi_high"] = None
                new_row["single_source"] = None
            rows.append(new_row)

    if not rows:
        logger.warning(
            "Table 3: no winners produced beta HDI rows (winners=%s, "
            "copy-through=%s). Skipping table generation.",
            winners,
            copy_through_winners,
        )
        return None

    df = pd.DataFrame(rows)
    df = _round_numeric_cols(df, decimals=3)

    caption_parts = [
        "Table 3. Winner Level-2 beta coefficients with 95% HDI and FDR-BH "
        "flag. M3/M5/M6a winners contribute two rows each (lec, iesr from "
        "the 2-cov L2 hook); M6b winners contribute up to 32 rows (4 "
        "covariate families × 8 parameters)."
    ]
    if copy_through_winners:
        caption_parts.append(
            "M1/M2 copy-through winners produce 0 rows (no L2-compatible "
            f"parameter target): {', '.join(copy_through_winners)}."
        )
    if subscale_section is not None:
        caption_parts.append(subscale_section)

    artefact = TableArtefact(
        name="table3_winner_betas",
        df=df,
        caption=" ".join(caption_parts),
        label="tbl-winner-betas",
    )
    _write_table_artefact(artefact, tables_dir)
    return artefact


# ---------------------------------------------------------------------------
# Figure 1 — forest plot (delegates to scripts/06_fit_analyses/07_bayesian_level2_effects.py)
# ---------------------------------------------------------------------------


def generate_figure1_forest(
    l2_dir: Path,
    figures_dir: Path,
    winners: list[str],
) -> list[Path]:
    """Generate forest plots for each winner via the legacy step-18 script.

    Per plan 21-10, we reuse ``scripts/06_fit_analyses/07_bayesian_level2_effects.py`` rather
    than re-implementing the forest plot. The script is invoked once per
    winner via subprocess with adjusted paths (``--posterior-path`` →
    ``models/bayesian/21_l2/{winner}_posterior.nc``).

    Parameters
    ----------
    l2_dir : Path
        Source directory holding ``{winner}_posterior.nc``.
    figures_dir : Path
        Where to write ``forest_{winner}.png``.
    winners : list[str]
        Display names from winners.txt.

    Returns
    -------
    list[Path]
        Generated forest PNG paths (may be empty if all winners are copy-through).
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for winner in winners:
        internal = DISPLAY_TO_INTERNAL.get(winner, winner)
        posterior = l2_dir / f"{internal}_posterior.nc"
        if not posterior.exists():
            logger.warning(
                "Forest plot: posterior missing for %s at %s; skipping.",
                winner,
                posterior,
            )
            continue
        out_png = figures_dir / f"forest_{internal}.png"
        cmd = [
            sys.executable,
            "scripts/06_fit_analyses/07_bayesian_level2_effects.py",
            "--posterior-path",
            str(posterior),
        ]
        logger.info("Forest plot: invoking %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            logger.error(
                "Forest plot script failed for %s: stderr=%s",
                winner,
                (exc.stderr or "")[-500:],
            )
            continue
        # The legacy script writes its own filenames; copy/rename if it
        # produced a recognisable output, else log a soft warning so the
        # pipeline doesn't fail on a cosmetic step.
        legacy_candidates = [
            REPORTS_FIGURES_BAYESIAN / f"{internal}_forest.png",
            REPORTS_FIGURES_BAYESIAN / "m6b_forest_lec5.png",
            MODELS_BAYESIAN_DIR / "level2" / f"{internal}_forest.png",
        ]
        copied = False
        for cand in legacy_candidates:
            if cand.exists():
                shutil.copy2(cand, out_png)
                generated.append(out_png)
                copied = True
                logger.info(
                    "Forest plot: copied %s -> %s (winner=%s)",
                    cand,
                    out_png,
                    winner,
                )
                break
        if not copied:
            logger.warning(
                "Forest plot for %s ran but no recognisable PNG was found in "
                "the legacy output directories; manual inspection needed.",
                winner,
            )
    return generated


# ---------------------------------------------------------------------------
# Null-result branch
# ---------------------------------------------------------------------------


def write_null_result_summary(
    baseline_dir: Path,
    l2_dir: Path,
    tables_dir: Path,
) -> Path:
    """Write a ``null_result_summary.md`` when audit pipeline_action is NULL.

    Parameters
    ----------
    baseline_dir : Path
        Directory containing ``loo_stacking_results.csv``.
    l2_dir : Path
        Directory containing ``scale_audit_report.md``.
    tables_dir : Path
        Output directory.

    Returns
    -------
    Path
        Written file path.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / "null_result_summary.md"

    stacking_path = baseline_dir / "loo_stacking_results.csv"
    audit_path = l2_dir / "scale_audit_report.md"

    weight_blurb = "(stacking results unavailable)"
    if stacking_path.exists():
        try:
            df = pd.read_csv(stacking_path)
            if "weight" in df.columns:
                top_idx = df["weight"].idxmax()
                top_w = float(df.loc[top_idx, "weight"])
                top_model = INTERNAL_TO_DISPLAY.get(
                    str(df.iloc[top_idx, 0]), str(df.iloc[top_idx, 0])
                )
                weight_blurb = (
                    f"Stacking-weight winner: **{top_model}** "
                    f"(w={top_w:.2f})."
                )
        except (KeyError, ValueError) as exc:
            logger.warning("Could not parse stacking results: %s", exc)

    body = (
        "# Phase 21 — Null-Result Summary\n\n"
        "The Phase 21 Bayesian model selection pipeline produced a "
        "**null result**: zero Level-2 beta sites had 95% highest-density "
        "intervals excluding zero after FDR-BH correction across the winner "
        "set.\n\n"
        f"{weight_blurb}\n\n"
        "## Interpretation\n\n"
        "Following the staged Bayesian workflow of Hess et al. (2025; "
        "DOI 10.5334/cpsy.116) and the troubleshooting protocol of "
        "Baribault & Collins (2023; DOI 10.1037/met0000554), a null result "
        "is treated as a valid scientific outcome — not a pipeline failure. "
        "The forest plot has been intentionally suppressed because rendering "
        "an all-null forest plot would mislead readers into searching for "
        "significant effects where none exist.\n\n"
        "## Sources\n\n"
        f"- Stacking weights: `{stacking_path}`\n"
        f"- Scale-fit audit: `{audit_path}`\n"
    )
    out_path.write_text(body, encoding="utf-8")
    logger.info("Wrote null-result summary: %s (%d bytes)", out_path, len(body))
    return out_path


# ---------------------------------------------------------------------------
# paper.qmd update
# ---------------------------------------------------------------------------

METHODS_PARAGRAPH_TEMPLATE = """\
### Bayesian Model Selection Pipeline {#sec-bayesian-selection}

Following the staged Bayesian workflow of Hess et al. (2025; DOI 10.5334/cpsy.116) and the troubleshooting protocol of Baribault & Collins (2023; DOI 10.1037/met0000554), we replaced AIC/BIC model comparison with a hierarchical-Bayesian pipeline comprising nine steps: (1) prior predictive checks on all six choice-only models; (2) Bayesian parameter recovery on N=50 synthetic datasets per model; (3) baseline hierarchical fits without trauma covariates; (4) convergence and fit-quality audit (R-hat $\\leq$ 1.05, ESS_bulk $\\geq$ 400, 0 divergences); (5) PSIS-LOO + stacking weights (Yao et al. 2018; DOI 10.1214/17-BA1091) as primary model ranking, with random-effects BMS + PXP (Stephan et al. 2009; Rigoux et al. 2014) as secondary; (6) refit of the stacking winner(s) with a Level-2 covariate design whose structure depends on the winning model --- M3, M5, and M6a use a 2-covariate design (lec_total and iesr_total, both z-scored), while M6b additionally includes the residualized IES-R intrusion and avoidance subscales as two further covariates (yielding 4 $\\times$ 8 = 32 beta sites for its stick-breaking parameterization); (7) scale-fit audit (FDR-BH adjusted HDI exclusion); (8) stacking-weighted model averaging of $\\beta$ coefficients, with the averaging applied only over the subset of $\\beta$ sites shared between winners (i.e., `beta_lec_{target}` and `beta_iesr_{target}`); subscale-exclusive $\\beta$s are reported from M6b alone; and (9) manuscript tables. AIC/BIC are reported for legacy comparability only; PSIS-LOO stacking is the primary selection criterion. The complete winner Level-2 beta coefficients with 95% HDI and FDR-BH flag are reported in @tbl-winner-betas; the LOO + stacking ranking is in @tbl-loo-stacking; the random-effects BMS PXP is in @tbl-rfx-bms; and the winner forest plots are in @fig-forest-21. The M6b subscale exploratory arm runs fire-and-forget — the resulting subscale beta table may be added via a post-phase quick task if the arm completes after the main manuscript build.

"""


def update_paper_qmd(paper_path: Path) -> bool:
    """Patch ``paper.qmd`` with the Phase 21 Methods paragraph + Results refs.

    1. Insert ``### Bayesian Model Selection Pipeline {#sec-bayesian-selection}``
       BEFORE the locked ``### Hierarchical Level-2 Trauma Associations
       {#sec-bayesian-regression}`` anchor (Phase 18-05 location).
    2. Adjust the Results "M6b is the winning model" sentence to reference
       the stacking-weight-based winner(s) — replace with a Quarto inline
       reference to ``winner_display`` so the manuscript stays in sync with
       ``loo_stacking_results.csv``.
    3. Idempotent: if the new section is already present (anchor
       ``{#sec-bayesian-selection}`` already in file), leave it alone.

    Parameters
    ----------
    paper_path : Path
        Path to ``paper.qmd`` (project default ``manuscript/paper.qmd``).

    Returns
    -------
    bool
        True if the file was modified, False if no changes were needed or
        the anchor was missing.
    """
    if not paper_path.exists():
        logger.error("paper.qmd not found at %s — paper edit skipped.", paper_path)
        return False

    text = paper_path.read_text(encoding="utf-8")

    if "{#sec-bayesian-selection}" in text:
        logger.info(
            "paper.qmd already contains #sec-bayesian-selection anchor; "
            "skipping insertion (idempotent)."
        )
        return False

    anchor = "### Hierarchical Level-2 Trauma Associations {#sec-bayesian-regression}"
    if anchor not in text:
        logger.error(
            "paper.qmd is missing the locked anchor '%s' — cannot insert "
            "Phase 21 Methods paragraph. Aborting paper edit.",
            anchor,
        )
        return False

    new_text = text.replace(anchor, METHODS_PARAGRAPH_TEMPLATE + anchor, 1)

    # Step 3: adjust the M6b sentence in Results (best-effort regex; leave
    # in place if not found rather than erroring out).
    m6b_sentence_re = re.compile(
        r"M6b received the\s+highest stacking weight among the six choice-only models\.",
        re.MULTILINE,
    )
    replacement = (
        "The stacking-weight winner is reported as `{python} winner_display` "
        "(see @tbl-loo-stacking)."
    )
    new_text, n_replaced = m6b_sentence_re.subn(replacement, new_text, count=1)
    if n_replaced:
        logger.info(
            "Replaced legacy 'M6b received the highest stacking weight...' "
            "sentence with stacking-weight-aware reference."
        )
    else:
        logger.info(
            "Legacy M6b winner sentence not matched — leaving Results "
            "section untouched (existing tbl-stacking-weights chunk still "
            "renders the winner table)."
        )

    paper_path.write_text(new_text, encoding="utf-8")
    logger.info(
        "Patched paper.qmd: inserted #sec-bayesian-selection (%d bytes added).",
        len(new_text) - len(text),
    )
    return True


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the argparse Namespace for CLI invocation."""
    p = argparse.ArgumentParser(
        prog="08_manuscript_tables",
        description=(
            "Step 21.9 — generate manuscript Tables 1/2/3 + Figure 1 forest "
            "plots from Phase 21 pipeline outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--baseline-dir",
        type=Path,
        default=MODELS_BAYESIAN_BASELINE,
        help="Directory holding loo_stacking_results.csv, rfx_bms_pxp.csv, winners.txt.",
    )
    p.add_argument(
        "--l2-dir",
        type=Path,
        default=MODELS_BAYESIAN_L2,
        help="Directory holding scale_audit_report.md, {winner}_beta_hdi_table.csv, etc.",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=REPORTS_FIGURES_BAYESIAN / "21_bayesian",
        help="Output directory for forest_{winner}.png figures.",
    )
    p.add_argument(
        "--tables-dir",
        type=Path,
        default=MODELS_BAYESIAN_DIR / "21_tables",
        help="Output directory for table1/2/3 .csv/.md/.tex artefacts.",
    )
    p.add_argument(
        "--paper",
        type=Path,
        default=Path("manuscript/paper.qmd"),
        help=(
            "Path to paper.qmd. Default points to manuscript/paper.qmd; "
            "the cluster invocation passes --no-paper-edit, so the paper "
            "is updated locally during plan execution and reviewed via Git diff."
        ),
    )
    p.add_argument(
        "--no-paper-edit",
        action="store_true",
        help="Skip the paper.qmd modification (cluster invocation default).",
    )
    p.add_argument(
        "--subscale-nc",
        type=Path,
        default=SUBSCALE_NC_DEFAULT,
        help=(
            "Path to the M6b subscale posterior NetCDF (Phase-16 canonical "
            "location). If missing, Table 3 subscale section is skipped "
            "with a NOTE log line (plan-checker Issue #9)."
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="DEBUG-level logging.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    _setup_logging(verbose=args.verbose)

    logger.info("=" * 60)
    logger.info("Phase 21 Step 21.9 — manuscript tables + figures")
    logger.info("=" * 60)
    logger.info("Baseline dir: %s", args.baseline_dir)
    logger.info("L2 dir:       %s", args.l2_dir)
    logger.info("Tables dir:   %s", args.tables_dir)
    logger.info("Figures dir:  %s", args.figures_dir)
    logger.info("paper.qmd:    %s (edit=%s)", args.paper, not args.no_paper_edit)

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    # Read pipeline action + winners.
    audit_action = _read_audit_pipeline_action(
        args.l2_dir / "scale_audit_report.md"
    )
    winners = _read_winners_txt(args.baseline_dir / "winners.txt")
    logger.info("Audit pipeline_action: %s", audit_action)
    logger.info("Winners: %s", winners)

    # ----- Tables 1 & 2 (always attempted) -----
    generate_table1_loo_stacking(args.baseline_dir, args.tables_dir)
    generate_table2_rfx_bms(args.baseline_dir, args.tables_dir)

    # ----- M6b subscale guard (plan-checker Issue #9) -----
    subscale_nc = args.subscale_nc
    if not subscale_nc.exists():
        logger.info(
            "NOTE: M6b subscale arm still running or not launched — "
            "subscale NetCDF not present at %s. Subscale table will be "
            "added in a post-phase quick task after the subscale fit "
            "completes. See plan 21-10 post-phase quick task.",
            subscale_nc,
        )
        subscale_section = (
            "M6b subscale exploratory arm: NetCDF not yet present at "
            f"`{subscale_nc}` at manuscript-build time; subscale table to "
            "be added via post-phase quick task."
        )
    else:
        logger.info("M6b subscale NetCDF present: %s", subscale_nc)
        subscale_section = (
            f"M6b subscale exploratory arm: NetCDF present at `{subscale_nc}` "
            "and included in winner beta table where applicable."
        )

    # ----- Null-result branch -----
    if audit_action == "NULL_RESULT":
        logger.warning(
            "Audit pipeline_action == NULL_RESULT — generating null-result "
            "summary and SKIPPING forest plot (would be misleading)."
        )
        write_null_result_summary(args.baseline_dir, args.l2_dir, args.tables_dir)
    else:
        # ----- Table 3 -----
        generate_table3_winner_betas(
            args.l2_dir,
            args.tables_dir,
            winners,
            subscale_section,
        )
        # ----- Figure 1 forest plot -----
        generate_figure1_forest(args.l2_dir, args.figures_dir, winners)

    # ----- paper.qmd edit -----
    if not args.no_paper_edit:
        update_paper_qmd(args.paper)
    else:
        logger.info(
            "--no-paper-edit set; skipping paper.qmd modification "
            "(cluster invocation default)."
        )

    logger.info("Phase 21 Step 21.9 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
