#!/usr/bin/env python
"""
18: Bayesian Level-2 Effects
=============================

.. deprecated:: v4.0

    Superseded by the Phase 21 Bayesian model selection pipeline.
    Kept in-tree because ``scripts/21_manuscript_tables.py`` delegates Figure 1
    forest-plot rendering to this module via subprocess (single source of truth
    for matplotlib styling). New Level-2 analysis should use
    ``scripts/21_fit_with_l2.py`` (winner L2 refit) and
    ``scripts/21_scale_audit.py`` (FDR-BH audit per-winner); this script is the
    rendering backend only, not a standalone entry point.

    Any new invocation outside the Phase 21 orchestrator (``bash
    cluster/21_submit_pipeline.sh``) should be considered exploratory and
    validated against the Phase 21 outputs before reporting results.

Forest plot generation for hierarchical Bayesian Level-2 regression coefficients.

This script visualizes posterior beta coefficients from hierarchical Bayesian
model fits (e.g., wmrl_m6b with subscale-level L2 regression). It generates
forest plots grouped by covariate type and a summary CSV of all coefficients.

The beta sites in the posterior encode trauma-parameter associations:
  beta_{covariate}_{param} maps trauma covariate onto model parameter group mean.

For the wmrl_m6b subscale model there are 32 beta sites (8 params x 4 covariates):
  Covariates: lec_total, iesr_total, iesr_intr_resid, iesr_avd_resid
  Parameters: alpha_pos, alpha_neg, phi, rho, capacity, epsilon,
               kappa_total, kappa_share

Forest plots:
  - m6b_forest_lec5.png        : LEC-5 total coefficients only
  - m6b_forest_iesr_residuals.png : IES-R total + residualized subscales
  - m6b_forest_all_l2.png      : Full plot with all beta coefficients

Coefficient summary:
  - output/bayesian/level2/{model}_l2_coefficient_summary.csv

Usage:
    # Default: wmrl_m6b posterior
    python scripts/18_bayesian_level2_effects.py

    # Specify posterior path explicitly
    python scripts/18_bayesian_level2_effects.py \\
        --posterior-path output/bayesian/wmrl_m6b_subscale_posterior.nc

    # Custom HDI probability
    python scripts/18_bayesian_level2_effects.py --hdi-prob 0.89
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from config import load_netcdf_with_validation  # noqa: E402

# Lazy/conditional arviz import — may not be in all environments
try:
    import arviz as az

    _ARVIZ_AVAILABLE = True
except ImportError:
    _ARVIZ_AVAILABLE = False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _require_arviz() -> None:
    """Raise ImportError with install instructions if arviz is unavailable."""
    if not _ARVIZ_AVAILABLE:
        raise ImportError(
            "arviz is required for this script. "
            "Install with: conda install -c conda-forge arviz"
        )


def discover_beta_vars(idata: "az.InferenceData") -> list[str]:
    """Discover all beta_ variable names in the posterior.

    Parameters
    ----------
    idata : az.InferenceData
        Posterior inference data object.

    Returns
    -------
    list[str]
        Sorted list of variable names starting with ``beta_``.
    """
    return sorted(
        v for v in idata.posterior.data_vars if str(v).startswith("beta_")
    )


def group_beta_vars(
    beta_vars: list[str],
) -> dict[str, list[str]]:
    """Group beta variable names by covariate prefix.

    Parameters
    ----------
    beta_vars : list[str]
        All beta_ variable names from the posterior.

    Returns
    -------
    dict[str, list[str]]
        Mapping from group label to variable names.
        Groups: ``"lec"`` (LEC-5 total), ``"iesr"`` (IES-R total +
        residualized subscales), ``"all"`` (every beta variable).
    """
    lec_vars = [v for v in beta_vars if v.startswith("beta_lec_total_")]
    iesr_vars = [
        v
        for v in beta_vars
        if v.startswith("beta_iesr_total_")
        or v.startswith("beta_iesr_intr_resid_")
        or v.startswith("beta_iesr_avd_resid_")
        or v.startswith("beta_iesr_hyp_resid_")
    ]
    return {
        "lec": lec_vars,
        "iesr": iesr_vars,
        "all": beta_vars,
    }


def compute_coefficient_summary(
    idata: "az.InferenceData",
    beta_vars: list[str],
    hdi_prob: float,
) -> pd.DataFrame:
    """Compute summary statistics for all beta coefficients.

    Parameters
    ----------
    idata : az.InferenceData
        Posterior inference data.
    beta_vars : list[str]
        Beta variable names to summarise.
    hdi_prob : float
        HDI probability (e.g. 0.95).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: variable, mean, sd, hdi_low, hdi_high,
        hdi_excludes_zero.
    """
    _require_arviz()
    rows = []
    for var in beta_vars:
        samples = idata.posterior[var].values.flatten()
        mean = float(np.mean(samples))
        sd = float(np.std(samples))
        hdi = az.hdi(samples, hdi_prob=hdi_prob)
        low = float(hdi[0])
        high = float(hdi[1])
        excludes_zero = (low > 0) or (high < 0)
        rows.append(
            {
                "variable": var,
                "mean": mean,
                "sd": sd,
                "hdi_low": low,
                "hdi_high": high,
                "hdi_excludes_zero": excludes_zero,
            }
        )
    return pd.DataFrame(rows)


# ============================================================================
# FOREST PLOT FUNCTIONS
# ============================================================================


def make_forest_plot(
    idata: "az.InferenceData",
    var_names: list[str],
    hdi_prob: float,
    title: str,
    output_path: Path,
) -> None:
    """Generate and save a forest plot for the specified variables.

    Parameters
    ----------
    idata : az.InferenceData
        Posterior inference data.
    var_names : list[str]
        Variable names to include in the plot.
    hdi_prob : float
        HDI probability.
    title : str
        Plot title.
    output_path : Path
        File path for the saved PNG.
    """
    _require_arviz()
    if not var_names:
        print(f"  [SKIP] No variables for plot '{title}' — skipping.")
        return

    n_vars = len(var_names)
    figsize = (10, max(6, n_vars * 0.5))

    ax = az.plot_forest(
        idata,
        var_names=var_names,
        combined=True,
        hdi_prob=hdi_prob,
        figsize=figsize,
    )

    # az.plot_forest returns an array of axes; grab the figure from the first
    fig = ax[0].get_figure() if hasattr(ax, "__len__") else ax.get_figure()
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    # Add vertical line at zero on all axes
    for a in (ax if hasattr(ax, "__len__") else [ax]):
        a.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {output_path}")


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    """Entry point for Level-2 forest plot generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate forest plots for Bayesian Level-2 regression "
            "coefficients from hierarchical model posteriors."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="wmrl_m6b",
        help="Model key; used to build default posterior path (default: wmrl_m6b)",
    )
    parser.add_argument(
        "--posterior-path",
        type=str,
        default=None,
        help=(
            "Path to NetCDF posterior file. "
            "Defaults to output/bayesian/{model}_posterior.nc"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/bayesian/figures",
        help="Directory for output PNG files (default: output/bayesian/figures)",
    )
    parser.add_argument(
        "--hdi-prob",
        type=float,
        default=0.95,
        help="HDI probability for forest plots (default: 0.95)",
    )
    args = parser.parse_args()

    _require_arviz()

    # Resolve paths relative to project root
    if args.posterior_path is None:
        posterior_path = project_root / "output" / "bayesian" / f"{args.model}_posterior.nc"
    else:
        posterior_path = Path(args.posterior_path)
        if not posterior_path.is_absolute():
            posterior_path = project_root / posterior_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    level2_dir = project_root / "output" / "bayesian" / "level2"

    print("=" * 70)
    print("BAYESIAN LEVEL-2 FOREST PLOTS")
    print("=" * 70)
    print(f"  Model:          {args.model}")
    print(f"  Posterior:      {posterior_path}")
    print(f"  Output dir:     {output_dir}")
    print(f"  HDI prob:       {args.hdi_prob}")
    print()

    # ------------------------------------------------------------------ #
    # Load posterior                                                       #
    # ------------------------------------------------------------------ #
    if not posterior_path.exists():
        print(
            f"[WARNING] Posterior file not found: {posterior_path}\n"
            "          The NetCDF file is created by the cluster MCMC job.\n"
            "          Run this script after the cluster job completes."
        )
        return

    print(f"Loading posterior from {posterior_path} ...")
    idata = load_netcdf_with_validation(posterior_path, args.model)
    print("  Loaded successfully.")

    # ------------------------------------------------------------------ #
    # Discover beta variables                                              #
    # ------------------------------------------------------------------ #
    beta_vars = discover_beta_vars(idata)

    if not beta_vars:
        print(
            "[WARNING] No beta_ variables found in posterior.\n"
            "          This posterior does not contain Level-2 regression "
            "coefficients.\n"
            "          Re-run with --subscale on a model that supports L2 "
            "regression."
        )
        return

    print(f"  Found {len(beta_vars)} beta_ variables:")
    for v in beta_vars:
        print(f"    {v}")

    groups = group_beta_vars(beta_vars)

    # ------------------------------------------------------------------ #
    # Forest plots                                                         #
    # ------------------------------------------------------------------ #
    print("\nGenerating forest plots ...")

    make_forest_plot(
        idata=idata,
        var_names=groups["lec"],
        hdi_prob=args.hdi_prob,
        title=f"{args.model}: LEC-5 Total Level-2 Coefficients",
        output_path=output_dir / f"{args.model}_forest_lec5.png",
    )

    make_forest_plot(
        idata=idata,
        var_names=groups["iesr"],
        hdi_prob=args.hdi_prob,
        title=f"{args.model}: IES-R Level-2 Coefficients (Total + Residualized Subscales)",
        output_path=output_dir / f"{args.model}_forest_iesr_residuals.png",
    )

    make_forest_plot(
        idata=idata,
        var_names=groups["all"],
        hdi_prob=args.hdi_prob,
        title=f"{args.model}: All Level-2 Regression Coefficients ({args.hdi_prob*100:.0f}% HDI)",
        output_path=output_dir / f"{args.model}_forest_all_l2.png",
    )

    # ------------------------------------------------------------------ #
    # Coefficient summary table                                            #
    # ------------------------------------------------------------------ #
    print("\nComputing coefficient summary ...")
    summary_df = compute_coefficient_summary(idata, beta_vars, args.hdi_prob)

    level2_dir.mkdir(parents=True, exist_ok=True)
    summary_path = level2_dir / f"{args.model}_l2_coefficient_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  [SAVED] {summary_path}")

    # Print table to stdout
    print("\nCoefficient summary:")
    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    n_sig = summary_df["hdi_excludes_zero"].sum()
    print(
        f"\n  {n_sig}/{len(summary_df)} coefficients have "
        f"{args.hdi_prob*100:.0f}% HDI excluding zero."
    )

    # ------------------------------------------------------------------ #
    # L2-08 Horseshoe Prior Decision                                       #
    # ------------------------------------------------------------------ #
    print()
    print("# " + "-" * 65)
    print("# L2-08 Horseshoe Prior Decision")
    print("# " + "-" * 65)
    print("#")
    print("# STATUS: DEFERRED")
    print("#")
    print(
        "# Rationale: The subscale model (wmrl_m6b with 32 beta sites) has "
    )
    print(
        "# not yet been run on the cluster. Normal(0,1) priors on beta "
    )
    print(
        "# coefficients have not been tested; it would be premature to add "
    )
    print(
        "# regularized horseshoe priors before observing whether the flat "
    )
    print(
        "# priors produce convergence issues (divergences, Rhat > 1.01) or "
    )
    print(
        "# implausibly diffuse posterior estimates."
    )
    print("#")
    print(
        "# Decision gate: After the cluster job completes and this script "
    )
    print(
        "# is re-run with the actual posterior, inspect:"
    )
    print(
        "#   1. max_rhat — if > 1.01 for beta_ sites, consider horseshoe."
    )
    print(
        "#   2. Posterior SD of beta_ sites — if >> 1 on the probit scale,"
    )
    print(
        "#      the flat prior is not regularizing enough."
    )
    print(
        "#   3. Number of 95% HDI excluions — if all 32 exclude zero, "
    )
    print(
        "#      horseshoe may be more appropriate as a skeptical prior."
    )
    print("#")
    print(
        "# This decision is documented in:"
    )
    print(
        "#   .planning/phases/16-choice-only-family-extension-subscale-level-2/"
    )
    print(
        "#   16-07-SUMMARY.md"
    )
    print("# " + "-" * 65)


if __name__ == "__main__":
    main()
