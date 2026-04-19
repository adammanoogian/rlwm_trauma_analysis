#!/usr/bin/env python
"""
17: Analyze Winner Heterogeneity.

Per-participant winning model analysis. Answers: when does M6b win over
M5 vs M6a vs M3? Is it about kappa_share, capacity, or RL forgetting?

Workflow
--------
1. Load per-participant AIC from all six choice-only model fits
   (M1, M2, M3, M5, M6a, M6b).
2. Assign each participant their AIC-winning model.
3. Merge with the M6b individual-fit parameters (M6b is the aggregate
   winner, so its parameters provide a common reference frame).
4. Compare M6b parameter distributions across winner groups using
   Kruskal-Wallis (non-parametric omnibus) plus pairwise effect sizes.
5. Emit a tidy CSV with per-participant winner + M6b parameters.
6. Generate a boxplot figure for manuscript inclusion.

Outputs
-------
- output/model_comparison/winner_heterogeneity.csv
- output/model_comparison/winner_heterogeneity_summary.csv
- figures/model_comparison/winner_heterogeneity_figure.png

Usage
-----
    python scripts/17_analyze_winner_heterogeneity.py

Notes
-----
- Three-layer naming: the script uses domain English in comments
  (perseveration, working-memory capacity), descriptive names at the
  public API (kappa_total, capacity), and can use math-symbol internals
  where they appear (e.g., kappa_share is k/(k+k_s)).
- Identifiability caveat: M6b parameter recovery for base RLWM params
  (alpha, phi, rho, capacity, epsilon) is poor (r<0.80). We still report
  the group differences because (a) they summarize within-sample
  best-fit configurations and (b) kappa_total/kappa_share recover well
  (r=0.997, r=0.931) so differences in those parameters are trustworthy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from config import FIGURES_DIR, MODEL_REGISTRY, load_fits_with_validation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHOICE_ONLY_MODELS: list[str] = [
    "qlearning",
    "wmrl",
    "wmrl_m3",
    "wmrl_m5",
    "wmrl_m6a",
    "wmrl_m6b",
]

SHORT_NAME_BY_MODEL: dict[str, str] = {
    model: MODEL_REGISTRY[model]["short_name"] for model in CHOICE_ONLY_MODELS
}

# Parameters from the M6b winning model to compare across winner groups.
M6B_PARAMETER_COLUMNS: list[str] = [
    "alpha_pos",
    "alpha_neg",
    "phi",
    "rho",
    "capacity",
    "kappa_total",
    "kappa_share",
    "epsilon",
]

# Parameters whose distributional differences carry the most scientific
# weight (recovery r >= 0.80 in quick-005 recovery metrics).
TRUSTED_PARAMETERS: list[str] = ["kappa_total", "kappa_share"]


# ---------------------------------------------------------------------------
# Data loading and winner assignment
# ---------------------------------------------------------------------------


def load_per_participant_aic(mle_dir: Path) -> pd.DataFrame:
    """Load per-participant AIC values for every choice-only model.

    Parameters
    ----------
    mle_dir
        Directory containing the individual-fit CSVs.

    Returns
    -------
    pandas.DataFrame
        Long dataframe with columns ``participant_id``, one ``aic_{short}``
        column per model, and the argmin ``winning_model`` short name.
    """
    merged: pd.DataFrame | None = None
    for model in CHOICE_ONLY_MODELS:
        csv_name = MODEL_REGISTRY[model]["csv_filename"]
        path = mle_dir / csv_name
        if not path.exists():
            raise FileNotFoundError(
                f"Missing fit CSV for {model}: expected {path}. "
                f"Run `python scripts/12_fit_mle.py --model {model}` first."
            )
        fits = load_fits_with_validation(path, model)
        short = SHORT_NAME_BY_MODEL[model]
        subset = fits[["participant_id", "aic"]].rename(
            columns={"aic": f"aic_{short}"}
        )
        merged = subset if merged is None else merged.merge(subset, on="participant_id", how="inner")

    assert merged is not None  # for type-checker
    aic_columns = [c for c in merged.columns if c.startswith("aic_")]
    merged = merged.dropna(subset=aic_columns).copy()
    merged["winning_model"] = (
        merged[aic_columns]
        .idxmin(axis=1)
        .str.replace("aic_", "", regex=False)
    )
    return merged


def attach_m6b_parameters(
    winners: pd.DataFrame, mle_dir: Path
) -> pd.DataFrame:
    """Merge per-participant M6b parameter estimates onto the winners table."""
    m6b_path = mle_dir / MODEL_REGISTRY["wmrl_m6b"]["csv_filename"]
    m6b_fits = load_fits_with_validation(m6b_path, "wmrl_m6b")
    columns_to_keep = ["participant_id", "nll", "aic", "n_trials"] + M6B_PARAMETER_COLUMNS
    m6b_subset = m6b_fits[columns_to_keep].rename(
        columns={col: f"m6b_{col}" for col in M6B_PARAMETER_COLUMNS}
        | {"nll": "m6b_nll", "aic": "m6b_aic", "n_trials": "m6b_n_trials"}
    )
    return winners.merge(m6b_subset, on="participant_id", how="inner")


# ---------------------------------------------------------------------------
# Statistical comparison across winner groups
# ---------------------------------------------------------------------------


def summarize_by_winner_group(
    combined: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute medians/IQRs and Kruskal-Wallis p-values across winner groups.

    Returns
    -------
    per_group
        Rows = (winner group x M6b parameter); columns = n, median,
        q25, q75, mean, sd.
    omnibus
        Rows = M6b parameter; columns = H, p_value, eta_squared_h.
    """
    group_rows: list[dict[str, object]] = []
    omnibus_rows: list[dict[str, object]] = []

    groups = sorted(combined["winning_model"].unique())

    for param in M6B_PARAMETER_COLUMNS:
        column = f"m6b_{param}"
        samples: list[np.ndarray] = []
        for group in groups:
            values = combined.loc[combined["winning_model"] == group, column].dropna().values
            samples.append(values)
            group_rows.append(
                {
                    "winning_model": group,
                    "m6b_parameter": param,
                    "n": len(values),
                    "median": float(np.median(values)) if len(values) > 0 else np.nan,
                    "q25": float(np.percentile(values, 25)) if len(values) > 0 else np.nan,
                    "q75": float(np.percentile(values, 75)) if len(values) > 0 else np.nan,
                    "mean": float(np.mean(values)) if len(values) > 0 else np.nan,
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                }
            )

        usable_samples = [s for s in samples if len(s) >= 2]
        if len(usable_samples) >= 2:
            h_stat, p_value = scipy_stats.kruskal(*usable_samples)
            # Effect size: eta-squared_H = (H - k + 1) / (N - k)
            n_total = int(sum(len(s) for s in usable_samples))
            k_groups = len(usable_samples)
            eta_squared_h = (h_stat - k_groups + 1) / (n_total - k_groups) if n_total > k_groups else np.nan
        else:
            h_stat, p_value, eta_squared_h = np.nan, np.nan, np.nan

        omnibus_rows.append(
            {
                "m6b_parameter": param,
                "H_statistic": h_stat,
                "p_value": p_value,
                "eta_squared_h": eta_squared_h,
                "is_identifiable": param in TRUSTED_PARAMETERS,
            }
        )

    return pd.DataFrame(group_rows), pd.DataFrame(omnibus_rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_winner_heterogeneity(
    combined: pd.DataFrame, output_path: Path
) -> None:
    """Create a 2x4 grid of boxplots: M6b params split by winning model."""
    parameters_to_plot = M6B_PARAMETER_COLUMNS
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes_flat = axes.flatten()

    group_order = ["M6b", "M5", "M6a", "M3", "M2", "M1"]
    observed_groups = [g for g in group_order if g in combined["winning_model"].unique()]
    palette = dict(zip(observed_groups, sns.color_palette("husl", len(observed_groups))))

    for ax, param in zip(axes_flat, parameters_to_plot):
        column = f"m6b_{param}"
        sns.boxplot(
            data=combined,
            x="winning_model",
            y=column,
            order=observed_groups,
            ax=ax,
            palette=palette,
            showfliers=False,
        )
        sns.stripplot(
            data=combined,
            x="winning_model",
            y=column,
            order=observed_groups,
            ax=ax,
            color="black",
            size=3,
            alpha=0.5,
            jitter=True,
        )
        title_suffix = " (trusted)" if param in TRUSTED_PARAMETERS else ""
        ax.set_title(f"M6b {param}{title_suffix}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Winning model (AIC)", fontsize=9)
        ax.set_ylabel(param, fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "M6b parameter distributions by per-participant winning model",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the heterogeneity analysis end-to-end."""
    parser = argparse.ArgumentParser(
        description='Per-participant winning model heterogeneity analysis'
    )
    parser.add_argument('--source', type=str, default='mle',
                        choices=['mle', 'bayesian'],
                        help='Fit source: mle (default) or bayesian')
    args = parser.parse_args()

    if args.source == 'bayesian':
        fits_dir = project_root / "output" / "bayesian"
        output_dir = project_root / "output" / "bayesian" / "model_comparison"
        figures_dir = FIGURES_DIR / "bayesian" / "model_comparison"
    else:
        fits_dir = project_root / "output" / "mle"
        output_dir = project_root / "output" / "model_comparison"
        figures_dir = FIGURES_DIR / "model_comparison"

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("WINNER HETEROGENEITY ANALYSIS")
    print("=" * 80)
    print(f"Source: {args.source.upper()}")

    print("\n[1/4] Loading per-participant AIC for all choice-only models...")
    winners = load_per_participant_aic(fits_dir)
    print(f"    Loaded {len(winners)} participants with complete AIC coverage.")

    win_counts = winners["winning_model"].value_counts().sort_values(ascending=False)
    print("\nWinning-model counts (AIC per participant):")
    for model, count in win_counts.items():
        pct = 100.0 * count / len(winners)
        print(f"    {model}: {count} ({pct:.1f}%)")

    print("\n[2/4] Attaching M6b parameters to each participant...")
    combined = attach_m6b_parameters(winners, fits_dir)
    print(f"    Combined table: {combined.shape[0]} rows, {combined.shape[1]} columns")

    print("\n[3/4] Computing per-group summary statistics and Kruskal-Wallis tests...")
    per_group, omnibus = summarize_by_winner_group(combined)

    print("\nKruskal-Wallis omnibus tests (M6b parameters across winner groups):")
    print(omnibus.to_string(index=False))

    print("\nPer-group medians (M6b parameters by winning model):")
    pivot = per_group.pivot(index="m6b_parameter", columns="winning_model", values="median")
    print(pivot.to_string())

    print("\n[4/4] Writing outputs...")
    combined_path = output_dir / "winner_heterogeneity.csv"
    combined_export = combined.copy()
    combined_export.to_csv(combined_path, index=False)
    print(f"    [SAVED] {combined_path}")

    summary_path = output_dir / "winner_heterogeneity_summary.csv"
    per_group_export = per_group.merge(
        omnibus[["m6b_parameter", "H_statistic", "p_value", "eta_squared_h"]],
        on="m6b_parameter",
        how="left",
    )
    per_group_export.to_csv(summary_path, index=False)
    print(f"    [SAVED] {summary_path}")

    figure_path = figures_dir / "winner_heterogeneity_figure.png"
    plot_winner_heterogeneity(combined, figure_path)
    print(f"    [SAVED] {figure_path}")

    # Also copy to output/model_comparison/ for consistency with other outputs
    secondary_figure_path = output_dir / "winner_heterogeneity_figure.png"
    plot_winner_heterogeneity(combined, secondary_figure_path)
    print(f"    [SAVED] {secondary_figure_path}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
