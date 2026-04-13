#!/usr/bin/env python
"""
18b: MLE vs Bayesian Reliability Scatterplots
===============================================

Generates per-(parameter, model) scatter plots comparing MLE point estimates
(x-axis) against Bayesian posterior means (y-axis) to validate that hierarchical
shrinkage is consistent with individual MLE fits.

Each plot includes:
  - Scatter of MLE vs posterior mean with Pearson r annotation
  - 45-degree reference line (identity line, dashed black)
  - For M6b (winning model): shrinkage direction arrows from the identity line
    toward each posterior mean, visualizing hierarchical pull-to-mean

Inputs:
    - output/mle/{model}_individual_fits.csv    (MLE point estimates, x-axis)
    - output/bayesian/{model}_individual_fits.csv (Bayesian posterior means, y-axis)

Outputs:
    - output/bayesian/figures/mle_vs_bayes/{model}_{param}.png  (one per cell)

Usage:
    python scripts/18b_mle_vs_bayes_reliability.py
    python scripts/18b_mle_vs_bayes_reliability.py --model wmrl_m6b
    python scripts/18b_mle_vs_bayes_reliability.py --output-dir output/bayesian/figures/mle_vs_bayes
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

# ---------------------------------------------------------------------------
# Project root on sys.path so config is importable from any working directory
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import MODEL_REGISTRY  # noqa: E402


def plot_mle_vs_bayes_reliability(
    mle_df: pd.DataFrame,
    bayes_df: pd.DataFrame,
    param: str,
    model_name: str,
    output_dir: Path,
    *,
    highlight_shrinkage: bool = False,
) -> Path:
    """Generate a scatter plot comparing MLE and Bayesian estimates for one parameter.

    Parameters
    ----------
    mle_df : pd.DataFrame
        DataFrame with MLE point estimates. Must contain columns
        ``participant_id`` and ``param``.
    bayes_df : pd.DataFrame
        DataFrame with Bayesian posterior means. Must contain columns
        ``participant_id`` and ``param``.
    param : str
        Parameter name to plot (e.g. ``'kappa_total'``).
    model_name : str
        Model key (e.g. ``'wmrl_m6b'``), used for figure title and filename.
    output_dir : Path
        Directory where the PNG will be saved.
    highlight_shrinkage : bool, optional
        When ``True`` (recommended for M6b), draw an arrow from the identity
        line ``(mle_val, mle_val)`` toward the posterior mean ``(mle_val,
        bayes_val)`` for each participant. Visualises the direction and
        magnitude of hierarchical shrinkage. Default ``False``.

    Returns
    -------
    Path
        Absolute path to the saved PNG file.

    Notes
    -----
    Rows are inner-joined on ``participant_id``, so only participants present
    in both DataFrames are plotted.  A Pearson r is annotated in the upper-left
    corner of the axes.
    """
    # ------------------------------------------------------------------
    # Align data by participant_id
    # ------------------------------------------------------------------
    merged = mle_df[["participant_id", param]].merge(
        bayes_df[["participant_id", param]],
        on="participant_id",
        suffixes=("_mle", "_bayes"),
    )
    if merged.empty:
        raise ValueError(
            f"No overlapping participants for param '{param}' in model '{model_name}'. "
            f"MLE had {len(mle_df)} rows, Bayesian had {len(bayes_df)} rows."
        )

    x = merged[f"{param}_mle"].to_numpy(dtype=float)
    y = merged[f"{param}_bayes"].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # Pearson r (mask NaN before computing)
    # ------------------------------------------------------------------
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() >= 2:
        r = float(np.corrcoef(x[valid], y[valid])[0, 1])
    else:
        r = float("nan")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    ax.scatter(x, y, alpha=0.5, s=20, color="steelblue", zorder=3)

    # 45-degree reference line
    all_vals = np.concatenate([x, y])
    finite_vals = all_vals[np.isfinite(all_vals)]
    if len(finite_vals):
        lo = float(finite_vals.min())
        hi = float(finite_vals.max())
        pad = (hi - lo) * 0.05
        ref_range = [lo - pad, hi + pad]
    else:
        ref_range = [0.0, 1.0]
    ax.plot(ref_range, ref_range, "--k", linewidth=1.0, zorder=2, label="Identity")
    ax.set_xlim(ref_range)
    ax.set_ylim(ref_range)

    # Shrinkage arrows: from (mle_val, mle_val) -> (mle_val, bayes_val)
    if highlight_shrinkage:
        for xi, yi in zip(x, y, strict=False):
            if not (np.isfinite(xi) and np.isfinite(yi)):
                continue
            dy = yi - xi  # deviation from identity
            if abs(dy) < 1e-6:
                continue
            ax.annotate(
                "",
                xy=(xi, yi),
                xytext=(xi, xi),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="steelblue",
                    lw=0.3,
                    mutation_scale=5,
                ),
                zorder=4,
            )

    # Annotations
    ax.set_xlabel(f"MLE {param}", fontsize=10)
    ax.set_ylabel(f"Posterior mean {param}", fontsize=10)
    ax.set_title(f"{model_name}: {param}", fontsize=11)
    ax.annotate(
        f"r = {r:.3f}",
        xy=(0.05, 0.92),
        xycoords="axes fraction",
        fontsize=9,
        color="black",
    )

    fig.tight_layout()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_name}_{param}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path.resolve()


def main() -> None:
    """Parse arguments and generate reliability scatterplots for all requested models.

    For each (model, parameter) cell:
      - Loads MLE CSV from ``output/mle/{model}_individual_fits.csv``
      - Loads Bayesian CSV from ``output/bayesian/{model}_individual_fits.csv``
      - Skips silently (with warning) if either CSV is missing
      - Calls :func:`plot_mle_vs_bayes_reliability`
      - For M6b (winning model): enables shrinkage arrows

    Prints a summary of how many plots were generated and the output directory.
    """
    model_choices = list(MODEL_REGISTRY.keys()) + ["all"]

    parser = argparse.ArgumentParser(
        description="MLE vs Bayesian reliability scatterplots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=model_choices,
        help="Model to plot. 'all' iterates over every model in MODEL_REGISTRY.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/bayesian/figures/mle_vs_bayes",
        help="Directory for output PNG files.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    project_root = _PROJECT_ROOT

    if args.model == "all":
        models_to_run = list(MODEL_REGISTRY.keys())
    else:
        models_to_run = [args.model]

    n_plots = 0
    n_skipped = 0

    for model_name in models_to_run:
        mle_path = project_root / "output" / "mle" / f"{model_name}_individual_fits.csv"
        bayes_path = (
            project_root / "output" / "bayesian" / f"{model_name}_individual_fits.csv"
        )

        if not mle_path.exists():
            print(
                f"[WARNING] Skipping {model_name}: MLE CSV not found at {mle_path}",
                file=sys.stderr,
            )
            n_skipped += 1
            continue

        if not bayes_path.exists():
            print(
                f"[WARNING] Skipping {model_name}: Bayesian CSV not found at {bayes_path}",
                file=sys.stderr,
            )
            n_skipped += 1
            continue

        mle_df = pd.read_csv(mle_path)
        bayes_df = pd.read_csv(bayes_path)

        # Normalise participant_id to string for reliable merge
        if "participant_id" not in mle_df.columns:
            print(
                f"[WARNING] Skipping {model_name}: 'participant_id' column missing "
                f"from MLE CSV {mle_path}",
                file=sys.stderr,
            )
            n_skipped += 1
            continue
        if "participant_id" not in bayes_df.columns:
            print(
                f"[WARNING] Skipping {model_name}: 'participant_id' column missing "
                f"from Bayesian CSV {bayes_path}",
                file=sys.stderr,
            )
            n_skipped += 1
            continue

        mle_df["participant_id"] = mle_df["participant_id"].astype(str)
        bayes_df["participant_id"] = bayes_df["participant_id"].astype(str)

        params = MODEL_REGISTRY[model_name]["params"]
        highlight = model_name == "wmrl_m6b"

        for param in params:
            if param not in mle_df.columns:
                print(
                    f"[WARNING] {model_name}: param '{param}' not in MLE CSV — skipping.",
                    file=sys.stderr,
                )
                continue
            if param not in bayes_df.columns:
                print(
                    f"[WARNING] {model_name}: param '{param}' not in Bayesian CSV — skipping.",
                    file=sys.stderr,
                )
                continue

            try:
                out_path = plot_mle_vs_bayes_reliability(
                    mle_df=mle_df,
                    bayes_df=bayes_df,
                    param=param,
                    model_name=model_name,
                    output_dir=output_dir,
                    highlight_shrinkage=highlight,
                )
                print(f"  Saved: {out_path}")
                n_plots += 1
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[WARNING] {model_name}/{param}: plot failed — {exc}",
                    file=sys.stderr,
                )

    print(
        f"\nDone. {n_plots} plot(s) generated"
        + (f", {n_skipped} model(s) skipped." if n_skipped else ".")
    )
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
