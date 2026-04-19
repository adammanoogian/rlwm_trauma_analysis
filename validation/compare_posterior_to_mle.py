"""Sanity-check: compare Bayesian posterior means to MLE point estimates.

This is a one-off check, not part of the pipeline.  Run after a hierarchical
Bayesian fit completes to verify the posterior is plausibly near the MLE
optimum (modulo expected hierarchical shrinkage).

Inputs
------
- ``output/bayesian/{model}_posterior.nc``  (ArviZ NetCDF, produced by
  ``scripts/fitting/fit_bayesian.py``)
- ``output/mle/{model}_individual_fits.csv`` (produced by
  ``scripts/fitting/fit_mle.py``)

Outputs
-------
- Prints per-parameter summary: posterior mean, MLE mean, mean absolute
  deviation, fraction of participants whose posterior mean is within
  2 × MLE SE of the MLE point estimate, and top outliers.
- Writes ``output/bayesian/{model}_posterior_vs_mle.csv`` with per-
  participant comparison rows.

Expected patterns (not failure criteria — the point is to *see* the pattern)
---------------------------------------------------------------------------
- Highly-identified parameters (kappa_total, kappa_share on M6b):
  posterior mean within ~1 MLE SE.
- Poorly-identified parameters (alpha_pos, alpha_neg, phi, rho, capacity):
  posterior mean shrinks toward the group mean — individual values can
  differ from MLE by 2+ SE.  This is partial pooling working as designed.
- If a highly-identified parameter drifts far from MLE, investigate:
  possible prior–data conflict, poor mixing, or bug in the refactor.

Usage
-----
    python validation/compare_posterior_to_mle.py --model wmrl_m6b
    python validation/compare_posterior_to_mle.py --model wmrl_m6b --n-worst 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import load_netcdf_with_validation  # noqa: E402


_MODEL_PARAM_KEYS: dict[str, list[str]] = {
    "qlearning": ["alpha_pos", "alpha_neg", "epsilon"],
    "wmrl": ["alpha_pos", "alpha_neg", "phi", "rho", "capacity", "epsilon"],
    "wmrl_m3": [
        "alpha_pos", "alpha_neg", "phi", "rho", "capacity", "kappa", "epsilon",
    ],
    "wmrl_m5": [
        "alpha_pos", "alpha_neg", "phi", "rho", "capacity",
        "kappa", "phi_rl", "epsilon",
    ],
    "wmrl_m6a": [
        "alpha_pos", "alpha_neg", "phi", "rho", "capacity", "kappa_s", "epsilon",
    ],
    "wmrl_m6b": [
        "alpha_pos", "alpha_neg", "phi", "rho", "capacity",
        "kappa_total", "kappa_share", "epsilon",
    ],
}


def _posterior_means(
    posterior_nc_path: Path,
    param_keys: list[str],
    model: str,
) -> tuple[pd.DataFrame, list]:
    """Load posterior and return (participants, n_params) DataFrame of means."""
    idata = load_netcdf_with_validation(posterior_nc_path, model)
    post = idata.posterior

    # Per-participant posterior means for each param site.  Participant
    # order follows the "participant" dim if present; otherwise assume
    # the same sorted-ID ordering used by fit_bayesian.
    rows: dict[str, np.ndarray] = {}
    participants = None

    for param in param_keys:
        if param not in post.data_vars:
            raise KeyError(
                f"Posterior file does not contain deterministic site '{param}'. "
                f"Available vars: {list(post.data_vars)}"
            )
        da = post[param]
        # Collapse (chain, draw) -> participant axis
        if "participant" in da.dims:
            mean = da.mean(dim=("chain", "draw")).values
            if participants is None:
                participants = da["participant"].values.tolist()
        else:
            # 3D array (chain, draw, participant_axis_unnamed)
            mean = da.mean(axis=(0, 1))
            if participants is None:
                participants = list(range(len(mean)))
        rows[param] = np.asarray(mean, dtype=float)

    df = pd.DataFrame(rows, index=participants)
    df.index.name = "participant_id"
    return df, participants


def _mle_point_and_se(
    mle_csv_path: Path,
    param_keys: list[str],
    participants: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load MLE CSV; return point estimates and SEs aligned to participants."""
    mle_df = pd.read_csv(mle_csv_path)
    mle_df = mle_df.set_index("participant_id")

    # Align to posterior participant order (ints)
    ordered = [int(p) for p in participants if int(p) in mle_df.index]
    mle_df = mle_df.loc[ordered]

    point = mle_df[param_keys].copy()
    se_cols = [f"{p}_se" for p in param_keys]
    missing_se = [c for c in se_cols if c not in mle_df.columns]
    if missing_se:
        print(
            f"  Note: MLE CSV missing SE columns {missing_se} — SE-based "
            "coverage stats will be skipped."
        )
        se = pd.DataFrame(
            np.nan, index=point.index, columns=[p for p in param_keys]
        )
    else:
        se = mle_df[se_cols].copy()
        se.columns = param_keys

    return point, se


def compare(
    model: str,
    posterior_nc: Path,
    mle_csv: Path,
    output_csv: Path,
    n_worst: int = 10,
) -> None:
    """Produce per-participant comparison and summary prints."""
    param_keys = _MODEL_PARAM_KEYS[model]
    print(f"\nModel: {model}")
    print(f"Params: {param_keys}")
    print(f"Posterior: {posterior_nc}")
    print(f"MLE CSV:   {mle_csv}")

    post_means, participants = _posterior_means(posterior_nc, param_keys, model)
    mle_point, mle_se = _mle_point_and_se(mle_csv, param_keys, participants)

    # Align to intersection
    shared = post_means.index.astype(int).intersection(mle_point.index)
    post_means = post_means.loc[shared]
    mle_point = mle_point.loc[shared]
    mle_se = mle_se.loc[shared]

    print(f"\nAligned N = {len(shared)} participants")

    rows: list[dict] = []
    print(
        f"\n{'Parameter':<15} {'Post Mean':>10} {'MLE Mean':>10} "
        f"{'|Diff|':>10} {'Rel':>8} {'Within 2SE':>12}"
    )
    print("-" * 72)
    for param in param_keys:
        post_vec = post_means[param].values
        mle_vec = mle_point[param].values
        se_vec = mle_se[param].values

        diff = post_vec - mle_vec
        abs_diff = np.abs(diff)
        mle_mean = float(np.nanmean(mle_vec))
        rel = float(np.nanmean(abs_diff) / max(abs(mle_mean), 1e-8))

        if np.all(np.isnan(se_vec)):
            within_2se = float("nan")
            within_str = "n/a"
        else:
            se_safe = np.where(se_vec > 0, se_vec, np.nan)
            z = abs_diff / se_safe
            within_2se = float(np.nanmean(z < 2.0))
            within_str = f"{within_2se * 100:.1f}%"

        print(
            f"{param:<15} {float(np.nanmean(post_vec)):>10.4f} "
            f"{mle_mean:>10.4f} {float(np.nanmean(abs_diff)):>10.4f} "
            f"{rel:>7.2%} {within_str:>12}"
        )

        for pid, pm, mm, d, se_i in zip(
            shared, post_vec, mle_vec, diff, se_vec
        ):
            rows.append({
                "participant_id": int(pid),
                "param": param,
                "posterior_mean": float(pm),
                "mle_point": float(mm),
                "diff": float(d),
                "mle_se": float(se_i) if np.isfinite(se_i) else np.nan,
                "abs_diff_over_se": (
                    float(abs(d) / se_i) if np.isfinite(se_i) and se_i > 0
                    else np.nan
                ),
            })

    comparison = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_csv, index=False)
    print(f"\nSaved per-participant comparison to: {output_csv}")

    # Top outliers by |diff|/SE
    if comparison["abs_diff_over_se"].notna().any():
        print(f"\nTop {n_worst} outliers (largest |posterior - MLE| / MLE SE):")
        worst = comparison.dropna(subset=["abs_diff_over_se"]).nlargest(
            n_worst, "abs_diff_over_se"
        )
        for _, row in worst.iterrows():
            print(
                f"  pid={int(row['participant_id']):>6}  "
                f"{row['param']:<15} "
                f"post={row['posterior_mean']:.4f}  "
                f"mle={row['mle_point']:.4f}  "
                f"diff/se={row['abs_diff_over_se']:.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare hierarchical Bayesian posterior means to MLE "
                    "point estimates."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(_MODEL_PARAM_KEYS.keys()),
    )
    parser.add_argument("--posterior-nc", type=Path, default=None)
    parser.add_argument("--mle-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--n-worst", type=int, default=10)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    posterior_nc = args.posterior_nc or (
        root / "output" / "bayesian" / f"{args.model}_posterior.nc"
    )
    mle_csv = args.mle_csv or (
        root / "output" / "mle" / f"{args.model}_individual_fits.csv"
    )
    output_csv = args.output_csv or (
        root / "output" / "bayesian" / f"{args.model}_posterior_vs_mle.csv"
    )

    if not posterior_nc.exists():
        raise SystemExit(f"Posterior file not found: {posterior_nc}")
    if not mle_csv.exists():
        raise SystemExit(f"MLE CSV not found: {mle_csv}")

    compare(args.model, posterior_nc, mle_csv, output_csv, args.n_worst)


if __name__ == "__main__":
    main()
