"""Aggregate permutation null test results for M3 LEC covariate (L2-06).

Globs ``models/bayesian/permutation/shuffle_*_results.json``, loads each
JSON, counts how many shuffles produced a ``beta_lec_kappa`` 95% HDI that
excludes zero, and reports the empirical false positive rate.

Writes a human-readable Markdown summary to
``models/bayesian/permutation/permutation_summary.md``.

Usage
-----
::

    python scripts/fitting/aggregate_permutation_results.py
    python scripts/fitting/aggregate_permutation_results.py \\
        --input-dir models/bayesian/permutation \\
        --output-dir models/bayesian/permutation

Examples
--------
After all 50 SLURM array tasks complete::

    python scripts/fitting/aggregate_permutation_results.py
    # Reports: 2/50 shuffles had HDI excluding zero (4.0%) — PASS

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import MODELS_BAYESIAN_DIR  # noqa: E402


def load_shuffle_results(input_dir: Path) -> list[dict]:
    """Load all shuffle JSON results from a directory.

    Parameters
    ----------
    input_dir : Path
        Directory containing ``shuffle_*_results.json`` files.

    Returns
    -------
    list[dict]
        List of result dicts sorted by ``shuffle_idx``.

    Raises
    ------
    FileNotFoundError
        If no ``shuffle_*_results.json`` files are found in ``input_dir``.
    """
    json_files = sorted(input_dir.glob("shuffle_*_results.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No shuffle_*_results.json files found in {input_dir}. "
            "Run the SLURM array job first: sbatch cluster/13_bayesian_permutation.slurm"
        )

    results = []
    for jf in json_files:
        with open(jf) as fh:
            results.append(json.load(fh))

    # Sort by shuffle_idx for consistent ordering
    results.sort(key=lambda r: r["shuffle_idx"])
    return results


def summarize_results(results: list[dict]) -> dict:
    """Compute summary statistics over permutation shuffles.

    Parameters
    ----------
    results : list[dict]
        List of result dicts from ``load_shuffle_results``.

    Returns
    -------
    dict
        Summary dict with keys:
        ``n_total``, ``n_surviving``, ``false_positive_rate``,
        ``mean_beta``, ``std_beta``, ``pass_verdict``.
    """
    n_total = len(results)
    n_surviving = sum(1 for r in results if r["excludes_zero"])
    fpr = n_surviving / n_total if n_total > 0 else float("nan")
    mean_beta = sum(r["beta_lec_kappa_mean"] for r in results) / n_total
    std_beta = sum(r["beta_lec_kappa_std"] for r in results) / n_total
    pass_verdict = fpr <= 0.05

    return {
        "n_total": n_total,
        "n_surviving": n_surviving,
        "false_positive_rate": fpr,
        "mean_beta": mean_beta,
        "std_beta": std_beta,
        "pass_verdict": pass_verdict,
    }


def write_summary(summary: dict, results: list[dict], output_path: Path) -> None:
    """Write a Markdown summary report to disk.

    Parameters
    ----------
    summary : dict
        Output of ``summarize_results``.
    results : list[dict]
        Full list of per-shuffle result dicts.
    output_path : Path
        Destination path for the Markdown file.
    """
    n_total = summary["n_total"]
    n_surviving = summary["n_surviving"]
    fpr = summary["false_positive_rate"]
    mean_beta = summary["mean_beta"]
    std_beta = summary["std_beta"]
    verdict = "PASS" if summary["pass_verdict"] else "FAIL"

    # Collect per-shuffle details for the table
    lines = [
        "# Permutation Null Test (L2-06)",
        "",
        "## Summary",
        "",
        f"- Shuffles completed: {n_total}",
        f"- Surviving effects (HDI excludes zero): {n_surviving}",
        f"- False positive rate: {fpr * 100:.1f}%",
        f"- Nominal alpha: 5%",
        f"- Mean beta_lec_kappa (across shuffles): {mean_beta:.4f}",
        f"- Mean beta_lec_kappa SD (across shuffles): {std_beta:.4f}",
        "",
        "## Verdict",
        "",
        f"**{verdict}** — empirical FPR {fpr * 100:.1f}% "
        + ("≤" if summary["pass_verdict"] else ">")
        + " 5% nominal alpha",
        "",
        "## Per-Shuffle Details",
        "",
        "| Shuffle | beta_mean | beta_std | HDI low | HDI high | Excludes 0 | Divergences |",
        "|---------|-----------|----------|---------|----------|------------|-------------|",
    ]

    for r in results:
        exc = "Yes" if r["excludes_zero"] else "No"
        lines.append(
            f"| {r['shuffle_idx']:>7} "
            f"| {r['beta_lec_kappa_mean']:>9.4f} "
            f"| {r['beta_lec_kappa_std']:>8.4f} "
            f"| {r['hdi_low']:>7.4f} "
            f"| {r['hdi_high']:>8.4f} "
            f"| {exc:>10} "
            f"| {r['n_divergences']:>11} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Aggregate permutation null test JSON files into a summary report."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate M3 permutation null test results and write a "
            "Markdown summary report."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(MODELS_BAYESIAN_DIR / "permutation"),
        help=(
            "Directory containing shuffle_*_results.json files "
            "(default: models/bayesian/permutation)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(MODELS_BAYESIAN_DIR / "permutation"),
        help=(
            "Directory for permutation_summary.md output "
            "(default: models/bayesian/permutation)"
        ),
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_path = output_dir / "permutation_summary.md"

    print(f"Loading shuffle results from: {input_dir}")
    results = load_shuffle_results(input_dir)
    print(f"  Loaded {len(results)} shuffle results")

    summary = summarize_results(results)
    n = summary["n_total"]
    n_surv = summary["n_surviving"]
    fpr_pct = summary["false_positive_rate"] * 100
    verdict = "PASS" if summary["pass_verdict"] else "FAIL"

    print(f"\nResults:")
    print(f"  {n_surv}/{n} shuffles had HDI excluding zero ({fpr_pct:.1f}%)")
    print(f"  Verdict: {verdict} (nominal alpha 5%)")

    write_summary(summary, results, output_path)
    print(f"\nSummary written to: {output_path}")


if __name__ == "__main__":
    main()
