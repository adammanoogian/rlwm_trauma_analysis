"""Render the 3-panel posterior-convergence diagnostics figure.

Produces ``manuscript/figures/{model}_posterior_diagnostics.png`` referenced by
``manuscript/paper.qmd`` (@fig-posterior-diagnostics):

- **(a)** R-hat for all parameter sites; dashed line at 1.05.
- **(b)** ESS bulk for group-level sites (``mu_*``, ``sigma_*``, ``beta_*``,
  ``*_mu_pr``, ``*_sigma_pr``); dashed line at 400.
- **(c)** Divergence count across the ``target_accept_prob`` auto-bump
  schedule (0.80 → 0.95 → 0.99). Per-chain counts are shown for the final
  accepted TAP (from the NetCDF's ``sample_stats.diverging``); earlier TAPs
  show totals parsed from the fitting log.

Usage
-----
>>> python scripts/visualization/plot_posterior_diagnostics.py \
...     --posterior output/bayesian/wmrl_m6b_posterior.nc \
...     --log logs/bayesian_m6b_54806682.out \
...     --output manuscript/figures/m6b_posterior_diagnostics.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "manuscript" / "figures"))

from config import load_netcdf_with_validation  # noqa: E402
from plot_utils import (  # noqa: E402
    MODEL_DISPLAY_NAMES,
    TEXT_WIDTH,
    apply_manuscript_style,
)

RHAT_THRESHOLD: float = 1.05
ESS_THRESHOLD: float = 400.0

GROUP_SITE_PATTERNS: tuple[str, ...] = (
    r"^mu_",
    r"^sigma_",
    r"^beta_",
    r"_mu_pr$",
    r"_sigma_pr$",
)

TAP_LINE_RE = re.compile(
    r"\[convergence-gate\]\s+target_accept_prob=(?P<tap>[\d.]+)"
    r"\s+divergences=(?P<div>\d+)"
)


def infer_model_from_path(posterior_path: Path) -> str:
    """Infer the internal model key from a posterior NetCDF filename.

    Parameters
    ----------
    posterior_path : Path
        Path whose stem begins with the internal model key (e.g.
        ``wmrl_m6b_posterior``).

    Returns
    -------
    str
        First matching key in ``MODEL_DISPLAY_NAMES``, or ``'unknown'`` if
        no prefix matches.
    """
    stem = posterior_path.stem.lower()
    for key in sorted(MODEL_DISPLAY_NAMES, key=len, reverse=True):
        if stem.startswith(key):
            return key
    return "unknown"


def is_group_site(site_name: str) -> bool:
    """Return ``True`` if ``site_name`` is a group-level parameter site.

    Parameters
    ----------
    site_name : str
        Parameter site label from ``az.summary`` index (e.g.
        ``mu_alpha_pos``, ``alpha_pos[12]``, ``beta_lec_kappa``).
    """
    base = site_name.split("[")[0]
    return any(re.search(pat, base) for pat in GROUP_SITE_PATTERNS)


def parse_tap_schedule_from_log(log_path: Path) -> list[tuple[float, int]]:
    """Parse ``[convergence-gate]`` lines from a fitting log.

    Parameters
    ----------
    log_path : Path
        Path to the slurm stdout/stderr log produced by ``13_fit_bayesian.py``.

    Returns
    -------
    list[tuple[float, int]]
        Ordered ``(target_accept_prob, total_divergences)`` tuples — one per
        attempt, in the order they appear in the log.
    """
    entries: list[tuple[float, int]] = []
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = TAP_LINE_RE.search(line)
            if m is not None:
                entries.append((float(m.group("tap")), int(m.group("div"))))
    return entries


def compute_per_chain_divergences(idata: az.InferenceData) -> np.ndarray:
    """Sum per-chain divergences from ``sample_stats.diverging``.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData with a ``sample_stats`` group containing ``diverging``.

    Returns
    -------
    np.ndarray
        1-D array of length ``num_chains`` with the divergence count in each
        chain, or ``np.zeros(0)`` if ``sample_stats.diverging`` is absent.
    """
    if "sample_stats" not in idata or "diverging" not in idata.sample_stats:
        return np.zeros(0, dtype=int)
    diverging = idata.sample_stats["diverging"].values
    return diverging.sum(axis=1).astype(int)


def summarize_sites(idata: az.InferenceData) -> pd.DataFrame:
    """Run ``az.summary`` and return all parameter sites.

    Parameters
    ----------
    idata : az.InferenceData
        Posterior samples.

    Returns
    -------
    pd.DataFrame
        Summary table indexed by site name with ``r_hat`` and ``ess_bulk``
        columns.
    """
    summary = az.summary(idata, kind="diagnostics")
    required = {"r_hat", "ess_bulk"}
    missing = required - set(summary.columns)
    if missing:
        raise KeyError(
            f"az.summary missing expected columns: expected {sorted(required)}, "
            f"actual columns={sorted(summary.columns)}"
        )
    return summary


def plot_rhat_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    """Draw panel (a): R-hat histogram across all sites.

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    summary : pd.DataFrame
        Output of ``summarize_sites``.
    """
    rhat = summary["r_hat"].dropna().to_numpy()
    n_sites = rhat.size
    max_rhat = float(np.nanmax(rhat)) if n_sites else float("nan")
    n_fail = int(np.sum(rhat >= RHAT_THRESHOLD))

    bin_upper = max(RHAT_THRESHOLD + 0.01, max_rhat if np.isfinite(max_rhat) else 1.1)
    bins = np.linspace(0.99, bin_upper, 40)
    ax.hist(rhat, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.axvline(
        RHAT_THRESHOLD,
        color="#D62246",
        linestyle="--",
        linewidth=1.0,
        label=f"threshold = {RHAT_THRESHOLD}",
    )
    ax.set_xlabel(r"$\hat{R}$")
    ax.set_ylabel("Count (parameter sites)")
    ax.set_title(
        f"(a) $\\hat{{R}}$ across {n_sites} sites\n"
        f"max = {max_rhat:.3f}, {n_fail} \u2265 {RHAT_THRESHOLD}"
    )
    ax.legend(loc="upper right", frameon=False)


def plot_ess_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    """Draw panel (b): ESS bulk for group-level sites.

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    summary : pd.DataFrame
        Output of ``summarize_sites``.
    """
    mask = summary.index.to_series().map(is_group_site)
    group_summary = summary.loc[mask, "ess_bulk"].dropna().sort_values()
    n_group = len(group_summary)
    if n_group == 0:
        ax.text(
            0.5,
            0.5,
            "No group-level sites detected",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    y = np.arange(n_group)
    colors = np.where(group_summary.to_numpy() < ESS_THRESHOLD, "#D62246", "#06A77D")
    ax.barh(y, group_summary.to_numpy(), color=colors, edgecolor="white", linewidth=0.3)
    ax.axvline(
        ESS_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"threshold = {int(ESS_THRESHOLD)}",
    )
    ax.set_yticks(y)
    tick_fontsize = 6 if n_group > 25 else 7 if n_group > 12 else 8
    ax.set_yticklabels(group_summary.index, fontsize=tick_fontsize)
    ax.set_xlabel("ESS bulk")
    ax.set_title(f"(b) ESS bulk, {n_group} group-level sites")
    ax.legend(loc="lower right", frameon=False)


def plot_divergence_panel(
    ax: plt.Axes,
    tap_entries: list[tuple[float, int]],
    per_chain_final: np.ndarray,
) -> None:
    """Draw panel (c): divergences across the TAP auto-bump schedule.

    Parameters
    ----------
    ax : plt.Axes
        Target axes.
    tap_entries : list[tuple[float, int]]
        ``(tap, total_divergences)`` tuples parsed from the fitting log, in
        attempt order. Empty if no log was provided.
    per_chain_final : np.ndarray
        Per-chain divergence counts from the accepted (final) run. Used both
        to annotate the last TAP bar and as a fallback when no log was
        provided.
    """
    num_chains = int(per_chain_final.size)

    if tap_entries:
        taps = [tap for tap, _ in tap_entries]
        totals = [n for _, n in tap_entries]
        x = np.arange(len(taps))
        ax.bar(x, totals, color="#4C72B0", edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t:.2f}" for t in taps])

        if num_chains and taps:
            jitter = np.linspace(-0.25, 0.25, num_chains) if num_chains > 1 else [0.0]
            for ci, val in enumerate(per_chain_final):
                ax.scatter(
                    x[-1] + jitter[ci],
                    int(val),
                    color="#D62246",
                    s=20,
                    zorder=3,
                    label="per-chain (final TAP)" if ci == 0 else None,
                )
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc="upper right", frameon=False)

        ax.set_xlabel(r"$\mathtt{target\_accept\_prob}$")
        ax.set_ylabel("Divergences (total across chains)")
        ax.set_title("(c) NUTS auto-bump schedule")
    elif num_chains:
        x = np.arange(num_chains)
        ax.bar(
            x,
            per_chain_final,
            color="#4C72B0",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"chain {i}" for i in range(num_chains)])
        ax.set_xlabel("Chain (final accepted TAP)")
        ax.set_ylabel("Divergences")
        ax.set_title("(c) Divergences per chain (final TAP)")
    else:
        ax.text(
            0.5,
            0.5,
            "No divergence data available\n(no log; no sample_stats.diverging)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    ymax = max(
        (max(totals) if tap_entries else 0),
        int(per_chain_final.max()) if num_chains else 0,
        1,
    )
    ax.set_ylim(0, ymax * 1.15 + 1)


def render_figure(
    posterior_path: Path,
    log_path: Path | None,
    output_path: Path,
) -> None:
    """Build the 3-panel diagnostics figure and write it to disk.

    Parameters
    ----------
    posterior_path : Path
        Input NetCDF posterior (from ``13_fit_bayesian.py``).
    log_path : Path or None
        Optional fitting log for panel (c) TAP-schedule parsing.
    output_path : Path
        Destination PNG path; parent directory is created if missing.
    """
    apply_manuscript_style()

    model_key = infer_model_from_path(posterior_path)
    model_label = MODEL_DISPLAY_NAMES.get(model_key, model_key)

    print(f">> Loading posterior: {posterior_path}")
    idata = load_netcdf_with_validation(posterior_path, model_key)
    print(f">> Model: {model_label}")

    print(">> Computing az.summary diagnostics...")
    summary = summarize_sites(idata)

    per_chain_final = compute_per_chain_divergences(idata)
    print(f">> Final-run per-chain divergences: {per_chain_final.tolist()}")

    if log_path is not None:
        print(f">> Parsing TAP schedule from log: {log_path}")
        tap_entries = parse_tap_schedule_from_log(log_path)
        print(f">> Parsed {len(tap_entries)} TAP entries: {tap_entries}")
    else:
        tap_entries = []
        print(">> No --log provided; panel (c) will show final-TAP per-chain only.")

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 3.2))
    plot_rhat_panel(axes[0], summary)
    plot_ess_panel(axes[1], summary)
    plot_divergence_panel(axes[2], tap_entries, per_chain_final)

    fig.suptitle(
        f"{model_label}: posterior convergence diagnostics",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f">> Wrote figure: {output_path}")


def main() -> None:
    """Parse CLI arguments and render the diagnostics figure."""
    parser = argparse.ArgumentParser(
        description=(
            "Render the 3-panel posterior-convergence diagnostics figure "
            "referenced by manuscript/paper.qmd (@fig-posterior-diagnostics)."
        )
    )
    parser.add_argument(
        "--posterior",
        type=Path,
        required=True,
        help="Path to ArviZ NetCDF posterior file.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help=(
            "Optional fitting log (slurm .out) for panel (c) TAP-schedule "
            "parsing. Looks for '[convergence-gate] target_accept_prob=... "
            "divergences=...' lines."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path. Defaults to "
            "manuscript/figures/{model}_posterior_diagnostics.png."
        ),
    )
    args = parser.parse_args()

    posterior_path: Path = args.posterior
    if not posterior_path.exists():
        raise FileNotFoundError(
            f"Posterior file not found: expected existing path, actual={posterior_path}"
        )

    if args.log is not None and not args.log.exists():
        raise FileNotFoundError(
            f"Log file not found: expected existing path, actual={args.log}"
        )

    model_key = infer_model_from_path(posterior_path)
    short_names: dict[str, str] = {
        "qlearning": "m1",
        "wmrl": "m2",
        "wmrl_m3": "m3",
        "wmrl_m4": "m4",
        "wmrl_m5": "m5",
        "wmrl_m6a": "m6a",
        "wmrl_m6b": "m6b",
    }
    short_name = short_names.get(model_key, model_key)

    if args.output is None:
        output_path = (
            PROJECT_ROOT
            / "manuscript"
            / "figures"
            / f"{short_name}_posterior_diagnostics.png"
        )
    else:
        output_path = args.output

    render_figure(posterior_path, args.log, output_path)


if __name__ == "__main__":
    main()
