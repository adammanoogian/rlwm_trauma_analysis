"""09 — Cross-model parameter x trauma significance heatmap.

Reads MLE regression CSVs for M3, M5, M6a, M6b and produces a 2x2
panel heatmap for the manuscript appendix. Cell color encodes Pearson r
(diverging blue-white-red); annotations encode uncorrected significance
(* / ** / ***) and correction survival. Within-model Bonferroni
correction (n_params x n_predictors tests per model).

Output
------
reports/figures/model_comparison/cross_model_significance_heatmap.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
REGRESSION_DIR = ROOT / "reports" / "tables" / "regressions"
OUTPUT_DIR = ROOT / "reports" / "figures" / "model_comparison"

MODELS = ["wmrl_m3", "wmrl_m5", "wmrl_m6a", "wmrl_m6b"]
MODEL_LABELS = {
    "wmrl_m3": "M3: WM-RL + κ",
    "wmrl_m5": "M5: WM-RL + φ_rl",
    "wmrl_m6a": "M6a: WM-RL + κ_s",
    "wmrl_m6b": "M6b: WM-RL + dual κ (winner)",
}

PREDICTOR_ORDER = [
    "lec_total_events",
    "lec_personal_events",
    "ies_total",
    "ies_hyperarousal",
    "ies_intrusion",
    "ies_avoidance",
]
PREDICTOR_LABELS = {
    "lec_total_events": "LEC-5\nTotal",
    "lec_personal_events": "LEC-5\nPersonal",
    "ies_total": "IES-R\nTotal",
    "ies_hyperarousal": "IES-R\nHyper.",
    "ies_intrusion": "IES-R\nIntrus.",
    "ies_avoidance": "IES-R\nAvoid.",
}

PARAM_ORDER: dict[str, list[str]] = {
    "wmrl_m3": [
        "kappa_mean",
        "phi_mean",
        "rho_mean",
        "wm_capacity_mean",
        "alpha_pos_mean",
        "alpha_neg_mean",
        "epsilon_mean",
    ],
    "wmrl_m5": [
        "kappa_mean",
        "phi_rl_mean",
        "phi_mean",
        "rho_mean",
        "wm_capacity_mean",
        "alpha_pos_mean",
        "alpha_neg_mean",
        "epsilon_mean",
    ],
    "wmrl_m6a": [
        "kappa_s_mean",
        "phi_mean",
        "rho_mean",
        "wm_capacity_mean",
        "alpha_pos_mean",
        "alpha_neg_mean",
        "epsilon_mean",
    ],
    "wmrl_m6b": [
        "kappa_total_mean",
        "kappa_share_mean",
        "phi_mean",
        "rho_mean",
        "wm_capacity_mean",
        "alpha_pos_mean",
        "alpha_neg_mean",
        "epsilon_mean",
    ],
}

PARAM_LABELS = {
    "kappa_mean": "κ (perseveration)",
    "kappa_s_mean": "κ_s (stim. persev.)",
    "kappa_total_mean": "κ_total (persev. budget)",
    "kappa_share_mean": "κ_share (global fraction)",
    "phi_mean": "φ (WM decay)",
    "phi_rl_mean": "φ_rl (RL forgetting)",
    "rho_mean": "ρ (WM weight)",
    "wm_capacity_mean": "K (WM capacity)",
    "alpha_pos_mean": "α+ (pos. learning)",
    "alpha_neg_mean": "α− (neg. learning)",
    "epsilon_mean": "ε (noise)",
}


def _parse_p(p_str: str | float) -> float:
    """Strip significance markers from CSV p-value strings."""
    if isinstance(p_str, (int, float)):
        return float(p_str)
    return float(str(p_str).replace("*", "").strip())


def load_model_data(model_key: str) -> pd.DataFrame:
    """Load regression CSV, compute exact p and Bonferroni correction."""
    path = REGRESSION_DIR / model_key / "regression_results_simple.csv"
    df = pd.read_csv(path)
    df["p_fdr_num"] = df["p_fdr"].apply(_parse_p)
    df["p_exact"] = 2 * (1 - stats.t.cdf(np.abs(df["t"]), df=df["N"] - 2))
    n_tests = len(df)
    df["p_bonf"] = np.minimum(df["p_exact"] * n_tests, 1.0)
    return df


def _draw_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    model_key: str,
    *,
    show_ylabel: bool,
    show_xlabel: bool,
) -> plt.cm.ScalarMappable:
    """Draw one model's heatmap panel."""
    params = PARAM_ORDER[model_key]
    n_p, n_pred = len(params), len(PREDICTOR_ORDER)

    r_mat = np.full((n_p, n_pred), np.nan)
    p_mat = np.full((n_p, n_pred), np.nan)
    fdr_mat = np.full((n_p, n_pred), np.nan)
    bonf_mat = np.full((n_p, n_pred), np.nan)

    for i, par in enumerate(params):
        for j, pred in enumerate(PREDICTOR_ORDER):
            row = df[(df["Parameter"] == par) & (df["Predictor"] == pred)]
            if len(row) == 1:
                r_mat[i, j] = row["r"].values[0]
                p_mat[i, j] = row["p_exact"].values[0]
                fdr_mat[i, j] = row["p_fdr_num"].values[0]
                bonf_mat[i, j] = row["p_bonf"].values[0]

    norm = TwoSlopeNorm(vmin=-0.30, vcenter=0, vmax=0.30)
    im = ax.imshow(r_mat, cmap="RdBu_r", norm=norm, aspect="auto")

    for i in range(n_p):
        for j in range(n_pred):
            p = p_mat[i, j]
            if np.isnan(p):
                ax.text(
                    j, i, "—", ha="center", va="center", fontsize=7, color="gray"
                )
                continue

            if p >= 0.05:
                ax.text(
                    j, i, "×", ha="center", va="center",
                    fontsize=9, color="gray", alpha=0.4,
                )
                continue

            stars = "***" if p < 0.001 else ("**" if p < 0.01 else "*")

            if bonf_mat[i, j] < 0.05:
                color, fw = "#B8860B", "bold"
                rect = mpatches.FancyBboxPatch(
                    (j - 0.46, i - 0.46), 0.92, 0.92,
                    boxstyle="round,pad=0.04",
                    linewidth=2.5, edgecolor="goldenrod", facecolor="none",
                )
                ax.add_patch(rect)
            elif fdr_mat[i, j] < 0.05:
                color, fw = "#006400", "bold"
                rect = mpatches.FancyBboxPatch(
                    (j - 0.46, i - 0.46), 0.92, 0.92,
                    boxstyle="round,pad=0.04",
                    linewidth=2.2, edgecolor="seagreen", facecolor="none",
                )
                ax.add_patch(rect)
            else:
                color, fw = "black", "normal"

            ax.text(
                j, i, stars, ha="center", va="center",
                fontsize=10, color=color, fontweight=fw,
            )

    ax.set_xticks(range(n_pred))
    ax.set_xticklabels(
        [PREDICTOR_LABELS[p] for p in PREDICTOR_ORDER] if show_xlabel else [],
        fontsize=7,
    )
    ax.set_yticks(range(n_p))
    ax.set_yticklabels(
        [PARAM_LABELS.get(p, p) for p in params] if show_ylabel else [],
        fontsize=8,
    )
    ax.set_title(MODEL_LABELS[model_key], fontsize=10, fontweight="bold", pad=6)

    for x in np.arange(-0.5, n_pred, 1):
        ax.axvline(x, color="white", linewidth=0.5)
    for y in np.arange(-0.5, n_p, 1):
        ax.axhline(y, color="white", linewidth=0.5)

    return im


def main() -> None:
    """Generate cross-model significance heatmap."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2, 2, figsize=(11, 9.5),
        gridspec_kw={"hspace": 0.35, "wspace": 0.35},
    )

    im = None
    for idx, model_key in enumerate(MODELS):
        ax = axes.flat[idx]
        df = load_model_data(model_key)
        im = _draw_panel(
            ax, df, model_key,
            show_ylabel=(idx % 2 == 0),
            show_xlabel=(idx >= 2),
        )

    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Pearson r")

    legend_handles = [
        plt.Line2D(
            [0], [0], marker="$×$", color="gray", linestyle="",
            markersize=10, alpha=0.5, label="p ≥ 0.05",
        ),
        plt.Line2D(
            [0], [0], marker="$*$", color="black", linestyle="",
            markersize=10, label="* p < 0.05 (uncorr.)",
        ),
        plt.Line2D(
            [0], [0], marker="$**$", color="black", linestyle="",
            markersize=12, label="** p < 0.01",
        ),
        plt.Line2D(
            [0], [0], marker="$***$", color="black", linestyle="",
            markersize=14, label="*** p < 0.001",
        ),
        mpatches.Patch(
            facecolor="none", edgecolor="seagreen", linewidth=2.2,
            label="Survives FDR-BH (q < 0.05)",
        ),
        mpatches.Patch(
            facecolor="none", edgecolor="goldenrod", linewidth=2.5,
            label="Survives Bonferroni",
        ),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=3,
        fontsize=8, bbox_to_anchor=(0.46, -0.01),
        frameon=True, fancybox=True, shadow=False,
    )

    fig.suptitle(
        "Cross-Model Parameter × Trauma Association Matrix (MLE, N = 154)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    for fmt in ("png", "pdf"):
        out = OUTPUT_DIR / f"cross_model_significance_heatmap.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved to {OUTPUT_DIR}/cross_model_significance_heatmap.{{png,pdf}}")

    # Print summary for verification
    print("\n=== Significant associations (p < 0.05 uncorrected) ===")
    for model_key in MODELS:
        df = load_model_data(model_key)
        sig = df[df["p_exact"] < 0.05].sort_values("p_exact")
        if len(sig) > 0:
            label = MODEL_LABELS[model_key]
            print(f"\n{label} ({len(df)} tests, Bonf threshold "
                  f"= {0.05 / len(df):.5f}):")
            for _, row in sig.iterrows():
                tag = ""
                if row["p_bonf"] < 0.05:
                    tag = " [BONF]"
                elif row["p_fdr_num"] < 0.05:
                    tag = " [FDR]"
                plabel = PARAM_LABELS.get(row["Parameter"], row["Parameter"])
                print(
                    f"  {plabel:32s} × {row['Predictor']:20s}: "
                    f"r={row['r']:+.3f}, p={row['p_exact']:.5f}{tag}"
                )


if __name__ == "__main__":
    main()
