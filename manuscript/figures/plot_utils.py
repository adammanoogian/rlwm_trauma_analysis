"""
Publication-quality matplotlib style utilities for the RLWM trauma analysis manuscript.

Mirrors the project's plotting_config.py conventions but tuned for journal
figures: serif fonts, smaller font sizes, 300 DPI, and standard column widths.

Usage
-----
>>> from plot_utils import apply_manuscript_style, GROUP_COLORS, COLUMN_WIDTH
>>> apply_manuscript_style()
>>> fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# ── Page geometry (inches, two-column journal standard) ───────────────────────

COLUMN_WIDTH: float = 3.5
"""Single-column figure width in inches (standard two-column journal)."""

TEXT_WIDTH: float = 7.0
"""Full-text-width figure width in inches (standard two-column journal)."""

# ── rcParams for publication figures ─────────────────────────────────────────

MANUSCRIPT_STYLE: dict[str, object] = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
}
"""Matplotlib rcParams dict for publication-quality manuscript figures."""

# ── Trauma group colors (keyed by actual group names from group_assignments.csv) ──

GROUP_COLORS: dict[str, str] = {
    "Trauma - No Ongoing Impact": "#06A77D",  # Green
    "Trauma - Ongoing Impact": "#D62246",     # Red
}
"""
Trauma group color palette.

Keys match the ``hypothesis_group`` column in group_assignments.csv.
Two groups: no ongoing impact (green) and ongoing impact (red).
"""

GROUP_SHORT_LABELS: dict[str, str] = {
    "Trauma - No Ongoing Impact": "No Impact",
    "Trauma - Ongoing Impact": "Ongoing Impact",
}
"""
Short axis tick labels for trauma groups.

Maps full group names (from ``hypothesis_group`` column) to compact labels
suitable for matplotlib axis tick marks.
"""

# ── Model short-name to internal key reverse lookup ──────────────────────────

SHORT_NAME_TO_KEY: dict[str, str] = {
    "M1": "qlearning",
    "M2": "wmrl",
    "M3": "wmrl_m3",
    "M4": "wmrl_m4",
    "M5": "wmrl_m5",
    "M6a": "wmrl_m6a",
    "M6b": "wmrl_m6b",
}
"""
Reverse lookup from comparison_results.csv short name to internal model key.

The ``model`` column in comparison_results.csv uses short names (M6b, M5, ...);
this dict maps back to the internal keys used in MODEL_REGISTRY and CSV filenames.
"""

# ── Model display names ───────────────────────────────────────────────────────

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "qlearning": "M1: Q-Learning",
    "wmrl": "M2: WM-RL",
    "wmrl_m3": "M3: WM-RL+kappa",
    "wmrl_m4": "M4: RLWM-LBA",
    "wmrl_m5": "M5: WM-RL+phi_rl",
    "wmrl_m6a": "M6a: WM-RL+kappa_s",
    "wmrl_m6b": "M6b: WM-RL+dual",
}
"""
Human-readable display names for all seven computational models.

Keys are the internal model identifiers used in CSV filenames and
``--model`` CLI arguments; values are publication-ready labels.
"""

# ── Parameter display names (LaTeX-formatted) ─────────────────────────────────

PARAM_DISPLAY_NAMES: dict[str, str] = {
    "alpha_pos": r"$\alpha_+$",
    "alpha_neg": r"$\alpha_-$",
    "phi": r"$\phi$",
    "rho": r"$\rho$",
    "K": r"$K$",
    "capacity": r"$K$",
    "kappa": r"$\kappa$",
    "kappa_s": r"$\kappa_s$",
    "kappa_total": r"$\kappa_{\mathrm{total}}$",
    "kappa_share": r"$\kappa_{\mathrm{share}}$",
    "phi_rl": r"$\phi_{\mathrm{RL}}$",
    "epsilon": r"$\varepsilon$",
}
"""
LaTeX-formatted display names for model parameters.

Keys are column names in MLE individual fits CSVs; values are
math-mode strings suitable for matplotlib axis labels and legend entries.
"""


def apply_manuscript_style() -> None:
    """
    Apply publication-quality rcParams to the current matplotlib session.

    Updates ``plt.rcParams`` with ``MANUSCRIPT_STYLE``. Call once at the
    top of each notebook or script that generates manuscript figures.

    Examples
    --------
    >>> apply_manuscript_style()
    >>> fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))
    """
    plt.rcParams.update(MANUSCRIPT_STYLE)
