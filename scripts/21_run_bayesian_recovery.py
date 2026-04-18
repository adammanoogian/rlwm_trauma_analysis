"""Step 21.2 — Bayesian parameter recovery for all 6 choice-only models.

Per Baribault & Collins (2023) gate 3 and Hess et al. (2025) stage 4, this
script validates that identifiable parameters (the kappa family) recover
with Pearson r >= 0.80 and 95% HDI coverage >= 0.90 under hierarchical
Bayesian inference.  Non-identifiable parameters (alpha_pos, alpha_neg,
phi, rho, capacity, epsilon) are labelled ``"descriptive only"`` in the
output — they match the MLE recovery findings (quick-005: r = 0.21 to
0.77) and serve only as individual-level descriptors.

Two CLI modes
-------------

``single-subject``
    Fit ONE synthetic dataset via hierarchical MCMC.  Designed to be
    dispatched by a SLURM array job (one task per synthetic subject).

    1. Draw a true parameter vector from the prior
       ``PARAM_PRIOR_DEFAULTS`` using the same ``phi_approx``
       transform the model uses at sampling time.  RNG key is derived
       from ``(subject_idx, seed)``.
    2. Generate trial data with
       ``scripts.fitting.model_recovery.generate_synthetic_participant``.
    3. Fit via ``scripts.fitting.fit_bayesian._fit_stacked_model`` with
       reduced budget (warmup=500, samples=1000, chains=2,
       max_tree_depth=8).
    4. Extract per-parameter ``posterior_mean``, ``hdi_low``,
       ``hdi_high`` and ``in_hdi`` from the MCMC samples via
       ``arviz.hdi``.
    5. Write ``{output_dir}/{model}_subject_{idx:03d}.json`` with the
       per-parameter comparison and MCMC metadata
       (``max_rhat``, ``min_ess``, ``num_divergences``).

``aggregate``
    Combine the per-subject JSONs into a single recovery table and
    evaluate the pass criterion.  Writes
    ``{output_dir}/{model}_recovery.csv`` and
    ``{output_dir}/{model}_recovery_summary.md``.  Exit code 0 if all
    kappa-family parameters pass (or if the model has no kappa
    parameters, i.e. M1 / M2); exit 1 if any kappa-family parameter
    fails.

Scope note (Phase 21, 2026-04-18)
---------------------------------

The 2-covariate L2 hook added to M3/M5/M6a in plan 21-11 is gated via
``scripts/fitting/tests/test_numpyro_models_2cov.py::test_recovery_2cov_m3``
— it is intentionally NOT part of this production recovery sweep.
Adding the second-covariate site to the production run would triple
the cluster cost for a confirmatory test the local pytest already
covers.  This script always exercises the baseline (no-covariate)
inference path.

References
----------

* Baribault, B. & Collins, A. G. E. (2023).  Troubleshooting Bayesian
  cognitive models.  Psychological Methods.
* Hess, S. et al. (2025).  Structured workflow for Bayesian model
  selection in cognitive tasks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

# -- Path bootstrap so this script runs both interactively and under SLURM
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import arviz as az  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import MODEL_REGISTRY  # noqa: E402
from scripts.fitting.fit_bayesian import STACKED_MODEL_DISPATCH, _fit_stacked_model  # noqa: E402
from scripts.fitting.model_recovery import generate_synthetic_participant  # noqa: E402
from scripts.fitting.numpyro_helpers import PARAM_PRIOR_DEFAULTS, phi_approx  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KAPPA_FAMILY: frozenset[str] = frozenset(
    {"kappa", "kappa_s", "kappa_total", "kappa_share"}
)
"""Parameters subject to the pass/fail recovery criterion.

All other parameters are flagged ``"descriptive only"`` in the
aggregate CSV.  This matches the MLE recovery findings (quick-005):
only kappa-family parameters are reliably identifiable under this
task structure.
"""

KAPPA_PEARSON_THRESHOLD: float = 0.80
KAPPA_HDI_COVERAGE_THRESHOLD: float = 0.90
HDI_PROB: float = 0.95

DEFAULT_WARMUP: int = 500
DEFAULT_SAMPLES: int = 1000
DEFAULT_CHAINS: int = 2
DEFAULT_MAX_TREE_DEPTH: int = 8

DEFAULT_N_BLOCKS: int = 17
DEFAULT_TRIALS_PER_BLOCK: int = 100


# ---------------------------------------------------------------------------
# Prior sampling (matches numpyro_helpers.sample_bounded_param semantics)
# ---------------------------------------------------------------------------


def sample_true_params_from_prior(
    model: str,
    subject_idx: int,
    seed: int,
) -> dict[str, float]:
    """Sample one true parameter vector from the prior for a synthetic subject.

    Mirrors the non-centered hBayesDM transform used inside every
    hierarchical model at sampling time:

        z          ~ Normal(0, 1)
        theta_unc  = mu_prior_loc + sigma_scale * z
        theta      = lower + (upper - lower) * Phi(theta_unc)

    With ``sigma_scale = 0.2`` matching the ``HalfNormal(0.2)`` prior on
    ``sigma_pr`` from ``sample_bounded_param`` — a fixed scale is used
    here because we are sampling a single subject's individual
    parameter vector, not inferring the group-level scale.

    For M6b, both ``kappa_total`` and ``kappa_share`` are sampled and
    then decoded into ``kappa`` and ``kappa_s`` so the simulator in
    ``model_recovery.generate_synthetic_participant`` receives the
    correct inputs.

    Parameters
    ----------
    model : str
        Model name (key of ``MODEL_REGISTRY``).  Must be one of the six
        choice-only models.
    subject_idx : int
        1-indexed subject number (from the SLURM array task id).
    seed : int
        Base RNG seed, combined with ``subject_idx`` to produce a unique
        key per array task.

    Returns
    -------
    dict[str, float]
        Mapping from parameter name to sampled scalar.  Includes all
        parameters in ``MODEL_REGISTRY[model]['params']``.

    Raises
    ------
    ValueError
        If ``model`` is not in ``MODEL_REGISTRY`` or if a required
        parameter is missing from ``PARAM_PRIOR_DEFAULTS``.
    """
    if model not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model!r}. "
            f"Expected one of: {sorted(MODEL_REGISTRY.keys())}"
        )

    params_list: list[str] = list(MODEL_REGISTRY[model]["params"])

    # Derive a unique RNG key for this (subject_idx, seed) combination.
    # Fold the subject index in so each array task gets distinct draws.
    key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), int(subject_idx))

    sigma_scale: float = 0.2  # Matches HalfNormal(0.2) prior on sigma_pr

    out: dict[str, float] = {}
    for pname in params_list:
        if pname not in PARAM_PRIOR_DEFAULTS:
            raise ValueError(
                f"Parameter {pname!r} missing from PARAM_PRIOR_DEFAULTS. "
                f"Expected keys: {sorted(PARAM_PRIOR_DEFAULTS.keys())}"
            )

        defaults = PARAM_PRIOR_DEFAULTS[pname]
        lower = float(defaults["lower"])
        upper = float(defaults["upper"])
        mu_prior_loc = float(defaults["mu_prior_loc"])

        key, subkey = jax.random.split(key)
        z = float(jax.random.normal(subkey))
        theta_unc = mu_prior_loc + sigma_scale * z
        theta = lower + (upper - lower) * float(phi_approx(jnp.asarray(theta_unc)))
        out[pname] = theta

    return out


# ---------------------------------------------------------------------------
# Single-subject mode
# ---------------------------------------------------------------------------


def _extract_hdi_per_param(
    mcmc: Any,
    param_names: list[str],
    hdi_prob: float = HDI_PROB,
) -> dict[str, dict[str, float]]:
    """Extract per-parameter posterior mean and HDI for ONE synthetic subject.

    The hierarchical model produces posterior samples of shape
    ``(n_chains * n_samples, n_participants)`` for each parameter.  With
    a single synthetic subject ``n_participants == 1``, so we squeeze
    the participant axis and compute posterior summaries over the
    draw axis.

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        Fitted MCMC object from ``_fit_stacked_model``.
    param_names : list[str]
        Parameter names to extract.  Must all be registered as
        ``numpyro.deterministic`` sites in the model.
    hdi_prob : float
        HDI probability mass.  Default ``0.95``.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping from parameter name to ``{"posterior_mean": float,
        "hdi_low": float, "hdi_high": float}``.
    """
    samples = mcmc.get_samples()
    out: dict[str, dict[str, float]] = {}

    for pname in param_names:
        if pname not in samples:
            # Parameter not sampled (e.g. LBA params for choice-only fit)
            warnings.warn(
                f"Parameter {pname!r} not found in MCMC samples. "
                f"Available: {sorted(samples.keys())}"
            )
            continue

        arr = np.asarray(samples[pname])
        # Shape expected: (n_draws, n_participants=1) — squeeze participant axis
        if arr.ndim == 2:
            arr = arr[:, 0]
        elif arr.ndim != 1:
            raise ValueError(
                f"Expected 1-D or 2-D posterior for {pname!r}, "
                f"got shape {arr.shape}"
            )

        posterior_mean = float(arr.mean())
        hdi_result = az.hdi(arr, hdi_prob=hdi_prob)
        hdi_low = float(np.asarray(hdi_result).ravel()[0])
        hdi_high = float(np.asarray(hdi_result).ravel()[1])

        out[pname] = {
            "posterior_mean": posterior_mean,
            "hdi_low": hdi_low,
            "hdi_high": hdi_high,
        }

    return out


def _extract_convergence_diagnostics(
    mcmc: Any, param_names: list[str]
) -> dict[str, float | int]:
    """Extract max R-hat, min ESS-bulk and divergence count.

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        Fitted MCMC object.
    param_names : list[str]
        Parameter names to include in the R-hat/ESS summary.

    Returns
    -------
    dict[str, float | int]
        Keys: ``max_rhat``, ``min_ess``, ``num_divergences``.
        ``max_rhat`` and ``min_ess`` can be NaN if ArviZ cannot compute
        them (e.g. single-chain fits).
    """
    try:
        idata = az.from_numpyro(mcmc)
        summary = az.summary(idata, var_names=param_names)
        max_rhat = float(summary["r_hat"].max())
        min_ess = float(summary["ess_bulk"].min())
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Could not compute R-hat/ESS summary: {exc!r}")
        max_rhat = float("nan")
        min_ess = float("nan")

    try:
        extra = mcmc.get_extra_fields()
        n_div = int(np.asarray(extra.get("diverging", np.array([]))).sum())
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Could not compute divergence count: {exc!r}")
        n_div = 0

    return {
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "num_divergences": n_div,
    }


def run_single_subject(
    model: str,
    subject_idx: int,
    seed: int,
    output_dir: Path,
    num_warmup: int = DEFAULT_WARMUP,
    num_samples: int = DEFAULT_SAMPLES,
    num_chains: int = DEFAULT_CHAINS,
    max_tree_depth: int = DEFAULT_MAX_TREE_DEPTH,
    n_blocks: int = DEFAULT_N_BLOCKS,
    n_trials_per_block: int = DEFAULT_TRIALS_PER_BLOCK,
) -> Path:
    """Run a single-subject Bayesian parameter recovery fit.

    Samples one true parameter vector, generates synthetic data, fits
    via hierarchical MCMC, and writes per-parameter recovery JSON.

    Parameters
    ----------
    model : str
        Model name (one of the 6 choice-only models).
    subject_idx : int
        1-indexed subject number (array task id).
    seed : int
        RNG seed for reproducibility.
    output_dir : Path
        Directory to write ``{model}_subject_{idx:03d}.json``.
    num_warmup, num_samples, num_chains, max_tree_depth : int
        MCMC budget overrides.  Defaults match plan 21-03 spec.
    n_blocks, n_trials_per_block : int
        Synthetic data size overrides.  Defaults match plan spec
        (17 blocks x 100 trials ~ 1700 trials per subject); set
        smaller for smoke tests.

    Returns
    -------
    Path
        Path to the written JSON file.

    Raises
    ------
    ValueError
        If ``model`` is unknown or not choice-only.
    """
    if model not in STACKED_MODEL_DISPATCH:
        raise ValueError(
            f"Recovery runner only supports choice-only models "
            f"(keys of STACKED_MODEL_DISPATCH). Got {model!r}, "
            f"expected one of: {sorted(STACKED_MODEL_DISPATCH.keys())}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print("=" * 80)
    print(f"STEP 21.2 SINGLE-SUBJECT RECOVERY: model={model}, subject={subject_idx}")
    print("=" * 80)
    print(f"  Seed: {seed}")
    print(f"  MCMC budget: warmup={num_warmup}, samples={num_samples}, "
          f"chains={num_chains}, max_tree_depth={max_tree_depth}")
    print(f"  Synthetic task: n_blocks={n_blocks}, "
          f"n_trials_per_block={n_trials_per_block}")

    # ------------------------------------------------------------------
    # 1. Sample true parameters from the prior
    # ------------------------------------------------------------------
    print("\n>> Sampling true parameters from prior...")
    true_params = sample_true_params_from_prior(model, subject_idx, seed)
    for pname, val in sorted(true_params.items()):
        print(f"  {pname:14s} = {val:.4f}")

    # ------------------------------------------------------------------
    # 2. Generate synthetic trial data.
    #    model_recovery.generate_synthetic_participant drives blocks 3..(3+21),
    #    so we slice to the requested n_blocks (default 17).  The seed here is
    #    derived from (subject_idx, base seed) so the synthetic stream differs
    #    per array task.
    # ------------------------------------------------------------------
    print("\n>> Generating synthetic trial data...")
    sim_seed = int(seed) * 7919 + int(subject_idx)  # Avoid collision with MCMC seed
    synth_df_full = generate_synthetic_participant(true_params, model, sim_seed)

    # Restrict to requested n_blocks (slicing sorted block numbers) for smoke
    # tests; in production n_blocks=17 approx matches real cohort block counts.
    unique_blocks = sorted(synth_df_full["block"].unique())
    selected_blocks = unique_blocks[: int(n_blocks)]
    synth_df = synth_df_full[synth_df_full["block"].isin(selected_blocks)].copy()

    # Trim to at most n_trials_per_block per block so smoke tests finish fast
    trimmed = []
    for block_num in selected_blocks:
        block_df = synth_df[synth_df["block"] == block_num]
        if len(block_df) > int(n_trials_per_block):
            trimmed.append(block_df.iloc[: int(n_trials_per_block)].copy())
        else:
            trimmed.append(block_df.copy())
    synth_df = pd.concat(trimmed, ignore_index=True)

    # Ensure expected columns and dtypes for prepare_stacked_participant_data.
    # model_recovery writes sona_id, block, stimulus, key_press, reward,
    # set_size, trial_in_block — which is exactly the stacking contract.
    synth_df["stimulus"] = synth_df["stimulus"].astype(int)
    synth_df["key_press"] = synth_df["key_press"].astype(int)
    synth_df["reward"] = synth_df["reward"].astype(float)

    print(f"  Generated {len(synth_df)} trials across "
          f"{synth_df['block'].nunique()} blocks")
    print(f"  Mean reward (proxy accuracy): {synth_df['reward'].mean():.3f}")

    # ------------------------------------------------------------------
    # 3. Fit via hierarchical MCMC (N=1 participant — hierarchy degenerates
    #    to flat priors, which is fine for recovery validation).
    # ------------------------------------------------------------------
    print("\n>> Fitting synthetic data via hierarchical MCMC...")
    mcmc_seed = int(seed) * 131 + int(subject_idx) * 17 + 42
    model_fn = STACKED_MODEL_DISPATCH[model]
    mcmc, _ = _fit_stacked_model(
        data=synth_df,
        model=model,
        model_fn=model_fn,
        num_warmup=int(num_warmup),
        num_samples=int(num_samples),
        num_chains=int(num_chains),
        seed=int(mcmc_seed),
        max_tree_depth=int(max_tree_depth),
    )

    # ------------------------------------------------------------------
    # 4. Extract per-parameter posterior summaries
    # ------------------------------------------------------------------
    print("\n>> Extracting posterior summaries and HDIs...")
    param_names = list(MODEL_REGISTRY[model]["params"])
    summaries = _extract_hdi_per_param(mcmc, param_names, hdi_prob=HDI_PROB)

    # Compute in_hdi per parameter
    per_param: dict[str, dict[str, float]] = {}
    for pname in param_names:
        if pname not in summaries:
            continue
        true_val = float(true_params[pname])
        hdi_low = float(summaries[pname]["hdi_low"])
        hdi_high = float(summaries[pname]["hdi_high"])
        in_hdi = int(hdi_low <= true_val <= hdi_high)
        per_param[pname] = {
            "true": true_val,
            "posterior_mean": float(summaries[pname]["posterior_mean"]),
            "hdi_low": hdi_low,
            "hdi_high": hdi_high,
            "in_hdi": in_hdi,
        }

    # ------------------------------------------------------------------
    # 5. Convergence diagnostics
    # ------------------------------------------------------------------
    diagnostics = _extract_convergence_diagnostics(mcmc, param_names)
    elapsed_sec = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # 6. Write JSON
    # ------------------------------------------------------------------
    result: dict[str, Any] = {
        "model": model,
        "subject_idx": int(subject_idx),
        "seed": int(seed),
        "sim_seed": int(sim_seed),
        "mcmc_seed": int(mcmc_seed),
        "mcmc_budget": {
            "num_warmup": int(num_warmup),
            "num_samples": int(num_samples),
            "num_chains": int(num_chains),
            "max_tree_depth": int(max_tree_depth),
        },
        "synthetic_task": {
            "n_blocks": int(synth_df["block"].nunique()),
            "n_trials": int(len(synth_df)),
        },
        "params": per_param,
        "diagnostics": diagnostics,
        "elapsed_sec": float(elapsed_sec),
        "hdi_prob": float(HDI_PROB),
    }

    out_path = output_dir / f"{model}_subject_{int(subject_idx):03d}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n  Wrote: {out_path}")
    print(f"  Elapsed: {elapsed_sec:.1f} s")
    print("=" * 80)

    return out_path


# ---------------------------------------------------------------------------
# Aggregate mode
# ---------------------------------------------------------------------------


def _safe_pearson_r(true_vals: np.ndarray, post_means: np.ndarray) -> float:
    """Compute Pearson r with robust handling of zero-variance edge cases.

    Returns NaN (with a warning) if either vector has zero variance —
    ``np.corrcoef`` would emit a runtime warning and return NaN anyway,
    but we want explicit control and log messaging.

    Parameters
    ----------
    true_vals, post_means : np.ndarray
        1-D arrays of the same length.

    Returns
    -------
    float
        Pearson r in ``[-1, 1]``, or NaN if undefined.
    """
    if len(true_vals) != len(post_means):
        raise ValueError(
            f"Length mismatch: true={len(true_vals)}, "
            f"post={len(post_means)}"
        )
    if len(true_vals) < 2:
        warnings.warn("Need >= 2 samples for Pearson r; returning NaN.")
        return float("nan")
    # Use peak-to-peak range instead of np.std to avoid floating-point
    # drift on constant inputs (np.std on 3x the same float can return
    # ~1e-17, not exactly zero — would pass the == 0 check incorrectly).
    if np.ptp(true_vals) == 0 or np.ptp(post_means) == 0:
        warnings.warn(
            "Zero variance in true or posterior-mean vector; "
            "Pearson r undefined — returning NaN."
        )
        return float("nan")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_mat = np.corrcoef(true_vals, post_means)
    return float(r_mat[0, 1])


def aggregate_recovery_results(
    model: str,
    n_subjects: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Aggregate per-subject recovery JSONs into a CSV + summary.md.

    Parameters
    ----------
    model : str
        Model name (key of ``MODEL_REGISTRY``).
    n_subjects : int
        Expected number of per-subject JSONs.  Missing subjects are
        reported in the summary but do not cause failure.
    output_dir : Path
        Directory containing ``{model}_subject_*.json`` and where the
        ``{model}_recovery.csv`` + ``{model}_recovery_summary.md`` are
        written.

    Returns
    -------
    dict[str, Any]
        Aggregation summary including the per-parameter status table
        and an overall ``verdict`` key: ``"PASS"``, ``"FAIL"`` or
        ``"NO_KAPPA"`` (for M1 / M2).
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    # Glob per-subject JSONs
    json_paths = sorted(output_dir.glob(f"{model}_subject_*.json"))
    if not json_paths:
        warnings.warn(
            f"No per-subject JSONs found in {output_dir} for model={model!r}"
        )

    # Load and collate
    per_param_data: dict[str, dict[str, list[float]]] = {}
    subjects_loaded: list[int] = []

    for path in json_paths:
        with path.open(encoding="utf-8") as fh:
            subject_result = json.load(fh)

        subject_idx = int(subject_result.get("subject_idx", -1))
        subjects_loaded.append(subject_idx)
        params = subject_result.get("params", {})

        for pname, pdata in params.items():
            if pname not in per_param_data:
                per_param_data[pname] = {
                    "true": [],
                    "posterior_mean": [],
                    "in_hdi": [],
                }
            per_param_data[pname]["true"].append(float(pdata["true"]))
            per_param_data[pname]["posterior_mean"].append(
                float(pdata["posterior_mean"])
            )
            per_param_data[pname]["in_hdi"].append(int(pdata["in_hdi"]))

    # Compute per-parameter recovery metrics
    rows: list[dict[str, Any]] = []
    n_loaded = len(subjects_loaded)
    any_kappa_failed = False
    any_kappa_param = False

    param_order = list(MODEL_REGISTRY[model]["params"])
    for pname in param_order:
        if pname not in per_param_data:
            continue
        data = per_param_data[pname]
        true_arr = np.asarray(data["true"], dtype=float)
        post_arr = np.asarray(data["posterior_mean"], dtype=float)
        in_hdi_arr = np.asarray(data["in_hdi"], dtype=float)

        pearson_r = _safe_pearson_r(true_arr, post_arr)
        hdi_coverage = float(in_hdi_arr.mean()) if len(in_hdi_arr) > 0 else float("nan")

        is_kappa = pname in KAPPA_FAMILY
        if is_kappa:
            any_kappa_param = True
            passes = (
                not np.isnan(pearson_r)
                and pearson_r >= KAPPA_PEARSON_THRESHOLD
                and hdi_coverage >= KAPPA_HDI_COVERAGE_THRESHOLD
            )
            status = "PASS" if passes else "FAIL"
            criterion = (
                f"r >= {KAPPA_PEARSON_THRESHOLD:.2f} "
                f"AND coverage >= {KAPPA_HDI_COVERAGE_THRESHOLD:.2f}"
            )
            if not passes:
                any_kappa_failed = True
        else:
            status = "descriptive only"
            criterion = "n/a (not kappa-family)"

        rows.append(
            {
                "model": model,
                "parameter": pname,
                "n_subjects": n_loaded,
                "pearson_r": pearson_r,
                "hdi_coverage": hdi_coverage,
                "pass_criterion": criterion,
                "status": status,
            }
        )

    # Overall verdict
    if not any_kappa_param:
        verdict = "NO_KAPPA"
    elif any_kappa_failed:
        verdict = "FAIL"
    else:
        verdict = "PASS"

    # Write CSV
    csv_path = output_dir / f"{model}_recovery.csv"
    result_df = pd.DataFrame(rows)
    result_df.to_csv(csv_path, index=False)
    print(f">> Wrote {csv_path}")

    # Write summary.md
    md_path = output_dir / f"{model}_recovery_summary.md"
    _write_recovery_summary_md(
        md_path=md_path,
        model=model,
        n_loaded=n_loaded,
        n_expected=int(n_subjects),
        subjects_loaded=sorted(subjects_loaded),
        rows=rows,
        verdict=verdict,
    )
    print(f">> Wrote {md_path}")

    return {
        "model": model,
        "n_subjects_loaded": n_loaded,
        "n_subjects_expected": int(n_subjects),
        "subjects_missing": [
            idx for idx in range(1, int(n_subjects) + 1)
            if idx not in subjects_loaded
        ],
        "rows": rows,
        "verdict": verdict,
        "csv_path": str(csv_path),
        "md_path": str(md_path),
    }


def _write_recovery_summary_md(
    md_path: Path,
    model: str,
    n_loaded: int,
    n_expected: int,
    subjects_loaded: list[int],
    rows: list[dict[str, Any]],
    verdict: str,
) -> None:
    """Emit a one-paragraph verdict + per-parameter table summary.

    Parameters
    ----------
    md_path : Path
        Target markdown path.
    model : str
        Model name.
    n_loaded : int
        Number of per-subject JSONs found.
    n_expected : int
        Expected number of per-subject JSONs (from --n-subjects).
    subjects_loaded : list[int]
        Subject indices successfully loaded (sorted).
    rows : list[dict]
        Per-parameter row dicts as written to the CSV.
    verdict : str
        Overall verdict: ``"PASS"``, ``"FAIL"`` or ``"NO_KAPPA"``.
    """
    missing = [
        idx for idx in range(1, int(n_expected) + 1) if idx not in subjects_loaded
    ]

    lines: list[str] = []
    lines.append(f"# {model} Bayesian Parameter Recovery Summary")
    lines.append("")
    lines.append(f"**Verdict:** {verdict}")
    lines.append(f"**Subjects loaded:** {n_loaded} / {n_expected}")
    if missing:
        lines.append(
            f"**Missing subject indices:** "
            f"{', '.join(str(i) for i in missing[:20])}"
            + (f" (+{len(missing) - 20} more)" if len(missing) > 20 else "")
        )
    lines.append("")

    if verdict == "PASS":
        lines.append(
            f"All kappa-family parameters met the recovery criterion "
            f"(Pearson r >= {KAPPA_PEARSON_THRESHOLD:.2f} AND 95% HDI coverage "
            f">= {KAPPA_HDI_COVERAGE_THRESHOLD:.2f}). Non-kappa parameters "
            f"are labelled `descriptive only` and are not subject to the "
            f"recovery gate (per quick-005 MLE recovery findings: r = 0.21 "
            f"to 0.77 under the task structure)."
        )
    elif verdict == "FAIL":
        lines.append(
            f"At least one kappa-family parameter failed the recovery "
            f"criterion (Pearson r >= {KAPPA_PEARSON_THRESHOLD:.2f} AND 95% "
            f"HDI coverage >= {KAPPA_HDI_COVERAGE_THRESHOLD:.2f}). See the "
            f"table below; the master pipeline should NOT proceed to L2 "
            f"inference on failing parameters."
        )
    else:  # NO_KAPPA
        lines.append(
            f"Model `{model}` has no kappa-family parameters; no pass/fail "
            f"gate applies. All parameters are reported as `descriptive "
            f"only` for informational purposes."
        )
    lines.append("")

    lines.append("## Per-parameter recovery")
    lines.append("")
    lines.append(
        "| Parameter | N | Pearson r | 95% HDI coverage | Criterion | Status |"
    )
    lines.append(
        "|-----------|---|-----------|------------------|-----------|--------|"
    )
    for row in rows:
        r = row["pearson_r"]
        cov = row["hdi_coverage"]
        r_str = f"{r:.3f}" if not np.isnan(r) else "nan"
        cov_str = f"{cov:.3f}" if not np.isnan(cov) else "nan"
        lines.append(
            f"| {row['parameter']} | {row['n_subjects']} | "
            f"{r_str} | {cov_str} | {row['pass_criterion']} | "
            f"{row['status']} |"
        )
    lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for both modes.

    Returns
    -------
    argparse.ArgumentParser
        Parser exposing a mutually-exclusive ``--mode`` with
        sub-parameters for ``single-subject`` vs ``aggregate``.
    """
    parser = argparse.ArgumentParser(
        prog="21_run_bayesian_recovery.py",
        description=(
            "Step 21.2 Bayesian parameter recovery runner. "
            "Modes: single-subject (one synthetic fit per SLURM array task), "
            "aggregate (combine per-subject JSONs into recovery CSV)."
        ),
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["single-subject", "aggregate"],
        help="Which mode to run.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(STACKED_MODEL_DISPATCH.keys()),
        help="Choice-only model name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/bayesian/21_recovery/"),
        help="Directory for per-subject JSONs + aggregate CSV/md.",
    )

    # single-subject args
    parser.add_argument(
        "--subject-idx",
        type=int,
        default=None,
        help="1-indexed subject number (SLURM_ARRAY_TASK_ID). "
             "Required for --mode single-subject.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Base RNG seed (single-subject mode).",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"MCMC warmup steps. Default {DEFAULT_WARMUP}.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"MCMC samples per chain. Default {DEFAULT_SAMPLES}.",
    )
    parser.add_argument(
        "--num-chains",
        type=int,
        default=DEFAULT_CHAINS,
        help=f"Number of parallel chains. Default {DEFAULT_CHAINS}.",
    )
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=DEFAULT_MAX_TREE_DEPTH,
        help=f"NUTS max tree depth. Default {DEFAULT_MAX_TREE_DEPTH}.",
    )
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=DEFAULT_N_BLOCKS,
        help=f"Synthetic n_blocks per subject. Default {DEFAULT_N_BLOCKS}.",
    )
    parser.add_argument(
        "--n-trials-per-block",
        type=int,
        default=DEFAULT_TRIALS_PER_BLOCK,
        help=f"Max synthetic trials per block. Default "
             f"{DEFAULT_TRIALS_PER_BLOCK}.",
    )

    # aggregate args
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=50,
        help="Expected number of per-subject JSONs (aggregate mode). "
             "Default 50.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Parameters
    ----------
    argv : list[str] or None
        Raw argv for testing; ``None`` consumes ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code: 0 on success or non-kappa-family model;
        1 if aggregate mode detects any kappa-family parameter failing
        the recovery criterion.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.mode == "single-subject":
        if args.subject_idx is None:
            parser.error("--subject-idx is required for --mode single-subject")
        run_single_subject(
            model=args.model,
            subject_idx=int(args.subject_idx),
            seed=int(args.seed),
            output_dir=args.output_dir,
            num_warmup=int(args.num_warmup),
            num_samples=int(args.num_samples),
            num_chains=int(args.num_chains),
            max_tree_depth=int(args.max_tree_depth),
            n_blocks=int(args.n_blocks),
            n_trials_per_block=int(args.n_trials_per_block),
        )
        return 0

    # aggregate mode
    summary = aggregate_recovery_results(
        model=args.model,
        n_subjects=int(args.n_subjects),
        output_dir=args.output_dir,
    )
    verdict = summary["verdict"]
    print(f"\n>> Aggregate verdict: {verdict}")
    if verdict == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
