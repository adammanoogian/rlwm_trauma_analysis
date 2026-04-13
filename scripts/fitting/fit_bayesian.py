"""
Bayesian Hierarchical Model Fitting using JAX/NumPyro

Fits choice-only hierarchical models (M1–M3, M5, M6a, M6b) using Hamiltonian
Monte Carlo (NUTS) via NumPyro.  All six models use the canonical stacked
hBayesDM non-centered parameterization introduced in Phase 15.

Usage:
------
# Fit Q-learning model (M1)
python scripts/fitting/fit_bayesian.py --model qlearning --data output/task_trials_long.csv

# Fit WM-RL+kappa model (M3)
python scripts/fitting/fit_bayesian.py --model wmrl_m3 --data output/task_trials_long.csv

# Fit dual-perseveration model (M6b) — winning model
python scripts/fitting/fit_bayesian.py --model wmrl_m6b --data output/task_trials_long.csv

# With custom MCMC settings
python scripts/fitting/fit_bayesian.py --model wmrl_m6b --data data.csv \\
    --chains 4 --warmup 1000 --samples 2000

Author: Generated for RLWM trauma analysis project
Updated: 2026-04-13 — refactored for all 6 choice-only models (16-04)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import numpyro
import arviz as az

from scripts.fitting.numpyro_models import (
    qlearning_hierarchical_model,
    wmrl_hierarchical_model,
    wmrl_m3_hierarchical_model,
    qlearning_hierarchical_model_stacked,
    wmrl_hierarchical_model_stacked,
    wmrl_m5_hierarchical_model,
    wmrl_m6a_hierarchical_model,
    wmrl_m6b_hierarchical_model,
    wmrl_m6b_hierarchical_model_subscale,
    prepare_data_for_numpyro,
    prepare_stacked_participant_data,
    run_inference,
    run_inference_with_bump,
    samples_to_arviz,
)
from scripts.fitting.bayesian_diagnostics import (
    compute_pointwise_log_lik,
    build_inference_data_with_loglik,
)
from scripts.fitting.bayesian_summary_writer import write_bayesian_summary

from config import OUTPUT_VERSION_DIR, EXCLUDED_PARTICIPANTS, ALL_MODELS, MODEL_REGISTRY

# ---------------------------------------------------------------------------
# Canonical stacked hierarchical models (hBayesDM non-centered convention)
# All 6 choice-only models dispatch through this table.
# ---------------------------------------------------------------------------
STACKED_MODEL_DISPATCH: dict[str, object] = {
    "qlearning": qlearning_hierarchical_model_stacked,
    "wmrl": wmrl_hierarchical_model_stacked,
    "wmrl_m3": wmrl_m3_hierarchical_model,
    "wmrl_m5": wmrl_m5_hierarchical_model,
    "wmrl_m6a": wmrl_m6a_hierarchical_model,
    "wmrl_m6b": wmrl_m6b_hierarchical_model,
}

# Models that support Level-2 LEC covariate regression on kappa.
# M1 and M2 raise NotImplementedError if covariate_lec is not None.
_L2_LEC_SUPPORTED: frozenset[str] = frozenset(
    {"wmrl_m3", "wmrl_m5", "wmrl_m6a", "wmrl_m6b"}
)


def load_and_prepare_data(
    data_path: Path,
    min_block: int = 3,
) -> pd.DataFrame:
    """Load and prepare human behavioral data.

    Handles both 'response' and 'key_press' column names.
    Excludes participants based on data quality criteria.

    Parameters
    ----------
    data_path : Path
        Path to CSV file with trial data.
    min_block : int
        Minimum block number to include (default: 3, excludes practice).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for fitting.
    """
    print(f"\n>> Loading data from: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load data
    df = pd.read_csv(data_path)

    # Remove rows with NaN participant IDs
    df = df.dropna(subset=["sona_id"]).copy()

    # Exclude participants based on data quality
    initial_n = df["sona_id"].nunique()
    df = df[~df["sona_id"].isin(EXCLUDED_PARTICIPANTS)].copy()
    n_excluded = initial_n - df["sona_id"].nunique()

    print(f"  Loaded {len(df)} trials from {initial_n} participants")
    if n_excluded > 0:
        print(f"  Excluded {n_excluded} participants (insufficient data/duplicates)")
        print(f"  Final sample: {df['sona_id'].nunique()} participants")

    # Filter blocks
    if min_block is not None:
        df = df[df["block"] >= min_block].copy()
        print(f"  Filtered to blocks >= {min_block}: {len(df)} trials remain")

    # Drop any NaN values in critical columns first
    df = df.dropna(subset=["stimulus", "key_press", "correct"]).copy()
    print(f"  Dropped NaN values in critical columns: {len(df)} trials remain")

    # Ensure correct data types
    df["stimulus"] = df["stimulus"].astype(int)
    df["key_press"] = df["key_press"].astype(int)
    df["reward"] = df["correct"].astype(float)

    # Check for set_size column (needed for WM-RL)
    if "set_size" not in df.columns:
        print("  Warning: 'set_size' column not found, using default value of 6")
        df["set_size"] = 6

    # Print summary
    print(f"\n>> Data summary:")
    print(f"  Participants: {df['sona_id'].nunique()}")
    print(f"  Blocks: {df['block'].nunique()}")
    print(f"  Total trials: {len(df)}")
    print(f"  Trials per participant: {len(df) / df['sona_id'].nunique():.0f}")

    # Per-participant summary
    for pid in sorted(df["sona_id"].unique()):
        pdata = df[df["sona_id"] == pid]
        print(
            f"    Participant {int(pid)}: {len(pdata)} trials, "
            f"{pdata['block'].nunique()} blocks"
        )

    return df


def _load_lec_covariate(
    participant_data_stacked: dict,
) -> jnp.ndarray | None:
    """Load and z-score the LEC total-events covariate.

    Reads ``output/summary_participant_metrics.csv``, aligns to the sorted
    participant order from ``participant_data_stacked``, and z-scores the
    ``less_total_events`` column.  Returns ``None`` with a warning if the
    file or column is missing or loading fails.

    Parameters
    ----------
    participant_data_stacked : dict
        Stacked participant data dict from ``prepare_stacked_participant_data``.
        Used only to obtain ``sorted(participant_data_stacked.keys())``.

    Returns
    -------
    jnp.ndarray or None
        Shape ``(n_participants,)`` float32 z-scored LEC covariate, or ``None``
        if unavailable.
    """
    metrics_path = Path("output/summary_participant_metrics.csv")
    if not metrics_path.exists():
        print(f"  WARNING: {metrics_path} not found. Running without L2 regression.")
        return None

    try:
        metrics_df = pd.read_csv(metrics_path)
        lec_col = "less_total_events"
        if lec_col not in metrics_df.columns:
            print(
                f"  WARNING: '{lec_col}' not found in {metrics_path}. "
                "Running without L2 regression."
            )
            return None

        sorted_pids = sorted(participant_data_stacked.keys())
        metrics_df = metrics_df.set_index("sona_id").reindex(sorted_pids)
        lec_raw = metrics_df[lec_col].values.astype(np.float32)
        lec_mean = float(np.nanmean(lec_raw))
        lec_std = float(np.nanstd(lec_raw))
        lec_z = (lec_raw - lec_mean) / (lec_std + 1e-8)
        covariate_lec = jnp.array(lec_z, dtype=jnp.float32)
        print(
            f"  LEC covariate loaded: mean={lec_mean:.2f}, "
            f"std={lec_std:.2f}, n={len(sorted_pids)}"
        )
        return covariate_lec
    except Exception as exc:
        print(
            f"  WARNING: Failed to load LEC covariate ({exc}). "
            "Running without L2 regression."
        )
        return None


def _load_subscale_design_matrix(
    participant_data_stacked: dict,
) -> tuple[jnp.ndarray | None, list[str]]:
    """Load the full Level-2 subscale design matrix for M6b subscale fitting.

    Reads ``output/summary_participant_metrics.csv``, aligns to the sorted
    participant order, and builds the 4-predictor design matrix via
    ``build_level2_design_matrix`` from ``scripts.fitting.level2_design``.

    Parameters
    ----------
    participant_data_stacked : dict
        Stacked participant data dict.  Used to obtain sorted participant IDs.

    Returns
    -------
    tuple[jnp.ndarray or None, list[str]]
        ``(covariate_matrix, covariate_names)`` where ``covariate_matrix``
        has shape ``(n_participants, 4)`` (float32 JAX array), or
        ``(None, [])`` if the metrics file is missing or loading fails.
    """
    from scripts.fitting.level2_design import (
        COVARIATE_NAMES,
        build_level2_design_matrix,
    )

    metrics_path = Path("output/summary_participant_metrics.csv")
    if not metrics_path.exists():
        print(
            f"  WARNING: {metrics_path} not found. "
            "Running subscale model without L2 regression."
        )
        return None, []

    try:
        metrics_df = pd.read_csv(metrics_path)
        sorted_pids = sorted(participant_data_stacked.keys())
        X, cov_names = build_level2_design_matrix(metrics_df, sorted_pids)
        covariate_matrix = jnp.array(X, dtype=jnp.float32)
        print(
            f"  Subscale design matrix loaded: shape={X.shape}, "
            f"covariates={cov_names}, "
            f"condition_number={float(np.linalg.cond(X)):.2f}"
        )
        return covariate_matrix, cov_names
    except Exception as exc:
        print(
            f"  WARNING: Failed to load subscale design matrix ({exc}). "
            "Running subscale model without L2 regression."
        )
        return None, []


def _fit_stacked_model(
    data: pd.DataFrame,
    model: str,
    model_fn: object,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    seed: int,
    subscale: bool = False,
    max_tree_depth: int = 10,
) -> tuple[object, dict]:
    """Fit a canonical stacked hierarchical model.

    Shared logic for all hBayesDM non-centered models:

    1. ``prepare_stacked_participant_data`` — builds padded stacked arrays.
    2. Load Level-2 covariate(s) from ``output/summary_participant_metrics.csv``.
    3. ``run_inference_with_bump`` — NUTS with automatic divergence bump.
    4. Print group-level ``*_mu_pr`` and ``beta_*`` posterior summaries.

    Parameters
    ----------
    data : pd.DataFrame
        Prepared trial-level DataFrame (output of ``load_and_prepare_data``).
    model : str
        Model name key (e.g. ``'wmrl_m6b'``).  Used for display only.
    model_fn : callable
        NumPyro model function from ``STACKED_MODEL_DISPATCH`` (or
        ``wmrl_m6b_hierarchical_model_subscale`` when ``subscale=True``).
    num_warmup : int
        Number of NUTS warmup steps.
    num_samples : int
        Number of posterior samples per chain.
    num_chains : int
        Number of parallel chains.
    seed : int
        JAX random seed.
    subscale : bool
        If ``True`` and ``model == 'wmrl_m6b'``, use the full subscale
        Level-2 design matrix (4 covariates, 32 beta sites).  The
        ``model_fn`` must already be set to
        ``wmrl_m6b_hierarchical_model_subscale`` by the caller.

    Returns
    -------
    tuple[MCMC, dict]
        ``(mcmc, participant_data_stacked)`` — the fitted MCMC object and the
        stacked participant data dict passed to the model.
    """
    print("\n>> Preparing stacked participant data...")
    participant_data_stacked = prepare_stacked_participant_data(
        data,
        participant_col="sona_id",
        block_col="block",
        stimulus_col="stimulus",
        action_col="key_press",
        reward_col="reward",
        set_size_col="set_size",
    )
    n_ppts = len(participant_data_stacked)
    print(f"  Prepared {n_ppts} participants (sorted by ID)")

    # ------------------------------------------------------------------
    # Build model_args based on subscale flag or standard L2 covariate
    # ------------------------------------------------------------------
    if subscale and model == "wmrl_m6b":
        # Full subscale design matrix path (L2-05)
        print(
            "\n>> Loading subscale Level-2 design matrix "
            "(4 covariates, 32 beta sites)..."
        )
        covariate_matrix, cov_names = _load_subscale_design_matrix(
            participant_data_stacked
        )
        model_args = {
            "participant_data_stacked": participant_data_stacked,
            "covariate_matrix": covariate_matrix,
            "covariate_names": cov_names,
        }
    else:
        # Standard path: single LEC covariate (or None for M1/M2)
        if model in _L2_LEC_SUPPORTED:
            covariate_lec = _load_lec_covariate(participant_data_stacked)
        else:
            covariate_lec = None
            print(
                f"  Skipping LEC covariate: model '{model}' does not support "
                "Level-2 regression (no natural L2 target parameter)."
            )

        model_args = {
            "participant_data_stacked": participant_data_stacked,
            "covariate_lec": covariate_lec,
        }

    print("\n>> Running MCMC inference (NUTS + convergence auto-bump)...")
    mcmc = run_inference_with_bump(
        model_fn,
        model_args,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed,
        max_tree_depth=max_tree_depth,
    )

    # Print group-level posterior summaries (all *_mu_pr sites)
    samples = mcmc.get_samples()
    display_name = model.upper().replace("_", "-")
    subscale_tag = " [SUBSCALE]" if subscale else ""
    print("\n" + "=" * 80)
    print(f"{display_name} POSTERIOR ESTIMATES — GROUP LEVEL{subscale_tag}")
    print("=" * 80)
    mu_pr_sites = sorted(k for k in samples if k.endswith("_mu_pr"))
    for pname in mu_pr_sites:
        arr = samples[pname]
        print(f"  {pname}: {float(arr.mean()):.3f} +/- {float(arr.std()):.3f}")

    # Print all beta_* sites with 95% HDI (covers both single-LEC and subscale)
    beta_sites = sorted(k for k in samples if k.startswith("beta_"))
    for bname in beta_sites:
        arr = samples[bname]
        beta_mean = float(arr.mean())
        beta_std = float(arr.std())
        hdi_result = az.hdi(np.array(arr), hdi_prob=0.95)
        hdi_low = float(hdi_result[0])
        hdi_high = float(hdi_result[1])
        print(
            f"  {bname}: mean={beta_mean:.4f} +/- {beta_std:.4f}, "
            f"95% HDI=[{hdi_low:.4f}, {hdi_high:.4f}]"
        )
        if hdi_low > 0:
            print(f"    HDI excludes zero (positive direction)")
        elif hdi_high < 0:
            print(f"    HDI excludes zero (negative direction)")
        else:
            print(f"    HDI includes zero")

    return mcmc, participant_data_stacked


def fit_model(
    data: pd.DataFrame,
    model: str,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    seed: int,
    subscale: bool = False,
    max_tree_depth: int = 10,
) -> tuple[object, dict]:
    """Fit a hierarchical Bayesian model.

    All six choice-only models (qlearning, wmrl, wmrl_m3, wmrl_m5, wmrl_m6a,
    wmrl_m6b) are dispatched through ``STACKED_MODEL_DISPATCH`` to the shared
    ``_fit_stacked_model`` helper.

    When ``subscale=True`` and ``model='wmrl_m6b'``, the subscale variant
    ``wmrl_m6b_hierarchical_model_subscale`` is used instead, accepting a full
    4-predictor design matrix (L2-05: 32 beta coefficient sites).

    Parameters
    ----------
    data : pd.DataFrame
        Prepared trial data (from ``load_and_prepare_data``).
    model : str
        Model name.  Must be a key of ``STACKED_MODEL_DISPATCH``.
    num_warmup : int
        Number of NUTS warmup steps.
    num_samples : int
        Number of posterior samples per chain.
    num_chains : int
        Number of parallel chains.
    seed : int
        JAX random seed.
    subscale : bool
        If ``True`` and ``model == 'wmrl_m6b'``, use
        ``wmrl_m6b_hierarchical_model_subscale`` with the full 4-predictor
        design matrix.  Raises ``ValueError`` if used with any other model
        (subscale L2 is only defined for M6b in this phase).

    Returns
    -------
    tuple[MCMC, dict]
        ``(mcmc, participant_data_stacked)`` for all stacked models.

    Raises
    ------
    NotImplementedError
        If ``model`` is not in ``STACKED_MODEL_DISPATCH``.
    ValueError
        If ``subscale=True`` is used with a model other than ``'wmrl_m6b'``.
    """
    if model not in STACKED_MODEL_DISPATCH:
        raise NotImplementedError(
            f"Bayesian fitting for '{model}' is not yet implemented. "
            f"Implemented models: {sorted(STACKED_MODEL_DISPATCH.keys())}. "
            f"Use scripts/12_fit_mle.py for MLE fitting of this model."
        )

    if subscale and model != "wmrl_m6b":
        raise ValueError(
            f"--subscale is only supported for model='wmrl_m6b'. "
            f"Got model='{model}'. The full subscale Level-2 design (L2-05) "
            "targets M6b parameters (kappa_total, kappa_share, and all 6 base "
            "params). Run without --subscale for other models."
        )

    # Select model function: subscale variant for M6b with --subscale flag
    if subscale and model == "wmrl_m6b":
        model_fn = wmrl_m6b_hierarchical_model_subscale
        variant_tag = " [SUBSCALE — 32 beta sites, 4 covariates]"
    else:
        model_fn = STACKED_MODEL_DISPATCH[model]
        variant_tag = ""

    display_name = model.upper().replace("_", "-")
    print(f"\n{'=' * 80}")
    print(f"FITTING {display_name} HIERARCHICAL MODEL{variant_tag}")
    print(f"{'=' * 80}")

    return _fit_stacked_model(
        data, model, model_fn, num_warmup, num_samples, num_chains, seed,
        subscale=subscale,
        max_tree_depth=max_tree_depth,
    )


def save_results(
    mcmc: object,
    data: pd.DataFrame,
    model: str,
    output_dir: Path,
    save_plots: bool = True,
    participant_data_stacked: dict | None = None,
) -> object:
    """Save fitting results to disk.

    For all stacked models (``model in STACKED_MODEL_DISPATCH``), applies the
    convergence gate (HIER-07) and writes:

    - Schema-parity CSV via ``write_bayesian_summary``
    - NetCDF posterior (``{model}_posterior.nc``)
    - WAIC and LOO information criteria
    - Shrinkage report (``{model}_shrinkage_report.md``)
    - Posterior predictive check results

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        Fitted MCMC object.
    data : pd.DataFrame
        Original trial-level data (used for n_trials computation).
    model : str
        Model identifier (e.g. ``'wmrl_m6b'``).
    output_dir : Path
        Root output directory for results.
    save_plots : bool
        Unused for stacked models; retained for API compatibility.
    participant_data_stacked : dict or None
        Stacked participant data dict from ``prepare_stacked_participant_data``.
        Required for all stacked models.

    Returns
    -------
    az.InferenceData or None
        ArviZ InferenceData if convergence gate passes, else ``None``.

    Raises
    ------
    ValueError
        If ``participant_data_stacked`` is ``None`` for a stacked model.
    """
    from scripts.fitting.bayesian_diagnostics import (
        filter_padding_from_loglik,
        compute_shrinkage_report,
        write_shrinkage_report,
        run_posterior_predictive_check,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Stacked-model path: convergence gate + schema-parity + shrinkage
    # Applies to all 6 choice-only models.
    # ------------------------------------------------------------------
    if model in STACKED_MODEL_DISPATCH:
        if participant_data_stacked is None:
            raise ValueError(
                f"participant_data_stacked is required for model='{model}'. "
                "Pass the dict returned by prepare_stacked_participant_data()."
            )

        pdata_stacked = participant_data_stacked
        participant_ids = sorted(pdata_stacked.keys())

        # Compute pointwise log-lik and filter padding before WAIC/LOO
        print("\n>> Computing pointwise log-likelihoods...")
        pointwise_loglik = compute_pointwise_log_lik(mcmc, pdata_stacked, model)
        print(f"  Pointwise log-lik shape: {pointwise_loglik.shape}")

        filtered_loglik = filter_padding_from_loglik(pointwise_loglik, pdata_stacked)
        print("  Padding positions set to NaN for WAIC/LOO.")

        # Build InferenceData with log_likelihood group
        print("\n>> Building ArviZ InferenceData...")
        idata = build_inference_data_with_loglik(
            mcmc, filtered_loglik, participant_ids=participant_ids
        )

        # ------------------------------------------------------------------
        # CONVERGENCE GATE (HIER-07) — refuse to write outputs if gate fails
        # ------------------------------------------------------------------
        model_params = MODEL_REGISTRY[model]["params"]
        print("\n>> Checking convergence gate...")
        convergence_summary = az.summary(idata, var_names=model_params)
        max_rhat = float(convergence_summary["r_hat"].max())
        min_ess = float(convergence_summary["ess_bulk"].min())
        extra = mcmc.get_extra_fields()
        n_div = int(extra["diverging"].sum()) if "diverging" in extra else 0
        converged = max_rhat < 1.01 and min_ess > 400 and n_div == 0

        if not converged:
            print(
                f"\n[CONVERGENCE GATE FAILED] max_rhat={max_rhat:.4f}, "
                f"min_ess_bulk={min_ess:.0f}, divergences={n_div}"
            )
            print("Refusing to write output files. Fix convergence issues and re-run.")
            return None  # Early return — no files written

        print(
            f"\n[CONVERGENCE GATE PASSED] max_rhat={max_rhat:.4f}, "
            f"min_ess_bulk={min_ess:.0f}, divergences={n_div}"
        )

        # ------------------------------------------------------------------
        # Write outputs (only reached if gate passes)
        # ------------------------------------------------------------------
        bayesian_dir = output_dir / "bayesian"
        bayesian_dir.mkdir(parents=True, exist_ok=True)

        # Schema-parity CSV
        print("\n>> Writing schema-parity CSV...")
        n_trials_per_ppt = [
            int(len(data[data["sona_id"] == pid])) for pid in participant_ids
        ]

        from config import EXPECTED_PARAMETERIZATION

        csv_path = write_bayesian_summary(
            idata,
            model,
            output_dir,
            participant_ids=participant_ids,
            parameterization_version=EXPECTED_PARAMETERIZATION.get(
                model, "v4.0-K[2,6]-phiapprox"
            ),
            n_trials_per_participant=n_trials_per_ppt,
        )
        print(f"  Saved: {csv_path}")

        # NetCDF posterior
        netcdf_path = bayesian_dir / f"{model}_posterior.nc"
        idata.to_netcdf(str(netcdf_path))
        print(f"  Saved posterior NetCDF: {netcdf_path}")

        # WAIC and LOO
        print("\n>> Computing WAIC and LOO...")
        try:
            waic_result = az.waic(idata)
            print(f"  WAIC: {waic_result.waic:.2f} (p_waic={waic_result.p_waic:.2f})")
        except Exception as exc:
            print(f"  WARNING: WAIC failed: {exc}")
        try:
            loo_result = az.loo(idata, pointwise=True)
            print(f"  LOO: {loo_result.loo:.2f} (p_loo={loo_result.p_loo:.2f})")
        except Exception as exc:
            print(f"  WARNING: LOO failed: {exc}")

        # Shrinkage report
        print("\n>> Computing shrinkage diagnostic...")
        shrinkage = compute_shrinkage_report(idata, model_params)
        print("  Shrinkage values:")
        for param, val in shrinkage.items():
            status = "identified" if val >= 0.3 else "WARNING: poorly identified"
            print(f"    {param}: {val:.3f} ({status})")

        shrinkage_path = bayesian_dir / f"{model}_shrinkage_report.md"
        write_shrinkage_report(shrinkage, shrinkage_path)
        print(f"  Saved: {shrinkage_path}")

        # Posterior predictive check (HIER-09)
        print("\n>> Running posterior predictive check...")
        ppc_result = run_posterior_predictive_check(
            mcmc,
            pdata_stacked,
            model,
            data,
            output_dir=output_dir,
        )
        covered = ppc_result["covered_count"]
        total_b = ppc_result["total_blocks"]
        print(f"  PPC: {covered}/{total_b} blocks covered by 95% envelope")

        print("\n>> All results saved successfully!")
        return idata

    # ------------------------------------------------------------------
    # Fallback: model not in STACKED_MODEL_DISPATCH (should not reach here
    # after fit_model() guards, but kept for safety)
    # ------------------------------------------------------------------
    raise NotImplementedError(
        f"save_results: model='{model}' not in STACKED_MODEL_DISPATCH. "
        f"Implemented: {sorted(STACKED_MODEL_DISPATCH.keys())}"
    )


def _run_permutation_shuffle(
    data: pd.DataFrame,
    shuffle_idx: int,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    seed: int,
    output_dir: Path,
) -> None:
    """Run a single permutation-null shuffle of the LEC covariate for M3.

    Permutes the participant-level LEC covariate with a deterministic RNG
    seeded by ``shuffle_idx``, runs M3 hierarchical NUTS with the shuffled
    covariate, extracts ``beta_lec_kappa`` posterior statistics, and saves a
    compact JSON summary.  Full ``save_results()`` pipeline is skipped.

    Parameters
    ----------
    data : pd.DataFrame
        Prepared trial-level DataFrame (from ``load_and_prepare_data``).
    shuffle_idx : int
        Permutation shuffle index (0-49).  Used as both the permutation RNG
        seed and as the output filename key.
    num_warmup : int
        Number of NUTS warmup steps (reduced budget recommended: 500).
    num_samples : int
        Number of posterior samples per chain (reduced budget: 1000).
    num_chains : int
        Number of parallel MCMC chains.
    seed : int
        JAX RNG seed for NUTS (independent of permutation RNG).
    output_dir : Path
        Root output directory; JSON files saved to
        ``output_dir/bayesian/permutation/``.
    """
    print(f"\n>> PERMUTATION TEST: shuffle index {shuffle_idx}")
    print(f"   num_warmup={num_warmup}, num_samples={num_samples}, seed={seed}")

    model = "wmrl_m3"
    model_fn = STACKED_MODEL_DISPATCH[model]

    print("\n>> Preparing stacked participant data...")
    participant_data_stacked = prepare_stacked_participant_data(
        data,
        participant_col="sona_id",
        block_col="block",
        stimulus_col="stimulus",
        action_col="key_press",
        reward_col="reward",
        set_size_col="set_size",
    )
    n_ppts = len(participant_data_stacked)
    print(f"  Prepared {n_ppts} participants (sorted by ID)")

    # Load real LEC covariate, then permute participant labels
    covariate_lec = _load_lec_covariate(participant_data_stacked)
    if covariate_lec is None:
        raise RuntimeError(
            "Permutation test requires LEC covariate. "
            "output/summary_participant_metrics.csv not found or missing column."
        )

    rng = np.random.default_rng(shuffle_idx)
    covariate_lec_shuffled = jnp.array(
        rng.permutation(np.array(covariate_lec)), dtype=jnp.float32
    )
    print(
        f"  LEC covariate shuffled with seed={shuffle_idx}: "
        f"first 5 values: {np.array(covariate_lec_shuffled)[:5]}"
    )

    model_args = {
        "participant_data_stacked": participant_data_stacked,
        "covariate_lec": covariate_lec_shuffled,
    }

    print("\n>> Running MCMC inference with shuffled covariate...")
    t0 = time.time()
    mcmc = run_inference_with_bump(
        model_fn,
        model_args,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed,
    )
    wall_clock_seconds = float(time.time() - t0)

    # Extract beta_lec_kappa posterior statistics
    samples = mcmc.get_samples()
    if "beta_lec_kappa" not in samples:
        raise KeyError(
            "beta_lec_kappa site not found in MCMC samples. "
            f"Available sites: {sorted(samples.keys())}"
        )

    beta_arr = np.array(samples["beta_lec_kappa"])
    beta_mean = float(beta_arr.mean())
    beta_std = float(beta_arr.std())
    hdi_result = az.hdi(beta_arr, hdi_prob=0.95)
    hdi_low = float(hdi_result[0])
    hdi_high = float(hdi_result[1])
    excludes_zero = bool(hdi_low > 0 or hdi_high < 0)

    # Count divergences
    extra = mcmc.get_extra_fields()
    n_div = int(extra["diverging"].sum()) if "diverging" in extra else 0

    result = {
        "shuffle_idx": shuffle_idx,
        "beta_lec_kappa_mean": beta_mean,
        "beta_lec_kappa_std": beta_std,
        "hdi_low": hdi_low,
        "hdi_high": hdi_high,
        "excludes_zero": excludes_zero,
        "n_divergences": n_div,
        "wall_clock_seconds": wall_clock_seconds,
    }

    # Save JSON
    perm_dir = output_dir / "bayesian" / "permutation"
    perm_dir.mkdir(parents=True, exist_ok=True)
    out_path = perm_dir / f"shuffle_{shuffle_idx}_results.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"\n>> Permutation shuffle {shuffle_idx} complete:")
    print(f"   beta_lec_kappa: mean={beta_mean:.4f}, std={beta_std:.4f}")
    print(f"   95% HDI: [{hdi_low:.4f}, {hdi_high:.4f}]")
    print(f"   HDI excludes zero: {excludes_zero}")
    print(f"   Divergences: {n_div}")
    print(f"   Wall clock: {wall_clock_seconds:.1f}s")
    print(f"   Saved to: {out_path}")


def main() -> None:
    """Main fitting workflow."""
    parser = argparse.ArgumentParser(
        description="Fit Bayesian hierarchical models using JAX/NumPyro"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=ALL_MODELS,
        help="Model to fit",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to trial data CSV",
    )
    parser.add_argument(
        "--min-block",
        type=int,
        default=None,
        help="Minimum block to include (default: 3, use 1 for all blocks)",
    )
    parser.add_argument(
        "--include-practice",
        action="store_true",
        help="Include practice blocks (1-2) in fitting. Equivalent to --min-block 1",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of MCMC chains (default: 4)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1000,
        help="Number of warmup samples (default: 1000)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of posterior samples per chain (default: 2000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_VERSION_DIR),
        help="Output directory for results",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save diagnostic plots (unused for stacked models; retained for compatibility)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=10,
        help=(
            "Maximum NUTS tree depth (default: 10 → up to 1024 leapfrog steps). "
            "Reducing to 8 (256 steps) cuts per-step time ~4x with minor "
            "sampling efficiency loss. Recommended for initial runs."
        ),
    )
    parser.add_argument(
        "--permutation-shuffle",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Permutation test: shuffle index (0-49). "
            "Permutes covariate_lec with RNG seed=shuffle_idx. "
            "Only valid with --model wmrl_m3."
        ),
    )
    parser.add_argument(
        "--subscale",
        action="store_true",
        help=(
            "Use full subscale Level-2 design matrix (only for --model wmrl_m6b). "
            "Loads 4-predictor design matrix from level2_design.py and fits "
            "wmrl_m6b_hierarchical_model_subscale with 32 beta coefficient sites "
            "(8 params x 4 covariates: lec_total, iesr_total, iesr_intr_resid, "
            "iesr_avd_resid). Requires output/summary_participant_metrics.csv. "
            "Extended wall-clock time recommended (12h on cluster)."
        ),
    )

    args = parser.parse_args()

    # Handle --include-practice flag: sets min_block to 1 (include all blocks)
    if args.include_practice:
        min_block = 1
    elif args.min_block is not None:
        min_block = args.min_block
    else:
        min_block = 3  # Default: exclude practice blocks

    model_display = args.model.upper()
    print("=" * 80)
    print(f"BAYESIAN {model_display} FIT WITH JAX/NUMPYRO")
    print("=" * 80)
    print(f"\nSettings:")
    print(f"  Model: {args.model}")
    print(f"  Data: {args.data}")
    print(
        f"  Min block: {min_block}"
        + (" (includes practice)" if min_block < 3 else " (excludes practice)")
    )
    print(f"  Chains: {args.chains}")
    print(f"  Warmup: {args.warmup}")
    print(f"  Samples: {args.samples}")
    print(f"  Max tree depth: {args.max_tree_depth}")
    print(f"  Output: {args.output}")
    print(f"  Seed: {args.seed}")
    if args.subscale:
        print(f"  Subscale: True (wmrl_m6b_hierarchical_model_subscale, 32 beta sites)")
    if args.permutation_shuffle is not None:
        print(f"  Permutation shuffle: {args.permutation_shuffle}")

    # Validate --subscale
    if args.subscale and args.model != "wmrl_m6b":
        parser.error(
            "--subscale is only supported with --model wmrl_m6b. "
            f"Got --model {args.model}."
        )

    # Validate --permutation-shuffle
    if args.permutation_shuffle is not None and args.model != "wmrl_m3":
        parser.error(
            "--permutation-shuffle is only supported with --model wmrl_m3. "
            f"Got --model {args.model}."
        )

    # Set NumPyro devices
    num_devices = min(args.chains, jax.local_device_count())
    numpyro.set_host_device_count(num_devices)
    print(f"  JAX devices: {num_devices}")

    # Load data
    data = load_and_prepare_data(Path(args.data), min_block=min_block)

    # ------------------------------------------------------------------
    # PERMUTATION NULL TEST PATH
    # Runs a reduced MCMC with shuffled LEC covariate.  Saves a compact
    # JSON summary per shuffle; skips full save_results() pipeline.
    # ------------------------------------------------------------------
    if args.permutation_shuffle is not None:
        _run_permutation_shuffle(
            data=data,
            shuffle_idx=args.permutation_shuffle,
            num_warmup=args.warmup,
            num_samples=args.samples,
            num_chains=args.chains,
            seed=args.seed,
            output_dir=Path(args.output),
        )
        return

    # Fit model
    mcmc, extra = fit_model(
        data,
        model=args.model,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        seed=args.seed,
        subscale=args.subscale,
        max_tree_depth=args.max_tree_depth,
    )

    # extra is participant_data_stacked for all stacked models
    save_results(
        mcmc,
        data,
        args.model,
        Path(args.output),
        args.save_plots,
        participant_data_stacked=extra if args.model in STACKED_MODEL_DISPATCH else None,
    )

    print("\n" + "=" * 80)
    print("FITTING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
