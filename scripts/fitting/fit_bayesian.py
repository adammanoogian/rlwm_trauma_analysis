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
import sys
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


def _fit_stacked_model(
    data: pd.DataFrame,
    model: str,
    model_fn: object,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    seed: int,
) -> tuple[object, dict]:
    """Fit a canonical stacked hierarchical model.

    Shared logic for all hBayesDM non-centered models:

    1. ``prepare_stacked_participant_data`` — builds padded stacked arrays.
    2. Load and z-score LEC covariate from ``output/summary_participant_metrics.csv``.
    3. ``run_inference_with_bump`` — NUTS with automatic divergence bump.
    4. Print group-level ``*_mu_pr`` posterior summaries.

    Parameters
    ----------
    data : pd.DataFrame
        Prepared trial-level DataFrame (output of ``load_and_prepare_data``).
    model : str
        Model name key (e.g. ``'wmrl_m6b'``).  Used for display only.
    model_fn : callable
        NumPyro model function from ``STACKED_MODEL_DISPATCH``.
    num_warmup : int
        Number of NUTS warmup steps.
    num_samples : int
        Number of posterior samples per chain.
    num_chains : int
        Number of parallel chains.
    seed : int
        JAX random seed.

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

    # Load and z-score LEC covariate
    covariate_lec = _load_lec_covariate(participant_data_stacked)

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
    )

    # Print group-level posterior summaries (all *_mu_pr sites)
    samples = mcmc.get_samples()
    display_name = model.upper().replace("_", "-")
    print("\n" + "=" * 80)
    print(f"{display_name} POSTERIOR ESTIMATES — GROUP LEVEL")
    print("=" * 80)
    mu_pr_sites = sorted(k for k in samples if k.endswith("_mu_pr"))
    for pname in mu_pr_sites:
        arr = samples[pname]
        print(f"  {pname}: {float(arr.mean()):.3f} +/- {float(arr.std()):.3f}")

    # Print beta_lec_* sites with 95% HDI
    beta_lec_sites = sorted(k for k in samples if k.startswith("beta_lec_"))
    for bname in beta_lec_sites:
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
) -> tuple[object, dict]:
    """Fit a hierarchical Bayesian model.

    All six choice-only models (qlearning, wmrl, wmrl_m3, wmrl_m5, wmrl_m6a,
    wmrl_m6b) are dispatched through ``STACKED_MODEL_DISPATCH`` to the shared
    ``_fit_stacked_model`` helper.

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

    Returns
    -------
    tuple[MCMC, dict]
        ``(mcmc, participant_data_stacked)`` for all stacked models.

    Raises
    ------
    NotImplementedError
        If ``model`` is not in ``STACKED_MODEL_DISPATCH``.
    """
    if model not in STACKED_MODEL_DISPATCH:
        raise NotImplementedError(
            f"Bayesian fitting for '{model}' is not yet implemented. "
            f"Implemented models: {sorted(STACKED_MODEL_DISPATCH.keys())}. "
            f"Use scripts/12_fit_mle.py for MLE fitting of this model."
        )

    model_fn = STACKED_MODEL_DISPATCH[model]
    display_name = model.upper().replace("_", "-")
    print(f"\n{'=' * 80}")
    print(f"FITTING {display_name} HIERARCHICAL MODEL")
    print(f"{'=' * 80}")

    return _fit_stacked_model(
        data, model, model_fn, num_warmup, num_samples, num_chains, seed
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
    print(f"  Output: {args.output}")
    print(f"  Seed: {args.seed}")

    # Set NumPyro devices
    num_devices = min(args.chains, jax.local_device_count())
    numpyro.set_host_device_count(num_devices)
    print(f"  JAX devices: {num_devices}")

    # Load data
    data = load_and_prepare_data(Path(args.data), min_block=min_block)

    # Fit model
    mcmc, extra = fit_model(
        data,
        model=args.model,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        seed=args.seed,
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
