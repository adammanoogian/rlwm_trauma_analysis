"""
Bayesian Hierarchical Model Fitting using JAX/NumPyro

Fits either Q-Learning or WM-RL models using Hamiltonian Monte Carlo (NUTS).
For model comparison (fitting both models), run this script twice with different
--model arguments and compare using ArviZ.

Usage:
------
# Fit Q-learning model
python scripts/fitting/fit_bayesian.py --model qlearning --data output/task_trials_long.csv

# Fit WM-RL model
python scripts/fitting/fit_bayesian.py --model wmrl --data output/task_trials_long.csv

# With custom MCMC settings
python scripts/fitting/fit_bayesian.py --model wmrl --data data.csv --chains 4 --warmup 1000 --samples 2000

# Save diagnostic plots
python scripts/fitting/fit_bayesian.py --model qlearning --data data.csv --save-plots

Author: Generated for RLWM trauma analysis project
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

from config import OUTPUT_VERSION_DIR, EXCLUDED_PARTICIPANTS, ALL_MODELS


def load_and_prepare_data(
    data_path: Path,
    min_block: int = 3
) -> pd.DataFrame:
    """
    Load and prepare human behavioral data.

    Handles both 'response' and 'key_press' column names.
    Excludes participants based on data quality criteria.

    Args:
        data_path: Path to CSV file with trial data
        min_block: Minimum block number to include (default: 3, excludes practice)

    Returns:
        Cleaned DataFrame ready for fitting
    """
    print(f"\n>> Loading data from: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load data
    df = pd.read_csv(data_path)

    # Remove rows with NaN participant IDs
    df = df.dropna(subset=['sona_id']).copy()

    # Exclude participants based on data quality
    initial_n = df['sona_id'].nunique()
    df = df[~df['sona_id'].isin(EXCLUDED_PARTICIPANTS)].copy()
    n_excluded = initial_n - df['sona_id'].nunique()

    print(f"  ✓ Loaded {len(df)} trials from {initial_n} participants")
    if n_excluded > 0:
        print(f"  ✓ Excluded {n_excluded} participants (insufficient data/duplicates)")
        print(f"  ✓ Final sample: {df['sona_id'].nunique()} participants")

    # Filter blocks
    if min_block is not None:
        df = df[df['block'] >= min_block].copy()
        print(f"  ✓ Filtered to blocks >= {min_block}: {len(df)} trials remain")

    # Drop any NaN values in critical columns first
    df = df.dropna(subset=['stimulus', 'key_press', 'correct']).copy()
    print(f"  ✓ Dropped NaN values in critical columns: {len(df)} trials remain")

    # Ensure correct data types
    df['stimulus'] = df['stimulus'].astype(int)
    df['key_press'] = df['key_press'].astype(int)
    df['reward'] = df['correct'].astype(float)  # Reward = 1 if correct, 0 otherwise

    # Check for set_size column (needed for WM-RL)
    if 'set_size' not in df.columns:
        print(f"  ⚠ Warning: 'set_size' column not found, using default value of 6")
        df['set_size'] = 6

    # Print summary
    print(f"\n>> Data summary:")
    print(f"  Participants: {df['sona_id'].nunique()}")
    print(f"  Blocks: {df['block'].nunique()}")
    print(f"  Total trials: {len(df)}")
    print(f"  Trials per participant: {len(df) / df['sona_id'].nunique():.0f}")

    # Per-participant summary
    for pid in sorted(df['sona_id'].unique()):
        pdata = df[df['sona_id'] == pid]
        print(f"    Participant {int(pid)}: {len(pdata)} trials, {pdata['block'].nunique()} blocks")

    return df


def fit_model(
    data: pd.DataFrame,
    model: str,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    seed: int
):
    """
    Fit a hierarchical Bayesian model.

    Args:
        data: Prepared trial data
        model: 'qlearning' or 'wmrl'
        num_warmup: Number of warmup samples
        num_samples: Number of posterior samples per chain
        num_chains: Number of MCMC chains
        seed: Random seed

    Returns:
        Tuple of (mcmc, participant_data)
    """
    BAYESIAN_IMPLEMENTED = {'qlearning', 'wmrl', 'wmrl_m3'}
    if model not in BAYESIAN_IMPLEMENTED:
        raise NotImplementedError(
            f"Bayesian fitting for '{model}' is not yet implemented. "
            f"Implemented models: {sorted(BAYESIAN_IMPLEMENTED)}. "
            f"Use scripts/12_fit_mle.py for MLE fitting of this model."
        )

    # ------------------------------------------------------------------
    # wmrl_m3 code path (hBayesDM non-centered, stacked data, L2 LEC)
    # ------------------------------------------------------------------
    if model == 'wmrl_m3':
        print("\n" + "=" * 80)
        print("FITTING WM-RL+KAPPA (M3) HIERARCHICAL MODEL")
        print("=" * 80)

        print("\n>> Preparing stacked participant data...")
        participant_data_stacked = prepare_stacked_participant_data(
            data,
            participant_col='sona_id',
            block_col='block',
            stimulus_col='stimulus',
            action_col='key_press',
            reward_col='reward',
            set_size_col='set_size',
        )
        n_ppts = len(participant_data_stacked)
        print(f"  Prepared {n_ppts} participants (sorted by ID)")

        # Load and z-score LEC covariate
        covariate_lec = None
        metrics_path = Path("output/summary_participant_metrics.csv")
        if metrics_path.exists():
            try:
                metrics_df = pd.read_csv(metrics_path)
                lec_col = "less_total_events"
                if lec_col not in metrics_df.columns:
                    print(
                        f"  WARNING: '{lec_col}' not found in {metrics_path}. "
                        "Running without L2 regression."
                    )
                else:
                    sorted_pids = sorted(participant_data_stacked.keys())
                    # Align metrics to sorted participant order
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
            except Exception as exc:
                print(f"  WARNING: Failed to load LEC covariate ({exc}). "
                      "Running without L2 regression.")
        else:
            print(
                f"  WARNING: {metrics_path} not found. Running without L2 regression."
            )

        model_args = {
            "participant_data_stacked": participant_data_stacked,
            "covariate_lec": covariate_lec,
        }

        print(f"\n>> Running MCMC inference (NUTS + convergence auto-bump)...")
        mcmc = run_inference_with_bump(
            wmrl_m3_hierarchical_model,
            model_args,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            seed=seed,
        )

        # Print M3 group-level posterior summaries
        samples = mcmc.get_samples()
        print("\n" + "=" * 80)
        print("WM-RL+KAPPA (M3) POSTERIOR ESTIMATES — GROUP LEVEL")
        print("=" * 80)
        m3_group_params = [
            "alpha_pos_mu_pr", "alpha_neg_mu_pr", "phi_mu_pr",
            "rho_mu_pr", "capacity_mu_pr", "epsilon_mu_pr",
            "kappa_mu_pr", "kappa_sigma_pr",
        ]
        for pname in m3_group_params:
            if pname in samples:
                arr = samples[pname]
                print(f"  {pname}: {float(arr.mean()):.3f} ± {float(arr.std()):.3f}")
        if covariate_lec is not None and "beta_lec_kappa" in samples:
            arr = samples["beta_lec_kappa"]
            print(f"  beta_lec_kappa: {float(arr.mean()):.3f} ± {float(arr.std()):.3f}")

        return mcmc, participant_data_stacked

    # ------------------------------------------------------------------
    # Legacy qlearning / wmrl code paths (unchanged)
    # ------------------------------------------------------------------
    model_name = "Q-LEARNING" if model == 'qlearning' else "WM-RL"
    model_fn = qlearning_hierarchical_model if model == 'qlearning' else wmrl_hierarchical_model

    print("\n" + "=" * 80)
    print(f"FITTING {model_name} MODEL")
    print("=" * 80)

    # Prepare data
    print("\n>> Preparing data...")
    participant_data = prepare_data_for_numpyro(
        data,
        participant_col='sona_id',
        block_col='block',
        stimulus_col='stimulus',
        action_col='key_press',
        reward_col='reward',
        set_size_col='set_size'
    )

    num_participants = len(participant_data)
    total_blocks = sum([len(pdata['stimuli_blocks']) for pdata in participant_data.values()])
    total_trials = sum([
        sum([len(block) for block in pdata['stimuli_blocks']])
        for pdata in participant_data.values()
    ])

    print(f"  ✓ Prepared {num_participants} participants")
    print(f"  ✓ Total blocks: {total_blocks}")
    print(f"  ✓ Total trials: {total_trials}")

    # Run inference
    print(f"\n>> Running MCMC inference...")
    print(f"  Sampler: NUTS (gradient-based)")
    print(f"  Chains: {num_chains}")
    print(f"  Warmup: {num_warmup}")
    print(f"  Samples: {num_samples}")

    mcmc = run_inference(
        model=model_fn,
        model_args={'participant_data': participant_data},
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        seed=seed
    )

    # Print summary
    samples = mcmc.get_samples()
    print("\n" + "=" * 80)
    print(f"{model_name} POSTERIOR ESTIMATES")
    print("=" * 80)

    if model in ('qlearning', 'wmrl'):
        print("\nGroup-level parameters:")
        print(f"  μ_α+ : {samples['mu_alpha_pos'].mean():.3f} ± {samples['mu_alpha_pos'].std():.3f}")
        print(f"  μ_α- : {samples['mu_alpha_neg'].mean():.3f} ± {samples['mu_alpha_neg'].std():.3f}")
        if 'mu_beta' in samples:
            print(f"  μ_β  : {samples['mu_beta'].mean():.3f} ± {samples['mu_beta'].std():.3f}")
        if model == 'wmrl':
            if 'mu_beta_wm' in samples:
                print(f"  μ_β_WM: {samples['mu_beta_wm'].mean():.3f} ± {samples['mu_beta_wm'].std():.3f}")
            print(f"  μ_φ   : {samples['mu_phi'].mean():.3f} ± {samples['mu_phi'].std():.3f}")
            print(f"  μ_ρ   : {samples['mu_rho'].mean():.3f} ± {samples['mu_rho'].std():.3f}")
            print(f"  μ_K   : {samples['mu_capacity'].mean():.3f} ± {samples['mu_capacity'].std():.3f}")

    return mcmc, participant_data


def save_results(
    mcmc,
    data: pd.DataFrame,
    model: str,
    output_dir: Path,
    save_plots: bool = True,
    participant_data_stacked: dict | None = None,
):
    """Save fitting results to disk.

    For wmrl_m3, applies the convergence gate (HIER-07) and writes schema-parity
    CSV via ``write_bayesian_summary``, a NetCDF posterior, and a shrinkage report.
    For legacy qlearning/wmrl models, preserves the previous saving behaviour.

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        Fitted MCMC object.
    data : pd.DataFrame
        Original trial-level data (used for legacy paths and n_trials).
    model : str
        Model identifier (e.g. ``'wmrl_m3'``).
    output_dir : Path
        Root output directory for results.
    save_plots : bool
        Whether to save diagnostic trace/posterior plots (legacy paths only).
    participant_data_stacked : dict or None
        Stacked participant data dict from ``prepare_stacked_participant_data``.
        Required for ``'wmrl_m3'``.  Ignored for legacy models.
    """
    from scripts.fitting.bayesian_diagnostics import (
        filter_padding_from_loglik,
        compute_shrinkage_report,
        write_shrinkage_report,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = model

    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # ------------------------------------------------------------------
    # wmrl_m3 path: convergence gate + schema-parity CSV + shrinkage report
    # ------------------------------------------------------------------
    if model == 'wmrl_m3':
        if participant_data_stacked is None:
            raise ValueError(
                "participant_data_stacked is required for model='wmrl_m3'. "
                "Pass the dict returned by prepare_stacked_participant_data()."
            )

        pdata_stacked = participant_data_stacked
        participant_ids = sorted(pdata_stacked.keys())

        # Compute pointwise log-lik and filter padding before WAIC/LOO
        print("\n>> Computing pointwise log-likelihoods...")
        pointwise_loglik = compute_pointwise_log_lik(
            mcmc, pdata_stacked, "wmrl_m3"
        )
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
        M3_PARAMS = ["alpha_pos", "alpha_neg", "phi", "rho", "capacity", "kappa", "epsilon"]
        print("\n>> Checking convergence gate...")
        convergence_summary = az.summary(idata, var_names=M3_PARAMS)
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
            return  # Early return — no files written

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
        # Compute n_trials per participant for AIC/BIC
        n_trials_per_ppt = []
        for pid in participant_ids:
            ppt_df = data[data['sona_id'] == pid]
            n_trials_per_ppt.append(int(len(ppt_df)))

        csv_path = write_bayesian_summary(
            idata,
            "wmrl_m3",
            output_dir,
            participant_ids=participant_ids,
            parameterization_version="v4.0-K[2,6]-phiapprox",
            n_trials_per_participant=n_trials_per_ppt,
        )
        print(f"  Saved: {csv_path}")

        # NetCDF posterior
        netcdf_path = bayesian_dir / "wmrl_m3_posterior.nc"
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
        shrinkage = compute_shrinkage_report(idata, M3_PARAMS)
        print("  Shrinkage values:")
        for param, val in shrinkage.items():
            status = "identified" if val >= 0.3 else "WARNING: poorly identified"
            print(f"    {param}: {val:.3f} ({status})")

        shrinkage_path = bayesian_dir / "wmrl_m3_shrinkage_report.md"
        write_shrinkage_report(shrinkage, shrinkage_path)
        print(f"  Saved: {shrinkage_path}")

        print("\n>> All results saved successfully!")
        return idata

    # ------------------------------------------------------------------
    # Legacy qlearning / wmrl paths (unchanged behaviour)
    # ------------------------------------------------------------------
    # Convert to ArviZ InferenceData
    print("\n>> Converting to ArviZ format...")
    idata = samples_to_arviz(mcmc, data)

    # Save NetCDF
    netcdf_file = output_dir / f'{prefix}_posterior_{timestamp}.nc'
    idata.to_netcdf(netcdf_file)
    print(f"  ✓ Saved posterior: {netcdf_file}")

    # Save summary CSV
    summary_file = output_dir / f'{prefix}_summary_{timestamp}.csv'
    summary = az.summary(idata)
    summary.to_csv(summary_file)
    print(f"  ✓ Saved summary: {summary_file}")

    # Save diagnostic plots
    if save_plots:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        # Define variables to plot
        if model == 'qlearning':
            var_names = ['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta']
        else:  # wmrl
            var_names = ['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta', 'mu_beta_wm', 'mu_phi', 'mu_rho', 'mu_capacity']

        # Trace plot
        trace_file = output_dir / f'{prefix}_trace_{timestamp}.png'
        az.plot_trace(idata, var_names=var_names)
        plt.tight_layout()
        plt.savefig(trace_file, dpi=150, bbox_inches='tight')
        plt.close('all')
        print(f"  ✓ Saved trace plot: {trace_file}")

        # Posterior plot
        posterior_file = output_dir / f'{prefix}_posterior_{timestamp}.png'
        az.plot_posterior(idata, var_names=var_names)
        plt.tight_layout()
        plt.savefig(posterior_file, dpi=150, bbox_inches='tight')
        plt.close('all')
        print(f"  ✓ Saved posterior plot: {posterior_file}")

    print("\n>> All results saved successfully!")
    return idata


def main():
    """Main fitting workflow."""
    parser = argparse.ArgumentParser(
        description='Fit Bayesian hierarchical models using JAX/NumPyro'
    )
    parser.add_argument('--model', type=str, required=True,
                        choices=ALL_MODELS,
                        help='Model to fit (only qlearning and wmrl have full Bayesian implementations)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to trial data CSV')
    parser.add_argument('--min-block', type=int, default=None,
                        help='Minimum block to include (default: 3, use 1 for all blocks)')
    parser.add_argument('--include-practice', action='store_true',
                        help='Include practice blocks (1-2) in fitting. Equivalent to --min-block 1')
    parser.add_argument('--chains', type=int, default=4,
                        help='Number of MCMC chains (default: 4)')
    parser.add_argument('--warmup', type=int, default=1000,
                        help='Number of warmup samples (default: 1000)')
    parser.add_argument('--samples', type=int, default=2000,
                        help='Number of posterior samples per chain (default: 2000)')
    parser.add_argument('--output', type=str, default=str(OUTPUT_VERSION_DIR),
                        help='Output directory for results')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save diagnostic plots')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

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
    print(f"  Min block: {min_block}" + (" (includes practice)" if min_block < 3 else " (excludes practice)"))
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

    # For wmrl_m3+, extra is participant_data_stacked; for legacy models, extra is None
    save_results(
        mcmc,
        data,
        args.model,
        Path(args.output),
        args.save_plots,
        participant_data_stacked=extra if args.model in ('wmrl_m3',) else None,
    )

    print("\n" + "=" * 80)
    print("FITTING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
