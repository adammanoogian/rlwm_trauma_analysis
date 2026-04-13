#!/usr/bin/env python
"""13: Fit Bayesian M4 (LBA) Hierarchical Model.

Self-contained pipeline script for M4 joint choice+RT hierarchical Bayesian
fitting using NumPyro NUTS.  Implements:

- M4H-01: Float64 process isolation (jax_enable_x64 before ANY other import)
- M4H-03: chain_method='vectorized' (NOT 'parallel')
- M4H-04: Checkpoint-and-resume via warmup state pickle
- M4H-05: Pareto-k gating with 5% threshold and MLE AIC fallback
- Convergence gate: max_rhat < 1.01, min_ess_bulk > 400, divergences == 0

This script is SEPARATE from scripts/fitting/fit_bayesian.py because float64
is process-wide.  Importing lba_likelihood via fit_bayesian.py would activate
float64 globally; M4 requires float64 while choice-only models must run in
float32 for performance.

Inputs
------
- output/task_trials_long.csv (default) -- trial-level behavioral data

Outputs
-------
- output/bayesian/wmrl_m4_individual_fits.csv  (schema-parity CSV)
- output/bayesian/wmrl_m4_posterior.nc          (ArviZ InferenceData)
- output/bayesian/wmrl_m4_shrinkage_report.md
- output/bayesian/wmrl_m4_pareto_k_report.md
- output/bayesian/wmrl_m4_pareto_k_report.json
- output/bayesian/wmrl_m4_run_metadata.json
- output/bayesian/m4_warmup_state.pkl            (checkpoint; removed on clean run)

Usage
-----
    # Standard run
    python scripts/13_fit_bayesian_m4.py

    # With custom MCMC budget
    python scripts/13_fit_bayesian_m4.py --chains 4 --warmup 1000 --samples 1500

    # Resume after SLURM wall-time kill (warmup state already saved)
    python scripts/13_fit_bayesian_m4.py  # resumes automatically

    # Include practice blocks
    python scripts/13_fit_bayesian_m4.py \\
        --data output/task_trials_long_all.csv --include-practice

References
----------
M4H-01..M4H-05 requirements (Phase 17 M4 Hierarchical LBA).
Brown & Heathcote (2008) LBA density.
McDougle & Collins (2021) M4 parameterization.
Senta, Bishop, Collins (2025) WM-RL parameter bounds.
"""

from __future__ import annotations

# =============================================================================
# FLOAT64 ISOLATION (M4H-01)
# =============================================================================
# These MUST be the first executable lines before ANY other import.
# JAX sets dtype mode at first array creation -- any import that
# materializes a JAX array before this will silently run float32.
import jax

jax.config.update("jax_enable_x64", True)
import numpyro

numpyro.enable_x64()
# =============================================================================
# END FLOAT64 ISOLATION
# =============================================================================

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import arviz as az
import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpyro.infer import MCMC, NUTS

# Add project root to sys.path before local imports
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.fitting.lba_likelihood import wmrl_m4_multiblock_likelihood_stacked
from scripts.fitting.numpyro_models import (
    prepare_stacked_participant_data_m4,
    wmrl_m4_hierarchical_model,
)
from scripts.fitting.bayesian_summary_writer import write_bayesian_summary
from scripts.fitting.bayesian_diagnostics import (
    compute_shrinkage_report,
    write_shrinkage_report,
)

from config import EXCLUDED_PARTICIPANTS, EXPECTED_PARAMETERIZATION, MODEL_REGISTRY


# =============================================================================
# HELPERS
# =============================================================================


def _build_argparser() -> argparse.ArgumentParser:
    """Build CLI argument parser for M4 Bayesian fitting.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    p = argparse.ArgumentParser(
        description=(
            "Fit M4 (WM-RL+LBA) hierarchical Bayesian model via NumPyro NUTS. "
            "Float64 is required; this script is separate from 13_fit_bayesian.py."
        )
    )
    p.add_argument(
        "--data",
        type=str,
        default="output/task_trials_long.csv",
        help="Path to trial-level CSV. Default: output/task_trials_long.csv",
    )
    p.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of MCMC chains. Default: 4",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=1000,
        help="Number of NUTS warmup steps. Default: 1000",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=1500,
        help="Number of posterior samples per chain. Default: 1500",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="JAX random seed. Default: 42",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Root output directory. Default: output",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default="output/bayesian",
        help="Directory for warmup state checkpoint. Default: output/bayesian",
    )
    p.add_argument(
        "--include-practice",
        action="store_true",
        help="Include practice blocks (blocks 1-2) in fitting. Default: excluded.",
    )
    return p


def _load_data(data_path: Path, include_practice: bool) -> pd.DataFrame:
    """Load and prepare trial-level behavioral data for M4.

    Applies the same exclusions and type coercions as fit_bayesian.py, plus
    requires the ``rt`` column (RTs in milliseconds) that M4 needs.

    Parameters
    ----------
    data_path : Path
        Path to CSV with trial data.
    include_practice : bool
        If False, exclude rows where ``is_practice == True`` (or
        ``block < 3`` if the column is absent).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist.
    ValueError
        If ``rt`` column is missing (M4 requires RTs).
    """
    print(f"\n>> Loading data from: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    df = df.dropna(subset=["sona_id"]).copy()

    # Participant exclusions
    initial_n = df["sona_id"].nunique()
    df = df[~df["sona_id"].isin(EXCLUDED_PARTICIPANTS)].copy()
    n_excluded = initial_n - df["sona_id"].nunique()
    if n_excluded > 0:
        print(
            f"  Excluded {n_excluded} participants "
            f"(data quality); {df['sona_id'].nunique()} remain."
        )

    # Practice filtering
    if not include_practice:
        if "is_practice" in df.columns:
            df = df[df["is_practice"] == False].copy()  # noqa: E712
            print("  Filtered to non-practice trials via is_practice column.")
        else:
            df = df[df["block"] >= 3].copy()
            print("  Filtered to blocks >= 3 (no is_practice column found).")

    # Drop NaN in required columns
    required_cols = ["stimulus", "key_press", "correct"]
    df = df.dropna(subset=required_cols).copy()

    # RT column is required for M4
    if "rt" not in df.columns:
        raise ValueError(
            "Column 'rt' (reaction times in ms) is required for M4 fitting. "
            f"Got columns: {list(df.columns)}. "
            "Expected vs actual: 'rt' missing from DataFrame."
        )
    df = df.dropna(subset=["rt"]).copy()

    # Type coercions
    df["stimulus"] = df["stimulus"].astype(int)
    df["key_press"] = df["key_press"].astype(int)
    df["reward"] = df["correct"].astype(float)
    df["rt"] = df["rt"].astype(float)

    if "set_size" not in df.columns:
        print("  WARNING: 'set_size' column not found; defaulting to 6.")
        df["set_size"] = 6

    n_ppts = df["sona_id"].nunique()
    n_trials = len(df)
    print(f"\n  Data summary:")
    print(f"    Participants: {n_ppts}")
    print(f"    Blocks: {df['block'].nunique()}")
    print(f"    Total trials: {n_trials}")
    print(f"    Trials per participant: {n_trials / n_ppts:.0f}")

    return df


def compute_m4_pointwise_loglik(
    mcmc: MCMC,
    participant_data_stacked: dict,
) -> np.ndarray:
    """Compute participant-level total log-likelihoods for M4 WAIC/LOO.

    M4-specific replacement for ``compute_pointwise_log_lik`` in
    bayesian_diagnostics.  ``wmrl_m4_multiblock_likelihood_stacked`` returns
    a scalar NLL per participant (not per-trial) because the LBA likelihood
    does not expose per-trial log-probs via ``return_pointwise``.

    Uses participant-level log-lik for Pareto-k diagnostics.  Pareto-k > 0.7
    is near-certain for LBA under NUTS regardless of granularity (as noted in
    STATE.md).  Participant-level granularity is the practical approach.

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        Fitted MCMC object (after ``mcmc.run()``).
    participant_data_stacked : dict
        Mapping from participant_id to stacked arrays from
        ``prepare_stacked_participant_data_m4``.

    Returns
    -------
    np.ndarray
        Shape ``(chains, samples_per_chain, n_participants)``.
        Each element is the total log-likelihood (negative of NLL) for that
        participant under that posterior draw.
    """
    print("  Computing M4 participant-level log-likelihoods...")

    # samples[param] has shape (chains, samples_per_chain, n_participants)
    # or (chains, samples_per_chain) for scalar hyperparams
    samples = mcmc.get_samples(group_by_chain=True)
    participant_ids = sorted(participant_data_stacked.keys())
    n_participants = len(participant_ids)

    # Extract shapes
    n_chains = samples["alpha_pos"].shape[0]
    n_samples_per_chain = samples["alpha_pos"].shape[1]
    n_total_samples = n_chains * n_samples_per_chain

    # M4 individual-level params: shape (chains, samples, n_participants)
    m4_indiv_params = ["alpha_pos", "alpha_neg", "phi", "rho", "capacity", "kappa",
                       "v_scale", "A", "delta", "t0"]

    # Build result array: (n_total_samples, n_participants)
    # We iterate over participants (outer) and flat samples (inner)
    # to avoid memory blow-up with vmap over all params simultaneously.
    loglik_flat = np.zeros((n_total_samples, n_participants), dtype=np.float64)

    for p_idx, pid in enumerate(participant_ids):
        pdata = participant_data_stacked[pid]

        stimuli_stacked = pdata["stimuli_stacked"]
        actions_stacked = pdata["actions_stacked"]
        rewards_stacked = pdata["rewards_stacked"]
        set_sizes_stacked = pdata["set_sizes_stacked"]
        rts_stacked = pdata["rts_stacked"]
        masks_stacked = pdata["masks_stacked"]

        # Extract flat parameter arrays: (n_total_samples,) for this participant
        def _flat_param(name: str) -> np.ndarray:
            arr = samples[name]  # (chains, n_samples, n_participants)
            return np.array(arr[:, :, p_idx].reshape(-1))

        alpha_pos_arr = _flat_param("alpha_pos")
        alpha_neg_arr = _flat_param("alpha_neg")
        phi_arr = _flat_param("phi")
        rho_arr = _flat_param("rho")
        capacity_arr = _flat_param("capacity")
        kappa_arr = _flat_param("kappa")
        v_scale_arr = _flat_param("v_scale")
        A_arr = _flat_param("A")
        delta_arr = _flat_param("delta")
        t0_arr = _flat_param("t0")

        # Decode b = A + delta (M4H-02)
        b_arr = A_arr + delta_arr

        # Per-sample NLL function (vmappable)
        def _per_sample_nll(
            alpha_pos, alpha_neg, phi, rho, capacity, kappa,
            v_scale, A, b, t0,
        ):
            nll = wmrl_m4_multiblock_likelihood_stacked(
                stimuli_stacked=stimuli_stacked,
                actions_stacked=actions_stacked,
                rewards_stacked=rewards_stacked,
                set_sizes_stacked=set_sizes_stacked,
                rts_stacked=rts_stacked,
                masks_stacked=masks_stacked,
                alpha_pos=alpha_pos,
                alpha_neg=alpha_neg,
                phi=phi,
                rho=rho,
                capacity=capacity,
                kappa=kappa,
                v_scale=v_scale,
                A=A,
                b=b,
                t0=t0,
            )
            return -nll  # return log-lik, not NLL

        vmapped_fn = jax.jit(jax.vmap(_per_sample_nll))

        loglik_p = vmapped_fn(
            jnp.array(alpha_pos_arr),
            jnp.array(alpha_neg_arr),
            jnp.array(phi_arr),
            jnp.array(rho_arr),
            jnp.array(capacity_arr),
            jnp.array(kappa_arr),
            jnp.array(v_scale_arr),
            jnp.array(A_arr),
            jnp.array(b_arr),
            jnp.array(t0_arr),
        )
        loglik_flat[:, p_idx] = np.array(loglik_p)

        if (p_idx + 1) % 10 == 0 or (p_idx + 1) == n_participants:
            print(f"    Participant {p_idx + 1}/{n_participants} done.")

    # Reshape to (chains, samples_per_chain, n_participants)
    loglik = loglik_flat.reshape(n_chains, n_samples_per_chain, n_participants)
    print(f"  M4 log-lik shape: {loglik.shape}")
    return loglik


def _build_inference_data_m4(
    mcmc: MCMC,
    loglik: np.ndarray,
    participant_ids: list,
) -> az.InferenceData:
    """Build ArviZ InferenceData for M4 with participant-level log-lik.

    Parameters
    ----------
    mcmc : MCMC
        Fitted MCMC object.
    loglik : np.ndarray
        Shape ``(chains, samples_per_chain, n_participants)`` from
        :func:`compute_m4_pointwise_loglik`.
    participant_ids : list
        Ordered participant IDs.

    Returns
    -------
    az.InferenceData
        With ``posterior``, ``sample_stats``, and ``log_likelihood`` groups.
    """
    n_participants = loglik.shape[2]

    idata = az.from_numpyro(mcmc)

    # Remove any auto-created log_likelihood group (shape mismatch for WAIC/LOO)
    if "log_likelihood" in idata._groups:
        del idata["log_likelihood"]

    idata.add_groups(
        log_likelihood={"obs": loglik},
        coords={"participant": participant_ids},
        dims={"obs": ["participant"]},
    )
    return idata


def _write_run_metadata(
    args: argparse.Namespace,
    wall_time: float,
    max_rhat: float,
    min_ess: float,
    n_div: int,
    converged: bool,
    n_participants: int,
    output_dir: Path,
) -> None:
    """Write run metadata JSON for debugging and downstream consumption.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    wall_time : float
        Wall-clock time in seconds.
    max_rhat : float
        Maximum R-hat across model parameters.
    min_ess : float
        Minimum ESS-bulk across model parameters.
    n_div : int
        Number of divergent transitions.
    converged : bool
        Whether the convergence gate passed.
    n_participants : int
        Number of participants fitted.
    output_dir : Path
        Output directory for bayesian files.
    """
    bayesian_dir = output_dir / "bayesian"
    bayesian_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "script": "13_fit_bayesian_m4.py",
        "model": "wmrl_m4",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "wall_time_hours": round(wall_time / 3600, 4),
        "n_participants": n_participants,
        "n_chains": args.chains,
        "n_warmup": args.warmup,
        "n_samples": args.samples,
        "seed": args.seed,
        "chain_method": "vectorized",
        "target_accept_prob": 0.95,
        "max_tree_depth": 10,
        "max_rhat": round(float(max_rhat), 6),
        "min_ess_bulk": round(float(min_ess), 2),
        "n_divergences": int(n_div),
        "converged": converged,
        "convergence_criteria": "max_rhat < 1.01 AND min_ess_bulk > 400 AND n_divergences == 0",
    }
    metadata_path = bayesian_dir / "wmrl_m4_run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Run metadata saved: {metadata_path}")


def _write_pareto_k_report(pareto_report: dict, report_path: Path) -> None:
    """Write Pareto-k diagnostic report as markdown.

    Parameters
    ----------
    pareto_report : dict
        Dict with keys: pareto_k_frac_bad, pareto_k_threshold,
        n_trials_checked, n_trials_bad, loo_unreliable, fallback (if bad),
        loo_elpd.
    report_path : Path
        Path to write the markdown file.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)

    frac_bad = pareto_report.get("pareto_k_frac_bad", float("nan"))
    threshold = pareto_report.get("pareto_k_threshold", 0.05)
    n_checked = pareto_report.get("n_trials_checked", "?")
    n_bad = pareto_report.get("n_trials_bad", "?")
    loo_unreliable = pareto_report.get("loo_unreliable", True)
    loo_elpd = pareto_report.get("loo_elpd", None)
    fallback = pareto_report.get("fallback", "mle_aic_track")

    verdict_header = "FALLBACK: LOO Unreliable" if loo_unreliable else "PASS: LOO Reliable"
    verdict_detail = (
        f"**{100 * frac_bad:.1f}%** of observations have Pareto-k > 0.7 "
        f"(threshold: {100 * threshold:.0f}%)."
    )

    if loo_unreliable:
        recommendation = (
            "LOO is unreliable for M4 in this dataset. "
            "Use MLE AIC track for cross-model comparison (M4 vs choice-only). "
            "LOO ELPD reported below as a standalone M4 quality metric only "
            "(not for cross-model ranking)."
        )
        fallback_note = f"Fallback: `{fallback}`"
    else:
        recommendation = (
            "LOO is reliable for M4 in this dataset. "
            "LOO ELPD can be used for cross-model comparison with care "
            "(M4 likelihood is not comparable to choice-only models)."
        )
        fallback_note = "No fallback required."

    elpd_line = f"LOO ELPD: {loo_elpd:.2f}" if loo_elpd is not None else "LOO ELPD: N/A"

    lines = [
        "# M4 Pareto-k Diagnostic Report",
        "",
        "Assesses reliability of Leave-One-Out Cross-Validation (LOO) for the",
        "M4 (WM-RL+LBA) joint choice+RT hierarchical model (M4H-05).",
        "",
        "## Results",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Observations with Pareto-k > 0.7 | {100 * frac_bad:.1f}% ({n_bad}/{n_checked}) |",
        f"| Threshold | {100 * threshold:.0f}% |",
        f"| LOO reliable | {'No' if loo_unreliable else 'Yes'} |",
        f"| {elpd_line} | |",
        "",
        "## Verdict",
        "",
        f"### {verdict_header}",
        "",
        verdict_detail,
        "",
        recommendation,
        "",
        fallback_note,
        "",
        "## Notes",
        "",
        "- Pareto-k > 0.7 is near-certain for LBA under NUTS (participant-level",
        "  observation granularity used since LBA does not expose per-trial",
        "  log-probs via `return_pointwise`).",
        "- LOO ELPD is still reported as a standalone M4 quality metric.",
        "- For cross-model comparison (M4 vs M1-M6b), use MLE AIC from",
        "  output/model_comparison/ (separate comparison track).",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Pareto-k report saved: {report_path}")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Run M4 hierarchical Bayesian fitting pipeline.

    Implements M4H-01 (float64), M4H-03 (chain_method=vectorized),
    M4H-04 (checkpoint-resume), M4H-05 (Pareto-k gating).

    Returns
    -------
    None
    """
    parser = _build_argparser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    bayesian_dir = output_dir / "bayesian"
    checkpoint_path = Path(args.checkpoint_dir) / "m4_warmup_state.pkl"

    # ------------------------------------------------------------------
    # M4H-01: Float64 verification
    # ------------------------------------------------------------------
    assert jnp.zeros(1).dtype == jnp.float64, (
        f"FATAL: float64 not active. Got dtype={jnp.zeros(1).dtype}. "
        "'jax.config.update(jax_enable_x64, True)' must be the first "
        "executable line in this script."
    )
    print(f"[M4H-01] Float64 verified: jnp.zeros(1).dtype={jnp.zeros(1).dtype}")

    # ------------------------------------------------------------------
    # Load and prepare data
    # ------------------------------------------------------------------
    data_path = Path(args.data)
    data = _load_data(data_path, include_practice=args.include_practice)
    n_participants = data["sona_id"].nunique()

    # ------------------------------------------------------------------
    # Prepare stacked participant data (M4 variant with RT)
    # ------------------------------------------------------------------
    print("\n>> Preparing stacked M4 participant data (with RT)...")
    participant_data_stacked = prepare_stacked_participant_data_m4(
        data,
        participant_col="sona_id",
        block_col="block",
        stimulus_col="stimulus",
        action_col="key_press",
        reward_col="reward",
        set_size_col="set_size",
        rt_col="rt",
    )
    participant_ids = sorted(participant_data_stacked.keys())
    print(f"  Prepared {len(participant_ids)} participants (sorted by ID)")

    # ------------------------------------------------------------------
    # M4H-03: MCMC setup with chain_method='vectorized'
    # ------------------------------------------------------------------
    print(f"\n>> Setting up NUTS (chains={args.chains}, "
          f"warmup={args.warmup}, samples={args.samples})...")
    nuts_kernel = NUTS(
        wmrl_m4_hierarchical_model,
        target_accept_prob=0.95,
        max_tree_depth=10,
    )
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        chain_method="vectorized",  # NOT 'parallel' (M4H-03)
        progress_bar=True,
    )

    model_kwargs = {"participant_data_stacked": participant_data_stacked}
    rng_key = jax.random.PRNGKey(args.seed)

    # ------------------------------------------------------------------
    # M4H-04: Checkpoint-and-resume
    # ------------------------------------------------------------------
    start_time = time.time()

    if checkpoint_path.exists():
        print(f"\n[M4H-04] Resuming from warmup checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "rb") as fh:
            warmup_state = pickle.load(fh)
        mcmc.post_warmup_state = warmup_state
        mcmc.run(warmup_state.rng_key, **model_kwargs)
    else:
        print(f"\n[M4H-04] Running warmup phase "
              f"({args.warmup} steps, {args.chains} chains)...")
        mcmc.warmup(rng_key, **model_kwargs)

        # jax.device_get ensures arrays are CPU-backed before pickling
        print(f"[M4H-04] Saving warmup state to: {checkpoint_path}")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        warmup_state_cpu = jax.device_get(mcmc.post_warmup_state)
        with open(checkpoint_path, "wb") as fh:
            pickle.dump(warmup_state_cpu, fh)
        print("[M4H-04] Warmup state saved. Running sampling phase...")
        mcmc.run(mcmc.post_warmup_state.rng_key, **model_kwargs)

    wall_time = time.time() - start_time
    print(f"\n[TIMING] Total MCMC wall time: {wall_time / 3600:.2f} hours "
          f"({wall_time:.0f}s)")

    # ------------------------------------------------------------------
    # Print group-level posterior summaries
    # ------------------------------------------------------------------
    samples = mcmc.get_samples()
    print("\n" + "=" * 80)
    print("M4 POSTERIOR ESTIMATES — GROUP LEVEL")
    print("=" * 80)
    mu_pr_sites = sorted(k for k in samples if k.endswith("_mu_pr"))
    for pname in mu_pr_sites:
        arr = samples[pname]
        print(f"  {pname}: {float(arr.mean()):.3f} +/- {float(arr.std()):.3f}")

    # ------------------------------------------------------------------
    # Compute M4 participant-level log-likelihoods (M4H-05 prerequisite)
    # ------------------------------------------------------------------
    print("\n>> Computing M4 participant-level log-likelihoods (inline)...")
    loglik = compute_m4_pointwise_loglik(mcmc, participant_data_stacked)

    # Build InferenceData with participant-level log-lik group
    print("\n>> Building ArviZ InferenceData...")
    idata = _build_inference_data_m4(mcmc, loglik, participant_ids)

    # ------------------------------------------------------------------
    # Convergence gate (M4H-03 + HIER-07 strict criterion)
    # max_rhat < 1.01 AND min_ess_bulk > 400 AND n_divergences == 0
    # ------------------------------------------------------------------
    model_params = MODEL_REGISTRY["wmrl_m4"]["params"]
    print("\n>> Checking convergence gate...")
    convergence_summary = az.summary(idata, var_names=model_params)
    max_rhat = float(convergence_summary["r_hat"].max())
    min_ess = float(convergence_summary["ess_bulk"].min())
    extra = mcmc.get_extra_fields()
    n_div = int(extra["diverging"].sum()) if "diverging" in extra else 0
    converged = max_rhat < 1.01 and min_ess > 400 and n_div == 0

    print(
        f"  max_rhat={max_rhat:.4f} "
        f"min_ess_bulk={min_ess:.0f} "
        f"divergences={n_div} "
        f"=> converged={converged}"
    )

    # Always write run metadata (useful for debugging even on failure)
    _write_run_metadata(
        args=args,
        wall_time=wall_time,
        max_rhat=max_rhat,
        min_ess=min_ess,
        n_div=n_div,
        converged=converged,
        n_participants=n_participants,
        output_dir=output_dir,
    )

    if not converged:
        print(
            f"\n[CONVERGENCE GATE FAILED] max_rhat={max_rhat:.4f}, "
            f"min_ess_bulk={min_ess:.0f}, divergences={n_div}"
        )
        print(
            "Refusing to write output files. "
            "Fix convergence issues and re-run (or extend budget)."
        )
        return

    print(
        f"\n[CONVERGENCE GATE PASSED] max_rhat={max_rhat:.4f}, "
        f"min_ess_bulk={min_ess:.0f}, divergences={n_div}"
    )

    # ------------------------------------------------------------------
    # Write outputs (only reached if convergence gate passes)
    # ------------------------------------------------------------------
    bayesian_dir.mkdir(parents=True, exist_ok=True)

    # Schema-parity CSV via write_bayesian_summary
    print("\n>> Writing schema-parity CSV (wmrl_m4_individual_fits.csv)...")
    n_trials_per_ppt = [
        int(len(data[data["sona_id"] == pid])) for pid in participant_ids
    ]
    csv_path = write_bayesian_summary(
        idata,
        "wmrl_m4",
        output_dir,
        participant_ids=participant_ids,
        parameterization_version=EXPECTED_PARAMETERIZATION.get(
            "wmrl_m4", "v4.0-K[2,6]-phiapprox-lba"
        ),
        n_trials_per_participant=n_trials_per_ppt,
    )
    print(f"  Saved: {csv_path}")

    # NetCDF posterior
    netcdf_path = bayesian_dir / "wmrl_m4_posterior.nc"
    idata.to_netcdf(str(netcdf_path))
    print(f"  Saved posterior NetCDF: {netcdf_path}")

    # Shrinkage diagnostic
    print("\n>> Computing shrinkage diagnostic...")
    shrinkage = compute_shrinkage_report(idata, model_params)
    print("  Shrinkage values:")
    for param, val in shrinkage.items():
        status = "identified" if val >= 0.3 else "WARNING: poorly identified"
        print(f"    {param}: {val:.3f} ({status})")

    shrinkage_path = bayesian_dir / "wmrl_m4_shrinkage_report.md"
    write_shrinkage_report(shrinkage, shrinkage_path)
    print(f"  Saved: {shrinkage_path}")

    # ------------------------------------------------------------------
    # M4H-05: Pareto-k gating
    # ------------------------------------------------------------------
    print("\n>> Pareto-k gating (M4H-05)...")
    try:
        loo_result = az.loo(idata, pointwise=True)
        pareto_k = loo_result.pareto_k.values
        frac_bad = float((pareto_k > 0.7).mean())
        n_bad = int((pareto_k > 0.7).sum())

        pareto_report: dict = {
            "pareto_k_frac_bad": frac_bad,
            "pareto_k_threshold": 0.05,
            "n_trials_checked": int(pareto_k.size),
            "n_trials_bad": n_bad,
            "loo_elpd": float(loo_result.elpd_loo),
        }

        if frac_bad > 0.05:
            print(
                f"[M4H-05 FALLBACK] {100 * frac_bad:.1f}% of observations have "
                f"Pareto-k > 0.7 ({n_bad}/{pareto_k.size})."
            )
            print(
                "LOO is unreliable for M4. "
                "Use MLE AIC track for cross-model comparison."
            )
            print("LOO ELPD reported as standalone M4 quality metric only.")
            pareto_report["loo_unreliable"] = True
            pareto_report["fallback"] = "mle_aic_track"
        else:
            print(
                f"[M4H-05 PASS] {100 * frac_bad:.1f}% Pareto-k > 0.7 "
                f"(threshold: 5%)"
            )
            pareto_report["loo_unreliable"] = False

    except Exception as exc:
        print(f"  WARNING: LOO computation failed: {exc}")
        pareto_report = {
            "pareto_k_frac_bad": float("nan"),
            "pareto_k_threshold": 0.05,
            "n_trials_checked": 0,
            "n_trials_bad": 0,
            "loo_unreliable": True,
            "fallback": "mle_aic_track",
            "loo_error": str(exc),
        }

    # Write Pareto-k report (markdown)
    pareto_md_path = bayesian_dir / "wmrl_m4_pareto_k_report.md"
    _write_pareto_k_report(pareto_report, pareto_md_path)

    # Write Pareto-k report (JSON for Phase 18 consumption)
    pareto_json_path = bayesian_dir / "wmrl_m4_pareto_k_report.json"
    with open(pareto_json_path, "w") as fh:
        json.dump(pareto_report, fh, indent=2)
    print(f"  Pareto-k JSON saved: {pareto_json_path}")

    print("\n>> All M4 results saved successfully!")
    print(f"   CSV: {csv_path}")
    print(f"   NetCDF: {netcdf_path}")
    print(f"   Shrinkage: {shrinkage_path}")
    print(f"   Pareto-k: {pareto_md_path}")
    print(f"   Metadata: {output_dir}/bayesian/wmrl_m4_run_metadata.json")


if __name__ == "__main__":
    main()
