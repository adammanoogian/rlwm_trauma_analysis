"""Prior-predictive check runner for RLWM choice-only models (Baribault gate).

Implements the step 21.1 prior-predictive check for the v4.0 principled
Bayesian model-selection pipeline.  For the named model, it:

1. Loads the canonical v4.0 cohort (N=138) via
   ``scripts.fitting.fit_bayesian.load_and_prepare_data``.
2. Builds the stacked-participant arrays consumed by the hierarchical model
   (``prepare_stacked_participant_data`` + ``stack_across_participants``).
3. Draws ``--num-draws`` samples from the *prior* of the hierarchical model
   via ``numpyro.infer.Predictive`` with no observations conditioned
   (``PARAM_PRIOR_DEFAULTS`` are the currently locked v4.0 priors with all
   ``mu_prior_loc=0.0`` except ``epsilon=-2.0``).
4. For each draw, simulates one participant's trial sequence under that
   draw's individual-level parameters using a model-specific choice policy
   (mirrors the likelihood update equations in
   ``scripts.fitting.jax_likelihoods``).  Uses the real participant trial
   structure (stimuli, set sizes, block boundaries) as the task template.
5. Applies the three-part Baribault & Collins (2023) gate:
   median accuracy in [0.4, 0.9], <10% draws sub-chance, <5% draws at-ceiling.
6. Writes ``{model}_prior_sim.nc`` (ArviZ NetCDF), ``{model}_prior_accuracy.csv``
   (per-draw table) and ``{model}_gate.md`` (PASS/FAIL verdict).

Exits 1 on FAIL so that a pipeline orchestrator can short-circuit.

References
----------
Baribault, B. & Collins, A. G. E. (2023).  Troubleshooting Bayesian cognitive
models.  *Psychological Methods*.  https://doi.org/10.1037/met0000554

Hess, B. et al. (2025).  A robust Bayesian workflow for computational
psychiatry.  *Computational Psychiatry*, 9(1):76-99.
https://doi.org/10.5334/cpsy.116

Usage
-----
python scripts/21_run_prior_predictive.py --model wmrl_m3 --num-draws 500
python scripts/21_run_prior_predictive.py --model qlearning --num-draws 20 \\
    --output-dir /tmp/ppc_smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpyro  # noqa: E402

from numpyro.infer import Predictive  # noqa: E402

from config import MODEL_REGISTRY  # noqa: E402
from scripts.fitting.fit_bayesian import (  # noqa: E402
    STACKED_MODEL_DISPATCH,
    load_and_prepare_data,
)
from scripts.fitting.numpyro_helpers import PARAM_PRIOR_DEFAULTS  # noqa: E402
from scripts.fitting.numpyro_models import (  # noqa: E402
    prepare_stacked_participant_data,
    stack_across_participants,
)

# Senta et al. (2025) canonical constants
FIXED_BETA: float = 50.0
NUM_ACTIONS: int = 3
Q_INIT: float = 0.5
WM_INIT: float = 1.0 / NUM_ACTIONS

# Working stimulus capacity for the simulator.  Real data CSVs store stimuli
# as 1-indexed ints in [1, 6] (see ``scripts/01_parse_raw_data.py`` notes);
# the fit_bayesian JAX likelihoods silently clip with ``num_stimuli=6`` by
# relying on JAX's out-of-bounds edge-clip semantics, which is safe but
# mirrors as ``num_stimuli=7`` in a plain-NumPy simulator so that index 6
# maps to its own row rather than aliasing onto index 5.
NUM_STIMULI: int = 7


# ---------------------------------------------------------------------------
# Per-trial policy helpers (NumPy) — mirror jax_likelihoods.*_block_likelihood
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray, beta: float) -> np.ndarray:
    """Numerically stable softmax with inverse-temperature ``beta``."""
    z = beta * (x - x.max())
    ez = np.exp(z)
    return ez / ez.sum()


def _apply_epsilon(probs: np.ndarray, epsilon: float, n_act: int) -> np.ndarray:
    """hBayesDM-style uniform-mixing epsilon noise."""
    return epsilon / n_act + (1.0 - epsilon) * probs


# ---------------------------------------------------------------------------
# Model-specific simulators — one participant, one draw
# ---------------------------------------------------------------------------


def _simulate_qlearning(
    stimuli_blocks: list[np.ndarray],
    set_sizes_blocks: list[np.ndarray],
    rng: np.random.Generator,
    *,
    alpha_pos: float,
    alpha_neg: float,
    epsilon: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Simulate M1 (Q-learning) actions+rewards given a real trial template."""
    actions_blocks: list[np.ndarray] = []
    rewards_blocks: list[np.ndarray] = []

    for stim in stimuli_blocks:
        n_trials = len(stim)
        Q = np.full((NUM_STIMULI, NUM_ACTIONS), Q_INIT, dtype=np.float32)
        correct_map = rng.integers(0, NUM_ACTIONS, size=NUM_STIMULI)

        acts = np.empty(n_trials, dtype=np.int32)
        rews = np.empty(n_trials, dtype=np.float32)

        for t in range(n_trials):
            s = int(stim[t])
            probs = _softmax(Q[s], FIXED_BETA)
            probs = _apply_epsilon(probs, epsilon, NUM_ACTIONS)
            a = int(rng.choice(NUM_ACTIONS, p=probs))
            r = 1.0 if a == int(correct_map[s]) else 0.0

            delta = r - Q[s, a]
            alpha = alpha_pos if delta > 0 else alpha_neg
            Q[s, a] += alpha * delta

            acts[t] = a
            rews[t] = r

        actions_blocks.append(acts)
        rewards_blocks.append(rews)

    return actions_blocks, rewards_blocks


def _simulate_wmrl_family(
    stimuli_blocks: list[np.ndarray],
    set_sizes_blocks: list[np.ndarray],
    rng: np.random.Generator,
    *,
    model: str,
    alpha_pos: float,
    alpha_neg: float,
    phi: float,
    rho: float,
    capacity: float,
    epsilon: float,
    kappa: float = 0.0,
    kappa_s: float = 0.0,
    phi_rl: float = 0.0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Simulate WM-RL family (M2, M3, M5, M6a, M6b) actions+rewards.

    Dispatch by ``model`` — each variant toggles a different set of kernel
    and decay terms from ``{kappa, kappa_s, phi_rl}``.
    """
    actions_blocks: list[np.ndarray] = []
    rewards_blocks: list[np.ndarray] = []
    Q0 = 1.0 / NUM_ACTIONS

    for stim, ss in zip(stimuli_blocks, set_sizes_blocks):
        n_trials = len(stim)
        Q = np.full((NUM_STIMULI, NUM_ACTIONS), Q_INIT, dtype=np.float32)
        WM = np.full((NUM_STIMULI, NUM_ACTIONS), WM_INIT, dtype=np.float32)
        WM_BASE = np.full((NUM_STIMULI, NUM_ACTIONS), WM_INIT, dtype=np.float32)
        last_action_global = -1
        last_actions_stim = np.full(NUM_STIMULI, -1, dtype=np.int32)
        correct_map = rng.integers(0, NUM_ACTIONS, size=NUM_STIMULI)

        acts = np.empty(n_trials, dtype=np.int32)
        rews = np.empty(n_trials, dtype=np.float32)

        for t in range(n_trials):
            s = int(stim[t])
            set_size = float(ss[t]) if ss[t] > 0 else 6.0

            # 1. WM decay
            WM = (1.0 - phi) * WM + phi * WM_BASE
            # 1a. M5 RL forgetting
            if model == "wmrl_m5" and phi_rl > 0.0:
                Q_eff = (1.0 - phi_rl) * Q + phi_rl * Q0
            else:
                Q_eff = Q

            # 2. Policy
            omega = rho * min(1.0, capacity / set_size)
            rl_probs = _softmax(Q_eff[s], FIXED_BETA)
            wm_probs = _softmax(WM[s], FIXED_BETA)
            base = omega * wm_probs + (1.0 - omega) * rl_probs
            base = base / base.sum()
            noisy = _apply_epsilon(base, epsilon, NUM_ACTIONS)

            # 3. Perseveration kernel mixing (model-specific)
            if model in ("wmrl", "qlearning"):
                probs = noisy
            elif model in ("wmrl_m3", "wmrl_m5"):
                if kappa > 0.0 and last_action_global >= 0:
                    ck = np.zeros(NUM_ACTIONS, dtype=np.float32)
                    ck[last_action_global] = 1.0
                    probs = (1.0 - kappa) * noisy + kappa * ck
                else:
                    probs = noisy
            elif model == "wmrl_m6a":
                last_a_s = int(last_actions_stim[s])
                if kappa_s > 0.0 and last_a_s >= 0:
                    ck = np.zeros(NUM_ACTIONS, dtype=np.float32)
                    ck[last_a_s] = 1.0
                    probs = (1.0 - kappa_s) * noisy + kappa_s * ck
                else:
                    probs = noisy
            elif model == "wmrl_m6b":
                # Dual perseveration: global + stim-specific blended
                has_global = kappa > 0.0 and last_action_global >= 0
                last_a_s = int(last_actions_stim[s])
                has_stim = kappa_s > 0.0 and last_a_s >= 0
                ck_g = np.zeros(NUM_ACTIONS, dtype=np.float32)
                ck_s = np.zeros(NUM_ACTIONS, dtype=np.float32)
                if has_global:
                    ck_g[last_action_global] = 1.0
                if has_stim:
                    ck_s[last_a_s] = 1.0
                eff_kappa = kappa if has_global else 0.0
                eff_kappa_s = kappa_s if has_stim else 0.0
                probs = (
                    (1.0 - eff_kappa - eff_kappa_s) * noisy
                    + eff_kappa * ck_g
                    + eff_kappa_s * ck_s
                )
            else:
                raise ValueError(f"Unknown model: {model}")

            # Normalize (guard against tiny numerical drift)
            probs = np.clip(probs, 0.0, 1.0)
            probs = probs / probs.sum()

            a = int(rng.choice(NUM_ACTIONS, p=probs))
            r = 1.0 if a == int(correct_map[s]) else 0.0

            # 4. Updates — WM immediate overwrite
            WM[s, a] = r
            # 5. Q-table asymmetric delta rule (use Q_eff state for M5)
            if model == "wmrl_m5" and phi_rl > 0.0:
                # Persist the forgotten state, then apply delta
                Q = Q_eff.copy()
            q_cur = Q[s, a]
            delta = r - q_cur
            alpha = alpha_pos if delta > 0 else alpha_neg
            Q[s, a] = q_cur + alpha * delta

            # 6. Perseveration state updates
            last_action_global = a
            last_actions_stim[s] = a

            acts[t] = a
            rews[t] = r

        actions_blocks.append(acts)
        rewards_blocks.append(rews)

    return actions_blocks, rewards_blocks


# ---------------------------------------------------------------------------
# Draw extraction — pluck one (participant, draw) param vector
# ---------------------------------------------------------------------------


def _extract_param_vector(
    prior_samples: dict[str, np.ndarray],
    model: str,
    draw_idx: int,
    ppt_idx: int,
) -> dict[str, float]:
    """Build a param dict for a single (draw, participant) combination.

    Handles the M6b stick-breaking decode: the model samples ``kappa_total``
    and ``kappa_share`` as deterministics, so we rebuild
    ``kappa = kappa_total * kappa_share`` and
    ``kappa_s = kappa_total * (1 - kappa_share)`` to feed the simulator.
    """
    out: dict[str, float] = {}
    params = MODEL_REGISTRY[model]["params"]

    # Standard bounded params are all registered as deterministic sites
    # named after the parameter.  Shape: (num_draws, n_participants).
    for pname in params:
        if pname in prior_samples:
            arr = np.asarray(prior_samples[pname])
            out[pname] = float(arr[draw_idx, ppt_idx])

    # M6b decode from (kappa_total, kappa_share)
    if model == "wmrl_m6b":
        if "kappa_total" in prior_samples and "kappa_share" in prior_samples:
            kt = float(np.asarray(prior_samples["kappa_total"])[draw_idx, ppt_idx])
            ks = float(np.asarray(prior_samples["kappa_share"])[draw_idx, ppt_idx])
            out["kappa"] = kt * ks
            out["kappa_s"] = kt * (1.0 - ks)

    return out


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def _unstack_participant_template(
    stacked: dict,
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    """Reconstruct per-participant lists of (stimuli, set_sizes) blocks.

    Uses ``masks`` to drop padded trials and ``n_blocks_per_ppt`` to drop
    padded blocks.  Returns one list-of-blocks per participant, matching
    the sorted participant order in ``stacked["participant_ids"]``.
    """
    stimuli_tensor = np.asarray(stacked["stimuli"])  # (N, B, T)
    set_sizes_tensor = np.asarray(stacked["set_sizes"])
    masks_tensor = np.asarray(stacked["masks"])
    n_blocks_per_ppt = np.asarray(stacked["n_blocks_per_ppt"])

    n_participants = stimuli_tensor.shape[0]
    stim_per_ppt: list[list[np.ndarray]] = []
    ss_per_ppt: list[list[np.ndarray]] = []

    for p in range(n_participants):
        n_blocks = int(n_blocks_per_ppt[p])
        blocks_stim: list[np.ndarray] = []
        blocks_ss: list[np.ndarray] = []
        for b in range(n_blocks):
            mask_bt = masks_tensor[p, b]
            valid = mask_bt > 0.5
            blocks_stim.append(stimuli_tensor[p, b][valid].astype(np.int32))
            blocks_ss.append(set_sizes_tensor[p, b][valid].astype(np.float32))
        stim_per_ppt.append(blocks_stim)
        ss_per_ppt.append(blocks_ss)

    return stim_per_ppt, ss_per_ppt


def _evaluate_gate(accuracies: np.ndarray) -> tuple[bool, dict[str, float]]:
    """Apply the three-part Baribault & Collins (2023) gate.

    Returns
    -------
    (passed, metrics) : tuple
        ``passed`` is True iff all three sub-gates pass.  ``metrics`` holds
        ``median``, ``frac_below_chance``, ``frac_at_ceiling`` and
        ``n_draws``.
    """
    median = float(np.median(accuracies))
    below = float(np.mean(accuracies < 0.33))
    ceiling = float(np.mean(accuracies > 0.95))
    passed = (0.4 <= median <= 0.9) and (below < 0.10) and (ceiling < 0.05)
    return passed, {
        "median": median,
        "frac_below_chance": below,
        "frac_at_ceiling": ceiling,
        "n_draws": int(len(accuracies)),
    }


def _write_gate_report(
    path: Path,
    model: str,
    metrics: dict[str, float],
    passed: bool,
    num_draws: int,
    seed: int,
) -> None:
    """Write the ``{model}_gate.md`` markdown report."""
    median_ok = 0.4 <= metrics["median"] <= 0.9
    below_ok = metrics["frac_below_chance"] < 0.10
    ceiling_ok = metrics["frac_at_ceiling"] < 0.05

    lines = [
        f"# Prior predictive gate: {model}",
        "",
        f"- **Verdict:** {'**PASS**' if passed else '**FAIL**'}",
        f"- Draws: {num_draws}",
        f"- Seed: {seed}",
        f"- Real draws simulated: {metrics['n_draws']}",
        "",
        "## Gate checks (Baribault & Collins 2023)",
        "",
        f"| Check | Threshold | Value | Pass |",
        f"|---|---|---|---|",
        (
            f"| Median accuracy | [0.40, 0.90] | "
            f"{metrics['median']:.3f} | {'yes' if median_ok else 'no'} |"
        ),
        (
            f"| Sub-chance fraction | < 10% | "
            f"{metrics['frac_below_chance'] * 100:.1f}% | "
            f"{'yes' if below_ok else 'no'} |"
        ),
        (
            f"| At-ceiling fraction | < 5% | "
            f"{metrics['frac_at_ceiling'] * 100:.1f}% | "
            f"{'yes' if ceiling_ok else 'no'} |"
        ),
        "",
        "## Priors used",
        "",
        "```",
        *[f"{k}: {v}" for k, v in sorted(PARAM_PRIOR_DEFAULTS.items())],
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_prior_predictive(
    model: str,
    data_path: Path,
    num_draws: int,
    seed: int,
    output_dir: Path,
) -> int:
    """Run the prior-predictive gate for a single model.

    Returns
    -------
    int
        Exit code: 0 on PASS, 1 on FAIL.
    """
    t0 = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)

    if model not in STACKED_MODEL_DISPATCH:
        raise ValueError(
            f"Model '{model}' not in STACKED_MODEL_DISPATCH "
            f"(expected one of {sorted(STACKED_MODEL_DISPATCH.keys())})."
        )

    print("=" * 72)
    print(f"Prior-predictive gate: {model}")
    print("=" * 72)
    print(f"Priors in use (PARAM_PRIOR_DEFAULTS):")
    for k, v in sorted(PARAM_PRIOR_DEFAULTS.items()):
        print(f"  {k}: {v}")
    print()

    # --- 1. Load data (canonical cohort) -----------------------------------
    df = load_and_prepare_data(data_path, use_cohort=True)

    # --- 2. Build stacked arrays ------------------------------------------
    ppt_data = prepare_stacked_participant_data(df)
    stacked = stack_across_participants(ppt_data)
    n_participants = len(stacked["participant_ids"])
    print(f"\n>> Stacked arrays built: N={n_participants} participants")

    # --- 3. Sample prior via Predictive -----------------------------------
    model_fn = STACKED_MODEL_DISPATCH[model]
    predictive = Predictive(model_fn, num_samples=num_draws)

    rng_key = jax.random.PRNGKey(seed)
    print(f"\n>> Drawing {num_draws} samples from the prior...")
    prior_samples = predictive(
        rng_key,
        participant_data_stacked=ppt_data,
        covariate_lec=None,
        stacked_arrays=stacked,
        use_pscan=False,
    )
    # Convert to numpy for downstream indexing
    prior_samples_np: dict[str, np.ndarray] = {
        k: np.asarray(v) for k, v in prior_samples.items()
    }
    print(f"   Prior sample keys: {sorted(prior_samples_np.keys())[:8]}...")

    # --- 4. Simulate — one (draw, participant) pair per draw --------------
    print("\n>> Building trial-sequence template per participant...")
    stim_template, ss_template = _unstack_participant_template(stacked)

    rng = np.random.default_rng(seed)
    accuracies = np.empty(num_draws, dtype=np.float32)
    print(f">> Simulating {num_draws} (draw x participant) pairs...")
    for d in range(num_draws):
        # Sample a participant index uniformly; avoids 500x138 cartesian
        p = int(rng.integers(0, n_participants))
        params = _extract_param_vector(prior_samples_np, model, d, p)
        stim_blocks = stim_template[p]
        ss_blocks = ss_template[p]

        if model == "qlearning":
            _, r_blocks = _simulate_qlearning(
                stim_blocks,
                ss_blocks,
                rng,
                alpha_pos=params["alpha_pos"],
                alpha_neg=params["alpha_neg"],
                epsilon=params["epsilon"],
            )
        else:
            _, r_blocks = _simulate_wmrl_family(
                stim_blocks,
                ss_blocks,
                rng,
                model=model,
                alpha_pos=params["alpha_pos"],
                alpha_neg=params["alpha_neg"],
                phi=params.get("phi", 0.0),
                rho=params.get("rho", 0.0),
                capacity=params.get("capacity", 3.0),
                epsilon=params["epsilon"],
                kappa=params.get("kappa", 0.0),
                kappa_s=params.get("kappa_s", 0.0),
                phi_rl=params.get("phi_rl", 0.0),
            )

        all_rewards = np.concatenate(r_blocks)
        accuracies[d] = float(all_rewards.mean()) if len(all_rewards) else 0.0

        if (d + 1) % max(1, num_draws // 10) == 0:
            print(
                f"   {d + 1}/{num_draws} draws, running median acc = "
                f"{float(np.median(accuracies[: d + 1])):.3f}"
            )

    # --- 5. Gate evaluation -----------------------------------------------
    passed, metrics = _evaluate_gate(accuracies)
    print("\n>> Gate metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")
    print(f">> Verdict: {'PASS' if passed else 'FAIL'}")

    # --- 6. Write artifacts -----------------------------------------------
    below_chance = (accuracies < 0.33).astype(np.int8)
    at_ceiling = (accuracies > 0.95).astype(np.int8)
    csv_df = pd.DataFrame(
        {
            "draw_idx": np.arange(num_draws),
            "accuracy": accuracies,
            "below_chance": below_chance,
            "at_ceiling": at_ceiling,
        }
    )
    csv_path = output_dir / f"{model}_prior_accuracy.csv"
    csv_df.to_csv(csv_path, index=False)

    # NetCDF via arviz.from_dict — wrap prior samples (filter out pscan/obs sites)
    prior_dict = {
        k: np.asarray(v)[None, ...]  # add chain dim
        for k, v in prior_samples_np.items()
        if np.asarray(v).ndim >= 1
    }
    prior_pred_dict = {"accuracy": accuracies[None, ...]}
    idata = az.from_dict(
        prior=prior_dict,
        prior_predictive=prior_pred_dict,
    )
    nc_path = output_dir / f"{model}_prior_sim.nc"
    idata.to_netcdf(str(nc_path))

    gate_path = output_dir / f"{model}_gate.md"
    _write_gate_report(
        gate_path,
        model=model,
        metrics=metrics,
        passed=passed,
        num_draws=num_draws,
        seed=seed,
    )

    elapsed = time.monotonic() - t0
    print(f"\n>> Wrote:")
    print(f"   {nc_path}")
    print(f"   {csv_path}")
    print(f"   {gate_path}")
    print(f">> Total wall-clock: {elapsed:.1f}s")

    return 0 if passed else 1


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Prior-predictive gate (Baribault & Collins 2023) for RLWM "
            "choice-only hierarchical models."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(STACKED_MODEL_DISPATCH.keys()),
        help="Choice-only hierarchical model name.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("output/task_trials_long.csv"),
        help="Path to trial-level CSV (default: output/task_trials_long.csv).",
    )
    parser.add_argument(
        "--num-draws",
        type=int,
        default=500,
        help="Number of prior draws to simulate (default: 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="PRNG seed (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/bayesian/21_prior_predictive"),
        help=(
            "Output directory (default: output/bayesian/21_prior_predictive)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Entry point — returns 0 on PASS, 1 on FAIL."""
    args = _parse_args()
    # Silence the NumPyro ``numpyro.set_host_device_count`` warning if any
    numpyro.set_host_device_count(1)
    return _run_prior_predictive(
        model=args.model,
        data_path=args.data,
        num_draws=args.num_draws,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
