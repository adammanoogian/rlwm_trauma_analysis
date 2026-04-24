"""Canonical single-source PPC simulator for prior and posterior predictive checks.

This module holds the one authoritative simulator used across the RLWM
pipeline. It supersedes the former per-stage copies — every orchestrator
now imports from here.

Layers (do not conflate)
------------------------
* This module (``scripts.utils.ppc``) is the *draws-level* simulator. Given
  a batch of posterior (or prior) draws from a hierarchical NumPyro model,
  it simulates per-participant trial sequences and aggregates behavioral
  metrics. Used by:

  - stage 03 prior PPC  → ``scripts/03_model_prefitting/04_run_prior_predictive.py``
  - stage 05 posterior PPC thin orchestrator
    → ``scripts/05_post_fitting_checks/03_run_posterior_ppc.py``

  (The former ``scripts/03_model_prefitting/09_run_ppc.py`` duplicate
  orchestrator was removed in plan 29-04b — stage 05 is now the sole
  entry point for posterior PPC per Scheme D.)

* For *single-agent* trajectory simulation given fixed parameters (used by
  the test-suite and parameter-sweep tooling), see
  ``scripts/simulations/unified_simulator.py``
  (``simulate_agent_fixed`` / ``simulate_agent_sampled``). That is a
  deliberately separate, lower-level layer that consumes the Gym env.

Public API
----------
* :func:`simulate_from_samples` — dispatch the per-trial simulator for a
  given (model, params) on a real participant's trial template.
* :func:`run_prior_ppc` — run the Baribault & Collins (2023) prior-
  predictive gate for a choice-only hierarchical model. Writes ArviZ
  NetCDF, per-draw CSV, and PASS/FAIL gate report.
* :func:`run_posterior_ppc` — run the posterior-predictive check
  pipeline (behavioral comparison + model recovery) for a fitted model.

References
----------
Baribault, B. & Collins, A. G. E. (2023). Troubleshooting Bayesian
cognitive models. *Psychological Methods*.
https://doi.org/10.1037/met0000554

Hess, B. et al. (2025). A robust Bayesian workflow for computational
psychiatry. *Computational Psychiatry*, 9(1):76-99.
https://doi.org/10.5334/cpsy.116

Wilson, R. C. & Collins, A. G. E. (2019). Ten simple rules for the
computational modeling of behavioral data. *eLife*, 8:e49547.

Senta et al. (2025). (Pre-registered RLWM reference implementation).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Senta et al. (2025) canonical constants
FIXED_BETA: float = 50.0
NUM_ACTIONS: int = 3
Q_INIT: float = 0.5
WM_INIT: float = 1.0 / NUM_ACTIONS

# Working stimulus capacity for the simulator. Real data CSVs store stimuli
# as 1-indexed ints in [1, 6] (see ``scripts/01_data_preprocessing/01_parse_raw_data.py``
# notes); the fit_bayesian JAX likelihoods silently clip with
# ``num_stimuli=6`` by relying on JAX's out-of-bounds edge-clip semantics,
# which is safe but mirrors as ``num_stimuli=7`` in a plain-NumPy simulator
# so that index 6 maps to its own row rather than aliasing onto index 5.
NUM_STIMULI: int = 7


# ---------------------------------------------------------------------------
# Per-trial policy helpers (NumPy) — mirror jax_likelihoods.*_block_likelihood
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray, beta: float) -> np.ndarray:
    """Compute a numerically stable softmax with inverse-temperature ``beta``.

    Parameters
    ----------
    x : np.ndarray
        Logits/action values for a single stimulus, shape ``(n_actions,)``.
    beta : float
        Inverse temperature; Senta et al. (2025) use ``FIXED_BETA = 50.0``.

    Returns
    -------
    np.ndarray
        Softmax probabilities, shape ``(n_actions,)``, summing to 1.
    """
    z = beta * (x - x.max())
    ez = np.exp(z)
    return ez / ez.sum()


def _apply_epsilon(probs: np.ndarray, epsilon: float, n_act: int) -> np.ndarray:
    """Apply hBayesDM-style uniform-mixing epsilon noise.

    Parameters
    ----------
    probs : np.ndarray
        Softmax action probabilities, shape ``(n_act,)``.
    epsilon : float
        Lapse rate in [0, 1].
    n_act : int
        Number of actions.

    Returns
    -------
    np.ndarray
        Mixed probabilities ``epsilon/n_act + (1-epsilon)*probs``.
    """
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
    """Simulate M1 (Q-learning) actions and rewards given a real trial template.

    Parameters
    ----------
    stimuli_blocks : list[np.ndarray]
        One array of stimulus indices per block.
    set_sizes_blocks : list[np.ndarray]
        One array of set sizes per block (unused by M1, kept for API symmetry).
    rng : np.random.Generator
        NumPy PRNG.
    alpha_pos, alpha_neg : float
        Asymmetric learning rates.
    epsilon : float
        Lapse rate.

    Returns
    -------
    actions_blocks, rewards_blocks : tuple[list[np.ndarray], list[np.ndarray]]
        One array per block of simulated actions and reinforcements.
    """
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
    """Simulate the WM-RL family (M2, M3, M5, M6a, M6b) actions and rewards.

    Dispatch by ``model`` — each variant toggles a different set of kernel
    and decay terms from ``{kappa, kappa_s, phi_rl}``.

    Parameters
    ----------
    stimuli_blocks, set_sizes_blocks : list[np.ndarray]
        Per-block stimulus indices and set sizes from a real participant.
    rng : np.random.Generator
    model : str
        One of ``{"wmrl", "wmrl_m3", "wmrl_m5", "wmrl_m6a", "wmrl_m6b"}``.
    alpha_pos, alpha_neg : float
        Asymmetric learning rates.
    phi, rho, capacity : float
        WM-RL hybrid parameters (decay, mix, capacity K).
    epsilon : float
        Lapse rate.
    kappa, kappa_s, phi_rl : float
        Variant-specific extras: global perseveration, stimulus-specific
        perseveration, RL forgetting.

    Returns
    -------
    actions_blocks, rewards_blocks : tuple[list[np.ndarray], list[np.ndarray]]
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


def simulate_from_samples(
    model: str,
    params: dict[str, float],
    stimuli_blocks: list[np.ndarray],
    set_sizes_blocks: list[np.ndarray],
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Dispatch the per-trial simulator for a single (model, params) draw.

    Thin wrapper that routes to :func:`_simulate_qlearning` for M1 and to
    :func:`_simulate_wmrl_family` for every WM-RL variant, using a common
    keyword-argument interface.

    Parameters
    ----------
    model : str
        Model identifier from :data:`config.ALL_MODELS`.
    params : dict[str, float]
        Parameter dictionary (uses keys ``alpha_pos``, ``alpha_neg``,
        ``epsilon``; plus ``phi``, ``rho``, ``capacity``, ``kappa``,
        ``kappa_s``, ``phi_rl`` for WM-RL variants).
    stimuli_blocks, set_sizes_blocks : list[np.ndarray]
        Per-block real trial template (stimuli, set sizes).
    rng : np.random.Generator

    Returns
    -------
    actions_blocks, rewards_blocks : tuple[list[np.ndarray], list[np.ndarray]]
    """
    if model == "qlearning":
        return _simulate_qlearning(
            stimuli_blocks,
            set_sizes_blocks,
            rng,
            alpha_pos=params["alpha_pos"],
            alpha_neg=params["alpha_neg"],
            epsilon=params["epsilon"],
        )
    return _simulate_wmrl_family(
        stimuli_blocks,
        set_sizes_blocks,
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


# ---------------------------------------------------------------------------
# Draw extraction — pluck one (participant, draw) param vector
# ---------------------------------------------------------------------------


def _extract_param_vector(
    prior_samples: dict[str, np.ndarray],
    model: str,
    draw_idx: int,
    ppt_idx: int,
) -> dict[str, float]:
    """Build a param dict for a single ``(draw, participant)`` combination.

    Handles the M6b stick-breaking decode: the model samples ``kappa_total``
    and ``kappa_share`` as deterministics, so we rebuild
    ``kappa = kappa_total * kappa_share`` and
    ``kappa_s = kappa_total * (1 - kappa_share)`` to feed the simulator.

    Parameters
    ----------
    prior_samples : dict[str, np.ndarray]
        Output of :class:`numpyro.infer.Predictive`. Each array is shaped
        ``(num_draws, n_participants)``.
    model : str
        Model identifier; used to look up the parameter list in
        :data:`config.MODEL_REGISTRY`.
    draw_idx, ppt_idx : int
        Draw and participant indices to pluck.

    Returns
    -------
    dict[str, float]
        Parameter dictionary ready to feed :func:`simulate_from_samples`.
    """
    from config import MODEL_REGISTRY

    out: dict[str, float] = {}
    params = MODEL_REGISTRY[model]["params"]

    # Standard bounded params are all registered as deterministic sites
    # named after the parameter. Shape: (num_draws, n_participants).
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
# Trial-template helpers
# ---------------------------------------------------------------------------


def _unstack_participant_template(
    stacked: dict,
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    """Reconstruct per-participant lists of ``(stimuli, set_sizes)`` blocks.

    Uses ``masks`` to drop padded trials and ``n_blocks_per_ppt`` to drop
    padded blocks.

    Parameters
    ----------
    stacked : dict
        Output of
        ``rlwm.fitting.numpyro_models.stack_across_participants``.

    Returns
    -------
    stim_per_ppt, ss_per_ppt : tuple[list[list[np.ndarray]], list[list[np.ndarray]]]
        One list-of-blocks per participant, matching the sorted participant
        order in ``stacked["participant_ids"]``.
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


# ---------------------------------------------------------------------------
# Baribault & Collins (2023) prior-predictive gate
# ---------------------------------------------------------------------------


def _evaluate_gate(accuracies: np.ndarray) -> tuple[bool, dict[str, float]]:
    """Apply the three-part Baribault & Collins (2023) gate.

    Parameters
    ----------
    accuracies : np.ndarray
        Per-draw mean accuracies, shape ``(num_draws,)``.

    Returns
    -------
    passed : bool
        True iff all three sub-gates pass.
    metrics : dict[str, float]
        Holds ``median``, ``frac_below_chance``, ``frac_at_ceiling``, and
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
    """Write the ``{model}_gate.md`` markdown report.

    Parameters
    ----------
    path : Path
        Output file path for the markdown report.
    model : str
        Model identifier.
    metrics : dict[str, float]
        Output of :func:`_evaluate_gate`.
    passed : bool
        Gate verdict.
    num_draws : int
        Number of prior draws simulated.
    seed : int
        PRNG seed used.
    """
    from rlwm.fitting.numpyro_helpers import PARAM_PRIOR_DEFAULTS

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
        "| Check | Threshold | Value | Pass |",
        "|---|---|---|---|",
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


def run_prior_ppc(
    model: str,
    data_path: Path,
    num_draws: int,
    seed: int,
    output_dir: Path,
) -> int:
    """Run the Baribault & Collins (2023) prior-predictive gate for one model.

    For the named choice-only hierarchical model, this function:

    1. Loads the canonical v4.0 cohort (N=138) via
       ``scripts.fitting.fit_bayesian.load_and_prepare_data``.
    2. Builds the stacked-participant arrays consumed by the hierarchical
       model (``prepare_stacked_participant_data`` +
       ``stack_across_participants``).
    3. Draws ``num_draws`` samples from the *prior* of the hierarchical
       model via ``numpyro.infer.Predictive`` with no observations
       conditioned (``PARAM_PRIOR_DEFAULTS`` is the currently locked v4.0
       prior set with all ``mu_prior_loc=0.0`` except ``epsilon=-2.0``).
    4. For each draw, simulates one participant's trial sequence under
       that draw's individual-level parameters using
       :func:`simulate_from_samples`, mirroring the likelihood update
       equations in ``rlwm.fitting.jax_likelihoods``.
    5. Applies the three-part Baribault & Collins (2023) gate: median
       accuracy in [0.4, 0.9], < 10% draws sub-chance, < 5% draws
       at-ceiling.
    6. Writes ``{model}_prior_sim.nc`` (ArviZ NetCDF),
       ``{model}_prior_accuracy.csv`` (per-draw table), and
       ``{model}_gate.md`` (PASS/FAIL verdict).

    Parameters
    ----------
    model : str
        One of the choice-only model identifiers in
        ``STACKED_MODEL_DISPATCH``.
    data_path : Path
        Trial-level CSV (default: ``data/processed/task_trials_long.csv``).
    num_draws : int
        Number of prior draws to simulate.
    seed : int
        PRNG seed.
    output_dir : Path
        Output directory for the three artifacts.

    Returns
    -------
    int
        Exit code: 0 on PASS, 1 on FAIL (so a pipeline orchestrator can
        short-circuit).
    """
    # Heavy imports done lazily to keep import-time cheap and allow tests
    # that import scripts.utils.ppc but never call run_prior_ppc to avoid
    # pulling in NumPyro/JAX/ArviZ.
    import arviz as az
    import jax
    from numpyro.infer import Predictive

    from rlwm.fitting.bayesian import (
        STACKED_MODEL_DISPATCH,
        load_and_prepare_data,
    )
    from rlwm.fitting.numpyro_helpers import PARAM_PRIOR_DEFAULTS
    from rlwm.fitting.core import (
        prepare_stacked_participant_data,
        stack_across_participants,
    )

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
    print("Priors in use (PARAM_PRIOR_DEFAULTS):")
    for k, v in sorted(PARAM_PRIOR_DEFAULTS.items()):
        print(f"  {k}: {v}")
    print()

    # --- 1. Load data (canonical cohort) ---------------------------------
    df = load_and_prepare_data(data_path, use_cohort=True)

    # --- 2. Build stacked arrays ----------------------------------------
    ppt_data = prepare_stacked_participant_data(df)
    stacked = stack_across_participants(ppt_data)
    n_participants = len(stacked["participant_ids"])
    print(f"\n>> Stacked arrays built: N={n_participants} participants")

    # --- 3. Sample prior via Predictive ---------------------------------
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

    # --- 4. Simulate — one (draw, participant) pair per draw ------------
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

        _, r_blocks = simulate_from_samples(
            model,
            params,
            stim_blocks,
            ss_blocks,
            rng,
        )

        all_rewards = np.concatenate(r_blocks)
        accuracies[d] = float(all_rewards.mean()) if len(all_rewards) else 0.0

        if (d + 1) % max(1, num_draws // 10) == 0:
            print(
                f"   {d + 1}/{num_draws} draws, running median acc = "
                f"{float(np.median(accuracies[: d + 1])):.3f}"
            )

    # --- 5. Gate evaluation ---------------------------------------------
    passed, metrics = _evaluate_gate(accuracies)
    print("\n>> Gate metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")
    print(f">> Verdict: {'PASS' if passed else 'FAIL'}")

    # --- 6. Write artifacts ---------------------------------------------
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

    # NetCDF via arviz.from_dict — wrap prior samples (filter out
    # pscan/obs sites)
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
    print("\n>> Wrote:")
    print(f"   {nc_path}")
    print(f"   {csv_path}")
    print(f"   {gate_path}")
    print(f">> Total wall-clock: {elapsed:.1f}s")

    return 0 if passed else 1


# ---------------------------------------------------------------------------
# Posterior-predictive check pipeline
# ---------------------------------------------------------------------------


def run_posterior_ppc(
    model: str,
    fitted_params_path: str,
    real_data_path: str,
    output_dir: Path,
    figures_dir: Path,
    skip_model_recovery: bool = False,
    use_gpu: bool = False,
    n_jobs: int = 1,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the full posterior-predictive check pipeline for a single model.

    The pipeline has two halves:

    1. **Behavioral comparison** — simulate trial-level data under the
       fitted point estimates, and compare real-vs-synthetic accuracy,
       learning curves, set-size effects, etc. Implemented by
       :func:`scripts.fitting.model_recovery.run_posterior_predictive_check`
       (which in turn calls :func:`simulate_from_samples` indirectly via
       its generative code path).

    2. **Model recovery** — refit *all* models (the full M1–M6 roster)
       to the synthetic data from step 1 and check that the generative
       model wins by AIC. Implemented by
       :func:`scripts.fitting.model_recovery.run_model_recovery_check`.

    Parameters
    ----------
    model : str
        Model identifier.
    fitted_params_path : str
        Path to ``models/mle/{model}_individual_fits.csv``.
    real_data_path : str
        Path to real trial-level CSV.
    output_dir : Path
        Directory for PPC CSVs (synthetic trials, behavioral comparison,
        MLE recovery fits).
    figures_dir : Path
        Directory for comparison plots.
    skip_model_recovery : bool, default False
        If True, only run the behavioral comparison (faster).
    use_gpu : bool, default False
        Pass ``--use-gpu`` through to the model-recovery MLE fits.
    n_jobs : int, default 1
        Parallel jobs for CPU model-recovery fits.
    verbose : bool, default True
        If False, suppress progress output.

    Returns
    -------
    dict[str, Any]
        Keys: ``model``, ``behavioral_comparison`` (DataFrame),
        ``model_recovery`` (dict or None).

    References
    ----------
    Wilson & Collins (2019), Palminteri et al. (2017), Senta et al.
    (2025).
    """
    # Heavy imports done lazily so cheap callers / tests can import
    # scripts.utils.ppc without paying JAX startup.
    from scripts.fitting.model_recovery import (
        run_model_recovery_check,
        run_posterior_predictive_check,
    )

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"POSTERIOR PREDICTIVE CHECK: {model.upper()}")
        print(f"{'=' * 60}")

    # 1. Generate synthetic data and compare behavior
    comparison_df = run_posterior_predictive_check(
        model=model,
        fitted_params_path=fitted_params_path,
        real_data_path=real_data_path,
        output_dir=output_dir,
        figures_dir=figures_dir,
        verbose=verbose,
    )

    # 2. Model recovery (unless skipped)
    model_recovery_result = None
    if not skip_model_recovery:
        synthetic_data_path = output_dir / "synthetic_trials.csv"
        model_recovery_result = run_model_recovery_check(
            synthetic_data_path=str(synthetic_data_path),
            generative_model=model,
            output_dir=output_dir,
            use_gpu=use_gpu,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    # 3. Summary
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"PPC SUMMARY: {model.upper()}")
        print(f"{'=' * 60}")
        print("\nBehavioral Comparison:")
        print(comparison_df.to_string(index=False))

        if model_recovery_result:
            status = "PASS" if model_recovery_result["generative_wins"] else "FAIL"
            print(f"\nModel Recovery: {status}")
            print(f"  Generative: {model_recovery_result['generative_model']}")
            print(f"  Winner:     {model_recovery_result['winning_model']}")

        print(f"{'=' * 60}\n")

    return {
        "model": model,
        "behavioral_comparison": comparison_df,
        "model_recovery": model_recovery_result,
    }


__all__ = [
    "FIXED_BETA",
    "NUM_ACTIONS",
    "NUM_STIMULI",
    "Q_INIT",
    "WM_INIT",
    "simulate_from_samples",
    "run_prior_ppc",
    "run_posterior_ppc",
]


# Quieten unused-import warnings for ``sys`` — kept for future CLI use.
_ = sys
