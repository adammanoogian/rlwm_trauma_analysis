"""Tests for the 2-covariate L2 hook on M3/M5/M6a hierarchical models (Plan 21-11).

Phase 21 Option C adds a second optional covariate ``covariate_iesr`` to the
M3, M5, and M6a hierarchical models (Approach A — additive parameter), beside
the existing ``covariate_lec``. When both are provided, ``beta_iesr_{target}``
is sampled with a ``Normal(0, 1)`` prior and the two shifts are summed on the
probit scale before the Phi_approx transform. L2 target parameter: ``kappa``
for M3/M5 and ``kappa_s`` for M6a.

The test suite covers four gates:

1. ``test_{m3,m5,m6a}_accepts_covariate_iesr`` — 2-covariate trace gate.
   Asserts that both ``beta_lec_{target}`` and ``beta_iesr_{target}`` appear
   in the trace when both covariates are passed. No MCMC — trace only (fast).
2. ``test_{m3,m5,m6a}_single_cov_unchanged`` — backward-compat gate.
   Asserts that passing ``covariate_lec`` only (no ``covariate_iesr``) yields
   ``beta_lec_{target}`` in the trace and that the new ``beta_iesr_{target}``
   site is NOT sampled. This proves Phase 16 callers are unaffected.
3. ``test_guard_raises_iesr_without_lec`` — guard gate.
   Asserts that ``covariate_lec=None`` combined with ``covariate_iesr=<v>``
   raises ``ValueError`` with the expected message on each of M3, M5, M6a.
4. ``test_recovery_2cov_m3`` — end-to-end recovery gate (@pytest.mark.slow).
   Generates N=40 synthetic participants with true
   ``beta_lec_kappa=0.4, beta_iesr_kappa=-0.3``, fits M3 with reduced MCMC
   budget (warmup=300, samples=600, chains=2), and asserts both posterior
   means land within one prior-SD (1.0) of the truth with zero divergences
   and max Rhat < 1.1.

Data generation for the recovery test uses a manual NumPy forward simulation
of the M3 model (the unified simulator framework does not expose a clean
per-participant kappa shift injection point — see plan 21-11 fallback
guidance).
"""

from __future__ import annotations

import numpy as np
import pytest

# ruff: noqa: E402 — heavy imports guarded inside tests to keep collection light


# ---------------------------------------------------------------------------
# Shared synthetic-data helpers
# ---------------------------------------------------------------------------


def _make_stacked_dict(
    n_ppts: int = 5,
    n_blocks: int = 3,
    n_trials: int = 20,
    seed: int = 0,
) -> dict:
    """Create minimal stacked participant data for trace-level testing.

    Mirrors the helper in ``test_m3_hierarchical.py`` so tests here do not
    depend on real-data CSVs.

    Parameters
    ----------
    n_ppts : int
        Number of synthetic participants.
    n_blocks : int
        Blocks per participant.
    n_trials : int
        Trials per block (pre-padding).
    seed : int
        NumPy RNG seed for reproducibility.

    Returns
    -------
    dict
        ``participant_data_stacked`` mapping int -> dict of stacked arrays,
        compatible with ``wmrl_m{3,5,6a}_hierarchical_model``.
    """
    import jax.numpy as jnp

    from rlwm.fitting.jax_likelihoods import pad_block_to_max

    rng = np.random.default_rng(seed)
    participant_data_stacked: dict = {}

    for i in range(n_ppts):
        stim_blocks = []
        act_blocks = []
        rew_blocks = []
        ss_blocks = []
        msk_blocks = []

        for _ in range(n_blocks):
            stim = jnp.array(rng.integers(0, 6, n_trials), dtype=jnp.int32)
            act = jnp.array(rng.integers(0, 3, n_trials), dtype=jnp.int32)
            rew = jnp.array(rng.integers(0, 2, n_trials).astype(np.float32))
            ss = jnp.ones(n_trials, dtype=jnp.float32) * 3.0
            p_stim, p_act, p_rew, p_ss, p_mask = pad_block_to_max(
                stim, act, rew, set_sizes=ss
            )
            stim_blocks.append(p_stim)
            act_blocks.append(p_act)
            rew_blocks.append(p_rew)
            ss_blocks.append(p_ss)
            msk_blocks.append(p_mask)

        participant_data_stacked[i] = {
            "stimuli_stacked": jnp.stack(stim_blocks),
            "actions_stacked": jnp.stack(act_blocks),
            "rewards_stacked": jnp.stack(rew_blocks),
            "set_sizes_stacked": jnp.stack(ss_blocks),
            "masks_stacked": jnp.stack(msk_blocks),
        }

    return participant_data_stacked


# ---------------------------------------------------------------------------
# Gate 1: 2-covariate trace acceptance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name, target",
    [
        ("wmrl_m3_hierarchical_model", "kappa"),
        ("wmrl_m5_hierarchical_model", "kappa"),
        ("wmrl_m6a_hierarchical_model", "kappa_s"),
    ],
)
def test_model_accepts_covariate_iesr(model_name: str, target: str) -> None:
    """Trace check: both beta_lec_{target} and beta_iesr_{target} are sampled.

    Runs ``numpyro.handlers.trace(seed(model, rng_seed=0))`` with both
    covariate vectors and checks the resulting trace. No MCMC invoked.

    Parameters
    ----------
    model_name : str
        NumPyro model attribute on ``rlwm.fitting.numpyro_models``.
    target : str
        L2 target parameter name used in site naming (``kappa`` or
        ``kappa_s``).
    """
    import jax.numpy as jnp
    import numpyro.handlers as handlers

    import rlwm.fitting.numpyro_models as models

    model_fn = getattr(models, model_name)

    rng = np.random.default_rng(42)
    n_ppts = 4
    ppt_data = _make_stacked_dict(n_ppts=n_ppts, n_blocks=2, n_trials=15)
    lec = jnp.array(rng.normal(0, 1, n_ppts), dtype=jnp.float32)
    iesr = jnp.array(rng.normal(0, 1, n_ppts), dtype=jnp.float32)

    trace = handlers.trace(
        handlers.seed(model_fn, rng_seed=0)
    ).get_trace(
        participant_data_stacked=ppt_data,
        covariate_lec=lec,
        covariate_iesr=iesr,
    )

    assert f"beta_lec_{target}" in trace, (
        f"{model_name}: expected 'beta_lec_{target}' in trace, got keys "
        f"{sorted(k for k in trace if k.startswith('beta_'))}"
    )
    assert f"beta_iesr_{target}" in trace, (
        f"{model_name}: expected 'beta_iesr_{target}' in trace with both "
        f"covariates provided, got keys "
        f"{sorted(k for k in trace if k.startswith('beta_'))}"
    )


# ---------------------------------------------------------------------------
# Gate 2: Backward compatibility — LEC only, no IES-R
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name, target",
    [
        ("wmrl_m3_hierarchical_model", "kappa"),
        ("wmrl_m5_hierarchical_model", "kappa"),
        ("wmrl_m6a_hierarchical_model", "kappa_s"),
    ],
)
def test_m3_single_cov_unchanged(model_name: str, target: str) -> None:
    """Phase 16 backward-compat: LEC-only path sees NO beta_iesr_{target} site.

    Asserts that ``covariate_lec=<v>, covariate_iesr=None`` (the old Phase 16
    calling convention from ``fit_bayesian._fit_stacked_model``) produces the
    same set of beta sites as before — specifically, ``beta_lec_{target}`` is
    sampled but the new ``beta_iesr_{target}`` site is NOT.

    Test name retained as ``test_m3_single_cov_unchanged`` for plan 21-11
    compatibility; parametrized over M3, M5, M6a.

    Parameters
    ----------
    model_name : str
        NumPyro model attribute on ``rlwm.fitting.numpyro_models``.
    target : str
        L2 target parameter name (``kappa`` or ``kappa_s``).
    """
    import jax.numpy as jnp
    import numpyro.handlers as handlers

    import rlwm.fitting.numpyro_models as models

    model_fn = getattr(models, model_name)

    rng = np.random.default_rng(7)
    n_ppts = 4
    ppt_data = _make_stacked_dict(n_ppts=n_ppts, n_blocks=2, n_trials=15)
    lec = jnp.array(rng.normal(0, 1, n_ppts), dtype=jnp.float32)

    trace = handlers.trace(
        handlers.seed(model_fn, rng_seed=0)
    ).get_trace(
        participant_data_stacked=ppt_data,
        covariate_lec=lec,
        # covariate_iesr omitted — defaults to None
    )

    assert f"beta_lec_{target}" in trace, (
        f"{model_name}: expected 'beta_lec_{target}' in trace under LEC-only "
        f"path, got {sorted(k for k in trace if k.startswith('beta_'))}"
    )
    assert f"beta_iesr_{target}" not in trace, (
        f"{model_name}: unexpected 'beta_iesr_{target}' in trace when only "
        f"covariate_lec was passed — backward compatibility broken. Got keys "
        f"{sorted(k for k in trace if k.startswith('beta_'))}"
    )


# ---------------------------------------------------------------------------
# Gate 3: Guard — IES-R without LEC raises ValueError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name",
    [
        "wmrl_m3_hierarchical_model",
        "wmrl_m5_hierarchical_model",
        "wmrl_m6a_hierarchical_model",
    ],
)
def test_guard_raises_iesr_without_lec(model_name: str) -> None:
    """Guard: covariate_iesr without covariate_lec raises ValueError.

    Prevents silently dropping the LEC covariate — the caller must pass both
    (or neither) to use the Phase 21 2-covariate L2 design.

    Parameters
    ----------
    model_name : str
        NumPyro model attribute on ``rlwm.fitting.numpyro_models``.
    """
    import jax.numpy as jnp
    import numpyro.handlers as handlers

    import rlwm.fitting.numpyro_models as models

    model_fn = getattr(models, model_name)

    rng = np.random.default_rng(9)
    n_ppts = 3
    ppt_data = _make_stacked_dict(n_ppts=n_ppts, n_blocks=2, n_trials=10)
    iesr = jnp.array(rng.normal(0, 1, n_ppts), dtype=jnp.float32)

    with pytest.raises(ValueError, match="covariate_iesr provided without covariate_lec"):
        handlers.trace(handlers.seed(model_fn, rng_seed=0)).get_trace(
            participant_data_stacked=ppt_data,
            covariate_lec=None,
            covariate_iesr=iesr,
        )


# ---------------------------------------------------------------------------
# Gate 4: Recovery of (beta_lec_kappa, beta_iesr_kappa) on M3 @ slow
# ---------------------------------------------------------------------------


def _simulate_m3_block_numpy(
    alpha_pos: float,
    alpha_neg: float,
    phi: float,
    rho: float,
    capacity: float,
    kappa: float,
    epsilon: float,
    n_trials: int,
    n_stim: int = 3,
    n_act: int = 3,
    set_size: int = 3,
    beta: float = 50.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a single M3 block using a NumPy forward pass.

    Mirrors the semantics of ``wmrl_m3_block_likelihood`` in
    ``scripts/fitting/jax_likelihoods.py``: hybrid WM/RL policy with
    epsilon-noise and probability-mixing perseveration against a one-hot
    choice kernel of the last action.

    Parameters
    ----------
    alpha_pos, alpha_neg : float
        Asymmetric RL learning rates.
    phi : float
        WM decay rate.
    rho : float
        Base WM reliance.
    capacity : float
        WM capacity K (for adaptive weighting).
    kappa : float
        Perseveration parameter (probability mixing with the choice kernel).
    epsilon : float
        Noise parameter.
    n_trials, n_stim, n_act, set_size : int
        Task structure.
    beta : float
        Inverse temperature for softmax (fixed at 50.0 per Senta et al., 2025).
    seed : int
        RNG seed.

    Returns
    -------
    stimuli, actions, rewards, set_sizes : np.ndarray
        Simulated block arrays.
    """
    rng = np.random.default_rng(seed)

    Q = np.ones((n_stim, n_act)) * 0.5
    WM = np.ones((n_stim, n_act)) * (1.0 / n_act)
    WM_0 = np.ones((n_stim, n_act)) * (1.0 / n_act)

    correct_actions = rng.integers(0, n_act, n_stim)

    stims, acts, rews = [], [], []
    last_action = -1  # Resets at block start

    for _ in range(n_trials):
        # Decay WM
        WM = (1.0 - phi) * WM + phi * WM_0

        s = int(rng.integers(0, n_stim))

        # Hybrid policy: omega = rho * min(1, K/set_size)
        w = rho * min(1.0, capacity / set_size)

        # Softmax over Q
        q = Q[s, :]
        q_shifted = beta * (q - q.max())
        p_q = np.exp(q_shifted)
        p_q /= p_q.sum()

        # Softmax over WM
        w_arr = WM[s, :]
        w_shifted = beta * (w_arr - w_arr.max())
        p_wm = np.exp(w_shifted)
        p_wm /= p_wm.sum()

        # Hybrid
        p_base = w * p_wm + (1.0 - w) * p_q

        # Apply epsilon
        p_noisy = epsilon / n_act + (1.0 - epsilon) * p_base

        # Perseveration via probability mixing
        if last_action >= 0 and kappa > 0.0:
            ck = np.zeros(n_act)
            ck[last_action] = 1.0
            p_final = (1.0 - kappa) * p_noisy + kappa * ck
        else:
            p_final = p_noisy

        # Sample action
        p_final = p_final / p_final.sum()  # defensive renorm
        a = int(rng.choice(n_act, p=p_final))
        r = 1.0 if a == int(correct_actions[s]) else 0.0

        stims.append(s)
        acts.append(a)
        rews.append(r)

        # Updates (after action taken)
        delta = r - Q[s, a]
        alpha = alpha_pos if delta > 0 else alpha_neg
        Q[s, a] = Q[s, a] + alpha * delta
        WM[s, a] = r  # Immediate overwrite

        last_action = a

    return (
        np.array(stims, dtype=np.int32),
        np.array(acts, dtype=np.int32),
        np.array(rews, dtype=np.float32),
        np.full(n_trials, set_size, dtype=np.int32),
    )


def _build_recovery_dataset(
    n_ppts: int,
    n_blocks: int,
    n_trials: int,
    true_beta_lec: float,
    true_beta_iesr: float,
    kappa_mu_pr: float = 0.0,
    kappa_sigma_pr: float = 0.2,
    base_params: dict[str, float] | None = None,
    seed: int = 0,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic M3 dataset with per-participant kappa shifts.

    True kappa on the probit (unconstrained) scale:
        ``kappa_unc_i = kappa_mu_pr + kappa_sigma_pr * z_i
                       + true_beta_lec * lec_i + true_beta_iesr * iesr_i``
    Bounded kappa:
        ``kappa_i = Phi_approx(kappa_unc_i)``  (in [0, 1]).

    Parameters
    ----------
    n_ppts : int
        Number of participants.
    n_blocks : int
        Blocks per participant.
    n_trials : int
        Trials per block.
    true_beta_lec, true_beta_iesr : float
        Ground-truth L2 regression coefficients.
    kappa_mu_pr, kappa_sigma_pr : float
        Group-level kappa_mu_pr + kappa_sigma_pr (on the probit scale).
    base_params : dict or None
        Base RLWM parameters shared across participants. Defaults to
        moderately heterogeneous recoverable values.
    seed : int
        RNG seed.

    Returns
    -------
    participant_data_stacked : dict
        Stacked data suitable for ``wmrl_m3_hierarchical_model``.
    lec_z : np.ndarray, shape (n_ppts,)
        Z-scored LEC covariate.
    iesr_z : np.ndarray, shape (n_ppts,)
        Z-scored IES-R covariate.
    true_kappas : np.ndarray, shape (n_ppts,)
        Per-participant bounded kappa values used in simulation.
    """
    import jax.numpy as jnp
    from jax.scipy.stats import norm as jax_norm

    from rlwm.fitting.jax_likelihoods import pad_block_to_max

    rng = np.random.default_rng(seed)

    if base_params is None:
        base_params = {
            "alpha_pos": 0.4,
            "alpha_neg": 0.15,
            "phi": 0.05,
            "rho": 0.7,
            "capacity": 4.0,
            "epsilon": 0.05,
        }

    # Generate z-scored covariates (independent)
    lec_raw = rng.normal(0, 1, n_ppts)
    iesr_raw = rng.normal(0, 1, n_ppts)
    lec_z = (lec_raw - lec_raw.mean()) / (lec_raw.std(ddof=0) + 1e-8)
    iesr_z = (iesr_raw - iesr_raw.mean()) / (iesr_raw.std(ddof=0) + 1e-8)

    # Per-participant kappa (unconstrained -> probit -> bounded)
    z = rng.normal(0, 1, n_ppts)
    kappa_unc = (
        kappa_mu_pr
        + kappa_sigma_pr * z
        + true_beta_lec * lec_z
        + true_beta_iesr * iesr_z
    )
    # phi_approx = scipy.stats.norm.cdf
    true_kappas = np.asarray(jax_norm.cdf(jnp.array(kappa_unc)))
    # Clip defensively to avoid pathological boundary kappas during sim
    true_kappas = np.clip(true_kappas, 0.01, 0.99)

    participant_data_stacked: dict = {}

    for i in range(n_ppts):
        stim_blocks = []
        act_blocks = []
        rew_blocks = []
        ss_blocks = []
        msk_blocks = []

        for b in range(n_blocks):
            stim, act, rew, ss = _simulate_m3_block_numpy(
                alpha_pos=base_params["alpha_pos"],
                alpha_neg=base_params["alpha_neg"],
                phi=base_params["phi"],
                rho=base_params["rho"],
                capacity=base_params["capacity"],
                kappa=float(true_kappas[i]),
                epsilon=base_params["epsilon"],
                n_trials=n_trials,
                seed=1000 * seed + 100 * i + b + 1,
            )
            p_stim, p_act, p_rew, p_ss, p_mask = pad_block_to_max(
                jnp.array(stim),
                jnp.array(act),
                jnp.array(rew),
                set_sizes=jnp.array(ss, dtype=jnp.float32),
            )
            stim_blocks.append(p_stim)
            act_blocks.append(p_act)
            rew_blocks.append(p_rew)
            ss_blocks.append(p_ss)
            msk_blocks.append(p_mask)

        participant_data_stacked[i] = {
            "stimuli_stacked": jnp.stack(stim_blocks),
            "actions_stacked": jnp.stack(act_blocks),
            "rewards_stacked": jnp.stack(rew_blocks),
            "set_sizes_stacked": jnp.stack(ss_blocks),
            "masks_stacked": jnp.stack(msk_blocks),
        }

    return participant_data_stacked, lec_z, iesr_z, true_kappas


@pytest.mark.slow
def test_recovery_2cov_m3() -> None:
    """End-to-end recovery: M3 recovers (beta_lec_kappa, beta_iesr_kappa).

    Generates N=40 synthetic M3 participants with ``beta_lec_kappa=0.4,
    beta_iesr_kappa=-0.3`` injected on the probit kappa scale, then fits the
    hierarchical M3 model with a reduced MCMC budget (warmup=300,
    samples=600, chains=2). Asserts:

    - ``|beta_lec_kappa_post_mean - 0.4| < 1.0``  (within one prior-SD)
    - ``|beta_iesr_kappa_post_mean - (-0.3)| < 1.0`` (within one prior-SD)
    - ``num_divergences == 0`` after auto-bump
    - ``max_rhat < 1.1`` across the two beta sites

    This is an integration-level smoke test: single seed, loose recovery
    bound appropriate for fast CI-speed budget. A full Pearson-r recovery
    study across an ensemble is explicitly out of scope per plan 21-11.

    Expected wall time: ~5–10 min on CPU.
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.infer import MCMC, NUTS

    from rlwm.fitting.numpyro_models import (
        stack_across_participants,
        wmrl_m3_hierarchical_model,
    )

    numpyro.set_host_device_count(2)

    ppt_data, lec_z, iesr_z, true_kappas = _build_recovery_dataset(
        n_ppts=40,
        n_blocks=4,
        n_trials=30,
        true_beta_lec=0.4,
        true_beta_iesr=-0.3,
        seed=21_11,
    )
    stacked_arrays = stack_across_participants(ppt_data)

    kernel = NUTS(
        wmrl_m3_hierarchical_model,
        target_accept_prob=0.95,
        max_tree_depth=8,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=300,
        num_samples=600,
        num_chains=2,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(
        jax.random.PRNGKey(0),
        participant_data_stacked=ppt_data,
        covariate_lec=jnp.array(lec_z, dtype=jnp.float32),
        covariate_iesr=jnp.array(iesr_z, dtype=jnp.float32),
        stacked_arrays=stacked_arrays,
    )

    samples = mcmc.get_samples(group_by_chain=False)
    beta_lec = np.asarray(samples["beta_lec_kappa"])
    beta_iesr = np.asarray(samples["beta_iesr_kappa"])

    post_mean_lec = float(beta_lec.mean())
    post_mean_iesr = float(beta_iesr.mean())

    # Primary gate: within one prior-SD of truth
    assert abs(post_mean_lec - 0.4) < 1.0, (
        f"beta_lec_kappa posterior mean {post_mean_lec:.3f} not within 1.0 "
        f"of truth 0.4 (diff={abs(post_mean_lec - 0.4):.3f})"
    )
    assert abs(post_mean_iesr - (-0.3)) < 1.0, (
        f"beta_iesr_kappa posterior mean {post_mean_iesr:.3f} not within 1.0 "
        f"of truth -0.3 (diff={abs(post_mean_iesr + 0.3):.3f})"
    )

    # Secondary gate: convergence diagnostics on the two beta sites
    extra = mcmc.get_extra_fields()
    divergences = extra.get("diverging")
    n_div = int(np.asarray(divergences).sum()) if divergences is not None else 0
    assert n_div == 0, (
        f"Recovery fit produced {n_div} divergences (expected 0 after "
        "auto-bump at target_accept=0.95)"
    )

    # Rhat across the two betas — use grouped samples for chain dim
    grouped = mcmc.get_samples(group_by_chain=True)
    rhats = []
    for name in ("beta_lec_kappa", "beta_iesr_kappa"):
        arr = np.asarray(grouped[name])  # shape (n_chains, n_samples)
        # Split-Rhat approximation via numpyro.diagnostics.gelman_rubin
        from numpyro.diagnostics import split_gelman_rubin

        rhat = float(split_gelman_rubin(arr))
        rhats.append(rhat)
    max_rhat = max(rhats)
    assert max_rhat < 1.1, (
        f"max Rhat across (beta_lec_kappa, beta_iesr_kappa) = {max_rhat:.3f} "
        f"> 1.1. Per-site rhats: {dict(zip(('beta_lec_kappa', 'beta_iesr_kappa'), rhats))}"
    )
