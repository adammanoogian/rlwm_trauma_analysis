"""Unit tests for scripts/fitting/numpyro_helpers.py.

Covers:
- phi_approx known values and bounds
- sample_bounded_param range correctness
- sample_capacity range correctness
- MCMC parameter recovery (marked slow)
- PARAM_PRIOR_DEFAULTS completeness
- load_fits_with_validation error handling
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import pytest

from config import EXPECTED_PARAMETERIZATION, MODEL_REGISTRY, load_fits_with_validation
from scripts.fitting.numpyro_helpers import (
    PARAM_PRIOR_DEFAULTS,
    phi_approx,
    sample_bounded_param,
    sample_capacity,
    sample_model_params,
)


# ---------------------------------------------------------------------------
# phi_approx tests
# ---------------------------------------------------------------------------


def test_phi_approx_known_values():
    """Phi_approx matches standard normal CDF at key points."""
    assert jnp.allclose(phi_approx(jnp.array(0.0)), jnp.array(0.5), atol=1e-3)
    assert jnp.allclose(phi_approx(jnp.array(-3.0)), jnp.array(0.00135), atol=1e-3)
    assert jnp.allclose(phi_approx(jnp.array(3.0)), jnp.array(0.99865), atol=1e-3)


def test_phi_approx_bounds():
    """Phi_approx stays strictly inside (0, 1) within float32 dynamic range.

    float32 saturates at approximately ±6 sigma; use ±5 which remains
    non-degenerate (phi_approx(-5) ~ 2.9e-7, phi_approx(5) ~ 1 - 3e-7).
    """
    assert float(phi_approx(jnp.array(-5.0))) > 0.0
    assert float(phi_approx(jnp.array(5.0))) < 1.0


# ---------------------------------------------------------------------------
# sample_bounded_param range tests
# ---------------------------------------------------------------------------


def test_bounded_param_range():
    """All sampled alpha values lie in [0, 1]."""

    def model():
        sample_bounded_param(
            "test",
            lower=0.0,
            upper=1.0,
            n_participants=100,
        )

    with numpyro.handlers.seed(rng_seed=0):
        trace = numpyro.handlers.trace(model).get_trace()

    values = trace["test"]["value"]
    assert jnp.all(values >= 0.0), f"min={float(values.min())}"
    assert jnp.all(values <= 1.0), f"max={float(values.max())}"


def test_capacity_range():
    """All sampled K values lie in [2, 6]."""

    def model():
        sample_capacity(n_participants=100)

    with numpyro.handlers.seed(rng_seed=1):
        trace = numpyro.handlers.trace(model).get_trace()

    values = trace["capacity"]["value"]
    assert jnp.all(values >= 2.0), f"min={float(values.min())}"
    assert jnp.all(values <= 6.0), f"max={float(values.max())}"


# ---------------------------------------------------------------------------
# MCMC parameter recovery tests (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_bounded_param_recovery_alpha():
    """Recover alpha_pos group mean within 5% relative error under MCMC.

    Ground truth: mu_pr=0.0, sigma_pr=0.1 (tight population).
    Observation: Normal(ground_truth, 0.01).
    """
    from numpyro.infer import MCMC, NUTS

    rng = np.random.default_rng(42)
    n_participants = 30
    true_mu_pr = 0.0
    true_sigma_pr = 0.1

    # Generate synthetic individual-level alpha values
    z_true = rng.standard_normal(n_participants)
    alpha_true = 0.0 + 1.0 * phi_approx(
        jnp.array(true_mu_pr + true_sigma_pr * z_true)
    )

    def recovery_model(obs):
        theta = sample_bounded_param(
            "alpha_pos",
            lower=0.0,
            upper=1.0,
            n_participants=n_participants,
            mu_prior_loc=0.0,
            mu_prior_scale=1.0,
            sigma_prior_scale=0.2,
        )
        numpyro.sample("obs", dist.Normal(theta, 0.01), obs=obs)

    nuts = NUTS(recovery_model)
    mcmc = MCMC(nuts, num_warmup=500, num_samples=500, num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), obs=alpha_true)
    samples = mcmc.get_samples()

    recovered_mu = float(jnp.mean(samples["alpha_pos_mu_pr"]))
    rel_error = abs(recovered_mu - true_mu_pr) / max(abs(true_mu_pr), 0.1)
    assert rel_error < 0.05, (
        f"alpha_pos_mu_pr recovery failed: expected={true_mu_pr:.4f}, "
        f"recovered={recovered_mu:.4f}, rel_error={rel_error:.4f}"
    )


@pytest.mark.slow
def test_bounded_param_recovery_capacity():
    """Recover capacity group mean within 5% relative error under MCMC.

    Ground truth: mu_K_pr=0.5 => K ~ 2 + 4 * Phi(0.5) ~ 4.77.
    Observation: Normal(ground_truth, 0.1).
    """
    from numpyro.infer import MCMC, NUTS

    rng = np.random.default_rng(99)
    n_participants = 30
    true_mu_pr = 0.5
    true_sigma_pr = 0.1

    z_true = rng.standard_normal(n_participants)
    k_true = 2.0 + 4.0 * phi_approx(
        jnp.array(true_mu_pr + true_sigma_pr * z_true)
    )

    def recovery_model(obs):
        theta = sample_capacity(
            n_participants=n_participants,
            mu_prior_loc=0.0,
            mu_prior_scale=1.0,
            sigma_prior_scale=0.2,
        )
        numpyro.sample("obs", dist.Normal(theta, 0.1), obs=obs)

    nuts = NUTS(recovery_model)
    mcmc = MCMC(nuts, num_warmup=500, num_samples=500, num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(1), obs=k_true)
    samples = mcmc.get_samples()

    recovered_mu = float(jnp.mean(samples["capacity_mu_pr"]))
    rel_error = abs(recovered_mu - true_mu_pr) / max(abs(true_mu_pr), 0.1)
    assert rel_error < 0.05, (
        f"capacity_mu_pr recovery failed: expected={true_mu_pr:.4f}, "
        f"recovered={recovered_mu:.4f}, rel_error={rel_error:.4f}"
    )


@pytest.mark.slow
def test_bounded_param_recovery_stick_breaking():
    """Recover kappa_total and kappa_share group means under stick-breaking decode.

    M6b uses two independent [0,1] bounded params decoded inside the likelihood:
        kappa   = kappa_total * kappa_share
        kappa_s = kappa_total * (1 - kappa_share)

    This test verifies that the non-centered parameterization recovers both
    group-level mu_pr values when observations are the decoded kappa/kappa_s.
    """
    from numpyro.infer import MCMC, NUTS

    rng = np.random.default_rng(77)
    n_participants = 30
    true_mu_pr_total = -2.0
    true_mu_pr_share = 0.0
    true_sigma_pr = 0.1

    z_total = rng.standard_normal(n_participants)
    z_share = rng.standard_normal(n_participants)
    kappa_total_true = phi_approx(
        jnp.array(true_mu_pr_total + true_sigma_pr * z_total)
    )
    kappa_share_true = phi_approx(
        jnp.array(true_mu_pr_share + true_sigma_pr * z_share)
    )
    kappa_true = kappa_total_true * kappa_share_true
    kappa_s_true = kappa_total_true * (1.0 - kappa_share_true)

    def recovery_model(obs_kappa, obs_kappa_s):
        kt = sample_bounded_param(
            "kappa_total", lower=0.0, upper=1.0,
            n_participants=n_participants,
            mu_prior_loc=-2.0, sigma_prior_scale=0.2,
        )
        ks = sample_bounded_param(
            "kappa_share", lower=0.0, upper=1.0,
            n_participants=n_participants,
            mu_prior_loc=0.0, sigma_prior_scale=0.2,
        )
        kappa = kt * ks
        kappa_s = kt * (1.0 - ks)
        numpyro.sample("obs_kappa", dist.Normal(kappa, 0.005), obs=obs_kappa)
        numpyro.sample("obs_kappa_s", dist.Normal(kappa_s, 0.005), obs=obs_kappa_s)

    nuts = NUTS(recovery_model)
    mcmc = MCMC(
        nuts, num_warmup=1000, num_samples=1000,
        num_chains=1, progress_bar=False,
    )
    mcmc.run(
        jax.random.PRNGKey(77),
        obs_kappa=kappa_true,
        obs_kappa_s=kappa_s_true,
    )
    samples = mcmc.get_samples()

    recovered_total = float(jnp.mean(samples["kappa_total_mu_pr"]))
    rel_err_total = abs(recovered_total - true_mu_pr_total) / max(
        abs(true_mu_pr_total), 0.1
    )
    assert rel_err_total < 0.05, (
        f"kappa_total_mu_pr recovery: expected={true_mu_pr_total:.4f}, "
        f"recovered={recovered_total:.4f}, rel_error={rel_err_total:.4f}"
    )

    recovered_share = float(jnp.mean(samples["kappa_share_mu_pr"]))
    rel_err_share = abs(recovered_share - true_mu_pr_share) / max(
        abs(true_mu_pr_share), 0.1
    )
    assert rel_err_share < 0.05, (
        f"kappa_share_mu_pr recovery: expected={true_mu_pr_share:.4f}, "
        f"recovered={recovered_share:.4f}, rel_error={rel_err_share:.4f}"
    )


# ---------------------------------------------------------------------------
# PARAM_PRIOR_DEFAULTS completeness
# ---------------------------------------------------------------------------


def test_param_prior_defaults_completeness():
    """PARAM_PRIOR_DEFAULTS covers all bounded params across all models.

    LBA-specific params (v_scale, A, delta, t0) are excluded because they
    use non-standard transforms handled separately in the M4 model.
    """
    lba_only = {"v_scale", "A", "delta", "t0"}

    for model_name, model_info in MODEL_REGISTRY.items():
        for param in model_info["params"]:
            if param in lba_only:
                continue
            assert param in PARAM_PRIOR_DEFAULTS, (
                f"Model '{model_name}' has param '{param}' not in PARAM_PRIOR_DEFAULTS"
            )


# ---------------------------------------------------------------------------
# load_fits_with_validation tests
# ---------------------------------------------------------------------------


def test_load_fits_with_validation_missing_column():
    """Raises ValueError with 'lacks' when parameterization_version absent."""
    df = pd.DataFrame({"alpha_pos": [0.3, 0.4], "epsilon": [0.05, 0.06]})
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f, index=False)
        tmp_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="lacks"):
            load_fits_with_validation(tmp_path, "wmrl_m3")
    finally:
        tmp_path.unlink(missing_ok=True)


def test_load_fits_with_validation_mismatch():
    """Raises ValueError with 'mismatch' when version string is wrong."""
    df = pd.DataFrame({
        "alpha_pos": [0.3],
        "parameterization_version": ["v3.0-legacy"],
    })
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f, index=False)
        tmp_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="mismatch"):
            load_fits_with_validation(tmp_path, "wmrl_m3")
    finally:
        tmp_path.unlink(missing_ok=True)


def test_load_fits_with_validation_success():
    """Returns DataFrame without error when version matches expected."""
    model = "wmrl_m3"
    expected_version = EXPECTED_PARAMETERIZATION[model]
    df = pd.DataFrame({
        "alpha_pos": [0.3, 0.4],
        "parameterization_version": [expected_version, expected_version],
    })
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f, index=False)
        tmp_path = Path(f.name)

    try:
        result = load_fits_with_validation(tmp_path, model)
        assert len(result) == 2
        assert "alpha_pos" in result.columns
    finally:
        tmp_path.unlink(missing_ok=True)
