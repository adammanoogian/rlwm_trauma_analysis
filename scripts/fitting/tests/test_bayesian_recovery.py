"""Smoke tests for scripts/21_run_bayesian_recovery.py.

Fast unit tests exercising the single-subject and aggregate modes
under a minimal MCMC budget.  The single-subject smoke test runs a
short warmup=50 / samples=100 / chains=1 fit on qlearning with only
3 blocks x 20 trials — this typically completes in 30–90 s on CPU.

The aggregate test uses stub JSONs (no MCMC) to verify CSV schema,
pass-criterion logic, and missing-subject handling.

Both tests are intended to run in CI; the slow single-subject fit
may be skipped locally with ``pytest -m "not slow"`` if needed (it
is not marked slow because the budget is already small).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Ensure the project root is importable for `import scripts...` and `import config`
_TEST_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _TEST_FILE.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import the recovery runner as a module.  ``scripts/21_run_bayesian_recovery.py``
# is not a conventional package path, so we use importlib to load it by path.
import importlib.util

_RUNNER_PATH = _PROJECT_ROOT / "scripts" / "21_run_bayesian_recovery.py"
_spec = importlib.util.spec_from_file_location(
    "_bayes_recovery_runner", str(_RUNNER_PATH)
)
assert _spec is not None and _spec.loader is not None, (
    f"Could not load {_RUNNER_PATH}"
)
recovery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recovery)


# ---------------------------------------------------------------------------
# Single-subject smoke test
# ---------------------------------------------------------------------------


def test_sample_true_params_from_prior_has_all_keys() -> None:
    """Every parameter in MODEL_REGISTRY[model]['params'] must be sampled.

    Covers all 6 choice-only models to ensure no PARAM_PRIOR_DEFAULTS
    lookup regression sneaks in when a new model is added.
    """
    from config import MODEL_REGISTRY

    for model in ["qlearning", "wmrl", "wmrl_m3", "wmrl_m5", "wmrl_m6a", "wmrl_m6b"]:
        params = recovery.sample_true_params_from_prior(
            model=model, subject_idx=1, seed=42
        )
        expected = set(MODEL_REGISTRY[model]["params"])
        assert set(params.keys()) == expected, (
            f"Expected params for {model!r}: {expected}, got {set(params.keys())}"
        )
        # All values must lie in their bounded intervals (per
        # PARAM_PRIOR_DEFAULTS lower/upper).
        for pname, val in params.items():
            if pname == "capacity":
                assert 2.0 <= val <= 6.0, f"{pname}={val} outside [2, 6]"
            else:
                assert 0.0 <= val <= 1.0, f"{pname}={val} outside [0, 1]"


def test_sample_true_params_different_subjects_give_different_draws() -> None:
    """Folding subject_idx into the RNG key must change the draw.

    Guards against a regression where subject_idx is accidentally
    ignored, which would produce identical ``true`` values across all
    50 array tasks and tank the Pearson r.
    """
    p1 = recovery.sample_true_params_from_prior("qlearning", 1, 42)
    p2 = recovery.sample_true_params_from_prior("qlearning", 2, 42)
    assert p1 != p2, "subject_idx must affect the draw"


def test_single_subject_recovery_qlearning_smoke(tmp_path: Path) -> None:
    """End-to-end smoke: fit one synthetic qlearning subject and read the JSON.

    Budget is intentionally tiny so this runs in CI.  We do NOT assert
    r > 0.80 or HDI coverage — with warmup=50 that's not meaningful.
    We only verify the JSON structure is correct.
    """
    out_path = recovery.run_single_subject(
        model="qlearning",
        subject_idx=1,
        seed=123,
        output_dir=tmp_path,
        num_warmup=50,
        num_samples=100,
        num_chains=1,
        max_tree_depth=6,
        n_blocks=3,
        n_trials_per_block=20,
    )
    assert out_path.exists(), f"Expected JSON at {out_path}"

    with out_path.open(encoding="utf-8") as fh:
        result = json.load(fh)

    # Top-level structure
    assert result["model"] == "qlearning"
    assert result["subject_idx"] == 1
    assert "params" in result
    assert "diagnostics" in result
    assert "mcmc_budget" in result
    assert result["mcmc_budget"]["num_warmup"] == 50

    # Per-parameter entries must exist for all qlearning params with all keys
    expected_params = {"alpha_pos", "alpha_neg", "epsilon"}
    assert set(result["params"].keys()) == expected_params

    for pname in expected_params:
        pdata = result["params"][pname]
        for key in ("true", "posterior_mean", "hdi_low", "hdi_high", "in_hdi"):
            assert key in pdata, (
                f"Missing key {key!r} for parameter {pname!r} in JSON"
            )
        # in_hdi must be 0 or 1
        assert pdata["in_hdi"] in (0, 1), (
            f"in_hdi must be 0 or 1, got {pdata['in_hdi']!r}"
        )
        # HDI bounds must be ordered
        assert pdata["hdi_low"] <= pdata["hdi_high"], (
            f"HDI bounds reversed for {pname}: {pdata}"
        )


# ---------------------------------------------------------------------------
# Aggregate mode tests (no MCMC — use stub JSONs)
# ---------------------------------------------------------------------------


def _write_stub_subject_json(
    out_dir: Path,
    model: str,
    subject_idx: int,
    params: dict[str, dict[str, float]],
) -> Path:
    """Write a minimal stub per-subject JSON for aggregate-mode tests.

    Parameters
    ----------
    out_dir : Path
        Output directory.
    model : str
        Model name.
    subject_idx : int
        1-indexed subject number.
    params : dict
        Per-parameter dict with keys ``true``, ``posterior_mean``,
        ``hdi_low``, ``hdi_high``, ``in_hdi``.

    Returns
    -------
    Path
        Path to the written JSON.
    """
    payload = {
        "model": model,
        "subject_idx": subject_idx,
        "seed": 0,
        "params": params,
        "diagnostics": {"max_rhat": 1.0, "min_ess": 400, "num_divergences": 0},
    }
    path = out_dir / f"{model}_subject_{subject_idx:03d}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_aggregate_handles_missing(tmp_path: Path) -> None:
    """Aggregate must not crash when fewer JSONs exist than --n-subjects.

    Also verifies the CSV schema and summary.md contents.
    """
    # Two stub qlearning subjects out of 5 expected
    _write_stub_subject_json(
        tmp_path, "qlearning", 1,
        {
            "alpha_pos": {"true": 0.3, "posterior_mean": 0.32,
                          "hdi_low": 0.2, "hdi_high": 0.45, "in_hdi": 1},
            "alpha_neg": {"true": 0.2, "posterior_mean": 0.25,
                          "hdi_low": 0.1, "hdi_high": 0.40, "in_hdi": 1},
            "epsilon": {"true": 0.1, "posterior_mean": 0.08,
                        "hdi_low": 0.01, "hdi_high": 0.20, "in_hdi": 1},
        },
    )
    _write_stub_subject_json(
        tmp_path, "qlearning", 2,
        {
            "alpha_pos": {"true": 0.7, "posterior_mean": 0.68,
                          "hdi_low": 0.55, "hdi_high": 0.85, "in_hdi": 1},
            "alpha_neg": {"true": 0.5, "posterior_mean": 0.45,
                          "hdi_low": 0.30, "hdi_high": 0.60, "in_hdi": 1},
            "epsilon": {"true": 0.4, "posterior_mean": 0.50,
                        "hdi_low": 0.25, "hdi_high": 0.60, "in_hdi": 1},
        },
    )

    summary = recovery.aggregate_recovery_results(
        model="qlearning", n_subjects=5, output_dir=tmp_path
    )

    # CSV schema
    csv_path = tmp_path / "qlearning_recovery.csv"
    assert csv_path.exists()
    import pandas as pd  # local import to keep top clean

    df = pd.read_csv(csv_path)
    expected_cols = {
        "model", "parameter", "n_subjects",
        "pearson_r", "hdi_coverage", "pass_criterion", "status",
    }
    assert set(df.columns) == expected_cols

    # n_subjects should be 2 (loaded), not 5 (expected)
    assert (df["n_subjects"] == 2).all()

    # qlearning has no kappa-family params => verdict NO_KAPPA
    assert summary["verdict"] == "NO_KAPPA"
    # All three qlearning params must be labelled "descriptive only"
    assert set(df["status"].unique()) == {"descriptive only"}

    # Missing subjects 3, 4, 5
    assert set(summary["subjects_missing"]) == {3, 4, 5}

    # summary.md must exist and mention NO_KAPPA
    md_path = tmp_path / "qlearning_recovery_summary.md"
    assert md_path.exists()
    md_text = md_path.read_text(encoding="utf-8")
    assert "NO_KAPPA" in md_text
    assert "qlearning" in md_text


def test_aggregate_kappa_pass_and_fail(tmp_path: Path) -> None:
    """Verify PASS vs FAIL verdict logic for kappa-family parameters.

    Builds 5 M3 stub subjects where:

    * ``kappa`` recovers well — true and posterior_mean correlate strongly
      with all five in the HDI (coverage = 1.0).
    * ``alpha_pos`` recovers poorly (not kappa-family, so descriptive only
      — must NOT drive the verdict either way).
    """
    import numpy as np

    np.random.seed(0)
    # Create strongly-correlated true vs posterior_mean for kappa
    true_kappa = np.linspace(0.05, 0.8, 5)
    post_kappa = true_kappa + np.random.normal(0, 0.02, size=5)

    for idx, (tk, pk) in enumerate(zip(true_kappa, post_kappa), start=1):
        params = {
            "alpha_pos": {"true": 0.3, "posterior_mean": 0.5,
                          "hdi_low": 0.1, "hdi_high": 0.8, "in_hdi": 1},
            "alpha_neg": {"true": 0.3, "posterior_mean": 0.5,
                          "hdi_low": 0.1, "hdi_high": 0.8, "in_hdi": 1},
            "phi": {"true": 0.2, "posterior_mean": 0.3,
                    "hdi_low": 0.05, "hdi_high": 0.55, "in_hdi": 1},
            "rho": {"true": 0.6, "posterior_mean": 0.65,
                    "hdi_low": 0.5, "hdi_high": 0.85, "in_hdi": 1},
            "capacity": {"true": 4.0, "posterior_mean": 4.2,
                         "hdi_low": 2.5, "hdi_high": 5.5, "in_hdi": 1},
            "kappa": {"true": float(tk), "posterior_mean": float(pk),
                      "hdi_low": float(tk - 0.05), "hdi_high": float(tk + 0.05),
                      "in_hdi": 1},
            "epsilon": {"true": 0.1, "posterior_mean": 0.12,
                        "hdi_low": 0.02, "hdi_high": 0.25, "in_hdi": 1},
        }
        _write_stub_subject_json(tmp_path, "wmrl_m3", idx, params)

    summary = recovery.aggregate_recovery_results(
        model="wmrl_m3", n_subjects=5, output_dir=tmp_path
    )
    assert summary["verdict"] == "PASS", (
        f"Expected PASS for strong kappa recovery, got {summary['verdict']}"
    )

    import pandas as pd
    df = pd.read_csv(tmp_path / "wmrl_m3_recovery.csv")
    kappa_row = df[df["parameter"] == "kappa"].iloc[0]
    assert kappa_row["status"] == "PASS"
    assert kappa_row["pearson_r"] >= 0.80
    assert kappa_row["hdi_coverage"] >= 0.90


def test_aggregate_kappa_fail_low_coverage(tmp_path: Path) -> None:
    """Zero HDI coverage on kappa must produce a FAIL verdict.

    Even with perfect correlation, if the HDI never contains truth
    (coverage = 0), the recovery gate must FAIL — this matches the
    Baribault & Collins (2023) joint r + coverage requirement.
    """
    import numpy as np

    np.random.seed(1)
    true_kappa = np.linspace(0.05, 0.8, 5)
    post_kappa = true_kappa.copy()  # Perfect correlation

    for idx, (tk, pk) in enumerate(zip(true_kappa, post_kappa), start=1):
        params = {
            "alpha_pos": {"true": 0.3, "posterior_mean": 0.3,
                          "hdi_low": 0.2, "hdi_high": 0.4, "in_hdi": 1},
            "alpha_neg": {"true": 0.3, "posterior_mean": 0.3,
                          "hdi_low": 0.2, "hdi_high": 0.4, "in_hdi": 1},
            "phi": {"true": 0.2, "posterior_mean": 0.2,
                    "hdi_low": 0.1, "hdi_high": 0.3, "in_hdi": 1},
            "rho": {"true": 0.6, "posterior_mean": 0.6,
                    "hdi_low": 0.5, "hdi_high": 0.7, "in_hdi": 1},
            "capacity": {"true": 4.0, "posterior_mean": 4.0,
                         "hdi_low": 3.0, "hdi_high": 5.0, "in_hdi": 1},
            # kappa: perfect correlation but HDI NEVER contains truth
            "kappa": {"true": float(tk), "posterior_mean": float(pk),
                      "hdi_low": float(tk + 0.2), "hdi_high": float(tk + 0.3),
                      "in_hdi": 0},
            "epsilon": {"true": 0.1, "posterior_mean": 0.1,
                        "hdi_low": 0.05, "hdi_high": 0.2, "in_hdi": 1},
        }
        _write_stub_subject_json(tmp_path, "wmrl_m3", idx, params)

    summary = recovery.aggregate_recovery_results(
        model="wmrl_m3", n_subjects=5, output_dir=tmp_path
    )
    assert summary["verdict"] == "FAIL"


def test_safe_pearson_r_handles_zero_variance() -> None:
    """Zero-variance true or posterior must return NaN, not crash."""
    import numpy as np

    # Zero variance in posterior
    r = recovery._safe_pearson_r(
        np.array([0.1, 0.3, 0.5]), np.array([0.2, 0.2, 0.2])
    )
    assert np.isnan(r)

    # Valid case
    r = recovery._safe_pearson_r(
        np.array([0.1, 0.3, 0.5]), np.array([0.15, 0.32, 0.48])
    )
    assert r > 0.9
