"""Micro-benchmark: sequential vs vectorized pscan for RLWM likelihoods.

Compares single-participant likelihood evaluation wall time between sequential
(``lax.scan``) and fully vectorized pscan variants of all 6 choice-only models.
The pscan variants use Phase 19 associative scan for Q/WM updates (O(log T))
and Phase 20 vectorized policy computation (O(1) depth) -- together forming a
fully parallel likelihood with no sequential passes.

Reports speedup ratios, numerical agreement, and JIT compilation overhead.

Usage
-----
    python tests/scientific/benchmark_parallel_scan.py [--model MODEL] [--n-repeats N]

    # All 6 models, 20 repeats (default)
    python tests/scientific/benchmark_parallel_scan.py

    # Single model, 5 repeats (quick)
    python tests/scientific/benchmark_parallel_scan.py --model wmrl_m3 --n-repeats 5

Outputs
-------
    - Console: timing table with speedup ratios
    - JSON: models/bayesian/pscan_benchmark_{cpu|gpu}.json

Notes
-----
    Requires JAX (``pip install jax jaxlib``). On Windows without JAX, the
    script will fail at import time. Use the ``rlwm_gpu`` conda environment
    on the Monash M3 cluster for GPU benchmarks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root so ``rlwm.fitting.*`` imports resolve.
# tests/scientific/<file>.py is 2 levels below repo root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from rlwm.fitting.core import MAX_TRIALS_PER_BLOCK
from rlwm.fitting.models.qlearning import (
    q_learning_multiblock_likelihood_stacked,
    q_learning_multiblock_likelihood_stacked_pscan,
)
from rlwm.fitting.models.wmrl import (
    wmrl_multiblock_likelihood_stacked,
    wmrl_multiblock_likelihood_stacked_pscan,
)
from rlwm.fitting.models.wmrl_m3 import (
    wmrl_m3_multiblock_likelihood_stacked,
    wmrl_m3_multiblock_likelihood_stacked_pscan,
)
from rlwm.fitting.models.wmrl_m5 import (
    wmrl_m5_multiblock_likelihood_stacked,
    wmrl_m5_multiblock_likelihood_stacked_pscan,
)
from rlwm.fitting.models.wmrl_m6a import (
    wmrl_m6a_multiblock_likelihood_stacked,
    wmrl_m6a_multiblock_likelihood_stacked_pscan,
)
from rlwm.fitting.models.wmrl_m6b import (
    wmrl_m6b_multiblock_likelihood_stacked,
    wmrl_m6b_multiblock_likelihood_stacked_pscan,
)

# ---------------------------------------------------------------------------
# Model registry: functions + typical parameter sets
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, dict] = {
    "qlearning": {
        "seq_fn": q_learning_multiblock_likelihood_stacked,
        "pscan_fn": q_learning_multiblock_likelihood_stacked_pscan,
        "params": {
            "alpha_pos": 0.3,
            "alpha_neg": 0.2,
            "epsilon": 0.05,
        },
        "needs_set_sizes": False,
    },
    "wmrl": {
        "seq_fn": wmrl_multiblock_likelihood_stacked,
        "pscan_fn": wmrl_multiblock_likelihood_stacked_pscan,
        "params": {
            "alpha_pos": 0.3,
            "alpha_neg": 0.2,
            "phi": 0.3,
            "rho": 0.7,
            "capacity": 4.0,
            "epsilon": 0.05,
        },
        "needs_set_sizes": True,
    },
    "wmrl_m3": {
        "seq_fn": wmrl_m3_multiblock_likelihood_stacked,
        "pscan_fn": wmrl_m3_multiblock_likelihood_stacked_pscan,
        "params": {
            "alpha_pos": 0.3,
            "alpha_neg": 0.2,
            "phi": 0.3,
            "rho": 0.7,
            "capacity": 4.0,
            "kappa": 0.1,
            "epsilon": 0.05,
        },
        "needs_set_sizes": True,
    },
    "wmrl_m5": {
        "seq_fn": wmrl_m5_multiblock_likelihood_stacked,
        "pscan_fn": wmrl_m5_multiblock_likelihood_stacked_pscan,
        "params": {
            "alpha_pos": 0.3,
            "alpha_neg": 0.2,
            "phi": 0.3,
            "rho": 0.7,
            "capacity": 4.0,
            "kappa": 0.1,
            "phi_rl": 0.1,
            "epsilon": 0.05,
        },
        "needs_set_sizes": True,
    },
    "wmrl_m6a": {
        "seq_fn": wmrl_m6a_multiblock_likelihood_stacked,
        "pscan_fn": wmrl_m6a_multiblock_likelihood_stacked_pscan,
        "params": {
            "alpha_pos": 0.3,
            "alpha_neg": 0.2,
            "phi": 0.3,
            "rho": 0.7,
            "capacity": 4.0,
            "kappa_s": 0.1,
            "epsilon": 0.05,
        },
        "needs_set_sizes": True,
    },
    "wmrl_m6b": {
        "seq_fn": wmrl_m6b_multiblock_likelihood_stacked,
        "pscan_fn": wmrl_m6b_multiblock_likelihood_stacked_pscan,
        "params": {
            "alpha_pos": 0.3,
            "alpha_neg": 0.2,
            "phi": 0.3,
            "rho": 0.7,
            "capacity": 4.0,
            "kappa": 0.1,
            "kappa_s": 0.05,
            "epsilon": 0.05,
        },
        "needs_set_sizes": True,
    },
}


def _generate_synthetic_data(
    n_blocks: int = 17,
    trials_per_block: int = MAX_TRIALS_PER_BLOCK,
    num_stimuli: int = 6,
    num_actions: int = 3,
    seed: int = 42,
) -> dict[str, jnp.ndarray]:
    """Generate synthetic stacked data matching real-data format.

    Parameters
    ----------
    n_blocks : int
        Number of blocks per participant (typical main task: 17-21).
    trials_per_block : int
        Trials per block (padded to ``MAX_TRIALS_PER_BLOCK``).
    num_stimuli : int
        Number of distinct stimuli.
    num_actions : int
        Number of possible actions.
    seed : int
        NumPy random seed.

    Returns
    -------
    dict[str, jnp.ndarray]
        Keys: ``stimuli_stacked``, ``actions_stacked``, ``rewards_stacked``,
        ``set_sizes_stacked``, ``masks_stacked`` -- each shape
        ``(n_blocks, MAX_TRIALS_PER_BLOCK)``.
    """
    rng = np.random.default_rng(seed)

    stimuli = rng.integers(0, num_stimuli, size=(n_blocks, trials_per_block))
    actions = rng.integers(0, num_actions, size=(n_blocks, trials_per_block))
    rewards = rng.choice([0.0, 1.0], size=(n_blocks, trials_per_block)).astype(
        np.float32
    )
    # Typical set sizes: blocks alternate between 3 and 6
    set_size_values = [3, 6]
    set_sizes = np.zeros((n_blocks, trials_per_block), dtype=np.int32)
    for b in range(n_blocks):
        set_sizes[b, :] = set_size_values[b % len(set_size_values)]

    masks = np.ones((n_blocks, trials_per_block), dtype=np.float32)

    return {
        "stimuli_stacked": jnp.array(stimuli, dtype=jnp.int32),
        "actions_stacked": jnp.array(actions, dtype=jnp.int32),
        "rewards_stacked": jnp.array(rewards, dtype=jnp.float32),
        "set_sizes_stacked": jnp.array(set_sizes, dtype=jnp.int32),
        "masks_stacked": jnp.array(masks, dtype=jnp.float32),
    }


def _build_call_kwargs(
    model_name: str,
    data: dict[str, jnp.ndarray],
) -> dict:
    """Build keyword arguments for a likelihood function call.

    Parameters
    ----------
    model_name : str
        Model key (e.g. ``'wmrl_m3'``).
    data : dict
        Stacked data arrays from ``_generate_synthetic_data``.

    Returns
    -------
    dict
        Keyword arguments ready to pass to the likelihood function.
    """
    reg = _MODEL_REGISTRY[model_name]
    kwargs: dict = {
        "stimuli_stacked": data["stimuli_stacked"],
        "actions_stacked": data["actions_stacked"],
        "rewards_stacked": data["rewards_stacked"],
        "masks_stacked": data["masks_stacked"],
        "num_stimuli": 6,
        "num_actions": 3,
        "q_init": 0.5,
        "return_pointwise": False,
    }

    if reg["needs_set_sizes"]:
        kwargs["set_sizes_stacked"] = data["set_sizes_stacked"]
        kwargs["wm_init"] = 1.0 / 3.0

    # Add model-specific parameters
    kwargs.update(reg["params"])

    return kwargs


def _time_function(
    fn: object,
    kwargs: dict,
    n_repeats: int,
) -> tuple[float, float]:
    """Time a likelihood function call.

    Performs one warmup call (triggers internal JIT compilation, excluded from
    timing) then ``n_repeats`` timed calls using ``time.perf_counter()`` with
    ``block_until_ready()``.

    The likelihood functions use internal ``lax.fori_loop`` / ``lax.scan`` /
    ``jax.lax.associative_scan`` which are JIT-traced on first call.  We do NOT
    wrap in an outer ``jax.jit`` because that would trace ``return_pointwise``
    and other Python-control-flow arguments.

    Parameters
    ----------
    fn : callable
        Likelihood function to time (called directly, not JIT-wrapped).
    kwargs : dict
        Keyword arguments for the function.
    n_repeats : int
        Number of timed repetitions.

    Returns
    -------
    tuple[float, float]
        ``(mean_ms, std_ms)`` -- mean and std of wall-clock time in
        milliseconds across ``n_repeats`` calls.
    """
    # Warmup (triggers internal JIT compilation)
    result = fn(**kwargs)
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()

    # Timed runs
    times_ms: list[float] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = fn(**kwargs)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    return float(np.mean(times_ms)), float(np.std(times_ms))


def _time_compilation(fn: object, kwargs: dict) -> float:
    """Measure compilation time (first call overhead).

    Calls the function once (triggering internal JIT tracing) and returns the
    wall-clock time.  Subsequent calls to ``_time_function`` will be fast.

    Parameters
    ----------
    fn : callable
        Likelihood function (not externally JIT-wrapped).
    kwargs : dict
        Keyword arguments for the function.

    Returns
    -------
    float
        First-call time in milliseconds (includes JIT compilation).
    """
    t0 = time.perf_counter()
    result = fn(**kwargs)
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def benchmark_model(
    model_name: str,
    data: dict[str, jnp.ndarray],
    n_repeats: int = 20,
) -> dict:
    """Benchmark sequential vs pscan for a single model.

    Parameters
    ----------
    model_name : str
        Model key from ``_MODEL_REGISTRY``.
    data : dict
        Stacked synthetic data arrays.
    n_repeats : int
        Number of timed repetitions.

    Returns
    -------
    dict
        Benchmark results including timing, speedup, and NLL agreement.
    """
    reg = _MODEL_REGISTRY[model_name]
    kwargs = _build_call_kwargs(model_name, data)

    # Measure compilation time
    print(f"  Compiling sequential...", end="", flush=True)
    seq_compile_ms = _time_compilation(reg["seq_fn"], kwargs)
    print(f" {seq_compile_ms:.0f}ms")

    print(f"  Compiling pscan...", end="", flush=True)
    pscan_compile_ms = _time_compilation(reg["pscan_fn"], kwargs)
    print(f" {pscan_compile_ms:.0f}ms")

    # Numerical agreement check (functions use internal JIT)
    seq_nll = float(reg["seq_fn"](**kwargs))
    pscan_nll = float(reg["pscan_fn"](**kwargs))
    nll_diff = abs(seq_nll - pscan_nll)
    nll_rel_diff = nll_diff / (abs(seq_nll) + 1e-10)

    print(f"  NLL sequential: {seq_nll:.6f}")
    print(f"  NLL pscan:      {pscan_nll:.6f}")
    print(f"  NLL agreement:  {nll_rel_diff:.2e} (abs diff: {nll_diff:.2e})")

    if nll_rel_diff > 1e-4:
        print(f"  WARNING: NLL disagreement > 1e-4!")

    # Timed runs
    print(f"  Timing sequential ({n_repeats} repeats)...", end="", flush=True)
    seq_mean_ms, seq_std_ms = _time_function(reg["seq_fn"], kwargs, n_repeats)
    print(f" {seq_mean_ms:.2f} +/- {seq_std_ms:.2f} ms")

    print(f"  Timing pscan ({n_repeats} repeats)...", end="", flush=True)
    pscan_mean_ms, pscan_std_ms = _time_function(
        reg["pscan_fn"], kwargs, n_repeats
    )
    print(f" {pscan_mean_ms:.2f} +/- {pscan_std_ms:.2f} ms")

    speedup = seq_mean_ms / pscan_mean_ms if pscan_mean_ms > 0 else float("inf")
    print(f"  Speedup: {speedup:.2f}x")

    return {
        "n_blocks": int(data["stimuli_stacked"].shape[0]),
        "trials_per_block": int(data["stimuli_stacked"].shape[1]),
        "n_repeats": n_repeats,
        "seq_ms": round(seq_mean_ms, 3),
        "seq_std_ms": round(seq_std_ms, 3),
        "pscan_ms": round(pscan_mean_ms, 3),
        "pscan_std_ms": round(pscan_std_ms, 3),
        "speedup": round(speedup, 3),
        "nll_sequential": round(seq_nll, 6),
        "nll_pscan": round(pscan_nll, 6),
        "nll_agreement": round(nll_rel_diff, 10),
        "nll_abs_diff": round(nll_diff, 10),
        "seq_compile_ms": round(seq_compile_ms, 1),
        "pscan_compile_ms": round(pscan_compile_ms, 1),
    }


def _get_device_info() -> str:
    """Return a human-readable string describing the JAX compute device."""
    devices = jax.devices()
    if not devices:
        return "unknown"
    d = devices[0]
    if d.platform == "gpu":
        # Try to get GPU name
        try:
            return str(d.device_kind)
        except Exception:
            return f"gpu:{d.id}"
    return d.platform


def _get_gpu_memory() -> dict | None:
    """Return GPU memory stats if available, else None."""
    devices = jax.devices()
    if not devices or devices[0].platform != "gpu":
        return None
    try:
        stats = devices[0].memory_stats()
        return {
            "peak_bytes_in_use": stats.get("peak_bytes_in_use", 0),
            "bytes_in_use": stats.get("bytes_in_use", 0),
        }
    except Exception:
        return None


def main() -> None:
    """Run the parallel scan micro-benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark sequential vs pscan likelihood evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(_MODEL_REGISTRY.keys()),
        help="Model to benchmark (default: all 6 choice-only models)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=20,
        help="Number of timed repetitions per variant (default: 20)",
    )
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=17,
        help="Number of blocks in synthetic data (default: 17)",
    )
    args = parser.parse_args()

    models = [args.model] if args.model else list(_MODEL_REGISTRY.keys())

    device_info = _get_device_info()
    timestamp = datetime.now(timezone.utc).isoformat()

    print("=" * 72)
    print("SEQUENTIAL vs VECTORIZED PSCAN LIKELIHOOD BENCHMARK")
    print("(Phase 19 associative scan + Phase 20 vectorized policy)")
    print("=" * 72)
    print(f"Device:        {device_info}")
    print(f"JAX version:   {jax.__version__}")
    print(f"Models:        {', '.join(models)}")
    print(f"N repeats:     {args.n_repeats}")
    print(f"N blocks:      {args.n_blocks}")
    print(f"Trials/block:  {MAX_TRIALS_PER_BLOCK}")
    print(f"Timestamp:     {timestamp}")
    print("=" * 72)

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    data = _generate_synthetic_data(n_blocks=args.n_blocks)
    print(
        f"  Shape: {data['stimuli_stacked'].shape} "
        f"(blocks x trials_per_block)"
    )

    # Benchmark each model
    results: dict = {}
    for model_name in models:
        print(f"\n{'─' * 60}")
        print(f"Benchmarking: {model_name}")
        print(f"{'─' * 60}")
        results[model_name] = benchmark_model(model_name, data, args.n_repeats)

    # GPU memory stats (after all models)
    gpu_mem = _get_gpu_memory()

    # Print summary table
    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    header = (
        f"{'Model':<12} {'Sequential (ms)':>16} {'PScan (ms)':>12} "
        f"{'Speedup':>9} {'NLL Agree':>12}"
    )
    print(header)
    print("-" * len(header))
    for model_name, r in results.items():
        print(
            f"{model_name:<12} {r['seq_ms']:>12.2f}     "
            f"{r['pscan_ms']:>8.2f}     "
            f"{r['speedup']:>6.2f}x  "
            f"{r['nll_agreement']:>10.2e}"
        )
    print(f"{'=' * 72}")

    if gpu_mem:
        peak_mb = gpu_mem["peak_bytes_in_use"] / (1024 * 1024)
        print(f"GPU peak memory: {peak_mb:.1f} MB")

    # Save JSON — filename includes backend so CPU and GPU results coexist
    output_dir = _PROJECT_ROOT / "models" / "bayesian"
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = jax.default_backend().lower()  # "cpu" or "gpu"
    output_path = output_dir / f"pscan_benchmark_{backend}.json"

    output_data = {
        "phase": "phase_20_vectorized",
        "description": (
            "Phase 19 associative scan (O(log T) Q/WM updates) "
            "+ Phase 20 vectorized policy (O(1) depth)"
        ),
        "device": device_info,
        "jax_version": jax.__version__,
        "timestamp": timestamp,
        "n_blocks": args.n_blocks,
        "trials_per_block": MAX_TRIALS_PER_BLOCK,
        "n_repeats": args.n_repeats,
        "models": results,
    }
    if gpu_mem:
        output_data["gpu_memory"] = gpu_mem

    with open(output_path, "w") as fh:
        json.dump(output_data, fh, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
