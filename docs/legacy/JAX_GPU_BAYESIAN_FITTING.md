# JAX GPU Bayesian Fitting Guide

**Practical end-to-end guide for running hierarchical Bayesian RLWM models
with GPU-accelerated parallel likelihoods.**

This guide covers the full pipeline from environment setup through MCMC
sampling with parallel scan likelihoods. It synthesizes findings from
Phase 19 (associative scan for Q/WM updates) and Phase 20 (vectorized
policy computation), providing actionable guidance for both CPU and GPU
execution.

**Cross-references:**
- [PARALLEL_SCAN_LIKELIHOOD.md](PARALLEL_SCAN_LIKELIHOOD.md) -- Implementation
  details for the associative scan and vectorized policy approach
- [DEER_NONLINEAR_PARALLELIZATION.md](DEER_NONLINEAR_PARALLELIZATION.md) --
  DEER no-go analysis and theoretical background

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Likelihood Architecture](#3-likelihood-architecture)
4. [Using --use-pscan](#4-using---use-pscan)
5. [Performance Characteristics](#5-performance-characteristics)
6. [DEER Investigation Summary](#6-deer-investigation-summary)
7. [Model-Specific Notes](#7-model-specific-notes)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Overview

The RLWM Bayesian fitting pipeline uses NumPyro for hierarchical MCMC
sampling with JAX-based likelihood functions. The `--use-pscan` flag swaps
the default sequential `lax.scan` likelihoods for fully parallel variants
that use `jax.lax.associative_scan` for Q/WM state trajectories and
vectorized array operations for policy computation.

**Pipeline summary:**

```
Data preparation (CPU)
    |
    v
Precompute perseveration arrays (CPU, once)   <-- parameter-independent
    |
    v
MCMC sampling (CPU or GPU)
    |-- Each likelihood evaluation:
    |   Phase 1: associative_scan for Q/WM     <-- O(log T) depth
    |   Phase 2: vectorized policy + log-prob   <-- O(1) depth
    |
    v
Posterior analysis (CPU)
```

The parallel likelihoods produce numerically identical results to the
sequential variants (validated to within 1e-4 relative tolerance across
all 6 choice-only models).

---

## 2. Prerequisites

### JAX with GPU support

```bash
# CPU-only (for development/testing)
pip install jax jaxlib

# GPU (CUDA 12.x) -- required for GPU acceleration
pip install jax[cuda12]

# Verify GPU is detected
python -c "import jax; print(jax.devices())"
# Expected: [CudaDevice(id=0)]
```

### NumPyro

```bash
pip install numpyro
```

### Monash M3 cluster environment

The `rlwm_gpu` conda environment on the M3 cluster has all dependencies
pre-configured:

```bash
module load cuda/12.2
conda activate rlwm_gpu

# Verify
python -c "import jax; print(jax.default_backend())"
# Expected: "gpu"
```

### Minimum versions

| Package | Minimum version | Notes |
|---|---|---|
| jax | 0.4.20 | `associative_scan` stability |
| jaxlib | 0.4.20 | Matching JAX version |
| numpyro | 0.13.0 | Hierarchical model support |
| Python | 3.10 | Type hint syntax |

---

## 3. Likelihood Architecture

### Two-phase parallel evaluation

Every pscan likelihood function follows a two-phase architecture:

**Phase 1 -- Parallel scan (O(log T) depth):**

Q-value updates and WM decay/overwrite are expressed as AR(1) linear
recurrences and solved via `jax.lax.associative_scan`. The associative
operator for the recurrence `x_t = a_t * x_{t-1} + b_t` is:

```
(a_2, b_2) . (a_1, b_1) = (a_2 * a_1, a_2 * b_1 + b_2)
```

This computes the full `(T, S, A)` Q-value and WM trajectories in
O(log T) parallel steps. See
[PARALLEL_SCAN_LIKELIHOOD.md](PARALLEL_SCAN_LIKELIHOOD.md) Sections 1-2
for the full derivation.

**Phase 2 -- Vectorized policy (O(1) depth):**

Given the Q/WM trajectories from Phase 1 and precomputed perseveration
arrays, all T trial log-probabilities are computed simultaneously via
array broadcasting:

```python
# Pseudocode -- actual implementation uses jax.vmap
q_vals = q_traj[trial_idx, stimuli, :]       # (T, nA)
wm_vals = wm_traj[trial_idx, stimuli, :]     # (T, nA)
omega = rho * jnp.minimum(1.0, K / set_sizes)  # (T,)
policy = omega[:, None] * wm_probs + (1 - omega[:, None]) * rl_probs
log_probs = jnp.log(policy[trial_idx, actions]) * mask
```

No sequential dependency remains. See
[PARALLEL_SCAN_LIKELIHOOD.md](PARALLEL_SCAN_LIKELIHOOD.md) Section 7 for
the Phase 20 implementation details.

### Perseveration precomputation

The one parameter-independent sequential operation is precomputing the
perseveration arrays from observed data. This runs once before MCMC:

| Function | Used by | Complexity |
|---|---|---|
| `precompute_last_action_global` | M3, M5, M6b | O(T) per block |
| `precompute_last_actions_per_stimulus` | M6a, M6b | O(T) per block |

These are called inside each block's likelihood function and are
amortized across MCMC iterations.

---

## 4. Using --use-pscan

### CLI examples

```bash
# Bayesian fitting with parallel scan likelihoods (CPU)
python scripts/13_fit_bayesian.py --model wmrl_m3 \
    --data output/task_trials_long.csv \
    --use-pscan

# GPU-accelerated Bayesian fitting
python scripts/13_fit_bayesian.py --model wmrl_m3 \
    --data output/task_trials_long.csv \
    --use-pscan \
    --chains 4 --warmup 1000 --samples 2000

# All choice-only models with pscan
for model in qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b; do
    python scripts/13_fit_bayesian.py --model $model \
        --data output/task_trials_long.csv \
        --use-pscan
done
```

### SLURM submission (Monash M3 GPU)

```bash
# Single model GPU job
sbatch cluster/13_bayesian_gpu.slurm

# Or with explicit model
sbatch --export=MODEL=wmrl_m3,USE_PSCAN=1 cluster/13_bayesian_gpu.slurm
```

### What --use-pscan changes

| Aspect | Without `--use-pscan` | With `--use-pscan` |
|---|---|---|
| Phase 1 (Q/WM) | Sequential `lax.scan` | `associative_scan` |
| Phase 2 (policy) | Sequential `lax.scan` | Vectorized array ops |
| Numerical result | Baseline | Equivalent (< 1e-4 rel. diff) |
| CPU performance | Faster (lower overhead) | Slower (~3-4x on CPU) |
| GPU performance | Baseline | Expected speedup (TBD) |
| Compilation time | ~250ms | ~650ms |

---

## 5. Performance Characteristics

### CPU performance (current benchmark data)

Phase 19 CPU benchmark (`output/bayesian/pscan_benchmark.json`) established
that the parallel scan approach is **slower on CPU** due to overhead:

| Variant | Time (ms) | Relative |
|---|---|---|
| Sequential (`lax.scan`) | 164 | 1.0x |
| PScan (Phase 19+20) | 624 | 0.26x (3.8x slower) |

This is expected. For T=100 trials per block and 17 blocks, the O(log T)
depth advantage of `associative_scan` does not overcome:
- Scan operator overhead (function call per tree node)
- Memory allocation for intermediate arrays
- CPU sequential execution already efficient for small T

### GPU performance (expected, not yet benchmarked)

GPU acceleration is expected to provide speedups for:
- **Large batch sizes:** vmap across participants (N=154)
- **Large T:** Future experiments with more trials per block
- **M4 LBA:** Joint choice+RT model with higher arithmetic intensity

The GPU benchmark will be run when cluster access is available. The
primary benefit of the fully parallel likelihood is enabling this future
GPU acceleration.

### When to use --use-pscan

| Scenario | Recommendation |
|---|---|
| CPU, single participant | Do NOT use (slower) |
| CPU, vmap across participants | Test both (may break even) |
| GPU, any configuration | Use `--use-pscan` |
| Development/debugging | Do NOT use (slower compilation) |
| Production MCMC on cluster | Use `--use-pscan` with GPU |

---

## 6. DEER Investigation Summary

Phase 20 investigated DEER (Deep Equilibrium Recurrences) as a potential
approach to parallelize the non-linear Phase 2 components. The conclusion:

**DEER: NO-GO**

- The Phase 2 sequential dependency was a phantom -- an implementation
  artifact of using `lax.scan` to track `last_action`, not a true
  non-linear recurrence.
- Perseveration state is a discrete integer (action index), which violates
  DEER's smoothness assumptions for Newton-based linearization.
- For scalar state (D=1) and short sequences (T=100), DEER's per-iteration
  overhead exceeds sequential evaluation cost.

**Vectorized policy: GO**

- Precomputing `last_action` arrays from observed data eliminates all
  sequential dependency in Phase 2.
- The resulting vectorized policy computation is O(1) depth, numerically
  exact, and simpler than the `lax.scan` implementation it replaced.

For full details, see
[DEER_NONLINEAR_PARALLELIZATION.md](DEER_NONLINEAR_PARALLELIZATION.md).

---

## 7. Model-Specific Notes

### Choice-only models (M1-M3, M5, M6a, M6b)

All choice-only models support `--use-pscan` with the same two-phase
architecture. Key differences:

| Model | Phase 1 scans | Phase 2 extras | Perseveration type |
|---|---|---|---|
| M1 (Q-learning) | Q only | None | None |
| M2 (WM-RL) | Q + WM | WM-Q mixing | None |
| M3 (WM-RL+kappa) | Q + WM | Mixing + global persev. | Global |
| M5 (WM-RL+phi_rl) | Q + WM + RL decay | Mixing + global persev. | Global |
| M6a (WM-RL+kappa_s) | Q + WM | Mixing + stimulus persev. | Per-stimulus |
| M6b (WM-RL+dual) | Q + WM | Mixing + dual persev. | Global + per-stimulus |

**M1 (Q-learning):** Simplest case. Only Q-value scan, no WM, no
perseveration. Phase 2 is just softmax + epsilon noise.

**M2 (WM-RL):** Adds WM decay/overwrite scan and WM-Q mixing in Phase 2.
No perseveration precomputation needed.

**M3, M5:** Use `precompute_last_action_global` for global perseveration
(`kappa` parameter). M5 additionally includes an RL forgetting scan
(`phi_rl` parameter).

**M6a:** Uses `precompute_last_actions_per_stimulus` for stimulus-specific
perseveration (`kappa_s` parameter). Per-stimulus tracking requires a
`(num_stimuli,)` state array.

**M6b:** Uses both precomputation functions for dual perseveration
(`kappa_total`, `kappa_share` with stick-breaking parameterization).
Most complex Phase 2 but still fully vectorized.

### M4 (RLWM-LBA) -- separate track

M4 is the joint choice+RT model using the Linear Ballistic Accumulator
(LBA). It has fundamentally different requirements:

- **Float64 precision:** Required for LBA density computation (log-pdf
  involves subtraction of near-equal quantities)
- **Higher arithmetic intensity:** LBA density is ~10x more compute per
  trial than choice-only softmax
- **Separate AIC track:** M4's AIC is not comparable to choice-only models

M4 does NOT currently have a pscan variant. The LBA density computation
adds non-trivial sequential dependencies (drift rate depends on Q/WM
state). A pscan M4 variant would require:

1. Phase 1 scan for Q/WM (same as other models)
2. Vectorized LBA density computation in Phase 2 (feasible but not yet
   implemented)

M4 is the strongest candidate for GPU acceleration due to its higher
arithmetic intensity.

---

## 8. Troubleshooting

### GPU not detected

```python
import jax
print(jax.devices())
# If only [CpuDevice(id=0)], GPU is not detected
```

**Common causes:**
- JAX installed without CUDA support (`pip install jax` instead of
  `pip install jax[cuda12]`)
- CUDA version mismatch (check `nvidia-smi` vs JAX requirements)
- Missing `module load cuda/12.2` on M3 cluster
- `XLA_FLAGS` or `CUDA_VISIBLE_DEVICES` set incorrectly

**Fix:**
```bash
# Check CUDA availability
nvidia-smi

# Reinstall JAX with correct CUDA version
pip install --upgrade "jax[cuda12]"

# Force GPU backend (errors if GPU unavailable)
JAX_PLATFORMS=cuda python scripts/13_fit_bayesian.py --model wmrl_m3 --use-pscan
```

### Out of memory (OOM)

**Symptom:** `XlaRuntimeError: RESOURCE_EXHAUSTED`

**Common causes:**
- Too many participants in vmap batch
- Large number of blocks (>20 per participant)
- M4 LBA with float64 (doubles memory)

**Fixes:**
```bash
# Reduce chain count
python scripts/13_fit_bayesian.py --model wmrl_m3 --use-pscan --chains 2

# Limit memory pre-allocation (JAX allocates 75% of GPU by default)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python scripts/13_fit_bayesian.py ...

# Use memory-efficient scan (sequential fallback)
python scripts/13_fit_bayesian.py --model wmrl_m3  # without --use-pscan
```

### Float precision issues

**Choice-only models (M1-M3, M5, M6a, M6b):** Use float32 throughout.
No precision issues expected.

**M4 LBA:** Requires float64. If you see NaN or inf in log-likelihoods:
```bash
# Force float64 globally
JAX_ENABLE_X64=1 python scripts/12_fit_mle.py --model wmrl_m4
```

### PScan NLL disagreement

If the pscan and sequential NLL values disagree by more than 1e-4
relative tolerance:

1. **Check alpha approximation:** The pscan uses reward-based alpha
   selection (`r==1` -> `alpha_pos`) instead of delta-sign. For extreme
   alpha values (>0.9), small disagreements are expected. See
   [PARALLEL_SCAN_LIKELIHOOD.md](PARALLEL_SCAN_LIKELIHOOD.md) Section 5.

2. **Check data padding:** Ensure `masks_stacked` correctly marks padded
   trials. Incorrect masking can cause divergent Q/WM trajectories.

3. **Run the benchmark** to confirm agreement on synthetic data:
   ```bash
   python validation/benchmark_parallel_scan.py --model wmrl_m3 --n-repeats 5
   ```

### Slow compilation

First-call JIT compilation takes ~650ms for pscan variants (vs ~250ms for
sequential). This is a one-time cost per model per session. To minimize
impact:

- Use `--warmup` with sufficient samples (JIT happens during warmup)
- Avoid restarting Python processes between chains
- Pre-compile by running a single likelihood evaluation before MCMC

---

*Last updated: 2026-04-14*
*Phases: 19 (associative scan) + 20 (vectorized policy)*
