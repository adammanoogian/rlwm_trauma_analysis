---
phase: 20-deer-nonlinear-parallelization
verified: 2026-04-14T15:10:00Z
status: passed
score: 5/5 must-haves verified
gaps: []
---

# Phase 20: DEER Non-Linear Parallelization Verification Report

**Phase Goal:** Investigate DEER-style fixed-point iteration approach to parallelize remaining non-linear RLWM likelihood components. Research phase: outcome may be no-go (which it was) or go. If no-go, document why with empirical evidence and update docs/benchmark.

**Verified:** 2026-04-14T15:10:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Research document covers all 5 required subsections: (a) DEER algorithm, (b) convergence analysis, (c) alternatives comparison, (d) Unifying Framework, (e) go/no-go recommendation | VERIFIED | `docs/DEER_NONLINEAR_PARALLELIZATION.md` (449 lines). Section headers at lines 38 (DEER Algorithm), 166 (Convergence Analysis), 213 (Alternative Approaches Comparison), 257 (Unifying Framework Perspective), 315 (Go/No-Go Recommendation). All substantive with worked examples, tables, and citations. |
| 2 | NO-GO on DEER documented with empirical evidence; GO on vectorized policy alternative | VERIFIED | Lines 317-377: NO-GO with 4 supporting arguments (code inspection, discrete state, complexity overhead, CPU target platform). References Phase 19 benchmark data (0.26x speedup). Lines 349-376: GO on vectorized policy with justification. |
| 3 | All 12 pscan likelihood variants fully vectorized (no sequential Phase 2 lax.scan for policy) | VERIFIED | 6 block + 6 multiblock pscan functions confirmed. Zero occurrences of `policy_step` in jax_likelihoods.py. Block functions use `jax.vmap(softmax_policy, ...)` and array broadcasting. Multiblock wrappers delegate to block functions. Perseveration precomputed via `precompute_last_action_global` (M3, M5, M6b) and `precompute_last_actions_per_stimulus` (M6a, M6b). |
| 4 | JAX_GPU_BAYESIAN_FITTING.md created/updated with Phase 20 findings | VERIFIED | `docs/JAX_GPU_BAYESIAN_FITTING.md` (432 lines, 8 sections). Section 6 "DEER Investigation Summary" covers NO-GO with reasons and GO on vectorized policy. Cross-references both DEER and PARALLEL_SCAN docs. |
| 5 | Reproducible benchmark exists in `validation/benchmark_parallel_scan.py` with Phase 20 metadata | VERIFIED | 548 lines. Docstring references Phase 20 vectorized policy. JSON output includes `"phase": "phase_20_vectorized"`. Existing benchmark JSON at `output/bayesian/pscan_benchmark.json` confirms infrastructure works (CPU, JAX 0.4.31). |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `docs/DEER_NONLINEAR_PARALLELIZATION.md` | DEER research document with 5 subsections | VERIFIED (449 lines, no stubs) | All sections substantive with algorithm detail, convergence analysis, comparison table, worked examples, citations |
| `scripts/fitting/jax_likelihoods.py` | Precomputation functions + vectorized pscan block functions | VERIFIED (5507 lines) | `precompute_last_action_global` at line 589, `precompute_last_actions_per_stimulus` at line 643. 6 block pscan functions vectorized. |
| `scripts/fitting/tests/test_pscan_likelihoods.py` | Precomputation tests + vectorized agreement tests | VERIFIED (1407 lines) | TestPrecomputeLastActionGlobal (3 tests), TestPrecomputeLastActionsPerStimulus (2 tests), TestPrecomputeAgreesWithScan (2 tests), TestVectorizedPhase2 (4+ tests) |
| `docs/JAX_GPU_BAYESIAN_FITTING.md` | GPU Bayesian fitting guide with Phase 20 findings | VERIFIED (432 lines, no stubs) | 8 sections including DEER summary, model-specific notes, troubleshooting |
| `docs/PARALLEL_SCAN_LIKELIHOOD.md` | Updated with Phase 20 vectorized policy section | VERIFIED (451 lines) | Section 7 "Phase 20: Vectorized Policy Pass" at line 316. Architecture table shows all 6 models vectorized. |
| `validation/benchmark_parallel_scan.py` | Benchmark script with Phase 20 metadata | VERIFIED (548 lines, no stubs) | Updated docstring, console header, JSON output fields reference Phase 20 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| M3 pscan block (line 3837) | precompute_last_action_global | Direct call | WIRED | Global perseveration precomputed from observed actions |
| M5 pscan block (line 4084) | precompute_last_action_global | Direct call | WIRED | Same pattern as M3 |
| M6a pscan block (line 4277) | precompute_last_actions_per_stimulus | Direct call | WIRED | Per-stimulus perseveration precomputed |
| M6b pscan block (line 4470-4471) | Both precompute functions | Direct calls | WIRED | Dual perseveration uses both global and per-stimulus |
| Multiblock pscan (e.g., line 3893) | Block pscan functions | lax.scan over blocks | WIRED | Multiblock wrappers delegate to vectorized block functions |
| DEER doc (line 5) | PARALLEL_SCAN_LIKELIHOOD.md | Cross-reference link | WIRED | Bidirectional cross-references confirmed |
| GPU guide (lines 13-16) | Both PARALLEL_SCAN and DEER docs | Cross-reference links | WIRED | Three-way cross-referencing confirmed |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | -- | -- | -- | No TODO, FIXME, placeholder, or stub patterns found in any Phase 20 artifact |

### Human Verification Required

### 1. GPU Benchmark Execution

**Test:** Run `python validation/benchmark_parallel_scan.py` on the Monash M3 cluster with an A100 GPU to measure actual GPU speedup of vectorized pscan variants.
**Expected:** Speedup ratio > 1.0x on GPU (vs CPU sequential baseline). JSON output to `output/bayesian/pscan_benchmark_gpu.json`.
**Why human:** Requires GPU hardware (A100) and cluster access. CPU benchmark exists (0.26x = slowdown on CPU, which is expected for T=100 with scan overhead), but the GPU result is the motivating use case.

### 2. Numerical Agreement Under MCMC

**Test:** Run M3 hierarchical fit with `--use-pscan` on cluster and compare posterior means with sequential fit.
**Expected:** Group-level posterior means agree within 5% relative error; WAIC/LOO agree within 1.0.
**Why human:** Requires full MCMC run on cluster (hours of compute). Structural verification confirms vectorized functions call the same underlying operations, but MCMC noise and compilation differences could introduce discrepancies.

### Gaps Summary

No gaps found. All 5 must-haves verified against actual codebase artifacts:

1. The DEER research document is a substantial 449-line analysis covering all 5 required subsections with worked examples, convergence analysis, comparison tables, and explicit NO-GO recommendation supported by 4 empirical arguments.

2. The NO-GO decision is well-justified: the Phase 2 sequential dependency was correctly identified as a phantom (actions are observed data, not model outputs), making DEER inapplicable. The vectorized policy alternative was implemented instead.

3. All 12 pscan likelihood variants are genuinely vectorized -- `policy_step` is completely gone, replaced by `jax.vmap` and array broadcasting. Perseveration precomputation functions correctly handle global (M3/M5/M6b) and per-stimulus (M6a/M6b) cases.

4. Documentation ecosystem is complete with three-way cross-referencing between DEER doc, GPU guide, and parallel scan doc.

5. The benchmark script exists with Phase 20 metadata and an existing benchmark JSON from Phase 19 demonstrates the infrastructure works.

---

*Verified: 2026-04-14T15:10:00Z*
*Verifier: Claude (gsd-verifier)*
