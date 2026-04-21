# Phase 19: Associative Scan Likelihood Parallelization - Context

**Gathered:** 2026-04-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace O(T) sequential `lax.scan`/`lax.fori_loop` in RLWM likelihood evaluation with O(log T) `jax.lax.associative_scan` for the linear-recurrence components (Q-value updates and WM forgetting). Benchmark against CPU baseline from Phases 15-16. Non-linear components (WM-Q mixing, softmax) stay sequential — that's Phase 20 (DEER).

</domain>

<decisions>
## Implementation Decisions

### Research depth
- **Implementation guide only** — not academic supplementary material, not future-me deep reference
- Non-RLWM references (PaMoRL, S4/Mamba, parallel Kalman): **one paragraph + citation each** explaining relevance to what we're doing. No worked examples for those.
- RLWM-specific content: **full worked examples** for the AR(1) reformulation and associative operator on actual RLWM equations
- **WM overwrite: formal treatment required** — derive how the hard `WM(s,a) ← r` overwrite can be encoded as a multiplicative reset within the associative scan operator. This is the tricky implementation detail.
- **Numerical stability analysis included** — document edge cases when α near 0/1 and φ near 0/1, and their impact on agreement between parallel and sequential implementations. Informs tolerance thresholds.

### Integration strategy
- **Separate functions** — keep sequential implementations untouched, add `*_pscan` variant functions alongside. No risk to existing Phases 15-17 fits.
- **Same STACKED_MODEL_DISPATCH, flag argument** — model functions accept a `use_pscan` kwarg. Minimal change to dispatch machinery. No separate PSCAN_MODEL_DISPATCH dict.
- **Two-stage benchmark** — (1) standalone micro-benchmark in `validation/benchmark_parallel_scan.py` for fast iteration, then (2) wire `--use-pscan` into `13_fit_bayesian.py` for full A/B MCMC comparison (success criterion 6).
- **All 6 choice-only models** — not just M3. If the scan operator is generic (same Q/WM recurrence structure), porting to M1, M2, M5, M6a, M6b is low marginal effort. Proves model-agnostic approach. M3 is the primary benchmark target; others get unit tests + smoke tests.

### Claude's Discretion
- Exact associative operator implementation (tuple structure, carry format)
- Whether to use a single generic scan operator or per-model variants
- Benchmark iteration count and warm-up strategy
- SLURM script configuration for GPU benchmark

</decisions>

<specifics>
## Specific Ideas

- The Q-update recurrence `Q_t = (1-α)Q_{t-1} + α*r_t` maps directly to `x_t = a_t * x_{t-1} + b_t` with `a_t = 1-α`, `b_t = α*r_t`
- The WM forgetting recurrence `WM_t = φ·WM_{t-1} + (1-φ)/nA` is the same form, but the hard overwrite `WM(s,a) ← r` on feedback trials breaks the recurrence — needs a reset mechanism inside the scan operator
- The non-linear parts (WM-Q mixing weight `w = ρ/(ρ + set_size)` → combined policy → softmax → log-prob) remain as a sequential post-scan pass in Phase 19

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 19-associative-scan-likelihood*
*Context gathered: 2026-04-14*
