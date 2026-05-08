# Phase 34 — Context: WMH (WM + H-agent) follow-up

**Status:** Queued, non-blocking. Provisional. Likely a separate paper rather
than a fix within the current manuscript.
**Date opened:** 2026-05-08
**Origin:** 2026-05-08 conversation — Collins (2024) "RL or not RL?" review
during Phase 33-01 scoping triggered consideration of the WMH (WM + H-agent)
architecture as an alternative to single-alpha RL.

---

## Premise

Collins (2024) reanalyzed 7 RLWM datasets and found:

> "best fitting model across 6 data sets... was a model with fixed r0 = 1,
> such that receiving incorrect feedback led to the same positive prediction
> error as correct feedback would. ... this slow module cannot be
> interpreted as an RL module anymore, as the association weights track a
> relative frequency of stimulus-action choice, irrespective of outcomes,
> rather than an estimated value."

The WMH architecture replaces the RL slow module with an H-agent (Hebbian /
habit-like) using update rule:

```
H_{t+1}(s_t, a_t) = H_t(s_t, a_t) + alpha_H(r_t) * (SR(r_t) - H_t(s_t, a_t))
```

where `SR(1) = 1` (correct outcomes) and `SR(0) = r_0 in [0, 1]` (incorrect
outcomes; free or fixed). With `r_0 = 0`: H reduces to RL. With `r_0 = 1`:
H tracks selection frequency only — pure Hebbian / habit.

Exceedance probability > 0.93 in all 6 deterministic datasets and > 0.99 in
the probabilistic RLWMP dataset.

---

## Why this is Phase 34, not Phase 33

Considered and rejected as a Phase 33 deliverable:

1. **Architectural change**, not a parameterization fix. New likelihood
   functions, new hierarchical wrappers, new parameter recovery, new
   PPC simulators. ~2-4 weeks effort.
2. **Loses comparability with Senta (2025)**, our reference paper. To
   maintain a clean reference comparison we'd need to add Senta's eta
   (negative-feedback neglect) parameterization too — a third architecture
   to compare and validate.
3. **Reframes the manuscript** from "WM-RL with extended perseveration"
   to "WM with Hebbian/habit slow module" — which is a different scientific
   claim than the one this manuscript is making. The current trauma effect
   on kappa_total is preserved either way (kappa is orthogonal to the slow
   module choice), but the discussion / interpretation paragraphs would
   need substantial rewrite.
4. **Phase 33-01 (Option A: drop alpha_neg) already lands the empirical
   improvement** (7/7 ICC vs 7/8) and the literature alignment (Senta,
   Sugawara, Collins all cited). The WMH question becomes "was the right
   parameterization H-agent rather than single-alpha RL?" — which is
   a head-to-head comparison question, not a fix.

---

## Tentative scope (provisional, finalize when greenlit)

**Goal:** Test the WMH parameterization (Collins 2024) head-to-head against
single-alpha RL (Phase 33-01 baseline) on the trauma dataset, and against
Senta's eta parameterization. Decide via WAIC / LOO / exceedance probability
which architecture best describes the slow learning process in our task.

**Depends on:** Phase 33-01 (single-alpha baseline must exist as the
reference point).

**Hypothetical plans (4-5):**

1. **34-01: Implement H-agent likelihood** — new `src/rlwm/fitting/core.py:
   associative_scan_h_update` function (mirror of `associative_scan_q_update`
   with `target = jnp.where(r == 1, 1.0, r0)` replacing the literal reward
   in the delta). Add new model files `src/rlwm/fitting/models/wmh.py`,
   `wmh_m3.py`, `wmh_m6b.py` mirroring the wmrl variants. ~10 lines of new
   core code + ~6 model files of mostly-copied wrapper code.

2. **34-02: Implement Senta eta parameterization** — for fair three-way
   comparison. Add `--eta` flag to existing model files; eta multiplicatively
   reduces both RL and WM updates following negative outcomes. Cleaner than
   adding a third model family.

3. **34-03: Parameter recovery** — full recovery study for both H-agent
   and eta variants on synthetic data; verify identifiability of `r0`,
   `eta`, `alpha_H`.

4. **34-04: Three-way fit + comparison** — refit M6b under three
   architectures (single-alpha RL, H-agent, eta-RL) on the trauma dataset.
   Compare via LOO, WAIC, exceedance probability. Report kappa_total x LEC
   effect under each architecture.

5. **34-05: Manuscript rewrite (or separate paper)** — depending on the
   34-04 result. If H-agent or eta wins decisively, decide whether to
   rewrite this manuscript's framing or carve out a separate
   methodological paper. User decision point.

**Out of scope for Phase 34:**
- Changes to the WM module (stays one-shot overwrite).
- Changes to perseveration (kappa_total / kappa_share / kappa_s preserved).
- Neural-substrate analysis (no fMRI / EEG in this dataset).

---

## Side-by-side: H-agent vs single-alpha RL (math reference)

```
RL slow module (Phase 33-01 baseline):
  delta_t = r_t - Q_t(s_t, a_t)
  alpha_t = alpha                         # single learning rate
  Q_{t+1}(s_t, a_t) = Q_t(s_t, a_t) + alpha_t * delta_t

H-agent slow module (Phase 34 candidate):
  SR(r_t) = 1.0       if r_t == 1.0
            r_0       if r_t == 0.0      # parameter, fixed at 1 in Collins's winner
  delta_t = SR(r_t) - H_t(s_t, a_t)
  alpha_t = alpha_H                       # Collins also has bias_H multiplying alpha_H
                                          #   when r_t == 0; for parsimony fix bias_H = 1
  H_{t+1}(s_t, a_t) = H_t(s_t, a_t) + alpha_t * delta_t
```

The implementation difference is one line in the delta — `r_t` becomes
`jnp.where(r_t == 1.0, 1.0, r_0)`. Everything else (WM module, hybrid policy,
kappa, epsilon) is preserved. M6b -> M6b-WMH is a one-parameter swap, not an
architectural rewrite.

Diagram of the spectrum:

```
r0 = 0:  pure RL    (Q tracks expected value; delta uses literal reward)
r0 ~ 0.5: hybrid    (Q tracks discounted value; partial habit)
r0 = 1:  pure H     (H tracks selection frequency; delta is direction-only)
```

The free-r0 model lets the data choose where it sits.

---

## Decision deferral

Phase 34 will not be planned in detail until Phase 33-01 lands and the
manuscript is in a stable state. If user decides to ship the manuscript
with Phase 33-01 alone (single-alpha RL + Collins citation), Phase 34
becomes a future paper opportunity rather than an active effort.

---

## References

- Collins, A. G. E. (2024). RL or not RL? Parsing the processes that
  support human reward-based learning. (Preprint;
  `Downloads/RLorNotRL_Collins2024.pdf`.)
- Senta, I., Rmus, M., Hartley, C. A., & Collins, A. G. E. (2025).
  Working memory and reinforcement learning interactions in human
  decision-making. *PLOS Computational Biology*, 21(9), e1012872.
- Toyama, A., Katahira, K., & Kunisato, Y. (2023). Examinations of
  biases by model misspecification and parameter reliability of
  reinforcement learning models. *Computational Brain & Behavior*, 6(4),
  651-670.
