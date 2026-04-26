# Phase 24 Canary Acceptance Report

- **Canary timestamp (UTC):** 20260425T193756Z
- **HEAD commit at submission:** 97aa22093d3a7bc90953e28c3f5185a649de99c1
- **Submit host:** m3-login2
- **STEP / Tag:** prior_predictive / 21_canary
- **Canary JobIDs:** qlearning=54957781, wmrl=54957194, wmrl_m3=54957195, wmrl_m5=54957196, wmrl_m6a=54957197, wmrl_m6b=54957198
- **Report generated (UTC):** 2026-04-26T09:11:47Z

## Four-criteria acceptance gate (CONTEXT.md §Submission strategy)

| Criterion | Verdict | Evidence |
|---|---|---|
| (a) SLURM exit 0 + 6 .nc files | **PASS** | 6 jobs COMPLETED 0:0; all 6 prior_sim.nc files present |
| (b) ArviZ load + dim check | **PASS** | all 6 loaded — qlearning(2), wmrl(2), wmrl_m3(2), wmrl_m5(2), wmrl_m6a(2), wmrl_m6b(2) |
| (c) Autopush commit on main | **PASS** | 6 commit(s) since 2026-04-25 19:37:56 |
| (d) CCDS-canonical path landing | **PASS** | CCDS populated; legacy output/bayesian/21_prior_predictive absent or empty |

## Per-model Baribault gate verdicts (advisory, SOFT_PASS not blocking)

Three-tier policy (scripts/utils/ppc.py:_classify_gate_verdict):

- **HARD_PASS** — within original Baribault & Collins (2023) hard band.
- **SOFT_PASS** — within documented soft margin; advisory; monitor in stage 04b fitting.
- **FAIL** — outside both bands; canary REJECTED.

| Model | JID | Verdict | Median acc | Source |
|---|---|---|---|---|
| qlearning | 54957781 | **SOFT_PASS** | 0.901 | qlearning_gate.md |
| wmrl | 54957194 | **HARD_PASS** | 0.808 | wmrl_gate.md |
| wmrl_m3 | 54957195 | **HARD_PASS** | 0.573 | wmrl_m3_gate.md |
| wmrl_m5 | 54957196 | **HARD_PASS** | 0.437 | wmrl_m5_gate.md |
| wmrl_m6a | 54957197 | **HARD_PASS** | 0.697 | wmrl_m6a_gate.md |
| wmrl_m6b | 54957198 | **HARD_PASS** | 0.614 | wmrl_m6b_gate.md |

## Overall verdict

**APPROVED**

All four cluster→repo flow criteria passed; per-model FAIL count is zero (SOFT_PASS treated as advisory). Wave 1 Task 4 may proceed.

Recommended next step:

```bash
bash cluster/submit_all.sh --from-stage 4   # full chain from stage 04b onwards
```
