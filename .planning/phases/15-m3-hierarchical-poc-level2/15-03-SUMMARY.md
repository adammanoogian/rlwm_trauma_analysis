---
phase: 15-m3-hierarchical-poc-level2
plan: 03
subsystem: bayesian-pipeline
requires:
  - "15-01"
  - "15-02"
provides:
  - "Absorbed into Phases 16, 18, and 21 — deliverable-by-deliverable mapping below"
affects:
  - "16-choice-only-family-extension-subscale-level-2"
  - "18-integration-comparison-manuscript"
  - "21-principled-bayesian-model-selection-pipeline"
key-files:
  created: []
  modified: []
key-decisions:
  - "Plan 15-03 deliverables absorbed rather than shipped as a standalone plan — downstream phases superseded the need for a separate execution"
duration: "absorbed (no standalone execution)"
completed: "2026-04-19 (closure audit resolution — actual absorption occurred incrementally across Phases 16, 18, and 21)"
status: "absorbed"
---

# Phase 15 Plan 03: Absorption Summary

Plan 15-03 was drafted to deliver PPC stratified by trauma group (HIER-09),
an M3 cluster SLURM script, and end-to-end pipeline validation with a
human-verification checkpoint for real-data fit results.

Its deliverables were absorbed into three downstream phases rather than
shipped as a standalone plan. No code was committed under plan 15-03 itself.
The plan is preserved at `15-03-PLAN.md` as the historical record of what
was originally scoped.

## Deliverable-by-Deliverable Absorption Mapping

| 15-03 Deliverable | Absorbed Into | Evidence (grep invariant) |
|---|---|---|
| HIER-09: PPC infrastructure stratified by trauma group | Phase 15 plans 01-02 + Phase 18 integration | `grep -n "def run_posterior_predictive_check" scripts/fitting/bayesian_diagnostics.py` returns line 631 |
| M3 cluster SLURM script | Phase 15 plan 02 Task 1d + Phase 21 step 21.3 | `ls cluster/13_bayesian_m3.slurm` succeeds; `ls cluster/21_3_fit_baseline.slurm` succeeds |
| End-to-end pipeline validation | Phase 21 master orchestrator `cluster/21_submit_pipeline.sh` | `grep -c "afterok" cluster/21_submit_pipeline.sh` returns 19 |
| Level-2 beta_lec_kappa credible interval | Phase 15 plans 01-02 (model + gate) + Phase 21 plan 21-07 (winner L2 refit) | `grep -n "beta_lec_kappa.*numpyro.sample" scripts/fitting/numpyro_models.py` returns line 1238 |

## What Was Completed Under Plans 15-01 and 15-02

Plan 15-01 delivered the `wmrl_m3_hierarchical_model` function (HIER-01), the
`test_smoke_dispatch` pytest (HIER-10), and the L2-01 `beta_lec_kappa` parameter
sampling. Plan 15-02 delivered `run_inference_with_bump` (HIER-07), the convergence
gate with file-write refusal, `compute_shrinkage_report` and `write_shrinkage_report`
(HIER-08), and WAIC/LOO infrastructure.

These plans fulfilled the core requirements. The checkpoint:human-verify in plan
15-03 required cluster execution (N=154 real data), which was deferred and
superseded by the Phase 21 baseline-fit pipeline (step 21.3) that fits all six
choice-only hierarchical models end-to-end.

## Why No Standalone Execution Occurred

By the time Phase 16 extended the hierarchical infrastructure to all five
choice-only models and Phase 21 designed the full 9-step pipeline, running
plan 15-03's M3-only cluster job as a standalone checkpoint became redundant.
The Phase 21 pipeline (cluster/21_submit_pipeline.sh) subsumes step 21.3
(M3 baseline fit among all models) with a stronger convergence gate and
end-to-end wire-up. Executing plan 15-03 independently would have produced
a subset of the Phase 21 deliverables without the surrounding integration.

## Historical Record

`15-03-PLAN.md` is preserved and documents:
- Original scope: PPC, SLURM script, end-to-end human-verify checkpoint
- Success criteria: PPC 18/21 blocks covered, beta_lec_kappa HDI excludes zero
- The checkpoint:human-verify gate that was never triggered

The plan is NOT deleted; it serves as the planning record for this absorption
decision and for future readers tracing the Phase 15 scope.

---
*Phase: 15-m3-hierarchical-poc-level2*
*Absorption documented: 2026-04-19*
