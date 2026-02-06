# Roadmap: RLWM Trauma Analysis

## Milestones

- ✅ **v1.0 M3 Infrastructure** - Phases 1-3 (shipped 2026-01-30)
- 🚧 **v2.0 Post-Fitting Validation & Publication Readiness** - Phases 4-7 (in progress)

## Phases

<details>
<summary>✅ v1.0 M3 Infrastructure (Phases 1-3) - SHIPPED 2026-01-30</summary>

### Phase 1: Perseveration Extension
**Goal**: Extend WM-RL model with perseveration parameter κ
**Plans**: 2 plans

Plans:
- [x] 01-01: Core M3 implementation
- [x] 01-02: Backward compatibility validation

### Phase 2: MLE Infrastructure
**Goal**: Complete MLE fitting infrastructure for M3
**Plans**: 2 plans

Plans:
- [x] 02-01: MLE parameter bounds and transforms
- [x] 02-02: CLI integration and testing

### Phase 3: Model Comparison
**Goal**: N-model comparison with Akaike weights
**Plans**: 2 plans

Plans:
- [x] 03-01: Comparison framework
- [x] 03-02: Output formatting and validation

</details>

## 🚧 v2.0 Post-Fitting Validation & Publication Readiness

**Milestone Goal:** Validate fitted models through parameter recovery and posterior predictive checks, enhance cluster monitoring, and produce publication-ready model comparison and trauma-association outputs.

### Phase 4: Regression Visualization
**Goal**: Enhanced visualization and organization for continuous regression analysis
**Depends on**: Nothing (first v2 phase, uses existing Scripts 15-16)
**Requirements**: REGR-01, REGR-02, REGR-03
**Success Criteria** (what must be TRUE):
  1. Script 16 output file is structured with clear sections grouping each scale x parameter regression
  2. User can run Scripts 15-16 with `--color-by trauma_group` to see group-colored scatter plots
  3. User can run Scripts 15-16 with `--color-by gender` (or any categorical column) to visualize different groupings
  4. All regression plots display colored data points matching the specified grouping variable
**Plans**: 2 plans

Plans:
- [x] 04-01-PLAN.md — Shared plotting utility + Script 15 (M3, --model, --color-by, full plot coverage)
- [x] 04-02-PLAN.md — Script 16 (M3, --color-by, model subdirectories, structured output)

### Phase 5: Parameter Recovery
**Goal**: Complete parameter recovery pipeline validating MLE fitting quality per Senta et al. (2025)
**Depends on**: Nothing (independent infrastructure work)
**Requirements**: RECV-01, RECV-02, RECV-03, RECV-04, RECV-05, RECV-06
**Success Criteria** (what must be TRUE):
  1. User can run `python scripts/fitting/model_recovery.py --model wmrl_m3 --n-subjects 50 --n-datasets 10` from command line
  2. Model recovery generates synthetic data from sampled parameters, fits via MLE, and collects recovered parameters
  3. Recovery metrics (Pearson r, RMSE, bias) are computed and displayed for each parameter
  4. Scatter plots are generated showing true vs. recovered parameters with r-squared annotations
  5. Recovery results CSV contains true parameters, recovered parameters, and metrics in structured format
  6. Script 11 invokes recovery pipeline and reports pass/fail against r >= 0.80 criterion
**Plans**: 3 plans

Plans:
- [ ] 05-01-PLAN.md — Core recovery pipeline (synthetic data generator + recovery loop + metrics)
- [ ] 05-02-PLAN.md — CLI, output, and visualization (argparse CLI + CSV output + scatter/KDE plots)
- [ ] 05-03-PLAN.md — Script 11 wrapper + end-to-end verification

### Phase 6: Cluster Monitoring
**Goal**: GPU utilization monitoring and memory checkpoint persistence for cluster execution
**Depends on**: Nothing (cluster infrastructure improvements)
**Requirements**: MNTR-01, MNTR-02
**Success Criteria** (what must be TRUE):
  1. GPU SLURM script runs background nvidia-smi polling, logging utilization to timestamped file
  2. User can configure monitoring interval via SLURM script variable
  3. fit_mle.py writes [MEMORY] stdout lines to persistent CSV alongside fit results
  4. Memory CSV contains timestamp, participant_id, model, checkpoint_count columns
**Plans**: TBD

Plans:
- [ ] 06-01: TBD

### Phase 7: Publication Polish
**Goal**: Publication-ready model comparison by group and combined results summary
**Depends on**: Phase 4 (uses regression outputs for summary table)
**Requirements**: PUBL-01, PUBL-02
**Success Criteria** (what must be TRUE):
  1. User can run `14_compare_models.py --by-group` to generate separate AIC/BIC tables per trauma group
  2. By-group comparison identifies winning model separately for each trauma category
  3. Combined results summary table exists showing winning model per group plus key parameter-trauma associations from Scripts 14-16
  4. Summary table is publication-ready (formatted, clear column names, statistical annotations)
**Plans**: TBD

Plans:
- [ ] 07-01: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 4 → 5 → 6 → 7

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Perseveration Extension | v1.0 | 2/2 | Complete | 2026-01-30 |
| 2. MLE Infrastructure | v1.0 | 2/2 | Complete | 2026-01-30 |
| 3. Model Comparison | v1.0 | 2/2 | Complete | 2026-01-30 |
| 4. Regression Visualization | v2.0 | 2/2 | Complete | 2026-02-06 |
| 5. Parameter Recovery | v2.0 | 0/3 | Planned | - |
| 6. Cluster Monitoring | v2.0 | 0/TBD | Not started | - |
| 7. Publication Polish | v2.0 | 0/TBD | Not started | - |
