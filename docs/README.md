# Documentation Index

```
docs/
├── 01_project_protocol/
│   ├── PARTICIPANT_EXCLUSIONS.md        # Exclusion criteria (superseded by HIERARCHICAL_BAYESIAN.md §4)
│   └── plotting_config_guide.md         # Figure styling (PlotConfig classes)
│
├── 02_pipeline_guide/
│   ├── ANALYSIS_PIPELINE.md             # Full pipeline walkthrough (scripts 01-16)
│   └── PLOTTING_REFERENCE.md            # ArviZ posterior visualization
│
├── 03_methods_reference/
│   ├── MODEL_REFERENCE.md               # Model math (Q-learning, WM-RL, priors)
│   ├── TASK_AND_ENVIRONMENT.md          # Task structure, environment API
│   └── references/                      # Key papers (Senta et al., 2025)
│
├── HIERARCHICAL_BAYESIAN.md             # Hierarchical architecture + validation checklist
├── SCALES_AND_FITTING_AUDIT.md          # LEC-5/IES-R distributions + fitting audit vs literature
├── CLUSTER_GPU_LESSONS.md               # JAX/NumPyro/GPU/SLURM lessons and pitfalls
├── PARALLEL_SCAN_LIKELIHOOD.md          # pscan architecture deep-dive
├── JAX_GPU_BAYESIAN_FITTING.md          # early JAX/GPU setup notes (superseded partly by CLUSTER_GPU_LESSONS)
├── K_PARAMETERIZATION.md                # Working-memory capacity K ∈ [2, 6] decision record
├── CONVERGENCE_ASSESSMENT.md            # Convergence diagnostics reference
├── DEER_NONLINEAR_PARALLELIZATION.md    # DEER investigation (Phase 20-01; NO-GO decision)
│
└── legacy/                              # Archived docs (do not use)
```

**Start here:**
- [ANALYSIS_PIPELINE.md](02_pipeline_guide/ANALYSIS_PIPELINE.md) — full pipeline walkthrough
- [HIERARCHICAL_BAYESIAN.md](HIERARCHICAL_BAYESIAN.md) — Bayesian architecture + validation checklist
- [CLUSTER_GPU_LESSONS.md](CLUSTER_GPU_LESSONS.md) — read before writing new cluster code or debugging a slow MCMC run

**Model details:** [MODEL_REFERENCE.md](03_methods_reference/MODEL_REFERENCE.md) — read before modifying fitting code.

**Before a refit:** [SCALES_AND_FITTING_AUDIT.md](SCALES_AND_FITTING_AUDIT.md) — scale usage, prior choices, fitting-procedure audit.
