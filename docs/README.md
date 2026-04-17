# Documentation Index

```
docs/
├── 01_project_protocol/
│   ├── PARTICIPANT_EXCLUSIONS.md        # Exclusion criteria (superseded by HIERARCHICAL_BAYESIAN.md §4)
│   └── plotting_config_guide.md         # Figure styling (PlotConfig classes)
│
├── 02_pipeline_guide/
│   ├── ANALYSIS_PIPELINE.md             # Full pipeline walkthrough (scripts 01-18)
│   └── PLOTTING_REFERENCE.md            # All scripts/visualization/* tools
│
├── 03_methods_reference/
│   ├── MODEL_REFERENCE.md               # Model math (Q-learning, WM-RL, priors)
│   ├── TASK_AND_ENVIRONMENT.md          # Task structure, environment API
│   └── references/                      # Key papers (Senta et al., 2025)
│
├── 04_methods/
│   └── README.md                        # Methods index (published + validation; see for 09-11 outputs)
│
├── 04_results/
│   └── README.md                        # Pipeline results index (all artifacts incl. orphaned/supplementary)
│
├── HIERARCHICAL_BAYESIAN.md             # Hierarchical architecture + validation checklist
├── SCALES_AND_FITTING_AUDIT.md          # LEC-5/IES-R distributions + fitting audit vs literature
├── CLUSTER_GPU_LESSONS.md               # JAX/NumPyro/GPU/SLURM lessons and pitfalls
├── PARALLEL_SCAN_LIKELIHOOD.md          # pscan architecture deep-dive
├── K_PARAMETERIZATION.md                # Working-memory capacity K ∈ [2, 6] decision record
│
└── legacy/                              # Archived superseded docs — see docs/legacy/README.md
```

**Start here:**
- [ANALYSIS_PIPELINE.md](02_pipeline_guide/ANALYSIS_PIPELINE.md) — full pipeline walkthrough
- [HIERARCHICAL_BAYESIAN.md](HIERARCHICAL_BAYESIAN.md) — Bayesian architecture + validation checklist
- [CLUSTER_GPU_LESSONS.md](CLUSTER_GPU_LESSONS.md) — read before writing new cluster code or debugging a slow MCMC run

**Model details:** [MODEL_REFERENCE.md](03_methods_reference/MODEL_REFERENCE.md) — read before modifying fitting code.

**Before a refit:** [SCALES_AND_FITTING_AUDIT.md](SCALES_AND_FITTING_AUDIT.md) — scale usage, prior choices, fitting-procedure audit.
