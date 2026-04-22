# Documentation Index

```
docs/
├── 01_project_protocol/
│   ├── PARTICIPANT_EXCLUSIONS.md        # Exclusion criteria (superseded by 04_methods/README.md §4.1)
│   └── plotting_config_guide.md         # Figure styling (PlotConfig classes)
│
├── 02_pipeline_guide/
│   ├── ANALYSIS_PIPELINE.md             # Full pipeline walkthrough (scripts 01-18)
│   └── PLOTTING_REFERENCE.md            # All scripts/visualization/* tools
│
├── 03_methods_reference/
│   ├── MODEL_REFERENCE.md               # Model math (Q-learning, WM-RL, priors, K parameterization)
│   ├── TASK_AND_ENVIRONMENT.md          # Task structure, environment API
│   └── references/                      # Key papers (Senta et al., 2025)
│
├── 04_methods/
│   └── README.md                        # Methods index + Bayesian architecture + scales audit
│
├── 04_results/
│   └── README.md                        # Pipeline results index (all artifacts incl. orphaned/supplementary)
│
├── CLUSTER_GPU_LESSONS.md               # JAX/NumPyro/GPU/SLURM lessons and pitfalls
├── PARALLEL_SCAN_LIKELIHOOD.md          # pscan architecture deep-dive
│
└── legacy/                              # Archived superseded docs — see docs/legacy/README.md
```

**Start here:**
- [ANALYSIS_PIPELINE.md](02_pipeline_guide/ANALYSIS_PIPELINE.md) — full pipeline walkthrough
- [04_methods/README.md](04_methods/README.md#hierarchical-bayesian-architecture) — Bayesian architecture + validation checklist
- [CLUSTER_GPU_LESSONS.md](CLUSTER_GPU_LESSONS.md) — read before writing new cluster code or debugging a slow MCMC run

**Model details:** [MODEL_REFERENCE.md](03_methods_reference/MODEL_REFERENCE.md) — read before modifying fitting code.

**Before a refit:** [04_methods/README.md](04_methods/README.md#scales-orthogonalization-and-audit) — scale usage, prior choices, fitting-procedure audit.
