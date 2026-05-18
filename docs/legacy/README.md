# docs/legacy/ — Archived Documentation

Superseded or closed-out documents retained for history. Do not use these as
references for current code.

| File | Reason for archival | Replacement |
|---|---|---|
| JAX_GPU_BAYESIAN_FITTING.md | Early JAX/GPU setup notes from Phase 13-14 — superseded by the operational writeup in CLUSTER_GPU_LESSONS.md | ../CLUSTER_GPU_LESSONS.md |
| CONVERGENCE_ASSESSMENT.md | Standalone convergence-diagnostics reference — content now covered by 04_methods/README.md §4.1 (merged from HIERARCHICAL_BAYESIAN.md) | ../04_methods/README.md#hierarchical-bayesian-architecture |
| HIERARCHICAL_BAYESIAN.md | Hierarchical Bayesian architecture doc — content merged into 04_methods/README.md (Phase 29 Plan 02) | ../04_methods/README.md#hierarchical-bayesian-architecture |
| SCALES_AND_FITTING_AUDIT.md | LEC-5/IES-R distributions + fitting audit — content merged into 04_methods/README.md (Phase 29 Plan 02) | ../04_methods/README.md#scales-orthogonalization-and-audit |
| K_PARAMETERIZATION.md | Working-memory capacity K ∈ [2, 6] decision record — content merged into 03_methods_reference/MODEL_REFERENCE.md section 12 (Phase 29 Plan 02) | ../03_methods_reference/MODEL_REFERENCE.md#k-parameterization |
| DEER_NONLINEAR_PARALLELIZATION.md | Phase 20-01 investigation (NO-GO decision locked) | Decision recorded in PARALLEL_SCAN_LIKELIHOOD.md (also archived here) |
| PARTICIPANT_EXCLUSIONS.md | N=48/54 exclusion list from pre-v4.0 sample — superseded by v4.0 canonical cohort (N=154) in config.get_analysis_cohort() | ../04_methods/README.md, manuscript/paper.qmd §Exclusions |
| PARALLEL_SCAN_LIKELIHOOD.md | Phase 19-20 associative-scan implementation guide — pscan slower on CPU for T<1000, standard lax.scan retained | ../CLUSTER_GPU_LESSONS.md for current GPU guidance |
| PARALLEL_SCAN_RLWM_LESSONS.md | Phase 19-20 lessons-learned (AR(1) formulation, phantom-recurrence insight, DEER no-go) | Lessons incorporated into fitting code comments |
| PLOTTING_REFERENCE.md | Posterior visualization reference for scripts/legacy/visualization/ tools — superseded by scripts/06_fit_analyses/ | scripts/06_fit_analyses/ for current figure generation |
