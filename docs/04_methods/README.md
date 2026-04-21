# docs/04_methods/ — Methods Documentation Index

Methods notes for published analyses AND supplementary/validation analyses
that do not appear in manuscript/paper.qmd. Each entry points to the
producing script and a short method writeup. Populate entries as new
methods are added or old ones documented.

## Published-in-paper methods

| Topic | Producing script | Method doc |
|---|---|---|
| Task structure and environment | src/rlwm/envs/rlwm_env.py | ../03_methods_reference/TASK_AND_ENVIRONMENT.md |
| Model mathematics | scripts/fitting/jax_likelihoods.py | ../03_methods_reference/MODEL_REFERENCE.md |
| Hierarchical Bayesian architecture | scripts/fitting/numpyro_models.py | ../HIERARCHICAL_BAYESIAN.md |
| Scale orthogonalization (IES-R) | scripts/fitting/level2_design.py | ../SCALES_AND_FITTING_AUDIT.md |

## Supplementary / validation methods

| Topic | Producing script | Method doc |
|---|---|---|
| Posterior predictive checks | scripts/simulations_recovery/09_run_ppc.py | _TODO_ |
| Synthetic-data generation | scripts/simulations_recovery/09_generate_synthetic_data.py | _TODO_ |
| Parameter sweep | scripts/simulations_recovery/10_run_parameter_sweep.py | _TODO_ |
| Parameter recovery | scripts/simulations_recovery/11_run_model_recovery.py | _TODO_ |
| Posterior-vs-MLE sanity check | validation/compare_posterior_to_mle.py | _TODO_ |

Entries marked _TODO_ are scaffolding. Add short method writeups here
as results are produced or as reviewers ask for them.
