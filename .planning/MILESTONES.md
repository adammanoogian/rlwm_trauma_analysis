# Project Milestones: WM-RL M3 (Perseveration Extension)

## v1 M3 Infrastructure (Shipped: 2026-01-30)

**Delivered:** Complete MLE fitting infrastructure for WM-RL M3 model with perseveration parameter (κ), enabling dissociation of outcome-insensitive action repetition from learning-rate effects.

**Phases completed:** 1-3 (6 plans total)

**Key accomplishments:**

- Implemented JAX likelihood functions (`wmrl_m3_block_likelihood()`, `wmrl_m3_multiblock_likelihood()`) with κ·Rep(a) perseveration term
- Extended WMRLHybridAgent with optional kappa parameter, maintaining exact M2 backward compatibility
- Added complete MLE infrastructure (WMRL_M3_BOUNDS, WMRL_M3_PARAMS, `--model wmrl_m3` CLI)
- Created 24+ backward compatibility tests validating M3(κ=0) ≡ M2 to rtol=1e-5
- Fixed critical backward compatibility bug: M3 now branches on κ=0 for M2 probability mixing
- Extended compare_mle_models.py for N-model comparison with Akaike weights

**Stats:**

- 26 files created/modified
- 4,368 lines in key source files
- 3 phases, 6 plans, ~15 tasks
- 2 days from start to ship (2026-01-29 → 2026-01-30)

**Git range:** `d4647a9` → `98a13b2`

**What's next:** Run M3 fits on cluster data, perform M1/M2/M3 model comparison, analyze κ parameter in trauma populations.

---
