# Phase 28 Context — Scope & Sequencing Decision

**Decided:** 2026-04-21 via `/gsd:plan-phase 28` orchestrator checkpoint.

---

## Sequencing Decision (Gate 1 — resolves ROADMAP.md "Sequencing Consideration")

**Chosen:** Option A-modified.

**Execution order:** 23 → **28** → 24 → 25 → 26 → 27.

(Phase 28 executes *before* Phase 24, not before Phase 26 as the roadmap's original Option A suggested.)

**Why this order and not numeric (Option C):**
- Phase 24 submits `cluster/21_submit_pipeline.sh` (the master pipeline orchestrator). That pipeline is a load-bearing structural artifact — per-model SLURM templates, output paths, figure paths are all hard-coded into it.
- If Phase 24 runs against the *old* structure and Phase 28 consolidates afterward, all Phase 24 artifacts need to be re-produced against the new layout → duplicate cluster-time cost.
- User is currently GPU-constrained ("we don't have the GPU cluster currently"), so running Phase 24 now is blocked anyway. Use the blocked window to consolidate.
- Result: when GPU access returns, one `sbatch cluster/<new-orchestrator>.sh` runs the entire pipeline against the final structure and artifacts land in the canonical paths the paper already references.

**Why not Option B (spawn new milestone):**
- Repo consolidation benefits v5.0's closure just as much as a future milestone — closure guard and manuscript both become simpler against a clean layout.
- Pushing this to v5.1 defers the very simplification that makes v5.0 easier to ship.

**Numbering:** Phase 28 keeps its numeric slot (28). Execution order ≠ numeric order for v5.0 only. STATE.md should document this (phases 24–27 still listed in numeric order, with a note that phase 28 executes first).

---

## Scope Decision (Gate 3 — bounds the planner)

**Chosen:** Repo-consolidation + paper.qmd structural scaffolding. No fit execution. No populating real numbers/figures.

### In scope for Phase 28

1. **`src/` consolidation** — authoritative MLE likelihoods + NumPyro hierarchical models live under `src/rlwm/`; scripts import from `src/` directly; no backward-compat shims. Resolve `environments/` vs `src/environments/` split (consolidate unless documented architectural reason).
2. **Script consolidation (01–04)** — data-processing stages into one module with shared utilities; eliminate duplication.
3. **Script consolidation (05–08)** — behavioral analysis grouped cleanly (summary + viz + trauma grouping + descriptives).
4. **Script consolidation (09–11)** — simulations/recovery under one entry point (likely subcommands `ppc`, `sweep`, `recovery`).
5. **Script consolidation (16–21)** — audit the Phase 21 script explosion (`21_compute_loo_stacking.py`, `21_scale_audit.py`, `21_model_averaging.py`, `21_baseline_audit.py`, `21_manuscript_tables.py`, etc.); collapse into one "post-Bayesian-fit analyses" step with subcommands or coherent module layout. MLE regression (15, 16) → "post-MLE-fit analyses".
6. **`figures/` + `output/` reorg** — paths mirror new pipeline stages (pre-fit / MLE / Bayesian / post-fit). Update CLAUDE.md + README.
7. **`cluster/` consolidation** — per-model SLURM templates (`13_bayesian_m1.slurm` … `13_bayesian_m6b.slurm`) → one parameterized template dispatched via `--export=MODEL=...`. Shell wrappers for grouped stages mirroring `cluster/21_submit_pipeline.sh`.
8. **`validation/` + `tests/` pruning** — audit each file; move stale ones to `legacy/` (or delete if equivalent logic lives elsewhere). Keep only tests guarding load-bearing invariants.
9. **Docs refresh** — `docs/` updated to new structure. README + CLAUDE.md Quick Reference reflect new entry points.
10. **`paper.qmd` structural scaffolding (NEW vs. original scope)** — Results section reordered to Bayesian-first canonical order (summary → Bayesian fits → L2 trauma regression → subscale breakdown → Appendix: MLE + recovery + Bayesian↔MLE scatter). Quarto `{python}` inline refs and `@tbl-*` / `@fig-*` cross-refs point to artifact paths that *will* be produced by Phase 24+26 (not populated now). Goal: `quarto render paper.qmd` produces a PDF with placeholders in the correct slots, so when Phase 26 runs, real values drop in without layout changes.

### Explicitly out of scope for Phase 28

- Running any Bayesian fits (no GPU available).
- Running any MLE fits (could be run but no value without the Bayesian story around them).
- Populating real winner names, stacking weights, Pareto-k percentages, forest plots into `paper.qmd`. Those are `MANU-01..05`, which stay assigned to Phase 26.
- Full manuscript writing/editing beyond structural scaffolding.
- `quarto render` producing a *final* PDF (only produces a scaffold PDF with placeholders).
- Closure-guard updates (`validation/check_v5_closure.py`) — Phase 27's job.

### Requirements — new REFAC-* category (planner enumerates)

Planner will create new `REFAC-01..REFAC-NN` requirements covering items 1–10 above. `MANU-*` requirements remain with Phase 26. `CLEAN-*` are already complete (Phase 23). No changes to `EXEC-*` (Phase 24), `REPRO-*` (Phase 25), `CLOSE-*` (Phase 27).

---

## Constraints the planner should honor

- **No breaking of existing v4.0 closure invariants.** `validation/check_v4_closure.py` must still pass after refactor. Any test that currently passes must still pass.
- **Grep-based invariants for import hygiene.** After consolidation, grep for `from scripts.fitting.jax_likelihoods` (and similar) across the codebase — should all route through `src/` imports, zero direct script-to-script imports for core math.
- **Commits are atomic per refactor unit.** Do not bundle "consolidate 01-04" with "consolidate 05-08" into one commit — each gets its own plan + commit so rollback is cheap if something breaks.
- **Prefer `git mv` over delete+create** for moved files, so `git log --follow` preserves history.
- **Legacy handling:** the user's private global CLAUDE.md says "Avoid backwards-compatibility hacks… If you are certain that something is unused, you can delete it completely." Apply this — no shim files, no deprecation warnings, no `# removed` comments.

---

## Expected plan shape (for the planner's budget)

Rough estimate (planner to refine): 8–12 plans across 3–4 waves.

- **Wave 1 (foundation):** `src/` consolidation + environments/ resolution (blocks everything else).
- **Wave 2 (parallel consolidation):** scripts 01-04, 05-08, 09-11, 16-21 — independent module groups.
- **Wave 3 (parallel infra):** cluster scripts, figures/output reorg, validation/tests pruning.
- **Wave 4 (sequential):** paper.qmd structural scaffolding, docs refresh, final grep-audit + test run.

---

## Open questions the researcher should answer

1. `environments/rlwm_env.py` — is it imported by any `src/` module, or only by scripts? If only scripts, what's the load-bearing reason to keep it separate from `src/`?
2. Which of the `21_*.py` scripts subsume which? Are any redundant pairs?
3. `cluster/13_bayesian_m1.slurm` through `m6b.slurm` — what actually differs line-by-line? Model name? Wall time? Memory? All parameterizable?
4. `validation/` — which files guard current-pipeline invariants vs. which reference deleted/legacy code paths?
5. `figures/` + `output/` — current structure and what a Bayesian-first layout would look like.
6. `paper.qmd` — does it currently exist with placeholder content, or is it the "manuscript placeholder" from commit `66aadda`? What's in it?
