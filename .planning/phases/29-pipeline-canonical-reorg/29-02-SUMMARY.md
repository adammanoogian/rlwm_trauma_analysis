---
phase: 29-pipeline-canonical-reorg
plan: 02
subsystem: docs
tags: [docs, MODEL_REFERENCE, hierarchical-bayesian, scales-audit, K-parameterization, legacy-archive, git-history]

# Dependency graph
requires:
  - phase: 28-paper-finalization
    provides: stable docs layout with 03_methods_reference/ and 04_methods/ subdirectories
provides:
  - docs/04_methods/README.md with full Hierarchical Bayesian + Scales Orthogonalization content
  - docs/03_methods_reference/MODEL_REFERENCE.md with K Parameterization section 12
  - docs/legacy/{HIERARCHICAL_BAYESIAN,SCALES_AND_FITTING_AUDIT,K_PARAMETERIZATION}.md archives
  - .planning/phases/29-pipeline-canonical-reorg/artifacts/pre_phase29_cluster_gpu_lessons.sha256
affects:
  - 29-07-closure-guard-extension (consumes sha256 artifact for byte-identical invariant check)
  - 29-06-paper-qmd-smoke-render (owns paper.qmd line 166 caption update to 04_methods/README.md#scales-orthogonalization-and-audit)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Legacy archive via git mv: originals in docs/legacy/ preserve full git history with --follow"
    - "Merge provenance footnotes: 'Merged from docs/X.md on DATE (Phase Y Plan Z)' + historical-source link"
    - "Section heading demotion: H1/H2 source docs have headings demoted by one level when embedded under new H2"

key-files:
  created:
    - docs/legacy/HIERARCHICAL_BAYESIAN.md
    - docs/legacy/SCALES_AND_FITTING_AUDIT.md
    - docs/legacy/K_PARAMETERIZATION.md
    - .planning/phases/29-pipeline-canonical-reorg/artifacts/pre_phase29_cluster_gpu_lessons.sha256
  modified:
    - docs/04_methods/README.md
    - docs/03_methods_reference/MODEL_REFERENCE.md
    - docs/README.md
    - docs/legacy/README.md
    - src/rlwm/fitting/numpyro_helpers.py

key-decisions:
  - "docs/CLUSTER_GPU_LESSONS.md left fully untouched — user directive; stale reference at line 975 is intentionally preserved"
  - "manuscript/paper.qmd line 166 caption (docs/SCALES_AND_FITTING_AUDIT.md) update DEFERRED to Plan 29-06 — avoids Wave 1 parallel-write race; 29-06 is single owner of paper.qmd edits in Phase 29"
  - "docs/PARALLEL_SCAN_LIKELIHOOD.md left at top level — 18 KB technical companion doc, not a merge candidate per CONTEXT.md"
  - "Merge provenance footnotes use 'Merged from docs/X.md' (attribution) not anchor-broken link — safe to leave in merged files"
  - "MODEL_REFERENCE.md 'See Also' section renumbered 12→13; K Parameterization inserted as new section 12 (natural parameter-reference section after section 11 Bayesian)"

patterns-established:
  - "Merge pattern: source file → target H2 section → legacy/ archive via git mv; historical-source footnote at section bottom"
  - "sha256 invariant: capture before-phase hash of untouched files so closure guard can verify byte-identical post-phase"

# Metrics
duration: ~20min
completed: 2026-04-22
---

# Phase 29 Plan 02: Docs Spare File Integration Summary

**Three orphan top-level docs merged into structured methods references via git-history-preserving git mv; sha256 invariant committed for CLUSTER_GPU_LESSONS.md closure-guard check**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-04-22T (concurrent Wave 1 execution with 29-01)
- **Completed:** 2026-04-22
- **Tasks:** 3
- **Files created/modified:** 8 (+ 3 legacy archives)

## Accomplishments

- `docs/HIERARCHICAL_BAYESIAN.md` (~10 KB) merged into `docs/04_methods/README.md` as new `## Hierarchical Bayesian Architecture` section; original archived to `docs/legacy/` with `git mv`
- `docs/SCALES_AND_FITTING_AUDIT.md` (~13 KB) merged into `docs/04_methods/README.md` as new `## Scales Orthogonalization and Audit` section; original archived to `docs/legacy/`
- `docs/K_PARAMETERIZATION.md` (~7.6 KB) merged into `docs/03_methods_reference/MODEL_REFERENCE.md` as new section 12 "K Parameterization"; original archived to `docs/legacy/`; inline MODEL_REFERENCE cross-references in sections 1 and 11 updated to anchor links
- `docs/README.md` and `docs/legacy/README.md` navigation tables updated to new merged locations
- `src/rlwm/fitting/numpyro_helpers.py` docstring reference updated from stale `K_PARAMETERIZATION.md` to `docs/03_methods_reference/MODEL_REFERENCE.md section 12`
- SHA-256 hash artifact for `CLUSTER_GPU_LESSONS.md` committed to `.planning/phases/29-pipeline-canonical-reorg/artifacts/pre_phase29_cluster_gpu_lessons.sha256` for Plan 29-07 closure guard

## Task Commits

Each task was committed atomically:

1. **Task 1: Merge HIERARCHICAL_BAYESIAN + SCALES_AND_FITTING_AUDIT into 04_methods/README.md** - `59cdef2` (docs)
2. **Task 2: Merge K_PARAMETERIZATION into MODEL_REFERENCE.md section 12** - `7d3f8fb` (docs)
3. **Task 3: Capture CLUSTER_GPU_LESSONS.md sha256 invariant + finalize** - `56e5ea5` (docs)

**Plan metadata:** (staged next — docs(29-02): complete docs-spare-file-integration plan)

## Merge Details

| Source file | Size | Target location | Anchor |
|---|---|---|---|
| docs/HIERARCHICAL_BAYESIAN.md | ~10 KB | docs/04_methods/README.md | `#hierarchical-bayesian-architecture` |
| docs/SCALES_AND_FITTING_AUDIT.md | ~13 KB | docs/04_methods/README.md | `#scales-orthogonalization-and-audit` |
| docs/K_PARAMETERIZATION.md | ~7.6 KB | docs/03_methods_reference/MODEL_REFERENCE.md | `#k-parameterization` (section 12) |

## Files Created/Modified

- `docs/04_methods/README.md` — rewritten: TOC anchors updated + two new H2 sections (Bayesian architecture + scales audit) with historical-source footnotes
- `docs/03_methods_reference/MODEL_REFERENCE.md` — section 12 K Parameterization added; section 13 "See Also" renumbered; two inline refs updated to anchor links
- `docs/README.md` — navigation updated: 3 old top-level file links replaced with merged-section anchors
- `docs/legacy/README.md` — 3 new inventory rows added
- `docs/legacy/HIERARCHICAL_BAYESIAN.md` — git mv destination (history preserved)
- `docs/legacy/SCALES_AND_FITTING_AUDIT.md` — git mv destination (history preserved)
- `docs/legacy/K_PARAMETERIZATION.md` — git mv destination (history preserved)
- `src/rlwm/fitting/numpyro_helpers.py` — docstring L143 updated from stale K_PARAMETERIZATION.md reference
- `.planning/phases/29-pipeline-canonical-reorg/artifacts/pre_phase29_cluster_gpu_lessons.sha256` — hash `b39e24c5...` for 29-07 invariant check

## Decisions Made

1. **CLUSTER_GPU_LESSONS.md untouched (user directive):** The stale reference at line 975 (`docs/HIERARCHICAL_BAYESIAN.md`) is intentionally preserved. This is documented in SUMMARY for Plan 29-07 awareness.

2. **paper.qmd line 166 deferred to 29-06:** `docs/SCALES_AND_FITTING_AUDIT.md` appears at line 166 of `manuscript/paper.qmd`. That edit belongs to Plan 29-06 (paper-qmd-smoke-render), which is the single owner of paper.qmd edits in Phase 29. Target replacement: `docs/SCALES_AND_FITTING_AUDIT.md` → `docs/04_methods/README.md#scales-orthogonalization-and-audit`.

3. **PARALLEL_SCAN_LIKELIHOOD.md stays at top level:** 18 KB technical companion doc, per CONTEXT.md recommendation (too large for a subsection merge; functions as a standalone 1:1 implementation reference).

4. **Section 12 insertion point for K Parameterization:** Inserted after section 11 (Hierarchical Bayesian Fitting Pipeline) and before the old "See Also" (renumbered 13). This is the natural location — section 11 covers the Bayesian priors/transforms that K inherits, and readers needing K bounds details would logically look after the Bayesian section.

5. **Merge footnote format:** Used `*Merged from docs/X.md on DATE (Phase 29 Plan 02).*` as a provenance header line (not a broken link — attribution only). Historical-source footnote at bottom: `*Historical source: see [../legacy/X.md](...) ...*`. This two-footnote pattern is now established for future merges.

## Referrers Updated

| Stale reference | File | Updated to |
|---|---|---|
| `../HIERARCHICAL_BAYESIAN.md` | docs/04_methods/README.md (table) | `#hierarchical-bayesian-architecture` |
| `../SCALES_AND_FITTING_AUDIT.md` | docs/04_methods/README.md (table) | `#scales-orthogonalization-and-audit` |
| `HIERARCHICAL_BAYESIAN.md` | docs/README.md | `04_methods/README.md#hierarchical-bayesian-architecture` |
| `SCALES_AND_FITTING_AUDIT.md` | docs/README.md | `04_methods/README.md#scales-orthogonalization-and-audit` |
| `../HIERARCHICAL_BAYESIAN.md` | docs/legacy/README.md (CONVERGENCE_ASSESSMENT row) | `../04_methods/README.md#hierarchical-bayesian-architecture` |
| `docs/K_PARAMETERIZATION.md` | docs/03_methods_reference/MODEL_REFERENCE.md (section 1 inline) | `#k-parameterization` anchor |
| `docs/K_PARAMETERIZATION.md` | docs/03_methods_reference/MODEL_REFERENCE.md (section 11 inline) | `#k-parameterization` anchor |
| `docs/K_PARAMETERIZATION.md` | docs/03_methods_reference/MODEL_REFERENCE.md (See Also) | self-anchor in same doc |
| `K_PARAMETERIZATION.md` (docstring) | src/rlwm/fitting/numpyro_helpers.py L143 | `docs/03_methods_reference/MODEL_REFERENCE.md section 12` |

**Intentionally NOT updated (flagged for downstream plans):**
- `manuscript/paper.qmd:166` — `docs/SCALES_AND_FITTING_AUDIT.md` → Plan 29-06 owns this
- `docs/CLUSTER_GPU_LESSONS.md:975` — `docs/HIERARCHICAL_BAYESIAN.md` → untouched per user directive

## Deviations from Plan

None from plan logic. One minor deviation from execution context:

**Plan 29-01 staged renames absorbed into Task 1 commit:** Because Plans 29-01 and 29-02 ran concurrently in Wave 1 sharing the same git index, Plan 29-01's staged script renames (data_processing/ → 01_data_preprocessing/, etc.) were staged when Task 1 was committed. This is expected Wave 1 parallel behavior — the commit is larger than docs-only but all changes are correct and were staged by the respective plans. No content overlap or conflict occurred.

## Verification Results

All success criteria pass:

- `test ! -f docs/{HIERARCHICAL_BAYESIAN,K_PARAMETERIZATION,SCALES_AND_FITTING_AUDIT}.md` PASS
- `test -f docs/legacy/{HIERARCHICAL_BAYESIAN,K_PARAMETERIZATION,SCALES_AND_FITTING_AUDIT}.md` PASS
- `grep "## Hierarchical Bayesian Architecture" docs/04_methods/README.md` → line 32 PASS
- `grep "## Scales Orthogonalization and Audit" docs/04_methods/README.md` → line 282 PASS
- `grep "## 12. K Parameterization" docs/03_methods_reference/MODEL_REFERENCE.md` → line 1446 PASS
- CLUSTER_GPU_LESSONS.md sha256 hash match PASS
- Zero stale refs outside legacy (except intentionally excluded CLUSTER_GPU_LESSONS.md:975 and paper.qmd:166)

## Issues Encountered

None.

## Next Phase Readiness

- Plan 29-06 (paper-qmd-smoke-render) needs to update `paper.qmd` line 166: `docs/SCALES_AND_FITTING_AUDIT.md` → `docs/04_methods/README.md#scales-orthogonalization-and-audit`
- Plan 29-07 (closure-guard-extension) can consume `.planning/phases/29-pipeline-canonical-reorg/artifacts/pre_phase29_cluster_gpu_lessons.sha256` (hash `b39e24c5c6de543717ec9fdb30ec5d77bcb3396be211dfe1c5398d3ba64ef30b`) to verify CLUSTER_GPU_LESSONS.md remained byte-identical throughout Phase 29
- `docs/CLUSTER_GPU_LESSONS.md:975` has a stale reference to `docs/HIERARCHICAL_BAYESIAN.md` — this is a known accepted deviation; future cleanup could update it but was explicitly excluded per user directive

---
*Phase: 29-pipeline-canonical-reorg*
*Completed: 2026-04-22*
