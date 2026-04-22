---
phase: 29-pipeline-canonical-reorg
plan: 02
type: execute
wave: 1
depends_on: []
files_modified:
  - docs/04_methods/README.md                          (content merged: HIERARCHICAL_BAYESIAN + SCALES_AND_FITTING_AUDIT)
  - docs/03_methods_reference/MODEL_REFERENCE.md       (content merged: K_PARAMETERIZATION section added)
  - docs/HIERARCHICAL_BAYESIAN.md                      (deleted — merged into 04_methods/README.md)
  - docs/K_PARAMETERIZATION.md                         (deleted — merged into 03_methods_reference/MODEL_REFERENCE.md)
  - docs/SCALES_AND_FITTING_AUDIT.md                   (deleted — merged into 04_methods/README.md)
  - docs/legacy/HIERARCHICAL_BAYESIAN.md               (git mv from docs/ — historical archive)
  - docs/legacy/K_PARAMETERIZATION.md                  (git mv from docs/)
  - docs/legacy/SCALES_AND_FITTING_AUDIT.md            (git mv from docs/)
  - docs/README.md                                     (updated cross-references)
  - manuscript/paper.qmd                               (line 166: docs ref updated for SCALES merge)
autonomous: true

must_haves:
  truths:
    - "docs/ top level no longer contains HIERARCHICAL_BAYESIAN.md, K_PARAMETERIZATION.md, or SCALES_AND_FITTING_AUDIT.md (merged into structured method docs)"  # SC#5
    - "docs/CLUSTER_GPU_LESSONS.md byte-identical to pre-phase content (untouched per user directive)"  # SC#6
    - "Merged content is reachable from docs/README.md and docs/04_methods/README.md navigation tables"
    - "paper.qmd cross-reference to `docs/SCALES_AND_FITTING_AUDIT.md` (line 166) updated to point at merged location"
  artifacts:
    - path: "docs/legacy/HIERARCHICAL_BAYESIAN.md"
      provides: "historical archive of pre-merge hierarchical-Bayesian design doc"
    - path: "docs/legacy/K_PARAMETERIZATION.md"
      provides: "historical archive of pre-merge K parameterization note"
    - path: "docs/legacy/SCALES_AND_FITTING_AUDIT.md"
      provides: "historical archive of pre-merge scales audit"
    - path: "docs/04_methods/README.md"
      provides: "Bayesian fitting + scales orthogonalization narrative (Hierarchical + SCALES merged)"
    - path: "docs/03_methods_reference/MODEL_REFERENCE.md"
      provides: "Model mathematics + K parameterization section (K_PARAMETERIZATION merged)"
    - path: "pre_phase29_cluster_gpu_lessons.sha256"
      provides: "hash manifest so 29-07 closure guard can verify CLUSTER_GPU_LESSONS.md untouched"
      contains: "sha256 hash"
  key_links:
    - from: "docs/04_methods/README.md"
      to: "docs/legacy/HIERARCHICAL_BAYESIAN.md"
      via: "historical-source footnote at bottom of merged section"
      pattern: "legacy/HIERARCHICAL_BAYESIAN\\.md"
    - from: "docs/03_methods_reference/MODEL_REFERENCE.md"
      to: "docs/legacy/K_PARAMETERIZATION.md"
      via: "historical-source footnote"
      pattern: "legacy/K_PARAMETERIZATION\\.md"
---

<objective>
Merge three orphan top-level docs (`HIERARCHICAL_BAYESIAN.md`, `K_PARAMETERIZATION.md`, `SCALES_AND_FITTING_AUDIT.md`) into their natural home in the structured `03_methods_reference/` and `04_methods/` subdirectories. Move the originals to `docs/legacy/` so git history is preserved and rollback is trivial. Leave `docs/CLUSTER_GPU_LESSONS.md` and `docs/PARALLEL_SCAN_LIKELIHOOD.md` in place (user directive for the former; the latter is 18 KB and a 1:1 implementation reference that fits top-level). Runs in Wave 1 alongside 29-01 because docs work has zero overlap with `scripts/` paths.

Purpose: Reduce top-level docs clutter and surface the merged content under canonical methods-reference navigation so manuscript readers and future maintainers find it.

Output: 3 merged sections, 3 legacy archives, hash manifest for CLUSTER_GPU_LESSONS.md invariant, updated paper.qmd cross-reference.
</objective>

<execution_context>
@C:\Users\aman0087\.claude\get-shit-done\workflows\execute-plan.md
@C:\Users\aman0087\.claude\get-shit-done\templates\summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/phases/29-pipeline-canonical-reorg/29-CONTEXT.md
@docs/README.md
@docs/04_methods/README.md
@docs/03_methods_reference/MODEL_REFERENCE.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Merge HIERARCHICAL_BAYESIAN.md and SCALES_AND_FITTING_AUDIT.md into docs/04_methods/README.md</name>
  <files>
    - docs/04_methods/README.md (modified)
    - docs/legacy/HIERARCHICAL_BAYESIAN.md (new — git mv destination)
    - docs/legacy/SCALES_AND_FITTING_AUDIT.md (new — git mv destination)
    - docs/HIERARCHICAL_BAYESIAN.md (deleted via git mv)
    - docs/SCALES_AND_FITTING_AUDIT.md (deleted via git mv)
  </files>
  <action>
    1. Read the two orphan files in full:
       - `docs/HIERARCHICAL_BAYESIAN.md` (~10 KB, hierarchical-Bayesian fitting architecture — priors, partial pooling, MCMC settings)
       - `docs/SCALES_AND_FITTING_AUDIT.md` (~13 KB, IES-R Gram-Schmidt orthogonalization + scale-handling audit)
    2. Open `docs/04_methods/README.md` — currently it's a 28-line index/TOC. Rewrite into a structured README with sections:
       - Keep the existing "Published-in-paper methods" and "Supplementary / validation methods" tables at the TOP (update row pointers: instead of `../HIERARCHICAL_BAYESIAN.md` use `#hierarchical-bayesian-architecture` anchor; instead of `../SCALES_AND_FITTING_AUDIT.md` use `#scales-orthogonalization-and-audit` anchor).
       - Add a new H2 "## Hierarchical Bayesian Architecture" section — paste the full content of `docs/HIERARCHICAL_BAYESIAN.md` verbatim (preserve its H2/H3 levels by demoting to H3/H4 as needed to fit under the new H2).
       - Add a new H2 "## Scales Orthogonalization and Audit" section — paste the full content of `docs/SCALES_AND_FITTING_AUDIT.md`.
       - At the bottom of each merged section, add a footnote: `*Historical source: see [legacy/HIERARCHICAL_BAYESIAN.md](../legacy/HIERARCHICAL_BAYESIAN.md) for the original standalone version.*` (equivalent for SCALES).
    3. `git mv docs/HIERARCHICAL_BAYESIAN.md docs/legacy/HIERARCHICAL_BAYESIAN.md`
    4. `git mv docs/SCALES_AND_FITTING_AUDIT.md docs/legacy/SCALES_AND_FITTING_AUDIT.md`
    5. Update any other referrers:
       - `grep -rn "HIERARCHICAL_BAYESIAN.md\|SCALES_AND_FITTING_AUDIT.md" . --exclude-dir=.planning --exclude-dir=.git --exclude-dir=docs/legacy`
       - Known hit: `manuscript/paper.qmd` line 166 references `docs/SCALES_AND_FITTING_AUDIT.md` (in a figure caption). Rewrite to: `docs/04_methods/README.md#scales-orthogonalization-and-audit`.
       - Known hit (potentially): `docs/README.md` if it has a top-level index — update to point at the merged sections.
  </action>
  <verify>
    - `test ! -f docs/HIERARCHICAL_BAYESIAN.md && test ! -f docs/SCALES_AND_FITTING_AUDIT.md`
    - `test -f docs/legacy/HIERARCHICAL_BAYESIAN.md && test -f docs/legacy/SCALES_AND_FITTING_AUDIT.md`
    - `grep -n "## Hierarchical Bayesian Architecture" docs/04_methods/README.md` returns 1 match
    - `grep -n "## Scales Orthogonalization and Audit" docs/04_methods/README.md` returns 1 match
    - `grep -rn "docs/HIERARCHICAL_BAYESIAN\.md\|docs/SCALES_AND_FITTING_AUDIT\.md" scripts/ tests/ validation/ cluster/ manuscript/ docs/ src/ --exclude-dir=legacy` returns ZERO (all references rewritten, except inside `docs/legacy/` which is fine)
    - `git log --follow --oneline docs/legacy/HIERARCHICAL_BAYESIAN.md | head -3` shows original file history preserved
  </verify>
  <done>Merged content lives under `docs/04_methods/README.md` H2 sections with historical-source footnotes; originals archived in `docs/legacy/`; paper.qmd caption updated.</done>
</task>

<task type="auto">
  <name>Task 2: Merge K_PARAMETERIZATION.md into docs/03_methods_reference/MODEL_REFERENCE.md</name>
  <files>
    - docs/03_methods_reference/MODEL_REFERENCE.md (modified)
    - docs/legacy/K_PARAMETERIZATION.md (new)
    - docs/K_PARAMETERIZATION.md (deleted)
  </files>
  <action>
    1. Read `docs/K_PARAMETERIZATION.md` (~7.6 KB — covers K parameter choice, bounds, identifiability, Collins-K normalization).
    2. Open `docs/03_methods_reference/MODEL_REFERENCE.md`. Find the parameter-list section (or section on WM capacity K if present). Add a new H2 "## K Parameterization" at an appropriate location (probably right after the main parameter table, or as a dedicated subsection under "WM capacity").
    3. Paste the full content verbatim, demoting heading levels as needed to fit.
    4. Add historical footnote at end of section: `*Historical source: see [../legacy/K_PARAMETERIZATION.md](../legacy/K_PARAMETERIZATION.md).*`
    5. `git mv docs/K_PARAMETERIZATION.md docs/legacy/K_PARAMETERIZATION.md`
    6. Grep for other referrers:
       - `grep -rn "K_PARAMETERIZATION.md" . --exclude-dir=.planning --exclude-dir=.git --exclude-dir=docs/legacy`
       - Rewrite hits to `docs/03_methods_reference/MODEL_REFERENCE.md#k-parameterization`.
  </action>
  <verify>
    - `test ! -f docs/K_PARAMETERIZATION.md`
    - `test -f docs/legacy/K_PARAMETERIZATION.md`
    - `grep -n "## K Parameterization\|^# K Parameterization" docs/03_methods_reference/MODEL_REFERENCE.md` returns at least 1 match
    - `grep -rn "docs/K_PARAMETERIZATION\.md" scripts/ tests/ validation/ cluster/ manuscript/ docs/ src/ --exclude-dir=legacy` returns ZERO
    - `git log --follow --oneline docs/legacy/K_PARAMETERIZATION.md | head -3` shows original file history preserved
  </verify>
  <done>K parameterization content lives under MODEL_REFERENCE.md as an H2 section; original archived; zero stale references.</done>
</task>

<task type="auto">
  <name>Task 3: Capture CLUSTER_GPU_LESSONS.md hash invariant + update docs/README.md + commit</name>
  <files>
    - pre_phase29_cluster_gpu_lessons.sha256 (new — tracked in repo root for closure guard)
    - docs/README.md (updated cross-references)
  </files>
  <action>
    1. Compute SHA-256 hash of `docs/CLUSTER_GPU_LESSONS.md` BEFORE any other edits in this plan touched anything, and write it to a manifest file that 29-07 closure guard will consume:
       ```
       python -c "import hashlib, pathlib; p=pathlib.Path('docs/CLUSTER_GPU_LESSONS.md'); print(hashlib.sha256(p.read_bytes()).hexdigest())" > pre_phase29_cluster_gpu_lessons.sha256
       ```
       Note: because 29-02 does NOT edit `CLUSTER_GPU_LESSONS.md`, the hash is the current file's hash — that's exactly what 29-07 will re-hash and diff against to prove no accidental edits slipped in during subsequent phase work.
    2. Update `docs/README.md` if it maintains a top-level index — remove references to the now-moved files (HIERARCHICAL_BAYESIAN.md, K_PARAMETERIZATION.md, SCALES_AND_FITTING_AUDIT.md) and point at the merged locations.
    3. Verify CLUSTER_GPU_LESSONS.md and PARALLEL_SCAN_LIKELIHOOD.md are UNTOUCHED:
       - `git diff docs/CLUSTER_GPU_LESSONS.md` → empty
       - `git diff docs/PARALLEL_SCAN_LIKELIHOOD.md` → empty (not moved per user directive — 18 KB implementation reference stays top-level)
    4. Commit the docs merge as a single commit:
       ```
       docs(29-02): merge HIERARCHICAL_BAYESIAN, K_PARAMETERIZATION, SCALES_AND_FITTING_AUDIT into structured method docs

       - docs/HIERARCHICAL_BAYESIAN.md → docs/04_methods/README.md#hierarchical-bayesian-architecture
       - docs/SCALES_AND_FITTING_AUDIT.md → docs/04_methods/README.md#scales-orthogonalization-and-audit
       - docs/K_PARAMETERIZATION.md → docs/03_methods_reference/MODEL_REFERENCE.md#k-parameterization
       - Originals archived under docs/legacy/ with git mv (history preserved)
       - CLUSTER_GPU_LESSONS.md and PARALLEL_SCAN_LIKELIHOOD.md UNTOUCHED (user directive)
       - paper.qmd caption (line 166) updated to new SCALES location
       - pre_phase29_cluster_gpu_lessons.sha256 manifest added for 29-07 closure guard
       ```
  </action>
  <verify>
    - `test -f pre_phase29_cluster_gpu_lessons.sha256` and file contains a 64-char hex string
    - `python -c "import hashlib, pathlib; actual=hashlib.sha256(pathlib.Path('docs/CLUSTER_GPU_LESSONS.md').read_bytes()).hexdigest(); expected=pathlib.Path('pre_phase29_cluster_gpu_lessons.sha256').read_text().strip(); assert actual == expected, f'{actual} != {expected}'"` passes silently
    - `git diff --name-only docs/CLUSTER_GPU_LESSONS.md docs/PARALLEL_SCAN_LIKELIHOOD.md` shows nothing
    - `git log -1 --stat` shows the single docs merge commit
  </verify>
  <done>Hash manifest committed; CLUSTER_GPU_LESSONS.md verifiably unchanged; docs/README.md cross-refs updated; commit landed.</done>
</task>

</tasks>

<verification>
```bash
# 3 orphan files relocated to legacy/
for f in HIERARCHICAL_BAYESIAN.md K_PARAMETERIZATION.md SCALES_AND_FITTING_AUDIT.md; do
  test ! -f docs/$f || { echo "STILL AT TOP: $f"; exit 1; }
  test -f docs/legacy/$f || { echo "MISSING FROM LEGACY: $f"; exit 1; }
done

# CLUSTER_GPU_LESSONS.md untouched (hash invariant)
python -c "
import hashlib, pathlib
actual = hashlib.sha256(pathlib.Path('docs/CLUSTER_GPU_LESSONS.md').read_bytes()).hexdigest()
expected = pathlib.Path('pre_phase29_cluster_gpu_lessons.sha256').read_text().strip()
assert actual == expected, f'hash mismatch: {actual} != {expected}'
print('CLUSTER_GPU_LESSONS.md byte-identical')
"

# Merged content exists at expected anchors
grep -n "## Hierarchical Bayesian Architecture" docs/04_methods/README.md
grep -n "## Scales Orthogonalization and Audit" docs/04_methods/README.md
grep -n "K Parameterization" docs/03_methods_reference/MODEL_REFERENCE.md

# Zero stale references outside legacy/
grep -rn "docs/HIERARCHICAL_BAYESIAN\.md\|docs/K_PARAMETERIZATION\.md\|docs/SCALES_AND_FITTING_AUDIT\.md" \
  scripts/ tests/ validation/ cluster/ manuscript/ docs/ src/ \
  --exclude-dir=legacy --exclude-dir=.planning \
  || echo "OK: zero stale refs"

# paper.qmd caption updated
grep -n "SCALES_AND_FITTING_AUDIT\|scales-orthogonalization-and-audit" manuscript/paper.qmd
```
</verification>

<success_criteria>
1. `docs/{HIERARCHICAL_BAYESIAN,K_PARAMETERIZATION,SCALES_AND_FITTING_AUDIT}.md` no longer exist at top level (SC#5).
2. The three originals live at `docs/legacy/` with `git mv` history preserved.
3. Merged content is findable under `docs/04_methods/README.md` (2 H2 sections) and `docs/03_methods_reference/MODEL_REFERENCE.md` (1 H2 section).
4. `docs/CLUSTER_GPU_LESSONS.md` and `docs/PARALLEL_SCAN_LIKELIHOOD.md` bit-identical to pre-phase content (SC#6); hash manifest captured.
5. `manuscript/paper.qmd` line 166 reference updated.
6. Zero stale `docs/HIERARCHICAL_BAYESIAN.md`/`docs/K_PARAMETERIZATION.md`/`docs/SCALES_AND_FITTING_AUDIT.md` references outside `docs/legacy/` and `.planning/`.
</success_criteria>

<output>
After completion, create `.planning/phases/29-pipeline-canonical-reorg/29-02-SUMMARY.md` with:
- Each merge: source → destination anchor, approximate byte count of merged content
- Hash captured for CLUSTER_GPU_LESSONS.md
- List of referrers rewritten
- Commit SHA
</output>
