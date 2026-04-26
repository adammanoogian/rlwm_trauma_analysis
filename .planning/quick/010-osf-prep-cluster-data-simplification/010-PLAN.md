---
phase: quick-010
plan: 010
type: execute
wave: 1
depends_on: []
files_modified:
  - .gitignore
  - scripts/01_data_preprocessing/01_parse_raw_data.py
  - scripts/01_data_preprocessing/04_create_summary_csv.py
  - docs/CODEBOOK.md
  - cluster/01_data_processing.slurm
  - cluster/submit_all.sh
  - data/processed/task_trials_long.csv
  - data/processed/task_trials_long_all.csv
  - data/processed/summary_participant_metrics.csv
autonomous: true

must_haves:
  truths:
    - "data/raw/* and data/interim/* are gitignored except .gitkeep (PII never leaks)"
    - "data/processed/*.csv and data/processed/*.json are tracked in git"
    - "summary_participant_metrics.csv contains ALL parseable participants (~250) with included_in_analysis flag and exclusion_reason string"
    - "task_trials_long.csv remains filtered to included-only participants (analysis-ready)"
    - "Cluster pipeline can cold-start from stage 02 when data/raw/ is empty AND data/processed/ is populated"
    - "docs/CODEBOOK.md exists as a self-contained data dictionary referencing actual CSV columns (not fabricated)"
    - "Five atomic commits land in order, each scoped to one of the five sub-tasks"
  artifacts:
    - path: ".gitignore"
      provides: "wildcard-form gitignore for data/raw and data/interim with .gitkeep negation; explicit non-ignore for data/processed CSVs/JSONs"
      contains: "data/raw/*"
    - path: "scripts/01_data_preprocessing/04_create_summary_csv.py"
      provides: "summary CSV that preserves all participants with included_in_analysis (bool) and exclusion_reason (str) columns"
      contains: "included_in_analysis"
    - path: "scripts/01_data_preprocessing/01_parse_raw_data.py"
      provides: "parse loop with one-line WHY note explaining task_trials_long.csv stays filtered while summary preserves all"
      contains: "MIN_TRIALS"
    - path: "docs/CODEBOOK.md"
      provides: "data dictionary for OSF deposit covering three canonical processed CSVs"
      min_lines: 80
    - path: "cluster/01_data_processing.slurm"
      provides: "pre-flight that exits 0 (skip stage 01) when data/raw/ is empty AND data/processed/task_trials_long.csv exists"
      contains: "data/processed/task_trials_long.csv"
    - path: "data/processed/task_trials_long.csv"
      provides: "trial-level main-task CSV, included participants only, regenerated under v5.0 cohort"
    - path: "data/processed/task_trials_long_all.csv"
      provides: "trial-level CSV including practice blocks (is_practice + phase_type cols)"
    - path: "data/processed/summary_participant_metrics.csv"
      provides: "per-participant summary, all ~250 parseable participants, with included_in_analysis + exclusion_reason"
  key_links:
    - from: "cluster/01_data_processing.slurm"
      to: "data/processed/task_trials_long.csv"
      via: "pre-flight conditional: empty raw + tracked processed -> exit 0"
      pattern: "data/processed/task_trials_long.csv"
    - from: "scripts/01_data_preprocessing/04_create_summary_csv.py"
      to: "data/processed/summary_participant_metrics.csv"
      via: "writes included_in_analysis + exclusion_reason columns for all parseable participants"
      pattern: "included_in_analysis"
    - from: "docs/CODEBOOK.md"
      to: "data/processed/{task_trials_long,task_trials_long_all,summary_participant_metrics}.csv"
      via: "documents real columns of each CSV (read from disk, not invented)"
      pattern: "summary_participant_metrics.csv"
---

<objective>
Switch the project from "scp PII raw data to cluster" to "commit de-identified
processed data to git, cluster pulls it." Implements five sequential sub-tasks:
gitignore tightening, excluded-as-flag in summary CSV, data dictionary, cluster
stage-01 graceful skip, and regenerated processed artifacts.

Purpose: OSF-prep + cluster cold-start no longer requires manual rsync of PII.
Output: tracked processed/ CSVs, codebook for deposit, cluster pipeline that
skips stage 01 when processed data already exists.
</objective>

<execution_context>
@C:\Users\aman0087\.claude/get-shit-done/workflows/execute-plan.md
@C:\Users\aman0087\.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/STATE.md

@CLAUDE.md
@.gitignore
@config.py
@scripts/01_data_preprocessing/01_parse_raw_data.py
@scripts/01_data_preprocessing/04_create_summary_csv.py
@cluster/01_data_processing.slurm
@cluster/submit_all.sh
</context>

<tasks>

<task type="auto">
  <name>Task 1: Gitignore tightening — wildcard data/raw + data/interim, explicit-track data/processed</name>
  <files>.gitignore</files>
  <action>
Replace the per-file ignore patterns under the "Phase 31 — CCDS-aligned layout"
block with wildcard form:

Current state (lines ~59-85) lists individual files for data/raw and data/interim.
Rewrite that block so it reads:

```
# Raw participant data — sensitive, NEVER tracked (wildcard form: any new
# raw artifact is gitignored by default; only .gitkeep is preserved).
data/raw/*
!data/raw/.gitkeep

# Interim (parsed/collated) — sensitive PII, gitignored under the same
# wildcard pattern.
data/interim/*
!data/interim/.gitkeep

# Processed — analysis-ready, EXPLICITLY tracked. CSVs and JSONs ship with
# the repo for OSF deposit + cluster cold-start (see docs/CODEBOOK.md).
# (No ignore pattern needed; listed here for intent visibility.)
!data/processed/.gitkeep
!data/processed/*.csv
!data/processed/*.json
```

Leave the models/ + reports/ + logs/ blocks (lines ~77-94) untouched.

Then decide on the legacy-alias file `data/processed/task_trials_long_all_participants.csv`:
verify it exists locally (`ls data/processed/`); the docstring in
`scripts/01_data_preprocessing/01_parse_raw_data.py` mentions it as a legacy
alias. If it is identical (same row count, same content) to
`task_trials_long_all.csv`, delete the alias file from disk and remove any
remaining references in the parse script's docstring. If it differs, keep it
for now and add it to docs/CODEBOOK.md in Task 3. Document the decision inline
in the commit message.

Verification commands (must run, must show expected outputs):

```bash
git check-ignore -v data/raw/rlwm_trauma_PARTICIPANT_SESSION_2025-11-05_19h07.07.413.csv
# Expect: matches data/raw/* pattern (ignored)

git check-ignore -v data/processed/task_trials_long.csv
# Expect: exit 1, no output (NOT ignored)

git check-ignore -v data/interim/parsed_demographics.csv
# Expect: matches data/interim/* (ignored)

git check-ignore -v data/raw/.gitkeep
# Expect: exit 1 (negated by !data/raw/.gitkeep)
```

Commit:
```
git add .gitignore
# Plus any legacy-alias removal (git rm if applicable)
git commit -m "refactor(data): gitignore tighten — track data/processed/ for OSF + cluster pull"
```
  </action>
  <verify>
git check-ignore -v on the four paths above produces the expected ignored/not-ignored
outcomes. `git status` shows no unexpected files newly tracked or newly ignored.
  </verify>
  <done>
data/raw/* and data/interim/* are wildcard-ignored (.gitkeep negated); data/processed/*.csv
and data/processed/*.json are explicitly tracked; legacy-alias decision recorded
in commit; one atomic commit landed.
  </done>
</task>

<task type="auto">
  <name>Task 2: Excluded-as-flag — preserve all participants in summary CSV with inclusion flag</name>
  <files>
scripts/01_data_preprocessing/04_create_summary_csv.py
scripts/01_data_preprocessing/01_parse_raw_data.py
  </files>
  <action>
**Read first**: confirm current logic in `04_create_summary_csv.py` (which already
adds an `excluded` bool — that single column is insufficient: we need a
human-readable reason string and a clearer flag name).

**Patch `04_create_summary_csv.py`**:

1. Replace the existing single `excluded` column with two columns:
   - `included_in_analysis` (bool): True iff the participant passes ALL inclusion
     criteria (sufficient trials AND sufficient blocks AND not in
     EXCLUDED_PARTICIPANTS).
   - `exclusion_reason` (str): empty string when `included_in_analysis is True`;
     otherwise a `|`-joined string of any of:
       * `low_trial_count_<MIN_TRIALS`
       * `low_block_count_<MIN_BLOCKS`
       * `manual_excluded_list`
       * `corrupted_csv` (see step 3 below)

2. Compute the flag inside the existing `for sona_id in task_trials['sona_id'].unique():`
   loop (around line 149). Use `block_counts` and `trial_counts` already
   computed earlier in the script. Build `exclusion_reason` by appending strings
   to a list and `'|'.join(...)` at the end; empty list -> empty string.

3. After computing `task_metrics`, the merge logic stays the same. The summary
   `summary` will already contain all parseable participants because
   `demographics` (the LHS of the chain of merges) contains everyone whose CSV
   parsed. Confirm with a print line:
   ```python
   print(f"  Included in analysis: {summary['included_in_analysis'].sum()}")
   print(f"  Excluded with reason: {(~summary['included_in_analysis']).sum()}")
   ```

4. **Corrupted-CSV surfacing**: in `01_parse_raw_data.py`, find the
   `parse_single_file()` call site (around line 305). Currently a corrupted CSV
   raises and either crashes or skips silently. Wrap the call in try/except
   `pd.errors.ParserError` and `(UnicodeDecodeError, ValueError)`; on failure,
   record `{'sona_id': assigned_id, 'corrupted': True}` into a new
   `corrupted_participants` list and continue. After the parse loop, write the
   list to `data/interim/corrupted_participants.csv` (gitignored by Task 1's
   wildcard). Then in `04_create_summary_csv.py`, read that file (if present)
   and union those sona_ids into the summary with
   `included_in_analysis=False`, `exclusion_reason='corrupted_csv'`. If the file
   doesn't exist (no corruptions this run), proceed normally — do not error.

5. **Keep `task_trials_long.csv` filtered to included-only.** Add a single-line
   WHY comment at the top of the parse loop block in `01_parse_raw_data.py`:
   ```python
   # NOTE: task_trials_long.csv stays filtered to included participants
   # (analysis-ready). Per-participant inclusion flag with exclusion_reason
   # lives in summary_participant_metrics.csv (see docs/CODEBOOK.md).
   ```

6. **Audit downstream consumers.** Run:
   ```bash
   grep -rln "summary_participant_metrics" scripts/ src/
   ```
   For each consumer that joins summary on participant_id and feeds into a
   model fit, an analysis, or a regression: add a defensive
   `summary = summary[summary['included_in_analysis'] == True]` filter
   immediately after the read_csv call. Specifically check:
   - `scripts/02_behav_analyses/01_summarize_behavioral_data.py`
   - `scripts/02_behav_analyses/03_analyze_trauma_groups.py`
   - `scripts/02_behav_analyses/04_run_statistical_analyses.py`
   - `scripts/06_fit_analyses/04_analyze_mle_by_trauma.py`
   - `scripts/06_fit_analyses/05_regress_parameters_on_scales.py`
   - `scripts/04_model_fitting/c_level2/fit_with_l2.py`
   - `scripts/fitting/level2_design.py`
   - `src/rlwm/fitting/bayesian.py`
   - `run_data_pipeline.py`

   For consumers that already filter by `excluded == False` (the old column
   name), update them to `included_in_analysis == True`. For consumers that
   read summary purely for descriptive stats (e.g. demographic table over the
   full cohort), leave them unfiltered — but document the choice in a one-line
   WHY comment.

   Consumers reading `task_trials_long.csv` need no change (already filtered
   by construction).

Verification:
```bash
python scripts/01_data_preprocessing/01_parse_raw_data.py
python scripts/01_data_preprocessing/02_create_collated_csv.py
python scripts/01_data_preprocessing/03_create_task_trials_csv.py
python scripts/01_data_preprocessing/04_create_summary_csv.py

python -c "
import pandas as pd
df = pd.read_csv('data/processed/summary_participant_metrics.csv')
print(f'rows: {len(df)}')
print(f'columns include included_in_analysis: {\"included_in_analysis\" in df.columns}')
print(f'columns include exclusion_reason: {\"exclusion_reason\" in df.columns}')
print(f'included: {df[\"included_in_analysis\"].sum()}, excluded: {(~df[\"included_in_analysis\"]).sum()}')
print('Reason counts:')
print(df.loc[~df['included_in_analysis'], 'exclusion_reason'].value_counts())
"
```

Expect: rows ~ 250 (matches mapping JSON), included ~ 169 (current cohort
size with MIN_TRIALS=100), excluded ~ 81, exclusion_reason populated.

Commit:
```
git add scripts/01_data_preprocessing/01_parse_raw_data.py \
        scripts/01_data_preprocessing/04_create_summary_csv.py \
        scripts/02_behav_analyses/*.py \
        scripts/06_fit_analyses/04_analyze_mle_by_trauma.py \
        scripts/06_fit_analyses/05_regress_parameters_on_scales.py \
        scripts/04_model_fitting/c_level2/fit_with_l2.py \
        scripts/fitting/level2_design.py \
        src/rlwm/fitting/bayesian.py \
        run_data_pipeline.py
git commit -m "feat(01-data): excluded-as-flag in summary_participant_metrics.csv (preserve all participants for OSF transparency)"
```

Only `git add` files actually modified — use `git status` to confirm before
staging.
  </action>
  <verify>
The verification python -c block above prints rows ~ 250, included flag and
exclusion_reason both present, and exclusion_reason value_counts shows the
expected reason strings (low_trial_count, manual_excluded_list, optionally
corrupted_csv). All four pipeline scripts run without error.
  </verify>
  <done>
summary_participant_metrics.csv contains all ~250 parseable participants;
included_in_analysis + exclusion_reason columns populated correctly; downstream
consumers filter by included_in_analysis where appropriate; task_trials_long.csv
is still included-only with a WHY comment explaining the asymmetry.
  </done>
</task>

<task type="auto">
  <name>Task 3: Write docs/CODEBOOK.md from the regenerated CSVs</name>
  <files>docs/CODEBOOK.md</files>
  <action>
Create `docs/CODEBOOK.md` as a self-contained data dictionary. Per project
CLAUDE.md, each major topic has ONE authoritative document — this is the OSF
deposit codebook.

**Step 1: enumerate columns from disk** (do not invent):

```python
import pandas as pd
for path in [
    'data/processed/task_trials_long.csv',
    'data/processed/task_trials_long_all.csv',
    'data/processed/summary_participant_metrics.csv',
]:
    df = pd.read_csv(path, nrows=5)
    print(f'=== {path} ({len(pd.read_csv(path))} rows) ===')
    for col in df.columns:
        print(f'  {col}: {df[col].dtype}')
    print()
```

Capture the actual columns + dtypes. Use this output to populate the codebook
tables.

**Step 2: write the codebook** with this structure (fill column tables from
Step 1 output):

```markdown
# CODEBOOK — RLWM Trauma Analysis

Companion to OSF deposit. Documents every column of the three canonical
datasets shipped with the repo:

- `data/processed/task_trials_long.csv` — trial-level, included participants, main task only
- `data/processed/task_trials_long_all.csv` — trial-level, included participants, includes practice
- `data/processed/summary_participant_metrics.csv` — one row per parseable participant

## Provenance

- Raw data source: jsPsych task hosted on Pavlovia (gitlab.pavlovia.org/amanoogian/rlwm_trauma)
- Processing pipeline: `scripts/01_data_preprocessing/01..04` (run in order)
- Anonymization: real Pavlovia sona_ids replaced with sequential `assigned_id` (10000+) via `scripts/_maintenance/update_participant_mapping.py` — mapping JSON kept private, never committed (see `.gitignore` `data/raw/*` rule)
- Inclusion criteria (combined into `included_in_analysis` flag in summary):
    - `n_task_trials >= MIN_TRIALS` (currently 100; see `config.py:DataParams.MIN_TRIALS`)
    - `n_blocks >= MIN_BLOCKS` (currently MIN_BLOCKS in config.py)
    - Participant not in `EXCLUDED_PARTICIPANTS` manual list
    - CSV parses cleanly (no `corrupted_csv` flag)
- Code state: this CODEBOOK.md is valid for git rev `<filled at deposit time>`. Pin via `git rev-parse HEAD` at the moment of OSF upload.

## task_trials_long.csv

Trial-level table, included participants only, main task only (block >= MAIN_TASK_START_BLOCK).
N rows = <fill from Step 1>.

| Column | Type | Range/Units | Description |
|--------|------|-------------|-------------|
| <enumerate from Step 1, populated with concise description> |

## task_trials_long_all.csv

Same schema as task_trials_long.csv plus practice trials. Distinguish practice
from main task via `is_practice` (bool) and `phase_type` (str: practice_static,
practice_dynamic, main_task).
N rows = <fill from Step 1>.

| Column | Type | Range/Units | Description |
|--------|------|-------------|-------------|
| <enumerate columns NEW relative to task_trials_long.csv (is_practice, phase_type)> |

## summary_participant_metrics.csv

One row per parseable participant. Includes ALL participants (~250) whose
raw CSV could be parsed; use `included_in_analysis` to subset.

| Column | Type | Range/Units | Description |
|--------|------|-------------|-------------|
| sona_id | int | 10000+ | Anonymous sequential ID. Never maps back to real identity. |
| included_in_analysis | bool | {True, False} | Inclusion gate (see Provenance). |
| exclusion_reason | str | "" or "low_trial_count_<100" | "low_block_count_<N>" | "manual_excluded_list" | "corrupted_csv" | "|"-joined if multiple. Empty string when included_in_analysis is True. |
| <enumerate the rest from Step 1, group by demographic / LEC / IES / task metric> |

## Instrument citations

- **LEC-5** (Life Events Checklist for DSM-5; Weathers et al., 2013) — scored to
  LESS (Life Events Stress Score) via `scripts/utils/scoring.py:score_less`.
  Output columns: `less_total_events`, `less_personal_events`.
- **IES-R** (Impact of Event Scale - Revised; Weiss & Marmar, 1997) — scored
  via `scripts/utils/scoring.py:score_ies_r`. Output columns: `ies_total`,
  `ies_intrusion`, `ies_avoidance`, `ies_hyperarousal`.
- Demographics: `age_years`, `gender`, `country`, `primary_language`,
  `education`, `relationship_status`, `living_arrangement`, `screen_time`.

## Exclusion criteria summary

| Reason string | Meaning | Source |
|---------------|---------|--------|
| `low_trial_count_<100` | Fewer than `DataParams.MIN_TRIALS` task trials | `04_create_summary_csv.py` |
| `low_block_count_<N>` | Fewer than `DataParams.MIN_BLOCKS` blocks | `04_create_summary_csv.py` |
| `manual_excluded_list` | In `config.py:EXCLUDED_PARTICIPANTS` | `04_create_summary_csv.py` |
| `corrupted_csv` | CSV failed `pd.read_csv` parse | `01_parse_raw_data.py` |

Multiple reasons concatenated with `|`.

## License + reuse

Data: <fill at deposit time — likely CC BY 4.0 with attribution>.
Code: see repo `LICENSE` file.

## Citation

If you use this dataset, please cite:
> <placeholder for paper citation once published>

## Versioning

This codebook documents the v5.0 cohort (regenerated <date of regeneration>).
Earlier cohort snapshots are not retained in git.
```

Step 3: open the file you just wrote, scan every column row, and confirm
NO column was invented (cross-check against Step 1 stdout). If unsure about
any column's meaning, look it up in `scripts/utils/scoring.py` or
`scripts/01_data_preprocessing/01_parse_raw_data.py` rather than guessing.

Commit:
```
git add docs/CODEBOOK.md
git commit -m "docs(codebook): add CODEBOOK.md as data dictionary for OSF deposit"
```
  </action>
  <verify>
docs/CODEBOOK.md exists; every column listed in the three tables matches the
output of the Step 1 enumeration script; no fabricated columns; sections present
for Provenance, three CSV tables, instrument citations, exclusion criteria,
license, citation, versioning.
  </verify>
  <done>
Self-contained codebook ready for OSF deposit; columns match disk reality;
exclusion-reason vocabulary documented and matches Task 2 implementation.
  </done>
</task>

<task type="auto">
  <name>Task 4: Cluster simplification — stage 01 SLURM gracefully skips when processed/ is tracked</name>
  <files>
cluster/01_data_processing.slurm
cluster/submit_all.sh
  </files>
  <action>
Implement Option A (SLURM-level skip — keeps the unit of skipping at the SLURM
job, doesn't alter dispatcher logic). Per standing user feedback "cluster
pre-flight should be SLURM-automated" — this stays in the SLURM script.

**Patch `cluster/01_data_processing.slurm`** — relax the existing pre-flight
check (lines ~66-80, added in commit 8385a98). Replace the existing block
with:

```bash
# Pre-flight: data sourcing strategy.
#   - If data/raw/ contains rlwm_trauma_*.csv: parse fresh (full stage 01).
#   - Else if data/processed/task_trials_long.csv exists: skip stage 01 entirely
#     (cluster pulled processed CSVs from git; raw PII stays on the local
#     workstation per .gitignore data/raw/* rule).
#   - Else: hard fail — neither raw input nor processed output is available.
shopt -s nullglob
RAW_CSVS=(data/raw/rlwm_trauma_*.csv)
shopt -u nullglob
if [[ ${#RAW_CSVS[@]} -eq 0 ]]; then
    if [[ -f data/processed/task_trials_long.csv \
       && -f data/processed/task_trials_long_all.csv \
       && -f data/processed/summary_participant_metrics.csv ]]; then
        echo "Pre-flight: data/raw/ empty AND data/processed/ populated — skipping stage 01"
        echo "  task_trials_long.csv:           $(wc -l < data/processed/task_trials_long.csv) lines"
        echo "  task_trials_long_all.csv:       $(wc -l < data/processed/task_trials_long_all.csv) lines"
        echo "  summary_participant_metrics.csv: $(wc -l < data/processed/summary_participant_metrics.csv) lines"
        echo "Stage 01 skipped — downstream stages will read committed processed/ CSVs"
        # Auto-push log even on skip so the chain has a record
        if [[ -f cluster/autopush.sh ]]; then
            source cluster/autopush.sh
        fi
        exit 0
    fi
    echo "ERROR: neither data/raw/ nor data/processed/ has data" >&2
    echo "       Expected EITHER: rlwm_trauma_*.csv files in data/raw/" >&2
    echo "       OR:              task_trials_long.csv + task_trials_long_all.csv" >&2
    echo "                        + summary_participant_metrics.csv in data/processed/" >&2
    echo "       For OSF/cluster cold-start workflow:" >&2
    echo "         git pull (processed/ CSVs are tracked since v5.0)" >&2
    exit 1
fi
echo "Pre-flight: found ${#RAW_CSVS[@]} raw participant CSV files in data/raw/ — running full stage 01"
```

The exit-0-on-skip means the afterok dependency in the chain is satisfied and
stage 02 proceeds normally with the git-tracked processed/ CSVs.

**Patch `cluster/submit_all.sh` header**: in the `# Usage:` block (lines ~17-23),
add a line documenting the cold-start entry:

```
#   bash cluster/submit_all.sh                          # full chain, real submission
#                                                      #   stage 01 auto-skips if data/raw/ empty AND data/processed/ tracked
#   bash cluster/submit_all.sh --from-stage 2          # explicit skip of stage 01 entirely
#                                                      #   (use when you've git-pulled processed/ and want to bypass the SLURM pre-flight cost)
```

No change to the dispatch logic — Option A keeps everything in the SLURM script.

Verification (local — bash syntax check, since we're not on M3):

```bash
bash -n cluster/01_data_processing.slurm
bash -n cluster/submit_all.sh

# Sanity: verify the dry-run path still works
bash cluster/submit_all.sh --dry-run --from-stage 1
```

Expect: dry-run succeeds, every python target resolves, no MISSING errors.

Commit:
```
git add cluster/01_data_processing.slurm cluster/submit_all.sh
git commit -m "feat(cluster): allow stage 01 to skip when data/processed/ is git-tracked"
```
  </action>
  <verify>
bash -n on both scripts passes. submit_all.sh --dry-run --from-stage 1 succeeds.
The relaxed pre-flight in 01_data_processing.slurm encodes three branches: raw
present (run stage), raw empty + processed present (exit 0 skip), neither
(hard fail with actionable message).
  </verify>
  <done>
Cluster cold-start workflow `git pull && bash cluster/submit_all.sh` will
correctly skip stage 01 on the cluster (which never has raw PII) and proceed
into stage 02 with the committed processed/ CSVs. Documentation in submit_all.sh
header explains both implicit-skip and explicit-skip paths.
  </done>
</task>

<task type="auto">
  <name>Task 5: Regenerate processed/ artifacts and commit data files</name>
  <files>
data/processed/task_trials_long.csv
data/processed/task_trials_long_all.csv
data/processed/summary_participant_metrics.csv
  </files>
  <action>
Now that Tasks 1-4 are committed, regenerate the canonical processed/ artifacts
under the new excluded-as-flag logic. Run the four-stage pipeline locally:

```bash
python scripts/01_data_preprocessing/01_parse_raw_data.py
python scripts/01_data_preprocessing/02_create_collated_csv.py
python scripts/01_data_preprocessing/03_create_task_trials_csv.py
python scripts/01_data_preprocessing/04_create_summary_csv.py
```

Expected outputs in data/processed/:

- `task_trials_long.csv` — included participants only, main task only (~112k rows
  matches current state pre-flag because included cohort is unchanged)
- `task_trials_long_all.csv` — included participants only, with practice
- `summary_participant_metrics.csv` — ALL ~250 parseable participants now
  (currently 169 → expected ~250); included_in_analysis + exclusion_reason
  columns populated

If `summary_participant_metrics_all.csv` was emitted by an older version of the
pipeline, decide whether the pipeline still produces it (the new flag-based
summary makes the `_all` variant redundant). If redundant, delete it from disk
and remove its production code in `04_create_summary_csv.py`. If still produced,
add it to docs/CODEBOOK.md (revisit Task 3 codebook).

Sanity-check the regenerated CSVs:

```bash
python -c "
import pandas as pd
t = pd.read_csv('data/processed/task_trials_long.csv')
ta = pd.read_csv('data/processed/task_trials_long_all.csv')
s = pd.read_csv('data/processed/summary_participant_metrics.csv')
print(f'task_trials_long.csv:           {len(t)} rows, {t[\"sona_id\"].nunique()} participants')
print(f'task_trials_long_all.csv:       {len(ta)} rows, {ta[\"sona_id\"].nunique()} participants')
print(f'summary_participant_metrics.csv: {len(s)} rows')
print(f'  included_in_analysis: {s[\"included_in_analysis\"].sum()} / {len(s)}')
print(f'  exclusion_reason value_counts:')
print(s.loc[~s['included_in_analysis'], 'exclusion_reason'].value_counts())
print(f'  task_trials_long unique participants == summary included: {t[\"sona_id\"].nunique() == s[\"included_in_analysis\"].sum()}')
"
```

Expect: `task_trials_long unique participants == summary included` is True.
This is the consistency invariant.

If the codebook now needs row-count updates (e.g. summary went from 169 to 250),
patch docs/CODEBOOK.md inline with the actual numbers and amend the codebook
within this commit (this is a coupled data + doc change — single commit is
correct; keeps documented row counts in sync with shipped CSVs).

Stage and commit:

```bash
git add data/processed/task_trials_long.csv \
        data/processed/task_trials_long_all.csv \
        data/processed/summary_participant_metrics.csv \
        docs/CODEBOOK.md
# Plus any deletions if summary_participant_metrics_all.csv was removed
git commit -m "data(processed): regenerate v5.0 cohort with excluded-as-flag (250 raw, N≈<actual> included)"
```

Replace `<actual>` with the actual included count from the sanity-check.

Verify the commit went through:
```bash
git log -1 --stat
git status
```

Do NOT push automatically — leave that to the user. Print the suggested push
command in the task summary:
```
git push origin main
```
  </action>
  <verify>
The four pipeline scripts run without error end-to-end. The
`task_trials_long.csv` participant count exactly equals the
`summary_participant_metrics.csv` `included_in_analysis.sum()`. The commit
contains the three processed CSVs (and the codebook if row counts changed).
git status clean afterwards.
  </verify>
  <done>
v5.0 cohort artifacts regenerated under the new flag-based logic, committed to
git, and ready for `git push`. The cluster cold-start workflow is now
end-to-end functional: `git pull` on M3 will deliver included-only trial CSVs +
all-participants summary, and `bash cluster/submit_all.sh` will skip stage 01
and proceed into stage 02.
  </done>
</task>

</tasks>

<verification>

End-to-end verification (run after Task 5 lands):

```bash
# 1. .gitignore correctness
git check-ignore -v data/raw/rlwm_trauma_PARTICIPANT_SESSION_2025-11-05_19h07.07.413.csv
git check-ignore -v data/interim/parsed_demographics.csv
git check-ignore -v data/processed/task_trials_long.csv     # NOT ignored
git check-ignore -v data/processed/summary_participant_metrics.csv  # NOT ignored

# 2. Processed CSVs are tracked
git ls-files data/processed/ | sort
# Expect: .gitkeep, summary_participant_metrics.csv, task_trials_long.csv, task_trials_long_all.csv

# 3. Summary has flag + reason
python -c "
import pandas as pd
s = pd.read_csv('data/processed/summary_participant_metrics.csv')
assert 'included_in_analysis' in s.columns
assert 'exclusion_reason' in s.columns
assert len(s) >= 200, f'expected >= 200 rows, got {len(s)}'
print('summary CSV invariants: PASS')
"

# 4. task_trials_long matches summary inclusion
python -c "
import pandas as pd
t = pd.read_csv('data/processed/task_trials_long.csv')
s = pd.read_csv('data/processed/summary_participant_metrics.csv')
assert t['sona_id'].nunique() == s['included_in_analysis'].sum(), \
       f'mismatch: trials={t[\"sona_id\"].nunique()} included={s[\"included_in_analysis\"].sum()}'
print('inclusion-consistency invariant: PASS')
"

# 5. Cluster scripts syntax-clean
bash -n cluster/01_data_processing.slurm
bash -n cluster/submit_all.sh
bash cluster/submit_all.sh --dry-run --from-stage 1

# 6. Codebook exists and references real columns
test -f docs/CODEBOOK.md && grep -q "included_in_analysis" docs/CODEBOOK.md \
    && grep -q "exclusion_reason" docs/CODEBOOK.md \
    && echo "codebook present + flag columns documented: PASS"

# 7. Five atomic commits land in order
git log --oneline -10
# Expect (top to bottom, newest first):
#   data(processed): regenerate v5.0 cohort with excluded-as-flag ...
#   feat(cluster): allow stage 01 to skip when data/processed/ is git-tracked
#   docs(codebook): add CODEBOOK.md as data dictionary for OSF deposit
#   feat(01-data): excluded-as-flag in summary_participant_metrics.csv ...
#   refactor(data): gitignore tighten — track data/processed/ for OSF + cluster pull
```

</verification>

<success_criteria>

- [ ] `.gitignore` uses wildcard form for data/raw + data/interim with .gitkeep negation
- [ ] data/processed/*.csv and *.json explicitly tracked in git
- [ ] summary_participant_metrics.csv has all ~250 parseable participants
- [ ] included_in_analysis (bool) + exclusion_reason (str) columns populated correctly
- [ ] exclusion_reason vocabulary: low_trial_count_<N>, low_block_count_<N>, manual_excluded_list, corrupted_csv (|-joined if multiple)
- [ ] task_trials_long.csv unique participant count == summary included count
- [ ] All downstream consumers either filter by included_in_analysis or carry a WHY comment for not filtering
- [ ] docs/CODEBOOK.md exists, columns enumerated from actual CSVs (not invented)
- [ ] cluster/01_data_processing.slurm pre-flight has 3 branches: raw present (run), raw empty + processed present (exit 0 skip), neither (hard fail)
- [ ] cluster/submit_all.sh dry-run succeeds; header documents both implicit and explicit skip
- [ ] Five atomic commits land in order (refactor, feat, docs, feat, data)
- [ ] No comments narrating WHAT — only WHY when non-obvious
- [ ] Pipeline scripts use NumPy-style docstrings + `from __future__ import annotations` + Python 3.10+ type hints

</success_criteria>

<output>
After completion, create `.planning/quick/010-osf-prep-cluster-data-simplification/010-SUMMARY.md`
documenting:
- Five atomic commits with their SHAs
- Final cohort numbers (raw / parseable / included)
- Suggested `git push origin main` command
- Cluster cold-start verification path: `git pull && bash cluster/submit_all.sh`
- Any deviations from the plan (e.g. legacy alias kept vs deleted, whether
  summary_participant_metrics_all.csv was retained)
</output>
