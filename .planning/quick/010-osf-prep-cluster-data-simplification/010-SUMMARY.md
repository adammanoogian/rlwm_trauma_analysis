---
quick_task: 010
slug: osf-prep-cluster-data-simplification
status: complete
date: 2026-04-26
---

# Quick Task 010: OSF-prep + cluster-data simplification

## What changed

**1. Gitignore tightening (b3f9d96)**
Replaced per-file ignore patterns for `data/raw/` and `data/interim/` with wildcard
`data/raw/*` + `!data/raw/.gitkeep` and `data/interim/*` + `!data/interim/.gitkeep`.
Added explicit `!data/processed/*.csv` + `!data/processed/*.json` to make tracking
intent visible. Legacy alias `task_trials_long_all_participants.csv` confirmed
identical to `task_trials_long.csv` (same rows, columns, content) — removed from
disk and git (`git rm`). `TASK_TRIALS_LEGACY` constant removed from `config.py`.

**2. Excluded-as-flag in summary CSV (3781746 + 1cf962e)**
Replaced the single `excluded` bool in `summary_participant_metrics.csv` with two
columns: `included_in_analysis` (bool) and `exclusion_reason` (str, pipe-joined).
Reason vocabulary: `low_trial_count_400`, `low_block_count_8`,
`manual_excluded_list`, `corrupted_csv`. Summary now covers ALL parseable
participants (196 total); downstream consumers (`03_analyze_trauma_groups.py`,
`04_run_statistical_analyses.py`, `04_analyze_mle_by_trauma.py`,
`05_regress_parameters_on_scales.py`, `fit_with_l2.py`, `bayesian.py`) updated
to filter on `included_in_analysis == True` before joining fit parameters or
running regressions. `01_summarize_behavioral_data.py` intentionally left
unfiltered with a WHY comment (descriptive cohort stats). Two bugs fixed in
the same commit range: `01_parse_raw_data.py` MIN_TRIALS aligned to
`DataParams.MIN_TRIALS` (400) so the consistency invariant holds;
`04_create_summary_csv.py` pointed at `TASK_TRIALS_ALL` instead of stale
`PARSED_TASK_TRIALS` interim file.

**3. CODEBOOK.md for OSF deposit (f10ba40)**
Created `docs/CODEBOOK.md` as self-contained data dictionary. Columns enumerated
from actual on-disk CSVs (not invented). Covers all three canonical processed
CSVs with column tables, type, range/units, and description. Documents instrument
citations (IES-R: Weiss & Marmar 1997; LEC-5: Weathers et al. 2013), exclusion
criteria vocabulary, anonymization approach, and inclusion criteria. Updated with
actual v5.0 cohort row counts after Task 5 regeneration (129,104 rows /
178 participants in `task_trials_long.csv`; 196 total in summary).

**4. Cluster stage-01 graceful skip (e36e5f3)**
Added three-branch pre-flight to `cluster/01_data_processing.slurm`:
- raw present → run full stage 01 python pipeline
- raw empty AND all three processed CSVs present → `exit 0` (downstream stages
  proceed via afterok; logs the file sizes)
- raw empty AND processed missing → hard fail with actionable message

`cluster/submit_all.sh` header updated to document `--from-stage 2` as the
canonical cluster cold-start entry (explicit skip of stage 01 SLURM overhead).
Both scripts pass `bash -n` and `--dry-run --from-stage 1`.

**5. Regenerated v5.0 processed artifacts (02c5fb8)**
Ran `01_parse_raw_data.py` + `04_create_summary_csv.py` under the new
excluded-as-flag logic. Final cohort: 196 parseable participants, 178 included
(18 excluded: all `low_trial_count_400|low_block_count_8`), 0 corrupted CSVs.
`task_trials_long.csv`: 129,104 rows (178 participants).
`task_trials_long_all.csv`: 144,238 rows (178 participants, +15,134 practice).
`summary_participant_metrics.csv`: 196 rows, 51 columns.
Consistency invariant: `task_trials_long.unique(sona_id) == included.sum() = 178`.
Legacy `summary_participant_metrics_all.csv` (17-row stale artifact) removed.
All three CSVs committed and pushed to origin/main.

## Verification

- [x] `git check-ignore -v data/raw/rlwm_trauma_*.csv` → ignored (data/raw/*)
- [x] `git check-ignore -v data/processed/task_trials_long.csv` → not ignored (exit 1)
- [x] `data/processed/summary_participant_metrics.csv` has `included_in_analysis` + `exclusion_reason` columns
- [x] All downstream consumers of summary_participant_metrics.csv now filter on the flag
- [x] `docs/CODEBOOK.md` exists, lists actual on-disk columns (not invented)
- [x] `cluster/01_data_processing.slurm` skips gracefully when raw empty + processed present
- [x] Inclusion-consistency invariant: `len(task_trials_long.participant_id.unique()) == summary.included_in_analysis.sum()` = 178
- [x] All 3 processed CSVs committed + pushed to origin/main

## Phase 24 unblock impact

Cluster cold-start path is now: `git pull` + `bash cluster/submit_all.sh --from-stage 2`
(no scp needed). Previous strategy required PII raw data on cluster; new strategy
keeps PII strictly local + on encrypted M3 home dir. The `01_data_processing.slurm`
pre-flight will auto-skip stage 01 even if `--from-stage 1` is used, as long as
`data/processed/` is populated from `git pull`.

## Deviations from plan

**[Rule 1 - Bug] 01_parse_raw_data.py MIN_TRIALS=100 vs DataParams.MIN_TRIALS=400**

- Found during: Task 5 regeneration
- Issue: `01_parse_raw_data.py` used a hard-coded `MIN_TRIALS=100` but
  `04_create_summary_csv.py` used `DataParams.MIN_TRIALS=400` for inclusion.
  This caused 183 participants in trial CSV vs 178 included in summary —
  breaking the consistency invariant.
- Fix: Changed `MIN_TRIALS = 100` to `MIN_TRIALS = DataParams.MIN_TRIALS`
  in `01_parse_raw_data.py`. Added WHY comment explaining the invariant.
- Commit: 1cf962e

**[Rule 1 - Bug] 04_create_summary_csv.py reading stale interim parsed_task_trials.csv**

- Found during: Task 5 regeneration
- Issue: `04_create_summary_csv.py` pointed at `DataParams.PARSED_TASK_TRIALS`
  (`data/interim/parsed_task_trials.csv`, written by legacy
  `03_create_task_trials_csv.py`) which only had 3 participants. The canonical
  output from `01_parse_raw_data.py` is `DataParams.TASK_TRIALS_ALL`.
- Fix: Changed `task_path = DataParams.PARSED_TASK_TRIALS` to
  `task_path = DataParams.TASK_TRIALS_ALL`.
- Commit: 1cf962e

**[Rule 2 - Missing Critical] NaN guard in included_in_analysis column**

- Found during: Task 2/5 integration
- Issue: Outer join in `04_create_summary_csv.py` could produce NaN in
  `included_in_analysis` if a participant appeared in demographics but not in
  `all_parsed_ids`. Added `fillna(False)` guard.
- Commit: 3781746

**Legacy summary_participant_metrics_all.csv deleted**

- Decision: 17-row artifact, not produced by current pipeline, redundant with
  the new all-participants summary. Removed via `git rm`.
- Commit: 02c5fb8

## Commits

- b3f9d96: refactor(data): gitignore tighten — track data/processed/ for OSF + cluster pull
- 3781746: feat(01-data): excluded-as-flag in summary_participant_metrics.csv (preserve all participants for OSF transparency)
- f10ba40: docs(codebook): add CODEBOOK.md as data dictionary for OSF deposit
- e36e5f3: feat(cluster): allow stage 01 to skip when data/processed/ is git-tracked
- 1cf962e: fix(01-data): align MIN_TRIALS threshold and task_path for consistency invariant
- 02c5fb8: data(processed): regenerate v5.0 cohort with excluded-as-flag (196 raw, 178 included)
