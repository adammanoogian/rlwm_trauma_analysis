# CODEBOOK — RLWM Trauma Analysis

Companion to OSF deposit. Documents every column of the three canonical
datasets shipped with the repo:

- `data/processed/task_trials_long.csv` — trial-level, included participants only, main task only
- `data/processed/task_trials_long_all.csv` — trial-level, included participants only, all blocks (practice + main)
- `data/processed/summary_participant_metrics.csv` — one row per parseable participant (~250 total)

---

## Provenance

- **Raw data source:** jsPsych task hosted on Pavlovia (gitlab.pavlovia.org/amanoogian/rlwm_trauma)
- **Processing pipeline:** `scripts/01_data_preprocessing/01..04` run in order
- **Anonymization:** real Pavlovia `sona_id` values replaced with sequential `assigned_id`
  (10000+) via `scripts/_maintenance/update_participant_mapping.py` — the mapping JSON is
  never committed (covered by `.gitignore data/raw/*`).
- **Inclusion criteria** (combined into `included_in_analysis` flag in summary):
  - `n_main_task_trials >= MIN_TRIALS` (currently 400; see `config.py:MIN_TRIALS_THRESHOLD`)
  - `n_main_task_blocks >= MIN_BLOCKS` (currently 8; see `config.py:DataParams.MIN_BLOCKS`)
  - Participant not in `EXCLUDED_PARTICIPANTS` manual list (`config.py`)
  - Raw CSV parsed without errors (no `corrupted_csv` flag)
- **Code state:** this CODEBOOK.md is valid for git rev `<fill via git rev-parse HEAD at deposit>`.
  Pin at OSF upload time.

---

## task_trials_long.csv

Trial-level table. Included participants only (see `included_in_analysis` in
`summary_participant_metrics.csv`). Main task only — practice blocks excluded.
Filter: `block >= 3` AND `is_practice == False`.

Row count (v5.0 cohort): see `git log --oneline -1 data/processed/task_trials_long.csv`
for the commit that regenerated this file.

| Column | Type | Range / Units | Description |
|--------|------|---------------|-------------|
| sona_id | int64 | 10000+ | Anonymous sequential participant ID. Never maps back to real identity. |
| trial_in_experiment | int64 | 1–N | Trial counter across the entire jsPsych session (resets to 1 per participant). |
| block | int64 | 3–23 | Block number (1–2 = practice, excluded from this file). |
| stimulus | int64 | 0–5 | 0-indexed stimulus ID within the block's stimulus set (converted from 1-indexed jsPsych data). |
| key_press | int64 | 0–2 | Participant response: 0 = J, 1 = K, 2 = L (decoded from jsPsych keyCode). |
| correct | float64 | {0.0, 1.0} | 1.0 if the response matched the correct action for this stimulus, else 0.0. |
| rt | float64 | ms | Reaction time from stimulus onset to key press. |
| time_elapsed | int64 | ms | Cumulative time since session start (jsPsych `time_elapsed` field). |
| set_size | float64 | {2, 3, 5, 6} | Number of distinct stimuli in this block. Set size 4 is excluded from the main task. |
| load_condition | object | low / high | Working memory load: low = set_size ≤ 3; high = set_size ≥ 4. |
| phase_type | object | main_task | Phase label from jsPsych (`practice_static` / `practice_dynamic` / `main_task`). All rows in this file have `main_task`. |
| source_file | object | filename | Original raw jsPsych CSV filename (PII-stripped; retains timestamp for audit). |
| is_practice | bool | False | Always False in this file (main task only). Kept for schema consistency with `task_trials_long_all.csv`. |
| trial_in_block | int64 | 1–N | Trial counter within a block (resets to 1 per block per participant). |
| reward | float64 | {0.0, 1.0} | Copy of `correct` cast to float; used as the scalar reward signal by model-fitting code. |

---

## task_trials_long_all.csv

Same schema as `task_trials_long.csv` plus practice trials. Distinguish
practice from main task via `is_practice` (bool) and `phase_type` (str).

Included participants only. Block range: 1–23 (blocks 1–2 = practice).

Row count (v5.0 cohort): ~125,900 (main task rows ≈ 112,500 + practice ≈ 13,400).

All columns are identical to `task_trials_long.csv`. Key differences:

| Column | Value in practice rows | Value in main-task rows |
|--------|------------------------|-------------------------|
| block | 1 or 2 | 3–23 |
| phase_type | practice_static (block 1) or practice_dynamic (block 2) | main_task |
| is_practice | True | False |

Use `--include-practice` flag with `fit_mle.py` / `fit_bayesian.py` to fit
models on the combined dataset.

---

## summary_participant_metrics.csv

One row per parseable participant. Includes ALL participants (~250) whose raw
CSV was successfully parsed. Use `included_in_analysis == True` to filter
to the analysis cohort (N ≈ 169 for the v5.0 cohort before regeneration;
see Task 5 of quick-task-010 for updated counts).

### Identification

| Column | Type | Range / Units | Description |
|--------|------|---------------|-------------|
| sona_id | int64 | 10000+ | Anonymous sequential participant ID. Matches `sona_id` in trial CSVs. |

### Inclusion gate

| Column | Type | Range / Units | Description |
|--------|------|---------------|-------------|
| included_in_analysis | bool | {True, False} | True iff the participant passes ALL inclusion criteria (see Provenance). |
| exclusion_reason | str | "" or pipe-joined list | Empty string when `included_in_analysis` is True. Otherwise one or more of: `low_trial_count_400`, `low_block_count_8`, `manual_excluded_list`, `corrupted_csv`. Multiple reasons joined with `\|`. |

### Demographics

| Column | Type | Range / Units | Description |
|--------|------|---------------|-------------|
| age_years | object | e.g. "25" | Self-reported age in years (stored as string due to free-text entry). |
| country | object | ISO-like string | Self-reported country of residence. |
| primary_language | object | string | Self-reported primary language. |
| gender | float64 | Likert | Gender identity item. Exact coding depends on jsPsych survey version; see raw survey. |
| education | float64 | Likert | Highest education level completed. |
| relationship_status | float64 | Likert | Current relationship status. |
| living_arrangement | float64 | Likert | Current living arrangement. |
| screen_time | float64 | hours/day | Self-reported daily screen time. |

### IES-R (Impact of Event Scale – Revised)

Scored via `scripts/utils/scoring.py:score_ies_r`.
Reference: Weiss & Marmar (1997). Higher scores = greater trauma-related distress.

| Column | Type | Range / Units | Description |
|--------|------|---------------|-------------|
| ies_total | float64 | 0–88 | Sum of all 22 IES-R items (0–4 each). |
| ies_intrusion | float64 | 0–35 | Intrusion subscale sum (items 1, 2, 3, 6, 9, 16, 20). |
| ies_avoidance | float64 | 0–40 | Avoidance subscale sum (items 5, 7, 8, 11, 12, 13, 17, 22). |
| ies_hyperarousal | float64 | 0–35 | Hyperarousal subscale sum (items 4, 10, 14, 15, 18, 19, 21). |

Note: `ies_total = ies_intrusion + ies_avoidance + ies_hyperarousal` exactly.
This linear dependence precludes using all four as independent predictors
in a regression — see `scripts/fitting/level2_design.py` module docstring.

### LEC-5 (Life Events Checklist for DSM-5)

Scored via `scripts/utils/scoring.py:score_less`.
Reference: Weathers et al. (2013).

| Column | Type | Range / Units | Description |
|--------|------|---------------|-------------|
| less_total_events | float64 | 0–N | LESS (Life Events Stress Score) total event count across all LEC-5 categories. |
| less_personal_events | float64 | 0–N | Number of events the participant experienced personally (vs. witnessed or learned of). |

Note: columns are named with `less_` prefix in the CSV to indicate LESS scoring,
but several downstream scripts rename them to `lec_*` for display purposes.

### Task performance metrics

Computed from main-task trials only (blocks ≥ 3) by
`scripts/utils/scoring.py:calculate_all_task_metrics`.

#### Overall

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| n_trials_total | float64 | count | Total main-task trials for this participant. |
| n_trials_completed | float64 | count | Trials with a valid (non-NaN) response. |
| accuracy_overall | float64 | proportion [0,1] | Proportion correct across all main-task trials. |
| mean_rt_overall | float64 | ms | Mean reaction time across all valid main-task trials. |
| median_rt_overall | float64 | ms | Median reaction time. |

#### By load condition

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| accuracy_low_load | float64 | proportion | Mean accuracy for set_size ≤ 3 (WM load low). |
| mean_rt_low_load | float64 | ms | Mean RT for low-load trials. |
| n_trials_low_load | float64 | count | Number of low-load trials. |
| accuracy_high_load | float64 | proportion | Mean accuracy for set_size ≥ 4 (WM load high). |
| mean_rt_high_load | float64 | ms | Mean RT for high-load trials. |
| n_trials_high_load | float64 | count | Number of high-load trials. |

#### By set size

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| accuracy_setsize_2 | float64 | proportion | Mean accuracy for set_size = 2. |
| mean_rt_setsize_2 | float64 | ms | Mean RT for set_size = 2 trials. |
| n_trials_setsize_2 | float64 | count | Number of set_size = 2 trials. |
| accuracy_setsize_3 | float64 | proportion | Mean accuracy for set_size = 3. |
| mean_rt_setsize_3 | float64 | ms | Mean RT for set_size = 3 trials. |
| n_trials_setsize_3 | float64 | count | Number of set_size = 3 trials. |
| accuracy_setsize_5 | float64 | proportion | Mean accuracy for set_size = 5. |
| mean_rt_setsize_5 | float64 | ms | Mean RT for set_size = 5 trials. |
| n_trials_setsize_5 | float64 | count | Number of set_size = 5 trials. |
| accuracy_setsize_6 | float64 | proportion | Mean accuracy for set_size = 6. |
| mean_rt_setsize_6 | float64 | ms | Mean RT for set_size = 6 trials. |
| n_trials_setsize_6 | float64 | count | Number of set_size = 6 trials. |

#### Learning trajectory (block epoch)

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| accuracy_early_blocks | float64 | proportion | Mean accuracy in early blocks (first third of the session). |
| mean_rt_early_blocks | float64 | ms | Mean RT in early blocks. |
| accuracy_middle_blocks | float64 | proportion | Mean accuracy in middle blocks. |
| mean_rt_middle_blocks | float64 | ms | Mean RT in middle blocks. |
| accuracy_late_blocks | float64 | proportion | Mean accuracy in late blocks (last third). |
| mean_rt_late_blocks | float64 | ms | Mean RT in late blocks. |
| learning_slope | float64 | proportion/block | Linear regression slope of block-level accuracy over block number (positive = improving). |
| learning_improvement_early_to_late | float64 | proportion | `accuracy_late_blocks - accuracy_early_blocks`. |

#### Reversal learning

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| n_reversals | float64 | count | Number of contingency reversals detected in this participant's session. |
| performance_drop_post_reversal | float64 | proportion | Mean accuracy drop in the 5 trials immediately following a reversal (relative to pre-reversal baseline). |
| adaptation_rate_post_reversal | float64 | proportion/trial | Rate of accuracy recovery after a reversal (linear slope of post-reversal accuracy). |

---

## Exclusion criteria vocabulary

| Reason string | Meaning | Source |
|---------------|---------|--------|
| `low_trial_count_400` | Fewer than `MIN_TRIALS_THRESHOLD` (400) main-task trials | `04_create_summary_csv.py` |
| `low_block_count_8` | Fewer than `DataParams.MIN_BLOCKS` (8) main-task blocks | `04_create_summary_csv.py` |
| `manual_excluded_list` | In `EXCLUDED_PARTICIPANTS` list in `config.py` | `04_create_summary_csv.py` |
| `corrupted_csv` | Raw jsPsych CSV failed `pd.read_csv` parsing | `01_parse_raw_data.py` |

Multiple reasons are `|`-joined. Empty string = included.

---

## Instrument citations

- **IES-R**: Weiss, D. S., & Marmar, C. R. (1997). The Impact of Event Scale–Revised.
  In J. P. Wilson & T. M. Keane (Eds.), *Assessing psychological trauma and PTSD* (pp. 399–411). Guilford.
- **LEC-5**: Weathers, F. W., Blake, D. D., Schnurr, P. P., Kaloupek, D. G., Marx, B. P.,
  & Keane, T. M. (2013). *The Life Events Checklist for DSM-5 (LEC-5)*. PTSD: National Center
  for PTSD.

---

## License + reuse

- **Data**: fill at deposit time (likely CC BY 4.0 with attribution).
- **Code**: see repo `LICENSE` file.

---

## Citation

If you use this dataset, please cite:
> Placeholder — fill with paper citation upon publication (Senta et al., 2025).

---

## Versioning

This codebook documents the v5.0 cohort (regenerated 2026-04-26 by quick-task-010).
Earlier cohort snapshots are not retained in git history for individual CSVs;
use `git log data/processed/` to identify the commit for a given snapshot.
