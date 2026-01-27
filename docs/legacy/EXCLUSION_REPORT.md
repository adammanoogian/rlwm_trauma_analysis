PARTICIPANT EXCLUSION REPORT
================================================================================

SAMPLE SIZES AT EACH STAGE:
- Demographics extracted: 54 participants
- Behavioral summary: 54 participants  
- Statistical data file: 51 participants (3 excluded)
- ANOVA long data: 51 unique IDs present
- ANOVA degrees of freedom: DF2=48 (indicates N=49)

================================================================================
EXCLUSIONS IDENTIFIED:
================================================================================

FROM DEMOGRAPHICS → STATISTICAL ANALYSIS (3 excluded):

1. ID 10012 - Insufficient trials
   - 87 trials (expected 807-1077)
   - Reason: Did not complete experiment
   
2. ID 10040 - Insufficient trials  
   - 12 trials, 0% accuracy
   - Reason: Abandoned experiment immediately
   
3. ID 10045 OR 10049 - Insufficient trials
   - 10045: 39 trials, 23% accuracy
   - 10049: 18 trials, 28% accuracy
   - One or both excluded for incomplete data

================================================================================
DUPLICATE PARTICIPANTS IDENTIFIED:
================================================================================

IDs 10043 and 10044 appear to be the SAME participant:
- Identical data: 807 trials, 71.4% accuracy
- Identical questionnaire scores: LESS=12/8, IES-R=33 (10/17/6)
- Both appear in statistical data file AND ANOVA data

IDs 10072 and 10073 appear to be the SAME participant:
- Identical data: 807 trials, 79.4% accuracy  
- Identical questionnaire scores: LESS=6/2, IES-R=50 (18/22/10)
- Both appear in statistical data file AND ANOVA data

================================================================================
DISCREPANCY BETWEEN DATA FILE (N=51) AND ANOVA (N=49):
================================================================================

The statistical data file contains 51 participants.
The ANOVA degrees of freedom (DF2=48) indicates N=49 participants.

This 2-participant discrepancy is NOT explained by the 4 duplicate IDs,
since all duplicates appear in the ANOVA data.

POSSIBLE EXPLANATIONS:
1. The ANOVA script drops duplicates before analysis (keep first occurrence)
2. Two participants have missing load-specific data preventing ANOVA inclusion
3. Error in ANOVA DF calculation or data preparation

================================================================================
PARTICIPANTS WITH PARTIAL DATA (RETAINED):
================================================================================

ID 10001: 207 trials (~4 blocks) - RETAINED in ANOVA
ID 10025: 507 trials (~10 blocks) - RETAINED in ANOVA  
ID 10053: 220 trials (~4 blocks) - RETAINED in ANOVA

These participants completed fewer than the standard 12-21 blocks but were
retained in the analysis.

================================================================================
RECOMMENDATIONS:
================================================================================

1. RESOLVE DUPLICATES:
   - Investigate why 10043/10044 and 10072/10073 have identical data
   - Likely data collection error or duplicate SONA IDs
   - Remove one from each pair before final analysis
   
2. CLARIFY EXCLUSION CRITERIA:
   - Decide on minimum trial threshold (current implicit threshold ~100 trials)
   - Document why 10001, 10025, 10053 retained despite <12 blocks
   
3. VERIFY FINAL N:
   - After removing duplicates: N = 51 - 2 = 49 (matches ANOVA DF)
   - This would give final N=49 for manuscript
   
4. UPDATE MANUSCRIPT:
   - Current manuscript states N=48
   - Actual analysis appears to use N=49 (after duplicate removal)
   - OR exclude one more participant to reach N=48

================================================================================
