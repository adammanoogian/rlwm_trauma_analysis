"""
Extract demographic information from all participants with complete survey data.
"""
import pandas as pd
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Load participant mapping
with open('data/participant_id_mapping.json', 'r') as f:
    id_mapping = json.load(f)

# Load participants with complete survey data
survey_df = pd.read_csv('output/summary_participant_metrics.csv')
participants_with_surveys = set(survey_df['sona_id'].astype(int).values)

print(f"Extracting demographics for {len(participants_with_surveys)} participants with survey data...")

demographics_list = []

for filename, info in id_mapping.items():
    participant_id = info['assigned_id']
    
    # Only process participants with complete survey data
    if participant_id not in participants_with_surveys:
        continue
    
    filepath = Path('data') / filename
    
    if not filepath.exists():
        print(f"  Warning: File not found for ID {participant_id}")
        continue
    
    try:
        # Load the CSV
        df = pd.read_csv(filepath)
        
        # Find demographics rows (usually has 'demographics_mc' or 'demographics_text' in section column)
        if 'section' not in df.columns:
            continue
            
        demo_rows = df[df['section'].str.contains('demo', case=False, na=False)]
        
        if len(demo_rows) == 0:
            continue
        
        # Extract demographic information
        demo_data = {'sona_id': participant_id}
        
        # Try to get age
        if 'age_years' in df.columns:
            age_values = df['age_years'].dropna()
            if len(age_values) > 0:
                demo_data['age_years'] = age_values.iloc[0]
        
        # Try to get gender
        if 'gender' in df.columns:
            gender_values = df['gender'].dropna()
            if len(gender_values) > 0:
                demo_data['gender'] = gender_values.iloc[0]
        
        # Try to get country
        if 'country' in df.columns:
            country_values = df['country'].dropna()
            if len(country_values) > 0:
                demo_data['country'] = country_values.iloc[0]
        
        # Try to get language
        if 'primary_language' in df.columns:
            lang_values = df['primary_language'].dropna()
            if len(lang_values) > 0:
                demo_data['primary_language'] = lang_values.iloc[0]
        
        # Try to get education
        if 'education' in df.columns:
            edu_values = df['education'].dropna()
            if len(edu_values) > 0:
                demo_data['education'] = edu_values.iloc[0]
        
        # Try to get relationship status
        if 'relationship_status' in df.columns:
            rel_values = df['relationship_status'].dropna()
            if len(rel_values) > 0:
                demo_data['relationship_status'] = rel_values.iloc[0]
        
        # Try to get living arrangement
        if 'living_arrangement' in df.columns:
            living_values = df['living_arrangement'].dropna()
            if len(living_values) > 0:
                demo_data['living_arrangement'] = living_values.iloc[0]
        
        # Only add if we got at least age
        if 'age_years' in demo_data:
            demographics_list.append(demo_data)
            print(f"  ✓ ID {participant_id}: age={demo_data.get('age_years', 'N/A')}, gender={demo_data.get('gender', 'N/A')}")
        
    except Exception as e:
        print(f"  Error processing ID {participant_id}: {e}")
        continue

# Create DataFrame
demographics_df = pd.DataFrame(demographics_list)

# Sort by sona_id
demographics_df = demographics_df.sort_values('sona_id').reset_index(drop=True)

print(f"\n{'='*80}")
print(f"EXTRACTION COMPLETE")
print(f"{'='*80}")
print(f"Total participants with demographics: {len(demographics_df)}")
print(f"\nColumns extracted: {list(demographics_df.columns)}")

# Save
output_path = Path('output/demographics_complete.csv')
demographics_df.to_csv(output_path, index=False)
print(f"\n[SAVED] {output_path}")

# Print summary statistics
print(f"\n{'='*80}")
print(f"DEMOGRAPHIC SUMMARY")
print(f"{'='*80}")

if 'age_years' in demographics_df.columns:
    age_data = demographics_df['age_years'].dropna()
    print(f"\nAge (N={len(age_data)}):")
    print(f"  Mean: {age_data.mean():.1f} years")
    print(f"  SD: {age_data.std():.1f}")
    print(f"  Range: {int(age_data.min())}-{int(age_data.max())}")

if 'gender' in demographics_df.columns:
    print(f"\nGender:")
    gender_counts = demographics_df['gender'].value_counts(dropna=False)
    for gender, count in gender_counts.items():
        print(f"  {gender}: {count}")

if 'country' in demographics_df.columns:
    print(f"\nCountry:")
    country_counts = demographics_df['country'].value_counts(dropna=False)
    for country, count in country_counts.items():
        print(f"  {country}: {count}")

if 'primary_language' in demographics_df.columns:
    print(f"\nPrimary Language:")
    lang_counts = demographics_df['primary_language'].value_counts(dropna=False)
    for lang, count in lang_counts.items():
        print(f"  {lang}: {count}")

if 'education' in demographics_df.columns:
    print(f"\nEducation:")
    edu_counts = demographics_df['education'].value_counts(dropna=False)
    for edu, count in edu_counts.items():
        print(f"  {edu}: {count}")

print(f"\n{'='*80}")
