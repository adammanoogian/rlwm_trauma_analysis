import pandas as pd

# Load summary with all participants
df = pd.read_csv('output/summary_participant_metrics_all.csv')

print(f"Total participants: {len(df)}")
print(f"\nColumns ({len(df.columns)}):")
print([c for c in df.columns if any(x in c for x in ['age', 'gender', 'country', 'language', 'education'])])

# Check if demographic columns exist
if any('age' in c for c in df.columns):
    age_col = 'age' if 'age' in df.columns else 'age_years'
    age = df[age_col].dropna()
    
    print(f"\nAge statistics:")
    print(f"  N with age data: {len(age)}")
    print(f"  Mean: {age.mean():.1f} years")
    print(f"  SD: {age.std():.1f}")
    print(f"  Range: {int(age.min())}-{int(age.max())}")
    
if 'gender' in df.columns:
    print(f"\nGender distribution:")
    print(df['gender'].value_counts(dropna=False))
    
if 'country' in df.columns:
    print(f"\nCountry:")
    print(df['country'].value_counts(dropna=False))
