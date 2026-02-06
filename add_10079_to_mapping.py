import json

# Load current mapping
mapping = json.load(open('data/participant_id_mapping.json'))

# Add the January 27 file as participant 10079
filename = 'rlwm_trauma_PARTICIPANT_SESSION_2026-01-27_13h51.52.432.csv'

mapping[filename] = {
    'assigned_id': 10079,
    'n_trials': 807,
    'filename': filename
}

# Save updated mapping
with open('data/participant_id_mapping.json', 'w') as f:
    json.dump(mapping, f, indent=2)

print("✓ Updated participant_id_mapping.json")
print(f"  Added: {filename}")
print(f"  Mapped to: participant 10079")
print(f"  Trials: 807")
print("\nTotal participants in mapping:", len(mapping))
