import pandas as pd
import os

# Check available files
m1 = pd.read_csv('output/mle/qlearning_individual_fits_matched.csv')
m2_matched = pd.read_csv('output/mle/wmrl_individual_fits_matched.csv')

print(f'Q-learning (matched from daa3fc8): N={len(m1)}')
print(f'WM-RL (matched): N={len(m2_matched)}')

if os.path.exists('output/mle/wmrl_individual_fits.csv'):
    m2_full = pd.read_csv('output/mle/wmrl_individual_fits.csv')
    print(f'WM-RL (full/recent): N={len(m2_full)}')
    print(f'IDs in M1 but not M2_full: {set(m1.sona_id) - set(m2_full.sona_id)}')
    print(f'IDs in M2_full but not M1: {set(m2_full.sona_id) - set(m1.sona_id)}')
else:
    print('WM-RL (full): File not found')
