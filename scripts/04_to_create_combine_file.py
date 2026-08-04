# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 23:18:32 2026

@author: H.A.R
"""

# create_combined.py
import pandas as pd
import os

SPLITS_DIR = r"D:\uni_prot2\revision\data\clean_splits\homology_aware"

train = pd.read_csv(os.path.join(SPLITS_DIR, 'train.csv'))
val = pd.read_csv(os.path.join(SPLITS_DIR, 'val.csv'))

combined = pd.concat([train, val], ignore_index=True)
combined.to_csv(os.path.join(SPLITS_DIR, 'combined_train_val.csv'), index=False)

print(f"✅ Combined: {len(combined)} samples")