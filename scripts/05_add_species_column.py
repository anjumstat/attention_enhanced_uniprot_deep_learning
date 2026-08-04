# -*- coding: utf-8 -*-
"""
add_species_column.py
Add species column to clean plant data for LOSO evaluation
"""

import pandas as pd
import os

# Paths
CLEAN_DATA_PATH = r"D:\uni_prot2\revision\data\clean_splits\combined_clean.csv"
ORIGINAL_DATA_PATH = r"D:\uni_prot2\revision\data\processed_results_v3\combined_all_species.csv"
OUTPUT_PATH = r"D:\uni_prot2\revision\data\clean_splits\combined_clean_with_species.csv"

print("=" * 80)
print("ADD SPECIES COLUMN TO CLEAN PLANT DATA")
print("=" * 80)

# =============================================
# 1. Load clean data (embeddings + target only)
# =============================================
print("\n📊 Loading clean data...")
clean_df = pd.read_csv(CLEAN_DATA_PATH)
print(f"   Clean data shape: {clean_df.shape}")
print(f"   Columns: {clean_df.columns.tolist()[:5]}...")

# =============================================
# 2. Load original data (has protein_id and species)
# =============================================
print("\n📊 Loading original data with species...")
original_df = pd.read_csv(ORIGINAL_DATA_PATH)
print(f"   Original data shape: {original_df.shape}")
print(f"   Columns: {original_df.columns.tolist()[:5]}...")

# Check if protein_id exists in original
if 'protein_id' not in original_df.columns:
    print("❌ 'protein_id' not found in original data!")
    print(f"   Available columns: {original_df.columns.tolist()}")
    # Try to find ID column
    possible_id_cols = ['Entry', 'UniProt_ID', 'id', 'protein_id']
    for col in possible_id_cols:
        if col in original_df.columns:
            print(f"   Using '{col}' as ID column")
            id_col = col
            break
else:
    id_col = 'protein_id'

# =============================================
# 3. Create species mapping
# =============================================
print(f"\n📊 Creating species mapping from '{id_col}'...")
species_map = dict(zip(original_df[id_col], original_df['species']))
print(f"   Species map size: {len(species_map)}")

# Check sample
sample_keys = list(species_map.keys())[:3]
print(f"   Sample mapping: {sample_keys[0]} -> {species_map[sample_keys[0]]}")

# =============================================
# 4. Add species column to clean data
# =============================================
print("\n📊 Adding species column to clean data...")

# Since clean data has no protein_id, we need to figure out
# how to map species. The clean data was created from original data
# but row order may be different.

# Option A: If clean data has same index as original (unlikely)
# Option B: If clean data has a protein_id column hidden
# Option C: If clean data rows are in same order as original

# Check if any column looks like protein_id
for col in clean_df.columns:
    if col.lower() in ['protein_id', 'entry', 'uniprot_id', 'id', 'protein']:
        print(f"   Found ID column in clean data: '{col}'")
        id_col_clean = col
        break
else:
    # If no ID column, we need to check if rows are in same order
    print("   No ID column found in clean data.")
    print("   Assuming rows are in same order as original data...")
    
    # Check if length matches
    if len(clean_df) == len(original_df):
        print(f"   ✅ Length matches: {len(clean_df)}")
        # Add species from original (assuming same order)
        clean_df['species'] = original_df['species'].values
        print("   ✅ Species column added (same order assumed)")
    else:
        print(f"   ❌ Length mismatch: clean={len(clean_df)}, original={len(original_df)}")
        print("   Cannot add species column automatically.")
        exit()

# =============================================
# 5. Check species distribution
# =============================================
print("\n📊 Species distribution:")
species_counts = clean_df['species'].value_counts()
for species, count in species_counts.items():
    print(f"   {species}: {count} samples")

# =============================================
# 6. Save new file
# =============================================
print(f"\n💾 Saving to: {OUTPUT_PATH}")
clean_df.to_csv(OUTPUT_PATH, index=False)
print(f"   ✅ Saved! Shape: {clean_df.shape}")
print(f"   Columns: {clean_df.columns.tolist()[:5]}... + species")

print("\n" + "=" * 80)
print("✅ DONE! Species column added successfully.")
print("=" * 80)