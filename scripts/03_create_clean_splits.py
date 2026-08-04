# -*- coding: utf-8 -*-
"""
03_create_clean_splits.py
Create clean CSV files with ONLY embeddings and target variable
No protein_id, no species, no ec_numbers - just features and labels
"""

import pandas as pd
import os
import numpy as np

# =============================================
# CONFIGURATION
# =============================================

class Config:
    INPUT_DIR = r"D:\uni_prot2\revision\data\evaluation_splits"
    OUTPUT_DIR = r"D:\uni_prot2\revision\data\clean_splits"
    EMBEDDING_DIM = 1024

# =============================================
# CREATE CLEAN SPLIT
# =============================================

def create_clean_split(input_file, output_file, embedding_dim=1024):
    """
    Read a CSV file and save only embeddings + target variable
    """
    print(f"📄 Processing: {os.path.basename(input_file)}")
    
    # Load data with low_memory=False to avoid dtype warnings
    df = pd.read_csv(input_file, low_memory=False)
    print(f"   Original shape: {df.shape}")
    
    # Get embedding column names
    emb_cols = [f'emb_{i}' for i in range(embedding_dim)]
    
    # Check if all embedding columns exist
    available_cols = [col for col in emb_cols if col in df.columns]
    if len(available_cols) != embedding_dim:
        print(f"   ⚠️ Found {len(available_cols)}/{embedding_dim} embedding columns")
    
    # Create clean dataframe with only embeddings and target
    clean_df = df[available_cols + ['is_enzyme']].copy()
    
    # Convert to proper types
    for col in available_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce').astype(np.float32)
    
    # Fix is_enzyme - convert to int (True/False -> 1/0)
    clean_df['is_enzyme'] = clean_df['is_enzyme'].astype(str).str.lower().map({'true': 1, 'false': 0, '1': 1, '0': 0}).fillna(0).astype(np.int8)
    
    # Calculate correctly
    total_samples = len(clean_df)
    enzymes = clean_df['is_enzyme'].sum()
    non_enzymes = total_samples - enzymes
    
    print(f"   Clean shape: {clean_df.shape}")
    print(f"   Total samples: {total_samples}")
    print(f"   Enzymes: {enzymes}")
    print(f"   Non-enzymes: {non_enzymes}")
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    clean_df.to_csv(output_file, index=False)
    print(f"   ✅ Saved to: {output_file}\n")
    
    return clean_df

# =============================================
# MAIN EXECUTION
# =============================================

def main():
    print("=" * 80)
    print("CREATE CLEAN SPLITS (Embeddings + Target Only)")
    print("=" * 80)
    
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # =============================================
    # 1. Process Homology-Aware Splits
    # =============================================
    print("\n" + "=" * 80)
    print("PROCESSING HOMOLOGY-AWARE SPLITS")
    print("=" * 80)
    
    homology_dir = os.path.join(config.INPUT_DIR, 'homology_aware')
    if os.path.exists(homology_dir):
        for split_name in ['train', 'val', 'test']:
            input_file = os.path.join(homology_dir, f'{split_name}.csv')
            if os.path.exists(input_file):
                output_file = os.path.join(config.OUTPUT_DIR, 'homology_aware', f'{split_name}.csv')
                create_clean_split(input_file, output_file, config.EMBEDDING_DIM)
    else:
        print("⚠️ Homology-aware splits not found!")
    
    # =============================================
    # 2. Process LOSO Splits
    # =============================================
    print("\n" + "=" * 80)
    print("PROCESSING LOSO SPLITS")
    print("=" * 80)
    
    loso_dir = os.path.join(config.INPUT_DIR, 'loso')
    if os.path.exists(loso_dir):
        for species in os.listdir(loso_dir):
            species_dir = os.path.join(loso_dir, species)
            if os.path.isdir(species_dir):
                print(f"\n📁 Species: {species}")
                
                for split_name in ['train', 'test']:
                    input_file = os.path.join(species_dir, f'{split_name}.csv')
                    if os.path.exists(input_file):
                        output_file = os.path.join(config.OUTPUT_DIR, 'loso', species, f'{split_name}.csv')
                        create_clean_split(input_file, output_file, config.EMBEDDING_DIM)
    else:
        print("⚠️ LOSO splits not found!")
    
    # =============================================
    # 3. Create Combined Clean Dataset
    # =============================================
    print("\n" + "=" * 80)
    print("CREATING COMBINED CLEAN DATASET")
    print("=" * 80)
    
    # Load the original combined data
    combined_path = os.path.join(config.INPUT_DIR, '..', 'processed_results_v3', 'combined_all_species.csv')
    if os.path.exists(combined_path):
        df = pd.read_csv(combined_path, low_memory=False)
        
        emb_cols = [f'emb_{i}' for i in range(config.EMBEDDING_DIM)]
        available_cols = [col for col in emb_cols if col in df.columns]
        
        clean_df = df[available_cols + ['is_enzyme']].copy()
        
        for col in available_cols:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce').astype(np.float32)
        
        clean_df['is_enzyme'] = clean_df['is_enzyme'].astype(str).str.lower().map({'true': 1, 'false': 0, '1': 1, '0': 0}).fillna(0).astype(np.int8)
        
        total_samples = len(clean_df)
        enzymes = clean_df['is_enzyme'].sum()
        non_enzymes = total_samples - enzymes
        
        output_file = os.path.join(config.OUTPUT_DIR, 'combined_clean.csv')
        clean_df.to_csv(output_file, index=False)
        print(f"✅ Combined clean dataset saved to: {output_file}")
        print(f"   Shape: {clean_df.shape}")
        print(f"   Total samples: {total_samples}")
        print(f"   Enzymes: {enzymes}")
        print(f"   Non-enzymes: {non_enzymes}")
    
    # =============================================
    # 4. Summary
    # =============================================
    print("\n" + "=" * 80)
    print("✅ CLEAN SPLITS CREATION COMPLETE")
    print("=" * 80)
    print(f"All clean splits saved to: {config.OUTPUT_DIR}")
    print("\n📁 Directory structure:")
    print(f"  {config.OUTPUT_DIR}/")
    print(f"  ├── combined_clean.csv")
    print(f"  ├── homology_aware/")
    print(f"  │   ├── train.csv  (Features: 1024 embeddings, Target: is_enzyme)")
    print(f"  │   ├── val.csv")
    print(f"  │   └── test.csv")
    print(f"  └── loso/")
    print(f"      ├── Arabidopsis_thaliana/")
    print(f"      │   ├── train.csv")
    print(f"      │   └── test.csv")
    print(f"      ├── Brassica_spp/")
    print(f"      │   ├── train.csv")
    print(f"      │   └── test.csv")
    print(f"      ├── Oryza_sativa/")
    print(f"      │   ├── train.csv")
    print(f"      │   └── test.csv")
    print(f"      └── Triticum_aestivum/")
    print(f"          ├── train.csv")
    print(f"          └── test.csv")
    print("=" * 80)

if __name__ == "__main__":
    main()