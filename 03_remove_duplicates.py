# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:51:41 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Remove Duplicate Rows from Combined Dataset
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# =============================================
# Configuration
# =============================================
DATA_PATH = 'D:/uni_prot2/processed_results/combined_data/all_embeddings_combined_processed.csv'
OUTPUT_DIR = 'D:/uni_prot2/processed_results/combined_data'
BACKUP_DIR = 'D:/uni_prot2/processed_results/backups'

# Create backup directory if it doesn't exist
os.makedirs(BACKUP_DIR, exist_ok=True)

def remove_duplicates_and_save():
    """Remove duplicate rows from the dataset and save cleaned version"""
    
    print("="*80)
    print("REMOVING DUPLICATE ROWS FROM DATASET")
    print("="*80)
    
    # Load the data
    print(f"\n📂 Loading data from: {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"✅ Data loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Display initial information
    print(f"\n📊 INITIAL DATASET INFORMATION:")
    print(f"   - Total samples: {len(df):,}")
    print(f"   - Total features: {df.shape[1]}")
    print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Check class distribution before removal
    if 'category' in df.columns:
        print(f"\n   Class distribution BEFORE duplicate removal:")
        print(f"   - Enzymes (1): {(df['category']==1).sum():,} ({((df['category']==1).sum()/len(df))*100:.2f}%)")
        print(f"   - Non-enzymes (0): {(df['category']==0).sum():,} ({((df['category']==0).sum()/len(df))*100:.2f}%)")
    
    # =============================================
    # Identify and remove duplicates
    # =============================================
    print(f"\n🔍 Analyzing duplicates...")
    
    # Find duplicate rows
    duplicate_mask = df.duplicated(keep='first')
    duplicate_count = duplicate_mask.sum()
    duplicate_percentage = (duplicate_count / len(df)) * 100
    
    print(f"   - Duplicate rows found: {duplicate_count:,} ({duplicate_percentage:.2f}%)")
    
    if duplicate_count == 0:
        print(f"\n✅ No duplicate rows found! Dataset is clean.")
        return df
    
    # Show sample of duplicates
    if duplicate_count > 0:
        duplicate_rows = df[duplicate_mask]
        print(f"\n   Sample of duplicate rows (first 5):")
        print(duplicate_rows.head(5).to_string())
        
        # Show which rows are duplicated most frequently
        duplicate_groups = df[df.duplicated(keep=False)].groupby(list(df.columns)).size().sort_values(ascending=False)
        if len(duplicate_groups) > 0:
            print(f"\n   Most frequent duplicates (top 5):")
            for i, (_, count) in enumerate(duplicate_groups.head(5).items()):
                if i < 5:
                    print(f"      - Appears {count} times")
    
    # =============================================
    # Create backup before removal
    # =============================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"all_species_backup_{timestamp}.csv"
    backup_path = os.path.join(BACKUP_DIR, backup_filename)
    
    print(f"\n💾 Creating backup...")
    df.to_csv(backup_path, index=False)
    print(f"   ✅ Backup saved to: {backup_path}")
    
    # =============================================
    # Remove duplicates (keep first occurrence)
    # =============================================
    print(f"\n🗑️ Removing duplicates...")
    
    # Option 1: Keep first occurrence of each duplicate
    df_cleaned = df.drop_duplicates(keep='first')
    
    # Alternative options (uncomment if needed):
    # df_cleaned = df.drop_duplicates(keep='last')  # Keep last occurrence
    # df_cleaned = df.drop_duplicates(keep=False)   # Remove all duplicates
    
    print(f"   - Samples before removal: {len(df):,}")
    print(f"   - Samples after removal: {len(df_cleaned):,}")
    print(f"   - Removed: {len(df) - len(df_cleaned):,} duplicate rows")
    
    # =============================================
    # Verify removal
    # =============================================
    print(f"\n🔍 Verifying removal...")
    
    # Check if any duplicates remain
    remaining_duplicates = df_cleaned.duplicated().sum()
    print(f"   - Duplicates remaining: {remaining_duplicates}")
    
    if remaining_duplicates == 0:
        print(f"   ✅ No duplicates remain!")
    else:
        print(f"   ⚠️ Warning: {remaining_duplicates} duplicates still exist!")
    
    # Check class distribution after removal
    if 'category' in df_cleaned.columns:
        print(f"\n   Class distribution AFTER duplicate removal:")
        enzymes_after = (df_cleaned['category']==1).sum()
        non_enzymes_after = (df_cleaned['category']==0).sum()
        print(f"   - Enzymes (1): {enzymes_after:,} ({((enzymes_after)/len(df_cleaned))*100:.2f}%)")
        print(f"   - Non-enzymes (0): {non_enzymes_after:,} ({((non_enzymes_after)/len(df_cleaned))*100:.2f}%)")
        
        # Show change in distribution
        enzymes_before = (df['category']==1).sum()
        print(f"\n   Change in class distribution:")
        print(f"   - Enzymes: {enzymes_before:,} → {enzymes_after:,} ({(enzymes_after - enzymes_before):+,})")
        print(f"   - Non-enzymes: {len(df)-enzymes_before:,} → {len(df_cleaned)-enzymes_after:,} ({(len(df_cleaned)-enzymes_after) - (len(df)-enzymes_before):+,})")
    
    # =============================================
    # Save cleaned dataset
    # =============================================
    print(f"\n💾 Saving cleaned dataset...")
    
    # Save as CSV
    cleaned_path = os.path.join(OUTPUT_DIR, "all_species_cleaned.csv")
    df_cleaned.to_csv(cleaned_path, index=False)
    print(f"   ✅ Cleaned dataset saved to: {cleaned_path}")
    
    # Also save as compressed CSV (optional)
    cleaned_gz_path = os.path.join(OUTPUT_DIR, "all_species_cleaned.csv.gz")
    df_cleaned.to_csv(cleaned_gz_path, index=False, compression='gzip')
    print(f"   ✅ Compressed version saved to: {cleaned_gz_path}")
    
    # Save summary statistics
    summary = {
        'Original_Samples': len(df),
        'Duplicate_Samples': duplicate_count,
        'Duplicate_Percentage': duplicate_percentage,
        'Cleaned_Samples': len(df_cleaned),
        'Removed_Samples': len(df) - len(df_cleaned),
        'Enzymes_Before': enzymes_before if 'category' in df.columns else 'N/A',
        'Enzymes_After': enzymes_after if 'category' in df_cleaned.columns else 'N/A',
        'Non_Enzymes_Before': len(df) - enzymes_before if 'category' in df.columns else 'N/A',
        'Non_Enzymes_After': len(df_cleaned) - enzymes_after if 'category' in df_cleaned.columns else 'N/A',
        'Backup_File': backup_filename,
        'Cleaned_File': "all_species_cleaned.csv",
        'Timestamp': timestamp
    }
    
    summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
    summary_path = os.path.join(OUTPUT_DIR, "duplicate_removal_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"   ✅ Summary saved to: {summary_path}")
    
    # =============================================
    # Final summary
    # =============================================
    print("\n" + "="*80)
    print("✅ DUPLICATE REMOVAL COMPLETE!")
    print("="*80)
    print(f"\n📊 SUMMARY:")
    print(f"   - Original samples: {len(df):,}")
    print(f"   - Duplicates removed: {duplicate_count:,} ({duplicate_percentage:.2f}%)")
    print(f"   - Final samples: {len(df_cleaned):,}")
    print(f"   - Reduction: {(1 - len(df_cleaned)/len(df))*100:.2f}%")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"   - Cleaned dataset: {cleaned_path}")
    print(f"   - Compressed version: {cleaned_gz_path}")
    print(f"   - Backup (original): {backup_path}")
    print(f"   - Summary report: {summary_path}")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Update your deep learning code to use: {cleaned_path}")
    print(f"   2. Re-run verification on cleaned dataset")
    print(f"   3. Proceed with deep learning training")
    
    return df_cleaned

# =============================================
# Optional: Create a function to verify the cleaned data
# =============================================

def verify_cleaned_data(cleaned_path):
    """Quick verification of cleaned dataset"""
    
    print("\n" + "="*80)
    print("VERIFYING CLEANED DATASET")
    print("="*80)
    
    df = pd.read_csv(cleaned_path)
    
    print(f"\n📊 Cleaned Dataset Statistics:")
    print(f"   - Total samples: {len(df):,}")
    print(f"   - Total features: {df.shape[1]}")
    print(f"   - Duplicate rows: {df.duplicated().sum()}")
    
    if 'category' in df.columns:
        print(f"\n   Class Distribution:")
        print(f"   - Enzymes (1): {(df['category']==1).sum():,} ({((df['category']==1).sum()/len(df))*100:.2f}%)")
        print(f"   - Non-enzymes (0): {(df['category']==0).sum():,} ({((df['category']==0).sum()/len(df))*100:.2f}%)")
    
    print(f"\n   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

# =============================================
# Update your deep learning code to use cleaned data
# =============================================

def update_deep_learning_code():
    """Print instructions for updating deep learning code"""
    
    print("\n" + "="*80)
    print("📝 UPDATE YOUR DEEP LEARNING CODE")
    print("="*80)
    
    print("\nReplace this line in your deep learning code:")
    print("   DATA_PATH = 'D:/uni_prot1/processed_results/combined_data/all_species.csv'")
    print("\nWith:")
    print("   DATA_PATH = 'D:/uni_prot1/processed_results/combined_data/all_species_cleaned.csv'")
    
    print("\nOr add this at the beginning of your deep learning code:")
    print("""
    # Load cleaned data (no duplicates)
    DATA_PATH = 'D:/uni_prot1/processed_results/combined_data/all_species_cleaned.csv'
    data = pd.read_csv(DATA_PATH)
    
    # Verify no duplicates
    print(f"Dataset shape: {data.shape}")
    print(f"Duplicate rows: {data.duplicated().sum()}")
    """)

# =============================================
# Main execution
# =============================================

if __name__ == "__main__":
    # Remove duplicates and save cleaned dataset
    cleaned_df = remove_duplicates_and_save()
    
    if cleaned_df is not None:
        # Verify the cleaned data
        cleaned_path = os.path.join(OUTPUT_DIR, "all_species_cleaned.csv")
        verify_cleaned_data(cleaned_path)
        
        # Show how to update deep learning code
        update_deep_learning_code()
        
        print("\n" + "="*80)
        print("✅ All tasks completed successfully!")
        print("="*80)