# -*- coding: utf-8 -*-
"""
Unified UniProt Data Processor
Processes multiple folders of UniProt data, combines embeddings with enzyme classification,
and provides detailed tracking statistics.
"""

import h5py
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import glob

# =============================================
# 1. Configuration and Path Setup
# =============================================
base_data_dir = r"D:\uni_prot\data"
output_base_dir = r"D:\uni_prot2\processed_results"
os.makedirs(output_base_dir, exist_ok=True)

# Create subdirectories for organized output
processed_dir = os.path.join(output_base_dir, "processed_embeddings")
combined_dir = os.path.join(output_base_dir, "combined_data")
stats_dir = os.path.join(output_base_dir, "statistics")
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(combined_dir, exist_ok=True)
os.makedirs(stats_dir, exist_ok=True)

# =============================================
# 2. Helper Functions
# =============================================

def find_files(folder_path, folder_name):
    """Find HDF5 and CSV files with flexible naming"""
    print(f"  Searching for files in {folder_name}...")
    
    # List all files in the folder
    all_files = os.listdir(folder_path)
    print(f"  All files in folder: {all_files}")
    
    # Look for HDF5 files - check ALL files without extension filtering
    hdf5_files = []
    csv_files = []
    
    for file in all_files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            # Check if it's a CSV file
            if file.lower().endswith('.csv'):
                csv_files.append(file_path)
            # Check if it's an Excel file (we can use these too)
            elif file.lower().endswith(('.xlsx', '.xls')):
                csv_files.append(file_path)  # We'll handle Excel files separately
            # Check if it's a gzip file (skip these for now)
            elif file.lower().endswith('.gz'):
                continue
            # For all other files, test if they are HDF5
            else:
                if is_hdf5_file(file_path):
                    hdf5_files.append(file_path)
                    print(f"    Found HDF5 file: {file}")
                else:
                    print(f"    Skipping non-HDF5 file: {file}")
    
    print(f"  Final HDF5 files: {[os.path.basename(f) for f in hdf5_files]}")
    print(f"  Final CSV files: {[os.path.basename(f) for f in csv_files]}")
    
    return hdf5_files, csv_files

def is_hdf5_file(file_path):
    """Check if file is actually an HDF5 file"""
    try:
        with h5py.File(file_path, 'r') as f:
            # Try to read some basic info to verify it's HDF5
            keys = list(f.keys())
            if len(keys) > 0:
                print(f"    ✓ Valid HDF5: {os.path.basename(file_path)} with {len(keys)} keys")
                return True
            return False
    except Exception as e:
        return False

def process_hdf5_embeddings(hdf5_path, folder_name):
    """Process HDF5 embeddings file and return structured data"""
    print(f"  Processing HDF5 embeddings for {folder_name}...")
    
    uniprot_ids = []
    embeddings = []
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            protein_ids = list(f.keys())
            print(f"    Found {len(protein_ids)} protein entries")
            
            # Show first few protein IDs as samples
            for i, pid in enumerate(protein_ids[:3]):
                embedding_shape = f[pid][:].shape
                print(f"      Sample {i+1}: {pid} -> embedding shape: {embedding_shape}")
            
            for pid in protein_ids:
                uniprot_ids.append(pid)
                embeddings.append(f[pid][:])
        
        # Convert to arrays
        uniprot_ids = np.array(uniprot_ids)
        embeddings = np.vstack(embeddings).astype(np.float32)
        
        print(f"    Successfully processed {len(uniprot_ids)} embeddings")
        print(f"    Final embedding matrix shape: {embeddings.shape}")
        
        return uniprot_ids, embeddings, len(uniprot_ids)
    
    except Exception as e:
        print(f"  ERROR processing HDF5 for {folder_name}: {str(e)}")
        return None, None, 0

def classify_enzymes(csv_path, folder_name):
    """Classify proteins as enzymes based on EC numbers"""
    print(f"  Classifying enzymes for {folder_name}...")
    
    try:
        # Handle different file types
        if csv_path.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(csv_path)
            print(f"    Reading Excel file: {os.path.basename(csv_path)}")
        else:
            # Try different encodings for CSV files
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding)
                    print(f"    Successfully read CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                print(f"    ERROR: Could not read CSV file with any encoding")
                return None, 0, 0
        
        # Show available columns for debugging
        print(f"    File shape: {df.shape}")
        print(f"    Available columns: {df.columns.tolist()}")
        
        # Show first column sample as that's usually the ID column
        if len(df) > 0:
            first_col = df.columns[0]
            sample_ids = df[first_col].head(3).tolist()
            print(f"    First column '{first_col}' samples: {sample_ids}")
        
        # Find EC column (look for columns that might contain EC information)
        ec_column = None
        possible_ec_columns = ['EC number', 'EC numbers', 'EC', 'EC_number', 'Protein names', 
                              'Function [CC]', 'Description', 'Annotation', 'Function']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(ec_term in col_lower for ec_term in ['ec', 'enzyme']):
                ec_column = col
                print(f"    Found EC column by name: {col}")
                break
        
        # If no specific EC column found, try to find it by content
        if ec_column is None:
            for col in df.columns:
                if len(df) > 0:
                    # Sample some values to check if they contain EC patterns
                    sample_values = df[col].dropna().head(5).astype(str)
                    ec_patterns = ['EC:', 'EC ', 'EC-', 'EC_', 'EC']
                    if any(any(pattern in val for pattern in ec_patterns) for val in sample_values):
                        ec_column = col
                        print(f"    Found EC column by content pattern: {col}")
                        print(f"    Sample values: {sample_values.tolist()}")
                        break
        
        # If still no EC column found, use the 4th column (index 3) as in original code
        if ec_column is None and len(df.columns) > 3:
            ec_column = df.columns[3]
            print(f"    Using default column {ec_column} (index 3) for EC classification")
        elif ec_column:
            print(f"    Using identified EC column: {ec_column}")
        else:
            print(f"    WARNING: No suitable EC column found in {folder_name}")
            # Try all columns to see if any contain EC patterns
            print(f"    Checking all columns for EC patterns:")
            for col in df.columns:
                if len(df) > 0:
                    sample_val = str(df[col].iloc[0]) if pd.notna(df[col].iloc[0]) else "NaN"
                    print(f"      '{col}': {sample_val[:80]}...")
            return None, 0, 0
        
        # Function to determine if a gene is an enzyme
        def is_enzyme(ec_text):
            if pd.isna(ec_text):
                return "Non-enzyme"
            ec_str = str(ec_text)
            # Check for EC number pattern (EC followed by numbers)
            if "EC " in ec_str or ("EC" in ec_str and any(char.isdigit() for char in ec_str)):
                return "Enzyme"
            # Also check for EC: pattern
            if "EC:" in ec_str:
                return "Enzyme"
            return "Non-enzyme"
        
        # Add classification column
        df['Enzyme_Classification'] = df[ec_column].apply(is_enzyme)
        
        enzyme_count = (df['Enzyme_Classification'] == 'Enzyme').sum()
        non_enzyme_count = (df['Enzyme_Classification'] == 'Non-enzyme').sum()
        
        print(f"    Classification results: {enzyme_count} enzymes, {non_enzyme_count} non-enzymes")
        
        # Show some examples
        enzyme_samples = df[df['Enzyme_Classification'] == 'Enzyme'].head(2)
        if len(enzyme_samples) > 0:
            print(f"    Enzyme samples: {enzyme_samples[ec_column].tolist()}")
        
        return df, enzyme_count, non_enzyme_count
        
    except Exception as e:
        print(f"  ERROR classifying enzymes for {folder_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, 0, 0

def merge_embeddings_with_labels(embeddings_df, enzyme_df, folder_name):
    """Merge embeddings with enzyme classification labels"""
    print(f"  Merging embeddings with labels for {folder_name}...")
    
    try:
        # Get the gene ID column names
        enzyme_gene_col = enzyme_df.columns[0]
        embeddings_gene_col = embeddings_df.columns[0]
        
        print(f"    Enzyme gene column: {enzyme_gene_col}")
        print(f"    Embeddings gene column: {embeddings_gene_col}")
        
        # Show sample IDs from both dataframes for debugging
        enzyme_sample_ids = enzyme_df[enzyme_gene_col].head(3).tolist()
        embeddings_sample_ids = embeddings_df[embeddings_gene_col].head(3).tolist()
        print(f"    Enzyme sample IDs: {enzyme_sample_ids}")
        print(f"    Embeddings sample IDs: {embeddings_sample_ids}")
        
        # Create mapping from Gene ID to Enzyme Classification
        id_to_label = dict(zip(enzyme_df[enzyme_gene_col], enzyme_df['Enzyme_Classification']))
        
        # Add labels to embeddings dataframe
        embeddings_df['Enzyme_Classification'] = embeddings_df[embeddings_gene_col].map(id_to_label)
        
        # Count matching results
        total_genes = len(embeddings_df)
        matched_genes = embeddings_df['Enzyme_Classification'].notna().sum()
        missing_genes = total_genes - matched_genes
        
        print(f"    Matching results: {matched_genes}/{total_genes} genes matched with labels")
        
        if missing_genes > 0:
            print(f"    WARNING: {missing_genes} genes without labels")
            # Show some unmatched genes for debugging
            unmatched = embeddings_df[embeddings_df['Enzyme_Classification'].isna()][embeddings_gene_col].head(5)
            print(f"    Sample unmatched genes: {list(unmatched)}")
            
            # Also show some enzyme IDs that didn't match
            enzyme_ids_set = set(enzyme_df[enzyme_gene_col])
            embeddings_ids_set = set(embeddings_df[embeddings_gene_col])
            missing_in_embeddings = enzyme_ids_set - embeddings_ids_set
            if missing_in_embeddings:
                print(f"    Sample enzyme IDs missing in embeddings: {list(missing_in_embeddings)[:3]}")
        
        # Fill missing labels with 'Unknown'
        embeddings_df['Enzyme_Classification'] = embeddings_df['Enzyme_Classification'].fillna('Unknown')
        
        # Add folder name for tracking
        embeddings_df['Data_Source'] = folder_name
        
        return embeddings_df, total_genes, matched_genes, missing_genes
        
    except Exception as e:
        print(f"  ERROR merging data for {folder_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, 0, 0, 0

# =============================================
# 3. Main Processing Pipeline
# =============================================

def main():
    print("Starting unified UniProt data processing...")
    print("=" * 60)
    
    # Statistics tracking
    statistics = {
        'Folder': [],
        'HDF5_Genes': [],
        'CSV_Genes': [],
        'Enzymes_Count': [],
        'Non_Enzymes_Count': [],
        'Matched_Genes': [],
        'Missing_Labels': [],
        'Final_Genes': []
    }
    
    all_combined_data = []
    
    # Get all folders in the data directory
    folders = [f for f in os.listdir(base_data_dir) 
               if os.path.isdir(os.path.join(base_data_dir, f))]
    
    print(f"Found {len(folders)} folders to process: {folders}")
    print("=" * 60)
    
    for folder_name in folders:
        print(f"\nProcessing folder: {folder_name}")
        print("-" * 40)
        
        folder_path = os.path.join(base_data_dir, folder_name)
        
        # Find files with improved detection
        hdf5_files, csv_files = find_files(folder_path, folder_name)
        
        if not hdf5_files:
            print(f"  ❌ No HDF5 files found in {folder_name}")
            continue
        
        if not csv_files:
            print(f"  ❌ No CSV files found in {folder_name}")
            continue
        
        hdf5_path = hdf5_files[0]  # Use first valid HDF5 file
        csv_path = csv_files[0]    # Use first CSV/Excel file
        
        print(f"  Using HDF5 file: {os.path.basename(hdf5_path)}")
        print(f"  Using data file: {os.path.basename(csv_path)}")
        
        # Step 1: Process HDF5 embeddings
        uniprot_ids, embeddings, hdf5_gene_count = process_hdf5_embeddings(hdf5_path, folder_name)
        if uniprot_ids is None:
            continue
            
        # Step 2: Save processed embeddings for individual folder
        folder_output_dir = os.path.join(processed_dir, folder_name)
        os.makedirs(folder_output_dir, exist_ok=True)
        
        # Save embeddings
        np.save(os.path.join(folder_output_dir, "uniprot_ids.npy"), uniprot_ids)
        np.save(os.path.join(folder_output_dir, "embeddings.npy"), embeddings)
        
        # Create and save embeddings DataFrame
        embeddings_df = pd.DataFrame(embeddings, index=uniprot_ids)
        embeddings_df.reset_index(inplace=True)
        embeddings_df.columns = ['UniProt_ID'] + [f'Embedding_{i}' for i in range(embeddings.shape[1])]
        
        embeddings_csv_path = os.path.join(folder_output_dir, "embeddings.csv")
        embeddings_df.to_csv(embeddings_csv_path, index=False)
        print(f"    Saved embeddings to: {embeddings_csv_path}")
        
        # Step 3: Classify enzymes from CSV/Excel
        enzyme_df, enzyme_count, non_enzyme_count = classify_enzymes(csv_path, folder_name)
        if enzyme_df is None:
            continue
        
        # Step 4: Merge embeddings with labels
        merged_df, total_genes, matched_genes, missing_genes = merge_embeddings_with_labels(
            embeddings_df, enzyme_df, folder_name
        )
        
        if merged_df is not None:
            # Save individual folder results
            individual_output_path = os.path.join(folder_output_dir, "labeled_embeddings.csv")
            merged_df.to_csv(individual_output_path, index=False)
            print(f"    Saved labeled embeddings to: {individual_output_path}")
            
            # Add to combined data
            all_combined_data.append(merged_df)
            
            # Update statistics
            statistics['Folder'].append(folder_name)
            statistics['HDF5_Genes'].append(hdf5_gene_count)
            statistics['CSV_Genes'].append(len(enzyme_df))
            statistics['Enzymes_Count'].append(enzyme_count)
            statistics['Non_Enzymes_Count'].append(non_enzyme_count)
            statistics['Matched_Genes'].append(matched_genes)
            statistics['Missing_Labels'].append(missing_genes)
            statistics['Final_Genes'].append(len(merged_df))
            
            print(f"  ✅ Successfully processed {folder_name}: {len(merged_df)} genes")
        
        print("-" * 40)
    
    # =============================================
    # 4. Combine All Data and Save Results
    # =============================================
    
    if all_combined_data:
        print(f"\nCombining data from all folders...")
        
        # Combine all data
        combined_df = pd.concat(all_combined_data, ignore_index=True)
        
        # Save combined data
        combined_output_path = os.path.join(combined_dir, "all_embeddings_combined.csv")
        combined_df.to_csv(combined_output_path, index=False)
        
        print(f"✅ Combined data saved: {combined_output_path}")
        print(f"  Total genes in combined dataset: {len(combined_df)}")
        
        # Save detailed statistics
        stats_df = pd.DataFrame(statistics)
        stats_output_path = os.path.join(stats_dir, "processing_statistics.csv")
        stats_df.to_csv(stats_output_path, index=False)
        
        print(f"✅ Statistics saved: {stats_output_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        for i, folder in enumerate(statistics['Folder']):
            print(f"\n{folder}:")
            print(f"  HDF5 Genes: {statistics['HDF5_Genes'][i]}")
            print(f"  CSV Genes: {statistics['CSV_Genes'][i]}")
            print(f"  Enzymes: {statistics['Enzymes_Count'][i]}")
            print(f"  Non-enzymes: {statistics['Non_Enzymes_Count'][i]}")
            print(f"  Matched genes: {statistics['Matched_Genes'][i]}")
            print(f"  Missing labels: {statistics['Missing_Labels'][i]}")
            print(f"  Final genes: {statistics['Final_Genes'][i]}")
        
        # Overall statistics
        print(f"\nOVERALL TOTALS:")
        print(f"  Total folders processed: {len(statistics['Folder'])}")
        print(f"  Total genes in combined dataset: {len(combined_df)}")
        print(f"  Distribution by folder:")
        folder_distribution = combined_df['Data_Source'].value_counts()
        for folder, count in folder_distribution.items():
            print(f"    {folder}: {count} genes")
        
        print(f"\n  Enzyme classification distribution:")
        enzyme_distribution = combined_df['Enzyme_Classification'].value_counts()
        for classification, count in enzyme_distribution.items():
            print(f"    {classification}: {count} genes")
    
    else:
        print("\n❌ No data was successfully processed!")
    
    print(f"\nAll results saved in: {output_base_dir}")
    print("Processing complete!")

# =============================================
# 5. Run the Main Function
# =============================================

if __name__ == "__main__":
    main()