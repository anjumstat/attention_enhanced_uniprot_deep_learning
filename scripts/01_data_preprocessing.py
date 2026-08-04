# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 20:10:44 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
UniProt Data Processor for Plant Enzyme Classification
Corrected EC number extraction based on actual data format
"""

import h5py
import numpy as np
import pandas as pd
import os
import json
import re
import gzip
import shutil
from datetime import datetime

# =============================================
# 1. Configuration
# =============================================

class Config:
    DATA_DIR = r"D:\uni_prot2\revision\data"
    OUTPUT_DIR = r"D:\uni_prot2\revision\data\processed_results_v3"
    
    # Species mapping with actual folder names
    SPECIES = {
        '3702': {
            'name': 'Arabidopsis_thaliana',
            'folder': 'thailian'
        },
        '3705': {
            'name': 'Brassica_spp',
            'folder': 'brassica'
        },
        '4530': {
            'name': 'Oryza_sativa',
            'folder': 'rice'
        },
        '4565': {
            'name': 'Triticum_aestivum',
            'folder': 'wheat'
        }
    }
    
    RANDOM_SEED = 42

# =============================================
# 2. Core Processing Functions
# =============================================

class UniProtDataProcessor:
    def __init__(self, config):
        self.config = config
        self.stats = {}
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        print(f"📁 Output directory: {self.config.OUTPUT_DIR}")
    
    def decompress_file(self, gz_path):
        """Decompress .gz file if needed"""
        if not gz_path.endswith('.gz'):
            return gz_path
        
        output_path = gz_path[:-3]
        if os.path.exists(output_path):
            print(f"  ✅ Decompressed file already exists: {os.path.basename(output_path)}")
            return output_path
        
        print(f"  📦 Decompressing: {os.path.basename(gz_path)}")
        try:
            with gzip.open(gz_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"  ✅ Decompressed to: {os.path.basename(output_path)}")
            return output_path
        except Exception as e:
            print(f"  ❌ Error decompressing: {str(e)}")
            return None
    
    def find_files_in_folder(self, folder_path):
        """Find TSV and HDF5 files in a folder"""
        tsv_file = None
        h5_file = None
        
        print(f"  📁 Scanning folder: {os.path.basename(folder_path)}")
        
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                # Find TSV file (not compressed)
                if file.endswith('.tsv') and not file.endswith('.gz'):
                    tsv_file = file_path
                    print(f"    📄 Found TSV: {file}")
                # Find HDF5 file (not compressed)
                elif file.endswith('.h5') and not file.endswith('.gz'):
                    h5_file = file_path
                    print(f"    📄 Found H5: {file}")
                # Also check for random name files that might be HDF5
                elif not file.endswith('.tsv') and not file.endswith('.tsv.gz') and not file.endswith('.gz'):
                    # Check if it might be an HDF5 file
                    if self.is_hdf5_file(file_path):
                        h5_file = file_path
                        print(f"    📄 Found HDF5 file: {file}")
        
        # If no TSV found but we have a compressed one, decompress it
        if tsv_file is None:
            for file in os.listdir(folder_path):
                if file.endswith('.tsv.gz'):
                    gz_path = os.path.join(folder_path, file)
                    tsv_file = self.decompress_file(gz_path)
                    if tsv_file:
                        print(f"    📄 Decompressed TSV: {os.path.basename(tsv_file)}")
                    break
        
        return tsv_file, h5_file
    
    def is_hdf5_file(self, file_path):
        """Check if file is an HDF5 file"""
        try:
            with h5py.File(file_path, 'r') as f:
                keys = list(f.keys())
                return len(keys) > 0
        except:
            return False
    
    def load_tsv_file(self, tsv_path):
        """Load TSV file with proper encoding"""
        print(f"  📄 Loading TSV: {os.path.basename(tsv_path)}")
        
        try:
            df = pd.read_csv(tsv_path, sep='\t', low_memory=False, encoding='utf-8')
            print(f"  ✅ Loaded {len(df)} entries")
            print(f"  📊 Columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            print(f"  ❌ Error loading TSV: {str(e)}")
            return None
    
    def extract_ec_numbers_corrected(self, ec_text):
        """
        Corrected EC number extraction based on actual data format.
        EC numbers appear as: 2.7.11.1, 3.6.4.-, 1.10.3.3, etc.
        Multiple ECs separated by semicolons: 2.5.1.3; 2.7.1.49
        """
        if pd.isna(ec_text):
            return []
        
        ec_text = str(ec_text).strip()
        if not ec_text:
            return []
        
        # Split by semicolon (multiple EC numbers)
        if ';' in ec_text:
            parts = ec_text.split(';')
        else:
            parts = [ec_text]
        
        ec_numbers = []
        
        # Pattern for EC numbers: digits.digits.digits.digits
        # Also handles partial numbers with - like 3.6.4.-, 3.1.-.-, etc.
        patterns = [
            r'(\d+\.\d+\.\d+\.\d+)',    # Full: 2.7.11.1
            r'(\d+\.\d+\.\d+\.-)',      # 3.6.4.-
            r'(\d+\.\d+\.-\.-)',        # 3.1.-.-
            r'(\d+\.-\.-\.-)',          # 3.-.-.-
            r'(\d+\.\d+\.\d+)',         # Partial: 1.2.3
        ]
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            extracted = False
            for pattern in patterns:
                match = re.search(pattern, part)
                if match:
                    ec_numbers.append(match.group(1))
                    extracted = True
                    break
            
            # If no pattern matched but it looks like an EC number format
            if not extracted and re.match(r'^[\d\.\-]+$', part):
                ec_numbers.append(part)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_ec = []
        for ec in ec_numbers:
            if ec not in seen:
                seen.add(ec)
                unique_ec.append(ec)
        
        return unique_ec
    
    def classify_enzymes(self, df):
        """Classify enzymes based on EC numbers"""
        # Find EC column - should be 'EC number'
        ec_col = None
        if 'EC number' in df.columns:
            ec_col = 'EC number'
        elif 'Catalytic activity' in df.columns:
            # Check if catalytic activity contains EC numbers
            ec_col = 'Catalytic activity'
        else:
            # Try to find by content
            for col in df.columns:
                if df[col].dtype == 'object':
                    sample = df[col].dropna().head(100)
                    if sample.astype(str).str.contains(r'\d+\.\d+\.\d+\.\d+', regex=True).any():
                        ec_col = col
                        break
        
        if ec_col is None:
            print(f"  ⚠️ No EC column found! Available columns: {df.columns.tolist()}")
            df['EC_list'] = [[] for _ in range(len(df))]
            df['is_enzyme'] = False
            df['num_ec'] = 0
            return df
        
        print(f"  ✅ Using EC column: '{ec_col}'")
        
        # Show sample values from EC column (non-NaN)
        sample_values = df[ec_col].dropna().head(10)
        print(f"  📋 Sample EC values: {sample_values.tolist()}")
        
        # Extract EC numbers using corrected function
        df['EC_list'] = df[ec_col].apply(self.extract_ec_numbers_corrected)
        df['is_enzyme'] = df['EC_list'].apply(lambda x: len(x) > 0)
        df['num_ec'] = df['EC_list'].apply(len)
        
        enzyme_count = df['is_enzyme'].sum()
        non_enzyme_count = len(df) - enzyme_count
        
        print(f"  ✅ Classification: {enzyme_count} enzymes, {non_enzyme_count} non-enzymes")
        
        # Show some examples of extracted EC numbers
        enzyme_samples = df[df['is_enzyme']].head(5)
        if len(enzyme_samples) > 0:
            print(f"  📋 Sample extracted EC numbers:")
            for idx, row in enzyme_samples.iterrows():
                print(f"    - {row['Entry']}: {row['EC_list']}")
        
        return df
    
    def load_hdf5_embeddings(self, h5_path):
        """Load embeddings from HDF5 file"""
        print(f"  📄 Loading HDF5: {os.path.basename(h5_path)}")
        
        try:
            with h5py.File(h5_path, 'r') as f:
                keys = list(f.keys())
                print(f"  ✅ Found {len(keys)} embeddings")
                
                if len(keys) == 0:
                    print(f"  ❌ No data in HDF5 file")
                    return None
                
                # Get embedding dimension
                first_key = keys[0]
                if len(f[first_key].shape) == 1:
                    embedding_dim = f[first_key].shape[0]
                else:
                    embedding_dim = f[first_key].shape[1]
                print(f"  📊 Embedding dimension: {embedding_dim}")
                
                # Extract all embeddings
                ids = []
                embeddings = []
                
                for key in keys:
                    ids.append(key)
                    embeddings.append(f[key][:])
                
                ids_array = np.array(ids)
                embeddings_array = np.vstack(embeddings).astype(np.float32)
                
                print(f"  ✅ Final matrix shape: {embeddings_array.shape}")
                
                return {
                    'ids': ids_array,
                    'embeddings': embeddings_array,
                    'count': len(ids_array),
                    'dimension': embedding_dim
                }
                
        except Exception as e:
            print(f"  ❌ Error loading HDF5: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def merge_embeddings_with_labels(self, tsv_df, embedding_data, species_name):
        """Merge embeddings with enzyme labels"""
        print(f"\n  🔗 Merging embeddings with labels for {species_name}...")
        
        # Get protein ID column (usually 'Entry')
        id_col = 'Entry' if 'Entry' in tsv_df.columns else tsv_df.columns[0]
        print(f"  📌 Using ID column: '{id_col}'")
        
        # Create mapping from protein ID to enzyme status
        id_to_enzyme = dict(zip(tsv_df[id_col], tsv_df['is_enzyme']))
        id_to_ec = dict(zip(tsv_df[id_col], tsv_df['EC_list']))
        
        matched_indices = []
        matched_ids = []
        matched_labels = []
        matched_ec = []
        
        for idx, prot_id in enumerate(embedding_data['ids']):
            # Try exact match first
            if prot_id in id_to_enzyme:
                matched_indices.append(idx)
                matched_ids.append(prot_id)
                matched_labels.append(id_to_enzyme[prot_id])
                matched_ec.append(id_to_ec[prot_id])
            else:
                # Try removing version suffix (e.g., P12345.1 -> P12345)
                prot_id_base = prot_id.split('.')[0]
                if prot_id_base in id_to_enzyme:
                    matched_indices.append(idx)
                    matched_ids.append(prot_id_base)
                    matched_labels.append(id_to_enzyme[prot_id_base])
                    matched_ec.append(id_to_ec[prot_id_base])
        
        if not matched_indices:
            print(f"  ❌ No matches found!")
            print(f"  Sample embedding IDs: {embedding_data['ids'][:5]}")
            print(f"  Sample TSV IDs: {list(id_to_enzyme.keys())[:5]}")
            return None
        
        print(f"  ✅ Found {len(matched_indices)} matches out of {len(embedding_data['ids'])} embeddings")
        match_percentage = (len(matched_indices) / len(embedding_data['ids'])) * 100
        print(f"  📊 Match rate: {match_percentage:.2f}%")
        
        # Create merged dataframe
        merged_data = {
            'protein_id': matched_ids,
            'is_enzyme': matched_labels,
            'species': species_name,
            'ec_numbers': matched_ec
        }
        
        # Add embeddings
        for i in range(embedding_data['dimension']):
            merged_data[f'emb_{i}'] = embedding_data['embeddings'][matched_indices, i]
        
        merged_df = pd.DataFrame(merged_data)
        
        print(f"\n  📊 Merged dataset summary:")
        print(f"    - Total samples: {len(merged_df)}")
        print(f"    - Enzymes: {merged_df['is_enzyme'].sum()}")
        print(f"    - Non-enzymes: {(~merged_df['is_enzyme']).sum()}")
        
        # Show some enzyme examples
        enzyme_samples = merged_df[merged_df['is_enzyme']].head(5)
        if len(enzyme_samples) > 0:
            print(f"\n  📋 Sample enzymes:")
            for idx, row in enzyme_samples.iterrows():
                ec_str = '; '.join(row['ec_numbers']) if row['ec_numbers'] else 'N/A'
                print(f"    - {row['protein_id']}: {ec_str}")
        
        return merged_df
    
    def process_species(self, tax_id, species_info):
        """Process a single species"""
        species_name = species_info['name']
        folder_name = species_info['folder']
        folder_path = os.path.join(self.config.DATA_DIR, folder_name)
        
        print(f"\n" + "=" * 80)
        print(f"Processing: {species_name} (taxon_{tax_id})")
        print("=" * 80)
        print(f"📁 Folder: {folder_name}")
        
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            return None
        
        # Find files
        tsv_path, h5_path = self.find_files_in_folder(folder_path)
        
        if not tsv_path:
            print(f"❌ No TSV file found in {folder_path}")
            print(f"   Files in folder: {os.listdir(folder_path)}")
            return None
        
        if not h5_path:
            print(f"❌ No HDF5 file found in {folder_path}")
            print(f"   Files in folder: {os.listdir(folder_path)}")
            return None
        
        # Step 1: Load TSV
        print(f"\n📄 Step 1: Loading TSV file...")
        print("-" * 40)
        tsv_df = self.load_tsv_file(tsv_path)
        if tsv_df is None:
            return None
        
        # Step 2: Classify enzymes
        print(f"\n🧬 Step 2: Classifying enzymes...")
        print("-" * 40)
        tsv_df = self.classify_enzymes(tsv_df)
        
        # Step 3: Load HDF5 embeddings
        print(f"\n💾 Step 3: Loading HDF5 embeddings...")
        print("-" * 40)
        embedding_data = self.load_hdf5_embeddings(h5_path)
        if embedding_data is None:
            return None
        
        # Step 4: Merge embeddings with labels
        print(f"\n🔗 Step 4: Merging embeddings with labels...")
        print("-" * 40)
        merged_df = self.merge_embeddings_with_labels(tsv_df, embedding_data, species_name)
        if merged_df is None:
            return None
        
        return merged_df
    
    def process_all_species(self):
        """Process all species"""
        print("=" * 80)
        print("UNIPROT DATA PROCESSOR - SPECIES-SPECIFIC (CORRECTED EC EXTRACTION)")
        print("=" * 80)
        print(f"Data directory: {self.config.DATA_DIR}")
        print(f"Output directory: {self.config.OUTPUT_DIR}")
        print("=" * 80)
        
        # Check if data directory exists
        if not os.path.exists(self.config.DATA_DIR):
            print(f"❌ Data directory not found: {self.config.DATA_DIR}")
            return None
        
        # List all folders
        all_folders = [f for f in os.listdir(self.config.DATA_DIR) 
                      if os.path.isdir(os.path.join(self.config.DATA_DIR, f))]
        print(f"\n📁 Found folders: {all_folders}")
        
        all_data = []
        species_stats = []
        
        for tax_id, species_info in self.config.SPECIES.items():
            merged_df = self.process_species(tax_id, species_info)
            
            if merged_df is not None and len(merged_df) > 0:
                all_data.append(merged_df)
                
                stats = {
                    'species': species_info['name'],
                    'taxonomy_id': tax_id,
                    'total_samples': len(merged_df),
                    'enzymes': int(merged_df['is_enzyme'].sum()),
                    'non_enzymes': int((~merged_df['is_enzyme']).sum()),
                    'enzyme_percentage': float(merged_df['is_enzyme'].mean() * 100) if len(merged_df) > 0 else 0
                }
                species_stats.append(stats)
                
                # Save individual species data
                self.save_species_data(merged_df, species_info['name'])
            else:
                print(f"⚠️ No data processed for {species_info['name']}")
        
        # Combine all species
        if all_data:
            print("\n" + "=" * 80)
            print("COMBINING ALL SPECIES")
            print("=" * 80)
            
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Save combined data
            combined_path = os.path.join(self.config.OUTPUT_DIR, 'combined_all_species.csv')
            combined_df.to_csv(combined_path, index=False)
            print(f"✅ Combined data saved to: {combined_path}")
            print(f"Total samples: {len(combined_df)}")
            
            # Save statistics
            self.save_statistics(combined_df, species_stats)
            
            # Generate report
            self.generate_report(combined_df, species_stats)
            
            return combined_df
        else:
            print("\n❌ No data was successfully processed!")
            return None
    
    def save_species_data(self, merged_df, species_name):
        """Save individual species data"""
        species_dir = os.path.join(self.config.OUTPUT_DIR, species_name)
        os.makedirs(species_dir, exist_ok=True)
        
        # Save merged data
        merged_path = os.path.join(species_dir, f'{species_name}_processed.csv')
        merged_df.to_csv(merged_path, index=False)
        print(f"  ✅ Saved to: {merged_path}")
        
        # Save positive and negative sets
        enzymes_df = merged_df[merged_df['is_enzyme']]
        if len(enzymes_df) > 0:
            enzymes_path = os.path.join(species_dir, f'{species_name}_enzymes.csv')
            enzymes_df.to_csv(enzymes_path, index=False)
            print(f"  ✅ Enzymes saved to: {enzymes_path}")
        
        non_enzymes_df = merged_df[~merged_df['is_enzyme']]
        if len(non_enzymes_df) > 0:
            non_enzymes_path = os.path.join(species_dir, f'{species_name}_non_enzymes.csv')
            non_enzymes_df.to_csv(non_enzymes_path, index=False)
            print(f"  ✅ Non-enzymes saved to: {non_enzymes_path}")
    
    def save_statistics(self, combined_df, species_stats):
        """Save statistics"""
        total_enzymes = combined_df['is_enzyme'].sum()
        total_samples = len(combined_df)
        
        stats = {
            'total_samples': total_samples,
            'total_enzymes': int(total_enzymes),
            'total_non_enzymes': int(total_samples - total_enzymes),
            'enzyme_percentage': float(total_enzymes / total_samples * 100) if total_samples > 0 else 0,
            'species': species_stats
        }
        
        stats_path = os.path.join(self.config.OUTPUT_DIR, 'statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✅ Statistics saved to: {stats_path}")
        
        stats_df = pd.DataFrame(species_stats)
        stats_csv_path = os.path.join(self.config.OUTPUT_DIR, 'species_statistics.csv')
        stats_df.to_csv(stats_csv_path, index=False)
        print(f"✅ Species statistics saved to: {stats_csv_path}")
    
    def generate_report(self, combined_df, species_stats):
        """Generate comprehensive report"""
        report_path = os.path.join(self.config.OUTPUT_DIR, 'processing_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("UNIPROT DATA PROCESSING REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data directory: {self.config.DATA_DIR}\n")
            f.write(f"Output directory: {self.config.OUTPUT_DIR}\n\n")
            
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total samples: {len(combined_df)}\n")
            f.write(f"Total enzymes: {combined_df['is_enzyme'].sum()}\n")
            f.write(f"Total non-enzymes: {(~combined_df['is_enzyme']).sum()}\n")
            f.write(f"Enzyme percentage: {combined_df['is_enzyme'].mean() * 100:.2f}%\n\n")
            
            f.write("SPECIES BREAKDOWN\n")
            f.write("-" * 40 + "\n")
            species_counts = combined_df['species'].value_counts()
            for species, count in species_counts.items():
                f.write(f"{species}: {count} samples\n")
            
            f.write("\n" + "-" * 40 + "\n")
            
            f.write("DETAILED SPECIES STATISTICS\n")
            f.write("-" * 40 + "\n")
            for stats in species_stats:
                f.write(f"\n{stats['species']} (taxon_{stats['taxonomy_id']}):\n")
                f.write(f"  Total samples: {stats['total_samples']}\n")
                f.write(f"  Enzymes: {stats['enzymes']}\n")
                f.write(f"  Non-enzymes: {stats['non_enzymes']}\n")
                f.write(f"  Enzyme percentage: {stats['enzyme_percentage']:.2f}%\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"✅ Report saved to: {report_path}")

# =============================================
# 4. Main Execution
# =============================================

def main():
    print("=" * 80)
    print("UNIPROT DATA PROCESSOR - STARTING (CORRECTED EC EXTRACTION)")
    print("=" * 80)
    
    processor = UniProtDataProcessor(Config)
    combined_data = processor.process_all_species()
    
    if combined_data is not None:
        print("\n" + "=" * 80)
        print("✅ PROCESSING COMPLETE")
        print("=" * 80)
        print(f"All outputs saved to: {Config.OUTPUT_DIR}")
        print("\nKey files:")
        print(f"  - Combined data: {os.path.join(Config.OUTPUT_DIR, 'combined_all_species.csv')}")
        print(f"  - Statistics: {os.path.join(Config.OUTPUT_DIR, 'statistics.json')}")
        print(f"  - Processing report: {os.path.join(Config.OUTPUT_DIR, 'processing_report.txt')}")
        print("\nSpecies-specific files:")
        for species in Config.SPECIES.values():
            species_dir = os.path.join(Config.OUTPUT_DIR, species['name'])
            if os.path.exists(species_dir):
                print(f"  - {species['name']}/")
                print(f"      ├── {species['name']}_processed.csv")
                print(f"      ├── {species['name']}_enzymes.csv")
                print(f"      └── {species['name']}_non_enzymes.csv")
        print("=" * 80)
        
        # Print summary of enzyme counts
        print("\n📊 ENZYME SUMMARY:")
        print("-" * 40)
        for species in Config.SPECIES.values():
            species_dir = os.path.join(Config.OUTPUT_DIR, species['name'])
            if os.path.exists(species_dir):
                enzymes_path = os.path.join(species_dir, f'{species['name']}_enzymes.csv')
                if os.path.exists(enzymes_path):
                    enzymes_df = pd.read_csv(enzymes_path)
                    print(f"  {species['name']}: {len(enzymes_df)} enzymes")
                else:
                    print(f"  {species['name']}: 0 enzymes")
    else:
        print("\n❌ Processing failed - no data was generated")

if __name__ == "__main__":
    main()