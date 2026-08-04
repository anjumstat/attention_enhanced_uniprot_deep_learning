# -*- coding: utf-8 -*-
"""
02_create_evaluation_splits.py
CREATE EVALUATION SPLITS WITH HOMOLOGY-AWARE CLUSTERING
"""

import pandas as pd
import numpy as np
import os
import json
import re
import subprocess
import tempfile
from sklearn.model_selection import train_test_split
from datetime import datetime
import shutil

# =============================================
# CONFIGURATION
# =============================================

class Config:
    DATA_DIR = r"D:\uni_prot2\revision\data"
    OUTPUT_DIR = r"D:\uni_prot2\revision\data\evaluation_splits"
    
    SPECIES_FOLDERS = {
        'Arabidopsis_thaliana': 'thailian',
        'Brassica_spp': 'brassica',
        'Oryza_sativa': 'rice',
        'Triticum_aestivum': 'wheat'
    }
    
    CDHIT_IDENTITY = 0.6  # 60% identity (use -n 4)
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1

# =============================================
# DATA LOADING
# =============================================

def load_processed_data(config):
    print("=" * 80)
    print("LOADING PROCESSED DATA")
    print("=" * 80)
    
    combined_path = os.path.join(config.DATA_DIR, 'processed_results_v3', 'combined_all_species.csv')
    
    if not os.path.exists(combined_path):
        print(f"❌ Combined data not found: {combined_path}")
        return None
    
    df = pd.read_csv(combined_path)
    print(f"✅ Loaded {len(df)} samples")
    print(f"   Species: {df['species'].unique()}")
    print(f"   Enzymes: {df['is_enzyme'].sum()}")
    print(f"   Non-enzymes: {(~df['is_enzyme']).sum()}")
    
    return df

# =============================================
# LOAD FASTA SEQUENCES
# =============================================

def extract_uniprot_id(header):
    if '.' in header:
        header = header.split('.')[0]
    if '|' in header:
        parts = header.split('|')
        if len(parts) >= 2:
            return parts[1]
    return header.split()[0]

def load_sequences_from_folders(config):
    print("\n" + "=" * 80)
    print("LOADING SEQUENCES FROM SPECIES FOLDERS")
    print("=" * 80)
    
    all_sequences = {}
    
    for species, folder in config.SPECIES_FOLDERS.items():
        folder_path = os.path.join(config.DATA_DIR, folder)
        
        if not os.path.exists(folder_path):
            print(f"⚠️ Folder not found: {folder_path}")
            continue
        
        print(f"\n📁 Searching in: {folder}/")
        
        fasta_files = [f for f in os.listdir(folder_path) if f.endswith('.fasta') or f.endswith('.fa')]
        
        if not fasta_files:
            print(f"   ⚠️ No FASTA file found")
            continue
        
        fasta_path = os.path.join(folder_path, fasta_files[0])
        print(f"   📄 Loading: {fasta_files[0]}")
        
        count = 0
        current_id = None
        sequence_lines = []
        
        with open(fasta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id and sequence_lines:
                        all_sequences[current_id] = ''.join(sequence_lines)
                        count += 1
                    header = line[1:]
                    current_id = extract_uniprot_id(header)
                    sequence_lines = []
                elif current_id:
                    sequence_lines.append(line)
            
            if current_id and sequence_lines:
                all_sequences[current_id] = ''.join(sequence_lines)
                count += 1
        
        print(f"   ✅ Loaded {count} sequences")
    
    print(f"\n✅ Total sequences loaded: {len(all_sequences)}")
    return all_sequences

# =============================================
# CREATE COMBINED FASTA
# =============================================

def create_combined_fasta(df, sequences, output_path):
    print("\n" + "=" * 80)
    print("CREATING COMBINED FASTA FOR CD-HIT")
    print("=" * 80)
    
    df = df.copy()
    
    def find_sequence(protein_id):
        if protein_id in sequences:
            return sequences[protein_id]
        if '.' in protein_id:
            base_id = protein_id.split('.')[0]
            if base_id in sequences:
                return sequences[base_id]
        return None
    
    df['sequence'] = df['protein_id'].apply(find_sequence)
    
    found = df['sequence'].notna().sum()
    total = len(df)
    print(f"Sequences found for {found}/{total} proteins ({found/total*100:.1f}%)")
    
    if found == 0:
        print("\n❌ No sequences matched!")
        return None
    
    print("\nSequence coverage by species:")
    for species in df['species'].unique():
        species_df = df[df['species'] == species]
        species_found = species_df['sequence'].notna().sum()
        species_total = len(species_df)
        print(f"  {species}: {species_found}/{species_total} ({species_found/species_total*100:.1f}%)")
    
    with open(output_path, 'w') as f:
        for idx, row in df.iterrows():
            if pd.notna(row['sequence']):
                f.write(f">{row['protein_id']}\n")
                f.write(f"{row['sequence']}\n")
    
    print(f"\n✅ Created combined FASTA: {output_path}")
    print(f"   Contains {found} sequences")
    
    return output_path

# =============================================
# RUN CD-HIT CLUSTERING (using WSL) - FIXED
# =============================================

def run_cdhit(fasta_path, identity=0.6):
    print("\n" + "=" * 80)
    print("RUNNING CD-HIT CLUSTERING (via WSL)")
    print("=" * 80)
    print(f"Identity threshold: {identity*100}%")
    
    # Convert Windows path to WSL path
    def win_to_wsl_path(win_path):
        win_path = win_path.replace('\\', '/')
        import re
        match = re.match(r'([A-Za-z]):/(.*)', win_path)
        if match:
            drive = match.group(1).lower()
            path = match.group(2)
            return f"/mnt/{drive}/{path}"
        return win_path
    
    fasta_wsl = win_to_wsl_path(fasta_path)
    
    temp_wsl = "/tmp/cdhit_temp"
    output_prefix_wsl = f"{temp_wsl}/cdhit_output"
    cluster_file_wsl = f"{output_prefix_wsl}.clstr"
    
    # Use -n 4 for 60% identity threshold (CD-HIT requirement)
    # For other thresholds: -n 5 for >=75%, -n 4 for 60-75%, -n 3 for 50-60%, -n 2 for <50%
    cmd = f"""
    mkdir -p {temp_wsl} && \
    cd-hit -i {fasta_wsl} -o {output_prefix_wsl} -c {identity} -n 4 -M 2000 -T 4 -d 0 && \
    cat {cluster_file_wsl}
    """
    
    print(f"Running CD-HIT in WSL with word length 4...")
    
    try:
        result = subprocess.run(
            ["wsl", "-e", "bash", "-c", cmd],
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ CD-HIT completed successfully")
        
        # Show last 500 characters of output
        if result.stdout:
            print(result.stdout[-500:])
        
        temp_dir = tempfile.mkdtemp()
        cluster_file_win = os.path.join(temp_dir, "cdhit_output.clstr")
        
        copy_cmd = f"cp {cluster_file_wsl} {win_to_wsl_path(cluster_file_win)}"
        subprocess.run(["wsl", "-e", "bash", "-c", copy_cmd], capture_output=True, text=True)
        
        clusters = parse_cdhit_clusters(cluster_file_win)
        
        print(f"✅ Created {len(clusters)} clusters")
        
        # Clean up
        clean_cmd = f"rm -rf {temp_wsl}"
        subprocess.run(["wsl", "-e", "bash", "-c", clean_cmd], capture_output=True)
        shutil.rmtree(temp_dir)
        
        return clusters
        
    except subprocess.CalledProcessError as e:
        print(f"❌ CD-HIT failed: {e.stderr}")
        return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def parse_cdhit_clusters(cluster_file):
    clusters = {}
    current_cluster = None
    
    with open(cluster_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>Cluster'):
                current_cluster = int(line.split()[1])
                clusters[current_cluster] = []
            elif line and current_cluster is not None:
                match = re.search(r'>(\S+)\.\.\.', line)
                if match:
                    prot_id = match.group(1)
                    if '.' in prot_id:
                        prot_id = prot_id.split('.')[0]
                    clusters[current_cluster].append(prot_id)
    
    return clusters

# =============================================
# CREATE HOMOLOGY-AWARE SPLITS
# =============================================

def create_homology_aware_splits(df, clusters, test_size=0.2, val_size=0.1, random_seed=42):
    print("\n" + "=" * 80)
    print("CREATING HOMOLOGY-AWARE SPLITS")
    print("=" * 80)
    
    protein_to_cluster = {}
    for cluster_id, proteins in clusters.items():
        for protein in proteins:
            protein_to_cluster[protein] = cluster_id
    
    df['cluster_id'] = df['protein_id'].map(protein_to_cluster)
    
    unassigned = df['cluster_id'].isna().sum()
    if unassigned > 0:
        print(f"⚠️ {unassigned} proteins not assigned to clusters")
        for idx in df[df['cluster_id'].isna()].index:
            df.loc[idx, 'cluster_id'] = f"singleton_{idx}"
    
    unique_clusters = df['cluster_id'].unique()
    print(f"Total clusters: {len(unique_clusters)}")
    
    np.random.seed(random_seed)
    shuffled_clusters = np.random.permutation(unique_clusters)
    
    n_clusters = len(shuffled_clusters)
    n_test = int(n_clusters * test_size)
    n_val = int(n_clusters * val_size)
    n_train = n_clusters - n_test - n_val
    
    train_clusters = shuffled_clusters[:n_train]
    val_clusters = shuffled_clusters[n_train:n_train+n_val]
    test_clusters = shuffled_clusters[n_train+n_val:]
    
    print(f"\nCluster split:")
    print(f"  Train: {len(train_clusters)} clusters")
    print(f"  Val:   {len(val_clusters)} clusters")
    print(f"  Test:  {len(test_clusters)} clusters")
    
    train_df = df[df['cluster_id'].isin(train_clusters)]
    val_df = df[df['cluster_id'].isin(val_clusters)]
    test_df = df[df['cluster_id'].isin(test_clusters)]
    
    print(f"\nProtein split:")
    print(f"  Train: {len(train_df)} proteins ({train_df['is_enzyme'].sum()} enzymes)")
    print(f"  Val:   {len(val_df)} proteins ({val_df['is_enzyme'].sum()} enzymes)")
    print(f"  Test:  {len(test_df)} proteins ({test_df['is_enzyme'].sum()} enzymes)")
    
    train_df = train_df.drop('cluster_id', axis=1)
    val_df = val_df.drop('cluster_id', axis=1)
    test_df = test_df.drop('cluster_id', axis=1)
    
    return train_df, val_df, test_df

# =============================================
# CREATE LEAVE-ONE-SPECIES-OUT SPLITS
# =============================================

def create_loso_splits(df):
    print("\n" + "=" * 80)
    print("CREATING LEAVE-ONE-SPECIES-OUT SPLITS")
    print("=" * 80)
    
    species_list = df['species'].unique()
    loso_splits = {}
    
    for test_species in species_list:
        train_df = df[df['species'] != test_species]
        test_df = df[df['species'] == test_species]
        
        loso_splits[test_species] = {
            'train': train_df,
            'test': test_df,
            'train_species': train_df['species'].unique().tolist(),
            'test_species': test_species,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'train_enzymes': int(train_df['is_enzyme'].sum()),
            'test_enzymes': int(test_df['is_enzyme'].sum())
        }
        
        print(f"\n{test_species}:")
        print(f"  Train: {len(train_df)} proteins (train on: {', '.join(train_df['species'].unique())})")
        print(f"  Test:  {len(test_df)} proteins")
    
    return loso_splits

# =============================================
# SAVE SPLITS
# =============================================

def save_splits(train_df, val_df, test_df, loso_splits, config, homology_success):
    print("\n" + "=" * 80)
    print("SAVING SPLITS")
    print("=" * 80)
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    split_info = {
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_source': config.DATA_DIR,
        'random_seed': config.RANDOM_SEED,
        'cdhit_identity': config.CDHIT_IDENTITY,
        'homology_aware_created': homology_success,
        'total_samples': len(train_df) + len(val_df) + len(test_df),
        'splits': {}
    }
    
    if homology_success:
        homology_dir = os.path.join(config.OUTPUT_DIR, 'homology_aware')
        os.makedirs(homology_dir, exist_ok=True)
        
        train_df.to_csv(os.path.join(homology_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(homology_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(homology_dir, 'test.csv'), index=False)
        
        split_info['splits']['homology_aware'] = {
            'train': len(train_df),
            'val': len(val_df),
            'test': len(test_df),
            'train_enzymes': int(train_df['is_enzyme'].sum()),
            'val_enzymes': int(val_df['is_enzyme'].sum()),
            'test_enzymes': int(test_df['is_enzyme'].sum())
        }
        
        print(f"✅ Homology-aware splits saved to: {homology_dir}")
    else:
        print("⚠️ Homology-aware splits NOT created")
    
    loso_dir = os.path.join(config.OUTPUT_DIR, 'loso')
    os.makedirs(loso_dir, exist_ok=True)
    
    split_info['splits']['loso'] = {}
    
    for species, split in loso_splits.items():
        species_dir = os.path.join(loso_dir, species.replace(' ', '_'))
        os.makedirs(species_dir, exist_ok=True)
        
        split['train'].to_csv(os.path.join(species_dir, 'train.csv'), index=False)
        split['test'].to_csv(os.path.join(species_dir, 'test.csv'), index=False)
        
        info = {
            'test_species': species,
            'train_species': split['train_species'],
            'train_size': split['train_size'],
            'test_size': split['test_size'],
            'train_enzymes': split['train_enzymes'],
            'test_enzymes': split['test_enzymes']
        }
        with open(os.path.join(species_dir, 'split_info.json'), 'w') as f:
            json.dump(info, f, indent=2)
        
        split_info['splits']['loso'][species] = {
            'train_size': split['train_size'],
            'test_size': split['test_size'],
            'train_enzymes': split['train_enzymes'],
            'test_enzymes': split['test_enzymes']
        }
    
    print(f"✅ LOSO splits saved to: {loso_dir}")
    
    with open(os.path.join(config.OUTPUT_DIR, 'splits_summary.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"✅ Summary saved to: {os.path.join(config.OUTPUT_DIR, 'splits_summary.json')}")

# =============================================
# MAIN EXECUTION
# =============================================

def main():
    print("=" * 80)
    print("CREATE EVALUATION SPLITS")
    print("Plant Enzyme Classification")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    df = load_processed_data(config)
    if df is None:
        return
    
    sequences = load_sequences_from_folders(config)
    
    homology_success = False
    train_df = val_df = test_df = None
    
    fasta_path = os.path.join(config.OUTPUT_DIR, "sequences.fasta")
    fasta_created = create_combined_fasta(df, sequences, fasta_path)
    
    if fasta_created is not None:
        clusters = run_cdhit(fasta_path, config.CDHIT_IDENTITY)
        
        if clusters is not None:
            train_df, val_df, test_df = create_homology_aware_splits(
                df, clusters,
                test_size=config.TEST_SIZE,
                val_size=config.VAL_SIZE,
                random_seed=config.RANDOM_SEED
            )
            homology_success = True
        else:
            print("\n❌ CD-HIT failed. Creating random splits as fallback.")
            train_df, temp_df = train_test_split(
                df, 
                test_size=(config.TEST_SIZE + config.VAL_SIZE),
                random_state=config.RANDOM_SEED,
                stratify=df['is_enzyme']
            )
            val_df, test_df = train_test_split(
                temp_df,
                test_size=config.TEST_SIZE / (config.TEST_SIZE + config.VAL_SIZE),
                random_state=config.RANDOM_SEED,
                stratify=temp_df['is_enzyme']
            )
    else:
        print("\n❌ Could not create FASTA. Creating random splits as fallback.")
        train_df, temp_df = train_test_split(
            df, 
            test_size=(config.TEST_SIZE + config.VAL_SIZE),
            random_state=config.RANDOM_SEED,
            stratify=df['is_enzyme']
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=config.TEST_SIZE / (config.TEST_SIZE + config.VAL_SIZE),
            random_state=config.RANDOM_SEED,
            stratify=temp_df['is_enzyme']
        )
    
    loso_splits = create_loso_splits(df)
    save_splits(train_df, val_df, test_df, loso_splits, config, homology_success)
    
    print("\n" + "=" * 80)
    print("✅ SPLITS CREATION COMPLETE")
    print("=" * 80)
    print(f"All splits saved to: {config.OUTPUT_DIR}")
    print(f"Homology-aware splits: {'✅ Created' if homology_success else '⚠️ Fallback'}")
    print("=" * 80)

if __name__ == "__main__":
    main()