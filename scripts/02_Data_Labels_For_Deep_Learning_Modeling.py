# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 15:57:57 2026

@author: H.A.R
"""

import pandas as pd
import os

# File paths
input_file = r"D:\uni_prot2\processed_results\combined_data\all_embeddings_combined.csv"
output_file = r"D:\uni_prot2\processed_results\combined_data\all_embeddings_combined_processed.csv"

# Load CSV
df = pd.read_csv(input_file)

# Drop first column (index 0, UniProt_ID) and last column (Data_Source)
df = df.drop(columns=[df.columns[0], df.columns[-1]])

# Rename last column (was Enzyme_Classification) to 'category'
df.rename(columns={df.columns[-1]: 'category'}, inplace=True)

# Map 'Enzyme' -> 1, 'Non-enzyme' -> 0
df['category'] = df['category'].map({'Enzyme': 1, 'Non-enzyme': 0})

# Save to CSV
df.to_csv(output_file, index=False)

print(f"✅ Processed file saved at: {output_file}")