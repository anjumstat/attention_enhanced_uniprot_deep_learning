# UniProt Data Processing Pipeline for Enzyme Classification

## Overview
This pipeline processes manually downloaded UniProt protein embeddings and annotation files for enzyme classification. The pipeline handles multiple plant species, extracts embeddings, classifies enzymes based on EC numbers, and combines all data into a unified dataset ready for deep learning.

## Data Source
Data was manually downloaded from [UniProt](https://www.uniprot.org/) for four plant species used in enzyme classification research.

### Species and Data Files

The following species were downloaded with their corresponding files:

| Species | Proteome ID | Embeddings File | Annotations File |
|---------|-------------|-----------------|------------------|
| Arabidopsis thaliana | UP000006548 | embeddings.h5 | annotations.xlsx |
| Brassica species | Various | embeddings.h5 | annotations.csv |
| Oryza sativa (Rice) | UP000059680 | embeddings.h5 | annotations.xlsx |
| Triticum aestivum (Wheat) | UP000019116 | embeddings.h5 | annotations.csv |

### Downloaded File Types

**1. Embeddings Files (HDF5 format)**
- Contains protein sequence embeddings from UniProt's protein language model
- Each protein has a 1024-dimensional embedding vector
- Format: HDF5 with protein IDs as keys and embedding arrays as values

**2. Annotation Files (CSV/Excel format)**
- Contains protein metadata including:
  - Protein IDs (matching embedding files)
  - EC numbers (Enzyme Commission numbers)
  - Protein names and descriptions
  - Functional annotations

## File Organization

After manual download, organize your files in this structure:
D:/uni_prot/data/
├── Arabidopsis_thaliana/
│ ├── embeddings.h5 # HDF5 embeddings file
│ └── annotations.xlsx # EC numbers and annotations
├── Brassica_species/
│ ├── embeddings.h5
│ └── annotations.csv
├── Oryza_sativa/
│ ├── embeddings.h5
│ └── annotations.xlsx
└── Triticum_aestivum/
├── embeddings.h5
└── annotations.csv
