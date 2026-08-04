# Plant Enzyme Classification using Attention-Enhanced Deep Neural Networks with UniProt Protein Embeddings

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Overview

This repository contains the complete code and data processing pipeline for the paper:

**"Transfer learning with attention-enhanced deep neural networks for plant enzyme classification using UniProt protein embeddings"**

The framework leverages pre-computed UniProt protein embeddings with an attention-enhanced deep neural network to accurately classify proteins as enzymes or non-enzymes across four major plant species: *Arabidopsis thaliana*, *Brassica* species, *Oryza sativa* (rice), and *Triticum aestivum* (wheat).

### Key Features

- 🔬 **Species-specific data curation** with UniProt taxonomy IDs (3702, 3705, 4530, 4565)
- 🧬 **Homology-aware evaluation** using CD-HIT clustering (60% identity threshold)
- 🌿 **Cross-species generalization** via Leave-One-Species-Out (LOSO) evaluation
- 🤖 **10 models** including Attention-Enhanced DNN, DNN, Logistic Regression, Random Forest, SVM, MLP variants, and ablation studies
- 📊 **Comprehensive evaluation** with Accuracy, AUC, F1, MCC, Precision, Recall
- 📈 **Statistical analysis** with Friedman test and Repeated Measures ANOVA
- 🔄 **Checkpoint/Resume** functionality for long-running experiments

---

## 📁 Repository Structure
├── codes/
│ ├── 01_data_preprocessing.py # Clean and curate data with taxonomy filtering
│ ├── 02_create_evaluation_splits.py # CD-HIT clustering and homology-aware splits
│ ├── 03_create_clean_splits.py # Create clean CSV files (embeddings + target only)
│ ├── 04_train_models.py # Attention-enhanced deep learning experiments
│ ├── 05_add_species_column.py # Add species column for LOSO evaluation
│ ├── 06_plant_loso_complete.py # Complete LOSO evaluation (all models, all species)
│ ├── 07_baseline_comparisons_cd_hit_dat_checkpoints.py # Baseline models with checkpoint
│ ├── 08_extract_f1_mcc_update_tables.py # Extract F1 and MCC from results
│ ├── 09_combine_csv_cd_hit.py # Combine LOSO results into summary tables
        10_combine_loso_results
│ ├── 11_statistical_tests.py # Friedman test and Repeated Measures ANOVA
│ └── 12_Generate_Figures.py # Generate all paper figures (Figure 1-6)
│
├── data/
│ ├── raw/ # Raw UniProt data (TSV + HDF5 + FASTA)
│ ├── processed_results_v3/ # Cleaned and processed data
│ ├── evaluation_splits/ # CD-HIT and LOSO splits
│ └── clean_splits/ # Clean embeddings-only data
│
├── results/
│ ├── homology_aware_with_test/ # Attention models results
│ ├── homology_aware_baselines_with_checkpoint/ # Baseline models results
│ ├── plant_loso_complete/ # Complete LOSO results
│ ├── combined_results_CD_HIT/ # Combined results with F1/MCC
│ ├── combined_loso_results/ # Combined LOSO results
│ └── Figures_for_Paper6/ # All paper figures (Figure 1-6)
│
├── requirements.txt # Python dependencies
└── README.md # This file

---

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.8 or higher
- **TensorFlow**: 2.12.0 or higher
- **WSL** (Windows Subsystem for Linux) if running CD-HIT on Windows
- **CD-HIT** installed for homology clustering

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/plant-enzyme-classification.git
cd plant-enzyme-classification
2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Install CD-HIT (for homology-aware clustering)
# On Ubuntu/Debian
sudo apt-get install cd-hit

# On Windows (via WSL)
wsl --install
# Then in WSL: sudo apt-get install cd-hit

# On macOS
brew install cd-hit
# Requirements
numpy>=1.24.0
pandas>=1.5.0
scikit-learn>=1.2.0
tensorflow>=2.12.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.10.0
statsmodels>=0.14.0
h5py>=3.8.0
biopython>=1.81
# Data Preparation
1. Download Data from UniProt
Download the following files for each species:

Species	Taxonomy ID	Files Needed
Arabidopsis thaliana	3702	TSV, HDF5 (embeddings), FASTA
Brassica spp.	3705	TSV, HDF5 (embeddings), FASTA
Oryza sativa	4530	TSV, HDF5 (embeddings), FASTA
Triticum aestivum	4565	TSV, HDF5 (embeddings), FASTA
Place files in the following folder structure:

data/
├── thailian/          # Arabidopsis thaliana
│   ├── uniprotkb_taxonomy_id_3702_AND_reviewed_*.tsv
│   ├── uniprotkb_taxonomy_id_3702_AND_reviewed_*.fasta
│   └── embeddings.h5   # HDF5 file with embeddings
├── brassica/          # Brassica species
│   ├── uniprotkb_taxonomy_id_3705_AND_reviewed_*.tsv
│   ├── uniprotkb_taxonomy_id_3705_AND_reviewed_*.fasta
│   └── embeddings.h5
├── rice/              # Oryza sativa
│   ├── uniprotkb_taxonomy_id_4530_AND_reviewed_*.tsv
│   ├── uniprotkb_taxonomy_id_4530_AND_reviewed_*.fasta
│   └── embeddings.h5
└── wheat/             # Triticum aestivum
    ├── uniprotkb_taxonomy_id_4565_AND_reviewed_*.tsv
    ├── uniprotkb_taxonomy_id_4565_AND_reviewed_*.fasta
    └── embeddings.h5
2. Run Data Preprocessing
# Step 1: Clean and curate data
python codes/01_data_preprocessing.py

# Step 2: Create homology-aware and LOSO splits
python codes/02_create_evaluation_splits.py

# Step 3: Create clean embeddings-only splits
python codes/03_create_clean_splits.py

# Step 4: Add species column for LOSO
python codes/05_add_species_column.py
Model Training
Attention-Enhanced Models (6 models)
# Train all attention-enhanced models with 10-fold CV
python codes/04_train_models.py
Models Included:

Attention_Enhanced_Basic - Proposed model with attention + residuals

DNN_Baseline - Standard DNN without attention

Logistic_Baseline - Simple linear classifier

Ablation_No_Attention - Attention removed

Ablation_No_Residual - Residual connections removed

Ablation_50Percent_Data - Only 50% of training data
Baseline Models (4 models)
# Train all baseline models with checkpoint/resume
python codes/07_baseline_comparisons_cd_hit_dat_checkpoints.py
Models Included:
Random_Forest - Random Forest classifier

SVM - Support Vector Machine with RBF kernel

MLP_256 - Multi-layer perceptron with 256 units

MLP_512 - Multi-layer perceptron with 512 units
LOSO Evaluation (All species, all models)
# Run complete LOSO evaluation with checkpoint/resume
python codes/06_plant_loso_complete.py
This will run:

10 models × 4 species × 3 learning rates × 4 batch sizes = 480 experiments

10-fold cross-validation per experiment = 4,800 model trainings

Checkpoint system allows resuming if interrupted
Results Extraction
# Extract F1 and MCC from all results
python codes/08_extract_f1_mcc_update_tables.py

# Combine CD-HIT results into summary tables
python codes/09_combine_loso_results.py
Statistical Analysis
# Run Friedman test and Repeated Measures ANOVA
python codes/10_statistical_tests.py
Figure Generation
# Generate all paper figures (Figure 1-6)
python codes/11_generate_figures_final.py
 Quick Start Summary
# 1. Prepare data
python codes/01_data_preprocessing.py
python codes/02_create_evaluation_splits.py
python codes/03_create_clean_splits.py

# 2. Train models
python codes/04_train_models.py
python codes/07_baseline_comparisons_cd_hit_dat_checkpoints.py
python codes/06_plant_loso_complete.py

# 3. Extract results
python codes/08_extract_f1_mcc_update_tables.py
python codes/09_combine_loso_results.py

# 4. Statistical analysis
python codes/10_statistical_tests.py

# 5. Generate figures
python codes/11_generate_figures_final.py
