markdown
# Plant Enzyme Classification Using Attention-Enhanced Deep Neural Networks with UniProt Protein Embeddings

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21871663.svg)](https://doi.org/10.5281/zenodo.21871663)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.12+](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python TensorFlow License

## 📋 Overview
This repository contains the complete source code, data processing pipeline, and experimental framework for the paper:

**Transfer Learning with Attention-Enhanced Deep Neural Networks for Plant Enzyme Classification Using UniProt Protein Embeddings**

The proposed framework leverages pre-computed UniProt protein embeddings together with an attention-enhanced deep neural network to accurately classify plant proteins into enzyme and non-enzyme classes across four major plant species:

- *Arabidopsis thaliana*
- *Brassica* spp.
- *Oryza sativa* (Rice)
- *Triticum aestivum* (Wheat)

The repository includes data preprocessing, homology-aware dataset construction, model training, cross-species evaluation, statistical analysis, and automatic generation of publication-quality figures.

## ✨ Key Features
- 🔬 **Species-specific data curation** using UniProt taxonomy IDs (3702, 3705, 4530, 4565)
- 🧬 **Homology-aware evaluation** using CD-HIT clustering (60% sequence identity)
- 🌿 **Leave-One-Species-Out (LOSO)** evaluation for cross-species generalization
- 🤖 **Ten machine learning and deep learning models**, including the proposed Attention-Enhanced DNN
- 📊 **Performance evaluation** using Accuracy, Precision, Recall, F1-score, MCC, and ROC-AUC
- 📈 **Statistical comparison** using Friedman Test and Repeated-Measures ANOVA
- 🔄 **Checkpoint/Resume functionality** for long-running experiments
- 📑 **Automatic generation** of publication-ready figures and summary tables
- 🎯 **Confusion matrix generation** for both article and supplementary materials

## 📁 Repository Structure
.
├── codes/
│ ├── 01_data_preprocessing.py
│ ├── 02_create_evaluation_splits.py
│ ├── 03_create_clean_splits.py
│ ├── 04_to_create_combine_file.py
│ ├── 05_add_species_column.py
│ ├── 06_attention_dnn_training.py
│ ├── 07_baseline_comparisons_cd_hit.py
│ ├── 08_loso_training.py
│ ├── 09_combined_csv_cd_hit.py
│ ├── 10_combine_loso_results.py
│ ├── 11_Statistical_Analysis_Final_code.py
│ ├── 12_Generate_Figures.py # Main manuscript figures
│ └── 13_confusion_matrix.py # Confusion matrices (article + supplementary)
│
├── data/
│ ├── arabidopsis/
│ ├── brassica/
│ ├── rice/
│ ├── wheat/
│ ├── evaluation_splits/
│ ├── clean_splits/
│ └── processed_results/
│
├── results/
├── figures/
│ └── confusion_matrices1/ # Generated confusion matrices
│ ├── article/
│ └── supplementary/
│
├── requirements.txt
├── LICENSE
└── README.md

text

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or above
- TensorFlow 2.12 or above
- CD-HIT
- WSL (Windows users only)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/plant-enzyme-classification.git
cd plant-enzyme-classification
Create a virtual environment

bash
python -m venv venv
Activate the environment

bash
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
Install the required packages

bash
pip install -r requirements.txt
Install CD-HIT

bash
# Ubuntu/Debian
sudo apt install cd-hit

# macOS
brew install cd-hit

# Windows
wsl --install
# Then inside WSL:
sudo apt install cd-hit
📦 Python Requirements
text
numpy>=1.24
pandas>=1.5
scikit-learn>=1.2
tensorflow>=2.12
matplotlib>=3.6
seaborn>=0.12
scipy>=1.10
statsmodels>=0.14
h5py>=3.8
biopython>=1.81
📂 Data Preparation
Download reviewed protein datasets from UniProt for the following species:

Species	Taxonomy ID
Arabidopsis thaliana	3702
Brassica spp.	3705
Oryza sativa	4530
Triticum aestivum	4565
For each species, download:

TSV annotation file

FASTA sequence file

HDF5 protein embedding file (1024 dimensions)

Arrange the files as follows:

text
data/
├── arabidopsis/
│   ├── *.tsv
│   ├── *.fasta
│   └── embeddings.h5
├── brassica/
│   ├── *.tsv
│   ├── *.fasta
│   └── embeddings.h5
├── rice/
│   ├── *.tsv
│   ├── *.fasta
│   └── embeddings.h5
└── wheat/
    ├── *.tsv
    ├── *.fasta
    └── embeddings.h5
🧹 Data Preprocessing
Run the preprocessing pipeline in the following order:

bash
# Step 1: Clean and curate UniProt datasets
python codes/01_data_preprocessing.py

# Step 2: Generate CD-HIT and LOSO evaluation splits
python codes/02_create_evaluation_splits.py

# Step 3: Create clean embedding datasets
python codes/03_create_clean_splits.py

# Step 4: Combine processed datasets
python codes/04_to_create_combine_file.py

# Step 5: Add species labels
python codes/05_add_species_column.py
🤖 Model Training
Attention-Enhanced Deep Learning Models
bash
python codes/06_attention_dnn_training.py
Models included:

Attention-Enhanced DNN (Proposed)

DNN Baseline

Logistic Regression

Ablation without Attention

Ablation without Residual Connections

Ablation using 50% of the training data

Machine Learning Baseline Models
bash
python codes/07_baseline_comparisons_cd_hit.py
Models included:

Random Forest

Support Vector Machine (SVM)

MLP-256

MLP-512

🌿 Leave-One-Species-Out (LOSO) Evaluation
bash
python codes/08_loso_training.py
The LOSO framework trains the models on three plant species and evaluates them on the remaining unseen species.

Total experiments:

10 Models × 4 Species × 3 Learning Rates × 4 Batch Sizes = 480 experiments

Each experiment performs 10-fold cross-validation = ~4,800 model trainings

💡 Checkpoint functionality enables interrupted experiments to resume automatically.

📊 Results Extraction
bash
# Extract CD-HIT evaluation results
python codes/09_combined_csv_cd_hit.py

# Combine LOSO evaluation results
python codes/10_combine_loso_results.py
📈 Statistical Analysis
bash
python codes/11_Statistical_Analysis_Final_code.py
This script performs:

Friedman Test

Repeated-Measures ANOVA

Post-hoc statistical comparisons

📉 Figure Generation
Main Manuscript Figures
bash
python codes/12_Generate_Figures.py
This script automatically generates all publication-quality figures used in the manuscript, including:

Performance comparison bar charts

ROC curves

Species-wise performance analysis

Model comparison visualizations

Confusion Matrices (Article + Supplementary)
bash
python codes/13_confusion_matrix.py
This script generates confusion matrices for both the main article and supplementary materials:

Article Figure:

2×2 grid showing the best method per species (4 species)

Highlighted with accuracy annotations

Publication-ready format

Supplementary Materials:

All 10 methods × 4 species = 40 confusion matrices

Individual PNG files for each combination

Combined PDF for easy review

Summary CSV and LaTeX tables with metrics (Accuracy, Precision, Recall, F1)

Species comparison figures showing all methods

Output Location: figures/confusion_matrices1/

article/ - Main figure for manuscript

supplementary/ - All supplementary materials

⚡ Quick Start
bash
# ============================================
# Step 1: Data Preprocessing
# ============================================
python codes/01_data_preprocessing.py
python codes/02_create_evaluation_splits.py
python codes/03_create_clean_splits.py
python codes/04_to_create_combine_file.py
python codes/05_add_species_column.py

# ============================================
# Step 2: Model Training
# ============================================
python codes/06_attention_dnn_training.py
python codes/07_baseline_comparisons_cd_hit.py
python codes/08_loso_training.py

# ============================================
# Step 3: Extract Results
# ============================================
python codes/09_combined_csv_cd_hit.py
python codes/10_combine_loso_results.py

# ============================================
# Step 4: Statistical Analysis
# ============================================
python codes/11_Statistical_Analysis_Final_code.py

# ============================================
# Step 5: Generate Figures
# ============================================
python codes/12_Generate_Figures.py          # Main figures
python codes/13_confusion_matrix.py          # Confusion matrices
📊 Evaluation Metrics
The models are evaluated using:

Accuracy

Precision

Recall

F1-score

Matthews Correlation Coefficient (MCC)

ROC-AUC
## 🔬 Reproducibility

All results in this paper are fully reproducible using the following:

| Aspect | Implementation |
|--------|----------------|
| **Versioned Code** | Zenodo DOI: [10.5281/zenodo.21871663](https://doi.org/10.5281/zenodo.21871663) |
| **Commit Hash** | `2e269f30f985322bad58b1fb3cac527e5a91e64b` |
| **GitHub Release** | [v1.0.0](https://github.com/YOUR_USERNAME/plant-enzyme-classification/releases/tag/v1.0.0) |
| **Random Seeds** | `RANDOM_SEED = 42` for all experiments |
| **Data Source** | UniProt with taxonomy IDs: 3702, 3705, 4530, 4565 |
| **Embedding Source** | UniProt embeddings (1024-dim, HDF5 format) |
| **Train/Test Splits** | CD-HIT clusters: 9,650/1,378/2,756 (train/val/test) |
| **Final Dataset** | 22,267 proteins (post-deduplication) |
| **Figure Generation** | `12_Generate_Figures.py` (main) and `13_confusion_matrix.py` (confusion matrices) |
| **Statistical Tests** | Friedman Test and Repeated-Measures ANOVA |
🔬 Reproducibility
All results in this paper are fully reproducible using the following:

Aspect	Implementation
Versioned Code	Zenodo DOI: 10.5281/zenodo.XXXXXXXXXX
Commit Hash	[Insert commit hash from latest release]
Random Seeds	RANDOM_SEED = 42 for all experiments
Data Source	UniProt with taxonomy IDs: 3702, 3705, 4530, 4565
Embedding Source	UniProt embeddings (1024-dim, HDF5 format)
Train/Test Splits	CD-HIT clusters: 9,650/1,378/2,756 (train/val/test)
Final Dataset	22,267 proteins (post-deduplication)
Figure Generation	12_Generate_Figures.py (main) and 13_confusion_matrix.py (confusion matrices)
Statistical Tests	Friedman Test and Repeated-Measures ANOVA
📝 Citation
If you use this repository in your research, please cite:
## 📝 Citation

If you use this repository in your research, please cite:

```bibtex
@article{shahzad2026transfer,
  title={Transfer Learning with Attention-Enhanced Deep Neural Networks for Plant Enzyme Classification Using UniProt Protein Embeddings},
  author={Shahzad, Anjum and Akram, Sheeraz and Mehmood, Tahir},
  journal={BMC Bioinformatics},
  year={2026},
  note={In Press},
  doi={[ADD_DOI_AFTER_PUBLICATION]},
  url={https://github.com/YOUR_USERNAME/plant-enzyme-classification}
}
Zenodo Archive: 10.5281/zenodo.21871663

Corresponding Author: Anjum Shahzad (anjumstat@yahoo.com)

🙏 Acknowledgements
This work makes use of the following open-source resources:

UniProt

TensorFlow

Scikit-learn

CD-HIT

NumPy

Pandas

SciPy

Matplotlib

Biopython

📄 License
This repository is licensed under the MIT License. See the LICENSE file for details.

Questions or issues? Please open an issue on GitHub or contact the corresponding author.






