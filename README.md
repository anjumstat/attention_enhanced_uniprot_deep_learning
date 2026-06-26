# Plant Enzyme Classification - Data Preprocessing

## Overview
This script processes UniProt data for plant enzyme classification. It handles:
- Loading TSV and HDF5 files for 4 plant species
- Extracting EC numbers from the data
- Classifying proteins as enzymes or non-enzymes
- Merging embeddings with labels
- Saving processed datasets

## Species Processed
- *Arabidopsis thaliana* (taxon: 3702)
- *Brassica* species (taxon: 3705)
- *Oryza sativa* (Rice, taxon: 4530)
- *Triticum aestivum* (Wheat, taxon: 4565)

## Requirements
- Python 3.8+
- pandas, numpy, h5py, etc.

## Usage
```python
python 01_data_preprocessing.py
## Output
Combined dataset: combined_all_species.csv

Species-specific datasets

Statistics and reports

### 3. `requirements.txt`

```txt
pandas>=1.3.0
numpy>=1.21.0
h5py>=3.6.0
