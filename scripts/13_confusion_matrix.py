# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 22:17:56 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Generate Confusion Matrices for Article and Supplementary Materials
Based on LOSO results from plant_loso_complete
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import json
from sklearn.metrics import confusion_matrix

# =============================================
# CONFIGURATION
# =============================================

LOSO_BASE_DIR = "D:/uni_prot2/revision/results/plant_loso_complete"
OUTPUT_DIR = "D:/uni_prot2/revision/figures/confusion_matrices1"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "article"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "supplementary"), exist_ok=True)

# Species list
SPECIES_LIST = ['Arabidopsis_thaliana', 'Brassica_spp', 'Oryza_sativa', 'Triticum_aestivum']
SPECIES_DISPLAY = {
    'Arabidopsis_thaliana': 'A. thaliana',
    'Brassica_spp': 'Brassica spp.',
    'Oryza_sativa': 'O. sativa',
    'Triticum_aestivum': 'T. aestivum'
}

# Method display names
METHOD_DISPLAY = {
    'results_Attention_Basic': 'Attn-DNN',
    'results_DNN_Baseline': 'DNN',
    'results_Logistic_Baseline': 'LogReg',
    'ablation_no_attention': 'No-Attn',
    'ablation_no_residual': 'No-Res',
    'ablation_50percent_data': '50% Data',
    'results_Random_Forest': 'RF',
    'results_SVM': 'SVM',
    'results_MLP_256': 'MLP-256',
    'results_MLP_512': 'MLP-512'
}

# Best method per species from LOSO results
BEST_PER_SPECIES = {
    'Triticum_aestivum': 'Ablation: 50% Data',
    'Oryza_sativa': 'Ablation: No Residual',
    'Brassica_spp': 'DNN Baseline',
    'Arabidopsis_thaliana': 'SVM'
}

# Method folder mapping
METHOD_FOLDER_MAP = {
    'Attention_Enhanced_Basic': 'results_Attention_Basic',
    'DNN_Baseline': 'results_DNN_Baseline',
    'Logistic_Baseline': 'results_Logistic_Baseline',
    'Ablation_No_Attention': 'ablation_no_attention',
    'Ablation_No_Residual': 'ablation_no_residual',
    'Ablation_50Percent_Data': 'ablation_50percent_data',
    'Random_Forest': 'results_Random_Forest',
    'SVM': 'results_SVM',
    'MLP_256': 'results_MLP_256',
    'MLP_512': 'results_MLP_512'
}

# Reverse mapping for best method
BEST_METHOD_MAP = {
    'Attention-Enhanced DNN': 'Attention_Enhanced_Basic',
    'DNN Baseline': 'DNN_Baseline',
    'Logistic Regression': 'Logistic_Baseline',
    'Ablation: No Attention': 'Ablation_No_Attention',
    'Ablation: No Residual': 'Ablation_No_Residual',
    'Ablation: 50% Data': 'Ablation_50Percent_Data',
    'Random Forest': 'Random_Forest',
    'SVM': 'SVM',
    'MLP-256': 'MLP_256',
    'MLP-512': 'MLP_512'
}

# =============================================
# FUNCTION: Load Confusion Matrix from NPY
# =============================================

def load_confusion_matrix(species, method, lr, bs):
    """Load confusion matrix from NPY file"""
    
    lr_str = f"{lr:.4f}".replace('.', '_')
    folder_name = METHOD_FOLDER_MAP.get(method, method)
    
    npy_path = os.path.join(
        LOSO_BASE_DIR, species, 
        f"lr_{lr_str}_bs_{bs}", 
        folder_name, 
        "npy_files", 
        "confusion_matrices.npy"
    )
    
    if os.path.exists(npy_path):
        try:
            cm = np.load(npy_path, allow_pickle=True)
            return np.mean(cm, axis=0)  # Average across folds
        except:
            return None
    return None

# =============================================
# FUNCTION: Get Best Configuration for Method
# =============================================

def get_best_config(species, method_name):
    """Get best configuration for a method on a species"""
    
    # Try to find LOSO summary
    loso_summary_path = os.path.join(
        LOSO_BASE_DIR, "LOSO_Complete_Summary.csv"
    )
    
    if os.path.exists(loso_summary_path):
        try:
            df = pd.read_csv(loso_summary_path)
            
            # Check column names
            print(f"  Columns in LOSO summary: {df.columns.tolist()}")
            
            # Try different possible column names
            species_col = None
            method_col = None
            auc_col = None
            
            for col in df.columns:
                if 'species' in col.lower() or 'Species' in col:
                    species_col = col
                if 'method' in col.lower() or 'Method' in col:
                    method_col = col
                if 'auc' in col.lower() or 'AUC' in col:
                    auc_col = col
            
            if species_col and method_col and auc_col:
                subset = df[(df[species_col] == species) & (df[method_col] == method_name)]
                if not subset.empty:
                    best = subset.loc[subset[auc_col].idxmax()]
                    
                    # Try to find learning_rate and batch_size columns
                    lr_col = None
                    bs_col = None
                    for col in df.columns:
                        if 'learning' in col.lower() or 'lr' in col.lower():
                            lr_col = col
                        if 'batch' in col.lower() or 'bs' in col.lower():
                            bs_col = col
                    
                    if lr_col and bs_col:
                        return best[lr_col], best[bs_col]
        except Exception as e:
            print(f"  Warning: Could not read summary: {e}")
    
    # If summary not available, search directory structure
    print(f"  Searching directory for {species} - {method_name}...")
    
    # Try to find any existing configuration
    species_path = os.path.join(LOSO_BASE_DIR, species)
    if os.path.exists(species_path):
        # Look for any lr_*_bs_* directory
        config_dirs = [d for d in os.listdir(species_path) 
                      if d.startswith('lr_') and os.path.isdir(os.path.join(species_path, d))]
        
        for config_dir in config_dirs:
            # Extract lr and bs from directory name
            parts = config_dir.split('_')
            try:
                lr = float(f"{parts[1]}.{parts[2]}")
                bs = int(parts[4])
                
                # Check if this configuration exists for this method
                folder_name = METHOD_FOLDER_MAP.get(method_name, method_name)
                method_path = os.path.join(species_path, config_dir, folder_name)
                if os.path.exists(method_path):
                    return lr, bs
            except:
                continue
    
    # Default fallback
    print(f"  Using default config for {species} - {method_name}")
    return 0.0001, 64

# =============================================
# FUNCTION: Plot Confusion Matrix
# =============================================

def plot_confusion_matrix(cm, title, output_path, 
                          xticklabels=['Non-Enzyme', 'Enzyme'],
                          yticklabels=['Non-Enzyme', 'Enzyme'],
                          figsize=(6, 5), cmap='Blues',
                          annot=True, fmt='.0f'):
    """Plot a single confusion matrix"""
    
    # Set font size for all text elements
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'figure.titlesize': 16,
        'axes.titlesize': 15,
        'axes.labelsize': 14
    })
    
    plt.figure(figsize=figsize)
    
    # Create heatmap with bold annotations
    ax = sns.heatmap(cm, annot=annot, fmt=fmt, cmap=cmap,
                     xticklabels=xticklabels, 
                     yticklabels=yticklabels,
                     cbar=True, cbar_kws={'label': 'Count'},
                     annot_kws={'weight': 'bold', 'size': 13})
    
    # Bold cbar label
    cbar = ax.collections[0].colorbar
    cbar.set_label('Count', weight='bold', size=14)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=15, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=15, fontweight='bold')
    
    # Make tick labels bold
    ax.set_xticklabels(ax.get_xticklabels(), weight='bold', size=13)
    ax.set_yticklabels(ax.get_yticklabels(), weight='bold', size=13)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path}")

# =============================================
# FUNCTION: Create Figure for Article
# =============================================

def create_article_figure():
    """Create Figure with 4 confusion matrices (best method per species)"""
    
    # Set global font sizes
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'figure.titlesize': 16,
        'axes.titlesize': 15,
        'axes.labelsize': 14
    })
    
    print("\n" + "=" * 80)
    print("CREATING ARTICLE FIGURE (4 Best Methods)")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, (species, best_method_display) in enumerate(BEST_PER_SPECIES.items()):
        method_name = BEST_METHOD_MAP.get(best_method_display, best_method_display)
        lr, bs = get_best_config(species, method_name)
        
        print(f"  {species} - {method_name}: LR={lr}, BS={bs}")
        
        cm = load_confusion_matrix(species, method_name, lr, bs)
        
        if cm is not None:
            display_name = SPECIES_DISPLAY.get(species, species)
            method_short = METHOD_DISPLAY.get(
                METHOD_FOLDER_MAP.get(method_name, method_name), 
                method_name
            )
            title = f'({chr(97+idx)}) {display_name} ({method_short})'
            
            # Create heatmap with bold annotations
            ax = sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues',
                             xticklabels=['Non-Enzyme', 'Enzyme'],
                             yticklabels=['Non-Enzyme', 'Enzyme'],
                             ax=axes[idx], cbar=True,
                             cbar_kws={'label': 'Count'},
                             annot_kws={'weight': 'bold', 'size': 13})
            
            # Bold cbar label
            cbar = ax.collections[0].colorbar
            cbar.set_label('Count', weight='bold', size=14)
            
            axes[idx].set_title(title, fontsize=16, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=15, fontweight='bold')
            axes[idx].set_xlabel('Predicted Label', fontsize=15, fontweight='bold')
            
            # Make tick labels bold
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), weight='bold', size=13)
            axes[idx].set_yticklabels(axes[idx].get_yticklabels(), weight='bold', size=13)
            
            # Add accuracy annotation with bold font
            total = np.sum(cm)
            correct = np.trace(cm)
            acc = correct / total
            axes[idx].text(0.5, -0.15, f'Accuracy: {acc:.3f}', 
                          transform=axes[idx].transAxes,
                          ha='center', va='top', fontsize=14, weight='bold')
        else:
            axes[idx].text(0.5, 0.5, f'No data for {species}', 
                          ha='center', va='center', fontsize=14, weight='bold')
            axes[idx].set_title(f'({chr(97+idx)}) {SPECIES_DISPLAY.get(species, species)}', 
                               fontsize=16, fontweight='bold')
    
    plt.suptitle('Confusion Matrices - Best Method per Species (LOSO)', 
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "article", 
                               "Figure_Confusion_Matrices_Article.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "article", 
                            "Figure_Confusion_Matrices_Article.pdf"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Article figure saved to: {output_path}")

# =============================================
# FUNCTION: Create Supplementary Materials
# =============================================

def create_supplementary_figures():
    """Create confusion matrices for all methods × all species"""
    
    # Set global font sizes
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'figure.titlesize': 16,
        'axes.titlesize': 15,
        'axes.labelsize': 14
    })
    
    print("\n" + "=" * 80)
    print("CREATING SUPPLEMENTARY MATERIALS (All Methods × All Species)")
    print("=" * 80)
    
    # All methods
    all_methods = [
        'Attention_Enhanced_Basic',
        'DNN_Baseline',
        'Logistic_Baseline',
        'Ablation_No_Attention',
        'Ablation_No_Residual',
        'Ablation_50Percent_Data',
        'Random_Forest',
        'SVM',
        'MLP_256',
        'MLP_512'
    ]
    
    summary_data = []
    
    for species in SPECIES_LIST:
        species_display = SPECIES_DISPLAY.get(species, species)
        
        for method_name in all_methods:
            method_display = METHOD_DISPLAY.get(
                METHOD_FOLDER_MAP.get(method_name, method_name), 
                method_name
            )
            
            # Get best config for this method on this species
            lr, bs = get_best_config(species, method_name)
            cm = load_confusion_matrix(species, method_name, lr, bs)
            
            if cm is not None:
                # Calculate metrics
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                else:
                    # If confusion matrix is not 2x2, handle differently
                    tn = cm[0, 0] if cm.shape[0] > 0 else 0
                    fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
                    fn = cm[1, 0] if cm.shape[0] > 1 else 0
                    tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
                
                total = np.sum(cm)
                accuracy = (tp + tn) / total if total > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                summary_data.append({
                    'Species': species_display,
                    'Method': method_display,
                    'TN': int(tn),
                    'FP': int(fp),
                    'FN': int(fn),
                    'TP': int(tp),
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1': f1
                })
                
                # Save individual confusion matrix (using the updated plot function)
                title = f'{species_display} - {method_display}'
                output_path = os.path.join(
                    OUTPUT_DIR, "supplementary",
                    f'CM_{species}_{method_name}.png'
                )
                plot_confusion_matrix(
                    cm, title, output_path,
                    figsize=(5, 4)
                )
                print(f"  ✅ {species} - {method_display}: Accuracy={accuracy:.3f}")
            else:
                print(f"  ⚠️ No data for {species} - {method_display}")
    
    # Save summary table
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(OUTPUT_DIR, "supplementary", 
                                    "Confusion_Matrix_Summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✅ Summary table saved to: {summary_path}")
        
        # Also save as LaTeX table
        latex_path = os.path.join(OUTPUT_DIR, "supplementary", 
                                  "Confusion_Matrix_Summary.tex")
        with open(latex_path, 'w') as f:
            f.write(summary_df.to_latex(index=False, float_format="%.3f"))
        print(f"✅ LaTeX table saved to: {latex_path}")
    
    # Create a multi-page PDF for supplementary
    print("\n📄 Creating combined PDF for supplementary materials...")
    
    # Get all PNG files
    png_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "supplementary", "CM_*.png")))
    
    if png_files:
        try:
            from PIL import Image
            
            images = []
            for png_file in png_files:
                img = Image.open(png_file)
                images.append(img)
            
            if images:
                pdf_path = os.path.join(OUTPUT_DIR, "supplementary", 
                                       "Supplementary_Confusion_Matrices.pdf")
                images[0].save(pdf_path, save_all=True, append_images=images[1:])
                print(f"✅ Combined PDF saved to: {pdf_path}")
        except ImportError:
            print("  ⚠️ PIL not installed. Skipping PDF creation.")

# =============================================
# FUNCTION: Create Individual Species Comparison
# =============================================

def create_species_comparison_figure():
    """Create comparison figure showing all methods for a species"""
    
    # Set global font sizes
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16,
        'axes.titlesize': 14,
        'axes.labelsize': 13
    })
    
    print("\n" + "=" * 80)
    print("CREATING SPECIES COMPARISON FIGURES")
    print("=" * 80)
    
    all_methods = [
        'Attention_Enhanced_Basic',
        'DNN_Baseline',
        'Logistic_Baseline',
        'Ablation_No_Attention',
        'Ablation_No_Residual',
        'Ablation_50Percent_Data',
        'Random_Forest',
        'SVM',
        'MLP_256',
        'MLP_512'
    ]
    
    method_display_short = {
        'Attention_Enhanced_Basic': 'Attn-DNN',
        'DNN_Baseline': 'DNN',
        'Logistic_Baseline': 'LogReg',
        'Ablation_No_Attention': 'No-Attn',
        'Ablation_No_Residual': 'No-Res',
        'Ablation_50Percent_Data': '50% Data',
        'Random_Forest': 'RF',
        'SVM': 'SVM',
        'MLP_256': 'MLP-256',
        'MLP_512': 'MLP-512'
    }
    
    for species in SPECIES_LIST:
        species_display = SPECIES_DISPLAY.get(species, species)
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, method_name in enumerate(all_methods):
            method_short = method_display_short.get(method_name, method_name)
            lr, bs = get_best_config(species, method_name)
            cm = load_confusion_matrix(species, method_name, lr, bs)
            
            if cm is not None:
                ax = sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues',
                                 xticklabels=['Non-Enzyme', 'Enzyme'],
                                 yticklabels=['Non-Enzyme', 'Enzyme'],
                                 ax=axes[idx], cbar=False,
                                 annot_kws={'weight': 'bold', 'size': 11})
                
                axes[idx].set_title(method_short, fontsize=13, fontweight='bold')
                
                # Make tick labels bold
                axes[idx].set_xticklabels(axes[idx].get_xticklabels(), weight='bold', size=11)
                axes[idx].set_yticklabels(axes[idx].get_yticklabels(), weight='bold', size=11)
                
                # Add accuracy with bold font
                total = np.sum(cm)
                correct = np.trace(cm)
                acc = correct / total if total > 0 else 0
                axes[idx].text(0.5, -0.12, f'Acc: {acc:.3f}', 
                              transform=axes[idx].transAxes,
                              ha='center', va='top', fontsize=12, weight='bold')
            else:
                axes[idx].text(0.5, 0.5, 'No Data', 
                              ha='center', va='center', fontsize=13, weight='bold')
                axes[idx].set_title(method_short, fontsize=13, fontweight='bold')
                axes[idx].set_xticklabels([])
                axes[idx].set_yticklabels([])
        
        plt.suptitle(f'Confusion Matrices - All Methods ({species_display})', 
                     fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(OUTPUT_DIR, "supplementary", 
                                   f'Species_Comparison_{species}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {output_path}")

# =============================================
# MAIN EXECUTION
# =============================================

if __name__ == "__main__":
    
    print("=" * 80)
    print("CONFUSION MATRIX GENERATOR")
    print("For Article and Supplementary Materials")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)
    
    # 1. Create Article Figure (4 best matrices)
    create_article_figure()
    
    # 2. Create Supplementary Materials (All methods × All species)
    create_supplementary_figures()
    
    # 3. Create Species Comparison Figures
    create_species_comparison_figure()
    
    print("\n" + "=" * 80)
    print("✅ ALL CONFUSION MATRICES GENERATED")
    print("=" * 80)
    print(f"Article figure: {os.path.join(OUTPUT_DIR, 'article')}")
    print(f"Supplementary figures: {os.path.join(OUTPUT_DIR, 'supplementary')}")
    print("=" * 80)