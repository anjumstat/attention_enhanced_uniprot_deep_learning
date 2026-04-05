"""
Results Section Generator for Attention-Enhanced Deep Learning Paper
Enzyme Classification using UniProt Embeddings
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
BASE_DIR = "D:/uni_prot2/NAE/Novel_Attention_Enhanced"
OUTPUT_DIR = "D:/uni_prot2/NAE/Paper_Results_1_1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define methods and their display names with abbreviations
METHODS_CONFIG = {
    'Attention_Enhanced_Basic': {'display': 'Attn-DNN', 'full_name': 'Attention-Enhanced DNN', 'color': '#2E86AB', 'type': 'Proposed'},
    'DNN_Baseline': {'display': 'DNN', 'full_name': 'DNN Baseline', 'color': '#A23B72', 'type': 'Baseline'},
    'Logistic_Baseline': {'display': 'LogReg', 'full_name': 'Logistic Regression', 'color': '#F18F01', 'type': 'Baseline'},
    'Ablation_No_Attention': {'display': 'No-Attn', 'full_name': 'Ablation: No Attention', 'color': '#C73E1D', 'type': 'Ablation'},
    'Ablation_No_Residual': {'display': 'No-Res', 'full_name': 'Ablation: No Residual', 'color': '#6A4E9B', 'type': 'Ablation'},
    'Ablation_50Percent_Data': {'display': '50% Data', 'full_name': 'Ablation: 50% Data', 'color': '#3B8EA5', 'type': 'Ablation'}
}

LEARNING_RATES = [0.01, 0.001, 0.0001]
BATCH_SIZES = [32, 64, 128, 256, 512]  # Added 32 and 64

def load_all_results():
    """Load all experiment results"""
    all_results = []
    
    for lr in LEARNING_RATES:
        for bs in BATCH_SIZES:
            for method_key, config in METHODS_CONFIG.items():
                lr_str = f"{lr:.4f}".replace('.', '_')
                
                # Determine folder name based on method
                if method_key == 'Attention_Enhanced_Basic':
                    folder_name = "results_Attention_Basic"
                elif method_key == 'DNN_Baseline':
                    folder_name = "results_DNN_Baseline"
                elif method_key == 'Logistic_Baseline':
                    folder_name = "results_Logistic_Baseline"
                elif method_key == 'Ablation_No_Attention':
                    folder_name = "ablation_no_attention"
                elif method_key == 'Ablation_No_Residual':
                    folder_name = "ablation_no_residual"
                elif method_key == 'Ablation_50Percent_Data':
                    folder_name = "ablation_50percent_data"
                else:
                    folder_name = f"results_{method_key}"
                
                folder_path = os.path.join(BASE_DIR, f"lr_{lr_str}_bs_{bs}", folder_name)
                npy_dir = os.path.join(folder_path, "npy_files")
                
                if os.path.exists(npy_dir):
                    result = load_experiment_results(npy_dir, lr, bs, method_key, config)
                    if result:
                        all_results.append(result)
    
    return pd.DataFrame(all_results)

def load_experiment_results(npy_dir, lr, bs, method_key, config):
    """Load results from a single experiment"""
    
    result = {
        'method': method_key,
        'method_display': METHODS_CONFIG[method_key]['display'],
        'method_full_name': METHODS_CONFIG[method_key]['full_name'],
        'method_type': config['type'],
        'learning_rate': lr,
        'batch_size': bs,
        'color': config['color']
    }
    
    try:
        # Load fold metrics
        metrics_file = os.path.join(npy_dir, "fold_metrics.npy")
        if os.path.exists(metrics_file):
            metrics = np.load(metrics_file, allow_pickle=True).item()
            
            # Test metrics
            if 'test' in metrics:
                result['test_accuracy'] = np.mean(metrics['test']['accuracy'])
                result['test_accuracy_std'] = np.std(metrics['test']['accuracy'])
                result['test_precision'] = np.mean(metrics['test']['precision'])
                result['test_recall'] = np.mean(metrics['test']['recall'])
                result['test_f1'] = np.mean(metrics['test']['f1'])
                result['test_mcc'] = np.mean(metrics['test']['mcc'])
                result['test_auc'] = np.mean(metrics['test']['auc'])
                result['test_auc_std'] = np.std(metrics['test']['auc'])
                result['test_acc_values'] = metrics['test']['accuracy']
                result['test_auc_values'] = metrics['test']['auc']
            
            # Validation metrics
            if 'val' in metrics:
                result['val_accuracy'] = np.mean(metrics['val']['accuracy'])
                result['val_accuracy_std'] = np.std(metrics['val']['accuracy'])
                result['val_auc'] = np.mean(metrics['val']['auc'])
                result['val_acc_values'] = metrics['val']['accuracy']
            
            # Training metrics
            if 'train' in metrics:
                result['train_accuracy'] = np.mean(metrics['train']['accuracy'])
                result['train_acc_values'] = metrics['train']['accuracy']
            
            # Training time
            if 'training_time' in metrics:
                result['training_time'] = np.mean(metrics['training_time'])
        
        # Load stability metrics
        stability_file = os.path.join(npy_dir, "stability_metrics.npy")
        if os.path.exists(stability_file):
            stability = np.load(stability_file, allow_pickle=True).item()
            result['stability'] = stability.get('overall_stability', 0)
        
        return result
        
    except Exception as e:
        print(f"Error loading {method_key} LR={lr} BS={bs}: {e}")
        return None

# ============================================================
# OVERFITTING ANALYSIS FUNCTIONS
# ============================================================

def calculate_overfitting_metrics(results_df):
    """Calculate overfitting metrics for each method"""
    
    overfitting_data = []
    
    for method in METHODS_CONFIG.keys():
        method_df = results_df[results_df['method'] == method]
        
        if not method_df.empty:
            # Calculate average metrics across all configurations
            avg_train_acc = method_df['train_accuracy'].mean()
            avg_val_acc = method_df['val_accuracy'].mean()
            avg_test_acc = method_df['test_accuracy'].mean()
            
            # Calculate overfitting gap
            train_val_gap = avg_train_acc - avg_val_acc
            train_test_gap = avg_train_acc - avg_test_acc
            val_test_gap = avg_val_acc - avg_test_acc
            
            # Calculate overfitting ratio
            overfitting_ratio = (avg_train_acc - avg_test_acc) / avg_train_acc if avg_train_acc > 0 else 0
            
            # Calculate variance across folds
            avg_val_std = method_df['val_accuracy_std'].mean()
            avg_test_std = method_df['test_accuracy_std'].mean()
            
            # Classification: Low, Medium, High overfitting
            if overfitting_ratio < 0.02:
                overfitting_level = "Low"
            elif overfitting_ratio < 0.05:
                overfitting_level = "Medium"
            else:
                overfitting_level = "High"
            
            overfitting_data.append({
                'Method': METHODS_CONFIG[method]['full_name'],
                'Method Abbr': METHODS_CONFIG[method]['display'],
                'Train Acc': f"{avg_train_acc:.4f}",
                'Val Acc': f"{avg_val_acc:.4f}",
                'Test Acc': f"{avg_test_acc:.4f}",
                'Train-Val Gap': f"{train_val_gap:.4f}",
                'Train-Test Gap': f"{train_test_gap:.4f}",
                'Overfitting Ratio': f"{overfitting_ratio:.4f}",
                'Val Std': f"{avg_val_std:.4f}",
                'Test Std': f"{avg_test_std:.4f}",
                'Overfitting Level': overfitting_level,
                'Color': METHODS_CONFIG[method]['color']
            })
    
    return pd.DataFrame(overfitting_data)

# ============================================================
# TABLE GENERATION FUNCTIONS
# ============================================================

def generate_table1_dataset_summary():
    """Table 1: Dataset summary statistics"""
    
    data = {
        'Species': ['A. thaliana', 'Brassica species', 'O. sativa (Rice)', 'T. aestivum (Wheat)', 'Total'],
        'Total Genes': [16682, 614, 14697, 1535, 33528],
        'Enzymes': [6381, 184, 5530, 779, 12874],
        'Non-Enzymes': [10301, 430, 9167, 756, 20654],
        'Enzyme (%)': [38.25, 29.97, 37.63, 50.75, 38.40]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(OUTPUT_DIR, "Table1_Dataset_Summary.csv"), index=False)
    
    print("✅ Table 1 generated: Dataset Summary")
    return df

def generate_table2_dataset_cleaned():
    """Table 2: Dataset after duplicate removal"""
    
    data = {
        'Species': ['A. thaliana', 'Brassica species', 'O. sativa (Rice)', 'T. aestivum (Wheat)', 'Total'],
        'Total Genes': [15487, 588, 13695, 1460, 31230],
        'Enzymes': [5945, 174, 5176, 848, 12143],
        'Non-Enzymes': [9542, 414, 8519, 612, 19087],
        'Enzyme (%)': [38.39, 29.59, 37.80, 58.08, 38.88]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(OUTPUT_DIR, "Table2_Dataset_Cleaned.csv"), index=False)
    
    print("✅ Table 2 generated: Dataset After Cleaning")
    return df

def generate_table3_method_performance():
    """Table 3: Method performance comparison (best hyperparameters)"""
    
    results_df = load_all_results()
    
    if results_df.empty:
        print("No results found!")
        return None
    
    # Get best configuration for each method
    table_data = []
    
    for method in METHODS_CONFIG.keys():
        method_df = results_df[results_df['method'] == method]
        if not method_df.empty:
            # Find best configuration by test accuracy
            best_idx = method_df['test_accuracy'].idxmax()
            best_row = method_df.loc[best_idx]
            
            table_data.append({
                'Method': METHODS_CONFIG[method]['full_name'],
                'Type': METHODS_CONFIG[method]['type'],
                'LR': best_row['learning_rate'],
                'Batch Size': best_row['batch_size'],
                'Accuracy': f"{best_row['test_accuracy']:.4f} +/- {best_row['test_accuracy_std']:.4f}",
                'Precision': f"{best_row['test_precision']:.4f}",
                'Recall': f"{best_row['test_recall']:.4f}",
                'F1-Score': f"{best_row['test_f1']:.4f}",
                'MCC': f"{best_row['test_mcc']:.4f}",
                'AUC': f"{best_row['test_auc']:.4f} +/- {best_row['test_auc_std']:.4f}",
                'Stability': f"{best_row.get('stability', 0):.4f}"
            })
    
    df = pd.DataFrame(table_data)
    df.to_csv(os.path.join(OUTPUT_DIR, "Table3_Method_Performance.csv"), index=False)
    
    print("✅ Table 3 generated: Method Performance Comparison")
    return df

def generate_table4_ablation_study():
    """Table 4: Ablation study results"""
    
    results_df = load_all_results()
    
    if results_df.empty:
        return None
    
    ablation_methods = ['Attention_Enhanced_Basic', 'Ablation_No_Attention', 'Ablation_No_Residual', 'Ablation_50Percent_Data']
    
    table_data = []
    
    # Initialize baseline variables
    baseline_acc = None
    baseline_auc = None
    
    for method in ablation_methods:
        method_df = results_df[results_df['method'] == method]
        if not method_df.empty:
            # Average across all configurations for ablation comparison
            avg_acc = method_df['test_accuracy'].mean()
            avg_auc = method_df['test_auc'].mean()
            avg_stability = method_df['stability'].mean()
            
            # Calculate improvement over baseline (Attention_Enhanced_Basic)
            if method == 'Attention_Enhanced_Basic':
                baseline_acc = avg_acc
                baseline_auc = avg_auc
                improvement_acc = "-"
                improvement_auc = "-"
            else:
                if baseline_acc is not None:
                    imp_acc = ((avg_acc - baseline_acc) / baseline_acc) * 100
                    improvement_acc = f"{imp_acc:+.1f}%"
                else:
                    improvement_acc = "N/A"
                
                if baseline_auc is not None:
                    imp_auc = ((avg_auc - baseline_auc) / baseline_auc) * 100
                    improvement_auc = f"{imp_auc:+.1f}%"
                else:
                    improvement_auc = "N/A"
            
            table_data.append({
                'Ablation': METHODS_CONFIG[method]['full_name'],
                'Avg Accuracy': f"{avg_acc:.4f}",
                'Avg AUC': f"{avg_auc:.4f}",
                'Avg Stability': f"{avg_stability:.4f}",
                'Change Accuracy': improvement_acc,
                'Change AUC': improvement_auc
            })
    
    df = pd.DataFrame(table_data)
    df.to_csv(os.path.join(OUTPUT_DIR, "Table4_Ablation_Study.csv"), index=False)
    
    print("✅ Table 4 generated: Ablation Study")
    return df

def generate_table5_hyperparameter_analysis():
    """Table 5: Hyperparameter analysis (best method)"""
    
    results_df = load_all_results()
    
    if results_df.empty:
        return None
    
    # Focus on Attention_Enhanced_Basic
    method_df = results_df[results_df['method'] == 'Attention_Enhanced_Basic']
    
    table_data = []
    
    for lr in LEARNING_RATES:
        for bs in BATCH_SIZES:
            subset = method_df[(method_df['learning_rate'] == lr) & (method_df['batch_size'] == bs)]
            if not subset.empty:
                row = subset.iloc[0]
                table_data.append({
                    'Learning Rate': lr,
                    'Batch Size': bs,
                    'Accuracy': f"{row['test_accuracy']:.4f} +/- {row['test_accuracy_std']:.4f}",
                    'AUC': f"{row['test_auc']:.4f} +/- {row['test_auc_std']:.4f}",
                    'F1-Score': f"{row['test_f1']:.4f}",
                    'Stability': f"{row.get('stability', 0):.4f}"
                })
    
    df = pd.DataFrame(table_data)
    df.to_csv(os.path.join(OUTPUT_DIR, "Table5_Hyperparameter_Analysis.csv"), index=False)
    
    print("✅ Table 5 generated: Hyperparameter Analysis")
    return df

def generate_table6_overfitting_summary(df_overfitting):
    """Table 6: Overfitting analysis summary"""
    
    table_data = df_overfitting[['Method', 'Train Acc', 'Val Acc', 'Test Acc', 
                                  'Train-Val Gap', 'Train-Test Gap', 
                                  'Overfitting Ratio', 'Overfitting Level']].copy()
    
    table_data.to_csv(os.path.join(OUTPUT_DIR, "Table6_Overfitting_Analysis.csv"), index=False)
    
    print("✅ Table 6 generated: Overfitting Analysis")
    return table_data

# ============================================================
# GRAPH GENERATION FUNCTIONS
# ============================================================

def set_plot_style():
    """Set consistent plotting style for publication with 500 dpi and font size 14 bold"""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Set font parameters - minimum 14pt bold
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['legend.title_fontsize'] = 11
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['figure.dpi'] = 500
    plt.rcParams['savefig.dpi'] = 500

def generate_figure1_performance_comparison():
    """Figure 1: Overall performance comparison bar chart"""
    
    set_plot_style()
    results_df = load_all_results()
    
    if results_df.empty:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    
    # Calculate mean performance for each method
    method_performance = results_df.groupby('method').agg({
        'test_accuracy': 'mean',
        'test_auc': 'mean',
        'test_f1': 'mean',
        'test_mcc': 'mean'
    }).round(4)
    
    # Reindex to maintain order
    method_order = list(METHODS_CONFIG.keys())
    method_performance = method_performance.reindex(method_order)
    method_labels = [METHODS_CONFIG[m]['display'] for m in method_order]
    colors = [METHODS_CONFIG[m]['color'] for m in method_order]
    
    # Plot 1: Accuracy
    bars1 = axes[0].bar(method_labels, method_performance['test_accuracy'], color=colors, alpha=0.8)
    axes[0].set_ylabel('Accuracy', fontweight='bold')
    axes[0].set_title('(a) Test Accuracy', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45, labelsize=11)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels above bars
    for bar, v in zip(bars1, method_performance['test_accuracy']):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008, 
                    f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')
    
    # Plot 2: AUC
    bars2 = axes[1].bar(method_labels, method_performance['test_auc'], color=colors, alpha=0.8)
    axes[1].set_ylabel('AUC', fontweight='bold')
    axes[1].set_title('(b) ROC-AUC', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45, labelsize=11)
    axes[1].set_ylim(0.5, 1.0)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, v in zip(bars2, method_performance['test_auc']):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008, 
                    f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')
    
    # Plot 3: F1 and MCC grouped
    x = np.arange(len(method_labels))
    width = 0.35
    
    bars3a = axes[2].bar(x - width/2, method_performance['test_f1'], width, label='F1-Score', color='#2E86AB', alpha=0.8)
    bars3b = axes[2].bar(x + width/2, method_performance['test_mcc'], width, label='MCC', color='#A23B72', alpha=0.8)
    axes[2].set_ylabel('Score', fontweight='bold')
    axes[2].set_title('(c) F1-Score and MCC', fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(method_labels, rotation=45, ha='right', fontsize=11)
    axes[2].legend(loc='lower right', prop={'size': 11, 'weight': 'bold'})
    axes[2].set_ylim(0.4, 1.0)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, v in zip(bars3a, method_performance['test_f1']):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008, 
                    f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
    for bar, v in zip(bars3b, method_performance['test_mcc']):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008, 
                    f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, wspace=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure1_Performance_Comparison.png"), dpi=500, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure1_Performance_Comparison.pdf"), dpi=500, bbox_inches='tight')
    plt.close()
    
    print("✅ Figure 1 generated: Performance Comparison")

def generate_figure2_roc_curves():
    """Figure 2: ROC curves for all methods (average of 10 folds)"""
    
    set_plot_style()
    results_df = load_all_results()
    
    if results_df.empty:
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Colors for different method types
    colors = {
        'Proposed': '#2E86AB',
        'Baseline': '#A23B72',
        'Ablation': '#C73E1D'
    }
    
    line_styles = {
        'Proposed': '-',
        'Baseline': '--',
        'Ablation': ':'
    }
    
    for method in METHODS_CONFIG.keys():
        method_df = results_df[results_df['method'] == method]
        if not method_df.empty:
            # Get best configuration
            best_idx = method_df['test_auc'].idxmax()
            best_row = method_df.loc[best_idx]
            
            # Try to load actual ROC data
            lr_str = f"{best_row['learning_rate']:.4f}".replace('.', '_')
            bs = best_row['batch_size']
            
            # Determine folder name
            if method == 'Attention_Enhanced_Basic':
                folder_name = "results_Attention_Basic"
            elif method == 'DNN_Baseline':
                folder_name = "results_DNN_Baseline"
            elif method == 'Logistic_Baseline':
                folder_name = "results_Logistic_Baseline"
            elif method == 'Ablation_No_Attention':
                folder_name = "ablation_no_attention"
            elif method == 'Ablation_No_Residual':
                folder_name = "ablation_no_residual"
            elif method == 'Ablation_50Percent_Data':
                folder_name = "ablation_50percent_data"
            else:
                folder_name = f"results_{method}"
            
            npy_dir = os.path.join(BASE_DIR, f"lr_{lr_str}_bs_{bs}", folder_name, "npy_files")
            
            roc_file = os.path.join(npy_dir, "roc_data_all_folds.npy")
            if os.path.exists(roc_file):
                try:
                    roc_data = np.load(roc_file, allow_pickle=True)
                    # Calculate mean ROC across all folds
                    all_fpr = []
                    all_tpr = []
                    for fold_data in roc_data:
                        if len(fold_data) >= 4:
                            all_fpr.append(fold_data[1])
                            all_tpr.append(fold_data[2])
                    
                    if all_fpr and all_tpr:
                        mean_fpr = np.linspace(0, 1, 100)
                        mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)], axis=0)
                        auc = best_row['test_auc']
                        
                        ax.plot(mean_fpr, mean_tpr, linewidth=2.5, 
                               label=f"{METHODS_CONFIG[method]['display']} (AUC = {auc:.3f})",
                               color=colors[METHODS_CONFIG[method]['type']],
                               linestyle=line_styles[METHODS_CONFIG[method]['type']])
                except Exception as e:
                    print(f"Error loading ROC for {method}: {e}")
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random Chance')
    
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('ROC Curves for All Methods (Average of 10-Fold CV)', fontweight='bold')
    ax.legend(loc='lower right', prop={'size': 11, 'weight': 'bold'})
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure2_ROC_Curves.png"), dpi=500, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure2_ROC_Curves.pdf"), dpi=500, bbox_inches='tight')
    plt.close()
    
    print("✅ Figure 2 generated: ROC Curves")

def generate_figure3_ablation_bars():
    """Figure 3: Ablation study bar chart"""
    
    set_plot_style()
    results_df = load_all_results()
    
    if results_df.empty:
        return
    
    ablation_methods = ['Attention_Enhanced_Basic', 'Ablation_No_Attention', 'Ablation_No_Residual', 'Ablation_50Percent_Data']
    ablation_labels = [METHODS_CONFIG[m]['display'] for m in ablation_methods]
    colors = [METHODS_CONFIG[m]['color'] for m in ablation_methods]
    
    # Calculate metrics
    metrics = []
    for method in ablation_methods:
        method_df = results_df[results_df['method'] == method]
        metrics.append({
            'accuracy': method_df['test_accuracy'].mean(),
            'auc': method_df['test_auc'].mean(),
            'f1': method_df['test_f1'].mean(),
            'stability': method_df['stability'].mean()
        })
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    # Plot 1: Accuracy and AUC
    x = np.arange(len(ablation_labels))
    width = 0.35
    
    bars1a = axes[0].bar(x - width/2, [m['accuracy'] for m in metrics], width, label='Accuracy', color='#2E86AB', alpha=0.8)
    bars1b = axes[0].bar(x + width/2, [m['auc'] for m in metrics], width, label='AUC', color='#F18F01', alpha=0.8)
    axes[0].set_ylabel('Score', fontweight='bold')
    axes[0].set_title('(a) Accuracy and AUC Comparison', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ablation_labels, fontsize=11, fontweight='bold')
    axes[0].legend(loc='upper right', bbox_to_anchor=(0.86, 1.0), prop={'size': 11, 'weight': 'bold'})
    axes[0].set_ylim(0.5, 1.0)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels at the TOP of bars for No-Res
    for i, (bar, v) in enumerate(zip(bars1a, [m['accuracy'] for m in metrics])):
        if ablation_labels[i] == 'No-Res':
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, 
                        f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
        else:
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008, 
                        f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
    
    for i, (bar, v) in enumerate(zip(bars1b, [m['auc'] for m in metrics])):
        if ablation_labels[i] == 'No-Res':
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, 
                        f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
        else:
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008, 
                        f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
    
    # Plot 2: F1-Score only - shift legend to top of No-Res bar
    bars2a = axes[1].bar(x, [m['f1'] for m in metrics], width, label='F1-Score', color='#A23B72', alpha=0.8)
    axes[1].set_ylabel('Score', fontweight='bold')
    axes[1].set_title('(b) F1-Score', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ablation_labels, fontsize=11, fontweight='bold')
    axes[1].set_ylim(0.4, 1.0)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels at the TOP of bars for No-Res
    for i, (bar, v) in enumerate(zip(bars2a, [m['f1'] for m in metrics])):
        if ablation_labels[i] == 'No-Res':
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, 
                        f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
        else:
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008, 
                        f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
    
    # Place legend at the top center of the plot
    axes[1].legend(loc='upper center', prop={'size': 11, 'weight': 'bold'})
    
    # Plot 3: Percentage drop from baseline
    baseline_acc = metrics[0]['accuracy']
    baseline_auc = metrics[0]['auc']
    
    drops = []
    for i in range(1, len(metrics)):
        drops.append({
            'accuracy_drop': ((baseline_acc - metrics[i]['accuracy']) / baseline_acc) * 100,
            'auc_drop': ((baseline_auc - metrics[i]['auc']) / baseline_auc) * 100
        })
    
    x_drop = np.arange(len(drops))
    bars3a = axes[2].bar(x_drop - width/2, [d['accuracy_drop'] for d in drops], width, label='Accuracy Drop', color='#C73E1D', alpha=0.8)
    bars3b = axes[2].bar(x_drop + width/2, [d['auc_drop'] for d in drops], width, label='AUC Drop', color='#3B8EA5', alpha=0.8)
    axes[2].set_ylabel('Performance Drop (%)', fontweight='bold')
    axes[2].set_title('(c) Performance Degradation', fontweight='bold')
    axes[2].set_xticks(x_drop)
    axes[2].set_xticklabels(ablation_labels[1:], fontsize=11, fontweight='bold')
    
    # Move legend to upper left
    axes[2].legend(loc='upper left', prop={'size': 11, 'weight': 'bold'})
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Add value labels at the top of bars
    for bar, v in zip(bars3a, [d['accuracy_drop'] for d in drops]):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, 
                    f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
    for bar, v in zip(bars3b, [d['auc_drop'] for d in drops]):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                    f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, wspace=0.35)
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure3_Ablation_Study.png"), dpi=500, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure3_Ablation_Study.pdf"), dpi=500, bbox_inches='tight')
    plt.close()
    
    print("✅ Figure 3 generated: Ablation Study")

def generate_figure4_hyperparameter_heatmap():
    """Figure 4: Hyperparameter optimization heatmap with batch sizes 32, 64, 128, 256, 512"""
    
    set_plot_style()
    results_df = load_all_results()
    
    if results_df.empty:
        return
    
    method_df = results_df[results_df['method'] == 'Attention_Enhanced_Basic']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Heatmap 1: Accuracy
    acc_pivot = method_df.pivot_table(values='test_accuracy', index='batch_size', columns='learning_rate')
    # Sort index to show batch sizes in order
    acc_pivot = acc_pivot.reindex(sorted(acc_pivot.index))
    # Sort columns to show learning rates in order
    acc_pivot = acc_pivot.reindex(sorted(acc_pivot.columns), axis=1)
    
    sns.heatmap(acc_pivot, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[0], 
                cbar_kws={'label': 'Accuracy'}, annot_kws={'weight': 'bold', 'size': 11})
    axes[0].set_title('(a) Test Accuracy Heatmap', fontweight='bold')
    axes[0].set_xlabel('Learning Rate', fontweight='bold')
    axes[0].set_ylabel('Batch Size', fontweight='bold')
    axes[0].tick_params(labelsize=11)
    
    # Heatmap 2: AUC
    auc_pivot = method_df.pivot_table(values='test_auc', index='batch_size', columns='learning_rate')
    auc_pivot = auc_pivot.reindex(sorted(auc_pivot.index))
    auc_pivot = auc_pivot.reindex(sorted(auc_pivot.columns), axis=1)
    
    sns.heatmap(auc_pivot, annot=True, fmt='.4f', cmap='YlGnBu', ax=axes[1],
                cbar_kws={'label': 'AUC'}, annot_kws={'weight': 'bold', 'size': 11})
    axes[1].set_title('(b) AUC Heatmap', fontweight='bold')
    axes[1].set_xlabel('Learning Rate', fontweight='bold')
    axes[1].set_ylabel('Batch Size', fontweight='bold')
    axes[1].tick_params(labelsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure4_Hyperparameter_Heatmap.png"), dpi=500, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure4_Hyperparameter_Heatmap.pdf"), dpi=500, bbox_inches='tight')
    plt.close()
    
    print("✅ Figure 4 generated: Hyperparameter Heatmap")

def generate_figure5_cross_validation_boxplot():
    """Figure 5: Cross-validation fold distribution boxplot"""
    
    set_plot_style()
    results_df = load_all_results()
    
    if results_df.empty:
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Collect fold-wise accuracies for each method
    plot_data = []
    labels = []
    colors = []
    
    for method in METHODS_CONFIG.keys():
        method_df = results_df[results_df['method'] == method]
        all_accuracies = []
        
        for _, row in method_df.iterrows():
            if 'test_acc_values' in row and row['test_acc_values'] is not None:
                all_accuracies.extend(row['test_acc_values'])
        
        if all_accuracies:
            plot_data.append(all_accuracies)
            labels.append(METHODS_CONFIG[method]['display'])
            colors.append(METHODS_CONFIG[method]['color'])
    
    # Create boxplot
    bp = ax.boxplot(plot_data, labels=labels, patch_artist=True, showmeans=True,
                    meanprops={'marker': 'D', 'markerfacecolor': 'red', 'markersize': 8, 'markeredgecolor': 'red'})
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Test Accuracy', fontweight='bold')
    ax.set_title('Cross-Validation Accuracy Distribution Across 10 Folds', fontweight='bold')
    ax.tick_params(axis='x', labelsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure5_CrossValidation_Boxplot.png"), dpi=500, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure5_CrossValidation_Boxplot.pdf"), dpi=500, bbox_inches='tight')
    plt.close()
    
    print("✅ Figure 5 generated: Cross-Validation Boxplot")

def generate_figure6_training_convergence():
    """Figure 6: Training convergence curves (average of 10 folds for each method)"""
    
    set_plot_style()
    results_df = load_all_results()
    
    if results_df.empty:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # For each method, plot average training/validation accuracy across all folds
    for idx, method in enumerate(METHODS_CONFIG.keys()):
        row = idx // 3
        col = idx % 3
        
        method_df = results_df[results_df['method'] == method]
        
        if not method_df.empty:
            # Get best configuration
            best_idx = method_df['test_auc'].idxmax()
            best_row = method_df.loc[best_idx]
            
            # Load training history
            lr_str = f"{best_row['learning_rate']:.4f}".replace('.', '_')
            bs = best_row['batch_size']
            
            # Determine folder name
            if method == 'Attention_Enhanced_Basic':
                folder_name = "results_Attention_Basic"
            elif method == 'DNN_Baseline':
                folder_name = "results_DNN_Baseline"
            elif method == 'Logistic_Baseline':
                folder_name = "results_Logistic_Baseline"
            elif method == 'Ablation_No_Attention':
                folder_name = "ablation_no_attention"
            elif method == 'Ablation_No_Residual':
                folder_name = "ablation_no_residual"
            elif method == 'Ablation_50Percent_Data':
                folder_name = "ablation_50percent_data"
            else:
                folder_name = f"results_{method}"
            
            npy_dir = os.path.join(BASE_DIR, f"lr_{lr_str}_bs_{bs}", folder_name, "npy_files")
            history_file = os.path.join(npy_dir, "training_history.npy")
            
            if os.path.exists(history_file):
                history = np.load(history_file, allow_pickle=True).item()
                
                # Collect all fold histories and average
                all_train_acc = []
                all_val_acc = []
                min_epochs = float('inf')
                
                for fold, fold_history in history.items():
                    train_acc = fold_history.get('accuracy', [])
                    val_acc = fold_history.get('val_accuracy', [])
                    
                    if len(train_acc) > 0 and len(val_acc) > 0:
                        all_train_acc.append(train_acc)
                        all_val_acc.append(val_acc)
                        min_epochs = min(min_epochs, len(train_acc), len(val_acc))
                
                if all_train_acc and all_val_acc:
                    # Trim to same length and average
                    trimmed_train = [acc[:min_epochs] for acc in all_train_acc]
                    trimmed_val = [acc[:min_epochs] for acc in all_val_acc]
                    
                    mean_train_acc = np.mean(trimmed_train, axis=0)
                    mean_val_acc = np.mean(trimmed_val, axis=0)
                    
                    epochs = range(1, min_epochs + 1)
                    
                    axes[row, col].plot(epochs, mean_train_acc, 'b-', linewidth=2, label='Training', alpha=0.8)
                    axes[row, col].plot(epochs, mean_val_acc, 'r--', linewidth=2, label='Validation', alpha=0.8)
                    axes[row, col].set_title(METHODS_CONFIG[method]['display'], fontweight='bold')
                    axes[row, col].set_xlabel('Epoch', fontweight='bold')
                    axes[row, col].set_ylabel('Accuracy', fontweight='bold')
                    axes[row, col].legend(loc='lower right', prop={'size': 10})
                    axes[row, col].grid(True, alpha=0.3)
                    axes[row, col].set_ylim(0.4, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure6_Training_Convergence.png"), dpi=500, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure6_Training_Convergence.pdf"), dpi=500, bbox_inches='tight')
    plt.close()
    
    print("✅ Figure 6 generated: Training Convergence Curves")

def generate_figure7_feature_stability():
    """Figure 7: Feature stability analysis"""
    
    set_plot_style()
    results_df = load_all_results()
    
    if results_df.empty:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Stability scores by method
    stability_data = []
    for method in METHODS_CONFIG.keys():
        method_df = results_df[results_df['method'] == method]
        stability_scores = method_df['stability'].dropna().values
        if len(stability_scores) > 0:
            stability_data.append({
                'method': METHODS_CONFIG[method]['display'],
                'stability': stability_scores.mean(),
                'std': stability_scores.std(),
                'color': METHODS_CONFIG[method]['color']
            })
    
    if stability_data:
        methods = [d['method'] for d in stability_data]
        scores = [d['stability'] for d in stability_data]
        stds = [d['std'] for d in stability_data]
        colors = [d['color'] for d in stability_data]
        
        bars = axes[0].bar(methods, scores, yerr=stds, capsize=5, color=colors, alpha=0.8)
        axes[0].set_ylabel('Stability Score', fontweight='bold')
        axes[0].set_title('(a) Feature Stability by Method', fontweight='bold')
        axes[0].tick_params(axis='x', labelsize=11, rotation=45)
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, v in zip(bars, scores):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
    
    # Plot 2: Stability vs Accuracy scatter
    for method in METHODS_CONFIG.keys():
        method_df = results_df[results_df['method'] == method]
        acc_vals = method_df['test_accuracy'].values
        stab_vals = method_df['stability'].dropna().values
        
        # Ensure same length
        if len(acc_vals) == len(stab_vals) and len(acc_vals) > 0:
            axes[1].scatter(stab_vals, acc_vals, 
                          label=METHODS_CONFIG[method]['display'],
                          color=METHODS_CONFIG[method]['color'],
                          s=120, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    axes[1].set_xlabel('Feature Stability Score', fontweight='bold')
    axes[1].set_ylabel('Test Accuracy', fontweight='bold')
    axes[1].set_title('(b) Stability vs Accuracy Trade-off', fontweight='bold')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10, 'weight': 'bold'})
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0.5, 1)
    
    plt.tight_layout()
    
    # Fix the OSError by ensuring directory exists and filename is valid
    try:
        plt.savefig(os.path.join(OUTPUT_DIR, "Figure7_Feature_Stability.png"), dpi=500, bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, "Figure7_Feature_Stability.pdf"), dpi=500, bbox_inches='tight')
        plt.close()
        print("✅ Figure 7 generated: Feature Stability Analysis")
    except OSError as e:
        print(f"Warning: Could not save Figure 7 - {e}")
        # Try alternative filename
        try:
            plt.savefig(os.path.join(OUTPUT_DIR, "Figure7_Stability.png"), dpi=500, bbox_inches='tight')
            plt.close()
            print("✅ Figure 7 generated as 'Figure7_Stability.png'")
        except:
            print("❌ Could not save Figure 7")
            plt.close()

def generate_figure8_overfitting_analysis(df_overfitting):
    """Figure 8: Overfitting analysis bar chart with vertical bar labels and adjusted legend"""
    
    set_plot_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    methods = df_overfitting['Method Abbr'].tolist()
    colors = df_overfitting['Color'].tolist()
    
    # Plot 1: Training, Validation, Test Accuracy comparison
    x = np.arange(len(methods))
    width = 0.25
    
    train_acc = df_overfitting['Train Acc'].astype(float).values
    val_acc = df_overfitting['Val Acc'].astype(float).values
    test_acc = df_overfitting['Test Acc'].astype(float).values
    
    bars1 = axes[0].bar(x - width, train_acc, width, label='Train', color='#2E86AB', alpha=0.8)
    bars2 = axes[0].bar(x, val_acc, width, label='Valid', color='#F18F01', alpha=0.8)
    bars3 = axes[0].bar(x + width, test_acc, width, label='Test', color='#A23B72', alpha=0.8)
    
    axes[0].set_ylabel('Accuracy', fontweight='bold')
    axes[0].set_title('(a) Training vs Validation vs Test Accuracy', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add legend
    axes[0].legend(bbox_to_anchor=(0.75, 0.92), loc='center', prop={'size': 10, 'weight': 'bold'})
    
    # Add value labels above bars - vertical orientation for better readability
    for bar in bars1:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008, 
                    f'{bar.get_height():.2f}', ha='center', va='bottom', 
                    fontsize=8, fontweight='bold', rotation=90)
    for bar in bars2:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008, 
                    f'{bar.get_height():.2f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold', rotation=90)
    for bar in bars3:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008, 
                    f'{bar.get_height():.2f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold', rotation=90)
    
    # Rotate x-axis labels to prevent overlapping
    axes[0].set_xticklabels(methods, fontsize=11, fontweight='bold', rotation=45, ha='right')
    
    # Plot 2: Overfitting Gaps with legend at top left
    train_val_gap = df_overfitting['Train-Val Gap'].astype(float).values
    train_test_gap = df_overfitting['Train-Test Gap'].astype(float).values
    
    x2 = np.arange(len(methods))
    width2 = 0.35
    
    bars4 = axes[1].bar(x2 - width2/2, train_val_gap, width2, label='Train-Val Gap', color='#C73E1D', alpha=0.8)
    bars5 = axes[1].bar(x2 + width2/2, train_test_gap, width2, label='Train-Test Gap', color='#6A4E9B', alpha=0.8)
    
    axes[1].set_ylabel('Accuracy Gap', fontweight='bold')
    axes[1].set_title('(b) Overfitting Gaps', fontweight='bold')
    axes[1].set_xticks(x2)
    axes[1].legend(loc='upper left', prop={'size': 10, 'weight': 'bold'})
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels above bars - vertical orientation
    for bar in bars4:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                    f'{bar.get_height():.4f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold', rotation=90)
    for bar in bars5:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                    f'{bar.get_height():.4f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold', rotation=90)
    
    # Rotate x-axis labels for plot 2
    axes[1].set_xticklabels(methods, fontsize=11, fontweight='bold', rotation=45, ha='right')
    
    # Plot 3: Overfitting Ratio - adjust to keep labels inside border
    overfitting_ratio = df_overfitting['Overfitting Ratio'].astype(float).values * 100
    overfitting_levels = df_overfitting['Overfitting Level'].tolist()
    
    bars6 = axes[2].bar(methods, overfitting_ratio, color=colors, alpha=0.8)
    axes[2].set_ylabel('Overfitting Ratio (%)', fontweight='bold')
    axes[2].set_title('(c) Overfitting Ratio by Method', fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Add value labels inside the bars to prevent going out of border
    for bar, ratio, level in zip(bars6, overfitting_ratio, overfitting_levels):
        # Position label inside the bar if bar is tall enough, otherwise above
        if ratio > 15:
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2, 
                        f'{ratio:.1f}%\n({level})', ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white')
        else:
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{ratio:.1f}%\n({level})', ha='center', va='bottom',
                        fontsize=9, fontweight='bold')
    
    # Rotate x-axis labels for plot 3
    axes[2].set_xticklabels(methods, fontsize=11, fontweight='bold', rotation=45, ha='right')
    
    # Adjust y-axis limit to accommodate labels if needed
    max_ratio = max(overfitting_ratio)
    if max_ratio > 0:
        axes[2].set_ylim(0, max_ratio + max_ratio * 0.2)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, wspace=0.3, bottom=0.15)
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure8_Overfitting_Analysis.png"), dpi=500, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure8_Overfitting_Analysis.pdf"), dpi=500, bbox_inches='tight')
    plt.close()
    
    print("✅ Figure 8 generated: Overfitting Analysis")

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Generate all tables and figures for results section"""
    
    print("="*60)
    print("RESULTS SECTION GENERATOR")
    print("Attention-Enhanced Deep Learning for Enzyme Classification")
    print("="*60)
    
    # Load results
    results_df = load_all_results()
    
    if results_df.empty:
        print("❌ No results found! Please check your BASE_DIR path.")
        return
    
    # Calculate overfitting metrics
    print("\n" + "-"*40)
    print("CALCULATING OVERFITTING METRICS")
    print("-"*40)
    df_overfitting = calculate_overfitting_metrics(results_df)
    
    # Generate Tables
    print("\n" + "-"*40)
    print("GENERATING TABLES")
    print("-"*40)
    
    generate_table1_dataset_summary()
    generate_table2_dataset_cleaned()
    generate_table3_method_performance()
    generate_table4_ablation_study()
    generate_table5_hyperparameter_analysis()
    generate_table6_overfitting_summary(df_overfitting)
    
    # Generate Figures
    print("\n" + "-"*40)
    print("GENERATING FIGURES")
    print("-"*40)
    
    generate_figure1_performance_comparison()
    generate_figure2_roc_curves()
    generate_figure3_ablation_bars()
    generate_figure4_hyperparameter_heatmap()
    generate_figure5_cross_validation_boxplot()
    generate_figure6_training_convergence()
    generate_figure7_feature_stability()
    generate_figure8_overfitting_analysis(df_overfitting)
    
    # Generate summary report
    print("\n" + "-"*40)
    print("GENERATING SUMMARY REPORT")
    print("-"*40)
    
    summary = f"""
    ============================================================
    RESULTS SECTION GENERATION COMPLETE
    ============================================================
    
    Output Directory: {OUTPUT_DIR}
    
    TABLES GENERATED (6 tables):
    ----------------------------
    1. Table1_Dataset_Summary.csv
    2. Table2_Dataset_Cleaned.csv
    3. Table3_Method_Performance.csv
    4. Table4_Ablation_Study.csv
    5. Table5_Hyperparameter_Analysis.csv
    6. Table6_Overfitting_Analysis.csv
    
    FIGURES GENERATED (8 figures, 500 DPI):
    ---------------------------------------
    1. Figure1_Performance_Comparison.png/pdf
    2. Figure2_ROC_Curves.png/pdf
    3. Figure3_Ablation_Study.png/pdf
    4. Figure4_Hyperparameter_Heatmap.png/pdf
    5. Figure5_CrossValidation_Boxplot.png/pdf
    6. Figure6_Training_Convergence.png/pdf
    7. Figure7_Feature_Stability.png/pdf
    8. Figure8_Overfitting_Analysis.png/pdf
    
    ============================================================
    """
    
    with open(os.path.join(OUTPUT_DIR, "RESULTS_SECTION_README.txt"), 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(summary)
    print("\n✅ ALL RESULTS GENERATED SUCCESSFULLY!")

if __name__ == "__main__":
    main()