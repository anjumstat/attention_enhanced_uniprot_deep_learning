# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:24:01 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 10:17:51 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
08_extract_f1_mcc_update_tables.py
Extract F1, MCC, Precision, and Recall from saved fold_metrics.npy files
Update all tables with F1, MCC, Precision, and Recall values
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

# =============================================
# CONFIGURATION
# =============================================

ATTENTION_DIR = r"D:\uni_prot2\revision\results\homology_aware_with_test"
BASELINE_DIR = r"D:\uni_prot2\revision\results\homology_aware_baselines_with_checkpoint"
OUTPUT_DIR = r"D:\uni_prot2\revision\results\combined_results_CD_HIT"

# Method display names
METHOD_DISPLAY_NAMES = {
    'results_Attention_Basic': 'Attention-Enhanced DNN',
    'results_DNN_Baseline': 'DNN Baseline',
    'results_Logistic_Baseline': 'Logistic Regression',
    'ablation_no_attention': 'Ablation: No Attention',
    'ablation_no_residual': 'Ablation: No Residual',
    'ablation_50percent_data': 'Ablation: 50% Data',
    'results_Random_Forest': 'Random Forest',
    'results_SVM': 'SVM',
    'results_MLP_256': 'MLP-256',
    'results_MLP_512': 'MLP-512'
}

# =============================================
# FUNCTION TO EXTRACT F1, MCC, PRECISION, RECALL
# =============================================

def extract_all_metrics_from_folds(base_dir):
    """Extract F1, MCC, Precision, and Recall from saved fold_metrics.npy files"""
    
    all_metrics = []
    
    # Find all fold_metrics.npy files
    pattern = os.path.join(base_dir, "lr_*_bs_*", "*", "npy_files", "fold_metrics.npy")
    files = glob.glob(pattern)
    
    print(f"  Found {len(files)} fold_metrics files in {os.path.basename(base_dir)}")
    
    for file_path in files:
        try:
            fold_metrics = np.load(file_path, allow_pickle=True).item()
            
            # Extract method and hyperparameters from path
            path_parts = file_path.split(os.sep)
            
            # Find method name
            method = None
            for i, part in enumerate(path_parts):
                if part == 'npy_files' and i > 0:
                    method = path_parts[i-1]
                    break
            
            if method is None:
                continue
            
            # Extract learning rate and batch size
            lr = None
            bs = None
            for part in path_parts:
                if part.startswith('lr_'):
                    parts = part.split('_')
                    if len(parts) >= 3:
                        try:
                            lr = float(f"{parts[1]}.{parts[2]}")
                        except:
                            lr = float(f"0.{parts[2]}") if parts[1] == '0' else float(parts[1])
                        if len(parts) >= 5:
                            try:
                                bs = int(parts[4])
                            except:
                                bs = None
                    break
            
            # Extract metrics from test_fixed
            if 'test_fixed' in fold_metrics:
                test_metrics = fold_metrics['test_fixed']
                
                # F1
                f1_values = test_metrics.get('f1', [])
                f1_mean = np.mean(f1_values) if f1_values else None
                f1_std = np.std(f1_values) if f1_values else None
                
                # MCC
                mcc_values = test_metrics.get('mcc', [])
                mcc_mean = np.mean(mcc_values) if mcc_values else None
                mcc_std = np.std(mcc_values) if mcc_values else None
                
                # Precision
                precision_values = test_metrics.get('precision', [])
                precision_mean = np.mean(precision_values) if precision_values else None
                precision_std = np.std(precision_values) if precision_values else None
                
                # Recall
                recall_values = test_metrics.get('recall', [])
                recall_mean = np.mean(recall_values) if recall_values else None
                recall_std = np.std(recall_values) if recall_values else None
                
                # Accuracy
                acc_values = test_metrics.get('accuracy', [])
                acc_mean = np.mean(acc_values) if acc_values else None
                acc_std = np.std(acc_values) if acc_values else None
                
                # AUC
                auc_values = test_metrics.get('auc', [])
                auc_mean = np.mean(auc_values) if auc_values else None
                auc_std = np.std(auc_values) if auc_values else None
            else:
                f1_mean = f1_std = mcc_mean = mcc_std = None
                precision_mean = precision_std = recall_mean = recall_std = None
                acc_mean = acc_std = auc_mean = auc_std = None
            
            result = {
                'method': method,
                'method_display': METHOD_DISPLAY_NAMES.get(method, method),
                'learning_rate': lr,
                'batch_size': bs,
                'test_accuracy': acc_mean,
                'test_accuracy_std': acc_std,
                'test_auc': auc_mean,
                'test_auc_std': auc_std,
                'test_f1': f1_mean,
                'test_f1_std': f1_std,
                'test_mcc': mcc_mean,
                'test_mcc_std': mcc_std,
                'test_precision': precision_mean,
                'test_precision_std': precision_std,
                'test_recall': recall_mean,
                'test_recall_std': recall_std
            }
            
            all_metrics.append(result)
            
        except Exception as e:
            print(f"  ⚠️ Error reading {file_path}: {e}")
    
    return pd.DataFrame(all_metrics)

# =============================================
# FORMAT METRIC FUNCTION
# =============================================

def format_metric(value, std):
    """Format metric with mean ± std, handling None values"""
    if value is None or pd.isna(value):
        return 'N/A'
    if std is None or pd.isna(std):
        return f"{value:.4f}"
    return f"{value:.4f} ± {std:.4f}"

# =============================================
# MAIN FUNCTION
# =============================================

def main():
    
    print("=" * 80)
    print("EXTRACT F1, MCC, PRECISION, AND RECALL FROM RESULTS")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # =============================================
    # 1. Extract Metrics from Attention Models
    # =============================================
    print("\n📊 Extracting Metrics from Attention Models...")
    attention_metrics = extract_all_metrics_from_folds(ATTENTION_DIR)
    print(f"   Extracted {len(attention_metrics)} attention model configurations")
    
    # =============================================
    # 2. Extract Metrics from Baseline Models
    # =============================================
    print("\n📊 Extracting Metrics from Baseline Models...")
    baseline_metrics = extract_all_metrics_from_folds(BASELINE_DIR)
    print(f"   Extracted {len(baseline_metrics)} baseline model configurations")
    
    # =============================================
    # 3. Combine All Results
    # =============================================
    all_metrics = pd.concat([attention_metrics, baseline_metrics], ignore_index=True)
    print(f"\n✅ Total configurations: {len(all_metrics)}")
    
    # Drop rows with missing AUC
    all_metrics = all_metrics[all_metrics['test_auc'].notna()]
    print(f"   After removing missing AUC values: {len(all_metrics)}")
    
    # =============================================
    # 4. Table 1: Best Performance Per Method (with all metrics)
    # =============================================
    print("\n" + "=" * 80)
    print("TABLE 1: BEST PERFORMANCE PER METHOD (with F1, MCC, Precision, Recall)")
    print("=" * 80)
    
    best_per_method = all_metrics.loc[all_metrics.groupby('method')['test_auc'].idxmax()]
    best_per_method = best_per_method.sort_values('test_auc', ascending=False)
    
    # Format for display
    display_table1 = best_per_method.copy()
    display_table1['Accuracy'] = display_table1.apply(
        lambda x: format_metric(x['test_accuracy'], x['test_accuracy_std']), axis=1)
    display_table1['AUC'] = display_table1.apply(
        lambda x: format_metric(x['test_auc'], x['test_auc_std']), axis=1)
    display_table1['F1'] = display_table1.apply(
        lambda x: format_metric(x['test_f1'], x['test_f1_std']), axis=1)
    display_table1['MCC'] = display_table1.apply(
        lambda x: format_metric(x['test_mcc'], x['test_mcc_std']), axis=1)
    display_table1['Precision'] = display_table1.apply(
        lambda x: format_metric(x['test_precision'], x['test_precision_std']), axis=1)
    display_table1['Recall'] = display_table1.apply(
        lambda x: format_metric(x['test_recall'], x['test_recall_std']), axis=1)
    
    table1_output = display_table1[['method_display', 'learning_rate', 'batch_size', 
                                    'Accuracy', 'AUC', 'F1', 'MCC', 'Precision', 'Recall']]
    table1_output.columns = ['Method', 'LR', 'Batch Size', 'Accuracy', 'AUC', 'F1', 'MCC', 'Precision', 'Recall']
    
    print("\n" + table1_output.to_string(index=False))
    
    # Save
    table1_output.to_csv(os.path.join(OUTPUT_DIR, 'Table1_Best_Per_Method_Updated.csv'), index=False)
    
    # =============================================
    # 5. Table 2: Average Performance Per Method (with all metrics)
    # =============================================
    print("\n" + "=" * 80)
    print("TABLE 2: AVERAGE PERFORMANCE PER METHOD (with F1, MCC, Precision, Recall)")
    print("=" * 80)
    
    avg_per_method = all_metrics.groupby('method_display').agg({
        'test_accuracy': ['mean', 'std'],
        'test_auc': ['mean', 'std'],
        'test_f1': ['mean', 'std'],
        'test_mcc': ['mean', 'std'],
        'test_precision': ['mean', 'std'],
        'test_recall': ['mean', 'std']
    }).round(4)
    
    avg_per_method.columns = ['_'.join(col).strip() for col in avg_per_method.columns.values]
    avg_per_method = avg_per_method.sort_values('test_auc_mean', ascending=False)
    
    # Format for display
    display_table2 = avg_per_method.copy()
    display_table2['Accuracy'] = display_table2.apply(
        lambda x: format_metric(x['test_accuracy_mean'], x['test_accuracy_std']), axis=1)
    display_table2['AUC'] = display_table2.apply(
        lambda x: format_metric(x['test_auc_mean'], x['test_auc_std']), axis=1)
    display_table2['F1'] = display_table2.apply(
        lambda x: format_metric(x['test_f1_mean'], x['test_f1_std']), axis=1)
    display_table2['MCC'] = display_table2.apply(
        lambda x: format_metric(x['test_mcc_mean'], x['test_mcc_std']), axis=1)
    display_table2['Precision'] = display_table2.apply(
        lambda x: format_metric(x['test_precision_mean'], x['test_precision_std']), axis=1)
    display_table2['Recall'] = display_table2.apply(
        lambda x: format_metric(x['test_recall_mean'], x['test_recall_std']), axis=1)
    
    table2_output = display_table2[['Accuracy', 'AUC', 'F1', 'MCC', 'Precision', 'Recall']]
    table2_output.index.name = 'Method'
    
    print("\n" + table2_output.to_string())
    
    # Save
    table2_output.to_csv(os.path.join(OUTPUT_DIR, 'Table2_Avg_Per_Method_Updated.csv'))
    
    # =============================================
    # 6. Table 3: Paper-Ready Summary Table
    # =============================================
    print("\n" + "=" * 80)
    print("TABLE 3: PAPER-READY SUMMARY TABLE (with F1, MCC, Precision, Recall)")
    print("=" * 80)
    
    paper_table = best_per_method[['method_display', 'learning_rate', 'batch_size',
                                   'test_accuracy', 'test_accuracy_std',
                                   'test_auc', 'test_auc_std',
                                   'test_f1', 'test_f1_std',
                                   'test_mcc', 'test_mcc_std',
                                   'test_precision', 'test_precision_std',
                                   'test_recall', 'test_recall_std']].copy()
    
    paper_table['Accuracy'] = paper_table.apply(
        lambda x: format_metric(x['test_accuracy'], x['test_accuracy_std']), axis=1)
    paper_table['AUC'] = paper_table.apply(
        lambda x: format_metric(x['test_auc'], x['test_auc_std']), axis=1)
    paper_table['F1'] = paper_table.apply(
        lambda x: format_metric(x['test_f1'], x['test_f1_std']), axis=1)
    paper_table['MCC'] = paper_table.apply(
        lambda x: format_metric(x['test_mcc'], x['test_mcc_std']), axis=1)
    paper_table['Precision'] = paper_table.apply(
        lambda x: format_metric(x['test_precision'], x['test_precision_std']), axis=1)
    paper_table['Recall'] = paper_table.apply(
        lambda x: format_metric(x['test_recall'], x['test_recall_std']), axis=1)
    
    paper_summary = paper_table[['method_display', 'learning_rate', 'batch_size', 
                                 'Accuracy', 'AUC', 'F1', 'MCC', 'Precision', 'Recall']]
    paper_summary.columns = ['Method', 'LR', 'Batch Size', 'Accuracy', 'AUC', 'F1', 'MCC', 'Precision', 'Recall']
    
    # Reorder methods for paper
    method_order = [
        'Attention-Enhanced DNN',
        'DNN Baseline',
        'Logistic Regression',
        'Ablation: No Attention',
        'Ablation: No Residual',
        'Ablation: 50% Data',
        'Random Forest',
        'SVM',
        'MLP-256',
        'MLP-512'
    ]
    paper_summary['Method_Order'] = paper_summary['Method'].map({m: i for i, m in enumerate(method_order)})
    paper_summary = paper_summary.sort_values('Method_Order').drop('Method_Order', axis=1)
    
    print("\n" + paper_summary.to_string(index=False))
    
    # Save
    paper_summary.to_csv(os.path.join(OUTPUT_DIR, 'Table3_Paper_Summary_Updated.csv'), index=False)
    
    # =============================================
    # 7. Table 4: Ablation Study with Precision and Recall
    # =============================================
    print("\n" + "=" * 80)
    print("TABLE 4: ABLATION STUDY (with Precision and Recall)")
    print("=" * 80)
    
    ablation_methods = [
        'results_Attention_Basic',
        'ablation_no_attention',
        'ablation_no_residual',
        'ablation_50percent_data'
    ]
    
    ablation_data = all_metrics[all_metrics['method'].isin(ablation_methods)]
    
    ablation_best = ablation_data.loc[ablation_data.groupby('method')['test_auc'].idxmax()]
    
    ablation_table = ablation_best[['method_display', 'learning_rate', 'batch_size',
                                    'test_accuracy', 'test_accuracy_std',
                                    'test_auc', 'test_auc_std',
                                    'test_f1', 'test_f1_std',
                                    'test_mcc', 'test_mcc_std',
                                    'test_precision', 'test_precision_std',
                                    'test_recall', 'test_recall_std']].copy()
    
    ablation_table['Accuracy'] = ablation_table.apply(
        lambda x: format_metric(x['test_accuracy'], x['test_accuracy_std']), axis=1)
    ablation_table['AUC'] = ablation_table.apply(
        lambda x: format_metric(x['test_auc'], x['test_auc_std']), axis=1)
    ablation_table['F1'] = ablation_table.apply(
        lambda x: format_metric(x['test_f1'], x['test_f1_std']), axis=1)
    ablation_table['MCC'] = ablation_table.apply(
        lambda x: format_metric(x['test_mcc'], x['test_mcc_std']), axis=1)
    ablation_table['Precision'] = ablation_table.apply(
        lambda x: format_metric(x['test_precision'], x['test_precision_std']), axis=1)
    ablation_table['Recall'] = ablation_table.apply(
        lambda x: format_metric(x['test_recall'], x['test_recall_std']), axis=1)
    
    ablation_output = ablation_table[['method_display', 'learning_rate', 'batch_size',
                                      'Accuracy', 'AUC', 'F1', 'MCC', 'Precision', 'Recall']]
    ablation_output.columns = ['Method', 'LR', 'Batch Size', 'Accuracy', 'AUC', 'F1', 'MCC', 'Precision', 'Recall']
    
    print("\n" + ablation_output.to_string(index=False))
    
    # Save
    ablation_output.to_csv(os.path.join(OUTPUT_DIR, 'Table4_Ablation_Study_Updated.csv'), index=False)
    
    # =============================================
    # 8. Best Overall
    # =============================================
    print("\n" + "=" * 80)
    print("🏆 BEST OVERALL PERFORMER")
    print("=" * 80)
    
    best_overall = all_metrics.loc[all_metrics['test_auc'].idxmax()]
    
    print(f"   Method: {best_overall['method_display']}")
    print(f"   Learning Rate: {best_overall['learning_rate']}")
    print(f"   Batch Size: {best_overall['batch_size']}")
    print(f"   Test Accuracy: {format_metric(best_overall['test_accuracy'], best_overall['test_accuracy_std'])}")
    print(f"   Test AUC: {format_metric(best_overall['test_auc'], best_overall['test_auc_std'])}")
    print(f"   Test F1: {format_metric(best_overall['test_f1'], best_overall['test_f1_std'])}")
    print(f"   Test MCC: {format_metric(best_overall['test_mcc'], best_overall['test_mcc_std'])}")
    print(f"   Test Precision: {format_metric(best_overall['test_precision'], best_overall['test_precision_std'])}")
    print(f"   Test Recall: {format_metric(best_overall['test_recall'], best_overall['test_recall_std'])}")
    
    best_overall_df = pd.DataFrame([{
        'Method': best_overall['method_display'],
        'LR': best_overall['learning_rate'],
        'Batch Size': best_overall['batch_size'],
        'Accuracy': format_metric(best_overall['test_accuracy'], best_overall['test_accuracy_std']),
        'AUC': format_metric(best_overall['test_auc'], best_overall['test_auc_std']),
        'F1': format_metric(best_overall['test_f1'], best_overall['test_f1_std']),
        'MCC': format_metric(best_overall['test_mcc'], best_overall['test_mcc_std']),
        'Precision': format_metric(best_overall['test_precision'], best_overall['test_precision_std']),
        'Recall': format_metric(best_overall['test_recall'], best_overall['test_recall_std'])
    }])
    best_overall_df.to_csv(os.path.join(OUTPUT_DIR, 'Best_Overall_Updated.csv'), index=False)
    
    # =============================================
    # 9. Summary by Model Type
    # =============================================
    print("\n" + "=" * 80)
    print("TABLE 5: SUMMARY BY MODEL TYPE (with Precision and Recall)")
    print("=" * 80)
    
    attention_methods = ['results_Attention_Basic', 'results_DNN_Baseline', 'results_Logistic_Baseline',
                         'ablation_no_attention', 'ablation_no_residual', 'ablation_50percent_data']
    
    def categorize_model(method):
        if method in attention_methods:
            return 'Attention Models'
        else:
            return 'Baseline Models'
    
    all_metrics['model_type'] = all_metrics['method'].apply(categorize_model)
    
    summary_by_type = all_metrics.groupby('model_type').agg({
        'test_accuracy': ['mean', 'std'],
        'test_auc': ['mean', 'std'],
        'test_f1': ['mean', 'std'],
        'test_mcc': ['mean', 'std'],
        'test_precision': ['mean', 'std'],
        'test_recall': ['mean', 'std']
    }).round(4)
    
    summary_by_type.columns = ['_'.join(col).strip() for col in summary_by_type.columns.values]
    
    display_summary = summary_by_type.copy()
    display_summary['Accuracy'] = display_summary.apply(
        lambda x: format_metric(x['test_accuracy_mean'], x['test_accuracy_std']), axis=1)
    display_summary['AUC'] = display_summary.apply(
        lambda x: format_metric(x['test_auc_mean'], x['test_auc_std']), axis=1)
    display_summary['F1'] = display_summary.apply(
        lambda x: format_metric(x['test_f1_mean'], x['test_f1_std']), axis=1)
    display_summary['MCC'] = display_summary.apply(
        lambda x: format_metric(x['test_mcc_mean'], x['test_mcc_std']), axis=1)
    display_summary['Precision'] = display_summary.apply(
        lambda x: format_metric(x['test_precision_mean'], x['test_precision_std']), axis=1)
    display_summary['Recall'] = display_summary.apply(
        lambda x: format_metric(x['test_recall_mean'], x['test_recall_std']), axis=1)
    
    print("\n" + display_summary[['Accuracy', 'AUC', 'F1', 'MCC', 'Precision', 'Recall']].to_string())
    
    summary_by_type.to_csv(os.path.join(OUTPUT_DIR, 'Table5_Summary_By_Type_Updated.csv'), index=False)
    
    # =============================================
    # 10. Save Full Results
    # =============================================
    all_metrics.to_csv(os.path.join(OUTPUT_DIR, 'all_results_with_all_metrics.csv'), index=False)
    
    print("\n" + "=" * 80)
    print("✅ ALL TABLES UPDATED WITH F1, MCC, PRECISION, AND RECALL")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nFiles generated:")
    print(f"  - Table1_Best_Per_Method_Updated.csv")
    print(f"  - Table2_Avg_Per_Method_Updated.csv")
    print(f"  - Table3_Paper_Summary_Updated.csv")
    print(f"  - Table4_Ablation_Study_Updated.csv")
    print(f"  - Table5_Summary_By_Type_Updated.csv")
    print(f"  - Best_Overall_Updated.csv")
    print(f"  - all_results_with_all_metrics.csv")
    print("=" * 80)

if __name__ == "__main__":
    main()