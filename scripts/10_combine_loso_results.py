# -*- coding: utf-8 -*-
"""
09_combine_loso_results.py
Combine ALL LOSO results (Attention Models + Baselines)
Generates summary tables with F1 and MCC for all species
UPDATED: Extracts F1 and MCC from fold_metrics.npy for ALL models

Input: D:/uni_prot2/revision/results/plant_loso_complete/ (ALL models)
Output: D:/uni_prot2/revision/results/combined_loso_results/
"""

import pandas as pd
import numpy as np
import os
import glob
import json
import pickle
from datetime import datetime

# =============================================
# CONFIGURATION
# =============================================

# Path to ALL LOSO results (plant_loso_complete has all models)
LOSO_BASE_DIR = r"D:\uni_prot2\revision\results\plant_loso_complete"

# Output directory for combined LOSO results
OUTPUT_DIR = r"D:\uni_prot2\revision\results\combined_loso_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Species list
SPECIES_LIST = ['Arabidopsis_thaliana', 'Brassica_spp', 'Oryza_sativa', 'Triticum_aestivum']

# Species display names
SPECIES_NAMES = {
    'Arabidopsis_thaliana': 'Arabidopsis thaliana',
    'Brassica_spp': 'Brassica spp.',
    'Oryza_sativa': 'Oryza sativa',
    'Triticum_aestivum': 'Triticum aestivum'
}

# Method display names
METHOD_DISPLAY_NAMES = {
    'Attention_Enhanced_Basic': 'Attention-Enhanced DNN',
    'DNN_Baseline': 'DNN Baseline',
    'Logistic_Baseline': 'Logistic Regression',
    'Ablation_No_Attention': 'Ablation: No Attention',
    'Ablation_No_Residual': 'Ablation: No Residual',
    'Ablation_50Percent_Data': 'Ablation: 50% Data',
    'Random_Forest': 'Random Forest',
    'SVM': 'SVM',
    'MLP_256': 'MLP-256',
    'MLP_512': 'MLP-512'
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

# Method order for paper
METHOD_ORDER = [
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

# =============================================
# FUNCTION TO EXTRACT FOLD METRICS FROM NPY
# =============================================

def extract_fold_metrics_from_npy(species, method, lr, bs):
    """Extract fold metrics from fold_metrics.npy file"""
    
    lr_str = f"{lr:.4f}".replace('.', '_')
    folder_name = METHOD_FOLDER_MAP.get(method, method)
    
    # Construct path
    npy_path = os.path.join(LOSO_BASE_DIR, species, f"lr_{lr_str}_bs_{bs}", folder_name, "npy_files", "fold_metrics.npy")
    
    if os.path.exists(npy_path):
        try:
            fold_metrics = np.load(npy_path, allow_pickle=True).item()
            
            result = {}
            
            # Extract test metrics
            if 'test_fixed' in fold_metrics:
                test_metrics = fold_metrics['test_fixed']
                
                # Extract all metrics
                result['test_accuracy'] = np.mean(test_metrics.get('accuracy', [np.nan]))
                result['test_accuracy_std'] = np.std(test_metrics.get('accuracy', [np.nan]))
                result['test_auc'] = np.mean(test_metrics.get('auc', [np.nan]))
                result['test_auc_std'] = np.std(test_metrics.get('auc', [np.nan]))
                result['test_f1'] = np.mean(test_metrics.get('f1', [np.nan]))
                result['test_f1_std'] = np.std(test_metrics.get('f1', [np.nan]))
                result['test_mcc'] = np.mean(test_metrics.get('mcc', [np.nan]))
                result['test_mcc_std'] = np.std(test_metrics.get('mcc', [np.nan]))
                result['test_precision'] = np.mean(test_metrics.get('precision', [np.nan]))
                result['test_recall'] = np.mean(test_metrics.get('recall', [np.nan]))
            
            # Extract training time
            if 'training_time' in fold_metrics:
                result['training_time'] = np.mean(fold_metrics['training_time'])
            
            return result
            
        except Exception as e:
            pass
    
    return None

# =============================================
# FUNCTION TO GET BEST CONFIG FOR METHOD
# =============================================

def get_best_config(species, method, metric='test_auc'):
    """Get best configuration for a method based on metric"""
    
    best_result = None
    best_value = -np.inf
    
    for lr in [0.01, 0.001, 0.0001]:
        for bs in [32, 64, 128, 256]:
            result = extract_fold_metrics_from_npy(species, method, lr, bs)
            if result and metric in result:
                val = result[metric]
                if not np.isnan(val) and val > best_value:
                    best_value = val
                    best_result = {
                        'learning_rate': lr,
                        'batch_size': bs,
                        'metrics': result
                    }
    
    return best_result

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
    print("COMBINE ALL LOSO RESULTS (FROM plant_loso_complete)")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    if not os.path.exists(LOSO_BASE_DIR):
        print(f"❌ Directory not found: {LOSO_BASE_DIR}")
        return
    
    print(f"\n📁 Base directory: {LOSO_BASE_DIR}")
    
    # =============================================
    # 1. Load ALL LOSO Results
    # =============================================
    print("\n📊 Loading LOSO results from NPY files...")
    
    all_results = []
    
    for species in SPECIES_LIST:
        print(f"\n  📊 Species: {species}")
        species_path = os.path.join(LOSO_BASE_DIR, species)
        
        if not os.path.exists(species_path):
            print(f"    ⚠️ Species folder not found: {species}")
            continue
        
        # Get all methods available
        method_list = list(METHOD_FOLDER_MAP.keys())
        
        for method in method_list:
            display_name = METHOD_DISPLAY_NAMES.get(method, method)
            
            # Get best config
            best = get_best_config(species, method)
            
            if best:
                metrics = best['metrics']
                
                result = {
                    'species': species,
                    'species_display': SPECIES_NAMES.get(species, species),
                    'method': method,
                    'method_display': display_name,
                    'learning_rate': best['learning_rate'],
                    'batch_size': best['batch_size'],
                    'test_accuracy': metrics.get('test_accuracy', np.nan),
                    'test_accuracy_std': metrics.get('test_accuracy_std', np.nan),
                    'test_auc': metrics.get('test_auc', np.nan),
                    'test_auc_std': metrics.get('test_auc_std', np.nan),
                    'test_f1': metrics.get('test_f1', np.nan),
                    'test_f1_std': metrics.get('test_f1_std', np.nan),
                    'test_mcc': metrics.get('test_mcc', np.nan),
                    'test_mcc_std': metrics.get('test_mcc_std', np.nan),
                    'test_precision': metrics.get('test_precision', np.nan),
                    'test_recall': metrics.get('test_recall', np.nan),
                    'training_time': metrics.get('training_time', np.nan)
                }
                all_results.append(result)
                print(f"    ✅ {display_name}: AUC={result['test_auc']:.4f}")
            else:
                print(f"    ❌ {display_name}: No data found")
    
    if not all_results:
        print("\n❌ No results found!")
        return
    
    # Convert to DataFrame
    all_loso = pd.DataFrame(all_results)
    print(f"\n✅ Total LOSO configurations loaded: {len(all_loso)}")
    
    # Drop rows with missing AUC
    all_loso = all_loso[all_loso['test_auc'].notna()]
    print(f"   After removing missing AUC values: {len(all_loso)}")
    
    # =============================================
    # 2. Table 1: Best Performance Per Species
    # =============================================
    print("\n" + "=" * 80)
    print("TABLE 1: BEST PERFORMANCE PER SPECIES (LOSO)")
    print("=" * 80)
    
    best_per_species = all_loso.loc[all_loso.groupby('species')['test_auc'].idxmax()]
    best_per_species = best_per_species[['species_display', 'method_display', 'learning_rate', 'batch_size',
                                         'test_accuracy', 'test_accuracy_std',
                                         'test_auc', 'test_auc_std',
                                         'test_f1', 'test_f1_std',
                                         'test_mcc', 'test_mcc_std',
                                         'test_precision', 'test_recall']]
    best_per_species = best_per_species.sort_values('test_auc', ascending=False)
    
    # Format for display
    display_table1 = best_per_species.copy()
    display_table1['Accuracy'] = display_table1.apply(
        lambda x: format_metric(x['test_accuracy'], x['test_accuracy_std']), axis=1)
    display_table1['AUC'] = display_table1.apply(
        lambda x: format_metric(x['test_auc'], x['test_auc_std']), axis=1)
    display_table1['F1'] = display_table1.apply(
        lambda x: format_metric(x['test_f1'], x['test_f1_std']), axis=1)
    display_table1['MCC'] = display_table1.apply(
        lambda x: format_metric(x['test_mcc'], x['test_mcc_std']), axis=1)
    
    table1_output = display_table1[['species_display', 'method_display', 
                                    'Accuracy', 'AUC', 'F1', 'MCC']]
    table1_output.columns = ['Species', 'Best Method', 'Accuracy', 'AUC', 'F1', 'MCC']
    
    print(table1_output.to_string(index=False))
    
    # Save
    table1_output.to_csv(os.path.join(OUTPUT_DIR, 'Table1_LOSO_Best_Per_Species.csv'), index=False)
    
    # =============================================
    # 3. Table 2: Average Performance Per Method
    # =============================================
    print("\n" + "=" * 80)
    print("TABLE 2: AVERAGE PERFORMANCE PER METHOD (Across 4 Species)")
    print("=" * 80)
    
    avg_per_method = all_loso.groupby('method_display').agg({
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
    
    table2_output = display_table2[['Accuracy', 'AUC', 'F1', 'MCC']]
    table2_output.index.name = 'Method'
    
    print(table2_output.to_string())
    
    # Save
    table2_output.to_csv(os.path.join(OUTPUT_DIR, 'Table2_LOSO_Avg_Per_Method.csv'))
    
    # =============================================
    # 4. Table 3: Paper-Ready Summary Table
    # =============================================
    print("\n" + "=" * 80)
    print("TABLE 3: PAPER-READY LOSO SUMMARY TABLE")
    print("=" * 80)
    
    # Get best config per method
    best_per_method = all_loso.loc[all_loso.groupby('method')['test_auc'].idxmax()]
    best_per_method = best_per_method[['method_display', 'learning_rate', 'batch_size',
                                       'test_accuracy', 'test_accuracy_std',
                                       'test_auc', 'test_auc_std',
                                       'test_f1', 'test_f1_std',
                                       'test_mcc', 'test_mcc_std']]
    best_per_method = best_per_method.sort_values('test_auc', ascending=False)
    
    paper_table = best_per_method.copy()
    paper_table['Accuracy'] = paper_table.apply(
        lambda x: format_metric(x['test_accuracy'], x['test_accuracy_std']), axis=1)
    paper_table['AUC'] = paper_table.apply(
        lambda x: format_metric(x['test_auc'], x['test_auc_std']), axis=1)
    paper_table['F1'] = paper_table.apply(
        lambda x: format_metric(x['test_f1'], x['test_f1_std']), axis=1)
    paper_table['MCC'] = paper_table.apply(
        lambda x: format_metric(x['test_mcc'], x['test_mcc_std']), axis=1)
    
    paper_summary = paper_table[['method_display', 'learning_rate', 'batch_size', 
                                 'Accuracy', 'AUC', 'F1', 'MCC']]
    paper_summary.columns = ['Method', 'LR', 'Batch Size', 'Accuracy', 'AUC', 'F1', 'MCC']
    
    # Reorder methods
    paper_summary['Method_Order'] = paper_summary['Method'].map({m: i for i, m in enumerate(METHOD_ORDER)})
    paper_summary = paper_summary.sort_values('Method_Order').drop('Method_Order', axis=1)
    
    print(paper_summary.to_string(index=False))
    
    # Save
    paper_summary.to_csv(os.path.join(OUTPUT_DIR, 'Table3_LOSO_Paper_Summary.csv'), index=False)
    
    # =============================================
    # 5. Table 4: Average LOSO Across All Species
    # =============================================
    print("\n" + "=" * 80)
    print("TABLE 4: AVERAGE LOSO PERFORMANCE (Across All Species)")
    print("=" * 80)
    
    avg_all = all_loso.groupby('method_display').agg({
        'test_accuracy': ['mean', 'std'],
        'test_auc': ['mean', 'std'],
        'test_f1': ['mean', 'std'],
        'test_mcc': ['mean', 'std']
    }).round(4)
    
    avg_all.columns = ['_'.join(col).strip() for col in avg_all.columns.values]
    
    display_avg = avg_all.copy()
    display_avg['Accuracy'] = display_avg.apply(
        lambda x: format_metric(x['test_accuracy_mean'], x['test_accuracy_std']), axis=1)
    display_avg['AUC'] = display_avg.apply(
        lambda x: format_metric(x['test_auc_mean'], x['test_auc_std']), axis=1)
    display_avg['F1'] = display_avg.apply(
        lambda x: format_metric(x['test_f1_mean'], x['test_f1_std']), axis=1)
    display_avg['MCC'] = display_avg.apply(
        lambda x: format_metric(x['test_mcc_mean'], x['test_mcc_std']), axis=1)
    
    table4_output = display_avg[['Accuracy', 'AUC', 'F1', 'MCC']]
    table4_output.index.name = 'Method'
    
    print(table4_output.to_string())
    
    # Save
    table4_output.to_csv(os.path.join(OUTPUT_DIR, 'Table4_LOSO_Average_All_Species.csv'))
    
    # =============================================
    # 6. Best Overall LOSO Performer
    # =============================================
    print("\n" + "=" * 80)
    print("🏆 BEST OVERALL LOSO PERFORMER")
    print("=" * 80)
    
    best_overall = all_loso.loc[all_loso['test_auc'].idxmax()]
    
    print(f"   Species: {best_overall['species_display']}")
    print(f"   Method: {best_overall['method_display']}")
    print(f"   Learning Rate: {best_overall['learning_rate']}")
    print(f"   Batch Size: {best_overall['batch_size']}")
    print(f"   Test Accuracy: {format_metric(best_overall['test_accuracy'], best_overall['test_accuracy_std'])}")
    print(f"   Test AUC: {format_metric(best_overall['test_auc'], best_overall['test_auc_std'])}")
    print(f"   Test F1: {format_metric(best_overall['test_f1'], best_overall['test_f1_std'])}")
    print(f"   Test MCC: {format_metric(best_overall['test_mcc'], best_overall['test_mcc_std'])}")
    
    # Save
    best_overall_df = pd.DataFrame([{
        'Species': best_overall['species_display'],
        'Method': best_overall['method_display'],
        'LR': best_overall['learning_rate'],
        'Batch Size': best_overall['batch_size'],
        'Accuracy': format_metric(best_overall['test_accuracy'], best_overall['test_accuracy_std']),
        'AUC': format_metric(best_overall['test_auc'], best_overall['test_auc_std']),
        'F1': format_metric(best_overall['test_f1'], best_overall['test_f1_std']),
        'MCC': format_metric(best_overall['test_mcc'], best_overall['test_mcc_std'])
    }])
    best_overall_df.to_csv(os.path.join(OUTPUT_DIR, 'Best_LOSO_Overall.csv'), index=False)
    
    # =============================================
    # 7. Summary by Model Type
    # =============================================
    print("\n" + "=" * 80)
    print("TABLE 5: SUMMARY BY MODEL TYPE (LOSO)")
    print("=" * 80)
    
    attention_methods = ['Attention_Enhanced_Basic', 'DNN_Baseline', 'Logistic_Baseline',
                         'Ablation_No_Attention', 'Ablation_No_Residual', 'Ablation_50Percent_Data']
    
    def categorize_model(method):
        if method in attention_methods:
            return 'Attention Models'
        else:
            return 'Baseline Models'
    
    all_loso['model_type'] = all_loso['method'].apply(categorize_model)
    
    summary_by_type = all_loso.groupby('model_type').agg({
        'test_accuracy': ['mean', 'std'],
        'test_auc': ['mean', 'std'],
        'test_f1': ['mean', 'std'],
        'test_mcc': ['mean', 'std']
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
    
    print(display_summary[['Accuracy', 'AUC', 'F1', 'MCC']].to_string())
    
    summary_by_type.to_csv(os.path.join(OUTPUT_DIR, 'Table5_LOSO_Summary_By_Type.csv'))
    
    # =============================================
    # 8. Save Full Results
    # =============================================
    all_loso.to_csv(os.path.join(OUTPUT_DIR, 'all_loso_results_detailed.csv'), index=False)
    
    print("\n" + "=" * 80)
    print("✅ ALL LOSO TABLES GENERATED")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nFiles generated:")
    print(f"  - Table1_LOSO_Best_Per_Species.csv")
    print(f"  - Table2_LOSO_Avg_Per_Method.csv")
    print(f"  - Table3_LOSO_Paper_Summary.csv")
    print(f"  - Table4_LOSO_Average_All_Species.csv")
    print(f"  - Table5_LOSO_Summary_By_Type.csv")
    print(f"  - Best_LOSO_Overall.csv")
    print(f"  - all_loso_results_detailed.csv")
    print("=" * 80)

if __name__ == "__main__":
    main()