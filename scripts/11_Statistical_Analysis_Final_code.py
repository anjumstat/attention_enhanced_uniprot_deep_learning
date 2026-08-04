# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:25:47 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Statistical Analysis: Friedman & Repeated Measures ANOVA
PER LEARNING RATE AND BATCH SIZE COMBINATION
Based on ACTUAL 10-Fold CV Values from NPY Files
UPDATED: Includes ALL models (Attention + Baselines) for CD-HIT
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon, shapiro
from statsmodels.stats.anova import AnovaRM
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# =============================================
# CONFIGURATION
# =============================================

# CD-HIT directories (Attention + Baselines)
CDHIT_ATTENTION_DIR = "D:/uni_prot2/revision/results/homology_aware_with_test"
CDHIT_BASELINE_DIR = "D:/uni_prot2/revision/results/homology_aware_baselines_with_checkpoint"

# LOSO directory
LOSO_BASE_DIR = "D:/uni_prot2/revision/results/plant_loso_complete"

OUTPUT_DIR = "D:/uni_prot2/revision/results/combined_results_CD_HIT/Statistical_Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# All methods
ATTENTION_METHODS = [
    'results_Attention_Basic',
    'results_DNN_Baseline',
    'results_Logistic_Baseline',
    'ablation_no_attention',
    'ablation_no_residual',
    'ablation_50percent_data'
]

BASELINE_METHODS = [
    'results_Random_Forest',
    'results_SVM',
    'results_MLP_256',
    'results_MLP_512'
]

ALL_METHODS = ATTENTION_METHODS + BASELINE_METHODS

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

# Method folder mapping for CD-HIT Attention
ATTENTION_FOLDER_MAP = {
    'results_Attention_Basic': 'results_Attention_Basic',
    'results_DNN_Baseline': 'results_DNN_Baseline',
    'results_Logistic_Baseline': 'results_Logistic_Baseline',
    'ablation_no_attention': 'ablation_no_attention',
    'ablation_no_residual': 'ablation_no_residual',
    'ablation_50percent_data': 'ablation_50percent_data'
}

# Method folder mapping for CD-HIT Baselines
BASELINE_FOLDER_MAP = {
    'results_Random_Forest': 'results_Random_Forest',
    'results_SVM': 'results_SVM',
    'results_MLP_256': 'results_MLP_256',
    'results_MLP_512': 'results_MLP_512'
}

# Method folder mapping for LOSO
LOSO_FOLDER_MAP = {
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

# LOSO method names (different from CD-HIT)
LOSO_METHODS = [
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

LOSO_METHOD_DISPLAY = {
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

LEARNING_RATES = [0.01, 0.001, 0.0001]
BATCH_SIZES = [32, 64, 128, 256]
SPECIES_LIST = ['Arabidopsis_thaliana', 'Brassica_spp', 'Oryza_sativa', 'Triticum_aestivum']

# =============================================
# FUNCTION: Extract 10-Fold CV Values from NPY (CD-HIT)
# =============================================

def extract_cdhit_fold_values(method, lr, bs):
    """Extract 10-fold CV values from CD-HIT NPY files (Attention + Baselines)"""
    
    lr_str = f"{lr:.4f}".replace('.', '_')
    
    # Determine which directory and folder map to use
    if method in ATTENTION_METHODS:
        base_dir = CDHIT_ATTENTION_DIR
        folder_map = ATTENTION_FOLDER_MAP
    elif method in BASELINE_METHODS:
        base_dir = CDHIT_BASELINE_DIR
        folder_map = BASELINE_FOLDER_MAP
    else:
        return None
    
    folder_name = folder_map.get(method, method)
    
    # Construct path
    npy_path = os.path.join(base_dir, f"lr_{lr_str}_bs_{bs}", folder_name, "npy_files", "fold_metrics.npy")
    
    if os.path.exists(npy_path):
        try:
            fold_metrics = np.load(npy_path, allow_pickle=True).item()
            
            # Extract test metrics (10 folds)
            if 'test_fixed' in fold_metrics:
                test_metrics = fold_metrics['test_fixed']
                
                # Get AUC values for all 10 folds
                auc_values = test_metrics.get('auc', None)
                if auc_values is not None and len(auc_values) == 10:
                    return np.array(auc_values)
                
                # If AUC not available, try accuracy
                acc_values = test_metrics.get('accuracy', None)
                if acc_values is not None and len(acc_values) == 10:
                    return np.array(acc_values)
            
            # Try 'test' key if 'test_fixed' not available
            if 'test' in fold_metrics:
                test_metrics = fold_metrics['test']
                auc_values = test_metrics.get('auc', None)
                if auc_values is not None and len(auc_values) == 10:
                    return np.array(auc_values)
                
                acc_values = test_metrics.get('accuracy', None)
                if acc_values is not None and len(acc_values) == 10:
                    return np.array(acc_values)
                    
        except Exception as e:
            pass
    
    return None

# =============================================
# FUNCTION: Extract 10-Fold CV Values from NPY (LOSO)
# =============================================

def extract_loso_fold_values(species, method, lr, bs):
    """Extract 10-fold CV values from LOSO NPY files"""
    
    lr_str = f"{lr:.4f}".replace('.', '_')
    folder_name = LOSO_FOLDER_MAP.get(method, method)
    
    # Construct path
    npy_path = os.path.join(LOSO_BASE_DIR, species, f"lr_{lr_str}_bs_{bs}", folder_name, "npy_files", "fold_metrics.npy")
    
    if os.path.exists(npy_path):
        try:
            fold_metrics = np.load(npy_path, allow_pickle=True).item()
            
            # Extract test metrics (10 folds)
            if 'test_fixed' in fold_metrics:
                test_metrics = fold_metrics['test_fixed']
                
                # Get AUC values for all 10 folds
                auc_values = test_metrics.get('auc', None)
                if auc_values is not None and len(auc_values) == 10:
                    return np.array(auc_values)
                
                # If AUC not available, try accuracy
                acc_values = test_metrics.get('accuracy', None)
                if acc_values is not None and len(acc_values) == 10:
                    return np.array(acc_values)
            
            # Try 'test' key if 'test_fixed' not available
            if 'test' in fold_metrics:
                test_metrics = fold_metrics['test']
                auc_values = test_metrics.get('auc', None)
                if auc_values is not None and len(auc_values) == 10:
                    return np.array(auc_values)
                
                acc_values = test_metrics.get('accuracy', None)
                if acc_values is not None and len(acc_values) == 10:
                    return np.array(acc_values)
                    
        except Exception as e:
            pass
    
    return None

# =============================================
# FUNCTION: Get 10-Fold Values for All CD-HIT Methods
# =============================================

def get_cdhit_fold_values(lr, bs):
    """Get 10-fold CV values for ALL CD-HIT methods for a specific LR and BS"""
    
    fold_data = {}
    
    for method in ALL_METHODS:
        fold_vals = extract_cdhit_fold_values(method, lr, bs)
        
        if fold_vals is not None:
            fold_data[method] = fold_vals
    
    return fold_data

# =============================================
# FUNCTION: Get 10-Fold Values for All LOSO Methods
# =============================================

def get_loso_fold_values(species, lr, bs):
    """Get 10-fold CV values for ALL LOSO methods for a specific species, LR and BS"""
    
    fold_data = {}
    
    for method in LOSO_METHODS:
        fold_vals = extract_loso_fold_values(species, method, lr, bs)
        
        if fold_vals is not None:
            fold_data[method] = fold_vals
    
    return fold_data

# =============================================
# FUNCTION: Check Assumptions
# =============================================

def check_assumptions(data_matrix, methods):
    """Check normality and sphericity assumptions"""
    
    n_folds = data_matrix.shape[1]
    n_methods = data_matrix.shape[0]
    
    # 1. Normality (Shapiro-Wilk on residuals)
    residuals = []
    for i in range(n_methods):
        row_mean = np.mean(data_matrix[i])
        residuals.extend(data_matrix[i] - row_mean)
    
    shapiro_stat, shapiro_p = shapiro(residuals)
    normality_passed = shapiro_p > 0.05
    
    # 2. Sphericity (variance ratio approach)
    diff_variances = []
    for i in range(n_methods):
        for j in range(i+1, n_methods):
            diff = data_matrix[i] - data_matrix[j]
            diff_variances.append(np.var(diff))
    
    if diff_variances:
        variance_ratio = np.max(diff_variances) / np.min(diff_variances) if np.min(diff_variances) > 0 else 100
        sphericity_passed = variance_ratio < 3
    else:
        variance_ratio = float('inf')
        sphericity_passed = False
    
    return {
        'normality_passed': normality_passed,
        'normality_p': shapiro_p,
        'sphericity_passed': sphericity_passed,
        'variance_ratio': variance_ratio,
        'assumptions_passed': normality_passed and sphericity_passed
    }

# =============================================
# FUNCTION: Perform Friedman Test
# =============================================

def perform_friedman(data_matrix, methods):
    """Perform Friedman Test"""
    
    try:
        stat, p_val = friedmanchisquare(*data_matrix)
        
        means = {m: np.mean(data_matrix[i]) for i, m in enumerate(methods)}
        best_method = max(means, key=means.get)
        best_mean = means[best_method]
        
        return {
            'friedman_stat': stat,
            'p_val': p_val,
            'significant': p_val < 0.05,
            'best_method': best_method,
            'best_mean': best_mean
        }
    except Exception as e:
        return None

# =============================================
# FUNCTION: Perform Repeated Measures ANOVA
# =============================================

def perform_anova(data_matrix, methods):
    """Perform Repeated Measures ANOVA"""
    
    n_folds = data_matrix.shape[1]
    n_methods = data_matrix.shape[0]
    
    data_long = []
    for i, method in enumerate(methods):
        for fold in range(n_folds):
            data_long.append({
                'Method': METHOD_DISPLAY.get(method, method) if method in METHOD_DISPLAY else method,
                'Fold': fold + 1,
                'Value': data_matrix[i, fold]
            })
    
    df_long = pd.DataFrame(data_long)
    
    try:
        model = AnovaRM(df_long, 'Value', 'Fold', within=['Method'])
        anova_results = model.fit()
        
        f_stat = anova_results.anova_table['F Value']['Method']
        p_val = anova_results.anova_table['Pr > F']['Method']
        df_num = anova_results.anova_table['num df']['Method']
        df_den = anova_results.anova_table['den df']['Method']
        
        return {
            'f_stat': f_stat,
            'df_num': df_num,
            'df_den': df_den,
            'p_val': p_val,
            'significant': p_val < 0.05
        }
    except Exception as e:
        return None

# =============================================
# MAIN ANALYSIS: CD-HIT
# =============================================

def analyze_cdhit():
    """Analyze CD-HIT data per LR and BS using NPY files (ALL models)"""
    
    print("\n" + "=" * 80)
    print("CD-HIT STATISTICAL ANALYSIS (Per LR and BS)")
    print("ALL Models (Attention + Baselines)")
    print("=" * 80)
    
    all_results = []
    
    for lr in LEARNING_RATES:
        for bs in BATCH_SIZES:
            print(f"\n📊 LR={lr}, BS={bs}")
            print("-" * 40)
            
            # Get fold metrics from NPY files for ALL methods
            fold_data = get_cdhit_fold_values(lr, bs)
            
            if len(fold_data) < 3:
                print(f"  ⚠️ Not enough methods (need >=3, have {len(fold_data)})")
                continue
            
            methods = list(fold_data.keys())
            data_matrix = np.array([fold_data[m] for m in methods])
            
            print(f"  Methods with data: {len(methods)}")
            for m in methods:
                print(f"    {METHOD_DISPLAY.get(m, m)}")
            
            # Check assumptions
            assumptions = check_assumptions(data_matrix, methods)
            
            print(f"  Normality: {'✅ PASSED' if assumptions['normality_passed'] else '❌ FAILED'} (p={assumptions['normality_p']:.4f})")
            print(f"  Sphericity: {'✅ PASSED' if assumptions['sphericity_passed'] else '❌ FAILED'}")
            print(f"  Assumptions Met: {'✅ YES' if assumptions['assumptions_passed'] else '❌ NO'}")
            
            # Choose test
            if assumptions['assumptions_passed']:
                print("  ✅ Using Repeated Measures ANOVA")
                result = perform_anova(data_matrix, methods)
                if result:
                    print(f"  F({result['df_num']:.1f}, {result['df_den']:.1f}) = {result['f_stat']:.3f}, p={result['p_val']:.6f} {'✅' if result['significant'] else '❌'}")
            else:
                print("  ❌ Using Friedman Test")
                result = perform_friedman(data_matrix, methods)
                if result:
                    print(f"  χ² = {result['friedman_stat']:.3f}, p={result['p_val']:.6f} {'✅' if result['significant'] else '❌'}")
                    print(f"  Best: {METHOD_DISPLAY.get(result['best_method'], result['best_method'])} ({result['best_mean']:.4f})")
            
            # Store results
            all_results.append({
                'LR': lr,
                'BS': bs,
                'assumptions': assumptions,
                'result': result,
                'methods': methods,
                'data_matrix': data_matrix
            })
    
    return all_results

# =============================================
# MAIN ANALYSIS: LOSO
# =============================================

def analyze_loso():
    """Analyze LOSO data per LR, BS, and Species using NPY files"""
    
    print("\n" + "=" * 80)
    print("LOSO STATISTICAL ANALYSIS (Per LR, BS, and Species)")
    print("=" * 80)
    
    all_results = []
    
    for species in SPECIES_LIST:
        print(f"\n📊 Species: {species}")
        print("=" * 50)
        
        for lr in LEARNING_RATES:
            for bs in BATCH_SIZES:
                print(f"\n  LR={lr}, BS={bs}")
                print("-" * 30)
                
                # Get fold metrics from NPY files
                fold_data = get_loso_fold_values(species, lr, bs)
                
                if len(fold_data) < 3:
                    print(f"    ⚠️ Not enough methods (need >=3, have {len(fold_data)})")
                    continue
                
                methods = list(fold_data.keys())
                data_matrix = np.array([fold_data[m] for m in methods])
                
                print(f"    Methods with data: {len(methods)}")
                for m in methods:
                    print(f"      {LOSO_METHOD_DISPLAY.get(m, m)}")
                
                # Check assumptions
                assumptions = check_assumptions(data_matrix, methods)
                
                print(f"    Normality: {'✅ PASSED' if assumptions['normality_passed'] else '❌ FAILED'} (p={assumptions['normality_p']:.4f})")
                print(f"    Sphericity: {'✅ PASSED' if assumptions['sphericity_passed'] else '❌ FAILED'}")
                print(f"    Assumptions Met: {'✅ YES' if assumptions['assumptions_passed'] else '❌ NO'}")
                
                # Choose test
                if assumptions['assumptions_passed']:
                    print("    ✅ Using Repeated Measures ANOVA")
                    result = perform_anova(data_matrix, methods)
                    if result:
                        print(f"    F({result['df_num']:.1f}, {result['df_den']:.1f}) = {result['f_stat']:.3f}, p={result['p_val']:.6f} {'✅' if result['significant'] else '❌'}")
                else:
                    print("    ❌ Using Friedman Test")
                    result = perform_friedman(data_matrix, methods)
                    if result:
                        print(f"    χ² = {result['friedman_stat']:.3f}, p={result['p_val']:.6f} {'✅' if result['significant'] else '❌'}")
                        print(f"    Best: {LOSO_METHOD_DISPLAY.get(result['best_method'], result['best_method'])} ({result['best_mean']:.4f})")
                
                # Store results
                all_results.append({
                    'Species': species,
                    'LR': lr,
                    'BS': bs,
                    'assumptions': assumptions,
                    'result': result,
                    'methods': methods,
                    'data_matrix': data_matrix
                })
    
    return all_results

# =============================================
# GENERATE SUMMARY TABLES
# =============================================

def generate_summary_tables(cdhit_results, loso_results):
    """Generate summary tables for paper"""
    
    # CD-HIT Summary
    cdhit_summary = []
    for res in cdhit_results:
        if res['result']:
            cdhit_summary.append({
                'LR': res['LR'],
                'BS': res['BS'],
                'Test_Used': 'Friedman' if not res['assumptions']['assumptions_passed'] else 'ANOVA',
                'Assumptions_Met': res['assumptions']['assumptions_passed'],
                'Statistic': res['result'].get('friedman_stat', res['result'].get('f_stat', 'N/A')),
                'P_Value': res['result']['p_val'],
                'Significant': res['result']['significant'],
                'Best_Method': METHOD_DISPLAY.get(res['result']['best_method'], res['result']['best_method']) if 'best_method' in res['result'] else 'N/A',
                'Best_Mean': res['result'].get('best_mean', 'N/A')
            })
    
    if cdhit_summary:
        cdhit_df = pd.DataFrame(cdhit_summary)
        cdhit_df.to_csv(os.path.join(OUTPUT_DIR, "CDHIT_Statistical_Summary_Per_LR_BS.csv"), index=False)
        print(f"\n✅ CD-HIT summary saved: {len(cdhit_df)} comparisons")
    
    # LOSO Summary
    loso_summary = []
    for res in loso_results:
        if res['result']:
            loso_summary.append({
                'Species': res['Species'],
                'LR': res['LR'],
                'BS': res['BS'],
                'Test_Used': 'Friedman' if not res['assumptions']['assumptions_passed'] else 'ANOVA',
                'Assumptions_Met': res['assumptions']['assumptions_passed'],
                'Statistic': res['result'].get('friedman_stat', res['result'].get('f_stat', 'N/A')),
                'P_Value': res['result']['p_val'],
                'Significant': res['result']['significant'],
                'Best_Method': LOSO_METHOD_DISPLAY.get(res['result']['best_method'], res['result']['best_method']) if 'best_method' in res['result'] else 'N/A',
                'Best_Mean': res['result'].get('best_mean', 'N/A')
            })
    
    if loso_summary:
        loso_df = pd.DataFrame(loso_summary)
        loso_df.to_csv(os.path.join(OUTPUT_DIR, "LOSO_Statistical_Summary_Per_LR_BS.csv"), index=False)
        print(f"✅ LOSO summary saved: {len(loso_df)} comparisons")
    
    return cdhit_df if cdhit_summary else None, loso_df if loso_summary else None

# =============================================
# PRINT SUMMARY STATISTICS
# =============================================

def print_summary_statistics(cdhit_results, loso_results):
    """Print summary statistics"""
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # CD-HIT
    cdhit_sig = sum(1 for r in cdhit_results if r['result'] and r['result']['significant'])
    cdhit_total = len([r for r in cdhit_results if r['result']])
    
    print(f"\n📊 CD-HIT:")
    print(f"  Total comparisons: {cdhit_total}")
    print(f"  Significant: {cdhit_sig}/{cdhit_total} ({cdhit_sig/cdhit_total*100:.1f}%)")
    
    # By LR
    print(f"\n  By Learning Rate:")
    for lr in LEARNING_RATES:
        lr_results = [r for r in cdhit_results if r['LR'] == lr and r['result']]
        if lr_results:
            sig = sum(1 for r in lr_results if r['result']['significant'])
            print(f"    LR={lr}: {sig}/{len(lr_results)} ({sig/len(lr_results)*100:.1f}%)")
    
    # By BS
    print(f"\n  By Batch Size:")
    for bs in BATCH_SIZES:
        bs_results = [r for r in cdhit_results if r['BS'] == bs and r['result']]
        if bs_results:
            sig = sum(1 for r in bs_results if r['result']['significant'])
            print(f"    BS={bs}: {sig}/{len(bs_results)} ({sig/len(bs_results)*100:.1f}%)")
    
    # LOSO
    loso_sig = sum(1 for r in loso_results if r['result'] and r['result']['significant'])
    loso_total = len([r for r in loso_results if r['result']])
    
    print(f"\n📊 LOSO:")
    print(f"  Total comparisons: {loso_total}")
    print(f"  Significant: {loso_sig}/{loso_total} ({loso_sig/loso_total*100:.1f}%)")
    
    # By Species
    print(f"\n  By Species:")
    for species in SPECIES_LIST:
        sp_results = [r for r in loso_results if r['Species'] == species and r['result']]
        if sp_results:
            sig = sum(1 for r in sp_results if r['result']['significant'])
            print(f"    {species}: {sig}/{len(sp_results)} ({sig/len(sp_results)*100:.1f}%)")
    
    # Best methods
    print(f"\n🏆 Best Method by Wins (CD-HIT):")
    cdhit_wins = {}
    for r in cdhit_results:
        if r['result'] and 'best_method' in r['result']:
            method = r['result']['best_method']
            cdhit_wins[method] = cdhit_wins.get(method, 0) + 1
    
    if cdhit_wins:
        sorted_wins = sorted(cdhit_wins.items(), key=lambda x: x[1], reverse=True)
        for method, count in sorted_wins[:5]:
            print(f"    {METHOD_DISPLAY.get(method, method)}: {count} wins")
    
    print(f"\n🏆 Best Method by Wins (LOSO):")
    loso_wins = {}
    for r in loso_results:
        if r['result'] and 'best_method' in r['result']:
            method = r['result']['best_method']
            loso_wins[method] = loso_wins.get(method, 0) + 1
    
    if loso_wins:
        sorted_wins = sorted(loso_wins.items(), key=lambda x: x[1], reverse=True)
        for method, count in sorted_wins[:5]:
            print(f"    {LOSO_METHOD_DISPLAY.get(method, method)}: {count} wins")

# =============================================
# MAIN EXECUTION
# =============================================

if __name__ == "__main__":
    
    print("=" * 80)
    print("STATISTICAL ANALYSIS: PER LR AND BS COMBINATION")
    print("Using ACTUAL 10-Fold CV Values from NPY Files")
    print("ALL Models Included (Attention + Baselines)")
    print("=" * 80)
    
    # Analyze CD-HIT
    print("\n🔍 Analyzing CD-HIT from NPY files...")
    cdhit_results = analyze_cdhit()
    
    # Analyze LOSO
    print("\n🔍 Analyzing LOSO from NPY files...")
    loso_results = analyze_loso()
    
    # Generate summary tables
    cdhit_summary, loso_summary = generate_summary_tables(cdhit_results, loso_results)
    
    # Print summary statistics
    print_summary_statistics(cdhit_results, loso_results)
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)