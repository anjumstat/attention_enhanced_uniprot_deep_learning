"""
Comprehensive Analysis for Attention-Enhanced Deep Learning Results
Enzyme Classification using UniProt Embeddings
"""

import numpy as np
import pandas as pd
import os
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
BASE_DIR = "D:/uni_prot2/NAE/Novel_Attention_Enhanced"
OUTPUT_DIR = "D:/uni_prot2/NAE/Analysis_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# METHODS based on your actual code
METHODS = {
    'Attention_Enhanced_Basic': 'results_Attention_Basic',
    'DNN_Baseline': 'results_DNN_Baseline',
    'Logistic_Baseline': 'results_Logistic_Baseline',
    'Ablation_No_Attention': 'ablation_no_attention',
    'Ablation_No_Residual': 'ablation_no_residual',
    'Ablation_50Percent_Data': 'ablation_50percent_data'
}

LEARNING_RATES = [0.01, 0.001, 0.0001]
BATCH_SIZES = [32, 64, 128, 256, 512]  # Added batch sizes 32 and 64

print(f"Input data from: {BASE_DIR}")
print(f"Output will be saved to: {OUTPUT_DIR}")
print(f"Methods to analyze: {list(METHODS.keys())}")
print(f"Learning rates: {LEARNING_RATES}")
print(f"Batch sizes: {BATCH_SIZES}")

def load_all_results_from_npy(base_dir):
    """Load all results from folders using .npy files"""
    
    all_results = []
    total_found = 0
    total_missing = 0
    
    for lr in LEARNING_RATES:
        for bs in BATCH_SIZES:
            for method_name, folder_name in METHODS.items():
                lr_str = f"{lr:.4f}".replace('.', '_')
                folder_path = os.path.join(base_dir, f"lr_{lr_str}_bs_{bs}", folder_name)
                npy_dir = os.path.join(folder_path, "npy_files")
                
                if os.path.exists(npy_dir):
                    print(f"✅ Loading: LR={lr}, BS={bs}, Method={method_name}")
                    
                    try:
                        result_data = load_single_experiment_npy(npy_dir, lr, bs, method_name)
                        if result_data:
                            all_results.append(result_data)
                            total_found += 1
                    except Exception as e:
                        print(f"❌ Error loading {folder_path}: {e}")
                        total_missing += 1
                else:
                    print(f"❌ Missing: {method_name} for LR={lr}, BS={bs}")
                    total_missing += 1
    
    print(f"\n📊 Loading Summary:")
    print(f"   ✅ Successfully loaded: {total_found} experiments")
    print(f"   ❌ Missing: {total_missing} experiments")
    print(f"   📈 Total experiments: {len(all_results)}")
    
    # Print summary of found batch sizes
    if all_results:
        df_temp = pd.DataFrame(all_results)
        print(f"   📊 Batch sizes found: {sorted(df_temp['batch_size'].unique())}")
        print(f"   📊 Learning rates found: {sorted(df_temp['learning_rate'].unique())}")
    
    return pd.DataFrame(all_results)

def load_single_experiment_npy(npy_dir, lr, bs, method):
    """Load results from a single experiment's npy_files directory"""
    
    result_data = {
        'learning_rate': lr,
        'batch_size': bs,
        'method': method,
        'npy_dir': npy_dir
    }
    
    try:
        # Try to load all_predictions.npy first to get fold info
        predictions_file = os.path.join(npy_dir, "all_predictions.npy")
        if os.path.exists(predictions_file):
            predictions = np.load(predictions_file, allow_pickle=True).item()
            result_data['n_folds'] = len(predictions.keys())
        else:
            result_data['n_folds'] = 10
        
        # Load fold_metrics.npy
        metrics_file = os.path.join(npy_dir, "fold_metrics.npy")
        if os.path.exists(metrics_file):
            metrics = np.load(metrics_file, allow_pickle=True).item()
            
            # Extract test metrics
            if 'test' in metrics:
                result_data['test_accuracy_mean'] = np.mean(metrics['test']['accuracy'])
                result_data['test_accuracy_std'] = np.std(metrics['test']['accuracy'])
                result_data['test_accuracy_values'] = metrics['test']['accuracy']
                
                result_data['test_precision_mean'] = np.mean(metrics['test']['precision'])
                result_data['test_recall_mean'] = np.mean(metrics['test']['recall'])
                result_data['test_f1_mean'] = np.mean(metrics['test']['f1'])
                result_data['test_mcc_mean'] = np.mean(metrics['test']['mcc'])
                result_data['test_auc_mean'] = np.mean(metrics['test']['auc'])
                result_data['test_auc_std'] = np.std(metrics['test']['auc'])
                result_data['test_auc_values'] = metrics['test']['auc']
            
            # Extract validation metrics
            if 'val' in metrics:
                result_data['val_accuracy_mean'] = np.mean(metrics['val']['accuracy'])
                result_data['val_accuracy_std'] = np.std(metrics['val']['accuracy'])
                result_data['val_accuracy_values'] = metrics['val']['accuracy']
                result_data['val_auc_mean'] = np.mean(metrics['val']['auc'])
            
            # Extract training metrics
            if 'train' in metrics:
                result_data['train_accuracy_mean'] = np.mean(metrics['train']['accuracy'])
                result_data['train_loss_mean'] = np.mean(metrics.get('training_time', [0]))
            
            # Training time
            if 'training_time' in metrics:
                result_data['training_time_mean'] = np.mean(metrics['training_time'])
                result_data['training_time_std'] = np.std(metrics['training_time'])
        
        # Load stability metrics
        stability_file = os.path.join(npy_dir, "stability_metrics.npy")
        if os.path.exists(stability_file):
            stability = np.load(stability_file, allow_pickle=True).item()
            result_data['feature_stability'] = stability.get('overall_stability', 0)
            result_data['jaccard_stability'] = stability.get('jaccard_stability', 0)
            result_data['rank_stability'] = stability.get('rank_stability', 0)
            result_data['consistency_ratio'] = stability.get('consistency_ratio', 0)
        
        # Try to load confusion matrices
        cm_file = os.path.join(npy_dir, "confusion_matrices.npy")
        if os.path.exists(cm_file):
            confusion_matrices = np.load(cm_file, allow_pickle=True)
            result_data['confusion_matrices'] = confusion_matrices
        
        # Try to load training history for convergence analysis
        history_file = os.path.join(npy_dir, "training_history.npy")
        if os.path.exists(history_file):
            history = np.load(history_file, allow_pickle=True).item()
            result_data['training_history'] = history
        
        return result_data
        
    except Exception as e:
        print(f"❌ Error processing {npy_dir}: {e}")
        return None

def run_comprehensive_analysis(df):
    """Run comprehensive analysis on all loaded results"""
    
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    # Create subdirectories
    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    data_dir = os.path.join(OUTPUT_DIR, "data")
    reports_dir = os.path.join(OUTPUT_DIR, "reports")
    
    for dir_path in [plots_dir, data_dir, reports_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 1. Performance comparison plots
    plot_performance_comparison(df, plots_dir)
    
    # 2. Hyperparameter analysis
    plot_hyperparameter_analysis(df, plots_dir)
    
    # 3. Ablation study analysis
    plot_ablation_analysis(df, plots_dir)
    
    # 4. Statistical significance testing
    statistical_results = run_statistical_tests(df, reports_dir)
    
    # 5. Method rankings
    ranking_results = calculate_method_rankings(df, reports_dir)
    
    # 6. Generate comprehensive report
    generate_report(df, reports_dir, statistical_results, ranking_results)
    
    # 7. Save all processed data
    save_processed_data(df, data_dir)
    
    return statistical_results, ranking_results

def plot_performance_comparison(df, plots_dir):
    """Plot overall performance comparison across methods"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Test Accuracy by Method
    ax1 = axes[0, 0]
    method_perf = df.groupby('method')['test_accuracy_mean'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    ax1.bar(range(len(method_perf)), method_perf['mean'], yerr=method_perf['std'], capsize=5, alpha=0.7, color='steelblue')
    ax1.set_xticks(range(len(method_perf)))
    ax1.set_xticklabels(method_perf.index, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Test Accuracy by Method')
    ax1.grid(True, alpha=0.3)
    
    # 2. AUC by Method
    ax2 = axes[0, 1]
    if 'test_auc_mean' in df.columns:
        auc_perf = df.groupby('method')['test_auc_mean'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        ax2.bar(range(len(auc_perf)), auc_perf['mean'], yerr=auc_perf['std'], capsize=5, alpha=0.7, color='coral')
        ax2.set_xticks(range(len(auc_perf)))
        ax2.set_xticklabels(auc_perf.index, rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('AUC')
        ax2.set_title('AUC by Method')
        ax2.grid(True, alpha=0.3)
    
    # 3. Training vs Validation Accuracy
    ax3 = axes[0, 2]
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        ax3.scatter(method_data['train_accuracy_mean'], method_data['val_accuracy_mean'], 
                   label=method, s=80, alpha=0.6)
    max_acc = max(df['train_accuracy_mean'].max(), df['val_accuracy_mean'].max())
    ax3.plot([0, max_acc], [0, max_acc], 'k--', alpha=0.3, label='Ideal')
    ax3.set_xlabel('Training Accuracy')
    ax3.set_ylabel('Validation Accuracy')
    ax3.set_title('Training vs Validation Accuracy')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. F1 Score by Method
    ax4 = axes[1, 0]
    if 'test_f1_mean' in df.columns:
        f1_perf = df.groupby('method')['test_f1_mean'].mean().sort_values(ascending=False)
        ax4.barh(range(len(f1_perf)), f1_perf.values, alpha=0.7, color='green')
        ax4.set_yticks(range(len(f1_perf)))
        ax4.set_yticklabels(f1_perf.index, fontsize=9)
        ax4.set_xlabel('F1 Score')
        ax4.set_title('F1 Score by Method')
        ax4.grid(True, alpha=0.3)
    
    # 5. MCC by Method
    ax5 = axes[1, 1]
    if 'test_mcc_mean' in df.columns:
        mcc_perf = df.groupby('method')['test_mcc_mean'].mean().sort_values(ascending=False)
        ax5.barh(range(len(mcc_perf)), mcc_perf.values, alpha=0.7, color='purple')
        ax5.set_yticks(range(len(mcc_perf)))
        ax5.set_yticklabels(mcc_perf.index, fontsize=9)
        ax5.set_xlabel('Matthews Correlation Coefficient')
        ax5.set_title('MCC by Method')
        ax5.grid(True, alpha=0.3)
    
    # 6. Feature Stability
    ax6 = axes[1, 2]
    stability_data = df[df['feature_stability'].notna()]
    if not stability_data.empty:
        stability_perf = stability_data.groupby('method')['feature_stability'].mean().sort_values(ascending=False)
        ax6.bar(range(len(stability_perf)), stability_perf.values, alpha=0.7, color='teal')
        ax6.set_xticks(range(len(stability_perf)))
        ax6.set_xticklabels(stability_perf.index, rotation=45, ha='right', fontsize=9)
        ax6.set_ylabel('Stability Score')
        ax6.set_title('Feature Stability by Method')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: performance_comparison.png")

def plot_hyperparameter_analysis(df, plots_dir):
    """Analyze effect of learning rate and batch size"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Learning rate effect
    ax1 = axes[0, 0]
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        lr_summary = method_data.groupby('learning_rate')['test_accuracy_mean'].mean()
        ax1.plot(lr_summary.index, lr_summary.values, 'o-', label=method, markersize=6, linewidth=2)
    ax1.set_xscale('log')
    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Effect of Learning Rate on Accuracy')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Batch size effect
    ax2 = axes[0, 1]
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        bs_summary = method_data.groupby('batch_size')['test_accuracy_mean'].mean()
        ax2.plot(bs_summary.index, bs_summary.values, 's-', label=method, markersize=6, linewidth=2)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Effect of Batch Size on Accuracy')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Heatmap for best method
    ax3 = axes[1, 0]
    best_method = df.groupby('method')['test_accuracy_mean'].mean().idxmax()
    best_method_data = df[df['method'] == best_method]
    heatmap_data = best_method_data.pivot_table(
        values='test_accuracy_mean', 
        index='batch_size', 
        columns='learning_rate'
    )
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='viridis', ax=ax3)
    ax3.set_title(f'Accuracy Heatmap: {best_method}')
    
    # 4. Method comparison across hyperparameters
    ax4 = axes[1, 1]
    pivot_data = df.pivot_table(
        values='test_accuracy_mean',
        index='method',
        columns='learning_rate',
        aggfunc='mean'
    )
    sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax4)
    ax4.set_title('Method Performance vs Learning Rate')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'hyperparameter_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: hyperparameter_analysis.png")

def plot_ablation_analysis(df, plots_dir):
    """Plot ablation study results"""
    
    ablation_methods = ['Attention_Enhanced_Basic', 'Ablation_No_Attention', 'Ablation_No_Residual', 'Ablation_50Percent_Data']
    ablation_df = df[df['method'].isin(ablation_methods)]
    
    if ablation_df.empty:
        print("No ablation data found")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Accuracy comparison
    ax1 = axes[0]
    acc_summary = ablation_df.groupby('method')['test_accuracy_mean'].agg(['mean', 'std']).reindex(ablation_methods)
    ax1.bar(range(len(acc_summary)), acc_summary['mean'], yerr=acc_summary['std'], capsize=5, alpha=0.7)
    ax1.set_xticks(range(len(acc_summary)))
    ax1.set_xticklabels(acc_summary.index, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Ablation Study: Accuracy')
    ax1.grid(True, alpha=0.3)
    
    # 2. AUC comparison
    ax2 = axes[1]
    if 'test_auc_mean' in ablation_df.columns:
        auc_summary = ablation_df.groupby('method')['test_auc_mean'].agg(['mean', 'std']).reindex(ablation_methods)
        ax2.bar(range(len(auc_summary)), auc_summary['mean'], yerr=auc_summary['std'], capsize=5, alpha=0.7, color='coral')
        ax2.set_xticks(range(len(auc_summary)))
        ax2.set_xticklabels(auc_summary.index, rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('AUC')
        ax2.set_title('Ablation Study: AUC')
        ax2.grid(True, alpha=0.3)
    
    # 3. Stability comparison
    ax3 = axes[2]
    stability_summary = ablation_df.groupby('method')['feature_stability'].mean().reindex(ablation_methods)
    ax3.bar(range(len(stability_summary)), stability_summary.values, alpha=0.7, color='teal')
    ax3.set_xticks(range(len(stability_summary)))
    ax3.set_xticklabels(stability_summary.index, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Stability Score')
    ax3.set_title('Ablation Study: Feature Stability')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'ablation_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: ablation_analysis.png")

def run_statistical_tests(df, reports_dir):
    """Run statistical significance tests"""
    
    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("="*60)
    
    statistical_results = {}
    
    # Prepare fold-wise accuracies for each method
    method_fold_accuracies = {}
    
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        all_fold_accuracies = []
        
        for _, row in method_data.iterrows():
            if 'test_accuracy_values' in row and row['test_accuracy_values'] is not None:
                all_fold_accuracies.extend(row['test_accuracy_values'])
        
        if all_fold_accuracies:
            method_fold_accuracies[method] = all_fold_accuracies
    
    statistical_results['method_fold_accuracies'] = method_fold_accuracies
    
    # Friedman test
    valid_methods = {k: v for k, v in method_fold_accuracies.items() if len(v) >= 5}
    
    if len(valid_methods) >= 3:
        min_folds = min(len(acc) for acc in valid_methods.values())
        balanced_data = [acc[:min_folds] for acc in valid_methods.values()]
        
        try:
            friedman_stat, friedman_p = stats.friedmanchisquare(*balanced_data)
            statistical_results['friedman_stat'] = friedman_stat
            statistical_results['friedman_p'] = friedman_p
            statistical_results['friedman_significant'] = friedman_p < 0.05
            
            print(f"Friedman Test: χ² = {friedman_stat:.3f}, p = {friedman_p:.4f}")
            
            if friedman_p < 0.05:
                print("Significant differences found between methods (p < 0.05)")
                
                # Pairwise Wilcoxon tests
                pairwise_results = []
                methods_list = list(valid_methods.keys())
                
                for i in range(len(methods_list)):
                    for j in range(i+1, len(methods_list)):
                        data_i = valid_methods[methods_list[i]]
                        data_j = valid_methods[methods_list[j]]
                        min_len = min(len(data_i), len(data_j))
                        stat, p_val = stats.wilcoxon(data_i[:min_len], data_j[:min_len])
                        
                        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        
                        pairwise_results.append({
                            'method1': methods_list[i],
                            'method2': methods_list[j],
                            'p_value': p_val,
                            'significance': significance,
                            'mean1': np.mean(data_i[:min_len]),
                            'mean2': np.mean(data_j[:min_len])
                        })
                        
                        print(f"  {methods_list[i]:30s} vs {methods_list[j]:30s}: p = {p_val:.4f} {significance}")
                
                statistical_results['pairwise_tests'] = pairwise_results
                
                # Save pairwise results
                pairwise_df = pd.DataFrame(pairwise_results)
                pairwise_df.to_csv(os.path.join(reports_dir, 'statistical_pairwise_tests.csv'), index=False)
                
        except Exception as e:
            print(f"Statistical tests failed: {e}")
            statistical_results['error'] = str(e)
    
    return statistical_results

def calculate_method_rankings(df, reports_dir):
    """Calculate and save method rankings"""
    
    print("\n" + "="*60)
    print("METHOD RANKINGS")
    print("="*60)
    
    ranking_results = {}
    
    # 1. Accuracy ranking
    accuracy_ranks = df.groupby('method')['test_accuracy_mean'].mean().sort_values(ascending=False)
    ranking_results['accuracy_ranks'] = accuracy_ranks.to_dict()
    
    print("\nRank by Test Accuracy:")
    for i, (method, acc) in enumerate(accuracy_ranks.items(), 1):
        print(f"  {i}. {method:30s}: {acc:.4f}")
    
    # 2. AUC ranking
    if 'test_auc_mean' in df.columns:
        auc_ranks = df.groupby('method')['test_auc_mean'].mean().sort_values(ascending=False)
        ranking_results['auc_ranks'] = auc_ranks.to_dict()
        
        print("\nRank by AUC:")
        for i, (method, auc) in enumerate(auc_ranks.items(), 1):
            print(f"  {i}. {method:30s}: {auc:.4f}")
    
    # 3. Stability ranking
    stability_data = df[df['feature_stability'].notna()]
    if not stability_data.empty:
        stability_ranks = stability_data.groupby('method')['feature_stability'].mean().sort_values(ascending=False)
        ranking_results['stability_ranks'] = stability_ranks.to_dict()
        
        print("\nRank by Feature Stability:")
        for i, (method, stab) in enumerate(stability_ranks.items(), 1):
            print(f"  {i}. {method:30s}: {stab:.4f}")
    
    # Save rankings
    rankings_df = pd.DataFrame([
        {'metric': 'Test Accuracy', **ranking_results.get('accuracy_ranks', {})},
        {'metric': 'AUC', **ranking_results.get('auc_ranks', {})},
        {'metric': 'Stability', **ranking_results.get('stability_ranks', {})}
    ])
    rankings_df.to_csv(os.path.join(reports_dir, 'method_rankings.csv'), index=False)
    
    return ranking_results

def generate_report(df, reports_dir, statistical_results, ranking_results):
    """Generate comprehensive analysis report"""
    
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("ATTENTION-ENHANCED DEEP LEARNING ANALYSIS REPORT")
    report_lines.append("Enzyme Classification using UniProt Embeddings")
    report_lines.append("="*70)
    report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total experiments: {len(df)}")
    report_lines.append(f"Methods: {', '.join(df['method'].unique())}")
    report_lines.append(f"Learning rates: {', '.join(map(str, sorted(df['learning_rate'].unique())))}")
    report_lines.append(f"Batch sizes: {', '.join(map(str, sorted(df['batch_size'].unique())))}")
    
    # Best overall configuration
    if 'test_accuracy_mean' in df.columns:
        best_idx = df['test_accuracy_mean'].idxmax()
        best_config = df.loc[best_idx]
        report_lines.append("\n" + "-"*50)
        report_lines.append("BEST OVERALL CONFIGURATION")
        report_lines.append("-"*50)
        report_lines.append(f"  Method: {best_config['method']}")
        report_lines.append(f"  Learning Rate: {best_config['learning_rate']}")
        report_lines.append(f"  Batch Size: {best_config['batch_size']}")
        report_lines.append(f"  Test Accuracy: {best_config['test_accuracy_mean']:.4f} ± {best_config['test_accuracy_std']:.4f}")
        if 'test_auc_mean' in best_config:
            report_lines.append(f"  Test AUC: {best_config['test_auc_mean']:.4f} ± {best_config['test_auc_std']:.4f}")
        if 'feature_stability' in best_config and not pd.isna(best_config['feature_stability']):
            report_lines.append(f"  Feature Stability: {best_config['feature_stability']:.4f}")
    
    # Best method overall
    best_method = df.groupby('method')['test_accuracy_mean'].mean().idxmax()
    best_method_acc = df.groupby('method')['test_accuracy_mean'].mean().max()
    report_lines.append(f"\nBEST METHOD OVERALL: {best_method} (Avg Accuracy: {best_method_acc:.4f})")
    
    # Method rankings summary
    report_lines.append("\n" + "-"*50)
    report_lines.append("METHOD RANKINGS SUMMARY")
    report_lines.append("-"*50)
    
    if 'accuracy_ranks' in ranking_results:
        report_lines.append("\nTest Accuracy Ranking:")
        for i, (method, acc) in enumerate(ranking_results['accuracy_ranks'].items(), 1):
            report_lines.append(f"  {i}. {method}: {acc:.4f}")
    
    if 'auc_ranks' in ranking_results:
        report_lines.append("\nAUC Ranking:")
        for i, (method, auc) in enumerate(ranking_results['auc_ranks'].items(), 1):
            report_lines.append(f"  {i}. {method}: {auc:.4f}")
    
    # Statistical results
    if 'friedman_significant' in statistical_results:
        report_lines.append("\n" + "-"*50)
        report_lines.append("STATISTICAL SIGNIFICANCE")
        report_lines.append("-"*50)
        report_lines.append(f"Friedman Test: χ² = {statistical_results['friedman_stat']:.3f}, p = {statistical_results['friedman_p']:.4f}")
        if statistical_results['friedman_significant']:
            report_lines.append("Result: Significant differences found between methods (p < 0.05)")
        else:
            report_lines.append("Result: No significant differences found between methods")
    
    # Save report
    report_path = os.path.join(reports_dir, 'analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"\n✅ Saved: analysis_report.txt")
    
    # Print summary to console
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*60)
    for line in report_lines[-20:]:
        print(line)

def save_processed_data(df, data_dir):
    """Save all processed data"""
    
    # Main dataframe
    df.to_csv(os.path.join(data_dir, 'all_experiment_results.csv'), index=False)
    
    # Method summary
    summary_cols = ['test_accuracy_mean', 'test_auc_mean', 'test_f1_mean', 'test_mcc_mean', 'feature_stability']
    available_cols = [col for col in summary_cols if col in df.columns]
    
    if available_cols:
        summary_stats = df.groupby('method')[available_cols].agg(['mean', 'std']).round(4)
        summary_stats.to_csv(os.path.join(data_dir, 'method_summary.csv'))
    
    # Hyperparameter summary
    hyperparam_summary = df.groupby(['method', 'learning_rate', 'batch_size'])['test_accuracy_mean'].mean().unstack()
    hyperparam_summary.to_csv(os.path.join(data_dir, 'hyperparameter_summary.csv'))
    
    print(f"\n✅ Saved processed data to: {data_dir}")

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("ATTENTION-ENHANCED DEEP LEARNING ANALYSIS")
    print("Enzyme Classification using UniProt Embeddings")
    print("="*60)
    
    # Load all results
    all_results_df = load_all_results_from_npy(BASE_DIR)
    
    if not all_results_df.empty:
        print(f"\n✅ Loaded {len(all_results_df)} experiments")
        
        # Run comprehensive analysis
        statistical_results, ranking_results = run_comprehensive_analysis(all_results_df)
        
        print("\n" + "="*60)
        print("🎉 ANALYSIS COMPLETE!")
        print("="*60)
        print(f"📁 Results saved to: {OUTPUT_DIR}")
        print(f"   📊 Plots: {OUTPUT_DIR}/plots/")
        print(f"   📈 Data: {OUTPUT_DIR}/data/")
        print(f"   📄 Reports: {OUTPUT_DIR}/reports/")
    else:
        print("\n❌ No results found! Please check your BASE_DIR path.")