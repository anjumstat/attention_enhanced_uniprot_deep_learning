# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 18:18:09 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
05_baseline_comparisons_with_checkpoint.py
Baseline Models Comparison with CHECKPOINT/RESUME functionality
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix, 
                             roc_auc_score, roc_curve)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import os
import time
import shutil
import json
import pickle
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# =============================================
# CONFIGURATION
# =============================================

DATA_PATH = r"D:\uni_prot2\revision\data\clean_splits\homology_aware\combined_train_val.csv"
TEST_DATA_PATH = r"D:\uni_prot2\revision\data\clean_splits\homology_aware\test.csv"
BASE_DIR = r"D:\uni_prot2\revision\results\homology_aware_baselines_with_checkpoint"

# Checkpoint file
CHECKPOINT_FILE = os.path.join(BASE_DIR, "baseline_checkpoint.json")

LEARNING_RATES = [0.01, 0.001, 0.0001]
BATCH_SIZES = [32, 64, 128, 256]

METHODS = {
    'Random_Forest': {'dir': 'results_Random_Forest', 'model_type': 'sklearn'},
    'SVM': {'dir': 'results_SVM', 'model_type': 'sklearn'},
    'MLP_256': {'dir': 'results_MLP_256', 'model_type': 'mlp', 'hidden_units': 256},
    'MLP_512': {'dir': 'results_MLP_512', 'model_type': 'mlp', 'hidden_units': 512},
}

# =============================================
# CHECKPOINT MANAGER
# =============================================

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except:
            return {"completed_runs": []}
    return {"completed_runs": []}

def save_checkpoint(completed_runs):
    checkpoint = {
        'completed_runs': completed_runs,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_completed': len(completed_runs)
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)

def is_completed(method, lr, bs):
    checkpoint = load_checkpoint()
    key = f"{method}_lr_{lr}_bs_{bs}"
    return key in checkpoint["completed_runs"]

def mark_completed(method, lr, bs):
    checkpoint = load_checkpoint()
    key = f"{method}_lr_{lr}_bs_{bs}"
    if key not in checkpoint["completed_runs"]:
        checkpoint["completed_runs"].append(key)
        save_checkpoint(checkpoint["completed_runs"])

# =============================================
# MODEL BUILDERS (same as before)
# =============================================

def build_sklearn_model(model_type, random_state=42):
    if model_type == 'Random_Forest':
        return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state,
                                      n_jobs=-1, class_weight='balanced')
    elif model_type == 'SVM':
        return SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
                   random_state=random_state, class_weight='balanced')
    else:
        raise ValueError(f"Unknown sklearn model: {model_type}")

def build_mlp_model(input_dim, hidden_units=256, dropout_rate=0.3, learning_rate=0.001):
    model = models.Sequential([
        layers.Dense(hidden_units, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(dropout_rate),
        layers.Dense(hidden_units // 2, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'precision', 'recall', tf.keras.metrics.AUC(name='auc')])
    return model

# =============================================
# NUMPY ENCODER
# =============================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# =============================================
# PLOT ROC CURVES
# =============================================

def plot_roc_curves(roc_data, output_dir, method_name):
    plt.figure(figsize=(10, 8))
    for fold_data in roc_data:
        plt.plot(fold_data['fpr'], fold_data['tpr'], 
                label=f"Fold {fold_data['fold']} (AUC = {fold_data['auc']:.3f})", 
                alpha=0.7, linewidth=1.5)
    if len(roc_data) > 1:
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        for fold_data in roc_data:
            tprs.append(np.interp(mean_fpr, fold_data['fpr'], fold_data['tpr']))
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = np.mean([d['auc'] for d in roc_data])
        std_auc = np.std([d['auc'] for d in roc_data])
        plt.plot(mean_fpr, mean_tpr, color='black', 
                label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})',
                linewidth=2.5, linestyle='--')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - {method_name} (10-Fold CV)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ROC_Curves_{method_name}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"ROC_Curves_{method_name}.pdf"), bbox_inches='tight')
    plt.close()

# =============================================
# MAIN EXPERIMENT WITH CHECKPOINT
# =============================================

def run_baseline_experiment(method_name, config, learning_rate, batch_size):
    """Run baseline experiment with checkpoint/resume"""
    
    # Check if already completed
    if is_completed(method_name, learning_rate, batch_size):
        print(f"\n⏭️ SKIPPING: {method_name} (already completed)")
        return None
    
    lr_str = f"{learning_rate:.4f}".replace('.', '_')
    batch_str = str(batch_size)
    output_dir = os.path.join(BASE_DIR, f"lr_{lr_str}_bs_{batch_str}", config['dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    npy_dir = os.path.join(output_dir, "npy_files")
    csv_dir = os.path.join(output_dir, "csv_files")
    plots_dir = os.path.join(output_dir, "plots")
    models_dir = os.path.join(output_dir, "models")
    
    for dir_path in [npy_dir, csv_dir, plots_dir, models_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    try:
        print(f"\n🚀 Running: {method_name} | LR={learning_rate} | BS={batch_size}")
        
        # Load data
        train_data = pd.read_csv(DATA_PATH)
        test_data = pd.read_csv(TEST_DATA_PATH)
        
        X_train_all = train_data.iloc[:, :-1].values
        y_train_all = train_data.iloc[:, -1].values
        X_test_fixed = test_data.iloc[:, :-1].values
        y_test_fixed = test_data.iloc[:, -1].values
        
        scaler = StandardScaler()
        X_train_all = scaler.fit_transform(X_train_all)
        X_test_fixed = scaler.transform(X_test_fixed)
        
        y_train_all_cat = y_train_all.reshape(-1, 1)
        y_test_fixed_cat = y_test_fixed.reshape(-1, 1)
        
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        fold_metrics = {
            'train': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'mcc': [], 'auc': []},
            'val': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'mcc': [], 'auc': []},
            'test_fixed': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'mcc': [], 'auc': []},
            'training_time': []
        }
        
        all_predictions = {}
        confusion_matrices, roc_data = [], []
        
        best_val_auc, best_fold = -np.inf, -1
        model_type = config.get('model_type', 'sklearn')
        full_model_start_time = time.time()
        
        for fold, (train_index, val_index) in enumerate(skf.split(X_train_all, y_train_all)):
            print(f"  Fold {fold+1}/10")
            
            X_train_fold, X_val_fold = X_train_all[train_index], X_train_all[val_index]
            y_train_fold, y_val_fold = y_train_all_cat[train_index], y_train_all_cat[val_index]
            
            start_time = time.time()
            
            if model_type == 'sklearn':
                model = build_sklearn_model(method_name)
                model.fit(X_train_fold, y_train_fold.flatten())
                
                y_train_pred = model.predict(X_train_fold)
                y_train_prob = model.predict_proba(X_train_fold)[:, 1]
                y_val_pred = model.predict(X_val_fold)
                y_val_prob = model.predict_proba(X_val_fold)[:, 1]
                y_test_pred = model.predict(X_test_fixed)
                y_test_prob = model.predict_proba(X_test_fixed)[:, 1]
                
                model_path = os.path.join(models_dir, f"fold{fold+1}_model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
            else:
                hidden_units = config.get('hidden_units', 256)
                model = build_mlp_model(X_train_all.shape[1], hidden_units, learning_rate=learning_rate)
                
                early_stop = callbacks.EarlyStopping(
                    monitor='val_auc', patience=10, min_delta=0.001,
                    mode='max', restore_best_weights=True, verbose=0
                )
                
                history = model.fit(
                    X_train_fold, y_train_fold,
                    epochs=50, batch_size=batch_size,
                    validation_data=(X_val_fold, y_val_fold),
                    verbose=0, callbacks=[early_stop]
                )
                
                y_train_prob = model.predict(X_train_fold, verbose=0).flatten()
                y_train_pred = (y_train_prob > 0.5).astype(int)
                y_val_prob = model.predict(X_val_fold, verbose=0).flatten()
                y_val_pred = (y_val_prob > 0.5).astype(int)
                y_test_prob = model.predict(X_test_fixed, verbose=0).flatten()
                y_test_pred = (y_test_prob > 0.5).astype(int)
                
                model_path = os.path.join(models_dir, f"fold{fold+1}_model.h5")
                model.save(model_path)
                tf.keras.backend.clear_session()
            
            fold_time = time.time() - start_time
            fold_metrics['training_time'].append(fold_time)
            
            # Evaluation
            datasets = [
                ('train', y_train_fold.flatten(), y_train_pred, y_train_prob),
                ('val', y_val_fold.flatten(), y_val_pred, y_val_prob),
                ('test_fixed', y_test_fixed.flatten(), y_test_pred, y_test_prob)
            ]
            
            for dataset_name, y_true, y_pred, y_prob in datasets:
                fold_metrics[dataset_name]['accuracy'].append(accuracy_score(y_true, y_pred))
                fold_metrics[dataset_name]['precision'].append(precision_score(y_true, y_pred, zero_division=0))
                fold_metrics[dataset_name]['recall'].append(recall_score(y_true, y_pred, zero_division=0))
                fold_metrics[dataset_name]['f1'].append(f1_score(y_true, y_pred, zero_division=0))
                fold_metrics[dataset_name]['mcc'].append(matthews_corrcoef(y_true, y_pred))
                fold_metrics[dataset_name]['auc'].append(roc_auc_score(y_true, y_prob))
            
            # ROC data
            fpr, tpr, _ = roc_curve(y_test_fixed, y_test_prob)
            auc_score = roc_auc_score(y_test_fixed, y_test_prob)
            roc_data.append({'fpr': fpr, 'tpr': tpr, 'fold': fold+1, 'auc': auc_score})
            
            # Store predictions
            all_predictions[f'fold{fold+1}'] = {
                'train': {'true': y_train_fold.flatten(), 'pred': y_train_pred, 'prob': y_train_prob},
                'val': {'true': y_val_fold.flatten(), 'pred': y_val_pred, 'prob': y_val_prob},
                'test_fixed': {'true': y_test_fixed, 'pred': y_test_pred, 'prob': y_test_prob}
            }
            
            cm = confusion_matrix(y_test_fixed, y_test_pred)
            confusion_matrices.append(cm)
            
            val_auc = np.mean(fold_metrics['val']['auc'])
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_fold = fold

        full_model_time_minutes = (time.time() - full_model_start_time) / 60
        
        # Save NPY files
        print("💾 Saving NPY files...")
        
        roc_data_array = np.array([(d['fold'], d['fpr'], d['tpr'], d['auc']) for d in roc_data], dtype=object)
        np.save(os.path.join(npy_dir, "roc_data_all_folds.npy"), roc_data_array)
        
        for i, roc_d in enumerate(roc_data):
            fold_roc = {'fold': roc_d['fold'], 'fpr': roc_d['fpr'], 'tpr': roc_d['tpr'], 'auc': roc_d['auc']}
            np.save(os.path.join(npy_dir, f"roc_data_fold_{roc_d['fold']}.npy"), fold_roc)
        
        np.save(os.path.join(npy_dir, "all_predictions.npy"), all_predictions)
        np.save(os.path.join(npy_dir, "confusion_matrices.npy"), np.array(confusion_matrices))
        np.save(os.path.join(npy_dir, "fold_metrics.npy"), fold_metrics)
        
        # Generate ROC curves
        print("📈 Generating ROC curves...")
        plot_roc_curves(roc_data, plots_dir, method_name)
        
        # Save CSV files
        print("💾 Saving CSV files...")
        
        # Average_Metrics_With_Std.csv
        avg_metrics = {
            'Dataset': ['Train', 'Validation', 'Test (Fixed)'],
            'Accuracy_Mean': [
                np.mean(fold_metrics['train']['accuracy']),
                np.mean(fold_metrics['val']['accuracy']),
                np.mean(fold_metrics['test_fixed']['accuracy'])
            ],
            'Accuracy_Std': [
                np.std(fold_metrics['train']['accuracy']),
                np.std(fold_metrics['val']['accuracy']),
                np.std(fold_metrics['test_fixed']['accuracy'])
            ],
            'AUC_Mean': [
                np.mean(fold_metrics['train']['auc']),
                np.mean(fold_metrics['val']['auc']),
                np.mean(fold_metrics['test_fixed']['auc'])
            ],
            'AUC_Std': [
                np.std(fold_metrics['train']['auc']),
                np.std(fold_metrics['val']['auc']),
                np.std(fold_metrics['test_fixed']['auc'])
            ],
            'F1_Mean': [
                np.mean(fold_metrics['train']['f1']),
                np.mean(fold_metrics['val']['f1']),
                np.mean(fold_metrics['test_fixed']['f1'])
            ],
            'F1_Std': [
                np.std(fold_metrics['train']['f1']),
                np.std(fold_metrics['val']['f1']),
                np.std(fold_metrics['test_fixed']['f1'])
            ],
            'MCC_Mean': [
                np.mean(fold_metrics['train']['mcc']),
                np.mean(fold_metrics['val']['mcc']),
                np.mean(fold_metrics['test_fixed']['mcc'])
            ],
            'MCC_Std': [
                np.std(fold_metrics['train']['mcc']),
                np.std(fold_metrics['val']['mcc']),
                np.std(fold_metrics['test_fixed']['mcc'])
            ]
        }
        avg_metrics_df = pd.DataFrame(avg_metrics)
        avg_metrics_df.to_csv(os.path.join(csv_dir, "Average_Metrics_With_Std.csv"), index=False)
        
        # Test_Metrics_Summary.csv
        test_summary = {
            'Metric': ['Accuracy', 'AUC', 'F1', 'MCC', 'Precision', 'Recall'],
            'Mean': [
                np.mean(fold_metrics['test_fixed']['accuracy']),
                np.mean(fold_metrics['test_fixed']['auc']),
                np.mean(fold_metrics['test_fixed']['f1']),
                np.mean(fold_metrics['test_fixed']['mcc']),
                np.mean(fold_metrics['test_fixed']['precision']),
                np.mean(fold_metrics['test_fixed']['recall'])
            ],
            'Std': [
                np.std(fold_metrics['test_fixed']['accuracy']),
                np.std(fold_metrics['test_fixed']['auc']),
                np.std(fold_metrics['test_fixed']['f1']),
                np.std(fold_metrics['test_fixed']['mcc']),
                np.std(fold_metrics['test_fixed']['precision']),
                np.std(fold_metrics['test_fixed']['recall'])
            ]
        }
        test_summary_df = pd.DataFrame(test_summary)
        test_summary_df.to_csv(os.path.join(csv_dir, "Test_Metrics_Summary.csv"), index=False)
        
        # Training_Time_Summary.csv
        time_df = pd.DataFrame({
            'Method': [method_name],
            'Learning_Rate': [learning_rate],
            'Batch_Size': [batch_size],
            'Total_Training_Time_Minutes': [full_model_time_minutes],
            'Average_Fold_Time_Seconds': [np.mean(fold_metrics['training_time'])],
            'Std_Fold_Time_Seconds': [np.std(fold_metrics['training_time'])]
        })
        time_df.to_csv(os.path.join(csv_dir, "Training_Time_Summary.csv"), index=False)
        
        # All_Predictions_Detailed.csv
        predictions_data = []
        for fold_num, preds in all_predictions.items():
            for dataset in ['train', 'val', 'test_fixed']:
                true = preds[dataset]['true']
                pred = preds[dataset]['pred']
                prob = preds[dataset]['prob']
                for i, (t, p, pr) in enumerate(zip(true, pred, prob)):
                    predictions_data.append({
                        'fold': fold_num,
                        'dataset': dataset,
                        'sample_id': i,
                        'true_label': int(t),
                        'predicted_label': int(p),
                        'predicted_probability': float(pr),
                        'correct': int(t == p)
                    })
        
        predictions_df = pd.DataFrame(predictions_data)
        predictions_df.to_csv(os.path.join(csv_dir, "All_Predictions_Detailed.csv"), index=False)
        
        # Experiment_Configuration.csv
        config_df = pd.DataFrame({
            'Parameter': ['Method', 'Learning_Rate', 'Batch_Size', 'Model_Type', 'Number_of_Folds', 'Test_Size'],
            'Value': [method_name, learning_rate, batch_size, 
                     config.get('model_type', 'unknown'),
                     10, len(y_test_fixed)]
        })
        config_df.to_csv(os.path.join(csv_dir, "Experiment_Configuration.csv"), index=False)
        
        # Experiment_Summary.csv
        summary_stats = {
            'Method': method_name,
            'Best_Validation_AUC': float(best_val_auc),
            'Best_Fold': int(best_fold + 1) if best_fold >= 0 else -1,
            'Mean_Test_Accuracy': float(np.mean(fold_metrics['test_fixed']['accuracy'])),
            'Std_Test_Accuracy': float(np.std(fold_metrics['test_fixed']['accuracy'])),
            'Mean_Test_AUC': float(np.mean(fold_metrics['test_fixed']['auc'])),
            'Std_Test_AUC': float(np.std(fold_metrics['test_fixed']['auc'])),
            'Mean_Training_Time_Per_Fold': float(np.mean(fold_metrics['training_time'])),
            'Total_Training_Time_Minutes': float(full_model_time_minutes),
            'Test_Size': int(len(y_test_fixed))
        }
        
        summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
        summary_df.to_csv(os.path.join(csv_dir, "Experiment_Summary.csv"), index=False)
        
        print(f"✅ {method_name} completed successfully")
        print(f"   Mean Test Accuracy: {np.mean(fold_metrics['test_fixed']['accuracy']):.4f} ± {np.std(fold_metrics['test_fixed']['accuracy']):.4f}")
        print(f"   Mean Test AUC: {np.mean(fold_metrics['test_fixed']['auc']):.4f} ± {np.std(fold_metrics['test_fixed']['auc']):.4f}")
        
        # Mark as completed
        mark_completed(method_name, learning_rate, batch_size)
        
        return {
            "status": "success", 
            "method": method_name, 
            "test_accuracy": np.mean(fold_metrics['test_fixed']['accuracy']),
            "test_auc": np.mean(fold_metrics['test_fixed']['auc'])
        }
    
    except Exception as e:
        print(f"❌ Error in {method_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "method": method_name, "error": str(e)}

# =============================================
# MAIN EXECUTION WITH RESUME
# =============================================

if __name__ == "__main__":
    
    if os.path.exists(BASE_DIR):
        print("📁 Existing results directory found. Will resume if needed.")
    else:
        os.makedirs(BASE_DIR, exist_ok=True)
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    completed_runs = checkpoint.get("completed_runs", [])
    
    print("=" * 80)
    print("BASELINE MODELS WITH CHECKPOINT/RESUME")
    print("=" * 80)
    print(f"Base directory: {BASE_DIR}")
    print(f"Completed runs: {len(completed_runs)}")
    print("=" * 80)
    
    results = []
    
    # Total experiments
    total_experiments = len(LEARNING_RATES) * len(BATCH_SIZES) * len(METHODS)
    print(f"\n📊 Progress: {len(completed_runs)}/{total_experiments} experiments completed")
    
    for learning_rate in LEARNING_RATES:
        for batch_size in BATCH_SIZES:
            for method_name, config in METHODS.items():
                
                # Skip if already completed
                if is_completed(method_name, learning_rate, batch_size):
                    print(f"\n⏭️ SKIPPING: {method_name} (already completed)")
                    continue
                
                print(f"\n{'='*60}")
                print(f"Testing {method_name} with LR={learning_rate}, BS={batch_size}")
                print(f"Progress: {len(completed_runs)+1}/{total_experiments}")
                print('='*60)
                
                result = run_baseline_experiment(method_name, config, learning_rate, batch_size)
                if result is not None:
                    results.append(result)
                    completed_runs = load_checkpoint().get("completed_runs", [])
                    print(f"💾 Checkpoint saved. Progress: {len(completed_runs)}/{total_experiments}")
    
    # Final summary
    successful_results = [r for r in results if r.get("status") == "success"]
    
    print(f"\n{'='*60}")
    print("🎯 BASELINE RESULTS SUMMARY")
    print('='*60)
    
    for result in successful_results:
        print(f"✅ {result['method']}:")
        print(f"   Test Accuracy: {result.get('test_accuracy', 0):.4f}")
        print(f"   Test AUC: {result.get('test_auc', 0):.4f}")
    
    if successful_results:
        best_method = max(successful_results, key=lambda x: x.get('test_auc', 0))
        print(f"\n🏆 BEST BASELINE PERFORMER: {best_method['method']}")
        print(f"   AUC: {best_method.get('test_auc', 0):.4f}")
    
    print(f"\n📁 All results saved to: {BASE_DIR}")
    print(f"📊 Total experiments completed: {len(completed_runs)}/{total_experiments}")
    print("=" * 80)