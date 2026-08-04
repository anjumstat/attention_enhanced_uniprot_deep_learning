# -*- coding: utf-8 -*-
"""
Modified Original Code - WITH TEST.CSV SUPPORT
Saves NPY files, ROC curves, AND test.csv results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix, 
                             classification_report, roc_auc_score, roc_curve)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers, backend as K
import os
import time
import shutil
import json
import warnings
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

# =============================================
# CONFIGURATION - MODIFIED PATHS
# =============================================

# Path to combined train+val data
DATA_PATH = r"D:\uni_prot2\revision\data\clean_splits\homology_aware\combined_train_val.csv"

# Path to your dedicated test set
TEST_DATA_PATH = r"D:\uni_prot2\revision\data\clean_splits\homology_aware\test.csv"

# Output directory
BASE_DIR = r"D:\uni_prot2\revision\results\homology_aware_with_test"

# Hyperparameters (reduced for speed)
LEARNING_RATES = [0.01, 0.001,0.0001]  # [0.01, 0.001, 0.0001]
BATCH_SIZES = [32,64,128,256]        # [32, 64, 128, 256, 512]

# NOVEL ATTENTION-ENHANCED METHODS WITH 3 IMPORTANT ABLATIONS
METHODS = {
    'Attention_Enhanced_Basic': {'dir': 'results_Attention_Basic', 'attention_type': 'basic', 'layers': 2},
    'DNN_Baseline': {'dir': 'results_DNN_Baseline', 'attention_type': 'none', 'layers': 2},
    'Logistic_Baseline': {'dir': 'results_Logistic_Baseline', 'attention_type': 'none', 'layers': 1},
    'Ablation_No_Attention': {'dir': 'ablation_no_attention', 'attention_type': 'none', 'layers': 2},
    'Ablation_No_Residual': {'dir': 'ablation_no_residual', 'attention_type': 'basic_no_residual', 
                             'layers': 2, 'use_residual': False},
    'Ablation_50Percent_Data': {'dir': 'ablation_50percent_data', 'attention_type': 'basic', 
                                 'layers': 2, 'data_fraction': 0.5},
}

# =============================================================================
# ATTENTION MECHANISM
# =============================================================================

class FeatureAttention(layers.Layer):
    def __init__(self, attention_units=64, dropout_rate=0.2, use_residual=True, **kwargs):
        super(FeatureAttention, self).__init__(**kwargs)
        self.attention_units = attention_units
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        
    def build(self, input_shape):
        self.feature_dim = input_shape[-1]
        self.query_dense = layers.Dense(self.attention_units, activation='relu')
        self.key_dense = layers.Dense(self.attention_units, activation='relu')
        self.value_dense = layers.Dense(self.feature_dim, activation='linear')
        self.attention_dense = layers.Dense(1, activation='sigmoid')
        self.layer_norm = layers.LayerNormalization()
        self.dropout = layers.Dropout(self.dropout_rate)
        self.built = True
        
    def call(self, inputs, training=False):
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        attention_input = tf.concat([query, key], axis=-1)
        attention_scores = self.attention_dense(attention_input)
        attended_features = value * attention_scores
        if self.use_residual:
            attended_features = attended_features + inputs
        attended_features = self.layer_norm(attended_features)
        attended_features = self.dropout(attended_features, training=training)
        return attended_features
    
    def get_config(self):
        return {
            'attention_units': self.attention_units,
            'dropout_rate': self.dropout_rate,
            'use_residual': self.use_residual
        }

# =============================================================================
# STABILITY ANALYZER
# =============================================================================

class CorrectedStabilityAnalyzer:
    def __init__(self):
        self.stability_metrics = {}
    
    def extract_model_importance(self, model, X_sample=None):
        importance_scores = []
        try:
            for layer in model.layers:
                if isinstance(layer, layers.Dense) and layer.get_weights():
                    weights = layer.get_weights()
                    if weights and len(weights) > 0:
                        if 'input' in layer.name or 'dense_input' in layer.name:
                            layer_weights = weights[0]
                            if layer_weights.ndim == 2:
                                importance = np.mean(np.abs(layer_weights), axis=1)
                                importance_scores.append(importance)
                                break
            if X_sample is not None and len(importance_scores) == 0:
                grad_importance = self._gradient_importance(model, X_sample)
                if grad_importance is not None:
                    importance_scores.append(grad_importance)
            if not importance_scores:
                n_features = X_sample.shape[1] if X_sample is not None else 1024
                importance_scores.append(np.ones(n_features))
            if importance_scores:
                combined_importance = np.mean(np.array(importance_scores), axis=0)
                if np.max(combined_importance) > 0:
                    combined_importance = combined_importance / np.max(combined_importance)
                return combined_importance
        except Exception as e:
            print(f"Warning: Importance extraction failed: {e}")
        n_features = X_sample.shape[1] if X_sample is not None else 1024
        return np.ones(n_features)
    
    def _gradient_importance(self, model, X_sample):
        try:
            X_tensor = tf.convert_to_tensor(X_sample[:5], dtype=tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(X_tensor)
                predictions = model(X_tensor)
            gradients = tape.gradient(predictions, X_tensor)
            importance = tf.reduce_mean(tf.abs(gradients), axis=0)
            return importance.numpy()
        except:
            return None
    
    def calculate_jaccard_stability(self, feature_sets):
        if len(feature_sets) < 2:
            return 0.0
        jaccard_scores = []
        for i in range(len(feature_sets)):
            for j in range(i+1, len(feature_sets)):
                set1, set2 = feature_sets[i], feature_sets[j]
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                if union > 0:
                    jaccard_scores.append(intersection / union)
        return np.mean(jaccard_scores) if jaccard_scores else 0.0
    
    def calculate_rank_stability(self, importance_matrices):
        if len(importance_matrices) < 2:
            return 0.0
        correlations = []
        for i in range(len(importance_matrices)):
            for j in range(i+1, len(importance_matrices)):
                if len(importance_matrices[i]) == len(importance_matrices[j]):
                    corr, _ = spearmanr(importance_matrices[i], importance_matrices[j])
                    if not np.isnan(corr):
                        correlations.append(corr)
        return np.mean(correlations) if correlations else 0.0
    
    def comprehensive_stability_analysis(self, fold_models, X_sample=None, top_k=50):
        feature_sets = []
        importance_matrices = []
        print(f"🔍 Analyzing stability across {len(fold_models)} folds...")
        for i, model in enumerate(fold_models):
            importance = self.extract_model_importance(model, X_sample)
            if importance is not None and len(importance) > 0:
                top_features = set(np.argsort(importance)[-top_k:])
                feature_sets.append(top_features)
                importance_matrices.append(importance)
                print(f"  Fold {i+1}: Extracted {len(top_features)} top features")
            else:
                feature_sets.append(set())
                if X_sample is not None:
                    importance_matrices.append(np.ones(X_sample.shape[1]))
                else:
                    importance_matrices.append(np.ones(1024))
        n_folds = len(feature_sets)
        if n_folds < 2:
            return self._default_stability_metrics()
        jaccard_stability = self.calculate_jaccard_stability(feature_sets)
        rank_stability = self.calculate_rank_stability(importance_matrices)
        consistency_ratio = self._calculate_consistency_ratio(importance_matrices)
        feature_agreement = self._calculate_feature_agreement(feature_sets, len(importance_matrices[0]))
        stability_metrics = {
            'jaccard_stability': jaccard_stability,
            'rank_stability': rank_stability,
            'consistency_ratio': consistency_ratio,
            'feature_agreement': feature_agreement,
            'n_folds_analyzed': n_folds
        }
        weights = [0.3, 0.3, 0.2, 0.2]
        stability_metrics['overall_stability'] = (
            weights[0] * stability_metrics['jaccard_stability'] +
            weights[1] * stability_metrics['rank_stability'] + 
            weights[2] * stability_metrics['consistency_ratio'] +
            weights[3] * stability_metrics['feature_agreement']
        )
        print(f"✅ Stability analysis completed: {stability_metrics['overall_stability']:.4f}")
        return stability_metrics
    
    def _calculate_consistency_ratio(self, importance_matrices, threshold=0.01):
        if len(importance_matrices) == 0:
            return 0.0
        n_features = len(importance_matrices[0])
        consistent_count = 0
        for feature_idx in range(n_features):
            fold_importances = [imp[feature_idx] for imp in importance_matrices]
            mean_importance = np.mean(fold_importances)
            std_importance = np.std(fold_importances)
            if mean_importance > 0 and (std_importance / mean_importance) < 1.0:
                consistent_count += 1
        return consistent_count / n_features
    
    def _calculate_feature_agreement(self, feature_sets, total_features):
        if not feature_sets:
            return 0.0
        agreement_scores = []
        for feature_idx in range(total_features):
            presence_count = sum(1 for feature_set in feature_sets if feature_idx in feature_set)
            agreement = presence_count / len(feature_sets)
            agreement_scores.append(agreement)
        return np.mean(agreement_scores)
    
    def _default_stability_metrics(self):
        return {
            'jaccard_stability': 0.0,
            'rank_stability': 0.0,
            'consistency_ratio': 0.0,
            'feature_agreement': 0.0,
            'n_folds_analyzed': 0,
            'overall_stability': 0.0
        }

# =============================================================================
# MODEL BUILDERS
# =============================================================================

def build_corrected_attention_model(config, input_shape, num_classes, learning_rate):
    input_layer = layers.Input(shape=(input_shape,))
    x = input_layer
    attention_units = config.get('attention_units', 128)
    dropout_rate = config.get('dropout_rate', 0.2)
    use_residual = config.get('use_residual', True)
    x = layers.Dense(512, activation='relu', name='dense_input_1')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation='relu', name='dense_input_2')(x)
    x = layers.Dropout(dropout_rate)(x)
    attention_type = config['attention_type']
    if attention_type == 'basic':
        x = FeatureAttention(attention_units=attention_units, 
                            dropout_rate=dropout_rate,
                            use_residual=use_residual, 
                            name='feature_attention')(x)
    elif attention_type == 'basic_no_residual':
        x = FeatureAttention(attention_units=attention_units, 
                            dropout_rate=dropout_rate,
                            use_residual=False, 
                            name='feature_attention_no_residual')(x)
    for i in range(config['layers'] - 1):
        units = max(128 // (2 ** i), 32)
        x = layers.Dense(units, activation='relu', name=f'dense_post_{i}')(x)
        x = layers.Dropout(dropout_rate)(x)
    output_layer = layers.Dense(num_classes, activation='sigmoid', name='output')(x)
    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def build_baseline_model(model_type, input_shape, num_classes, learning_rate):
    if model_type == 'DNN_Baseline':
        model = models.Sequential([
            layers.Dense(512, activation='relu', input_shape=(input_shape,), name='dense_input_1'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu', name='dense_input_2'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu', name='dense_post_0'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='sigmoid', name='output')
        ])
    else:
        model = models.Sequential([
            layers.Dense(num_classes, activation='sigmoid', input_shape=(input_shape,), name='dense_input_1')
        ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def build_model(method, input_shape, num_classes, learning_rate, ablation_mode=None):
    if 'Attention' in method or 'Ablation' in method:
        config = METHODS[method]
        return build_corrected_attention_model(config, input_shape, num_classes, learning_rate)
    else:
        return build_baseline_model(method, input_shape, num_classes, learning_rate)

# =============================================================================
# NUMPY ENCODER
# =============================================================================

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

# =============================================================================
# PLOT ROC CURVES
# =============================================================================

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

# =============================================================================
# COMPREHENSIVE EXPERIMENT WITH TEST.CSV SUPPORT
# =============================================================================

def run_comprehensive_experiment(method_name, config, learning_rate, batch_size):
    """Modified to use test.csv for final evaluation"""
    
    data_fraction = config.get('data_fraction', 1.0)
    
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
        # =============================================
        # LOAD DATA - MODIFIED SECTION
        # =============================================
        print("📊 Loading data...")
        
        # Load combined train+val data
        train_data = pd.read_csv(DATA_PATH)
        
        # Load test data
        test_data = pd.read_csv(TEST_DATA_PATH)
        print(f"   Test data: {len(test_data)} samples ({test_data['is_enzyme'].sum()} enzymes)")
        
        # Apply data fraction ablation if specified
        if data_fraction < 1.0:
            print(f"⚠️ Using {data_fraction*100}% of data (Ablation)")
            train_data = train_data.sample(frac=data_fraction, random_state=42)
        
        # Extract features and labels
        X_train_all = train_data.iloc[:, :-1].values
        y_train_all = train_data.iloc[:, -1].values
        
        X_test_fixed = test_data.iloc[:, :-1].values
        y_test_fixed = test_data.iloc[:, -1].values
        
        feature_names = train_data.columns[:-1].tolist()
        
        print(f"   Training data: {X_train_all.shape[0]} samples")
        print(f"   Test data: {X_test_fixed.shape[0]} samples")
        
        # =============================================
        # SCALING
        # =============================================
        scaler = StandardScaler()
        X_train_all = scaler.fit_transform(X_train_all)
        X_test_fixed = scaler.transform(X_test_fixed)
        
        num_classes = 1
        y_train_all_cat = y_train_all.reshape(-1, 1)
        y_test_fixed_cat = y_test_fixed.reshape(-1, 1)
        
        # Save dataset info
        dataset_info = {
            'n_samples': int(X_train_all.shape[0]),
            'n_test_samples': int(X_test_fixed.shape[0]),
            'n_features': int(X_train_all.shape[1]),
            'n_classes': int(len(np.unique(y_train_all))),
            'class_distribution': {str(k): int(v) for k, v in dict(zip(*np.unique(y_train_all, return_counts=True))).items()},
            'test_class_distribution': {str(k): int(v) for k, v in dict(zip(*np.unique(y_test_fixed, return_counts=True))).items()},
            'data_fraction': data_fraction
        }
        
        with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f, indent=4, cls=NumpyEncoder)

        # =============================================
        # 10-FOLD CROSS VALIDATION
        # =============================================
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        fold_metrics = {
            'train': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'mcc': [], 'auc': []},
            'val': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'mcc': [], 'auc': []},
            'test_fixed': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'mcc': [], 'auc': []},
            'training_time': []
        }
        
        all_history, all_predictions, all_feature_importances = {}, {}, {}
        confusion_matrices, roc_data = [], []
        
        best_model_path = os.path.join(models_dir, f"Best_{method_name}_Model.h5")
        best_val_auc, best_fold = -np.inf, -1
        fold_models = []
        
        full_model_start_time = time.time()

        for fold, (train_index, val_index) in enumerate(skf.split(X_train_all, y_train_all)):
            print(f"  Fold {fold+1}/10")
            
            X_train_fold, X_val_fold = X_train_all[train_index], X_train_all[val_index]
            y_train_fold, y_val_fold = y_train_all_cat[train_index], y_train_all_cat[val_index]
            
            model = build_model(method_name, X_train_all.shape[1], num_classes, learning_rate)
            
            early_stop = callbacks.EarlyStopping(
                monitor='val_auc', patience=10, min_delta=0.001, 
                mode='max', restore_best_weights=True, verbose=0
            )
            
            start_time = time.time()
            history = model.fit(
                X_train_fold, y_train_fold,
                epochs=50, batch_size=batch_size,
                validation_data=(X_val_fold, y_val_fold),
                verbose=0, callbacks=[early_stop]
            )
            
            all_history[f'fold{fold+1}'] = history.history
            
            # =============================================
            # EVALUATION ON TRAIN, VAL, AND TEST (FIXED)
            # =============================================
            datasets = [
                ('train', X_train_fold, y_train_fold),
                ('val', X_val_fold, y_val_fold),
                ('test_fixed', X_test_fixed, y_test_fixed_cat)
            ]
            
            for dataset_name, X_data, y_data in datasets:
                y_pred = model.predict(X_data, verbose=0)
                y_pred_classes = (y_pred > 0.5).astype(int).flatten()
                y_true = y_data.flatten()
                
                fold_metrics[dataset_name]['accuracy'].append(accuracy_score(y_true, y_pred_classes))
                fold_metrics[dataset_name]['precision'].append(precision_score(y_true, y_pred_classes, zero_division=0))
                fold_metrics[dataset_name]['recall'].append(recall_score(y_true, y_pred_classes, zero_division=0))
                fold_metrics[dataset_name]['f1'].append(f1_score(y_true, y_pred_classes, zero_division=0))
                fold_metrics[dataset_name]['mcc'].append(matthews_corrcoef(y_true, y_pred_classes))
                fold_metrics[dataset_name]['auc'].append(roc_auc_score(y_true, y_pred))
            
            # =============================================
            # ROC DATA FOR TEST (FIXED)
            # =============================================
            y_test_pred = model.predict(X_test_fixed, verbose=0)
            y_test_true = y_test_fixed.flatten()
            fpr, tpr, _ = roc_curve(y_test_true, y_test_pred)
            auc_score = roc_auc_score(y_test_true, y_test_pred)
            roc_data.append({'fpr': fpr, 'tpr': tpr, 'fold': fold+1, 'auc': auc_score})
            
            # Store predictions
            all_predictions[f'fold{fold+1}'] = {
                'train': {'true': y_train_fold.flatten(), 'pred': (model.predict(X_train_fold, verbose=0) > 0.5).astype(int).flatten(), 'prob': model.predict(X_train_fold, verbose=0).flatten()},
                'val': {'true': y_val_fold.flatten(), 'pred': (model.predict(X_val_fold, verbose=0) > 0.5).astype(int).flatten(), 'prob': model.predict(X_val_fold, verbose=0).flatten()},
                'test_fixed': {'true': y_test_true, 'pred': (y_test_pred > 0.5).astype(int).flatten(), 'prob': y_test_pred.flatten()}
            }
            
            # Store confusion matrix for test
            cm = confusion_matrix(y_test_true, (y_test_pred > 0.5).astype(int).flatten())
            confusion_matrices.append(cm)
            
            # Feature importance
            analyzer = CorrectedStabilityAnalyzer()
            feature_importance = analyzer.extract_model_importance(model, X_test_fixed[:10])
            if feature_importance is not None:
                all_feature_importances[f'fold{fold+1}'] = feature_importance
            
            fold_models.append(model)
            fold_time = time.time() - start_time
            fold_metrics['training_time'].append(fold_time)
            
            model.save(os.path.join(models_dir, f"fold{fold+1}_model.h5"))
            
            val_auc = np.max(history.history['val_auc'])
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_fold = fold
                model.save(best_model_path)
            
            del model
            tf.keras.backend.clear_session()

        full_model_time_minutes = (time.time() - full_model_start_time) / 60
        
        # =============================================
        # STABILITY ANALYSIS
        # =============================================
        print("📈 Running stability analysis...")
        stability_analyzer = CorrectedStabilityAnalyzer()
        stability_metrics = stability_analyzer.comprehensive_stability_analysis(fold_models, X_test_fixed[:10])
        
        # =============================================
        # SAVE NPY FILES
        # =============================================
        print("💾 Saving NPY files...")
        
        roc_data_array = np.array([(d['fold'], d['fpr'], d['tpr'], d['auc']) for d in roc_data], dtype=object)
        np.save(os.path.join(npy_dir, "roc_data_all_folds.npy"), roc_data_array)
        
        for i, roc_d in enumerate(roc_data):
            fold_roc = {
                'fold': roc_d['fold'],
                'fpr': roc_d['fpr'],
                'tpr': roc_d['tpr'],
                'auc': roc_d['auc']
            }
            np.save(os.path.join(npy_dir, f"roc_data_fold_{roc_d['fold']}.npy"), fold_roc)
        
        np.save(os.path.join(npy_dir, "all_predictions.npy"), all_predictions)
        if all_feature_importances:
            np.save(os.path.join(npy_dir, "feature_importances.npy"), all_feature_importances)
        np.save(os.path.join(npy_dir, "training_history.npy"), all_history)
        np.save(os.path.join(npy_dir, "confusion_matrices.npy"), np.array(confusion_matrices))
        np.save(os.path.join(npy_dir, "fold_metrics.npy"), fold_metrics)
        np.save(os.path.join(npy_dir, "stability_metrics.npy"), stability_metrics)
        
        # =============================================
        # GENERATE ROC CURVES
        # =============================================
        print("📈 Generating ROC curves...")
        plot_roc_curves(roc_data, plots_dir, method_name)
        
        # =============================================
        # SAVE CSV FILES
        # =============================================
        print("💾 Saving CSV files...")
        
        # 1. Average_Metrics_With_Std.csv
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
        
        # 2. Test_Metrics_Summary.csv (NEW - dedicated test results)
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
        
        # 3. Training_Time_Stability_Summary.csv
        time_df = pd.DataFrame({
            'Method': [method_name],
            'Learning_Rate': [learning_rate],
            'Batch_Size': [batch_size],
            'Total_Training_Time_Minutes': [full_model_time_minutes],
            'Average_Fold_Time_Seconds': [np.mean(fold_metrics['training_time'])],
            'Std_Fold_Time_Seconds': [np.std(fold_metrics['training_time'])],
            'Overall_Stability_Score': [stability_metrics['overall_stability']],
            'Jaccard_Stability': [stability_metrics['jaccard_stability']],
            'Rank_Stability': [stability_metrics['rank_stability']],
            'Consistency_Ratio': [stability_metrics['consistency_ratio']],
            'Feature_Agreement': [stability_metrics['feature_agreement']]
        })
        time_df.to_csv(os.path.join(csv_dir, "Training_Time_Stability_Summary.csv"), index=False)
        
        # 4. Best_Model_Performance.csv
        if best_fold >= 0:
            custom_objects = {'FeatureAttention': FeatureAttention}
            try:
                best_model = tf.keras.models.load_model(best_model_path, custom_objects=custom_objects)
            except:
                best_model = tf.keras.models.load_model(best_model_path)
            
            # Calculate on best fold's train/val and test
            metrics_sets = {}
            
            for dataset_name, X_data, y_data in [
                ('train', X_train_all[train_index], y_train_all[train_index]),
                ('validation', X_train_all[val_index], y_train_all[val_index]),
                ('test_fixed', X_test_fixed, y_test_fixed)
            ]:
                y_pred = best_model.predict(X_data, verbose=0)
                y_pred_classes = (y_pred > 0.5).astype(int).flatten()
                y_true = y_data.flatten()
                
                metrics_sets[dataset_name] = {
                    'accuracy': accuracy_score(y_true, y_pred_classes),
                    'precision': precision_score(y_true, y_pred_classes, zero_division=0),
                    'recall': recall_score(y_true, y_pred_classes, zero_division=0),
                    'f1': f1_score(y_true, y_pred_classes, zero_division=0),
                    'mcc': matthews_corrcoef(y_true, y_pred_classes),
                    'auc': roc_auc_score(y_true, y_pred),
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'best_fold': best_fold + 1
                }
            
            best_metrics_df = pd.DataFrame(metrics_sets).T
            best_metrics_df.to_csv(os.path.join(csv_dir, "Best_Model_Performance.csv"))
        
        # 5. All_Predictions_Detailed.csv
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
        
        # 6. Feature_Importance_Analysis.csv
        if all_feature_importances:
            feature_importance_data = []
            for fold, importance in all_feature_importances.items():
                for feature_idx, imp_value in enumerate(importance):
                    feature_name = f'Feature_{feature_idx}'
                    if feature_names and feature_idx < len(feature_names):
                        feature_name = feature_names[feature_idx]
                    feature_importance_data.append({
                        'fold': fold,
                        'feature_index': int(feature_idx),
                        'feature_name': feature_name,
                        'importance': float(imp_value)
                    })
            if feature_importance_data:
                feature_importance_df = pd.DataFrame(feature_importance_data)
                feature_importance_df.to_csv(os.path.join(csv_dir, "Feature_Importance_Analysis.csv"), index=False)
        
        # 7. Experiment_Configuration.csv
        config_df = pd.DataFrame({
            'Parameter': ['Method', 'Learning_Rate', 'Batch_Size', 'Attention_Type', 'Number_of_Folds', 'Test_Size', 'Data_Fraction'],
            'Value': [method_name, learning_rate, batch_size, 
                     config.get('attention_type', 'none'),
                     10, len(y_test_fixed), data_fraction]
        })
        config_df.to_csv(os.path.join(csv_dir, "Experiment_Configuration.csv"), index=False)
        
        # 8. Experiment_Summary.csv
        summary_stats = {
            'Best_Validation_AUC': float(best_val_auc),
            'Best_Fold': int(best_fold + 1),
            'Mean_Test_Accuracy': float(np.mean(fold_metrics['test_fixed']['accuracy'])),
            'Std_Test_Accuracy': float(np.std(fold_metrics['test_fixed']['accuracy'])),
            'Mean_Test_AUC': float(np.mean(fold_metrics['test_fixed']['auc'])),
            'Std_Test_AUC': float(np.std(fold_metrics['test_fixed']['auc'])),
            'Mean_Training_Time_Per_Fold': float(np.mean(fold_metrics['training_time'])),
            'Total_Training_Time_Minutes': float(full_model_time_minutes),
            'Overall_Stability_Score': float(stability_metrics['overall_stability']),
            'Jaccard_Stability': float(stability_metrics['jaccard_stability']),
            'Rank_Stability': float(stability_metrics['rank_stability']),
            'Consistency_Ratio': float(stability_metrics['consistency_ratio']),
            'Feature_Agreement': float(stability_metrics['feature_agreement']),
            'Number_of_Folds_Analyzed': int(stability_metrics['n_folds_analyzed']),
            'Test_Size': int(len(y_test_fixed))
        }
        
        summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
        summary_df.to_csv(os.path.join(csv_dir, "Experiment_Summary.csv"), index=False)
        
        print(f"✅ {method_name} completed successfully")
        print(f"   Best validation AUC: {best_val_auc:.4f}")
        print(f"   Mean Test Accuracy: {np.mean(fold_metrics['test_fixed']['accuracy']):.4f} ± {np.std(fold_metrics['test_fixed']['accuracy']):.4f}")
        print(f"   Mean Test AUC: {np.mean(fold_metrics['test_fixed']['auc']):.4f} ± {np.std(fold_metrics['test_fixed']['auc']):.4f}")
        print(f"   Overall stability: {stability_metrics['overall_stability']:.4f}")
        print(f"   NPY files saved for all 10 folds in: {npy_dir}")
        
        return {
            "status": "success", 
            "method": method_name, 
            "test_accuracy": np.mean(fold_metrics['test_fixed']['accuracy']),
            "test_auc": np.mean(fold_metrics['test_fixed']['auc']),
            "stability_score": stability_metrics['overall_stability']
        }
    
    except Exception as e:
        print(f"❌ Error in {method_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "method": method_name, "error": str(e)}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    if os.path.exists(BASE_DIR):
        shutil.rmtree(BASE_DIR)
    os.makedirs(BASE_DIR, exist_ok=True)
    
    print("=" * 80)
    print("ATTENTION-ENHANCED DEEP LEARNING - WITH TEST.CSV SUPPORT")
    print("=" * 80)
    print(f"📁 NPY files will be saved for all 10 folds")
    print(f"📊 ROC curves will be generated automatically")
    print(f"📁 Test results will use your test.csv")
    print(f"Base directory: {BASE_DIR}")
    print("=" * 80)
    
    results = []
    
    for learning_rate in LEARNING_RATES:
        for batch_size in BATCH_SIZES:
            for method_name, config in METHODS.items():
                print(f"\n{'='*60}")
                print(f"Testing {method_name} with LR={learning_rate}, BS={batch_size}")
                if 'Attention' in method_name:
                    print("⭐ ATTENTION-ENHANCED METHOD")
                elif 'Ablation' in method_name:
                    print("🔬 ABLATION STUDY")
                print('='*60)
                
                result = run_comprehensive_experiment(method_name, config, learning_rate, batch_size)
                results.append(result)
    
    # Final summary
    successful_methods = [r for r in results if r["status"] == "success"]
    
    print(f"\n{'='*60}")
    print("🎯 FINAL RESULTS SUMMARY (USING test.csv)")
    print('='*60)
    
    for result in successful_methods:
        print(f"✅ {result['method']}:")
        print(f"   Test Accuracy: {result.get('test_accuracy', 0):.4f}")
        print(f"   Test AUC: {result.get('test_auc', 0):.4f}")
        print(f"   Stability Score: {result.get('stability_score', 0):.4f}")
        print()
    
    if successful_methods:
        best_method = max(successful_methods, key=lambda x: x.get('test_auc', 0))
        print(f"🏆 BEST PERFORMER: {best_method['method']}")
        print(f"   AUC: {best_method.get('test_auc', 0):.4f}")
        print(f"   Stability: {best_method.get('stability_score', 0):.4f}")
    
    print(f"\n📁 All results saved to: {BASE_DIR}")
    print("📁 NPY files for all 10 folds are in the 'npy_files' subdirectory")
    print("📊 ROC curves are in the 'plots' subdirectory")
    print("📊 Test results use your test.csv for final evaluation")
    print("=" * 80)