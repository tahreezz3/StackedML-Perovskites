# -*- coding: utf-8 -*-
"""
Lightweight Test Script for Stacked ML Pipeline

Demonstrates the complete ML pipeline with minimal computational cost:
- Data loading and preprocessing
- Feature selection
- Hyperparameter tuning of base models
- Stacking ensemble creation and selection
- Evaluation and interpretation

Each step prints detailed information about values and decision logic.
"""

import time
import os
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split, KFold

# Import from project modules
from config import (
    DATA_FILE, TEST_SIZE, RANDOM_STATE, 
    FEATURE_SELECTION_METHOD, K_BEST_FEATURES, 
    STRATIFY_SPLIT, OUTPUT_DIR,
    SAVE_MODELS, SAVE_FEATURES, SAVE_RESULTS
)
from utils import run_optuna_study, get_compute_device_params
from model_definitions import (
    # Model getters
    get_base_regressors, get_base_classifiers,
    get_meta_regressor_candidates, get_meta_classifier_candidates,
    select_best_stack,
    # Optimizer functions - regression
    optimize_lgbm_reg, optimize_xgb_reg, optimize_rf_reg, optimize_gb_reg,
    optimize_ridge, optimize_bayes_ridge, optimize_mlp_reg, optimize_svr, optimize_knn_reg,
    # Optimizer functions - classification
    optimize_lgbm_cls, optimize_xgb_cls, optimize_logistic_regression, 
    optimize_rf_cls, optimize_gb_cls, optimize_mlp_cls, optimize_svc, optimize_knn_cls
)
from outer_cv import run_outer_cv_loop
from pipeline_steps import (
    aggregate_cv_results, select_final_model, 
    evaluate_on_test_set, run_shap_analysis, save_artifacts
)

# =============================================================================
# TEST CONFIGURATION - REDUCED FOR SPEED
# =============================================================================
print("\n========== PEROVSKITES ML PIPELINE TEST ==========\n")
print("TEST CONFIGURATION - REDUCED FOR SPEED:")

# Reduce computational cost
TEST_OPTUNA_TRIALS_MAIN = 3   # Reduced from default
TEST_OPTUNA_TRIALS_OTHER = 2  # Reduced from default
TEST_N_SPLITS_OUTER_CV = 2    # Reduced from default 5 folds
TEST_STACKING_CV_FOLDS = 2    # Reduced from default

print(f"  Data Source: {DATA_FILE}")
print(f"  Test Split: {TEST_SIZE}")
print(f"  Feature Selection: {FEATURE_SELECTION_METHOD}")
print(f"  Outer CV Folds: {TEST_N_SPLITS_OUTER_CV} (reduced)")
print(f"  Stacking CV Folds: {TEST_STACKING_CV_FOLDS} (reduced)")
print(f"  Optuna Trials: {TEST_OPTUNA_TRIALS_MAIN} (main) / {TEST_OPTUNA_TRIALS_OTHER} (other)")
print(f"  Random State: {RANDOM_STATE}")

# =============================================================================
# 1. DATA LOADING AND PREPARATION
# =============================================================================
print("\n1. LOADING DATA AND PREPARING SPLITS")
start_time = time.time()

try:
    print(f"  Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    print(f"  Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    
    # Display data summary
    print(f"  Feature columns: {df.shape[1]-2}")
    print(f"  Regression target: {df.columns[-2]}")
    print(f"  Classification target: {df.columns[-1]}")
    
    # Extract features and targets
    X = df.iloc[:, :-2]
    y_reg = df.iloc[:, -2]
    y_cls = df.iloc[:, -1]
    feature_names = X.columns.tolist()
    
    # Print target statistics
    print(f"  Regression target statistics:")
    print(f"    Min: {y_reg.min():.4f}, Max: {y_reg.max():.4f}, Mean: {y_reg.mean():.4f}, Std: {y_reg.std():.4f}")
    print(f"  Classification target distribution:")
    print(f"    {Counter(y_cls)}")
    
    # Create train/test split
    print("\n  Creating train/test split...")
    stratify_option = y_cls if STRATIFY_SPLIT and y_cls.nunique() > 1 else None
    if stratify_option is not None:
        print(f"    Using stratification for split based on classification target")
    else:
        print(f"    Not using stratification for split")
        
    X_train_val, X_test, y_train_val_reg, y_test_reg, y_train_val_cls, y_test_cls = train_test_split(
        X, y_reg, y_cls, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_option
    )
    
    print(f"    Training/validation set: {X_train_val.shape[0]} samples")
    print(f"    Test set: {X_test.shape[0]} samples")
    
    # Set up cross-validation
    print("\n  Setting up cross-validation...")
    kf_outer = KFold(n_splits=TEST_N_SPLITS_OUTER_CV, shuffle=True, random_state=RANDOM_STATE)
    print(f"    Created {TEST_N_SPLITS_OUTER_CV}-fold cross-validation for outer loop")

except Exception as e:
    print(f"ERROR: Failed to load data or prepare splits: {e}")
    raise

# =============================================================================
# 2. PREPARE MODELS AND OPTIMIZATION FUNCTIONS
# =============================================================================
print("\n2. PREPARING MODELS AND OPTIMIZATION FUNCTIONS")

# Get base models
print("  Initializing base regression models...")
MODEL_REGRESSORS = get_base_regressors(RANDOM_STATE)
for name in MODEL_REGRESSORS.keys():
    print(f"    ✓ {name}")

print("  Initializing base classification models...")
MODEL_CLASSIFIERS = get_base_classifiers(RANDOM_STATE)
for name in MODEL_CLASSIFIERS.keys():
    print(f"    ✓ {name}")

# Get meta-model candidates
print("  Initializing meta-regressor candidates...")
STACKING_META_REGRESSOR_CANDIDATES = get_meta_regressor_candidates(RANDOM_STATE)
for name in STACKING_META_REGRESSOR_CANDIDATES.keys():
    print(f"    ✓ {name}")

print("  Initializing meta-classifier candidates...")
STACKING_META_CLASSIFIER_CANDIDATES = get_meta_classifier_candidates(RANDOM_STATE)
for name in STACKING_META_CLASSIFIER_CANDIDATES.keys():
    print(f"    ✓ {name}")

# Set up optimization functions
print("  Setting up optimization functions...")
OPTIMIZATION_FUNCTIONS_REG = {
    'LGBM': optimize_lgbm_reg,
    'XGB': optimize_xgb_reg,
    'RandomForest': optimize_rf_reg,
    'GradientBoosting': optimize_gb_reg,
    'Ridge': optimize_ridge,
    'BayesianRidge': optimize_bayes_ridge,
    'MLP': optimize_mlp_reg,
    'SVR': optimize_svr,
    'KNN': optimize_knn_reg
}

OPTIMIZATION_FUNCTIONS_CLS = {
    'LGBM': optimize_lgbm_cls,
    'XGB': optimize_xgb_cls,
    'LogisticRegression': optimize_logistic_regression,
    'RandomForest': optimize_rf_cls,
    'GradientBoosting': optimize_gb_cls,
    'MLP': optimize_mlp_cls,
    'SVC': optimize_svc,
    'KNN': optimize_knn_cls
}
print(f"    ✓ {len(OPTIMIZATION_FUNCTIONS_REG)} regression optimizers")
print(f"    ✓ {len(OPTIMIZATION_FUNCTIONS_CLS)} classification optimizers")

# =============================================================================
# 3. RUN OUTER CROSS-VALIDATION LOOP
# =============================================================================
print("\n3. RUNNING OUTER CROSS-VALIDATION LOOP")
start_outer_cv = time.time()

try:
    # Execute the main cross-validation loop
    outer_fold_results_reg, outer_fold_results_cls, fold_selected_features_list, \
    fold_best_params_reg, fold_best_params_cls, fold_scalers, fold_selectors, \
    all_fold_models_reg, all_fold_models_cls = run_outer_cv_loop(
        X_train_val=X_train_val,
        y_train_val_reg=y_train_val_reg,
        y_train_val_cls=y_train_val_cls,
        kf_outer=kf_outer,
        feature_names=feature_names,
        FEATURE_SELECTION_METHOD=FEATURE_SELECTION_METHOD,
        K_BEST_FEATURES=K_BEST_FEATURES,
        TUNE_ALL_BASE_MODELS=True,  # Test all hyperparameter tuning
        OPTUNA_TRIALS_MAIN=TEST_OPTUNA_TRIALS_MAIN,
        OPTUNA_TRIALS_OTHER=TEST_OPTUNA_TRIALS_OTHER,
        MODEL_REGRESSORS=MODEL_REGRESSORS,
        MODEL_CLASSIFIERS=MODEL_CLASSIFIERS,
        STACKING_META_REGRESSOR_CANDIDATES=STACKING_META_REGRESSOR_CANDIDATES,
        STACKING_META_CLASSIFIER_CANDIDATES=STACKING_META_CLASSIFIER_CANDIDATES,
        STACKING_CV_FOLDS=TEST_STACKING_CV_FOLDS,
        get_compute_device_params=get_compute_device_params,
        run_optuna_study=run_optuna_study,
        select_best_stack=select_best_stack,
        OPTIMIZATION_FUNCTIONS_REG=OPTIMIZATION_FUNCTIONS_REG,
        OPTIMIZATION_FUNCTIONS_CLS=OPTIMIZATION_FUNCTIONS_CLS
    )
    
    outer_cv_time = time.time() - start_outer_cv
    print(f"\nOuter CV completed in {outer_cv_time:.2f} seconds")

except Exception as e:
    print(f"ERROR: Failed during outer CV: {e}")
    import traceback
    traceback.print_exc()
    raise

# =============================================================================
# 4. AGGREGATE CROSS-VALIDATION RESULTS
# =============================================================================
print("\n4. AGGREGATING CROSS-VALIDATION RESULTS")

try:
    # Get aggregate metrics across folds
    mean_r2, std_r2, mean_mae, std_mae, mean_acc, std_acc, mean_roc_auc, std_roc_auc = aggregate_cv_results(
        outer_fold_results_reg, outer_fold_results_cls
    )
    
    # Print detailed feature selection results
    print("\n  Feature Selection Results:")
    for i, features in enumerate(fold_selected_features_list):
        print(f"    Fold {i+1}: Selected {len(features)} features")
        print(f"      Top 5 features: {features[:5]}")
    
    # Print hyperparameter tuning results (sample)
    print("\n  Hyperparameter Tuning Results (Sample):")  
    for model_name, folds_params in fold_best_params_reg.items():
        print(f"    {model_name}:")
        # Show first fold only as example
        if 0 in folds_params:
            print(f"      Fold 1 Best Params: {folds_params[0]}")
        else:
            print("      No parameters found for Fold 1")

except Exception as e:
    print(f"ERROR: Failed to aggregate results: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# 5. SELECT FINAL MODELS
# =============================================================================
print("\n5. SELECTING FINAL MODELS BASED ON BEST FOLD PERFORMANCE")

try:
    # Select best models from cross-validation
    final_regressor, final_classifier, final_scaler, final_selector, \
    selected_features_final, best_fold_idx_reg, best_fold_idx_cls = select_final_model(
        outer_fold_results_reg, 
        outer_fold_results_cls,
        all_fold_models_reg,
        all_fold_models_cls,
        fold_scalers,
        fold_selectors,
        fold_selected_features_list,
        TEST_N_SPLITS_OUTER_CV
    )
    
    # Print detailed information about selected models
    if final_regressor:
        print(f"\n  Selected final regressor from fold {best_fold_idx_reg + 1}:")
        print(f"    Model type: {type(final_regressor).__name__}")
        print(f"    R2 score: {outer_fold_results_reg['R2'][best_fold_idx_reg]:.4f}")
        print(f"    MAE: {outer_fold_results_reg['MAE'][best_fold_idx_reg]:.4f}")
    
    if final_classifier:
        print(f"\n  Selected final classifier from fold {best_fold_idx_cls + 1}:")
        print(f"    Model type: {type(final_classifier).__name__}")
        print(f"    Accuracy: {outer_fold_results_cls['Accuracy'][best_fold_idx_cls]:.4f}")
        print(f"    ROC-AUC: {outer_fold_results_cls['ROC-AUC'][best_fold_idx_cls]:.4f}")
    
    if selected_features_final:
        print(f"\n  Selected {len(selected_features_final)} features for final models:")
        print(f"    Top 10: {selected_features_final[:10]}")

except Exception as e:
    print(f"ERROR: Failed to select final models: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# 6. EVALUATE ON TEST SET
# =============================================================================
print("\n6. EVALUATING ON TEST SET")

try:
    # Apply preprocessing and evaluate on test set
    X_test_scaled_df, X_test_sel_df = evaluate_on_test_set(
        X_test, y_test_reg, y_test_cls,
        final_regressor, final_classifier,
        final_scaler, final_selector,
        selected_features_final,
        feature_names
    )
    
    print("\n  Test set evaluation complete")
    if X_test_sel_df is not None:
        print(f"    Processed test data shape: {X_test_sel_df.shape}")

except Exception as e:
    print(f"ERROR: Failed to evaluate on test set: {e}")
    import traceback
    traceback.print_exc()
    X_test_scaled_df, X_test_sel_df = None, None

# =============================================================================
# 7. SHAP ANALYSIS
# =============================================================================
print("\n7. RUNNING SHAP ANALYSIS")

try:
    if X_test_sel_df is not None and final_regressor is not None:
        run_shap_analysis(
            final_regressor, final_classifier,
            X_test_sel_df,
            selected_features_final,
            X_train_val,
            y_train_val_reg,
            kf_outer,
            best_fold_idx_reg,
            final_scaler,
            final_selector,
            FEATURE_SELECTION_METHOD
        )
    else:
        print("  SHAP analysis skipped: missing test data or final models")

except Exception as e:
    print(f"ERROR: Failed to run SHAP analysis: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# 8. SAVE RESULTS
# =============================================================================
print("\n8. SAVING RESULTS")

try:
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"  Using output directory: {OUTPUT_DIR}")
    
    # Save models and results
    save_artifacts(
        final_regressor, final_classifier,
        selected_features_final,
        outer_fold_results_reg, outer_fold_results_cls,
        output_dir=OUTPUT_DIR,
        save_models=SAVE_MODELS,
        save_features=SAVE_FEATURES,
        save_results=SAVE_RESULTS
    )

except Exception as e:
    print(f"ERROR: Failed to save results: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# SUMMARY
# =============================================================================
total_time = time.time() - start_time
print("\n========== TEST EXECUTION SUMMARY ==========")
print(f"Total execution time: {total_time:.2f} seconds")
print(f"Cross-validation: {outer_cv_time:.2f} seconds ({outer_cv_time/total_time*100:.1f}% of total)")

print("\nKey Results:")
if 'mean_r2' in locals() and not np.isnan(mean_r2):
    print(f"  Mean R2: {mean_r2:.4f} +/- {std_r2:.4f}")
if 'mean_mae' in locals() and not np.isnan(mean_mae):
    print(f"  Mean MAE: {mean_mae:.4f} +/- {std_mae:.4f}")
if 'mean_acc' in locals() and not np.isnan(mean_acc):
    print(f"  Mean Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
if 'mean_roc_auc' in locals() and not np.isnan(mean_roc_auc):
    print(f"  Mean ROC-AUC: {mean_roc_auc:.4f} +/- {std_roc_auc:.4f}")

print("\nTest completed successfully!")
