# -*- coding: utf-8 -*-
"""
Test script for outer_cv.py functionality.
Focuses on testing run_outer_cv_loop with all available models.
"""

# =============================================================================
# 1. LIBRARY IMPORTS (Reduced for testing)
# =============================================================================
import pandas as pd
import time
import numpy as np # Added numpy
from collections import Counter, defaultdict

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler # Done inside outer_cv
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, roc_auc_score, mean_squared_error # Done inside outer_cv
from sklearn.ensemble import StackingRegressor, StackingClassifier, RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge, BayesianRidge #, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression, mutual_info_regression # Done inside outer_cv

import lightgbm
import xgboost


import config # <<< Added import

from utils import run_optuna_study, get_compute_device_params

from config import * # Import necessary configurations
from model_definitions import (
    get_base_regressors, get_base_classifiers,
    get_meta_regressor_candidates, get_meta_classifier_candidates,
    get_final_model_instance, # Not needed for this test
    # Import *only* the optimization functions we need
    optimize_lgbm_reg, optimize_xgb_reg,
    optimize_rf_reg, optimize_gb_reg, 
    optimize_bayes_ridge,
    optimize_lgbm_cls, optimize_xgb_cls, optimize_logistic_regression, optimize_rf_cls, optimize_gb_cls,
    optimize_mlp_reg,
    optimize_svr,
    optimize_knn_reg,
    optimize_mlp_cls,
    optimize_svc,
    optimize_knn_cls
)
from outer_cv import run_outer_cv_loop
# from pipeline_steps import * # Not needed for this test


# =============================================================================
# 2. DATA LOADING AND INITIAL SPLIT (Same as main.py)
# =============================================================================
print(f"--- 2. Loading Data from {DATA_FILE} ---")
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_FILE}")
    print("Please ensure the data file exists and the path in config.py is correct.")
    exit()


# Assuming last two columns are targets (regression then classification)
if df.shape[1] < 3:
    print("Error: Data file requires at least 3 columns (features + regression target + classification target).")
    exit()

X = df.iloc[:, :-2]
y_reg = df.iloc[:, -2]
y_cls = df.iloc[:, -1] # Keep for stratify split, but won't be used in CV loop test
feature_names = X.columns.tolist()

# --- Hold-out Test Set (Keep split logic, but test set won't be used here) ---
X_train_val, X_test, y_train_val_reg, y_test_reg, y_train_val_cls, y_test_cls = train_test_split(
    X, y_reg, y_cls, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_cls if STRATIFY_SPLIT and y_cls.nunique() > 1 else None
)

print(f"Using Train/Val data = {X_train_val.shape[0]} samples for Outer CV test")
# print(f"Test data = {X_test.shape[0]} samples (not used in this test)") # Commented out


# =============================================================================
# 3. OUTER CROSS-VALIDATION SETUP (Same as main.py)
# =============================================================================
print(f"\n--- 3. Setting up Outer {N_SPLITS_OUTER_CV}-Fold Cross-Validation ---")
kf_outer = KFold(n_splits=N_SPLITS_OUTER_CV, shuffle=True, random_state=RANDOM_STATE)

# =============================================================================
# 4. PREPARE INPUTS FOR run_outer_cv_loop (All Models)
# =============================================================================
print(f"\n--- 4. Preparing Inputs for Outer CV Loop Test ---")

# === Test-Specific Model/Optimization Definitions ===
# Import necessary functions and classes
from model_definitions import (
    get_base_regressors,
    get_meta_regressor_candidates,
    get_base_classifiers, 
    get_meta_classifier_candidates, 
    # Regression Optimizers
    optimize_lgbm_reg, 
    optimize_xgb_reg,  
    optimize_rf_reg,
    optimize_gb_reg,
    optimize_ridge, 
    optimize_bayes_ridge,
    optimize_mlp_reg,
    optimize_svr,
    optimize_knn_reg,
    # Classification Optimizers 
    optimize_lgbm_cls, optimize_xgb_cls, optimize_logistic_regression, optimize_rf_cls, optimize_gb_cls,
    optimize_mlp_cls,
    optimize_svc,
    optimize_knn_cls
)

# Define the set of models for THIS test run
MODEL_REGRESSORS_TEST = {
    'LGBM': get_base_regressors(RANDOM_STATE)['LGBM'],
    'XGB': get_base_regressors(RANDOM_STATE)['XGB'],
    'Ridge': get_base_regressors(RANDOM_STATE)['Ridge'],
    'RandomForest': get_base_regressors(RANDOM_STATE)['RandomForest'],
    'GradientBoosting': get_base_regressors(RANDOM_STATE)['GradientBoosting'],
    'BayesianRidge': get_base_regressors(RANDOM_STATE)['BayesianRidge'],
    'MLP': get_base_regressors(RANDOM_STATE)['MLP'],
    'SVR': get_base_regressors(RANDOM_STATE)['SVR'],
    'KNN': get_base_regressors(RANDOM_STATE)['KNN']
}

STACKING_META_REGRESSOR_CANDIDATES_TEST = {
    'XGB': get_meta_regressor_candidates(RANDOM_STATE)['XGB'],
    'Ridge': get_meta_regressor_candidates(RANDOM_STATE)['Ridge'],
    'BayesianRidge': get_meta_regressor_candidates(RANDOM_STATE)['BayesianRidge'],
    'RandomForest': get_meta_regressor_candidates(RANDOM_STATE)['RandomForest'],
    'LGBM': get_meta_regressor_candidates(RANDOM_STATE)['LGBM']
}

MODEL_CLASSIFIERS_TEST = get_base_classifiers(RANDOM_STATE)

STACKING_META_CLASSIFIER_CANDIDATES_TEST = {
    'LogisticRegression': get_meta_classifier_candidates(RANDOM_STATE)['LogisticRegression'],
    'RandomForest': get_meta_classifier_candidates(RANDOM_STATE)['RandomForest'],
    'LGBM': get_meta_classifier_candidates(RANDOM_STATE)['LGBM'],
    'XGB': get_meta_classifier_candidates(RANDOM_STATE)['XGB'],
    'SVC': get_meta_classifier_candidates(RANDOM_STATE)['SVC'],
    'KNN': get_meta_classifier_candidates(RANDOM_STATE)['KNN']
}

# Define Optimization functions for the selected test models
OPTIMIZATION_FUNCTIONS_REG_TEST = {
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

OPTIMIZATION_FUNCTIONS_CLS_TEST = {
    'LGBM': optimize_lgbm_cls,
    'XGB': optimize_xgb_cls,
    'LogisticRegression': optimize_logistic_regression,
    'RandomForest': optimize_rf_cls,
    'GradientBoosting': optimize_gb_cls,
    'MLP': optimize_mlp_cls,
    'SVC': optimize_svc,
    'KNN': optimize_knn_cls
}

# --- Other Parameters ---
# Use parameters from config.py where possible
FEATURE_SELECTION_METHOD_TEST = FEATURE_SELECTION_METHOD # Or set to 'none' to speed up test
K_BEST_FEATURES_TEST = K_BEST_FEATURES
TUNE_ALL_BASE_MODELS_TEST = TUNE_ALL_BASE_MODELS # Keep True to test tuning
OPTUNA_TRIALS_MAIN_TEST = 5 # REDUCED trials for faster testing
OPTUNA_TRIALS_OTHER_TEST = 3 # REDUCED trials for faster testing

print(f"  Feature Selection: {FEATURE_SELECTION_METHOD_TEST}")
print(f"  Tune Base Models: {TUNE_ALL_BASE_MODELS_TEST}")
print(f"  Optuna Trials (Main/Other): {OPTUNA_TRIALS_MAIN_TEST} / {OPTUNA_TRIALS_OTHER_TEST}")


# =============================================================================
# 5. RUN OUTER CROSS-VALIDATION LOOP (Test Call)
# =============================================================================
print(f"\n--- 5. Running Outer CV Loop Test ---")
start_time_test = time.time()

# Call the outer loop function with the test setup
try:
    (
        outer_fold_results_reg,
        outer_fold_results_cls, 
        fold_selected_features_list,
        fold_best_params_reg,
        fold_best_params_cls, 
        fold_scalers,
        fold_selectors,
        all_fold_models_reg,
        all_fold_models_cls 
    ) = run_outer_cv_loop(
        X_train_val=X_train_val,
        y_train_val_reg=y_train_val_reg,
        y_train_val_cls=y_train_val_cls, 
        kf_outer=kf_outer,
        feature_names=feature_names,
        FEATURE_SELECTION_METHOD=FEATURE_SELECTION_METHOD_TEST,
        K_BEST_FEATURES=K_BEST_FEATURES_TEST,
        TUNE_ALL_BASE_MODELS=TUNE_ALL_BASE_MODELS_TEST,
        OPTUNA_TRIALS_MAIN=OPTUNA_TRIALS_MAIN_TEST, 
        OPTUNA_TRIALS_OTHER=OPTUNA_TRIALS_OTHER_TEST, 
        MODEL_REGRESSORS=MODEL_REGRESSORS_TEST, 
        MODEL_CLASSIFIERS=MODEL_CLASSIFIERS_TEST, 
        STACKING_META_REGRESSOR_CANDIDATES=STACKING_META_REGRESSOR_CANDIDATES_TEST, 
        STACKING_META_CLASSIFIER_CANDIDATES=STACKING_META_CLASSIFIER_CANDIDATES_TEST, 
        STACKING_CV_FOLDS=config.STACKING_CV_FOLDS,
        get_compute_device_params=get_compute_device_params,
        run_optuna_study=run_optuna_study,
        select_best_stack=select_best_stack,
        OPTIMIZATION_FUNCTIONS_REG=OPTIMIZATION_FUNCTIONS_REG_TEST, 
        OPTIMIZATION_FUNCTIONS_CLS=OPTIMIZATION_FUNCTIONS_CLS_TEST 
    )

    end_time_test = time.time()
    print(f"\n--- Outer CV Loop Test Completed in {end_time_test - start_time_test:.2f} seconds ---")

    # =============================================================================
    # 6. DISPLAY RESULTS (Simplified)
    # =============================================================================
    print("\n--- 6. Test Results ---")

    # Regression Results
    print("\nRegression Fold Results:")
    if outer_fold_results_reg:
        for metric, values in outer_fold_results_reg.items():
            print(f"  {metric}: {values}")
        # Calculate and print means/stds if desired
        mean_r2 = np.mean(outer_fold_results_reg.get('R2', [np.nan]))
        std_r2 = np.std(outer_fold_results_reg.get('R2', [np.nan]))
        mean_mae = np.mean(outer_fold_results_reg.get('MAE', [np.nan]))
        std_mae = np.std(outer_fold_results_reg.get('MAE', [np.nan]))
        print(f"\n  Aggregated Regression:")
        print(f"    Mean R2: {mean_r2:.4f} +/- {std_r2:.4f}")
        print(f"    Mean MAE: {mean_mae:.4f} +/- {std_mae:.4f}")
    else:
        print("  No regression results collected.")

    print("\nFold Selected Features (First 10):")
    for i, features in enumerate(fold_selected_features_list):
        print(f"  Fold {i+1}: {features[:10]}...")

    print("\nFold Best Regressor Params (Example):")
    if fold_best_params_reg:
        for model_name, folds_params in fold_best_params_reg.items():
             print(f"  {model_name}:")
             for fold_idx, params in folds_params.items():
                 # Print only the first fold's params for brevity
                 if fold_idx == 0:
                     print(f"    Fold 1 Best Params: {params}")
                 break # Only show first fold

    print("\nStored Scalers count:", len(fold_scalers))
    print("Stored Selectors count:", len(fold_selectors))
    print("Stored Regressor Models count:", len(all_fold_models_reg))

    # Classification results are expected to be empty/NaN
    # print("\nClassification Fold Results:")
    # if outer_fold_results_cls:
    #     for metric, values in outer_fold_results_cls.items():
    #         print(f"  {metric}: {values}")
    # else:
    #      print("  No classification results collected (as expected).")

except Exception as e:
    print(f"\n--- ERROR during Outer CV Loop Test ---")
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Test Script Finished ---")