# =============================================================================
# outer_cv.py
# =============================================================================


import time
import pandas as pd
import numpy as np # Added numpy
from collections import Counter, defaultdict # Added defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression, mutual_info_regression
# Stacking imports moved inside the function where needed, or ensure they are passed
from sklearn.metrics import (r2_score, mean_absolute_error, accuracy_score, roc_auc_score)

# Import config variables directly (alternative: pass a config object)
from config import (
    RANDOM_STATE,
    OPTUNA_TIMEOUT,
    N_SPLITS_OUTER_CV, # Used for printing fold numbers
    STACKING_CV_FOLDS, # Added parameter
    # Add any other config vars needed *directly* within the loop if not passed
)
# Import necessary functions/definitions (alternative: pass them as arguments)
# from model_definitions import get_final_model_instance # Might not be needed inside loop
# from utils import run_optuna_study # Assume passed as arg
# from model_tuning import select_best_stack # Assume passed as arg

# Import model classes (if not dynamically created via get_final_model_instance)
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb


def run_outer_cv_loop(
    X_train_val,
    y_train_val_reg,
    y_train_val_cls,
    kf_outer,
    feature_names, # Added
    # Config params (explicitly passed)
    FEATURE_SELECTION_METHOD,
    K_BEST_FEATURES,
    TUNE_ALL_BASE_MODELS,
    OPTUNA_TRIALS_MAIN,
    OPTUNA_TRIALS_OTHER,
    MODEL_REGRESSORS, # Dict defining base regressors
    MODEL_CLASSIFIERS, # Dict defining base classifiers
    STACKING_META_REGRESSOR_CANDIDATES,
    STACKING_META_CLASSIFIER_CANDIDATES,
    STACKING_CV_FOLDS, # Added parameter
    # Helper functions (passed as arguments)
    get_compute_device_params, # Function to get compute settings
    run_optuna_study, # Function to run hyperparameter tuning
    select_best_stack, # Function to select best stacking model
    OPTIMIZATION_FUNCTIONS_REG, # Dict mapping model names to objective functions
    OPTIMIZATION_FUNCTIONS_CLS # Dict mapping model names to objective functions
):
    """
    Runs the entire Outer Cross-Validation loop, performing:
      - Train/val splitting for the fold
      - Scaling
      - Feature Selection
      - Hyperparameter Tuning for base models
      - Stacking Ensemble training & selection
      - Performance evaluation on the fold's validation set

    Returns:
      outer_fold_results_reg (dict): Dictionary containing lists of R2, MAE per fold.
      outer_fold_results_cls (dict): Dictionary containing lists of Accuracy, ROC-AUC per fold.
      fold_selected_features_list (list): List of selected feature names for each fold.
      fold_best_params_reg (dict): Dictionary storing best hyperparameters for each regressor type across folds.
      fold_best_params_cls (dict): Dictionary storing best hyperparameters for each classifier type across folds.
      fold_scalers (list): List of fitted scalers for each fold.
      fold_selectors (list): List of fitted feature selectors for each fold.
      all_fold_models_reg (list): List of trained best regressor stack models for each fold.
      all_fold_models_cls (list): List of trained best classifier stack models for each fold.
    """

    # Store results from each outer fold
    outer_fold_results_reg = defaultdict(list) # Stores R2, MAE per fold
    outer_fold_results_cls = defaultdict(list) # Stores Accuracy, ROC-AUC per fold
    fold_selected_features_list = [] # Store selected features per fold
    fold_best_params_reg = defaultdict(lambda: defaultdict(list)) # Store best params for each regressor type across folds
    fold_best_params_cls = defaultdict(lambda: defaultdict(list)) # Store best params for each classifier type across folds
    fold_scalers = [] # Store scalers
    fold_selectors = [] # Store selectors
    all_fold_models_reg = [] # Store best regressor stack model per fold
    all_fold_models_cls = [] # Store best classifier stack model per fold

    # Get compute params once (assuming they don't change per fold)
    COMPUTE_PARAMS = get_compute_device_params()

    # Begin outer CV
    start_time_cv = time.time()

    for fold, (train_outer_idx, val_outer_idx) in enumerate(kf_outer.split(X_train_val, y_train_val_reg)): # Split based on X
        try:
            print(f"\n===== Outer Fold {fold+1}/{N_SPLITS_OUTER_CV} ====")
            fold_start_time = time.time()

            X_tr_fold, X_va_fold = X_train_val.iloc[train_outer_idx], X_train_val.iloc[val_outer_idx]
            y_tr_reg_fold, y_va_reg_fold = y_train_val_reg.iloc[train_outer_idx], y_train_val_reg.iloc[val_outer_idx]
            y_tr_cls_fold, y_va_cls_fold = y_train_val_cls.iloc[train_outer_idx], y_train_val_cls.iloc[val_outer_idx]

            print(f"  Train fold shape: {X_tr_fold.shape}, Validation fold shape: {X_va_fold.shape}")
            print(f"  Train fold class distribution: {Counter(y_tr_cls_fold)}")
            print(f"  Validation fold class distribution: {Counter(y_va_cls_fold)}")

            # --- 5.2 Preprocessing: Scaling (Fit on fold's train data) ---
            print(f"\n  --- 5.2 Scaling Features (Fold {fold+1}) ---")
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr_fold)
            X_va_scaled = scaler.transform(X_va_fold)
            # Keep DataFrames for easier feature selection referencing
            X_tr_scaled_df = pd.DataFrame(X_tr_scaled, index=X_tr_fold.index, columns=feature_names)
            X_va_scaled_df = pd.DataFrame(X_va_scaled, index=X_va_fold.index, columns=feature_names)
            fold_scalers.append(scaler)

            # --- 5.3 Feature Selection (On fold's scaled train data) ---
            print(f"\n  --- 5.3 Feature Selection (Method: {FEATURE_SELECTION_METHOD}, Fold {fold+1}) ---")
            selector = None
            X_tr_sel, X_va_sel = X_tr_scaled, X_va_scaled # Default if no selection
            selected_features = feature_names
            fold_selector_instance = None # Store the fitted selector

            if FEATURE_SELECTION_METHOD == 'lgbm':
                # Use LGBM Regressor for feature importance
                lgbm_params = {
                    'random_state': RANDOM_STATE,
                    'n_estimators': 100, # Consider making this configurable
                    'device': COMPUTE_PARAMS['lgbm_device'],
                    'verbosity': -1
                }
                # Conditionally add GPU parameters
                if COMPUTE_PARAMS.get('lgbm_device') == 'gpu':
                    lgbm_params['gpu_platform_id'] = COMPUTE_PARAMS.get('lgbm_gpu_platform_id', -1) # Default to -1 if key missing
                    lgbm_params['gpu_device_id'] = COMPUTE_PARAMS.get('lgbm_gpu_device_id', -1)   # Default to -1 if key missing

                lgbm_selector_model = lgb.LGBMRegressor(**lgbm_params)

                selector = SelectFromModel(lgbm_selector_model, threshold='median') # Threshold can be adjusted
                selector.fit(X_tr_scaled, y_tr_reg_fold)
                fold_selector_instance = selector
            elif FEATURE_SELECTION_METHOD == 'kbest_f_reg':
                k_val = min(K_BEST_FEATURES, X_tr_scaled.shape[1]) # Ensure k is not > n_features
                selector = SelectKBest(score_func=f_regression, k=k_val)
                selector.fit(X_tr_scaled, y_tr_reg_fold)
                fold_selector_instance = selector
            elif FEATURE_SELECTION_METHOD == 'kbest_mutual_info':
                k_val = min(K_BEST_FEATURES, X_tr_scaled.shape[1])
                selector = SelectKBest(score_func=mutual_info_regression, k=k_val)
                selector.fit(X_tr_scaled, y_tr_reg_fold)
                fold_selector_instance = selector
            elif FEATURE_SELECTION_METHOD == 'none':
                print("    Skipping feature selection.")
            else:
                print(f"    Warning: Unknown feature selection method '{FEATURE_SELECTION_METHOD}'. Skipping selection.")

            # Apply selection if a method was used
            if selector:
                X_tr_sel = selector.transform(X_tr_scaled)
                X_va_sel = selector.transform(X_va_scaled)
                selected_indices = selector.get_support(indices=True)
                selected_features = [feature_names[i] for i in selected_indices]
                print(f"    Selected {len(selected_features)} features: {selected_features[:10]}...")
            else:
                print(f"    Using all {len(selected_features)} features.")

            fold_selected_features_list.append(selected_features)
            fold_selectors.append(fold_selector_instance) # Store None if no selection

            # Convert selected data back to DataFrame for consistency if needed by tuning funcs
            X_tr_sel_df = pd.DataFrame(X_tr_sel, index=X_tr_fold.index, columns=selected_features)
            X_va_sel_df = pd.DataFrame(X_va_sel, index=X_va_fold.index, columns=selected_features)

            # --- 5.4 Hyperparameter Tuning (On fold's selected train/val data) ---
            print(f"\n  --- 5.4 Hyperparameter Tuning (Fold {fold+1}) ---")
            fold_tuned_params_reg = {} # Store best params for this fold's regressors
            fold_tuned_params_cls = {} # Store best params for this fold's classifiers

            # Define which models to tune based on config AND available optimization functions
            regressors_to_tune_config = {}

            # --- Regression Tuning --- #
            print("\n    Tuning Regressors...")
            for name, model_class in MODEL_REGRESSORS.items(): # Iterate through models provided
                if name in OPTIMIZATION_FUNCTIONS_REG: # Check if optimization function is available
                    objective_fn = OPTIMIZATION_FUNCTIONS_REG[name]
                    # Determine n_trials based on whether it's a 'main' model or 'other'
                    # Heuristic: Assume LGBM/XGB are main, others are 'other'
                    is_main_model = name in ['LGBM', 'XGB']
                    n_trials = OPTUNA_TRIALS_MAIN if is_main_model else OPTUNA_TRIALS_OTHER

                    # Determine optimization direction (common practice: min MAE/RMSE, max R2/AUC)
                    # This requires knowing what the objective function returns.
                    # Hardcoding based on common practice for now. Adjust if objectives change.
                    if name in ['LGBM', 'XGB']: # Assuming MAE/RMSE minimization
                        direction = 'minimize'
                    else: # Assuming R2 maximization for others
                        direction = 'maximize'

                    # Add to tuning config only if tuning is enabled for this type
                    if is_main_model or TUNE_ALL_BASE_MODELS:
                         regressors_to_tune_config[name] = (objective_fn, direction, n_trials)
                elif TUNE_ALL_BASE_MODELS:
                     print(f"      Warning: Tuning enabled, but no optimization function found for {name} in OPTIMIZATION_FUNCTIONS_REG. Skipping tuning.")

            # Now run the tuning loop using the dynamically built config
            for name, (objective_fn, direction, n_trials) in regressors_to_tune_config.items():
                print(f"      Tuning {name} ({n_trials} trials, direction={direction})...")
                study_name = f"reg_{name}_fold{fold+1}_{int(time.time())}" # Add timestamp for uniqueness
                study = run_optuna_study(
                    objective_fn,
                    X_tr_sel_df, y_tr_reg_fold, # Pass dataframes or arrays as expected by objective
                    X_va_sel_df, y_va_reg_fold,
                    n_trials=n_trials,
                    direction=direction,
                    study_name=study_name,
                    timeout=OPTUNA_TIMEOUT
                )
                if study and study.best_trial:
                    fold_tuned_params_reg[name] = study.best_params
                    fold_best_params_reg[name][fold] = study.best_params # Store globally too
                else:
                    print(f"      Warning: Optuna study for {name} did not yield a best trial.")
                    fold_tuned_params_reg[name] = {} # Store empty dict if tuning failed

            # --- Classification Tuning --- #
            # Similar dynamic logic for classifiers
            classifiers_to_tune_config = {}
            print("\n    Tuning Classifiers...")
            for name, model_class in MODEL_CLASSIFIERS.items():
                if name in OPTIMIZATION_FUNCTIONS_CLS:
                    objective_fn = OPTIMIZATION_FUNCTIONS_CLS[name]
                    is_main_model = name in ['LGBM', 'XGB'] # Adjust if needed
                    n_trials = OPTUNA_TRIALS_MAIN if is_main_model else OPTUNA_TRIALS_OTHER
                    direction = 'maximize' # Assume maximizing ROC-AUC or Accuracy

                    if is_main_model or TUNE_ALL_BASE_MODELS:
                        classifiers_to_tune_config[name] = (objective_fn, direction, n_trials)
                elif TUNE_ALL_BASE_MODELS:
                    print(f"      Warning: Tuning enabled, but no optimization function found for {name} in OPTIMIZATION_FUNCTIONS_CLS. Skipping tuning.")

            for name, (objective_fn, direction, n_trials) in classifiers_to_tune_config.items():
                 print(f"      Tuning {name} ({n_trials} trials, direction={direction})...")
                 study_name = f"cls_{name}_fold{fold+1}_{int(time.time())}"
                 study = run_optuna_study(
                     objective_fn,
                     X_tr_sel_df, y_tr_cls_fold,
                     X_va_sel_df, y_va_cls_fold,
                     n_trials=n_trials,
                     direction=direction,
                     study_name=study_name,
                     timeout=OPTUNA_TIMEOUT
                 )
                 if study and study.best_trial:
                     fold_tuned_params_cls[name] = study.best_params
                     fold_best_params_cls[name][fold] = study.best_params
                 else:
                     print(f"      Warning: Optuna study for {name} did not yield a best trial.")
                     fold_tuned_params_cls[name] = {}

            print("DEBUG: Finished or skipped classifier tuning.")

            print("DEBUG: Proceeding to select regression stack.")

            # Regression Stack
            print("\n    Selecting best REGRESSION stack...")
            best_stack_reg, best_base_models_reg, best_meta_reg = select_best_stack(
                base_model_definitions=MODEL_REGRESSORS, # Original definitions
                tuned_base_params=fold_tuned_params_reg, # Tuned params for this fold
                meta_candidates=STACKING_META_REGRESSOR_CANDIDATES,
                X_tr=X_tr_sel_df, # Use selected features df
                y_tr=y_tr_reg_fold,
                X_va=X_va_sel_df, # Use selected features df
                y_va=y_va_reg_fold,
                task='regression',
                cv_folds=STACKING_CV_FOLDS,
                random_state=RANDOM_STATE,
                compute_params=COMPUTE_PARAMS
            )
            all_fold_models_reg.append(best_stack_reg) # Store the best stack for this fold

            # Classification Stack
            print("\n    Selecting best CLASSIFICATION stack...")
            best_stack_cls, best_base_models_cls, best_meta_cls = select_best_stack(
                base_model_definitions=MODEL_CLASSIFIERS,
                tuned_base_params=fold_tuned_params_cls,
                meta_candidates=STACKING_META_CLASSIFIER_CANDIDATES,
                X_tr=X_tr_sel_df,
                y_tr=y_tr_cls_fold,
                X_va=X_va_sel_df,
                y_va=y_va_cls_fold,
                task='classification',
                cv_folds=STACKING_CV_FOLDS,
                random_state=RANDOM_STATE,
                compute_params=COMPUTE_PARAMS
            )
            all_fold_models_cls.append(best_stack_cls) # Store the best stack for this fold


            # --- 5.8 Evaluate Best Stack Model on Fold's Validation Data ---
            print(f"\n  --- 5.8 Evaluating Best Stack on Fold {fold+1} Validation Data ---")

            # Regression Evaluation
            if best_stack_reg:
                y_va_pred_reg = best_stack_reg.predict(X_va_sel_df) # Predict on validation set (selected features)
                r2_va = r2_score(y_va_reg_fold, y_va_pred_reg)
                mae_va = mean_absolute_error(y_va_reg_fold, y_va_pred_reg)
                outer_fold_results_reg['R2'].append(r2_va)
                outer_fold_results_reg['MAE'].append(mae_va)
                print(f"    Regression Stack Validation - R2: {r2_va:.4f}, MAE: {mae_va:.4f}")
            else:
                outer_fold_results_reg['R2'].append(np.nan)
                outer_fold_results_reg['MAE'].append(np.nan)
                print("    Regression Stack - No model selected.")

            # Classification Evaluation
            if best_stack_cls:
                y_va_pred_cls = best_stack_cls.predict(X_va_sel_df)
                y_va_pred_proba_cls = best_stack_cls.predict_proba(X_va_sel_df)[:, 1]
                acc_va = accuracy_score(y_va_cls_fold, y_va_pred_cls)
                try:
                    roc_auc_va = roc_auc_score(y_va_cls_fold, y_va_pred_proba_cls)
                except ValueError:
                    print("    Warning: ROC AUC score could not be calculated (e.g., only one class present in validation fold). Setting to NaN.")
                    roc_auc_va = np.nan
                outer_fold_results_cls['Accuracy'].append(acc_va)
                outer_fold_results_cls['ROC-AUC'].append(roc_auc_va)
                print(f"    Classification Stack Validation - Accuracy: {acc_va:.4f}, ROC-AUC: {roc_auc_va:.4f}")
            else:
                outer_fold_results_cls['Accuracy'].append(np.nan)
                outer_fold_results_cls['ROC-AUC'].append(np.nan)
                print("    Classification Stack - No model selected.")

            fold_end_time = time.time()
            print(f"===== Outer Fold {fold+1} completed in {fold_end_time - fold_start_time:.2f} seconds ====")

        except Exception as e:
            import traceback
            print(f"\n!!!!!! ERROR CAUGHT IN OUTER FOLD {fold+1} !!!!!")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {e}")
            print("Traceback:")
            print(traceback.format_exc())
            print("Continuing to next fold if possible, or returning current results...")
            # Depending on severity, you might want to 'break' or 'raise e' here.
            # For debugging, we'll let it potentially continue to see if other folds work.

    # End of outer loop
    total_cv_time = time.time() - start_time_cv
    print(f"\n--- Outer Cross-Validation completed in {total_cv_time:.2f} seconds ---")

    return (
        outer_fold_results_reg,
        outer_fold_results_cls,
        fold_selected_features_list,
        fold_best_params_reg,
        fold_best_params_cls,
        fold_scalers,
        fold_selectors,
        all_fold_models_reg,
        all_fold_models_cls
    )


# --- (Optional) Helper functions previously in utils or model_tuning could be moved here ---
# For example, if select_best_stack was not complex, it could live here.
# However, keeping utils.py and model_tuning.py separate is often better for organization.