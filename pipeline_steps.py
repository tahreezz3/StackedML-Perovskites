# -*- coding: utf-8 -*-
"""
pipeline_steps.py

Contains functions defining distinct steps of the ML pipeline,
called from main.py.
"""

import numpy as np
import pandas as pd
import shap
import joblib
import time
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, roc_auc_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, mutual_info_regression # For recreating selector if needed

# Import necessary config or utility functions if needed, or pass them
# from config import ...
# from utils import ...


def aggregate_cv_results(outer_fold_results_reg, outer_fold_results_cls):
    """
    Calculates and prints the aggregated cross-validation results.

    Args:
        outer_fold_results_reg (dict): Dictionary with lists of regression metrics per fold.
        outer_fold_results_cls (dict): Dictionary with lists of classification metrics per fold.

    Returns:
        tuple: (mean_r2, std_r2, mean_mae, std_mae, mean_acc, std_acc, mean_roc_auc, std_roc_auc)
               Returns NaNs if metrics couldn't be calculated for any fold.
    """
    print("\n--- 6. Aggregating Cross-Validation Results ---")

    # Regression Results
    r2_scores = outer_fold_results_reg.get('R2', [])
    mae_scores = outer_fold_results_reg.get('MAE', [])
    mean_r2, std_r2, mean_mae, std_mae = np.nan, np.nan, np.nan, np.nan # Defaults

    if r2_scores and not all(np.isnan(r2_scores)):
        mean_r2 = np.nanmean(r2_scores)
        std_r2 = np.nanstd(r2_scores)
        print(f"  Average Outer CV R2: {mean_r2:.4f} +/- {std_r2:.4f}")
    else:
        print("  Average Outer CV R2: N/A (No valid scores)")

    if mae_scores and not all(np.isnan(mae_scores)):
        mean_mae = np.nanmean(mae_scores)
        std_mae = np.nanstd(mae_scores)
        print(f"  Average Outer CV MAE: {mean_mae:.4f} +/- {std_mae:.4f}")
    else:
        print("  Average Outer CV MAE: N/A (No valid scores)")

    # Classification Results
    acc_scores = outer_fold_results_cls.get('Accuracy', [])
    roc_auc_scores = outer_fold_results_cls.get('ROC-AUC', [])
    mean_acc, std_acc, mean_roc_auc, std_roc_auc = np.nan, np.nan, np.nan, np.nan # Defaults

    if acc_scores and not all(np.isnan(acc_scores)):
        mean_acc = np.nanmean(acc_scores)
        std_acc = np.nanstd(acc_scores)
        print(f"  Average Outer CV Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
    else:
        print("  Average Outer CV Accuracy: N/A (No valid scores)")

    if roc_auc_scores and not all(np.isnan(roc_auc_scores)):
        mean_roc_auc = np.nanmean(roc_auc_scores)
        std_roc_auc = np.nanstd(roc_auc_scores)
        print(f"  Average Outer CV ROC-AUC: {mean_roc_auc:.4f} +/- {std_roc_auc:.4f}")
    else:
        print("  Average Outer CV ROC-AUC: N/A (No valid scores)")

    return mean_r2, std_r2, mean_mae, std_mae, mean_acc, std_acc, mean_roc_auc, std_roc_auc


def select_final_model(
    outer_fold_results_reg,
    outer_fold_results_cls,
    all_fold_models_reg,
    all_fold_models_cls,
    fold_scalers,
    fold_selectors,
    fold_selected_features_list,
    n_splits_outer_cv # For context/logging
):
    """
    Selects the final regressor and classifier based on the best performing fold
    in the outer cross-validation.
    Uses the scaler, selector, and features from the best *regression* fold for consistency
    when processing the final test set later.

    Args:
        outer_fold_results_reg (dict): Regression metrics per fold.
        outer_fold_results_cls (dict): Classification metrics per fold.
        all_fold_models_reg (list): List of trained regressor models per fold.
        all_fold_models_cls (list): List of trained classifier models per fold.
        fold_scalers (list): List of fitted scalers per fold.
        fold_selectors (list): List of fitted selectors per fold.
        fold_selected_features_list (list): List of selected features per fold.
        n_splits_outer_cv (int): Number of outer CV splits.

    Returns:
        tuple: (
            final_regressor, final_classifier, final_scaler, final_selector,
            selected_features_final, best_fold_idx_reg, best_fold_idx_cls
        )
    """
    print(f"\n--- 7. Selecting Final Model (Based on best CV Fold Performance) ---")

    final_regressor = None
    final_classifier = None
    final_scaler = None
    final_selector = None
    selected_features_final = None
    best_fold_idx_reg = -1
    best_fold_idx_cls = -1

    # --- Select Best Fold Based on Validation ROC-AUC (for Classifier) and R2 (for Regressor) ---
    roc_auc_scores = outer_fold_results_cls.get('ROC-AUC', [])
    r2_scores = outer_fold_results_reg.get('R2', [])

    if roc_auc_scores and not all(np.isnan(roc_auc_scores)):
        best_fold_idx_cls = np.nanargmax(roc_auc_scores)
        final_classifier = all_fold_models_cls[best_fold_idx_cls]
        print(f"  Best Classification Fold (Max ROC-AUC): {best_fold_idx_cls + 1}/{n_splits_outer_cv} (ROC-AUC: {roc_auc_scores[best_fold_idx_cls]:.4f})")
        if final_classifier:
            print(f"  Selected Final Classifier: {final_classifier.__class__.__name__} (from fold {best_fold_idx_cls + 1})")
        else:
             print(f"  Warning: Best classification fold {best_fold_idx_cls + 1} had no valid model stored.")
    else:
        print("  Could not determine best classification fold (no valid ROC-AUC scores). No final classifier selected.")

    if r2_scores and not all(np.isnan(r2_scores)):
        best_fold_idx_reg = np.nanargmax(r2_scores)
        final_regressor = all_fold_models_reg[best_fold_idx_reg]
        # Use the scaler, selector, and features from the best *regression* fold for test set processing
        final_scaler = fold_scalers[best_fold_idx_reg]
        final_selector = fold_selectors[best_fold_idx_reg]
        selected_features_final = fold_selected_features_list[best_fold_idx_reg]
        print(f"  Best Regression Fold (Max R2): {best_fold_idx_reg + 1}/{n_splits_outer_cv} (R2: {r2_scores[best_fold_idx_reg]:.4f})")
        if final_regressor:
            print(f"  Selected Final Regressor: {final_regressor.__class__.__name__} (from fold {best_fold_idx_reg + 1})")
        else:
            print(f"  Warning: Best regression fold {best_fold_idx_reg + 1} had no valid model stored.")
        if final_scaler:
            print(f"  Using Scaler from fold {best_fold_idx_reg + 1}")
        if final_selector:
             print(f"  Using Selector ({final_selector.__class__.__name__}) from fold {best_fold_idx_reg + 1}")
        if selected_features_final:
             print(f"  Using Selected Features ({len(selected_features_final)} count) from fold {best_fold_idx_reg + 1}")

    else:
        print("  Could not determine best regression fold (no valid R2 scores). No final regressor, scaler, selector, or features selected.")

    # Handle case where best folds might have yielded None models (e.g., stacking failed)
    if best_fold_idx_cls != -1 and final_classifier is None:
        print(f"  Warning: Model from best classification fold ({best_fold_idx_cls + 1}) was None.")
    if best_fold_idx_reg != -1 and final_regressor is None:
         print(f"  Warning: Model from best regression fold ({best_fold_idx_reg + 1}) was None.")

    return (
        final_regressor, final_classifier, final_scaler, final_selector,
        selected_features_final, best_fold_idx_reg, best_fold_idx_cls
    )


def evaluate_on_test_set(
    X_test, y_test_reg, y_test_cls,
    final_regressor, final_classifier,
    final_scaler, final_selector,
    selected_features_final,
    feature_names # Original full feature names list
):
    """
    Evaluates the selected final models on the hold-out test set.
    Applies the scaling and feature selection steps derived from the best CV fold.

    Args:
        X_test (pd.DataFrame): Test set features.
        y_test_reg (pd.Series): Test set regression target.
        y_test_cls (pd.Series): Test set classification target.
        final_regressor: The selected final regression model.
        final_classifier: The selected final classification model.
        final_scaler: The fitted scaler from the best CV fold.
        final_selector: The fitted feature selector from the best CV fold.
        selected_features_final (list): List of feature names selected by the final_selector.
        feature_names (list): List of all original feature names.

    Returns:
        tuple: (X_test_scaled_df, X_test_sel_df) - Processed test dataframes for SHAP.
               Returns (None, None) if preprocessing fails.
    """
    print("\n--- 8. Final Evaluation on Unseen Test Set ---")

    if final_scaler is None:
        print("  Error: Final scaler is not available. Cannot evaluate on test set.")
        return None, None
    if selected_features_final is None:
         print("  Error: Final selected features list is not available. Cannot evaluate on test set.")
         return None, None

    # --- Apply the SAME preprocessing as the best fold --- #
    # 1. Scale the test data using the scaler fitted on the best fold's *training* data
    X_test_scaled = final_scaler.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=feature_names)
    print(f"  Scaled test data shape: {X_test_scaled_df.shape}")

    # 2. Apply feature selection using the selector fitted on the best fold's *training* data
    X_test_sel = X_test_scaled # Default if no selector was used
    if final_selector:
        try:
            X_test_sel = final_selector.transform(X_test_scaled)
            print(f"  Applied feature selection. Selected test data shape: {X_test_sel.shape}")
        except ValueError as e:
            print(f"  Error applying feature selector to test data: {e}")
            print(f"  Expected {final_selector.n_features_in_} features, got {X_test_scaled.shape[1]}")
            return None, None # Cannot proceed if selection fails
    else:
        print("  No feature selector applied (selector was None).")

    # Convert selected test data to DataFrame using the selected feature names
    X_test_sel_df = pd.DataFrame(X_test_sel, index=X_test.index, columns=selected_features_final)


    # --- Evaluate Regressor --- #
    if final_regressor:
        print("\n  Evaluating Final Regressor on Test Set...")
        try:
            y_pred_reg_test = final_regressor.predict(X_test_sel_df)
            test_r2 = r2_score(y_test_reg, y_pred_reg_test)
            test_mae = mean_absolute_error(y_test_reg, y_pred_reg_test)
            test_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg_test))
            print(f"    Test R2: {test_r2:.4f}")
            print(f"    Test MAE: {test_mae:.4f}")
            print(f"    Test RMSE: {test_rmse:.4f}")
        except Exception as e:
            print(f"    Error during regression evaluation: {e}")
    else:
        print("\n  Skipping Test Set Regression Evaluation (No final model selected)." )

    # --- Evaluate Classifier --- #
    if final_classifier:
        print("\n  Evaluating Final Classifier on Test Set...")
        try:
            y_pred_cls_test = final_classifier.predict(X_test_sel_df)
            test_accuracy = accuracy_score(y_test_cls, y_pred_cls_test)
            print(f"    Test Accuracy: {test_accuracy:.4f}")

            # Calculate ROC AUC if possible
            if hasattr(final_classifier, "predict_proba"):
                try:
                    y_pred_proba_cls_test = final_classifier.predict_proba(X_test_sel_df)[:, 1]
                    test_roc_auc = roc_auc_score(y_test_cls, y_pred_proba_cls_test)
                    print(f"    Test ROC-AUC: {test_roc_auc:.4f}")
                except ValueError as e:
                    print(f"    Warning: Could not calculate Test ROC-AUC: {e}")
                except Exception as e:
                    print(f"    Error calculating Test ROC-AUC: {e}")
            else:
                print("    Test ROC-AUC: Not calculated (model lacks predict_proba)")
        except Exception as e:
            print(f"    Error during classification evaluation: {e}")
    else:
        print("\n  Skipping Test Set Classification Evaluation (No final model selected)." )

    return X_test_scaled_df, X_test_sel_df # Return processed test data for SHAP

def run_shap_analysis(
    final_regressor, final_classifier,
    X_test_sel_df, # Selected features test data
    selected_features_final,
    X_train_val, # Full training data for background sample
    y_train_val_reg, # Needed for some selectors if re-fitting needed (though ideally not)
    kf_outer, # To get the training indices of the best fold
    best_fold_idx_reg, # Index of the fold whose scaler/selector were used
    final_scaler, # Scaler from the best fold
    final_selector, # Selector from the best fold
    feature_selection_method # To know if selector exists
):
    """
    Calculates and potentially plots SHAP values for the final models.

    Args:
        final_regressor: The final trained regression model.
        final_classifier: The final trained classification model.
        X_test_sel_df (pd.DataFrame): Test data preprocessed with selected features.
        selected_features_final (list): List of final selected feature names.
        X_train_val (pd.DataFrame): The original combined training/validation data.
        y_train_val_reg (pd.Series): Regression target for train/val (for selector fit if needed).
        kf_outer: The outer cross-validation splitter.
        best_fold_idx_reg (int): Index of the best regression fold (used for scaler/selector).
        final_scaler: The scaler fitted on the best fold's training data.
        final_selector: The selector fitted on the best fold's training data.
        feature_selection_method (str): The method used for feature selection.

    Returns:
        None
    """
    print("\n--- 9. SHAP Interpretability --- ")

    if best_fold_idx_reg == -1:
        print("  Skipping SHAP: Cannot determine the best fold for background data.")
        return

    if not final_regressor and not final_classifier:
        print("  Skipping SHAP: No final models available.")
        return

    # --- Prepare Background Data --- #
    # Use a sample of the training data from the *best fold* (whose scaler/selector we used)
    print(f"  Preparing SHAP background data from best fold's training set (Fold {best_fold_idx_reg + 1})")
    try:
        train_indices, _ = list(kf_outer.split(X_train_val, y_train_val_reg))[best_fold_idx_reg]
        X_tr_final_fold = X_train_val.iloc[train_indices]

        # Take a subset for performance
        background_sample_size = min(100, X_tr_final_fold.shape[0])
        background_data_idx = np.random.choice(X_tr_final_fold.index, size=background_sample_size, replace=False)
        X_tr_fold_sample = X_tr_final_fold.loc[background_data_idx]

        # Scale and select features for the background data using the *final* fold's scaler/selector
        X_tr_fold_sample_scaled = final_scaler.transform(X_tr_fold_sample)

        if final_selector:
            X_tr_fold_sample_sel = final_selector.transform(X_tr_fold_sample_scaled)
        elif feature_selection_method == 'none':
             X_tr_fold_sample_sel = X_tr_fold_sample_scaled # No selection applied
        else:
             print("  Error: Could not apply feature selection to SHAP background data.")
             return # Cannot proceed without background data

        X_tr_fold_sample_sel_df = pd.DataFrame(
            X_tr_fold_sample_sel,
            index=X_tr_fold_sample.index,
            columns=selected_features_final
        )
        print(f"  Background data shape for SHAP: {X_tr_fold_sample_sel_df.shape}")

    except Exception as e:
        print(f"  Error preparing SHAP background data: {e}")
        return

    # Select a sample of the test data for SHAP calculation (for performance)
    test_sample_size = min(200, X_test_sel_df.shape[0])
    test_sample_idx = np.random.choice(X_test_sel_df.index, size=test_sample_size, replace=False)
    X_test_sel_sample_df = X_test_sel_df.loc[test_sample_idx]
    print(f"  Calculating SHAP values for {test_sample_size} test samples.")

    # --- SHAP for Regressor --- #
    if final_regressor:
        print("\n  Calculating SHAP values for Regressor...")
        try:
            # Define prediction function (needed for KernelExplainer)
            def reg_predict_fn(data):
                data_df = pd.DataFrame(data, columns=selected_features_final)
                return final_regressor.predict(data_df)

            # Use KernelExplainer (safer for complex models like Stacking)
            reg_explainer = shap.KernelExplainer(reg_predict_fn, X_tr_fold_sample_sel_df)
            shap_values_reg = reg_explainer.shap_values(X_test_sel_sample_df)

            print("    Generating Regression SHAP summary plot...")
            shap.summary_plot(shap_values_reg, X_test_sel_sample_df, feature_names=selected_features_final, show=False)
            # TODO: Add saving plot logic if needed (e.g., using matplotlib)
            # import matplotlib.pyplot as plt
            # plt.savefig("shap_summary_regressor.png", bbox_inches='tight')
            # plt.close()
            print("    Regression SHAP summary plot generated.")

        except Exception as e:
            print(f"    Error generating SHAP values/plot for regressor: {e}")
    else:
        print("\n  Skipping SHAP for regressor (no final model)." )

    # --- SHAP for Classifier --- #
    if final_classifier:
        print("\n  Calculating SHAP values for Classifier...")
        try:
            # Define prediction function (outputting probabilities for class 1)
            def cls_predict_proba_fn(data):
                data_df = pd.DataFrame(data, columns=selected_features_final)
                # Ensure predict_proba exists and returns probabilities for the positive class
                if hasattr(final_classifier, 'predict_proba'):
                    return final_classifier.predict_proba(data_df)[:, 1]
                else:
                    # Handle models without predict_proba if necessary (e.g., return decisions)
                    # This might require a different SHAP explainer or approach
                    print("    Warning: Classifier lacks predict_proba for SHAP.")
                    return final_classifier.predict(data_df) # Fallback, may not be ideal for SHAP

            # Use KernelExplainer
            cls_explainer = shap.KernelExplainer(cls_predict_proba_fn, X_tr_fold_sample_sel_df)
            shap_values_cls = cls_explainer.shap_values(X_test_sel_sample_df)

            print("    Generating Classification SHAP summary plot...")
            shap.summary_plot(shap_values_cls, X_test_sel_sample_df, feature_names=selected_features_final, show=False)
            # plt.savefig("shap_summary_classifier.png", bbox_inches='tight')
            # plt.close()
            print("    Classification SHAP summary plot generated.")

        except Exception as e:
            print(f"    Error generating SHAP values/plot for classifier: {e}")
    else:
        print("\n  Skipping SHAP for classifier (no final model)." )


def save_artifacts(
    final_regressor, final_classifier,
    selected_features_final,
    outer_fold_results_reg, outer_fold_results_cls,
    output_dir="output", # Allow specifying output directory
    save_models=SAVE_MODELS,
    save_features=SAVE_FEATURES,
    save_results=SAVE_RESULTS
):
    """
    Saves the final models, selected features, and CV results to disk.

    Args:
        final_regressor: The final trained regression model.
        final_classifier: The final trained classification model.
        selected_features_final (list): List of final selected feature names.
        outer_fold_results_reg (dict): Aggregated regression CV results.
        outer_fold_results_cls (dict): Aggregated classification CV results.
        output_dir (str): Directory to save the artifacts.
        save_models (bool): Whether to save the final models.
        save_features (bool): Whether to save the list of selected features.
        save_results (bool): Whether to save the cross-validation results.

    Returns:
        None
    """
    import os
    print(f"\n--- 10. Saving Artifacts (Output Dir: {output_dir}) --- ")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # --- Save Final Models --- #
    if save_models:
        if final_regressor:
            try:
                reg_filename = os.path.join(output_dir, 'final_regressor.joblib')
                joblib.dump(final_regressor, reg_filename)
                print(f"  Saved final regressor to: {reg_filename}")
            except Exception as e:
                print(f"  Error saving final regressor: {e}")
        else:
            print("  Skipping saving final regressor (not available).")

        if final_classifier:
            try:
                cls_filename = os.path.join(output_dir, 'final_classifier.joblib')
                joblib.dump(final_classifier, cls_filename)
                print(f"  Saved final classifier to: {cls_filename}")
            except Exception as e:
                print(f"  Error saving final classifier: {e}")
        else:
            print("  Skipping saving final classifier (not available).")

    # --- Save Selected Features --- #
    if save_features:
        if selected_features_final:
            try:
                features_filename = os.path.join(output_dir, 'selected_features.txt')
                with open(features_filename, 'w') as f:
                    for feature in selected_features_final:
                        f.write(f"{feature}\n")
                print(f"  Saved selected features ({len(selected_features_final)} count) to: {features_filename}")
            except Exception as e:
                print(f"  Error saving selected features: {e}")
        else:
            print("  Skipping saving selected features (list is empty or None).")

    # --- Save CV Results --- #
    if save_results:
        try:
            results_df_reg = pd.DataFrame(outer_fold_results_reg)
            results_df_cls = pd.DataFrame(outer_fold_results_cls)
            results_filename_reg = os.path.join(output_dir, 'cv_results_regression.csv')
            results_filename_cls = os.path.join(output_dir, 'cv_results_classification.csv')
            results_df_reg.to_csv(results_filename_reg, index=False)
            results_df_cls.to_csv(results_filename_cls, index=False)
            print(f"  Saved regression CV results to: {results_filename_reg}")
            print(f"  Saved classification CV results to: {results_filename_cls}")
        except Exception as e:
            print(f"  Error saving CV results: {e}")


# --- Add other pipeline step functions below ---
