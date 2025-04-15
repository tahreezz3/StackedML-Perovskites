# Flowchart: Outer Cross-Validation Loop (`outer_cv.py`)

## 1. Purpose

The `outer_cv.py` script orchestrates the outer loop of a nested cross-validation process. Its main responsibilities within each fold are:
1.  **Data Splitting**: Divide the training/validation data into a fold-specific training set and validation set.
2.  **Preprocessing**: Apply scaling and feature selection, fitting transformers only on the fold's training data.
3.  **Hyperparameter Tuning**: Optimize hyperparameters for base models using the fold's train/validation split via Optuna.
4.  **Stacking Ensemble Selection**: Build and evaluate stacking ensembles (for both regression and classification) using the tuned base models and select the best meta-learner for each task.
5.  **Performance Evaluation**: Assess the performance of the selected stacking models on the fold's validation set.
6.  **Result Aggregation**: Collect metrics, selected features, fitted preprocessors, tuned parameters, and trained models from each fold.

## 2. Main Function: `run_outer_cv_loop()`

This is the core function driving the outer cross-validation process.

### Inputs

-   `X_train_val`, `y_train_val_reg`, `y_train_val_cls`: The combined training and validation dataset features and targets.
-   `kf_outer`: The scikit-learn cross-validation splitter object (e.g., `KFold`).
-   `feature_names`: List of original feature names.
-   Configuration Parameters (from `config.py` or passed):
    -   `FEATURE_SELECTION_METHOD`, `K_BEST_FEATURES`
    -   `TUNE_ALL_BASE_MODELS`, `OPTUNA_TRIALS_MAIN`, `OPTUNA_TRIALS_OTHER`, `OPTUNA_TIMEOUT`
    -   `STACKING_CV_FOLDS`
-   Model Dictionaries:
    -   `MODEL_REGRESSORS`, `MODEL_CLASSIFIERS`
    -   `STACKING_META_REGRESSOR_CANDIDATES`, `STACKING_META_CLASSIFIER_CANDIDATES`
-   Helper Functions:
    -   `get_compute_device_params`: Retrieves compute parameters.
    -   `run_optuna_study`: Executes an Optuna hyperparameter search.
    -   `select_best_stack`: Selects the best stacking ensemble configuration.
    -   `OPTIMIZATION_FUNCTIONS_REG`, `OPTIMIZATION_FUNCTIONS_CLS`: Dictionaries mapping model names to their Optuna objective functions.

### Outputs

-   `outer_fold_results_reg`, `outer_fold_results_cls`: Dictionaries containing performance metrics (R2, MAE, Accuracy, ROC-AUC) for each fold.
-   `fold_selected_features_list`: A list where each element is the list of selected feature names for that fold.
-   `fold_best_params_reg`, `fold_best_params_cls`: Dictionaries storing the best hyperparameters found for each base model type across folds.
-   `fold_scalers`: List of fitted `StandardScaler` instances, one per fold.
-   `fold_selectors`: List of fitted feature selector instances (e.g., `SelectFromModel`, `SelectKBest`), one per fold (can be `None`).
-   `all_fold_models_reg`, `all_fold_models_cls`: Lists containing the final trained stacking regressor and classifier model for each fold.

## 3. Workflow per Fold

The following diagram illustrates the sequence of operations performed within *each* fold of the outer loop:

```mermaid
graph TD
    A[Start Outer Fold] --> B(Split Data into Fold Train/Val);
    B --> C{Preprocessing};
    C -- Scaling --> D[Fit & Apply StandardScaler];
    C -- Feature Selection --> E[Fit & Apply Selector (e.g., LGBM, KBest)];
    D --> F{Store Scaler};
    E --> G{Store Selector & Features};
    G --> H[Prepare Selected Data (X_tr_sel_df, X_va_sel_df)];

    H --> I{Hyperparameter Tuning (Optuna)};
    I -- Regression --> J[Tune Base Regressors];
    I -- Classification --> K[Tune Base Classifiers];

    J --> L{Store Best Reg Params};
    K --> M{Store Best Cls Params};

    L & M --> N{Stacking Ensemble Selection};
    N -- Regression --> O[Select Best Regressor Stack];
    N -- Classification --> P[Select Best Classifier Stack];

    O --> Q{Store Best Regressor Stack};
    P --> R{Store Best Classifier Stack};

    Q & R --> S{Performance Evaluation};
    S -- Regression --> T[Predict & Calc Reg Metrics (R2, MAE) on Val Set];
    S -- Classification --> U[Predict & Calc Cls Metrics (Acc, AUC) on Val Set];

    T & U --> V{Store Fold Metrics};
    V --> W[End Outer Fold];

    %% Error Handling Overlay (Conceptual)
    subgraph Error Handling
        direction LR
        X(Try Block Start) --> Y{Operation}
        Y -- Success --> Z(Continue)
        Y -- Failure --> AA(Catch Exception)
        AA --> AB(Log Error & Store Defaults/NaN)
        AB --> Z
    end

    %% Apply Error Handling Concept to major steps
    classDef errorHandling fill:#f9f,stroke:#333,stroke-width:2px;
    class B,D,E,J,K,O,P,T,U errorHandling; %% Example steps wrapped in try-except
```

## 4. Sub-Process Details

### 4.1 Feature Selection (`FEATURE_SELECTION_METHOD`)

-   **`lgbm`**: Uses `lightgbm.LGBMRegressor` feature importances via `SelectFromModel`. Threshold typically 'median'.
-   **`kbest_f_reg`**: Uses `SelectKBest` with `f_regression` scoring. Selects top `K_BEST_FEATURES`.
-   **`kbest_mutual_info`**: Uses `SelectKBest` with `mutual_info_regression` scoring. Selects top `K_BEST_FEATURES`.
-   **`none`**: Skips feature selection; uses all features after scaling.
-   *Default*: If an unknown method is provided, it defaults to `none`.
-   The fitted selector instance (or `None`) is stored for each fold.

### 4.2 Hyperparameter Tuning (`run_optuna_study`)

-   Iterates through `MODEL_REGRESSORS` and `MODEL_CLASSIFIERS`.
-   Checks `TUNE_ALL_BASE_MODELS` flag or if the model is a 'main' model (e.g., LGBM, XGB).
-   Retrieves the corresponding objective function from `OPTIMIZATION_FUNCTIONS_REG` or `OPTIMIZATION_FUNCTIONS_CLS`.
-   Sets the number of trials (`OPTUNA_TRIALS_MAIN` or `OPTUNA_TRIALS_OTHER`).
-   Calls `run_optuna_study` with the selected fold data (`X_tr_sel_df`, `y_tr_...`, `X_va_sel_df`, `y_va_...`).
-   `run_optuna_study` handles the Optuna study creation, execution (with timeout), and error catching per trial.
-   Stores the `best_params` found for each tuned model in `fold_tuned_params_reg`/`fold_tuned_params_cls` for the current fold.

### 4.3 Stacking Ensemble Selection (`select_best_stack`)

-   Called separately for regression and classification tasks.
-   **Inputs**:
    -   Base model definitions (`MODEL_REGRESSORS`/`MODEL_CLASSIFIERS`).
    -   Tuned hyperparameters for base models (`fold_tuned_params_reg`/`fold_tuned_params_cls`).
    -   Meta-learner candidates (`STACKING_META_...`).
    -   Fold's selected training data (`X_tr_sel_df`, `y_tr_...`).
    -   Stacking internal CV folds (`STACKING_CV_FOLDS`).
-   **Process**: Likely involves instantiating base models with tuned parameters, training them, generating out-of-fold predictions, trying different meta-learners on these predictions, and selecting the meta-learner that performs best according to internal validation.
-   **Output**: Returns the best-performing, fully trained `StackingRegressor` or `StackingClassifier` for the fold.

## 5. Error Handling

-   The main loop (`for fold...`) is wrapped in a `try...except` block.
-   Individual critical steps like tuning and stack selection often have their own internal error handling (e.g., within `run_optuna_study`).
-   If an error occurs during a fold:
    -   An error message is printed (often with traceback).
    -   Default values (e.g., `np.nan` for metrics, empty dicts for params, `None` for models/selectors) are appended to the results lists for that fold.
    -   The loop continues to the next fold, ensuring the overall process completes even if some folds fail.

## 6. Configuration Dependencies

Key parameters from `config.py` influencing this module:

```python
# Feature Selection
FEATURE_SELECTION_METHOD = 'lgbm' # 'kbest_f_reg', 'kbest_mutual_info', 'none'
K_BEST_FEATURES = 50

# Hyperparameter Tuning
TUNE_ALL_BASE_MODELS = True
OPTUNA_TRIALS_MAIN = 50 # Trials for 'main' models (LGBM, XGB)
OPTUNA_TRIALS_OTHER = 20 # Trials for other models
OPTUNA_TIMEOUT = None # Timeout per Optuna study (seconds)

# Stacking
STACKING_CV_FOLDS = 5 # Inner CV folds for meta-learner training

# General
RANDOM_STATE = 42
N_SPLITS_OUTER_CV = 5 # For logging fold progress
```

## 7. Notes

-   Preprocessing (scaling, selection) is strictly fitted *only* on the training portion of each fold to prevent data leakage.
-   The validation portion of the fold (`X_va_...`, `y_va_...`) is used for evaluating base model hyperparameters during tuning *and* for evaluating the final stacked model performance for that fold.
-   Results, parameters, scalers, selectors, and models are collected across all folds to be aggregated and analyzed later (e.g., in `pipeline_steps.py`).
