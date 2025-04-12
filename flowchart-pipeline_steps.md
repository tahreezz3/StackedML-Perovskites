# Machine Learning Pipeline Steps Flowchart

This document explains the logic and workflow of the ML pipeline implemented in `pipeline_steps.py`. This file contains the key functions that make up the later stages of the machine learning pipeline for perovskite materials analysis.

## Pipeline Overview

The `pipeline_steps.py` module contains functions for the following key stages of the machine learning pipeline:

1. **Aggregating Cross-Validation Results** - Function: `aggregate_cv_results()`
2. **Selecting Final Models** - Function: `select_final_model()`
3. **Evaluating on Test Set** - Function: `evaluate_on_test_set()`
4. **SHAP Interpretability Analysis** - Function: `run_shap_analysis()`
5. **Saving Artifacts** - Function: `save_artifacts()`

These functions are designed to be called sequentially, with each step building on the results of the previous one. The pipeline handles both regression and classification tasks simultaneously.

## Detailed Function Workflows

### 1. `aggregate_cv_results()`

**Purpose**: Calculates and reports the average cross-validation performance metrics across all folds.

**Workflow**:
1. Takes regression and classification results dictionaries as input
2. For regression metrics (R2, MAE):
   - Calculates mean and standard deviation across all folds
   - Handles NaN values gracefully
   - Prints formatted results
3. For classification metrics (Accuracy, ROC-AUC):
   - Calculates mean and standard deviation across all folds
   - Handles NaN values gracefully
   - Prints formatted results
4. Returns all calculated metrics for further use

### 2. `select_final_model()`

**Purpose**: Selects the best performing models, scalers, and feature selectors from cross-validation folds.

**Workflow**:
1. Takes CV results, all trained models, scalers, and feature selectors as input
2. For classification models:
   - Selects the fold with the best ROC-AUC score
   - Retrieves the corresponding classifier model
3. For regression models:
   - Selects the fold with the best R2 score
   - Retrieves the corresponding regressor model
   - Also retrieves the scaler, selector, and selected features from that fold
4. Provides detailed logging about which models were chosen
5. Returns the final models and preprocessing components

**Key Logic**:
- The regression fold's preprocessing components (scaler, selector, feature list) are used for consistency across both regression and classification tasks
- Handles cases where models might be None due to training failures

### 3. `evaluate_on_test_set()`

**Purpose**: Applies the preprocessing pipeline and evaluates the final models on the unseen test set.

**Workflow**:
1. Takes test data, final models, and preprocessing components as input
2. Preprocessing pipeline:
   - Applies the final scaler to the test data
   - Applies the final feature selector to the scaled test data
   - Creates DataFrame with selected features
3. For regression evaluation:
   - Generates predictions using the final regressor
   - Calculates R2, MAE, and RMSE metrics
   - Prints test performance metrics
4. For classification evaluation:
   - Generates predictions using the final classifier
   - Calculates accuracy and ROC-AUC (if available)
   - Prints test performance metrics
5. Returns processed test data for SHAP analysis

**Key Logic**:
- Applies the exact same preprocessing steps that were used on the training data
- Handles cases where models might be unavailable
- Error handling for potential issues in the preprocessing pipeline

### 4. `run_shap_analysis()`

**Purpose**: Calculates and visualizes SHAP explanations for model predictions.

**Workflow**:
1. Takes final models, processed test data, and training data as input
2. Prepares background data for SHAP:
   - Uses samples from the best fold's training set
   - Applies same preprocessing (scaling and feature selection)
3. Selects a manageable sample of test data for efficiency
4. For the regression model:
   - Creates a KernelExplainer with prediction function
   - Calculates SHAP values on test samples
   - Generates and displays summary plot
5. For the classification model:
   - Creates a KernelExplainer with probability prediction function
   - Calculates SHAP values on test samples
   - Generates and displays summary plot

**Key Logic**:
- Uses a subset of data to make SHAP computation feasible
- Handles models without predict_proba for classification
- Consistent preprocessing across background and test data
- Error handling for SHAP calculation failures

### 5. `save_artifacts()`

**Purpose**: Persists the final models, selected features, and CV results to disk.

**Workflow**:
1. Takes final models, selected features, and CV results as input
2. Creates output directory if it doesn't exist
3. For model saving (if enabled):
   - Saves regression model using joblib
   - Saves classification model using joblib
4. For feature saving (if enabled):
   - Saves the list of selected features to a text file
5. For results saving (if enabled):
   - Converts results dictionaries to DataFrames
   - Saves regression and classification results as CSV files

**Key Logic**:
- Controlled by configuration flags (save_models, save_features, save_results)
- Robust error handling for each saving operation
- Uses standard formats for compatibility (joblib, txt, csv)

## Pipeline Data Flow

```
Input Data → [Preprocessing, CV Training - implemented elsewhere] → 
    CV Results, Models → aggregate_cv_results() → 
    Performance Metrics → select_final_model() → 
    Final Models, Preprocessors → evaluate_on_test_set() → 
    Test Performance, Processed Test Data → run_shap_analysis() → 
    Model Explanations → save_artifacts() → 
    Saved Models, Features, Results
```

## Configuration

The pipeline leverages configuration variables from `config.py` including:
- `SAVE_MODELS`: Flag to control model persistence
- `SAVE_FEATURES`: Flag to control feature list persistence
- `SAVE_RESULTS`: Flag to control CV results persistence
- `SHAP_BACKGROUND_SAMPLES`: Number of samples to use for SHAP background
- `SHAP_EXPLAIN_SAMPLES`: Number of test samples to explain with SHAP
- `OUTPUT_DIR`: Directory for saving artifacts

## Notes

- The pipeline is designed to handle both regression and classification tasks simultaneously
- Preprocessing steps (scaling, feature selection) are consistently applied across training, validation, and test sets
- Extensive error handling ensures the pipeline can continue even if certain steps fail
- Performance metrics are logged at each step for transparency
- The SHAP analysis provides interpretability for both regression and classification models
