# FLOWCHART: Stacked ML Process in outer_cv.py

## OVERVIEW

The outer_cv.py file implements a nested cross-validation approach for machine learning with stacking ensembles. The primary goal is to:
1. Tune hyperparameters for base models
2. Use tuned base models for stacking ensemble creation
3. Select the best meta-model (stacking model) based on performance
4. Handle errors gracefully and maintain robustness

## DETAILED PROCESS FLOW

```
INPUT → OUTER CV LOOP → [PROCESS EACH FOLD] → OUTPUT RESULTS
```

### 1. INPUTS
- Feature data (X_train_val)
- Regression target (y_train_val_reg)
- Classification target (y_train_val_cls)
- Outer cross-validation split iterator (kf_outer)
- Feature names
- Configuration parameters
- Helper functions and compute device parameters

### 2. OUTER CV LOOP PROCESS (For each fold)

```
┌────────────────────────────┐
│ Begin Outer CV Loop        │
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│ Initialize Result Storage  │
│ - Metrics per fold         │
│ - Selected features        │
│ - Best parameters          │
│ - Models and components    │
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│ Split Data into Train/Val  │
│ Print fold statistics      │
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│ Apply Feature Scaling      │
│ Store scaler for fold      │
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│ Feature Selection:         │
│ - LGBM-based               │
│ - K-best (f_regression)    │
│ - K-best (mutual_info)     │
│ - None (use all features)  │
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│ Hyperparameter Tuning      │
│ With Error Handling        │
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│ Build & Select Best Stack  │
│ For Both Tasks:            │
│ - Regression               │
│ - Classification           │
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│ Evaluate on Validation Set │
│ Store Performance Metrics  │
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│ Error Handling:            │
│ - Catch exceptions         │
│ - Store default values     │
│ - Continue to next fold    │
└────────────────────────────┘
```

### 3. HYPERPARAMETER TUNING DETAILS

```
┌───────────────────────────────┐
│ Begin Hyperparameter Tuning   │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ Configure Models to Tune:     │
│ - Main models (more trials)   │
│ - Other models (fewer trials) │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ Regression Models:            │
│ - Check optimization function │
│ - Set trials and direction    │
│ - Configure model parameters  │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ Run Optuna Studies:           │
│ - With timeout                │
│ - Error handling per model    │
│ - Store best parameters       │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ Classification Models:        │
│ - Similar process as above    │
│ - Maximize metrics           │
└───────────────────────────────┘
```

### 4. STACKING AND META-MODEL SELECTION

```
┌───────────────────────────────┐
│ Begin Meta-Model Selection    │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ Regression Stack:             │
│ - Use tuned base models       │
│ - Try meta-regressors         │
│ - Cross-validation            │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ Classification Stack:         │
│ - Use tuned base models       │
│ - Try meta-classifiers        │
│ - Cross-validation            │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ Evaluate Both Stacks:         │
│ - R2 and MAE for regression   │
│ - Accuracy and ROC-AUC for    │
│   classification              │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ Store Best Models and Results │
└───────────────────────────────┘
```

## IMPORTANT POINTS

1. **Robust Error Handling:**
   - Each major step has try-except blocks
   - Default values stored for failed folds
   - Detailed error logging and tracebacks
   - Continues to next fold on failure

2. **Hyperparameter Tuning:**
   - Different trial counts for main vs other models
   - Configurable optimization settings
   - Timeout limits for optimization
   - Stores parameters per model and fold

3. **Feature Selection Options:**
   - LGBM-based importance selection
   - K-best with f_regression
   - K-best with mutual_info_regression
   - Option to skip selection

4. **Stacking Process:**
   - Separate stacks for regression and classification
   - Cross-validation during stacking
   - Multiple meta-model candidates
   - Comprehensive metric tracking

5. **Performance Tracking:**
   - Regression: R2 and MAE
   - Classification: Accuracy and ROC-AUC
   - Stores results per fold
   - Handles missing/failed metrics

## OUTPUTS

The process returns:
1. Outer fold results (regression and classification)
2. Selected features per fold
3. Best parameters for all models
4. Fitted preprocessing components
5. Trained stack models for both tasks

## IMPLEMENTATION NOTES

1. **Compute Configuration:**
   - Configurable compute resources
   - Device-specific settings
   - Resource-aware optimization

2. **Data Management:**
   - Maintains DataFrame structure
   - Tracks feature names
   - Handles data transformations

3. **Validation:**
   - Comprehensive metric tracking
   - Proper handling of edge cases
   - Robust error management
