import numpy as np
import lightgbm as lgb
import xgboost as xgb
import shap
from optuna.exceptions import TrialPruned
from functools import partial
from utils import get_compute_device_params

# scikit-learn imports
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier, 
    GradientBoostingRegressor, GradientBoostingClassifier,
    StackingRegressor, StackingClassifier
)
from sklearn.linear_model import Ridge, BayesianRidge, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, roc_auc_score
)

# Constants
RANDOM_STATE = 42   

# Get compute parameters once
COMPUTE_PARAMS = get_compute_device_params()

# =============================================================================
# 4. OPTUNA OBJECTIVE FUNCTIONS (Defined before the loop)
# =============================================================================
# Objective functions return the metric to be optimized (e.g., R2, -MSE, ROC-AUC)

# --- Custom Optuna Pruning Callback for LightGBM ---
def optuna_lgbm_pruning_callback(trial, metric, env):
    """
    Custom Optuna pruning callback for LightGBM.
    Monitors the specified metric on the first validation set ('valid_0').
    Reports the metric to Optuna and checks for pruning.
    """
    current_score = None
    # env.evaluation_result_list format: [('data_name', 'metric_name', value, is_higher_better), ...]
    for data_name, metric_name, value, _ in env.evaluation_result_list:
        if data_name == 'valid_0' and metric_name == metric:
            current_score = value
            break

    if current_score is None:
        # Metric not found, could happen in initial iterations or if eval_set is different
        # print(f"Warning: Metric '{metric}' not found at iteration {env.iteration} in results: {env.evaluation_result_list}")
        return # Continue training if metric not available yet

    trial.report(current_score, step=env.iteration)

    if trial.should_prune():
        message = f"Trial was pruned at iteration {env.iteration} with score {current_score:.4f}."
        raise TrialPruned(message)


# --- Regression Objectives ---
def optimize_lgbm_reg(trial, X_tr, y_tr, X_va, y_va):
    params = {
        'objective': 'regression_l1', 'metric': 'l1', 'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'device': COMPUTE_PARAMS['lgbm_device'], 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': -1
    }
    model = lgb.LGBMRegressor(**params)
    try:
        # Create the partial function for the custom callback
        pruning_callback = partial(optuna_lgbm_pruning_callback, trial, 'l1')

        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='l1',
                  callbacks=[pruning_callback, # Use custom pruning callback
                             lgb.early_stopping(stopping_rounds=50, verbose=False)])
        preds = model.predict(X_va)
        score = mean_absolute_error(y_va, preds)
        if np.isnan(score): raise TrialPruned("MAE score is NaN")
        return score
    except TrialPruned as e: # Catch pruned trials specifically
        print(f"Trial pruned during optimization: {e}")
        raise # Re-raise TrialPruned for Optuna
    except Exception as e:
        print(f"Trial failed for LGBM Reg: {e}")
        raise TrialPruned() # Prune on other errors

def optimize_xgb_reg(trial, X_tr, y_tr, X_va, y_va):
    params = {
        'objective': 'reg:squarederror', 'n_estimators': 1000,
        'eta': trial.suggest_float('eta', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'tree_method': COMPUTE_PARAMS['xgb_tree_method'], 'random_state': RANDOM_STATE, 'verbosity': 0
    }
    xgb_params = params.copy()
    xgb_params['alpha'] = xgb_params.pop('reg_alpha')
    xgb_params['lambda'] = xgb_params.pop('reg_lambda')

    model = xgb.XGBRegressor(**xgb_params)
    try:
        # Add early stopping to prevent overfitting and reduce computation time
        model.fit(
            X_tr, y_tr, 
            eval_set=[(X_va, y_va)],
            eval_metric='rmse',
            early_stopping_rounds=50,
            verbose=False
        )
        preds = model.predict(X_va)
        mse = mean_squared_error(y_va, preds)
        score = np.sqrt(mse)
        if np.isnan(score): raise TrialPruned("RMSE score is NaN")
        return score
    except Exception as e:
        print(f"Trial failed for XGB Reg: {e}")
        raise TrialPruned()

def optimize_rf_reg(trial, X_tr, y_tr, X_va, y_va):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 5, 50, log=True)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_float('max_features', 0.1, 1.0)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                  min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                  max_features=max_features, random_state=RANDOM_STATE, n_jobs=-1)
    try:
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        score = r2_score(y_va, preds)
        if np.isnan(score): raise TrialPruned("R2 score is NaN")
        return score
    except Exception as e:
        print(f"Trial failed for RF Reg: {e}")
        raise TrialPruned()

def optimize_ridge(trial, X_tr, y_tr, X_va, y_va):
    alpha = trial.suggest_float('alpha', 1e-4, 1e2, log=True)
    model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
    try:
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        score = r2_score(y_va, preds)
        if np.isnan(score): raise TrialPruned("R2 score is NaN")
        return score
    except Exception as e:
        print(f"Trial failed for Ridge: {e}")
        raise TrialPruned()

def optimize_mlp_reg(trial, X_tr, y_tr, X_va, y_va):
    hidden_layer_sizes = (trial.suggest_int('n_units_l1', 32, 128), trial.suggest_int('n_units_l2', 16, 64))
    activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
    solver = trial.suggest_categorical('solver', ['adam', 'sgd'])
    alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver,
                         alpha=alpha, learning_rate_init=learning_rate_init, max_iter=500,
                         early_stopping=True, random_state=RANDOM_STATE)
    try:
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        score = r2_score(y_va, preds)
        if np.isnan(score): raise TrialPruned("R2 score is NaN")
        return score
    except Exception as e:
        print(f"Trial failed for MLP Reg: {e}")
        raise TrialPruned()

def optimize_knn_reg(trial, X_tr, y_tr, X_va, y_va):
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', 3, 30),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'p': trial.suggest_int('p', 1, 2)
    }
    model = KNeighborsRegressor(**params, n_jobs=-1)
    try:
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        score = r2_score(y_va, preds)
        if np.isnan(score): raise TrialPruned("R2 score is NaN")
        return score
    except Exception as e:
        print(f"Trial failed for KNN Reg: {e}")
        raise TrialPruned()

# --- Classification Objectives ---
def optimize_lgbm_cls(trial, X_tr, y_tr, X_va, y_va):
    params = {
        'objective': 'binary', 'metric': 'auc', 'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'device': COMPUTE_PARAMS['lgbm_device'], 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': -1
    }
    model = lgb.LGBMClassifier(**params)
    try:
        # Create the partial function for the custom callback
        pruning_callback = partial(optuna_lgbm_pruning_callback, trial, 'auc')

        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc',
                  callbacks=[pruning_callback, # Use custom pruning callback
                             lgb.early_stopping(stopping_rounds=50, verbose=False)])
        preds = model.predict_proba(X_va)[:, 1]
        score = roc_auc_score(y_va, preds)
        if np.isnan(score): raise TrialPruned("ROC-AUC score is NaN")
        return score
    except TrialPruned as e: # Catch pruned trials specifically
        print(f"Trial pruned during optimization: {e}")
        raise # Re-raise TrialPruned for Optuna
    except Exception as e:
        print(f"Trial failed for LGBM Cls: {e}")
        raise TrialPruned() # Prune on other errors

def optimize_xgb_cls(trial, X_tr, y_tr, X_va, y_va):
    params = {
        'objective': 'binary:logistic', 'n_estimators': 1000,
        'eta': trial.suggest_float('eta', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'tree_method': COMPUTE_PARAMS['xgb_tree_method'], 'random_state': RANDOM_STATE, 'verbosity': 0
    }
    xgb_params = params.copy()
    xgb_params['alpha'] = xgb_params.pop('reg_alpha')
    xgb_params['lambda'] = xgb_params.pop('reg_lambda')

    model = xgb.XGBClassifier(**xgb_params)
    try:
        # Simplest fit call for diagnostics - no early stopping/pruning here
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_va)[:, 1]
        score = roc_auc_score(y_va, preds)
        if np.isnan(score): raise TrialPruned("ROC-AUC score is NaN")
        return score
    except Exception as e:
        print(f"Trial failed for XGB Cls: {e}")
        raise TrialPruned()

def optimize_rf_cls(trial, X_tr, y_tr, X_va, y_va):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 50, log=True),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'random_state': RANDOM_STATE, 'n_jobs': -1
    }
    model = RandomForestClassifier(**params)
    try:
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_va)[:, 1]
        score = roc_auc_score(y_va, preds)
        if np.isnan(score): raise TrialPruned("ROC-AUC score is NaN")
        return score
    except Exception as e:
        print(f"Trial failed for RF Cls: {e}")
        raise TrialPruned()

def optimize_logistic_regression(trial, X_tr, y_tr, X_va, y_va):
    C = trial.suggest_float('C', 1e-4, 1e2, log=True)
    solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    valid_combo = False
    if solver == 'liblinear' and penalty in ['l1', 'l2']: valid_combo = True
    elif solver == 'saga' and penalty in ['l1', 'l2']: valid_combo = True

    if not valid_combo:
      raise TrialPruned(f"Invalid combo: solver={solver}, penalty={penalty}")

    model = LogisticRegression(C=C, solver=solver, penalty=penalty, max_iter=2000,
                               random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')
    try:
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_va)[:, 1]
        score = roc_auc_score(y_va, preds)
        if np.isnan(score): raise TrialPruned("ROC-AUC score is NaN")
        return score
    except ValueError as e:
        print(f"Trial failed for LR: {e}")
        raise TrialPruned()

def optimize_mlp_cls(trial, X_tr, y_tr, X_va, y_va):
    hidden_layer_sizes = (trial.suggest_int('n_units_l1', 32, 128), trial.suggest_int('n_units_l2', 16, 64))
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
    solver = trial.suggest_categorical('solver', ['adam', 'sgd'])
    alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
    params = {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'solver': solver,
        'alpha': alpha,
        'learning_rate_init': learning_rate_init,
        'max_iter': 500,
        'early_stopping': True,
        'random_state': RANDOM_STATE
    }
    model = MLPClassifier(**params)
    try:
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_va)[:, 1]
        score = roc_auc_score(y_va, preds)
        if np.isnan(score): raise TrialPruned("ROC-AUC score is NaN")
        return score
    except Exception as e:
        print(f"Trial failed for MLP Cls: {e}")
        raise TrialPruned()

def optimize_svc(trial, X_tr, y_tr, X_va, y_va):
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
    
    if kernel == 'linear':
        gamma = 'scale'
        degree = 3
    elif kernel == 'poly':
        gamma = trial.suggest_float('gamma', 1e-4, 1e0, log=True)
        degree = trial.suggest_int('degree', 2, 5)
    else:  
        gamma = trial.suggest_float('gamma', 1e-4, 1e0, log=True)
        degree = 3

    params = {
        'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
        'kernel': kernel,
        'gamma': gamma,
        'degree': degree,
        'probability': True,
        'random_state': RANDOM_STATE
    }

    model = SVC(**params)
    try:
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_va)[:, 1]
        score = roc_auc_score(y_va, preds)
        if np.isnan(score): raise TrialPruned("ROC-AUC score is NaN")
        return score
    except Exception as e:
        print(f"Trial failed for SVC: {e}")
        raise TrialPruned()

# --- Helper Function for Model Instantiation ---
def get_final_model_instance(ModelClass, best_params, **kwargs):
    if not best_params:
        print(f"Warning: No best parameters found for {ModelClass.__name__}. Using default parameters.")
        return ModelClass(random_state=RANDOM_STATE, **kwargs) if 'random_state' in ModelClass().get_params() else ModelClass(**kwargs)

    current_best_params = best_params.copy()
    final_params = {**current_best_params, **kwargs}

    valid_params = ModelClass().get_params().keys()
    cleaned_params = {k: v for k, v in final_params.items() if k in valid_params}

    if 'random_state' in valid_params:
        cleaned_params['random_state'] = RANDOM_STATE
    if 'n_jobs' in valid_params:
         cleaned_params['n_jobs'] = -1
    if ModelClass == SVC and 'probability' in valid_params:
        cleaned_params['probability'] = True

    try:
        final_model = ModelClass(**cleaned_params)
    except TypeError as e:
        print(f"Error instantiating {ModelClass.__name__} with params {cleaned_params}: {e}")
        print("Attempting instantiation with default parameters.")
        final_model = ModelClass(random_state=RANDOM_STATE, **kwargs) if 'random_state' in ModelClass().get_params() else ModelClass(**kwargs)

    return final_model, current_best_params


# --- Model Definition Functions ---
def get_base_regressors(random_state=42):
    """
    Returns a dictionary of base regression models (6 total).
    Each is instantiated with default or minimal parameters, 
    plus a random_state for reproducibility.
    """
    return {
        # Tree-based learners
        'LGBM': lgb.LGBMRegressor(random_state=random_state, n_jobs=-1, verbosity=-1),
        'XGB': xgb.XGBRegressor(random_state=random_state, verbosity=0),
        'RandomForest': RandomForestRegressor(random_state=random_state, n_jobs=-1),

        # Linear model
        'Ridge': Ridge(random_state=random_state),

        # Neural network
        'MLP': MLPRegressor(random_state=random_state, max_iter=500, hidden_layer_sizes=(128, 64), activation='relu', solver='adam', alpha=1e-5, learning_rate_init=1e-3, early_stopping=True, n_iter_no_change=100),

        # Distance-based
        'KNN': KNeighborsRegressor(n_jobs=-1),
    }


def get_base_classifiers(random_state=42):
    """
    Returns a dictionary of base classification models (6 total).
    Each is instantiated with default or minimal parameters, 
    plus a random_state for reproducibility.
    """
    return {
        # Tree-based learners
        'LGBM': lgb.LGBMClassifier(random_state=random_state, n_jobs=-1, verbosity=-1),
        'XGB': xgb.XGBClassifier(random_state=random_state, verbosity=0),
        'RandomForest': RandomForestClassifier(random_state=random_state, n_jobs=-1),

        # Linear model
        'LogisticRegression': LogisticRegression(random_state=random_state, max_iter=1000, n_jobs=-1),

        # Neural network
        'MLP': MLPClassifier(random_state=random_state, max_iter=500, hidden_layer_sizes=(128, 64), activation='relu', solver='adam', alpha=1e-5, learning_rate_init=1e-3),

        # Support Vector
        'SVC': SVC(probability=True, random_state=random_state),
    }


def get_meta_regressor_candidates(random_state=42):
    """
    Returns a dictionary of meta regression models (3 total).
    Each model is well-suited as a meta-learner to stack base regressors.
    """
    return {
        # One tree-based option
        'XGB': xgb.XGBRegressor(random_state=random_state, verbosity=0, n_estimators=100),

        # Linear options
        'Ridge': Ridge(random_state=random_state),
        'BayesianRidge': BayesianRidge(random_state=random_state),
    }


def get_meta_classifier_candidates(random_state=42):
    """
    Returns a dictionary of meta classification models (3 total).
    Each model is well-suited as a meta-learner to stack base classifiers.
    """
    return {
        # Linear meta-learner
        'LogisticRegression': LogisticRegression(
            random_state=random_state, 
            class_weight='balanced', 
            max_iter=1000, 
            n_jobs=-1
        ),

        # Tree-based meta-learner
        'RandomForest': RandomForestClassifier(
            random_state=random_state, 
            n_jobs=-1, 
            n_estimators=100
        ),

        # Another tree-based or boosting option
        'XGB': xgb.XGBClassifier(random_state=random_state, verbosity=0),
    }

# select_best_stack is defined in this file and passed as an arg to outer_cv.py
def select_best_stack(base_model_definitions, tuned_base_params, meta_candidates,
                      X_tr, y_tr, X_va, y_va, task='regression', cv_folds=5,
                      random_state=42, compute_params=None):
    """
    Builds and selects the best stacking ensemble from tuned base models and meta-learner candidates.
    
    Args:
        base_model_definitions (dict): Dictionary of base model classes
        tuned_base_params (dict): Dictionary of tuned parameters for base models
        meta_candidates (dict): Dictionary of meta-learner candidates
        X_tr (pd.DataFrame): Training features
        y_tr (pd.Series): Training targets
        X_va (pd.DataFrame): Validation features
        y_va (pd.Series): Validation targets
        task (str): 'regression' or 'classification'
        cv_folds (int): Number of CV folds for stacking
        random_state (int): Random state for reproducibility
        compute_params (dict): Dictionary of compute-related parameters
    
    Returns:
        tuple: (best_stack, best_base_models, best_meta)
            - best_stack: Best stacking ensemble model
            - best_base_models: Dictionary of tuned base models used in the stack
            - best_meta: Meta-learner used in the best stack
    """
    print(f"  Building stacking ensemble for {task}...")
    
    # Instantiate tuned base models
    tuned_base_models = {}
    for model_name, model_instance in base_model_definitions.items():
        if model_name in tuned_base_params and tuned_base_params[model_name]:
            print(f"    Instantiating tuned {model_name}...")
            try:
                # Use helper function to get correct instance with tuned parameters
                model, _ = get_final_model_instance(model_instance.__class__, tuned_base_params[model_name])
                tuned_base_models[model_name] = model
            except Exception as e:
                print(f"    Error instantiating {model_name}: {e}")
                # Fallback to default model
                tuned_base_models[model_name] = model_instance
        else:
            print(f"    Using default {model_name} (no tuned params)...")
            tuned_base_models[model_name] = model_instance

    # Try each meta-learner and select the best one
    best_score = -np.inf if task == 'regression' else 0 # Initialize regression best score to negative infinity
    best_stack = None
    best_meta = None
    best_meta_name = None
    
    for meta_name, meta_model in meta_candidates.items():
        print(f"    Trying meta-learner: {meta_name}...")
        
        try:
            # Create named tuned estimators for the stack
            named_estimators = [(name, model) for name, model in tuned_base_models.items()]
            
            # Create and fit the stacking ensemble
            if task == 'regression':
                stack = StackingRegressor(
                    estimators=named_estimators,
                    final_estimator=meta_model,
                    cv=cv_folds,
                    n_jobs=-1,
                    passthrough=False
                )
            else:  # classification
                stack = StackingClassifier(
                    estimators=named_estimators,
                    final_estimator=meta_model,
                    cv=cv_folds,
                    n_jobs=-1,
                    passthrough=False
                )
            
            # Fit on training data
            print(f"      Fitting stack with {meta_name}...")
            stack.fit(X_tr, y_tr)
            print(f"      Stack fitting finished for {meta_name}.")

            # Predict and evaluate on validation data
            if task == 'regression':
                y_pred = stack.predict(X_va)
                score = r2_score(y_va, y_pred)
                print(f"      Validation R2 for {meta_name}: {score:.4f} (is_finite: {np.isfinite(score)})") # Added finiteness check
            else:
                y_pred_proba = stack.predict_proba(X_va)[:, 1]
                score = roc_auc_score(y_va, y_pred_proba)
                print(f"      Validation ROC-AUC for {meta_name}: {score:.4f} (is_finite: {np.isfinite(score)})") # Added finiteness check

            # Update best if improved and score is valid
            if np.isfinite(score): # Only update if score is a valid number
                if (task == 'regression' and score > best_score) or (task == 'classification' and score > best_score):
                    best_score = score
                    best_stack = stack
                    best_meta = meta_model
                    best_meta_name = meta_name
                    print(f"      >>> New best meta-learner: {meta_name} (Score: {score:.4f})")
            else:
                 print(f"      Score for {meta_name} is not finite. Skipping update.")

        except Exception as e:
            import traceback # Import traceback for detailed error info
            print(f"      ERROR with meta-learner {meta_name}: {e}")
            print(f"      Traceback: {traceback.format_exc()}") # Print full traceback
            continue

    if best_stack is None:
        print("  Failed to build any valid stacking ensemble. Returning None.")
        return None, {}, None
    
    print(f"  Selected best stacking ensemble with meta-learner: {best_meta_name} (Score: {best_score:.4f})")
    return best_stack, tuned_base_models, best_meta
