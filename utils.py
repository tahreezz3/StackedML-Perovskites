"""
Utility functions for the ML pipeline.
"""

import optuna
import time
import warnings # Ensure warnings is imported
import uuid


# Suppress Optuna trial INFO logging & specific warnings
# optuna.logging.set_verbosity(optuna.logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.INFO) # Changed to INFO for debugging
warnings.filterwarnings('ignore', category=UserWarning, message='.*Starting from version 2.2.1.*')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero.*')
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning) # For LightGBMPruningCallback

def get_compute_device_params():
    """Returns hardcoded CPU device parameters."""
    params = {'lgbm_device': 'cpu', 'xgb_tree_method': 'auto', 'xgb_device': 'cpu'}
    print("Using hardcoded CPU parameters: lgbm_device='cpu', xgb_tree_method='auto'")
    return params

def run_optuna_study(objective, X_tr, y_tr, X_va, y_va, n_trials, direction, study_name, timeout=None):
    """Runs an Optuna study with fault-tolerant SQLite loading and fallback."""
    print(f"Starting Optuna study: {study_name} ({n_trials} trials, direction={direction})")
    start_time = time.time()

    func = lambda trial: objective(trial, X_tr, y_tr, X_va, y_va)

    base_name = study_name.replace(' ', '_').lower()
    storage_name = f"sqlite:///{base_name}.db"

    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction=direction,
            load_if_exists=True
        )
        print(f"Loaded or created study '{study_name}' from {storage_name}")
    except Exception as e:
        print(f"Failed to load Optuna study '{study_name}': {e}")
        fallback_name = f"{base_name}_{uuid.uuid4().hex[:8]}"
        fallback_storage = f"sqlite:///{fallback_name}.db"
        print(f"Falling back to study '{fallback_name}' at {fallback_storage}")
        study = optuna.create_study(
            study_name=fallback_name,
            storage=fallback_storage,
            direction=direction
        )

    # Study metadata tracking
    if not study.user_attrs.get('last_run_timestamp') or study.user_attrs.get('run_count', 0) == 0:
        study.set_user_attr('creation_timestamp', time.strftime('%Y-%m-%d %H:%M:%S'))
    study.set_user_attr('last_run_timestamp', time.strftime('%Y-%m-%d %H:%M:%S'))
    study.set_user_attr('run_count', study.user_attrs.get('run_count', 0) + 1)

    # Only run remaining trials
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    to_run = max(0, n_trials - completed)
    if to_run > 0:
        print(f"Running {to_run} new trials (target: {n_trials})...")
        study.optimize(func, n_trials=to_run, timeout=timeout, n_jobs=1)
    else:
        print(f"Study already has {completed} completed trials.")

    print(f"Optuna study finished in {time.time() - start_time:.2f} seconds.")
    print(f"Best trial: Value={study.best_value:.4f}, Params={study.best_params}")

    return study