import os
import pandas as pd
import numpy as np
import pickle
import json
import glob
from datetime import datetime
from collections import defaultdict

from autogluon.tabular import TabularPredictor
from autogluon.tabular import TabularDataset

from ngboost import NGBRegressor
from ngboost.distns import Normal
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RandomizedSearchCV
from ngboost import NGBRegressor
from ngboost.distns import Normal
from scipy.stats import loguniform, randint
import warnings

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from functools import partial

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Utility Functions
# =============================================================================

def detect_problem_type(series, threshold=15):
    """Detect if target is classification or regression."""
    n_unique = series.nunique()
    n_total = len(series)
    unique_ratio = n_unique / n_total
    is_numeric = pd.api.types.is_numeric_dtype(series)

    if not is_numeric:
        return 'classification'
    if n_unique <= 2:
        return 'classification'
    # elif unique_ratio < (threshold / 100) and series.dtype in ['int64', 'int32']:
    #     return 'classification'
    else:
        return 'regression'


def aggregate_targets_to_distribution(target_values, problem_type='regression'):
    """
    Convert multiple target values to a distribution representation.
    
    Parameters:
    -----------
    target_values : list or array
        Multiple target values for a single scenario
    problem_type : str
        'regression' or 'classification'
    
    Returns:
    --------
    dict with distribution info (for regression) or mode counts (for classification)
    """
    if len(target_values) == 0:
        return None
    
    # Ensure we're working with a numpy array
    target_values = np.array(target_values)
    
    if problem_type == 'classification':
        # For classification: count frequencies, don't compute mean/std
        unique, counts = np.unique(target_values, return_counts=True)
        mode_val = unique[np.argmax(counts)]
        mode_count = np.max(counts)
        
        return {
            'mode': mode_val,
            'mode_frequency': int(mode_count),
            'mode_probability': float(mode_count / len(target_values)),
            'n_samples': len(target_values),
            'distribution': {str(u): int(c) for u, c in zip(unique, counts)},
            'values': target_values.tolist()
        }
    
    else:  # regression
        # Ensure numeric type for regression
        try:
            numeric_values = target_values.astype(float)
        except ValueError as e:
            raise ValueError(
                f"Cannot convert target values to float for regression: {target_values[:5]}... "
                f"Error: {e}"
            )
        
        mean_val = np.mean(numeric_values)
        std_val = np.std(numeric_values) if len(numeric_values) > 1 else 0.0
        
        return {
            'mean': float(mean_val),
            'std': float(std_val),
            'min': float(np.min(numeric_values)),
            'max': float(np.max(numeric_values)),
            'n_samples': len(numeric_values),
            'values': numeric_values.tolist()
        }


def get_already_trained_classification_targets(output_dir="models/multilabel_cls"):
    """Check which classification targets already have trained models."""
    trained = set()
    if not os.path.exists(output_dir):
        return trained
    meta_files = glob.glob(os.path.join(output_dir, "multilabel_metadata_*.json"))
    for mf in meta_files:
        with open(mf, 'r') as f:
            meta = json.load(f)
        for t in meta.get('targets', []):
            model_path = meta['per_target'][t].get('model_path', '')
            if os.path.isdir(model_path):
                trained.add(t)
    for d in os.listdir(output_dir):
        full = os.path.join(output_dir, d)
        if os.path.isdir(full) and d != '__pycache__':
            parts = d.rsplit('_', 2)
            if len(parts) >= 3:
                target_name = '_'.join(parts[:-2])
                if os.path.exists(os.path.join(full, 'predictor.pkl')) or \
                   os.path.exists(os.path.join(full, 'models')):
                    trained.add(target_name)
    return trained


def get_already_trained_regression_targets(output_dir="models/multitarget_reg"):
    """Check which regression targets already have trained models."""
    trained = set()
    if not os.path.exists(output_dir):
        return trained
    meta_files = glob.glob(os.path.join(output_dir, "ngboost_metadata_*.json"))
    for mf in meta_files:
        with open(mf, 'r') as f:
            meta = json.load(f)
        bundle_path = meta.get('model_bundle_path', '')
        if os.path.exists(bundle_path):
            trained.update(meta.get('targets', []))
    return trained


# =============================================================================
# NEW: Grouped Data Loading by base_scenario
# =============================================================================

def load_and_merge_datasets_grouped(training_dir="training", group_col='scenario_name'):
    """
    Load datasets and group by base_scenario.
    Each scenario has: single features + multiple target values (distribution).
    """
    all_targets = {}
    feature_frames = []
    
    index_col_candidates = ['scenario_name', 'base_scenario']

    files = sorted([f for f in os.listdir(training_dir) if f.endswith('.csv')])
    if not files:
        raise FileNotFoundError(f"No CSV files found in {training_dir}/")

    print(f"Found {len(files)} CSV file(s) in '{training_dir}/'")

    reference_features = None
    raw_dfs = {}

    for file in files:
        file_path = os.path.join(training_dir, file)
        df = pd.read_csv(file_path)
        
        train_list_path = "train.csv"
        if os.path.exists(train_list_path):
            train_list = pd.read_csv(train_list_path)["scenario_name"].tolist()
            df = df[df["scenario_name"].str.startswith(tuple(train_list))]
        
        target = file.replace("training_dataset_", "").split(".")[0]

        if group_col not in df.columns:
            raise ValueError(f"Grouping column '{group_col}' not found in {file}. Available: {df.columns.tolist()}")

        cols_to_drop = [c for c in index_col_candidates if c in df.columns] + [target]
        feature_df = df.drop(columns=cols_to_drop, errors='ignore')
        numeric_cols = feature_df.select_dtypes(include=['number']).columns.tolist()

        raw_dfs[file] = {
            'df': df,
            'target': target,
            'numeric_features': numeric_cols,
            'group_col': group_col
        }

        if reference_features is None:
            reference_features = set(numeric_cols)
        else:
            reference_features = reference_features.intersection(numeric_cols)

    shared_features = sorted(reference_features)
    print(f"Shared numeric features across all files: {len(shared_features)}")

    # Group by base_scenario for each file
    grouped_scenarios = defaultdict(lambda: {
        'features': None,
        'targets': {},
        'n_observations': {}
    })

    target_types = {}

    for file, info in raw_dfs.items():
        df = info['df']
        target = info['target']
        print(f"\nProcessing {file}: target='{target}'")

        for scenario_name, group in df.groupby(group_col):
            feat_values = group[shared_features].iloc[0].values
            tgt_values = group[target].values.tolist()
            
            if grouped_scenarios[scenario_name]['features'] is None:
                grouped_scenarios[scenario_name]['features'] = feat_values
            
            grouped_scenarios[scenario_name]['targets'][target] = tgt_values
            grouped_scenarios[scenario_name]['n_observations'][target] = len(tgt_values)
        
        all_tgt_values = df[target].values
        target_types[target] = detect_problem_type(pd.Series(all_tgt_values))
        print(f"  -> {target_types[target]} ({len(df)} total rows, {df[group_col].nunique()} unique scenarios)")

    # Convert to DataFrames
    scenario_names = list(grouped_scenarios.keys())
    features_array = np.array([grouped_scenarios[s]['features'] for s in scenario_names])
    features_df = pd.DataFrame(features_array, columns=shared_features, index=scenario_names)
    features_df.index.name = group_col

    # Aggregate targets
    targets_aggregated = {}
    
    for target in target_types.keys():
        if target_types[target] == 'classification':
            targets_aggregated[target] = [
                pd.Series(grouped_scenarios[s]['targets'][target]).mode()[0]
                if len(grouped_scenarios[s]['targets'][target]) > 0 else np.nan
                for s in scenario_names
            ]
        else:
            targets_aggregated[target] = [
                np.mean(grouped_scenarios[s]['targets'][target])
                if len(grouped_scenarios[s]['targets'][target]) > 0 else np.nan
                for s in scenario_names
            ]
    
    targets_df = pd.DataFrame(targets_aggregated, index=scenario_names)
    targets_df.index.name = group_col

    # Store full distribution info
    # In load_and_merge_datasets_grouped(), change the aggregation call to:
    distribution_info = {
        s: {
            'targets': {
                t: aggregate_targets_to_distribution(
                    grouped_scenarios[s]['targets'][t],
                    problem_type=target_types[t]  # Pass the detected type
                )
                for t in grouped_scenarios[s]['targets'].keys()
            },
            'n_observations': grouped_scenarios[s]['n_observations']
        }
        for s in scenario_names
    }

    print(f"\n{'='*70}")
    print(f"GROUPED DATA SUMMARY")
    print(f"{'='*70}")
    print(f"Unique scenarios: {len(scenario_names)}")
    print(f"Features per scenario: {len(shared_features)}")
    avg_obs = np.mean([sum(d['n_observations'].values()) for d in distribution_info.values()])
    print(f"Average observations per scenario: {avg_obs:.1f}")

    return features_df, targets_df, target_types, shared_features, distribution_info


# =============================================================================
# Classification: AutoGluon (handles aggregated targets)
# =============================================================================

def train_multilabel_classification(features_df, targets_df, cls_targets, 
                                    distribution_info=None, output_dir="models/multilabel_cls"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    already_trained = get_already_trained_classification_targets(output_dir)
    targets_to_train = [t for t in cls_targets if t not in already_trained]
    skipped = [t for t in cls_targets if t in already_trained]

    if skipped:
        print(f"\n  SKIPPING already-trained classification targets: {skipped}")
    if not targets_to_train:
        print(f"\n  All classification targets already trained. Nothing to do.")
        return {}, {}

    print(f"  Training classification targets: {targets_to_train}")

    train_df = features_df.copy()
    for t in targets_to_train:
        train_df[t] = targets_df[t].values

    predictors = {}
    all_metadata = {}

    for target in targets_to_train:
        print(f"\n{'='*70}")
        print(f"CLASSIFICATION TARGET: {target}")
        print(f"{'='*70}")

        n_unique = train_df[target].nunique()
        problem_type = 'binary' if n_unique <= 2 else 'multiclass'
        eval_metric = 'roc_auc' if problem_type == 'binary' else 'accuracy'

        print(f"  Problem type: {problem_type}")
        print(f"  Eval metric:  {eval_metric}")
        print(f"  Classes:      {n_unique}")
        print(f"  Note: Training on aggregated targets (majority class per scenario)")

        predictor_path = os.path.join(output_dir, f"{target}_{timestamp}")

        predictor = TabularPredictor(
            label=target,
            path=predictor_path,
            eval_metric=eval_metric,
            problem_type=problem_type,
        ).fit(
            train_data=train_df[list(features_df.columns) + [target]],
            presets='best_quality',
        )

        lb = predictor.leaderboard(silent=True)
        print(f"\n  Top 5 models:")
        print(lb[['model', 'score_val', 'fit_time']].head(5).to_string(index=False))

        best_model = lb.iloc[0]['model']
        best_score = lb.iloc[0]['score_val']
        print(f"\n  Best model: {best_model}  |  Score: {best_score:.6f}")

        predictors[target] = predictor
        all_metadata[target] = {
            'problem_type': problem_type,
            'eval_metric': eval_metric,
            'best_model': best_model,
            'best_score': float(best_score),
            'model_path': predictor_path,
            'n_classes': int(n_unique),
            'aggregation_method': 'majority_vote',
            'leaderboard': lb[['model', 'score_val', 'fit_time']].head(10).to_dict('records'),
        }

    bundle_meta = {
        'type': 'multilabel_classification',
        'targets': targets_to_train,
        'skipped_targets': skipped,
        'timestamp': timestamp,
        'features': list(features_df.columns),
        'grouping': 'base_scenario',
        'per_target': all_metadata,
    }
    meta_path = os.path.join(output_dir, f"multilabel_metadata_{timestamp}.json")
    with open(meta_path, 'w') as f:
        json.dump(bundle_meta, f, indent=2, default=str)
    print(f"\nClassification bundle metadata saved to: {meta_path}")

    return predictors, bundle_meta


# =============================================================================
# Regression: NGBoost with Distribution-Aware Training
# =============================================================================

def objective(trial, X, y, kf, feature_names):
    """Optuna objective function for NGBoost hyperparameter optimization."""
    
    # Define hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'minibatch_frac': trial.suggest_categorical('minibatch_frac', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'col_sample': trial.suggest_categorical('col_sample', [0.6, 0.7, 0.8, 0.9, 1.0]),
    }
    
    cv_r2_scores = []
    
    # Cross-validation with early stopping support
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        try:
            model = NGBRegressor(
                Dist=Normal,
                verbose=False,
                random_state=42,
                natural_gradient=True,
                **params
            )
            
            # Fit with early stopping to enable pruning
            model.fit(
                X_tr, y_tr,
                X_val=X_val, 
                Y_val=y_val,
                early_stopping_rounds=50
            )
            
            # Calculate R² score
            y_dist = model.pred_dist(X_val)
            y_pred = y_dist.mean()
            fold_r2 = r2_score(y_val, y_pred)
            cv_r2_scores.append(fold_r2)
            
            # Report intermediate result for pruning (using first fold as proxy)
            if fold_idx == 0:
                trial.report(fold_r2, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                    
        except Exception as e:
            # Return poor score if training fails
            return -np.inf
    
    if not cv_r2_scores:
        return -np.inf
    
    return np.mean(cv_r2_scores)

def train_multitarget_ngboost(features_df, targets_df, reg_targets, 
                              distribution_info=None, output_dir="models/multitarget_reg",
                              n_trials=100, n_threads=32):
    """
    Train NGBoost models with TPE hyperparameter tuning for each target separately.
    Each model is saved to its own .pkl file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    already_trained = get_already_trained_regression_targets(output_dir)
    targets_to_train = [t for t in reg_targets if t not in already_trained]
    skipped = [t for t in reg_targets if t in already_trained]

    if skipped:
        print(f"\n  SKIPPING already-trained regression targets: {skipped}")
    if not targets_to_train:
        print(f"\n  All regression targets already trained. Nothing to do.")
        return {}, None, {}

    print(f"  Training regression targets: {targets_to_train}")
    print(f"  Hyperparameter tuning: {n_trials} trials using TPE sampler ({n_threads} threads)")
    print(f"  Each model will be saved to separate .pkl file")

    X = features_df.values.copy()
    feature_names = list(features_df.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler separately for reuse
    scaler_path = os.path.join(output_dir, f"scaler_{timestamp}.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved to: {scaler_path}")

    all_metadata = {}
    trained_models_info = {}

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for target in targets_to_train:
        print(f"\n{'='*70}")
        print(f"REGRESSION TARGET: {target}")
        print(f"{'='*70}")

        y = targets_df[target].values.astype(float)

        # Check distribution info if available
        has_distribution = distribution_info is not None
        if has_distribution:
            try:
                stds = [distribution_info[idx]['targets'][target]['std'] 
                        for idx in features_df.index 
                        if target in distribution_info[idx]['targets']]
                avg_std = np.mean(stds) if stds else 0
                print(f"  Average target std within scenarios: {avg_std:.4f}")
            except (KeyError, TypeError):
                print(f"  Warning: Could not extract distribution info for {target}")

        print(f"\n  Starting TPE hyperparameter optimization ({n_trials} trials)...")

        # Create Optuna study with TPE sampler and pruner
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(
                n_startup_trials=10,  # Random sampling first 10 trials
                n_ei_candidates=24,   # Number of candidate samples for expected improvement
                multivariate=True,    # Consider parameter correlations
                seed=42
            ),
            pruner=MedianPruner(
                n_startup_trials=5,   # Don't prune first 5 trials
                n_warmup_steps=2,     # Allow 2 folds before pruning
                interval_steps=1       # Check pruning every step
            )
        )

        # Optimize with parallel execution
        objective_func = partial(objective, X=X_scaled, y=y, kf=kf, feature_names=feature_names)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(objective_func, n_trials=n_trials, n_jobs=n_threads, show_progress_bar=True)

        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        print(f"\n  Best CV R²: {best_value:.6f}")
        print(f"  Best parameters: {best_params}")
        
        # Detailed statistics about optimization
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        print(f"  Trials: {completed_trials} completed, {pruned_trials} pruned")

        # Final cross-validation with best parameters to get detailed metrics
        print(f"\n  Calculating detailed metrics with best parameters...")
        
        cv_r2_scores = []
        cv_rmse_scores = []
        cv_nll_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            fold_model = NGBRegressor(
                Dist=Normal,
                verbose=False,
                random_state=42,
                natural_gradient=True,
                **best_params
            )
            fold_model.fit(X_tr, y_tr, X_val=X_val, Y_val=y_val, early_stopping_rounds=50)

            y_dist = fold_model.pred_dist(X_val)
            y_pred = y_dist.mean()
            
            fold_r2 = r2_score(y_val, y_pred)
            fold_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            fold_nll = -np.mean(y_dist.logpdf(y_val))

            cv_r2_scores.append(fold_r2)
            cv_rmse_scores.append(fold_rmse)
            cv_nll_scores.append(fold_nll)
            
            print(f"  Fold {fold_idx+1}: R²={fold_r2:.6f}  RMSE={fold_rmse:.6f}  NLL={fold_nll:.4f}")

        mean_r2 = np.mean(cv_r2_scores)
        std_r2 = np.std(cv_r2_scores)
        mean_rmse = np.mean(cv_rmse_scores)
        mean_nll = np.mean(cv_nll_scores)
        
        print(f"\n  Final CV R²:   {mean_r2:.6f} ± {std_r2:.6f}")
        print(f"  Final CV RMSE: {mean_rmse:.6f}")
        print(f"  Final CV NLL:  {mean_nll:.4f}")

        # Train final model on full dataset with best parameters
        print(f"\n  Training final model on full dataset...")
        final_model = NGBRegressor(
            Dist=Normal,
            verbose=False,
            random_state=42,
            natural_gradient=True,
            **best_params
        )
        final_model.fit(X_scaled, y)

        # Save individual model to separate .pkl file
        model_filename = f"ngboost_{target}_{timestamp}.pkl"
        model_path = os.path.join(output_dir, model_filename)
        
        model_bundle = {
            'model': final_model,
            'target': target,
            'scaler': scaler,
            'feature_names': feature_names,
            'best_params': best_params,
            'timestamp': timestamp,
            'optuna_study': study,  # Include full study for later analysis
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_bundle, f)
        
        print(f"\n  Model saved to: {model_path}")

        # Store metadata
        all_metadata[target] = {
            'model_path': model_path,
            'best_params': best_params,
            'best_cv_r2': float(best_value),
            'cv_r2_mean': float(mean_r2),
            'cv_r2_std': float(std_r2),
            'cv_rmse_mean': float(mean_rmse),
            'cv_nll_mean': float(mean_nll),
            'cv_r2_per_fold': [float(s) for s in cv_r2_scores],
            'cv_rmse_per_fold': [float(s) for s in cv_rmse_scores],
            'cv_nll_per_fold': [float(s) for s in cv_nll_scores],
            'target_distribution': 'Normal (mean and std predicted)',
            'aggregation_method': 'mean_per_scenario',
            'n_trials': n_trials,
            'n_threads': n_threads,
            'completed_trials': completed_trials,
            'pruned_trials': pruned_trials,
            'optimization_method': 'TPE (Tree-structured Parzen Estimator)',
        }
        
        trained_models_info[target] = {
            'model': final_model,
            'path': model_path
        }

    # Save comprehensive metadata JSON
    bundle_meta = {
        'type': 'multitarget_regression',
        'targets': targets_to_train,
        'skipped_targets': skipped,
        'timestamp': timestamp,
        'features': feature_names,
        'scaler_path': scaler_path,
        'n_trials': n_trials,
        'n_threads': n_threads,
        'optimization_method': 'TPE (Tree-structured Parzen Estimator)',
        'per_target': all_metadata,
        'grouping': 'scenario_name',
    }
    meta_path = os.path.join(output_dir, f"ngboost_metadata_{timestamp}.json")
    with open(meta_path, 'w') as f:
        json.dump(bundle_meta, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Metadata saved to: {meta_path}")
    print(f"Individual models: {len(targets_to_train)} files in {output_dir}")

    return trained_models_info, scaler, bundle_meta


# =============================================================================
# Inference Helpers (Updated for Grouped Data)
# =============================================================================

def predict_classification(predictors, new_data):
    """
    Predict classification targets.
    new_data should be a DataFrame with base_scenario as index or column.
    """
    predictions = {}
    probabilities = {}
    
    if isinstance(new_data, pd.Series):
        new_data = new_data.to_frame().T
    
    for target, predictor in predictors.items():
        predictions[target] = predictor.predict(new_data)
        try:
            probabilities[target] = predictor.predict_proba(new_data)
        except Exception:
            probabilities[target] = None
    
    return predictions, probabilities


def predict_regression(bundle_path, new_data_df):
    """
    Predict regression targets with full distribution.
    Returns mean and std for each prediction.
    """
    with open(bundle_path, 'rb') as f:
        bundle = pickle.load(f)

    scaler = bundle['scaler']
    feature_names = bundle['feature_names']
    models = bundle['models']

    if isinstance(new_data_df, dict):
        new_data_df = pd.DataFrame([new_data_df])
    
    X = new_data_df[feature_names].values
    X_scaled = scaler.transform(X)

    results = {}
    for target, model in models.items():
        dist = model.pred_dist(X_scaled)
        results[target] = {
            'mean': dist.mean(),
            'std': dist.std(),
            'distribution': 'Normal',
            'ci_lower': dist.mean() - 1.96 * dist.std(),
            'ci_upper': dist.mean() + 1.96 * dist.std(),
        }
    
    return results


def predict_with_uncertainty(bundle_path, new_data_df, n_samples=100):
    """
    Generate multiple samples from predicted distributions.
    Useful for downstream Monte Carlo analysis.
    """
    with open(bundle_path, 'rb') as f:
        bundle = pickle.load(f)

    scaler = bundle['scaler']
    feature_names = bundle['feature_names']
    models = bundle['models']

    X = new_data_df[feature_names].values
    X_scaled = scaler.transform(X)

    results = {}
    for target, model in models.items():
        dist = model.pred_dist(X_scaled)
        means = dist.mean()
        stds = dist.std()
        
        samples = np.array([
            np.random.normal(m, s, n_samples) 
            for m, s in zip(means, stds)
        ])
        
        results[target] = {
            'mean': means,
            'std': stds,
            'samples': samples,
        }
    
    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MULTI-TARGET TRAINING PIPELINE (GROUPED BY base_scenario)")
    print("  Classification -> AutoGluon (per-label, majority vote agg)")
    print("  Regression     -> NGBoost (probabilistic, distribution-aware)")
    print("=" * 70)

    # 1. Load and merge all datasets with grouping
    features_df, targets_df, target_types, shared_features, distribution_info = \
        load_and_merge_datasets_grouped("training_multi", group_col='base_scenario')

    print(f"\nDataset shape:   {features_df.shape[0]} scenarios x {features_df.shape[1]} features")
    print(f"Total targets:   {len(target_types)}")
    print(f"Grouping column: base_scenario")

    # 2. Split targets by type
    cls_targets = [t for t, tp in target_types.items() if tp == 'classification']
    reg_targets = [t for t, tp in target_types.items() if tp == 'regression']

    print(f"\nClassification targets ({len(cls_targets)}): {cls_targets}")
    print(f"Regression targets     ({len(reg_targets)}): {reg_targets}")

    already_cls = get_already_trained_classification_targets()
    already_reg = get_already_trained_regression_targets()
    print(f"\nAlready trained (classification): {already_cls or 'none'}")
    print(f"Already trained (regression):     {already_reg or 'none'}")

    # 3. Train classification models
    cls_predictors, cls_meta = None, None
    if cls_targets:
        print(f"\n\n{'#'*70}")
        print("# CLASSIFICATION TRAINING (AutoGluon)")
        print(f"{'#'*70}")
        cls_predictors, cls_meta = train_multilabel_classification(
            features_df, targets_df, cls_targets, distribution_info
        )
    else:
        print("\nNo classification targets found. Skipping.")

    # 4. Train regression models
    reg_models, reg_scaler, reg_meta = None, None, None
    reg_targets.sort()
    if reg_targets:
        print(f"\n\n{'#'*70}")
        print("# REGRESSION TRAINING (NGBoost)")
        print(f"{'#'*70}")
        reg_models, reg_scaler, reg_meta = train_multitarget_ngboost(
            features_df, targets_df, reg_targets, distribution_info
        )
    else:
        print("\nNo regression targets found. Skipping.")

    # 5. Summary
    print(f"\n\n{'='*70}")
    print("TRAINING COMPLETE - SUMMARY")
    print(f"{'='*70}")

    if cls_meta and cls_meta.get('per_target'):
        print(f"\nCLASSIFICATION (newly trained):")
        for t in cls_meta.get('targets', []):
            m = cls_meta['per_target'][t]
            print(f"  {t:30s}  best_score={m['best_score']:.6f}  model={m['best_model']}")
        if cls_meta.get('skipped_targets'):
            print(f"  Skipped: {cls_meta['skipped_targets']}")

    if reg_meta and reg_meta.get('per_target'):
        print(f"\nREGRESSION (newly trained):")
        for t in reg_meta.get('targets', []):
            m = reg_meta['per_target'][t]
            print(f"  {t:30s}  CV R²={m['cv_r2_mean']:.6f} ± {m['cv_r2_std']:.6f}  (NLL={m['cv_nll_mean']:.4f})")
        if reg_meta.get('skipped_targets'):
            print(f"  Skipped: {reg_meta['skipped_targets']}")

    print(f"\n{'='*70}")
    print("ALL DONE - Models trained on scenario-level distributions")
    print(f"{'='*70}")