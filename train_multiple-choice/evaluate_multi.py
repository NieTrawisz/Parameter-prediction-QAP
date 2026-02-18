import os
import pandas as pd
import numpy as np
import pickle
import json
import glob
from datetime import datetime

from autogluon.tabular import TabularPredictor
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Data Loading
# =============================================================================

def load_scenario_data(train_csv="train.csv", test_csv="test.csv", training_dir="training_multi"):
    """
    Load data grouped by scenario_name.
    Returns: dict[scenario_name] = {
        'features': pd.Series (same for all rows),
        'targets': {target_name: [list of available values]},
        'split': 'train' or 'test'
    }
    """
    train_scenarios = set(pd.read_csv(train_csv)["scenario_name"].tolist())
    test_scenarios = set(pd.read_csv(test_csv)["scenario_name"].tolist())
    
    print(f"Train scenarios: {len(train_scenarios)}")
    print(f"Test scenarios: {len(test_scenarios)}")
    
    # Load all target files
    files = sorted([f for f in os.listdir(training_dir) if f.endswith('.csv')])
    print(f"\nFound {len(files)} target files")
    
    # First pass: collect all data organized by scenario
    scenario_data = {}  # scenario_name -> {target_name -> [values]}
    feature_cols = None
    index_cols = ['scenario_name', 'base_scenario']
    
    for file in files:
        file_path = os.path.join(training_dir, file)
        df = pd.read_csv(file_path)
        target_name = file.replace("training_dataset_", "").split(".")[0]
        
        # Get feature columns from first file
        if feature_cols is None:
            feature_cols = [c for c in df.columns 
                          if c not in index_cols and not c.startswith('training_dataset_')]
            feature_cols = [c for c in feature_cols if c != target_name]
            # Keep only numeric features
            feature_cols = df[feature_cols].select_dtypes(include=['number']).columns.tolist()
            print(f"Feature columns: {len(feature_cols)}")
        
        # Group by scenario
        for scenario_name, group in df.groupby('scenario_name'):
            if scenario_name not in scenario_data:
                # Determine split
                if scenario_name in train_scenarios:
                    split = 'train'
                elif scenario_name in test_scenarios:
                    split = 'test'
                else:
                    continue  # Skip if not in train or test
                
                # Store features (same for all rows in scenario)
                scenario_data[scenario_name] = {
                    'features': group[feature_cols].iloc[0],  # Take first row
                    'targets': {},
                    'split': split
                }
            
            # Store all target values for this scenario
            scenario_data[scenario_name]['targets'][target_name] = group[target_name].values.tolist()
    
    # Organize by split
    train_scenarios_data = {k: v for k, v in scenario_data.items() if v['split'] == 'train'}
    test_scenarios_data = {k: v for k, v in scenario_data.items() if v['split'] == 'test'}
    
    print(f"\nTotal scenarios: {len(scenario_data)}")
    print(f"  Train: {len(train_scenarios_data)}")
    print(f"  Test: {len(test_scenarios_data)}")
    
    # Detect target types
    target_types = {}
    for target_name in files:
        target_name = target_name.replace("training_dataset_", "").split(".")[0]
        # Get all values for this target to determine type
        all_values = []
        for sc in scenario_data.values():
            if target_name in sc['targets']:
                all_values.extend(sc['targets'][target_name])
        target_types[target_name] = detect_problem_type(pd.Series(all_values))
    
    print(f"\nTarget types:")
    for t, typ in target_types.items():
        print(f"  {t}: {typ}")
    
    return train_scenarios_data, test_scenarios_data, target_types, feature_cols


def detect_problem_type(series, threshold=15):
    n_unique = series.nunique()
    n_total = len(series)
    unique_ratio = n_unique / n_total
    is_numeric = pd.api.types.is_numeric_dtype(series)

    if not is_numeric:
        return 'classification'
    if n_unique <= 2:
        return 'classification'
    elif unique_ratio < (threshold / 100) and series.dtype in ['int64', 'int32']:
        return 'classification'
    else:
        return 'regression'


# =============================================================================
# Model Loading (same as before)
# =============================================================================

def find_latest_classification_models(output_dir="models/multilabel_cls"):
    if not os.path.exists(output_dir):
        return {}
    meta_files = glob.glob(os.path.join(output_dir, "multilabel_metadata_*.json"))
    if not meta_files:
        return {}
    
    meta_files.sort()
    latest_meta_file = meta_files[-1]
    
    with open(latest_meta_file, 'r') as f:
        meta = json.load(f)
    
    predictors = {}
    for target, info in meta.get('per_target', {}).items():
        model_path = info.get('model_path', '')
        if os.path.exists(model_path):
            try:
                predictor = TabularPredictor.load(model_path)
                predictors[target] = predictor
                print(f"  Loaded classification model for '{target}'")
            except Exception as e:
                print(f"  Error loading '{target}': {e}")
    return predictors


def find_latest_regression_models(output_dir="models/multitarget_reg"):
    """
    Load the latest trained regression models from separate .pkl files.
    Returns a dictionary compatible with the old bundle format for easy integration.
    """
    if not os.path.exists(output_dir):
        return None
    
    # Find latest metadata file
    meta_files = glob.glob(os.path.join(output_dir, "ngboost_metadata_*.json"))
    if not meta_files:
        return None
    
    meta_files.sort()
    latest_meta_file = meta_files[-1]
    
    with open(latest_meta_file, 'r') as f:
        meta = json.load(f)
    
    timestamp = meta.get('timestamp', '')
    targets = meta.get('targets', [])
    
    if not targets:
        print(f"  No targets found in metadata: {latest_meta_file}")
        return None
    
    # Load individual model files
    models = {}
    all_metadata = {}
    scaler = None
    feature_names = None
    
    print(f"  Loading regression models from timestamp: {timestamp}")
    
    for target in targets:
        # Construct expected model filename
        model_filename = f"ngboost_{target}_{timestamp}.pkl"
        model_path = os.path.join(output_dir, model_filename)
        
        # Fallback: search for any matching file if exact name not found
        if not os.path.exists(model_path):
            pattern = os.path.join(output_dir, f"ngboost_{target}_*.pkl")
            matches = glob.glob(pattern)
            if matches:
                matches.sort()
                model_path = matches[-1]  # Take latest if multiple exist
            else:
                print(f"  Warning: Model file not found for target '{target}'")
                continue
        
        try:
            with open(model_path, 'rb') as f:
                model_bundle = pickle.load(f)
            
            models[target] = model_bundle['model']
            
            # Extract metadata from individual model bundle
            target_meta = {
                'model_path': model_path,
                'best_params': model_bundle.get('best_params', {}),
                'timestamp': model_bundle.get('timestamp', timestamp),
            }
            
            # Add CV metrics from main metadata if available
            if target in meta.get('per_target', {}):
                target_meta.update(meta['per_target'][target])
            
            all_metadata[target] = target_meta
            
            # Get scaler and feature names from first successful load
            if scaler is None and 'scaler' in model_bundle:
                scaler = model_bundle['scaler']
            if feature_names is None and 'feature_names' in model_bundle:
                feature_names = model_bundle['feature_names']
                
            print(f"    ✓ Loaded {target}: {os.path.basename(model_path)}")
            
        except Exception as e:
            print(f"  Error loading model for {target}: {e}")
            continue
    
    if not models:
        print(f"  No models could be loaded from {latest_meta_file}")
        return None
    
    # Load scaler from dedicated file if not found in models
    if scaler is None:
        scaler_path = meta.get('scaler_path', '')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            # Fallback: search for any scaler file with same timestamp
            scaler_pattern = os.path.join(output_dir, f"scaler_{timestamp}.pkl")
            scaler_matches = glob.glob(scaler_pattern)
            if scaler_matches:
                with open(scaler_matches[0], 'rb') as f:
                    scaler = pickle.load(f)
    
    # Reconstruct bundle format compatible with old code
    bundle = {
        'models': models,
        'scaler': scaler,
        'feature_names': feature_names or meta.get('features', []),
        'targets': list(models.keys()),
        'metadata': all_metadata,
        'grouping': meta.get('grouping', 'scenario_name'),
        'timestamp': timestamp,
        'optimization_method': meta.get('optimization_method', 'TPE'),
        '_meta_file': latest_meta_file,  # Track source for debugging
    }
    
    print(f"  Loaded {len(models)} regression models (TPE optimized)")
    return bundle


# =============================================================================
# Evaluation with Set-Based Metrics
# =============================================================================

def evaluate_classification_set(predictor, scenario_data, target_name):
    """
    For classification: prediction is correct if it matches ANY available target 
    for that scenario.
    """
    correct = 0
    total = 0
    predictions = []
    
    for scenario_name, data in scenario_data.items():
        if target_name not in data['targets']:
            continue
        
        available_targets = set(data['targets'][target_name])
        
        # Prepare features as DataFrame
        features_df = pd.DataFrame([data['features']])
        
        # Predict
        pred = predictor.predict(features_df)
        if hasattr(pred, 'values'):
            pred = pred.values[0]
        else:
            pred = pred[0]
        
        predictions.append({
            'scenario': scenario_name,
            'prediction': pred,
            'available': list(available_targets),
            'correct': pred in available_targets
        })
        
        if pred in available_targets:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'target': target_name,
        'type': 'classification',
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'predictions': predictions
    }


def evaluate_regression_nearest(bundle, scenario_data, target_name):
    """
    For regression: compute error as distance to nearest available target.
    Also compute R² treating nearest target as "true" value.
    """
    scaler = bundle['scaler']
    feature_names = bundle['feature_names']
    model = bundle['models'].get(target_name)
    
    if model is None:
        return None
    
    errors = []
    nearest_targets = []
    predictions = []
    all_preds = []
    all_nearest = []
    
    for scenario_name, data in scenario_data.items():
        if target_name not in data['targets']:
            continue
        
        available_targets = np.array(data['targets'][target_name])
        
        # Prepare features
        features_df = pd.DataFrame([data['features']])[feature_names]
        X = features_df.values
        X_scaled = scaler.transform(X)
        
        # Predict
        pred = model.predict(X_scaled)[0]
        all_preds.append(pred)
        
        # Find nearest available target
        distances = np.abs(available_targets - pred)
        nearest_idx = np.argmin(distances)
        nearest_target = available_targets[nearest_idx]
        nearest_targets.append(nearest_target)
        all_nearest.append(nearest_target)
        
        error = distances[nearest_idx]
        errors.append(error)
        
        predictions.append({
            'scenario': scenario_name,
            'prediction': float(pred),
            'nearest_target': float(nearest_target),
            'available_targets': available_targets.tolist(),
            'distance_to_nearest': float(error)
        })
    
    if not errors:
        return None
    
    errors = np.array(errors)
    all_preds = np.array(all_preds)
    all_nearest = np.array(all_nearest)
    
    # Metrics
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors**2))
    r2 = r2_score(all_nearest, all_preds)  # R² using nearest target as ground truth
    
    return {
        'target': target_name,
        'type': 'regression',
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mean_distance_to_nearest': np.mean(errors),
        'max_distance_to_nearest': np.max(errors),
        'n_scenarios': len(errors),
        'predictions': predictions
    }


def evaluate_all(scenario_data, target_types, feature_cols, cls_predictors, reg_bundle):
    """Evaluate all models on scenario-level data."""
    results = []
    
    # Get unique targets from scenario data
    all_targets = set()
    for data in scenario_data.values():
        all_targets.update(data['targets'].keys())
    
    for target_name in all_targets:
        problem_type = target_types.get(target_name, 'regression')
        print(f"\nEvaluating {target_name} ({problem_type})...")
        
        if problem_type == 'classification':
            if target_name in cls_predictors:
                result = evaluate_classification_set(
                    cls_predictors[target_name], scenario_data, target_name
                )
                results.append(result)
                print(f"  Accuracy: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")
            else:
                print(f"  No model found")
        else:
            if reg_bundle and target_name in reg_bundle['models']:
                result = evaluate_regression_nearest(
                    reg_bundle, scenario_data, target_name
                )
                if result:
                    results.append(result)
                    print(f"  R²: {result['r2']:.4f}, MAE: {result['mae']:.4f}, RMSE: {result['rmse']:.4f}")
            else:
                print(f"  No model found")
    
    return results


# =============================================================================
# Reporting
# =============================================================================

def print_results(train_results, test_results):
    """Print comparison of train vs test results."""
    
    print("\n" + "="*90)
    print("EVALUATION RESULTS (Scenario-Level)")
    print("="*90)
    print("Classification: Accuracy = prediction matches ANY available target for scenario")
    print("Regression: R² computed against nearest available target for scenario")
    print("="*90)
    
    for split_name, results in [('TRAIN', train_results), ('TEST', test_results)]:
        print(f"\n{split_name} SET:")
        print("-"*90)
        
        if not results:
            print("  No results")
            continue
        
        cls_results = [r for r in results if r['type'] == 'classification']
        reg_results = [r for r in results if r['type'] == 'regression']
        
        if cls_results:
            print("\n  Classification:")
            print(f"  {'Target':<35} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
            print("  " + "-"*67)
            for r in cls_results:
                print(f"  {r['target']:<35} {r['accuracy']:>10.4f} {r['correct']:>10} {r['total']:>10}")
        
        if reg_results:
            print("\n  Regression:")
            print(f"  {'Target':<35} {'R²':>10} {'MAE':>12} {'RMSE':>12} {'N':>8}")
            print("  " + "-"*73)
            for r in reg_results:
                print(f"  {r['target']:<35} {r['r2']:>10.4f} {r['mae']:>12.4f} {r['rmse']:>12.4f} {r['n_scenarios']:>8}")
    
    # Summary
    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    
    for split_name, results in [('Train', train_results), ('Test', test_results)]:
        if not results:
            continue
            
        cls_results = [r for r in results if r['type'] == 'classification']
        reg_results = [r for r in results if r['type'] == 'regression']
        
        print(f"\n{split_name}:")
        if cls_results:
            accs = [r['accuracy'] for r in cls_results]
            print(f"  Classification - Mean Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        if reg_results:
            r2s = [r['r2'] for r in reg_results]
            maes = [r['mae'] for r in reg_results]
            print(f"  Regression     - Mean R²:       {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
            print(f"                   Mean MAE:       {np.mean(maes):.4f} ± {np.std(maes):.4f}")


def save_detailed_results(train_results, test_results, output_file="evaluation_results.json"):
    """Save detailed results including per-scenario predictions."""
    
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    output = {
        'train': convert(train_results),
        'test': convert(test_results),
        'evaluation_method': {
            'classification': 'Prediction matches any available target for scenario',
            'regression': 'Error computed as distance to nearest available target, R² against nearest'
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("="*90)
    print("SCENARIO-LEVEL MODEL EVALUATION")
    print("="*90)
    
    # 1. Load data grouped by scenario
    print("\n[1/5] Loading and grouping data by scenario...")
    train_data, test_data, target_types, feature_cols = load_scenario_data()
    
    # 2. Load models
    print("\n[2/5] Loading classification models...")
    cls_predictors = find_latest_classification_models()
    
    print("\n[3/5] Loading regression models...")
    reg_bundle = find_latest_regression_models()
    
    # 3. Evaluate
    print("\n[4/5] Evaluating on train set...")
    train_results = evaluate_all(train_data, target_types, feature_cols, cls_predictors, reg_bundle)
    
    print("\n[5/5] Evaluating on test set...")
    test_results = evaluate_all(test_data, target_types, feature_cols, cls_predictors, reg_bundle)
    
    # 4. Report
    print_results(train_results, test_results)
    save_detailed_results(train_results, test_results)
    
    print("\n" + "="*90)
    print("EVALUATION COMPLETE")
    print("="*90)