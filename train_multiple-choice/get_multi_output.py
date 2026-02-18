import os
import pandas as pd
import numpy as np
import pickle
import json
import glob
from datetime import datetime

from autogluon.tabular import TabularPredictor
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
        'base_scenario': base_scenario value,
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
                    'base_scenario': group['base_scenario'].iloc[0],  # Store base_scenario
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
    else:
        return 'regression'


# =============================================================================
# Model Loading
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
# Prediction Functions
# =============================================================================

def predict_classification(predictor, scenario_data, target_name):
    """
    For classification: predict the class for each scenario.
    Returns DataFrame with base_scenario and prediction.
    """
    predictions = []
    
    for scenario_name, data in scenario_data.items():
        if target_name not in data['targets']:
            continue
        
        # Prepare features as DataFrame
        features_df = pd.DataFrame([data['features']])
        
        # Predict
        pred = predictor.predict(features_df)
        if hasattr(pred, 'values'):
            pred = pred.values[0]
        else:
            pred = pred[0]
        
        predictions.append({
            'base_scenario': data['base_scenario'],
            target_name: pred
        })
    
    if not predictions:
        return None
    
    return pd.DataFrame(predictions)


def predict_regression(bundle, scenario_data, target_name):
    """
    For regression: predict the value for each scenario.
    Returns DataFrame with base_scenario and prediction.
    """
    scaler = bundle['scaler']
    feature_names = bundle['feature_names']
    model = bundle['models'].get(target_name)
    
    if model is None:
        return None
    
    predictions = []
    
    for scenario_name, data in scenario_data.items():
        if target_name not in data['targets']:
            continue
        
        # Prepare features
        features_df = pd.DataFrame([data['features']])[feature_names]
        X = features_df.values
        X_scaled = scaler.transform(X)
        
        # Predict
        pred = model.predict(X_scaled)[0]
        
        predictions.append({
            'base_scenario': data['base_scenario'],
            target_name: pred
        })
    
    if not predictions:
        return None
    
    return pd.DataFrame(predictions)


def predict_all(scenario_data, target_types, feature_cols, cls_predictors, reg_bundle):
    """Generate predictions for all models on scenario-level data."""
    all_predictions = []
    
    # Get unique targets from scenario data
    all_targets = set()
    for data in scenario_data.values():
        all_targets.update(data['targets'].keys())
    
    for target_name in all_targets:
        # Skip targets ending with "_dual"
        if target_name.endswith("_dual"):
            print(f"Skipping {target_name} (ends with _dual)")
            continue
            
        problem_type = target_types.get(target_name, 'regression')
        print(f"Predicting {target_name} ({problem_type})...")
        
        pred_df = None
        if problem_type == 'classification':
            if target_name in cls_predictors:
                pred_df = predict_classification(
                    cls_predictors[target_name], scenario_data, target_name
                )
            else:
                print(f"  No classification model found for {target_name}")
        else:
            if reg_bundle and target_name in reg_bundle['models']:
                pred_df = predict_regression(
                    reg_bundle, scenario_data, target_name
                )
            else:
                print(f"  No regression model found for {target_name}")
        
        if pred_df is not None and not pred_df.empty:
            all_predictions.append(pred_df)
            print(f"  Generated {len(pred_df)} predictions")
    
    return all_predictions


# =============================================================================
# Combine and Save Predictions
# =============================================================================

def combine_predictions(list_of_predictions, output_file):
    """
    Combine all prediction DataFrames by joining on 'base_scenario'.
    Filter out columns ending with '_dual'.
    """
    if not list_of_predictions:
        print(f"No predictions to save for {output_file}")
        return None
    
    # Start with first prediction dataframe
    combined = list_of_predictions[0].copy()
    print(f"\nStarting with {len(combined)} rows and columns: {list(combined.columns)}")
    
    # Merge remaining predictions
    for i, pred_df in enumerate(list_of_predictions[1:], 1):
        # Check for duplicate columns (excluding base_scenario)
        existing_cols = set(combined.columns) - {'base_scenario'}
        new_cols = set(pred_df.columns) - {'base_scenario'}
        
        duplicates = existing_cols & new_cols
        if duplicates:
            print(f"  Warning: Duplicate columns found in prediction {i}: {duplicates}")
            # Drop duplicate columns from new dataframe (keep first occurrence)
            pred_df = pred_df.drop(columns=list(duplicates))
        
        combined = combined.merge(pred_df, on='base_scenario', how='outer')
        print(f"  Merged prediction {i}: now {len(combined)} rows, {len(combined.columns)} columns")
    
    # Filter out columns ending with "_dual"
    dual_cols = [c for c in combined.columns if c.endswith('_dual')]
    if dual_cols:
        print(f"\nRemoving {len(dual_cols)} columns ending with '_dual': {dual_cols}")
        combined = combined.drop(columns=dual_cols)
    
    # Sort by base_scenario for consistency
    combined = combined.sort_values('base_scenario').reset_index(drop=True)
    
    print(f"\nFinal predictions: {len(combined)} rows, {len(combined.columns)} columns")
    print(f"Columns: {list(combined.columns)}")
    
    # Save to CSV
    combined.to_csv(output_file, index=False)
    print(f"\nPredictions saved to: {output_file}")
    
    return combined


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("="*90)
    print("SCENARIO-LEVEL PREDICTION GENERATION")
    print("="*90)
    
    # 1. Load data grouped by scenario
    print("\n[1/4] Loading and grouping data by scenario...")
    train_data, test_data, target_types, feature_cols = load_scenario_data()
    
    # 2. Load models
    print("\n[2/4] Loading classification models...")
    cls_predictors = find_latest_classification_models()
    
    print("\n[3/4] Loading regression models...")
    reg_bundle = find_latest_regression_models()
    
    # 4. Generate predictions for both train and test
    print("\n[4/4] Generating predictions...")
    
    # print("\n--- Train Set Predictions ---")
    # train_predictions = predict_all(train_data, target_types, feature_cols, cls_predictors, reg_bundle)
    
    print("\n--- Test Set Predictions ---")
    test_predictions = predict_all(test_data, target_types, feature_cols, cls_predictors, reg_bundle)
    
    # 5. Combine and save separately
    print("\n" + "="*90)
    print("SAVING PREDICTIONS")
    print("="*90)
    
    # # Save train predictions
    # if train_predictions:
    #     print("\nProcessing train predictions...")
    #     combine_predictions(train_predictions, output_file="predictions_train.csv")
    
    # Save test predictions
    if test_predictions:
        print("\nProcessing test predictions...")
        combine_predictions(test_predictions, output_file="predictions_test.csv")
    
    print("\n" + "="*90)
    print("PREDICTION GENERATION COMPLETE")
    print("="*90)