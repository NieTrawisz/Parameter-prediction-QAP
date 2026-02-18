import os
import pandas as pd
import numpy as np
import json
import glob
from datetime import datetime

from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Data Loading
# =============================================================================

def load_predictions_csv(csv_path="single_target_predictions.csv"):
    """
    Load predictions directly from CSV file.
    Returns: dict[scenario_name] = {
        'predictions': {target_name: predicted_value},
        'features': pd.Series (original features if needed)
    }
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded predictions CSV: {len(df)} scenarios")
    
    # Define target columns by type
    classification_targets = ['migration', 'release', 'reproduction', 'succession']
    regression_targets = [
        'islands_num', 'migration_batch', 'migration_freq',
        'probCXcrossover', 'probMutateInverse', 'probMutateShift', 
        'probMutateSwap', 'probNothing', 'probOXcrossover', 'probPMXcrossover'
    ]
    
    # Group by scenario
    predictions = {}
    for _, row in df.iterrows():
        scenario_name = row['base_scenario']
        
        pred_dict = {}
        for target in classification_targets + regression_targets:
            pred_dict[target] = row[target]
        
        predictions[scenario_name] = {
            'predictions': pred_dict,
            'base_scenario': row['base_scenario']
        }
    
    print(f"  Classification targets: {classification_targets}")
    print(f"  Regression targets: {len(regression_targets)}")
    
    return predictions, classification_targets, regression_targets


def load_scenario_ground_truth(train_csv="train.csv", test_csv="test.csv", 
                                training_dir="training_multi"):
    """
    Load ground truth data grouped by scenario_name.
    Returns: dict[scenario_name] = {
        'targets': {target_name: [list of available values]},
        'split': 'train' or 'test'
    }
    """
    train_scenarios = set(pd.read_csv(train_csv)["scenario_name"].tolist())
    test_scenarios = set(pd.read_csv(test_csv)["scenario_name"].tolist())
    
    print(f"\nTrain scenarios: {len(train_scenarios)}")
    print(f"Test scenarios: {len(test_scenarios)}")
    
    # Load all target files
    files = sorted([f for f in os.listdir(training_dir) if f.endswith('.csv')])
    print(f"Found {len(files)} target files")
    
    scenario_data = {}
    
    for file in files:
        file_path = os.path.join(training_dir, file)
        df = pd.read_csv(file_path)
        target_name = file.replace("training_dataset_", "").split(".")[0]
        
        # Group by scenario
        for scenario_name, group in df.groupby('scenario_name'):
            if scenario_name not in scenario_data:
                # Determine split
                if scenario_name in train_scenarios:
                    split = 'train'
                elif scenario_name in test_scenarios:
                    split = 'test'
                else:
                    continue
                
                scenario_data[scenario_name] = {
                    'targets': {},
                    'split': split
                }
            
            # Store all target values for this scenario
            scenario_data[scenario_name]['targets'][target_name] = group[target_name].values.tolist()
    
    # Organize by split
    train_scenarios_data = {k: v for k, v in scenario_data.items() if v['split'] == 'train'}
    test_scenarios_data = {k: v for k, v in scenario_data.items() if v['split'] == 'test'}
    
    print(f"\nTotal scenarios with ground truth: {len(scenario_data)}")
    print(f"  Train: {len(train_scenarios_data)}")
    print(f"  Test: {len(test_scenarios_data)}")
    
    return train_scenarios_data, test_scenarios_data


# =============================================================================
# Evaluation with Set-Based Metrics (using CSV predictions)
# =============================================================================

def evaluate_classification_set(predictions, ground_truth, target_name):
    """
    For classification: prediction is correct if it matches ANY available target 
    for that scenario.
    """
    correct = 0
    total = 0
    prediction_details = []
    
    for scenario_name, pred_data in predictions.items():
        if scenario_name not in ground_truth:
            continue
        if target_name not in ground_truth[scenario_name]['targets']:
            continue
        
        available_targets = set(ground_truth[scenario_name]['targets'][target_name])
        pred_value = pred_data['predictions'][target_name]
        
        is_correct = pred_value in available_targets
        
        prediction_details.append({
            'scenario': scenario_name,
            'prediction': pred_value,
            'available': list(available_targets),
            'correct': is_correct
        })
        
        if is_correct:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'target': target_name,
        'type': 'classification',
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'predictions': prediction_details
    }


def evaluate_regression_nearest(predictions, ground_truth, target_name):
    """
    For regression: compute error as distance to nearest available target.
    Also compute R² treating nearest target as "true" value.
    """
    errors = []
    nearest_targets = []
    all_preds = []
    all_nearest = []
    prediction_details = []
    
    for scenario_name, pred_data in predictions.items():
        if scenario_name not in ground_truth:
            continue
        if target_name not in ground_truth[scenario_name]['targets']:
            continue
        
        available_targets = np.array(ground_truth[scenario_name]['targets'][target_name]).astype(np.float16)
        pred_value = pred_data['predictions'][target_name]
        
        all_preds.append(pred_value)
        
        # Find nearest available target
        distances = np.abs(available_targets - pred_value)
        nearest_idx = np.argmin(distances)
        nearest_target = available_targets[nearest_idx]
        
        nearest_targets.append(nearest_target)
        all_nearest.append(nearest_target)
        
        error = distances[nearest_idx]
        errors.append(error)
        
        prediction_details.append({
            'scenario': scenario_name,
            'prediction': float(pred_value),
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
    
    # R² using nearest target as ground truth
    try:
        r2 = r2_score(all_nearest, all_preds)
    except:
        r2 = float('nan')
    
    return {
        'target': target_name,
        'type': 'regression',
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mean_distance_to_nearest': np.mean(errors),
        'max_distance_to_nearest': np.max(errors),
        'n_scenarios': len(errors),
        'predictions': prediction_details
    }


def evaluate_all(predictions, train_ground_truth, test_ground_truth, 
                 classification_targets, regression_targets):
    """Evaluate all targets using CSV predictions."""
    
    train_results = []
    test_results = []
    
    # Evaluate classification targets
    for target_name in classification_targets:
        print(f"\nEvaluating {target_name} (classification)...")
        
        # Train set
        result_train = evaluate_classification_set(predictions, train_ground_truth, target_name)
        if result_train['total'] > 0:
            train_results.append(result_train)
            print(f"  Train - Accuracy: {result_train['accuracy']:.4f} ({result_train['correct']}/{result_train['total']})")
        
        # Test set
        result_test = evaluate_classification_set(predictions, test_ground_truth, target_name)
        if result_test['total'] > 0:
            test_results.append(result_test)
            print(f"  Test  - Accuracy: {result_test['accuracy']:.4f} ({result_test['correct']}/{result_test['total']})")
    
    # Evaluate regression targets
    for target_name in regression_targets:
        print(f"\nEvaluating {target_name} (regression)...")
        
        # Train set
        result_train = evaluate_regression_nearest(predictions, train_ground_truth, target_name)
        if result_train and result_train['n_scenarios'] > 0:
            train_results.append(result_train)
            print(f"  Train - R²: {result_train['r2']:.4f}, MAE: {result_train['mae']:.4f}, RMSE: {result_train['rmse']:.4f}")
        
        # Test set
        result_test = evaluate_regression_nearest(predictions, test_ground_truth, target_name)
        if result_test and result_test['n_scenarios'] > 0:
            test_results.append(result_test)
            print(f"  Test  - R²: {result_test['r2']:.4f}, MAE: {result_test['mae']:.4f}, RMSE: {result_test['rmse']:.4f}")
    
    return train_results, test_results


# =============================================================================
# Reporting (same as before)
# =============================================================================

def print_results(train_results, test_results):
    """Print comparison of train vs test results."""
    
    print("\n" + "="*90)
    print("EVALUATION RESULTS (Scenario-Level) - Using CSV Predictions")
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
            for r in cls_results:
                print(f"    - {r['target']}: {r['accuracy']:.4f}")
        if reg_results:
            r2s = [r['r2'] for r in reg_results if not np.isnan(r['r2'])]
            maes = [r['mae'] for r in reg_results]
            print(f"  Regression     - Mean R²:       {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
            print(f"                   Mean MAE:       {np.mean(maes):.4f} ± {np.std(maes):.4f}")


def save_detailed_results(train_results, test_results, output_file="evaluation_results_csv.json"):
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
            'regression': 'Error computed as distance to nearest available target, R² against nearest',
            'source': 'CSV predictions file'
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
    print("SCENARIO-LEVEL EVALUATION (CSV-Based Predictions)")
    print("="*90)
    
    # 1. Load predictions from CSV
    print("\n[1/3] Loading predictions from CSV...")
    predictions, classification_targets, regression_targets = load_predictions_csv(
        "predictions.csv"
    )
    
    # 2. Load ground truth data
    print("\n[2/3] Loading ground truth data...")
    train_ground_truth, test_ground_truth = load_scenario_ground_truth()
    
    # 3. Evaluate predictions against ground truth
    print("\n[3/3] Evaluating predictions...")
    train_results, test_results = evaluate_all(
        predictions, 
        train_ground_truth, 
        test_ground_truth,
        classification_targets,
        regression_targets
    )
    
    # 4. Report results
    print_results(train_results, test_results)
    save_detailed_results(train_results, test_results)
    
    print("\n" + "="*90)
    print("EVALUATION COMPLETE")
    print("="*90)