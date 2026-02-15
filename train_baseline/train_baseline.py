import os
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from autogluon.tabular import TabularPredictor
import warnings
warnings.filterwarnings('ignore')

def detect_problem_type(df, target, threshold=15):
    """
    Auto-detect if problem is classification or regression.
    Uses unique value ratio heuristic: if unique/total < threshold% and dtype is int-like, treat as classification.
    """
    target_series = df[target]
    n_unique = target_series.nunique()
    n_total = len(target_series)
    unique_ratio = n_unique / n_total
    
    # Check if target is numeric
    is_numeric = pd.api.types.is_numeric_dtype(target_series)
    
    # If not numeric, definitely classification
    if not is_numeric:
        return 'multiclass'
    
    # If unique values are few relative to dataset size, likely classification
    if n_unique <= 2:
        return 'binary'
    # elif unique_ratio < (threshold / 100) and target_series.dtype in ['int64', 'int32']:
    #     return 'multiclass'
    else:
        return 'regression'

def get_eval_metric(problem_type):
    """Get appropriate evaluation metric based on problem type."""
    metrics = {
        'binary': 'roc_auc',
        'multiclass': 'balanced_accuracy',
        'regression': 'r2'
    }
    return metrics.get(problem_type, 'auto')

def save_model_bundle(predictor, target, problem_type, output_dir="models"):
    """
    Save model with metadata for reproducibility.
    Creates: predictor folder, metadata JSON, and quick-load pickle.
    """
    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{target}_{timestamp}"
    save_path = os.path.join(output_dir, model_name)
    os.makedirs(save_path, exist_ok=True)
    
    # Save AutoGluon predictor (native format - best for reloading)
    predictor_path = os.path.join(save_path, "autogluon_model")
    predictor.save(predictor_path)
    
    # Save metadata
    leaderboard = predictor.leaderboard(silent=True)
    metadata = {
        'target': target,
        'problem_type': problem_type,
        'eval_metric': predictor.eval_metric,
        'best_model': leaderboard.iloc[0]['model'] if not leaderboard.empty else None,
        'best_score': float(leaderboard.iloc[0]['score_val']) if not leaderboard.empty else None,
        'training_timestamp': timestamp,
        'features': predictor.features(),
        'model_count': len(leaderboard),
        'leaderboard_summary': leaderboard[['model', 'score_val', 'fit_time']].head(10).to_dict('records')
    }
    
    metadata_path = os.path.join(save_path, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Save quick-reference pickle for fast inference without AutoGluon reload
    # Note: This is a lightweight reference, actual model stays in autogluon_model folder
    quick_ref = {
        'predictor_path': predictor_path,
        'target': target,
        'problem_type': problem_type,
        'features': predictor.features()
    }
    quick_ref_path = os.path.join(save_path, "model_reference.pkl")
    with open(quick_ref_path, 'wb') as f:
        pickle.dump(quick_ref, f)
    
    return save_path, metadata

def load_model_bundle(model_path):
    """Helper to reload a saved model bundle."""
    metadata_path = os.path.join(model_path, "metadata.json")
    predictor_path = os.path.join(model_path, "autogluon_model")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    predictor = TabularPredictor.load(predictor_path)
    return predictor, metadata

# Main training loop
for file in os.listdir("training"):
    if not file.endswith('.csv'):
        continue
        
    # Load the dataset
    file_path = os.path.join("training", file)
    df = pd.read_csv(file_path)
    train_list = pd.read_csv("train.csv")["scenario_name"].tolist()
        
    # Filtering df to only contain scenario from train dataset
    df = df[df["scenario_name"].str.startswith(tuple(train_list))]
    
    target = file.replace("training_dataset_","").split(".")[0]

    # Skip already trained targets
    if os.path.exists(f"single_target/{target}"):
        print(f"Skipping {target} - already trained")
        continue

    # Prepare the data - drop string columns and target
    cols_to_drop = ['scenario_name', 'base_scenario', target]
    train_df = df.drop(columns=cols_to_drop, errors='ignore')

    # Keep only numeric columns
    numeric_cols = train_df.select_dtypes(include=['number']).columns.tolist()
    train_df = train_df[numeric_cols].copy()

    # Add target column back
    train_df[target] = df[target]

    # Auto-detect problem type
    problem_type = detect_problem_type(train_df, target)
    eval_metric = get_eval_metric(problem_type)
    
    print(f"\n{'='*70}")
    print(f"Processing: {file}")
    print(f"{'='*70}")
    print(f"Training data shape: {train_df.shape}")
    print(f"Features used: {len(numeric_cols)}")
    print(f"Target: {target}")
    print(f"Auto-detected problem type: {problem_type.upper()}")
    print(f"Evaluation metric: {eval_metric}")
    print(f"{'='*70}")
    print("Training AutoGluon model...")
    print(f"{'='*70}")

    try:
        # Train AutoGluon model with auto-detected settings
        predictor = TabularPredictor(
            label=target,
            path = f"single_target/{target}",
            eval_metric=eval_metric,
            problem_type=problem_type
        ).fit(
            train_data=train_df,
            # time_limit=3600,
            presets='best_quality'
        )

        print("\nTraining complete!")

        # Display leaderboard
        print(f"\n{'='*70}")
        print("MODEL LEADERBOARD")
        print(f"{'='*70}")
        leaderboard = predictor.leaderboard(silent=True)
        display_cols = ['model', 'score_val', 'fit_time']
        if problem_type == 'regression':
            # For regression, lower RMSE is better but R2 is displayed
            if 'rmse' in leaderboard.columns:
                display_cols = ['model', 'score_val', 'rmse', 'fit_time']
        print(leaderboard[display_cols].head(10).to_string())

        # Get best model
        best_model = leaderboard.iloc[0]['model']
        best_score = leaderboard.iloc[0]['score_val']

        print(f"\nBest Model: {best_model}")
        score_label = "Validation R²" if problem_type == 'regression' else "Validation Score"
        print(f"{score_label}: {best_score:.6f}")

    except Exception as e:
        print(f"\nERROR processing {file}: {str(e)}")
        print("Skipping to next file...")
        continue

print(f"\n{'='*70}")
print("ALL TRAINING COMPLETE")
print(f"{'='*70}")