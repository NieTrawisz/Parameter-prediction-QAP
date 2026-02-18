#!/usr/bin/env python3
"""
Script to merge features.csv and multi_results.csv into training datasets.
Features are used as input variables, and configuration parameters are used as targets.

Handles dual problems: dual scenarios (ending with _dual) use the same target
configurations as their base problems but have different features.

NOW SUPPORTS: Multiple configurations per scenario for larger datasets.
"""

import pandas as pd
import os
from pathlib import Path

def load_and_prepare_data(features_path, configs_path):
    """
    Load features and best configs, then merge them.
    Handles dual problems by mapping them to base problem configurations.
    SUPPORTS multiple configs per scenario (explodes the dataset).
    
    Args:
        features_path: Path to features.csv
        configs_path: Path to multi_results.csv
    
    Returns:
        Merged DataFrame with features and target columns (expanded for multiple configs)
    """
    # Load features CSV
    features_df = pd.read_csv(features_path, index_col=0)
    print(f"Loaded features: {features_df.shape[0]} instances, {features_df.shape[1]} features")
    
    # Load best configs CSV
    configs_df = pd.read_csv(configs_path)
    print(f"Loaded configs: {configs_df.shape[0]} total config rows, {configs_df.shape[1]} columns")
    
    # Reset index to make scenario_name a regular column
    features_df.index.name = 'scenario_name'
    features_df = features_df.reset_index()
    
    # Create a mapping for dual problems
    # Dual problems (ending with _dual) should use configs from base problems
    def get_base_scenario(scenario_name):
        """Get base scenario name (remove _dual suffix if present)"""
        if scenario_name.endswith('_dual'):
            return scenario_name[:-5]  # Remove '_dual'
        return scenario_name
    
    # Add a column for base scenario name
    features_df['base_scenario'] = features_df['scenario_name'].apply(get_base_scenario)
    configs_df['base_scenario'] = configs_df['scenario_name'].apply(get_base_scenario)
    
    # Count dual vs base scenarios
    dual_count = features_df['scenario_name'].str.endswith('_dual').sum()
    base_count = len(features_df) - dual_count
    print(f"\nScenario breakdown:")
    print(f"  Base scenarios: {base_count}")
    print(f"  Dual scenarios: {dual_count}")
    
    # Count configs per base scenario
    configs_per_scenario = configs_df.groupby('base_scenario').size()
    print(f"\nConfig distribution:")
    print(f"  Scenarios with 1 config: {(configs_per_scenario == 1).sum()}")
    print(f"  Scenarios with 2+ configs: {(configs_per_scenario > 1).sum()}")
    print(f"  Max configs per scenario: {configs_per_scenario.max()}")
    print(f"  Average configs per scenario: {configs_per_scenario.mean():.2f}")
    
    # IMPORTANT CHANGE: Keep ALL configs per base scenario (not just first)
    # This creates multiple rows per feature set
    configs_grouped = configs_df.drop('scenario_name', axis=1)  # Drop original scenario_name
    
    # Merge features with ALL configs using base_scenario
    # This will create a Cartesian product: each feature row × each config row
    merged_df = pd.merge(
        features_df, 
        configs_grouped,
        on='base_scenario', 
        how='inner'
    )
    
    print(f"\nMerged dataset: {merged_df.shape[0]} instances, {merged_df.shape[1]} total columns")
    print(f"  (Expanded from {features_df.shape[0]} features × multiple configs)")
    
    # Show which scenarios were matched
    matched_base = merged_df['base_scenario'].unique()
    print(f"Matched base scenarios: {len(matched_base)}")
    
    # Check for unmatched scenarios
    all_feature_bases = set(features_df['base_scenario'].unique())
    all_config_bases = set(configs_grouped['base_scenario'].unique())
    unmatched_in_features = all_feature_bases - all_config_bases
    unmatched_in_configs = all_config_bases - all_feature_bases
    
    if unmatched_in_features:
        print(f"\nWarning: {len(unmatched_in_features)} feature scenarios without configs:")
        for scenario in sorted(list(unmatched_in_features))[:5]:
            print(f"  - {scenario}")
        if len(unmatched_in_features) > 5:
            print(f"  ... and {len(unmatched_in_features) - 5} more")
    
    if unmatched_in_configs:
        print(f"\nWarning: {len(unmatched_in_configs)} config scenarios without features:")
        for scenario in sorted(list(unmatched_in_configs))[:5]:
            print(f"  - {scenario}")
        if len(unmatched_in_configs) > 5:
            print(f"  ... and {len(unmatched_in_configs) - 5} more")
    
    return merged_df, features_df, configs_df


def create_training_datasets(merged_df, feature_cols, output_dir='training_multi'):
    """
    Create training datasets with features and targets.
    
    Args:
        merged_df: Merged DataFrame with features and configs
        output_dir: Directory to save training datasets
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # # Define feature columns (from features.csv)
    # feature_cols = [
    #     'n', 'Frobenius inner product', 'Cosine similarity', 
    #     'Singular Value Alignment', 'Cross-Correlation',
    #     'Fsym', 'Favr', 'Fmax', 'Fmax_min', 'Fsdv', 'FzeroN', 
    #     'F_rank', 'F_frobenius_norm', 'F_eigen_real_min', 
    #     'F_eigen_real_max', 'F_eigen_real_var', 'F_singular_min', 
    #     'F_singular_max', 'F_singular_mean', 'F_singular_var',
    #     'Dsym', 'Davr', 'Dmax', 'Dmax_min', 'Dsdv', 'DzeroN', 
    #     'D_rank', 'D_frobenius_norm', 'D_eigen_real_min', 
    #     'D_eigen_real_max', 'D_eigen_real_var', 'D_singular_min', 
    #     'D_singular_max', 'D_singular_mean', 'D_singular_var'
    # ]
    
    # Define target columns (configuration parameters from multi_results.csv)
    # Exclude metadata columns
    metadata_cols = ['scenario_name', 'base_scenario', 'seed', 'config_id', 'avg_cost', 
                     'num_evaluations', 'total_trials', 'walltime', 'finished']
    
    target_cols = [col for col in merged_df.columns 
                   if col not in feature_cols and col not in metadata_cols]
    
    print(f"\nFeature columns: {len(feature_cols)}")
    print(f"Target columns: {len(target_cols)}")
    print(f"Target columns: {target_cols}")
    
    # Count base vs dual in final dataset
    dual_in_final = merged_df['scenario_name'].str.endswith('_dual').sum()
    base_in_final = len(merged_df) - dual_in_final
    
    # Count unique config combinations
    if 'config_id' in merged_df.columns:
        unique_configs = merged_df.groupby('base_scenario')['config_id'].nunique()
        print(f"\nFinal dataset composition:")
        print(f"  Base scenario instances: {base_in_final}")
        print(f"  Dual scenario instances: {dual_in_final}")
        print(f"  Total rows: {len(merged_df)}")
        print(f"  Average configs per scenario: {unique_configs.mean():.2f}")
    else:
        print(f"\nFinal dataset composition:")
        print(f"  Base scenario instances: {base_in_final}")
        print(f"  Dual scenario instances: {dual_in_final}")
        print(f"  Total: {len(merged_df)}")
    
    # Create main training dataset with all features and all targets
    id_cols = ['scenario_name', 'base_scenario']
    if 'config_id' in merged_df.columns:
        id_cols.append('config_id')
    if 'seed' in merged_df.columns:
        id_cols.append('seed')
        
    training_data = merged_df[id_cols + feature_cols + target_cols].copy()
    
    # Save main training dataset
    main_output_path = os.path.join(output_dir, 'training_dataset_full.csv')
    training_data.to_csv(main_output_path, index=False)
    print(f"\n✓ Saved: {main_output_path}")
    print(f"  Shape: {training_data.shape}")
    
    # Create separate datasets for each target (for single-target prediction)
    for target in target_cols:
        target_data = merged_df[id_cols + feature_cols + [target]].copy()
        output_path = os.path.join(output_dir, f'training_dataset_{target}.csv')
        target_data.to_csv(output_path, index=False)
    
    print(f"✓ Saved {len(target_cols)} individual target datasets")
    
    # Create dataset without scenario_name or base_scenario (for direct ML training)
    ml_cols = feature_cols + target_cols
    if 'config_id' in merged_df.columns:
        ml_cols.append('config_id')
    if 'seed' in merged_df.columns:
        ml_cols.append('seed')
        
    training_data_no_id = merged_df[ml_cols].copy()
    no_id_output_path = os.path.join(output_dir, 'training_dataset_no_id.csv')
    training_data_no_id.to_csv(no_id_output_path, index=False)
    print(f"\n✓ Saved: {no_id_output_path}")
    print(f"  Shape: {training_data_no_id.shape}")
    
    # Create summary statistics
    summary_path = os.path.join(output_dir, 'dataset_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Training Dataset Summary (Multi-Config Version)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total instances (rows): {training_data.shape[0]}\n")
        f.write(f"  Base scenario rows: {base_in_final}\n")
        f.write(f"  Dual scenario rows: {dual_in_final}\n")
        f.write(f"Unique scenarios: {training_data['base_scenario'].nunique()}\n")
        f.write(f"Total features: {len(feature_cols)}\n")
        f.write(f"Total targets: {len(target_cols)}\n\n")
        
        f.write("Note: Dual scenarios (ending with _dual) have different features\n")
        f.write("but share the same target configurations as their base scenarios.\n")
        f.write("Multiple configs per scenario are now included (dataset is expanded).\n\n")
        
        # Config distribution
        if 'config_id' in merged_df.columns:
            f.write("Config distribution per scenario:\n")
            config_counts = merged_df.groupby('base_scenario').size()
            f.write(f"  Min configs: {config_counts.min()}\n")
            f.write(f"  Max configs: {config_counts.max()}\n")
            f.write(f"  Mean configs: {config_counts.mean():.2f}\n\n")
        
        f.write("Feature columns:\n")
        for i, col in enumerate(feature_cols, 1):
            f.write(f"  {i}. {col}\n")
        
        f.write("\nTarget columns:\n")
        for i, col in enumerate(target_cols, 1):
            f.write(f"  {i}. {col}\n")
        
        f.write("\nDataset statistics (features):\n")
        f.write(str(training_data[feature_cols].describe()))
        
        # Add examples of dual pairs
        f.write("\n\nExample dual pairs (first 5):\n")
        base_scenarios = merged_df[~merged_df['scenario_name'].str.endswith('_dual')]['scenario_name'].unique()
        for base in list(base_scenarios)[:5]:
            dual = base + '_dual'
            if dual in merged_df['scenario_name'].values:
                base_count = len(merged_df[merged_df['scenario_name'] == base])
                dual_count = len(merged_df[merged_df['scenario_name'] == dual])
                f.write(f"  {base} ({base_count} configs) <-> {dual} ({dual_count} configs)\n")
    
    print(f"✓ Saved: {summary_path}")
    
    return training_data, target_cols


def main():
    """Main execution function."""
    # Define input file paths
    features_path = 'features.csv'
    configs_path = 'multi_configs.csv'
    output_dir = 'training_multi'
    
    print("=" * 80)
    print("Merging Features and Best Configs into Training Datasets")
    print("(Handling dual problems + MULTIPLE CONFIGS PER SCENARIO)")
    print("=" * 80)
    
    # Load and merge data
    merged_df, features_df, configs_df = load_and_prepare_data(features_path, configs_path)
    
    # Create training datasets
    feature_cols = features_df.columns.to_list()
    training_data, target_cols = create_training_datasets(merged_df, feature_cols, output_dir)
    
    print("\n" + "=" * 80)
    print("Training datasets created successfully!")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}/")
    print(f"Main dataset: training_dataset_full.csv")
    print(f"Individual target datasets: training_dataset_<target_name>.csv")
    print(f"Dataset without IDs: training_dataset_no_id.csv")
    print(f"Summary: dataset_summary.txt")
    print(f"\nNOTE: Dataset now includes ALL configurations per scenario!")
    print(f"      Previous size: ~{features_df.shape[0]} rows")
    print(f"      New size: {training_data.shape[0]} rows")
    print(f"      Expansion ratio: {training_data.shape[0] / max(features_df.shape[0], 1):.2f}x")


if __name__ == "__main__":
    main()