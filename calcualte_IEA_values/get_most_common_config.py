import pandas as pd
import numpy as np

def replace_columns_with_mean_or_mode(input_file, output_file):
    """
    Replace each column in a CSV with its mean (for numerical columns)
    or mode/most common value (for categorical/boolean columns).
    Keeps the first column (scenario_name) unchanged.
    """
    # Load the CSV file
    df = pd.read_csv(input_file)
    
    # Create a copy to avoid modifying the original
    df_replaced = df.copy()
    
    # Process each column
    for col in df.columns:
        if col == 'scenario_name':
            # Keep scenario_name as is
            continue
        elif df[col].dtype == 'object':
            # For categorical columns, use mode (most common value)
            mode_val = df[col].mode()[0]
            df_replaced[col] = mode_val
            print(f"{col}: replaced with mode = '{mode_val}'")
        elif df[col].dtype == 'bool':
            # For boolean columns, use mode (most common value)
            mode_val = df[col].mode()[0]
            df_replaced[col] = mode_val
            print(f"{col}: replaced with mode = {mode_val}")
        else:
            # For numerical columns, use mean
            mean_val = df[col].mean()
            df_replaced[col] = mean_val
            print(f"{col}: replaced with mean = {mean_val:.6f}")
    
    # Save the result
    df_replaced.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")
    
    return df_replaced


# Usage example
if __name__ == "__main__":
    input_path = 'best_configs.csv'
    output_path = 'most_common_configs.csv'
    
    df_result = replace_columns_with_mean_or_mode(input_path, output_path)
    
    # Display first few rows
    print("\nFirst 5 rows of the result:")
    print(df_result.head())