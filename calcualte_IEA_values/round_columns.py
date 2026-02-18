import pandas as pd

df = pd.read_csv("single_target_predictions.csv",index_col=0)

df['islands_num'] = df['islands_num'].astype(float).round().astype(int)
df['migration_freq'] = df['migration_freq'].astype(float).round().astype(int)

df.to_csv("single_target_predictions.csv")