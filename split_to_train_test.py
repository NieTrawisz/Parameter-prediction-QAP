import pandas as pd
from sklearn.model_selection import train_test_split

all_data = pd.read_csv("best_configs.csv")["scenario_name"]

train, test = train_test_split(all_data, test_size=0.1, random_state=42)

# Save to CSV
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)
