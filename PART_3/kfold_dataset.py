import pandas as pd
import os

# Define base directory
base_dir = "..\\PART_1\\data"

# Define paths to the individual CSV files
train_csv = os.path.join(base_dir, "train_dataset.csv")
validation_csv = os.path.join(base_dir, "validation_dataset.csv")
test_csv = os.path.join(base_dir, "test_dataset.csv")

# Read the CSV files
train_data = pd.read_csv(train_csv)
validation_data = pd.read_csv(validation_csv)
test_data = pd.read_csv(test_csv)

# Concatenate the datasets
kfold_data = pd.concat([train_data, validation_data, test_data], ignore_index=True)

# Shuffle the concatenated dataset to mitigate any problems with the KFOLD segmentations
kfold_data_shuffled = kfold_data.sample(frac=1, random_state=2024).reset_index(drop=True)

# Save the shuffled dataset to a new CSV file
kfold_csv = os.path.join(base_dir, "kfold_dataset.csv")
kfold_data_shuffled.to_csv(kfold_csv, index=False)

print(f"kfold_dataset.csv has been created and shuffled at {kfold_csv}")
