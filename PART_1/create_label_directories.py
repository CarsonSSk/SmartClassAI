import os
import shutil
import pandas as pd

# Paths to the CSV files
train_csv_path = "data/train_dataset.csv"
validation_csv_path = "data/validation_dataset.csv"
test_csv_path = "data/test_dataset.csv"

# Load the CSV files into DataFrames
train_df = pd.read_csv(train_csv_path)
validation_df = pd.read_csv(validation_csv_path)
test_df = pd.read_csv(test_csv_path)

# Add a column to each DataFrame to indicate the provenance set
train_df['Provenance'] = 'Train'
validation_df['Provenance'] = 'Validation'
test_df['Provenance'] = 'Test'

# Concatenate the DataFrames
combined_df = pd.concat([train_df, validation_df, test_df], ignore_index=True)

# Check for duplicates in the 'Image Name' column
duplicates = combined_df[combined_df.duplicated(subset='Image Name', keep=False)]

if not duplicates.empty:
    print("Duplicate images found across provenance sets:")
    print(duplicates)
else:
    print("No duplicate images found across provenance sets.")

# Directory to copy the images to
destination_dir = "data/additionalLabels/unlabelled"

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Copy images to the destination directory
for _, row in combined_df.iterrows():
    source_path = row['Path']
    destination_path = os.path.join(destination_dir, row['Image Name'])
    try:
        shutil.copy(source_path, destination_path)
    except Exception as e:
        print(f"Error copying {source_path} to {destination_path}: {e}")

# Save the combined DataFrame to a new CSV file
combined_csv_path = "combined_images_train_validation_test.csv"
combined_df.to_csv(combined_csv_path, index=False)

print(f"Combined CSV file saved to {combined_csv_path}")
print(f"Images copied to {destination_dir}")
