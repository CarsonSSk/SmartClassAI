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

# Define the root directory containing the gender and age labeled folders
root_dir = "data/additionalLabels"

# Initialize lists for gender and age labels
genders = []
ages = []

# Copy images to the destination directory and extract gender and age labels
for _, row in combined_df.iterrows():
    image_name = row['Image Name']
    # Find the image in the labeled folders
    found = False
    for gender in ['male', 'female', 'other']:
        for age in ['young', 'middle', 'old', 'other']:
            potential_path = os.path.join(root_dir, gender, age, image_name)
            if os.path.exists(potential_path):
                genders.append(gender)
                ages.append(age)
                destination_path = os.path.join(destination_dir, image_name)
                try:
                    shutil.copy(potential_path, destination_path)
                except Exception as e:
                    print(f"Error copying {potential_path} to {destination_path}: {e}")
                found = True
                break
        if found:
            break
    if not found:
        print(f"Image {image_name} not found in labeled directories.")

# Add Gender and Age columns to the DataFrame
combined_df['Gender'] = genders
combined_df['Age'] = ages

# Save the combined DataFrame to a new CSV file
combined_csv_path = "combined_images_train_validation_test.csv"
combined_df.to_csv(combined_csv_path, index=False)

print(f"Combined CSV file saved to {combined_csv_path}")
print(f"Images copied to {destination_dir}")
