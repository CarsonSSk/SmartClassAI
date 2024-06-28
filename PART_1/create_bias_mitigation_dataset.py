import os
import pandas as pd

# Path to the original dataset CSV
original_csv_path = 'combined_images_train_validation_test.csv'

# Read the original dataset CSV
original_df = pd.read_csv(original_csv_path)

# Directory containing extra images
extra_images_dir = 'data/biasMitigationExtraImageBatch'

# Read the image data CSV
image_data_csv_path = 'data/image_data.csv'
image_data_df = pd.read_csv(image_data_csv_path)

# Initialize lists to hold extra image data
extra_image_paths = []
composite_names = []
labels = []
image_names = []
sources = []
provenances = []
genders = []
ages = []

# Go through the extra images directory
for gender in ['female', 'male']:
    for age in ['old']:
        folder_path = os.path.join(extra_images_dir, gender, age)
        for image_name in os.listdir(folder_path):
            if image_name.endswith(('.jpg', '.png', '.jpeg')):  # Add other image extensions if needed
                image_path = os.path.join(folder_path, image_name)
                composite_name = f"{image_name.split('.')[0]}_{gender}_{age}_FER2013"

                # Check if the image is a duplicate of an image in the original dataset
                if image_name in original_df['Image Name'].values:
                    continue  # Skip duplicates from extra images

                # Match the label based on Image Name from image_data_df
                label = image_data_df.loc[image_data_df['Image Name'] == image_name, 'Label'].values
                if len(label) > 0:
                    labels.append(label[0])
                else:
                    labels.append('unknown')  # Set a default label if Image Name is not found

                extra_image_paths.append(image_path)
                composite_names.append(composite_name)
                image_names.append(image_name)
                sources.append('FER2013')
                provenances.append('Train')  # All extra images are part of the training dataset
                genders.append(gender)
                ages.append(age)

# Create a DataFrame for the extra images
extra_df = pd.DataFrame({
    'Path': extra_image_paths,
    'CompositeName': composite_names,
    'Label': labels,
    'Image Name': image_names,
    'Source': sources,
    'Provenance': provenances,
    'Gender': genders,
    'Age': ages
})

# Combine original and extra dataframes
combined_df = pd.concat([original_df, extra_df], ignore_index=True)

# Check for duplicates in the Image Name column
duplicate_image_names = combined_df[combined_df.duplicated(subset=['Image Name'], keep=False)]

# Count duplicates for each label
duplicate_counts = duplicate_image_names['Label'].value_counts()

# Print the number of duplicates for each label
print("Number of duplicates for each label:")
print(duplicate_counts)

# Print the duplicate image names if any
if not duplicate_image_names.empty:
    print("Duplicate Image Names:")
    print(duplicate_image_names[['Image Name']])

    # Print the duplicating paths for each duplicate based on the image name
    print("Duplicating Paths for each duplicate based on Image Name:")
    for name in duplicate_image_names['Image Name'].unique():
        duplicates = duplicate_image_names[duplicate_image_names['Image Name'] == name]
        paths = duplicates['Path'].unique()
        print(f"Image Name: {name}")
        print(f"Duplicating Paths: {', '.join(paths)}")
        print()
else:
    print("No duplicate image names found.")

# Save the combined DataFrame to a new CSV file
combined_csv_path = 'with_biasMitigation_combined_images_train_validation_test.csv'
combined_df.to_csv(combined_csv_path, index=False)

# Create a folder to store the unbiased datasets if it doesn't exist
unbiased_datasets_folder = 'unbiasedDatasets'
os.makedirs(unbiased_datasets_folder, exist_ok=True)

# Filter the combined dataset to get the unbiased datasets
train_dataset = combined_df[combined_df['Provenance'] == 'Train']
validation_dataset = combined_df[combined_df['Provenance'] == 'Validation']
test_dataset = combined_df[combined_df['Provenance'] == 'Test']

# Define paths for the unbiased datasets
train_dataset_path = os.path.join(unbiased_datasets_folder, 'unbiased_train_dataset.csv')
validation_dataset_path = os.path.join(unbiased_datasets_folder, 'unbiased_validation_dataset.csv')
test_dataset_path = os.path.join(unbiased_datasets_folder, 'unbiased_test_dataset.csv')

# Save the unbiased datasets to CSV files
train_dataset.to_csv(train_dataset_path, index=False)
validation_dataset.to_csv(validation_dataset_path, index=False)
test_dataset.to_csv(test_dataset_path, index=False)

print(f"Original dataset: {original_csv_path}")
print(f"Extra images dataset: DataFrame created from folders, all images marked as 'Train'")
print(f"Combined dataset: {combined_csv_path}")
print(f"Unbiased train dataset saved to: {train_dataset_path}")
print(f"Unbiased validation dataset saved to: {validation_dataset_path}")
print(f"Unbiased test dataset saved to: {test_dataset_path}")
