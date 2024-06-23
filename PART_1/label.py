import logging
import pandas as pd
import matplotlib.pyplot as plt
from deepface import DeepFace
from tqdm import tqdm

# Suppress TensorFlow logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Paths to the original CSV files
csv_paths = ["data/train_dataset.csv", "data/validation_dataset.csv", "data/test_dataset.csv"]

# Corresponding output CSV file paths
output_csv_paths = ["labeled_images_with_gender_age_train.csv", "labeled_images_with_gender_age_validation.csv",
                    "labeled_images_with_gender_age_test.csv"]

# Loop through each CSV file
for csv_path, output_csv_path in zip(csv_paths, output_csv_paths):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Initialize lists to store new columns
    genders = []
    ages = []

    # Loop over each image in the CSV file with a progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing images from {csv_path}"):
        image_path = row['Path']  # Assuming the path to the image is in the 'Path' column
        try:
            # Analyze the image using DeepFace
            attributes = DeepFace.analyze(image_path, actions=["age", "gender"])

            # Check if attributes is a list (multiple faces detected)
            if isinstance(attributes, list):
                attributes = attributes[0]  # Consider the first face detected

            # Extract the most probable gender
            gender = max(attributes['gender'], key=attributes['gender'].get)

            # Extract the age value
            age = attributes.get("age", "error")

        except ValueError as e:
            # If a face could not be detected, mark it as an error
            if "Face could not be detected" in str(e):
                age = "error"
                gender = "error"
            else:
                # For other exceptions, just log the error message
                print(f"Error processing {image_path}: {e}")
                age = "error"
                gender = "error"
        except Exception as e:
            # Log any other exceptions
            print(f"Unexpected error processing {image_path}: {e}")
            age = "error"
            gender = "error"

        # Append the results to the lists
        ages.append(age)
        genders.append(gender)

    # Add the new columns to the DataFrame
    df['Gender'] = genders
    df['Age'] = ages

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv_path, index=False)

    # Generate and display the age distribution plot for the current CSV file
    age_values = [age for age in ages if age != "error"]
    plt.hist(age_values, bins=20, edgecolor='k', alpha=0.7)
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title(f'Age Distribution for {csv_path}')
    plt.grid(True)
    plt.show()
