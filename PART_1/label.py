import logging
import os
import pandas as pd
from deepface import DeepFace
from tqdm import tqdm

# Suppress TensorFlow logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Base directory containing the sections
base_dir = "./cleanedData"

# List of sections
# sections = ["angry", "engaged", "happy", "neutral"]
sections = ["test"]

# Initialize a list to store results
results = []

# Loop over each section
for section in sections:
    section_dir = os.path.join(base_dir, section)

    # Check if the directory exists
    if os.path.exists(section_dir):
        # Get all image files in the section directory
        image_files = [
            f
            for f in os.listdir(section_dir)
            if f.endswith(".jpg") or f.endswith(".png")
        ]

        # change here if we want to label all the files
        # image_files = image_files[:100]

        # Loop over all images in the section directory with a progress bar
        for filename in tqdm(image_files, desc=f"Processing {section} images"):
            image_path = os.path.join(section_dir, filename)

            try:
                # Analyze the image using DeepFace
                attributes = DeepFace.analyze(image_path, actions=["age", "gender"])

                # Check if attributes is a list (multiple faces detected)
                if isinstance(attributes, list):
                    attributes = attributes[0]  # Consider the first face detected

                # Extract the relevant attributes
                age = attributes.get("age", "error")
                gender = attributes.get("gender", "error")

                # Append the results
                results.append(
                    {
                        "filename": filename,
                        "section": section,
                        "age": age,
                        "gender": gender,
                    }
                )
            except ValueError as e:
                # If a face could not be detected, mark it as an error
                if "Face could not be detected" in str(e):
                    results.append(
                        {
                            "filename": filename,
                            "section": section,
                            "age": "error",
                            "gender": "error",
                        }
                    )
                else:
                    # For other exceptions, just log the error message
                    print(f"Error processing {image_path}: {e}")
            except Exception as e:
                # Log any other exceptions
                print(f"Unexpected error processing {image_path}: {e}")

# Convert the results to a DataFrame
df = pd.DataFrame(results)

# Save the results to a CSV file
df.to_csv("labeled_images.csv", index=False)
