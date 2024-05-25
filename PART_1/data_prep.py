import os
import zipfile
import shutil
import numpy as np
from PIL import Image
import pandas as pd

# Ensure the directory exists
if not os.path.exists('data/fer2013'):
    os.makedirs('data/fer2013')

# Download the FER-2013 dataset using Kaggle API with --force option
os.system('kaggle datasets download -d msambare/fer2013 -p data/fer2013 --force')

# Extract the dataset if not already extracted
zip_path = 'data/fer2013/fer2013.zip'
extract_path = 'data/fer2013'
if not os.path.exists(os.path.join(extract_path, 'train')) or not os.path.exists(os.path.join(extract_path, 'test')):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# Define the required classes and corresponding subdirectories
required_classes = ['happy', 'angry', 'neutral']
class_mapping = {'happy': 'happy', 'angry': 'angry', 'neutral': 'neutral'}

# Create directories for organized data
base_dir = 'data/fer2013/organized'

# Remove the existing directories if they exist
for subset in ['train', 'test']:
    subset_dir = os.path.join(base_dir, subset)
    if os.path.exists(subset_dir):
        shutil.rmtree(subset_dir)

# Create directories for organized data
for subset in ['train', 'test']:
    subset_dir = os.path.join(base_dir, subset)
    if not os.path.exists(subset_dir):
        os.makedirs(subset_dir)
    for emotion in required_classes:
        emotion_dir = os.path.join(subset_dir, emotion)
        if not os.path.exists(emotion_dir):
            os.makedirs(emotion_dir)

# Helper function to process images
def process_images(src_dir, dest_dir, limit, img_size=(48, 48)):
    count = 0
    for filename in os.listdir(src_dir):
        if count >= limit:
            break
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            src_path = os.path.join(src_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            img = Image.open(src_path).convert('L')
            img = img.resize(img_size)
            img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
            img_array = (img_array * 255).astype(np.uint8)  # Convert back to uint8 format for saving
            Image.fromarray(img_array).save(dest_path)
            count += 1
    return count

# Copy and process images for each subset and each emotion
for subset in ['train', 'test']:
    for emotion in required_classes:
        src_dir = os.path.join(extract_path, subset, class_mapping[emotion])
        dest_dir = os.path.join(base_dir, subset, emotion)
        limit = 400 if subset == 'train' else 100
        processed_count = process_images(src_dir, dest_dir, limit)
        if processed_count < limit:
            print(f"Warning: Only {processed_count} images processed for {emotion} in {subset}. Needed {limit}.")

print("Data preparation complete.")

# Remove the original data directories to save space
for subset in ['train', 'test']:
    for emotion in required_classes:
        dir_to_remove = os.path.join(extract_path, subset, class_mapping[emotion])
        if os.path.exists(dir_to_remove):
            shutil.rmtree(dir_to_remove)

print("Original data removed.")
