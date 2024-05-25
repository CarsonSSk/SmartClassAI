import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

# Define the dataset path
dataset_path = 'data/fer2013/fer2013.zip'

# Download the FER-2013 dataset using Kaggle API
os.system('kaggle datasets download -d msambare/fer2013 -p data/fer2013')

# Extract the dataset
with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall('data/fer2013')

# Load the dataset
data = pd.read_csv('data/fer2013/fer2013.csv')

# Filter the dataset for the required classes
required_classes = ['happy', 'angry', 'neutral']
class_mapping = {'happy': 3, 'angry': 0, 'neutral': 6}
data = data[data['emotion'].isin(class_mapping.values())]

# Map numerical labels to class names
data['emotion'] = data['emotion'].map({v: k for k, v in class_mapping.items()})

# Create directories for organized data
base_dir = 'data/fer2013/organized'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

for subset in ['train', 'test']:
    subset_dir = os.path.join(base_dir, subset)
    if not os.path.exists(subset_dir):
        os.makedirs(subset_dir)
    for emotion in required_classes:
        emotion_dir = os.path.join(subset_dir, emotion)
        if not os.path.exists(emotion_dir):
            os.makedirs(emotion_dir)

# Split data into training and testing sets
train_data, test_data = train_test_split(data, stratify=data['emotion'], test_size=0.2, random_state=42)

# Ensure each class has the required number of images
train_data = train_data.groupby('emotion').head(400)
test_data = test_data.groupby('emotion').head(100)

# Helper function to save images
def save_images(data, subset):
    for i, row in data.iterrows():
        img = np.fromstring(row['pixels'], dtype=int, sep=' ').reshape(48, 48)
        img_path = os.path.join(base_dir, subset, row['emotion'], f"{i}.png")
        cv2.imwrite(img_path, img)

# Save training and testing images
save_images(train_data, 'train')
save_images(test_data, 'test')
