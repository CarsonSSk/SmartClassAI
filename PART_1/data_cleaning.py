import os
import cv2
import pandas as pd
from PIL import Image
import numpy as np

class DataCleaning:
    def __init__(self, test_csv, train_csv, output_dir, root_dir='PART_1', img_size=(48, 48)):
        self.test_csv = test_csv
        self.train_csv = train_csv
        self.output_dir = output_dir
        self.root_dir = root_dir
        self.img_size = img_size

        # Ensure output directory exists with the required subfolders
        self.setup_output_directories()

    def setup_output_directories(self):
        for subset in ['test', 'train']:
            for label in ['happy', 'angry', 'neutral', 'engaged']:
                path = os.path.join(self.output_dir, subset, label)
                os.makedirs(path, exist_ok=True)

    def read_csv(self, csv_file):
        return pd.read_csv(csv_file, header=None, names=['Path', 'CompositeName', 'Label', 'ImageName', 'Source'])

    def process_image(self, img_path):
        # Prepend root directory to the image path
        img_path = os.path.join(self.root_dir, img_path)
        
        # Normalize the path
        img_path = os.path.normpath(img_path)
        
        # Verify file existence
        if not os.path.exists(img_path):
            print(f"Error: File does not exist at {img_path}")
            return None

        # Read image
        print(f"Processing image: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to read image at {img_path}")
            return None
        
        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize image
        img_resized = cv2.resize(img_gray, self.img_size)

        #Commented cleaning functions are too aggressive for now.

        # Apply Gaussian Blur for noise reduction 
        # img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)

        # Apply Median Blur for noise reduction
        # img_blurred = cv2.medianBlur(img_resized, 5)

        # Apply Bilateral Filter for noise reduction while preserving edges
        # img_filtered = cv2.bilateralFilter(img_resized, d=9, sigmaColor=75, sigmaSpace=75)

        # Apply histogram equalization for contrast enhancement
        img_equalized = cv2.equalizeHist(img_resized)
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_equalized / 255.0

        # Apply binarization
        # img_binarized = cv2.threshold(img_normalized, 0.5, 1.0, cv2.THRESH_BINARY)

        return img_normalized

    def save_image(self, img_array, save_path):
        # Convert back to uint8 format for saving
        img_array = (img_array * 255).astype(np.uint8)
        # Save image
        Image.fromarray(img_array).save(save_path)

    def clean_data(self, csv_file, subset):
        data = self.read_csv(csv_file)

        for index, row in data.iterrows():
            img_path = row['Path']
            label = row['Label']
            
            # Process the image
            cleaned_img = self.process_image(img_path)
            if cleaned_img is None:
                continue

            # Create destination path
            save_dir = os.path.join(self.output_dir, subset, label)
            save_path = os.path.join(save_dir, row['ImageName'])
            
            # Save the cleaned image
            self.save_image(cleaned_img, save_path)

        print(f"Data cleaning for {subset} complete.")

#Parameters at runtime
if __name__ == "__main__":
    test_csv = 'PART_1/data/test_dataset.csv'
    train_csv = 'PART_1/data/train_dataset.csv'
    output_dir = 'PART_1/data/cleanedData'
    
    cleaner = DataCleaning(test_csv, train_csv, output_dir)
    
    cleaner.clean_data(test_csv, 'test')
    cleaner.clean_data(train_csv, 'train')
