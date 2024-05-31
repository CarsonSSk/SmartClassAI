import os
import cv2
import pandas as pd
from PIL import Image
import numpy as np
import csv

# Define the directories
base_dir = "data"
data = pd.read_csv(os.path.join(base_dir, "image_data.csv"))
image_data_csv = os.path.join(base_dir, "image_data.csv")
output_dir = os.path.join(base_dir, "cleanedData")
log_file = os.path.join(base_dir, "cleaned_image_data.csv")
class DataCleaning:
    def __init__(self, image_data_csv, output_dir, root_dir='PART_1', img_size=(48,48), log_file='PART_1/data/cleaned_image_data.csv'):
        self.image_data_csv = image_data_csv
        self.output_dir = output_dir
        self.root_dir = root_dir
        self.img_size = img_size
        self.log_file = log_file

        # Ensure output directory exists with the required subfolders
        self.setup_output_directories()

        # Ensure the log file exists and write headers if it doesn't
        self.setup_log_file()

    def setup_output_directories(self):
        for label in ['happy', 'angry', 'neutral', 'engaged']:
            path = os.path.join(self.output_dir, label)
            os.makedirs(path, exist_ok=True)

    def setup_log_file(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Image Name', 'Path', 'Label', 'Source', 'CompositeName'])

    def read_csv(self, csv_file):
        return pd.read_csv(csv_file, header=0, names=['ImageName', 'Path', 'Label', 'Source', 'CompositeName'])

    def process_image(self, img_path):

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
        #Convert from BGR to RGB
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        #Resize image
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)

        #Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        #Apply Global histogram equalization for contrast enhancement
        img = cv2.equalizeHist(img)

        # Apply Gaussian Blur for noise reduction - Feature disabled
        #img = cv2.GaussianBlur(img, (3, 3), 0)

        # Apply Median Blur for noise reduction - Feature disabled
        #img = cv2.medianBlur(img, 5)

        # Apply Bilateral Filter for noise reduction while preserving edges - Feature disabled
        # img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

        # Normalize pixel values to [0, 1]
        img = img / 255.0

        return img

    def save_image(self, img_array, save_path):
        img = Image.fromarray(np.uint8(img_array*255))
        img.save(save_path)

    def log_image_data(self, row, save_path):
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            file_is_empty = os.stat(self.log_file).st_size == 0
            if file_is_empty: # Writes csv file headers
                writer.writerow(['Image Name', 'Path', 'Label', 'Source', 'CompositeName'])
            writer.writerow([row['ImageName'], save_path, row['Label'], row['Source'], row['CompositeName']])

    def clean_data(self, csv_file):
        data = self.read_csv(csv_file)
        for index, row in data.iterrows():
            img_path = row['Path']
            label = row['Label']

            # Process the image
            cleaned_img = self.process_image(img_path)
            if cleaned_img is None:
                continue

            # Create destination path
            save_dir = os.path.join(self.output_dir, label)
            save_path = os.path.join(save_dir, row['ImageName'])

            # Save the cleaned image
            self.save_image(cleaned_img, save_path)

            # Log the image data in csv
            self.log_image_data(row, save_path)

        print(f"Data cleaning complete.")

# Runtime properties
if __name__ == "__main__":
    print(data)
    # Clear previous log file
    f = open(log_file, "w")
    f.truncate()
    f.close()

    # Clean the data
    cleaner = DataCleaning(image_data_csv, output_dir, log_file=log_file)
    cleaner.clean_data(image_data_csv)
