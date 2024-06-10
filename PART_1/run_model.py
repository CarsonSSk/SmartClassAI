import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import numpy as np
from data_cleaning import DataCleaning
import csv

image_path = "runFiles/PrivateTest_46114477.jpg"  # Specify the path to your image here
directory_path = "runFiles/imagesDirectory"  # Specify the path to your directory here
output_dir="runFiles/runCleanedImages"
directory_csv_file = os.path.join("runFiles", 'directory_images_data.csv')

cleaner=DataCleaning(directory_csv_file, output_dir, log_file="runFiles/run_cleaned_image_data.csv", setupOutputDir=False, setupLog=False)

# Store images to predict in CSV
data = []
images_folder = os.path.join(directory_path)
for filename in os.listdir(images_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        filepath = os.path.join(images_folder, filename)
        image_name = os.path.splitext(filename)[0]
        composite_name = f"{image_name}_PREDICT"
        data.append((filename, filepath, "N/A", "Predict Directory", composite_name))
# Write data to CSV
with open(directory_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Name', 'Path', 'Predicted Label', 'Source', 'CompositeName'])
    writer.writerows(data)

# Clean single image
cleanedImage=DataCleaning.process_image(cleaner, image_path)
img = Image.fromarray(np.uint8(cleanedImage*255))
img.save("runFiles/cleanedSingleImage.jpg")
image_path = "runFiles/cleanedSingleImage.jpg"

# Clean images directory
for image in os.listdir(images_folder):
    cleanImg = DataCleaning.process_image(cleaner, f"{directory_path}/{image}")
    img = Image.fromarray(np.uint8(cleanImg*255))
    img.save(f"runFiles/runCleanedImages/{image}")
directory_path = "runFiles/runCleanedImages"

# Define model parameters and settings
input_size = 48 * 48  # Input size for 48x48 grayscale images
hidden_size = 50  # Number of hidden units
output_size = 4  # Number of output classes

# Label mapping to facilitate use with PyTorch and tensor formats
label_mapping = {'happy': 0, 'angry': 1, 'neutral': 2, 'engaged': 3}
classes = ['happy', 'angry', 'neutral', 'engaged']

# Normalization values for our dataset
normalize = transforms.Normalize(mean=[0.5], std=[0.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# CNN (Based on CIFAR10 example in Lab 7)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(12 * 12 * 64, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        if x.dim() == 2:
            batch_size = x.size(0)
            x = x.view(batch_size, 1, 48, 48)
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

def load_model(model_path, model_class):
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_image(model, image_path):
    image = Image.open(image_path).convert('L')
    image = np.array(image)
    image = transform(image).unsqueeze(0)
    output = model(image)
    _, predicted = torch.max(output, 1)
    return classes[predicted.item()]

def predict_directory(model, directory_path):
    predictions = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(directory_path, filename)
            prediction = predict_image(model, image_path)
            predictions[filename] = prediction

    return predictions

def find_model_file(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".pth"):
            return os.path.join(directory, filename)
    return None


def update_predictions_in_csv(csv_file, predictions):
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        data = [row for row in reader]
    # Update predictions
    for row in data:
        image_name = row[0]
        if image_name in predictions:
            row[2] = predictions[image_name]  # Update the "Predicted Label" column

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)

# Example usage
model_dir = "models/best"  # Specify the model directory here
model_path = find_model_file(model_dir)  # Dynamically find the model file

if model_path:
    print(f"Found model file: {model_path}")
    model_class = CNN  # Specify the model class here

    # Load the model
    model = load_model(model_path, model_class)

    # Predict a single image
    print(f"Predicting facial emotion for single image: ")
    prediction = predict_image(model, image_path)
    print(f"Prediction for {image_path}: {prediction}")

    # Predict all images in a directory
    print(f"Predicting facial emotions for images in directory {directory_path}: ")
    predictions = predict_directory(model, directory_path)
    for image_name, prediction in predictions.items():
        print(f"Prediction for {image_name}: {prediction}")
    update_predictions_in_csv(directory_csv_file, predictions)

else:
    print("No model file found in the specified directory.")