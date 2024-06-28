import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import numpy as np
from data_cleaning import DataCleaning
#from model_training import CNN, CNNModelVariant1, CNNModelVariant2
import csv
import random

# Randomization parameters
randomseed = 2024 # Set the random seed to a specific integer. Changing the seed will yield different model results.
reproducibility = True # Set to false if you want to have different results each run. Can help with generating a high performing model with sequential runs

def set_seed(seed): # Random seed function to set the seeds to all PyTorch/Numpy randomization functions used in the training process, ensuring reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if reproducibility:
    set_seed(randomseed)

# Set the paths to the images you wish to predict, as well as the model to be used
image_path = "runFiles/happy_test7.jpg"  # Specify the path to your image here
directory_path = "runFiles/imagesDirectory"  # Specify the path to your directory here
model_dir = "models/model_2024-06-27_22-56-49"  # Specify the model directory here
model_type = 0 # 0 = Main model, 1 = Variant 1, 2 = Variant 2. You can easily figure the type of the model within the model name.

output_dir="runFiles/runCleanedImages"
directory_csv_file = os.path.join("runFiles", 'directory_images_data.csv')

cleaner=DataCleaning(directory_csv_file, output_dir, log_file="runFiles/run_cleaned_image_data.csv", setupOutputDir=False, setupLog=False)

class CNN(nn.Module): # 8 convolutional layers, 2 pooling layers, 2 FC layers, kernel size=3x3
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(3 * 3 * 256, 1000),
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

# Variant 1: Vary the Number of Convolutional Layers (2 layers -> 63.67% acc and 6 layers -> 69.33% acc)
class CNNModelVariant1(nn.Module): # 10 convolutional layers, 4 pooling layers, 2 FC layers, kernelsize= 3x3
    def __init__(self):
        super(CNNModelVariant1, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1),
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
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(3 * 3 * 512, 1000), # for 10 layers, 4 pooling, 3x3 kernel

            #How to adjust nn.Linear ( x * x * y, 1000) according to number of layers:
            # 1. y =  out_channels of the final convolutional layer/BatchNorm2d parameter.
            # 2. x = 48 / (2 ^ (Number of MaxPool2d layers). i.e. Divide the image size by 2 each time a
            # MaxPool2d is applied. If 4 MaxPool2d, then 48*48 divided by 2, 4 times in a row, gives 3*3.

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

# Variant 2: Experiment with Different Kernel Sizes (2x2 kernel size -> 60.00% acc and 5x5 kernel size -> 62.00% acc)
class CNNModelVariant2(nn.Module): # 8 convolutional layers, kernel size = 2
    def __init__(self):
        super(CNNModelVariant2, self).__init__() # 8 convolutional layers, kernelsize= 5x5
        #kernelsize = 2  # Set the kernel size to 2x2
        kernelsize=5 # Edit Kernel Size for convolutional layers (not MaxPool2d layer)
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=48, kernel_size=kernelsize, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=kernelsize, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=kernelsize, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernelsize, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernelsize, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=kernelsize, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kernelsize, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=kernelsize, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(1 * 1 * 256, 1000), #8 layers 5x5

            # How to adjust nn.Linear ( x * x * y, 1000) according to kernel size:
            # 1. y =  out_channels of the final convolutional layer/BatchNorm2d parameter.
            # 2. x = 48. For each convolutional layer, x = x - (kernelsize - 3). For each pooling layer, x = x/2.
            # These operations must be applied in the same sequence as the layers. Avoid having an uneven number of pixels before applying MaxPool2d..

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
#transform = transforms.Compose([transforms.ToTensor(), normalize])
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(48, padding=4),
    transforms.ToTensor(),
    normalize
])


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
model_path = find_model_file(model_dir)  # Dynamically find the model file

if model_path:
    print(f"Found model file: {model_path}")
    if model_type == 0:
        model_class = CNN
    elif model_type == 1:
        model_class = CNNModelVariant1
    elif model_type == 2:
        model_class = CNNModelVariant2

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

### Generate results on test dataset (for Part 3):
# Define the test dataset path
test_csv_path = "data/test_dataset.csv"  # Replace with the actual path to your test dataset CSV file

# Load the test dataset
test_df = pd.read_csv(test_csv_path)

# Initialize lists to store results
image_paths = []
true_labels = []
predicted_labels = []

# Define image transformations (if not already defined)
normalize = transforms.Normalize(mean=[0.5], std=[0.5])
#transform = transforms.Compose([transforms.ToTensor(), normalize])
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(48, padding=4),
    transforms.ToTensor(),
    normalize
])

# Loop through each image in the test dataset, make predictions and store results
for index, row in test_df.iterrows():
    image_path = row['Path']  # Replace with the column name for image path in your test CSV
    true_label = row['Label']  # Replace with the column name for the true label in your test CSV

    # Open the image
    image = Image.open(image_path)
    image = np.array(image)
    image = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_label = classes[predicted.item()]  # Map numerical index to label string

    # Append to results lists
    image_paths.append(image_path)
    true_labels.append(true_label)
    predicted_labels.append(predicted_label)

# Save results to a CSV file
results_csv_path = "results_run_test.csv"  # Replace with the desired path for the results CSV file
with open(results_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ImagePath", "TrueLabel", "PredictedLabel"])  # Write header
    for img_path, true_lbl, pred_lbl in zip(image_paths, true_labels, predicted_labels):
        writer.writerow([img_path, true_lbl, pred_lbl])

correct_predictions = 0
total_predictions = len(true_labels)

for true_label, predicted_label in zip(true_labels, predicted_labels):
    if true_label == predicted_label:
        correct_predictions += 1

accuracy = correct_predictions / total_predictions * 100
print(f"Test dataset Accuracy: {accuracy:.2f}%")
