## This code is heavily inspired by the Lab 6 questions 2-3 codes based on the CIFAR10 and Iris datasets, adapted to our own data structure.

import os
import torch
import torch.nn as nn
import torch.utils.data as td
import torch.nn.functional as F
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import datetime

# Defining the directories and CSV files containing the training and testing data (already split)
base_dir = "data"
train_data = pd.read_csv(os.path.join(base_dir, "train_dataset.csv"))
test_data = pd.read_csv(os.path.join(base_dir, "test_dataset.csv"))

# Define model parameters and settings
input_size = 48 * 48  # Input size for 48x48 grayscale images
hidden_size = 50  # Number of hidden units
output_size = 4  # Number of output classes
num_epochs = 10  # Number of training epochs
learning_rate = 0.005  # Learning rate

# Label mapping to facilitate use with PyTorch and tensor formats
label_mapping = {'happy': 0, 'angry': 1, 'neutral': 2, 'engaged': 3}
classes = ['happy', 'angry', 'neutral', 'engaged']

# Creating a dataframe to store the predicted labels for each image to allow subsequent evaluation and visualization.
results_df = pd.DataFrame({
    'Image Path': test_data['Path'],
    'Correct Label': test_data['Label'],
    'Predicted Label': "N/A"
})

# Function to load data and labels from CSV files
def LoadData(data):
    data_x = []
    data_y = []
    for index, row in data.iterrows():
        # Load the image using the path and convert to grayscale
        image = Image.open(row['Path']).convert('L')
        # Convert the image to a numpy array
        image = np.array(image)
        # Append the image array to data_x
        data_x.append(image)
        # Append the label to data_y
        data_y.append(label_mapping[row['Label']])
    # Convert lists to arrays
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    return data_x, data_y

# Normalization values for our dataset
normalize = transforms.Normalize(mean=[0.5], std=[0.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# Custom dataset class to handle the data loading with our specific data structure
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# Load training and test data
x_train, y_train = LoadData(train_data)
x_test, y_test = LoadData(test_data)
train_dataset = CustomDataset(x_train, y_train, transform=transform)
test_dataset = CustomDataset(x_test, y_test, transform=transform)

# Create data loaders for training and testing
train_loader = td.DataLoader(train_dataset, batch_size=20, shuffle=True)
test_loader = td.DataLoader(test_dataset, batch_size=20, shuffle=False)

# Define the multi-layer fully connected neural network
class MultiLayerFCNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MultiLayerFCNet, self).__init__()
        # Define the input layer (from input dimension to hidden dimension)
        self.linear1 = torch.nn.Linear(D_in, H)
        # Define two hidden layers
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        # Define the output layer (from hidden dimension to output dimension)
        self.linear4 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        # Pass input through the input layer and apply ReLU activation
        x = F.relu(self.linear1(x))
        # Pass through the first hidden layer and apply ReLU activation
        x = F.relu(self.linear2(x))
        # Pass through the second hidden layer and apply ReLU activation
        x = F.relu(self.linear3(x))
        # Pass through the output layer
        x = self.linear4(x)
        # Apply log_softmax to get log probabilities for multi-class classification
        return F.log_softmax(x, dim=1)

#Variant 1: Vary the Number of Convolutional Layers    
class CNNModelVariant1(nn.Module):
    def __init__(self):
        super(CNNModelVariant1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Additional layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 4)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

#Variant 2: Experiment with Different Kernel Sizes
class CNNModelVariant2(nn.Module):
    def __init__(self):
        super(CNNModelVariant2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)  # Larger kernel size
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 4)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_models(model, model_name):
# Initialize the model, loss function, and optimizer
#model = MultiLayerFCNet(input_size, hidden_size, output_size)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # Train the model
    for epoch in range(num_epochs):
        avg_loss_epoch = 0  # To keep track of average loss for each epoch
        batch_loss = 0  # Sum of losses for the batches processed
        total_batches = 0  # Total batches processed
        for images, labels in train_loader:
            if isinstance(model, MultiLayerFCNet):
                # Reshape images to match the input size of the model
                images = images.view(-1, 48 * 48)
            # Get model predictions for the current batch
            outputs = model(images)
            # Compute the loss between the predicted outputs and true labels
            loss = criterion(outputs, labels)
            # Clear previous gradients
            optimizer.zero_grad()
            # Backpropagate to compute gradients
            loss.backward()
            # Update model parameters
            optimizer.step()
            total_batches += 1
            batch_loss += loss.item()
        # Compute average loss for the current epoch
        avg_loss_epoch = batch_loss / total_batches
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss for {model_name} epoch[{epoch+1}] = {avg_loss_epoch:.4f}')
    
    # Evaluate the model on test data
    # Initialize counters
    correct = 0
    total = 0

    # Loop through test data (one by one and not batch by batch)
    for i in range(len(test_dataset)):
        image, label = test_dataset[i]
        if isinstance(model, MultiLayerFCNet):
            # Reshape the image to match the input size of the model
            image = image.view(-1, 48 * 48)
        else:
            # Add batch dimension to match the input size of the model
            image = image.unsqueeze(0)
        # Get prediction for current image
        output = model(image)

        # Get predicted class label using torch.argmax()
        predicted = torch.argmax(output, dim=1)

        # Insert predicted class in the results dataframe
        results_df.loc[i, 'Predicted Label'] = classes[predicted.item()]

        # Update total number of images processed
        total += 1

        # Update correct counter by comparing predicted labels to true labels
        correct += (predicted == label).item()

    # Calculate and print accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy of {model_name} on the test images: {accuracy:.2f}%')

    # Save the model and prediction data
    # Create a directory for the model
    model_dir = os.path.join("models", model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)

    # Save the results in CSV
    results_csv_path = os.path.join(model_dir, "results.csv")
    results_df.to_csv(results_csv_path, index=False)

    # Save the CSV file containing the training set
    train_csv_path = os.path.join(model_dir, "train_dataset.csv")
    train_data.to_csv(train_csv_path, index=False)

    # Save the CSV file containing the testing set
    test_csv_path = os.path.join(model_dir, "test_dataset.csv")
    test_data.to_csv(test_csv_path, index=False)

    # Save the CSV file containing the accuracy value of the model tested on the test set
    accuracy_csv_path = os.path.join(model_dir, "accuracy.csv")
    accuracy_df = pd.DataFrame({"Accuracy": [accuracy]})
    accuracy_df.to_csv(accuracy_csv_path, index=False)

    print(f"Model and related files saved in: {model_dir}")

# Train and evaluate the main model
main_model = MultiLayerFCNet(input_size, hidden_size, output_size)
train_models(main_model, f"model_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

# Train and evaluate variant 1
variant1_model = CNNModelVariant1()
train_models(variant1_model, f"variant1_model_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

# Train and evaluate variant 2
variant2_model = CNNModelVariant2()
train_models(variant2_model, f"variant2_model_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

### Visualization section
# Plot some example images with their predicted labels

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# Function to plot results
def plot_results(model, test_dataset, x=2, y=5):
    # Width per image (inches)
    width_per_image = 2.4
    # Shuffle data to ensure variety in labels
    indices = torch.randperm(len(test_dataset))[:x * y]
    images = torch.stack([test_dataset[i][0] for i in indices])
    labels = torch.tensor([test_dataset[i][1] for i in indices])
    # Get predictions for these images
    random_images_reshaped = images.view(-1, 48 * 48)
    outputs = model(random_images_reshaped)
    _, predicted = torch.max(outputs.data, 1)
    fig, axes = plt.subplots(x, y, figsize=(y * width_per_image, x * width_per_image))
    # Iterate over the random images and display them along with their predicted labels
    for i, ax in enumerate(axes.ravel()):
        # Denormalize image
        img = denormalize(images[i], [0.5], [0.5])
        img = img.squeeze().numpy()  # Convert image to numpy array
        true_label = classes[labels[i]]
        pred_label = classes[predicted[i]]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"true='{true_label}', pred='{pred_label}'", fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Plot results for the main model
plot_results(main_model, test_dataset)

# Plot results for variant 1
plot_results(variant1_model, test_dataset)

# Plot results for variant 2
plot_results(variant2_model, test_dataset)