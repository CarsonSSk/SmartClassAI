import os
import torch
import torch.nn as nn
import torch.utils.data as td
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import datetime
import matplotlib.pyplot as plt
import csv
import random

# Defining the directories and CSV files containing the training, validation, and testing data
base_dir = "data"
train_data = pd.read_csv(os.path.join("unbiasedDatasets/unbiased_train_dataset.csv"))
val_data = pd.read_csv(os.path.join("unbiasedDatasets/unbiased_validation_dataset.csv"))
test_data = pd.read_csv(os.path.join("unbiasedDatasets/unbiased_test_dataset.csv"))

# Define model parameters and settings
input_size = 48 * 48  # Input size for 48x48 grayscale images
hidden_size = 50  # Number of hidden units
output_size = 4  # Number of output classes
num_epochs = 10  # Minimum number of training epochs
learning_rate = 0.005  # Learning rate
patience = 5  # Early stopping patience
randomseed = 2042 # Set the random seed to a specific integer. Changing the seed will yield different model results.
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

# Label mapping to facilitate use with PyTorch and tensor formats
label_mapping = {'happy': 0, 'angry': 1, 'neutral': 2, 'engaged': 3}
classes = ['happy', 'angry', 'neutral', 'engaged']

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

# Normalization values for our dataset + Data augmentation techniques
# Data augmentation techniques used: random horizontal flip, random rotation, random crop
normalize = transforms.Normalize(mean=[0.5], std=[0.5])
#transform = transforms.Compose([transforms.ToTensor(), normalize]) # Without data augmentation techniques
transform = transforms.Compose([ # With data augmentation techniques
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(48, padding=4),
    transforms.ToTensor(),
    normalize
])

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

# Load training, validation, and test data
x_train, y_train = LoadData(train_data)
x_val, y_val = LoadData(val_data)
x_test, y_test = LoadData(test_data)

train_dataset = CustomDataset(x_train, y_train, transform=transform)
val_dataset = CustomDataset(x_val, y_val, transform=transform)
test_dataset = CustomDataset(x_test, y_test, transform=transform)

# Create data loaders for training, validation, and testing
train_loader = td.DataLoader(train_dataset, batch_size=20, shuffle=True)
val_loader = td.DataLoader(val_dataset, batch_size=20, shuffle=False)
test_loader = td.DataLoader(test_dataset, batch_size=20, shuffle=False)

# CNN (Based on CIFAR10 example in Lab 7)
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


def train_models(model, model_name):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    best_val_loss = float('inf')
    best_model_path = os.path.join("models", model_name, "best_model.pth")
    os.makedirs(os.path.join("models", model_name), exist_ok=True)
    patience_counter = 0
    limitHit = 0

    for epoch in range(30):  # Setting maximum epochs to 30
        model.train()
        avg_loss_epoch = 0
        total_batches = 0
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_batches += 1
            avg_loss_epoch += loss.item()

        avg_loss_epoch /= total_batches
        print(f'Epoch [{epoch+1}/30], Average Loss for {model_name} epoch[{epoch+1}] = {avg_loss_epoch:.4f}')

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f'Validation Loss for {model_name} epoch[{epoch+1}] = {val_loss:.4f}')

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # Print patience counter after each epoch
        print(f'Patience counter: {patience_counter} / {patience}')

        # Ensure training stops early at epoch 10 if patience counter is met before epoch 10
        if patience_counter == patience:
            limitHit +=1

        # Ensure training runs for at least 10 epochs before stopping
        if epoch >= 9 and limitHit > 0:
            print(f'Early stopping at epoch {epoch+1}')
            break

    # Save the validation loss
    validation_loss_csv_path = os.path.join("models", model_name, "validation_loss.csv")
    val_loss_df = pd.DataFrame({"Validation Loss": [best_val_loss]})
    val_loss_df.to_csv(validation_loss_csv_path, index=False)

    # Evaluate the model on test data
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    correct = 0
    total = 0

    results_df = pd.DataFrame({
        'Image Path': test_data['Path'],
        'Correct Label': test_data['Label'],
        'Predicted Label': "N/A"
    })

    for i in range(len(test_dataset)):
        image, label = test_dataset[i]
        image = image.unsqueeze(0)
        output = model(image)
        predicted = torch.argmax(output, dim=1)
        results_df.loc[i, 'Predicted Label'] = classes[predicted.item()]
        total += 1
        correct += (predicted == label).item()

    accuracy = 100 * correct / total
    print(f'Accuracy of {model_name} on the test images: {accuracy:.2f}%')

    # Save the results
    results_csv_path = os.path.join("models", model_name, "results.csv")
    results_df.to_csv(results_csv_path, index=False)
    accuracy_csv_path = os.path.join("models", model_name, "accuracy.csv")
    accuracy_df = pd.DataFrame({"Accuracy": [accuracy]})
    accuracy_df.to_csv(accuracy_csv_path, index=False)

    print(f"Model and related files saved in: models/{model_name}")

    return best_val_loss

# Track the best model across all variants
best_model_overall = None
best_val_loss_overall = float('inf')
best_model_name = ""

# Dictionaries to store the best model for each type
best_models = {
    "main": {"model": None, "val_loss": float('inf'), "name": ""},
    "variant1": {"model": None, "val_loss": float('inf'), "name": ""},
    "variant2": {"model": None, "val_loss": float('inf'), "name": ""}
}

# Train and evaluate the main model
main_model_name = f"model_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
main_model = CNN()
main_model_val_loss = train_models(main_model, main_model_name)
if main_model_val_loss < best_val_loss_overall:
    best_val_loss_overall = main_model_val_loss
    best_model_overall = main_model
    best_model_name = main_model_name
if main_model_val_loss < best_models["main"]["val_loss"]:
    best_models["main"]["val_loss"] = main_model_val_loss
    best_models["main"]["model"] = main_model
    best_models["main"]["name"] = main_model_name
### Visualization section
# Plot some example images with their predicted labels

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# Function to plot results
def plot_results(model, test_dataset, x=2, y=5):
    width_per_image = 2.4
    indices = torch.randperm(len(test_dataset))[:x * y]
    images = torch.stack([test_dataset[i][0] for i in indices])
    labels = torch.tensor([test_dataset[i][1] for i in indices])

    # Ensure the images are reshaped to [batch_size, 1, 48, 48]
    if images.dim() == 3:  # [batch_size, 48, 48]
        images = images.unsqueeze(1)  # [batch_size, 1, 48, 48]
    elif images.dim() == 4 and images.shape[1] != 1:  # [batch_size, channels, 48, 48]
        images = images[:, :1, :, :]  # Ensure single channel if multiple channels are present

    # Move images to the same device as the model
    images = images.to(next(model.parameters()).device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(images)

    _, predicted = torch.max(outputs.data, 1)
    fig, axes = plt.subplots(x, y, figsize=(y * width_per_image, x * width_per_image))
    for i, ax in enumerate(axes.ravel()):
        img = denormalize(images[i], [0.5], [0.5])
        img = img.squeeze().numpy()
        true_label = classes[labels[i]]
        pred_label = classes[predicted[i]]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"true='{true_label}', pred='{pred_label}'", fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Plot results for the main model
plot_results(main_model, test_dataset)
