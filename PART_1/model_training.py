import os
import torch
import torch.nn as nn
import torch.utils.data as td
import torch.nn.functional as F
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import datetime
import matplotlib.pyplot as plt

# Defining the directories and CSV files containing the training, validation, and testing data
base_dir = "data"
train_data = pd.read_csv(os.path.join(base_dir, "train_dataset.csv"))
val_data = pd.read_csv(os.path.join(base_dir, "validation_dataset.csv"))
test_data = pd.read_csv(os.path.join(base_dir, "test_dataset.csv"))

# Define model parameters and settings
input_size = 48 * 48  # Input size for 48x48 grayscale images
hidden_size = 50  # Number of hidden units
output_size = 4  # Number of output classes
num_epochs = 10  # Minimum number of training epochs
learning_rate = 0.005  # Learning rate
patience = 3  # Early stopping patience

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

# Define the multi-layer fully connected neural network
class MultiLayerFCNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MultiLayerFCNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return F.log_softmax(x, dim=1)

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

# Variant 1: Vary the Number of Convolutional Layers
class CNNModelVariant1(nn.Module):
    def __init__(self):
        super(CNNModelVariant1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 4)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        if x.dim() == 2:
            batch_size = x.size(0)
            x = x.view(batch_size, 1, 48, 48)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Variant 2: Experiment with Different Kernel Sizes
class CNNModelVariant2(nn.Module):
    def __init__(self):
        super(CNNModelVariant2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 4)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        if x.dim() == 2:
            batch_size = x.size(0)
            x = x.view(batch_size, 1, 48, 48)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_models(model, model_name):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    best_val_loss = float('inf')
    best_model_path = os.path.join("models", model_name, "best_model.pth")
    os.makedirs(os.path.join("models", model_name), exist_ok=True)
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        avg_loss_epoch = 0
        total_batches = 0
        for images, labels in train_loader:
            if isinstance(model, MultiLayerFCNet):
                images = images.view(-1, 48 * 48)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_batches += 1
            avg_loss_epoch += loss.item()
        
        avg_loss_epoch /= total_batches
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss for {model_name} epoch[{epoch+1}] = {avg_loss_epoch:.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                if isinstance(model, MultiLayerFCNet):
                    images = images.view(-1, 48 * 48)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f'Validation Loss for {model_name} epoch[{epoch+1}] = {val_loss:.4f}')
        
        if epoch >= num_epochs - 1:  # Ensure we run for at least num_epochs epochs
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
        
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
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
        if isinstance(model, MultiLayerFCNet):
            image = image.view(-1, 48 * 48)
        else:
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

# Train and evaluate the main model
main_model_name = f"model_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
main_model = CNN()
main_model_val_loss = train_models(main_model, main_model_name)
if main_model_val_loss < best_val_loss_overall:
    best_val_loss_overall = main_model_val_loss
    best_model_overall = main_model
    best_model_name = main_model_name

# Train and evaluate variant 1
variant1_model_name = f"variant1_model_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
variant1_model = CNNModelVariant1()
variant1_model_val_loss = train_models(variant1_model, variant1_model_name)
if variant1_model_val_loss < best_val_loss_overall:
    best_val_loss_overall = variant1_model_val_loss
    best_model_overall = variant1_model
    best_model_name = variant1_model_name

# Train and evaluate variant 2
variant2_model_name = f"variant2_model_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
variant2_model = CNNModelVariant2()
variant2_model_val_loss = train_models(variant2_model, variant2_model_name)
if variant2_model_val_loss < best_val_loss_overall:
    best_val_loss_overall = variant2_model_val_loss
    best_model_overall = variant2_model
    best_model_name = variant2_model_name

print(f"Best model overall: {best_model_name} with validation loss: {best_val_loss_overall:.4f}")

# Save the best model to models/best/
best_model_save_path = os.path.join("models", "best", f"{best_model_name}_best.pth")
os.makedirs(os.path.dirname(best_model_save_path), exist_ok=True)
torch.save(best_model_overall.state_dict(), best_model_save_path)
print(f"Best model saved at: {best_model_save_path}")

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

    # Debug: Print the shape of the images tensor
    print(f"Shape of images tensor: {images.shape}")

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

# Plot results for variant 1
plot_results(variant1_model, test_dataset)

# Plot results for variant 2
plot_results(variant2_model, test_dataset)
