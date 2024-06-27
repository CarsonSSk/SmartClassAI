import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support, confusion_matrix
from torchvision import transforms
from PIL import Image
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Add the PART_1 directory to the Python path
sys.path.append(os.path.abspath('../PART_1'))

# Define parameters
num_epochs = 30
learning_rate = 0.005
patience = 5
randomseed = 2026
batch_size = 20
base_dir = os.path.abspath('../PART_1')  # Define the base directory for PART_1

# Set seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(randomseed)

# Load combined k-fold dataset
kfold_data_csv = os.path.join(base_dir, 'data/kfold_dataset.csv')
data = pd.read_csv(kfold_data_csv)

# Label mapping
label_mapping = {'happy': 0, 'angry': 1, 'neutral': 2, 'engaged': 3}
classes = ['happy', 'angry', 'neutral', 'engaged']

# Function to perform k-fold cross-validation
def kfold_cross_validation(model_class, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=randomseed)

    X = data['Path'].values
    y = data['Label'].map(label_mapping).values

    fold_results = []
    overall_test_results = []

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Further split the training set into training and validation sets
        val_split = int(len(X_train) * 0.85)
        X_train, X_val = X_train[:val_split], X_train[val_split:]
        y_train, y_val = y_train[:val_split], y_train[val_split:]

        # Load data into custom datasets
        train_dataset = CustomDataset(*LoadDataFromPaths(X_train, y_train, base_dir), transform=transform)
        val_dataset = CustomDataset(*LoadDataFromPaths(X_val, y_val, base_dir), transform=transform)
        test_dataset = CustomDataset(*LoadDataFromPaths(X_test, y_test, base_dir), transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize the model
        model = model_class()

        # Train the model
        model_name = f"model_fold_{fold + 1}"
        train_models(model, train_loader, val_loader, model_name)

        # Evaluate the model
        model.load_state_dict(torch.load(os.path.join("models", model_name, "best_model.pth")))
        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.tolist())
                y_pred.extend(predicted.tolist())

        # Calculate metrics
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

        fold_results.append({
            'Fold': fold + 1,
            'Accuracy': accuracy,
            'Precision (Macro)': precision_macro,
            'Recall (Macro)': recall_macro,
            'F1-Score (Macro)': f1_macro,
            'Precision (Micro)': precision_micro,
            'Recall (Micro)': recall_micro,
            'F1-Score (Micro)': f1_micro
        })

        print(f"Fold {fold + 1} - Accuracy: {accuracy:.4f}, Precision (Macro): {precision_macro:.4f}, Recall (Macro): {recall_macro:.4f}, F1-Score (Macro): {f1_macro:.4f}")
        print(f"Fold {fold + 1} - Precision (Micro): {precision_micro:.4f}, Recall (Micro): {recall_micro:.4f}, F1-Score (Micro): {f1_micro:.4f}")

        # Append test results to overall test results
        for true_label, predicted_label in zip(y_true, y_pred):
            overall_test_results.append({
                'Fold': fold + 1,
                'True Label': classes[true_label],
                'Predicted Label': classes[predicted_label]
            })

    # Calculate average metrics across all folds
    avg_metrics = {
        'Fold': 'Average',
        'Accuracy': np.mean([result['Accuracy'] for result in fold_results]),
        'Precision (Macro)': np.mean([result['Precision (Macro)'] for result in fold_results]),
        'Recall (Macro)': np.mean([result['Recall (Macro)'] for result in fold_results]),
        'F1-Score (Macro)': np.mean([result['F1-Score (Macro)'] for result in fold_results]),
        'Precision (Micro)': np.mean([result['Precision (Micro)'] for result in fold_results]),
        'Recall (Micro)': np.mean([result['Recall (Micro)'] for result in fold_results]),
        'F1-Score (Micro)': np.mean([result['F1-Score (Micro)'] for result in fold_results])
    }
    fold_results.append(avg_metrics)

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv('kfold_results.csv', index=False)
    print("K-Fold cross-validation results saved to 'kfold_results.csv'")

    # Save overall test results
    overall_test_results_df = pd.DataFrame(overall_test_results)
    overall_test_results_df.to_csv('overall_test_results.csv', index=False)
    print("Overall test results saved to 'overall_test_results.csv'")

    # Generate confusion matrix
    generate_confusion_matrix(overall_test_results_df)

# Function to load data from paths
def LoadDataFromPaths(paths, labels, base_dir):
    data_x = []
    data_y = []
    for path, label in zip(paths, labels):
        try:
            full_path = os.path.join(base_dir, path)  # Construct the full path
            image = Image.open(full_path).convert('L')
            image = np.array(image)
            data_x.append(image)
            data_y.append(label)
        except FileNotFoundError:
            print(f"File not found: {full_path}")
            continue
    return np.array(data_x), np.array(data_y)

# Function to generate and save the confusion matrix
def generate_confusion_matrix(df):
    y_true = df['True Label'].map(label_mapping).values
    y_pred = df['Predicted Label'].map(label_mapping).values

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for K-Fold Validation')
    plt.savefig('confusion_matrix.png')
    plt.show()

# Normalization and transformations
normalize = transforms.Normalize(mean=[0.5], std=[0.5])
transform = transforms.Compose([
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

# CNN (Same model architecture as main model in PART 1 model_training.py)
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

# Early stopping training function (exact same code for early stopping)
def train_models(model, train_loader, val_loader, model_name):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    best_val_loss = float('inf')
    best_model_path = os.path.join("models", model_name, "best_model.pth")
    os.makedirs(os.path.join("models", model_name), exist_ok=True)
    patience_counter = 0
    limitHit = 0
    
    for epoch in range(num_epochs):  # Setting maximum epochs to 30
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
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss for {model_name} epoch[{epoch+1}] = {avg_loss_epoch:.4f}')
        
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

if __name__ == "__main__":
    kfold_cross_validation(CNN)
