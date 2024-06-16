import os
import csv
import pandas as pd

#Define base directory and subdirectories
base_dir = "data"
data_sources_dir = os.path.join(base_dir, 'dataSources')
source_folders = ['fer2013', 'lfiw', 'kdef', 'teammates']
labels = ['happy', 'angry', 'neutral', 'engaged']

#Define the output CSV files
csv_file = os.path.join(base_dir, 'image_data.csv')
csv_segmentation_file = os.path.join(base_dir, 'image_data_segmentation.csv')
csv_shuffled_file = os.path.join(base_dir, 'image_data_shuffled.csv')
csv_cleaned_data = os.path.join(base_dir, 'cleaned_image_data.csv')

#Parameters for data segmentation
subset_counts = {
    'happy': 500, #Define number of images from each class to be used in all sets
    'angry': 500,
    'neutral': 500,
    'engaged': 500
}
train_percent = 0.7 #Define the relative proportions of images in each set
validation_percent = 0.15
test_percent = 0.15
seed = 2024 #Seed for the randomization

#Function to generate the composite name
def generate_composite_name(image_name, label, source):
    return f"{image_name}_{label}_{source}"

#Collect data from all sources
data = []
for source in source_folders:
    for label in labels:
        label_folder = os.path.join(data_sources_dir, source, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(label_folder, filename)
                    image_name = os.path.splitext(filename)[0]
                    composite_name = generate_composite_name(image_name, label, source.upper())
                    data.append((filename, filepath, label, source.upper(), composite_name))

#Write data to CSV
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Name', 'Path', 'Label', 'Source', 'CompositeName'])
    writer.writerows(data)

print(f'CSV file "{csv_file}" has been created with the following columns: Image Name, Path, Label, Source, CompositeName.')

#Create empty datasets
train_dataset = pd.DataFrame(columns=['Path', 'CompositeName', 'Label'])
validation_dataset = pd.DataFrame(columns=['Path', 'CompositeName', 'Label'])
test_dataset = pd.DataFrame(columns=['Path', 'CompositeName', 'Label'])

#Shuffle the data using seed
data_df = pd.read_csv(csv_cleaned_data)
shuffled_data = data_df.sample(frac=1, random_state=seed)
shuffled_data.to_csv(csv_shuffled_file, index=False)

#Calculate number of images for each set from each class
train_counts = {label: int(count * train_percent) for label, count in subset_counts.items()}
validation_counts = {label: int(count * validation_percent) for label, count in subset_counts.items()}
test_counts = {label: int(count * test_percent) for label, count in subset_counts.items()}

#Assign images from each class to train, validation, and test sets
for label, count in subset_counts.items():
    label_data = shuffled_data[shuffled_data['Label'] == label]
    train_count = train_counts[label]
    validation_count = validation_counts[label]
    test_count = test_counts[label]

    train_dataset = pd.concat([train_dataset, label_data[:train_count]], ignore_index=True)
    validation_dataset = pd.concat([validation_dataset, label_data[train_count:train_count + validation_count]], ignore_index=True)
    test_dataset = pd.concat([test_dataset, label_data[train_count + validation_count:train_count + validation_count + test_count]], ignore_index=True)

#Write the train, validation and test datasets to CSV
train_dataset.to_csv(os.path.join(base_dir, 'train_dataset.csv'), index=False)
validation_dataset.to_csv(os.path.join(base_dir, 'validation_dataset.csv'), index=False)
test_dataset.to_csv(os.path.join(base_dir, 'test_dataset.csv'), index=False)

#Calculate total counts for each class
total_counts = {
    label: len(shuffled_data[shuffled_data['Label'] == label]) for label in labels
}

#Calculate total counts
total_count = sum(total_counts.values())

#Print the number of images from each class in each set
print("Distribution of images in each set:")
for label in labels:
    train_count = len(train_dataset[train_dataset['Label'] == label])
    validation_count = len(validation_dataset[validation_dataset['Label'] == label])
    test_count = len(test_dataset[test_dataset['Label'] == label])

    print(f"{label.capitalize()} images:")
    print(f"  Train: {train_count} images ({train_count/total_counts[label]:.2%} of total {label} images)")
    print(f"  Validation: {validation_count} images ({validation_count/total_counts[label]:.2%} of total {label} images)")
    print(f"  Test: {test_count} images ({test_count/total_counts[label]:.2%} of total {label} images)")