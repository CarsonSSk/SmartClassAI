import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np

#Data importation section
#base_dir = "Dataset"
#train_dir = os.path.join(base_dir, 'train')
#test_dir = os.path.join(base_dir, 'test')
base_dir="data"
data=pd.read_csv(os.path.join(base_dir,"image_data.csv"))
shuffled_data = pd.read_csv(os.path.join(base_dir, "image_data_shuffled.csv"))


# #Function to get image paths and labels, creates data array in the format (Path string, emotion label string)
# def get_image_paths_and_labels(base_folder):
#     data = []
#     for label in os.listdir(base_folder):
#         label_folder = os.path.join(base_folder, label)
#         if os.path.isdir(label_folder):
#             for filename in os.listdir(label_folder):
#                 if filename.endswith('.jpg'):
#                     filepath = os.path.join(label_folder, filename)
#                     data.append((filepath, label))
#     #print(data)
#     return data
#
# #Get the train and test data + Combination of all data
# train_data = get_image_paths_and_labels(train_dir)
# test_data = get_image_paths_and_labels(test_dir)
# all_data = train_data + test_data
#
# #Write data array to CSV
# csv_file = 'image_paths_and_labels.csv'
# with open(csv_file, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['file_path', 'label'])
#     writer.writerows(all_data)
#
# print(f'CSV file "{csv_file}" has been created.')

#Data visualization section

#Load dataset from the existing CSV file (all data)
#data = pd.read_csv('image_paths_and_labels.csv')

#Function to plot class (labels) distrirbution in dataset
def plot_class_distribution(data):
    plt.figure(figsize=(10, 6))

    class_counts = data['Label'].value_counts()
    bars = plt.bar(class_counts.index, class_counts.values)

    #Add title and labels
    plt.title('Class Distribution - Entire Dataset')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')

    #Annotate each bar with the count as an integer on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, int(yval), ha='center', va='bottom', fontsize=12,
                 color='black')

    plt.show()

#Run the plot_class_distribution function on all data
#plot_class_distribution(data)

#Function to plot n sample images of the class_name label within data, in a 3 by 5 grid with pixel intensity distribution (grayscale)
def plot_sample_images(data, class_name, image_column='Path', n_samples=15):

    sample_data = data[data['Label'] == class_name].sample(n_samples)
    sample_data = sample_data.sample(frac=1)  #Shuffle the samples

    plt.figure(figsize=(15, 10))

    for i, (index, row) in enumerate(sample_data.iterrows()):
        img = Image.open(row[image_column]).convert('L')  #Converts to grayscale, Change if we want to visualize RGB images

        #Plot image
        plt.subplot(3, 10, i*2 + 1)
        plt.imshow(img, cmap='gray', aspect='equal')  #Display as grayscale
        plt.axis('off')

        #Plot histogram
        plt.subplot(3, 10, i*2 + 2)
        plt.hist(np.array(img).flatten(), bins=256, range=(0, 256), density=True, color='gray', alpha=0.75)
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.gca().set_aspect('auto')

    plt.suptitle(f'Sample Images - {class_name}', size=16)
    plt.tight_layout()
    plt.show()

#Run the plot_sample_images function with a specific class_name
#plot_sample_images(data, 'angry', image_column='Path')
#plot_sample_images(data, 'happy', image_column='Path')
#plot_sample_images(data, 'neutral', image_column='Path')
#plot_sample_images(data, 'engaged', image_column='Path')

#Function to plot n sample images within all the data, in a 3 by 5 grid with pixel intensity distribution (grayscale)
def plot_random_images(data, image_column='Path', n_samples=15):
    shuffled_data = data.sample(n=n_samples).reset_index(drop=True)

    plt.figure(figsize=(15, 10))

    for i, (_, row) in enumerate(shuffled_data.iterrows()):
        img = Image.open(row[image_column]).convert('L')  # Convert to grayscale

        # Add label as annotation
        plt.subplot(3, 10, i * 2 + 1)
        plt.text(0.5, -0.15, row['Label'], fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)

        # Plot image
        plt.subplot(3, 10, i * 2 + 1)
        plt.imshow(img, cmap='gray', aspect='equal')  # Display as grayscale
        plt.axis('off')

        # Plot histogram
        plt.subplot(3, 10, i * 2 + 2)
        plt.hist(np.array(img).flatten(), bins=256, range=(0, 256), density=True, color='gray', alpha=0.75)
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.gca().set_aspect('auto')

    plt.suptitle('Sample Images - All Data', size=16)
    plt.tight_layout()
    plt.show()
#Test the function
#plot_random_images(data)

#Function to plot pixel intensity distribution per class
def plot_pixel_intensity_distribution(data, class_name, image_column='Path'):

    class_data = data[data['Label'] == class_name]
    pixel_intensities = []

    for i, (index, row) in enumerate(class_data.iterrows()):
        img = Image.open(row[image_column]).convert('L')  #Convert to grayscale
        pixel_values = np.array(img).flatten()
        pixel_intensities.extend(pixel_values)
        print(i)

    plt.figure(figsize=(10, 5))
    plt.hist(pixel_intensities, bins=256, range=(0, 256), density=True, color='gray', alpha=0.75)
    plt.title(f'Pixel Intensity Distribution - {class_name} - {len(class_data)} images', size=16)
    plt.xlabel('Pixel Intensity')
    plt.xlim(left=0, right=255)
    plt.ylabel('Frequency')
    plt.show()

#Test the plot_pixel_intensity_distribution function for a specific class
#plot_pixel_intensity_distribution(data, 'engaged', image_column='Path')
#plot_pixel_intensity_distribution(data, 'happy', image_column='Path')
#plot_pixel_intensity_distribution(data, 'neutral', image_column='Path')
#plot_pixel_intensity_distribution(data, 'angry', image_column='Path')

#Function to plot pixel intensity distribution for RGB images - not useable because no RGB images in our dataset currently
def plot_rgb_pixel_intensity_distribution(data, class_name, image_column='Path'):

    class_data = data[data['Label'] == class_name]
    red_intensities = []
    green_intensities = []
    blue_intensities = []
    RGB_count = 0

    for i, (index, row) in enumerate(class_data.iterrows()):
        img = Image.open(row[image_column])
        img_array = np.array(img)

        #Ensure image is in RGB mode
        if img_array.ndim == 3 and img_array.shape[2] == 3:
            red_intensities.extend(img_array[:, :, 0].flatten())
            green_intensities.extend(img_array[:, :, 1].flatten())
            blue_intensities.extend(img_array[:, :, 2].flatten())
            RGB_count = RGB_count + 1
            print(f"RGB: {i}")
        else:
            print(f"Skipping non-RGB image: {row[image_column]}")

    plt.figure(figsize=(10, 5))

    plt.hist(red_intensities, bins=256, range=(0, 256), density=True, color='red', alpha=0.5, label='Red Channel')
    plt.hist(green_intensities, bins=256, range=(0, 256), density=True, color='green', alpha=0.5, label='Green Channel')
    plt.hist(blue_intensities, bins=256, range=(0, 256), density=True, color='blue', alpha=0.5, label='Blue Channel')

    plt.title(f'RGB Pixel Intensity Distribution - {class_name} - {RGB_count} RGB images / {len(class_data)} total images', size=16)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim(left=0, right=255)
    plt.legend()
    plt.show()


#Test the function (may not work because no RGB data currently)
#plot_rgb_pixel_intensity_distribution(data, 'angry', image_column='Path')
#plot_rgb_pixel_intensity_distribution(data, 'happy', image_column='Path')
#plot_rgb_pixel_intensity_distribution(data, 'neutral', image_column='Path')
plot_rgb_pixel_intensity_distribution(data, 'engaged', image_column='Path')



#plot_rgb_pixel_intensity_distribution(data, 'disgust', image_column='file_path')