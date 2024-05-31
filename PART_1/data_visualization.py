import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

#Data importation section
base_dir="data"
data=pd.read_csv(os.path.join(base_dir,"cleaned_image_data.csv"))
shuffled_data = pd.read_csv(os.path.join(base_dir, "image_data_shuffled.csv"))

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

#Function to plot n sample images of the class_name label within data, in a 3 by 5 grid with pixel intensity distributions next to each image
def plot_sample_images(data, class_name, image_column='Path', n_samples=15):
    sample_data = data[data['Label'] == class_name].sample(n_samples)
    sample_data = sample_data.sample(frac=1)  #Shuffle the samples

    plt.figure(figsize=(20, 15))

    for i, (index, row) in enumerate(sample_data.iterrows()):
        img = Image.open(row[image_column])  #Convert to grayscale

        #Plot image
        plt.subplot(5, 6, i * 2 + 1)
        plt.imshow(img, cmap='gray', aspect='equal')
        plt.axis('off')

        #Plot histogram
        plt.subplot(5, 6, i * 2 + 2)
        plt.hist(np.array(img).flatten(), bins=256, range=(0, 256), density=True, color='gray', alpha=0.75)
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.gca().set_aspect('auto')
        plt.xlim(left=0, right=255)

    #Add title + layout adjustments
    plt.suptitle(f'Sample Images - {class_name}', size=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.5, bottom=0.06)
    plt.show()

#Function to plot pixel intensity distribution per class
def plot_pixel_intensity_distribution(data, class_name, image_column='Path'):

    class_data = data[data['Label'] == class_name]
    pixel_intensities = []

    for i, (index, row) in enumerate(class_data.iterrows()): #Iterate through images from the desired class to add pixel intensity values to pixel_intensity array
        img = Image.open(row[image_column]).convert('L')  #Convert to grayscale
        pixel_values = np.array(img).flatten()
        pixel_intensities.extend(pixel_values)

    #Plot the histogram
    plt.figure(figsize=(10, 5))
    plt.hist(pixel_intensities, bins=256, range=(0, 256), density=True, color='gray', alpha=0.75)
    plt.title(f'Pixel Intensity Distribution - {class_name} - {len(class_data)} images', size=16)
    plt.xlabel('Pixel Intensity')
    plt.xlim(left=0, right=255)
    plt.ylabel('Frequency')
    plt.show()

#Run the plot_class_distribution function on all data
plot_class_distribution(data)

#Run the plot_sample_images function for each class
plot_sample_images(data, 'angry', image_column='Path')
plot_sample_images(data, 'happy', image_column='Path')
plot_sample_images(data, 'neutral', image_column='Path')
plot_sample_images(data, 'engaged', image_column='Path')

#Run the plot_pixel_intensity_distribution function for a specific class
plot_pixel_intensity_distribution(data, 'engaged', image_column='Path')
plot_pixel_intensity_distribution(data, 'happy', image_column='Path')
plot_pixel_intensity_distribution(data, 'neutral', image_column='Path')
plot_pixel_intensity_distribution(data, 'angry', image_column='Path')