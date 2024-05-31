# SmartClass A.I.ssistant - PART 1

Welcome to the SmartClass A.I.ssistant project! This project is part of COMP 472 Artificial Intelligence (Summer 2024) and aims to develop a deep learning Convolutional Neural Network (CNN) using PyTorch to analyze images of students in a classroom or online meeting setting and categorize them into distinct states or activities.

## Team composition
Hao Yi Liu
Carson Senthilkumar
Qing Han Zhang

Supervised by teaching assistant Mr. Amin Karimi

## Project Overview

### PART 1: Data Collection, Cleaning, Labeling & Preliminary Analysis
In this part, we focus on:
- Collecting suitable training data
- Cleaning and labeling the dataset
- Visualizing the dataset to improve the cleaning methods and diagnose any issues with the dataset

## Table of Contents
Part 1: Contains all folders and .py scripts.
- data: Folder that contains all data images and .csv files
  - dataSources: Folder that contains the original images split into their sources and classes.
    - fer2013: Images from the FER-2013 dataset.
      - Angry, Happy, Neutral folders containing images from their respective class. Empty Engaged folder (no images from this source in this class)
    - kdef: Images from the KDEF dataset.
      - Engaged folder containing images for this class. Angry, Happy, Neutral folders are empty (no images from this source in these classes.)
    - lfiw: Images from the Labeled Faces In the Wild dataset.
      - Engaged folder containing images for this class. Angry, Happy, Neutral folders are empty (no images from this source in these classes.)
    - teammates: Images taken by each team member.
      - Angry, Happy, Neutral and Engaged folders containing images from their respective class.
  - cleanedData: Folder that contains the images after data cleaning methods were applied, split into the 4 classes.
    - Angry, Happy, Neutral and Engaged folders containing images from their respective class.
  - image_data.csv: CSV that stores the image data from the entire dataset from dataSources (Image Name, Path, Label, Source, CompositeName)
  - image_data_shuffled: CSV that stores the same data as image_data.csv, but in a shuffled order (can be reshuffled with dataset_segmentation.py).
  - cleaned_image_data.csv: CSV that stores the data of the CLEANED images from the entire dataset from cleanedData (Image Name, Path, Label, Source, CompositeName)
  - train_dataset.csv: CSV that stores the data of the images assigned to the training dataset (Path, CompositeName, Label, Image Name, Source)
  - validation_dataset.csv: CSV that stores the data of the images assigned to the validation dataset (Path, CompositeName, Label, Image Name, Source)
  - test_dataset.csv: CSV that stores the data of the images assigned to the testing dataset (Path, CompositeName, Label, Image Name, Source)
- data_cleaning.py: Script that cleans the original data according to cleaning techniques, which are easily modifiable.
- dataset_segmentation.py: Script that allows random selection of a specified number of images within the original data, for each class, and splits them into 3 datasets (training, validation, testing) to train, validate and test our classification algorithm. Number of images, proportion of images in each split, and randomization seed can be modified by the user. The script also creates a shuffled list of all images.
- data_visualization.py: Script to visualize the data in many ways: plot the class distribution of images, plot the pixel intensity distributions within each class, sample 15 random images from each class with their corresponding pixel intensity distribution.
- data_augmentation.py: Generate augmented images from the training dataset (created by dataset_segmentation.py), using augmentation methods in a random combination (flipping, rotation, color, brightness, contrast, sharpness enhancements).
- README.md: Read me file.

## Steps to execute code
### Part 1
a) Data cleaning:
1. First, make sure PART_1/data/dataSources contains images in their respective folders. These will be the ones to be cleaned into the PART_1/data/cleanedData folder.
2. Open data_cleaning.py.
3. In the process_image function, locate the various cleaning methods implemented. Comment in/out any method that you wish to apply or not on the original images (Grayscaling, Resizing, Histogram equalization, CLAHE, Blurring).
4. Run data_cleaning.py.
5. Cleaned images will now be in the cleanedData folder, and cleaned_image_data.csv will contain the corresponding data.

b) Data visualization (with Matplotlib):
1. First, make sure the PART_1/data/cleanedData folder contains images (the script has been configured to visualize post-cleaning data).
2. Open data_visualization.py.
3. At the end of the script, comment in/out any execution of the plot_class_distribution, plot_sample_images, plot_pixel_intensity_distribution functions that you wish to run or not.
4. Run data_visualization.py. Plots/sample images will pop-up sequentially. In order to view the next plot/image sample, the current pop-up must be closed to allow the program to continue to run. Plots and sample images can be saved.
