# SmartClass A.I.ssistant - PART 1+2

Welcome to the SmartClass A.I.ssistant project! This project is part of COMP 472 Artificial Intelligence (Summer 2024) and aims to develop a deep learning Convolutional Neural Network (CNN) using PyTorch to analyze images of students in a classroom or online meeting setting and categorize them into distinct states or activities.

## Team composition

- Hao Yi Liu
- Carson Senthilkumar
- Qing Han Zhang

Supervised by teaching assistant Mr. Amin Karimi

## Project Overview

### PART 1: Data Collection, Cleaning, Labeling & Preliminary Analysis

In this part, we focus on:

- Collecting suitable training data
- Cleaning and labeling the dataset
- Visualizing the dataset to improve the cleaning methods and diagnose any issues with the dataset

### PART 2: Deep Learning Model Development and Evaluation

In this part, we focus on:

- Creating an AI capable of analyzing facial images for classification using PyTorch
- Design and train this Convolutional Neural Network on the classes as outlined in Part I
- Generate a thorough evaluation of our model

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
  - models: Directory that contains all models. Inside each of the subfolders which are named after the respective models, you will find: the model paramters file (.pth), CSV files containing the results of that model on the testing dataset, CSV containing the validation loss from the training process, CSV containing the accuracy of the model.
  - runFiles: Directory containing images that you wish to run a specific model on (single image + batch of images are possible) using run_model.py.
    - imagesDirectory: Directory where you place the images you want to apply the specific model on.
    - runCleanedImages: Directory that contains the cleaned version of the images you wish to predict. The cleaned images will be generated automatically based on the same methodology used to clean the training/testing image data.
    - directory_images_data.csv: CSV file containing the predictions for all the images in the batch that you last ran the run_model.py script on.
  - image_data.csv: CSV that stores the image data from the entire dataset from dataSources (Image Name, Path, Label, Source, CompositeName)
  - image_data_shuffled: CSV that stores the same data as image_data.csv, but in a shuffled order (can be reshuffled with dataset_segmentation.py).
  - cleaned_image_data.csv: CSV that stores the data of the CLEANED images from the entire dataset from cleanedData (Image Name, Path, Label, Source, CompositeName)
  - train_dataset.csv: CSV that stores the data of the images assigned to the training dataset (Path, CompositeName, Label, Image Name, Source)
  - validation_dataset.csv: CSV that stores the data of the images assigned to the validation dataset (Path, CompositeName, Label, Image Name, Source)
  - test_dataset.csv: CSV that stores the data of the images assigned to the testing dataset (Path, CompositeName, Label, Image Name, Source)
  - kfold_dataset.csv: CSV that stores all the data used for training and testing our final model in Part II. (holds 2000 images)
- data_cleaning.py: Script that cleans the original data according to cleaning techniques, which are easily modifiable.
- dataset_segmentation.py: Script that allows random selection of a specified number of images within the original data, for each class, and splits them into 3 datasets (training, validation, testing) to train, validate and test our classification algorithm. Number of images, proportion of images in each split, and randomization seed can be modified by the user. The script also creates a shuffled list of all images.
- data_visualization.py: Script to visualize the data in many ways: plot the class distribution of images, plot the pixel intensity distributions within each class, sample 15 random images from each class with their corresponding pixel intensity distribution.
- bias_analysis_results.csv:
- bias_analysis.py:
- combined_images_train_validation_test.csv:
- create_bias_mitigation_dataset.py:
- labeled_images_with_gender_age_test.csv:
- labeled_images_with_gender_age_train.csv:
- labeled_images_with_gender_age_validation.csv:
- labeled_images_with_gender_age.csv:
- unbiased_model_evaluation.py:
- unbiased_model_training.py:
- with_biasMitigation_combined_images_train_validation_test.csv:

PART_3:
-labeled_data: 
  -Man:
    -middle age:
    -old:
    -young:
  -Woman:
    -middle age:
    -old:
    -young:
-Part2_kfold_models:
  -model_fold_1:
  -models 1-10
-Part3_kfold_models:
  -model_fold_1:
  -models 1-10
-kfold_dataset.py: Creates the combined dataset with all the data we used for training and testing our models in Part II. Creates the kfold_dataset.csv.
-kfold_script.py: Same code as our model_training.py but only has the main model implemented and has the k-fold validation incorporated within the training loop.
-labeled_images.csv:
- README.md: Read me file.

## Start up dependencies

### 1. Set Up the Environment Using `venv`

1. **Navigate to the project directory:**

   ```bash
   ex.: cd ~/Desktop/COMP472-Project/PART_1
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - **Windows Command Prompt:**
     ```bash
     venv\Scripts\activate
     ```
   - **Windows PowerShell:**
     ```bash
     .\venv\Scripts\Activate
     ```
   - **Git Bash or WSL:**
     ```bash
     sourceData venv/Scripts/activate
     ```
   - **Linux/macOS Terminal:**
     ```bash
     sourceData venv/bin/activate
     ```
   - **If `cannot be loaded because running scripts is disabled on this system`:**
     Try `Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser`
4. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

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

### Part 2 (you can find our final models and some of our sample models here: https://drive.google.com/drive/folders/1sjcfjRKqo8a35taRH3w1eqJgEFPJmiDV)
c) Model training (with PyTorch):

1. First, make sure the PART_1/data/cleanedData folder contains images (the script has been configured to train on the cleaned data).
2. Open model_training.py.
3. At the beginning of the script, establish the parameters for the training: num_epochs (minimum number of epochs, the maximum has been set to 30), learning rate (set to 0.005), patience parameter (how many epochs without decreasing validation loss before the training stops). Possibility to set a random seed if you want to obtain consistent models each run (same results). Set "reproducibility" to false if you want new results on each run of the script. Note: Reproducibility only works when run on the same machine, with the same Python version. Running on two different machines will guarantee DIFFERENT results.
4. Run model_training.py. Wait for the models to be trained (this step can take up to 15 minutes). Validation losses, epochs, accuracies and model names will be displayed in the output. At the end, there will be pop-ups of sample images from the testing dataset with predictions and their respective correct labels, for each of the three models trained (main, variant 1, variant 2). Each pop-up has to be closed for the next one to show up.
5. When the script is finished running, all 3 models will be saved in a unique directory in the models folder, as well as their respective prediction results, accuracies, and validation losses, all in csv format. The model is saved as well in .pth format.

d) Model evaluation (with Matplotlib and Sklearn)

1. First, make sure the PART_1/models contains the models you wish to evaluate, with their respective results.csv files (the script has been configured to generate the evaluations from the results csv which is generated when training the model, and then testing it on the testing dataset, all which is done when running model_training.py).
2. Open model_evaluation.py.
3. At the beginning, establish the model names of the models that you wish to evaluate. You can simply copy-paste the folder names from the models directory. The model folder must contain the results.csv of the model you wish to evaluate.
4. Run model_evaluation.py.
5. There will be pop-ups with the confusion matrices of each evaluated model, sequentially (close a pop-up to move on to the next).  Evaluation metrics for each model will be printed in the output, and saved in model_evaluation_summary.csv, located in the main directory.

e) Run/apply model on custom image / batch of images

1. First, open the "runFiles" directory in the main directory. Place the single image you wish to predict in this directory, and place the batch of images that you wish to predict in the "imagesDirectory" subdirectory. For the best results, the images must be: square format (will be resized to 48x48 so distortion may happen); can be RGB or grayscale; ideally, as much background should have been cropped out of the image manually in order to center the image on the face.
2. Open run_model.py.
3. At the beginning, set the paths to the images you wish to predict: image_path for the single image; directory_path for the batch of images. Set the path to the model you wish to use as well as the appropriate model type (0 = main model, 1 = variant 1, 2 = variant 2).
4. Run run_model.py.
5. In the output, you will see the prediction of the model for each image's facial emotion. You can check if it is right or wrong! The results for the batch of images will also be saved in a csv file: runFiles/directory_images_data.csv, which contains each of the batch's image name, path, and predicted label. This allows for further analysis of the predictions if desired by the user.
