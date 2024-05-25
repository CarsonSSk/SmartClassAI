# SmartClass A.I.ssistant - PART 1

Welcome to the SmartClass A.I.ssistant project! This project is part of COMP 472 Artificial Intelligence (Summer 2024) and aims to develop a deep learning Convolutional Neural Network (CNN) using PyTorch to analyze images of students in a classroom or online meeting setting and categorize them into distinct states or activities.

## Project Overview

### PART 1: Data Collection, Cleaning, Labeling & Preliminary Analysis
In this part, we focus on:
- Collecting suitable training data
- Performing exploratory data analysis (EDA)
- Cleaning and labeling the dataset

## How to Use

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
      source venv/Scripts/activate
      ```
    - **Linux/macOS Terminal:**
      ```bash
      source venv/bin/activate
      ```

4. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### 2. Download and Prepare the Dataset
1. **Run the `data_prep.py` script to download, extract, and organize the dataset:**
    ```bash
    python data_prep.py
    ```

### 3. Script Explanation
The `data_prep.py` script performs the following steps:
- Ensures the `data/fer2013` directory exists.
- Downloads the FER-2013 dataset using the Kaggle API with the `--force` option.
- Extracts the dataset if not already extracted.
- Defines the required classes (`happy`, `angry`, `neutral`) and corresponding subdirectories.
- Creates directories for organized data.
- Removes existing directories if they exist.
- Copies images for each subset (`train`, `test`) and each emotion to the organized directory, ensuring that exactly 400 images per class are used for training and 100 images per class for testing.
- Removes the original data directories to save space.

### 4. Directory Structure
After running the script, the directory structure will look like this:
COMP472-Project/
├── PART_1/
│ ├── data/
│ │ ├── fer2013/
│ │ │ ├── fer2013.zip
│ │ │ └── organized/
│ │ │ ├── train/
│ │ │ │ ├── angry/
│ │ │ │ ├── happy/
│ │ │ │ └── neutral/
│ │ │ └── test/
│ │ │ ├── angry/
│ │ │ ├── happy/
│ │ │ └── neutral/
│ ├── venv/
│ │ ├── Include/
│ │ ├── Lib/
│ │ ├── Scripts/
│ │ ├── pyvenv.cfg
│ │ └── ...
│ ├── requirements.txt
│ ├── data_preparation.py
│ └── README.md
├── .gitignore
└── LICENSE
