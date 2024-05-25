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
1. Create a virtual environment:
    ```bash
    python -m venv venv
    ```
2. Activate the virtual environment:
    - **Windows:**
      ```bash
      venv\Scripts\activate
      ```
    - **Linux/macOS:**
      ```bash
      source venv/bin/activate
      ```
3. Install the required packages:
    ```bash
    pip install -r PART_1/requirements.txt
    ```

### 2. Download and Prepare the Dataset
Run the `data_preparation.py` script to download and prepare the dataset:
```bash
python PART_1/data_preparation.py
