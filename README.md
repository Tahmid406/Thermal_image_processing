# Project: Thermal Image Preprocessing and Dataset Creation

## Overview

This project contains scripts for preprocessing thermal images and masks, performing five-fold cross-validation splits, and preparing datasets for machine learning workflows. The key features include:

1. **Five-fold cross-validation dataset creation** using thermal and mask images.
2. **Image preprocessing** to standardize data for training models.

## Files in this Repository

### 1. `foldcvmain.py`

This script creates five-fold cross-validation datasets. It:

- Splits thermal images and corresponding masks into training, validation, and test sets.
- Ensures proper folder structure for each fold.
- Randomizes the data to ensure an unbiased split.

**Usage**:

- Ensure you have thermal and mask images in the specified directories.
- Update the paths in the script to match your data locations.
- Run the script to generate the dataset.

### 3. `read_image.ipynb`

This Jupyter Notebook focuses on reading and visualizing images and masks. It is useful for:

- Checking the integrity of images and masks.
- Displaying samples to validate preprocessing steps.

**Usage**:

- Open the notebook in JupyterLab or any compatible environment.
- Update file paths to point to your dataset.
- Execute the cells to visualize the data.

## Folder Structure

Ensure the following folder structure for the scripts to work correctly:

```
Project Root
├── foldcvmain.py
├── read_image.ipynb
├── PHOTOVOLTAIC THERMAL IMAGES DATASET/
├── output/
└── combined_images/
```

## How to Use

### Step 1: Install Dependencies

Ensure you have Python 3.7 or later installed. Use the following command to install required packages:

```bash
pip install tqdm numpy jupyterlab
```

### Step 2: Update File Paths

- Modify file paths in `foldcvmain.py`, `image_preprocess.ipynb`, and `read_image.ipynb` to match your dataset locations.

### Step 3: Run Scripts

- Run `foldcvmain.py` to create cross-validation datasets:
  ```bash
  python foldcvmain.py
  ```
- Use the Jupyter Notebooks for preprocessing and visualization as needed.
