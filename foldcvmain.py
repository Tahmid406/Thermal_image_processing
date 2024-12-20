import os
import shutil
import random
import numpy as np
from tqdm import tqdm  # Progress bar library

# Paths to the thermal and mask images
thermal_image_path = r'P:\Thesis\Dataset\PV cell\thermal_projection\output\thermal_images'
mask_image_path = r'P:\Thesis\Dataset\PV cell\thermal_projection\output\mask_images'

# Output folder for the five-fold cross-validation sets
output_folder = r'P:\Thesis\Dataset\PV cell\thermal_projection\output\Data'

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Create subdirectories for each fold (training, validation, test)
folders = ['Train', 'Val', 'Test']
for folder in folders:
    folder_path = os.path.join(output_folder, folder)
    os.makedirs(folder_path, exist_ok=True)
    for fold in range(1, 6):
        fold_dir = os.path.join(folder_path, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'masks'), exist_ok=True)

# Get all image filenames from the thermal and mask image directories
thermal_images = os.listdir(thermal_image_path)
mask_images = os.listdir(mask_image_path)

# Ensure that the number of thermal and mask images match
assert len(thermal_images) == len(mask_images), "Number of thermal images and mask images do not match!"

# Shuffle the images randomly to ensure a good split
combined = list(zip(thermal_images, mask_images))
random.shuffle(combined)
thermal_images, mask_images = zip(*combined)

# Calculate the number of images for each split
total_images = len(thermal_images)
train_size = int(total_images * 0.8)
val_size = int(total_images * 0.1)
test_size = total_images - train_size - val_size

# Split the data into 5 folds and copy files
for fold in range(5):
    # Determine the indices for each fold's splits
    start_val_idx = fold * val_size
    end_val_idx = start_val_idx + val_size
    start_test_idx = end_val_idx
    end_test_idx = start_test_idx + test_size
    
    # Create training, validation, and test sets for the current fold
    val_images = thermal_images[start_val_idx:end_val_idx]
    val_masks = mask_images[start_val_idx:end_val_idx]
    
    test_images = thermal_images[start_test_idx:end_test_idx]
    test_masks = mask_images[start_test_idx:end_test_idx]
    
    train_images = thermal_images[:start_val_idx] + thermal_images[end_test_idx:]
    train_masks = mask_images[:start_val_idx] + mask_images[end_test_idx:]
    
    # Copy the images to their respective folders for the current fold with a progress bar
    for image, mask in tqdm(zip(train_images, train_masks), total=len(train_images), desc=f"Processing Fold {fold+1} - Train"):
        shutil.copy(os.path.join(thermal_image_path, image), os.path.join(output_folder, 'Train', f'fold_{fold+1}', 'images', image))
        shutil.copy(os.path.join(mask_image_path, mask), os.path.join(output_folder, 'Train', f'fold_{fold+1}', 'masks', mask))

    for image, mask in tqdm(zip(val_images, val_masks), total=len(val_images), desc=f"Processing Fold {fold+1} - Val"):
        shutil.copy(os.path.join(thermal_image_path, image), os.path.join(output_folder, 'Val', f'fold_{fold+1}', 'images', image))
        shutil.copy(os.path.join(mask_image_path, mask), os.path.join(output_folder, 'Val', f'fold_{fold+1}', 'masks', mask))

    for image, mask in tqdm(zip(test_images, test_masks), total=len(test_images), desc=f"Processing Fold {fold+1} - Test"):
        shutil.copy(os.path.join(thermal_image_path, image), os.path.join(output_folder, 'Test', f'fold_{fold+1}', 'images', image))
        shutil.copy(os.path.join(mask_image_path, mask), os.path.join(output_folder, 'Test', f'fold_{fold+1}', 'masks', mask))

print("Five-fold cross-validation dataset creation complete!")
