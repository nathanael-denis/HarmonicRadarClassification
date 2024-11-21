import os
import random
import shutil

def split_dataset(input_dir, output_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Splits the dataset into train, validation, and test sets without balancing classes.

    Parameters:
    - input_dir (str): Path to the input dataset directory.
    - output_dir (str): Path to the output directory where splits will be saved.
    - train_ratio (float): Proportion of data to be used for training.
    - val_ratio (float): Proportion of data to be used for validation.
    - test_ratio (float): Proportion of data to be used for testing.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory at: {output_dir}")

    # List all class directories
    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    if not classes:
        print("No class directories found in the input directory.")
        return

    print(f"Found classes: {classes}")

    # Create output directories for splits and classes
    for split in ['train', 'val', 'test']:
        for cls in classes:
            split_cls_dir = os.path.join(output_dir, split, cls)
            os.makedirs(split_cls_dir, exist_ok=True)
            print(f"Ensured directory exists: {split_cls_dir}")

    # Process each class
    for cls in classes:
        cls_dir = os.path.join(input_dir, cls)
        images = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]

        # Shuffle the images to ensure random selection
        random.shuffle(images)

        # Split the images into train, val, test
        total_images = len(images)
        train_split = int(total_images * train_ratio)
        val_split = int(total_images * val_ratio) + train_split

        train_files = images[:train_split]
        val_files = images[train_split:val_split]
        test_files = images[val_split:]

        # Define output directories
        train_dir = os.path.join(output_dir, 'train', cls)
        val_dir = os.path.join(output_dir, 'val', cls)
        test_dir = os.path.join(output_dir, 'test', cls)

        # Move files to the respective directories
        for file in train_files:
            shutil.move(os.path.join(cls_dir, file), os.path.join(train_dir, file))
        for file in val_files:
            shutil.move(os.path.join(cls_dir, file), os.path.join(val_dir, file))
        for file in test_files:
            shutil.move(os.path.join(cls_dir, file), os.path.join(test_dir, file))

        print(f"Processed Class '{cls}': {len(train_files)} for train, {len(val_files)} for val, {len(test_files)} for test.")

    print("Dataset splitting completed successfully.")

# Example usage using current directory
current_dir = os.getcwd()
input_dir = os.path.join(current_dir, 'images')
output_dir = os.path.join(current_dir, 'output')

split_dataset(input_dir=input_dir, output_dir=output_dir)