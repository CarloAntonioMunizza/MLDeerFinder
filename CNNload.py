from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import glob

# Function to preprocess input image
def preprocess_image(image, target_size):
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize image to [0, 1]
    return np.expand_dims(image, axis=0)

# Break image into 16x16 grid
def break_into_grid(image, grid_num = 16):
    
    # get dimensions
    if len(image.shape) == 4:
        image_num, height, width, channels = image.shape
    elif len(image.shape) == 3:
        height, width, channels = image.shape
        image = np.expand_dims(image, axis=0)
    else:
        raise ValueError("Unsupported image shape")
    
    grid_height = height // grid_num
    grid_width = width // grid_num
    
    
    divided_image = []
    
    # slice image into grids
    for x in range(0, height, grid_height):
        row = []
        for y in range(0, width, grid_width):
            grid = image[:, x : x+grid_height, y : y+grid_width, :]
            row.append(grid)
        divided_image.append(row)
    
    for row in divided_image:
        for grid in row:
            # Calculate center coords     
    
               
            center_cord = (grid_height // 2, grid_width // 2)
            print(center_cord)
            
            cv2.circle(grid[0], center_cord, 2, (128, 128, 128), -1)
    
    return divided_image

model = load_model('deer_forest_model.h5')
        
input_folder = '/Users/Anthraxlemonaid/OneDrive/Desktop/Programming/Datasets/dump'
output_folder = '/Users/Anthraxlemonaid/OneDrive/Desktop/Programming/Datasets/result'
input_size = (64, 64)

if not os.path.exists(output_folder):
        os.makedirs(output_folder)

# Get list of all image files in the input folder
image_files = glob.glob(os.path.join(input_folder, '*.jpg'))

for image_file in image_files:
    
    # Load and preprocess image
    image = cv2.imread(image_file)
    if image is None:
        continue  # Skip any unreadable images
    
    input_image = preprocess_image(image, input_size)

    image_grid = break_into_grid(input_image)
    
    # Make predictions
    predictions = model.predict(input_image)

    # Since it's a binary classification, we threshold the predictions
    is_deer_present = predictions[0][0] > 0.5

    # Draw bounding box if deer is present
    if is_deer_present:
        image_grid = break_into_grid(image)
        
        cv2.rectangle(image, (10, 10), (image.shape[1], image.shape[0]), (0, 255, 0), 2)

    # Save the resulting image
    base_filename = os.path.basename(image_file)
    output_path = os.path.join(output_folder, base_filename)
    cv2.imwrite(output_path, image)
    print(f"Processed and saved: {output_path}")
    
