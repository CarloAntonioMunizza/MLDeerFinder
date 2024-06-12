from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import glob
import tensorflow as tf

# Ensure TensorFlow is using the GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0]}")
else:
    print("No GPU found, using CPU.")

dims = 128

# Function to preprocess input image
def preprocess_image(image, target_size):
    try:
        image = cv2.resize(image, target_size)
        image = image / 255.0  # Normalize image to [0, 1]
        return np.expand_dims(image, axis=0)
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None

# Break image into 16x16 grid
def break_into_grid(image, grid_num=16):
    try:
        # Get dimensions
        if len(image.shape) == 4:
            _, height, width, _ = image.shape
        elif len(image.shape) == 3:
            height, width, _ = image.shape
            image = np.expand_dims(image, axis=0)
        else:
            raise ValueError("Unsupported image shape")
    except Exception as e:
        print(f"Error in getting image dimensions: {e}")
        return None


# Load the model
modelPath = 'Models/'
model = load_model(modelPath + 'deer_forest_model.keras')

input_folder = '/Users/Anthraxlemonaid/OneDrive/Desktop/Programming/Datasets/combined'
output_folder = '/Users/Anthraxlemonaid/OneDrive/Desktop/Programming/Datasets/result'
input_size = (dims, dims)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get list of all image files in the input folder
image_files = glob.glob(os.path.join(input_folder, '*.jpg'))

for image_file in image_files:
    try:
        # Load and preprocess image
        image = cv2.imread(image_file)
        if image is None:
            print(f"Failed to load image: {image_file}")
            continue  # Skip any unreadable images
        
        input_image = preprocess_image(image, input_size)
        if input_image is None:
            print(f"Failed to preprocess image: {image_file}")
            continue
        
        # Make predictions
        predictions = model.predict(input_image)

        # Since it's a binary classification, we threshold the predictions
        is_deer_present = predictions[0][0] > 0.5

        # Draw bounding box if deer is present
        if is_deer_present:
            image_grid = break_into_grid(image)
            if image_grid is None:
                print(f"Failed to break image into grid: {image_file}")
                continue
            
            cv2.rectangle(image, (10, 10), (image.shape[1]-10, image.shape[0]-10), (0, 255, 0), 2)

        # Save the resulting image
        base_filename = os.path.basename(image_file)
        output_path = os.path.join(output_folder, base_filename)
        if not cv2.imwrite(output_path, image):
            print(f"Failed to save image: {output_path}")
        else:
            print(f"Processed and saved: {output_path}")
    
    except Exception as e:
        print(f"Error processing image {image_file}: {e}")
