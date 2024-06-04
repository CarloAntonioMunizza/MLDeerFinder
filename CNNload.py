from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import glob

model = load_model('deer_forest_model.h5')

# Function to preprocess input image
def preprocess_image(image, target_size):
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize image to [0, 1]
    return np.expand_dims(image, axis=0)

# Function to process and save images
def process_and_save_images(input_folder, output_folder, input_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of all image files in the input folder
    image_files = glob.glob(os.path.join(input_folder, '*.*'))
    for image_file in image_files:
        # Load and preprocess image
        image = cv2.imread(image_file)
        if image is None:
            continue  # Skip any unreadable images
        input_image = preprocess_image(image, input_size)

        # Make predictions
        predictions = model.predict(input_image)

        # Since it's a binary classification, we threshold the predictions
        is_deer_present = predictions[0][0] > 0.5

        # Draw bounding box if deer is present
        if is_deer_present:
            cv2.rectangle(image, (10, 10), (image.shape[1]-10, image.shape[0]-10), (0, 255, 0), 2)

        # Save the resulting image
        base_filename = os.path.basename(image_file)
        output_path = os.path.join(output_folder, base_filename)
        cv2.imwrite(output_path, image)
        print(f"Processed and saved: {output_path}")
        
input_folder = '/Users/Anthraxlemonaid/OneDrive/Desktop/Programming/Datasets/dump'
output_folder = '/Users/Anthraxlemonaid/OneDrive/Desktop/Programming/Datasets/result'
input_size = (64, 64)

process_and_save_images(input_folder, output_folder, input_size)
