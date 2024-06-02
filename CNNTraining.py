import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Path to the data
pathDeer = '/Users/Anthraxlemonaid/OneDrive/Desktop/Programming/Datasets/deer/train'
pathForest = '/Users/Anthraxlemonaid/OneDrive/Desktop/Programming/Datasets/forest/train'

data = []
labels = []

# image preprocessing
def load_images(folder, label):
    for file in tqdm(os.listdir(folder)):
        if file.endswith('.jpg'):
            # Construct the full file path
            img_path = os.path.join(folder, file)
            # Load the image
            img = cv2.imread(img_path)
            
            # Check if image is loaded correctly
            if img is not None:
                img = cv2.resize(img, (64, 64))
                data.append(img)
                labels.append(label)
       
        
load_images(pathDeer, 1) # 1 is the label for deer

num_deer = len(data)
load_images(pathForest, 0) # 0 is the label for forest

# Convert data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Normalize the data
data = data.astype('float32') / 255.0

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

# Create the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test), callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# Save the model
model.save('deer_forest_model.h5')