import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback

# Set the GPU device
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.experimental.set_virtual_device_configuration(
            physical_devices[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
        print(f"GPU device set: {physical_devices[0]}")
    except RuntimeError as e:
        print(e)

# Path to the data
pathDeer = '/Users/Anthraxlemonaid/OneDrive/Desktop/Programming/Datasets/deer'
pathNotDeer = '/Users/Anthraxlemonaid/OneDrive/Desktop/Programming/Datasets/not_deer'

dims = 128
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
                img = cv2.resize(img, (dims, dims))
                data.append(img)
                labels.append(label)
       
        
load_images(pathDeer, 1) # 1 is the label for deer

num_deer = len(data)
load_images(pathNotDeer, 0) # 0 is the label for forest

# Convert data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Normalize the data
data = data.astype('float32') / 255.0

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

#class CustomDropoutScheduler(Callback):
#    def __init__(self, start_epoch, dropout_layer):
#        super(CustomDropoutScheduler, self).__init__()
#        self.start_epoch = start_epoch
#        self.dropout_layer = dropout_layer
#
#    def on_epoch_begin(self, epoch, logs=None):
#        if epoch >= self.start_epoch:
#            # ramp up the dropout rate by 0.1 every epoch
#            new_rate = min(self.dropout_layer.rate + 0.1, 0.5)
#            self.dropout_layer.rate = new_rate
#            print(f"Dropout rate updated to: {new_rate}")

# Create the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(dims, dims, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Set the starting epoch and dropout layer
start_epoch = 20  # Example: Start dropout after epoch 5
dropout_layer = model.layers[6]  # Assuming dropout is the second layer in the model

# Create the dropout scheduler callback
#dropout_scheduler = CustomDropoutScheduler(start_epoch, dropout_layer)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


# Train the model
with tf.device('/GPU:0'):
    history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test)) # callbacks=[dropout_scheduler]

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# Save the model
modelPath = 'Models/'
model.save(modelPath + 'deer_forest_model.keras')

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()