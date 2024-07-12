import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SeparableConv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping
import time

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
pathDeer = '/Users/Anthraxlemonaid/OneDrive/Desktop/Programming/Datasets/ann_deer'
pathNotDeer = '/Users/Anthraxlemonaid/OneDrive/Desktop/Programming/Datasets/not_deer'

dims = 128

# Image preprocessing
def load_images(folder, label):
    images = []
    bboxes = []
    for file in tqdm(os.listdir(folder)):
        if file.endswith('.jpg'):
            img_path = os.path.join(folder, file)
            json_path = os.path.join(img_path + '.json')
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (dims, dims))
                images.append(img)
                if label == 1:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        image_height = data['size']['height']
                        image_width = data['size']['width']
                        img_bboxes = []
                        for obj in data['objects']:
                            if obj['classTitle'] == 'Deer':
                                xmin, ymin = obj['points']['exterior'][0]
                                xmax, ymax = obj['points']['exterior'][1]
                                xmin /= image_width
                                ymin /= image_height
                                xmax /= image_width
                                ymax /= image_height
                                img_bboxes.append([xmin, ymin, xmax, ymax])
                        bboxes.append(img_bboxes)
                else:
                    bboxes.append([])
    return images, [label] * len(images), bboxes

def decode_bboxes(bboxes, image_shape):
    decoded_bboxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        xmin *= image_shape[1]
        ymin *= image_shape[0]
        xmax *= image_shape[1]
        ymax *= image_shape[0]
        decoded_bboxes.append([xmin, ymin, xmax, ymax])
    return decoded_bboxes

# Function to draw bounding boxes on an image
def draw_bboxes(image, bboxes):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.title('Predicted bounding boxes')
    plt.show()
    
class DropoutRateScheduler(Callback):
    def __init__(self, epoch_threshold, initial_rate, new_rate):
        super(DropoutRateScheduler, self).__init__()
        self.epoch_threshold = epoch_threshold
        self.initial_rate = initial_rate
        self.new_rate = new_rate

    def on_epoch_begin(self, epoch, logs=None):
        current_rate = self.new_rate if epoch >= self.epoch_threshold else self.initial_rate
        for layer in self.model.layers:
            if isinstance(layer, Dropout):
                layer.rate = current_rate
                print(f"Epoch {epoch + 1}: Setting dropout rate to {current_rate}")

deer_images, deer_labels, deer_bboxes = load_images(pathDeer, 1)
notdeer_images, notdeer_labels, notdeer_bboxes = load_images(pathNotDeer, 0)

images = deer_images + notdeer_images
labels = deer_labels + notdeer_labels
bboxes = deer_bboxes + notdeer_bboxes

data = np.array(images).astype('float32') / 255.0
labels = np.array(labels)

max_boxes = max(len(bbox) for bbox in bboxes)
padded_bboxes = []

for bbox in bboxes:
    while len(bbox) < max_boxes:
        bbox.append([0, 0, 0, 0])
    padded_bboxes.append(bbox)
bboxes = np.array(padded_bboxes)

# Flatten bboxes for training
bboxes = bboxes.reshape(len(bboxes), -1)

x_train, x_test, y_train, y_test, bbox_train, bbox_test = train_test_split(data, labels, bboxes, test_size=0.25, random_state=42)

# Create the model
inputs = Input(shape=(dims, dims, 3))
x = SeparableConv2D(32, (3, 3), activation='relu')(inputs)
x = SeparableConv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = SeparableConv2D(64, (3, 3), activation='relu')(x)
x = SeparableConv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = SeparableConv2D(128, (3, 3), activation='relu')(x)
x = SeparableConv2D(256, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.4)(x)

# Output for bounding boxes
bbox_output = Dense(max_boxes * 4, activation='sigmoid', name='bbox_output')(x)

model = Model(inputs, bbox_output)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['accuracy']) # 0.0001


#early_stopping = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=150, min_lr=0.00001)

# Use the custom callback
dropout_scheduler = DropoutRateScheduler(epoch_threshold=400, initial_rate=0.4, new_rate=0.6)

with tf.device('/GPU:0'):
    history = model.fit(x_train, bbox_train, epochs=700, batch_size=16, validation_data=(x_test, bbox_test), callbacks=[dropout_scheduler])

loss, accuracy = model.evaluate(x_test, bbox_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

modelPath = 'Models/'
model.save(modelPath + 'deer_forest_model.keras')

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

def measure_fps(model, x_test):
    num_images = len(x_test)
    start_time = time.time()
    for img in x_test:
        img = np.expand_dims(img, axis=0)
        model.predict(img)
    end_time = time.time()
    total_time = end_time - start_time
    fps = num_images / total_time
    return fps

fps = measure_fps(model, x_test)
print(f"Model processes images at {fps:.2f} FPS")

num_images_to_visualize = 5
selected_images = x_test[:num_images_to_visualize]
selected_bboxes = bbox_test[:num_images_to_visualize]

predicted_bboxes = model.predict(selected_images)

for i in range(num_images_to_visualize):
    image = selected_images[i]
    true_bboxes = decode_bboxes(selected_bboxes[i].reshape(-1, 4), image.shape)
    pred_bboxes = decode_bboxes(predicted_bboxes[i].reshape(-1, 4), image.shape)
    
    print("True bounding boxes:")
    draw_bboxes(image, true_bboxes)
    
    print("Predicted bounding boxes:")
    draw_bboxes(image, pred_bboxes)
