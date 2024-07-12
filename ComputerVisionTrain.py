import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, SeparableConv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
import time

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

ann_data_dir = os.path.join('/Users/Anthraxlemonaid/OneDrive/Desktop/Programming/Datasets/ann')
deer_dir = os.path.join('/Users/Anthraxlemonaid/OneDrive/Desktop/Programming/Datasets/ann_deer')

nrows = 4
ncols = 4

pic_index = 0

fig = plt.gcf()

fig.set_size_inches(ncols * 4, nrows * 4)
pic_index += 8

next_deer_pic = [os.path.join(deer_dir, fname) for fname in os.listdir(deer_dir)[pic_index-8:pic_index]]
for i, img_path in enumerate(next_deer_pic):
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off')
  img = plt.imread(img_path)
  plt.imshow(img)
plt.show()

# Create the model
model = Sequential([
    SeparableConv2D(32, (3, 3), activation='relu', input_shape=(dims, dims, 3)),
    MaxPooling2D((2, 2)),
    SeparableConv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])