import tensorflow as tf

# List all available devices
devices = tf.config.list_physical_devices()
print("Available devices:")
for device in devices:
    print(device)

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU is available.")
else:
    print("GPU is not available.")

# Run a simple computation to see if it's using the GPU
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print("Result of matrix multiplication on GPU:")
    print(c)