import tensorflow as tf             # type: ignore
import matplotlib.pyplot as plt     # type: ignore
import numpy as np                  # type: ignore
import time

def get_mnist_dataset():

    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)

    return x_train, y_train

def create_model():

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_and_time(device, x, y):

    with tf.device(device):
        model = create_model()
        start = time.time()
        history = model.fit(x, y, epochs=5, batch_size=64, verbose=0)
        end = time.time()
        final_accuracy = history.history['accuracy'][-1]

    return end - start, final_accuracy

x_train, y_train = get_mnist_dataset()

gpu_available = tf.config.list_physical_devices('GPU')
gpu_time = None
gpu_accuracy = None

if gpu_available:
    gpu_time, gpu_accuracy = train_and_time("/GPU:0", x_train, y_train)

if gpu_time:
    print(f"Training time on GPU: {gpu_time:.2f} seconds - Accuracy: {gpu_accuracy:.4f}")
else:
    print("No GPU available for testing.")
    
cpu_time, cpu_accuracy = train_and_time("/CPU:0", x_train, y_train)

print(f"Training time on CPU: {cpu_time:.2f} seconds - Accuracy: {cpu_accuracy:.4f}")
devices = ['CPU']
times = [cpu_time]
accuracies = [cpu_accuracy]
colors = ['blue']

if gpu_time:
    devices.append('GPU')
    times.append(gpu_time)
    accuracies.append(gpu_accuracy)
    colors.append('green')

plt.figure(figsize=(8, 5))
bars = plt.bar(devices, times, color=colors)
plt.ylabel('Time (seconds)')
plt.title('Training Time Comparison: CPU vs GPU')

for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0,
             height,
             f"Acc: {acc:.2%}",
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig("training_time_comparison.png")
print("Plot saved as training_time_comparison.png")