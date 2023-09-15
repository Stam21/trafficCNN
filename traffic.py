import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, models
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4
THREADS = 8

#-------------------------------------------------------
# RESULTS
#  loss: 0.0390 - accuracy: 0.9881 - precision: 0.9901 - recall: 0.9861 - f1_score: 0.4541 - 3s/epoch - 8ms/step
#-------------------------------------------------------

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    def process_image(image_path, category):
        image = cv2.imread(image_path)
        if image is not None:
            resized_image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            normalized_image = resized_image.astype(np.float32) / 255.0
            return normalized_image, category
        else:
            return None, None

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        for category in os.listdir(data_dir):
            category_path = os.path.join(data_dir, category)
            if os.path.isdir(category_path):
                image_paths = [os.path.join(category_path, image) for image in os.listdir(category_path)]
                category_labels = [category] * len(image_paths)
                batches = [image_paths[i:i+32] for i in range(0, len(image_paths), 32)]
                for batch in batches:
                    batch_labels = category_labels[:len(batch)]
                    results = executor.map(process_image, batch, batch_labels)
                    for img, lbl in results:
                        if img is not None and lbl is not None:
                            images.append(img)
                            labels.append(lbl)

    return images, labels

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    # TODO: Optimize the model by configuring the layers of the CNN
    def create_cnn_model():
        model = models.Sequential()

        # Add convolutional layers
        # RELU activation function is optimal in computer vision problems.
        model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_WIDTH,IMG_HEIGHT,3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization(axis=-1))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization(axis=-1))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization(axis=-1))
        # Flatten the output and add fully connected layers
        model.add(layers.Flatten())
        model.add(layers.BatchNormalization())
        # Add a dropout layer to prevent overfitting
        model.add(layers.Dropout(0.5))

        # Use softmax activation function to normalize the output.
        # Softmax is a Sigmoid function that handles multiclass problems.
        model.add(layers.Dense(NUM_CATEGORIES, activation='softmax'))

        return model
    
    model = create_cnn_model()

    
    
    # Define the optimizer with an initial learning rate
    initial_learning_rate = 0.001
    optimizer = Adam(learning_rate=initial_learning_rate)

    # Define the metrics
    metrics = ['accuracy', Precision(name='precision'), Recall(name='recall'), f1_score]
    
    # Compile the model
    model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=metrics)


    return model

def f1_score(y_true, y_pred):
    # Convert predictions to binary values (0 or 1)
    y_pred_bin = tf.keras.backend.round(y_pred)
    
    # Calculate true positives, false positives, and false negatives
    tp = tf.keras.backend.sum(tf.keras.backend.round(y_true * y_pred_bin), axis=0)
    fp = tf.keras.backend.sum(tf.keras.backend.round((1 - y_true) * y_pred_bin), axis=0)
    fn = tf.keras.backend.sum(tf.keras.backend.round(y_true * (1 - y_pred_bin)), axis=0)
    
    # Calculate precision and recall
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    
    # Calculate F1-score
    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    
    return tf.keras.backend.mean(f1)
if __name__ == "__main__":
    main()
