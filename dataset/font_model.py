from random import randint

import numpy
import cv2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from pathlib import Path

def create_model():
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    font_i = 0
    font_map = {}

    fonts = Path('./fonts')
    for font in fonts.iterdir():
        simple_name = str(font).replace('fonts\\', '')
        font_map[simple_name] = font_i
        font_i = font_i + 1
        for i in range(33, 127):
            path = Path('./generated/' + str(i) + '/' + simple_name)
            for img in path.iterdir():
                image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                decision = randint(1, 10)
                if decision <= 8:
                    train_images.append(image / 255)
                    train_labels.append(font_map[simple_name])
                else:
                    test_images.append(image / 255)
                    test_labels.append(font_map[simple_name])

    train_images = numpy.array(train_images)
    train_labels = numpy.array(train_labels)
    test_images = numpy.array(test_images)
    test_labels = numpy.array(test_labels)

    model = models.Sequential()
    model.add(layers.Conv1D(16, 2, activation='relu', input_shape=(16, 16)))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(32, 2, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(32, 2, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(9))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=500,
                        validation_data=(test_images, test_labels))

create_model()