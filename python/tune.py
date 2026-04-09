"""
Hyperparameter Tuning with Keras Tuner

Searches over key design-space parameters to find the best configuration
for face recognition accuracy on the ESP32-S3.
"""

import os
import numpy as np
import tensorflow as tf
import keras
import keras_tuner as kt
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.applications import MobileNetV2

from preprocess import IMG_SIZE, NUM_CLASSES, LABELS
from main import preprocess_and_load_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

GEN_DIR = os.path.join(os.path.dirname(__file__), 'gen')
TUNER_DIR = os.path.join(GEN_DIR, 'tuner')


def build_tunable_model(hp) -> keras.Model:
    """Build model with tunable hyperparameters."""
    alpha = hp.Choice('alpha', [0.25, 0.35, 0.5], default=0.35)
    dense_units = hp.Choice('dense_units', [32, 64, 128], default=64)
    dropout_1 = hp.Float('dropout_1', 0.2, 0.5, step=0.1, default=0.3)
    dropout_2 = hp.Float('dropout_2', 0.1, 0.3, step=0.1, default=0.2)
    learning_rate = hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4], default=1e-3)
    label_smoothing = hp.Choice('label_smoothing', [0.0, 0.05, 0.1, 0.15], default=0.1)

    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        alpha=alpha
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_1)(x)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(dropout_2)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=['accuracy']
    )
    return model


def main():
    print('Loading data...')
    x_train, y_train, x_test, y_test = preprocess_and_load_data()
    y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_cat = keras.utils.to_categorical(y_test, NUM_CLASSES)
    print(f'Train: {x_train.shape}, Test: {x_test.shape}')
    print()

    tuner = kt.Hyperband(
        build_tunable_model,
        objective='val_accuracy',
        max_epochs=20,
        factor=3,
        directory=TUNER_DIR,
        project_name='face_recognition',
        overwrite=True
    )

    print('Search space:')
    tuner.search_space_summary()
    print()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    tuner.search(
        x_train, y_train_cat,
        validation_data=(x_test, y_test_cat),
        callbacks=[early_stopping],
        batch_size=32,
    )

    print()
    print('=== Best Hyperparameters ===')
    best_hps = tuner.get_best_hyperparameters(num_trials=3)
    for i, hp in enumerate(best_hps):
        print(f'\nTrial {i + 1}:')
        print(f'  alpha:           {hp.get("alpha")}')
        print(f'  dense_units:     {hp.get("dense_units")}')
        print(f'  dropout_1:       {hp.get("dropout_1")}')
        print(f'  dropout_2:       {hp.get("dropout_2")}')
        print(f'  learning_rate:   {hp.get("learning_rate")}')
        print(f'  label_smoothing: {hp.get("label_smoothing")}')

    print()
    print('To use these results, update the design-space parameters at the top of main.py.')
    print('Then re-run main.py with the chosen configuration.')


if __name__ == '__main__':
    main()
