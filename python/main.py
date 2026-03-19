"""
Stage 3+4: Model Training and TFLite Export

Trains a MobileNetV2-based face classifier with transfer learning,
exports to INT8-quantized TFLite, and generates C source files for ESP32.
"""

import os
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.applications import MobileNetV2

from preprocess import preprocess_all, IMG_SIZE, NUM_CLASSES, LABELS
from utils.export_tflite import write_model_h_file, write_model_c_file
from utils.eval_utils import compute_precision_recall_f1, print_confusion_matrix

# Minimize TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
GEN_DIR = os.path.join(os.path.dirname(__file__), 'gen')
MODEL_C_PATH = os.path.join(GEN_DIR, 'model.c')
MODEL_H_PATH = os.path.join(GEN_DIR, 'model.h')
USE_CACHED_DATA = True


def preprocess_and_load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed data from .npy files, running preprocessing if needed."""
    gen_dir = os.path.abspath(GEN_DIR)

    if not USE_CACHED_DATA \
            or not os.path.exists(os.path.join(gen_dir, 'x_train.npy')) \
            or not os.path.exists(os.path.join(gen_dir, 'y_train.npy')) \
            or not os.path.exists(os.path.join(gen_dir, 'x_test.npy')) \
            or not os.path.exists(os.path.join(gen_dir, 'y_test.npy')):
        preprocess_all(DATA_DIR, GEN_DIR)

    x_train = np.load(os.path.join(gen_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(gen_dir, 'y_train.npy'))
    x_test = np.load(os.path.join(gen_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(gen_dir, 'y_test.npy'))

    return x_train, y_train, x_test, y_test


def build_model() -> keras.Model:
    """Build MobileNetV2-based classifier for face recognition."""
    # Load MobileNetV2 pre-trained on ImageNet, without top classification head
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    # Freeze all base model layers initially
    base_model.trainable = False

    # Build the full model
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model


def train_model(x_train: np.ndarray, y_train: np.ndarray,
                x_test: np.ndarray, y_test: np.ndarray) -> keras.Model:
    """Two-phase transfer learning training."""
    gen_dir = os.path.abspath(GEN_DIR)
    os.makedirs(gen_dir, exist_ok=True)
    model_path = os.path.join(gen_dir, 'model.keras')

    model = build_model()

    # --- Phase 1: Feature extraction (frozen base) ---
    print('Phase 1: Training classification head (base frozen)...')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        model_path, monitor='val_loss', save_best_only=True
    )

    model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping, model_checkpoint]
    )

    # Load best from phase 1
    model = keras.models.load_model(model_path)

    # --- Phase 2: Fine-tuning (unfreeze last ~30 layers of base) ---
    print()
    print('Phase 2: Fine-tuning (unfreezing last 30 layers)...')
    base_model = model.layers[1]  # MobileNetV2 is the second layer
    base_model.trainable = True

    # Freeze all layers except the last 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        model_path, monitor='val_loss', save_best_only=True
    )

    model.fit(
        x_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping, model_checkpoint]
    )

    # Load best model overall
    model = keras.models.load_model(model_path)
    return model


def evaluate_model(model: keras.Model, x_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the Keras model on the test set."""
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)
    y_pred_int = np.argmax(y_pred, axis=1)

    class_labels = [name for name, _ in sorted(LABELS.items(), key=lambda x: x[1])]

    print()
    print('Test loss:           %.4f' % test_loss)
    print('Test accuracy:       %.4f' % test_accuracy)
    print()

    # Per-class metrics
    for name, idx in sorted(LABELS.items(), key=lambda x: x[1]):
        precision, recall, f1 = compute_precision_recall_f1(y_test, y_pred_int, class_index=idx)
        print(f'  {name}: precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}')
    print()

    print_confusion_matrix(y_test, y_pred_int, class_labels)


def export_model_to_tflite(model: keras.Model, x_train: np.ndarray) -> bytes:
    """Convert to INT8-quantized TFLite and export C source files."""
    gen_dir = os.path.abspath(GEN_DIR)

    print()
    print('Converting to TensorFlow Lite model with INT8 quantization...')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Representative dataset for calibration
    def representative_dataset():
        # Use a subset for calibration (up to 200 samples)
        indices = np.random.choice(len(x_train), min(200, len(x_train)), replace=False)
        for i in indices:
            yield [x_train[i:i + 1].astype(np.float32)]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    # Print quantization parameters
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale = input_details[0]['quantization'][0]
    input_zero_point = input_details[0]['quantization'][1]
    output_scale = output_details[0]['quantization'][0]
    output_zero_point = output_details[0]['quantization'][1]

    print(f'Input scale:      {input_scale}')
    print(f'Input zero point: {input_zero_point}')
    print(f'Output scale:      {output_scale}')
    print(f'Output zero point: {output_zero_point}')
    print(f'Model size:        {len(tflite_model) / 1024:.1f} KB')

    # Export to C source files
    print()
    print('Exporting TFLite model to C source files...')
    defines = {
        'IMG_SIZE': IMG_SIZE,
        'NUM_CLASSES': NUM_CLASSES,
        'INPUT_SCALE': f'{input_scale}f',
        'INPUT_ZERO_POINT': input_zero_point,
        'OUTPUT_SCALE': f'{output_scale}f',
        'OUTPUT_ZERO_POINT': output_zero_point,
    }
    declarations = [
        f'// Labels: {", ".join(f"{name}={idx}" for name, idx in sorted(LABELS.items(), key=lambda x: x[1]))}',
    ]
    write_model_h_file(MODEL_H_PATH, defines, declarations)
    write_model_c_file(MODEL_C_PATH, tflite_model)

    # Save .tflite file
    tflite_path = os.path.join(gen_dir, 'model.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f'Saved: {tflite_path}')

    return tflite_model


def evaluate_tflite_model(tflite_model: bytes, x_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the quantized TFLite model on the test set."""
    print()
    print('Evaluating quantized TFLite model...')

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    # Quantize test inputs
    x_test_quantized = x_test / input_scale + input_zero_point
    x_test_quantized = np.clip(x_test_quantized, -128, 127).astype(np.int8)

    # Run inference on each sample
    y_pred_all = np.empty((len(x_test), NUM_CLASSES), dtype=np.int8)
    for i in range(len(x_test)):
        interpreter.set_tensor(
            input_details[0]['index'],
            x_test_quantized[i].reshape(1, IMG_SIZE, IMG_SIZE, 3)
        )
        interpreter.invoke()
        y_pred_all[i] = interpreter.get_tensor(output_details[0]['index'])[0]

    # Dequantize and get predictions
    y_pred_float = (y_pred_all.astype(np.float32) - output_zero_point) * output_scale
    y_pred_int = np.argmax(y_pred_float, axis=1)

    # Compute accuracy
    accuracy = np.mean(y_pred_int == y_test)
    class_labels = [name for name, _ in sorted(LABELS.items(), key=lambda x: x[1])]

    print(f'Quantized test accuracy: {accuracy:.4f}')
    print()

    for name, idx in sorted(LABELS.items(), key=lambda x: x[1]):
        precision, recall, f1 = compute_precision_recall_f1(y_test, y_pred_int, class_index=idx)
        print(f'  {name}: precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}')
    print()

    print_confusion_matrix(y_test, y_pred_int, class_labels)


if __name__ == '__main__':
    # Load data
    print('Loading data...')
    x_train, y_train, x_test, y_test = preprocess_and_load_data()
    print(f'Train: {x_train.shape}, Test: {x_test.shape}')
    print()

    # Train model
    model = train_model(x_train, y_train, x_test, y_test)

    # Evaluate Keras model
    print()
    print('=== Keras Model Evaluation ===')
    evaluate_model(model, x_test, y_test)

    # Export to TFLite
    tflite_model = export_model_to_tflite(model, x_train)

    # Evaluate TFLite model
    print()
    print('=== Quantized TFLite Model Evaluation ===')
    evaluate_tflite_model(tflite_model, x_test, y_test)

    print()
    print('Done.')
