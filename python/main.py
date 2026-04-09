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

# --- Design space parameters ---
ALPHA = 0.35                # MobileNetV2 depth multiplier (0.25/0.35/0.5/1.0)
DENSE_UNITS = 64            # Dense head units (32/64/128)
DROPOUT_1 = 0.3             # Dropout after global average pooling
DROPOUT_2 = 0.2             # Dropout after dense layer
LABEL_SMOOTHING = 0.1       # Prevents overconfident softmax (0.0–0.15)
FINE_TUNE_LAYERS = 20       # Layers to unfreeze in phase 2 (scale with alpha)
REJECTION_THRESHOLD = 0.90  # Confidence threshold for unknown rejection
USE_QAT = True              # Use Quantization-Aware Training


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


def build_model(alpha: float = ALPHA,
                dense_units: int = DENSE_UNITS,
                dropout_1: float = DROPOUT_1,
                dropout_2: float = DROPOUT_2) -> keras.Model:
    """Build MobileNetV2-based classifier for face recognition.

    Args:
        alpha: Width multiplier for MobileNetV2 (smaller = lighter model).
        dense_units: Number of units in the classification head dense layer.
        dropout_1: Dropout rate after global average pooling.
        dropout_2: Dropout rate after the dense layer.
    """
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
    return model


def train_model(x_train: np.ndarray, y_train: np.ndarray,
                x_test: np.ndarray, y_test: np.ndarray,
                label_smoothing: float = LABEL_SMOOTHING,
                fine_tune_layers: int = FINE_TUNE_LAYERS) -> keras.Model:
    """Two-phase transfer learning training with label smoothing."""
    gen_dir = os.path.abspath(GEN_DIR)
    os.makedirs(gen_dir, exist_ok=True)
    model_path = os.path.join(gen_dir, 'model.keras')

    model = build_model()

    # Convert labels to one-hot for label smoothing
    y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_cat = keras.utils.to_categorical(y_test, NUM_CLASSES)

    # --- Phase 1: Feature extraction (frozen base) ---
    print('Phase 1: Training classification head (base frozen)...')
    print(f'  alpha={ALPHA}, dense_units={DENSE_UNITS}, '
          f'label_smoothing={label_smoothing}, fine_tune_layers={fine_tune_layers}')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
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
        x_train, y_train_cat,
        epochs=20,
        batch_size=32,
        validation_data=(x_test, y_test_cat),
        callbacks=[early_stopping, model_checkpoint]
    )

    # Load best from phase 1
    model = keras.models.load_model(model_path)

    # --- Phase 2: Fine-tuning (unfreeze last N layers of base) ---
    print()
    print(f'Phase 2: Fine-tuning (unfreezing last {fine_tune_layers} layers)...')
    base_model = model.layers[1]  # MobileNetV2 is the second layer
    base_model.trainable = True

    # Freeze all layers except the last N
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=['accuracy']
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        model_path, monitor='val_loss', save_best_only=True
    )

    model.fit(
        x_train, y_train_cat,
        epochs=50,
        batch_size=32,
        validation_data=(x_test, y_test_cat),
        callbacks=[early_stopping, model_checkpoint]
    )

    # Load best model overall
    model = keras.models.load_model(model_path)
    return model


def train_model_qat(model: keras.Model,
                    x_train: np.ndarray, y_train: np.ndarray,
                    x_test: np.ndarray, y_test: np.ndarray,
                    label_smoothing: float = LABEL_SMOOTHING) -> keras.Model:
    """Phase 3: Quantization-Aware Training for better INT8 accuracy."""
    import tensorflow_model_optimization as tfmot

    gen_dir = os.path.abspath(GEN_DIR)
    model_path = os.path.join(gen_dir, 'model_qat.keras')

    print()
    print('Phase 3: Quantization-Aware Training (QAT)...')
    qat_model = tfmot.quantization.keras.quantize_model(model)

    y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_cat = keras.utils.to_categorical(y_test, NUM_CLASSES)

    qat_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=['accuracy']
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        model_path, monitor='val_loss', save_best_only=True
    )

    qat_model.fit(
        x_train, y_train_cat,
        epochs=10,
        batch_size=32,
        validation_data=(x_test, y_test_cat),
        callbacks=[early_stopping, model_checkpoint]
    )

    return qat_model


def evaluate_model(model: keras.Model, x_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the Keras model on the test set."""
    y_test_cat = keras.utils.to_categorical(y_test, NUM_CLASSES)
    test_loss, test_accuracy = model.evaluate(x_test, y_test_cat)
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

    # Rejection analysis
    evaluate_with_rejection(y_pred, y_test)


def evaluate_with_rejection(y_pred_probs: np.ndarray, y_test: np.ndarray,
                            thresholds: list[float] = None):
    """Evaluate model performance at different rejection thresholds.

    For each threshold, shows how many samples would be accepted vs rejected,
    and the accuracy on accepted samples only.
    """
    if thresholds is None:
        thresholds = [0.80, 0.85, 0.90, 0.95]

    print()
    print('Rejection threshold analysis:')
    print(f'  {"Threshold":>10}  {"Accepted":>10}  {"Rejected":>10}  {"Acc (accepted)":>15}')
    print(f'  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*15}')

    for threshold in thresholds:
        max_conf = np.max(y_pred_probs, axis=1)
        accepted = max_conf >= threshold
        rejected = ~accepted

        if accepted.sum() > 0:
            y_pred_int = np.argmax(y_pred_probs[accepted], axis=1)
            acc = np.mean(y_pred_int == y_test[accepted])
        else:
            acc = 0.0

        marker = ' <--' if abs(threshold - REJECTION_THRESHOLD) < 0.001 else ''
        print(f'  {threshold:>10.2f}  {accepted.sum():>10}/{len(y_test)}'
              f'  {rejected.sum():>10}  {acc:>14.4f}{marker}')


def export_model_to_tflite(model: keras.Model, x_train: np.ndarray,
                           is_qat: bool = False) -> bytes:
    """Convert to INT8-quantized TFLite and export C source files."""
    gen_dir = os.path.abspath(GEN_DIR)

    print()
    quantization_method = 'QAT' if is_qat else 'PTQ'
    print(f'Converting to TFLite with INT8 quantization ({quantization_method})...')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if not is_qat:
        # PTQ needs a representative dataset for calibration
        # Use ALL training samples for better calibration accuracy
        def representative_dataset():
            for i in range(len(x_train)):
                yield [x_train[i:i + 1].astype(np.float32)]
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

    model_size = len(tflite_model)
    print(f'Input scale:      {input_scale}')
    print(f'Input zero point: {input_zero_point}')
    print(f'Output scale:      {output_scale}')
    print(f'Output zero point: {output_zero_point}')
    print(f'Model size:        {model_size / 1024:.1f} KB')
    print(f'Quantization:      {quantization_method}')

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
        'MODEL_SIZE': model_size,
        'REJECTION_THRESHOLD_Q': int(REJECTION_THRESHOLD / output_scale + output_zero_point),
    }
    declarations = [
        f'// Labels: {", ".join(f"{name}={idx}" for name, idx in sorted(LABELS.items(), key=lambda x: x[1]))}',
        f'// Quantization method: {quantization_method}',
        f'// Alpha (depth multiplier): {ALPHA}',
        f'// Rejection threshold: {REJECTION_THRESHOLD}',
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

    # Rejection analysis on TFLite outputs
    evaluate_with_rejection(y_pred_float, y_test)


if __name__ == '__main__':
    # Print configuration
    print('Configuration:')
    print(f'  Alpha:            {ALPHA}')
    print(f'  Dense units:      {DENSE_UNITS}')
    print(f'  Dropout:          {DROPOUT_1}/{DROPOUT_2}')
    print(f'  Label smoothing:  {LABEL_SMOOTHING}')
    print(f'  Fine-tune layers: {FINE_TUNE_LAYERS}')
    print(f'  Rejection thresh: {REJECTION_THRESHOLD}')
    print(f'  QAT enabled:      {USE_QAT}')
    print()

    # Load data
    print('Loading data...')
    x_train, y_train, x_test, y_test = preprocess_and_load_data()
    print(f'Train: {x_train.shape}, Test: {x_test.shape}')
    print()

    # Train model (Phase 1 + Phase 2)
    model = train_model(x_train, y_train, x_test, y_test)

    # Evaluate Keras model
    print()
    print('=== Keras Model Evaluation ===')
    evaluate_model(model, x_test, y_test)

    # Optional Phase 3: QAT
    export_model = model
    is_qat = False
    if USE_QAT:
        try:
            qat_model = train_model_qat(model, x_train, y_train, x_test, y_test)
            print()
            print('=== QAT Model Evaluation ===')
            evaluate_model(qat_model, x_test, y_test)
            export_model = qat_model
            is_qat = True
        except (ImportError, AttributeError) as e:
            print()
            print(f'Warning: QAT unavailable ({e}), falling back to PTQ.')
            print('This is expected with Keras 3.x / TF 2.21+.')

    # Export to TFLite
    tflite_model = export_model_to_tflite(export_model, x_train, is_qat=is_qat)

    # Evaluate TFLite model
    print()
    print('=== Quantized TFLite Model Evaluation ===')
    evaluate_tflite_model(tflite_model, x_test, y_test)

    print()
    print('Done.')
