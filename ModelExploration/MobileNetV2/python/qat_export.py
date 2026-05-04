"""
Quantization-Aware Training (QAT) subprocess.

This script runs with TF_USE_LEGACY_KERAS=1 to use tensorflow-model-optimization
which is incompatible with Keras 3.x. It trains from scratch with legacy Keras,
applies QAT, and exports the quantized TFLite model.

Usage: TF_USE_LEGACY_KERAS=1 python3 python/qat_export.py
"""

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import tf_keras as keras
import tensorflow_model_optimization as tfmot

from utils.train_val_split import split_train_for_validation

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
GEN_DIR = os.path.join(os.path.dirname(__file__), 'gen')
IMG_SIZE = 96
NUM_CLASSES = 3
LABELS = {'Amine': 0, 'Rifki': 1, 'Jakub': 2}
LABEL_SMOOTHING = 0.05
REJECTION_THRESHOLD = 0.90
ALPHA = 0.35
FINE_TUNE_LAYERS = 20
DENSE_UNITS = 32
DROPOUT_1 = 0.4
DROPOUT_2 = 0.1
SEED = 42
VAL_FRACTION = 0.15


def load_data():
    gen_dir = os.path.abspath(GEN_DIR)
    x_train = np.load(os.path.join(gen_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(gen_dir, 'y_train.npy'))
    x_test = np.load(os.path.join(gen_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(gen_dir, 'y_test.npy'))
    return x_train, y_train, x_test, y_test


def build_flat_model():
    """Build a flat Functional model by inlining MobileNetV2 layers.

    quantize_model can't handle Model-inside-Model, so we connect
    MobileNetV2's internal layers directly into our model graph.
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        alpha=ALPHA
    )

    # Build flat model by connecting base_model's input/output directly
    inputs = base_model.input
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(DROPOUT_1)(x)
    x = keras.layers.Dense(DENSE_UNITS, activation='relu')(x)
    x = keras.layers.Dropout(DROPOUT_2)(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def main():
    gen_dir = os.path.abspath(GEN_DIR)

    print('Loading data...')
    x_train, y_train, x_test, y_test = load_data()
    x_fit, y_fit, x_val, y_val, split_info = split_train_for_validation(
        x_train, y_train, DATA_DIR, GEN_DIR, LABELS,
        val_fraction=VAL_FRACTION, seed=SEED
    )
    y_fit_cat = keras.utils.to_categorical(y_fit, NUM_CLASSES)
    y_val_cat = keras.utils.to_categorical(y_val, NUM_CLASSES)
    y_test_cat = keras.utils.to_categorical(y_test, NUM_CLASSES)
    print(f'Train cache: {x_train.shape}, Train: {x_fit.shape}, Val: {x_val.shape}, Test: {x_test.shape}')
    print(f'Val split manifest: {split_info["split_path"]}')

    # Build and train with legacy Keras (two-phase transfer learning)
    print('\n=== Phase 1: Feature extraction (frozen base) ===')
    model = build_flat_model()
    # Freeze all layers except the head (last 5 layers we added)
    for layer in model.layers[:-5]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )

    model.fit(x_fit, y_fit_cat, epochs=20, batch_size=32,
              validation_data=(x_val, y_val_cat),
              callbacks=[keras.callbacks.EarlyStopping(
                  monitor='val_loss', patience=10, restore_best_weights=True)])

    print('\n=== Phase 2: Fine-tuning ===')
    # Unfreeze last FINE_TUNE_LAYERS of the model (includes head layers)
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[-(FINE_TUNE_LAYERS + 5):]:
        layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )

    model.fit(x_fit, y_fit_cat, epochs=50, batch_size=32,
              validation_data=(x_val, y_val_cat),
              callbacks=[keras.callbacks.EarlyStopping(
                  monitor='val_loss', patience=10, restore_best_weights=True)])

    _, pre_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f'\nPre-QAT test accuracy: {pre_acc:.4f}')

    # Apply QAT
    print('\n=== Phase 3: Quantization-Aware Training ===')
    qat_model = tfmot.quantization.keras.quantize_model(model)
    qat_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )

    qat_model.fit(
        x_fit, y_fit_cat,
        epochs=10,
        batch_size=32,
        validation_data=(x_val, y_val_cat),
        callbacks=[keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True)]
    )

    _, post_acc = qat_model.evaluate(x_test, y_test_cat, verbose=0)
    print(f'\nPost-QAT test accuracy: {post_acc:.4f}')

    # Export to TFLite
    print('\nConverting QAT model to TFLite INT8...')
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for i in range(len(x_fit)):
            yield [x_fit[i:i + 1].astype(np.float32)]
    converter.representative_dataset = representative_dataset

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zp = input_details[0]['quantization']
    output_scale, output_zp = output_details[0]['quantization']

    model_size = len(tflite_model)
    print(f'Input scale:      {input_scale}')
    print(f'Input zero point: {input_zp}')
    print(f'Output scale:      {output_scale}')
    print(f'Output zero point: {output_zp}')
    print(f'Model size:        {model_size / 1024:.1f} KB')

    # Save TFLite
    tflite_path = os.path.join(gen_dir, 'model.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f'Saved: {tflite_path}')

    # Write C source files
    from utils.export_tflite import write_model_h_file, write_model_c_file
    model_c_path = os.path.join(gen_dir, 'model.c')
    model_h_path = os.path.join(gen_dir, 'model.h')

    defines = {
        'IMG_SIZE': IMG_SIZE,
        'NUM_CLASSES': NUM_CLASSES,
        'INPUT_SCALE': f'{input_scale}f',
        'INPUT_ZERO_POINT': input_zp,
        'OUTPUT_SCALE': f'{output_scale}f',
        'OUTPUT_ZERO_POINT': output_zp,
        'MODEL_SIZE': model_size,
        'REJECTION_THRESHOLD_Q': int(REJECTION_THRESHOLD / output_scale + output_zp),
    }
    declarations = [
        f'// Labels: {", ".join(f"{name}={idx}" for name, idx in sorted(LABELS.items(), key=lambda x: x[1]))}',
        '// Quantization method: QAT',
        f'// Alpha (depth multiplier): {ALPHA}',
        f'// Rejection threshold: {REJECTION_THRESHOLD}',
    ]
    write_model_h_file(model_h_path, defines, declarations)
    write_model_c_file(model_c_path, tflite_model)
    print(f'Saved: {model_h_path}')
    print(f'Saved: {model_c_path}')

    # Evaluate TFLite model
    print('\n=== QAT TFLite Evaluation ===')
    x_test_q = np.clip(x_test / input_scale + input_zp, -128, 127).astype(np.int8)
    y_pred_all = np.empty((len(x_test), NUM_CLASSES), dtype=np.int8)
    for i in range(len(x_test)):
        interpreter.set_tensor(input_details[0]['index'],
                               x_test_q[i].reshape(1, IMG_SIZE, IMG_SIZE, 3))
        interpreter.invoke()
        y_pred_all[i] = interpreter.get_tensor(output_details[0]['index'])[0]

    y_pred_float = (y_pred_all.astype(np.float32) - output_zp) * output_scale
    y_pred_int = np.argmax(y_pred_float, axis=1)
    accuracy = np.mean(y_pred_int == y_test)

    print(f'QAT TFLite accuracy: {accuracy:.4f}  (PTQ was: 0.8551)')
    print()

    # Per-class
    for name, idx in sorted(LABELS.items(), key=lambda x: x[1]):
        mask_true = (y_test == idx)
        mask_pred = (y_pred_int == idx)
        tp = np.sum(mask_true & mask_pred)
        fp = np.sum(~mask_true & mask_pred)
        fn = np.sum(mask_true & ~mask_pred)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f'  {name}: precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}')

    print('\nDone.')


if __name__ == '__main__':
    main()
