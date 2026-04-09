"""
Experiment Comparison Script

Trains multiple configurations and outputs a comparison table
for the report's design-space analysis section.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

from preprocess import IMG_SIZE, NUM_CLASSES, LABELS
from main import (preprocess_and_load_data, build_model, train_model,
                  train_model_qat, export_model_to_tflite,
                  evaluate_with_rejection)
from utils.eval_utils import compute_precision_recall_f1

GEN_DIR = os.path.join(os.path.dirname(__file__), 'gen')

# Configurations to compare
CONFIGS = [
    {'name': 'a035_ptq', 'alpha': 0.35, 'fine_tune': 20, 'qat': False, 'label_smoothing': 0.1},
    {'name': 'a035_qat', 'alpha': 0.35, 'fine_tune': 20, 'qat': True,  'label_smoothing': 0.1},
    {'name': 'a050_ptq', 'alpha': 0.50, 'fine_tune': 25, 'qat': False, 'label_smoothing': 0.1},
    {'name': 'a050_qat', 'alpha': 0.50, 'fine_tune': 25, 'qat': True,  'label_smoothing': 0.1},
    {'name': 'a035_no_ls', 'alpha': 0.35, 'fine_tune': 20, 'qat': True, 'label_smoothing': 0.0},
]


def run_experiment(config: dict, x_train, y_train, x_test, y_test) -> dict:
    """Run a single experiment and return metrics."""
    import main as m

    # Override globals for this experiment
    m.ALPHA = config['alpha']
    m.FINE_TUNE_LAYERS = config['fine_tune']
    m.LABEL_SMOOTHING = config['label_smoothing']

    print(f'\n{"="*60}')
    print(f'Experiment: {config["name"]}')
    print(f'  alpha={config["alpha"]}, fine_tune={config["fine_tune"]}, '
          f'qat={config["qat"]}, label_smoothing={config["label_smoothing"]}')
    print(f'{"="*60}')

    # Build and train
    model = build_model(alpha=config['alpha'])
    model = train_model(x_train, y_train, x_test, y_test,
                        label_smoothing=config['label_smoothing'],
                        fine_tune_layers=config['fine_tune'])

    # Evaluate Keras model
    y_test_cat = keras.utils.to_categorical(y_test, NUM_CLASSES)
    _, keras_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    y_pred = model.predict(x_test, verbose=0)
    y_pred_int = np.argmax(y_pred, axis=1)

    # Optional QAT
    export_model = model
    is_qat = config['qat']
    if config['qat']:
        try:
            export_model = train_model_qat(model, x_train, y_train, x_test, y_test,
                                           label_smoothing=config['label_smoothing'])
        except (ImportError, AttributeError, ValueError) as e:
            print(f'  QAT failed ({e}), falling back to PTQ.')
            is_qat = False

    # Export and evaluate TFLite
    tflite_model = export_model_to_tflite(export_model, x_train, is_qat=is_qat)
    tflite_size_kb = len(tflite_model) / 1024

    # Evaluate TFLite accuracy
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zp = input_details[0]['quantization']
    output_scale, output_zp = output_details[0]['quantization']

    x_q = np.clip(x_test / input_scale + input_zp, -128, 127).astype(np.int8)
    y_tflite = np.empty((len(x_test), NUM_CLASSES), dtype=np.int8)
    for i in range(len(x_test)):
        interpreter.set_tensor(input_details[0]['index'],
                               x_q[i].reshape(1, IMG_SIZE, IMG_SIZE, 3))
        interpreter.invoke()
        y_tflite[i] = interpreter.get_tensor(output_details[0]['index'])[0]

    y_tflite_float = (y_tflite.astype(np.float32) - output_zp) * output_scale
    tflite_acc = np.mean(np.argmax(y_tflite_float, axis=1) == y_test)

    # Rejection at 0.90 threshold
    max_conf = np.max(y_pred, axis=1)
    accepted_90 = max_conf >= 0.90
    if accepted_90.sum() > 0:
        acc_at_90 = np.mean(np.argmax(y_pred[accepted_90], axis=1) == y_test[accepted_90])
    else:
        acc_at_90 = 0.0
    reject_rate_90 = 1.0 - accepted_90.mean()

    return {
        'name': config['name'],
        'alpha': config['alpha'],
        'qat': config['qat'],
        'label_smoothing': config['label_smoothing'],
        'keras_acc': keras_acc,
        'tflite_acc': tflite_acc,
        'tflite_size_kb': tflite_size_kb,
        'acc_at_90': acc_at_90,
        'reject_rate_90': reject_rate_90,
    }


def main():
    print('Loading data...')
    x_train, y_train, x_test, y_test = preprocess_and_load_data()
    print(f'Train: {x_train.shape}, Test: {x_test.shape}')

    results = []
    for config in CONFIGS:
        result = run_experiment(config, x_train, y_train, x_test, y_test)
        results.append(result)

    # Print comparison table
    print()
    print('=' * 90)
    print('EXPERIMENT COMPARISON TABLE')
    print('=' * 90)
    header = (f'{"Config":<15} {"Alpha":>5} {"QAT":>4} {"LS":>4} '
              f'{"Keras":>7} {"TFLite":>7} {"Size(KB)":>8} '
              f'{"Acc@0.9":>8} {"Rej@0.9":>8}')
    print(header)
    print('-' * 90)

    for r in results:
        row = (f'{r["name"]:<15} {r["alpha"]:>5.2f} '
               f'{"Y" if r["qat"] else "N":>4} {r["label_smoothing"]:>4.2f} '
               f'{r["keras_acc"]:>6.1%} {r["tflite_acc"]:>6.1%} '
               f'{r["tflite_size_kb"]:>7.0f} '
               f'{r["acc_at_90"]:>7.1%} {r["reject_rate_90"]:>7.1%}')
        print(row)

    print()
    print('Use these results for the design-space analysis in your report.')


if __name__ == '__main__':
    main()
