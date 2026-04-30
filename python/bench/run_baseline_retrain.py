#!/usr/bin/env python3
"""Run the clean F2-fixed baseline retrain for the 96x96 Jakub pipeline."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras as keras
from sklearn.metrics import accuracy_score, f1_score

from bench.build_originals_test import build_arrays
from bench.eval_branches import evaluate as evaluate_tflite
from preprocess import IMG_SIZE, LABELS, NUM_CLASSES
from utils.export_tflite import write_model_h_file
from utils.train_val_split import split_train_for_validation

DATA_DIR = ROOT / "data"
GEN_DIR = PYTHON_DIR / "gen"
RESULTS_DIR = ROOT / "bench" / "results"
FACE_MODEL_PATH = PYTHON_DIR / "blaze_face_short_range.tflite"
EXISTING_MODEL_PATH = GEN_DIR / "model.tflite"

ALPHA = 0.35
DENSE_UNITS = 32
DROPOUT_1 = 0.4
DROPOUT_2 = 0.1
LEARNING_RATE = 5e-4
QAT_LEARNING_RATE = 1e-5
LABEL_SMOOTHING = 0.05
REJECTION_THRESHOLD = 0.90


@dataclass(frozen=True)
class EvalResult:
    name: str
    model_path: Path
    accuracy: float
    macro_f1: float
    correct: int
    n: int
    ci_low: float
    ci_high: float
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--qat-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--output-model",
        type=Path,
        default=GEN_DIR / "baseline_model.tflite",
    )
    parser.add_argument(
        "--output-header",
        type=Path,
        default=GEN_DIR / "baseline_model.h",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=RESULTS_DIR / "baseline_retrain_report.md",
    )
    parser.add_argument(
        "--history-json",
        type=Path,
        default=RESULTS_DIR / "baseline_training_history_seed42.json",
    )
    return parser.parse_args()


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if hasattr(keras.utils, "set_random_seed"):
        keras.utils.set_random_seed(seed)
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as exc:  # pragma: no cover - version dependent.
        print(f"warning: could not enable TF op determinism: {exc}", file=sys.stderr)


def load_cached_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.load(GEN_DIR / "x_train.npy")
    y_train = np.load(GEN_DIR / "y_train.npy")
    x_test = np.load(GEN_DIR / "x_test.npy")
    y_test = np.load(GEN_DIR / "y_test.npy")
    return x_train, y_train, x_test, y_test


def build_flat_model() -> keras.Model:
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        alpha=ALPHA,
    )
    inputs = base_model.input
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(DROPOUT_1)(x)
    x = keras.layers.Dense(DENSE_UNITS, activation="relu")(x)
    x = keras.layers.Dropout(DROPOUT_2)(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=outputs)


def history_summary(history: keras.callbacks.History) -> dict[str, float | int | None]:
    data = history.history
    epochs_run = len(data.get("loss", []))
    summary: dict[str, float | int | None] = {"epochs_run": epochs_run}
    for key in ("loss", "accuracy", "val_loss", "val_accuracy"):
        values = data.get(key, [])
        summary[f"final_{key}"] = float(values[-1]) if values else None
        if key.startswith("val_") and values:
            best = min(values) if key.endswith("loss") else max(values)
            summary[f"best_{key}"] = float(best)
    return summary


def train_float_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
) -> tuple[keras.Model, keras.callbacks.History]:
    model = build_flat_model()
    for layer in model.layers[:-5]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )
    callbacks: list[keras.callbacks.Callback] = []
    if epochs > 1:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        )
    y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_cat = keras.utils.to_categorical(y_val, NUM_CLASSES)
    history = model.fit(
        x_train,
        y_train_cat,
        validation_data=(x_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    return model, history


def train_qat_model(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
) -> tuple[keras.Model, keras.callbacks.History]:
    qat_model = tfmot.quantization.keras.quantize_model(model)
    qat_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=QAT_LEARNING_RATE),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )
    callbacks: list[keras.callbacks.Callback] = []
    if epochs > 1:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=min(5, max(1, epochs // 2)), restore_best_weights=True
            )
        )
    y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_cat = keras.utils.to_categorical(y_val, NUM_CLASSES)
    history = qat_model.fit(
        x_train,
        y_train_cat,
        validation_data=(x_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    return qat_model, history


def export_int8_tflite(
    model: keras.Model,
    representative_x: np.ndarray,
    output_model: Path,
    output_header: Path,
) -> tuple[bytes, dict[str, Any]]:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for idx in range(len(representative_x)):
            yield [representative_x[idx : idx + 1].astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_model.write_bytes(tflite_model)

    interpreter = tf.lite.Interpreter(model_content=tflite_model, num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_detail["quantization"]
    output_scale, output_zero_point = output_detail["quantization"]
    model_size = len(tflite_model)

    defines = {
        "IMG_SIZE": IMG_SIZE,
        "NUM_CLASSES": NUM_CLASSES,
        "INPUT_SCALE": f"{input_scale}f",
        "INPUT_ZERO_POINT": int(input_zero_point),
        "OUTPUT_SCALE": f"{output_scale}f",
        "OUTPUT_ZERO_POINT": int(output_zero_point),
        "MODEL_SIZE": model_size,
        "REJECTION_THRESHOLD_Q": int(REJECTION_THRESHOLD / output_scale + output_zero_point),
    }
    declarations = [
        f"// Labels: {', '.join(f'{name}={idx}' for name, idx in sorted(LABELS.items(), key=lambda item: item[1]))}",
        "// Quantization method: QAT + INT8 TFLite conversion",
        f"// Clean train/val split seed: 42",
        f"// Alpha (depth multiplier): {ALPHA}",
        f"// Dense units: {DENSE_UNITS}",
        f"// Dropout: {DROPOUT_1}/{DROPOUT_2}",
        f"// Label smoothing: {LABEL_SMOOTHING}",
    ]
    write_model_h_file(str(output_header), defines, declarations)

    metadata = {
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zero_point),
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zero_point),
        "input_dtype": str(np.dtype(input_detail["dtype"])),
        "output_dtype": str(np.dtype(output_detail["dtype"])),
        "model_size_bytes": model_size,
    }
    return tflite_model, metadata


def ensure_originals_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    x96, x160, y, capture_ids, class_counts = build_arrays(DATA_DIR, FACE_MODEL_PATH)
    np.save(RESULTS_DIR / "x_test_originals_96_pm1.npy", x96)
    np.save(RESULTS_DIR / "x_test_originals_160_raw.npy", x160)
    np.save(RESULTS_DIR / "y_test_originals.npy", y)
    np.save(RESULTS_DIR / "capture_ids_originals.npy", capture_ids, allow_pickle=True)
    print(
        "Originals-only test counts: "
        + ", ".join(f"{name}={class_counts.get(name, 0)}" for name in sorted(LABELS))
    )
    return x96, y, capture_ids


def wilson_ci(correct: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = correct / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    margin = z * np.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def run_eval(name: str, model_path: Path, x: np.ndarray, y: np.ndarray) -> EvalResult:
    predictions, _probs, metadata = evaluate_tflite(
        model_path=model_path,
        x=x,
        y=y,
        norm="pm1",
        num_threads=1,
    )
    correct = int(np.sum(predictions == y))
    n = int(len(y))
    ci_low, ci_high = wilson_ci(correct, n)
    return EvalResult(
        name=name,
        model_path=model_path,
        accuracy=float(accuracy_score(y, predictions)),
        macro_f1=float(f1_score(y, predictions, average="macro", zero_division=0)),
        correct=correct,
        n=n,
        ci_low=ci_low,
        ci_high=ci_high,
        metadata=metadata,
    )


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def metric_or_dash(value: float | int | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def phase_row(name: str, summary: dict[str, float | int | None]) -> str:
    return (
        f"| {name} | {metric_or_dash(summary.get('epochs_run'))} | "
        f"{metric_or_dash(summary.get('final_loss'))} | "
        f"{metric_or_dash(summary.get('final_accuracy'))} | "
        f"{metric_or_dash(summary.get('final_val_loss'))} | "
        f"{metric_or_dash(summary.get('final_val_accuracy'))} | "
        f"{metric_or_dash(summary.get('best_val_loss'))} | "
        f"{metric_or_dash(summary.get('best_val_accuracy'))} |"
    )


def eval_row(result: EvalResult) -> str:
    ci = f"[{percent(result.ci_low)}, {percent(result.ci_high)}]"
    return (
        f"| {result.name} | {result.correct}/{result.n} | {percent(result.accuracy)} | "
        f"{ci} | {result.macro_f1:.4f} | `{result.model_path.relative_to(ROOT)}` |"
    )


def write_report(
    report_path: Path,
    args: argparse.Namespace,
    split_info: dict[str, Any],
    float_history: keras.callbacks.History,
    qat_history: keras.callbacks.History,
    export_metadata: dict[str, Any],
    eval_results: list[EvalResult],
) -> None:
    float_summary = history_summary(float_history)
    qat_summary = history_summary(qat_history)
    baseline = next(result for result in eval_results if result.name == "baseline_retrain")
    existing = next((result for result in eval_results if result.name == "existing_model"), None)
    delta_text = "Existing model not available."
    if existing is not None:
        delta = baseline.accuracy - existing.accuracy
        delta_text = (
            f"Baseline retrain is {delta * 100:+.2f} pp vs the evaluated existing model "
            f"({percent(existing.accuracy)}). The planning baseline was 98.33%, so the retrain is "
            f"{(baseline.accuracy - 0.9833) * 100:+.2f} pp vs that reference."
        )

    class_lines = []
    for class_name, info in split_info["classes"].items():
        class_lines.append(
            f"- {class_name}: {info['train_capture_count']} train captures / "
            f"{len(info['val_captures'])} val captures; "
            f"{info['train_file_count']} train files / {info['val_file_count']} val files."
        )

    text = f"""# Clean F2-fixed baseline retrain

## Status

Completed single-seed insurance run on CPU with TensorFlow op determinism enabled.

## Configuration

| parameter | value |
| --- | --- |
| seed | {args.seed} |
| IMG_SIZE | {IMG_SIZE} |
| alpha | {ALPHA} |
| dense_units | {DENSE_UNITS} |
| dropout_1 / dropout_2 | {DROPOUT_1} / {DROPOUT_2} |
| learning_rate | {LEARNING_RATE} |
| label_smoothing | {LABEL_SMOOTHING} |
| train epochs | {args.epochs} |
| QAT epochs | {args.qat_epochs} |
| batch size | {args.batch_size} |
| validation split | {args.val_fraction:.0%} of train captures, stratified by class and grouped by capture prefix |

## F2 fix / validation split

Validation is held out from `data/*/train/` only; `x_test` is not used for early stopping or model selection. Manifest: `{Path(split_info['split_path']).relative_to(ROOT)}`.

{chr(10).join(class_lines)}

## Train/validation curves summary

| phase | epochs run | final loss | final acc | final val loss | final val acc | best val loss | best val acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{phase_row('float feature-extraction', float_summary)}
{phase_row('QAT', qat_summary)}

Full history JSON: `{args.history_json.relative_to(ROOT)}`.

## TFLite export

- Output model: `{args.output_model.relative_to(ROOT)}`
- Header: `{args.output_header.relative_to(ROOT)}`
- Size: {export_metadata['model_size_bytes']} bytes ({export_metadata['model_size_bytes'] / 1024:.1f} KiB)
- Input quantization: scale={export_metadata['input_scale']:.8g}, zero_point={export_metadata['input_zero_point']}
- Output quantization: scale={export_metadata['output_scale']:.8g}, zero_point={export_metadata['output_zero_point']}

## Originals-only test evaluation

Evaluation reuses `python/bench/build_originals_test.py` to filter augmented test files and `python/bench/eval_branches.py` for INT8 inference.

| model | correct/n | accuracy | Wilson 95% CI | macro-F1 | path |
| --- | ---: | ---: | ---: | ---: | --- |
{chr(10).join(eval_row(result) for result in eval_results)}

{delta_text}

## Known issue

None for this run. Multi-seed retraining is deferred; this is the requested seed-42 insurance baseline.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text)


def write_failure_report(report_path: Path, args: argparse.Namespace, trace: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Clean F2-fixed baseline retrain

## Status

Run failed before producing a complete baseline artifact. The F2 code changes and validation split helper are still present for inspection.

## Intended configuration

- seed: {args.seed}
- alpha: {ALPHA}
- dense_units: {DENSE_UNITS}
- dropout_1 / dropout_2: {DROPOUT_1} / {DROPOUT_2}
- learning_rate: {LEARNING_RATE}
- label_smoothing: {LABEL_SMOOTHING}
- train epochs: {args.epochs}
- QAT epochs: {args.qat_epochs}
- validation split: {args.val_fraction:.0%} of train captures, grouped by capture prefix

## Known issue

```text
{trace}
```
"""
    report_path.write_text(text)


def run(args: argparse.Namespace) -> None:
    configure_determinism(args.seed)
    x_train, y_train, _x_test, _y_test = load_cached_data()
    x_fit, y_fit, x_val, y_val, split_info = split_train_for_validation(
        x_train,
        y_train,
        DATA_DIR,
        GEN_DIR,
        LABELS,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(
        f"Cached train={x_train.shape}; F2-fixed train={x_fit.shape}; val={x_val.shape}; "
        f"manifest={split_info['split_path']}"
    )

    print("\n=== Float training ===")
    float_model, float_history = train_float_model(
        x_fit, y_fit, x_val, y_val, args.epochs, args.batch_size
    )
    print("\n=== QAT training ===")
    qat_model, qat_history = train_qat_model(
        float_model, x_fit, y_fit, x_val, y_val, args.qat_epochs, args.batch_size
    )

    args.history_json.parent.mkdir(parents=True, exist_ok=True)
    args.history_json.write_text(
        json.dumps(
            {
                "float": float_history.history,
                "qat": qat_history.history,
                "config": {
                    "seed": args.seed,
                    "alpha": ALPHA,
                    "dense_units": DENSE_UNITS,
                    "dropout_1": DROPOUT_1,
                    "dropout_2": DROPOUT_2,
                    "learning_rate": LEARNING_RATE,
                    "qat_learning_rate": QAT_LEARNING_RATE,
                    "label_smoothing": LABEL_SMOOTHING,
                    "epochs": args.epochs,
                    "qat_epochs": args.qat_epochs,
                    "batch_size": args.batch_size,
                },
            },
            indent=2,
        )
        + "\n"
    )

    print("\n=== INT8 export ===")
    _tflite_model, export_metadata = export_int8_tflite(
        qat_model, x_fit, args.output_model, args.output_header
    )
    print(
        f"Saved {args.output_model} ({export_metadata['model_size_bytes'] / 1024:.1f} KiB)"
    )

    print("\n=== Originals-only evaluation ===")
    x_orig, y_orig, _capture_ids = ensure_originals_arrays()
    eval_results = [run_eval("baseline_retrain", args.output_model, x_orig, y_orig)]
    if EXISTING_MODEL_PATH.exists():
        eval_results.append(run_eval("existing_model", EXISTING_MODEL_PATH, x_orig, y_orig))
    for result in eval_results:
        print(
            f"{result.name}: acc={result.accuracy:.4f} macro_f1={result.macro_f1:.4f} "
            f"n={result.n} Wilson=[{result.ci_low:.4f}, {result.ci_high:.4f}]"
        )

    write_report(
        args.report,
        args,
        split_info,
        float_history,
        qat_history,
        export_metadata,
        eval_results,
    )
    print(f"Report: {args.report}")


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception:
        trace = traceback.format_exc()
        write_failure_report(args.report, args, trace)
        raise


if __name__ == "__main__":
    main()
