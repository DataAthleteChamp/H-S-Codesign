"""Evaluate an INT8 TFLite model on a preprocessed NumPy test set."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _quant_params(detail: dict[str, Any]) -> tuple[float, int]:
    scale, zero_point = detail.get("quantization", (0.0, 0))
    if not scale:
        params = detail.get("quantization_parameters", {})
        scales = params.get("scales", [])
        zero_points = params.get("zero_points", [])
        if len(scales):
            scale = float(scales[0])
        if len(zero_points):
            zero_point = int(zero_points[0])
    return float(scale), int(zero_point)


def _validate_inputs(x: np.ndarray, y: np.ndarray, norm: str) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"x must have shape (N, H, W, 3), got {x.shape}")
    if len(y) != len(x):
        raise ValueError(f"x/y length mismatch: {len(x)} images vs {len(y)} labels")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains NaN or infinite values")

    x = x.astype(np.float32, copy=False)
    y = y.astype(np.int64, copy=False).reshape(-1)

    x_min = float(np.min(x)) if len(x) else 0.0
    x_max = float(np.max(x)) if len(x) else 0.0
    if norm == "pm1" and (x_min < -1.05 or x_max > 1.05):
        _warn(f"--norm pm1 but x range is [{x_min:.3f}, {x_max:.3f}]")
    if norm == "unit" and (x_min < -0.05 or x_max > 1.05):
        _warn(f"--norm unit but x range is [{x_min:.3f}, {x_max:.3f}]")
    return x, y


def _load_capture_ids(path: str | None, n: int) -> np.ndarray:
    if path is None:
        return np.array([], dtype=str)
    capture_ids = np.load(path, allow_pickle=True)
    capture_ids = np.asarray(capture_ids, dtype=str).reshape(-1)
    if len(capture_ids) != n:
        raise ValueError(
            f"capture-id length mismatch: {len(capture_ids)} ids vs {n} samples"
        )
    return capture_ids


def _prepare_interpreter(
    model_path: Path, sample_shape: tuple[int, int, int], num_threads: int
) -> tuple[tf.lite.Interpreter, dict[str, Any], dict[str, Any]]:
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path), num_threads=max(1, num_threads)
    )
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1:
        raise ValueError(f"expected one model input, got {len(input_details)}")
    if len(output_details) != 1:
        raise ValueError(f"expected one model output, got {len(output_details)}")

    input_detail = input_details[0]
    wanted_shape = [1, *sample_shape]
    current_shape = [int(v) for v in input_detail["shape"]]
    shape_signature = [int(v) for v in input_detail.get("shape_signature", current_shape)]
    if len(current_shape) != len(wanted_shape):
        raise ValueError(
            f"model input rank {len(current_shape)} does not match x rank {len(wanted_shape)}"
        )
    if current_shape != wanted_shape:
        fixed_bad = [
            idx
            for idx, (cur, want, sig) in enumerate(
                zip(current_shape, wanted_shape, shape_signature)
            )
            if cur != want and sig != -1
        ]
        if any(idx != 0 for idx in fixed_bad):
            raise ValueError(
                f"model expects input shape {current_shape}; x samples are {wanted_shape}"
            )
        interpreter.resize_tensor_input(input_detail["index"], wanted_shape, strict=False)

    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    return interpreter, input_detail, output_detail


def _quantize_input(sample: np.ndarray, input_detail: dict[str, Any]) -> np.ndarray:
    dtype = np.dtype(input_detail["dtype"])
    batched = sample[np.newaxis, ...].astype(np.float32, copy=False)
    if np.issubdtype(dtype, np.floating):
        return batched.astype(dtype, copy=False)

    scale, zero_point = _quant_params(input_detail)
    if scale == 0.0:
        raise ValueError("integer model input has no quantization scale")
    q = np.rint(batched / scale + zero_point)
    info = np.iinfo(dtype)
    return np.clip(q, info.min, info.max).astype(dtype)


def _dequantize_output(output: np.ndarray, output_detail: dict[str, Any]) -> np.ndarray:
    dtype = np.dtype(output_detail["dtype"])
    if np.issubdtype(dtype, np.integer):
        scale, zero_point = _quant_params(output_detail)
        if scale == 0.0:
            raise ValueError("integer model output has no quantization scale")
        output = scale * (output.astype(np.float32) - float(zero_point))
    else:
        output = output.astype(np.float32, copy=False)
    if output.ndim >= 2 and output.shape[0] == 1:
        output = output[0]
    return np.asarray(output, dtype=np.float32).reshape(-1)


def evaluate(
    model_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    norm: str,
    num_threads: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x, y = _validate_inputs(x, y, norm)
    interpreter, input_detail, output_detail = _prepare_interpreter(
        model_path, tuple(int(v) for v in x.shape[1:]), num_threads
    )

    input_scale, input_zero_point = _quant_params(input_detail)
    output_scale, output_zero_point = _quant_params(output_detail)

    probs: list[np.ndarray] = []
    predictions = np.empty(len(x), dtype=np.int64)
    input_index = input_detail["index"]
    output_index = output_detail["index"]

    for idx, sample in enumerate(x):
        interpreter.set_tensor(input_index, _quantize_input(sample, input_detail))
        interpreter.invoke()
        row = _dequantize_output(interpreter.get_tensor(output_index), output_detail)
        probs.append(row)
        predictions[idx] = int(np.argmax(row))

    if probs:
        probs_array = np.vstack(probs).astype(np.float32)
    else:
        probs_array = np.empty((0, 0), dtype=np.float32)

    metadata = {
        "input_shape": np.asarray(input_detail["shape"], dtype=np.int64),
        "input_scale": np.float32(input_scale),
        "input_zero_point": np.int32(input_zero_point),
        "output_scale": np.float32(output_scale),
        "output_zero_point": np.int32(output_zero_point),
        "input_dtype": str(np.dtype(input_detail["dtype"])),
        "output_dtype": str(np.dtype(output_detail["dtype"])),
    }
    return predictions, probs_array, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to .tflite model")
    parser.add_argument("--x", required=True, help="Path to preprocessed x_test .npy")
    parser.add_argument("--y", required=True, help="Path to y_test .npy")
    parser.add_argument(
        "--capture-ids", help="Optional .npy array of capture identifiers, length N"
    )
    parser.add_argument("--norm", choices=("pm1", "unit"), required=True)
    parser.add_argument("--out", required=True, help="Output .npz path")
    parser.add_argument("--num-threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    x_path = Path(args.x)
    y_path = Path(args.y)
    out_path = Path(args.out)

    x = np.load(x_path)
    y = np.load(y_path)
    x, y = _validate_inputs(x, y, args.norm)
    capture_ids = _load_capture_ids(args.capture_ids, len(x))

    predictions, probs, metadata = evaluate(
        model_path=model_path,
        x=x,
        y=y,
        norm=args.norm,
        num_threads=args.num_threads,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        predictions=predictions.astype(np.int64),
        probs=probs.astype(np.float32),
        labels=y.astype(np.int64),
        capture_ids=capture_ids,
        model_path=str(model_path),
        x_path=str(x_path),
        y_path=str(y_path),
        norm=args.norm,
        **metadata,
    )

    acc = accuracy_score(y, predictions)
    macro_f1 = f1_score(y, predictions, average="macro", zero_division=0)
    print(f"acc={acc:.4f}  macro_f1={macro_f1:.4f}  n={len(y)}")


if __name__ == "__main__":
    main()
