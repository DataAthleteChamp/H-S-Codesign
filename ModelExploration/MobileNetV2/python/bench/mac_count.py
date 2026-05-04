"""Estimate TFLite operator MAC counts and save a CSV summary."""

from __future__ import annotations

import argparse
import csv
import io
import os
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb

COMPUTE_OPS = {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"}


def run_analyzer(model_path: Path) -> str:
    analyzer = getattr(tf.lite.experimental, "Analyzer", None)
    if analyzer is None:
        print("TensorFlow Lite Analyzer is unavailable in this TensorFlow build.")
        return ""

    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            result = analyzer.analyze(model_path=str(model_path), gpu_compatibility=False)
    except TypeError:
        with redirect_stdout(buffer):
            result = analyzer.analyze(model_path=str(model_path))

    text = ""
    if isinstance(result, str):
        text += result
    text += buffer.getvalue()
    if text:
        print(text, end="" if text.endswith("\n") else "\n")
    return text


def builtin_name_map() -> dict[int, str]:
    names: dict[int, str] = {}
    for name in dir(schema_fb.BuiltinOperator):
        if name.startswith("_"):
            continue
        value = getattr(schema_fb.BuiltinOperator, name)
        if isinstance(value, int):
            names[int(value)] = name
    return names


def tensor_shape(tensor: Any) -> list[int]:
    if tensor is None:
        return []
    if hasattr(tensor, "ShapeAsNumpy"):
        shape = tensor.ShapeAsNumpy()
        if shape is not None:
            return [int(v) for v in shape]
    return [int(tensor.Shape(i)) for i in range(tensor.ShapeLength())]


def product(values: list[int]) -> int:
    if not values:
        return 1
    return int(np.prod(np.asarray(values, dtype=np.int64)))


def op_inputs(operator: Any) -> list[int]:
    return [int(operator.Inputs(i)) for i in range(operator.InputsLength())]


def op_outputs(operator: Any) -> list[int]:
    return [int(operator.Outputs(i)) for i in range(operator.OutputsLength())]


def get_tensor(subgraph: Any, index: int) -> Any | None:
    if index < 0:
        return None
    return subgraph.Tensors(index)


def conv2d_macs(subgraph: Any, operator: Any) -> int:
    inputs = op_inputs(operator)
    outputs = op_outputs(operator)
    if len(inputs) < 2 or not outputs:
        return 0
    filter_shape = tensor_shape(get_tensor(subgraph, inputs[1]))
    output_shape = tensor_shape(get_tensor(subgraph, outputs[0]))
    if len(filter_shape) != 4 or len(output_shape) != 4:
        return 0
    batch, out_h, out_w, out_c = output_shape
    _, filter_h, filter_w, in_c = filter_shape
    return product([batch, out_h, out_w, out_c, filter_h, filter_w, in_c])


def depthwise_conv2d_macs(subgraph: Any, operator: Any) -> int:
    inputs = op_inputs(operator)
    outputs = op_outputs(operator)
    if len(inputs) < 2 or not outputs:
        return 0
    filter_shape = tensor_shape(get_tensor(subgraph, inputs[1]))
    output_shape = tensor_shape(get_tensor(subgraph, outputs[0]))
    if len(filter_shape) != 4 or len(output_shape) != 4:
        return 0
    batch, out_h, out_w, out_c = output_shape
    _, filter_h, filter_w, _ = filter_shape
    return product([batch, out_h, out_w, out_c, filter_h, filter_w])


def fully_connected_macs(subgraph: Any, operator: Any) -> int:
    inputs = op_inputs(operator)
    outputs = op_outputs(operator)
    if len(inputs) < 2 or not outputs:
        return 0
    input_shape = tensor_shape(get_tensor(subgraph, inputs[0]))
    weight_shape = tensor_shape(get_tensor(subgraph, inputs[1]))
    output_shape = tensor_shape(get_tensor(subgraph, outputs[0]))

    batch = output_shape[0] if len(output_shape) > 1 else 1
    if len(weight_shape) >= 2:
        out_units = weight_shape[0]
        input_units = weight_shape[-1]
    else:
        out_units = output_shape[-1] if output_shape else 0
        input_units = product(input_shape[1:] if len(input_shape) > 1 else input_shape)
    return product([batch, out_units, input_units])


def estimate_operator_macs(subgraph: Any, op_name: str, operator: Any) -> int:
    if op_name == "CONV_2D":
        return conv2d_macs(subgraph, operator)
    if op_name == "DEPTHWISE_CONV_2D":
        return depthwise_conv2d_macs(subgraph, operator)
    if op_name == "FULLY_CONNECTED":
        return fully_connected_macs(subgraph, operator)
    return 0


def operator_name(model: Any, operator: Any, names: dict[int, str]) -> str:
    opcode = model.OperatorCodes(operator.OpcodeIndex())
    builtin_code = int(opcode.BuiltinCode())
    if builtin_code == schema_fb.BuiltinOperator.CUSTOM:
        custom_code = opcode.CustomCode()
        if isinstance(custom_code, bytes):
            return custom_code.decode("utf-8", errors="replace")
        return str(custom_code)
    return names.get(builtin_code, f"BUILTIN_{builtin_code}")


def estimate_macs(model_path: Path) -> dict[str, dict[str, int]]:
    model_buf = model_path.read_bytes()
    model = schema_fb.Model.GetRootAsModel(model_buf, 0)
    if model.SubgraphsLength() < 1:
        raise ValueError("TFLite model contains no subgraphs")
    subgraph = model.Subgraphs(0)
    names = builtin_name_map()
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "est_macs": 0})

    for idx in range(subgraph.OperatorsLength()):
        operator = subgraph.Operators(idx)
        name = operator_name(model, operator, names)
        stats[name]["count"] += 1
        stats[name]["est_macs"] += estimate_operator_macs(subgraph, name, operator)
    return dict(stats)


def write_csv(stats: dict[str, dict[str, int]], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_compute_macs = sum(
        row["est_macs"] for name, row in stats.items() if name in COMPUTE_OPS
    )
    total_compute_count = sum(row["count"] for name, row in stats.items() if name in COMPUTE_OPS)

    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("op_name", "count", "est_macs"))
        writer.writeheader()
        for name in sorted(stats):
            writer.writerow(
                {
                    "op_name": name,
                    "count": stats[name]["count"],
                    "est_macs": stats[name]["est_macs"],
                }
            )
        writer.writerow(
            {
                "op_name": "TOTAL_COMPUTE",
                "count": total_compute_count,
                "est_macs": total_compute_macs,
            }
        )
    return total_compute_macs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="python/gen/model.tflite")
    parser.add_argument("--out", default="bench/results/mac_count.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    out_path = Path(args.out)
    run_analyzer(model_path)
    stats = estimate_macs(model_path)
    total_macs = write_csv(stats, out_path)

    print("MAC summary:")
    for name in sorted(stats):
        print(
            f"  {name}: count={stats[name]['count']} "
            f"est_macs={stats[name]['est_macs']}"
        )
    print(f"  TOTAL_COMPUTE: est_macs={total_macs}")
    print(f"wrote {out_path}  total_compute_macs={total_macs}")


if __name__ == "__main__":
    main()
