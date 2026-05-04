"""Embedded-footprint summary for the deployed TFLite model.

Outputs `bench/results/footprint.md` and `bench/results/footprint.csv` with:

- model file size (flash budget)
- parameter count and weight bytes (INT8 + bias int32)
- input/output tensor shapes and dtypes
- declared firmware tensor arena size (parsed from
  `esp32/main/inference.cpp::TENSOR_ARENA_SIZE`)
- XIAO ESP32-S3 Sense capability ceilings (PSRAM, internal SRAM, flash)
  for context on the budget headroom

The numbers are derived from the artefacts already in the repo (no extra
training / no new measurements). For runtime arena_used_bytes() see
the on-device serial log produced by the firmware itself.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TFLITE = REPO_ROOT / "python" / "gen" / "model.tflite"
DEFAULT_BASELINE = REPO_ROOT / "python" / "gen" / "baseline_model.tflite"
DEFAULT_FIRMWARE_INFERENCE = REPO_ROOT / "esp32" / "main" / "inference.cpp"
DEFAULT_REPORT_MD = REPO_ROOT / "bench" / "results" / "footprint.md"
DEFAULT_REPORT_CSV = REPO_ROOT / "bench" / "results" / "footprint.csv"


# XIAO ESP32-S3 Sense official capability ceilings.
# Source: Seeed Studio XIAO ESP32-S3 Sense product wiki.
BOARD_PSRAM_BYTES = 8 * 1024 * 1024      # 8 MB external PSRAM
BOARD_SRAM_BYTES = 512 * 1024            # 512 KB internal SRAM
BOARD_FLASH_BYTES = 8 * 1024 * 1024      # 8 MB flash


def parse_arena_size(inference_cpp: Path) -> int | None:
    """Parse `#define TENSOR_ARENA_SIZE (...)` and evaluate the literal."""

    if not inference_cpp.exists():
        return None
    text = inference_cpp.read_text()
    match = re.search(
        r"#define\s+TENSOR_ARENA_SIZE\s+\(?\s*([0-9*\s]+?)\s*\)?\s*$",
        text,
        re.MULTILINE,
    )
    if not match:
        return None
    expr = match.group(1).replace(" ", "")
    if not re.fullmatch(r"[0-9*]+", expr):
        return None
    parts = [int(p) for p in expr.split("*") if p]
    product = 1
    for p in parts:
        product *= p
    return product


def summarize_model(tflite_path: Path) -> dict:
    file_bytes = tflite_path.stat().st_size

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    tensors = interpreter.get_tensor_details()

    # Heuristic: anything that is not an explicit input/output and has a
    # static shape with no batch dim (or batch=1) is treated as a parameter
    # tensor. This matches what TF Lite considers "weights/biases".
    input_indices = {t["index"] for t in inputs}
    output_indices = {t["index"] for t in outputs}

    weight_bytes = 0
    weight_count = 0
    int8_weight_count = 0
    int32_bias_count = 0
    for det in tensors:
        idx = det["index"]
        if idx in input_indices or idx in output_indices:
            continue
        try:
            arr = interpreter.get_tensor(idx)
        except (ValueError, RuntimeError):
            continue
        if arr.size == 0:
            continue
        weight_count += int(arr.size)
        weight_bytes += int(arr.nbytes)
        if arr.dtype == np.int8:
            int8_weight_count += int(arr.size)
        elif arr.dtype == np.int32:
            int32_bias_count += int(arr.size)

    in_det = inputs[0]
    out_det = outputs[0]
    in_shape = list(in_det["shape"])
    out_shape = list(out_det["shape"])
    in_dtype = np.dtype(in_det["dtype"]).name
    out_dtype = np.dtype(out_det["dtype"]).name
    in_quant = in_det.get("quantization", (0.0, 0))
    out_quant = out_det.get("quantization", (0.0, 0))

    return {
        "file_path": tflite_path,
        "file_bytes": file_bytes,
        "weight_count": weight_count,
        "weight_bytes": weight_bytes,
        "int8_weight_count": int8_weight_count,
        "int32_bias_count": int32_bias_count,
        "input_shape": in_shape,
        "input_dtype": in_dtype,
        "input_scale": float(in_quant[0]),
        "input_zero_point": int(in_quant[1]),
        "output_shape": out_shape,
        "output_dtype": out_dtype,
        "output_scale": float(out_quant[0]),
        "output_zero_point": int(out_quant[1]),
    }


def fmt_kib(n_bytes: int) -> str:
    return f"{n_bytes / 1024:.2f} KiB"


def fmt_mib(n_bytes: int) -> str:
    return f"{n_bytes / (1024 * 1024):.2f} MiB"


def write_report_md(
    path: Path,
    model: dict,
    baseline: dict | None,
    arena_bytes: int | None,
) -> None:
    lines: list[str] = []
    lines.append("# Embedded footprint")
    lines.append("")
    lines.append(
        "Numbers below describe the deployed model only and are derived "
        "from the artefacts in `python/gen/` plus the firmware constants in "
        "`esp32/main/inference.cpp`. No new measurements are performed."
    )
    lines.append("")
    lines.append("## Deployed model (`python/gen/model.tflite`)")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| file size (flash budget) | {model['file_bytes']:,} B "
                 f"({fmt_kib(model['file_bytes'])}) |")
    lines.append(f"| total parameter count | {model['weight_count']:,} |")
    lines.append(f"| INT8 weights | {model['int8_weight_count']:,} |")
    lines.append(f"| INT32 biases | {model['int32_bias_count']:,} |")
    lines.append(f"| weight + bias bytes | {model['weight_bytes']:,} B "
                 f"({fmt_kib(model['weight_bytes'])}) |")
    lines.append(f"| input shape / dtype | {tuple(model['input_shape'])} / "
                 f"{model['input_dtype']} |")
    lines.append(f"| input quant scale / zp | "
                 f"{model['input_scale']:.6g} / {model['input_zero_point']} |")
    lines.append(f"| output shape / dtype | {tuple(model['output_shape'])} / "
                 f"{model['output_dtype']} |")
    lines.append(f"| output quant scale / zp | "
                 f"{model['output_scale']:.6g} / {model['output_zero_point']} |")
    lines.append("")

    if baseline is not None:
        lines.append("## Baseline reference (`python/gen/baseline_model.tflite`)")
        lines.append("")
        lines.append("| metric | challenger | baseline | Δ |")
        lines.append("|---|---:|---:|---:|")
        delta_file = model["file_bytes"] - baseline["file_bytes"]
        delta_w = model["weight_count"] - baseline["weight_count"]
        lines.append(f"| file size | {model['file_bytes']:,} B | "
                     f"{baseline['file_bytes']:,} B | {delta_file:+,} B |")
        lines.append(f"| parameters | {model['weight_count']:,} | "
                     f"{baseline['weight_count']:,} | {delta_w:+,} |")
        lines.append("")

    lines.append("## Firmware-side budget")
    lines.append("")
    lines.append("| component | value | source |")
    lines.append("|---|---:|---|")
    if arena_bytes is not None:
        lines.append(f"| `TENSOR_ARENA_SIZE` (declared) | {arena_bytes:,} B "
                     f"({fmt_kib(arena_bytes)}) | "
                     "`esp32/main/inference.cpp` |")
        lines.append("| arena allocation region | PSRAM "
                     "(`heap_caps_malloc(... MALLOC_CAP_SPIRAM)`) | "
                     "`esp32/main/inference.cpp` |")
    else:
        lines.append("| `TENSOR_ARENA_SIZE` | _not parsed_ | "
                     "`esp32/main/inference.cpp` |")
    lines.append("| arena_used_bytes() (runtime) | _measured on device, "
                 "see ESP_LOGI \"Arena used: ... bytes\"_ | live serial log |")
    lines.append("")

    lines.append("## Board capability ceilings (XIAO ESP32-S3 Sense)")
    lines.append("")
    lines.append("| resource | total | used by model |")
    lines.append("|---|---:|---:|")
    flash_pct = 100.0 * model["file_bytes"] / BOARD_FLASH_BYTES
    lines.append(f"| flash | {fmt_mib(BOARD_FLASH_BYTES)} | "
                 f"{fmt_kib(model['file_bytes'])} ({flash_pct:.2f}%) |")
    if arena_bytes is not None:
        psram_pct = 100.0 * arena_bytes / BOARD_PSRAM_BYTES
        lines.append(f"| PSRAM | {fmt_mib(BOARD_PSRAM_BYTES)} | "
                     f"arena reserved {fmt_kib(arena_bytes)} "
                     f"({psram_pct:.2f}%) |")
    else:
        lines.append(f"| PSRAM | {fmt_mib(BOARD_PSRAM_BYTES)} | _arena size "
                     "not parsed_ |")
    lines.append(f"| internal SRAM | {fmt_kib(BOARD_SRAM_BYTES)} | _firmware "
                 "code + stacks; not measured here_ |")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Weight bytes are the on-flash representation: INT8 weights are "
        "1 byte each and INT32 biases are 4 bytes each. Activation tensors "
        "are not parameters and do not contribute to flash; they live in "
        "the runtime arena."
    )
    lines.append(
        "- `TENSOR_ARENA_SIZE` is a worst-case allocation; the actual "
        "arena_used_bytes() reported by the runtime can be smaller. Read "
        "the on-device serial log to confirm and tighten the bound."
    )
    lines.append(
        "- The arena lives in PSRAM, so the 512 KiB internal SRAM ceiling "
        "is not the binding constraint for inference memory."
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_report_csv(path: Path, model: dict, baseline: dict | None,
                     arena_bytes: int | None) -> None:
    rows: list[tuple[str, str]] = [
        ("model_file_bytes", str(model["file_bytes"])),
        ("model_weight_count", str(model["weight_count"])),
        ("model_int8_weight_count", str(model["int8_weight_count"])),
        ("model_int32_bias_count", str(model["int32_bias_count"])),
        ("model_weight_bytes", str(model["weight_bytes"])),
        ("model_input_shape", "x".join(str(d) for d in model["input_shape"])),
        ("model_input_dtype", model["input_dtype"]),
        ("model_input_scale", f"{model['input_scale']:.10g}"),
        ("model_input_zero_point", str(model["input_zero_point"])),
        ("model_output_shape", "x".join(str(d) for d in model["output_shape"])),
        ("model_output_dtype", model["output_dtype"]),
        ("model_output_scale", f"{model['output_scale']:.10g}"),
        ("model_output_zero_point", str(model["output_zero_point"])),
    ]
    if baseline is not None:
        rows.extend([
            ("baseline_file_bytes", str(baseline["file_bytes"])),
            ("baseline_weight_count", str(baseline["weight_count"])),
        ])
    if arena_bytes is not None:
        rows.append(("firmware_tensor_arena_size_bytes", str(arena_bytes)))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tflite", type=Path, default=DEFAULT_TFLITE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--firmware", type=Path,
                        default=DEFAULT_FIRMWARE_INFERENCE)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    args = parser.parse_args()

    if not args.tflite.exists():
        print(f"missing tflite: {args.tflite}")
        return 2

    model = summarize_model(args.tflite)
    baseline = summarize_model(args.baseline) if args.baseline.exists() else None
    arena_bytes = parse_arena_size(args.firmware)

    write_report_md(args.report_md, model, baseline, arena_bytes)
    write_report_csv(args.report_csv, model, baseline, arena_bytes)

    print(f"wrote {args.report_md.relative_to(REPO_ROOT)}")
    print(f"wrote {args.report_csv.relative_to(REPO_ROOT)}")
    print(f"  model file size      : {model['file_bytes']:,} B "
          f"({fmt_kib(model['file_bytes'])})")
    print(f"  parameter count      : {model['weight_count']:,}")
    if arena_bytes is not None:
        print(f"  TENSOR_ARENA_SIZE    : {arena_bytes:,} B "
              f"({fmt_kib(arena_bytes)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
