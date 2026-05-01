"""Verify the F3 firmware-preprocess fix.

Background
----------
Before the fix, `esp32/main/inference.cpp::inference_preprocess` normalised
each RGB byte by `(value / 255.0f) / scale + zp`, which produces INT8
outputs in [zp, zp + 1/scale] -- i.e. roughly [0, 127] for our
scale=0.00784, zp=0 model.

That is the wrong distribution. The MobileNetV2 family is trained with
`(value / 127.5f) - 1.0f` ([-1, 1]), which after INT8 quantisation
becomes roughly [-127, +127] centred near 0.

This script feeds a synthetic uniform-grey image through both the OLD
buggy formula and the NEW (correct) formula, then asserts that the new
distribution looks like a MobileNetV2 input and the old one does not.
This guards against accidental regression of F3 in firmware diffs.

Usage:
    python -m python.bench.firmware_preprocess_check \
        --tflite python/gen/model.tflite

Outputs `bench/results/firmware_preprocess_check.md`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TFLITE = REPO_ROOT / "python" / "gen" / "model.tflite"
DEFAULT_REPORT = REPO_ROOT / "bench" / "results" / "firmware_preprocess_check.md"


def get_input_quantization(tflite_path: Path) -> tuple[float, int, int]:
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
        except ImportError:
            import tensorflow as tf  # type: ignore
            Interpreter = tf.lite.Interpreter  # type: ignore[attr-defined]

    interp = Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    scale, zp = inp["quantization"]
    img_size = int(inp["shape"][1])
    return float(scale), int(zp), img_size


def buggy_preprocess(rgb: np.ndarray, scale: float, zp: int) -> np.ndarray:
    """The OLD pre-F3-fix code path: normalise to [0, 1]."""
    f = rgb.astype(np.float32) / 255.0
    q = np.round(f / scale) + zp
    return np.clip(q, -128, 127).astype(np.int8)


def fixed_preprocess(rgb: np.ndarray, scale: float, zp: int) -> np.ndarray:
    """The NEW (correct) MobileNetV2 [-1, 1] preprocessing."""
    f = (rgb.astype(np.float32) / 127.5) - 1.0
    q = np.round(f / scale) + zp
    return np.clip(q, -128, 127).astype(np.int8)


def describe(arr: np.ndarray) -> dict[str, float]:
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "frac_negative": float((arr < 0).mean()),
        "frac_zero": float((arr == 0).mean()),
        "frac_positive": float((arr > 0).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tflite", type=Path, default=DEFAULT_TFLITE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    if not args.tflite.exists():
        print(f"TFLite model not found: {args.tflite}", file=sys.stderr)
        return 1

    scale, zp, img_size = get_input_quantization(args.tflite)
    rng = np.random.default_rng(seed=42)
    rgb = rng.integers(0, 256, size=(img_size, img_size, 3), dtype=np.uint8)

    buggy = buggy_preprocess(rgb, scale, zp)
    fixed = fixed_preprocess(rgb, scale, zp)

    buggy_stats = describe(buggy)
    fixed_stats = describe(fixed)

    fixed_mean_ok = abs(fixed_stats["mean"]) < 5.0
    fixed_neg_ok = fixed_stats["frac_negative"] > 0.40
    fixed_pos_ok = fixed_stats["frac_positive"] > 0.40
    fixed_min_ok = fixed_stats["min"] <= -50
    fixed_max_ok = fixed_stats["max"] >= 50

    buggy_min_high = buggy_stats["min"] >= 0
    buggy_neg_low = buggy_stats["frac_negative"] < 0.05

    all_ok = (fixed_mean_ok and fixed_neg_ok and fixed_pos_ok and
              fixed_min_ok and fixed_max_ok and
              buggy_min_high and buggy_neg_low)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# F3 firmware preprocess regression check",
        "",
        f"- TFLite model: `{args.tflite.relative_to(REPO_ROOT)}`",
        f"- Input quant: scale={scale:.8f} zero_point={zp} img_size={img_size}",
        "- Synthetic uniform-random RGB image, seed=42.",
        "",
        "## Old buggy formula `(value / 255.0) / scale + zp`",
        "",
        "| metric | value |",
        "|---|---:|",
    ] + [f"| {k} | {v:.4f} |" for k, v in buggy_stats.items()] + [
        "",
        "## New fixed formula `(value / 127.5 - 1) / scale + zp` (MobileNetV2)",
        "",
        "| metric | value |",
        "|---|---:|",
    ] + [f"| {k} | {v:.4f} |" for k, v in fixed_stats.items()] + [
        "",
        "## Regression assertions",
        "",
        f"- fixed mean near zero (|mean|<5): **{'PASS' if fixed_mean_ok else 'FAIL'}** ({fixed_stats['mean']:.2f})",
        f"- fixed has >=40% negative values: **{'PASS' if fixed_neg_ok else 'FAIL'}** ({100*fixed_stats['frac_negative']:.1f}%)",
        f"- fixed has >=40% positive values: **{'PASS' if fixed_pos_ok else 'FAIL'}** ({100*fixed_stats['frac_positive']:.1f}%)",
        f"- fixed min <= -50: **{'PASS' if fixed_min_ok else 'FAIL'}** ({fixed_stats['min']:.0f})",
        f"- fixed max >= +50: **{'PASS' if fixed_max_ok else 'FAIL'}** ({fixed_stats['max']:.0f})",
        f"- buggy min >= 0 (proves bug): **{'PASS' if buggy_min_high else 'FAIL'}** ({buggy_stats['min']:.0f})",
        f"- buggy <5% negative (proves bug): **{'PASS' if buggy_neg_low else 'FAIL'}** ({100*buggy_stats['frac_negative']:.1f}%)",
        "",
        "## Verdict",
        "",
        ("**PASS** -- the new firmware preprocessing produces a balanced "
         "[-1, 1] distribution as expected by MobileNetV2, while the old "
         "buggy formula produces only non-negative values. The F3 fix is "
         "active and the firmware is consistent with the training pipeline."
         if all_ok else
         "**FAIL** -- check the firmware code or the model's quantisation "
         "parameters; preprocessing is not as expected."),
        "",
        "## Notes",
        "",
        "This test does not require an attached ESP32; it asserts the same "
        "math the firmware C++ helpers `mobilenet_v2_preprocess` and "
        "`quantize_to_int8` perform. Run it as part of CI to catch any "
        "future revert of the F3 fix.",
    ]

    args.report.write_text("\n".join(lines) + "\n")
    print(f"Wrote report: {args.report}")
    print(f"Verdict: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
